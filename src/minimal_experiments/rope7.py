import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, get_args

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


def repo_root() -> Path:
    """Walk up from the cwd to the folder holding pyproject.toml, so paths work no
    matter where the kernel is launched from."""
    here = Path(__file__)
    for d in here.parents:
        if (d / "pyproject.toml").exists():
            return d
    raise RuntimeError("no pyproject.toml found above " + str(here))


ROOT = repo_root()
DATA = ROOT / "data" / "input.txt"
text = DATA.read_text()
device = "cuda" if torch.cuda.is_available() else "cpu"


class CharTokenizer:
    def __init__(self, text: str):
        self.itoc = sorted(set(text))
        self.ctoi = {c: i for i, c in enumerate(self.itoc)}
        self.vocab_size = len(self.itoc)

    def encode(self, s: str) -> list[int]:
        return [self.ctoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itoc[i] for i in ids)


tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split: str, batch_size: int, block_size: int):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


Attention = Literal["flash", "gqa"]
RopeScaling = Literal["none", "pi", "ntk", "yarn"]


@dataclass(frozen=True, kw_only=True)
class GPTConfig:
    vocab_size: int  # V
    block_size: int  # T
    n_embed: int  # E
    n_head: int  # nh
    n_layer: int
    n_kv_head: int | None = None
    attention: Attention = "flash"
    rope_scale: float = 1.0  # context extension factor; 1.0 is plain RoPE
    rope_scaling: RopeScaling = "none"
    rope_alpha: float = 1.0  # yarn: under this many turns in training, interpolate
    rope_beta: float = 32.0  # yarn: over this many, leave the pair alone


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    batch_size: int  # B
    max_steps: int
    lr: float
    eval_interval: int
    eval_iters: int


class ResidualProj(nn.Linear):
    """Linear whose output is added to the residual stream; gets 1/sqrt(2*n_layer) init."""


def rope_inv_freq(head_size: int, base: float = 10000.0, device=None) -> Tensor:
    assert head_size % 2 == 0, "head_size must be even: the dims rotate in pairs"
    i = torch.arange(0, head_size, 2, device=device, dtype=torch.float32)
    return base ** (-i / head_size)  # [hs/2]


def pi_inv_freq(head_size: int, base: float, device, scale: float) -> Tensor:
    """Position interpolation: divide every position by scale, which is the same
    as dividing every theta. The baseline the other two improve on -- it fits the
    far positions back inside the trained range, but it slows the fast pairs by
    the same factor, and those were the model's only fine-grained ruler."""
    return rope_inv_freq(head_size, base, device) / scale


def ntk_inv_freq(head_size: int, base: float, device, scale: float) -> Tensor:
    """Stretch the base, not the positions. log theta is linear in i and anchored
    at pair 0, so one base change tilts the whole set: pair 0 pinned, the last
    pair shrunk by exactly scale, geometric in between."""
    assert head_size > 2, "NTK exponent divides by head_size - 2"
    base *= scale ** (head_size / (head_size - 2))
    return rope_inv_freq(head_size, base, device)


def yarn_inv_freq(
    head_size: int,
    base: float,
    device,
    scale: float,
    train_len: int,
    alpha: float = 1.0,
    beta: float = 32.0,
) -> Tensor:
    """NTK-by-parts: pick each pair's treatment from what training showed it.

    Under alpha turns the model only saw a sliver of that pair, so shrink it by
    the full scale; over beta turns it saw every phase, so leave it alone. NTK
    infers this ramp from one exponent, YaRN measures it per pair."""
    inv_freq = rope_inv_freq(head_size, base, device)
    turns = train_len * inv_freq / (2 * math.pi)  # cycles made inside training
    keep = ((turns - alpha) / (beta - alpha)).clamp(0, 1)  # 1 = extrapolate as is
    return (1 - keep) * inv_freq / scale + keep * inv_freq


def yarn_attn_scale(scale: float) -> float:
    """Interpolating narrows the spread of q.k, so YaRN scales the logits back up.
    Free here: q and k are both rotated, so a factor on cos and sin reaches the
    score squared, which is what the paper asks for."""
    return 0.1 * math.log(scale) + 1.0


def rope_angles(
    offset: int,
    seq_len: int,
    head_size: int,
    base: float = 10000.0,
    device=None,
    scale: float = 1.0,
    scaling: RopeScaling = "none",
    train_len: int = 0,
    alpha: float = 1.0,
    beta: float = 32.0,
) -> tuple[Tensor, Tensor]:
    mult = 1.0  # what cos and sin get multiplied by; only yarn moves it
    if scale == 1.0:
        scaling = "none"  # no stretch asked for, so no arithmetic to round
    match scaling:
        case "none":
            inv_freq = rope_inv_freq(head_size, base, device)
        case "pi":
            inv_freq = pi_inv_freq(head_size, base, device, scale)
        case "ntk":
            inv_freq = ntk_inv_freq(head_size, base, device, scale)
        case "yarn":
            assert train_len > 0, "yarn counts turns, so it needs the trained context"
            inv_freq = yarn_inv_freq(
                head_size, base, device, scale, train_len, alpha, beta
            )
            mult = yarn_attn_scale(scale)
        case _:
            raise ValueError(f"unknown rope scaling: {scaling}")
    pos = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    angles = pos.unsqueeze(1) * inv_freq  # [seq_len, hs/2]
    return angles.cos() * mult, angles.sin() * mult


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]  # [..., T, hs/2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class FlashAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = ResidualProj(n_embed, n_embed)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache=None,
        use_cache: bool = False,
    ):
        B, T, E = x.shape
        nh = self.n_head
        qkv = self.qkv(x)  # [B, T, 3E]
        qkv = rearrange(qkv, "b t (three e) -> three b t e", three=3)
        q, k, v = [rearrange(t, "b t (nh hs) -> b nh t hs", nh=nh) for t in qkv]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            # [B, nh, T_past, hs] ++ [B, nh, T, hs] -> [B, nh, T_kv, hs]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)
        T_kv = k.size(2)
        T_past = T_kv - T
        if T_kv == T:
            is_causal, attn_mask = True, None
        elif T == 1:
            is_causal, attn_mask = False, None
        else:
            is_causal = False
            q_pos = torch.arange(T_past, T_kv, device=x.device).unsqueeze(1)
            k_pos = torch.arange(T_kv, device=x.device)
            attn_mask = k_pos <= q_pos  # [T, T_kv]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal
        )  # [B, nh, T, hs]
        out = rearrange(out, "b nh t hs -> b t (nh hs)")  # [B, T, E]
        out = self.proj(out)
        return (out, new_cache) if use_cache else out


class GQAttention(nn.Module):
    def __init__(
        self,
        n_embed: int,
        n_head: int,
        n_kv_head: int | None = None,
    ):
        super().__init__()
        assert n_embed % n_head == 0
        n_kv_head = n_head if n_kv_head is None else n_kv_head
        assert n_head % n_kv_head == 0
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head  # q heads per kv head
        self.qkv = nn.Linear(
            n_embed,
            n_embed + 2 * n_kv_head * self.head_size,
            bias=False,
        )
        self.proj = ResidualProj(n_embed, n_embed)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache=None,
        use_cache: bool = False,
    ):
        B, T, E = x.shape
        nkv, hs = self.n_kv_head, self.head_size
        qkv: Tensor = self.qkv(x)
        q, k, v = qkv.split([E, nkv * hs, nkv * hs], dim=-1)
        q, k, v = [rearrange(t, "b t (n hs) -> b n t hs", hs=hs) for t in [q, k, v]]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            # [B, nkv, T_past, hs] ++ [B, nkv, T, hs] -> [B, nkv, T_kv, hs]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)
        T_kv = k.size(2)
        T_past = T_kv - T
        if T_kv == T:
            is_causal, attn_mask = True, None
        elif T == 1:
            is_causal, attn_mask = False, None
        else:
            is_causal = False
            q_pos = torch.arange(T_past, T_kv, device=x.device).unsqueeze(1)
            k_pos = torch.arange(T_kv, device=x.device)
            attn_mask = k_pos <= q_pos  # [T, T_kv]
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            enable_gqa=self.n_rep > 1,
        )  # [B, nh, T, hs]
        out = rearrange(out, "b nh t hs -> b t (nh hs)")  # [B, T, E]
        out = self.proj(out)
        return (out, new_cache) if use_cache else out


class FeedForward(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            ResidualProj(4 * n_embed, n_embed),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # [B, T, E]


def make_attention(kind: Attention, n_embed, n_head, n_kv_head=None):
    mha_only = "kv groups; it is MHA only, so n_kv_head must equal n_head"
    match kind:
        case "flash":
            assert n_kv_head in (None, n_head), f"{kind} has no {mha_only}"
            return FlashAttention(n_embed, n_head)
        case "gqa":
            return GQAttention(n_embed, n_head, n_kv_head)
        case _:
            raise ValueError(f"unknown attention: {kind}")


class Block(nn.Module):
    def __init__(
        self,
        n_embed: int,
        n_head: int,
        n_kv_head: int | None = None,
        attention: Attention = "gqa",
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.attn = make_attention(attention, n_embed, n_head, n_kv_head)
        self.ffwd = FeedForward(n_embed)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache=None,
        use_cache: bool = False,
    ):
        if use_cache:
            attn_out, new_cache = self.attn(
                self.ln1(x), cos, sin, kv_cache, use_cache=True
            )
        else:
            attn_out = self.attn(self.ln1(x), cos, sin)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return (x, new_cache) if use_cache else x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int,
        n_head: int,
        n_layer: int,
        n_kv_head: int | None = None,
        attention: Attention = "gqa",
        rope_scale: float = 1.0,
        rope_scaling: RopeScaling = "none",
        rope_alpha: float = 1.0,
        rope_beta: float = 32.0,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.block_size = block_size
        self.rope_scale = rope_scale
        self.rope_scaling = rope_scaling
        self.rope_alpha = rope_alpha
        self.rope_beta = rope_beta
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.head_size = n_embed // n_head
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head, n_kv_head, attention) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding_table.weight

    def _init_weights(self, module: nn.Module) -> None:
        with torch.no_grad():
            if isinstance(module, nn.Linear):
                std = 0.02
                if isinstance(module, ResidualProj):
                    std *= (2 * self.n_layer) ** -0.5
                module.weight.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.normal_(mean=0.0, std=0.02)

    def forward(
        self,
        idx: Tensor,
        kv_caches=None,
        use_cache: bool = False,
    ):
        _, T = idx.shape
        T_past = 0 if kv_caches is None else kv_caches[0][0].size(2)
        x = self.token_embedding_table(idx)
        cos, sin = rope_angles(
            T_past,
            T,
            self.head_size,
            device=x.device,
            scale=self.rope_scale,
            scaling=self.rope_scaling,
            train_len=self.block_size,  # what "seen during training" means
            alpha=self.rope_alpha,
            beta=self.rope_beta,
        )
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        new_caches = []
        for i, block in enumerate(self.blocks):
            if use_cache:
                layer_cache = kv_caches[i] if kv_caches is not None else None
                x, new_cache = block(x, cos, sin, layer_cache, use_cache=True)
                new_caches.append(new_cache)
            else:
                x = block(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return (logits, new_caches) if use_cache else logits


@torch.no_grad()
def generate(
    model: GPT,
    idx: Tensor,
    max_new_tokens: int,
    use_cache: bool = False,
):
    was_training = model.training
    model.eval()
    kv_caches = None
    for _ in range(max_new_tokens):
        if use_cache:
            step = idx if kv_caches is None else idx[:, -1:]
            logits, kv_caches = model(step, kv_caches, use_cache=True)
        else:
            logits = model(idx)
        logits = logits[:, -1, :]  # [B, V]
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    model.train(was_training)
    return idx


def batch_loss(model: GPT, x: Tensor, y: Tensor) -> Tensor:
    B, T = x.shape
    logits = model(x)  # [B, T, V]
    loss = F.cross_entropy(logits.reshape(B * T, -1), y.reshape(B * T))
    return loss


@torch.no_grad()
def full_val_loss(model: GPT, batch_size: int):
    was_training = model.training
    model.eval()
    B, T = batch_size, model.block_size
    nwin = (len(val_data) - 1) // T  # how many full windows fit
    x = val_data[: nwin * T].view(nwin, T)  # (nwin, T) inputs
    y = val_data[1 : nwin * T + 1].view(nwin, T)  # targets, shifted +1
    total = torch.tensor(0.0, device=device)
    count = 0
    for i in range(0, nwin, B):  # batch the windows through the model
        xb, yb = x[i : i + B].to(device), y[i : i + B].to(device)
        # weight by #tokens so the mean is correct over uneven chunks
        loss = batch_loss(model, xb, yb)
        total += loss * yb.numel()
        count += yb.numel()
    model.train(was_training)
    return total / count


@torch.no_grad()
def train_loss_est(model: GPT, batch_size: int, iters=100):
    """Cheap train-loss estimate from random batches (the train split is huge; no
    full pass needed)."""
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(iters):
        xb, yb = get_batch("train", batch_size, model.block_size)
        loss = batch_loss(model, xb, yb)
        losses.append(loss)
    est = torch.stack(losses).mean().item()
    model.train(was_training)
    return est


def train(
    model: GPT,
    train_cfg: TrainConfig,
):
    B, T = train_cfg.batch_size, model.block_size
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    def evaluate(it: int):
        tr, va = (
            train_loss_est(model, B, train_cfg.eval_iters),
            full_val_loss(model, B),
        )
        print(f"  step {it:>4} : train {tr:.3f}   val {va:.3f}")

    for it in range(train_cfg.max_steps):
        if it % train_cfg.eval_interval == 0:
            evaluate(it)
        xb, yb = get_batch("train", B, T)
        loss = batch_loss(model, xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    evaluate(train_cfg.max_steps)


# picked so yarn has something to decide. hs = 16 spreads 8 pairs over the
# spectrum, and 1024 positions is long enough that pairs 0 and 1 turn more than
# beta=32 times inside training -- so yarn keeps them exact while ntk, which
# only pins pair 0, still tilts pair 1
small_gpt_cfg = GPTConfig(
    vocab_size=tok.vocab_size,
    block_size=1024,
    n_embed=64,
    n_head=4,
    n_layer=3,
    n_kv_head=2,
    attention="gqa",
)
small_train_cfg = TrainConfig(
    batch_size=32,
    max_steps=500,
    lr=1e-2,
    eval_interval=100,
    eval_iters=20,
)


@torch.no_grad()
def extrapolation_report(model: GPT, window: int, scales=(2.0, 4.0, 8.0)):
    """Val loss per position band in one long window, per method and scale.

    Bands, not a mean: the mean hides where extrapolation fails. The first column
    is the trained range and is the one to watch -- yarn leaves the pairs that
    cover it exact, so it should give up less there than ntk does."""
    was = (model.training, model.rope_scale, model.rope_scaling)
    model.eval()
    edges = [0, model.block_size]
    while edges[-1] < window:
        edges.append(min(edges[-1] * 2, window))
    # block_size 1024, window 4096 -> [0, 1024, 2048, 4096]
    nwin = min(64, (len(val_data) - 1) // window)
    # enable_gqa drops sdpa off the flash path onto one that materialises the
    # scores, so peak memory grows with window**2, not window
    d = val_data[: nwin * window + 1]
    x = d[: nwin * window].view(nwin, window)  # cpu; moved a batch at a time
    y = d[1 : nwin * window + 1].view(nwin, window)
    spans = list(zip(edges, edges[1:]))

    def row(label: str):
        chunks = []
        for i in range(nwin):
            xb, yb = x[i : i + 1].to(device), y[i : i + 1].to(device)
            logits = model(xb)
            chunks.append(
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    yb.reshape(-1),
                    reduction="none",
                ).view(xb.size(0), window)
            )
        loss = torch.cat(chunks)  # [nwin, window], the batching undone
        cells = "".join(f"{loss[:, lo:hi].mean().item():10.3f}" for lo, hi in spans)
        print(f"    {label:<12}" + cells)

    print(f"  per-position val loss, window={window}, block_size={model.block_size}")
    print(
        "    "
        + "method".ljust(12)
        + "".join(f"{lo}-{hi - 1}".rjust(10) for lo, hi in spans)
    )
    model.rope_scale, model.rope_scaling = 1.0, "none"
    row("none")
    for mode in ("pi", "ntk", "yarn"):
        for s in scales:
            model.rope_scale, model.rope_scaling = s, mode
            row(f"{mode} x{s:g}")
    model.rope_scale, model.rope_scaling = was[1], was[2]
    model.train(was[0])


def run():
    torch.manual_seed(0)  # same init test() builds
    model = GPT(**asdict(small_gpt_cfg))
    model.to(device)
    torch.manual_seed(1)  # same batches evaluate(0) draws in test()
    train(model, small_train_cfg)
    # the point of the file: does stretching the base buy back the positions
    # the model never trained on? scale is an inference knob, so one trained
    # model answers for every row
    extrapolation_report(model, window=4 * model.block_size)
    prompt = torch.tensor([tok.encode("\n")], device=device)
    sample = tok.decode(
        generate(model, prompt, max_new_tokens=500, use_cache=True)[0].tolist()
    )
    print(sample)


def test():
    for kind in get_args(Attention):
        torch.manual_seed(0)
        mha_only = kind in ("flash",)
        cfg = GPTConfig(
            vocab_size=tok.vocab_size,
            block_size=32,
            n_embed=64,
            n_head=4,
            n_layer=2,
            n_kv_head=None if mha_only else 2,
            attention=kind,
        )
        model = GPT(**asdict(cfg)).to(device)
        model.eval()
        prompt = torch.tensor([tok.encode("hello")], device=device)

        # logits first: one full forward vs the same tokens fed through the cache
        # in chunks. sampled tokens can agree through a broken cache, logits
        # cannot. chunk=1 is decoding; chunk>1 is the case generate() never
        # produces -- T>1 against a live cache, where a [T, T] mask stops
        # matching [T, T_kv] scores
        with torch.no_grad():
            ids = generate(model, prompt, max_new_tokens=100)
            full = model(ids)
            for chunk in (1, 2, 5):
                caches, step = None, []
                for t in range(0, ids.size(1), chunk):
                    lg, caches = model(ids[:, t : t + chunk], caches, use_cache=True)
                    step.append(lg)
                d = (full - torch.cat(step, dim=1)).abs().max().item()
                assert d < 1e-4, (
                    f"{kind} chunk={chunk}: cached logits differ by {d:.2e}"
                )

        # then end to end: same seed in, same tokens out. several seeds, because
        # an untrained model's logits are near-uniform and one seed proves little
        for seed in range(4):
            torch.manual_seed(seed)
            a = generate(model, prompt, max_new_tokens=100)
            torch.manual_seed(seed)
            b = generate(model, prompt, max_new_tokens=100, use_cache=True)
            assert torch.equal(a, b), f"{kind} seed {seed}: cached generate diverged"

        print(f"    {kind:13} cache == no cache: logits {d:.2e}, samples identical")

    # asking an MHA-only kind for kv groups must fail loudly, not hand back MHA
    try:
        make_attention("flash", 64, 4, n_kv_head=2)
        raise SystemExit("make_attention silently ignored n_kv_head")
    except AssertionError:
        pass

    # scale=1.0 must be the plain file, bit for bit, in EVERY mode -- a default
    # that perturbs the untouched model would make every earlier number unreadable
    hs, L, scale = 16, small_gpt_cfg.block_size, 8.0
    plain = rope_angles(0, 40, hs)
    for mode in get_args(RopeScaling):
        got = rope_angles(0, 40, hs, scale=1.0, scaling=mode, train_len=L)
        assert all(torch.equal(x, y) for x, y in zip(plain, got)), (
            f"{mode} at scale=1.0 is not a no-op"
        )

    # both methods shrink theta more the slower the pair, and both pin the ends:
    # fastest pair untouched, slowest shrunk by the full scale. what differs is
    # the shape in between, which is the whole point of yarn
    f1 = rope_inv_freq(hs)
    shrink = {
        "ntk": (f1 / ntk_inv_freq(hs, 10000.0, None, scale)).tolist(),
        "yarn": (f1 / yarn_inv_freq(hs, 10000.0, None, scale, L)).tolist(),
    }
    # PI is the control: one divide for every pair, ends included. the fastest
    # pair -- the only ruler for "1 token vs 2" -- is slowed as hard as the rest
    pi = (f1 / pi_inv_freq(hs, 10000.0, None, scale)).tolist()
    assert all(abs(x - scale) < 1e-4 for x in pi), f"pi is not flat: {pi}"
    for mode, r in shrink.items():
        assert abs(r[0] - 1.0) < 0.05, f"{mode} moved the fastest pair: {r[0]:.3f}"
        assert abs(r[-1] - scale) / scale < 0.02, f"{mode} slowest: {r[-1]:.3f}"
        assert all(x <= y + 1e-6 for x, y in zip(r, r[1:])), f"{mode} not monotone"

    turns = (L * f1 / (2 * math.pi)).tolist()  # what yarn decides on
    # yarn's claim over ntk: a pair training turned more than beta times is left
    # EXACTLY alone, not merely nudged. ntk has no such flat region
    kept = [i for i, t in enumerate(turns) if t > 32.0]
    for i in kept:
        assert abs(shrink["yarn"][i] - 1.0) < 1e-5, (
            f"yarn moved pair {i} despite {turns[i]:.1f} turns in training"
        )
    # pair 0 is pinned for ntk too -- its exponent is zero, so no base can move
    # it. the contrast only exists on the pairs after it
    contrast = [i for i in kept if i > 0]
    assert contrast, (
        "no pair past 0 clears beta, so ntk and yarn cannot be told apart here; "
        "train longer or raise head_size"
    )
    for i in contrast:
        assert shrink["ntk"][i] > 1.0 + 1e-3, "ntk unexpectedly left a pair exact"

    # the property none of this may break: a score still depends on m - n alone.
    # R(m)^T R(n) = R(n - m) holds for any theta, so per-pair scaling is fine,
    # and so is a constant on cos and sin -- it factors straight out of the score
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 1, hs), torch.randn(1, 1, 1, hs)
    for mode in ("pi", "ntk", "yarn"):
        cos, sin = rope_angles(0, 300, hs, scale=scale, scaling=mode, train_len=L)
        scores = []
        for mm, nn in [(20, 5), (100, 85), (295, 280)]:  # all 15 apart
            qm = apply_rope(q, cos[mm : mm + 1], sin[mm : mm + 1])
            kn = apply_rope(k, cos[nn : nn + 1], sin[nn : nn + 1])
            scores.append((qm * kn).sum().item())
        spread = max(scores) - min(scores)
        assert spread < 1e-4, f"{mode} broke relative positions: {spread:.2e}"

    # the temperature is all yarn adds outside the frequencies, and it costs the
    # rotation its norm: every vector comes out longer by exactly that factor
    t = yarn_attn_scale(scale)
    cos, sin = rope_angles(0, 8, hs, scale=scale, scaling="yarn", train_len=L)
    x = torch.randn(1, 1, 8, hs)
    grew = (apply_rope(x, cos, sin).norm(dim=-1) / x.norm(dim=-1)).flatten()
    assert (grew - t).abs().max() < 1e-5, "yarn temperature is not a clean rescale"

    print(f"    turns made in training: {[f'{x:.2f}' for x in turns]}")
    print(f"    pi   theta shrink x{scale:g}: {[f'{x:.2f}' for x in pi]}")
    print(f"    ntk  theta shrink x{scale:g}: {[f'{x:.2f}' for x in shrink['ntk']]}")
    print(f"    yarn theta shrink x{scale:g}: {[f'{x:.2f}' for x in shrink['yarn']]}")
    print(f"    yarn leaves pairs {kept} exact, attn scale {t:.3f}")

    print("✅ all tests ok")


if __name__ == "__main__":
    test()
    run()
