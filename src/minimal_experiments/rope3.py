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


@dataclass(frozen=True, kw_only=True)
class GPTConfig:
    vocab_size: int  # V
    block_size: int  # T
    n_embed: int  # E
    n_head: int  # nh
    n_layer: int
    n_kv_head: int | None = None
    attention: Attention = "flash"


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    batch_size: int  # B
    max_steps: int
    lr: float
    eval_interval: int
    eval_iters: int


class ResidualProj(nn.Linear):
    """Linear whose output is added to the residual stream; gets 1/sqrt(2*n_layer) init."""


def rope_inv_freq(head_size: int, base: float = 10000.0) -> Tensor:
    assert head_size % 2 == 0, "head_size must be even: the dims rotate in pairs"
    i = torch.arange(0, head_size, 2, dtype=torch.float32)
    return base ** (-i / head_size)  # [hs/2]


def rope_tables(block_size: int, head_size: int, base: float = 10000.0):
    inv_freq = rope_inv_freq(head_size, base)
    pos = torch.arange(block_size, dtype=torch.float32)
    angles = pos.unsqueeze(1) * inv_freq  # [T, hs/2]
    return angles.cos(), angles.sin()


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
    rope_cos: Tensor
    rope_sin: Tensor

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int,
        n_head: int,
        n_layer: int,
        n_kv_head: int | None = None,
        attention: Attention = "gqa",
    ):
        super().__init__()
        self.n_layer = n_layer
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # one copy for the whole stack. handing the same tensor to every layer's
        # register_buffer would NOT work: .to() rebinds each buffer separately
        cos, sin = rope_tables(block_size, n_embed // n_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

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
        assert T_past + T <= self.block_size, (
            f"sequence {T_past + T} outgrew block_size"
        )
        x = self.token_embedding_table(idx)
        cos, sin = (
            self.rope_cos[T_past : T_past + T],
            self.rope_sin[T_past : T_past + T],
        )
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
    if use_cache:
        fed = idx.size(1) + max_new_tokens - 1  # the last token is never fed back
        assert fed <= model.block_size, (
            f"use_cache needs prompt+max_new_tokens-1 ({idx.size(1)}+"
            f"{max_new_tokens}-1={fed}) <= block_size ({model.block_size}). "
            f"Use use_cache=False to crop+recompute."
        )
    was_training = model.training
    model.eval()
    kv_caches = None
    for _ in range(max_new_tokens):
        if use_cache:
            step = idx if kv_caches is None else idx[:, -1:]
            logits, kv_caches = model(step, kv_caches, use_cache=True)
        else:
            logits = model(idx[:, -model.block_size :])
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


small_gpt_cfg = GPTConfig(
    vocab_size=tok.vocab_size,
    block_size=32,
    n_embed=64,
    n_head=16,
    n_layer=3,
    n_kv_head=4,
    attention="gqa",
)
small_train_cfg = TrainConfig(
    batch_size=32,
    max_steps=500,
    lr=1e-2,
    eval_interval=100,
    eval_iters=100,
)


def run():
    torch.manual_seed(0)  # same init test() builds
    model = GPT(**asdict(small_gpt_cfg))
    model.to(device)
    torch.manual_seed(1)  # same batches evaluate(0) draws in test()
    train(model, small_train_cfg)
    prompt = torch.tensor([tok.encode("\n")], device=device)
    sample = tok.decode(generate(model, prompt, max_new_tokens=100)[0].tolist())
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
            ids = generate(model, prompt, max_new_tokens=20)
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
            a = generate(model, prompt, max_new_tokens=25)
            torch.manual_seed(seed)
            b = generate(model, prompt, max_new_tokens=25, use_cache=True)
            assert torch.equal(a, b), f"{kind} seed {seed}: cached generate diverged"

        print(f"    {kind:13} cache == no cache: logits {d:.2e}, samples identical")

    # asking an MHA-only kind for kv groups must fail loudly, not hand back MHA
    try:
        make_attention("flash", 64, 4, n_kv_head=2)
        raise SystemExit("make_attention silently ignored n_kv_head")
    except AssertionError:
        pass

    print("✅ all tests ok")


if __name__ == "__main__":
    test()
    run()
