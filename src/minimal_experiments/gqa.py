from dataclasses import asdict, dataclass
from pathlib import Path

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


@dataclass(frozen=True, kw_only=True)
class GPTConfig:
    vocab_size: int  # V
    block_size: int  # T
    n_embed: int  # E
    n_head: int  # nh
    n_layer: int
    n_kv_head: int | None = None


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    batch_size: int  # B
    max_steps: int
    lr: float
    eval_interval: int
    eval_iters: int


class ResidualProj(nn.Linear):
    """Linear whose output is added to the residual stream; gets 1/sqrt(2*n_layer) init."""


class FlashAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = ResidualProj(n_embed, n_embed)

    def forward(self, x: Tensor):
        nh = self.n_head
        qkv = self.qkv(x)  # [B, T, 3E]
        qkv = rearrange(qkv, "b t (three e) -> three b t e", three=3)
        q, k, v = [rearrange(t, "b t (nh hs) -> b nh t hs", nh=nh) for t in qkv]
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )  # [B, nh, T, hs]
        out = rearrange(out, "b nh t hs -> b t (nh hs)")  # [B, T, E]
        out = self.proj(out)
        return out


class FusedQKVAttention(nn.Module):
    tril: Tensor

    def __init__(self, n_embed: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = ResidualProj(n_embed, n_embed)
        self.register_buffer(
            "tril",
            torch.ones(block_size, block_size, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, x: Tensor):
        B, T, E = x.shape
        nh, hs = self.n_head, self.head_size
        qkv = self.qkv(x)  # [B, T, 3E]
        qkv = rearrange(qkv, "b t (three e) -> three b t e", three=3)
        q, k, v = [rearrange(t, "b t (nh hs) -> b nh t hs", nh=nh) for t in qkv]
        scores = q @ rearrange(k, "b nh t hs -> b nh hs t") * hs**-0.5  # [B, nh, T, T]
        scores = scores.masked_fill(~self.tril[:T, :T], float("-inf"))
        w = F.softmax(scores, dim=-1)
        out = w @ v  # [B, nh, T, hs]
        out = rearrange(out, "b nh t hs -> b t (nh hs)")  # [B, T, E]
        out = self.proj(out)
        return out


class GQAttentionRepeatInterleave(nn.Module):
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

    def forward(self, x: Tensor):
        B, T, E = x.shape
        nkv, hs = self.n_kv_head, self.head_size
        qkv: Tensor = self.qkv(x)
        q, k, v = qkv.split([E, nkv * hs, nkv * hs], dim=-1)
        q, k, v = [rearrange(t, "b t (n hs) -> b n t hs", hs=hs) for t in [q, k, v]]
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )  # [B, nh, T, hs]
        out = rearrange(out, "b nh t hs -> b t (nh hs)")  # [B, T, E]
        out = self.proj(out)
        return out


class GQAttentionFusedRepeatInterleave(nn.Module):
    tril: Tensor

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        block_size: int,
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
        self.register_buffer(
            "tril",
            torch.ones(block_size, block_size, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, x: Tensor):
        B, T, E = x.shape
        nkv, hs = self.n_kv_head, self.head_size
        qkv: Tensor = self.qkv(x)
        q, k, v = qkv.split([E, nkv * hs, nkv * hs], dim=-1)
        q, k, v = [rearrange(t, "b t (n hs) -> b n t hs", hs=hs) for t in [q, k, v]]
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        scores = q @ rearrange(k, "b nh t hs -> b nh hs t") * hs**-0.5  # [B, nh, T, T]
        scores = scores.masked_fill(~self.tril[:T, :T], float("-inf"))
        w = F.softmax(scores, dim=-1)
        out = w @ v  # [B, nh, T, hs]
        out = rearrange(out, "b nh t hs -> b t (nh hs)")  # [B, T, E]
        out = self.proj(out)
        return out


class GQAttentionFused(nn.Module):
    tril: Tensor

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        block_size: int,
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
        self.register_buffer(
            "tril",
            torch.ones(block_size, block_size, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, x: Tensor):
        B, T, E = x.shape
        nh, nkv, hs = self.n_head, self.n_kv_head, self.head_size
        qkv: Tensor = self.qkv(x)
        q, k, v = qkv.split([E, nkv * hs, nkv * hs], dim=-1)
        q, k, v = [rearrange(t, "b t (n hs) -> b n t hs", hs=hs) for t in [q, k, v]]
        q = q.reshape(B, nkv, self.n_rep * T, hs)  # [B, nkv, n_rep*T, hs]
        scores = (
            q @ rearrange(k, "b nkv t hs -> b nkv hs t") * hs**-0.5
        )  # [B, nkv, n_rep*T, T]
        causal = self.tril[:T, :T].repeat(self.n_rep, 1)  # [n_rep*T, T]
        scores = scores.masked_fill(~causal, float("-inf"))
        w = F.softmax(scores, dim=-1)
        out = w @ v  # [B, nkv, n_rep*T, hs]
        out = out.view(B, nh, T, hs)
        out = rearrange(out, "b nh t hs -> b t (nh hs)")  # [B, T, E]
        out = self.proj(out)
        return out


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

    def forward(self, x: Tensor):
        B, T, E = x.shape
        nkv, hs = self.n_kv_head, self.head_size
        qkv: Tensor = self.qkv(x)
        q, k, v = qkv.split([E, nkv * hs, nkv * hs], dim=-1)
        q, k, v = [rearrange(t, "b t (n hs) -> b n t hs", hs=hs) for t in [q, k, v]]
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            enable_gqa=self.n_rep > 1,
        )  # [B, nh, T, hs]
        out = rearrange(out, "b nh t hs -> b t (nh hs)")  # [B, T, E]
        out = self.proj(out)
        return out


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


class Block(nn.Module):
    def __init__(
        self,
        n_embed: int,
        n_head: int,
        n_kv_head: int | None = None,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.attn = GQAttention(n_embed, n_head, n_kv_head)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x: Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embed: int,
        n_head: int,
        n_layer: int,
        n_kv_head: int | None = None,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head, n_kv_head) for _ in range(n_layer)]
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

    def forward(self, idx: Tensor):
        _, T = idx.shape
        x = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device)
        x = x + self.position_embedding_table(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


@torch.no_grad()
def generate(
    model: GPT,
    idx: Tensor,
    max_new_tokens: int,
):
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
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
)
small_train_cfg = TrainConfig(
    batch_size=32,
    max_steps=500,
    lr=1e-2,
    eval_interval=100,
    eval_iters=100,
)


def run():
    model = GPT(**asdict(small_gpt_cfg))
    model.to(device)
    train(model, small_train_cfg)
    prompt = torch.tensor([tok.encode("\n")], device=device)
    sample = tok.decode(generate(model, prompt, max_new_tokens=100)[0].tolist())
    print(sample)


def compare(
    a: nn.Module, b: nn.Module, x: Tensor, tol: float = 1e-6, load: bool = True
):
    """b takes a's weights; outputs and every grad must match.

    load=False keeps b's own weights, for deliberately-different negative controls.
    """
    ka, kb = a.state_dict().keys(), b.state_dict().keys()
    assert ka == kb, f"different params: {set(ka) ^ set(kb)}"
    if load:
        b.load_state_dict(a.state_dict())
    a.zero_grad()  # compare() may be called twice on one module
    b.zero_grad()

    xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    ya, yb = a(xa), b(xb)
    w = torch.randn_like(ya)  # random weighting, so a sign flip can't cancel
    (ya * w).sum().backward()
    (yb * w).sum().backward()

    diffs = {
        "out": (ya - yb).abs().max().item(),
        "x.grad": (xa.grad - xb.grad).abs().max().item(),
    }
    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        diffs[f"{name}.grad"] = (pa.grad - pb.grad).abs().max().item()

    for k, d in diffs.items():
        print(f"    {k:>12}  {d:.2e}")
    bad = {k: d for k, d in diffs.items() if not d <= tol}  # not (<=) also traps nan
    assert not bad, f"above tol {tol}: {bad}"


def test():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)  # float32 noise is as big as the tolerance
    B, T, E, NH, BS = 2, 8, 32, 4, 16  # BS > T, so tril[:T, :T] actually slices
    x = torch.randn(B, T, E)

    def impls(nkv):
        """Everything that can hold the same weights at this n_kv_head, torch's own
        causal kernel first -- matching it is what proves the rest are causal."""
        if nkv == NH:  # only here is qkv square, so the MHA-only pair fits
            yield FlashAttention(E, NH)
            yield FusedQKVAttention(E, NH, BS)
        yield GQAttention(E, NH, nkv)
        yield GQAttentionRepeatInterleave(E, NH, nkv)
        yield GQAttentionFused(E, NH, BS, nkv)
        yield GQAttentionFusedRepeatInterleave(E, NH, BS, nkv)

    for nkv in (NH, 2):  # n_rep 1 and 2: the fused reshape is an identity at 1
        ref, *rest = impls(nkv)
        for b in rest:
            print(f"n_kv_head={nkv}  {type(b).__name__} vs {type(ref).__name__}")
            compare(ref, b, x, tol=1e-12)

    # negative control: perturb AFTER the load, then tell compare not to reload
    print("perturbed (expect failure)")
    ref = FlashAttention(E, NH)
    bad = GQAttention(E, NH, NH)
    bad.load_state_dict(ref.state_dict())
    with torch.no_grad():
        bad.qkv.weight[0, 0] += 1e-3
    try:
        compare(ref, bad, x, load=False)
        raise SystemExit("compare() missed a real difference")
    except AssertionError:
        print("    caught")

    torch.set_default_dtype(torch.float32)
    print("✅ all tests ok")


if __name__ == "__main__":
    test()
    run()
