import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Literal
from utils import repo_root
from einops import rearrange
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype

ROOT = repo_root()
DATA = ROOT / "data" / "input.txt"
text = DATA.read_text(encoding="utf-8")
itoc = sorted(set(text))
ctoi = {c: i for i, c in enumerate(itoc)}


def encode(s: str):
    return [ctoi[c] for c in s]


def decode(ids: list[int]):
    return "".join(itoc[i] for i in ids)


data = torch.tensor(encode(text))
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

block_size = 8  # t
batch_size = 32  # b


@jaxtyped(typechecker=beartype)
def get_batch(
    split: Literal["train", "val"],
) -> tuple[Int[Tensor, "b t"], Int[Tensor, "b t"]]:
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


vocab_size = len(itoc)  # v


class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        idx: Int[Tensor, "b t"],
        targets: Int[Tensor, "b t"] | None = None,
    ) -> tuple[Float[Tensor, "b t v"], Float[Tensor, ""] | None]:
        logits = self.token_embedding_table(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                rearrange(logits, "b t v -> (b t) v"),
                rearrange(targets, "b t -> (b t)"),
            )
        return logits, loss

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def generate(
        self, idx: Int[Tensor, "b t"], max_new_tokens: int
    ) -> Int[Tensor, "b t_out"]:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)  # (b, v)
            nxt = torch.multinomial(probs, num_samples=1)  # (b, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


model = BigramLM()
lr = 1e-2
opt = torch.optim.AdamW(model.parameters(), lr=lr)
max_steps = 1000
for it in range(max_steps):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % 100 == 0 or it == max_steps - 1:
        print(f"it: {it}, loss: {loss.item()}")

start = torch.tensor([encode("\n")])
sample = decode(model.generate(start, max_new_tokens=500)[0].tolist())
print(sample)
