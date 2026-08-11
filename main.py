from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

DATA = Path(__file__).parent / "data" / "input.txt"
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

block_size = 8  # T
batch_size = 32  # B


def get_batch(split: Literal["train", "val"]):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


vocab_size = len(itoc)  # V


class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        logits = self.token_embedding_table(idx)  # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        # idx (B, T)
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)  # (B, V)
            nxt = torch.multinomial(probs, num_samples=1)  # (B, 1)
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
