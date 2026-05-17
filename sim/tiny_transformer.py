"""Minimal self-contained decoder-only GPT (PyTorch). Generator-F
language model. Self-contained at runtime: the artifact is the
state_dict (.pt) + a sidecar hyperparam JSON; zero external
dependency, no external LLM, no runtime corpus. ASCII only."""
from __future__ import annotations
import json
import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):
        n = x.size(1)
        # causal mask: True == NOT allowed to attend (j > i masked).
        mask = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=x.device),
            diagonal=1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask,
                         need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layer=4, n_head=4,
                 block_size=128, dropout=0.0):
        super().__init__()
        self.cfg = {"vocab_size": int(vocab_size),
                    "d_model": int(d_model), "n_layer": int(n_layer),
                    "n_head": int(n_head),
                    "block_size": int(block_size),
                    "dropout": float(dropout)}
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_Block(d_model, n_head, dropout)
             for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        n = idx.size(1)
        if n > self.cfg["block_size"]:
            raise ValueError(
                "sequence length %d exceeds block_size %d "
                "(positional embedding would be out of range)"
                % (n, self.cfg["block_size"]))
        pos = torch.arange(n, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None, :, :])
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path + ".pt")
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(self.cfg, f)

    @classmethod
    def load(cls, path: str) -> "TinyGPT":
        with open(path + ".meta.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        m = cls(**cfg)
        m.load_state_dict(torch.load(path + ".pt",
                                     map_location="cpu"))
        return m
