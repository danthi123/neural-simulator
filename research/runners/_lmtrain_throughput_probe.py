"""Throughput + VRAM de-risk: real 3090 tokens/sec for the multi-layer WKV/SSM at several sizes -> firms the
compute->quality budgets + the VRAM ceiling. Pre-norm residual stack of diagonal-decay SSM blocks (the real arch)."""
import time, torch, torch.nn as nn

dev = "cuda" if torch.cuda.is_available() else "cpu"

class Block(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.ln = nn.LayerNorm(D)
        self.Wv = nn.Linear(D, D, bias=False); self.Wr = nn.Linear(D, D, bias=False); self.Wo = nn.Linear(D, D, bias=False)
        self.decay = nn.Parameter(torch.zeros(D))
    def forward(self, x):
        h = self.ln(x); v = self.Wv(h); r = torch.sigmoid(self.Wr(h)); lam = torch.sigmoid(self.decay)
        a = torch.zeros(x.shape[0], x.shape[-1], device=x.device); outs = []
        for t in range(x.shape[1]):
            a = lam * a + (1 - lam) * v[:, t]; outs.append(r[:, t] * self.Wo(a))
        return x + torch.stack(outs, 1)

class WKV(nn.Module):
    def __init__(self, V, D, L):
        super().__init__()
        self.emb = nn.Embedding(V, D); self.blocks = nn.ModuleList([Block(D) for _ in range(L)])
        self.lnf = nn.LayerNorm(D); self.head = nn.Linear(D, V)
    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks: h = b(h)
        return self.head(self.lnf(h))

def probe(V, D, L, B, T, steps=4):
    torch.cuda.empty_cache() if dev == "cuda" else None
    if dev == "cuda": torch.cuda.reset_peak_memory_stats()
    m = WKV(V, D, L).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    P = sum(p.numel() for p in m.parameters())
    lossf = nn.CrossEntropyLoss()
    # warmup
    for _ in range(2):
        x = torch.randint(0, V, (B, T), device=dev)
        y = m(x)[:, :-1]; loss = lossf(y.reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        x = torch.randint(0, V, (B, T), device=dev)
        y = m(x)[:, :-1]; loss = lossf(y.reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    if dev == "cuda": torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    toks = B * T
    vram = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
    return P, toks / dt, vram

print(f"device={dev}  (B=16, T=256 -> 4096 tok/step)")
print(f"{'d_model':>7} {'layers':>6} {'params':>10} {'tok/s':>9} {'VRAM_GB':>8}  {'1day(B tok)':>11} {'1wk':>6} {'1mo':>6}")
for D, L in [(768,12),(1024,16),(1280,20),(1536,24)]:
    try:
        P, tps, vram = probe(8000, D, L, 16, 256)
        d1 = tps*86400/1e9; print(f"{D:>7} {L:>6} {P/1e6:>9.1f}M {tps:>9.0f} {vram:>8.2f}  {d1:>11.2f} {d1*7:>6.1f} {d1*30:>6.0f}")
    except RuntimeError as e:
        print(f"{D:>7} {L:>6}  OOM/err: {str(e)[:50]}")
