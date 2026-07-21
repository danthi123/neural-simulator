"""Optimization de-risk: how much can we reasonably speed up WKV/SSM training on the 3090? Measures the easy,
low-risk levers (bf16 mixed precision, batch-scaling into the free VRAM, torch.compile) at the ~67M 1-week-sweet-spot
config. The Python-loop recurrence is the known bottleneck; the chunked/parallel scan (bigger algorithmic win) is
flagged separately."""
import time, torch, torch.nn as nn
dev = "cuda"

class Block(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.ln = nn.LayerNorm(D); self.Wv = nn.Linear(D, D, bias=False)
        self.Wr = nn.Linear(D, D, bias=False); self.Wo = nn.Linear(D, D, bias=False)
        self.decay = nn.Parameter(torch.zeros(D))
    def forward(self, x):
        h = self.ln(x); v = self.Wv(h); r = torch.sigmoid(self.Wr(h)); lam = torch.sigmoid(self.decay)
        a = torch.zeros(x.shape[0], x.shape[-1], device=x.device, dtype=x.dtype); outs = []
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

def bench(V, D, L, B, T, amp=False, compile_=False, steps=4, label=""):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = WKV(V, D, L).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
    if compile_: m = torch.compile(m)
    lossf = nn.CrossEntropyLoss()
    def step():
        x = torch.randint(0, V, (B, T), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            y = m(x)[:, :-1]; loss = lossf(y.reshape(-1, V).float(), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    t_warm = time.time()
    for _ in range(3): step()          # warmup (compile happens here)
    torch.cuda.synchronize(); warm = time.time() - t_warm
    t0 = time.time()
    for _ in range(steps): step()
    torch.cuda.synchronize(); dt = (time.time() - t0) / steps
    tps = B * T / dt; vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {label:38} tok/s={tps:>7.0f}  VRAM={vram:>5.2f}GB  (warmup {warm:.0f}s)")
    return tps

V, D, L, T = 8000, 1024, 16, 256
print(f"~67M WKV (d={D}, L={L}, T={T}) on 3090 -- optimization levers:")
base = bench(V, D, L, 16, T, label="fp32, B=16 (baseline)")
b1 = bench(V, D, L, 16, T, amp=True, label="+ bf16, B=16")
b2 = bench(V, D, L, 64, T, amp=True, label="+ bf16, B=64 (fill VRAM)")
b3 = bench(V, D, L, 128, T, amp=True, label="+ bf16, B=128")
try:
    b4 = bench(V, D, L, 64, T, amp=True, compile_=True, label="+ bf16, B=64, torch.compile")
    print(f"  ==> best easy-win speedup vs baseline: {max(b2,b3,b4)/base:.1f}x")
except Exception as e:
    print(f"  torch.compile: {str(e)[:60]} (skipped)")
    print(f"  ==> best easy-win speedup vs baseline: {max(b2,b3)/base:.1f}x")
