"""CHUNKED PARALLEL SCAN for the WKV/SSM recurrence (training-throughput lever, 2026-07-21).

The SSM/WKV block's recurrence (`_lmtrain_optim_probe.py` Block, and `_emerge_wkv_lm_derisk.py --recurrence ssm`)
is a Python loop over T:  a_t = lam*a_{t-1} + (1-lam)*v_t  (lam = sigmoid(decay) per-channel in (0,1)),
then y_t = r_t * Wo(a_t). At T=256 that loop launches T sequential kernels/block AND torch.compile HANGS
unrolling it. This module implements a CHUNKED parallel scan that is NUMERICALLY IDENTICAL to the loop (the
diagonal decay parallelizes within a chunk) so the state work becomes T/C sequential steps of a parallel
matmul, and the shortened Python loop lets torch.compile succeed.

Math (chunk length C, a_prev = state at end of the previous chunk, zeros for chunk 0; local t,s in [0,C-1]):
    a_t = lam^(t+1) * a_prev  +  sum_{s<=t} lam^(t-s) * (1-lam) * v_s
        = carry_term          +  einsum('tsd,bsd->btd', L, v_chunk)
  where L[t,s,d] = lam_d^(t-s) * (1-lam_d) for s<=t else 0.
New carry (end of chunk) = a_{C-1}. The scan runs in fp32 (decay powers underflow-guarded); matmuls are
bf16-friendly for the forward.

NO `sim/` edit -- pure torch/runner.

  Correctness gate:  python -m research.runners._lmtrain_chunked_scan gate
  Speed (one variant): python -m research.runners._lmtrain_chunked_scan speed --variant {loop,chunked,chunked-compile} [--baseline]
"""
from __future__ import annotations
import argparse, time
import torch
import torch.nn as nn


# ------------------------------------------------------------------ the scans ----------------------------------------
def loop_ssm(v: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """REFERENCE loop (byte-for-byte the probe's Block recurrence). v:[B,T,D], lam:[D]. Returns a:[B,T,D]."""
    B, T, D = v.shape
    a = torch.zeros(B, D, device=v.device, dtype=v.dtype)
    outs = []
    for t in range(T):
        a = lam * a + (1 - lam) * v[:, t]
        outs.append(a)
    return torch.stack(outs, 1)


def chunked_ssm(v: torch.Tensor, lam: torch.Tensor, C: int = 16) -> torch.Tensor:
    """CHUNKED parallel scan == loop_ssm. v:[B,T,D], lam:[D] in (0,1), chunk length C. Returns a:[B,T,D].

    The scan interior is done in fp32 (decay powers can underflow for small lam; that is harmless -- the loop
    zeros those far contributions too). Output cast back to v.dtype."""
    B, T, D = v.shape
    dev, in_dtype = v.device, v.dtype
    vf = v.float()
    lamf = lam.float().clamp(1e-6, 1.0 - 1e-7)          # guard the ends of (0,1)
    one_m = 1.0 - lamf                                    # [D]

    # pad T up to a multiple of C
    pad = (-T) % C
    if pad:
        vf = torch.cat([vf, vf.new_zeros(B, pad, D)], dim=1)
    Tp = T + pad
    n_chunks = Tp // C
    vc = vf.view(B, n_chunks, C, D)                       # [B, nchunks, C, D]

    # intra-chunk decay matrix  L[t,s,d] = lam^(t-s) * (1-lam)  for s<=t else 0     -> [C,C,D]
    ar = torch.arange(C, device=dev)
    expo = ar.view(C, 1) - ar.view(1, C)                  # t - s   [C,C]
    tri = (expo >= 0)                                     # lower-triangular (incl diag)
    # lam^(t-s): compute in log space where valid, exp back; zero the strictly-upper triangle
    log_lam = torch.log(lamf)                             # [D]  (<0)
    expo_cl = expo.clamp(min=0).float()                   # [C,C]
    L = torch.exp(expo_cl.view(C, C, 1) * log_lam.view(1, 1, D)) * one_m.view(1, 1, D)
    L = L * tri.view(C, C, 1).to(L.dtype)                 # mask s>t to 0
    # carry powers  lam^(t+1), t in [0,C-1]                                          -> [C,D]
    tpow = (ar + 1).float()                               # [C]
    carry_pow = torch.exp(tpow.view(C, 1) * log_lam.view(1, D))   # [C,D]

    # move D into the batch dim so the intra-chunk term is one efficient bmm per chunk:
    #   out[b,t,d] = sum_s L[t,s,d] * v[b,s,d]  ==  bmm( L[d,t,s], v[d,s,b] ) -> [d,t,b]
    Ld = L.permute(2, 0, 1).contiguous()                 # [D,C,C]

    a_prev = vf.new_zeros(B, D)                           # fp32 carry
    outs = []
    for c in range(n_chunks):
        vchunk = vc[:, c]                                 # [B,C,D]
        vd = vchunk.permute(2, 1, 0)                      # [D,C,B]
        y_intra = torch.bmm(Ld, vd).permute(2, 1, 0)      # [B,C,D]
        y_carry = carry_pow.view(1, C, D) * a_prev.view(B, 1, D)
        a_chunk = y_intra + y_carry                       # [B,C,D]
        outs.append(a_chunk)
        a_prev = a_chunk[:, -1]                           # new carry = a_{C-1}
    a = torch.cat(outs, dim=1)[:, :T]                     # [B,T,D]
    return a.to(in_dtype)


# ------------------------------------------------------------------ blocks / model ------------------------------------
class LoopBlock(nn.Module):
    """== `_lmtrain_optim_probe.py` Block (the naive Python-loop recurrence)."""
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


class ChunkedBlock(nn.Module):
    """Same params/output as LoopBlock, recurrence via chunked_ssm."""
    def __init__(self, D, C=16):
        super().__init__()
        self.ln = nn.LayerNorm(D); self.Wv = nn.Linear(D, D, bias=False)
        self.Wr = nn.Linear(D, D, bias=False); self.Wo = nn.Linear(D, D, bias=False)
        self.decay = nn.Parameter(torch.zeros(D))
        self.C = C

    def forward(self, x):
        h = self.ln(x); v = self.Wv(h); r = torch.sigmoid(self.Wr(h)); lam = torch.sigmoid(self.decay)
        a = chunked_ssm(v, lam, self.C)
        return x + r * self.Wo(a)


class WKV(nn.Module):
    def __init__(self, V, D, L, block="loop", C=16):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        mk = (lambda: LoopBlock(D)) if block == "loop" else (lambda: ChunkedBlock(D, C))
        self.blocks = nn.ModuleList([mk() for _ in range(L)])
        self.lnf = nn.LayerNorm(D); self.head = nn.Linear(D, V)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.lnf(h))


# ------------------------------------------------------------------ correctness gate ----------------------------------
def correctness_gate():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CORRECTNESS GATE (device={dev}): chunked_ssm vs loop_ssm, fp32, gate = max|diff| < 1e-4")
    B, T, D = 4, 256, 256
    worst = 0.0
    passed = True
    for seed in (0, 1, 2, 42):
        torch.manual_seed(seed)
        v = torch.randn(B, T, D, device=dev)
        # lam in (0.01, 0.99), per-channel
        lam = 0.01 + 0.98 * torch.rand(D, device=dev)
        a_loop = loop_ssm(v, lam)
        for C in (8, 16, 32):
            a_ch = chunked_ssm(v, lam, C)
            diff = (a_ch - a_loop).abs().max().item()
            worst = max(worst, diff)
            ok = diff < 1e-4
            passed = passed and ok
            print(f"  seed={seed:>2}  C={C:>2}   max|chunked-loop| = {diff:.3e}   {'PASS' if ok else 'FAIL'}")
    # also assert the block-level output matches (params shared)
    torch.manual_seed(7)
    D2 = 128
    lb = LoopBlock(D2).to(dev); cb = ChunkedBlock(D2, 16).to(dev)
    cb.load_state_dict(lb.state_dict())
    x = torch.randn(4, 256, D2, device=dev)
    with torch.no_grad():
        bdiff = (lb(x) - cb(x)).abs().max().item()
    print(f"  block-level output max|loop-chunked| = {bdiff:.3e}   {'PASS' if bdiff < 1e-4 else 'FAIL'}")
    passed = passed and bdiff < 1e-4
    print(f"==> GATE {'PASSED' if passed else 'FAILED'}   worst state diff = {worst:.3e}")
    return passed


# ------------------------------------------------------------------ speed benchmark -----------------------------------
def bench(variant, V=8000, D=1024, L=16, B=64, T=256, C=16, amp=True, compile_=False, steps=6):
    dev = "cuda"
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    block = "loop" if variant == "loop" else "chunked"
    m = WKV(V, D, L, block=block, C=C).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
    if compile_:
        m = torch.compile(m)
    lossf = nn.CrossEntropyLoss()

    def step():
        x = torch.randint(0, V, (B, T), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            y = m(x)[:, :-1]
            loss = lossf(y.reshape(-1, V).float(), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

    t_warm = time.time()
    for _ in range(3):
        step()                                          # warmup (compile happens here)
    torch.cuda.synchronize(); warm = time.time() - t_warm
    t0 = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize(); dt = (time.time() - t0) / steps
    tps = B * T / dt
    vram = torch.cuda.max_memory_allocated() / 1e9
    label = f"{variant}{'+compile' if compile_ else ''} (B={B},amp={amp})"
    print(f"RESULT {label:34} tok/s={tps:>8.0f}  VRAM={vram:>5.2f}GB  warmup={warm:.0f}s")
    return tps


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gate")
    sp = sub.add_parser("speed")
    sp.add_argument("--variant", choices=["loop", "chunked", "chunked-compile", "baseline"], required=True)
    sp.add_argument("--B", type=int, default=64)
    sp.add_argument("--C", type=int, default=16)
    sp.add_argument("--amp", type=int, default=1)
    args = ap.parse_args()

    if args.cmd == "gate":
        ok = correctness_gate()
        raise SystemExit(0 if ok else 1)

    if args.cmd == "speed":
        if args.variant == "baseline":
            bench("loop", B=16, amp=False)              # naive fp32 B=16 loop (the 3072 tok/s reference)
        elif args.variant == "loop":
            bench("loop", B=args.B, amp=bool(args.amp))
        elif args.variant == "chunked":
            bench("chunked", B=args.B, C=args.C, amp=bool(args.amp))
        elif args.variant == "chunked-compile":
            bench("chunked", B=args.B, C=args.C, amp=bool(args.amp), compile_=True)


if __name__ == "__main__":
    main()
