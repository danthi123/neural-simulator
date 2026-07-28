"""torch.compile correctness smoke for the WKV LM trainer — does --compile 1 work with this architecture + preserve the
loss? (The real speedup is measured on the FLIP of the full run; a tiny model's compile speedup is NOT representative.)
Tiny model, few steps, GPU. Runs CONCURRENT with the real training (shares the 300W cap = thermally safe, just slower).
"""
import os, sys, time
import torch
from research.runners._lmtrain_chunked_scan import WKV as ChunkedWKV

DEV = "cuda"
V, D, L, C, T, B = 16000, 256, 4, 16, 256, 8
N_WARM, N_TIMED = 6, 15


def _build():
    torch.manual_seed(0)
    return ChunkedWKV(V, D, L, block="chunked", C=C).to(DEV)


def _run(compile_on):
    m = _build()
    if compile_on:
        m = torch.compile(m)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator(device=DEV).manual_seed(1)   # SAME data stream both arms
    losses = []
    for _ in range(N_WARM):                          # warmup (compile happens on the first step)
        x = torch.randint(0, V, (B, T), device=DEV, generator=g)
        y = m(x)
        loss = lossf(y[:, :-1].reshape(-1, V).float(), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(N_TIMED):
        x = torch.randint(0, V, (B, T), device=DEV, generator=g)
        y = m(x)
        loss = lossf(y[:, :-1].reshape(-1, V).float(), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    return losses, (time.time() - t0) / N_TIMED


def main():
    print(f"[compile-smoke] torch {torch.__version__} device={torch.cuda.get_device_name(0)}")
    try:
        l0, dt0 = _run(False)
        print(f"  eager   : loss[0]={l0[0]:.4f} loss[-1]={l0[-1]:.4f}  {dt0*1000:.1f} ms/step")
    except Exception as e:
        print(f"  eager FAILED: {e}"); sys.exit(1)
    try:
        l1, dt1 = _run(True)
        print(f"  compiled: loss[0]={l1[0]:.4f} loss[-1]={l1[-1]:.4f}  {dt1*1000:.1f} ms/step")
    except Exception as e:
        print(f"  !!! torch.compile FAILED on WKV: {type(e).__name__}: {e}")
        print("  -> --compile is NOT safe to flip; investigate the graph break before touching the real run.")
        sys.exit(2)
    # correctness: same seed + same data -> losses should match within fp/compile tolerance
    dl = max(abs(a - b) for a, b in zip(l0, l1))
    ok = dl < 0.05
    print(f"  max|eager-compiled| loss diff = {dl:.4f}  ({'OK: compile preserves the loss' if ok else 'DIVERGENT'})")
    print(f"  [tiny-model step time eager {dt0*1000:.1f}ms vs compiled {dt1*1000:.1f}ms -- NOT the real speedup; "
          f"the 83M flip is the real measurement]")
    print(f"[compile-smoke] VERDICT: {'GO -- torch.compile runs on WKV + preserves loss; safe to flip the real run' if ok else 'CHECK divergence'}")


if __name__ == "__main__":
    main()
