"""Cloud-GPU speedup benchmark for the run3 WKV LM -- run the SAME script on a cloud GPU and on the local 3090;
the tokens/sec RATIO is the real training speedup for THIS model (not the misleading raw-FLOPS ratio).

Faithful: builds the actual ChunkedWKV at the run3 config (d_model=1024, L=16, chunk_c=16, seq=256, batch=32,
vocab=16000) and times a full fwd+bwd+optimizer step with torch.compile ON (the real training uses --compile 1),
on RANDOM data (no corpus needed). Needs only torch + this repo.

Usage on any CUDA box (from the repo root):
    python -m research.runners.cloud_gpu_benchmark
    # smaller GPU (OOM at batch 32)? use the SAME --batch on BOTH machines for a valid ratio:
    python -m research.runners.cloud_gpu_benchmark --batch 16
Paste the RESULT line back. Local 3090 baseline is measured with the identical command.
"""
import argparse
import time

import torch

from research.runners._lmtrain_chunked_scan import WKV as ChunkedWKV


def _build(vocab, d_model, n_layers, chunk_c):
    torch.manual_seed(0)
    return ChunkedWKV(vocab, d_model, n_layers, block="chunked", C=chunk_c)


def bench(compile_on, vocab, d_model, n_layers, chunk_c, seq_len, batch, n_warm=8, n_timed=30):
    dev = "cuda"
    m = _build(vocab, d_model, n_layers, chunk_c).to(dev)
    if compile_on:
        m = torch.compile(m)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator(device=dev).manual_seed(1)

    def step():
        x = torch.randint(0, vocab, (batch, seq_len), device=dev, generator=g)
        y = m(x)
        loss = lossf(y[:, :-1].reshape(-1, vocab).float(), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(n_warm):          # warmup (torch.compile builds the graph on the first steps)
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_timed):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n_timed
    return dt, (batch * seq_len) / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=16)
    ap.add_argument("--chunk-c", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-compile", action="store_true", help="skip torch.compile (the real training uses it; keep it ON for a representative number)")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[cloud-bench] no CUDA GPU visible -- this must run on a GPU box.")
    n_params = sum(p.numel() for p in _build(a.vocab, a.d_model, a.n_layers, a.chunk_c).parameters())
    print(f"[cloud-bench] torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)} "
          f"| {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
    print(f"[cloud-bench] model d={a.d_model} L={a.n_layers} C={a.chunk_c} vocab={a.vocab} "
          f"seq={a.seq_len} batch={a.batch} -> {n_params/1e6:.1f}M params")
    try:
        dt, tps = bench(not a.no_compile, a.vocab, a.d_model, a.n_layers, a.chunk_c, a.seq_len, a.batch)
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(f"[cloud-bench] OOM at batch={a.batch}; rerun with a smaller --batch (use the SAME value on both machines).")
    print(f"[cloud-bench] RESULT: {dt*1000:.1f} ms/step | {tps:,.0f} tokens/sec | "
          f"compile={'off' if a.no_compile else 'on'} | batch={a.batch}")


if __name__ == "__main__":
    main()
