"""Burndown #3 K=32 speedup (3/3): PARALLEL per-seed launcher for the K=32 routing-margin battery.

The K=32 moat-at-scale battery (`_phaseB_onebrain_sequencerK_k32_margin_derisk.py`) is host-CPU-bound and runs
its seeds SERIALLY (`for K in ks: for s in seeds`). The seeds are fully independent (each builds its own composer
+ sequencer bridge from its own seed; nothing is shared), so on a multi-core host the whole sweep can run in ~one
seed's wall-clock by spawning one process PER SEED and merging the per-seed JSONs into the SAME combined output
the serial runner produces.

This is a pure orchestration wrapper -- it shells out to the EXISTING runner once per seed (so the per-seed
computation, the cache, the vectorized gate-couplings, and the moat are all the byte-identical paths) and then
re-assembles the per-K summary exactly as the serial runner's main() does. BYTE-IDENTITY: each per-seed run is
the unmodified runner on a single seed; merging just concatenates the per-seed result lists in seed order and
re-derives the summary verdict with the identical gate logic. The combined JSON is bit-for-bit what the serial
runner would have written (same per-seed rows/decisions/moat; summary recomputed from them).

  SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_sequencerK_k32_parallel \
      --seeds 42,43,44,100,101,102 --dim 128 --ks 32 --retreat divnorm --gain 0.11 --match-thresh 0.06 \
      --max-parallel 6 --out research/findings/raw/_phaseB_onebrain_sequencerK_k32_parallel.json

Pass-through args (forwarded verbatim to the per-seed runner): --dim --ks --retreat --input-gain --sigma --gain
--peak-mults --host-fallback-above --match-thresh. Launcher-only args: --seeds (the set to fan out), --max-parallel
(concurrent processes; default = min(#seeds, cpu_count)), --out (the merged combined JSON), --keep-per-seed (keep
the per-seed JSON shards instead of deleting them after the merge).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time


PER_SEED_MODULE = "research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk"
# args forwarded verbatim to the per-seed runner (everything EXCEPT --seeds / --out which the launcher controls)
_PASSTHROUGH = ["dim", "ks", "retreat", "input_gain", "sigma", "gain", "peak_mults", "host_fallback_above",
                "match_thresh"]


def _per_seed_cmd(seed, args, out_path):
    cmd = [sys.executable, "-u", "-m", PER_SEED_MODULE, "--seeds", str(seed), "--out", out_path]
    for name in _PASSTHROUGH:
        val = getattr(args, name)
        if val is None:
            continue
        cmd += [f"--{name.replace('_', '-')}", str(val)]
    return cmd


def _recompute_summary(per_K_results, ks, off_guard, off_ok, args):
    """Re-derive the serial runner's summary block from the merged per-seed results, using the IDENTICAL gate
    logic (mirrors `_phaseB_onebrain_sequencerK_k32_margin_derisk.main`). Returns (summary_dict, verdict,
    k_star, first_break_K)."""
    summary = {}
    overall_go = off_ok
    first_break_K = None
    for K in ks:
        rs = per_K_results[str(K)]
        n = len(rs)
        eq_n = sum(r["eq_all"] for r in rs)
        moat_n = sum(r["moat_ok"] for r in rs)
        les_n = sum(r["lesion_fails_safe"] for r in rs)
        perm_n = sum(r["permuted_inverts"] for r in rs)
        raw_n = sum(r["raw_fails"] for r in rs)
        fa_total = sum(r["false_accepts"] for r in rs)
        pr_n = sum(r["peak_robust"] for r in rs)
        any_host = any(r["used_host"] for r in rs)
        if any_host:
            go = (eq_n == n and moat_n == n and fa_total == 0)
        else:
            go = (eq_n == n and moat_n == n and les_n == n and perm_n == n and raw_n == n
                  and fa_total == 0 and pr_n == n)
        overall_go = overall_go and go
        if not go and first_break_K is None:
            first_break_K = K
        summary[str(K)] = dict(n=n, eq_n=eq_n, moat_n=moat_n, lesion_n=les_n, permuted_n=perm_n,
                               raw_fails_n=raw_n, fa_total=fa_total, peak_robust_n=pr_n, any_host=any_host,
                               verdict="GO" if go else "NEGATIVE")
    onbridge_go_ks = [int(k) for k in ks if summary[str(k)]["verdict"] == "GO" and not summary[str(k)]["any_host"]]
    k_star = max(onbridge_go_ks) if onbridge_go_ks else None
    verdict = "GO" if overall_go else "NEGATIVE"
    return summary, verdict, k_star, first_break_K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--ks", default="32")
    ap.add_argument("--retreat", default="divnorm", choices=["divnorm", "wta"])
    ap.add_argument("--input-gain", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=0.11)
    ap.add_argument("--peak-mults", default="1.0")
    ap.add_argument("--host-fallback-above", type=int, default=None)
    ap.add_argument("--match-thresh", type=float, default=0.06)
    ap.add_argument("--max-parallel", type=int, default=None, help="concurrent per-seed processes")
    ap.add_argument("--keep-per-seed", action="store_true", help="keep the per-seed JSON shards after merge")
    ap.add_argument("--out", default="research/findings/raw/_phaseB_onebrain_sequencerK_k32_parallel.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    ks = [int(k) for k in args.ks.split(",")]
    max_parallel = args.max_parallel or min(len(seeds), os.cpu_count() or 1)

    # cap each child's BLAS thread pool so N numpy processes don't oversubscribe the cores (each per-seed run is
    # mostly a Python/scipy-sparse loop; a few BLAS threads per process x N processes would thrash). Inherit-only.
    child_env = dict(os.environ)
    child_env.setdefault("SIM_BACKEND", "numpy")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        child_env.setdefault(var, "2")

    tmpdir = tempfile.mkdtemp(prefix="k32par_")
    seed_out = {s: os.path.join(tmpdir, f"seed{s}.json") for s in seeds}
    log_out = {s: os.path.join(tmpdir, f"seed{s}.log") for s in seeds}

    print(f"PARALLEL K=32: {len(seeds)} seeds {seeds}, max_parallel={max_parallel}, ks={ks}, "
          f"retreat={args.retreat} gain={args.gain} match_thresh={args.match_thresh} peak_mults={args.peak_mults}",
          flush=True)
    t0 = time.time()
    pending = list(seeds)
    running = {}   # seed -> (Popen, log file handle, start_time)

    def _launch(seed):
        lf = open(log_out[seed], "w")
        cmd = _per_seed_cmd(seed, args, seed_out[seed])
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=child_env)
        running[seed] = (p, lf, time.time())
        print(f"  [launch] seed {seed} pid {p.pid}", flush=True)

    seed_wall = {}
    failed = []
    while pending or running:
        while pending and len(running) < max_parallel:
            _launch(pending.pop(0))
        time.sleep(0.5)
        done = [s for s, (p, _lf, _t) in running.items() if p.poll() is not None]
        for s in done:
            p, lf, t_start = running.pop(s)
            lf.close()
            seed_wall[s] = time.time() - t_start
            rc = p.returncode
            ok = (rc == 0 and os.path.exists(seed_out[s]))
            print(f"  [done]   seed {s} rc={rc} wall={seed_wall[s]:.0f}s {'OK' if ok else 'FAILED'}", flush=True)
            if not ok:
                failed.append(s)
                # surface the tail of the child log for diagnosis
                try:
                    with open(log_out[s]) as f:
                        tail = "".join(f.readlines()[-12:])
                    print(f"  [seed {s} log tail]\n{tail}", flush=True)
                except OSError:
                    pass
    total_wall = time.time() - t0

    if failed:
        print(f"FAILED seeds: {failed} -- NOT merging (a partial merge would misreport the verdict).", flush=True)
        sys.exit(1)

    # merge: concatenate each seed's per-K result list in SEED ORDER (== the serial runner's order), then recompute
    # the summary with the identical gate logic. The off-guard is seed-independent; take it from the first shard.
    per_seed = {s: json.load(open(seed_out[s])) for s in seeds}
    off_guard = per_seed[seeds[0]]["summary"]["off_guard"]
    off_ok = per_seed[seeds[0]]["summary"]["off_ok"]
    gpu = per_seed[seeds[0]]["summary"]["gpu"]
    merged_results = {}
    for K in ks:
        rows = []
        for s in seeds:
            rs = per_seed[s]["results"][str(K)]
            assert len(rs) == 1, f"seed {s} K={K} produced {len(rs)} results (expected 1)"
            rows.append(rs[0])
        merged_results[str(K)] = rows

    summary, verdict, k_star, first_break_K = _recompute_summary(merged_results, ks, off_guard, off_ok, args)

    # per-seed console echo (mirrors the serial runner's per-seed line) + the per-K summary line
    for K in ks:
        for r in merged_results[str(K)]:
            eq = "==host" if r["eq_all"] else "!=host"
            moat = "moat-OK" if r["moat_ok"] else f"MOAT-BREACH(fa={r['false_accepts']})"
            les = "lesion-SAFE" if r["lesion_fails_safe"] else "lesion-UNSAFE"
            perm = "perm-inverts" if r["permuted_inverts"] else "perm-FAIL"
            raw = "raw-fails" if r["raw_fails"] else f"RAW-ALSO-PASSES(fa={r['raw_fa']})"
            nm = r["nominal_modes"]
            pr = "peak-robust" if r["peak_robust"] else "PEAK-VARIES"
            print(f"K={K} seed {r['seed']} D{args.dim}: {eq}  {moat}  {les}  {perm}  {raw}  "
                  f"modes(ex/xt/ms)={nm['exact']}/{nm['extra']}/{nm['miss']}  {pr}", flush=True)
        sm = summary[str(K)]
        print(f"\nK={K} SUMMARY: ==host {sm['eq_n']}/{sm['n']}  moat {sm['moat_n']}/{sm['n']} "
              f"(FA_total {sm['fa_total']})  lesion {sm['lesion_n']}/{sm['n']}  permuted {sm['permuted_n']}/{sm['n']}"
              f"  raw-fails {sm['raw_fails_n']}/{sm['n']}  peak-robust {sm['peak_robust_n']}/{sm['n']}"
              f"  -> {sm['verdict']}", flush=True)

    print(f"\nOVERALL: {verdict}  (K in {ks}, {len(seeds)} seeds, retreat={args.retreat}, gain={args.gain})",
          flush=True)
    print(f"on-bridge clean-GO K* = {k_star}  (first break at K={first_break_K})", flush=True)
    print(f"\n[PARALLEL] total wall {total_wall:.0f}s; per-seed wall {{ {', '.join(f'{s}:{seed_wall[s]:.0f}s' for s in seeds)} }}; "
          f"slowest seed {max(seed_wall.values()):.0f}s; speedup vs serial-sum "
          f"~{sum(seed_wall.values())/total_wall:.2f}x", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=dict(per_K=summary, off_guard=off_guard, off_ok=off_ok, verdict=verdict,
                                    k_star=k_star, first_break_K=first_break_K, retreat=args.retreat,
                                    gpu=gpu, input_gain=args.input_gain, sigma=args.sigma, gain=args.gain,
                                    match_thresh=args.match_thresh,
                                    peak_mults=[float(x) for x in args.peak_mults.split(",")],
                                    host_fallback_above=args.host_fallback_above,
                                    parallel=dict(n_seeds=len(seeds), max_parallel=max_parallel,
                                                  total_wall_s=round(total_wall, 1),
                                                  per_seed_wall_s={str(s): round(seed_wall[s], 1) for s in seeds})),
                       results=merged_results), f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)

    if not args.keep_per_seed:
        for s in seeds:
            for path in (seed_out[s], log_out[s]):
                try:
                    os.remove(path)
                except OSError:
                    pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


if __name__ == "__main__":
    main()
