"""RANK-6 production-flip gate (scaffold_retirement_backlog.md #6 / Vikunja #211): the ACTUAL cupy GPU VRAM
measurement the flip is gated on, at real knowledge-core scale.

WHY THIS FILE EXISTS (state contradiction resolved 2026-09-08, before writing a line of new code): two docs
disagreed about whether this measurement had already been taken.
  - `docs/plans/2026-09-07-rank6-substrate-store-vram-reduction-options.md` says the cupy VRAM measurement is
    "queued" and only ESTIMATES 3.8-5GB.
  - `research/coordination/scaffold_retirement_backlog.md`'s rank-6 STATUS UPDATE (2026-09-05) reports "MEASURED
    (not estimated) marginal memory cost: 50.26 KB/fact ... 3.78 GB at the full 78,857-fact core -- affordable".
These are NOT the same claim. Reading the cited source
(`research/findings/2026-09-05-rank6-knowledge-core-substrate-write-scaled-derisk-mixed.md`, section (b)) shows
the 3.78 GB figure is **peak host RSS** (`resource.getrusage(...).ru_maxrss`), measured by
`research/runners/_rank6_knowledge_core_substrate_write_derisk.py`, whose `main()` calls
`assert_backend("numpy", ...)` UNCONDITIONALLY -- that runner cannot run on cupy at all, by construction, and
never has. Host RAM and GPU VRAM are different resources on this machine (46GB RAM headroom vs the 24GB 3090);
the backlog's "affordable" verdict answers the RAM question, not the gate's actual VRAM question. The plan
doc's "queued" is the correct read. This file is that queued measurement.

METHOD (reuses the existing store path -- NO new store mechanism, per the task's own instruction). Imports
`build_store`/`load_real_facts`/`vocab_of` directly from the sibling RSS runner
(`_rank6_knowledge_core_substrate_write_derisk.py`) -- the identical `ShardedPhasorStore`/`RFPhasorComposer`
construction, just executed under `SIM_BACKEND=cupy` instead of numpy, with the measurement swapped from
`ru_maxrss` to GPU memory. Same earned methodology as that file's own (b): a background `nvidia-smi` poller
takes the TRUE peak (a checkpoint-time snapshot would miss a transient spike mid-build), a baseline reading is
taken AFTER the fixed CUDA-context + codebook allocation but BEFORE any fact is stored (so the marginal slope
is not swamped by fixed overhead, the identical trap that runner's own docstring names), and substrate=True vs
substrate=False run in SEPARATE spawned subprocesses (CUDA requires `spawn`, not `fork`; and a high-water mark
never falls within one process, so the two variants would contaminate each other in one process exactly as the
RSS runner's own note explains).

MUST run through `tools/gpu_queue.sh` (loads SimulationBridge/cupy onto the shared 3090 -- CLAUDE.md
cost-routing; two concurrent brain-adjacent cupy jobs risk OOM/card-hang). No `sim/` edit. Default-off; this
file does not touch `enable_substrate_store`'s production default.

Run (via the queue, not directly):
  tools/gpu_queue.sh add 'cd <this checkout> && SIM_BACKEND=cupy <venv>/bin/python -u -m \
      research.runners._rank6_substrate_store_cupy_vram_derisk \
      --cost-points 500 2000 8000 --target-n 78857 \
      --out research/findings/raw/_rank6_substrate_store_cupy_vram_derisk/full_run.json'
  # --smoke : tiny N for a fast end-to-end sanity pass before committing to the real checkpoints
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time

os.environ["SIM_BACKEND"] = "cupy"  # HARD-set (not setdefault) -- this file's whole point is the cupy reading

DEFAULT_BUNDLE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k"
D = 128


# ------------------------------------------------------------------------------------------------------------
def nvidia_smi_used_mib():
    """Whole-device `memory.used` (MiB) -- includes CUDA context overhead + every other process on the card,
    same convention `docs/FAILURE_GATE_MATRIX.md`'s own VRAM checks (vllm_sleep_mode_test.py etc.) use."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip().splitlines()
        vals = [int(x.strip()) for x in out if x.strip()]
        return max(vals) if vals else None
    except Exception:
        return None


class VramPoller:
    """Background thread sampling nvidia-smi every `interval_s`, tracking the peak -- a checkpoint-time
    snapshot alone can miss a transient allocation spike mid-build (e.g. a temporary buffer during a single
    fact's resonate steps that is freed before the next checkpoint is reached)."""

    def __init__(self, interval_s=0.25):
        self.interval_s = interval_s
        self.peak_mib = nvidia_smi_used_mib() or 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            v = nvidia_smi_used_mib()
            if v is not None and v > self.peak_mib:
                self.peak_mib = v
            time.sleep(self.interval_s)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        return self.peak_mib


# ------------------------------------------------------------------------------------------------------------
# COST worker -- mirrors _rank6_knowledge_core_substrate_write_derisk._cost_worker's earned design (baseline
# AFTER fixed cost, grow ONE store to the largest checkpoint, substrate variants in separate processes) but
# reads GPU memory (nvidia-smi peak + cupy's own pool accounting) instead of ru_maxrss.
# ------------------------------------------------------------------------------------------------------------
def _cost_worker(facts_path, vocab, checkpoints, substrate, seed, q):
    os.environ["SIM_BACKEND"] = "cupy"
    # reuse the EXISTING store-construction path, no new mechanism:
    from research.runners._rank6_knowledge_core_substrate_write_derisk import build_store  # noqa: F401 (documents reuse)
    from research.runners.sharded_phasor_store import ShardedPhasorStore
    from research.runners.tiered_fact_store import auto_n_shards
    import cupy as cp

    with open(facts_path) as fh:
        facts = [r["fact"] for r in json.load(fh)][: max(checkpoints)]

    poller = VramPoller().start()
    used0_mib = nvidia_smi_used_mib()  # BEFORE constructing anything (CUDA-context/driver floor for this process)

    n_shards = auto_n_shards(len(facts))
    store = ShardedPhasorStore(n_shards=n_shards, seed=seed, D=D, vocab=list(vocab), share_codebook=True,
                               enable_substrate_store=substrate)
    used_post_construct_mib = nvidia_smi_used_mib()  # after codebook alloc, BEFORE any fact stored -- the baseline
    pool = cp.get_default_memory_pool()

    rows = []
    done = 0
    t_start = time.time()
    for cpn in sorted(checkpoints):
        while done < cpn:
            f = facts[done]
            store.store(f["agent"], f["action"], f["patient"], polarity=f.get("polarity"))
            done += 1
        used_mib = nvidia_smi_used_mib()
        rows.append(dict(
            n=cpn, substrate=substrate, elapsed_s=round(time.time() - t_start, 3),
            used0_mib=used0_mib,
            used_post_construct_mib=used_post_construct_mib,
            used_mib_at_checkpoint=used_mib,
            marginal_vs_construct_mib=(None if used_mib is None or used_post_construct_mib is None
                                        else used_mib - used_post_construct_mib),
            cupy_pool_used_mib=round(pool.used_bytes() / (1024.0 * 1024.0), 3),
            cupy_pool_total_mib=round(pool.total_bytes() / (1024.0 * 1024.0), 3),
            nvidia_smi_peak_mib_so_far=poller.peak_mib,
        ))
    peak_mib = poller.stop()
    rows[-1]["nvidia_smi_peak_mib_final"] = peak_mib
    q.put(rows)


def run_cost(bundle_dir, checkpoints, seed=42, verbose=True):
    from research.runners._rank6_knowledge_core_substrate_write_derisk import load_real_facts, vocab_of
    facts_path = os.path.join(bundle_dir, "facts.json")
    all_facts = load_real_facts(bundle_dir)
    vocab = vocab_of(all_facts[: max(checkpoints)])
    ctx = mp.get_context("spawn")  # CUDA is not fork-safe
    all_rows = []
    for substrate in (False, True):
        q = ctx.Queue()
        p = ctx.Process(target=_cost_worker, args=(facts_path, vocab, checkpoints, substrate, seed, q))
        p.start()
        rows = q.get()
        p.join()
        all_rows.extend(rows)
        for row in rows:
            if verbose:
                print("  N=%-6d substrate=%-5s elapsed=%.2fs  used0=%sMiB post_construct=%sMiB "
                      "at_checkpoint=%sMiB marginal=%sMiB pool_used=%.1fMiB pool_total=%.1fMiB "
                      "nvidia_peak_so_far=%sMiB"
                      % (row["n"], substrate, row["elapsed_s"], row["used0_mib"], row["used_post_construct_mib"],
                         row["used_mib_at_checkpoint"], row["marginal_vs_construct_mib"], row["cupy_pool_used_mib"],
                         row["cupy_pool_total_mib"], row["nvidia_smi_peak_mib_so_far"]))
    return all_rows


def project_from_cost(rows, target_n, label):
    """Marginal MiB/fact from the LARGEST-vs-SMALLEST checkpoint of each variant (both already exclude the
    fixed CUDA-context+codebook cost via the post-construct baseline) -- an honest incremental slope, matching
    `_rank6_knowledge_core_substrate_write_derisk.project_from_cost`'s own convention for the RSS reading."""
    def slope(sub_rows, field):
        sub_rows = sorted(sub_rows, key=lambda r: r["n"])
        sub_rows = [r for r in sub_rows if r.get(field) is not None]
        if len(sub_rows) < 2:
            return None
        lo, hi = sub_rows[0], sub_rows[-1]
        dn = hi["n"] - lo["n"]
        if dn <= 0:
            return None
        return dict(per_fact_mib=(hi[field] - lo[field]) / dn, n_lo=lo["n"], n_hi=hi["n"])

    sub_rows = [r for r in rows if r["substrate"]]
    base_rows = [r for r in rows if not r["substrate"]]

    out = dict(label=label, target_n=target_n)
    for field, key in (("marginal_vs_construct_mib", "nvidia_smi"), ("cupy_pool_total_mib", "cupy_pool")):
        sub_slope = slope(sub_rows, field)
        base_slope = slope(base_rows, field)
        if sub_slope is None:
            print("  PROJECT[%s] %s: fewer than 2 substrate checkpoints -- UNDEFINED" % (key, label))
            continue
        proj_gb = sub_slope["per_fact_mib"] * target_n / 1024.0
        print("  PROJECT[%s] %-28s substrate slope (N=%d->%d): %.5f MiB/fact => %.3f GiB at N=%d"
              % (key, label, sub_slope["n_lo"], sub_slope["n_hi"], sub_slope["per_fact_mib"], proj_gb, target_n))
        entry = dict(substrate_slope_mib_per_fact=sub_slope["per_fact_mib"], proj_gib=round(proj_gb, 4))
        if base_slope is not None:
            base_proj_gb = base_slope["per_fact_mib"] * target_n / 1024.0
            entry["numpy_kb_slope_mib_per_fact"] = base_slope["per_fact_mib"]
            entry["numpy_kb_proj_gib"] = round(base_proj_gb, 4)
            print("    for comparison, numpy-kb path[%s]: %.5f MiB/fact => %.3f GiB at N=%d"
                  % (key, base_slope["per_fact_mib"], base_proj_gb, target_n))
        out[key] = entry
    return out


# ------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost-points", type=int, nargs="+", default=[500, 2000, 8000])
    ap.add_argument("--target-n", type=int, default=78857)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="research/findings/raw/_rank6_substrate_store_cupy_vram_derisk/full_run.json")
    a = ap.parse_args()

    if a.smoke:
        a.cost_points = [10, 30, 60]

    from tools.lab import assert_backend
    assert_backend("cupy", note="(this file's whole point -- the flip's gate is a CUPY VRAM reading, not RSS)")

    if not os.path.isdir(a.bundle):
        print("SKIPPED: bundle not found at %r" % a.bundle)
        out = dict(verdict="UNDEFINED", reason="bundle_not_found", bundle=a.bundle)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2)
        return 0

    print("[rank6-vram] whole-device memory.used BEFORE anything (this process's view): %s MiB"
          % nvidia_smi_used_mib())
    print("\n=== CUPY VRAM COST: subprocess-isolated nvidia-smi peak + cupy pool accounting, substrate vs numpy-kb ===")
    cost = run_cost(a.bundle, a.cost_points, seed=a.seed)
    projection = project_from_cost(cost, a.target_n, "wikidata_100k(%d facts)" % a.target_n)

    out = dict(bundle=a.bundle, target_n=a.target_n, seed=a.seed, cost_points=a.cost_points,
               device_used_mib_before=nvidia_smi_used_mib(), cost=cost, projection=projection)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n[rank6-vram] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
