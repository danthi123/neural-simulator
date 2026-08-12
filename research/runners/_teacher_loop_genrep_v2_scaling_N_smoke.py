"""TEACHER-LOOP GENERATIVE-REPLAY v2 -- BREADTH-CEILING SCALING SWEEP (2026-08-10, MEASUREMENT ONLY, NO new mechanism).

FRONTIER: how far does teacher-loop RETENTION hold as the fact-count N grows, with the CURRENT BEST retention
stack? The best stack is the NON-FORGETTING generative-replay generator `generative_v2`
(research/runners/_teacher_loop_generative_replay_v2_derisk.py; finding
2026-08-09-non-forgetting-generative-replay-generator-matches-flat-store.md; GO 5/6 @ N=20, matches the flat O(N)
store 0.958 vs 0.950 with a BOUNDED fixed-size neural generator). That finding measured retention ONLY to N=20 and
EXPLICITLY left N>20 untested: "N tested to 20 ... not asserted for N>>20", and flagged an upstream open wall:
"the N=100 acquisition wall (...SLIPS-at-N100) -- the shared readout struggling to ACQUIRE 100 facts -- is
upstream of this and still open." The separate CAPACITY (grown-reservoir / neurogenesis) lever was already swept
N in {20,50,100} and SLIPS at N=100 (0.967->0.913->0.727), but that is a DIFFERENT, inferior lever -- NOT this
best generative_v2 consolidation stack.

THIS SWEEP: run the BEST config (generative_v2, its runner's argparse-default GO config: gen_k=64, gen_hidden=96,
gen_tol=0.05, gen_max_epochs=120, gen_new_mult=3, gen_lr=0.8) over N in {20, 50, 100}, single-seed smoke, numpy,
and report WHERE retention breaks and WHY (the finding's known slip: shared-readout acquisition wall at large N /
generator query-code rank -- vs a fixed bug). NO new mechanism: this imports `run` from the committed v2 runner
unchanged (reuse-by-import) and only sweeps N. NO sim/ edit.

SKEPTICAL CONTROL (built into the imported runner, per N):
  * generative_v1  = the naive PRIOR generator (the KNOWN NEGATIVE -- it forgets; ~0.692 @ N=20). Must stay below
                     generative_v2 (and below flat) -- if v1 does NOT reproduce its forgetting, the harness is
                     broken. This is the no-lever/baseline arm.
  * flat           = the O(N) unbounded-store UPPER BOUND. generative_v2's whole claim is "matches flat with a
                     BOUNDED store"; the sweep asks whether that equality survives as N grows.
  * generative_v2  = TREATMENT (the best config). Reported: frac_recalled@N, generator fidelity (mean_cos),
                     immediate-acq. A POSITIVE (retention holds) must survive the v1 control (v2 > v1) at that N.

Also reports the generator's fixed-size anti-cheat (trained-param count constant in N) so a "retention holds"
cannot be an artifact of the store growing with N.

DISCIPLINE: reuse-by-import of the committed v2 runner's `run`/`_verdict`; NO sim/ edit; cfg.seed byte-identical
(asserted inside `run`); backend numpy (tiny launch-bound net). Single-seed SMOKE as instructed. Do NOT commit.

RUN (staged; N=100 is O(N^2) consolidation so it is the slow one -- run it last / alone if needed):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python -m research.runners._teacher_loop_genrep_v2_scaling_N_smoke \
      --seed 42 --n-list 20 50 100 --out /tmp/genrep_v2_scaling_s42.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# reuse-by-import: the committed v2 runner's run() (with all anti-cheats + the 3 arms). NO sim/ edit, NO new mechanism.
from research.runners._teacher_loop_generative_replay_v2_derisk import run as _v2_run  # noqa: E402

# The BEST-config defaults == the v2 runner's argparse defaults == the finding's GO config.
GO = dict(
    capacity=5, slow_hidden=100, gen_hidden=96, gen_k=64, settle=20, epochs=20, batch=16, eprop_lr=0.5,
    w_clip=4000.0, n_draws=16, d_p=12, noise=0.12, test_n=40, replay_epochs=12, replay_per_fact=8,
    replay_noise=0.10, gen_settle=15, gen_epochs=16, gen_lr=0.8, gen_tol=0.05, gen_max_epochs=120,
    gen_check_every=4, gen_new_mult=3, bdsp_wmax=1e9,
)


def _milestones_for(n):
    ms = [m for m in (10, 20, 50, 100) if m <= n]
    if n not in ms:
        ms.append(n)
    return sorted(set(ms))


def _frac(arms, arm, n):
    rc = arms.get(arm, {}).get("retention_curve", {})
    d = rc.get(str(n))
    return float(d["frac_recalled"]) if d else float("nan")


def _fid(arms, arm, n):
    fd = arms.get(arm, {}).get("generator_fidelity", {}).get(str(n), {})
    return float(fd.get("mean_cos", float("nan")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-list", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--arms", nargs="+", default=["generative_v2", "generative_v1", "flat"])
    ap.add_argument("--out", default="/tmp/genrep_v2_scaling_s42.json")
    a = ap.parse_args()

    rows = []
    per_n = {}
    for n in a.n_list:
        ms = _milestones_for(n)
        t0 = time.time()
        print(f"\n==== N={n}  (milestones {ms})  seed {a.seed}  arms {a.arms} ====", flush=True)
        res = _v2_run(seed=a.seed, n_max=n, milestones=ms, arms_to_run=a.arms, **GO)
        arms = res["arms"]
        row = dict(
            N=n,
            v2_frac=_frac(arms, "generative_v2", n),
            v1_frac=_frac(arms, "generative_v1", n),
            flat_frac=_frac(arms, "flat", n),
            v2_gen_cos=_fid(arms, "generative_v2", n),
            v1_gen_cos=_fid(arms, "generative_v1", n),
            v2_imm_acq=float(arms.get("generative_v2", {}).get("mean_acquire_acc_immediate", float("nan"))),
            v1_imm_acq=float(arms.get("generative_v1", {}).get("mean_acquire_acc_immediate", float("nan"))),
            flat_imm_acq=float(arms.get("flat", {}).get("mean_acquire_acc_immediate", float("nan"))),
            gen_trained_params=res.get("generator_trained_params"),
            gen_k_query_width=res.get("gen_k_query_width_v2"),
            substrate_byte_identical=res.get("substrate_byte_identical"),
            sim_diff_empty=res.get("sim_diff_empty"),
            wall_s=round(time.time() - t0, 1),
        )
        rows.append(row)
        per_n[str(n)] = res
        print(f"[N={n}] v2 {row['v2_frac']:.2f} | v1 {row['v1_frac']:.2f} | flat {row['flat_frac']:.2f} "
              f"| v2-cos {row['v2_gen_cos']:.3f} | v2-imm-acq {row['v2_imm_acq']:.3f} "
              f"| gen-params {row['gen_trained_params']} | {row['wall_s']:.0f}s", flush=True)
        # write incrementally so a kill at N=100 still keeps N=20/50
        Path(a.out).write_text(json.dumps({"seed": a.seed, "config": GO, "rows": rows}, indent=2, default=str))

    print("\n================ SCALING SWEEP (best config = generative_v2) ================")
    print(f"{'N':>4} | {'v2':>5} | {'v1(ctl)':>7} | {'flat(UB)':>8} | {'v2-cos':>6} | {'v2-imm':>6} | {'gen-par':>7} | wall")
    for r in rows:
        print(f"{r['N']:>4} | {r['v2_frac']:>5.2f} | {r['v1_frac']:>7.2f} | {r['flat_frac']:>8.2f} | "
              f"{r['v2_gen_cos']:>6.3f} | {r['v2_imm_acq']:>6.3f} | {str(r['gen_trained_params']):>7} | {r['wall_s']:.0f}s")

    # skeptical read: control must reproduce the known negative (v1 forgets: v1 < v2) at every N that ran v1
    ctl_ok = all((r["v1_frac"] < r["v2_frac"]) for r in rows if r["v1_frac"] == r["v1_frac"])
    print(f"\ncontrol (v1<v2 every N, known-negative reproduced): {ctl_ok}")
    print(f"gen fixed-size (trained-params constant across N): "
          f"{len(set(r['gen_trained_params'] for r in rows)) == 1} -> {[r['gen_trained_params'] for r in rows]}")
    Path(a.out).write_text(json.dumps(
        {"seed": a.seed, "config": GO, "rows": rows, "control_v1_below_v2": ctl_ok}, indent=2, default=str))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
