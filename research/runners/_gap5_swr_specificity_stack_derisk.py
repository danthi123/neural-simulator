"""gap#5 — SWR replay READOUT specificity: does the structured+potentiated SPARSE Schaffer (drop the dense-random
readout) give CA1-specific replay, closing the near-tie (distinct CA3 assemblies → near-identical CA1)?

Per the 2026-07-19 research gate (Valero 2017: CA1 replay selectivity lives in the CELL-SPECIFIC structured+potentiated
synaptic drive, NOT a dense-random Schaffer). The gate's decisive first rung: on DISTINCT (pre-assigned sparse)
assemblies, turn the readout STACK on — `swr_learn_schaffer=True` builds a distinct sparse CA1 TARGET per assembly and
potentiates ONLY that assembly→target Schaffer (structured sparse), vs the dense-random baseline. If ca1_match/cross
SEPARATE → the near-tie was the dense-random readout (fixable); the runner's own no-learn anti-cheat (cross 0.999→0.27)
already flags the Schaffer as load-bearing.

ONE VARIABLE: swr_learn_schaffer ON vs OFF (OFF = dense-random). GATE: ON ca1_match≥0.6, ca1_cross≤0.3, ratio≥3×,
6-seed; OFF cross≈1 (dense-random collapses — the anti-cheat). Reuse-by-import of the SWR runner's `run()`. GPU.
"""
import argparse
import json
import os
import numpy as np

# REPRODUCIBILITY (2026-07-21): the CLOSED 6/6 result requires PHASE-2 Schaffer STP-off — without it the no-learn
# dense-random anti-cheat does NOT collapse (the STP-crushed g_e can't saturate CA1 into the near-tie), so the readout
# looks "specific" for the WRONG reason. Default it ON here so the stack reproduces (subagent-verified: with it, 6-seed
# match 0.626 / cross 0.042 / ratio 14.9x, no-learn collapses to near-tie; without it match 0.397 / cross 0.117).
os.environ.setdefault("SWR_PHASE2_NOSTP", "1")

from research.runners._riii_ca3_synchronous_assembly_derisk import run  # noqa: E402

# THE SWR-CLOSED config (2026-07-19 6/6 GO + anti-cheat clean). = the bistable-completion ENCODING base
# (raw/_gap5_ca3_bistable_6seed.GO: hebb_lr/coact_thresh/lam_dep_wi/hebb_max/ca3_fb_inhib/k_thresh — the params my
# first COMPLETION_CFG OMITTED, giving a dead completion) + the SWR sparse+synchronous overrides (assembly_frac=0.03,
# no_sync=False, recall_k_thresh=30, hebb_max=150, recall_drive=1200, hebb_lr=4, lam_dep=1, swr_disjoint) + learned
# Schaffer + E%-max read. VERIFY-FIRST: the driver prints held_cue so a dead completion (=> cue-driven ca1 artifact)
# is caught before trusting ca1_match.
COMPLETION_CFG = dict(
    n_ca3=2000, ca3_density=0.05, assembly_frac=0.03, encode_drive=3000.0, no_sync=False,
    coact_thresh=0.02, hebb_lr=4.0, lam_dep_wi=1.0, hebb_max=150.0, ca3_fb_inhib=30.0, k_thresh=15.0,
    recall_k_thresh=30.0, recall_drive=1200, recall_steps=150, bistable=True, nmda_recurrent=False,
    enable_ou=False, selective_inhib=True, structural_sep=1, plateau_self_regen=0.15, apical_kir_g=3.0,
    apical_gc=1.0, apical_gc_read=5.0, swr_disjoint=True, swr_schaffer_hi=80.0, swr_schaffer_lo=0.0,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=2)
    ap.add_argument("--ca1-topk", type=float, default=None,
                    help="E%-max CA1 winner FRACTION (0..1, multiplied by n_ca1=120 in the runner); None=off")
    ap.add_argument("--out", type=str, default="research/findings/raw/_gap5_swr_specificity.json")
    args = ap.parse_args()

    rows = []
    for s in args.seeds:
        cfg = {**COMPLETION_CFG, "n_mem": args.n_mem}
        on = run(s, read_ca1=True, swr_learn_schaffer=True, swr_ca1_topk=args.ca1_topk, **cfg)
        off = run(s, read_ca1=True, swr_learn_schaffer=False, swr_ca1_topk=args.ca1_topk, **cfg)
        r = {"seed": s,
             "held_cue": on.get("held_cue", 0.0),  # VERIFY the completion is live (readout on a dead completion = void)
             "on_match": on.get("ca1_match", 0.0), "on_cross": on.get("ca1_cross", 0.0),
             "off_match": off.get("ca1_match", 0.0), "off_cross": off.get("ca1_cross", 0.0)}
        rows.append(r)
        print(f"  [seed {s}] completion held_cue={r['held_cue']:.3f} | STACK-ON match={r['on_match']:.3f} "
              f"cross={r['on_cross']:.3f} (ratio {r['on_match']/(r['on_cross']+1e-9):.2f}x) | "
              f"dense-random-OFF match={r['off_match']:.3f} cross={r['off_cross']:.3f}", flush=True)
    m_on = float(np.mean([r["on_match"] for r in rows])); c_on = float(np.mean([r["on_cross"] for r in rows]))
    c_off = float(np.mean([r["off_cross"] for r in rows]))
    go = (m_on >= 0.6 and c_on <= 0.3 and m_on >= 3 * (c_on + 1e-9))
    anticheat = c_off >= 0.7                                   # dense-random collapses (near-tie) => Schaffer load-bearing
    print("=" * 80)
    print(f"[gap5 SWR specificity] completing-cfg n_ca3={COMPLETION_CFG['n_ca3']} n_mem={args.n_mem} "
          f"topk={args.ca1_topk} | {len(rows)} seed(s)")
    print(f"  STACK-ON: ca1_match {m_on:.3f} | ca1_cross {c_on:.3f} | ratio {m_on/(c_on+1e-9):.2f}x")
    print(f"  dense-random-OFF: ca1_cross {c_off:.3f} (must be high >=0.7 = the near-tie the stack fixes)")
    print(f"  {'GO' if (go and anticheat) else 'BOUNDARY'}: ON match>=0.6({m_on>=0.6}) & cross<=0.3({c_on<=0.3}) & "
          f"ratio>=3x({m_on>=3*c_on}) & dense-random-collapses({anticheat})")
    json.dump({"rows": rows, "on_match": m_on, "on_cross": c_on, "off_cross": c_off, "go": bool(go and anticheat)},
              open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
