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
import numpy as np

from research.runners._riii_ca3_synchronous_assembly_derisk import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-ca3", type=int, default=1000)
    ap.add_argument("--n-mem", type=int, default=3)
    ap.add_argument("--ca1-topk", type=int, default=None, help="E%-max CA1 sparsification (top-k by g_e); None=off")
    ap.add_argument("--out", type=str, default="research/findings/raw/_gap5_swr_specificity.json")
    args = ap.parse_args()

    rows = []
    for s in args.seeds:
        on = run(s, n_ca3=args.n_ca3, n_mem=args.n_mem, read_ca1=True,
                 swr_learn_schaffer=True, swr_ca1_topk=args.ca1_topk)
        off = run(s, n_ca3=args.n_ca3, n_mem=args.n_mem, read_ca1=True,
                  swr_learn_schaffer=False, swr_ca1_topk=args.ca1_topk)
        r = {"seed": s,
             "on_match": on.get("ca1_match", 0.0), "on_cross": on.get("ca1_cross", 0.0),
             "off_match": off.get("ca1_match", 0.0), "off_cross": off.get("ca1_cross", 0.0)}
        rows.append(r)
        print(f"  [seed {s}] STACK-ON match={r['on_match']:.3f} cross={r['on_cross']:.3f} "
              f"(ratio {r['on_match']/(r['on_cross']+1e-9):.2f}x) | dense-random-OFF match={r['off_match']:.3f} "
              f"cross={r['off_cross']:.3f}", flush=True)
    m_on = float(np.mean([r["on_match"] for r in rows])); c_on = float(np.mean([r["on_cross"] for r in rows]))
    c_off = float(np.mean([r["off_cross"] for r in rows]))
    go = (m_on >= 0.6 and c_on <= 0.3 and m_on >= 3 * (c_on + 1e-9))
    anticheat = c_off >= 0.7                                   # dense-random collapses (near-tie) => Schaffer load-bearing
    print("=" * 80)
    print(f"[gap5 SWR specificity] n_ca3={args.n_ca3} n_mem={args.n_mem} topk={args.ca1_topk} | {len(rows)} seed(s)")
    print(f"  STACK-ON: ca1_match {m_on:.3f} | ca1_cross {c_on:.3f} | ratio {m_on/(c_on+1e-9):.2f}x")
    print(f"  dense-random-OFF: ca1_cross {c_off:.3f} (must be high >=0.7 = the near-tie the stack fixes)")
    print(f"  {'GO' if (go and anticheat) else 'BOUNDARY'}: ON match>=0.6({m_on>=0.6}) & cross<=0.3({c_on<=0.3}) & "
          f"ratio>=3x({m_on>=3*c_on}) & dense-random-collapses({anticheat})")
    json.dump({"rows": rows, "on_match": m_on, "on_cross": c_on, "off_cross": c_off, "go": bool(go and anticheat)},
              open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
