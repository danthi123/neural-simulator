"""2026-07-15 — EDGE 5 rung 3b: does the error-correcting DELTA write ENGAGE at HIGH interference? rung-3 found delta ≈
additive at the easy scale (KV=4, P≤4, distinct values) — the depression was idle because specific potentiation already
handled the mild interference. The numpy store's delta>additive gap opened only at HIGH load (many binds held at once →
additive saturates). This probe raises the load: KV=8 value pools, P up to 8 distinct binds. If DELTA(depress) now beats
additive(no-depress), the error-correction IS the multi-bind mechanism (just needs interference to engage); if delta ≈
additive still, the weight store is capacity-bound and the honest surpass is the binder's slot separation.

numpy-CPU; NO `sim/` edit (monkeypatches the store's KV/POOL to raise the value-pool count; reuses the rung-3 DeltaStore).

Run: SIM_BACKEND=numpy python -u -m research.runners._edge5_rung3b_high_interference_probe --seeds 42
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--kv", type=int, default=8)
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--out", default="research/findings/raw/_edge5_rung3b_high_interference.json")
    a = ap.parse_args()
    import research.runners._edge5_rung2_stp_store_onbridge_derisk as r2
    import research.runners._edge5_rung3_delta_store_onbridge_derisk as r3
    r2.KV = a.kv; r2.POOL = 30              # more value pools (higher interference); smaller pools to keep the bridge modest
    r3.KV = a.kv; r3.POOL = 30
    Ps = tuple(p for p in (2, 4, 6, 8) if p <= a.kv)
    rows = []
    for s in a.seeds:
        rd = r3.run_multipair(s, Ps=Ps, n_trials=a.n_trials, depress=True)
        ra = r3.run_multipair(s, Ps=Ps, n_trials=a.n_trials, depress=False)
        row = {"seed": s, "kv": a.kv, "chance": round(1.0 / a.kv, 4),
               "delta": rd["byP"], "additive": ra["byP"],
               "delta_beats_additive": {P: round(rd["byP"][P] - ra["byP"][P], 4) for P in Ps}}
        rows.append(row)
        db = row["delta_beats_additive"]
        print(f"[hi-interf s{s}] chance={row['chance']} kv={a.kv} || "
              + " ".join(f"P{P}:d={rd['byP'][P]:.2f}/a={ra['byP'][P]:.2f}(gap{db[P]:+.2f})" for P in Ps), flush=True)
    # verdict: at high load does the depression help (mean Δ at the hardest P > 0.05)?
    hardP = Ps[-1]
    mean_gap = float(np.mean([r["delta_beats_additive"][hardP] for r in rows]))
    engages = mean_gap > 0.05
    print(f"[hi-interf] mean DELTA-minus-additive at P={hardP}: {mean_gap:+.3f} -> error-correction "
          f"{'ENGAGES at high interference (the depression IS the multi-bind lever; rung-3 was just too easy to trigger it)' if engages else 'STILL IDLE (delta approx additive even at high load) -> the weight store is capacity-bound; the honest surpass is the binder slot separation (banked), not the error-correcting write'}.", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
