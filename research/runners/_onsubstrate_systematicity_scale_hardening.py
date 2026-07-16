"""2026-07-15 — the honest follow-up the RUNG-1/2 CORRECTION named: does the on-substrate systematicity hold at a LARGER task
scale (more cat×qt combos → more held-out → less small-sample leakage + tighter controls), or is the on-spikes bind genuinely
POINT-NEURON-NOISE-LIMITED (a real boundary)? At the small 7×7 task, held to the parent's FULL controls (1-NN memfloor +
linear-raw), RUNG-1 cleared +0.15 on only 1/6 seeds and RUNG-2 on 3/6 — the verifier showed spiking noise erodes the margin on
the leaky/near-degenerate seeds. If the win is real, a larger task (fewer degenerate held-out cells) should recover the margin.

Scale via a monkeypatch of `_fixedbind_systematicity_derisk.N_CAT/N_QTYPE` (build_task reads them at call time; the bind dim
D=NB+D_PAD is UNCHANGED, only the number of (cat,qt) combos grows). Re-runs RUNG-1 + RUNG-2's `run_one` verbatim (their FULL
controls already restored post-correction). numpy-CPU; NO `sim/` edit.

Run: SIM_BACKEND=numpy python -u -m research.runners._onsubstrate_systematicity_scale_hardening --n-cat 12 --n-qtype 12 --seeds 42 43 44 100 101 102
"""
import os, sys, json, argparse

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cat", type=int, default=12)
    ap.add_argument("--n-qtype", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--out", default="research/findings/raw/_onsubstrate_systematicity_scale_hardening.json")
    a = ap.parse_args()
    import research.runners._fixedbind_systematicity_derisk as fb
    fb.N_CAT = int(a.n_cat); fb.N_QTYPE = int(a.n_qtype)          # scale the task (more combos -> more held-out)
    from research.runners._onsubstrate_coincidence_systematicity_derisk import run_one as rung1
    from research.runners._onsubstrate_bind_learned_readout_derisk import run_one as rung2
    out = {"n_cat": a.n_cat, "n_qtype": a.n_qtype, "combos": a.n_cat * a.n_qtype, "rung1": [], "rung2": []}
    for s in a.seeds:
        r1 = rung1(s); out["rung1"].append(r1)
        print(f"[scale-R1 s{s}] n_held={r1['n_held']} spikebind={r1['spikebind_held']:.3f} | memfloor={r1['memfloor_held']:.3f} "
              f"| linear={r1['linear_held']:.3f} | MLP={r1['mlp_held']:.3f} || GO_full={'GO' if r1['GO'] else 'no'}", flush=True)
    for s in a.seeds:
        r2 = rung2(s); out["rung2"].append(r2)
        print(f"[scale-R2 s{s}] transportfree={r2['transportfree_held']:.3f} | memfloor={r2['memfloor_held']:.3f} "
              f"| linear={r2['linear_held']:.3f} || GO_full={'GO' if r2['GO'] else 'no'}", flush=True)
    n1 = sum(x["GO"] for x in out["rung1"]); n2 = sum(x["GO"] for x in out["rung2"])
    m1 = sum(x["spikebind_held"] for x in out["rung1"]) / len(out["rung1"])
    m2 = sum(x["transportfree_held"] for x in out["rung2"]) / len(out["rung2"])
    print(f"[scale {a.n_cat}x{a.n_qtype}={a.n_cat*a.n_qtype} combos] RUNG-1 full-controls GO {n1}/{len(a.seeds)} (mean bind {m1:.3f}); "
          f"RUNG-2 full-controls GO {n2}/{len(a.seeds)} (mean transportfree {m2:.3f}). "
          f"{'HARDENS (larger task recovers the margin = real, small-task was noise-limited)' if n1 >= 4 else 'still noise-limited/bounded at this scale (a genuine point-neuron residual, honestly)'}.", flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
