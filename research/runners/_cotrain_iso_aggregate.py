"""Aggregate the FUSED co-training isolation sweep: --homeostasis 0 vs 1, across seeds, at ONE per-learner budget.

Answers two questions in one table:
  (1) POST-BUDGET-FIX 6-seed: what is the honest co-training cost now that each learner gets the SAME data as its
      separate baseline? (The banked _cotrain_iso_6seed.json predates the per-learner cap -- it records
      cotrain=16000 AND sepA=sepB=16000, impossible under the per-learner code, so each co-trained learner got HALF
      its baseline's reinforcement. That table is confounded and must not be defended.)
  (2) RESIDUAL CAUSE: does disabling homeostasis close the residual? Mechanism (read from sim/kernels.py:348-361 +
      sim/config.py:430-432): threshold += (ema - target_rate) * adapt_rate, with target_rate=0.02 and
      ema_alpha=2e-4 (tau ~5000 steps) vs window_steps=2. The EMA is ~2500x too slow to track the A/B interleave,
      so it CANNOT "drift per window"; what it sees is that each CO-TRAINED learner fires in only ~half the
      wall-clock steps -> its long-run average rate is ~half the separate baseline's -> (ema - 0.02) is
      systematically negative -> thresholds drift DOWN -> hyper-excitable -> noisier co-occurrence.
      NOTE this is exactly why the data-budget fix could NOT have fixed it: that equalized WINDOWS per learner,
      but homeostasis averages over STEPS, and a co-trained learner is still idle half the steps.
      PREDICTION: --homeostasis 0 closes most of dA/dB.

Compare DELTAS (dA/dB), not absolute corr: the banked controls were computed on the old Windows box and the new
CUDA/CuPy stack makes absolutes cross-stack. The runner executes cotrain/sepA/sepB/shared in ONE process per seed,
so dA/dB is a WITHIN-run delta and stack drift cancels to first order.

Usage: python research/runners/_cotrain_iso_aggregate.py <homeo0.json> [more...] --vs <homeo1.json> [more...]
"""
import json, sys, statistics as st


def load(paths):
    rows = []
    for p in paths:
        try:
            rows.extend(json.load(open(p)))
        except FileNotFoundError:
            print(f"  (missing: {p})")
    return {r["seed"]: r for r in rows}


def summarize(tag, by_seed):
    if not by_seed:
        return None
    dA = [r["dA_vs_sep"] for r in by_seed.values()]
    dB = [r["dB_vs_sep"] for r in by_seed.values()]
    both = dA + dB
    go = sum(r["GO"] for r in by_seed.values())
    print(f"\n{tag}  (n={len(by_seed)} seeds: {sorted(by_seed)})")
    print(f"  {'seed':<6}{'dA':>9}{'dB':>9}   {'co A/B':>15}  {'sep A/B':>15}  {'shared A/B':>15}  GO")
    for s in sorted(by_seed):
        r = by_seed[s]
        co, sa, sb, sh = r["cotrain"], r["sepA"], r["sepB"], r["shared"]
        print(f"  {s:<6}{r['dA_vs_sep']:>+9.4f}{r['dB_vs_sep']:>+9.4f}   "
              f"{co['corrA']:>7.3f}/{co['corrB']:<7.3f}{sa['corrA']:>7.3f}/{sb['corrB']:<7.3f}"
              f"{sh['corrA']:>7.3f}/{sh['corrB']:<7.3f}  {'GO' if r['GO'] else 'no'}")
    print(f"  MEAN dA {st.mean(dA):+.4f}   MEAN dB {st.mean(dB):+.4f}   MEAN both {st.mean(both):+.4f}"
          f"   |   GO {go}/{len(by_seed)}")
    # retained fidelity = co / sep, the headline the finding quotes as "~90-95%"
    ret = [by_seed[s]["cotrain"]["corrA"] / by_seed[s]["sepA"]["corrA"] for s in by_seed] + \
          [by_seed[s]["cotrain"]["corrB"] / by_seed[s]["sepB"]["corrB"] for s in by_seed]
    print(f"  retained fidelity co/sep: mean {st.mean(ret)*100:.1f}%  (min {min(ret)*100:.1f}%)")
    return {"mean_both": st.mean(both), "dA": dA, "dB": dB, "n": len(by_seed), "go": go}


def main():
    argv = sys.argv[1:]
    if "--vs" not in argv:
        print(__doc__)
        raise SystemExit(2)
    i = argv.index("--vs")
    h0 = load(argv[:i])
    h1 = load(argv[i + 1:])
    s0 = summarize("--homeostasis 0  (probe: homeostasis OFF)", h0)
    s1 = summarize("--homeostasis 1  (control: homeostasis ON = HEAD behavior; ALSO the clean post-budget-fix table)", h1)

    if s0 and s1:
        shared = sorted(set(h0) & set(h1))
        print(f"\n{'='*100}\nSINGLE-VARIABLE VERDICT (paired on the {len(shared)} shared seeds: {shared})")
        print(f"  {'seed':<6}{'d_both ON':>12}{'d_both OFF':>12}{'closed?':>10}")
        closed = []
        for s in shared:
            on = (h1[s]["dA_vs_sep"] + h1[s]["dB_vs_sep"]) / 2
            off = (h0[s]["dA_vs_sep"] + h0[s]["dB_vs_sep"]) / 2
            frac = (1 - abs(off) / abs(on)) * 100 if abs(on) > 1e-9 else float("nan")
            closed.append(frac)
            print(f"  {s:<6}{on:>+12.4f}{off:>+12.4f}{frac:>9.0f}%")
        print(f"\n  mean gap ON  {st.mean([(h1[s]['dA_vs_sep']+h1[s]['dB_vs_sep'])/2 for s in shared]):+.4f}")
        print(f"  mean gap OFF {st.mean([(h0[s]['dA_vs_sep']+h0[s]['dB_vs_sep'])/2 for s in shared]):+.4f}")
        print(f"  mean % of the residual CLOSED by disabling homeostasis: {st.mean(closed):.0f}%")
        print("\n  READ: >~60% closed on a MAJORITY of seeds => homeostatic threshold drift CONFIRMED as the")
        print("        dominant residual cause. ~0% or negative => REFUTED; the residual is elsewhere and the")
        print("        next candidate must come from a fresh read of the substrate, not another config sweep.")
        print("        Judge the SIGN+MAGNITUDE per seed, not the mean alone (n is small; report all seeds).")
    print()


if __name__ == "__main__":
    main()
