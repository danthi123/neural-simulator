"""Option B REAL test (SPIKING, GPU): does a per-dimension winner-take-all (WTA) read of the accumulator recover
`sign(true s)` where the graded read-then-subtract gave cos 0.41?

The cheap-first numpy probe (`_b_bipolar_threshold_numpy_probe.py`) proved binarizing the bound vector to a per-dim
sign preserves the VSA unbind at 100% IF the sign is the TRUE sign. The in-network NEGATIVE
(`_b_innetwork_superposition_probe.py`) showed reading the two accumulator channels SEPARATELY and subtracting in
numpy gives a signed vector at only cos 0.41 (noise amplification of a small difference of correlated rates).

Option B (research synthesis): don't subtract-then-read; let the per-dim ON/OFF competition DECIDE the sign in the
analog stage (the common mode cancels in the competition; biology's push-pull as a DECISION, Kandel Ch 22 p543).
The accumulator already has mutual lateral inhibition `acc_on[k] -| acc_off[k]`. HARDENING it (cranking `w_opp`)
turns the soft shunt into a hard WTA: the winner fires, the loser silences, so `sign(bon' - boff')` reflects the
DIFFERENTIAL drive (= true s[k]) robustly. This probe reads the per-dim WTA SIGN as the bound vector and SWEEPS
`w_opp` to find the hard-WTA operating point.

Three arms per role (cleanup = numpy argmax oracle held constant; GATE = in-network == numpy):
  - numpy reference:  comp.bind_fact(fact) -> unbind                        (the truth the GATE compares to)
  - graded baseline:  raw (bon',boff') -> unbind                            (the in-network NEGATIVE arm, ~0.46-0.69)
  - BIPOLAR (Opt B):  onoff(sign(bon'-boff')) -> unbind                     (the per-dim WTA decision)
Plus the diagnostic `sign_agree` = mean(sign(bon'-boff') == sign(true s)) -- how well the WTA recovers the true sign.

  python -u -m research.findings.raw._b_bipolar_wta_spiking_probe --seed 42 \
      --w-opp-sweep 200 800 2000 5000 --out research/findings/raw/_b_bipolar_wta_spiking.json
"""
import argparse
import json
import numpy as np

from research.runners.core_sim_composition import CoreSimComposer
from research.findings.raw._b_innetwork_superposition_probe import (
    build_bind_accumulator_bridge, bind_fact_in_network, numpy_raw_superposition,
    onoff, _cos, ACC_OP, ROLES,
)


def eval_seed(seed, proj_dim, n_flat, n_attr, op):
    comp = CoreSimComposer(seed=seed, proj_dim=proj_dim)
    bridge, idx = build_bind_accumulator_bridge(seed, comp.D, op)
    usable = [w for w in comp.words if w not in ("AFFIRM", "NEGATE")]
    rng = np.random.default_rng(seed)

    def pick(k):
        return [str(x) for x in rng.choice(usable, size=k, replace=False)]

    facts = []
    for _ in range(n_flat):
        a, ac, p = pick(3)
        facts.append(({"agent": a, "action": ac, "patient": p}, ROLES))
    for _ in range(n_attr):
        a, ac, adj, noun = pick(4)
        facts.append(({"agent": a, "action": ac, "patient": noun, "attribute": adj},
                      ("agent", "action", "patient", "attribute")))

    n_total = 0
    n_graded = 0    # raw (bon',boff') unbind == numpy
    n_bipolar = 0   # onoff(sign(bon'-boff')) unbind == numpy
    sign_agree_list = []
    signed_cos_list = []
    for fact, roles in facts:
        B = comp.bind_fact(fact)                                  # numpy reference bound vector
        bon_p, boff_p = bind_fact_in_network(bridge, idx, comp, fact, op)
        s_acc = bon_p - boff_p
        raw_bon, raw_boff = numpy_raw_superposition(comp, fact)
        s_true = raw_bon - raw_boff
        signed_cos_list.append(_cos(s_acc, s_true))
        sign_agree_list.append(float(np.mean(np.sign(s_acc) == np.sign(s_true))))
        B_bipolar = onoff(np.sign(s_acc))                        # Option B: per-dim WTA decision
        B_graded = (bon_p, boff_p)                               # baseline: raw channels (in-network NEGATIVE arm)
        for role in roles:
            if role not in fact:
                continue
            e_on_np, e_off_np = comp._unbind_onoff(B, role)
            filler_np = comp._cleanup(e_on_np - e_off_np, comp.words)
            e_on_g, e_off_g = comp._unbind_onoff(B_graded, role)
            filler_g = comp._cleanup(e_on_g - e_off_g, comp.words)
            e_on_b, e_off_b = comp._unbind_onoff(B_bipolar, role)
            filler_b = comp._cleanup(e_on_b - e_off_b, comp.words)
            n_total += 1
            n_graded += int(filler_g == filler_np)
            n_bipolar += int(filler_b == filler_np)

    return {
        "seed": seed, "n_total": n_total,
        "graded_recovery": n_graded / max(n_total, 1),
        "bipolar_recovery": n_bipolar / max(n_total, 1),
        "mean_sign_agree": float(np.mean(sign_agree_list)),
        "mean_signed_cos": float(np.mean(signed_cos_list)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="multi-seed at the chosen w_opp")
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-flat", type=int, default=3)
    ap.add_argument("--n-attr", type=int, default=1)
    ap.add_argument("--w-opp-sweep", type=float, nargs="*", default=None,
                    help="sweep mutual-inhibition strength (the hard-WTA knob) on --seed")
    ap.add_argument("--w-opp", type=float, default=ACC_OP["w_opp"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {"sweep": {}, "per_seed": {}}

    if args.w_opp_sweep:
        for w_opp in args.w_opp_sweep:
            op = dict(ACC_OP); op["w_opp"] = w_opp
            r = eval_seed(args.seed, args.proj_dim, args.n_flat, args.n_attr, op)
            results["sweep"][w_opp] = r
            print(f"[B-WTA sweep] w_opp={w_opp:.0f}: bipolar={r['bipolar_recovery']:.3f} "
                  f"graded={r['graded_recovery']:.3f} sign_agree={r['mean_sign_agree']:.3f} "
                  f"signed_cos={r['mean_signed_cos']:.4f}", flush=True)
        best_w = max(results["sweep"], key=lambda w: results["sweep"][w]["bipolar_recovery"])
        print(f"[B-WTA] best w_opp={best_w:.0f} bipolar={results['sweep'][best_w]['bipolar_recovery']:.3f}")
        args.w_opp = best_w

    seeds = args.seeds if args.seeds else [args.seed]
    op = dict(ACC_OP); op["w_opp"] = args.w_opp
    for s in seeds:
        r = eval_seed(s, args.proj_dim, args.n_flat, args.n_attr, op)
        results["per_seed"][s] = r
        print(f"[B-WTA] seed {s} (w_opp={args.w_opp:.0f}): bipolar={r['bipolar_recovery']:.3f} "
              f"graded={r['graded_recovery']:.3f} sign_agree={r['mean_sign_agree']:.3f}", flush=True)

    min_bip = min(results["per_seed"][s]["bipolar_recovery"] for s in seeds)
    mean_bip = float(np.mean([results["per_seed"][s]["bipolar_recovery"] for s in seeds]))
    min_grad = min(results["per_seed"][s]["graded_recovery"] for s in seeds)
    verdict = "GO" if min_bip >= 0.95 else ("PARTIAL" if mean_bip > min_grad + 0.1 else "NEGATIVE")
    results["min_bipolar"] = min_bip; results["mean_bipolar"] = mean_bip
    results["min_graded"] = min_grad; results["w_opp"] = args.w_opp; results["verdict"] = verdict
    print(f"\n[B-WTA ROBUST] min_bipolar={min_bip:.3f} mean_bipolar={mean_bip:.3f} min_graded(baseline)={min_grad:.3f}")
    print(f"[VERDICT] per-dim WTA sign read recovers unbind parity -> {verdict} "
          f"(GATE: per-seed bipolar >= 0.95; cf. in-network graded NEGATIVE ~0.46-0.69)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=float)
        print(f"[B-WTA] wrote {args.out}")


if __name__ == "__main__":
    main()
