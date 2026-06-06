"""OPTION (a) FULL-PIPELINE follow-up (only meaningful because the isolation passed): on-bridge GRADED whitening
(the shipped cp_graded_lateral learning) -> PHASE read-out -> composer, gated on COMPOSITION (2026-06-06).

The isolation (phase_handoff_decorrelation_compose.py) proved the READ CHANNEL was the boundary: reading the
KNOWN-100%-composing code out in PHASE (vs the saturating clip-RATE) recovers composition 100%. This runner closes
the loop the task asks for IF the isolation passes: does reading the ON-BRIDGE GRADED LATERAL's learned whitening
out in PHASE make the on-substrate spiking whitening compose ~100%?

HONEST PRIOR (stated before the run): the boundary localized TWO things about the graded lateral — (1) the RATE
read-out degrades the gentle code (option a FIXES this), AND (2) the graded lateral itself lands in the
OVER-WHITENING regime (coh ~0.187, NOT the gentle composing coh ~0.043; coh 0.187 composes at the FLOOR per the
rate-model arc). Phase is magnitude-invariant -> it PRESERVES whatever coherence the graded lateral produces. So if
the graded lateral over-whitens to 0.187, the phase read-out faithfully carries 0.187 -> still the floor. The
read-channel is fixed; the over-whitening AMOUNT is a separate learning-rule/tuning issue phase cannot fix. This
run tests that prediction directly and reports it honestly either way.

Conditions (gated on composition, bracketed by RAW/CONCEPT):
  - GRADED-CLIP   : the boundary's graded readout (clip-rate `a`) -> composer  (== the 66.7% floor it reported).
  - GRADED-PHASE  : the SAME graded-lateral whitened code, but read out through the RF PHASE channel -> composer.
                    If phase fixes the full pipeline, this is ~100%; if the graded lateral over-whitens, this is
                    the floor (the read-out is fixed but the whitening AMOUNT is wrong).

Reuse-by-import; NO sim/ edits.
  REALOBJ_CIFAR=data/cifar10/cifar-10-batches-py/data_batch_1 \
      python -m research.runners.phase_handoff_fullpipeline_compose --seeds 42 43 44 --gain 40 --epochs 8 --signed --out out.json
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from research.runners.unified_agent_realobject_grounded import build_realobject_features
from research.runners.unified_agent_visual_grounded import _decorrelate
from research.runners.unified_agent_benchmark import build_vocab
from research.runners._visual_grounding_probe import _v1_matrix
from research.runners.graded_lgn_decorrelation_compose import graded_lgn_codes, coherence
from research.runners.phase_handoff_decorrelation_compose import compose, phase_direct_codes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--lam", type=float, default=0.01, help="graded_lateral_lambda (-lambda*M)")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--gain", type=float, default=40.0, help="graded_lateral_gain_pA (the boundary's signed run used 40)")
    ap.add_argument("--act-scale", type=float, default=15.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--drive-scale", type=float, default=1400.0)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--settle", type=int, default=15)
    ap.add_argument("--signed", action="store_true", help="ON/OFF 2K LGN pool (the boundary's pairwise-decorrelating variant)")
    ap.add_argument("--period", type=int, default=400, help="RF phasor period (400 = clean round-trip)")
    ap.add_argument("--bench-seeds", type=int, nargs="+", default=None)
    ap.add_argument("--out", default="research/findings/raw/_phase_handoff_fullpipeline.json")
    args = ap.parse_args()
    bench_seeds = args.bench_seeds if args.bench_seeds is not None else args.seeds

    nouns, verbs, adjs = build_vocab()
    W, _ = _v1_matrix()
    feats, dim, tokens, src = build_realobject_features(nouns, verbs, adjs, W, seed=args.seeds[0])
    print(f"=== OPTION (a) FULL-PIPELINE: graded lateral -> PHASE -> composer | {len(tokens)} concepts | K={args.K} "
          f"| gain={args.gain} epochs={args.epochs} signed={args.signed} ===", flush=True)

    out = {"source": src, "K": args.K, "gain": args.gain, "epochs": args.epochs, "signed": bool(args.signed),
           "seeds": args.seeds, "bench_seeds": bench_seeds}
    raw_ok, raw_tot = compose(feats, bench_seeds, tokens, nouns, verbs, adjs)
    cw_ok, cw_tot = compose(_decorrelate(feats), bench_seeds, tokens, nouns, verbs, adjs)
    out["RAW"] = [raw_ok, raw_tot]; out["CONCEPT-whiten"] = [cw_ok, cw_tot]
    print(f"  RAW floor {raw_ok}/{raw_tot}={raw_ok/raw_tot*100:.1f}%  |  CONCEPT-whiten {cw_ok}/{cw_tot}="
          f"{cw_ok/cw_tot*100:.1f}%", flush=True)
    harness_ok = (raw_ok / raw_tot < 0.85) and (cw_ok / cw_tot > 0.9)

    out["SEEDS"] = {}
    for s in args.seeds:
        # learn the ON-BRIDGE graded lateral (the boundary's mechanism); a_codes = the graded clip readout.
        a_codes, sp_codes, guards = graded_lgn_codes(
            feats, s, args.K, args.lam, args.lr, args.gain, args.act_scale, args.epochs,
            args.drive_scale, args.window, args.settle, proj_seed=s, signed=args.signed)
        gc_ok, gc_tot = compose(a_codes, bench_seeds, tokens, nouns, verbs, adjs)        # GRADED-CLIP (boundary)
        # route the SAME graded-whitened code out through the RF PHASE channel.
        Y_ph, gph = phase_direct_codes(a_codes, s, args.period, 0)
        gp_ok, gp_tot = compose(Y_ph, bench_seeds, tokens, nouns, verbs, adjs)            # GRADED-PHASE
        entry = {"graded_clip_compose": [gc_ok, gc_tot], "graded_phase_compose": [gp_ok, gp_tot],
                 "graded_coh": guards["graded_coh_mean"], "phase_read_coh": gph["read_coh_mean"],
                 "phase_roundtrip_corr": gph["roundtrip_phase_corr"], "phase_frac_zero": gph["frac_zero_phase"],
                 "n_silent_graded": guards["n_silent_graded"], "M_norm": guards["M_norm"]}
        out["SEEDS"][str(s)] = entry
        print(f"  [seed={s}] GRADED-CLIP {gc_ok}/{gc_tot}={gc_ok/gc_tot*100:.1f}% (coh {guards['graded_coh_mean']:.3f})"
              f"  |  GRADED-PHASE {gp_ok}/{gp_tot}={gp_ok/gp_tot*100:.1f}% (phase preserves coh "
              f"{gph['read_coh_mean']:.3f}, roundtrip {gph['roundtrip_phase_corr']:.3f})", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out}", flush=True)

    raw_rate = raw_ok / raw_tot
    gc_rates = [v["graded_clip_compose"][0] / v["graded_clip_compose"][1] for v in out["SEEDS"].values()]
    gp_rates = [v["graded_phase_compose"][0] / v["graded_phase_compose"][1] for v in out["SEEDS"].values()]
    print("\n" + "=" * 88, flush=True)
    print(f"VERDICT (OPTION a FULL-PIPELINE, {len(args.seeds)} seed(s)):", flush=True)
    print(f"  RAW floor {raw_rate*100:.1f}%  |  CONCEPT-whiten {cw_ok/cw_tot*100:.1f}%", flush=True)
    print(f"  GRADED-CLIP  (boundary readout): {['%.1f%%' % (r*100) for r in gc_rates]} (mean {np.mean(gc_rates)*100:.1f}%)",
          flush=True)
    print(f"  GRADED-PHASE (phase read-out)  : {['%.1f%%' % (r*100) for r in gp_rates]} (mean {np.mean(gp_rates)*100:.1f}%)",
          flush=True)
    if not harness_ok:
        verdict = "INVALID - controls did not bracket"
    elif np.mean(gp_rates) >= 0.95 and min(gp_rates) >= 0.90:
        verdict = "GO - the on-bridge GRADED whitening read out in PHASE COMPOSES ~100% (whitening resolved spike-native)"
    elif np.mean(gp_rates) > np.mean(gc_rates) + 0.1:
        verdict = (f"PARTIAL - phase read-out lifts the graded pipeline above its clip readout "
                   f"({np.mean(gp_rates)*100:.1f}% vs {np.mean(gc_rates)*100:.1f}%) but below target")
    else:
        verdict = (f"BOUNDARY (over-whitening, NOT the read-out) - phase faithfully carries the graded lateral's "
                   f"coherence but the graded lateral OVER-WHITENS (coh ~0.19 vs composing ~0.04), so PHASE composes "
                   f"at the floor ({np.mean(gp_rates)*100:.1f}%). The read channel is fixed (isolation GO); the "
                   f"whitening AMOUNT is the remaining issue.")
    print(f"  => {verdict}", flush=True)
    print("=" * 88, flush=True)


if __name__ == "__main__":
    main()
