"""gap#5 R1 readout-fix probe: does the DECOUPLED forward-asymmetric store reactivate DISCRETELY under the
RANK-1 BISTABLE reactivation readout (the readout that reactivated the symmetric store), where the intrinsic-fatigue
de-latch readout gave ev=0?

2026-07-23 (scoped by research/findings/2026-07-23-gap5-decoupled-lr-encode-GO-readout-reactivation-roadblock.md,
"Next" step 1). The DECOUPLED-lr encode (within-lr 0.05 + chain-lr 0.5 + freeze_between_refresh) is a 6/6-GO WEIGHT
store: within ~206 (reactivation-scale) + a strong FORWARD bias (adj_fwd ~38 / adj_rev ~5, ratio ~7.65x). But the
intrinsic-fatigue spiking READOUT on that store gives ev=0 (no discrete reactivation), sweeping self_regen_read.

The scoped continuation (NOT blind sweeping): feed the SAME decoupled store into the RANK-1 BISTABLE reactivation
readout `_rest_and_detect` (from `_gap5_spontaneous_reactivation_derisk`, the readout that reactivated the symmetric
store) INSTEAD of the intrinsic-fatigue de-latch -- to SEPARATE "the store can't reactivate" from "the intrinsic-
fatigue de-latch (self_regen_read=0) prevents ignition." If the within~206 assembly reactivates DISCRETELY under the
bistable readout, the readout is the fix (bistable-ignite -> then de-latch to transition). If not, the honest negative
scopes the next mechanism (research gate: a targeted / DG-detonator ignition per Kandel Ch 54, or a sharper within-
attractor with feedback inhibition so a single assembly bursts discretely rather than smears).

THE MATCHED CONTROL (the load-bearing "store vs readout" separator): a SYMMETRIC reference store built by the SAME
encode machinery with `freeze_between_refresh=False` -- which leaves the ~137 SYMMETRIC between-links (the ~142 the
findings cite) that the freeze removes. That strong-between store is the one that reactivated DIFFUSELY (all assemblies
co-igniting). Running it under the SAME bistable readout confirms the readout DOES ignite on this substrate -> a
decoupled ev=0 is a STORE property (weak between-links), not a broken readout. cross_frac distinguishes a CLEAN single-
assembly reactivation (low cross_frac) from the diffuse co-fire (high cross_frac) the whole arc fights.

Reuse-by-import of the decoupled encode (`_prepare_sequence`) + the bistable readout (`_rest_and_detect`,
`_detect_events`, `_shuffle_within_weights`) + the ordered-replay diagnostic (`_detect_sequence_events`,
`_scramble_between_weights`). NO `sim/` edit.

CPU-smoke: SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_decoupled_store_bistable_readout_derisk \
    --seeds 42 --n-ca3 2000 --rest-steps 1000
Full run (GPU): SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap5_decoupled_store_bistable_readout_derisk \
    --seeds 42 43 44 100 101 102 --n-ca3 2000 --rest-steps 1500
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

# the DECOUPLED forward-asymmetric encode (== _gap5_R1_hetero_encode_sweep's 6/6-GO store)
from research.runners._gap5_sequence_replay_derisk import (  # noqa: E402
    _prepare_sequence, SEQ_CFG, _detect_sequence_events, _scramble_between_weights,
)
# the RANK-1 BISTABLE reactivation readout (the readout that reactivated the symmetric store) + its anti-cheat ops
from research.runners._gap5_spontaneous_reactivation_derisk import (  # noqa: E402
    _rest_and_detect, _shuffle_within_weights, _noise_label, GO_CFG as _SPONT_GO_CFG,  # noqa: F401
)

OUT = _REPO / "research" / "findings" / "raw" / "_gap5_decoupled_store_bistable_readout_derisk.json"

# The DECOUPLED-lr forward-asymmetric GO store (== pool_dec_s42.json / _gap5_R1_hetero_encode_sweep 6/6 GO):
#   within-lr 0.05 + chain-lr 0.5 + freeze_between_refresh -> within ~206 + adj_fwd ~38 / adj_rev ~5 (ratio ~7.65x).
# Every knob explicit so the JSON records the exact store this readout is tested on.
DECOUPLED_CFG = dict(SEQ_CFG)
DECOUPLED_CFG.update(
    n_mem=3, within_events=30, within_refresh=8, chain_fwd=24, chain_rev=0,
    rank1_encode=True, overlap_draw=False, encode_btsp_hetero=0.0,
    freeze_between_refresh=True, chain_rule="btsp",
    btsp_lr=0.05,          # WITHIN-attractor lr (LOW -> reactivatable within ~206)
    chain_btsp_lr=0.5,     # CHAIN forward lr (HIGH -> strong forward bias adj_fwd/adj_rev ~7.65x)
)


def _weight_diag(prep):
    return dict(w_within=prep["w_within"], w_forward=prep.get("w_forward"), w_reverse=prep.get("w_reverse"),
                w_adj_fwd=prep.get("w_adj_fwd"), w_adj_rev=prep.get("w_adj_rev"),
                ratio_adj=(float(prep.get("w_adj_fwd", 0.0)) / max(abs(float(prep.get("w_adj_rev", 0.0))), 1e-6)),
                n_between_fwd=prep.get("n_between_fwd"), n_between_rev=prep.get("n_between_rev"),
                assembly_sizes=[int(len(a)) for a in prep["assemblies"]])


def one_seed(seed, cfg, noise_specs, rest_steps, W, ev_floor, ev_k, min_frac, assembly_idx,
             active_frac, onset_frac):
    t0 = time.time()
    out = {"seed": seed}

    # -- BUILD the DECOUPLED forward-asymmetric store (the store under test) --
    prep = _prepare_sequence(seed, cfg, do_encode=True)
    out["encode_decoupled"] = _weight_diag(prep)
    print(f"  [seed {seed}] DECOUPLED store: within={prep['w_within']:.1f} adj_fwd={prep['w_adj_fwd']:.2f} "
          f"adj_rev={prep['w_adj_rev']:.2f} ratio={out['encode_decoupled']['ratio_adj']:.2f}x "
          f"(n_fwd={prep['n_between_fwd']} n_rev={prep['n_between_rev']}) ({time.time()-t0:.0f}s)", flush=True)

    # -- GO: sweep the noise level on the decoupled bristable store (reuse ONE frozen bridge -> safe). Primary metric =
    #    single-assembly DISCRETE reactivation (_detect_events on assembly `assembly_idx`, via _rest_and_detect). --
    go_runs = {}
    best_ns, best, best_F = None, None, None
    for ns in noise_specs:
        ev, F = _rest_and_detect(prep, ns, rest_steps, seed, assembly_idx=assembly_idx, W=W,
                                 ev_floor=ev_floor, ev_k=ev_k, min_frac=min_frac)
        go_runs[_noise_label(ns)] = ev
        print(f"  [seed {seed}] GO {_noise_label(ns):>22}: events={ev['n_events']:>3} specific={ev['n_specific']:>3} "
              f"rate/1k={ev['event_rate_per1k']:.2f} duty={ev['duty_cycle']:.3f} memb={ev['member_frac']:.3f} "
              f"rand={ev['random_frac']:.3f} cross={ev['cross_frac']:.3f} spec={ev['specificity']:+.3f} "
              f"peak={ev['event_peak_frac']:.3f} pop={ev['pop_rate']:.4f} frozen={ev['weights_frozen']} "
              f"apical_max={ev['apical_rest_max']} latched={ev['apical_n_latched']} ({time.time()-t0:.0f}s)", flush=True)
        score = (ev["n_specific"], ev["specificity"])
        if best is None or score > best:
            best, best_ns, best_F = score, ns, F
    out["go_runs"] = go_runs
    out["best_noise"] = _noise_label(best_ns)
    go = go_runs[_noise_label(best_ns)]

    # ordered-replay DIAGNOSTIC on the best decoupled run (does the forward-asymmetric store, IF it reactivates, produce
    # forward-ordered multi-assembly replay? -- the ultimate RANK-2 question; secondary to discrete single-assy reactivation)
    seq = _detect_sequence_events(best_F, prep["assemblies_local"], W=W, ev_floor=ev_floor, ev_k=ev_k,
                                  active_frac=active_frac, onset_frac=onset_frac)
    out["ordered_replay_diag"] = {k: seq[k] for k in ("n_events", "n_multi", "n_full", "forward_frac", "reverse_frac",
                                                       "forward_frac_full", "reverse_frac_full", "mean_tau",
                                                       "chance_forward", "duty_cycle", "pop_rate", "per_asm_active")}
    print(f"  [seed {seed}] ORDERED-DIAG {out['best_noise']}: events={seq['n_events']} multi={seq['n_multi']} "
          f"full={seq['n_full']} FWD={seq['forward_frac']:.3f} REV={seq['reverse_frac']:.3f} "
          f"chance={seq['chance_forward']:.3f} asm_active={seq['per_asm_active']} ({time.time()-t0:.0f}s)", flush=True)

    # -- ANTI-CHEATS at best noise --
    # NO-NOISE on the SAME encoded bridge -> must be SILENT (acid test for the self-sustaining artifact)
    nn, _ = _rest_and_detect(prep, ("none",), rest_steps, seed, assembly_idx=assembly_idx, W=W,
                             ev_floor=ev_floor, ev_k=ev_k, min_frac=min_frac)
    out["nonoise"] = nn
    print(f"  [seed {seed}] NO-NOISE (acid): events={nn['n_events']} specific={nn['n_specific']} "
          f"duty={nn['duty_cycle']:.4f} memb={nn['member_frac']:.3f} pop={nn['pop_rate']:.5f} "
          f"apical_max={nn['apical_rest_max']} latched={nn['apical_n_latched']} ({time.time()-t0:.0f}s)", flush=True)

    # NO-ENCODING (fresh bridge, store skipped, same noise) -> no assembly-specific events
    prep_ne = _prepare_sequence(seed, cfg, do_encode=False)
    ne, _ = _rest_and_detect(prep_ne, best_ns, rest_steps, seed, assembly_idx=assembly_idx, W=W,
                             ev_floor=ev_floor, ev_k=ev_k, min_frac=min_frac)
    out["noencode"] = ne
    print(f"  [seed {seed}] NO-ENCODE {out['best_noise']}: events={ne['n_events']} specific={ne['n_specific']} "
          f"memb={ne['member_frac']:.3f} rand={ne['random_frac']:.3f} spec={ne['specificity']:+.3f} "
          f"pop={ne['pop_rate']:.4f} w_within(prepare)={prep_ne['w_within']:.2f} ({time.time()-t0:.0f}s)", flush=True)

    # SHUFFLED within-assembly weights (fresh encoded bridge, scramble, same noise) -> no assembly-specific events
    prep_sh = _prepare_sequence(seed, cfg, do_encode=True)
    n_shuf = _shuffle_within_weights(prep_sh, seed)
    sh, _ = _rest_and_detect(prep_sh, best_ns, rest_steps, seed, assembly_idx=assembly_idx, W=W,
                             ev_floor=ev_floor, ev_k=ev_k, min_frac=min_frac)
    out["shuffled"] = sh; out["shuffled"]["n_within_shuffled"] = n_shuf
    print(f"  [seed {seed}] SHUFFLED-W {out['best_noise']}: shuffled {n_shuf} edges; events={sh['n_events']} "
          f"specific={sh['n_specific']} memb={sh['member_frac']:.3f} rand={sh['random_frac']:.3f} "
          f"spec={sh['specificity']:+.3f} pop={sh['pop_rate']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # -- POSITIVE CONTROL (the load-bearing STORE-vs-READOUT separator): the SYMMETRIC reference store == the SAME encode
    #    machinery with freeze_between_refresh=False -> the ~137 SYMMETRIC between-links remain (the store that reactivated
    #    DIFFUSELY). Same bistable readout: if it ignites here (events>0) but NOT on the decoupled store, ev=0 is a STORE
    #    property (weak between-links), NOT a broken readout. --
    cfg_sym = {**cfg, "freeze_between_refresh": False}
    prep_sym = _prepare_sequence(seed, cfg_sym, do_encode=True)
    out["encode_symmetric"] = _weight_diag(prep_sym)
    sym, Fsym = _rest_and_detect(prep_sym, best_ns, rest_steps, seed, assembly_idx=assembly_idx, W=W,
                                 ev_floor=ev_floor, ev_k=ev_k, min_frac=min_frac)
    seq_sym = _detect_sequence_events(Fsym, prep_sym["assemblies_local"], W=W, ev_floor=ev_floor, ev_k=ev_k,
                                      active_frac=active_frac, onset_frac=onset_frac)
    out["symmetric_readout"] = sym
    out["symmetric_ordered_diag"] = {k: seq_sym[k] for k in ("n_events", "n_multi", "per_asm_active", "duty_cycle")}
    print(f"  [seed {seed}] SYM-CTRL (freeze OFF, within={prep_sym['w_within']:.1f} adj_fwd={prep_sym['w_adj_fwd']:.2f} "
          f"adj_rev={prep_sym['w_adj_rev']:.2f} n_fwd={prep_sym['n_between_fwd']}): events={sym['n_events']} "
          f"specific={sym['n_specific']} memb={sym['member_frac']:.3f} cross={sym['cross_frac']:.3f} "
          f"duty={sym['duty_cycle']:.3f} pop={sym['pop_rate']:.4f} co-fire_asm_active={seq_sym['per_asm_active']} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # -- PER-SEED VERDICT (the scoped question: does the DECOUPLED store reactivate DISCRETELY under the bistable readout?) --
    specific_events = (go["n_specific"] >= 1 and go["member_frac"] >= min_frac
                       and go["member_frac"] > 2.0 * (go["random_frac"] + 1e-6))
    discrete = (go["duty_cycle"] <= 0.40)                                          # NOT a continuous ON state
    acid_noise_off = (nn["n_specific"] == 0 and nn["assembly_rest_frac"] < 0.05)   # NO-NOISE -> silent
    frozen_ok = bool(go["weights_frozen"] and nn["weights_frozen"])
    dendrite_reset_ok = (go["apical_rest_max"] is None
                         or go["apical_rest_max"] <= float(_SPONT_GO_CFG["plateau_v_hold"]) + 1e-3)
    noencode_retired = (ne["n_specific"] == 0 or ne["member_frac"] < 0.5 * max(go["member_frac"], 1e-6))
    shuffle_retired = (sh["n_specific"] == 0 or sh["member_frac"] < 0.5 * max(go["member_frac"], 1e-6))
    permuted_retired = (go["member_frac"] > 2.0 * (go["random_frac"] + 1e-6))
    # clean single-assembly (LOW cross_frac) vs diffuse co-fire (HIGH cross_frac -- the co-ignition the arc fights)
    clean_single_assembly = (go["n_specific"] >= 1 and go["cross_frac"] < 0.5 * (go["member_frac"] + 1e-6))
    readout_works = (sym["n_events"] >= 1)   # the bistable readout DOES ignite the strong-between symmetric store

    decoupled_reactivates = bool(specific_events and discrete and acid_noise_off and frozen_ok
                                 and dendrite_reset_ok and noencode_retired and shuffle_retired and permuted_retired)
    out["checks"] = dict(specific_events=specific_events, discrete=discrete, acid_noise_off=acid_noise_off,
                         frozen_ok=frozen_ok, dendrite_reset_ok=dendrite_reset_ok, noencode_retired=noencode_retired,
                         shuffle_retired=shuffle_retired, permuted_retired=permuted_retired,
                         clean_single_assembly=clean_single_assembly, readout_works_on_symmetric=readout_works)
    out["decoupled_reactivates"] = decoupled_reactivates
    out["readout_works"] = readout_works
    out["seed_go"] = decoupled_reactivates
    print(f"  [seed {seed}] => decoupled_reactivates={'YES' if decoupled_reactivates else 'no'} "
          f"readout_works_on_symmetric={'YES' if readout_works else 'no'} checks={out['checks']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-ca3", type=int, default=2000, help="the store only completes at 2000 (RANK 1 finding)")
    ap.add_argument("--n-mem", type=int, default=3)
    ap.add_argument("--assembly-idx", type=int, default=0, help="the within~206 attractor to test reactivation on (0)")
    # noise == the RANK-1 spontaneous readout's validated CA3-EXC-targeted Poisson sweep (the readout that reactivated
    # the symmetric store); the intrinsic-fatigue readout that gave ev=0 is a DIFFERENT readout.
    ap.add_argument("--noise", choices=["poisson", "ou"], default="poisson")
    ap.add_argument("--poisson-rate", type=float, default=0.01)
    ap.add_argument("--poisson-pa", type=float, nargs="+", default=[500.0, 1000.0, 2000.0])
    ap.add_argument("--poisson-dur", type=int, default=5)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[100.0, 200.0, 400.0])
    ap.add_argument("--rest-steps", type=int, default=1000, help="smoke default 1000; full-run 1500")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ev-floor", type=float, default=0.5)
    ap.add_argument("--ev-k", type=float, default=4.0)
    ap.add_argument("--min-frac", type=float, default=0.30)
    ap.add_argument("--active-frac", type=float, default=0.12, help="ordered-replay diag: per-assembly peak ACTIVE frac")
    ap.add_argument("--onset-frac", type=float, default=0.08, help="ordered-replay diag: per-assembly ONSET frac")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    cfg = dict(DECOUPLED_CFG); cfg["n_ca3"] = int(a.n_ca3); cfg["n_mem"] = int(a.n_mem)
    if a.noise == "ou":
        noise_specs = [("ou", s) for s in a.sigmas]
    else:
        noise_specs = [("poisson", a.poisson_rate, p, a.poisson_dur) for p in a.poisson_pa]

    print(f"[gap5-dec-bistable] DECOUPLED store (within-lr {cfg['btsp_lr']} + chain-lr {cfg['chain_btsp_lr']} + "
          f"freeze={cfg['freeze_between_refresh']}) under the RANK-1 BISTABLE readout | n_ca3={cfg['n_ca3']} "
          f"n_mem={cfg['n_mem']} assy~{max(6, int(cfg['assembly_frac']*cfg['n_ca3']))} noise={a.noise} "
          f"levels={[_noise_label(n) for n in noise_specs]} rest_steps={a.rest_steps} seeds={a.seeds} "
          f"backend={os.environ.get('SIM_BACKEND','auto')}", flush=True)

    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(one_seed(s, cfg, noise_specs, a.rest_steps, a.window, a.ev_floor, a.ev_k, a.min_frac,
                                a.assembly_idx, a.active_frac, a.onset_frac))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None and per:
        n_react = sum(1 for p in per if p["decoupled_reactivates"])
        n_readout_ok = sum(1 for p in per if p["readout_works"])
        go = n_react >= max(1, (len(per) + 1) // 2)
        mg = [p["go_runs"][p["best_noise"]] for p in per]
        mm = float(np.mean([g["member_frac"] for g in mg])); mr = float(np.mean([g["random_frac"] for g in mg]))
        mc = float(np.mean([g["cross_frac"] for g in mg])); md = float(np.mean([g["duty_cycle"] for g in mg]))
        me = float(np.mean([g["n_events"] for g in mg])); ms = float(np.mean([g["n_specific"] for g in mg]))
        msym = float(np.mean([p["symmetric_readout"]["n_events"] for p in per]))
        if go:
            verdict = (f"READOUT-FIX GO {n_react}/{len(per)} -- the DECOUPLED forward-asymmetric store (within ~206, "
                       f"adj_fwd/adj_rev ~7.65x) DOES reactivate DISCRETELY under the RANK-1 bistable readout: "
                       f"events {me:.1f} specific {ms:.1f} member_frac {mm:.3f} vs random {mr:.3f} / cross-assembly "
                       f"{mc:.3f}, duty {md:.3f}. => the intrinsic-fatigue de-latch (self_regen_read=0) was preventing "
                       f"ignition; the bistable-ignite IS the readout fix. NEXT: bistable-ignite -> then de-latch to "
                       f"drive the forward transition (run the 6-seed GPU confirm).")
        elif n_readout_ok >= 1:
            verdict = (f"HONEST NEGATIVE {n_react}/{len(per)} -- the RANK-1 bistable readout IGNITES the strong-between "
                       f"SYMMETRIC store ({msym:.1f} events, {n_readout_ok}/{len(per)} seeds) but the weak-between "
                       f"DECOUPLED store does NOT reactivate discretely (events {me:.1f} specific {ms:.1f} member_frac "
                       f"{mm:.3f} duty {md:.3f}). => ev=0 is a STORE property (weak forward-asymmetric between-links do "
                       f"not spontaneously ignite a single within~206 assembly from noise), NOT a broken readout. "
                       f"Per THE LAW: the discrete single-assembly reactivation on weak-between stores is the open "
                       f"piece -- RESEARCH GATE the next ignition mechanism (targeted / DG-detonator ignition, Kandel "
                       f"Ch 54; or a sharper within-attractor with feedback inhibition so ONE assembly bursts "
                       f"discretely rather than smears).")
        else:
            verdict = (f"INCONCLUSIVE {n_react}/{len(per)} -- the bistable readout did NOT ignite even the strong-between "
                       f"symmetric control at this smoke scale/noise (sym events {msym:.1f}). Re-check n_ca3=2000 + the "
                       f"validated poisson pA/rate before concluding anything about the decoupled store.")
    else:
        go = False; n_react = 0; n_readout_ok = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"

    summary = {"probe": "gap5_decoupled_store_bistable_readout", "GO": go, "n_reactivate": n_react,
               "n_readout_works": n_readout_ok, "seeds": a.seeds,
               "decoupled_cfg": {k: cfg[k] for k in sorted(cfg)},   # every knob recorded
               "noise": a.noise, "noise_levels": [_noise_label(n) for n in noise_specs],
               "rest_steps": a.rest_steps, "window": a.window, "ev_floor": a.ev_floor, "ev_k": a.ev_k,
               "min_frac": a.min_frac, "assembly_idx": a.assembly_idx,
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118 + f"\n[gap5-dec-bistable] VERDICT: {verdict}\n[gap5-dec-bistable] wrote {a.out}\n"
          + "=" * 118, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
