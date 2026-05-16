"""Increment G1 Task 10 -- the PRE-REGISTERED held-out anti-cheat gate.

This orchestrator does NOT train. It READS the already-trained
song_g1.ckpt.npz + its frozen sidecar (song_g1.ckpt.npz.meta.json,
written ONCE at train start by research.runners.song_g1_train) and emits
the definitive, pre-registered G1 verdict on the HELD-OUT propositions
that were NEVER trained.

Six LOAD-BEARING anti-cheat invariants (pre-registered in the
implementation plan; violating ANY one silently invalidates the G1
verdict -- see "Pre-registration corrections 2/3/4" + Task 10):

(1) SIDECAR-FROZEN g1_abstain -- NEVER recompute, NEVER 650. The
    abstention floor for `gate_cleared` is EXACTLY the value Step 0
    froze at TRAIN START: `meta["calibration"]["g1_abstain"]` (the
    control-max separation point in the self_comprehend NO-DRIVE
    residual regime; the run that produced it had `smoke:false`). This
    gate does NOT re-run Step-0 calibration, does NOT recompute a floor,
    and does NOT use the literal 650 (650 was calibrated on the
    CONTINUOUS-DRIVE stim_recall regime -- a different magnitude regime,
    not comparable to the no-drive integrated residual this gate
    decodes; see corrections 2/3). If the sidecar is missing OR its
    `smoke` flag is True (a smoke-calibrated sidecar must NEVER gate the
    real verdict), this gate exits NON-ZERO (code 2 = not runnable) with
    a clear message -- it does NOT fall back to 650 and does NOT
    recompute.

(2) HELD-OUT ONLY, INTEGRATED decode (corrections 2 / M3). Evaluate
    ONLY the sidecar's frozen `heldout_propositions` (never trained).
    For each: rebuild the TRAINED SongHVC from the checkpoint (W from
    sim.train_checkpoint.load_checkpoint -- NOT re-randomized; the exact
    trainer resume idiom), `song.rollout(intention, len(intended))` ->
    the produced ORDERED concept sequence, then the M3 integrated
    decode (ignite the WHOLE ordered sequence ONCE via
    song_g1_ignite.ignite_sequence, THEN song_g1_ignite.self_comprehend
    ONCE on the integrated post-sequence residual -- NEVER per-slot).
    We REUSE song_g1_train._integrated_decode verbatim (DRY -- the exact
    M3 path Task 9 trained with; not re-implemented).
    `gate_cleared = (integrated decode top-rate >= sidecar g1_abstain)`;
    `true_score = score_order(decoded, intended)` (UNMODIFIED
    song_g1_core).

(3) PERMUTED-ORDER control is load-bearing. For each held-out prop:
    `permuted_order_controls(intended, rng, perm_n)` (UNMODIFIED
    song_g1_core; deterministic per-prop seed). Ignite each permuted
    ORDER through the SAME integrated-decode path; score each;
    `best_perm_score = max`. If `permuted_order_controls` returns []
    (a degenerate intended -- e.g. all-same multiset), that proposition
    CANNOT be gated: it is recorded EXCLUDED and NOT counted as PASS
    (consistent with g1_verdict's `best_perm_score > 0` guard).

(4) PURE pre-registered `g1_verdict` UNCHANGED. Per held-out prop:
    `v = g1_verdict(true_score, best_perm_score, gate_cleared)` (the
    UNMODIFIED song_g1_core function: PASS iff gate_cleared AND
    best_perm_score>0 AND true_score>=_G1_ABS_FLOOR(0.5) AND
    true_score>=best_perm*1.10). AGGREGATE verdict =
    `g1_verdict(mean_true_score, mean_best_perm_score,
    all_gate_cleared)` over the NON-EXCLUDED held-out props, where
    `all_gate_cleared` = every counted held-out prop cleared its gate.
    This REUSES g1_verdict on the means -- it does NOT invent a new
    aggregate rule. The FIXED bars (_G1_MARGIN=0.10, _G1_ABS_FLOOR=0.5)
    are NEVER touched here.

(5) Exit codes: 0 == verdict computed + written (PASS or FAIL are BOTH
    valid computed results -> 0); 2 == ckpt/sidecar not ready or
    smoke-tagged (not runnable -> the controller polls / fixes).

(6) HONEST PROPAGATION is the CONTROLLER's job (post-real-run). main()
    is compute+print+JSON ONLY: it writes song_g1_gate.json with full
    per-prop detail + the aggregate verdict + `g1_abstain_source:
    "sidecar-frozen"` + `meta_smoke:false`. It does NOT write the
    findings doc / capability_status (the controller does that honest
    propagation after the real run).

Heavy imports / IO are lazy (inside main): loading 5 sparse 320 bridges
is slow + GPU-bound (several minutes expected). `aggregate_verdict` and
`_check_sidecar_usable` are PURE (no IO, no heavy import) so they are
CPU-unit-tested without a checkpoint.

ASCII-only output (Windows cp1252 safe).
"""
from __future__ import annotations

import argparse

# Default checkpoint + sidecar (the trainer's _CKPT_DEFAULT; the sidecar
# is `<ckpt>.meta.json`, written ONCE at train start with the FROZEN
# train/held-out propositions + the Step-0 control-calibrated
# g1_abstain + a "smoke" flag).
_CKPT_DEFAULT = "research/findings/raw/g11_bg/song_g1.ckpt.npz"
_OUT_DEFAULT = "research/findings/raw/g11_bg/song_g1_gate.json"

# SongHVC chain geometry -- MUST match exactly what song_g1_train built
# (and the no-harm probe constructed): SongHVC(8, 64, seed). Mismatching
# these would rebuild a differently-shaped chain and silently misread
# the trained weights.
_SONG_N_STATES = 8
_SONG_N_CONCEPTS = 64

# Drive / decode windows for the M3 integrated decode. These mirror the
# validated stim_recall_sparse_rates magnitudes the comprehension path
# uses and MUST match song_g1_train's values (the trained chain + the
# Step-0 g1_abstain were both measured in this exact regime).
_DRIVE_PA = 1500.0
_STEPS_PER = 100
_DECODE_WINDOW = 100


def _sidecar_path(ckpt_path: str) -> str:
    """The trainer's sidecar path idiom (DRY): `<ckpt>.meta.json`."""
    return ckpt_path + ".meta.json"


def _check_sidecar_usable(meta):
    """PURE: may this sidecar gate the REAL G1 verdict? (no IO).

    Returns (ok: bool, reason: str). The sidecar is usable ONLY if:
      * it exists (meta is not None), AND
      * meta["smoke"] is NOT True -- a smoke-calibrated sidecar must
        NEVER gate the real verdict (its g1_abstain was frozen on a
        2-epoch / 2-prop toy run; absent key treated as full=False
        per the trainer's own additive-JSON contract), AND
      * meta["calibration"]["g1_abstain"] is present (the Step-0
        control-max floor frozen at TRAIN START).

    This is the anti-cheat guard for invariant (1): a missing or
    smoke-tagged sidecar is "not runnable" (caller exits code 2); we
    NEVER fall back to 650 and NEVER recompute the floor.
    """
    if meta is None:
        return False, "sidecar missing (no <ckpt>.meta.json)"
    if not isinstance(meta, dict):
        return False, "sidecar malformed (not a JSON object)"
    if bool(meta.get("smoke", False)):
        return False, ("sidecar is smoke-calibrated (smoke=True); a "
                       "smoke sidecar must NEVER gate the real G1 "
                       "verdict")
    calib = meta.get("calibration")
    if not isinstance(calib, dict) or "g1_abstain" not in calib:
        return False, ("sidecar has no calibration.g1_abstain "
                       "(Step-0 floor not frozen)")
    return True, "ok"


def aggregate_verdict(per_prop):
    """PURE aggregate G1 verdict over the held-out propositions (no IO).

    Reuses the UNMODIFIED pre-registered `g1_verdict` on the MEANS of
    the NON-EXCLUDED held-out propositions -- it does NOT invent a new
    aggregate rule (invariant 4).

    Parameters
    ----------
    per_prop : list of dict
        One dict per held-out proposition. Each MUST carry:
          "excluded"      : bool (True iff no permuted-ORDER control
                             existed -> cannot be gated; not counted)
          "true_score"    : float
          "best_perm_score": float
          "gate_cleared"  : bool
        (Extra keys are ignored -- callers add per-prop detail.)

    Aggregate rule (pre-registered):
      * counted = the non-excluded props (excluded props are NEVER
        counted as PASS -- consistent with g1_verdict's
        best_perm_score>0 guard).
      * mean_true_score / mean_best_perm_score = means over `counted`.
      * all_gate_cleared = EVERY counted prop cleared its gate.
      * AGGREGATE = g1_verdict(mean_true_score, mean_best_perm_score,
                               all_gate_cleared)  [UNMODIFIED].
      * zero counted props (all excluded / empty) -> aggregate FAIL
        with all-zero means (no evidence of ORDER-learning), reusing
        g1_verdict's own FAIL for ts=ps=0 / gate_cleared=False.

    Returns the g1_verdict dict (true_score == mean_true_score, etc.)
    augmented with: n_props, n_excluded, n_counted, n_gate_cleared,
    n_prop_pass (count of per-prop PASS over counted), all_gate_cleared,
    excluded_props (their identifiers if provided via "intention").
    """
    from research.runners.song_g1_core import g1_verdict

    n_props = len(per_prop)
    counted = [p for p in per_prop if not bool(p.get("excluded", False))]
    n_counted = len(counted)
    n_excluded = n_props - n_counted

    if n_counted == 0:
        # No gate-able held-out prop -> NO evidence of ORDER-learning.
        # Reuse g1_verdict's own FAIL (ts=ps=0, gate_cleared=False);
        # do NOT invent a separate aggregate rule.
        agg = g1_verdict(0.0, 0.0, False)
        agg.update({
            "n_props": n_props,
            "n_excluded": n_excluded,
            "n_counted": 0,
            "n_gate_cleared": 0,
            "n_prop_pass": 0,
            "all_gate_cleared": False,
            "mean_true_score": 0.0,
            "mean_best_perm_score": 0.0,
            "excluded_props": [p.get("intention") for p in per_prop
                               if bool(p.get("excluded", False))],
            "aggregate_note": ("no non-excluded held-out propositions "
                               "-> no ORDER-learning evidence -> FAIL"),
        })
        return agg

    mean_true = sum(float(p["true_score"]) for p in counted) / n_counted
    mean_perm = (sum(float(p["best_perm_score"]) for p in counted)
                 / n_counted)
    n_gate_cleared = sum(1 for p in counted
                         if bool(p.get("gate_cleared", False)))
    all_gate_cleared = (n_gate_cleared == n_counted)

    # per-prop PASS count (each via the UNMODIFIED g1_verdict) -- a
    # transparency number; the AGGREGATE verdict is g1_verdict on means.
    n_prop_pass = 0
    for p in counted:
        pv = g1_verdict(float(p["true_score"]),
                        float(p["best_perm_score"]),
                        bool(p.get("gate_cleared", False)))
        if pv["gate"]:
            n_prop_pass += 1

    agg = g1_verdict(mean_true, mean_perm, all_gate_cleared)
    agg.update({
        "n_props": n_props,
        "n_excluded": n_excluded,
        "n_counted": n_counted,
        "n_gate_cleared": n_gate_cleared,
        "n_prop_pass": n_prop_pass,
        "all_gate_cleared": bool(all_gate_cleared),
        "mean_true_score": float(mean_true),
        "mean_best_perm_score": float(mean_perm),
        "excluded_props": [p.get("intention") for p in per_prop
                           if bool(p.get("excluded", False))],
    })
    return agg


def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--ckpt", type=str, default=_CKPT_DEFAULT,
        help="trained SongHVC checkpoint .npz (its sidecar "
             "<ckpt>.meta.json holds the FROZEN train/held-out "
             "propositions + the Step-0 control-calibrated g1_abstain "
             "+ a 'smoke' flag). Default: %(default)s")
    ap.add_argument(
        "--perm-n", type=int, default=8,
        help="permuted-ORDER controls per held-out proposition "
             "(distinct non-identity orderings of the SAME concept "
             "multiset; the load-bearing anti-cheat). Default: "
             "%(default)s")
    ap.add_argument(
        "--seed", type=int, default=42,
        help="seed for the bridge load AND the deterministic "
             "permuted-ORDER control RNG. Default: %(default)s")
    ap.add_argument(
        "--out", type=str, default=_OUT_DEFAULT,
        help="where to write the verdict JSON. Default: %(default)s")
    return ap


def main():
    # --- lazy/heavy imports (loading 5 sparse 320 bridges is slow +
    #     GPU; aggregate_verdict / _check_sidecar_usable stay pure) ----
    import json
    import time
    from pathlib import Path

    import numpy as np

    args = _build_arg_parser().parse_args()

    sidecar = _sidecar_path(args.ckpt)

    print("=" * 64, flush=True)
    print("SONG G1 TASK 10 -- PRE-REGISTERED HELD-OUT ANTI-CHEAT GATE",
          flush=True)
    print("(sidecar-FROZEN g1_abstain; held-out ONLY; permuted-ORDER; "
          "pure g1_verdict)", flush=True)
    print("=" * 64, flush=True)

    # --- (1) sidecar gate: refuse missing / smoke-tagged -------------
    if not Path(args.ckpt).exists():
        print(f"[NOT READY] checkpoint not present: {args.ckpt}",
              flush=True)
        print("  (training not finished -- exit 2; controller polls)",
              flush=True)
        print("=" * 64, flush=True)
        return 2

    meta = None
    if Path(sidecar).exists():
        try:
            with open(sidecar, "r") as f:
                meta = json.load(f)
        except (ValueError, OSError) as e:
            print(f"[NOT READY] sidecar unreadable ({sidecar}): {e}",
                  flush=True)
            print("=" * 64, flush=True)
            return 2

    ok, reason = _check_sidecar_usable(meta)
    if not ok:
        print(f"[NOT READY] sidecar not usable: {reason}", flush=True)
        print(f"  sidecar = {sidecar}", flush=True)
        print("  REFUSING to fall back to literal 650 or recompute the "
              "floor (anti-cheat invariant 1).", flush=True)
        print("  Exit 2 (not a computed verdict -- not runnable).",
              flush=True)
        print("=" * 64, flush=True)
        return 2

    # The FROZEN floor -- EXACTLY the Step-0 value, never recomputed,
    # never 650. Assert the run that produced it was NOT a smoke run.
    meta_smoke = bool(meta.get("smoke", False))
    assert meta_smoke is False, (
        "smoke-tagged sidecar reached the gate body -- "
        "_check_sidecar_usable must have rejected it")
    g1_abstain = float(meta["calibration"]["g1_abstain"])
    heldout_props = list(meta.get("heldout_propositions", []))
    train_props = list(meta.get("train_propositions", []))
    all_props = train_props + heldout_props

    print(f"  sidecar             : {sidecar}", flush=True)
    print(f"  meta_smoke          : {meta_smoke}  (asserted False -- "
          f"a smoke sidecar may never gate the real verdict)",
          flush=True)
    print(f"  g1_abstain (FROZEN) : {g1_abstain:.2f}  "
          f"(Step-0 sidecar value; NOT 650, NOT recomputed)",
          flush=True)
    print(f"  operating criterion : "
          f"{meta['calibration'].get('operating_criterion', '?')}  "
          f"(self_comprehend NO-DRIVE residual regime)", flush=True)
    print(f"  held-out props      : {len(heldout_props)}  "
          f"(NEVER trained -- the only props this gate evaluates)",
          flush=True)

    if not heldout_props:
        print("[NOT READY] sidecar has no heldout_propositions to "
              "evaluate.", flush=True)
        print("=" * 64, flush=True)
        return 2

    t0 = time.time()

    # --- DRY reuse: trained chain, M3 decode, scoring, controls ------
    from research.runners.song_g1_core import (
        g1_verdict, permuted_order_controls, score_order,
    )
    from research.runners.song_g1_ignite import (
        ignite_sequence, load_members, self_comprehend,
    )
    # REUSE the trainer's exact M3 integrated-decode + inter-production
    # recovery (DRY -- do NOT re-implement the order-carrying decode).
    from research.runners.song_g1_train import (
        _integrated_decode, _recover, _seed_intention_biases,
        _IGNITE_RECOVERY,
    )
    from sim.song_hvc import SongHVC
    from sim.train_checkpoint import load_checkpoint

    # --- load the trained SongHVC weights from the checkpoint --------
    ckpt = load_checkpoint(args.ckpt)
    if ckpt is None:
        # exists() said yes but load failed -> treat as not-ready.
        print(f"[NOT READY] load_checkpoint returned None for "
              f"{args.ckpt}", flush=True)
        print("=" * 64, flush=True)
        return 2
    trained_epoch = int(ckpt["epoch"])

    # --- load the 5 validated G.20 320-sparse bridges (DRY) ----------
    print(f"\nLoading 5 sparse 320-tier bridges via "
          f"song_g1_ignite.load_members(seed={args.seed}) ...",
          flush=True)
    members = load_members(seed=int(args.seed))
    print(f"  loaded: {[m.name for m in members]} "
          f"({int(time.time()-t0)}s)", flush=True)

    # --- rebuild the TRAINED chain (W from ckpt; NOT re-randomized;
    #     exact trainer resume idiom) + re-seed intention biases from
    #     ALL frozen sidecar props so rollout reproduces training -----
    song = SongHVC(n_states=_SONG_N_STATES,
                   n_concepts=_SONG_N_CONCEPTS,
                   seed=int(args.seed))
    _seed_intention_biases(song, all_props)
    song.W = np.asarray(ckpt["weights"][0], dtype=np.float32).copy()
    print(f"  trained SongHVC restored from ckpt (epoch "
          f"{trained_epoch}; W NOT re-randomized)", flush=True)

    print(f"\n[GATE] held-out propositions ({len(heldout_props)}; "
          f"NEVER trained):", flush=True)
    for p in heldout_props:
        print(f"   int={p.get('intention')} {p.get('bridge')} "
              f"seq={p.get('concept_seq')} words={p.get('words')}",
              flush=True)

    # --- evaluate each HELD-OUT proposition (integrated decode) ------
    per_prop = []
    for p in heldout_props:
        intention = p["intention"]
        bidx = p["bridge_idx"]
        member = members[bidx]
        intended = list(p["concept_seq"])

        # produced ORDERED sequence from the TRAINED chain.
        produced = song.rollout(intention, len(intended))

        # M3: ignite the WHOLE produced ordered sequence ONCE, THEN
        # self_comprehend ONCE on the integrated residual (DRY: the
        # trainer's exact _integrated_decode). NEVER per-slot.
        decoded, top_rate = _integrated_decode(
            member, ignite_sequence, self_comprehend,
            produced, _DRIVE_PA, _STEPS_PER, _DECODE_WINDOW)
        gate_cleared = bool(top_rate >= g1_abstain)
        true_score = float(score_order(decoded, intended))
        _recover(member, _IGNITE_RECOVERY)

        # --- permuted-ORDER controls (load-bearing anti-cheat) -------
        # deterministic per-prop rng (seed + intention) so the controls
        # are reproducible and independent across props.
        perm_rng = np.random.default_rng(int(args.seed) * 1009
                                         + int(intention))
        perms = permuted_order_controls(intended, perm_rng,
                                        int(args.perm_n))

        if not perms:
            # degenerate intended (e.g. all-same multiset): NO
            # permuted-ORDER contrast exists -> this prop CANNOT be
            # gated. EXCLUDE it (not counted as PASS), consistent with
            # g1_verdict's best_perm_score>0 guard.
            v = g1_verdict(true_score, 0.0, gate_cleared)
            per_prop.append({
                "intention": intention,
                "bridge": p.get("bridge"),
                "concept_seq": intended,
                "words": p.get("words"),
                "produced": [int(x) for x in produced],
                "decoded": [int(x) for x in decoded],
                "top_rate": round(top_rate, 2),
                "gate_cleared": gate_cleared,
                "true_score": round(true_score, 4),
                "best_perm_score": 0.0,
                "perm_scores": [],
                "n_perm_controls": 0,
                "excluded": True,
                "exclude_reason": ("no non-identity permuted ORDER "
                                   "(degenerate multiset) -> cannot "
                                   "gate ORDER-learning"),
                "verdict": v,
            })
            print(f"  [EXC] int={intention} seq={intended} "
                  f"true={true_score:.3f} -- no permuted-ORDER "
                  f"control (EXCLUDED, not counted)", flush=True)
            continue

        perm_scores = []
        for perm in perms:
            p_dec, _p_rate = _integrated_decode(
                member, ignite_sequence, self_comprehend,
                perm, _DRIVE_PA, _STEPS_PER, _DECODE_WINDOW)
            perm_scores.append(float(score_order(p_dec, intended)))
            _recover(member, _IGNITE_RECOVERY)
        best_perm_score = max(perm_scores) if perm_scores else 0.0

        # per-prop verdict via the UNMODIFIED pre-registered g1_verdict.
        v = g1_verdict(true_score, best_perm_score, gate_cleared)
        per_prop.append({
            "intention": intention,
            "bridge": p.get("bridge"),
            "concept_seq": intended,
            "words": p.get("words"),
            "produced": [int(x) for x in produced],
            "decoded": [int(x) for x in decoded],
            "top_rate": round(top_rate, 2),
            "gate_cleared": gate_cleared,
            "true_score": round(true_score, 4),
            "best_perm_score": round(best_perm_score, 4),
            "perm_scores": [round(s, 4) for s in perm_scores],
            "n_perm_controls": len(perm_scores),
            "excluded": False,
            "exclude_reason": None,
            "verdict": v,
        })
        print(f"  [{'P' if v['gate'] else 'F'}] int={intention} "
              f"seq={intended} true={true_score:.3f} "
              f"best_perm={best_perm_score:.3f} "
              f"gate_cleared={'Y' if gate_cleared else 'N'} "
              f"top_rate={top_rate:.1f} -> {v['GATE']}", flush=True)

    # --- AGGREGATE verdict = g1_verdict on the means (PURE helper) ---
    agg = aggregate_verdict(per_prop)

    result = {
        "task": "song_g1 Task 10 pre-registered held-out anti-cheat "
                "gate",
        "substrate": "G.20 320-sparse 5-bridge",
        "seed": int(args.seed),
        "ckpt": args.ckpt,
        "sidecar": sidecar,
        "trained_epoch": trained_epoch,
        # invariant (1) provenance -- explicit + machine-checkable.
        "g1_abstain": g1_abstain,
        "g1_abstain_source": "sidecar-frozen",
        "g1_abstain_note": ("EXACTLY meta['calibration']['g1_abstain'] "
                            "frozen by Step 0 at TRAIN START; NOT the "
                            "literal 650 (different magnitude regime), "
                            "NOT recomputed here"),
        "meta_smoke": meta_smoke,          # asserted False
        "operating_criterion": meta["calibration"].get(
            "operating_criterion"),
        "calibration_auc_encoded_vs_control": meta["calibration"].get(
            "auc_encoded_vs_control"),
        "perm_n": int(args.perm_n),
        "g1_margin_pct": agg.get("margin_required_pct"),
        "g1_abs_floor": agg.get("abs_floor"),
        "n_heldout_props": len(heldout_props),
        "n_excluded": agg["n_excluded"],
        "n_counted": agg["n_counted"],
        "n_gate_cleared": agg["n_gate_cleared"],
        "n_prop_pass": agg["n_prop_pass"],
        "all_gate_cleared": agg["all_gate_cleared"],
        "mean_true_score": round(agg["mean_true_score"], 4),
        "mean_best_perm_score": round(agg["mean_best_perm_score"], 4),
        "per_prop": per_prop,
        "aggregate_verdict": agg,
        "GATE": agg["GATE"],
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    # --- ASCII verdict block ----------------------------------------
    print("\n" + "=" * 64, flush=True)
    print("SONG G1 TASK 10 PRE-REGISTERED HELD-OUT GATE VERDICT",
          flush=True)
    print("=" * 64, flush=True)
    print("  substrate           : G.20 320-sparse 5-bridge", flush=True)
    print(f"  ckpt                : {args.ckpt} (epoch "
          f"{trained_epoch})", flush=True)
    print(f"  g1_abstain          : {g1_abstain:.2f}  "
          f"[SOURCE = sidecar-frozen Step-0 value; NOT 650; NOT "
          f"recomputed]", flush=True)
    print(f"  meta_smoke          : {meta_smoke}  (asserted False)",
          flush=True)
    print(f"  perm-n              : {int(args.perm_n)}  "
          f"(permuted-ORDER controls per held-out prop)", flush=True)
    print(f"  FIXED bars          : margin >= "
          f"{agg.get('margin_required_pct')}%  abs_floor >= "
          f"{agg.get('abs_floor')}  (UNTOUCHED)", flush=True)
    print("  -" * 32, flush=True)
    for r in per_prop:
        v = r["verdict"]
        if r["excluded"]:
            flag = "EXC"
        elif v["gate"]:
            flag = "OK "
        else:
            flag = "BAD"
        print(f"  [{flag}] int={r['intention']:<2} "
              f"seq={str(r['concept_seq']):<10} "
              f"true={r['true_score']:>6.3f} "
              f"best_perm={r['best_perm_score']:>6.3f} "
              f"gate={'Y' if r['gate_cleared'] else 'N'} "
              f"top_rate={r['top_rate']:>8.1f} "
              f"nperm={r['n_perm_controls']:>2} -> {v['GATE']}",
              flush=True)
    print("  (EXC = no non-identity permuted ORDER (degenerate "
          "multiset) -> cannot gate ORDER-learning; NOT counted)",
          flush=True)
    print("  -" * 32, flush=True)
    print(f"  held-out props        : {len(heldout_props)}", flush=True)
    print(f"  excluded (no controls): {agg['n_excluded']}", flush=True)
    print(f"  counted (gate-able)   : {agg['n_counted']}", flush=True)
    print(f"  per-prop PASS         : {agg['n_prop_pass']}/"
          f"{agg['n_counted']}  (transparency; AGGREGATE is "
          f"g1_verdict on the means)", flush=True)
    print(f"  all_gate_cleared      : "
          f"{'Y' if agg['all_gate_cleared'] else 'N'} "
          f"({agg['n_gate_cleared']}/{agg['n_counted']} cleared the "
          f"frozen g1_abstain={g1_abstain:.1f})", flush=True)
    print(f"  mean_true_score       : "
          f"{agg['mean_true_score']:.4f}", flush=True)
    print(f"  mean_best_perm_score  : "
          f"{agg['mean_best_perm_score']:.4f}", flush=True)
    print(f"  pct_over_permuted     : "
          f"{agg['pct_over_permuted']:.2f}%  (need >= "
          f"{agg.get('margin_required_pct')}%)", flush=True)
    print("  -" * 32, flush=True)
    print(f"  AGGREGATE GATE : {agg['GATE']}  "
          f"(g1_verdict on the means; FIXED bars UNTOUCHED)",
          flush=True)
    if not agg["gate"]:
        why = []
        if not agg["all_gate_cleared"]:
            why.append(f"only {agg['n_gate_cleared']}/"
                       f"{agg['n_counted']} cleared the frozen "
                       f"g1_abstain")
        if agg["best_perm_score"] <= 0.0 and agg["n_counted"] > 0:
            why.append("no permuted-ORDER contrast (mean best_perm=0)")
        if agg["n_counted"] == 0:
            why.append("no non-excluded held-out props")
        if (agg["true_score"] < agg.get("abs_floor", 0.5)
                and agg["n_counted"] > 0):
            why.append(f"mean_true {agg['true_score']:.3f} < abs_floor "
                       f"{agg.get('abs_floor')}")
        if why:
            print(f"  WHY FAIL: {'; '.join(why)}", flush=True)
    print("  NOTE: honest propagation (findings doc + "
          "capability_status) is the CONTROLLER's job post-run; this "
          "runner only computes+prints+writes JSON.", flush=True)
    print(f"  -> {args.out}", flush=True)
    print("=" * 64, flush=True)

    # Exit 0 for BOTH PASS and FAIL: a FAIL is a VALID computed result
    # (the honest negative the project requires), not a runner error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
