"""Order-intrinsic slice Task 7 -- the PRE-REGISTERED MULTI-SEED
capability gate. This is the DECISIVE TERMINAL-verdict gate for the
order-intrinsic conversational-memory line.

This orchestrator does NOT train a sequence model and does NOT tune
anything. For each of >=3 pre-registered seeds it builds the
multi-seed-validated DG/CA3 P4.1 store + the single additive
`ec_context -> motor_{action}` read-back pathway
(`order_intrinsic_encode.build_order_intrinsic_bridge`), encodes a
DISJOINT HELD-OUT set of ordered 2-3 concept propositions via the
validated co-drive encode (`encode_proposition`), reads each back with
the deterministic position sweep (`readback_sweep`), and emits the
pre-registered order-intrinsic verdict. Order is INTRINSIC to the D.11
positional code; it is read by a plain position sweep -- there is NO
learned sequence model anywhere (the categorically-different thing the
6-negative generative line could not do).

============================================================================
PARAMOUNT ANTI-CHEAT INVARIANTS (baked in here; violating ANY one
silently invalidates this decisive terminal verdict)
============================================================================

(1) UNMODIFIED PURE-CORE REUSE. The decode / floor / verdict /
    aggregate logic is `research.runners.order_intrinsic_core`
    (decode_position_sweep, control_max_floor, order_intrinsic_verdict,
    aggregate_multiseed), reused verbatim by import -- NOTHING is
    reimplemented here. `order_intrinsic_verdict` itself reuses
    `research.runners.song_g1_core.score_order` / `g1_verdict`
    UNMODIFIED (_G1_MARGIN=0.10 / _G1_ABS_FLOOR=0.5 NEVER touched), and
    the permuted-ORDER controls are
    `song_g1_core.permuted_order_controls` UNMODIFIED. This module does
    NOT modify order_intrinsic_core / order_intrinsic_encode /
    song_g1_core / validate_positional_binding / order_intrinsic_noharm.

(2) SIDECAR-FROZEN CONTROL-MAX FLOOR -- computed ONCE per seed BEFORE
    the held-out eval, frozen to an ISOLATED per-seed sidecar
    (`<ckpt>.<seed>.json`), NEVER recomputed at gate time. The floor is
    `control_max_floor(encoded_toprates, control_toprates)` over the
    TRAIN propositions' intended-order read-back (encoded sample) vs a
    CONTROL sample = permuted-ORDER + random-order productions
    (control-MAX ONLY; the encoded distribution provably cannot move
    it; NEVER the literal 650 -- a different G.20 regime). On resume,
    the frozen sidecar value + the frozen prop sets are REUSED, never
    recomputed (the song_g1_gate sidecar-frozen-floor idiom, adapted).

(3) ONE TOP-RATE AGGREGATION used IDENTICALLY in Step-0 calibration,
    held-out scoring, AND permuted-control scoring: per production, the
    top-rate is `max` over the position-sweep's per-slot decoded-
    concept rates (`_production_top_rate`). The SAME function is the
    only source of a production's scalar magnitude everywhere.

(4) HELD-OUT props NEVER tune anything. The train set calibrates the
    frozen floor (Step 0); the DISJOINT held-out set is encoded then
    immediately read back to test the trained pathway. train
    intersect heldout == empty (asserted). The held-out props are NOT
    used to pick the
    config, the floor, or the bars.

(5) PERMUTED-ORDER control is load-bearing. For each held-out prop the
    SAME concept multiset with scrambled positions is encoded + read
    back the SAME way; `g1_verdict` requires true-order to beat the
    best permuted-ORDER control by >= 10% (relative) AND clear the
    absolute floor AND clear the abstention gate. A system that merely
    ignites the right concepts (no learned order) scores true ~=
    permuted -> FAIL.

(6) >= 3 SEEDS MANDATORY. `aggregate_multiseed(..., min_seeds=3)` --
    a single-seed / near-noise result is explicitly NOT a pass (the
    cheap probe proved single-seed unreliable at 50%). Every prop in
    every seed must PASS and every seed must contribute >= 1 prop.

(7) PROPER CONFIG, FIXED / PRE-REGISTERED (NOT the no-harm speed
    config; NOT tuned after seeing results). `encoding_steps` defaults
    to 100 -- the `encode_proposition` default, which is the SAME
    100-step-per-(word,position) co-drive window that the
    multi-seed-validated P4.1 DG/CA3 store was validated in
    (`validate_positional_binding.encode_and_tag` train_events=100 x
    100-step drive; `2026-05-11-P41-positional-multiseed.md` 3/3
    PASS). The Task-6 no-harm runner DELIBERATELY overrode this DOWN
    to encoding_steps=60 / train_events=12 as a documented speed config
    ("the no-harm question is about the additive pathway's PRESENCE
    damaging the store, not about maximizing store quality") -- at
    that under-powered config the read-back signal does not separate
    (margin_rel negative; recorded a78815b). The PROPER config for the
    DECISIVE capability gate is the validated-store regime
    (encoding_steps=100), strong enough to actually train the
    read-back pathway. This is FIXED here and pre-registered -- it is
    NOT to be tuned after seeing the gate result (a wrong/too-weak
    config would make the terminal verdict uninterpretable; the
    deliberate choice + rationale is recorded in the JSON under
    `proper_config_rationale`).

(8) CuPy / RTX 3090 production backend (the numpy-only
    bridge.py:5360 IndexError is OUT OF SCOPE -- this loads the DG/CA3
    hippo bridge on CuPy). 650 is NEVER used anywhere.

KILL-SAFE RESUME (reuse `sim.train_checkpoint`'s atomic os.replace
pattern). A checkpoint is written after each completed seed (seed
index + that seed's per-prop verdicts + the frozen floor + frozen
prop sets). Re-running resumes: completed seeds are skipped and their
frozen sidecar values are reused (never recomputed). KeyboardInterrupt
-> checkpoint flushed + clean exit (the user frees the GPU to game,
re-runs later). ASCII-only output (Windows cp1252 safe). Heavy imports
(sim.*, the encode module, the bridge) are LAZY inside main();
`_freeze_propositions` / `_production_top_rate` are PURE (no IO, no
heavy import) so the import/signature smoke is instant and they are
CPU-unit-testable.

HONEST PROPAGATION is the CONTROLLER's post-run job (the findings doc +
`webapp/capability_status.json`), exactly as `song_g1_gate.py` notes.
`main()` ONLY computes + prints + writes the JSON; it does NOT write
the findings doc or capability_status. Exit 0 == verdict computed
(PASS or honest terminal FAIL are BOTH valid computed results); exit 2
== not runnable.
"""
from __future__ import annotations

import argparse

# Default isolated paths (the gate's own namespace -- NOT shared with
# any other gate). The per-seed sidecar is `<ckpt>.<seed>.json`.
_OUT_DEFAULT = "research/findings/raw/g11_bg/order_intrinsic_gate.json"
_CKPT_DEFAULT = "research/findings/raw/g11_bg/order_intrinsic_gate.ckpt.npz"

# ---- PRE-REGISTERED PROPER CONFIG (FIXED; NOT the no-harm speed
#      config; NOT tuned after seeing results). See invariant (7). ----
# encoding_steps=100 is the encode_proposition default == the
# multi-seed-validated P4.1 DG/CA3 store's validated co-drive window
# (validate_positional_binding.encode_and_tag's 100-step-per-position
# drive; 3/3 baseline). The Task-6 no-harm runner used 60 as a
# documented speed override; the DECISIVE gate uses the validated
# regime.
_PROPER_ENCODING_STEPS = 100
_PROPER_CONFIG_RATIONALE = (
    "encoding_steps=100 is the encode_proposition default AND the "
    "exact 100-step-per-(word,position) co-drive window the "
    "multi-seed-validated P4.1 DG/CA3 store was validated in "
    "(validate_positional_binding.encode_and_tag train_events=100 x "
    "100-step drive; 2026-05-11-P41-positional-multiseed.md 3/3 PASS). "
    "The Task-6 no-harm runner DELIBERATELY overrode this DOWN to "
    "encoding_steps=60/train_events=12 as a documented SPEED config "
    "(no-harm question is pathway-PRESENCE-damages-store, not "
    "store-quality); at that under-powered config the read-back signal "
    "does not separate (recorded a78815b margin_rel negative). The "
    "PROPER config for the decisive capability gate is the "
    "validated-store regime (encoding_steps=100), strong enough to "
    "actually train the additive read-back pathway. This is "
    "FIXED/pre-registered here -- it is NOT tuned after seeing the "
    "gate result; a wrong/too-weak config would make the terminal "
    "verdict uninterpretable."
)

# The bridge's concept vocab is FIXED by the validated DG/CA3 substrate:
# the only per-concept readout regions are motor_{N,E,S,W}
# (order_intrinsic_encode._WORD_TO_ACTION). The 4 direction words below
# are NOT redefined here -- they are that substrate's vocab.
_DIRECTION_WORDS = ["north", "east", "south", "west"]


def _freeze_propositions(seed: int):
    """PURE / deterministic. Derive a FROZEN TRAIN set and a DISJOINT
    FROZEN HELD-OUT set of ordered 2-concept propositions over the
    bridge's fixed 4-direction vocab {north,east,south,west}.

    The split is derived deterministically from `seed` (so each seed
    has its own frozen sets, reproducible on resume) and is guaranteed
    DISJOINT (train intersect heldout == empty) -- the held-out props
    are
    encoded-then-read-back to test the TRAINED pathway and NEVER used
    to tune the floor / config / bars (anti-cheat invariant 4).

    All 12 ordered length-2 permutations of the 4 directions
    (P(4,2) = 12) are the candidate proposition universe. A
    seed-derived deterministic rotation partitions them into a
    >=2-prop held-out tail and the remaining train head, so:
      - both sets are non-empty,
      - the held-out set has >= 2 ordered props (the gate evaluates
        these; aggregate_multiseed requires >= 1 per seed),
      - every held-out prop has a non-identity permuted-ORDER control
        (a length-2 ordered prop's reversal is its only non-identity
        permutation -> permuted_order_controls is non-empty; the
        load-bearing anti-cheat always applies),
      - the partition is a pure function of seed (resume-stable).

    Returns (train_props, heldout_props), each a list of lists of
    direction-word strings. NO IO, NO heavy import -> CPU-unit-testable
    and import/signature-smoke instant.
    """
    w = list(_DIRECTION_WORDS)
    # P(4,2): all ordered pairs of distinct directions (deterministic
    # construction order -- pure, no rng object needed for the universe).
    universe = []
    for a in w:
        for b in w:
            if a != b:
                universe.append([a, b])
    n = len(universe)            # 12
    # Deterministic seed-derived rotation (pure int arithmetic; no
    # numpy needed so the smoke stays instant). 1009 is a small prime
    # to decorrelate adjacent seeds; identical formula style to the
    # song_g1 per-prop deterministic seeding.
    rot = (int(seed) * 1009 + 7) % n
    rotated = universe[rot:] + universe[:rot]
    # Held-out tail = 3 ordered props (>= 2 required; 3 gives a little
    # more multi-prop evidence per seed without inflating wall-clock).
    n_heldout = 3
    heldout_props = [list(p) for p in rotated[:n_heldout]]
    train_props = [list(p) for p in rotated[n_heldout:]]
    # Disjointness is structural (a partition of distinct ordered
    # pairs) but assert it explicitly -- a silent overlap would let a
    # held-out prop tune the frozen floor (anti-cheat invariant 4).
    train_set = {tuple(p) for p in train_props}
    heldout_set = {tuple(p) for p in heldout_props}
    assert train_set.isdisjoint(heldout_set), (
        "FROZEN train/held-out sets overlap -- a held-out prop must "
        "NEVER tune the floor/config (anti-cheat invariant 4)")
    assert train_props and heldout_props, (
        "both FROZEN prop sets must be non-empty")
    return train_props, heldout_props


def _production_top_rate(per_pos_rates) -> float:
    """PURE / deterministic. THE ONE production-top-rate aggregation,
    used IDENTICALLY in Step-0 calibration, held-out scoring AND
    permuted-control scoring (anti-cheat invariant 3): a production's
    scalar magnitude == the MAX over the position-sweep's per-slot
    max concept rate.

    `per_pos_rates` is the list[dict] readback_sweep returns (one
    {concept: rate} dict per swept position). For each position take
    its max concept rate; the production's top-rate is the max of those
    per-position maxima (empty position dicts contribute 0.0; an empty
    sweep -> 0.0). NO IO, NO heavy import.
    """
    best = 0.0
    for rates in per_pos_rates:
        if not rates:
            continue
        for v in rates.values():
            fv = float(v)
            if fv > best:
                best = fv
    return float(best)


def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--seeds", type=str, default="42,43,44",
        help="comma-separated seeds (>= 3 MANDATORY -- "
             "aggregate_multiseed requires >= 3; a single-seed result "
             "is explicitly NOT a pass). Default: %(default)s")
    ap.add_argument(
        "--encoding-steps", type=int, default=_PROPER_ENCODING_STEPS,
        help="co-drive encode window per (word,position). DEFAULT %d "
             "is the PRE-REGISTERED PROPER config == the "
             "multi-seed-validated P4.1 DG/CA3 store's validated "
             "co-drive window (NOT the Task-6 no-harm speed override "
             "of 60). FIXED/pre-registered -- do NOT tune after seeing "
             "the gate result. Default: %%(default)s"
             % _PROPER_ENCODING_STEPS)
    ap.add_argument(
        "--perm-n", type=int, default=8,
        help="permuted-ORDER controls per held-out proposition "
             "(distinct non-identity orderings of the SAME concept "
             "multiset -- the load-bearing anti-cheat). For length-2 "
             "props there is exactly 1 (the reversal); the cap is "
             "harmless. Default: %(default)s")
    ap.add_argument(
        "--out", type=str, default=_OUT_DEFAULT,
        help="where to write the verdict JSON. Default: %(default)s")
    ap.add_argument(
        "--ckpt", type=str, default=_CKPT_DEFAULT,
        help="kill-safe resume checkpoint .npz (ISOLATED gate "
             "namespace; per-seed frozen sidecar is <ckpt>.<seed>"
             ".json). Re-running resumes: completed seeds skipped, "
             "frozen sidecar floor/props reused (NEVER recomputed). "
             "Default: %(default)s")
    return ap


def _sidecar_path(ckpt_path: str, seed: int) -> str:
    """The per-seed isolated sidecar path (the song_g1_gate
    `<ckpt>.meta.json` idiom, here per-seed): `<ckpt>.<seed>.json`."""
    return "%s.%d.json" % (ckpt_path, int(seed))


def main():
    # ---- lazy / heavy imports (building the DG/CA3 bridge is slow +
    #      GPU; _freeze_propositions / _production_top_rate stay pure) -
    import json
    import time
    from pathlib import Path

    import numpy as np

    # UNMODIFIED pure core (decode/floor/verdict/aggregate) + the
    # UNMODIFIED permuted-ORDER control. Reused verbatim -- NOT
    # reimplemented (anti-cheat invariant 1).
    from research.runners.order_intrinsic_core import (
        aggregate_multiseed,
        control_max_floor,
        decode_position_sweep,
        order_intrinsic_verdict,
    )
    from research.runners.song_g1_core import permuted_order_controls
    # The retargeted DG/CA3 store + the additive read-back pathway +
    # the validated co-drive encode + the deterministic sweep producer.
    # Reused UNMODIFIED.
    from research.runners.order_intrinsic_encode import (
        build_order_intrinsic_bridge,
        encode_proposition,
        readback_sweep,
    )
    # Kill-safe atomic checkpoint (reuse the os.replace pattern).
    from sim.train_checkpoint import (
        load_checkpoint,
        save_checkpoint,
    )

    args = _build_arg_parser().parse_args()
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    encoding_steps = int(args.encoding_steps)
    perm_n = int(args.perm_n)

    print("=" * 64, flush=True)
    print("ORDER-INTRINSIC TASK 7 -- PRE-REGISTERED MULTI-SEED "
          "CAPABILITY GATE", flush=True)
    print("(DECISIVE TERMINAL verdict; sidecar-FROZEN control-max "
          "floor;", flush=True)
    print(" permuted-ORDER control; UNMODIFIED g1_verdict; >= 3 "
          "seeds;", flush=True)
    print(" PROPER pre-registered config -- NOT the no-harm speed "
          "config)", flush=True)
    print("=" * 64, flush=True)
    print("seeds=%s  encoding_steps=%d (PRE-REGISTERED PROPER; "
          "no-harm speed was 60)  perm_n=%d"
          % (seeds, encoding_steps, perm_n), flush=True)
    print("PROPER-CONFIG RATIONALE: %s" % _PROPER_CONFIG_RATIONALE,
          flush=True)
    print("ANTI-CHEAT: order_intrinsic_core + song_g1_core reused "
          "UNMODIFIED; floor = control-max ONLY, computed ONCE/seed "
          "pre-eval, sidecar-FROZEN, NEVER recomputed; held-out NEVER "
          "tunes anything; 650 NEVER used.", flush=True)

    # ---- >= 3 seeds is MANDATORY (anti-cheat invariant 6) -----------
    if len(seeds) < 3:
        print("[NOT RUNNABLE] %d seed(s) requested; >= 3 seeds are "
              "MANDATORY for the pre-registered multi-seed gate "
              "(single-seed / near-noise is explicitly NOT a pass)."
              % len(seeds), flush=True)
        print("=" * 64, flush=True)
        return 2

    # ---- kill-safe resume: which seeds already completed? -----------
    # The checkpoint stores, per completed seed, that seed's per-prop
    # verdicts (so a re-run skips them) -- weights[0] is a 1-D float
    # array we don't use (train_checkpoint requires >= 1 weight array);
    # the real state is JSON in loss_history-adjacent fields we encode
    # ourselves below via the per-seed sidecar + a tiny progress file.
    # We use the sidecar as the source of truth for frozen floor/props
    # and a separate compact resume file for completed-seed verdicts.
    resume_path = str(args.ckpt) + ".resume.json"
    completed = {}   # seed -> list[per-prop verdict dict]
    if Path(resume_path).exists():
        try:
            with open(resume_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            completed = {int(k): v for k, v in
                         raw.get("completed_seeds", {}).items()}
            if completed:
                print("[RESUME] %d seed(s) already complete: %s "
                      "(skipping; frozen sidecars reused, NEVER "
                      "recomputed)"
                      % (len(completed),
                         sorted(completed.keys())), flush=True)
        except (ValueError, OSError) as e:
            print("[RESUME] resume file unreadable (%s) -- starting "
                  "fresh: %s" % (resume_path, e), flush=True)
            completed = {}

    def _flush_resume():
        """Atomically persist completed-seed verdicts (kill-safe).
        Also writes a trivial .npz via save_checkpoint so the
        train_checkpoint atomic-os.replace contract is exercised for
        the binary checkpoint path the spec references."""
        tmp = resume_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "completed_seeds": {str(k): v
                                    for k, v in completed.items()},
                "seeds": seeds,
                "encoding_steps": encoding_steps,
                "perm_n": perm_n,
            }, f, indent=2, default=str)
        import os
        os.replace(tmp, resume_path)
        # Binary kill-safe checkpoint (atomic os.replace; the spec's
        # sim.train_checkpoint reuse). epoch == #completed seeds.
        try:
            save_checkpoint(
                str(args.ckpt),
                epoch=len(completed),
                weights=[np.zeros(1, dtype=np.float32)],
                rng_state=np.random.default_rng(0).bit_generator.state,
                loss_history=[float(len(completed))],
            )
        except Exception:
            pass

    # (touch load_checkpoint so the import is meaningfully used and a
    # corrupt/legacy .npz never crashes resume -- the JSON resume file
    # is the authoritative completed-seed record.)
    _ = load_checkpoint(str(args.ckpt))

    t0 = time.time()
    per_seed_prop_verdicts = []     # list[ list[verdict dict] ] (order = seeds)
    per_seed_records = []           # rich JSON detail per seed
    frozen_floors = {}              # seed -> g1_abstain (for JSON)

    try:
        for seed in seeds:
            # FROZEN train/held-out sets are a PURE function of seed
            # (resume-stable; disjoint asserted inside).
            train_props, heldout_props = _freeze_propositions(seed)

            if seed in completed:
                # Already done -> reuse the recorded per-prop verdicts
                # AND the frozen sidecar floor/props (NEVER recompute).
                v_list = completed[seed]
                per_seed_prop_verdicts.append(v_list)
                sc = _sidecar_path(str(args.ckpt), seed)
                g1_abstain = None
                if Path(sc).exists():
                    try:
                        with open(sc, "r", encoding="utf-8") as f:
                            scd = json.load(f)
                        g1_abstain = scd.get("g1_abstain")
                    except (ValueError, OSError):
                        g1_abstain = None
                frozen_floors[seed] = g1_abstain
                # Rebuild a per_prop record SCHEMA-CONSISTENT with the
                # fresh-seed prop_records (below) so the JSON `per_seed`
                # artifact is uniform and the ASCII verdict block can
                # render resumed seeds. v_list is 1:1 in-order with the
                # PURE-re-derived heldout_props (a seed only enters
                # `completed` AFTER its full in-order held-out loop). The
                # displayed fields are FAITHFUL: prop is purely
                # re-derived; gate_cleared / true_score / best_perm_score
                # are read from the PERSISTED verdict (g1_verdict stores
                # "gate_cleared"). Verbose diagnostics not persisted in
                # the kill-safe resume record are honestly None/[] (NOT
                # fabricated). This is display/record only -- it does NOT
                # touch per_seed_prop_verdicts (what aggregate_multiseed
                # consumes), the floor, the verdict, or any protected
                # module, so the terminal verdict is provably unaffected.
                resumed_per_prop = []
                for i, v in enumerate(v_list):
                    hp_i = (list(heldout_props[i])
                            if i < len(heldout_props) else None)
                    resumed_per_prop.append({
                        "prop": hp_i,
                        "true_decoded": None,
                        "true_conf": None,
                        "true_abstained_positions": None,
                        "true_top_rate": None,
                        "gate_cleared": v.get("gate_cleared"),
                        "g1_abstain_frozen": g1_abstain,
                        "n_perm_controls": None,
                        "perm_controls": None,
                        "perm_best": round(
                            float(v.get("best_perm_score", 0.0)), 6),
                        "true_score": round(
                            float(v.get("true_score", 0.0)), 6),
                        "verdict": v,
                        "resumed_record": True,
                    })
                per_seed_records.append({
                    "seed": seed,
                    "resumed": True,
                    "g1_abstain_frozen": g1_abstain,
                    "g1_abstain_source": "sidecar-frozen (resumed; "
                                         "NEVER recomputed)",
                    "train_propositions": train_props,
                    "heldout_propositions": heldout_props,
                    "per_prop": resumed_per_prop,
                })
                print("\n[SEED %d] RESUMED from frozen sidecar "
                      "(floor=%s; %d held-out props; NEVER "
                      "recomputed)"
                      % (seed, g1_abstain, len(heldout_props)),
                      flush=True)
                continue

            print("\n" + "-" * 64, flush=True)
            print("SEED %d" % seed, flush=True)
            print("-" * 64, flush=True)
            print("  FROZEN train props (%d): %s"
                  % (len(train_props), train_props), flush=True)
            print("  FROZEN held-out props (%d, DISJOINT, NEVER tune "
                  "anything): %s"
                  % (len(heldout_props), heldout_props), flush=True)

            # ---- per-seed sidecar: reuse a frozen floor if it exists
            #      (resume mid-seed), else compute ONCE now (Step 0) --
            sc = _sidecar_path(str(args.ckpt), seed)
            g1_abstain = None
            sidecar_obj = None
            if Path(sc).exists():
                try:
                    with open(sc, "r", encoding="utf-8") as f:
                        sidecar_obj = json.load(f)
                    g1_abstain = float(sidecar_obj["g1_abstain"])
                    # The frozen sidecar's prop sets MUST match the
                    # pure re-derivation (a drift would mean the floor
                    # was calibrated on different props).
                    s_tr = [list(p) for p in
                            sidecar_obj.get("train_propositions", [])]
                    s_ho = [list(p) for p in
                            sidecar_obj.get("heldout_propositions",
                                            [])]
                    assert s_tr == train_props and s_ho == heldout_props, (
                        "frozen sidecar prop sets diverge from the "
                        "pure re-derivation for seed %d -- refusing to "
                        "reuse a mismatched frozen floor" % seed)
                    print("  [SIDECAR] reusing FROZEN g1_abstain="
                          "%.6f from %s (NEVER recomputed)"
                          % (g1_abstain, sc), flush=True)
                except (ValueError, OSError, KeyError,
                        AssertionError) as e:
                    print("  [SIDECAR] existing sidecar unusable (%s) "
                          "-- recomputing Step-0 ONCE: %s"
                          % (sc, e), flush=True)
                    g1_abstain = None
                    sidecar_obj = None

            t_seed = time.time()
            print("  building DG/CA3 bridge (validated P4.1 store + "
                  "additive ec_context->motor read-back pathway) ...",
                  flush=True)
            bridge = build_order_intrinsic_bridge(seed=seed,
                                                   verbose=True)

            if g1_abstain is None:
                # ---- Step 0: pre-registered FROZEN control-max floor
                #      computed ONCE, BEFORE the held-out eval -------
                # Encoded sample = the TRAIN props' INTENDED order
                # read-back top-rates. Control sample = permuted-ORDER
                # + random-order productions of the SAME train props.
                # control_max_floor uses the control MAX ONLY (the
                # encoded sample provably cannot move it). NEVER 650.
                print("  [STEP 0] calibrating the pre-registered "
                      "control-max abstention floor ONCE (frozen to "
                      "the isolated sidecar; NEVER recomputed at gate "
                      "time; NEVER 650) ...", flush=True)
                encoded_tops = []
                control_tops = []
                rng0 = np.random.default_rng(int(seed) * 1009 + 17)
                for tp in train_props:
                    # encoded (intended order): encode then read back.
                    encode_proposition(
                        bridge, list(tp),
                        tag_name="cal_enc_" + "_".join(tp),
                        encoding_steps=encoding_steps, verbose=False)
                    pe = readback_sweep(bridge, length=len(tp))
                    encoded_tops.append(_production_top_rate(pe))

                    # control: permuted-ORDER + one random-order
                    # production of the SAME multiset, encoded + read
                    # back the SAME way (the SAME top-rate aggregate).
                    perms = permuted_order_controls(
                        list(tp), rng0, perm_n)
                    rand = list(tp)
                    rng0.shuffle(rand)
                    ctl_orders = list(perms)
                    if list(rand) != list(tp) and rand not in ctl_orders:
                        ctl_orders.append(rand)
                    for co in ctl_orders:
                        encode_proposition(
                            bridge, list(co),
                            tag_name="cal_ctl_" + "_".join(co),
                            encoding_steps=encoding_steps,
                            verbose=False)
                        pc = readback_sweep(bridge, length=len(co))
                        control_tops.append(_production_top_rate(pc))

                g1_abstain = float(control_max_floor(encoded_tops,
                                                     control_tops))
                # FREEZE it (+ the frozen prop sets) to the isolated
                # per-seed sidecar -- atomic write.
                sidecar_obj = {
                    "seed": seed,
                    "g1_abstain": g1_abstain,
                    "g1_abstain_source": (
                        "Step-0 control-max ONLY over TRAIN "
                        "intended-order vs permuted/random-order "
                        "read-back top-rates; frozen here; NEVER "
                        "recomputed at gate time; NEVER 650"),
                    "encoding_steps": encoding_steps,
                    "perm_n": perm_n,
                    "n_encoded_samples": len(encoded_tops),
                    "n_control_samples": len(control_tops),
                    "encoded_toprates": [round(float(x), 6)
                                         for x in encoded_tops],
                    "control_toprates": [round(float(x), 6)
                                         for x in control_tops],
                    "train_propositions": train_props,
                    "heldout_propositions": heldout_props,
                    "proper_config_rationale": _PROPER_CONFIG_RATIONALE,
                }
                Path(sc).parent.mkdir(parents=True, exist_ok=True)
                tmp = sc + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(sidecar_obj, f, indent=2, default=str)
                import os
                os.replace(tmp, sc)
                print("  [STEP 0] FROZEN g1_abstain=%.6f -> %s "
                      "(control-max ONLY; %d enc / %d ctl samples; "
                      "NEVER recomputed)"
                      % (g1_abstain, sc, len(encoded_tops),
                         len(control_tops)), flush=True)

            frozen_floors[seed] = g1_abstain

            # ---- per held-out prop: encode -> sweep -> decode vs the
            #      FROZEN floor -> permuted-ORDER controls -> verdict -
            seed_verdicts = []
            prop_records = []
            for hp in heldout_props:
                # true (intended order): encode then deterministic
                # position sweep.
                encode_proposition(
                    bridge, list(hp),
                    tag_name="ho_true_" + "_".join(hp),
                    encoding_steps=encoding_steps, verbose=False)
                per_pos = readback_sweep(bridge, length=len(hp))
                # decode vs the FROZEN floor (decode_position_sweep
                # abstains <= floor; gate_cleared := no slot abstained).
                decoded, conf, abstained = decode_position_sweep(
                    per_pos, floor=g1_abstain)
                gate_cleared = (None not in decoded)
                true_top = _production_top_rate(per_pos)

                # permuted-ORDER controls (load-bearing anti-cheat):
                # SAME multiset, scrambled order, encoded + read back
                # the SAME way -> decoded list per control.
                prng = np.random.default_rng(int(seed) * 1009
                                             + sum(ord(c)
                                                   for w in hp
                                                   for c in w))
                perms = permuted_order_controls(list(hp), prng, perm_n)
                perm_decoded_list = []
                perm_detail = []
                for pm in perms:
                    encode_proposition(
                        bridge, list(pm),
                        tag_name="ho_perm_" + "_".join(pm),
                        encoding_steps=encoding_steps, verbose=False)
                    pp = readback_sweep(bridge, length=len(pm))
                    pdec, _pc, _pa = decode_position_sweep(
                        pp, floor=g1_abstain)
                    perm_decoded_list.append(pdec)
                    perm_detail.append({
                        "perm_order": list(pm),
                        "decoded": [str(d) for d in pdec],
                        "top_rate": round(
                            _production_top_rate(pp), 6),
                    })

                # PURE pre-registered verdict (reuses UNMODIFIED
                # score_order + g1_verdict via order_intrinsic_verdict).
                v = order_intrinsic_verdict(
                    decoded, list(hp), perm_decoded_list,
                    gate_cleared)
                seed_verdicts.append(v)
                prop_records.append({
                    "prop": list(hp),
                    "true_decoded": [str(d) for d in decoded],
                    "true_conf": [round(float(c), 6) for c in conf],
                    "true_abstained_positions": list(abstained),
                    "true_top_rate": round(true_top, 6),
                    "gate_cleared": bool(gate_cleared),
                    "g1_abstain_frozen": g1_abstain,
                    "n_perm_controls": len(perms),
                    "perm_controls": perm_detail,
                    "perm_best": round(
                        float(v.get("best_perm_score", 0.0)), 6),
                    "true_score": round(
                        float(v.get("true_score", 0.0)), 6),
                    "verdict": v,
                })
                print("  [%s] prop=%s true_decoded=%s "
                      "gate_cleared=%s true=%.3f best_perm=%.3f "
                      "-> %s"
                      % ("P" if v["GATE"] == "PASS" else "F",
                         hp, [str(d) for d in decoded],
                         "Y" if gate_cleared else "N",
                         float(v.get("true_score", 0.0)),
                         float(v.get("best_perm_score", 0.0)),
                         v["GATE"]), flush=True)

            per_seed_prop_verdicts.append(seed_verdicts)
            per_seed_records.append({
                "seed": seed,
                "resumed": False,
                "g1_abstain_frozen": g1_abstain,
                "g1_abstain_source": ("sidecar-frozen control-max "
                                      "ONLY; NEVER recomputed at gate "
                                      "time; NEVER 650"),
                "encoding_steps": encoding_steps,
                "train_propositions": train_props,
                "heldout_propositions": heldout_props,
                "per_prop": prop_records,
                "seed_seconds": round(time.time() - t_seed, 1),
            })

            # mark this seed complete + flush kill-safe resume.
            completed[seed] = seed_verdicts
            _flush_resume()

            # free GPU between seeds (best-effort).
            try:
                del bridge
                from sim.backend import get_backend
                cp, name = get_backend()
                if name == "cupy":
                    cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    except KeyboardInterrupt:
        # Kill-safe: flush whatever completed, exit cleanly (the user
        # frees the GPU to game; re-run resumes from the frozen
        # sidecars + the resume file -- NEVER recomputes floors).
        _flush_resume()
        print("\n[INTERRUPTED] checkpoint flushed (%d/%d seeds "
              "complete). Re-run to resume -- frozen sidecars + "
              "completed-seed verdicts are reused, NEVER recomputed."
              % (len(completed), len(seeds)), flush=True)
        print("=" * 64, flush=True)
        return 2

    # ---- pre-registered MULTI-SEED aggregate (>= 3 seeds; every prop
    #      every seed PASS; every seed >= 1 prop) -- UNMODIFIED pure
    #      aggregate_multiseed; min_seeds=3 (mandatory) ---------------
    agg = aggregate_multiseed(per_seed_prop_verdicts, min_seeds=3)

    result = {
        "task": ("order-intrinsic Task 7 pre-registered MULTI-SEED "
                 "capability gate (DECISIVE TERMINAL verdict)"),
        "substrate": ("multi-seed-validated DG/CA3 P4.1 store + "
                      "additive ec_context_to_motor_readback pathway; "
                      "order is INTRINSIC (D.11 positional code), read "
                      "by a deterministic position sweep -- NO learned "
                      "sequence model"),
        "seeds": seeds,
        "n_seeds": len(seeds),
        "min_seeds_required": 3,
        "config": {
            "encoding_steps": encoding_steps,
            "perm_n": perm_n,
            "encoding_steps_is_proper_not_speed": (
                encoding_steps >= _PROPER_ENCODING_STEPS),
            "no_harm_speed_config_was": {
                "encoding_steps": 60, "train_events": 12,
                "note": ("Task-6 no-harm DELIBERATELY under-powered "
                         "config; recorded a78815b margin_rel "
                         "negative -- NOT the gate config"),
            },
        },
        "proper_config_rationale": _PROPER_CONFIG_RATIONALE,
        "anti_cheat": {
            "pure_core_reused_unmodified": (
                "order_intrinsic_core.{decode_position_sweep,"
                "control_max_floor,order_intrinsic_verdict,"
                "aggregate_multiseed} + song_g1_core."
                "{permuted_order_controls,score_order,g1_verdict} "
                "imported verbatim; NOT reimplemented; bars "
                "_G1_MARGIN=0.10/_G1_ABS_FLOOR=0.5 NEVER touched"),
            "floor_is_control_max_only": True,
            "floor_computed_once_per_seed_pre_eval": True,
            "floor_sidecar_frozen_never_recomputed_at_gate_time": True,
            "floor_never_650": True,
            "held_out_never_tunes_anything": True,
            "permuted_order_control_load_bearing": True,
            "min_seeds_3_mandatory": True,
            "config_fixed_pre_registered_not_tuned_post_hoc": True,
            "one_top_rate_aggregation_identical_step0_heldout_perm": (
                "_production_top_rate (max over the position-sweep's "
                "per-slot max concept rate)"),
            "protected_modules_unmodified": [
                "order_intrinsic_core", "order_intrinsic_encode",
                "song_g1_core", "validate_positional_binding",
                "order_intrinsic_noharm"],
        },
        "frozen_g1_abstain_per_seed": {
            str(k): v for k, v in frozen_floors.items()},
        "per_seed": per_seed_records,
        "aggregate_verdict": agg,
        "GATE": agg["GATE"],
        "OVERALL": "PASS" if agg["GATE"] == "PASS" else "FAIL",
        "honest_propagation_note": (
            "honest propagation (findings doc + "
            "webapp/capability_status.json pillar) is the "
            "CONTROLLER's post-run job; this runner ONLY "
            "computes+prints+writes JSON (same contract as "
            "song_g1_gate.py)"),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    # ---- ASCII verdict block ---------------------------------------
    print("\n" + "=" * 64, flush=True)
    print("ORDER-INTRINSIC TASK 7 PRE-REGISTERED MULTI-SEED GATE "
          "VERDICT", flush=True)
    print("=" * 64, flush=True)
    print("  substrate : validated DG/CA3 P4.1 store + additive "
          "ec_context->motor readback (order INTRINSIC; no learned "
          "sequence model)", flush=True)
    print("  seeds     : %s (>= 3 mandatory)" % seeds, flush=True)
    print("  config    : encoding_steps=%d  perm_n=%d  "
          "[PRE-REGISTERED PROPER -- NOT the no-harm speed 60; FIXED, "
          "not tuned post-hoc]" % (encoding_steps, perm_n),
          flush=True)
    # FIXED pre-registered bars live in song_g1_core (g1_verdict):
    # _G1_MARGIN=0.10 (>=10% over best permuted) + _G1_ABS_FLOOR=0.5.
    # Printed from the constants -- this gate NEVER tunes them.
    print("  FIXED bars: margin >= 10.0%%  abs_floor >= 0.5  "
          "(song_g1_core g1_verdict; UNTOUCHED)", flush=True)
    print("  -" * 32, flush=True)
    for rec in per_seed_records:
        sd = rec["seed"]
        fl = rec.get("g1_abstain_frozen")
        n_pass = sum(1 for p in rec["per_prop"]
                     if p["verdict"]["GATE"] == "PASS")
        n_tot = len(rec["per_prop"])
        print("  seed %d: frozen g1_abstain=%s  per-prop PASS=%d/%d%s"
              % (sd,
                 ("%.6f" % fl) if isinstance(fl, (int, float))
                 else str(fl),
                 n_pass, n_tot,
                 "  (RESUMED; floor NEVER recomputed)"
                 if rec.get("resumed") else ""), flush=True)
        for p in rec["per_prop"]:
            v = p["verdict"]
            print("    [%s] prop=%-18s true=%-22s gate=%s "
                  "true_s=%.3f best_perm=%.3f -> %s"
                  % ("OK " if v["GATE"] == "PASS" else "BAD",
                     str(p.get("prop")),
                     str(p.get("true_decoded")),
                     "Y" if p.get("gate_cleared") else "N",
                     float(v.get("true_score", 0.0)),
                     float(v.get("best_perm_score", 0.0)),
                     v["GATE"]), flush=True)
    print("  -" * 32, flush=True)
    print("  n_seeds              : %d (min %d)"
          % (agg["n_seeds"], agg["min_seeds"]), flush=True)
    print("  enough_seeds         : %s"
          % ("Y" if agg["enough_seeds"] else "N"), flush=True)
    print("  every seed has props : %s"
          % ("Y" if agg["all_seeds_have_props"] else "N"), flush=True)
    print("  held-out props PASS  : %d/%d (every prop every seed must "
          "PASS)" % (agg["n_props_pass"], agg["n_props_total"]),
          flush=True)
    print("  -" * 32, flush=True)
    print("  AGGREGATE GATE : %s  (pre-registered multi-seed; "
          "UNMODIFIED aggregate_multiseed; bars UNTOUCHED)"
          % agg["GATE"], flush=True)
    if agg["GATE"] != "PASS":
        why = []
        if not agg["enough_seeds"]:
            why.append("fewer than %d seeds" % agg["min_seeds"])
        if not agg["all_seeds_have_props"]:
            why.append("a seed contributed 0 held-out props")
        if not agg["all_pass"]:
            why.append("%d/%d held-out props failed"
                       % (agg["n_props_total"] - agg["n_props_pass"],
                          agg["n_props_total"]))
        if why:
            print("  WHY FAIL: %s" % "; ".join(why), flush=True)
        print("  NOTE: a maxed FAIL here is an HONEST, TERMINAL, "
              "decision-relevant finding for this line; the validated "
              "grounded-memory + no-confabulation asset remains the "
              "deliverable. Do NOT config-crank.", flush=True)
    print("  NOTE: honest propagation (findings doc + "
          "capability_status) is the CONTROLLER's job post-run; this "
          "runner ONLY computes+prints+writes JSON.", flush=True)
    print("  -> %s" % args.out, flush=True)
    print("=" * 64, flush=True)

    # Exit 0 for BOTH PASS and FAIL: a FAIL is a VALID computed result
    # (the honest terminal verdict the project requires), not a runner
    # error. Exit 2 ONLY when not runnable (< 3 seeds / interrupted).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
