"""Order-intrinsic slice Task 6 -- the LOAD-BEARING no-harm safety check.

CRITICAL / GATING. This MUST pass before the Task 7 pre-registered
multi-seed gate is trusted (the v12/v13/v15/G1 "first do no harm"
lesson). It is MEANINGFUL because Task 5 was retargeted (commit
9f9187f) to the design's MULTI-SEED-VALIDATED DG->CA3 hippocampal
store -- the exact store `research/runners/validate_positional_binding`
validates 3/3 (seeds 42/43/44, all CA3 cosines < 0.14;
`research/findings/2026-05-11-P41-positional-multiseed.md`).

THE NO-HARM QUESTION
--------------------
`order_intrinsic_encode.build_order_intrinsic_bridge` builds that SAME
validated DG/CA3 store PLUS exactly one net-new additive plastic
pathway `ec_context -> motor_{N,E,S,W}` gated
`ec_context_to_motor_readback`, DISJOINT from every validated-store
gate (`ec_context_to_dg`, `ec_to_dg`, `dg_to_ca3`, `ca3_swr_burst`,
`lang_to_ec`, `ec_to_ca1`, `ca3_to_ca1`, `ca1_to_motor`, ...). Does
the PRESENCE of (and the training of) that additive read-back pathway
REGRESS:
  (A) the validated (word,position)->distinct-CA3 store distinctness
      (same-word/diff-pos cos < 0.4 AND diff-word/same-pos cos < 0.4,
      multi-seed 3/3 baseline cos < 0.14), or
  (B) the no-confabulation abstention moat (a known/encoded
      (word,position) reactivates clearly above an un-encoded control;
      an unknown cue does NOT confabulate)?

WHAT THIS RUNNER DOES (DRY -- nothing reimplemented)
----------------------------------------------------
`validate_positional_binding.run_positional_validation` builds its OWN
bridge internally (no external-bridge injection point). So per the
Task 6 spec's branch (b):

 PART A (store-distinctness no-harm, the LOAD-BEARING axis):
   build the bridge via `build_order_intrinsic_bridge` (the additive
   `ec_context_to_motor_readback` pathway is PRESENT on it), then run
   the EXACT validated 4-tag encode + pairwise-CA3-cosine + PASS
   criteria by REUSING, verbatim by import, the UNCHANGED
   `validate_positional_binding` functions:
     - `build_word_pattern`        (the validated word code)
     - `encode_and_tag`           (the validated co-drive encode +
                                   engram tag; it opens ONLY the
                                   validated store gates -- it never
                                   references the readback gate, so
                                   reusing it verbatim correctly tests
                                   "pathway PRESENT, its plasticity not
                                   bleeding into the validated store
                                   path" -- exactly the no-harm Q)
     - `cosine_similarity_indices` (the validated metric)
   and apply the SAME `< 0.4` PASS criteria + the SAME 4 bindings
   [apple_pos0, apple_pos2, alice_pos0, alice_pos2]. The cosine/PASS
   logic is NOT reimplemented here; it is `validate_positional_binding`'s.
   We additionally compare against the multi-seed baseline cos < 0.14
   (a clear jump above 0.4 with the pathway present is a real
   regression; near-baseline is unregressed).

 PART B (no-confabulation abstention moat, unregressed):
   on the SAME bridge, REUSE verbatim (by import)
   `order_intrinsic_encode.encode_proposition` (trains the additive
   read-back pathway via its disjoint gate) + `.readback_sweep` (the
   deterministic ec_context(position)-alone producer), and the project's
   pure abstention idiom `order_intrinsic_core.decode_position_sweep`
   + `.control_max_floor`. The moat = the encoded proposition's
   per-position top motor rate clearly SEPARATES from a never-encoded
   control position's top rate (control-max-calibrated floor; this
   DG/CA3 read-back regime is NOT the G.20 650-regime -- no literal 650
   is used here, per the Task 6 spec). Encoded clearly > control AND
   the never-encoded control abstains under that control-max floor.

ANTI-CHEAT / SAFETY (non-negotiable -- run ONCE)
------------------------------------------------
Run ONCE. PASS -> commit runner + JSON; Task 7 gate is authorized.
FAIL -> STOP: do NOT re-run to chase a pass, do NOT weaken
`validate_positional_binding`'s `< 0.4` criteria, do NOT touch
`order_intrinsic_core`/`song_g1_core`. A genuine FAIL means the
additive read-back pathway regressed the validated DG/CA3 store
distinctness or the moat -> separation-of-concerns is wrong (the
readback pathway must not bleed into the ec_context->dg->CA3 store
path); that is a real BLOCKING finding, propagated with full evidence
+ an honest real-regression-vs-probe-noise assessment vs the 3/3
baseline cos < 0.14. The JSON is committed as a recorded finding
either way.

Production backend is CuPy/RTX 3090 (the numpy-only bridge.py:5360
IndexError is OUT OF SCOPE -- this loads the DG/CA3 hippo bridge on
CuPy). Heavy imports are LAZY inside main(). ASCII-only output
(Windows cp1252 safe).
"""
from __future__ import annotations

# Default no-harm config. Multi-seed (42/43/44) to MATCH the validated
# 3/3 baseline (`2026-05-11-P41-positional-multiseed.md`); a small
# train-events (the validated probe tolerates 8-20) for speed -- the
# no-harm question is about the additive pathway's PRESENCE/training
# damaging the store, not about maximizing store quality.
_SEEDS = [42, 43, 44]
_TRAIN_EVENTS = 12

# The validated store's PASS criterion (validate_positional_binding):
# same-word/diff-pos AND diff-word/same-pos CA3 cosine < this. NOT
# redefined here -- mirrored as a constant for the JSON record; the
# actual PASS decision REUSES validate_positional_binding's own logic
# pattern (we recompute with its cosine_similarity_indices + its 0.4).
_STORE_COS_CRITERION = 0.4
# The multi-seed-validated baseline (3/3 PASS): every CA3 cosine was
# < this. A clear jump above 0.4 WITH the additive pathway present is a
# real regression; near this baseline is unregressed. Record-only
# context for the honest regression-vs-noise assessment (NOT a bar --
# the bar is validate_positional_binding's < 0.4).
_MULTISEED_BASELINE_MAX_COS = 0.14

# Moat: a known/encoded proposition vs a never-encoded control cue.
# The encoded position's correct-motor top rate must clearly separate
# from the control-max floor (control-max-calibrated, the project's
# pure `control_max_floor` idiom). This DG/CA3 read-back regime is a
# DIFFERENT regime from the G.20 sparse 650-regime -- NO literal 650 is
# used (per the Task 6 spec). A small relative separation margin guards
# substrate stochasticity (the encoded signal must beat the control
# floor by more than noise to count as a real, unregressed moat).
_MOAT_REL_MARGIN = 0.05
# A known proposition over the 4 direction words (the only per-concept
# readout regions on the validated DG/CA3 bridge are motor_{N,E,S,W};
# this is order_intrinsic_encode._WORD_TO_ACTION's vocab -- NOT
# redefined). Position k of this list is the "encoded (word,position)"
# whose motor_{action} pool must reactivate above the control floor.
_MOAT_PROPOSITION = ["north", "east", "south", "west"]
# A never-encoded control cue: an extra position index that was NOT in
# the encoded proposition (length L -> sweep L+1; position L was never
# encoded). Its top motor rate is the control floor / abstention proof.

_OUT_JSON = "research/findings/raw/g11_bg/order_intrinsic_noharm.json"


def _run_store_distinctness_noharm(seed: int, train_events: int,
                                   log) -> dict:
    """PART A: run validate_positional_binding's EXACT validated 4-tag
    encode + pairwise-CA3-cosine + PASS criteria, REUSING its functions
    verbatim by import, against a bridge built by
    build_order_intrinsic_bridge (additive ec_context->motor read-back
    pathway PRESENT). DRY: cosine/PASS logic is NOT reimplemented.
    """
    import time

    # DRY: the UNCHANGED validated functions -- imported, not copied.
    from research.runners.validate_positional_binding import (
        build_word_pattern,
        encode_and_tag,
        cosine_similarity_indices,
    )
    from research.runners.order_intrinsic_encode import (
        build_order_intrinsic_bridge,
        _READBACK_GATE,
    )
    from sim.backend import to_host

    t0 = time.time()
    log("  building bridge via build_order_intrinsic_bridge "
        "(validated DG/CA3 store + additive %s pathway PRESENT) ..."
        % _READBACK_GATE)
    # n_lang_input=1024 etc are build_order_intrinsic_bridge defaults,
    # which equal validate_positional_binding.run_positional_validation
    # defaults (the 3/3 store config) -- the bridge build is byte-for-
    # byte the validated store + ONLY the appended readback pathway.
    bridge = build_order_intrinsic_bridge(seed=seed, verbose=False)
    n_total = int(bridge.cp_external_input_current.shape[0])
    try:
        nnz = int(bridge.cp_connections.nnz)
    except Exception:
        nnz = -1
    log("    built in %.1fs; %d neurons, %d synapses"
        % (time.time() - t0, n_total, nnz))

    # The 4 validated bindings -- IDENTICAL to
    # validate_positional_binding.run_positional_validation.
    n_lang_input = 1024  # build_order_intrinsic_bridge default
    n_ec_context = 200   # build_order_intrinsic_bridge default
    apple = build_word_pattern("apple", n_lang_input)
    alice = build_word_pattern("alice", n_lang_input)
    bindings = [
        ("apple_pos0", apple, 0),
        ("apple_pos2", apple, 2),
        ("alice_pos0", alice, 0),
        ("alice_pos2", alice, 2),
    ]
    for tag_name, word_indices, pos in bindings:
        t_enc = time.time()
        log("    encoding %s (position=%d) via validated "
            "encode_and_tag ..." % (tag_name, pos))
        # encode_and_tag opens ONLY the validated store gates
        # (ca3_swr_burst/dg_to_ca3/ec_to_dg/ec_context_to_dg/
        # lang_to_ec) -- it NEVER references _READBACK_GATE. So this is
        # the validated co-drive encode, run on a bridge that merely
        # HAS the additive pathway present: exactly the no-harm Q.
        stats = encode_and_tag(
            bridge, tag_name, word_indices, pos,
            n_lang_input=n_lang_input, n_ec_context=n_ec_context,
            train_events=train_events,
        )
        log("      -> %d CA3 neurons tagged (%.0fs)"
            % (stats["n_tagged"], time.time() - t_enc))

    # Pairwise CA3 cosines -- REUSE validate_positional_binding's
    # cosine_similarity_indices verbatim + its exact pair set.
    tag_indices = {
        name: to_host(bridge.get_engram_tag_indices(name))
        for name, _, _ in bindings
    }
    pairs = [
        ("apple_pos0", "apple_pos2", "SAME WORD, DIFFERENT POSITION"),
        ("alice_pos0", "alice_pos2", "SAME WORD, DIFFERENT POSITION"),
        ("apple_pos0", "alice_pos0", "DIFFERENT WORD, SAME POSITION"),
        ("apple_pos2", "alice_pos2", "DIFFERENT WORD, SAME POSITION"),
        ("apple_pos0", "alice_pos2",
         "DIFFERENT WORD, DIFFERENT POSITION"),
    ]
    pair_results = []
    log("  pairwise CA3 ensemble cosines (validated metric):")
    for a, b, label in pairs:
        cos = cosine_similarity_indices(
            tag_indices[a], tag_indices[b], n_total)
        log("    %s vs %s (%s): %.3f" % (a, b, label, cos))
        pair_results.append(
            {"a": a, "b": b, "label": label, "cosine": cos})

    # SAME PASS criteria as validate_positional_binding.
    pd = {(r["a"], r["b"]): r["cosine"] for r in pair_results}
    cos_apple_pos = pd.get(("apple_pos0", "apple_pos2"), 1.0)
    cos_alice_pos = pd.get(("alice_pos0", "alice_pos2"), 1.0)
    cos_pos0_word = pd.get(("apple_pos0", "alice_pos0"), 1.0)
    cos_pos2_word = pd.get(("apple_pos2", "alice_pos2"), 1.0)
    pass_position = ((cos_apple_pos < _STORE_COS_CRITERION)
                     and (cos_alice_pos < _STORE_COS_CRITERION))
    pass_word = ((cos_pos0_word < _STORE_COS_CRITERION)
                 and (cos_pos2_word < _STORE_COS_CRITERION))
    store_pass = bool(pass_position and pass_word)
    max_cos = max(cos_apple_pos, cos_alice_pos,
                  cos_pos0_word, cos_pos2_word)

    log("  store-distinctness PASS criteria (< %.1f, "
        "validate_positional_binding's own):" % _STORE_COS_CRITERION)
    log("    apple_pos0 vs apple_pos2: %.3f %s"
        % (cos_apple_pos,
           "PASS" if cos_apple_pos < _STORE_COS_CRITERION else "FAIL"))
    log("    alice_pos0 vs alice_pos2: %.3f %s"
        % (cos_alice_pos,
           "PASS" if cos_alice_pos < _STORE_COS_CRITERION else "FAIL"))
    log("    apple_pos0 vs alice_pos0: %.3f %s"
        % (cos_pos0_word,
           "PASS" if cos_pos0_word < _STORE_COS_CRITERION else "FAIL"))
    log("    apple_pos2 vs alice_pos2: %.3f %s"
        % (cos_pos2_word,
           "PASS" if cos_pos2_word < _STORE_COS_CRITERION else "FAIL"))
    log("    max CA3 cosine = %.3f (multi-seed 3/3 baseline < %.2f)"
        % (max_cos, _MULTISEED_BASELINE_MAX_COS))
    log("    STORE-DISTINCTNESS no-harm: %s"
        % ("PASS" if store_pass else "FAIL"))

    return {
        "seed": seed,
        "train_events": train_events,
        "n_neurons": n_total,
        "n_synapses": nnz,
        "pair_cosines": pair_results,
        "cos_apple_pos": cos_apple_pos,
        "cos_alice_pos": cos_alice_pos,
        "cos_pos0_word": cos_pos0_word,
        "cos_pos2_word": cos_pos2_word,
        "max_ca3_cosine": max_cos,
        "store_criterion": _STORE_COS_CRITERION,
        "multiseed_baseline_max_cos": _MULTISEED_BASELINE_MAX_COS,
        "pass_position": pass_position,
        "pass_word": pass_word,
        "store_distinctness_PASS": store_pass,
        # carried so PART B reuses the SAME bridge (the additive
        # pathway must be present + the store already encoded).
        "_bridge": bridge,
    }


def _run_moat_probe(bridge, seed: int, log) -> dict:
    """PART B: no-confabulation abstention moat, unregressed. On the
    SAME bridge, REUSE order_intrinsic_encode.encode_proposition (trains
    the additive read-back pathway via its DISJOINT gate) +
    .readback_sweep, and the project's pure
    order_intrinsic_core.decode_position_sweep / control_max_floor
    idiom. DRY: nothing reimplemented. The moat = an encoded
    (word,position)'s correct-motor top rate clearly SEPARATES from a
    never-encoded control position's top rate (control-max floor; this
    DG/CA3 regime is NOT the G.20 650-regime -- no literal 650).
    """
    from research.runners.order_intrinsic_encode import (
        encode_proposition, readback_sweep, _WORD_TO_ACTION,
    )
    from research.runners.order_intrinsic_core import (
        decode_position_sweep, control_max_floor,
    )

    prop = list(_MOAT_PROPOSITION)
    L = len(prop)
    log("  encoding KNOWN proposition %s via "
        "encode_proposition (trains the additive read-back "
        "pathway; its gate is DISJOINT from the validated store) ..."
        % prop)
    tag = encode_proposition(
        bridge, prop, tag_name="moat_prop",
        n_lang_input=1024, word_seed=seed,
        n_ec_context=200, encoding_steps=60, verbose=False)
    log("    encoded engram tag=%s" % tag)

    # Sweep L+1 positions: 0..L-1 are encoded; position L was NEVER
    # encoded -> its top motor rate is the control floor / the
    # no-confabulation abstention proof.
    log("  readback_sweep over %d positions (0..%d encoded; %d "
        "NEVER encoded = control / abstention proof) ..."
        % (L + 1, L - 1, L))
    per_pos = readback_sweep(bridge, length=L + 1,
                             n_ec_context=200, stim_steps=80)

    # Encoded-signal vs control-floor top rates.
    def _top(rates):
        if not rates:
            return (None, 0.0)
        bk, bv = None, None
        for k, v in rates.items():
            fv = float(v)
            if bv is None or fv > bv:
                bv, bk = fv, k
        return (bk, bv)

    encoded_tops = []          # (pos, top_word, top_rate, expected)
    encoded_correct_rates = []  # rate of the CORRECT motor at each pos
    for k in range(L):
        tw, tr = _top(per_pos[k])
        expected = prop[k]
        encoded_tops.append({
            "position": k, "expected_word": expected,
            "top_word": tw, "top_rate": round(tr, 5),
            "expected_rate": round(
                float(per_pos[k].get(expected, 0.0)), 5),
            "top_is_expected": bool(tw == expected),
        })
        # the encoded-signal strength at this slot = the correct
        # word's motor pool rate (the thing the trained read-back
        # pathway is supposed to reactivate).
        encoded_correct_rates.append(
            float(per_pos[k].get(expected, 0.0)))

    ctrl_word, ctrl_rate = _top(per_pos[L])
    log("    per-position top motor rates:")
    for e in encoded_tops:
        log("      pos=%d expected=%-5s top=%-5s rate=%.4f "
            "exp_rate=%.4f %s"
            % (e["position"], e["expected_word"], str(e["top_word"]),
               e["top_rate"], e["expected_rate"],
               "OK" if e["top_is_expected"] else "(top!=expected)"))
    log("      pos=%d NEVER-ENCODED control: top=%-5s rate=%.4f"
        % (L, str(ctrl_word), float(ctrl_rate)))

    # Control-max floor over the never-encoded control position's
    # per-word rates (the project's pure control_max_floor idiom; the
    # control-max operating point -- NOT a literal 650; DG/CA3 regime).
    control_rates = [float(v) for v in per_pos[L].values()]
    encoded_toprates = [float(e["top_rate"]) for e in encoded_tops]
    moat_floor = control_max_floor(encoded_toprates, control_rates)

    # Pure abstention idiom: decode the encoded slots vs the
    # control-max floor (decode_position_sweep == the no-confabulation
    # moat applied per position). Encoded slots should decode (not
    # abstain); the never-encoded control slot should abstain.
    decoded, conf, abstained = decode_position_sweep(
        per_pos, floor=moat_floor)
    encoded_decoded_any = any(
        decoded[k] is not None for k in range(L))
    control_abstained = (L in abstained)

    # Moat separation: the MEAN encoded-correct-motor rate must clearly
    # exceed the control-max floor by more than substrate noise (the
    # encoded signal >> control, the moat). A small relative margin
    # guards stochasticity -- this DG/CA3 read-back regime is a
    # different regime from G.20's 650 (per spec, no 650 here).
    mean_encoded = (sum(encoded_correct_rates)
                    / max(1, len(encoded_correct_rates)))
    margin_abs = mean_encoded - moat_floor
    denom = max(abs(mean_encoded), abs(moat_floor), 1e-9)
    margin_rel = margin_abs / denom
    separates = bool(margin_rel >= _MOAT_REL_MARGIN
                     and mean_encoded > moat_floor)

    moat_pass = bool(separates
                     and encoded_decoded_any
                     and control_abstained)

    log("  abstention moat (control-max floor = %.5f; NOT a "
        "literal 650 -- DG/CA3 read-back regime):" % moat_floor)
    log("    mean encoded-correct motor rate : %.5f" % mean_encoded)
    log("    control-max floor               : %.5f" % moat_floor)
    log("    separation margin (rel)         : %.4f "
        "(need >= %.2f)" % (margin_rel, _MOAT_REL_MARGIN))
    log("    encoded decodes (not abstain)   : %s"
        % ("YES" if encoded_decoded_any else "NO"))
    log("    never-encoded control abstains  : %s"
        % ("YES (no confab)" if control_abstained
           else "NO (CONFABULATED)"))
    log("    MOAT: %s" % ("PASS" if moat_pass else "FAIL"))

    return {
        "seed": seed,
        "proposition": prop,
        "encoded_tops": encoded_tops,
        "control_position": L,
        "control_top_word": ctrl_word,
        "control_top_rate": round(float(ctrl_rate), 5),
        "control_max_floor": round(float(moat_floor), 5),
        "mean_encoded_correct_rate": round(float(mean_encoded), 5),
        "moat_margin_abs": round(float(margin_abs), 5),
        "moat_margin_rel": round(float(margin_rel), 5),
        "moat_rel_margin_required": _MOAT_REL_MARGIN,
        "encoded_signal_separates": separates,
        "encoded_decoded_any": bool(encoded_decoded_any),
        "control_abstained": bool(control_abstained),
        "decoded_sweep": [str(d) for d in decoded],
        "abstained_positions": list(abstained),
        "moat_PASS": moat_pass,
    }


def main() -> int:
    import json
    import time
    from pathlib import Path

    t0 = time.time()
    print("=" * 64, flush=True)
    print("ORDER-INTRINSIC TASK 6 -- LOAD-BEARING NO-HARM CHECK",
          flush=True)
    print("(validated DG/CA3 (word,position)->distinct store "
          "distinctness", flush=True)
    print(" + no-confabulation abstention moat must stay UNREGRESSED",
          flush=True)
    print(" with the additive ec_context->motor read-back pathway "
          "PRESENT)", flush=True)
    print("=" * 64, flush=True)
    print("seeds=%s  train_events=%d  store_criterion=cos<%.1f  "
          "multiseed_baseline=cos<%.2f"
          % (_SEEDS, _TRAIN_EVENTS, _STORE_COS_CRITERION,
             _MULTISEED_BASELINE_MAX_COS), flush=True)
    print("DRY: reuses validate_positional_binding.{build_word_pattern,"
          "encode_and_tag,cosine_similarity_indices} +", flush=True)
    print("     order_intrinsic_encode.{build_order_intrinsic_bridge,"
          "encode_proposition,readback_sweep} +", flush=True)
    print("     order_intrinsic_core.{decode_position_sweep,"
          "control_max_floor} -- nothing reimplemented", flush=True)

    per_seed = []
    for seed in _SEEDS:
        print("\n" + "-" * 64, flush=True)
        print("SEED %d" % seed, flush=True)
        print("-" * 64, flush=True)
        log = (lambda *a: print(*a, flush=True))

        print("[PART A] store-distinctness no-harm "
              "(validate_positional_binding logic, REUSED)", flush=True)
        a = _run_store_distinctness_noharm(seed, _TRAIN_EVENTS, log)
        bridge = a.pop("_bridge")

        print("\n[PART B] no-confabulation abstention moat "
              "(unregressed)", flush=True)
        b = _run_moat_probe(bridge, seed, log)

        seed_pass = bool(a["store_distinctness_PASS"]
                         and b["moat_PASS"])
        print("\n  SEED %d: store=%s moat=%s -> %s"
              % (seed,
                 "PASS" if a["store_distinctness_PASS"] else "FAIL",
                 "PASS" if b["moat_PASS"] else "FAIL",
                 "PASS" if seed_pass else "FAIL"), flush=True)
        per_seed.append({
            "seed": seed,
            "store": a,
            "moat": b,
            "seed_PASS": seed_pass,
        })
        # free GPU between seeds (best-effort).
        try:
            del bridge
            from sim.backend import get_backend
            cp, name = get_backend()
            if name == "cupy":
                cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    n_seeds = len(per_seed)
    n_store_pass = sum(
        1 for s in per_seed if s["store"]["store_distinctness_PASS"])
    n_moat_pass = sum(1 for s in per_seed if s["moat"]["moat_PASS"])
    n_seed_pass = sum(1 for s in per_seed if s["seed_PASS"])
    # The LOAD-BEARING no-harm gate: EVERY seed's store distinctness
    # AND moat must stay PASS (matching the 3/3 multi-seed baseline).
    store_distinctness_PASS = (n_store_pass == n_seeds and n_seeds > 0)
    moat_PASS = (n_moat_pass == n_seeds and n_seeds > 0)
    overall_pass = bool(store_distinctness_PASS and moat_PASS)
    max_cos_all = max(
        (s["store"]["max_ca3_cosine"] for s in per_seed), default=1.0)
    min_moat_margin_rel = min(
        (s["moat"]["moat_margin_rel"] for s in per_seed), default=-1.0)

    result = {
        "task": "order-intrinsic Task 6 LOAD-BEARING no-harm check",
        "substrate": ("multi-seed-validated DG/CA3 P4.1 store "
                      "(validate_positional_binding) + additive "
                      "ec_context_to_motor_readback pathway PRESENT"),
        "method": ("PART A: REUSE validate_positional_binding's exact "
                   "4-tag encode + pairwise-CA3-cosine + < 0.4 PASS "
                   "criteria (build_word_pattern / encode_and_tag / "
                   "cosine_similarity_indices imported verbatim) on a "
                   "build_order_intrinsic_bridge bridge. PART B: REUSE "
                   "order_intrinsic_encode.encode_proposition + "
                   "readback_sweep + order_intrinsic_core."
                   "decode_position_sweep/control_max_floor; encoded "
                   "proposition's mean correct-motor rate must clearly "
                   "separate from a never-encoded control-max floor "
                   "(no literal 650 -- DG/CA3 regime) AND the "
                   "never-encoded control must abstain"),
        "seeds": _SEEDS,
        "train_events": _TRAIN_EVENTS,
        "store_cos_criterion": _STORE_COS_CRITERION,
        "multiseed_baseline_max_cos": _MULTISEED_BASELINE_MAX_COS,
        "moat_rel_margin_required": _MOAT_REL_MARGIN,
        "n_seeds": n_seeds,
        "n_store_distinctness_pass": n_store_pass,
        "n_moat_pass": n_moat_pass,
        "n_seed_pass": n_seed_pass,
        "max_ca3_cosine_all_seeds": max_cos_all,
        "min_moat_margin_rel_all_seeds": min_moat_margin_rel,
        "store_distinctness_PASS": store_distinctness_PASS,
        "moat_encoded_vs_control": {
            s["seed"]: {
                "mean_encoded_correct_rate":
                    s["moat"]["mean_encoded_correct_rate"],
                "control_max_floor":
                    s["moat"]["control_max_floor"],
                "margin_rel": s["moat"]["moat_margin_rel"],
            } for s in per_seed
        },
        "moat_PASS": moat_PASS,
        "OVERALL_PASS": overall_pass,
        "per_seed": per_seed,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    Path(_OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    # --- ASCII verdict block ---------------------------------------
    print("\n" + "=" * 64, flush=True)
    print("ORDER-INTRINSIC TASK 6 NO-HARM VERDICT", flush=True)
    print("=" * 64, flush=True)
    print("  substrate : validated DG/CA3 P4.1 store + additive "
          "ec_context->motor readback PRESENT", flush=True)
    print("  seeds     : %s (multi-seed, matches 3/3 baseline)"
          % _SEEDS, flush=True)
    print("  -" * 32, flush=True)
    for s in per_seed:
        st = s["store"]
        mt = s["moat"]
        print("  seed %d:" % s["seed"], flush=True)
        print("    store  max_cos=%.3f (need < %.1f; baseline < "
              "%.2f) -> %s"
              % (st["max_ca3_cosine"], _STORE_COS_CRITERION,
                 _MULTISEED_BASELINE_MAX_COS,
                 "PASS" if st["store_distinctness_PASS"]
                 else "FAIL"), flush=True)
        print("      apple_pos %.3f  alice_pos %.3f  "
              "pos0_word %.3f  pos2_word %.3f"
              % (st["cos_apple_pos"], st["cos_alice_pos"],
                 st["cos_pos0_word"], st["cos_pos2_word"]),
              flush=True)
        print("    moat   enc=%.4f ctrl_floor=%.4f rel_margin=%.3f "
              "(need >= %.2f) ctrl_abstain=%s -> %s"
              % (mt["mean_encoded_correct_rate"],
                 mt["control_max_floor"], mt["moat_margin_rel"],
                 _MOAT_REL_MARGIN,
                 "Y" if mt["control_abstained"] else "N",
                 "PASS" if mt["moat_PASS"] else "FAIL"), flush=True)
    print("  -" * 32, flush=True)
    print("  store-distinctness no-harm : %d/%d seeds PASS -> %s"
          % (n_store_pass, n_seeds,
             "PASS" if store_distinctness_PASS else "FAIL"),
          flush=True)
    print("    (max CA3 cosine across all seeds = %.3f; "
          "multi-seed 3/3 baseline < %.2f)"
          % (max_cos_all, _MULTISEED_BASELINE_MAX_COS), flush=True)
    print("  abstention moat unregressed: %d/%d seeds PASS -> %s"
          % (n_moat_pass, n_seeds,
             "PASS" if moat_PASS else "FAIL"), flush=True)
    print("  -" * 32, flush=True)
    print("  OVERALL : %s" % ("PASS" if overall_pass else "FAIL"),
          flush=True)
    if not overall_pass:
        why = []
        if not store_distinctness_PASS:
            why.append("validated DG/CA3 store distinctness "
                       "REGRESSED by the additive read-back pathway "
                       "(max_cos=%.3f vs baseline < %.2f)"
                       % (max_cos_all, _MULTISEED_BASELINE_MAX_COS))
        if not moat_PASS:
            why.append("no-confabulation abstention moat REGRESSED")
        print("  WHY FAIL: %s" % "; ".join(why), flush=True)
        print("  -> STOP. Do NOT proceed to Task 7. Do NOT re-run to "
              "chase a pass. Do NOT weaken validate_positional_binding.",
              flush=True)
        print("  -> The additive pathway's separation-of-concerns is "
              "wrong (it bleeds into the validated DG/CA3 store path).",
              flush=True)
    else:
        print("  -> Task 7 pre-registered multi-seed gate is "
              "AUTHORIZED.", flush=True)
    print("  -> %s" % _OUT_JSON, flush=True)
    print("=" * 64, flush=True)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
