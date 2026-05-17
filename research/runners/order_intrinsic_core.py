"""Pure read-back / scoring core for the order-intrinsic
conversational-memory cheap-first slice.

This module is PURE numpy/stdlib: no cupy, no sim.bridge import. It
holds only the deterministic read-back/decoding/scoring logic for the
order-intrinsic line, where order is INTRINSIC via the validated
D.11/P4.1 positional-context store (a genuinely different architecture
after the 6-negative generative line -- the network does not GENERATE
an ordered sequence; instead each position is read back by a
deterministic position sweep over the positional store).

Task 3 will REUSE song_g1_core.g1_verdict / score_order /
permuted_order_controls UNMODIFIED -- the pre-registered anti-cheat
bars (permuted-ORDER control, absolute floor, +10% margin) are NEVER
reimplemented here. This file only adds the decode/aggregation glue;
later tasks append more pure functions (control_max_floor,
order_intrinsic_verdict, aggregate_multiseed) to it.
"""
from __future__ import annotations


def decode_position_sweep(per_pos_rates, floor):
    """Decode an ordered concept list from per-position pool-firing-
    rate dicts (the shape query_position returns: {concept: rate}).

    For each position: pick the concept with the max rate (stable
    FIRST-max on a tie). If that max rate is <= floor, the slot is
    None (ABSTAIN -- the no-confabulation moat applied per position:
    never emit a low-confidence/confabulated slot) and the position
    index is recorded in `abstained`.

    Pure / deterministic. Returns (decoded, conf, abstained):
      decoded   : list -- concept per position, or None if abstained
      conf      : list -- the chosen max rate per position (float)
      abstained : list -- position indices that abstained
    """
    decoded = []
    conf = []
    abstained = []
    for i, rates in enumerate(per_pos_rates):
        if not rates:
            decoded.append(None)
            conf.append(0.0)
            abstained.append(i)
            continue
        # stable first-max: iterate insertion order, strict > to keep
        # the first on ties (dict preserves insertion order, py3.7+)
        best_k = None
        best_v = None
        for k, v in rates.items():
            fv = float(v)
            if best_v is None or fv > best_v:
                best_v = fv
                best_k = k
        conf.append(best_v)
        if best_v <= float(floor):
            decoded.append(None)
            abstained.append(i)
        else:
            decoded.append(best_k)
    return decoded, conf, abstained


def control_max_floor(encoded_toprates, control_toprates) -> float:
    """Pre-registered control-calibrated abstention floor =
    control-MAX (the EXACT operating criterion that produced the
    prior frozen floors: the gate is set just above the max control
    top-rate). `encoded_toprates` is accepted for signature parity /
    transparency logging (e.g. an AUC could be reported alongside)
    but does NOT affect the bar -- the bar is control-max ONLY, so it
    can never be tuned by the encoded distribution. No controls ->
    0.0. Pure / deterministic."""
    if not control_toprates:
        return 0.0
    return float(max(float(x) for x in control_toprates))


def order_intrinsic_verdict(true_decoded, intended, perm_decoded,
                            gate_cleared) -> dict:
    """Pure pre-registered verdict for one held-out proposition.
    REUSES song_g1_core.score_order + g1_verdict UNMODIFIED -- the
    anti-cheat bars (_G1_MARGIN=0.10, _G1_ABS_FLOOR=0.5) are NEVER
    reimplemented or touched here.

    true_decoded : decoded ordered concept list from the position
                   sweep (None slots allowed -> score_order treats a
                   mismatch).
    intended     : the intended ordered concept list.
    perm_decoded : list of decoded lists, one per permuted-ORDER
                   control (the load-bearing anti-cheat: same concept
                   multiset, scrambled positions).
    gate_cleared : did the production clear the pre-registered
                   control-max abstention floor (computed elsewhere)?

    Returns the g1_verdict dict augmented with true_score /
    best_perm_score for the JSON record.
    """
    from research.runners.song_g1_core import score_order, g1_verdict
    true_score = score_order(true_decoded, intended)
    best_perm = max(
        (score_order(pd, intended) for pd in perm_decoded),
        default=0.0)
    v = g1_verdict(true_score, best_perm, gate_cleared)
    v["true_score"] = true_score
    v["best_perm_score"] = best_perm
    return v


def aggregate_multiseed(per_seed_prop_verdicts, min_seeds: int = 3) -> dict:
    """Pre-registered MULTI-SEED aggregate. PASS iff:
      (a) at least `min_seeds` seeds were run (>=3 mandatory --
          single-seed / near-noise is explicitly NOT a pass; the
          cheap probe proved single-seed unreliable at 50%), AND
      (b) EVERY held-out proposition in EVERY seed has GATE == "PASS".
    Any seed-prop FAIL, or fewer than min_seeds seeds, -> FAIL.
    Pure / deterministic. Returns a dict with the aggregate GATE +
    counts for the JSON record."""
    n_seeds = len(per_seed_prop_verdicts)
    n_props_total = 0
    n_props_pass = 0
    for seed_props in per_seed_prop_verdicts:
        for v in seed_props:
            n_props_total += 1
            if v.get("GATE") == "PASS":
                n_props_pass += 1
    enough_seeds = n_seeds >= int(min_seeds)
    all_pass = (n_props_total > 0
                and n_props_pass == n_props_total)
    gate = bool(enough_seeds and all_pass)
    return {
        "GATE": "PASS" if gate else "FAIL",
        "n_seeds": n_seeds,
        "min_seeds": int(min_seeds),
        "enough_seeds": enough_seeds,
        "n_props_total": n_props_total,
        "n_props_pass": n_props_pass,
        "all_pass": all_pass,
    }
