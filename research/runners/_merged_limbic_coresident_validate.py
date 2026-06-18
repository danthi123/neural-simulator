"""Validate the SHARED LIMBIC CORE lifted onto the merged 'one brain' (co_resident_limbic) —
TRUE ONE BRAIN item #1 merge lift (the audit's highest-leverage consolidation step).

The validated standalone organ (finding 2026-06-18-limbic-core-rpe-battery-GO.md, Schultz RPE
battery 6/6) is now an additive default-off opt-in on build_merged_nav_conv_bridge. This runner
validates it co-resident on the one brain:

  (A) CO-RESIDENCE + NAV-INERTNESS : the 4 limbic_ regions exist as disjoint slices; the limbic
      slice has ZERO cp_connections out-edges into non-limbic neurons (nav-inert, like rf/cortex_it).
  (B) DEFAULT-OFF BYTE-PRESERVED   : co_resident_limbic=False -> the limbic regions are ABSENT and
      every non-limbic region's base index is UNCHANGED (the production agent is unaffected; the lift
      is opt-in).
  (C) THE SPIKING RPE ARITHMETIC WORKS CO-RESIDENT : drive the merged-bridge limbic slice and confirm
      the canonical signatures hold ON the one brain (no learning needed — the GABA_B value
      subtraction is tested by driving limbic_cue directly so limbic_striosome fires at the init
      weight): burst-on-US >= 3x, graded corr >= +0.8, the value subtraction (cue -> striosome ->
      GABA_B -> limbic_snc reduced => predicted < unpredicted), reward-lesion -> burst vanishes,
      critic-GABA_B-lesion -> the gap collapses.

The on-merge critic LEARNING (V via three-factor) + the shared-DA-gates-nav routing + the
moat-no-regression-with-limbic-ON are the NEXT increment (#2); the production default is unaffected
here because the lift is opt-in (B).

Usage
-----
    SIM_BACKEND=cupy python -m research.runners._merged_limbic_coresident_validate --seed 42
    SIM_BACKEND=numpy python -m research.runners._merged_limbic_coresident_validate --smoke   # tiny numpy build+arithmetic
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

LIMBIC = ("limbic_cue", "limbic_striosome", "limbic_reward_us", "limbic_snc")


def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def _limbic_idx_map(bridge, xp):
    """Map the de-risk's logical keys -> the merged-bridge limbic_ regions (so the de-risk's _drive
    works verbatim: it indexes idx_map['snc'] / ['striosome_value'] / ['reward_us'] / ['cue'])."""
    import numpy as np
    rm = bridge.region_manager

    def gi(n):
        return xp.asarray(np.asarray(rm.indices(n), dtype=np.int64))
    return {"cue": gi("limbic_cue"), "striosome_value": gi("limbic_striosome"),
            "reward_us": gi("limbic_reward_us"), "snc": gi("limbic_snc")}


def _lesion_edges(bridge, pre_name, post_name):
    """Zero every pre_name->post_name edge in cp_connections (CSR rows=post, cols=pre, with a
    fallback to the other orientation)."""
    import numpy as np
    rm = bridge.region_manager
    pre_set = set(int(i) for i in rm.indices(pre_name))
    post_set = set(int(i) for i in rm.indices(post_name))
    coo = bridge.cp_connections.tocoo()
    rows = np.asarray(_host(coo.row), dtype=np.int64); cols = np.asarray(_host(coo.col), dtype=np.int64)
    mask = np.array([(r in post_set and c in pre_set) for r, c in zip(rows, cols)])
    if not mask.any():
        mask = np.array([(r in pre_set and c in post_set) for r, c in zip(rows, cols)])
        pre = rows[mask]; post = cols[mask]
    else:
        pre = cols[mask]; post = rows[mask]
    if len(pre) == 0:
        return 0
    return bridge.set_pathway_weights(f"{pre_name}->{post_name}(lesion)", pre, post,
                                      np.zeros(len(pre), dtype=np.float32))


def _lesion_gabab_mask(bridge):
    m = getattr(bridge, "cp_gabab_synapse_mask", None)
    if m is None:
        return 0
    n_was = int(_host(m).sum())
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge.cp_gabab_synapse_mask = xp.zeros_like(m)
    if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
        bridge.cp_conductance_g_gabab[:] = 0.0
    return n_was


def validate_structure(seed=42, vocab=None):
    """(A) co-residence + nav-inertness; (B) default-off byte-preserved. Returns the on-bridge."""
    import numpy as np
    from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge

    b_on, _ = build_merged_nav_conv_bridge(seed=seed, vocab=vocab, co_resident_limbic=True)
    rm = b_on.region_manager
    names_on = set(r.name for r in rm.regions())
    missing = [n for n in LIMBIC if n not in names_on]
    assert not missing, f"(A) FAIL limbic regions missing: {missing}"

    # nav-inertness: no cp_connections edge from a limbic PRE to a non-limbic POST.
    limbic_idx = set()
    for n in LIMBIC:
        limbic_idx |= set(int(i) for i in rm.indices(n))
    coo = b_on.cp_connections.tocoo()
    rows = np.asarray(_host(coo.row), dtype=np.int64)   # post
    cols = np.asarray(_host(coo.col), dtype=np.int64)   # pre
    out_edges = int(sum(1 for r, c in zip(rows, cols) if int(c) in limbic_idx and int(r) not in limbic_idx))
    assert out_edges == 0, f"(A) FAIL limbic slice not nav-inert: {out_edges} out-edges into non-limbic"
    # and confirm the limbic INTERNAL edges exist (the organ is wired).
    in_edges = int(sum(1 for r, c in zip(rows, cols) if int(c) in limbic_idx and int(r) in limbic_idx))
    assert in_edges > 0, "(A) FAIL limbic slice has no internal edges (organ not wired)"

    # (B) default-off: build WITHOUT limbic; assert limbic absent + non-limbic bases identical.
    b_off, _ = build_merged_nav_conv_bridge(seed=seed, vocab=vocab, co_resident_limbic=False)
    rm_off = b_off.region_manager
    names_off = set(r.name for r in rm_off.regions())
    leaked = [n for n in LIMBIC if n in names_off]
    assert not leaked, f"(B) FAIL limbic leaked into default-off build: {leaked}"
    drift = []
    for r in rm_off.regions():
        if int(rm_off.indices(r.name)[0]) != int(rm.indices(r.name)[0]):
            drift.append(r.name)
    assert not drift, f"(B) FAIL non-limbic base drift (byte-identity broken): {drift}"
    n_neurons_off = b_off.runtime_state.num_neurons if hasattr(b_off.runtime_state, "num_neurons") else None
    print(f"  (A) co-residence + nav-inertness: PASS ({len(LIMBIC)} limbic regions, "
          f"{out_edges} out-edges into nav, {in_edges} internal edges)")
    print(f"  (B) default-off byte-preserved: PASS (limbic absent, {len(list(rm_off.regions()))} non-limbic "
          f"regions all base-identical)")
    return b_on


def validate_arithmetic(bridge, *, snc_tonic_pa=220.0, us_drive_pa=600.0, cue_drive_pa=600.0, hold_steps=40):
    """(C) the spiking RPE arithmetic on the merged-bridge limbic slice (no learning; the GABA_B
    value subtraction is exercised by driving limbic_cue at the init critic weight)."""
    import numpy as np
    from sim.backend import get_backend
    from research.runners._limbic_core_rpe_battery_derisk import _drive, _settle
    xp, _ = get_backend()
    idx = _limbic_idx_map(bridge, xp)

    # The organ's operating point (finding 2026-06-18-limbic-core-rpe-battery-GO.md) was pinned WITH OU
    # spontaneous activity (sigma~100, the CoreSimConfig default). The merged bridge sets OU OFF for the
    # resting-nav DETERMINISM config (not a biological constraint — the brain always has spontaneous
    # synaptic bombardment). The merged builder KEEPS the OU state allocated precisely so a read can
    # re-enable it per the builder note. Re-enable it for the limbic measurement = the faithful
    # re-validation at the pinned operating point; restore the resting config after.
    cc = bridge.core_config
    _ou_saved = (cc.enable_ou_process, cc.ou_std_current_pA)
    cc.enable_ou_process = True
    cc.ou_std_current_pA = 100.0

    def measure(drives):
        _settle(bridge, xp)
        return _drive(bridge, idx, drives, hold_steps, xp)[0]

    base = measure({"snc": snc_tonic_pa})
    unpred = measure({"reward_us": us_drive_pa, "snc": snc_tonic_pa})               # burst on US
    pred = measure({"cue": cue_drive_pa, "reward_us": us_drive_pa, "snc": snc_tonic_pa})  # cue->striosome->GABA_B subtracts
    mags = [0.25, 0.5, 0.75, 1.0]
    graded = [measure({"reward_us": us_drive_pa * m, "snc": snc_tonic_pa}) for m in mags]
    corr = float(np.corrcoef(mags, graded)[0, 1]) if len(set(graded)) > 1 else 0.0
    burst_ratio = unpred / max(base, 1e-6)
    gap = unpred / max(pred, 1e-6)

    # (6) CRITIC GABA_B LESION FIRST (on the INTACT reward path): zero the GABA_B routing mask so the
    # cue-driven value subtraction vanishes -> the predicted burst rises to match the unpredicted ->
    # the gap collapses to ~1.0. This proves the subtraction WAS the synaptic GABA_B (not host arithmetic).
    # Done before the reward lesion so the burst is intact (a degenerate post-reward-lesion gap would be ~1.0
    # for the wrong reason — both fall to tonic).
    n_cl = _lesion_gabab_mask(bridge)
    pred_cl = measure({"cue": cue_drive_pa, "reward_us": us_drive_pa, "snc": snc_tonic_pa})
    unpred_cl = measure({"reward_us": us_drive_pa, "snc": snc_tonic_pa})
    gap_cl = unpred_cl / max(pred_cl, 1e-6)
    gap_collapses = gap_cl <= 1.2

    # (5) REWARD LESION (the GABA_B is already cut, irrelevant to the no-cue burst): zero
    # limbic_reward_us->limbic_snc -> the burst vanishes to tonic.
    n_rl = _lesion_edges(bridge, "limbic_reward_us", "limbic_snc")
    unpred_rl = measure({"reward_us": us_drive_pa, "snc": snc_tonic_pa})
    reward_vanishes = abs(unpred_rl - base) <= 0.20 * max(base, 1e-6)

    res = dict(base_hz=base, unpredicted_hz=unpred, predicted_hz=pred, burst_ratio=burst_ratio,
               gap_ratio=gap, corr_mag=corr, graded_hz=graded,
               reward_lesion_unpred_hz=unpred_rl, reward_vanishes=bool(reward_vanishes), n_reward_edges=n_rl,
               gabab_lesion_gap=gap_cl, gap_collapses=bool(gap_collapses), n_gabab_syn=n_cl)
    burst_ok = burst_ratio >= 3.0
    graded_ok = corr >= 0.8
    subtract_ok = gap > 1.2   # predicted clearly < unpredicted (the GABA_B value subtraction)
    print(f"  (C) arithmetic on the merged limbic slice:")
    print(f"      tonic={base:.1f} unpred={unpred:.1f} pred={pred:.1f} Hz | burst {burst_ratio:.2f}x (>=3: {burst_ok}) "
          f"| graded corr {corr:+.2f} (>=+0.8: {graded_ok}) | value-subtract gap {gap:.2f} (>1.2: {subtract_ok})")
    print(f"      reward-lesion unpred {unpred_rl:.1f}Hz vs tonic {base:.1f}Hz (vanishes: {reward_vanishes}, {n_rl} edges) "
          f"| GABA_B-lesion gap {gap_cl:.2f} (collapses <=1.2: {gap_collapses}, {n_cl} syn)")
    res["pass_core"] = bool(burst_ok and graded_ok and subtract_ok and reward_vanishes and gap_collapses)
    cc.enable_ou_process, cc.ou_std_current_pA = _ou_saved   # restore the resting-nav config
    return res


def validate_diag(bridge):
    """Instrument the per-region limbic firing rates on the merged bridge under each drive, with learning
    FROZEN, to localize why the on-merge operating point differs from the standalone de-risk."""
    import numpy as np
    from sim.backend import get_backend
    from research.runners._limbic_core_rpe_battery_derisk import _settle
    xp, _ = get_backend()
    rm = bridge.region_manager
    idx = {n: xp.asarray(np.asarray(rm.indices(n), dtype=np.int64)) for n in LIMBIC}
    cc = bridge.core_config
    _ou = (cc.enable_ou_process, cc.ou_std_current_pA)
    cc.enable_ou_process = True; cc.ou_std_current_pA = 100.0
    _lr = cc.reward_learning_rate; cc.reward_learning_rate = 0.0   # freeze learning

    def rates(drives, steps=60):
        _settle(bridge, xp)
        bridge.cp_external_input_current[:] = 0.0
        for nm, pa in drives.items():
            bridge.cp_external_input_current[idx[nm]] = xp.float32(pa)
        c = {n: 0 for n in LIMBIC}
        for _ in range(steps):
            bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
            bridge.runtime_state.current_time_ms = bridge.runtime_state.current_time_step * cc.dt_ms
            for n in LIMBIC:
                c[n] += int(bridge.cp_firing_states[idx[n]].sum())
        return {n: c[n] / max(len(list(rm.indices(n))), 1) / (steps * 1e-3) for n in LIMBIC}

    print("  [DIAG] per-region limbic firing rates (Hz), learning frozen, OU on:")
    for label, d in [("tonic", {"limbic_snc": 220}),
                     ("US600", {"limbic_reward_us": 600, "limbic_snc": 220}),
                     ("US1500", {"limbic_reward_us": 1500, "limbic_snc": 220}),
                     ("cue600", {"limbic_cue": 600, "limbic_snc": 220}),
                     ("cue1500", {"limbic_cue": 1500, "limbic_snc": 220}),
                     ("cue+US", {"limbic_cue": 600, "limbic_reward_us": 600, "limbic_snc": 220})]:
        r = rates(d)
        print(f"    {label:8} reward_us={r['limbic_reward_us']:6.1f}  striosome={r['limbic_striosome']:6.1f}  "
              f"snc={r['limbic_snc']:6.1f}")
    cc.enable_ou_process, cc.ou_std_current_pA = _ou
    cc.reward_learning_rate = _lr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny numpy build+arithmetic only (skip the 2nd build)")
    ap.add_argument("--diag", action="store_true", help="instrument per-region limbic rates on the merged bridge")
    ap.add_argument("--moat", action="store_true", help="moat-no-regression: conversation survives the shared DA modulator")
    ap.add_argument("--sweep", action="store_true", help="systematic operating-point search (one build, drive grid)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.diag:
        from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge
        b_on, _ = build_merged_nav_conv_bridge(seed=args.seed, co_resident_limbic=True)
        validate_diag(b_on)
        return

    if args.sweep:
        # Systematic operating-point search on the merged bridge (ONE build): grid (tonic, us, cue) and read
        # base/unpred/pred NON-destructively (no lesions), so the working point (burst>=3x AND gap>1.2 with the
        # right direction pred<unpred) is found in one shot instead of guess-and-check rebuilds.
        import numpy as np
        from sim.backend import get_backend
        from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge
        from research.runners._limbic_core_rpe_battery_derisk import _drive, _settle
        xp, _ = get_backend()
        het = os.environ.get("MERGED_HET_TEST") == "1"   # de-risk: does heterogeneity restore the arithmetic?
        b, _h = build_merged_nav_conv_bridge(seed=args.seed, co_resident_limbic=True, _global_het_test=het)
        print(f"  (global heterogeneity test: {het})")
        idx = _limbic_idx_map(b, xp)
        cc = b.core_config
        cc.enable_ou_process = True; cc.ou_std_current_pA = 100.0
        cc.reward_learning_rate = 0.0

        def meas(drives):
            _settle(b, xp)
            return _drive(b, idx, drives, 40, xp)[0]
        print("  [SWEEP] merged limbic operating point (burst=unpred/base, gap=unpred/pred; want burst>=3 gap>1.2):")
        best = None
        for tonic in (160, 185, 210):
            for us in (400, 600, 800):
                for cue in (800, 1200):
                    base = meas({"snc": tonic})
                    unpred = meas({"reward_us": us, "snc": tonic})
                    pred = meas({"cue": cue, "reward_us": us, "snc": tonic})
                    burst = unpred / max(base, 1e-6); gap = unpred / max(pred, 1e-6)
                    ok = (burst >= 3.0) and (gap > 1.2)
                    print(f"    tonic={tonic} us={us} cue={cue} | base={base:5.1f} unpred={unpred:6.1f} "
                          f"pred={pred:6.1f} | burst={burst:5.2f}x gap={gap:5.2f} {'<== OK' if ok else ''}")
                    if ok and (best is None or gap > best[-1]):
                        best = (tonic, us, cue, burst, gap)
        print(f"\n  BEST: {best}")
        return

    if args.moat:
        # (D) MOAT-NO-REGRESSION: the shared `dopamine` modulator (scope="all", threshold-0 neutral-at-rest) must
        # NOT break the conversational comprehension/abstention. Build the agent WITH the limbic core co-resident
        # (+ the co-resident composer) and run who/what + the no-confab abstain — the same surface the conversational
        # gate b asserts. PASS = recall correct AND the moat abstains on an unheard cue (the parser/composer survive
        # the modulator's presence).
        from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
        ag = MergedNavConvAgent(seed=args.seed, co_resident_composer=True, co_resident_limbic=True)
        ag.hear("dog go north")   # vocab words (the conversational gate's validated sentence)
        recall = ag.what_does("dog", "go")
        unheard = ag.what_does("river", "look")   # never stored -> the moat must abstain (None)
        recall_ok = (recall == "north")
        moat_ok = (unheard is None)
        print(f"  (D) moat-no-regression WITH limbic co-resident: recall what_does('dog','go')={recall!r} "
              f"(==north: {recall_ok}) | unheard what_does('river','look')={unheard!r} (abstains: {moat_ok})")
        verdict = "GO" if (recall_ok and moat_ok) else "REGRESSION"
        print(f"\n  MOAT (seed {args.seed}): {verdict}  [the shared DA modulator does not perturb conversation]")
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"mode": "merged_limbic_moat", "seed": args.seed, "recall": recall,
                           "recall_ok": recall_ok, "moat_ok": moat_ok, "verdict": verdict}, f, indent=2)
        return

    from sim.backend import get_backend
    _, backend = get_backend()
    print(f"[merged-limbic validate seed={args.seed}] backend={backend}")

    if args.smoke:
        from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge
        b_on, _ = build_merged_nav_conv_bridge(seed=args.seed, co_resident_limbic=True)
        rm = b_on.region_manager
        assert all(n in set(r.name for r in rm.regions()) for n in LIMBIC), "limbic regions missing"
        print(f"  smoke build OK: {len(list(rm.regions()))} regions incl. {LIMBIC}")
        res = validate_arithmetic(b_on)
        print(f"\n  SMOKE arithmetic pass_core={res['pass_core']}")
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"mode": "merged_limbic_smoke", "seed": args.seed, "arithmetic": res}, f, indent=2)
        return

    b_on = validate_structure(seed=args.seed)
    res = validate_arithmetic(b_on)
    verdict = "GO" if res["pass_core"] else "PARTIAL"
    print(f"\n  MERGED LIMBIC LIFT (seed {args.seed}): {verdict}  "
          f"[co-residence+inertness PASS, default-off PASS, arithmetic pass_core={res['pass_core']}]")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mode": "merged_limbic_validate", "seed": args.seed, "arithmetic": res,
                       "verdict": verdict}, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
