"""Cheap-first de-risk for TRUE-ONE-BRAIN ROADMAP #3 — CONSOLIDATE the validated A-CSC TD
cue-shift machinery onto the merged "one brain" (build_merged_nav_conv_bridge).

Scoping: research/findings/2026-06-18-TD-cueshift-dendrite-decision-scoping.md §3 (the
recommended de-risk + the frozen GO bar). The TD cue-shift is ALREADY a multi-seed POINT-neuron
GO on the standalone CPU probe (research/findings/2026-06-10-N9-TD-cue-shift-A-CSC-GO.md:
migration r = -0.80/-0.77/-0.89, full Schultz signature, both anti-cheats decisive). The merged
"one brain" currently has only the Rescorla-Wagner limbic core (delta=r-V). This de-risk lifts
the A-CSC TD machinery onto the merge as an additive, default-OFF `co_resident_td_cueshift` slice
(mirroring the validated co_resident_limbic lift) and re-runs the SAME Pavlovian battery the
standalone passed, PLUS the two consolidation gates.

THE QUESTION (NOT "can point neurons do the cue-shift" — answered yes; it is): does the validated
A-CSC cue-shift SURVIVE co-resident on the merged bridge, alongside the conversational moat + the
nav byte-identity, WITHOUT dendrites?

REUSE-BY-IMPORT: the validated A-CSC recipe (snc_stageb_critic_probe.py helpers) is lifted
VERBATIM; only the BRIDGE is the merged one (TD slice via build_merged_nav_conv_bridge). NO new
`sim/` edit (the B-2 conductance-derivative is already shipped + byte-approved; it is ON for the TD
slice, byte-identical when the slice is OFF).

THE A-CSC TD CUE-SHIFT (the reward is the dependent variable; behavior is NOT the test)
---------------------------------------------------------------------------------------
  td_csc_0..td_csc_{K-1}  (the tapped-delay cue, each ITS OWN plastic synapse onto the critic)
       |  (plastic, the per-tap value w_k)
       v
  td_striosome (GABAergic MSN-D1 critic; learns V)  --GABA_B/GIRK--> td_snc (-V LEVEL + dV/dt source)
       |  (the critic INHIBITS the reward relay => r - V at the SNc)            ^
       v                                                                        | tonic pacemaker
  td_reward_us (excitatory reward relay) ----------------------------------> td_snc (DOPAMINE)
  + td_fs (the production FS-clamp: holds the critic SPARSE as the per-tap weights grow)
  dopamine modulator: from_region_firing_signed over [td_snc] -> plasticity_rate scope=all.
  B-2 conductance-derivative (+dV/dt) at the td_snc membrane = the bootstrap gamma*V(s')-V(s).

FROZEN GO BAR (pre-registered, inherited from the A-CSC GO so the bar is NOT tuned on the test):
  (headline) migration r < -0.7, sign-consistent (cue-ward), >= 5/6 seeds.
  early-burst-at-US -> late-burst-at-CS (genuine transfer, graded HS98-faithful PASS ok).
  omission dip at the expected-reward time; no burst in the CS->US gap (value flat); cue value grows.
  CONSOLIDATION GATES (decisive for "one brain"):
    (1) MOAT byte-intact: MergedNavConvAgent(co_resident_td_cueshift=True).what_does('dog','go')=='north'
        AND what_does('river','look') is None (the dopamine scope=all broadcast must NOT perturb the
        frozen conversational slice).
    (2) NAV byte-identity: the TD slice is additive/default-off + appended LAST -> the nav/parser/dlPFC/
        rf index bases are byte-unchanged (the existing nav/conv builds are untouched).
  ANTI-CHEATS (the TD error is NEURAL, the migration not a host/co-residence artifact):
    cue-pathway lesion -> migration vanishes, the US reflex survives (decisive);
    unpaired-timing control -> no migration (DISCRIMINATING).

Three-outcome framing (scoping §3.4): GO = roadmap #3 lands on the one brain + the dendrite
question stays closed-NEGATIVE; BOUNDARY = a co-residence/merge-engineering finding (NOT a dendrite
finding); NEGATIVE = the unlikely case that would re-open a temporal-dendrite candidacy (localize WHY
— the conductance filter under co-residence).

CPU-friendly first (SIM_BACKEND=numpy). The merged bridge is ~3300 neurons (the full nav cascade +
dlPFC), so each BUILD is ~100s — bound the seed count; commit early.

Usage
-----
    SIM_BACKEND=numpy python -m research.runners._merged_td_cueshift_consolidation_derisk --seed 42
    SIM_BACKEND=numpy python -m research.runners._merged_td_cueshift_consolidation_derisk \
        --seeds 42,43,44,100,101,102 --out research/findings/raw/_merged_td_cueshift_6seed.json
    # one-off gates:
    SIM_BACKEND=numpy python -m research.runners._merged_td_cueshift_consolidation_derisk --seed 42 --lesion
    SIM_BACKEND=numpy python -m research.runners._merged_td_cueshift_consolidation_derisk --seed 42 --unpaired
    SIM_BACKEND=numpy python -m research.runners._merged_td_cueshift_consolidation_derisk --seed 42 --moat-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Reuse-by-import the VALIDATED A-CSC machinery (the standalone GO recipe's helpers, lifted verbatim).
from research.runners.snc_stageb_critic_probe import (
    _drive_timecourse, _calibrate_da_threshold, _calibrate_da_baseline, _pearson_r,
    _lesion_pathway, _csc_substate_weights, _mean_pathway_weight,
)
from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge


# The locked A-CSC production recipe (finding 2026-06-10-N9-TD-cue-shift-A-CSC-GO.md §production recipe).
# The WIRING (weights) lives in build_merged_nav_conv_bridge's td slice; these are the DRIVES + protocol.
RECIPE = dict(
    csc_n=8, csc_drive_pa=600.0, snc_tonic_pa=300.0,
    csc_critic_tonic_pa=140.0, csc_critic_teacher_pa=700.0,
    csc_reward_us_drive_pa=600.0,
    bin_steps=20, n_post_bins=3, us_dur_bins=1, csc_iti_bins=8,
    n_train=50,
)

# The CO-RESIDENT operating-point overrides (the merged-bridge fix for the runaway-critic / tonic-death that the
# uncapped first run hit). The merged bridge pins stdp_w_max=400 (the 5a conversational-weight clip), which REMOVES
# the per-tap weight cap (40) the standalone CSC bridge used -> the critic ran to ~240, its GABA_B -V saturated, td_snc
# died, migration impossible. The fix: (a) the GIRK conductance cap bounds -V so td_snc survives at any critic rate;
# (b) a stronger FS-clamp keeps the critic sparse; (c) the runner re-clips the td_value weights to a LOCAL cap per
# trial (the standalone's weight-bound, enforced co-resident). These default into the builder kwargs.
OP = dict(
    td_csc_to_strio_weight=14.0, td_to_fs_weight=16.0, td_fs_to_strio_weight=10.0,
    td_strio_to_snc_weight=1.5, td_gabab_prop=0.105, td_gabab_conductance_max=0.0, td_stdp_w_max=0.0,
    td_derivative_gain=1.0, td_slow_tau_ms=130.0,
)


def _idx_arr(bridge, name, xp):
    import numpy as np
    return xp.asarray(np.asarray(bridge.region_manager.indices(name), dtype=np.int64))


def _build(seed, td_csc_n=8, vocab=None, op=None, global_het_test=False):
    """Build the merged nav+conv bridge WITH the co-resident A-CSC TD cue-shift slice. global_het_test=True is a
    DIAGNOSTIC hook ONLY (turns on global parameter heterogeneity, perturbing nav/conv determinism) to test whether
    the standalone's het-ON operating point restores the migration co-resident — if so, the named BOUNDARY fix is
    per-region heterogeneity for the td slice (a small additive sim/ analogue of the per-region NMDA mask)."""
    t0 = time.time()
    op = dict(OP, **(op or {}))
    bridge, handles = build_merged_nav_conv_bridge(
        seed=int(seed), vocab=vocab, co_resident_td_cueshift=True, td_csc_n=int(td_csc_n),
        _global_het_test=bool(global_het_test), **op)
    return bridge, handles, time.time() - t0


class _frozen_homeostasis:
    """Context manager that FREEZES homeostatic threshold adaptation (+ synaptic scaling) for a probe/test window.

    WHY (the lesion-non-discrimination root cause): the merged-config fix gives the td slice per-region homeostasis
    (`cp_homeostasis_neuron_mask`). During a frozen LESION test the cue->critic value conduit is cut, so td_snc fires
    only at its tonic floor (~3.5 Hz). Homeostasis SEES that low rate as below-target and keeps lowering td_snc's
    threshold across the settle window -> the tonic baseline CLIMBS (measured 3.5 -> 44 Hz, lesion_diag_s42.json), so a
    cue-bin "burst" of ~60 Hz is really the homeostatically-inflated tonic + a tiny transient, NOT a surviving
    value-driven burst. That inflated floor is exactly what makes the cue-pathway lesion non-discriminating co-resident
    (opsearch BOUNDARY). A frozen test must freeze ALL plasticity, not just reward-STDP -- homeostatic threshold
    adaptation IS plasticity. This pins the thresholds (and disables both the global flag AND the per-region mask) for
    the duration, then restores the live state EXACTLY. Runner-side measurement protocol; NO `sim/` edit, NO mechanism
    change (the value learning / derivative / burst stay 100% neural and unchanged; this only stops the probe-window
    threshold drift)."""
    def __init__(self, bridge):
        self.b = bridge
        self.cfg = bridge.core_config

    def __enter__(self):
        b, cfg = self.b, self.cfg
        self._saved_enable = cfg.enable_homeostasis
        self._saved_scaling = getattr(cfg, "enable_synaptic_scaling", False)
        self._saved_mask = getattr(b, "cp_homeostasis_neuron_mask", None)
        # Snapshot thresholds so any residual write is reverted on exit (belt-and-suspenders).
        thr = getattr(b, "cp_neuron_firing_thresholds", None)
        self._saved_thr = thr.copy() if thr is not None else None
        cfg.enable_homeostasis = False
        if hasattr(cfg, "enable_synaptic_scaling"):
            cfg.enable_synaptic_scaling = False
        if self._saved_mask is not None:
            b.cp_homeostasis_neuron_mask = None   # the mask path keeps homeostasis active even with the flag off
        return self

    def __exit__(self, *exc):
        b, cfg = self.b, self.cfg
        cfg.enable_homeostasis = self._saved_enable
        if hasattr(cfg, "enable_synaptic_scaling"):
            cfg.enable_synaptic_scaling = self._saved_scaling
        if self._saved_mask is not None:
            b.cp_homeostasis_neuron_mask = self._saved_mask
        if self._saved_thr is not None and getattr(b, "cp_neuron_firing_thresholds", None) is not None:
            b.cp_neuron_firing_thresholds[:] = self._saved_thr
        return False


def _clip_td_value_weights(bridge, K, w_max):
    """Re-clip ONLY the td_csc_k->td_striosome (td_value-gated) synapses to w_max via bridge.set_pathway_weights (the
    same (pre,post)->CSR mapper _lesion_pathway uses, so it is orientation-safe and writes the LIVE CSR the matvec +
    STDP read). This enforces the standalone CSC bridge's per-tap weight cap (stdp_w_max=40) co-resident, where the
    GLOBAL stdp_w_max is pinned at 400 for the conversational weights. A weight-BOUND, NOT a host computation of
    value/reward/delta — the cue-shift (value learning, the derivative, the burst, the credit) stays 100% neural.
    No-op when w_max<=0. Returns the number of edges clamped (for verification)."""
    if w_max is None or w_max <= 0:
        return 0
    import numpy as np
    from research.runners.snc_stageb_critic_probe import _host
    pre_set, post_set = set(), set(int(i) for i in bridge.region_manager.indices("td_striosome"))
    for k in range(K):
        pre_set |= set(int(i) for i in bridge.region_manager.indices(f"td_csc_{k}"))
    coo = bridge.cp_connections.tocoo()
    rows = np.asarray(_host(coo.row), dtype=np.int64)
    cols = np.asarray(_host(coo.col), dtype=np.int64)
    data = np.asarray(_host(coo.data), dtype=np.float32)
    # CSR orientation is rows=post, cols=pre — fall back to the other orientation if no edges match (orientation-safe,
    # exactly like _mean_pathway_weight / _lesion_pathway, which DO find these edges).
    mask = np.fromiter(((r in post_set and c in pre_set) for r, c in zip(rows, cols)), dtype=bool, count=len(rows))
    if mask.any():
        pre = cols[mask]; post = rows[mask]
    else:
        mask = np.fromiter(((r in pre_set and c in post_set) for r, c in zip(rows, cols)), dtype=bool, count=len(rows))
        pre = rows[mask]; post = cols[mask]
    if not mask.any():
        return 0
    vals = data[mask].copy()
    over = vals > w_max
    if not over.any():
        return 0
    vals[over] = w_max
    # Write ALL matched edges at their (clamped) current value — set_pathway_weights maps (pre,post)->CSR position.
    # Defensive: if STDP has mid-run made any (pre,post) pair un-writable, degrade gracefully (the GIRK cap + FS-clamp
    # are the primary, robust levers; this host weight-clip is a secondary belt-and-suspenders).
    try:
        bridge.set_pathway_weights("td_value(clip)", pre, post, vals.astype(np.float32))
    except Exception:
        return -1
    return int(over.sum())


def run_td_csc_merged(seed, *, lesion_cue=False, unpaired=False, verbose=True, td_csc_n=8, op=None, n_train_override=0,
                      global_het_test=False):
    """The A-CSC TD cue-shift Pavlovian protocol on the MERGED-bridge td_ slice. Mirrors
    snc_stageb_critic_probe.run_td_csc, but the bridge is build_merged_nav_conv_bridge's, the
    regions are td_-prefixed, and the calibrations/timecourses use the td slice indices."""
    from sim.backend import get_backend
    import numpy as np
    xp, _ = get_backend()
    K = int(td_csc_n)
    reward_bin = K - 1
    bridge, handles, build_s = _build(seed, td_csc_n=K, op=op, global_het_test=global_het_test)
    w_clip = float(handles.get("td_stdp_w_max", 0.0) or 0.0)
    if verbose:
        print(f"  [build] merged bridge in {build_s:.0f}s; num_neurons={bridge.core_config.num_neurons}; "
              f"td_snc base={handles['td']['td_snc']['base']}")

    # The td slice region names (so _drive_timecourse / the calibrations address the td_ slice). The standalone
    # helpers key on idx_map["snc"] and idx_map["striosome_value"] -> alias the td_ regions to those names.
    region_names = tuple(f"td_csc_{k}" for k in range(K)) + ("td_striosome", "td_snc", "td_reward_us", "td_fs")
    idx_map = {n: _idx_arr(bridge, n, xp) for n in region_names}
    # Alias to the names the standalone helpers expect (snc / striosome_value), AND keep csc_k as the standalone names.
    idx_map["snc"] = idx_map["td_snc"]
    idx_map["striosome_value"] = idx_map["td_striosome"]
    idx_map["reward_us"] = idx_map["td_reward_us"]
    for k in range(K):
        idx_map[f"csc_{k}"] = idx_map[f"td_csc_{k}"]

    cfg = bridge.core_config
    # PROVENANCE / anti-cheat (3): no host TD term reaches td_snc. The td_snc drive is
    # tonic(direct) + td_reward_us(synaptic relay; critic inhibits = r-V) + synaptic GABA_B(-V) +
    # synaptic conductance-derivative(+dV/dt) ONLY. current_reward_signal stays 0 (brain-based).
    assert cfg.current_reward_signal == 0.0, "host reward scalar must be 0 (brain-based)"
    assert cfg.reward_baseline == 0.0, "host reward baseline must be 0 (brain-based)"
    assert cfg.enable_td_value_derivative is True, "the B-2 conductance-derivative must be ON for the TD slice"
    assert cfg.reward_eligibility_tau_ms == 40.0, "the SHORT (tap-local) eligibility tau must be set"
    prov = {
        "snc_gets_direct_reward": False,            # reward enters synaptically (td_reward_us->td_snc)
        "reward_is_synaptic_relay": True,
        "host_reward_signal": float(cfg.current_reward_signal),
        "host_value_term": False,
        "snc_drive_terms": ("tonic(direct) + td_reward_us(synaptic relay; critic inhibits = r-V) + "
                            "synaptic GABA_B(-V) + synaptic conductance-derivative(+dV/dt) only"),
        "enable_td_value_derivative": bool(cfg.enable_td_value_derivative),
        "enable_gabab": bool(cfg.enable_gabab),
        "eligibility_tau_ms": float(cfg.reward_eligibility_tau_ms),
        "co_resident": "merged nav+conv bridge (parser + dlPFC + nav cascade co-resident)",
    }

    crit_tonic = {"td_striosome": RECIPE["csc_critic_tonic_pa"]}
    snc_tonic_pa = RECIPE["snc_tonic_pa"]
    csc_drive_pa = RECIPE["csc_drive_pa"]
    teacher_pa = RECIPE["csc_critic_teacher_pa"]
    reward_us_drive_pa = RECIPE["csc_reward_us_drive_pa"]
    bin_steps = RECIPE["bin_steps"]
    n_post_bins = RECIPE["n_post_bins"]
    us_dur_bins = RECIPE["us_dur_bins"]
    iti_bins = RECIPE["csc_iti_bins"]
    n_train = int(n_train_override) if n_train_override and n_train_override > 0 else RECIPE["n_train"]

    # Calibrate the dopamine threshold + baseline at the tonic (floor) condition (snc + critic tonic).
    tonic_drives = {"td_snc": snc_tonic_pa, **crit_tonic}
    tonic_frac = _calibrate_da_threshold(bridge, cfg, idx_map, tonic_drives, xp)
    tonic_conc = _calibrate_da_baseline(bridge, cfg, idx_map, tonic_drives, xp)
    if verbose:
        print(f"  [calib] K={K}, reward_bin={reward_bin}; td_snc tonic frac={tonic_frac:.4f} -> threshold; "
              f"tonic da conc={tonic_conc:.4f} -> baseline")

    n_win_bins = K + int(n_post_bins)
    win_steps = n_win_bins * bin_steps
    floor = {"td_snc": snc_tonic_pa, **crit_tonic}

    import random as _random
    rng = _random.Random(seed)

    peak_bins, snc_per_bin_hist = [], []
    v_substates_hist, w_substates_hist = [], []
    us_bin_rates, cue_bin_rates, floor_rates = [], [], []
    snc_tc_first = snc_tc_last = None

    def _build_events(us_bin):
        ev = [(k * bin_steps, (k + 1) * bin_steps, {f"td_csc_{k}": csc_drive_pa}) for k in range(K)]
        us0 = us_bin * bin_steps
        us1 = (us_bin + max(1, int(us_dur_bins))) * bin_steps
        # CRITIC TEACHER (innate-reflex-teaches-learned-circuit): the US fires the critic during the reward window,
        # so the reward-overlapping tap forms CAUSAL eligibility -> seeds the value gradient that back-propagates.
        ev.append((us0, us1, {"td_striosome": RECIPE["csc_critic_tonic_pa"] + teacher_pa}))
        # The reward enters at the EXCITATORY relay td_reward_us (critic inhibits it => r-V at td_snc).
        ev.append((us0, us1, {"td_reward_us": reward_us_drive_pa}))
        return ev

    for t in range(n_train):
        _, _, _f_snc, _ = _drive_timecourse(
            bridge, idx_map, floor, max(2, int(iti_bins)) * bin_steps, xp, bin_steps)
        floor_rates.append(float(_f_snc))
        us_bin = rng.randint(0, max(0, n_win_bins - 1)) if unpaired else reward_bin
        events = _build_events(us_bin)
        snc_bins, strio_bins, _, _ = _drive_timecourse(
            bridge, idx_map, floor, win_steps, xp, bin_steps, events=events)
        # Enforce the per-tap weight cap (the standalone's sparse-critic bound; co-resident the global stdp_w_max=400
        # would let the critic run away -> -V saturates -> td_snc dies -> no migration). No-op when w_clip<=0.
        _n_clamped = _clip_td_value_weights(bridge, K, w_clip)
        if verbose and t == 0 and w_clip > 0:
            print(f"  [clip] w_clip={w_clip}: clamped {_n_clamped} td_value edges after trial 0")
        peak_bin = int(np.argmax(snc_bins)) if snc_bins else 0
        peak_bins.append(peak_bin)
        snc_per_bin_hist.append(list(snc_bins))
        v_subs = [float(strio_bins[k]) if k < len(strio_bins) else 0.0 for k in range(K)]
        v_substates_hist.append(v_subs)
        w_substates_hist.append([_mean_pathway_weight(bridge, f"td_csc_{k}", "td_striosome") for k in range(K)])
        us_bin_rates.append(float(snc_bins[reward_bin]) if reward_bin < len(snc_bins) else 0.0)
        cue_bin_rates.append(float(snc_bins[0]) if snc_bins else 0.0)
        if t == 0:
            snc_tc_first = list(snc_bins)
        if t == n_train - 1:
            snc_tc_last = list(snc_bins)
        if verbose and (t < 3 or t % 10 == 0 or t == n_train - 1):
            wstr = " ".join(f"{w:.1f}" for w in w_substates_hist[-1])
            print(f"  [csc t={t:02d}] peak_bin={peak_bin}  cue-bin={cue_bin_rates[-1]:5.1f}  "
                  f"US-bin={us_bin_rates[-1]:5.1f}Hz  w[k]=[{wstr}]")

    if lesion_cue:
        n_cut = sum(_lesion_pathway(bridge, f"td_csc_{k}", "td_striosome") for k in range(K))
        if verbose:
            print(f"  [lesion-cue] zeroed {n_cut} td_csc_k->td_striosome edges (the value conduit)")

    # Frozen test block: a long settle then baseline + omission (mirrors the standalone). FREEZE HOMEOSTASIS for the
    # whole test block (not just the reward LR) -- otherwise the per-region td_snc threshold drifts during the settle
    # (the post-lesion floor climbed 3.5 -> 44 Hz, inflating the baseline so the cue-pathway lesion can't discriminate;
    # lesion_diag_s42.json + opsearch BOUNDARY). A frozen probe must freeze ALL plasticity.
    _settle_bins = max(8, int(iti_bins) * 2)
    omit_events = [(k * bin_steps, (k + 1) * bin_steps, {f"td_csc_{k}": csc_drive_pa}) for k in range(K)]
    with _frozen_homeostasis(bridge):
        _drive_timecourse(bridge, idx_map, floor, _settle_bins * bin_steps, xp, bin_steps, freeze_lr=0.0, cfg=cfg)
        base_bins, _, _, _ = _drive_timecourse(bridge, idx_map, floor, win_steps, xp, bin_steps, freeze_lr=0.0, cfg=cfg)
        _drive_timecourse(bridge, idx_map, floor, _settle_bins * bin_steps, xp, bin_steps, freeze_lr=0.0, cfg=cfg)
        # For the lesion test, also measure a predicted (chain + US) WITHOUT the teacher (the trained response must
        # stand on the learned csc->critic synapses, which the lesion cut) so the US reflex can be read.
        if lesion_cue:
            les_ev = [(k * bin_steps, (k + 1) * bin_steps, {f"td_csc_{k}": csc_drive_pa}) for k in range(K)]
            les_ev.append((reward_bin * bin_steps, (reward_bin + max(1, us_dur_bins)) * bin_steps,
                           {"td_reward_us": reward_us_drive_pa}))
            pred_bins, pred_strio, _, _ = _drive_timecourse(
                bridge, idx_map, floor, win_steps, xp, bin_steps, events=les_ev, freeze_lr=0.0, cfg=cfg)
        omit_bins, _, _, _ = _drive_timecourse(
            bridge, idx_map, floor, win_steps, xp, bin_steps, events=omit_events, freeze_lr=0.0, cfg=cfg)

    trial_idx = list(range(n_train))
    r_migration = _pearson_r(trial_idx, peak_bins)
    early = slice(0, max(1, n_train // 5)); late = slice(-max(1, n_train // 5), None)
    peak_early = float(np.mean(peak_bins[early])); peak_late = float(np.mean(peak_bins[late]))
    cue_early = float(np.mean(cue_bin_rates[early])); cue_late = float(np.mean(cue_bin_rates[late]))
    us_early = float(np.mean(us_bin_rates[early])); us_late = float(np.mean(us_bin_rates[late]))
    base_rate = float(np.mean(base_bins)) if base_bins else 0.0
    tonic_rate = float(np.mean(floor_rates[late])) if floor_rates else base_rate
    v_arr = np.asarray(v_substates_hist, dtype=np.float64)
    cue_v_early = float(v_arr[early, 0].mean()); cue_v_late = float(v_arr[late, 0].mean())
    w_arr = np.asarray(w_substates_hist, dtype=np.float64)
    w_sub_early = w_arr[early].mean(axis=0).tolist(); w_sub_late = w_arr[late].mean(axis=0).tolist()

    half = max(1, int(us_dur_bins))
    omit_at_reward = float(np.mean(omit_bins[reward_bin:reward_bin + half])) if omit_bins else 0.0
    omit_at_cue = float(omit_bins[0]) if omit_bins else 0.0
    base_at_reward = float(np.mean(base_bins[reward_bin:reward_bin + half])) if base_bins else 0.0
    dip_at_reward_depth = omit_at_cue - omit_at_reward
    gap_lo = max(1, reward_bin - 3); gap_hi = max(gap_lo + 1, reward_bin - 1)
    snc_last = snc_tc_last or []
    gap_late = (float(np.mean(snc_last[gap_lo:gap_hi])) if len(snc_last) > gap_hi else tonic_rate)

    # Gates (design §4.2, identical to the standalone).
    migration_r_pass = (r_migration < -0.7)
    migration_dir_pass = (peak_late < peak_early - 0.5)
    cue_ref = max(tonic_rate, gap_late)
    late_burst_at_cue = (cue_late > 1.15 * cue_ref) and (us_late <= 1.40 * cue_ref + 1e-6)
    early_burst_at_us = (us_early > 1.15 * tonic_rate)
    omission_dip_at_reward = (dip_at_reward_depth > 0) and (omit_at_reward < omit_at_cue + 1e-6)
    cue_value_grows = (cue_v_late > 1.20 * cue_v_early) if cue_v_early > 1e-6 else (cue_v_late > 1e-6)

    gates = {
        "migration_r_pass": bool(migration_r_pass), "migration_dir_pass": bool(migration_dir_pass),
        "early_burst_at_us": bool(early_burst_at_us), "late_burst_at_cue": bool(late_burst_at_cue),
        "omission_dip_at_reward": bool(omission_dip_at_reward), "cue_value_grows": bool(cue_value_grows),
    }
    out = {
        "seed": seed, "lesion_cue": lesion_cue, "unpaired": unpaired, "mode": "td_csc_merged",
        "n_train": n_train, "n_csc": K, "reward_bin": reward_bin,
        "r_migration": r_migration, "peak_bin_early": peak_early, "peak_bin_late": peak_late,
        "cue_rate_early": cue_early, "cue_rate_late": cue_late,
        "us_rate_early": us_early, "us_rate_late": us_late, "tonic_rate": tonic_rate,
        "base_rate_bare_hz": base_rate, "gap_late_hz": gap_late,
        "cue_v_early_hz": cue_v_early, "cue_v_late_hz": cue_v_late,
        "w_sub_early": w_sub_early, "w_sub_late": w_sub_late,
        "omit_at_reward_hz": omit_at_reward, "omit_at_cue_hz": omit_at_cue,
        "base_at_reward_hz": base_at_reward, "dip_at_reward_depth_hz": dip_at_reward_depth,
        "peak_bins": peak_bins, "snc_tc_first": snc_tc_first, "snc_tc_last": snc_tc_last,
        "omit_tc": list(omit_bins), "base_tc": list(base_bins),
        "gates": gates, "provenance": prov,
    }
    if lesion_cue:
        us_rate = float(np.mean(pred_bins[reward_bin:reward_bin + half])) if pred_bins else 0.0
        cue_rate = float(pred_bins[0]) if pred_bins else 0.0
        v_cue = float(pred_strio[0]) if pred_strio else 0.0
        # no_cue_burst REFERENCE = the derivative-active NO-CUE base window (base_bins), NOT the ITI tonic floor.
        # On the MERGED bridge the B-2 conductance-derivative converts the critic-tonic-driven GABA_B ripples into a
        # sustained td_snc baseline (~38 Hz with NO cue) that the bare ITI floor (~3.75 Hz) does not see. The correct
        # lesion contrast is cue-ON vs cue-OFF *in the same derivative-active window* (the standalone's tonic+base were
        # both ~60 Hz so this distinction was invisible there). With the proper reference, a cue burst that SURVIVES the
        # value-conduit lesion would lift cue >> base; a burst carried BY the (cut) value conduit collapses to ~base.
        base_window_hz = float(np.mean(base_bins)) if base_bins else tonic_rate
        cue_ref = max(base_window_hz, tonic_rate)
        out.update({
            "lesion_n_cut": n_cut, "lesion_v_cue_hz": v_cue, "lesion_cue_rate_hz": cue_rate,
            "lesion_us_rate_hz": us_rate, "lesion_tonic_hz": tonic_rate,
            "lesion_base_window_hz": base_window_hz, "lesion_cue_ref_hz": cue_ref,
            "cue_silenced": bool(v_cue <= 1e-2),
            # cue collapses to within the no-cue base (the value-driven burst is gone with the conduit cut).
            "no_cue_burst": bool(cue_rate <= 1.30 * cue_ref + 1e-6),
            # US reflex still bursts ABOVE the no-cue base (the reward relay survives, fires td_snc).
            "us_reflex_intact": bool(us_rate > 1.30 * cue_ref),
        })
    return out


def _print(r):
    print()
    print(f"  SNc time-of-peak   : trial-early bin {r['peak_bin_early']:.2f} -> trial-late bin "
          f"{r['peak_bin_late']:.2f}   (reward bin = {r['reward_bin']}, cue = bin 0)")
    print(f"  migration r        : {r['r_migration']:+.3f}   (MIGRATION = peak moves cue-ward => r < -0.7)")
    print(f"  cue-bin SNc rate   : {r['cue_rate_early']:.2f} -> {r['cue_rate_late']:.2f} Hz   "
          f"(tonic {r['tonic_rate']:.2f}Hz)")
    print(f"  US-bin SNc rate    : {r['us_rate_early']:.2f} -> {r['us_rate_late']:.2f} Hz   "
          f"(transferred if late US ~ tonic)")
    print(f"  V(strio) on cue    : {r['cue_v_early_hz']:.2f} -> {r['cue_v_late_hz']:.2f} Hz   "
          f"(cue value grows: {r['gates']['cue_value_grows']})")
    we = " ".join(f"{w:.1f}" for w in r["w_sub_early"]); wl = " ".join(f"{w:.1f}" for w in r["w_sub_late"])
    print(f"  w[k] early -> late : [{we}] -> [{wl}]")
    print(f"  omission @ reward  : {r['omit_at_reward_hz']:.2f}Hz vs @cue {r['omit_at_cue_hz']:.2f}Hz  "
          f"(dip depth {r['dip_at_reward_depth_hz']:+.2f})")
    g = r["gates"]
    print(f"  gates: migration_r {g['migration_r_pass']} | dir {g['migration_dir_pass']} | "
          f"early@US {g['early_burst_at_us']} | late@cue {g['late_burst_at_cue']} | "
          f"omit-dip@reward {g['omission_dip_at_reward']} | cue-value-grows {g['cue_value_grows']}")


def _verdict(g, r_migration):
    headline = g["migration_r_pass"] and g["migration_dir_pass"]
    support = sum([g["early_burst_at_us"], g["late_burst_at_cue"],
                   g["omission_dip_at_reward"], g["cue_value_grows"]])
    if headline and support >= 3:
        return "GO", support
    if g["migration_dir_pass"] or support >= 2:
        return "PARTIAL", support
    return "NEGATIVE", support


def run_moat_gate(seed, vocab=None, verbose=True):
    """CONSOLIDATION GATE (1): the no-confab MOAT is byte-intact with the TD slice + shared DA broadcast co-resident.
    Build a MergedNavConvAgent WITH co_resident_td_cueshift=True; assert a stored fact retrieves AND an unstored cue
    abstains (`is None`). The dopamine scope=all broadcast (over td_snc, threshold-calibrated) must not perturb the
    frozen conversational comprehension."""
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
    t0 = time.time()
    agent = MergedNavConvAgent(seed=int(seed), vocab=vocab, co_resident_td_cueshift=True)
    # Teach a fact, then probe the moat (this is the EXACT test_nav_conv_merged_agent moat assertion surface).
    agent.hear("dog go north", voice="active")
    stored = agent.what_does("dog", "go")           # must == 'north'
    abstain1 = agent.what_does("river", "look")     # must be None (no-confab)
    abstain2 = agent.what_does("cat", "go")         # must be None (agent matches none)
    desc_unknown = agent.describe("river")          # must be None (no fact about river)
    ok_stored = (stored == "north")
    ok_abstain = (abstain1 is None) and (abstain2 is None) and (desc_unknown is None)
    if verbose:
        print(f"  [moat] built agent(co_resident_td_cueshift=True) in {time.time()-t0:.0f}s")
        print(f"  [moat] what_does('dog','go') = {stored!r}  (== 'north': {ok_stored})")
        print(f"  [moat] what_does('river','look') = {abstain1!r} ; what_does('cat','go') = {abstain2!r} ; "
              f"describe('river') = {desc_unknown!r}  (all None: {ok_abstain})")
    return {"seed": seed, "moat_stored": stored, "moat_stored_ok": bool(ok_stored),
            "moat_abstain_ok": bool(ok_abstain), "moat_intact": bool(ok_stored and ok_abstain)}


def run_nav_byte_identity(seed, vocab=None, verbose=True):
    """CONSOLIDATION GATE (2): the TD slice is additive/default-off + appended LAST -> the nav/parser/dlPFC/rf index
    bases are BYTE-UNCHANGED (the existing nav/conv builds are untouched). Build TD-off and TD-on, compare the
    per-region base indices of every NON-td region. They must be IDENTICAL (append-last preserves the bases), which
    is the concrete byte-identity argument: the existing builds are bit-for-bit the TD-off case."""
    t0 = time.time()
    b_off, h_off = build_merged_nav_conv_bridge(seed=int(seed), vocab=vocab, co_resident_td_cueshift=False)
    b_on, h_on = build_merged_nav_conv_bridge(seed=int(seed), vocab=vocab, co_resident_td_cueshift=True)
    rm_off, rm_on = b_off.region_manager, b_on.region_manager
    off_names = [r.name for r in rm_off.regions()]
    bases_off = {n: int(rm_off.indices(n)[0]) for n in off_names}
    bases_on = {n: int(rm_on.indices(n)[0]) for n in off_names}   # every TD-off region must exist on TD-on
    mismatches = {n: (bases_off[n], bases_on[n]) for n in off_names if bases_off[n] != bases_on[n]}
    n_off = int(b_off.core_config.num_neurons); n_on = int(b_on.core_config.num_neurons)
    # The td slice must be appended AFTER the last non-td neuron (so it cannot shift any nav/conv index).
    max_nontd = max(int(rm_on.indices(n)[-1]) for n in off_names)
    td_base = int(rm_on.indices("td_csc_0")[0])
    bases_preserved = (len(mismatches) == 0)
    td_appended_last = (td_base > max_nontd)
    if verbose:
        print(f"  [nav-byte] built TD-off + TD-on in {time.time()-t0:.0f}s; "
              f"num_neurons off={n_off} on={n_on} (delta={n_on-n_off})")
        print(f"  [nav-byte] all {len(off_names)} non-td region bases preserved: {bases_preserved} "
              f"({len(mismatches)} mismatch)")
        print(f"  [nav-byte] td slice appended LAST (td_base {td_base} > max non-td idx {max_nontd}): {td_appended_last}")
        if mismatches:
            print(f"  [nav-byte] MISMATCHES: {dict(list(mismatches.items())[:6])}")
    return {"seed": seed, "bases_preserved": bool(bases_preserved), "td_appended_last": bool(td_appended_last),
            "n_neurons_off": n_off, "n_neurons_on": n_on, "n_mismatch": len(mismatches),
            "nav_byte_identical": bool(bases_preserved and td_appended_last)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None, help="comma seeds for the multi-seed migration battery")
    ap.add_argument("--td-csc-n", type=int, default=8)
    # CO-RESIDENT operating-point knobs (the merged-bridge runaway-critic / tonic-death fix; defaults = standalone GO).
    ap.add_argument("--td-stdp-w-max", type=float, default=0.0,
                    help="per-tap td_value weight cap re-clipped per trial (the standalone used 40; the merged bridge "
                         "pins the GLOBAL stdp_w_max=400 for conversation, so re-clip ONLY the td slice). 0=off")
    ap.add_argument("--td-gabab-cmax", type=float, default=0.0,
                    help="GIRK saturation cap on the td GABA_B conductance (bounds -V so a hot critic can't clamp "
                         "td_snc dead). 0=off")
    ap.add_argument("--td-gabab-prop", type=float, default=0.105, help="td GABA_B per-spike conductance increment")
    ap.add_argument("--td-to-fs-weight", type=float, default=16.0, help="td_csc_k->td_fs (FS-clamp drive) weight")
    ap.add_argument("--td-fs-to-strio-weight", type=float, default=10.0, help="td_fs->td_striosome (FS-clamp) weight")
    ap.add_argument("--td-csc-to-strio-weight", type=float, default=14.0, help="td_csc_k->td_striosome init weight")
    ap.add_argument("--td-derivative-gain", type=float, default=1.0,
                    help="B-2 conductance-derivative gain (the bootstrap +dV/dt). RAISE co-resident to lift the cue burst "
                         "so the peak migrates (the merged GIRK cap throttles the derivative; a higher gain compensates)")
    ap.add_argument("--td-slow-tau-ms", type=float, default=130.0, help="td conductance-derivative slow-EMA tau (ms)")
    ap.add_argument("--n-train", type=int, default=0, help="override n_train (0 = recipe default 50; use ~30 for faster op-point search)")
    ap.add_argument("--global-het-test", action="store_true",
                    help="DIAGNOSTIC: turn on global parameter heterogeneity (perturbs nav/conv determinism) to test "
                         "whether the standalone's het-ON operating point restores migration co-resident")
    ap.add_argument("--lesion", action="store_true", help="anti-cheat: cue-pathway lesion (migration must vanish, US reflex survives)")
    ap.add_argument("--unpaired", action="store_true", help="anti-cheat: unpaired timing (no migration)")
    ap.add_argument("--moat-only", action="store_true", help="run ONLY consolidation gate (1) the moat")
    ap.add_argument("--nav-byte-only", action="store_true", help="run ONLY consolidation gate (2) nav byte-identity")
    ap.add_argument("--no-gates", action="store_true", help="skip the consolidation gates (migration battery only)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    op = dict(td_csc_to_strio_weight=args.td_csc_to_strio_weight, td_to_fs_weight=args.td_to_fs_weight,
              td_fs_to_strio_weight=args.td_fs_to_strio_weight, td_gabab_prop=args.td_gabab_prop,
              td_gabab_conductance_max=args.td_gabab_cmax, td_stdp_w_max=args.td_stdp_w_max,
              td_derivative_gain=args.td_derivative_gain, td_slow_tau_ms=args.td_slow_tau_ms)

    if args.moat_only:
        rs = [run_moat_gate(s) for s in seeds]
        n_ok = sum(1 for r in rs if r["moat_intact"])
        print(f"\n=== MOAT GATE: {n_ok}/{len(rs)} byte-intact (co_resident_td_cueshift) ===")
        if args.out:
            json.dump({"mode": "moat", "results": rs}, open(args.out, "w"), indent=2)
        return
    if args.nav_byte_only:
        rs = [run_nav_byte_identity(s) for s in seeds]
        n_ok = sum(1 for r in rs if r["nav_byte_identical"])
        print(f"\n=== NAV BYTE-IDENTITY: {n_ok}/{len(rs)} (bases preserved + td appended last) ===")
        if args.out:
            json.dump({"mode": "nav_byte", "results": rs}, open(args.out, "w"), indent=2)
        return

    results = []
    for s in seeds:
        if args.lesion:
            print(f"[merged-CSC seed={s}] CUE-LESION anti-cheat — train then zero ALL td_csc_k->td_striosome:")
            r = run_td_csc_merged(s, lesion_cue=True, td_csc_n=args.td_csc_n, op=op, n_train_override=args.n_train)
            print(f"  V(strio) on cue after lesion = {r['lesion_v_cue_hz']:.2f}Hz (cue silenced: {r['cue_silenced']})")
            print(f"  cue-rate={r['lesion_cue_rate_hz']:.2f}Hz  US-rate={r['lesion_us_rate_hz']:.2f}Hz  "
                  f"tonic={r['lesion_tonic_hz']:.2f}Hz")
            ok = r["cue_silenced"] and r["no_cue_burst"] and r["us_reflex_intact"]
            print(f"  CUE-LESION anti-cheat (seed {s}): {'PASS' if ok else 'UNEXPECTED'}  "
                  f"[cue-silenced {r['cue_silenced']}, no-cue-burst {r['no_cue_burst']}, "
                  f"us-reflex-intact {r['us_reflex_intact']}]")
            r["_mode"] = "lesion"; results.append(r); print()
            continue
        if args.unpaired:
            print(f"[merged-CSC seed={s}] UNPAIRED anti-cheat — US at a random bin (no contingency):")
            r = run_td_csc_merged(s, unpaired=True, td_csc_n=args.td_csc_n, op=op, n_train_override=args.n_train)
            _print(r)
            g = r["gates"]
            no_mig = not (g["migration_r_pass"] and g["migration_dir_pass"])
            print(f"\n  UNPAIRED anti-cheat (seed {s}): {'PASS' if no_mig else 'UNEXPECTED'}  "
                  f"[no-migration {no_mig}, r={r['r_migration']:+.3f}]")
            r["_mode"] = "unpaired"; results.append(r); print()
            continue
        print(f"[merged-CSC seed={s}] A-CSC TD cue-shift on the MERGED 'one brain' — does the burst MIGRATE?")
        r = run_td_csc_merged(s, td_csc_n=args.td_csc_n, op=op, n_train_override=args.n_train,
                              global_het_test=args.global_het_test)
        _print(r)
        verdict, support = _verdict(r["gates"], r["r_migration"])
        print(f"\n  A-CSC migration (seed {s}): {verdict}  [HEADLINE migration_r {r['gates']['migration_r_pass']} "
              f"(r={r['r_migration']:+.3f}), dir {r['gates']['migration_dir_pass']}; support {support}/4]")
        r["_verdict"] = verdict; r["_mode"] = "csc"; results.append(r); print()

    # Multi-seed migration roll-up.
    if results and results[0].get("_mode") == "csc":
        n_go = sum(1 for r in results if r.get("_verdict") == "GO")
        n_partial = sum(1 for r in results if r.get("_verdict") == "PARTIAL")
        rs = ["{}={:+.3f}".format(r["seed"], r["r_migration"]) for r in results]
        print(f"=== MULTI-SEED MERGED A-CSC: {n_go} GO + {n_partial} PARTIAL / {len(results)} ===")
        print("=== migration r per seed: " + ", ".join(rs) + " ===")
        signs = [(_r["r_migration"] < 0) for _r in results]
        dips = [_r["gates"]["omission_dip_at_reward"] for _r in results]
        print(f"=== sign-consistent (all cue-ward): {all(signs)} | omission-dip-at-reward all seeds: {all(dips)} ===")

    # Consolidation gates (on the first seed; the moat/nav-byte are config-level so 1 is conclusive for the wiring).
    gate_out = {}
    if results and results[0].get("_mode") == "csc" and not args.no_gates:
        print("=== CONSOLIDATION GATES ===")
        gate_out["moat"] = run_moat_gate(seeds[0])
        gate_out["nav_byte"] = run_nav_byte_identity(seeds[0])
        print(f"  GATE (1) MOAT byte-intact      : {gate_out['moat']['moat_intact']}")
        print(f"  GATE (2) NAV byte-identical    : {gate_out['nav_byte']['nav_byte_identical']}")

    if args.out:
        json.dump({"mode": "merged_td_cueshift_consolidation", "results": results, "gates": gate_out},
                  open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
