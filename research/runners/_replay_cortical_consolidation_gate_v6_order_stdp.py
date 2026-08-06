"""Bounded v6 calibration: v5+SFA + order-sensitive STDP consolidation.

Built ON v5+SFA (``_replay_cortical_consolidation_gate_v5_sfa.py``), which
CLOSED the shared-cue-cell interference wall on both calibration seeds (retest
false recall 0.066 / 0.080, far under the frozen 0.15 ceiling) while keeping
every CLS signature, but was NO-GO on the SOLE residual control
``intact_beats_shuffled_order`` (margins +0.0014 / +0.0056, below +0.01). The
v5+SFA finding localised the root cause precisely: the sleep consolidation rule
is ORDER-BLIND. It is rate-window Hebbian coactivity, and each replay event is
run through a full down-state reset (``_clear_fast_dynamics``), so permuting the
event ORDER preserves the per-event coactivity MULTISET and intact/shuffled
consolidate near-identical weights.

The named surpass (v5+SFA "Decision and next mechanism"): an ORDER-SENSITIVE
spike-timing-dependent / sequence-replay plasticity rule during sleep, so an
ORDERED replay potentiates a directional cue->target trace a SHUFFLED replay
does not.

Two coupled, biology-grounded changes make ORDER carry a consolidation signal
that STDP can read (nothing else in v5+SFA changes):

1. ORDER-SENSITIVE PLASTICITY (the substrate's own STDP). ``enable_stdp`` is
   turned on so the cortical cue->target association is trained by the substrate's
   intrinsic spike-timing rule (Bi & Poo 1998 asymmetric window in
   ``sim/kernels.fused_stdp_weight_update``), NOT a host-computed timing rule.
   STDP respects the per-pathway plasticity gate (``cp_plasticity_rate_gain``)
   and the per-synapse plastic mask, and during sleep only ``CORTICAL_GATE`` is
   open, so STDP acts on cue->target ONLY. It is kept INERT outside sleep by
   never advancing ``runtime_state.current_time_ms`` during wake encode/probe
   (the documented bridge.py:9382 guard: with a frozen clock every delta_t==0
   and every STDP update is exactly 0.0); the clock is advanced ONLY during
   sleep, and ``cp_last_spike_time`` is cleared at sleep onset so no wake spike
   pairs across the phase boundary. STDP weight bounds are set to the Hebbian
   scale (w_max=hebbian_max_weight) so STDP does not crush the trace.

2. ORDER-CARRYING DYNAMICS (contiguous replay, not a per-event down-state). In
   v5+SFA every event is separated by a full fast-dynamics reset, which erases
   any dependence of event i+1 on event i and makes order invisible to any local
   rule. Biologically, coherently sequenced replay is a CONTIGUOUS reactivation
   (adjacent events overlap -- the ``mean_adjacent_input_overlap`` the frozen
   ``temporal_control_changes_order`` check already verifies is HIGH for intact,
   LOW for shuffled); disorganised replay is fragmented. v6 replaces the
   per-event reset with a configurable inter-event handling
   (``sleep_interevent_reset``): the default keeps the membrane/conductance
   down-state boundary but preserves ``cp_last_spike_time`` so STDP timing
   carries ACROSS the boundary, and (``sleep_carry_residual``) optionally lets a
   short residual of the previous event's activity bleed in. With ordered
   replay, adjacent same-trajectory events sustain a coherent cue-before-target
   flow -> consistent LTP; with shuffled replay the adjacency is broken and the
   cue->target timing is inconsistent -> weaker/undirected trace. This makes the
   per-event coactivity MULTISET no longer sufficient to determine the weights:
   ADJACENCY (order) now matters, which is exactly what the v5+SFA root-cause
   analysis said was required.

Everything else is inherited UNCHANGED from v5+SFA: the learned CA1->cortex
reinstatement, the intrinsic SFA one-of-N eviction and its ``target_sfa_lesion``
power control, all v5 controls (no_sleep, shuffled_replay_order,
shuffled_target_index, ca3_ca1_lesion, cortical_plasticity_off,
target_inhibition_lesion, ca1_target_reinstatement_lesion), and every frozen
criterion (false_recall_bounded<=0.15, intact_beats_shuffled_order margin +0.01,
the load-bearing lesions, memory-selectivity). The reward: BOTH seeds 412 AND
413 pass ALL controls including ``intact_beats_shuffled_order`` WITHOUT breaking
false-recall<0.15 or any CLS signature.

Fresh seed partition (inherited): calibration 412/413, smoke 416, development
414/415/410 and held-out 417/418/419 mechanically rejected until calibration
lands a clean verdict.

CPU smoke:
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v6_order_stdp --smoke

Calibration:
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v6_order_stdp \
        --seeds 412 413 --out research/findings/raw/replay_v5_sfa_order/replay_v6_order_stdp_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners import _replay_cortical_consolidation_gate as v1  # noqa: E402
from research.runners import _replay_cortical_consolidation_gate_v2 as v2  # noqa: E402
from research.runners import _replay_cortical_consolidation_gate_v5 as v5  # noqa: E402
from research.runners import _replay_cortical_consolidation_gate_v5_sfa as v5s  # noqa: E402
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import UNDEFINED, Verdict  # noqa: E402


CALIBRATION_SEEDS = v5.CALIBRATION_SEEDS
DEVELOPMENT_SEEDS = v5.DEVELOPMENT_SEEDS
HELD_OUT_SEEDS = v5.HELD_OUT_SEEDS
SMOKE_SEED = v5.SMOKE_SEED
CONDITIONS = v5s.CONDITIONS

CA3_GATE = v5.CA3_GATE
INDEX_CUE_GATE = v5.INDEX_CUE_GATE
INDEX_TARGET_GATE = v5.INDEX_TARGET_GATE
CORTICAL_GATE = v5.CORTICAL_GATE
SCHAFFER_GATE = v5.SCHAFFER_GATE
TARGET_INHIBITION_GATE = v5.TARGET_INHIBITION_GATE
REINSTATEMENT_GATE = v5.REINSTATEMENT_GATE


@dataclass(frozen=True)
class GateConfig(v5s.GateConfig):
    """v5+SFA anatomy/timing/eviction + order-sensitive STDP consolidation."""

    # Order-sensitive spike-timing plasticity on the cortical cue->target trace.
    stdp_sleep: bool = True
    # STDP amplitudes (Bi & Poo asymmetric window). LTP-biased in EFFECT via the
    # timing (the causal cue-before-target ordering dominates delta_t), with a
    # strong LTD term to prune the anti-causal cross-memory pairings that a
    # cross-boundary timing carry would otherwise deposit (the seed-413 false-
    # recall driver). Amplitudes calibrated jointly with the SFA eviction below.
    stdp_a_plus: float = 0.008
    stdp_a_minus: float = 0.03
    stdp_tau_plus_ms: float = 20.0
    stdp_tau_minus_ms: float = 20.0
    # STDP soft-bound as a FRACTION of the Hebbian weight scale. The default
    # STDP w_max=2 would collapse a ~90-scale weight (the documented STDP
    # w_bounds gotcha); 1.0 matches the Hebbian scale. 0.5 caps how large STDP
    # can drive any single cue->target synapse, which is what keeps the harder
    # seed's false recall bounded while preserving the ordered-replay margin.
    stdp_w_max_scale: float = 0.5
    # During sleep, keep rate-window Hebbian ON (baseline v5+SFA transfer) and
    # ADD STDP for the directional/order term. STDP-only (False) isolates the
    # order signal but removes the established order-blind baseline.
    sleep_hebbian_on: bool = True
    # Inter-event handling during sleep. "full" == v5+SFA per-event down-state
    # reset (order-blind). "timing" keeps the membrane/conductance down-state but
    # preserves cp_last_spike_time so STDP timing carries across the boundary --
    # this is what makes cross-event ADJACENCY (order) visible to the local rule.
    sleep_interevent_reset: str = "timing"
    # Let a short residual of the previous event bleed into the next (do not zero
    # membrane/conductance). True makes ordered-vs-shuffled maximally distinct but
    # blows up false recall; the calibrated point uses False (clean down-state,
    # order carried purely by cp_last_spike_time timing across the boundary).
    sleep_carry_residual: bool = False
    # Stronger intrinsic SFA than v5+SFA's d=120: the order-STDP trace is broader,
    # so the one-of-N eviction must evict harder at retest to hold false recall
    # under 0.15 on the harder seed. (v5+SFA d=120 is inherited otherwise.)
    target_sfa_d_increment: float | None = 180.0


def smoke_config() -> GateConfig:
    base = v5s.smoke_config()
    return GateConfig(**asdict(base))


def validate_calibration_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    checked = tuple(int(seed) for seed in seeds)
    invalid = [
        seed for seed in checked if seed not in CALIBRATION_SEEDS and seed != SMOKE_SEED
    ]
    if invalid:
        raise ValueError(
            f"This bounded v6 runner accepts calibration seeds {CALIBRATION_SEEDS} "
            f"(or smoke seed {SMOKE_SEED}) only; refusing reserved seeds {invalid}."
        )
    if not checked:
        raise ValueError("At least one calibration seed is required.")
    return checked


def build_bridge(seed: int, config: GateConfig) -> tuple[object, dict]:
    """v5+SFA bridge, with the substrate STDP allocated so it can train sleep."""
    bridge, handles = v5.build_bridge(seed, config)
    if config.stdp_sleep:
        cfg = bridge.core_config
        # Allocate STDP state (cp_last_spike_time) by enabling the substrate STDP.
        # It stays INERT outside sleep because the clock is frozen in wake/probe
        # (bridge.py:9382 guard: every delta_t==0 => every update exactly 0.0).
        cfg.enable_stdp = True
        cfg.stdp_a_plus = float(config.stdp_a_plus)
        cfg.stdp_a_minus = float(config.stdp_a_minus)
        cfg.stdp_tau_plus_ms = float(config.stdp_tau_plus_ms)
        cfg.stdp_tau_minus_ms = float(config.stdp_tau_minus_ms)
        cfg.stdp_w_min = 0.0
        cfg.stdp_w_max = float(config.hebbian_max_weight) * float(config.stdp_w_max_scale)
        # Build the STDP timing array now (bridge already initialised).
        from sim.backend import get_backend

        xp, _ = get_backend()
        n = int(bridge.cp_membrane_potential_v.shape[0])
        if getattr(bridge, "cp_last_spike_time", None) is None:
            bridge.cp_last_spike_time = xp.full(n, -1000.0, dtype=xp.float32)
    return bridge, handles


def _apply_target_sfa(bridge, handles, config, *, enabled):
    return v5s._apply_target_sfa(bridge, handles, config, enabled=enabled)


def _sleep(bridge, handles: dict, condition: str, seed: int, config: GateConfig) -> dict:
    """v5+SFA sleep, made order-sensitive: contiguous replay + substrate STDP.

    Only two things differ from ``v5._sleep``: (1) the inter-event reset is
    softened so ``cp_last_spike_time`` (and optionally a short residual) carries
    across the down-state boundary, making cross-event ADJACENCY physically
    present; (2) ``runtime_state.current_time_ms`` is advanced each step so the
    substrate STDP is LIVE during sleep (and only during sleep).
    """
    from sim.backend import get_backend

    xp, _ = get_backend()
    v1._clear_dynamics(bridge)
    if config.stdp_sleep:
        # No wake spike may pair across the phase boundary.
        if getattr(bridge, "cp_last_spike_time", None) is not None:
            bridge.cp_last_spike_time[:] = xp.float32(-1000.0)
        # Advance the clock so STDP is live (start above the STDP window).
        bridge.runtime_state.current_time_ms = float(
            max(bridge.core_config.stdp_tau_plus_ms, bridge.core_config.stdp_tau_minus_ms) * 10.0
        )
    cortical_on = condition != "cortical_plasticity_off"
    v5._set_phase_gates(bridge, sleep=True, cortical=cortical_on)
    # Rate-window Hebbian baseline transfer (order-blind) stays as in v5+SFA;
    # STDP adds the order/directional term. Setting the rate to 0 would isolate
    # STDP-only (sleep_hebbian_on=False).
    if config.sleep_hebbian_on:
        bridge.core_config.hebbian_learning_rate = float(config.cortical_sleep_learning_rate)
    else:
        bridge.core_config.hebbian_learning_rate = 0.0
    if condition == "ca3_ca1_lesion":
        bridge.set_transmission_gate(SCHAFFER_GATE, 0.0)
    inhibition_gain = 0.0 if condition == "target_inhibition_lesion" else 1.0
    reinstatement_gain = (
        0.0
        if condition == "ca1_target_reinstatement_lesion"
        else float(config.reinstatement_sleep_gain)
    )
    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, inhibition_gain)
    bridge.set_transmission_gate(REINSTATEMENT_GATE, reinstatement_gain)
    shuffled_edges = (
        v1._shuffle_target_index(bridge, seed)
        if condition == "shuffled_target_index"
        else 0
    )

    regions = handles["regions"]
    events = v2._ordered_sleep_events(
        seed, config, regions["ca3"], shuffle=condition == "shuffled_replay_order",
    )
    ca3_dev = xp.asarray(regions["ca3"], dtype=xp.int64)
    ca1_dev = xp.asarray(regions["ca1"], dtype=xp.int64)
    cue_dev = xp.asarray(regions["cortical_cue"], dtype=xp.int64)
    target_dev = xp.asarray(regions["cortical_target"], dtype=xp.int64)
    fs_dev = xp.asarray(regions["cortical_target_fs"], dtype=xp.int64)
    ca3_a = handles["device_patterns"]["A"]["ca3"]
    ca3_b = handles["device_patterns"]["B"]["ca3"]
    target_a = handles["device_patterns"]["A"]["target"]
    target_b = handles["device_patterns"]["B"]["target"]
    event_winners: list[str] = []
    target_winners: list[str] = []
    spike_totals = {
        "ca3": 0,
        "ca1": 0,
        "cortical_cue": 0,
        "cortical_target": 0,
        "cortical_target_fs": 0,
    }

    cortical_w0 = None
    if config.stdp_sleep:
        cortical_w0 = v1._path_weights(bridge, CORTICAL_GATE).copy()

    def _interevent_reset():
        """Down-state boundary that keeps STDP timing (and optional residual)."""
        if config.sleep_interevent_reset == "full" or not config.stdp_sleep:
            v2._clear_fast_dynamics(bridge)
            return
        # "timing": reset membrane/conductance unless residual carry is on;
        # ALWAYS preserve cp_last_spike_time / cp_firing_states so STDP timing and
        # a short residual can carry the ADJACENCY across the boundary.
        if not config.sleep_carry_residual:
            if getattr(bridge, "cp_izh_c_reset", None) is not None:
                bridge.cp_membrane_potential_v[:] = bridge.cp_izh_c_reset
            else:
                bridge.cp_membrane_potential_v[:] = -65.0
            if getattr(bridge, "cp_recovery_variable_u", None) is not None:
                bridge.cp_recovery_variable_u[:] = 0.0
            for name in (
                "cp_conductance_g_e",
                "cp_conductance_g_i",
                "cp_conductance_g_nmda",
                "cp_conductance_g_nmda_rise",
                "cp_conductance_g_nmda_recurrent",
                "cp_conductance_g_nmda_recurrent_rise",
            ):
                arr = getattr(bridge, name, None)
                if arr is not None:
                    arr[:] = 0
        v1._zero_current(bridge)

    for event in events:
        _interevent_reset()
        event_a = event_b = 0
        target_ev_a = target_ev_b = 0
        if condition == "no_sleep":
            # no_sleep never drives; keep it fully quiescent (frozen check).
            v2._clear_fast_dynamics(bridge)
            v1._step(bridge, config.sleep_noise_steps + config.sleep_free_steps)
        else:
            background_dev = xp.asarray(event, dtype=xp.int64)
            for step in range(config.sleep_noise_steps + config.sleep_free_steps):
                v1._zero_current(bridge)
                if step < config.sleep_noise_steps:
                    bridge.cp_external_input_current[background_dev] = config.sleep_drive_pA
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                if config.stdp_sleep:
                    bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
                firing = bridge.cp_firing_states
                event_a += int(firing[ca3_a].sum())
                event_b += int(firing[ca3_b].sum())
                target_ev_a += int(firing[target_a].sum())
                target_ev_b += int(firing[target_b].sum())
                spike_totals["ca3"] += int(firing[ca3_dev].sum())
                spike_totals["ca1"] += int(firing[ca1_dev].sum())
                spike_totals["cortical_cue"] += int(firing[cue_dev].sum())
                spike_totals["cortical_target"] += int(firing[target_dev].sum())
                spike_totals["cortical_target_fs"] += int(firing[fs_dev].sum())
        if event_a == event_b == 0:
            event_winners.append("none")
        elif event_a > event_b:
            event_winners.append("A")
        elif event_b > event_a:
            event_winners.append("B")
        else:
            event_winners.append("tie")
        if target_ev_a == target_ev_b == 0:
            target_winners.append("none")
        elif target_ev_a > target_ev_b:
            target_winners.append("A")
        elif target_ev_b > target_ev_a:
            target_winners.append("B")
        else:
            target_winners.append("tie")

    v1._zero_current(bridge)
    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, 1.0)
    bridge.set_transmission_gate(REINSTATEMENT_GATE, 1.0)
    reinstatement_match = sum(
        1
        for ca3_win, tgt_win in zip(event_winners, target_winners)
        if ca3_win in ("A", "B") and tgt_win == ca3_win
    )
    reinstatement_mismatch = sum(
        1
        for ca3_win, tgt_win in zip(event_winners, target_winners)
        if ca3_win in ("A", "B") and tgt_win in ("A", "B") and tgt_win != ca3_win
    )
    stdp_cortical_delta = None
    if config.stdp_sleep and cortical_w0 is not None:
        stdp_cortical_delta = float(
            np.mean(v1._path_weights(bridge, CORTICAL_GATE) - cortical_w0)
        )
    return {
        "events": int(len(events)),
        "event_winners": event_winners,
        "target_winners": target_winners,
        "reactivated_events": int(sum(winner != "none" for winner in event_winners)),
        "replayed_A": int(sum(winner == "A" for winner in event_winners)),
        "replayed_B": int(sum(winner == "B" for winner in event_winners)),
        "target_reinstated_A": int(sum(winner == "A" for winner in target_winners)),
        "target_reinstated_B": int(sum(winner == "B" for winner in target_winners)),
        "reinstatement_match_events": int(reinstatement_match),
        "reinstatement_mismatch_events": int(reinstatement_mismatch),
        "spikes": spike_totals,
        "shuffled_edges": int(shuffled_edges),
        "event_content_multiset_digest": v2._event_digest(events, order_sensitive=False),
        "event_order_digest": v2._event_digest(events, order_sensitive=True),
        "mean_adjacent_input_overlap": v2._mean_adjacent_overlap(events),
        "target_inhibition_gain_during_sleep": inhibition_gain,
        "reinstatement_gain_during_sleep": reinstatement_gain,
        "stdp_cortical_delta_during_sleep": stdp_cortical_delta,
        "host_selected_episode_for_replay": False,
        "host_selected_target_drive": False,
    }


def run_condition(seed: int, condition: str, config: GateConfig | None = None) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}.")
    validate_calibration_seeds([seed])
    cfg = config or GateConfig()
    bridge, handles = build_bridge(seed, cfg)

    sfa_report = _apply_target_sfa(
        bridge, handles, cfg, enabled=(condition != "target_sfa_lesion")
    )

    bridge_ids = [id(bridge)]
    phase_trace: list[str] = []

    bridge.set_transmission_gate(TARGET_INHIBITION_GATE, 0.0)

    before = {
        "ca3": v1._path_weights(bridge, CA3_GATE),
        "index_cue": v1._path_weights(bridge, INDEX_CUE_GATE),
        "index_target": v1._path_weights(bridge, INDEX_TARGET_GATE),
        "cortical": v1._path_weights(bridge, CORTICAL_GATE),
    }
    encode_a = v5._encode_memory(bridge, handles, "A", cfg.encode_a_events, cfg)
    phase_trace.append("encode_A")
    bridge_ids.append(id(bridge))
    after_a = {
        "ca3": v1._path_weights(bridge, CA3_GATE),
        "cortical": v1._path_weights(bridge, CORTICAL_GATE),
    }
    encode_b = v5._encode_memory(bridge, handles, "B", cfg.encode_b_events, cfg)
    phase_trace.append("encode_B")
    bridge_ids.append(id(bridge))
    after_b = {
        "ca3": v1._path_weights(bridge, CA3_GATE),
        "index_cue": v1._path_weights(bridge, INDEX_CUE_GATE),
        "index_target": v1._path_weights(bridge, INDEX_TARGET_GATE),
        "cortical": v1._path_weights(bridge, CORTICAL_GATE),
    }
    sleep = _sleep(bridge, handles, condition, seed, cfg)
    phase_trace.append("sleep")
    bridge_ids.append(id(bridge))
    after_sleep = {
        "ca3": v1._path_weights(bridge, CA3_GATE),
        "index_cue": v1._path_weights(bridge, INDEX_CUE_GATE),
        "index_target": v1._path_weights(bridge, INDEX_TARGET_GATE),
        "cortical": v1._path_weights(bridge, CORTICAL_GATE),
    }

    # Retest hippocampus-disabled: CA1 silent, reinstatement off, STDP clock
    # left where sleep ended but plasticity gate CORTICAL is 0 at retest, so no
    # further consolidation occurs during probing.
    v5._set_phase_gates(bridge)
    recall = {memory: v1._probe_memory(bridge, handles, memory, cfg) for memory in ("A", "B")}
    phase_trace.append("retest")
    bridge_ids.append(id(bridge))

    def mean_delta(later: np.ndarray, earlier: np.ndarray) -> float:
        return float(np.mean(later - earlier))

    return {
        "seed": int(seed),
        "condition": condition,
        "config": asdict(cfg),
        "phase_trace": phase_trace,
        "single_bridge_persisted": len(set(bridge_ids)) == 1,
        "wiring_counts": handles["wiring_counts"],
        "reinstatement_memory_specific": bool(handles["reinstatement_memory_specific"]),
        "inhibitory_neuron_count": int(len(handles["inhibitory_indices"])),
        "target_sfa": sfa_report,
        "encode_A": encode_a,
        "encode_B": encode_b,
        "sleep": sleep,
        "recall": recall,
        "weight_deltas": {
            "ca3_during_encode_A": mean_delta(after_a["ca3"], before["ca3"]),
            "ca3_during_encode_B": mean_delta(after_b["ca3"], after_a["ca3"]),
            "ca3_during_sleep": mean_delta(after_sleep["ca3"], after_b["ca3"]),
            "index_cue_during_wake": mean_delta(after_b["index_cue"], before["index_cue"]),
            "reinstatement_during_wake": mean_delta(
                after_b["index_target"], before["index_target"]
            ),
            "index_cue_during_sleep": mean_delta(after_sleep["index_cue"], after_b["index_cue"]),
            "reinstatement_during_sleep": mean_delta(
                after_sleep["index_target"], after_b["index_target"]
            ),
            "cortical_during_wake": mean_delta(after_b["cortical"], before["cortical"]),
            "cortical_during_sleep": mean_delta(after_sleep["cortical"], after_b["cortical"]),
        },
    }


def _calibration_verdict(conditions: dict[str, dict]) -> dict:
    """v5+SFA's frozen verdict (unchanged criteria), plus an order-STDP diagnostic."""
    verdict = v5s._calibration_verdict(conditions)
    intact = conditions["intact"]
    shuffled = conditions["shuffled_replay_order"]
    verdict["intact_stdp_cortical_delta"] = intact["sleep"].get("stdp_cortical_delta_during_sleep")
    verdict["shuffled_stdp_cortical_delta"] = shuffled["sleep"].get(
        "stdp_cortical_delta_during_sleep"
    )
    verdict["intact_vs_shuffled_recovery_margin"] = (
        verdict["intact_mean_recovery"] - verdict["control_mean_recovery"]["shuffled_replay_order"]
    )
    # Whose is the ordered-vs-shuffled recovery difference? treatment = intact
    # (ordered replay), control = shuffled_replay_order (same event content,
    # permuted order). A positive fraction attributes the recovery gap to the
    # ORDER of replay -- which the stdp_sleep=False power control confirms is
    # created by the order-sensitive STDP, not the SFA or the contiguous reset.
    verdict["order_recovery_attribution"] = attributable_to(
        "order-sensitive STDP consolidation on retest recovery",
        verdict["intact_mean_recovery"],
        verdict["control_mean_recovery"]["shuffled_replay_order"],
    )
    return verdict


def run_seed(seed: int, config: GateConfig | None = None) -> dict:
    validate_calibration_seeds([seed])
    cfg = config or GateConfig()
    conditions = {condition: run_condition(seed, condition, cfg) for condition in CONDITIONS}
    verdict = _calibration_verdict(conditions)
    return {
        "seed": int(seed),
        "conditions": conditions,
        "calibration": verdict,
        "calibration_status": verdict["calibration_status"],
        "verdict": verdict["verdict"],
    }


def run_calibration(seeds: Iterable[int], config: GateConfig | None = None) -> dict:
    checked = validate_calibration_seeds(seeds)
    started = time.time()
    rows = [run_seed(seed, config) for seed in checked]
    statuses = [row["calibration_status"] for row in rows]
    if any(status == "UNDEFINED" for status in statuses):
        aggregate_status = "UNDEFINED"
    elif all(status == "CALIBRATION_PROMISING" for status in statuses):
        aggregate_status = "CALIBRATION_PROMISING"
    else:
        aggregate_status = "CALIBRATION_NEEDS_REVISION"
    return {
        "gate": "replay_cortical_consolidation_v6_order_stdp",
        "phase": "calibration",
        "mechanism": "v5 learned CA1->cortex reinstatement + intrinsic SFA one-of-N eviction + order-sensitive STDP consolidation",
        "calibration_status": aggregate_status,
        "seeds": list(checked),
        "reserved_seeds_inspected": False,
        "rows": rows,
        "remaining_scaffolds": [
            "host-defined wake episode populations and partial probe cues",
            "opponent inhibitory channel membership fixed from calibration assemblies",
            "host-scheduled sleep down-state boundaries and episode-agnostic CA3 background current",
            "host spike/weight measurement against known calibration assemblies",
            "fixed assembly anatomy",
            "SFA parameters (d_increment/a) set on the target slice at build, not developmentally tuned",
            "STDP amplitudes/bounds set at build, not developmentally tuned",
        ],
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.smoke:
        seeds = (SMOKE_SEED,)
        config = smoke_config()
    else:
        seeds = args.seeds
        config = GateConfig()
    payload = run_calibration(seeds, config)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
