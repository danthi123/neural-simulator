"""Self-normalised (FAIR) lateral inhibition for the source-monitor lane.

The overlap-sweep NO-GO (2026-08-06-source-monitor-episode-overlap-makes-criterion
-satisfiable-but-symmetric-GABA-A-competition-fails-weakest-source-NO-GO) isolated
the binding constraint as DIRECTION, not magnitude: v6's symmetric GABA-A circuit
is RICH-GET-RICHER because each source drives its OWN fast-spiking interneuron,
which inhibits only its RIVALS (self is excluded from its own inhibition). So the
inhibition a pool receives scales with the WINNER's output -- the strong source is
spared its own inhibition while the weak source is crushed by both strong rivals,
and min(M) < min(L) (competition degrades the weakest source).

The fix is the canonical divisive / feedback normalization of Carandini & Heeger:
a SHARED normalization interneuron pool driven by ALL source-memory pools that
feeds GABA-A inhibition back to ALL source-memory pools -- INCLUDING each pool's
own drive. The normalization signal is then the SAME for every pool (proportional
to the TOTAL source drive, self + rivals), so the strong pool inhibits itself as
hard as it inhibits the others and cannot run away. This is "fair"/"self-normalised"
inhibition: balanced across pools regardless of who is winning. Biologically it is
the PV+ basket-cell blanket/pooled inhibition that implements cortical gain control
(one common interneuron population summing local excitation and shunting it back),
NOT a host rate-normalization -- the pooling and feedback are spiking synaptic
pathways on the shared bridge, gated by the same SOURCE_COMPETITION_GATE.

This is DISTINCT from v8 (multiplicative synaptic scaling), which drove each
neuron's incoming recall weight toward a COMMON firing set-point and thereby
EQUALIZED per-source rates (compressing the discrimination margins -> NO-GO).
Divisive normalization divides every source rate by the SAME denominator: it
PRESERVES the rate ordering and ratios, only compressing magnitude, so a NEGATIVE
weakest margin becomes LESS negative (min(M) > min(L)) without equalizing rates.

The instrument is UNCHANGED: the v6 fixed recall protocol (per-recall
``reset_dynamical_state`` + settle-to-quiescence), the overlapping episode
patterns from ``make_overlapping_episode_patterns``, ``_source_margin``, and the
``weakest_source_strictly_improved`` predicate are all reused verbatim. At every
overlap level the zero-learned-weight control must still yield strict=False. Only
the lateral-inhibition WIRING changes (rival-only -> shared self-inclusive pool).
NumPy backend, deterministic.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger("SIM_BRIDGE").setLevel(logging.WARNING)

from research.runners._laneC_source_monitor_coresidency_gate import (
    ACC_GATE,
    ACC_REGION,
    APFC_GATE,
    APFC_SOURCE,
    EPISODE_REGION,
    SOURCE_AFFERENT,
    SOURCE_AFFERENT_GATE,
    SOURCE_LEARNING_GATE,
    SOURCE_MEMORY,
    SOURCE_RECALL_GATE,
    SOURCES,
    SourceMonitorCoresidencyGate,
    _dominant_source,
    _source_margin,
)
from research.runners._laneC_source_monitor_coresidency_gate_v5 import (
    SOURCE_COMPETITION_GATE,
    SourceMonitorConfigV2,
)
from research.runners._laneC_source_monitor_coresidency_gate_v6 import (
    SourceMonitorCoresidencyGateV6,
)
from research.runners._laneC_source_monitor_overlap_sweep import (
    make_overlapping_episode_patterns,
    _rival_burden,
)
from sim.backend import to_host
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
from sim.regions import BrainRegion, RegionPathway
from tools.lab import attributable_to

CALIBRATION_SEEDS = (650, 651)
DEVELOPMENT_SEEDS = (652, 653, 654)
DEFAULT_OVERLAPS = (0.0, 0.2, 0.4)

# One shared normalization interneuron population (the fair-inhibition pool).
SHARED_INTERNEURON = "source_competition_fs_shared"


@dataclass(frozen=True)
class SourceMonitorConfigFair(SourceMonitorConfigV2):
    """v2 operating point with ONE shared normalization interneuron pool.

    The two competition weights are inherited unchanged from v2:
    ``source_to_interneuron_weight`` (memory -> shared pool, excitatory) and
    ``interneuron_to_rival_weight`` (shared pool -> every memory pool, GABA-A).
    The pool is sized so its pooled excitatory drive (now from all three sources
    rather than one) and its fan-out onto all three memory pools give an
    inhibitory operating point comparable to the per-source circuit.
    """

    n_shared_interneuron: int = 12
    # Inhibitory reversal on the source-memory pools. The default GABA-A Cl-
    # reversal is -75 mV (hyperpolarizing, below the RS rest ~-60 mV), which lets
    # blanket inhibition rebound-fire an otherwise SILENT rival pool. Setting the
    # reversal to the resting potential makes the shared-pool inhibition SHUNTING
    # (divisive gain control, the true cortical normalization) rather than
    # hyperpolarizing, so a silent pool is not driven to spike. None keeps the
    # global -75 mV default.
    memory_inh_reversal_mV: float | None = None


class SourceMonitorFairGate(SourceMonitorCoresidencyGateV6):
    """v6 fixed recall protocol with self-inclusive shared divisive inhibition.

    Everything except the lateral-inhibition wiring is inherited from v6/v5/base:
    the per-recall ``reset_dynamical_state`` + settle-to-quiescence read, the
    Hebbian episode->source learning, the aPFC/ACC read-out, and the
    competition-off rest. Only ``_build_bridge`` changes -- the per-source
    rival-only interneurons are replaced by ONE shared pool that pools all source
    drive and inhibits every source pool (including its own driver).
    """

    def __init__(
        self,
        *,
        seed: int,
        config: SourceMonitorConfigFair | Mapping | None = None,
    ):
        c = (
            config
            if isinstance(config, SourceMonitorConfigFair)
            else SourceMonitorConfigFair(**(dict(config) if config else {}))
        )
        # Bypass the v2 __init__ (it looks up per-source interneuron regions that
        # the fair build does not create) and call the BASE __init__ directly; it
        # runs our _build_bridge override and sets up episode/source/aPFC/ACC
        # indices, zeros the learned weights, and captures the clean dynamical
        # snapshot. We then wire the shared-pool competition indices + gate that
        # v2 would otherwise have installed. The v5/v6 methods (_rest,
        # _settle_to_quiescence, recall) resolve via the MRO unchanged.
        SourceMonitorCoresidencyGate.__init__(self, seed=seed, config=c)
        rm = self.bridge.region_manager
        shared = np.asarray(rm.indices(SHARED_INTERNEURON), dtype=np.int64)
        # v5.recall reads competition_spikes[source] via _competition_indices;
        # map every source to the one shared population (count unused by the
        # sweep but must not KeyError).
        self._competition_indices = {source: shared for source in SOURCES}
        self._shared_interneuron_indices = shared
        self.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)

    def _memory_region(self, name: str, n_neurons: int) -> BrainRegion:
        """Source-memory pool, optionally with a shunting inhibitory reversal."""

        region = self._region(name, n_neurons)
        rev = self.config.memory_inh_reversal_mV
        if rev is not None:
            region = replace(region, syn_reversal_potential_i_override=float(rev))
        return region

    def _build_bridge(self) -> SimulationBridge:
        c = self.config
        regions = [self._region(EPISODE_REGION, c.n_episode)]
        for source in SOURCES:
            regions.extend(
                [
                    self._region(SOURCE_AFFERENT[source], c.n_source_afferent),
                    self._memory_region(SOURCE_MEMORY[source], c.n_source_memory),
                    self._region(APFC_SOURCE[source], c.n_apfc),
                ]
            )
        # One shared normalization interneuron pool (fast-spiking cortical).
        regions.append(
            self._fs_region(SHARED_INTERNEURON, c.n_shared_interneuron)
        )
        regions.append(self._region(ACC_REGION, c.n_acc))

        pathways: list[RegionPathway] = []
        for source in SOURCES:
            pathways.extend(
                [
                    RegionPathway(
                        from_region=EPISODE_REGION,
                        to_region=SOURCE_MEMORY[source],
                        density=1.0,
                        weight_mean=0.0,
                        weight_jitter=0.0,
                        plastic=True,
                        plasticity_gate=SOURCE_LEARNING_GATE,
                        transmission_gate=SOURCE_RECALL_GATE,
                    ),
                    RegionPathway(
                        from_region=SOURCE_AFFERENT[source],
                        to_region=SOURCE_MEMORY[source],
                        density=1.0,
                        weight_mean=float(c.source_afferent_weight),
                        weight_jitter=0.0,
                        plastic=False,
                        transmission_gate=SOURCE_AFFERENT_GATE,
                    ),
                    # FAIR: every source pool DRIVES the one shared normalization
                    # pool (the pool sums the TOTAL source drive, self + rivals).
                    RegionPathway(
                        from_region=SOURCE_MEMORY[source],
                        to_region=SHARED_INTERNEURON,
                        density=1.0,
                        weight_mean=float(c.source_to_interneuron_weight),
                        weight_jitter=0.0,
                        plastic=False,
                        transmission_gate=SOURCE_COMPETITION_GATE,
                    ),
                    # FAIR: the shared pool inhibits EVERY source pool, INCLUDING
                    # this one -- self-inclusive feedback/divisive normalization,
                    # so the winner shunts itself and cannot bury the weak pool.
                    RegionPathway(
                        from_region=SHARED_INTERNEURON,
                        to_region=SOURCE_MEMORY[source],
                        density=1.0,
                        weight_mean=float(c.interneuron_to_rival_weight),
                        weight_jitter=0.0,
                        plastic=False,
                        transmission_gate=SOURCE_COMPETITION_GATE,
                        receptor="gaba_a",
                    ),
                    RegionPathway(
                        from_region=SOURCE_MEMORY[source],
                        to_region=APFC_SOURCE[source],
                        density=1.0,
                        weight_mean=float(c.source_to_apfc_weight),
                        weight_jitter=0.0,
                        plastic=False,
                        transmission_gate=APFC_GATE,
                    ),
                    RegionPathway(
                        from_region=SOURCE_MEMORY[source],
                        to_region=ACC_REGION,
                        density=1.0,
                        weight_mean=float(c.source_to_acc_weight),
                        weight_jitter=0.0,
                        plastic=False,
                        transmission_gate=ACC_GATE,
                    ),
                ]
            )

        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.brain_regions = regions
        cfg.region_pathways = pathways
        cfg.seed = self.seed
        cfg.dt_ms = 0.5
        cfg.enable_parameter_heterogeneity = False
        cfg.enable_stdp = False
        cfg.enable_hebbian_learning = True
        cfg.hebbian_symmetric = True
        cfg.hebbian_learning_rate = float(c.hebbian_learning_rate)
        cfg.hebbian_max_weight = float(c.hebbian_max_weight)
        cfg.hebbian_min_weight = 0.0
        cfg.hebbian_weight_decay = 0.0
        cfg.enable_reward_modulation = False
        cfg.enable_structural_plasticity = False
        cfg.enable_short_term_plasticity = False
        cfg.enable_homeostasis = False
        cfg.ou_std_current_pA = 0.0
        cfg.fast_spike_reset = True
        bridge = SimulationBridge(
            core_config=cfg,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=GPUConfig(),
        )
        bridge._initialize_simulation_data(called_from_playback_init=False)
        return bridge


def _recall_three(
    gate: SourceMonitorFairGate, patterns: Sequence[np.ndarray]
) -> dict[str, dict]:
    return {
        "seen": gate.recall(patterns[0]),
        "heard": gate.recall(patterns[1]),
        "self_generated": gate.recall(patterns[2]),
    }


def evaluate_fair_overlap(
    seed: int,
    overlap_fraction: float,
    config: SourceMonitorConfigFair | None = None,
) -> dict:
    """Fair (self-normalised) inhibition at one overlap level, one seed."""

    c = config or SourceMonitorConfigFair()
    patterns, core = make_overlapping_episode_patterns(seed, c, overlap_fraction)
    t0 = time.time()

    # -- Instrument verification: zero-learned-weight control (no experience) ---
    ctrl = SourceMonitorFairGate(seed=seed + 30000, config=c)
    ctrl_on = _recall_three(ctrl, patterns)
    ctrl.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
    try:
        ctrl_off = _recall_three(ctrl, patterns)
    finally:
        ctrl.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)
    ctrl_M = {s: _source_margin(ctrl_on[s], s) for s in SOURCES}
    ctrl_L = {s: _source_margin(ctrl_off[s], s) for s in SOURCES}
    control_strict = bool(min(ctrl_M.values()) > min(ctrl_L.values()))

    # -- The real arm: learn, then competition ON (M) vs OFF (L) ----------------
    intact = SourceMonitorFairGate(seed=seed, config=c)
    initial = intact.weight_summary()
    intact.experience(patterns[0], visual_activity=True)
    intact.experience(patterns[1], auditory_activity=True)
    intact.experience(patterns[2], corollary_discharge=True)
    intact.experience(patterns[3], visual_activity=True, auditory_activity=True)
    learned = intact.weight_summary()

    on = _recall_three(intact, patterns)
    intact.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
    try:
        off = _recall_three(intact, patterns)
    finally:
        intact.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)

    margins_M = {s: _source_margin(on[s], s) for s in SOURCES}
    margins_L = {s: _source_margin(off[s], s) for s in SOURCES}
    rival_burden_on = {s: _rival_burden(on[s], s) for s in SOURCES}
    rival_burden_off = {s: _rival_burden(off[s], s) for s in SOURCES}
    dominant_correct = {s: bool(_dominant_source(on[s]) == s) for s in SOURCES}
    weakest_strict = bool(min(margins_M.values()) > min(margins_L.values()))
    floor_cleared = bool(min(margins_M.values()) >= 0.15)

    weak_src = min(SOURCES, key=lambda s: margins_L[s])
    weak_margin_attribution = attributable_to(
        f"weakest-source ({weak_src}) margin vs SOURCE_COMPETITION_GATE",
        treatment_value=margins_M[weak_src],
        control_value=margins_L[weak_src],
    )

    return {
        "seed": int(seed),
        "overlap_fraction": float(overlap_fraction),
        "core_size": int(core.size),
        "mechanism": "fair-shared-divisive-normalization",
        "weights_initial_l1": float(initial["l1"]),
        "weights_learned_l1": float(learned["l1"]),
        "control_zero_weight_strict": control_strict,
        "control_min_M": float(min(ctrl_M.values())),
        "control_min_L": float(min(ctrl_L.values())),
        "rival_burden_off": rival_burden_off,
        "rival_burden_on": rival_burden_on,
        "min_rival_burden_off": float(min(rival_burden_off.values())),
        "margins_M": margins_M,
        "margins_L": margins_L,
        "min_margin_M": float(min(margins_M.values())),
        "min_margin_L": float(min(margins_L.values())),
        "weakest_source_strictly_improved": weakest_strict,
        "min_margin_M_meets_0p15_floor": floor_cleared,
        "weakest_source_by_L": weak_src,
        "weak_margin_attributable_to_competition": weak_margin_attribution,
        "dominant_source_correct": dominant_correct,
        "all_dominant_correct": bool(all(dominant_correct.values())),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def evaluate_own_gain_ceiling(
    seed: int,
    overlap_fraction: float,
    config: SourceMonitorConfigV2 | None = None,
    gains: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0),
) -> dict:
    """CEILING probe for the BCM/own-gain alternative (host-oracle shortcut).

    Bounds whether raising the WEAK source's OWN firing gain can clear the floor,
    the alternative named by the overlap NO-GO. Uses the v6 circuit with lateral
    competition OFF (own-gain only), then for each source multiplies ITS OWN
    episode->source recall synapses (the exact synapses BCM own-gain would
    potentiate -- the CSR post-neuron column selects the target pool) by an oracle
    ``gains`` factor and records the best own-recall margin reachable. The lever
    is verified to ENGAGE: the target's own rate responds to the scaling (it is
    NOT the firing-threshold array, which is a spike-DETECTION set-point that does
    not change Izhikevich v-peak spiking). If the best margin still cannot clear
    the floor -- because the target's rate saturates at its refractory/adaptation
    ceiling while the rivals fire freely on the shared cells -- a biologically
    earned BCM own-gain rule cannot clear it either. Explicit host shortcut, used
    ONLY to bound the arc, not a mechanism claim.
    """

    c = config or SourceMonitorConfigV2()
    patterns, core = make_overlapping_episode_patterns(seed, c, overlap_fraction)
    t0 = time.time()
    g = SourceMonitorCoresidencyGateV6(seed=seed, config=c)
    g.experience(patterns[0], visual_activity=True)
    g.experience(patterns[1], auditory_activity=True)
    g.experience(patterns[2], corollary_discharge=True)
    g.experience(patterns[3], visual_activity=True, auditory_activity=True)
    g.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)

    learned = np.asarray(g._learned_synapse_indices(), dtype=np.int64)
    post = np.asarray(to_host(g.bridge.cp_connections.indices), dtype=np.int64)[learned]
    data0 = np.asarray(to_host(g.bridge.cp_connections.data), dtype=np.float64).copy()

    base = {s: _source_margin(g.recall(patterns[i]), s) for i, s in enumerate(SOURCES)}
    best = {}
    best_own_rate = {}
    base_own_rate = {}
    best_curve = {}
    for i, s in enumerate(SOURCES):
        target_syn = learned[np.isin(post, g._source_memory_indices[s])]
        curve = {}
        rate_curve = {}
        for gain in gains:
            d = data0.copy()
            d[target_syn] = data0[target_syn] * float(gain)
            g.bridge.cp_connections.data[:] = d
            rec = g.recall(patterns[i])
            curve[float(gain)] = _source_margin(rec, s)
            rate_curve[float(gain)] = float(rec["source_rates"][s])
        g.bridge.cp_connections.data[:] = data0
        best_gain = max(curve, key=curve.get)
        best[s] = float(curve[best_gain])
        best_own_rate[s] = float(rate_curve[best_gain])
        base_own_rate[s] = float(rate_curve[1.0])
        best_curve[s] = {str(k): round(v, 4) for k, v in curve.items()}

    # The lever ENGAGED iff scaling moved the target's own rate somewhere.
    lever_engaged = any(
        abs(best_own_rate[s] - base_own_rate[s]) > 1e-6 for s in SOURCES
    )
    own_gain_cannot_clear_floor = bool(min(best.values()) < 0.15)
    own_gain_improvement = float(min(best.values()) - min(base.values()))
    return {
        "seed": int(seed),
        "overlap_fraction": float(overlap_fraction),
        "core_size": int(core.size),
        "probe": "own-gain-episode-to-source-weight-scaling ceiling (host oracle)",
        "competition": "OFF (own-gain only)",
        "lever_engaged": bool(lever_engaged),
        "base_own_rate": base_own_rate,
        "best_own_rate": best_own_rate,
        "base_margins": {s: float(base[s]) for s in SOURCES},
        "best_own_gain_margin": {s: float(best[s]) for s in SOURCES},
        "best_own_gain_margin_curve": best_curve,
        "min_base_margin": float(min(base.values())),
        "min_best_own_gain_margin": float(min(best.values())),
        "own_gain_min_margin_improvement": own_gain_improvement,
        "own_gain_cannot_clear_floor": own_gain_cannot_clear_floor,
        "best_clears_0p15_floor": bool(min(best.values()) >= 0.15),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def _run_own_gain_ceiling(args) -> dict:
    rows = []
    for overlap in args.overlaps:
        for seed in args.seeds:
            row = evaluate_own_gain_ceiling(int(seed), float(overlap))
            rows.append(row)
            print(
                "[own-gain-ceiling] "
                f"seed={row['seed']} overlap={row['overlap_fraction']:.2f} "
                f"engaged={row['lever_engaged']} "
                f"base_min={row['min_base_margin']:.4f} "
                f"best_min={row['min_best_own_gain_margin']:.4f} "
                f"improve={row['own_gain_min_margin_improvement']:+.4f} "
                f"clears_floor={row['best_clears_0p15_floor']}",
                flush=True,
            )
    return {
        "runner": "research/runners/_laneC_source_monitor_fair_inhibition_sweep.py",
        "mode": "own-gain-ceiling",
        "seeds": list(args.seeds),
        "overlaps": list(args.overlaps),
        "instrument": "v6 fixed (per-recall reset_dynamical_state + settle), competition OFF",
        "probe": "intrinsic own-gain (firing-threshold) ceiling, host oracle",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fair (self-normalised) inhibition overlap sweep for the source-monitor lane."
    )
    parser.add_argument(
        "--mode",
        choices=("fair", "own-gain-ceiling"),
        default="fair",
        help="fair = shared self-inclusive divisive inhibition; own-gain-ceiling = intrinsic own-gain oracle bound",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS))
    parser.add_argument("--overlaps", type=float, nargs="+", default=list(DEFAULT_OVERLAPS))
    parser.add_argument("--n-shared", type=int, default=SourceMonitorConfigFair().n_shared_interneuron)
    parser.add_argument("--inh-weight", type=float, default=SourceMonitorConfigFair().interneuron_to_rival_weight)
    parser.add_argument(
        "--inh-reversal-mV",
        type=float,
        default=None,
        help="source-memory inhibitory reversal (shunting near rest ~-60 mV); default keeps -75 mV",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if args.mode == "own-gain-ceiling":
        out = _run_own_gain_ceiling(args)
        if args.json:
            out_path = Path(args.json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=2))
            print(f"[own-gain-ceiling] wrote {out_path}", flush=True)
        return 0

    cfg = SourceMonitorConfigFair(
        n_shared_interneuron=int(args.n_shared),
        interneuron_to_rival_weight=float(args.inh_weight),
        memory_inh_reversal_mV=(None if args.inh_reversal_mV is None else float(args.inh_reversal_mV)),
    )
    rows = []
    for overlap in args.overlaps:
        for seed in args.seeds:
            row = evaluate_fair_overlap(int(seed), float(overlap), config=cfg)
            rows.append(row)
            print(
                "[fair-sweep] "
                f"seed={row['seed']} overlap={row['overlap_fraction']:.2f} "
                f"core={row['core_size']} n_shared={cfg.n_shared_interneuron} "
                f"inhW={cfg.interneuron_to_rival_weight} "
                f"ctrl_strict={row['control_zero_weight_strict']} "
                f"min_rival_off={row['min_rival_burden_off']:.4f} "
                f"minM={row['min_margin_M']:.4f} minL={row['min_margin_L']:.4f} "
                f"strict={row['weakest_source_strictly_improved']} "
                f"floor={row['min_margin_M_meets_0p15_floor']} "
                f"dom_ok={row['all_dominant_correct']}",
                flush=True,
            )

    out = {
        "runner": "research/runners/_laneC_source_monitor_fair_inhibition_sweep.py",
        "seeds": list(args.seeds),
        "overlaps": list(args.overlaps),
        "n_shared_interneuron": int(cfg.n_shared_interneuron),
        "interneuron_to_rival_weight": float(cfg.interneuron_to_rival_weight),
        "instrument": "v6 fixed (per-recall reset_dynamical_state + settle)",
        "mechanism": "fair shared self-inclusive divisive normalization",
        "rows": rows,
    }
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[fair-sweep] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
