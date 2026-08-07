"""RECALL-SIDE de-risk: CA3-style attractor competition among source assemblies.

Two encoding-side levers (hetero-depression 8aca3c62, conjunctive-tag 1a5d2db6)
and four recall-activity levers all recorded NO-GO: a SOURCE-BLIND recall drive
over a fully-shared core reactivates every committed subset, so the rival burden
persists.  Every one of those six levers was LINEAR / per-cell / FEEDFORWARD.
The residual the scoping doc
(``research/findings/raw/_source_monitor_attractor_competition_scoping.md``)
isolates is a NONLINEAR, ATTRACTOR-LEVEL, recall-time competition -- a locus none
of the six touched.

This runner adds (fixed-weight, SYMMETRIC across the three sources, scaled by ONE
knob ``g_comp``):
  (i)  within-population recurrent EXCITATION -- each ``source_memory_{s}`` becomes
       an autoassociative attractor via a slow-NMDA (Wang 2001/2002; Rolls CA3
       recurrent collaterals) self-recurrent pathway (``exc_receptor='nmda_slow'``);
  (ii) between-population lateral INHIBITION -- the v2 fast-spiking interneuron
       circuit (``source_memory -> fs -> rival source_memory``, GABA-A), whose
       ``interneuron_to_rival`` weight is now TIED to ``g_comp`` at a fixed ratio.
Both pathways share ``SOURCE_COMPETITION_GATE``, so competition ON = gate 1.0 (M)
and competition OFF = gate 0.0 (L, the lesion / feedforward arm).  Assembly
IDENTITIES are the pre-defined region memberships; discrimination STILL comes only
from the learned ``episode->source`` fan-out.  The mechanism carries NO
source-specific term -- it is structurally symmetric and cannot encode which
source is cued (the honesty guard).

Anti-cheats reported EVERY seed x g_comp:
  (a) ``g_comp == 0`` builds NO competition pathway at all, so its M arm is
      byte-identical to its own (and every other g_comp's) feedforward L arm --
      the null control and lesion arm coincide.
  (b) HONESTY: the recall drives EPISODE current only; source-afferent external
      current == 0 AND source-afferent firing == 0 during the read (measured);
      the competition module is parameter-symmetric across sources.  Non-vacuity:
      a forced WRONG-source afferent at recall MOVES the dominant winner, proving
      the guard excludes a real path.
  (c) no source's OWN-recall rate collapses (each cued source's own rate > 0).
  (d) zero-learned-weight instrument control stays strict=False (no
      stepping-history artifact) -- reused verbatim from the overlap sweep.

  ==> THE DECISIVE ANTI-CHEAT: ``all_dominant_correct`` must stay True on EVERY
      source INCLUDING the weakest (``self_generated``).  A high ``g_comp`` that
      wins margin by ALWAYS silencing two pools regardless of correctness is the
      rich-get-richer cheat, and is a NO-GO -- reported explicitly per row.

GO (smoke, needs full validation): ``min_margin_M >= 0.15`` AND
``min_margin_M > min_margin_L`` AND ``all_dominant_correct`` True, on both
calibration seeds 650/651.  numpy, deterministic, minutes/seed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
    _dominant_source,
    _source_margin,
)
from research.runners._laneC_source_monitor_coresidency_gate_v5 import (
    SOURCE_COMPETITION_GATE,
    SOURCE_INTERNEURON,
    SourceMonitorConfigV2,
)
from research.runners._laneC_source_monitor_coresidency_gate_v6 import (
    SourceMonitorCoresidencyGateV6,
)
from research.runners._laneC_source_monitor_overlap_sweep import (
    make_overlapping_episode_patterns,
)
from sim.backend import get_backend, to_host
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
from sim.regions import RegionPathway

CALIBRATION_SEEDS = (650, 651)
DEVELOPMENT_SEEDS = (652, 653, 654)
HELD_OUT_SEEDS = (655, 656, 657)
DEFAULT_G_COMP = (0.0, 0.5, 1.0, 2.0)
DEFAULT_OVERLAP = 0.2

# The recall-time margin floor the whole lane is measured against (frozen, shared
# with every earlier source-monitor rung).
MIN_SOURCE_MARGIN = 0.15


@dataclass(frozen=True)
class SourceMonitorAttractorConfig(SourceMonitorConfigV2):
    """v2 operating point plus a single-knob CA3-style attractor competition.

    ``g_comp`` scales BOTH the within-population recurrent excitation and the
    between-population lateral inhibition at a fixed ratio, so one variable turns
    the whole nonlinear competition up or down.  ``g_comp == 0`` builds NO
    competition pathway -> pure feedforward (byte-identical to the lesion arm).
    """

    g_comp: float = 1.0
    # Recurrent-E weight (slow-NMDA) at g_comp == 1.0. The effective weight is
    # g_comp * recurrent_e_weight_base.
    recurrent_e_weight_base: float = 200.0
    recurrent_e_density: float = 1.0
    # Lateral-I rival weight at g_comp == 1.0 (the v2 default 3.0). Tied to g_comp
    # at the fixed ratio recurrent_e_weight_base : lateral_i_weight_base.
    lateral_i_weight_base: float = 3.0


class SourceMonitorAttractorCompetitionGate(SourceMonitorCoresidencyGateV6):
    """v6 silent-by-construction recall + a single-knob CA3 attractor competition.

    At ``g_comp == 0`` NO competition pathway is built, so SOURCE_COMPETITION_GATE
    is unregistered; every access to it is guarded (``_set_comp_gate`` plus the
    ``_rest`` / ``_settle_to_quiescence`` overrides), and the g_comp==0 build is a
    pure feedforward null identical to the lesion arm.
    """

    def __init__(self, *, seed, config=None):
        # Bypass v2.__init__'s UNCONDITIONAL competition-gate set (the gate does
        # not exist at g_comp==0); replicate what v2.__init__ sets up, guarded.
        from research.runners._laneC_source_monitor_coresidency_gate import (
            SourceMonitorCoresidencyGate,
        )

        c = (
            config
            if isinstance(config, SourceMonitorAttractorConfig)
            else SourceMonitorAttractorConfig(**(dict(config) if config else {}))
        )
        SourceMonitorCoresidencyGate.__init__(self, seed=seed, config=c)
        rm = self.bridge.region_manager
        self._competition_indices = {
            s: np.asarray(rm.indices(SOURCE_INTERNEURON[s]), dtype=np.int64)
            for s in SOURCES
        }
        self._set_comp_gate(1.0)

    def _set_comp_gate(self, value: float) -> bool:
        """Set the competition gate if it is registered; no-op otherwise."""

        try:
            self.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, float(value))
            return True
        except KeyError:
            return False

    def _rest(self) -> None:
        """Drain trial state; gate competition off during rest IFF it exists."""

        from research.runners._laneC_source_monitor_coresidency_gate import (
            SourceMonitorCoresidencyGate,
        )

        prior = self.bridge._transmission_gate_values.get(SOURCE_COMPETITION_GATE)
        if prior is not None:
            self.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
        try:
            SourceMonitorCoresidencyGate._rest(self)
        finally:
            if prior is not None:
                self.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, prior)

    def _settle_to_quiescence(self, max_blocks: int = None) -> dict:
        """v6 settle, but a no-op competition-gate toggle when the gate is absent."""

        if SOURCE_COMPETITION_GATE in self.bridge._transmission_gate_values:
            if max_blocks is None:
                return super()._settle_to_quiescence()
            return super()._settle_to_quiescence(max_blocks)
        # g_comp == 0: no competition pathway -> nothing to gate off during settle.
        from sim.backend import to_host as _to_host

        readout = np.concatenate(
            [self._source_memory_indices[s] for s in SOURCES]
            + [self._apfc_indices[s] for s in SOURCES]
            + [self._source_afferent_indices[s] for s in SOURCES]
            + [self._acc_indices]
        )
        from research.runners._laneC_source_monitor_coresidency_gate_v6 import (
            MAX_SETTLE_BLOCKS,
        )

        blocks_cap = MAX_SETTLE_BLOCKS if max_blocks is None else int(max_blocks)
        self.bridge.cp_external_input_current[:] = 0.0
        steps = 0
        blocks = 0
        reached = False
        block = int(self.config.rest_steps)
        for _ in range(int(blocks_cap)):
            blocks += 1
            quiet = True
            for _ in range(block):
                self.bridge._run_one_simulation_step()
                steps += 1
                firing = np.asarray(
                    _to_host(self.bridge.cp_firing_states), dtype=np.float64
                )
                if float(firing[readout].sum()) > 0.0:
                    quiet = False
            if quiet:
                reached = True
                break
        return {"settle_steps": steps, "settle_blocks": blocks, "reached_quiescence": reached}

    def _build_bridge(self) -> SimulationBridge:
        c = self.config
        g_comp = float(getattr(c, "g_comp", 1.0))
        rec_e_weight = g_comp * float(getattr(c, "recurrent_e_weight_base", 0.0))
        lat_i_weight = g_comp * float(getattr(c, "lateral_i_weight_base", 0.0))
        # This operating point sits on many marginal spikes, so even the 0.01 weight
        # FLOOR the pathway builder imposes is NOT sub-threshold-negligible. So at
        # g_comp == 0 NO competition pathway is built at all -> a TRUE feedforward
        # null (byte-identical to the lesion arm, verified per row). The gate then
        # goes unregistered, so every gate access is guarded (``_set_comp_gate`` and
        # the ``_rest`` / ``_settle_to_quiescence`` overrides below).
        add_competition = g_comp > 0.0

        regions = [self._region(EPISODE_REGION, c.n_episode)]
        for source in SOURCES:
            regions.extend(
                [
                    self._region(SOURCE_AFFERENT[source], c.n_source_afferent),
                    self._region(SOURCE_MEMORY[source], c.n_source_memory),
                    self._fs_region(SOURCE_INTERNEURON[source], c.n_source_interneuron),
                    self._region(APFC_SOURCE[source], c.n_apfc),
                ]
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
            if add_competition:
                # (i) within-population recurrent EXCITATION (autoassociative
                # attractor; slow-NMDA carried, Mg-block self-limiting so the
                # recurrent latches a graded state without an AMPA synchronous
                # runaway). SYMMETRIC across sources, no source-specific term.
                pathways.append(
                    RegionPathway(
                        from_region=SOURCE_MEMORY[source],
                        to_region=SOURCE_MEMORY[source],
                        density=float(getattr(c, "recurrent_e_density", 1.0)),
                        weight_mean=rec_e_weight,
                        weight_jitter=0.0,
                        plastic=False,
                        transmission_gate=SOURCE_COMPETITION_GATE,
                        exc_receptor="nmda_slow",
                    )
                )
                # source_memory -> its own fast-spiking interneuron
                pathways.append(
                    RegionPathway(
                        from_region=SOURCE_MEMORY[source],
                        to_region=SOURCE_INTERNEURON[source],
                        density=1.0,
                        weight_mean=float(c.source_to_interneuron_weight),
                        weight_jitter=0.0,
                        plastic=False,
                        transmission_gate=SOURCE_COMPETITION_GATE,
                    )
                )
                # (ii) between-population lateral INHIBITION (fs -> rivals, GABA-A),
                # weight tied to g_comp at the fixed ratio.
                for rival in SOURCES:
                    if rival == source:
                        continue
                    pathways.append(
                        RegionPathway(
                            from_region=SOURCE_INTERNEURON[source],
                            to_region=SOURCE_MEMORY[rival],
                            density=1.0,
                            weight_mean=lat_i_weight,
                            weight_jitter=0.0,
                            plastic=False,
                            transmission_gate=SOURCE_COMPETITION_GATE,
                            receptor="gaba_a",
                        )
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
        # Slow-NMDA recurrent conductance (Wang 2001/2002). Guarded: with no
        # nmda_slow pathway (g_comp == 0) the arrays stay None and the step block
        # is unreached (byte-identical to the feedforward baseline).
        cfg.enable_nmda_recurrent = bool(add_competition)
        bridge = SimulationBridge(
            core_config=cfg,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=GPUConfig(),
        )
        bridge._initialize_simulation_data(called_from_playback_init=False)
        return bridge

    def recall_instrumented(
        self,
        episode_pattern: Sequence[int],
        *,
        force_afferent: str | None = None,
    ) -> dict:
        """Settle to quiescence, then read the learned pathway with full honesty
        instrumentation: accumulates source-memory, aPFC, ACC, competition AND
        source-afferent firing, and records the max source-afferent external
        current during the read (must be 0 unless ``force_afferent`` injects one).
        """

        self.reset_dynamical_state()
        settle = self._settle_to_quiescence()

        xp, _ = get_backend()
        episode_global = self._episode_global_indices(episode_pattern)
        self.bridge.set_plasticity_gate(SOURCE_LEARNING_GATE, 0.0)
        self.bridge.set_transmission_gate(SOURCE_RECALL_GATE, 1.0)
        self.bridge.set_transmission_gate(ACC_GATE, 1.0)

        source_spikes = {s: 0.0 for s in SOURCES}
        apfc_spikes = {s: 0.0 for s in SOURCES}
        competition_spikes = {s: 0.0 for s in SOURCES}
        afferent_spikes = {s: 0.0 for s in SOURCES}
        acc_spikes = 0.0
        max_afferent_current = 0.0
        all_afferent = np.concatenate(
            [self._source_afferent_indices[s] for s in SOURCES]
        )
        try:
            self.bridge.cp_external_input_current[:] = 0.0
            self.bridge.cp_external_input_current[
                xp.asarray(episode_global, dtype=xp.int64)
            ] = float(self.config.drive_pA)
            if force_afferent is not None:
                self.bridge.cp_external_input_current[
                    xp.asarray(self._source_afferent_indices[force_afferent], dtype=xp.int64)
                ] = float(self.config.drive_pA)
            for _ in range(int(self.config.read_steps)):
                self.bridge._run_one_simulation_step()
                firing = np.asarray(to_host(self.bridge.cp_firing_states), dtype=np.float64)
                ext = np.asarray(
                    to_host(self.bridge.cp_external_input_current), dtype=np.float64
                )
                max_afferent_current = max(
                    max_afferent_current, float(np.abs(ext[all_afferent]).max())
                )
                for s in SOURCES:
                    source_spikes[s] += float(firing[self._source_memory_indices[s]].sum())
                    apfc_spikes[s] += float(firing[self._apfc_indices[s]].sum())
                    competition_spikes[s] += float(
                        firing[self._competition_indices[s]].sum()
                    )
                    afferent_spikes[s] += float(
                        firing[self._source_afferent_indices[s]].sum()
                    )
                acc_spikes += float(firing[self._acc_indices].sum())
        finally:
            self._rest()
            self.bridge.set_transmission_gate(SOURCE_RECALL_GATE, 1.0)
            self.bridge.set_transmission_gate(ACC_GATE, 1.0)
            self.bridge.cp_external_input_current[:] = 0.0

        source_rates = {
            s: source_spikes[s]
            / (float(self.config.read_steps) * float(self.config.n_source_memory))
            for s in SOURCES
        }
        return {
            "source_spikes": source_spikes,
            "source_rates": source_rates,
            "apfc_source_spikes": apfc_spikes,
            "competition_spikes": competition_spikes,
            "afferent_spikes": afferent_spikes,
            "acc_spikes": float(acc_spikes),
            "max_afferent_current": float(max_afferent_current),
            "settle": settle,
        }


# Local alias so the copied _build_bridge stays readable.
APFC_SOURCE = {source: f"apfc_source_{source}" for source in SOURCES}


def _rival_burden(record: dict, expected: str) -> float:
    rates = record["source_rates"]
    return float(sum(rates[s] for s in SOURCES if s != expected))


def evaluate_attractor(
    seed: int,
    g_comp: float,
    overlap_fraction: float,
    *,
    config: SourceMonitorAttractorConfig | None = None,
) -> dict:
    """One seed x g_comp x overlap. Competition ON (M) vs OFF (L), all anti-cheats."""

    base = config or SourceMonitorAttractorConfig()
    c = SourceMonitorAttractorConfig(
        **{
            **{k: getattr(base, k) for k in base.__dataclass_fields__},
            "g_comp": float(g_comp),
        }
    )
    patterns, core = make_overlapping_episode_patterns(seed, c, overlap_fraction)
    t0 = time.time()

    # -- (d) zero-learned-weight instrument control: strict must be False --------
    ctrl = SourceMonitorAttractorCompetitionGate(seed=seed + 30000, config=c)
    ctrl_on = {s: ctrl.recall_instrumented(patterns[i]) for i, s in enumerate(SOURCES)}
    ctrl._set_comp_gate(0.0)
    try:
        ctrl_off = {s: ctrl.recall_instrumented(patterns[i]) for i, s in enumerate(SOURCES)}
    finally:
        ctrl._set_comp_gate(1.0)
    ctrl_M = {s: _source_margin(ctrl_on[s], s) for s in SOURCES}
    ctrl_L = {s: _source_margin(ctrl_off[s], s) for s in SOURCES}
    control_strict = bool(min(ctrl_M.values()) > min(ctrl_L.values()))

    # -- the real arm: learn, then competition ON (M) vs OFF (L) ----------------
    intact = SourceMonitorAttractorCompetitionGate(seed=seed, config=c)
    initial = intact.weight_summary()
    intact.experience(patterns[0], visual_activity=True)
    intact.experience(patterns[1], auditory_activity=True)
    intact.experience(patterns[2], corollary_discharge=True)
    intact.experience(patterns[3], visual_activity=True, auditory_activity=True)
    learned = intact.weight_summary()

    on = {s: intact.recall_instrumented(patterns[i]) for i, s in enumerate(SOURCES)}
    intact._set_comp_gate(0.0)
    try:
        off = {s: intact.recall_instrumented(patterns[i]) for i, s in enumerate(SOURCES)}
    finally:
        intact._set_comp_gate(1.0)

    margins_M = {s: _source_margin(on[s], s) for s in SOURCES}
    margins_L = {s: _source_margin(off[s], s) for s in SOURCES}
    own_rate_M = {s: float(on[s]["source_rates"][s]) for s in SOURCES}
    own_rate_L = {s: float(off[s]["source_rates"][s]) for s in SOURCES}
    rival_burden_on = {s: _rival_burden(on[s], s) for s in SOURCES}
    rival_burden_off = {s: _rival_burden(off[s], s) for s in SOURCES}
    dominant_correct = {s: bool(_dominant_source(on[s]) == s) for s in SOURCES}
    all_dominant_correct = bool(all(dominant_correct.values()))
    weakest_strict = bool(min(margins_M.values()) > min(margins_L.values()))
    min_M = float(min(margins_M.values()))
    min_L = float(min(margins_L.values()))
    clears_floor = bool(min_M >= MIN_SOURCE_MARGIN)

    # -- anti-cheat (a): g_comp == 0 M arm is byte-identical to L (feedforward) --
    byte_identical_null = None
    if float(g_comp) == 0.0:
        byte_identical_null = bool(
            all(abs(margins_M[s] - margins_L[s]) < 1e-12 for s in SOURCES)
        )

    # -- anti-cheat (b): honesty -- afferent current == 0 AND firing == 0 -------
    afferent_current_zero = bool(
        all(on[s]["max_afferent_current"] == 0.0 for s in SOURCES)
    )
    afferent_firing_zero = bool(
        all(sum(on[s]["afferent_spikes"].values()) == 0.0 for s in SOURCES)
    )
    # parameter symmetry: the recurrent-E and lateral-I weights carry no source
    # term (the same scalar wires every source) -- structural, asserted here.
    competition_param_symmetric = True

    # non-vacuity: on the UNSEEN (unlearned) episode cue -- which alone drives ~no
    # source memory -- force the 'seen' afferent. The winner must MOVE to 'seen',
    # proving the (normally-silent, honesty-guarded) afferent path is a real path
    # the guard genuinely excludes. patterns[4] is the disjoint unseen pattern.
    forced = "seen"
    forced_rec = intact.recall_instrumented(patterns[4], force_afferent=forced)
    forced_moves_winner = bool(_dominant_source(forced_rec) == forced)

    # -- anti-cheat (c): no source's own-recall rate collapses ------------------
    no_own_rate_collapse = bool(all(own_rate_M[s] > 0.0 for s in SOURCES))

    smoke_go = bool(clears_floor and weakest_strict and all_dominant_correct)

    return {
        "seed": int(seed),
        "g_comp": float(g_comp),
        "overlap_fraction": float(overlap_fraction),
        "core_size": int(core.size),
        "episode_pattern_size": int(c.episode_pattern_size),
        "recurrent_e_weight": float(g_comp * c.recurrent_e_weight_base),
        "lateral_i_weight": float(g_comp * c.lateral_i_weight_base),
        "weights_initial_l1": float(initial["l1"]),
        "weights_learned_l1": float(learned["l1"]),
        # decisive metrics
        "margins_M": margins_M,
        "margins_L": margins_L,
        "min_margin_M": min_M,
        "min_margin_L": min_L,
        "clears_floor": clears_floor,
        "weakest_source_strictly_improved": weakest_strict,
        "dominant_source_correct": dominant_correct,
        "all_dominant_correct": all_dominant_correct,
        "smoke_go": smoke_go,
        # supporting
        "own_rate_M": own_rate_M,
        "own_rate_L": own_rate_L,
        "rival_burden_on": rival_burden_on,
        "rival_burden_off": rival_burden_off,
        # anti-cheats
        "anti_cheats": {
            "control_zero_weight_strict": control_strict,  # (d) must be False
            "byte_identical_null_at_g0": byte_identical_null,  # (a); None unless g_comp==0
            "afferent_current_zero_at_recall": afferent_current_zero,  # (b)
            "afferent_firing_zero_at_recall": afferent_firing_zero,  # (b)
            "competition_param_symmetric": competition_param_symmetric,  # (b)
            "forced_afferent_moves_winner": forced_moves_winner,  # (b) non-vacuity
            "no_own_rate_collapse": no_own_rate_collapse,  # (c)
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CA3-style attractor-competition de-risk for co-resident source monitoring."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CALIBRATION_SEEDS))
    parser.add_argument("--g-comp", type=float, nargs="+", default=list(DEFAULT_G_COMP))
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    for g_comp in args.g_comp:
        for seed in args.seeds:
            row = evaluate_attractor(int(seed), float(g_comp), float(args.overlap))
            rows.append(row)
            ac = row["anti_cheats"]
            print(
                "[attractor-competition] "
                f"seed={row['seed']} g_comp={row['g_comp']:.2f} "
                f"overlap={row['overlap_fraction']:.2f} core={row['core_size']} "
                f"minM={row['min_margin_M']:.4f} minL={row['min_margin_L']:.4f} "
                f"clears={row['clears_floor']} strict={row['weakest_source_strictly_improved']} "
                f"dom_ok={row['all_dominant_correct']} SMOKE_GO={row['smoke_go']} "
                f"| ctrl_strict={ac['control_zero_weight_strict']} "
                f"aff0={ac['afferent_firing_zero_at_recall']}/{ac['afferent_current_zero_at_recall']} "
                f"forced_moves={ac['forced_afferent_moves_winner']} "
                f"no_collapse={ac['no_own_rate_collapse']} "
                f"byte_null={ac['byte_identical_null_at_g0']}",
                flush=True,
            )

    out = {
        "runner": "research/runners/_laneC_source_monitor_attractor_competition.py",
        "seeds": list(args.seeds),
        "g_comp": list(args.g_comp),
        "overlap": float(args.overlap),
        "min_source_margin_floor": MIN_SOURCE_MARGIN,
        "mechanism": (
            "within-population slow-NMDA recurrent excitation + between-population "
            "GABA-A lateral inhibition, symmetric across sources, one knob g_comp"
        ),
        "rows": rows,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[attractor-competition] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
