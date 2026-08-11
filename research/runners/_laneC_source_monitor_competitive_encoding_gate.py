"""Source-monitor competitive ENCODING: heterosynaptic LTD / target-source
commitment at the LEARNING step — the encoding-side surpass of the weak-source
residual that recall-time gain could not clear from EITHER side.

Read the two priors before touching this:
  * `_laneC_source_monitor_popcode_homeostasis_gate` (the recall-time baseline):
    independent per-source POPULATION-CODED pools (n=24) + up-only homeostatic
    synaptic scaling STRUCTURALLY resolve the no-harm boundary and clear the 0.15
    source-margin floor on 5/6 fresh seeds — but NOT universally. ~2/55 observed
    seeds (244, 259) stay below the floor at every pool size, and the finding named
    the convergent next mechanism: COMPETITIVE ENCODING (heterosynaptic LTD at the
    ENCODING step).  See
    research/findings/2026-08-11-source-monitor-no-harm-boundary-is-an-operating-point-population-coding-derisk.md
  * The disjoint-side and overlap-side recall-time NO-GOs both pointed at the
    encoding step.

WHAT THE WEAK SEEDS ACTUALLY ARE (diagnostic, seeds 244 + 259, popcode+homeo
baseline).  The weak source is HEARD, and it fires at a HEALTHY rate (~0.19,
comparable to the strong sources' ~0.20-0.24).  The margin is small because a RIVAL
pool (SEEN) spuriously CO-FIRES at heard's recall (seen=0.0600 on seed 244,
seen=0.1175 on seed 259), eating the margin:
    seed 244 heard recall: heard=0.1988  seen=0.0600  -> margin 0.1388  (< floor)
    seed 259 heard recall: heard=0.1871  seen=0.1175  -> margin 0.0696  (< floor)
So the residual is NOT weak own-encoding (recall-time gain over-drives it into the
f-I ceiling, exactly the prior NO-GOs).  It is a CROSS-TALK / orthogonalization
deficit — precisely the class competitive encoding / heterosynaptic LTD is for.

THE MECHANISM (heterosynaptic LTD, activity-local: pre active, post silent -> LTD).
During source s's encoding window its episode pattern drives episode neurons while
the RIVAL pools' cells are silent (their afferents are off).  The classic
heterosynaptic / anti-Hebbian arm depresses those pre-active/post-silent synapses:
the p_s -> rival-pool synapses are depressed (a target-source COMMITMENT — each pool
commits to the patterns it was co-active with, decorrelating the codes).  This is
computed here as a one-shot consolidation from the recorded encoding activity (the
same host-computed/host-timed scaffold the homeostatic scaling introduced; its
spiking online-anti-Hebbian implementation is deferred and named).

STRUCTURAL NO-HARM — STRONGER than the homeostatic proof.  Competitive encoding
depresses ONLY foreign-pre synapses (pre active for another source, post in THIS
pool).  At recall of any source's OWN pattern, only that pattern's episode neurons
fire; the target pool's OWN synapses are never depressed (so its rate is unchanged),
and the ONLY synapses depressed anywhere are cross-talk synapses that make a RIVAL
fire — so every source's rivals can only fire LESS.  Therefore EVERY source's margin
can only rise or stay equal.  No weight that drives a source's own recall is touched;
no cross-pool budget exists.

SCAFFOLD LEDGER (unchanged from popcode + one): caller-supplied sparse episode
activity, physical source-afferent identity, an externally timed learning window,
host spike-count evaluation, and the host-computed/host-timed homeostatic scaling all
remain scaffolds.  NEW scaffold: the heterosynaptic-LTD consolidation is likewise
host-computed and host-timed (a one-shot commitment derived from the recorded
per-source encoding activity — the SAME information the existing source-afferent /
learning-window scaffolds already supply).  The biology it stands in for
(heterosynaptic / anti-Hebbian depression, pre-active/post-silent -> LTD) is real;
its spiking online implementation is deferred and named.  No language, confidence
scalar, or response policy is claimed.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger("SIM_BRIDGE").setLevel(logging.WARNING)

from research.runners._laneC_source_monitor_coresidency_gate import (
    ACC_GATE,
    ACC_REGION,
    DEVELOPMENT_MIN_ATTRIBUTION_FRACTION,
    DEVELOPMENT_MIN_SOURCE_MARGIN,
    EPISODE_REGION,
    SOURCE_AFFERENT,
    SOURCE_MEMORY,
    SOURCE_RECALL_GATE,
    SOURCES,
    APFC_SOURCE,
    _dominant_source,
    _source_margin,
    make_episode_patterns,
)
from research.runners._laneC_source_monitor_popcode_homeostasis_gate import (
    SourceMonitorConfigPop,
    SourceMonitorPopcodeHomeostasisGate,
)
from sim.backend import to_host
from tools.lab import attributable_to, lever, zero_lever_control
from tools.verdict import UNDEFINED, Verdict


# The weak-encoding class is OBSERVED exploration (used to freeze the mechanism):
# the finding named 244 and 259 as the ~2/55 seeds that stay below the floor at
# every pool size and are not rescued by recall-time gain.  The fresh 700-705 seeds
# are the popcode finding's decisive partition; here they are no-harm / generalization
# guards (they already clear the floor at the popcode+homeo baseline, so competitive
# encoding must PRESERVE them, not damage them).
WEAK_OBSERVED_SEEDS = (244, 259)
DECISIVE_FRESH_SEEDS = (700, 701, 702, 703, 704, 705)

MIN_SOURCE_MARGIN = DEVELOPMENT_MIN_SOURCE_MARGIN          # 0.15, unchanged
MIN_ATTRIBUTION_FRACTION = DEVELOPMENT_MIN_ATTRIBUTION_FRACTION
NO_HARM_EPSILON = 0.01

# Which episode patterns were co-active with each source's afferent during the fixed
# experience schedule below (p0 seen, p1 heard, p2 self, p3 mixed seen+heard).  This
# is the recorded encoding activity the heterosynaptic-LTD consolidation reads — the
# same source-afferent / learning-window information the existing scaffolds supply.
OWN_PATTERN_INDICES = {"seen": (0, 3), "heard": (1, 3), "self_generated": (2,)}


@dataclass(frozen=True)
class SourceMonitorConfigCompete(SourceMonitorConfigPop):
    """Popcode+homeo operating point plus the competitive-encoding knobs."""

    # Heterosynaptic LTD / target-source commitment at the encoding step.  Default
    # OFF so the runner is byte-identical to the popcode+homeo baseline when off.
    enable_competitive_encoding: bool = False
    # Depression depth for a foreign (cross-talk) synapse: w *= (1 - depth).
    # depth=1.0 is a full target-source commitment (the cross-talk synapse is
    # zeroed).  Not tuned per seed; frozen.
    hetero_ltd_depth: float = 1.0


class SourceMonitorCompetitiveEncodingGate(SourceMonitorPopcodeHomeostasisGate):
    """Popcode + homeostasis, plus a one-shot heterosynaptic-LTD (target-source
    commitment) consolidation at the encoding step.  Inherits the whole bridge and
    the homeostatic scaling unchanged; adds only the encoding-time decorrelation."""

    def __init__(
        self,
        *,
        seed: int,
        config: SourceMonitorConfigCompete | Mapping | None = None,
    ):
        c = (
            config
            if isinstance(config, SourceMonitorConfigCompete)
            else SourceMonitorConfigCompete(**(dict(config) if config else {}))
        )
        super().__init__(seed=seed, config=c)
        self._pre_ce_weights: np.ndarray | None = None

    # --- per-learned-synapse (pre, post) global indices -------------------------
    def _learned_pre_post(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        learned = np.asarray(self._learned_synapse_indices(), dtype=np.int64)
        coo = self.bridge.cp_connections.tocoo(copy=False)
        pre = np.asarray(to_host(coo.row), dtype=np.int64)[learned]
        post = np.asarray(to_host(coo.col), dtype=np.int64)[learned]
        return learned, pre, post

    def _own_pre_global(
        self, own_patterns: Mapping[str, Sequence[Sequence[int]]]
    ) -> dict[str, set[int]]:
        out: dict[str, set[int]] = {}
        for source in SOURCES:
            g: set[int] = set()
            for local in own_patterns[source]:
                g |= set(int(i) for i in self._episode_global_indices(local).tolist())
            out[source] = g
        return out

    def apply_competitive_encoding(
        self, own_patterns: Mapping[str, Sequence[Sequence[int]]]
    ) -> dict:
        """One-shot heterosynaptic LTD / target-source commitment at encoding.

        For each source pool s, depress incoming learned synapses whose PRE episode
        neuron was active during ANOTHER source's encoding window but NOT during s's
        own (a foreign / cross-talk synapse: pre-active-elsewhere, post-silent-here).

        Returns the applied depression and the STRUCTURAL no-harm proof:
          * only foreign-pre synapses are touched (own-pattern synapses are never
            depressed), and each pool's foreign set targets only its own pool cells,
            so no synapse that drives a source's OWN recall is reduced.
        """

        learned, pre, post = self._learned_pre_post()
        own_pre = self._own_pre_global(own_patterns)
        all_own: set[int] = set().union(*own_pre.values())
        mem = {
            source: set(int(i) for i in self._source_memory_indices[source].tolist())
            for source in SOURCES
        }

        # freeze pre-CE weights (lesion target + the baseline OFF arm)
        data = self.bridge.cp_connections.data
        self._pre_ce_weights = np.asarray(to_host(data), dtype=np.float64).copy()

        depth = float(self.config.hetero_ltd_depth)
        touched_own = 0
        n_depressed = 0
        foreign_l1_before = 0.0
        foreign_l1_after = 0.0
        touched_pools_disjoint = True
        seen_touched: set[int] = set()
        if self.config.enable_competitive_encoding and depth > 0.0:
            host_learned = np.asarray(to_host(data), dtype=np.float64)[learned]
            for source in SOURCES:
                foreign_pre = all_own - own_pre[source]  # active elsewhere, not here
                in_pool = np.array([int(p) in mem[source] for p in post], dtype=bool)
                is_foreign = np.array([int(p) in foreign_pre for p in pre], dtype=bool)
                mask = in_pool & is_foreign
                # structural guard: a depressed synapse must never be an OWN synapse
                is_own = np.array([int(p) in own_pre[source] for p in pre], dtype=bool)
                touched_own += int(np.count_nonzero(in_pool & is_own & mask))
                flat = learned[mask]
                if flat.size:
                    fl = set(int(i) for i in flat.tolist())
                    if fl & seen_touched:
                        touched_pools_disjoint = False
                    seen_touched |= fl
                    foreign_l1_before += float(host_learned[mask].sum())
                    self.bridge.cp_connections.data[flat] *= (1.0 - depth)
                    n_depressed += int(flat.size)
            after = np.asarray(to_host(self.bridge.cp_connections.data), dtype=np.float64)
            for source in SOURCES:
                foreign_pre = all_own - own_pre[source]
                in_pool = np.array([int(p) in mem[source] for p in post], dtype=bool)
                is_foreign = np.array([int(p) in foreign_pre for p in pre], dtype=bool)
                foreign_l1_after += float(after[learned][in_pool & is_foreign].sum())

        return {
            "enabled": bool(self.config.enable_competitive_encoding),
            "hetero_ltd_depth": depth,
            "n_foreign_synapses_depressed": int(n_depressed),
            "foreign_l1_before": float(foreign_l1_before),
            "foreign_l1_after": float(foreign_l1_after),
            "no_own_synapse_touched": bool(touched_own == 0),
            "per_pool_touched_disjoint": bool(touched_pools_disjoint),
            "engaged": bool(n_depressed > 0),
        }

    def lesion_competitive_encoding(self) -> None:
        """Restore the pre-competitive-encoding weights (the mechanism lesion =
        the popcode+homeo baseline)."""
        if self._pre_ce_weights is not None:
            xp_data = self.bridge.cp_connections.data
            self.bridge.cp_connections.data[:] = np.asarray(
                self._pre_ce_weights, dtype=xp_data.dtype
            )


def phase_for(seed: int) -> str:
    if seed in WEAK_OBSERVED_SEEDS:
        return "weak_observed"
    if seed in DECISIVE_FRESH_SEEDS:
        return "decisive_fresh"
    raise ValueError(
        f"seed {seed} is not declared; weak_observed={WEAK_OBSERVED_SEEDS}, "
        f"decisive_fresh={DECISIVE_FRESH_SEEDS}"
    )


def _margins_all(gate, pat) -> dict[str, float]:
    gate.reset_dynamical_state()
    return {s: _source_margin(gate.recall(pat[s]), s) for s in SOURCES}


def evaluate_seed(seed: int, *, config: SourceMonitorConfigCompete | None = None) -> dict:
    """Run one seed with the full popcode control suite plus the competitive-encoding
    no-harm + load-bearing controls.  ON = popcode + homeo + competitive encoding;
    OFF (baseline) = popcode + homeo (the prior recall-gain method)."""

    ph = phase_for(int(seed))
    c = config or SourceMonitorConfigCompete(enable_competitive_encoding=True)
    patterns = make_episode_patterns(seed, 5, c)
    pat = {"seen": patterns[0], "heard": patterns[1], "self_generated": patterns[2]}
    own_patterns = {s: [patterns[i] for i in OWN_PATTERN_INDICES[s]] for s in SOURCES}
    t0 = time.time()

    intact = SourceMonitorCompetitiveEncodingGate(seed=seed, config=c)
    initial = intact.weight_summary()
    intact.experience(patterns[0], visual_activity=True)
    intact.experience(patterns[1], auditory_activity=True)
    intact.experience(patterns[2], corollary_discharge=True)
    intact.experience(patterns[3], visual_activity=True, auditory_activity=True)
    learned = intact.weight_summary()

    # Baseline mechanism (the OFF arm the prior finding earned): popcode + homeo.
    homeo = intact.apply_homeostatic_scaling(pat)

    # OFF (competitive-encoding-lesioned) margins = the popcode+homeo baseline.
    baseline_margins = _margins_all(intact, pat)

    # ON: apply the competitive-encoding (heterosynaptic LTD) consolidation.
    ce = intact.apply_competitive_encoding(own_patterns)
    ce_margins = _margins_all(intact, pat)

    # Load-bearing lesion: restore the pre-CE (baseline) weights, re-measure.
    intact.lesion_competitive_encoding()
    lesion_margins = _margins_all(intact, pat)

    # Re-apply CE for the downstream (ON-state) controls.
    intact.apply_competitive_encoding(own_patterns)
    intact.reset_dynamical_state()
    seen = intact.recall(patterns[0])
    heard = intact.recall(patterns[1])
    self_generated = intact.recall(patterns[2])
    mixed = intact.recall(patterns[3])
    unseen = intact.recall(patterns[4])
    source_lesion = intact.recall(patterns[0], source_path_lesion=True)
    acc_lesion = intact.recall(patterns[0], acc_lesion=True)

    margins = {
        "seen": _source_margin(seen, "seen"),
        "heard": _source_margin(heard, "heard"),
        "self_generated": _source_margin(self_generated, "self_generated"),
    }
    no_harm_gains = {s: ce_margins[s] - baseline_margins[s] for s in SOURCES}
    weakest_source = min(SOURCES, key=lambda s: baseline_margins[s])

    # --- anti-cheats (execute, do not remember) --------------------------------
    # (d) the lever MOVED: foreign (cross-talk) synaptic weight was actually depressed.
    lever(
        "competitive-encoding foreign L1",
        round(ce["foreign_l1_before"], 6),
        round(ce["foreign_l1_after"], 6),
        required=bool(ce["enabled"]),
    )
    # (a) load-bearing on the weak class: the weakest-source floor criterion must FAIL
    #     when the mechanism is a no-op (competitive-encoding OFF == baseline).  Only
    #     enforced where the baseline is genuinely below the floor (the weak seeds).
    baseline_min = min(baseline_margins.values())
    ce_min = min(ce_margins.values())
    if ce["enabled"] and baseline_min < MIN_SOURCE_MARGIN:
        zero_lever_control(
            "weakest_margin_meets_floor",
            criterion_on=bool(ce_min >= MIN_SOURCE_MARGIN),
            criterion_off=bool(baseline_min >= MIN_SOURCE_MARGIN),
        )
    # (attributable_to) the floor gain is attributable to the mechanism, not carried state.
    weak_gain_fraction = attributable_to(
        "weakest-source margin (competitive encoding)",
        float(ce_margins[weakest_source]),
        float(baseline_margins[weakest_source]),
    )

    intact_source_total = float(sum(seen["source_spikes"].values()))
    lesioned_source_total = float(sum(source_lesion["source_spikes"].values()))
    source_path_fraction = attributable_to(
        "source recall pathway", intact_source_total, lesioned_source_total
    )
    acc_path_fraction = attributable_to(
        "source-to-ACC pathway",
        float(seen["acc_spikes"]),
        float(acc_lesion["acc_spikes"]),
    )

    # structural no-harm proof for competitive encoding
    ce_structural_no_harm = bool(
        (not ce["enabled"]) or (ce["no_own_synapse_touched"] and ce["per_pool_touched_disjoint"])
    )
    # lesion restores the baseline exactly (load-bearing + clean lever)
    lesion_restores_baseline = bool(
        max(abs(lesion_margins[s] - baseline_margins[s]) for s in SOURCES) <= 1e-9
    )

    swapped = SourceMonitorCompetitiveEncodingGate(seed=seed + 10000, config=c)
    swapped.experience(patterns[0], auditory_activity=True)
    swapped.experience(patterns[1], visual_activity=True)
    swap_zero = swapped.recall(patterns[0])
    swap_one = swapped.recall(patterns[1])

    learning_off = SourceMonitorCompetitiveEncodingGate(seed=seed + 20000, config=c)
    off_initial = learning_off.weight_summary()
    learning_off.experience(patterns[0], visual_activity=True, learning_enabled=False)
    off_after = learning_off.weight_summary()
    off_recall = learning_off.recall(patterns[0])

    components = {
        "learned_routes_start_zero": bool(initial["l1"] == 0.0),
        "experience_changes_synaptic_weights": bool(learned["l1"] > initial["l1"]),
        "seen_source_recalled": bool(
            _dominant_source(seen) == "seen" and margins["seen"] > 0.0
        ),
        "heard_source_recalled": bool(
            _dominant_source(heard) == "heard" and margins["heard"] > 0.0
        ),
        "self_source_recalled": bool(
            _dominant_source(self_generated) == "self_generated"
            and margins["self_generated"] > 0.0
        ),
        "all_source_margins_meet_fixed_floor": bool(min(margins.values()) >= MIN_SOURCE_MARGIN),
        "source_swap_follows_afferent_activity": bool(
            _dominant_source(swap_zero) == "heard" and _dominant_source(swap_one) == "seen"
        ),
        "mixed_source_reinstates_both": bool(
            mixed["source_spikes"]["seen"] > 0.0 and mixed["source_spikes"]["heard"] > 0.0
        ),
        "source_path_lesion_collapses_recall": bool(
            lesioned_source_total <= 0.10 * max(intact_source_total, 1.0)
        ),
        "source_path_attribution_meets_fixed_floor": bool(
            source_path_fraction is not None and source_path_fraction >= MIN_ATTRIBUTION_FRACTION
        ),
        "acc_lesion_preserves_source_and_silences_acc": bool(
            sum(acc_lesion["source_spikes"].values()) >= 0.90 * intact_source_total
            and acc_lesion["acc_spikes"] == 0.0
        ),
        "acc_path_attribution_meets_fixed_floor": bool(
            acc_path_fraction is not None and acc_path_fraction >= MIN_ATTRIBUTION_FRACTION
        ),
        "learning_off_keeps_weights_zero": bool(
            off_initial["l1"] == 0.0 and off_after["l1"] == 0.0
        ),
        "learning_off_has_no_source_recall": bool(sum(off_recall["source_spikes"].values()) == 0.0),
        "unseen_episode_has_negligible_source_recall": bool(
            sum(unseen["source_spikes"].values()) <= 0.02 * max(intact_source_total, 1.0)
        ),
        "source_spikes_reach_apfc_and_acc": bool(
            seen["apfc_source_spikes"]["seen"] > 0.0 and seen["acc_spikes"] > 0.0
        ),
        # (b) NO-HARM: competitive encoding must not reduce ANY source's margin.
        "competitive_encoding_no_harm": bool(
            ce_structural_no_harm and min(no_harm_gains.values()) >= -NO_HARM_EPSILON
        ),
        # (a) LOAD-BEARING: where the baseline fails the floor, competitive encoding
        #     must flip it; where the baseline already passes, this is vacuously true.
        "competitive_encoding_flips_weak_seed": bool(
            (baseline_min >= MIN_SOURCE_MARGIN) or (ce_min >= MIN_SOURCE_MARGIN)
        ),
        "competitive_encoding_lesion_restores_baseline": lesion_restores_baseline,
    }

    region_names = {region.name for region in intact.bridge.region_manager.regions()}
    expected_regions = {EPISODE_REGION, ACC_REGION}
    expected_regions.update(SOURCE_AFFERENT.values())
    expected_regions.update(SOURCE_MEMORY.values())
    expected_regions.update(APFC_SOURCE.values())
    recall_parameters = list(
        inspect.signature(SourceMonitorCompetitiveEncodingGate.recall).parameters
    )
    earned = Verdict("source-monitor competitive encoding")
    earned.require(
        "episode, source, aPFC, and ACC populations share one bridge",
        expected_regions.issubset(region_names),
        expect=True,
    )
    earned.require(
        "recall accepts episode activity without source metadata",
        recall_parameters == ["self", "episode_pattern", "source_path_lesion", "acc_lesion"],
        expect=True,
    )
    earned.require(
        "population-coded pools (n_source_memory > V1's 12), no cross-pool competition",
        int(c.n_source_memory) > 12,
        expect=True,
    )
    earned.require(
        "competitive encoding touches ONLY foreign (cross-talk) synapses (structural no-harm)",
        ce_structural_no_harm,
        expect=True,
    )
    earned.require(
        "competitive-encoding lesion restores the popcode+homeo baseline exactly",
        lesion_restores_baseline,
        expect=True,
    )
    earned.reaches(
        "episode-to-source transmission lesion",
        before=intact_source_total,
        after=lesioned_source_total,
    )
    earned.reaches(
        "source-to-ACC transmission lesion",
        before=float(seen["acc_spikes"]),
        after=float(acc_lesion["acc_spikes"]),
    )
    earned.disabled(
        "STDP, reward modulation, homeostasis (engine), short-term & structural plasticity",
        why="this version isolates population-coded Hebbian source association, an up-only "
        "homeostatic scaling consolidation, and an explicit heterosynaptic-LTD "
        "(target-source commitment) consolidation at the encoding step",
    )
    decided = earned.decide(go=all(components.values()), verbose=False)
    status = (
        "UNDEFINED"
        if decided["status"] == UNDEFINED
        else f"{ph.upper()}_PASS" if decided["go"] else f"{ph.upper()}_FAIL"
    )
    return {
        "seed": int(seed),
        "phase": ph,
        "status": status,
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "components": components,
        "metrics": {
            "seen_margin": margins["seen"],
            "heard_margin": margins["heard"],
            "self_generated_margin": margins["self_generated"],
            "minimum_source_margin": min(margins.values()),
            "baseline_seen_margin": baseline_margins["seen"],
            "baseline_heard_margin": baseline_margins["heard"],
            "baseline_self_generated_margin": baseline_margins["self_generated"],
            "baseline_minimum_source_margin": baseline_min,
            "competitive_encoding_minimum_source_margin": ce_min,
            "weakest_source": weakest_source,
            "weakest_baseline_margin": baseline_margins[weakest_source],
            "weakest_competitive_encoding_margin": ce_margins[weakest_source],
            "no_harm_margin_gains": no_harm_gains,
            "no_harm_min_gain": min(no_harm_gains.values()),
            "lesion_margins": lesion_margins,
            "weakest_margin_gain_fraction": weak_gain_fraction,
            "homeostasis_factors": homeo["factors"],
            "intact_source_spikes": intact_source_total,
            "source_lesion_spikes": lesioned_source_total,
            "intact_acc_spikes": float(seen["acc_spikes"]),
            "acc_lesion_spikes": float(acc_lesion["acc_spikes"]),
            "mixed_seen_spikes": float(mixed["source_spikes"]["seen"]),
            "mixed_heard_spikes": float(mixed["source_spikes"]["heard"]),
        },
        "competitive_encoding": ce,
        "homeostasis": homeo,
        "attribution": {
            "weakest_source_margin": {
                "competitive_encoding_margin": float(ce_margins[weakest_source]),
                "baseline_margin": float(baseline_margins[weakest_source]),
                "attributable_fraction": weak_gain_fraction,
            },
            "source_recall_path": {
                "intact_source_spikes": intact_source_total,
                "lesioned_source_spikes": lesioned_source_total,
                "lesion_delta": intact_source_total - lesioned_source_total,
                "attributable_to": SOURCE_RECALL_GATE,
                "attributable_fraction": source_path_fraction,
            },
            "acc_output_path": {
                "intact_acc_spikes": float(seen["acc_spikes"]),
                "lesioned_acc_spikes": float(acc_lesion["acc_spikes"]),
                "lesion_delta": float(seen["acc_spikes"] - acc_lesion["acc_spikes"]),
                "attributable_to": ACC_GATE,
                "attributable_fraction": acc_path_fraction,
            },
        },
        "records": {
            "seen": seen,
            "heard": heard,
            "self_generated": self_generated,
            "mixed": mixed,
            "unseen": unseen,
            "source_path_lesion": source_lesion,
            "acc_lesion": acc_lesion,
            "swap_pattern_zero": swap_zero,
            "swap_pattern_one": swap_one,
            "learning_off": off_recall,
        },
        "weights": {
            "initial": initial,
            "learned": learned,
            "learning_off_initial": off_initial,
            "learning_off_after": off_after,
        },
        "config": asdict(c),
        "interface_guards": {
            "recall_parameters": recall_parameters,
            "no_source_argument_at_inference": "source"
            not in inspect.signature(SourceMonitorCompetitiveEncodingGate.recall).parameters,
            "episode_activity_is_caller_supplied": True,
            "cross_pool_competition": False,
            "shared_inhibitory_budget": False,
            "host_homeostatic_scaling": True,
            "host_heterosynaptic_ltd_consolidation": True,
            "stronger_source_specific_drive": False,
            "host_confidence_scalar": False,
            "host_response_decision": False,
        },
        "seed_policy": {
            "weak_observed": list(WEAK_OBSERVED_SEEDS),
            "decisive_fresh": list(DECISIVE_FRESH_SEEDS),
        },
        "fixed_criteria": {
            "minimum_source_margin": MIN_SOURCE_MARGIN,
            "minimum_attribution_fraction": MIN_ATTRIBUTION_FRACTION,
            "no_harm_epsilon": NO_HARM_EPSILON,
        },
        "honest_scope": (
            "Independent per-source population-coded pools (no cross-pool competition, no "
            "shared inhibitory budget) plus an up-only per-source homeostatic scaling "
            "consolidation plus a one-shot heterosynaptic-LTD (target-source COMMITMENT) "
            "consolidation at the encoding step: for each pool, foreign (cross-talk) synapses "
            "whose presynaptic episode neuron was active during another source's encoding are "
            "depressed.  No-harm is structural: only foreign synapses are touched, no synapse "
            "that drives a source's OWN recall is reduced, and a rival can only fire LESS, so "
            "every source's margin can only rise.  Scaffolds: caller-supplied sparse episode "
            "activity, physical source afferents, an externally timed learning window, host "
            "spike-count evaluation, and the host-computed/host-timed homeostatic + "
            "heterosynaptic-LTD consolidations (the biology — anti-Hebbian pre-active/post-silent "
            "depression — is real; its spiking online implementation is deferred and named).  No "
            "language, confidence scalar, or response policy is claimed."
        ),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Source-monitor competitive-encoding (heterosynaptic LTD) gate."
    )
    parser.add_argument("--seed", type=int, default=WEAK_OBSERVED_SEEDS[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="self-sweep: run several seeds and aggregate (decisive use)")
    parser.add_argument("--competitive-encoding", choices=("on", "off"), default="on",
                        help="on = popcode+homeo+competitive encoding; off = popcode+homeo baseline")
    parser.add_argument("--hetero-ltd-depth", type=float, default=1.0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    cfg = SourceMonitorConfigCompete(
        enable_competitive_encoding=(args.competitive_encoding == "on"),
        hetero_ltd_depth=float(args.hetero_ltd_depth),
    )
    seeds = args.seeds if args.seeds is not None else [args.seed]
    rows = []
    for s in seeds:
        row = evaluate_seed(s, config=cfg)
        rows.append(row)
        m = row["metrics"]
        print(
            f"[competitive-encoding] seed={row['seed']} phase={row['phase']} status={row['status']} "
            f"ce={args.competitive_encoding} weakest={m['weakest_source']} "
            f"baseline_min={m['baseline_minimum_source_margin']:+.4f} "
            f"ce_min={m['competitive_encoding_minimum_source_margin']:+.4f} "
            f"no_harm_min_gain={m['no_harm_min_gain']:+.4f} "
            f"n_depressed={row['competitive_encoding']['n_foreign_synapses_depressed']}",
            flush=True,
        )

    go = all(r["status"].endswith("_PASS") for r in rows)
    n_pass = sum(r["status"].endswith("_PASS") for r in rows)
    print(f"[competitive-encoding] AGGREGATE {n_pass}/{len(rows)} PASS  ->  {'GO' if go else 'NO-GO'}",
          flush=True)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = rows[0] if len(rows) == 1 else {
            "aggregate": {"n_pass": n_pass, "n_total": len(rows), "go": go},
            "rows": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[competitive-encoding] wrote {out_path}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
