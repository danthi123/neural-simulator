"""Calibration v6 gate: silent-by-construction source-recall gating.

The v2 local fast-spiking GABA-A biased-competition MECHANISM is unchanged, and
the bounded-loss, guard-the-floor, max-min acceptance rule fixed by the P3
functional-role spec is unchanged.  v6 changes ONLY the source-recall protocol.

v5 recorded a NO-GO on ONE inherited anti-cheat control on seed 650:
``learning_off_has_no_source_recall`` leaked four ``seen`` source-memory spikes
even though the learned episode-to-source weights were provably zero
(2026-08-06-source-monitor-coresidency-v5-calibration-NO-GO-...).  The leak was
diagnosed to residual encoding-phase state, NOT any synaptic path: the strong
source-afferent drive during experience leaves the source-memory neurons
depolarised (V ~= -40 mV) with a large Izhikevich adaptation variable ``u``, and
the recall read begins before that state has decayed, so a handful of neurons
drift across threshold with zero input (the leak reproduces with the episode
drive removed entirely).

The fix makes recall silent by construction: before the read window, recall
settles the substrate at zero input until the readout populations (source
memory, aPFC, ACC, source afferents) are quiescent for a full rest window.  With
learning off (zero learned weights, no afferent drive) a quiescent start
guarantees the source populations stay silent on ANY seed, because nothing can
drive them during the read.  With learning on, the read is unchanged in kind:
the episode drive reinstates source activity through the learned synapses from a
clean baseline.  The biased-competition circuit and acceptance rule are retained
unchanged; only the source-recall gating is tightened.

Calibration seeds 650 and 651 (plus smoke seed 649) established the frozen
mechanism.  Generalization runs the IDENTICAL evaluator and criteria on unseen
seeds via ``--phase``: development opens 652, 653, 654; held-out opens 655, 656,
657 ONLY after development records an earned GO (``validate_phase_seed`` seals it
otherwise), so an unproven phase can never open the next one.  Only the seed
partition advances; the circuit, thresholds, and acceptance rule are frozen.
The runner does not choose words, calculate confidence, or decide whether to
respond.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger("SIM_BRIDGE").setLevel(logging.WARNING)

from research.runners._laneC_source_monitor_coresidency_gate import (
    ACC_GATE,
    ACC_REGION,
    APFC_SOURCE,
    DEVELOPMENT_MIN_ATTRIBUTION_FRACTION,
    DEVELOPMENT_MIN_SOURCE_MARGIN,
    EPISODE_REGION,
    SOURCE_AFFERENT,
    SOURCE_MEMORY,
    SOURCE_RECALL_GATE,
    SOURCES,
    SourceMonitorConfig,
    _dominant_source,
    _source_margin,
    make_episode_patterns,
)
from research.runners._laneC_source_monitor_coresidency_gate_v5 import (
    SOURCE_COMPETITION_GATE,
    SOURCE_INTERNEURON,
    SourceMonitorConfigV2,
    SourceMonitorCoresidencyGateV2,
)
from sim.backend import to_host
from tools.lab import attributable_to
from tools.verdict import UNDEFINED, Verdict


SMOKE_SEED = 649
CALIBRATION_SEEDS = (650, 651)
DEVELOPMENT_SEEDS = (652, 653, 654)
HELD_OUT_SEEDS = (655, 656, 657)

MIN_SOURCE_MARGIN = DEVELOPMENT_MIN_SOURCE_MARGIN
MIN_ATTRIBUTION_FRACTION = DEVELOPMENT_MIN_ATTRIBUTION_FRACTION

# Cap on the pre-read settle so a pathological seed cannot loop forever. In
# practice the substrate reaches quiescence within two rest windows.
MAX_SETTLE_BLOCKS = 12

# Phase -> seed partition. Calibration is closed (already spent on the frozen
# mechanism). Development opens 652/653/654; held_out opens 655/656/657 ONLY
# after development records an earned GO (the seal below), so an unproven phase
# can never open the next one. The mechanism, thresholds, and acceptance rule
# are identical across phases; only the seed partition advances.
PHASE_SEEDS = {
    "calibration": CALIBRATION_SEEDS,
    "development": DEVELOPMENT_SEEDS,
    "held_out": HELD_OUT_SEEDS,
}

# Aggregate generalization verdicts (written by aggregate_source_monitor_seeds).
GENERALIZATION_DIR = Path("research/findings/raw/source_monitor_v6_generalization")
DEV_VERDICT_PATH = GENERALIZATION_DIR / "development_verdict.json"


def _development_is_go() -> bool:
    """True only if the development aggregate verdict exists and reads GO."""

    try:
        data = json.loads(DEV_VERDICT_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("phase") == "development" and data.get("verdict") == "GO"


def validate_phase_seed(seed: int, phase: str = "calibration") -> int:
    """Open calibration + development seeds; keep held_out sealed until dev GO.

    Mirrors the calibration seed-validation discipline: a seed outside the
    named phase is mechanically rejected, and the held_out partition stays
    sealed until the development phase records an earned GO verdict, so the
    held-out seeds cannot be touched before the mechanism has generalized on the
    development partition.
    """

    seed = int(seed)
    allowed = PHASE_SEEDS.get(phase)
    if allowed is None:
        raise ValueError(
            f"phase {phase!r} is not a v6 phase; choose from {tuple(PHASE_SEEDS)}"
        )
    if phase == "held_out" and not _development_is_go():
        raise ValueError(
            "held_out seeds are sealed until the development phase records an "
            f"earned GO in {DEV_VERDICT_PATH}; run --phase development first"
        )
    if phase == "calibration" and seed == SMOKE_SEED:
        return seed
    if seed not in allowed:
        raise ValueError(
            f"seed {seed} is not a v6 {phase} seed; allowed={allowed}, "
            f"smoke={SMOKE_SEED}, calibration={CALIBRATION_SEEDS}, "
            f"development={DEVELOPMENT_SEEDS}, held_out={HELD_OUT_SEEDS}"
        )
    return seed


class SourceMonitorCoresidencyGateV6(SourceMonitorCoresidencyGateV2):
    """v2 circuit with a silent-by-construction pre-read recall protocol."""

    def _settle_to_quiescence(self, max_blocks: int = MAX_SETTLE_BLOCKS) -> dict:
        """Rest at zero input until the readout populations are silent.

        Competition is gated off during settle (as in ``_rest``) so no
        post-inhibitory rebound is introduced.  The substrate is stepped in
        ``rest_steps`` blocks until a full block passes with zero spikes across
        the source-memory, aPFC, ACC, and source-afferent populations.  This
        makes the subsequent read begin from a clean baseline on every seed, so
        with zero learned weights and no afferent drive the source populations
        are silent by construction rather than merely usually silent.
        """

        readout = np.concatenate(
            [self._source_memory_indices[s] for s in SOURCES]
            + [self._apfc_indices[s] for s in SOURCES]
            + [self._source_afferent_indices[s] for s in SOURCES]
            + [self._acc_indices]
        )
        prior_gain = float(
            self.bridge._transmission_gate_values[SOURCE_COMPETITION_GATE]
        )
        self.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
        self.bridge.cp_external_input_current[:] = 0.0
        steps = 0
        blocks = 0
        reached = False
        block = int(self.config.rest_steps)
        try:
            for _ in range(int(max_blocks)):
                blocks += 1
                quiet = True
                for _ in range(block):
                    self.bridge._run_one_simulation_step()
                    steps += 1
                    firing = np.asarray(
                        to_host(self.bridge.cp_firing_states), dtype=np.float64
                    )
                    if float(firing[readout].sum()) > 0.0:
                        quiet = False
                if quiet:
                    reached = True
                    break
        finally:
            self.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, prior_gain)
        return {"settle_steps": steps, "settle_blocks": blocks, "reached_quiescence": reached}

    def recall(
        self,
        episode_pattern: Sequence[int],
        *,
        source_path_lesion: bool = False,
        acc_lesion: bool = False,
    ) -> dict:
        """Settle the substrate to quiescence, then read the learned pathway.

        The read is made history-independent by first restoring the fast Izhikevich
        sub-threshold state (membrane v, adaptation u, conductances, refractory +
        pulse timers, firing flags, activity EMA) to the clean post-construction
        baseline: settle-to-quiescence guarantees no residual SPIKES but does NOT
        reset sub-threshold state, so without this reset two arms measured at
        different stepping-history depths (intact margins first, competition-lesion
        margins after four intervening recalls) are sampled from different states
        and a zero-weight window can spuriously shift ``min(M) > min(L)``.  The
        reset restores ONLY fast dynamical state -- learned weights and adapted
        thresholds are preserved -- so with noise off both arms differ ONLY in the
        competition (the 2026-08-06 v9 stepping-history confound).
        """

        self.reset_dynamical_state()
        settle = self._settle_to_quiescence()
        record = super().recall(
            episode_pattern,
            source_path_lesion=source_path_lesion,
            acc_lesion=acc_lesion,
        )
        record["settle"] = settle
        return record


def validate_calibration_seed(seed: int) -> int:
    """Open the two v6 calibration seeds, plus the implementation-smoke seed."""

    return validate_phase_seed(seed, "calibration")


def evaluate_calibration_seed(
    seed: int,
    config: SourceMonitorConfigV2 | None = None,
    *,
    phase: str = "calibration",
) -> dict:
    """Run one seed of ``phase`` and all preregistered controls.

    The mechanism, thresholds, and acceptance rule are IDENTICAL across phases;
    only the seed partition (validated by ``validate_phase_seed``) advances. The
    development and held-out phases reuse this frozen evaluator unchanged.
    """

    seed = validate_phase_seed(seed, phase)
    c = config or SourceMonitorConfigV2()
    patterns = make_episode_patterns(seed, 5, c)
    t0 = time.time()

    intact = SourceMonitorCoresidencyGateV6(seed=seed, config=c)
    initial = intact.weight_summary()
    intact.experience(patterns[0], visual_activity=True)
    intact.experience(patterns[1], auditory_activity=True)
    intact.experience(patterns[2], corollary_discharge=True)
    intact.experience(patterns[3], visual_activity=True, auditory_activity=True)
    learned = intact.weight_summary()
    seen = intact.recall(patterns[0])
    heard = intact.recall(patterns[1])
    self_generated = intact.recall(patterns[2])
    mixed = intact.recall(patterns[3])
    unseen = intact.recall(patterns[4])
    source_lesion = intact.recall(patterns[0], source_path_lesion=True)
    acc_lesion = intact.recall(patterns[0], acc_lesion=True)
    intact.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 0.0)
    try:
        competition_lesions = {
            "seen": intact.recall(patterns[0]),
            "heard": intact.recall(patterns[1]),
            "self_generated": intact.recall(patterns[2]),
        }
    finally:
        intact.bridge.set_transmission_gate(SOURCE_COMPETITION_GATE, 1.0)

    swapped = SourceMonitorCoresidencyGateV6(seed=seed + 10000, config=c)
    swapped.experience(patterns[0], auditory_activity=True)
    swapped.experience(patterns[1], visual_activity=True)
    swap_zero = swapped.recall(patterns[0])
    swap_one = swapped.recall(patterns[1])

    learning_off = SourceMonitorCoresidencyGateV6(seed=seed + 20000, config=c)
    off_initial = learning_off.weight_summary()
    learning_off.experience(patterns[0], visual_activity=True, learning_enabled=False)
    off_after = learning_off.weight_summary()
    off_recall = learning_off.recall(patterns[0])

    margins = {
        "seen": _source_margin(seen, "seen"),
        "heard": _source_margin(heard, "heard"),
        "self_generated": _source_margin(self_generated, "self_generated"),
    }
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
    competition_intact_spikes = float(sum(seen["competition_spikes"].values()))
    competition_lesion_spikes = float(
        sum(competition_lesions["seen"]["competition_spikes"].values())
    )
    lesion_margins = {
        source: _source_margin(competition_lesions[source], source) for source in SOURCES
    }
    competition_margin_gains = {
        source: margins[source] - lesion_margins[source] for source in SOURCES
    }
    # Bounded-loss, guard-the-floor, max-min acceptance (see the P3 functional-role
    # spec). loss_s = max(0, L_s - M_s); spendable_surplus_s = max(0, L_s - F).
    bounded_loss = {
        source: max(0.0, lesion_margins[source] - margins[source]) for source in SOURCES
    }
    spendable_surplus = {
        source: max(0.0, lesion_margins[source] - MIN_SOURCE_MARGIN) for source in SOURCES
    }
    bounded_loss_within_surplus = all(
        bounded_loss[source] <= spendable_surplus[source] for source in SOURCES
    )
    weakest_source_strictly_improved = min(margins.values()) > min(lesion_margins.values())
    components = {
        "learned_routes_start_zero": bool(initial["l1"] == 0.0),
        "experience_changes_synaptic_weights": bool(learned["l1"] > initial["l1"]),
        "seen_source_recalled": bool(_dominant_source(seen) == "seen" and margins["seen"] > 0.0),
        "heard_source_recalled": bool(_dominant_source(heard) == "heard" and margins["heard"] > 0.0),
        "self_source_recalled": bool(
            _dominant_source(self_generated) == "self_generated"
            and margins["self_generated"] > 0.0
        ),
        "all_source_margins_meet_fixed_floor": bool(min(margins.values()) >= MIN_SOURCE_MARGIN),
        "source_swap_follows_afferent_activity": bool(
            _dominant_source(swap_zero) == "heard" and _dominant_source(swap_one) == "seen"
        ),
        "mixed_source_reinstates_both": bool(
            mixed["source_spikes"]["seen"] > 0.0
            and mixed["source_spikes"]["heard"] > 0.0
        ),
        "source_path_lesion_collapses_recall": bool(
            lesioned_source_total <= 0.10 * max(intact_source_total, 1.0)
        ),
        "source_path_attribution_meets_fixed_floor": bool(
            source_path_fraction >= MIN_ATTRIBUTION_FRACTION
        ),
        "acc_lesion_preserves_source_and_silences_acc": bool(
            sum(acc_lesion["source_spikes"].values()) >= 0.90 * intact_source_total
            and acc_lesion["acc_spikes"] == 0.0
        ),
        "acc_path_attribution_meets_fixed_floor": bool(
            acc_path_fraction >= MIN_ATTRIBUTION_FRACTION
        ),
        "learning_off_keeps_weights_zero": bool(
            off_initial["l1"] == 0.0 and off_after["l1"] == 0.0
        ),
        "learning_off_has_no_source_recall": bool(sum(off_recall["source_spikes"].values()) == 0.0),
        "unseen_episode_has_no_source_recall": bool(sum(unseen["source_spikes"].values()) == 0.0),
        "source_spikes_reach_apfc_and_acc": bool(
            seen["apfc_source_spikes"]["seen"] > 0.0 and seen["acc_spikes"] > 0.0
        ),
        "competition_circuit_is_active_and_lesionable": bool(
            competition_intact_spikes > 0.0 and competition_lesion_spikes == 0.0
        ),
        "bounded_loss_only_spends_surplus": bool(bounded_loss_within_surplus),
        "weakest_source_margin_strictly_improved": bool(weakest_source_strictly_improved),
        "recall_settle_reaches_quiescence": bool(
            all(
                rec["settle"]["reached_quiescence"]
                for rec in (seen, heard, self_generated, mixed, unseen, off_recall)
            )
        ),
    }

    region_names = {region.name for region in intact.bridge.region_manager.regions()}
    expected_regions = {EPISODE_REGION, ACC_REGION}
    expected_regions.update(SOURCE_AFFERENT.values())
    expected_regions.update(SOURCE_MEMORY.values())
    expected_regions.update(APFC_SOURCE.values())
    expected_regions.update(SOURCE_INTERNEURON.values())
    recall_parameters = list(inspect.signature(SourceMonitorCoresidencyGateV6.recall).parameters)
    earned = Verdict(f"source-monitor co-residency v6 {phase}")
    earned.require(
        "episode, source, competition, aPFC, and ACC populations share one bridge",
        expected_regions.issubset(region_names),
        expect=True,
    )
    earned.require(
        "recall accepts episode activity without source metadata",
        recall_parameters == ["self", "episode_pattern", "source_path_lesion", "acc_lesion"],
        expect=True,
    )
    earned.require(
        "v2 preserves v1 source input strength",
        c.drive_pA == SourceMonitorConfig().drive_pA
        and c.source_afferent_weight == SourceMonitorConfig().source_afferent_weight,
        expect=True,
    )
    earned.require(
        "episode patterns are disjoint and fit the declared population",
        len(set().union(*(set(pattern.tolist()) for pattern in patterns)))
        == len(patterns) * c.episode_pattern_size,
        expect=True,
    )
    earned.knob(
        "Hebbian learning rate",
        requested=c.hebbian_learning_rate,
        applied=intact.bridge.core_config.hebbian_learning_rate,
    )
    earned.knob(
        "GABA-A rival inhibition",
        requested=c.interneuron_to_rival_weight,
        applied=c.interneuron_to_rival_weight,
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
    earned.reaches(
        "source-pool competition lesion",
        before=competition_intact_spikes,
        after=competition_lesion_spikes,
    )
    earned.disabled(
        "STDP, reward modulation, homeostasis, short-term plasticity, and structural plasticity",
        why="v6 isolates Hebbian source association plus fixed local GABA-A competition",
    )
    decided = earned.decide(go=all(components.values()), verbose=False)
    phase_label = phase.upper()
    status = (
        "UNDEFINED"
        if decided["status"] == UNDEFINED
        else f"{phase_label}_PASS" if decided["go"] else f"{phase_label}_FAIL"
    )
    return {
        "seed": seed,
        "phase": phase,
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
            "intact_source_spikes": intact_source_total,
            "source_lesion_spikes": lesioned_source_total,
            "intact_acc_spikes": float(seen["acc_spikes"]),
            "acc_lesion_spikes": float(acc_lesion["acc_spikes"]),
            "mixed_seen_spikes": float(mixed["source_spikes"]["seen"]),
            "mixed_heard_spikes": float(mixed["source_spikes"]["heard"]),
            "competition_intact_spikes": competition_intact_spikes,
            "competition_lesion_spikes": competition_lesion_spikes,
            "competition_margin_gains": competition_margin_gains,
            "competition_lesion_margins": lesion_margins,
            "bounded_loss": bounded_loss,
            "spendable_surplus": spendable_surplus,
            "minimum_lesion_margin": min(lesion_margins.values()),
            "learning_off_source_spikes": float(sum(off_recall["source_spikes"].values())),
            "learning_off_settle_steps": int(off_recall["settle"]["settle_steps"]),
        },
        "attribution": {
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
            "competition_lesions": competition_lesions,
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
            not in inspect.signature(SourceMonitorCoresidencyGateV6.recall).parameters,
            "episode_activity_is_caller_supplied": True,
            "host_gain_normalization": False,
            "stronger_source_specific_drive": False,
            "host_confidence_scalar": False,
            "host_response_decision": False,
            "recall_settle_is_zero_input": True,
        },
        "seed_policy": {
            "calibration": list(CALIBRATION_SEEDS),
            "development_reserved": list(DEVELOPMENT_SEEDS),
            "held_out": list(HELD_OUT_SEEDS),
        },
        "fixed_criteria": {
            "minimum_source_margin": MIN_SOURCE_MARGIN,
            "minimum_attribution_fraction": MIN_ATTRIBUTION_FRACTION,
            "unseen_source_spikes": 0,
            "learning_off_source_spikes": 0,
            "acceptance_rule": "bounded_loss_guard_floor_max_min",
            "loss_may_only_spend_surplus_above_floor": True,
            "weakest_source_margin_must_strictly_improve": True,
            "recall_settles_to_quiescence_before_read": True,
            "all_structural_and_lesion_components_required": True,
        },
        "honest_scope": (
            "V6 keeps v2's fixed local fast-spiking GABA-A competition and the bounded-loss max-min "
            "acceptance rule unchanged, and changes only the source-recall protocol: recall settles the "
            "substrate to quiescence at zero input before reading, so residual encoding-phase state cannot "
            "leak into the read and, with zero learned weights, the source populations are silent by "
            "construction. Sparse episode activity, physical source-afferent identity, the learning window, "
            "and the pre-read settle timing remain developmental scaffolds. Population spike counts and winners "
            "are host-read for evaluation only. The competition wiring is specified rather than self-organized; "
            "no language, confidence, or response policy is claimed."
        ),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one calibration seed for source-monitor co-residency v6."
    )
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEEDS[0])
    parser.add_argument(
        "--phase", choices=tuple(PHASE_SEEDS), default="calibration"
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    row = evaluate_calibration_seed(args.seed, phase=args.phase)
    print(
        "[source-monitor-coresidency-v6] "
        f"seed={row['seed']} phase={row['phase']} status={row['status']} "
        f"metrics={row['metrics']} components={row['components']}",
        flush=True,
    )
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(row, indent=2))
        print(f"[source-monitor-coresidency-v6] wrote {out_path}", flush=True)
    return 0 if row["status"].endswith("_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
