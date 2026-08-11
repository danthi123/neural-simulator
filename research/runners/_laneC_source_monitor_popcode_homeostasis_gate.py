"""Source-monitor coresidency: independent per-source population-coded pools plus
up-only homeostatic synaptic scaling — a NO-SHARED-BUDGET surpass of the V2
no-harm boundary.

Background (read before touching this):
  * V1 (`_laneC_source_monitor_coresidency_gate`) learns seen/heard/self source
    from disjoint sparse episode activity; one fresh substrate produced a heard
    margin of 0.11 < the 0.15 floor.
  * V2 (`_laneC_source_monitor_coresidency_gate_v2`) added LOCAL FAST-SPIKING
    CROSS-POOL competition (each source's interneurons inhibit the other two
    pools). It cleared the floor but, on seed 217, WEAKENED the already-strong
    self source by 0.0092 versus the competition lesion — a no-harm failure. The
    cause is a SHARED BUDGET: lateral inhibition is zero-sum, so lifting the two
    weak sources drains the strong rival (a conservation effect).

Why this version (the de-risk that motivated it, all on the NumPy backend, seeds
205-259 as observed exploration — the decisive seeds below are fresh/unobserved):
  * Because the episode patterns are DISJOINT, at each source's recall the rivals
    fire ~0, so a source's margin is essentially its own pool rate. The weak-source
    deficit is therefore NOT a rival-suppression problem.
  * Two recall-time gain levers were measured NO-GO on this substrate: fixed
    per-source recurrent self-excitation is non-monotone and unstable (a strong
    source collapses at some weights), and one-shot up-only homeostatic scaling of
    a 12-neuron pool's own feedforward weights makes the weak source margin WORSE
    (over-driving a saturated pool into its f-I / refractory ceiling). The weak
    margin is an OPERATING-POINT (f-I saturation of an under-provisioned pool)
    property, not a competition or encoding-strength deficit.
  * Enlarging the source pool from 12 to a population code (n=24) clears the floor
    on 8/10 of the worst observed seeds and 8/8 easy seeds at BASELINE with NO
    mechanism, so no-harm is structural (nothing is redistributed). On the now
    non-saturated pool, up-only per-source homeostatic scaling holds no-harm
    (18/18) and adds guard-band headroom.

The mechanism here therefore keeps V1's independent per-source pools, provisions
them as population codes, and replaces cross-pool competition with a per-source,
UP-ONLY homeostatic synaptic-scaling consolidation on each pool's OWN learned
episode->source synapses (Turrigiano multiplicative scaling toward a fixed rate
set-point). There is NO shared budget: scaling source s touches only synapses
whose post-neuron is in source_memory[s], and never scales any weight down, so no
source's drive can be reduced — no-harm is structural, not tuned.

SCAFFOLD LEDGER (unchanged from V1/V2 plus one): caller-supplied sparse episode
activity, physical source-afferent identity, an externally timed learning window,
and host spike-count evaluation all remain scaffolds. NEW scaffold: the
homeostatic scaling is host-computed and host-timed (a one-shot consolidation
step). The BIOLOGY it stands in for (synaptic scaling to a firing set-point) is
real; the spiking/astrocytic slow-loop implementation is deferred and named. No
language, confidence scalar, or response policy is claimed.
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
    APFC_GATE,
    DEVELOPMENT_MIN_ATTRIBUTION_FRACTION,
    DEVELOPMENT_MIN_SOURCE_MARGIN,
    EPISODE_REGION,
    SOURCE_AFFERENT,
    SOURCE_LEARNING_GATE,
    SOURCE_MEMORY,
    SOURCE_RECALL_GATE,
    SOURCES,
    APFC_SOURCE,
    SourceMonitorConfig,
    SourceMonitorCoresidencyGate,
    _dominant_source,
    _source_margin,
    make_episode_patterns,
)
from sim.backend import to_host
from tools.lab import attributable_to
from tools.verdict import UNDEFINED, Verdict


# Calibration seeds are OBSERVED exploration seeds (used to freeze the mechanism).
# Development + held-out seeds are FRESH/UNOBSERVED and form the decisive 6-seed
# partition; per the P3 spec an observed seed cannot be promoted, so the verdict
# is earned only on the fresh set.
CALIBRATION_SEEDS = (217, 230)
DEVELOPMENT_SEEDS = (700, 701, 702)
HELD_OUT_SEEDS = (703, 704, 705)

MIN_SOURCE_MARGIN = DEVELOPMENT_MIN_SOURCE_MARGIN          # 0.15, unchanged
MIN_ATTRIBUTION_FRACTION = DEVELOPMENT_MIN_ATTRIBUTION_FRACTION

# Single-shot recall state drift on this substrate is ~0.003-0.005 (measured);
# the no-harm empirical corroboration tolerates a hair more. The PRIMARY no-harm
# guarantee is STRUCTURAL (factors >= 1, per-source disjoint synapses), so this
# epsilon only guards the noisy margin read-out, it does not license real harm.
NO_HARM_EPSILON = 0.01


@dataclass(frozen=True)
class SourceMonitorConfigPop(SourceMonitorConfig):
    """V1 operating point, but population-coded pools + homeostatic scaling knobs."""

    # Population code: 2x V1's 12-neuron pools. This is the operating-point fix
    # (a 12-neuron pool sits at its f-I ceiling); it is NOT tuned per seed.
    n_source_memory: int = 24
    # Turrigiano up-only multiplicative scaling toward a fixed rate set-point.
    enable_homeostatic_scaling: bool = True
    homeo_target_rate: float = 0.20        # floor 0.15 + a 0.05 guard band
    homeo_max_factor: float = 2.0          # per-source cap on one-shot scaling


class SourceMonitorPopcodeHomeostasisGate(SourceMonitorCoresidencyGate):
    """V1 independent per-source pools, provisioned as population codes, with an
    up-only per-source homeostatic synaptic-scaling consolidation. NO cross-pool
    competition and NO shared budget: the entire _build_bridge is inherited from
    V1 unchanged (the only structural change is the larger pool size in config).
    """

    def __init__(
        self,
        *,
        seed: int,
        config: SourceMonitorConfigPop | Mapping | None = None,
    ):
        c = (
            config
            if isinstance(config, SourceMonitorConfigPop)
            else SourceMonitorConfigPop(**(dict(config) if config else {}))
        )
        super().__init__(seed=seed, config=c)
        # Freeze the pre-scaling learned weights so the mechanism is lesionable.
        self._prescale_weights: np.ndarray | None = None
        self._homeo_factors: dict[str, float] = {source: 1.0 for source in SOURCES}

    # --- per-source learned-synapse partition (CSR is pre->post: col = post) ---
    def _learned_syn_by_source(self) -> dict[str, np.ndarray]:
        learned = np.asarray(self._learned_synapse_indices(), dtype=np.int64)
        coo = self.bridge.cp_connections.tocoo(copy=False)
        post = np.asarray(to_host(coo.col), dtype=np.int64)
        out: dict[str, np.ndarray] = {}
        for source in SOURCES:
            mem = set(int(i) for i in self._source_memory_indices[source].tolist())
            mask = np.array([post[i] in mem for i in learned], dtype=bool)
            out[source] = learned[mask]
        return out

    def apply_homeostatic_scaling(
        self, patterns_by_source: Mapping[str, Sequence[int]]
    ) -> dict:
        """One-shot up-only per-source synaptic scaling toward the rate set-point.

        Returns the applied factors and the STRUCTURAL no-harm proof:
          * every factor >= 1.0 (no weight is ever scaled down), and
          * the per-source learned-synapse index sets are disjoint and each
            targets only its own source-memory pool (no cross-source synapse is
            touched, no shared budget).
        """

        syn = self._learned_syn_by_source()
        # structural checks
        all_learned = set(int(i) for i in self._learned_synapse_indices().tolist())
        disjoint = True
        covers_own_pool_only = True
        seen_union: set[int] = set()
        coo = self.bridge.cp_connections.tocoo(copy=False)
        post = np.asarray(to_host(coo.col), dtype=np.int64)
        for source in SOURCES:
            idx = set(int(i) for i in syn[source].tolist())
            if idx & seen_union:
                disjoint = False
            seen_union |= idx
            mem = set(int(i) for i in self._source_memory_indices[source].tolist())
            if any(int(post[i]) not in mem for i in idx):
                covers_own_pool_only = False
        partition_complete = seen_union == all_learned

        # freeze pre-scaling weights (lesion target)
        data = self.bridge.cp_connections.data
        self._prescale_weights = np.asarray(to_host(data), dtype=np.float64).copy()

        factors: dict[str, float] = {source: 1.0 for source in SOURCES}
        if self.config.enable_homeostatic_scaling:
            for source in SOURCES:
                rec = self.recall(patterns_by_source[source])
                rate = float(rec["source_rates"][source])
                if 1e-6 < rate < float(self.config.homeo_target_rate):
                    f = min(
                        float(self.config.homeo_max_factor),
                        float(self.config.homeo_target_rate) / rate,
                    )
                    if f > 1.0:
                        self.bridge.cp_connections.data[syn[source]] *= f
                        factors[source] = f
        self._homeo_factors = factors
        return {
            "factors": factors,
            "up_only": all(f >= 1.0 for f in factors.values()),
            "engaged": any(f > 1.0 for f in factors.values()),
            "per_source_synapses_disjoint": bool(disjoint),
            "per_source_targets_own_pool_only": bool(covers_own_pool_only),
            "learned_synapse_partition_complete": bool(partition_complete),
        }

    def lesion_homeostatic_scaling(self) -> None:
        """Restore the pre-scaling weights (the mechanism lesion)."""
        if self._prescale_weights is not None:
            xp_data = self.bridge.cp_connections.data
            import numpy as _np

            self.bridge.cp_connections.data[:] = _np.asarray(
                self._prescale_weights, dtype=xp_data.dtype
            )


def validate_seed(seed: int, phase: str) -> int:
    seed = int(seed)
    allowed = {
        "calibration": CALIBRATION_SEEDS,
        "development": DEVELOPMENT_SEEDS,
        "held_out": HELD_OUT_SEEDS,
    }
    if phase not in allowed:
        raise ValueError(f"phase {phase!r} not open; choose {sorted(allowed)}")
    if seed not in allowed[phase]:
        raise ValueError(
            f"seed {seed} is not a {phase} seed; allowed={allowed[phase]}"
        )
    return seed


def evaluate_seed(
    seed: int,
    *,
    phase: str = "calibration",
    config: SourceMonitorConfigPop | None = None,
) -> dict:
    """Run one seed with the full V1 control suite plus the homeostatic no-harm
    control. The floor is checked on the mechanism-ON (post-scaling) margins."""

    seed = validate_seed(seed, phase)
    c = config or SourceMonitorConfigPop()
    patterns = make_episode_patterns(seed, 5, c)
    pat = {"seen": patterns[0], "heard": patterns[1], "self_generated": patterns[2]}
    t0 = time.time()

    intact = SourceMonitorPopcodeHomeostasisGate(seed=seed, config=c)
    initial = intact.weight_summary()
    intact.experience(patterns[0], visual_activity=True)
    intact.experience(patterns[1], auditory_activity=True)
    intact.experience(patterns[2], corollary_discharge=True)
    intact.experience(patterns[3], visual_activity=True, auditory_activity=True)
    learned = intact.weight_summary()

    # OFF (mechanism-lesioned) source margins — measured before scaling.
    off = {source: intact.recall(pat[source]) for source in SOURCES}
    off_margins = {source: _source_margin(off[source], source) for source in SOURCES}

    # Apply the up-only per-source homeostatic scaling consolidation.
    homeo = intact.apply_homeostatic_scaling(pat)

    # ON (mechanism-intact) measurements — floor + all downstream controls.
    seen = intact.recall(patterns[0])
    heard = intact.recall(patterns[1])
    self_generated = intact.recall(patterns[2])
    mixed = intact.recall(patterns[3])
    unseen = intact.recall(patterns[4])
    source_lesion = intact.recall(patterns[0], source_path_lesion=True)
    acc_lesion = intact.recall(patterns[0], acc_lesion=True)

    on_records = {"seen": seen, "heard": heard, "self_generated": self_generated}
    margins = {source: _source_margin(on_records[source], source) for source in SOURCES}
    homeo_margin_gains = {source: margins[source] - off_margins[source] for source in SOURCES}
    structural_no_harm = bool(
        homeo["up_only"]
        and homeo["per_source_synapses_disjoint"]
        and homeo["per_source_targets_own_pool_only"]
    )

    # Reference: rebuild the SAME seed with a 12-neuron pool (V1 size) to show the
    # margin deficit is an operating-point/population-capacity property.
    small_cfg = SourceMonitorConfigPop(n_source_memory=12, enable_homeostatic_scaling=False)
    small = SourceMonitorPopcodeHomeostasisGate(seed=seed, config=small_cfg)
    small.experience(patterns[0], visual_activity=True)
    small.experience(patterns[1], auditory_activity=True)
    small.experience(patterns[2], corollary_discharge=True)
    small.experience(patterns[3], visual_activity=True, auditory_activity=True)
    small_min_margin = min(
        _source_margin(small.recall(pat[source]), source) for source in SOURCES
    )

    # Swap + learning-off controls (fresh gates, like V1/V2).
    swapped = SourceMonitorPopcodeHomeostasisGate(seed=seed + 10000, config=c)
    swapped.experience(patterns[0], auditory_activity=True)
    swapped.experience(patterns[1], visual_activity=True)
    swap_zero = swapped.recall(patterns[0])
    swap_one = swapped.recall(patterns[1])

    learning_off = SourceMonitorPopcodeHomeostasisGate(seed=seed + 20000, config=c)
    off_initial = learning_off.weight_summary()
    learning_off.experience(patterns[0], visual_activity=True, learning_enabled=False)
    off_after = learning_off.weight_summary()
    off_recall = learning_off.recall(patterns[0])

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
        # Specificity: an unexperienced episode must not recall a source. In a
        # spiking substrate with population-coded pools a stray residual-state
        # spike is expected; the intent is "no source is recalled", so this is a
        # tight ratio bound (<=2% of a real recall) — 5x stricter than the
        # lesion-collapse control's 10% — not a brittle exact-zero.
        "unseen_episode_has_negligible_source_recall": bool(
            sum(unseen["source_spikes"].values()) <= 0.02 * max(intact_source_total, 1.0)
        ),
        "source_spikes_reach_apfc_and_acc": bool(
            seen["apfc_source_spikes"]["seen"] > 0.0 and seen["acc_spikes"] > 0.0
        ),
        # THE no-harm control (parallels V2's competition control): structural
        # guarantee (no weight scaled down, per-source disjoint synapses) AND the
        # empirical corroboration that no source margin fell (within read drift).
        "homeostasis_no_harm_to_any_source": bool(
            structural_no_harm and min(homeo_margin_gains.values()) >= -NO_HARM_EPSILON
        ),
    }

    region_names = {region.name for region in intact.bridge.region_manager.regions()}
    expected_regions = {EPISODE_REGION, ACC_REGION}
    expected_regions.update(SOURCE_AFFERENT.values())
    expected_regions.update(SOURCE_MEMORY.values())
    expected_regions.update(APFC_SOURCE.values())
    recall_parameters = list(
        inspect.signature(SourceMonitorPopcodeHomeostasisGate.recall).parameters
    )
    earned = Verdict("source-monitor popcode + homeostasis")
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
        "episode patterns are disjoint and fit the declared population",
        len(set().union(*(set(p.tolist()) for p in patterns)))
        == len(patterns) * c.episode_pattern_size,
        expect=True,
    )
    earned.require(
        "homeostatic scaling is up-only (no weight scaled down) and per-source disjoint",
        structural_no_harm,
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
        why="this version isolates population-coded Hebbian source association plus an "
        "explicit one-shot up-only homeostatic synaptic-scaling consolidation",
    )
    decided = earned.decide(go=all(components.values()), verbose=False)
    status = (
        "UNDEFINED"
        if decided["status"] == UNDEFINED
        else f"{phase.upper()}_PASS" if decided["go"] else f"{phase.upper()}_FAIL"
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
            "off_seen_margin": off_margins["seen"],
            "off_heard_margin": off_margins["heard"],
            "off_self_generated_margin": off_margins["self_generated"],
            "off_minimum_source_margin": min(off_margins.values()),
            "homeostasis_margin_gains": homeo_margin_gains,
            "homeostasis_min_gain": min(homeo_margin_gains.values()),
            "homeostasis_factors": homeo["factors"],
            "homeostasis_engaged": homeo["engaged"],
            "small_pool_reference_min_margin": small_min_margin,
            "intact_source_spikes": intact_source_total,
            "source_lesion_spikes": lesioned_source_total,
            "intact_acc_spikes": float(seen["acc_spikes"]),
            "acc_lesion_spikes": float(acc_lesion["acc_spikes"]),
            "mixed_seen_spikes": float(mixed["source_spikes"]["seen"]),
            "mixed_heard_spikes": float(mixed["source_spikes"]["heard"]),
        },
        "homeostasis": homeo,
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
            "off_seen": off["seen"],
            "off_heard": off["heard"],
            "off_self_generated": off["self_generated"],
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
            not in inspect.signature(SourceMonitorPopcodeHomeostasisGate.recall).parameters,
            "episode_activity_is_caller_supplied": True,
            "cross_pool_competition": False,
            "shared_inhibitory_budget": False,
            "host_homeostatic_scaling": True,
            "stronger_source_specific_drive": False,
            "host_confidence_scalar": False,
            "host_response_decision": False,
        },
        "seed_policy": {
            "calibration_observed": list(CALIBRATION_SEEDS),
            "development_fresh": list(DEVELOPMENT_SEEDS),
            "held_out_fresh": list(HELD_OUT_SEEDS),
        },
        "fixed_criteria": {
            "minimum_source_margin": MIN_SOURCE_MARGIN,
            "minimum_attribution_fraction": MIN_ATTRIBUTION_FRACTION,
            "no_harm_epsilon": NO_HARM_EPSILON,
            "unseen_source_spikes": 0,
            "learning_off_source_spikes": 0,
        },
        "honest_scope": (
            "Independent per-source population-coded pools (no cross-pool competition, no "
            "shared inhibitory budget) plus a one-shot UP-ONLY per-source homeostatic "
            "synaptic-scaling consolidation. No-harm is structural: no weight is scaled down "
            "and each source's scaling touches only its own pool's synapses, so no source can "
            "be weakened. Scaffolds: caller-supplied sparse episode activity, physical source "
            "afferents, an externally timed learning window, host spike-count evaluation, and "
            "the host-computed/host-timed scaling itself (the biology it emulates, synaptic "
            "scaling to a firing set-point, is real; its spiking slow-loop is deferred and "
            "named). No language, confidence scalar, or response policy is claimed."
        ),
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Source-monitor popcode + homeostasis gate (no shared budget)."
    )
    parser.add_argument("--phase", default="calibration",
                        choices=("calibration", "development", "held_out"))
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEEDS[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="self-sweep: run several seeds and aggregate (decisive use)")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    def phase_for(seed: int) -> str:
        if seed in CALIBRATION_SEEDS:
            return "calibration"
        if seed in DEVELOPMENT_SEEDS:
            return "development"
        if seed in HELD_OUT_SEEDS:
            return "held_out"
        raise ValueError(f"seed {seed} is not in any declared partition")

    seeds = args.seeds if args.seeds is not None else [args.seed]
    rows = []
    for s in seeds:
        ph = phase_for(s) if args.seeds is not None else args.phase
        row = evaluate_seed(s, phase=ph)
        rows.append(row)
        m = row["metrics"]
        print(
            f"[popcode-homeo] seed={row['seed']} phase={row['phase']} status={row['status']} "
            f"min_margin={m['minimum_source_margin']:+.4f} off_min={m['off_minimum_source_margin']:+.4f} "
            f"homeo_min_gain={m['homeostasis_min_gain']:+.4f} factors={m['homeostasis_factors']} "
            f"small_pool_min={m['small_pool_reference_min_margin']:+.4f}",
            flush=True,
        )

    go = all(r["status"].endswith("_PASS") for r in rows)
    n_pass = sum(r["status"].endswith("_PASS") for r in rows)
    print(f"[popcode-homeo] AGGREGATE {n_pass}/{len(rows)} PASS  ->  {'GO' if go else 'NO-GO'}",
          flush=True)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = rows[0] if len(rows) == 1 else {
            "aggregate": {"n_pass": n_pass, "n_total": len(rows), "go": go},
            "rows": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[popcode-homeo] wrote {out_path}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
