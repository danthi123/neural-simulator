"""Bounded v5+SFA calibration: intrinsic spike-frequency-adaptation one-of-N eviction.

Built ON v5 (learned, encoding-potentiated CA1->cortex reinstatement,
``_replay_cortical_consolidation_gate_v5.py``). v5 established the CAPABILITY --
a learned, memory-specific CA1->cortical_target reinstatement pathway makes
replay consolidation causal AND hippocampus-independent at retest on BOTH
calibration seeds (the CLS signature) -- but was NO-GO at the 2-seed bar: seed
412 GO, seed 413 NO-GO on retest false recall (0.180 vs the 0.15 ceiling). The
v5 finding localised the cause to SHARED-cue-cell interference (``cue_overlap``
6 of 16): at the hippocampus-disabled retest, a partial cue for one memory
includes cue cells shared with the other, which drive the consolidated
cue->target association of BOTH memories and leak ~5 spikes to the wrong target
assembly. Point-neuron opponent fast-spiking competition reduces but does not
clear this on the harder seed -- the point-neuron competition limit.

The named surpass (2026-08-06 gate, and v5's own "Decision and next mechanism"):
INTRINSIC spike-frequency-adaptation-driven one-of-N eviction on the cortical
target attractor. During a probe the correct target assembly is strongly and
recurrently driven; the interfering (wrong) assembly is only weakly driven by
the few shared cue cells and has no recurrent support. Spike-frequency
adaptation (the Izhikevich recovery variable u, incremented by ``d`` on every
spike; Dehaene-Changeux 2011 metastability / Ecker 2022 adaptation-driven
transitions) accumulates faster on a persistently-firing neuron than on a
transiently-driven one, so it silences the weak, unsupported false assembly
while the recurrently-sustained correct assembly rides through it -- a
one-of-N eviction that sharpens the biased competition the opponent FS pool
already starts. SFA is INTRINSIC per-neuron biology, not a host computation:
it is realised through the substrate's own ``cp_izh_d_increment`` / ``cp_izh_a``
on the cortical_target slice (RS default d_increment=100, a=0.03; precedent
``_gap5_intrinsic_fatigue_replay_derisk.py`` and ``_affect_eviction_derisk.py``
-- a transmission gate CANNOT reach an intrinsic mechanism, so eviction is
controlled by writing the per-neuron parameters and lesioned by restoring them).

Structural change vs v5 (v5 otherwise inherited whole):
* the cortical_target neurons carry stronger intrinsic spike-frequency
  adaptation (``target_sfa_d_increment`` / ``target_sfa_a``), applied to the
  attractor in EVERY phase (adaptation is intrinsic, always on);
* a new ``target_sfa_lesion`` control restores the RS-default adaptation on the
  SAME neurons/wiring, so the false-recall reduction can be attributed to the
  SFA eviction rather than to anything else -- the sAHP power control that the
  affect arc found was missing (no transmission gate reaches intrinsic sAHP).

Every v5 control and gating criterion is inherited UNCHANGED (no_sleep,
shuffled_replay_order, shuffled_target_index, ca3_ca1_lesion,
cortical_plasticity_off, target_inhibition_lesion,
ca1_target_reinstatement_lesion, memory-selectivity, and the frozen
``false_recall_bounded <= 0.15``). ``target_sfa_lesion`` is ADDITIVE: it does
not weaken any threshold. The reward is BOTH seeds 412 and 413 passing the
frozen gate, i.e. seed-413 false recall below 0.15 without harming seed 412 or
the consolidation/lesion signatures.

Fresh seed partition (disjoint, inherited from v5): calibration 412/413, smoke
416, development 414/415/410 and held-out 417/418/419 stay mechanically
rejected until calibration lands a clean verdict.

CPU smoke:
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v5_sfa --smoke

Calibration:
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_cortical_consolidation_gate_v5_sfa \
        --seeds 412 413 --out research/findings/raw/replay_v5_sfa/replay_v5_sfa_calibration.json
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
from research.runners import _replay_cortical_consolidation_gate_v5 as v5  # noqa: E402
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import UNDEFINED, Verdict  # noqa: E402


CALIBRATION_SEEDS = v5.CALIBRATION_SEEDS
DEVELOPMENT_SEEDS = v5.DEVELOPMENT_SEEDS
HELD_OUT_SEEDS = v5.HELD_OUT_SEEDS
SMOKE_SEED = v5.SMOKE_SEED
# v5's eight conditions, plus the additive SFA power control.
CONDITIONS = v5.CONDITIONS + ("target_sfa_lesion",)

CA3_GATE = v5.CA3_GATE
INDEX_CUE_GATE = v5.INDEX_CUE_GATE
INDEX_TARGET_GATE = v5.INDEX_TARGET_GATE
CORTICAL_GATE = v5.CORTICAL_GATE


@dataclass(frozen=True)
class GateConfig(v5.GateConfig):
    """v5 anatomy/timing/reinstatement + intrinsic SFA on the cortical target attractor."""

    # Intrinsic spike-frequency adaptation on the cortical_target neurons.
    # RS default is d_increment=100.0, a=0.03 (sim/enums.py). Stronger d and a
    # slower a give the sustained sAHP that evicts a weakly-driven false
    # assembly while a recurrently-supported correct assembly rides through.
    # None => leave the substrate default (i.e. no eviction, == v5).
    target_sfa_d_increment: float | None = 120.0
    target_sfa_a: float | None = 0.02


def smoke_config() -> GateConfig:
    base = v5.smoke_config()
    return GateConfig(**asdict(base))


def validate_calibration_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    checked = tuple(int(seed) for seed in seeds)
    invalid = [
        seed for seed in checked if seed not in CALIBRATION_SEEDS and seed != SMOKE_SEED
    ]
    if invalid:
        raise ValueError(
            f"This bounded v5+SFA runner accepts calibration seeds {CALIBRATION_SEEDS} "
            f"(or smoke seed {SMOKE_SEED}) only; refusing reserved seeds {invalid}."
        )
    if not checked:
        raise ValueError("At least one calibration seed is required.")
    return checked


def _apply_target_sfa(bridge, handles: dict, config: GateConfig, *, enabled: bool) -> dict:
    """Write intrinsic spike-frequency adaptation onto the cortical_target slice.

    Brain-based: this is the substrate's own per-neuron Izhikevich adaptation
    (u += d on spike, u relaxes at rate a), NOT a host computation. The
    ``target_sfa_lesion`` condition passes ``enabled=False`` so the SAME neurons
    keep the RS default -- the sAHP power control the affect arc found missing.
    """
    from sim.backend import get_backend, to_host

    xp, _ = get_backend()
    target = np.asarray(handles["regions"]["cortical_target"], dtype=np.int64)
    d_arr = getattr(bridge, "cp_izh_d_increment", None)
    a_arr = getattr(bridge, "cp_izh_a", None)
    if d_arr is None or a_arr is None:
        return {
            "applied": False,
            "reason": "substrate exposes no cp_izh_d_increment/cp_izh_a",
            "n_target": int(target.size),
        }
    tdev = xp.asarray(target, dtype=xp.int64)
    d_before = float(np.asarray(to_host(d_arr[tdev])).mean())
    a_before = float(np.asarray(to_host(a_arr[tdev])).mean())
    report = {
        "applied": bool(enabled and config.target_sfa_d_increment is not None),
        "n_target": int(target.size),
        "d_before": d_before,
        "a_before": a_before,
    }
    if enabled and config.target_sfa_d_increment is not None:
        d_val = float(config.target_sfa_d_increment)
        d_arr[tdev] = xp.float32(d_val)
        report["d_after"] = d_val
        if config.target_sfa_a is not None:
            a_val = float(config.target_sfa_a)
            a_arr[tdev] = xp.float32(a_val)
            report["a_after"] = a_val
        else:
            report["a_after"] = a_before
    else:
        report["d_after"] = d_before
        report["a_after"] = a_before
        report["reason"] = (
            "target_sfa_lesion: RS-default adaptation restored on the target attractor"
            if not enabled
            else "target_sfa_d_increment is None (== v5, no eviction)"
        )
    return report


def run_condition(seed: int, condition: str, config: GateConfig | None = None) -> dict:
    """v5.run_condition's phase sequence, with intrinsic SFA applied to the target."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}.")
    validate_calibration_seeds([seed])
    cfg = config or GateConfig()
    bridge, handles = v5.build_bridge(seed, cfg)

    # Intrinsic SFA eviction on the cortical target attractor. ON for every
    # condition (adaptation is intrinsic) EXCEPT the dedicated power control.
    sfa_report = _apply_target_sfa(
        bridge, handles, cfg, enabled=(condition != "target_sfa_lesion")
    )

    bridge_ids = [id(bridge)]
    phase_trace: list[str] = []

    # Wake teaching first: opponent target inhibition must not suppress the
    # externally presented target while the hippocampal index is being learned.
    bridge.set_transmission_gate(v5.TARGET_INHIBITION_GATE, 0.0)

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
    sleep = v5._sleep(bridge, handles, condition, seed, cfg)
    phase_trace.append("sleep")
    bridge_ids.append(id(bridge))
    after_sleep = {
        "ca3": v1._path_weights(bridge, CA3_GATE),
        "index_cue": v1._path_weights(bridge, INDEX_CUE_GATE),
        "index_target": v1._path_weights(bridge, INDEX_TARGET_GATE),
        "cortical": v1._path_weights(bridge, CORTICAL_GATE),
    }

    # Retest with hippocampus disabled: CA1 silent (Schaffer off) and the
    # reinstatement wire off, so recall can only come from the consolidated
    # intracortical cue->target association -- now sharpened by SFA eviction.
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
    """v5's frozen verdict, plus an ADDITIVE (non-gating) SFA-eviction diagnostic."""
    verdict = v5._calibration_verdict(conditions)
    intact_false = verdict["intact_mean_false_recall"]
    if "target_sfa_lesion" in conditions:
        lesion_false = v5.v2._mean_false_recall(conditions["target_sfa_lesion"])
        verdict["sfa_lesion_mean_false_recall"] = lesion_false
        verdict["sfa_false_recall_reduction"] = lesion_false - intact_false
        verdict["sfa_reduces_false_recall"] = bool(intact_false < lesion_false)
        # Whose is the false-recall change? treatment = intact (SFA on),
        # control = target_sfa_lesion (SFA restored to RS default). A negative
        # fraction reads correctly as "the manipulation REDUCED the effect"
        # (retest false recall), attributing the reduction to SFA eviction and
        # not to anything running identically in both arms.
        verdict["sfa_false_recall_attribution"] = attributable_to(
            "SFA one-of-N eviction on retest false recall", intact_false, lesion_false
        )
        verdict["target_sfa"] = conditions["intact"]["target_sfa"]
        verdict["target_sfa_lesion_report"] = conditions["target_sfa_lesion"]["target_sfa"]
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
        "gate": "replay_cortical_consolidation_v5_sfa",
        "phase": "calibration",
        "mechanism": "v5 learned CA1->cortex reinstatement + intrinsic SFA one-of-N eviction on the target attractor",
        "calibration_status": aggregate_status,
        "seeds": list(checked),
        "reserved_seeds_inspected": False,
        "rows": rows,
        "remaining_scaffolds": [
            "host-defined wake episode populations and partial probe cues",
            "opponent inhibitory channel membership fixed from calibration assemblies",
            "host-scheduled sleep down-state boundaries and episode-agnostic CA3 background current",
            "host spike/weight measurement against known calibration assemblies",
            "rate-window Hebbian plasticity and fixed assembly anatomy",
            "SFA parameters (d_increment/a) set on the target slice at build, not developmentally tuned",
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
