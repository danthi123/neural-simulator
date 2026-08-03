"""Gate intrinsic spiking vocal exploration and same-brain convention reversal.

The brain receives context, perception, one shared arousal signal, and social
consequences. It never receives a desired vocal channel. Independent spiking
competition produces exploratory actions; reward-US -> SNc dopamine bursts
strengthen successful routes, while negative feedback -> RMTg -> SNc dips
weaken unsuccessful routes.

This is a small preverbal learning rung. Passing it does not imply natural
language, open conversation, or human-like development.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sim.backend import get_backend
from tools.lab import attributable_to
from research.runners._developmental_vocal_convention_derisk import (
    ALL_CASES,
    HELD_OUT_CASES,
    TRAIN_CASES,
    VocalConvention,
    _accuracy,
    _build_agent,
    _connection_mask,
    _evaluate,
    _factor_accuracy,
    _release,
    _snapshot_trial_state,
    calibrate_snc_tonic,
    settle_after_training,
    train_by_consequence,
)
from research.runners.nav_conv_merged_bridge import (
    GEN_PERCEPTION,
    VOCAL_EXPLORATION_AROUSAL,
    VOCAL_EXPLORE_PREFIX,
    VOCAL_NEGATIVE_FEEDBACK,
    VOCAL_RMTG,
    VOCAL_SOCIAL_CUE,
    _developmental_vocal_regions_pathways,
)


DEFAULT_SEEDS = (42, 43, 44, 100, 101, 102)
INITIAL_TRIALS = 200
EXTINCTION_TRIALS = 120
REVERSAL_TRIALS = 240


def _train(
    agent,
    convention,
    trials,
    *,
    seed,
    mode="contingent",
    negative_feedback=True,
    negative_learning_scale=0.10,
    arousal_pA=700.0,
    yoked_schedule=None,
):
    return train_by_consequence(
        agent,
        convention,
        trials=trials,
        exploration_seed=int(seed),
        mode=mode,
        yoked_schedule=yoked_schedule,
        exploration_mode="intrinsic",
        negative_feedback=negative_feedback,
        snc_tonic_pA=220.0,
        intrinsic_action_steps=120,
        intrinsic_noise_pA=20.0,
        exploration_arousal_pA=arousal_pA,
        phasic_update_window=True,
        phasic_consolidation_steps=30,
        negative_feedback_steps=18,
        negative_consolidation_steps=5,
        negative_learning_scale=negative_learning_scale,
        vocal_weight_max=3.0,
        context_learning_gain=0.35,
        visual_learning_gain=4.0,
        intertrial_steps=60,
    )


def _evaluate_from_origin(agent, convention, origin):
    rows = _evaluate(agent, convention, origin)
    return {
        "joint_accuracy": _accuracy(rows),
        "intent_accuracy": _factor_accuracy(rows, "intent"),
        "referent_accuracy": _factor_accuracy(rows, "referent"),
        "held_out_accuracy": float(np.mean([
            row["listener"]["success"]
            for row in rows
            if tuple(row["evaluation_target"]) in HELD_OUT_CASES
        ])),
        "raw_actions": [row["neural"]["raw_action"] for row in rows],
        "rows": rows,
    }


def _feedback_metrics(*trainings):
    events = [event for training in trainings for event in training["events"]]
    rewarded = [event for event in events if event["reward_delivered"]]
    errors = [event for event in events if event["negative_feedback_delivered"]]
    return {
        "reward_events": len(rewarded),
        "error_events": len(errors),
        "peak_dopamine": max(
            (event["reward_trace"]["peak_dopamine"] for event in rewarded),
            default=0.5,
        ),
        "minimum_dopamine": min(
            (event["reward_trace"]["minimum_dopamine"] for event in errors),
            default=0.5,
        ),
        "reward_snc_spikes": sum(
            event["reward_trace"]["snc_spikes"] for event in rewarded
        ),
        "error_rmtg_spikes": sum(
            event["reward_trace"]["rmtg_spikes"] for event in errors
        ),
    }


def _exploration_metrics(*trainings):
    events = [event for training in trainings for event in training["events"]]
    raw_actions = [
        event["neural"]["raw_action"]
        for event in events
        if event["neural"]["raw_action"] is not None
    ]
    unique_actions = sorted({
        (action["intent_channel"], action["referent_channel"])
        for action in raw_actions
    })
    outcomes = {
        outcome: sum(event["neural"]["motor_outcome"] == outcome for event in events)
        for outcome in ("emitted", "silence", "timeout")
    }
    return {
        "recorded_events": len(events),
        "unique_emitted_actions": [list(action) for action in unique_actions],
        "motor_outcomes": outcomes,
        "all_actions_from_shared_arousal": all(
            event["neural"]["exploration_channels"] is None
            and not event["neural"]["injected_output_current"]
            for event in events
        ),
    }


def _anatomy_checks():
    _, pathways = _developmental_vocal_regions_pathways(
        intrinsic_exploration=True,
        error_feedback=True,
    )
    arousal_routes = [
        route for route in pathways
        if route.from_region == VOCAL_EXPLORATION_AROUSAL
        and route.to_region.startswith(VOCAL_EXPLORE_PREFIX)
    ]
    explore_targets = {route.to_region for route in arousal_routes}
    prohibited = [
        route for route in pathways
        if route.to_region in explore_targets
        and route.from_region in {"drive_agrp", VOCAL_SOCIAL_CUE, GEN_PERCEPTION}
    ]
    symmetric = bool(
        len(arousal_routes) == 6
        and all(
            route.density == 1.0
            and route.weight_mean == 24.0
            and route.weight_jitter == 0.0
            for route in arousal_routes
        )
    )
    error_route = any(
        route.from_region == VOCAL_NEGATIVE_FEEDBACK
        and route.to_region == VOCAL_RMTG
        for route in pathways
    )
    inhibitory_snc_route = any(
        route.from_region == VOCAL_RMTG
        and route.to_region == "limbic_snc"
        and route.receptor == "gaba_a"
        for route in pathways
    )
    return {
        "six_equal_shared_arousal_routes": symmetric,
        "no_context_or_perception_to_explore_route": not prohibited,
        "negative_feedback_reaches_rmtg": error_route,
        "rmtg_inhibits_snc": inhibitory_snc_route,
    }


def _zero_pathway(agent, source_name, target_name):
    bridge = agent._merged_bridge
    source = np.asarray(bridge.region_manager.indices(source_name), dtype=np.int64)
    target = np.asarray(bridge.region_manager.indices(target_name), dtype=np.int64)
    mask_x, mask_h = _connection_mask(bridge, source, target)
    bridge.cp_connections.data[mask_x] = 0.0
    return int(mask_h.sum())


def _run_acquisition_control(seed, *, mode, arousal_pA=700.0, yoked_schedule=None):
    identity = VocalConvention.identity()
    agent = _build_agent(seed, intrinsic_exploration=True, error_feedback=True)
    settle_after_training(agent, steps=300)
    calibration = calibrate_snc_tonic(agent, tonic_pA=220.0)
    training = _train(
        agent,
        identity,
        INITIAL_TRIALS,
        seed=seed,
        mode=mode,
        negative_feedback=mode in ("contingent", "da_lesion"),
        arousal_pA=arousal_pA,
        yoked_schedule=yoked_schedule,
    )
    settle_after_training(agent, steps=300)
    origin = _snapshot_trial_state(agent)
    evaluation = _evaluate_from_origin(agent, identity, origin)
    result = {
        "calibration": calibration,
        "training": training,
        "evaluation": evaluation,
    }
    del agent
    _release()
    return result


def _run_error_path_lesion_control(seed):
    identity = VocalConvention.identity()
    swapped = VocalConvention.swapped()
    agent = _build_agent(seed, intrinsic_exploration=True, error_feedback=True)
    settle_after_training(agent, steps=300)
    calibrate_snc_tonic(agent, tonic_pA=220.0)
    initial = _train(agent, identity, INITIAL_TRIALS, seed=seed)
    settle_after_training(agent, steps=300)
    initial_origin = _snapshot_trial_state(agent)
    initial_evaluation = _evaluate_from_origin(agent, identity, initial_origin)
    lesioned = (
        _zero_pathway(agent, VOCAL_NEGATIVE_FEEDBACK, VOCAL_RMTG)
        + _zero_pathway(agent, VOCAL_RMTG, "limbic_snc")
    )
    extinction = _train(
        agent,
        swapped,
        EXTINCTION_TRIALS,
        seed=seed,
        mode="negative_only",
        negative_learning_scale=0.25,
    )
    reversal = _train(agent, swapped, REVERSAL_TRIALS, seed=seed)
    settle_after_training(agent, steps=300)
    origin = _snapshot_trial_state(agent)
    evaluation = _evaluate_from_origin(agent, swapped, origin)
    result = {
        "lesioned_synapses": lesioned,
        "initial_training": initial,
        "initial_evaluation": initial_evaluation,
        "extinction_training": extinction,
        "reversal_training": reversal,
        "evaluation": evaluation,
    }
    del agent
    _release()
    return result


def run_seed(seed, *, full_controls=False, verbose=True):
    identity = VocalConvention.identity()
    swapped = VocalConvention.swapped()
    agent = _build_agent(seed, intrinsic_exploration=True, error_feedback=True)
    bridge_identity = id(agent._merged_bridge)
    settle_after_training(agent, steps=300)
    calibration = calibrate_snc_tonic(agent, tonic_pA=220.0)

    initial = _train(agent, identity, INITIAL_TRIALS, seed=seed)
    settle_after_training(agent, steps=300)
    initial_origin = _snapshot_trial_state(agent)
    initial_evaluation = _evaluate_from_origin(agent, identity, initial_origin)

    extinction = _train(
        agent,
        swapped,
        EXTINCTION_TRIALS,
        seed=seed,
        mode="negative_only",
        negative_learning_scale=0.25,
    )
    reversal = _train(agent, swapped, REVERSAL_TRIALS, seed=seed)
    settle_after_training(agent, steps=300)
    final_origin = _snapshot_trial_state(agent)
    reversal_evaluation = _evaluate_from_origin(agent, swapped, final_origin)
    old_convention_evaluation = _evaluate_from_origin(agent, identity, final_origin)

    feedback = _feedback_metrics(initial, extinction, reversal)
    exploration = _exploration_metrics(initial, extinction, reversal)
    anatomy = _anatomy_checks()
    outside_changes = [
        phase["outside_vocal_changed_synapses"]
        for phase in (initial, extinction, reversal)
    ]
    checks = {
        **anatomy,
        "one_shared_brain_through_reversal": id(agent._merged_bridge) == bridge_identity,
        "initial_convention_acquired": initial_evaluation["joint_accuracy"] == 1.0,
        "initial_held_out_composition": initial_evaluation["held_out_accuracy"] == 1.0,
        "reversed_convention_acquired": reversal_evaluation["joint_accuracy"] == 1.0,
        "reversed_held_out_composition": reversal_evaluation["held_out_accuracy"] == 1.0,
        "old_convention_no_longer_controls": old_convention_evaluation["joint_accuracy"] == 0.0,
        "all_four_raw_actions_explored": len(exploration["unique_emitted_actions"]) == 4,
        "no_channel_specific_exploration_or_output_current": exploration[
            "all_actions_from_shared_arousal"
        ],
        "dopamine_reward_burst": (
            feedback["reward_snc_spikes"] > 0 and feedback["peak_dopamine"] > 0.5
        ),
        "rmtg_error_dip": (
            feedback["error_rmtg_spikes"] > 0 and feedback["minimum_dopamine"] < 0.5
        ),
        "only_vocal_synapses_changed": all(count == 0 for count in outside_changes),
    }

    controls = {}
    if full_controls:
        main_reward_schedule = initial["reward_schedule"]
        del agent
        _release()
        controls = {
            "no_consequence": _run_acquisition_control(seed, mode="none"),
            "yoked_reward": _run_acquisition_control(
                seed,
                mode="yoked",
                yoked_schedule=main_reward_schedule,
            ),
            "dopamine_lesion": _run_acquisition_control(seed, mode="da_lesion"),
            "exploration_arousal_lesion": _run_acquisition_control(
                seed,
                mode="contingent",
                arousal_pA=0.0,
            ),
            "rmtg_error_path_lesion": _run_error_path_lesion_control(seed),
        }
        control_acc = {
            name: control["evaluation"]["joint_accuracy"]
            for name, control in controls.items()
        }
        attribution = {
            "acquisition_to_dopamine": attributable_to(
                "intrinsic vocal acquisition to dopamine",
                initial_evaluation["joint_accuracy"],
                control_acc["dopamine_lesion"],
                warn_below=0.75,
            ),
            "acquisition_to_exploration_arousal": attributable_to(
                "intrinsic vocal acquisition to exploration arousal",
                initial_evaluation["joint_accuracy"],
                control_acc["exploration_arousal_lesion"],
                warn_below=0.75,
            ),
            "reversal_to_rmtg_error_path": attributable_to(
                "same-brain vocal reversal to RMTg error path",
                reversal_evaluation["joint_accuracy"],
                control_acc["rmtg_error_path_lesion"],
                warn_below=0.50,
            ),
        }
        controls["attribution"] = attribution
        checks.update({
            "no_consequence_does_not_learn": control_acc["no_consequence"] <= 0.25,
            "yoked_reward_does_not_learn": control_acc["yoked_reward"] <= 0.50,
            "dopamine_lesion_does_not_learn": control_acc["dopamine_lesion"] <= 0.25,
            "exploration_lesion_blocks_learning": (
                control_acc["exploration_arousal_lesion"] <= 0.25
            ),
            "rmtg_lesion_blocks_reversal": control_acc["rmtg_error_path_lesion"] <= 0.50,
            "rmtg_control_acquires_before_lesion": (
                controls["rmtg_error_path_lesion"]["initial_evaluation"][
                    "joint_accuracy"
                ] == 1.0
            ),
            "causal_attribution_is_substantial": all(
                value is not None and value >= 0.50 for value in attribution.values()
            ),
        })
    else:
        del agent
        _release()

    row = {
        "seed": int(seed),
        "calibration": calibration,
        "initial_training": initial,
        "initial_evaluation": initial_evaluation,
        "extinction_training": extinction,
        "reversal_training": reversal,
        "reversal_evaluation": reversal_evaluation,
        "old_convention_evaluation": old_convention_evaluation,
        "feedback": feedback,
        "exploration": exploration,
        "controls": controls,
        "checks": checks,
        "go": bool(all(checks.values())),
    }
    if verbose:
        failed = [name for name, passed in checks.items() if not passed]
        print(
            f"[intrinsic-vocal seed={seed}] initial="
            f"{initial_evaluation['joint_accuracy']:.2f} reversed="
            f"{reversal_evaluation['joint_accuracy']:.2f} old="
            f"{old_convention_evaluation['joint_accuracy']:.2f} -> "
            f"{'GO' if row['go'] else 'NO-GO'}",
            flush=True,
        )
        if failed:
            print(f"  failed: {failed}", flush=True)
    return row


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance():
    xp, _ = get_backend()
    gpu = None
    if xp.__name__ == "cupy":
        props = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
        name = props["name"]
        gpu = name.decode() if isinstance(name, bytes) else str(name)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    sources = [
        Path(__file__).resolve(),
        _REPO / "research/runners/_developmental_vocal_convention_derisk.py",
        _REPO / "research/runners/nav_conv_merged_bridge.py",
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "source_sha256": {str(path.relative_to(_REPO)): _sha256(path) for path in sources},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": xp.__name__,
        "gpu": gpu,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--full-controls", action="store_true")
    parser.add_argument(
        "--out",
        default="research/findings/raw/developmental_vocal_intrinsic_reversal_6seed.json",
    )
    args = parser.parse_args()
    rows = [run_seed(seed, full_controls=args.full_controls) for seed in args.seeds]
    report = {
        "probe": "developmental_vocal_intrinsic_reversal",
        "scope": (
            "preverbal two-intent by two-referent convention learning with intrinsic "
            "spiking exploration and same-brain reversal; not natural language"
        ),
        "protocol": {
            "train_cases": [list(case) for case in TRAIN_CASES],
            "held_out_cases": [list(case) for case in HELD_OUT_CASES],
            "all_evaluation_cases": [list(case) for case in ALL_CASES],
            "initial_trials": INITIAL_TRIALS,
            "negative_only_extinction_trials": EXTINCTION_TRIALS,
            "reversal_trials": REVERSAL_TRIALS,
            "initial_convention": asdict(VocalConvention.identity()),
            "reversed_convention": asdict(VocalConvention.swapped()),
        },
        "seeds": args.seeds,
        "full_controls": bool(args.full_controls),
        "rows": rows,
        "n_go": int(sum(row["go"] for row in rows)),
        "all_go": bool(all(row["go"] for row in rows)),
        "provenance": _provenance(),
    }
    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[intrinsic-vocal] {report['n_go']}/{len(rows)} seeds GO -> {out}", flush=True)
    return 0 if report["all_go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
