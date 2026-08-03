"""Minimal grounded speech-action loop on one persistent spiking bridge.

This runner joins mechanisms that previously worked only beside one another:

    visual scene -> learned food cue + hunger -> request action
    request action -> social/world consequence -> satiety -> silence

The world is allowed to render sensory and interoceptive input, decode a motor-
like speech output, and deliver food.  It never receives the intended decision,
the expected answer, or a host-computed hunger score.  The decision is the
request-versus-silence spike race on the same ``SimulationBridge`` as the
learned visual convergence and AgRP/POMC drive.

Honest first-rung scope:

* the developmental caregiver explicitly pairs the apple percept with one
  food-cue population while a local Hebbian pathway learns the association;
* the preverbal output decoder has one fixed meaning: ``request apple``;
* need/cue convergence weights and the speech motor decoder are hand-designed;
* no text generator is involved.  A WKV renderer belongs only after this
  conceptual causal loop works.

The controls require the behavior to disappear when hunger, perception, the
learned visual route, or learning itself is removed.  If the social consequence
is removed, the unchanged hungry brain must request again.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import get_backend, to_host
from research.runners.navigate_to_compose_then_answer import (
    GEN_PERC_DRIVE_PA,
    OBJECT_WORDS,
)
from research.runners.nav_conv_merged_bridge import (
    MergedNavConvAgent,
    SPEECH_FOOD_CUE,
    SPEECH_GROUNDING_GATE,
    SPEECH_REQUEST,
    SPEECH_SILENCE,
    SPEECH_WTA_FS,
)


FOOD_WORD = "apple"
VOCAB = list(OBJECT_WORDS) + ["need", "request", "silence"]


@dataclass(frozen=True)
class SpeechAction:
    intent: str
    referent: str


@dataclass
class SocialFoodWorld:
    """Legitimate outside-world code: body energy, scene, and food delivery."""

    energy: float = 0.25
    visible_object: str = FOOD_WORD
    refill: float = 0.75

    def apply(self, action: SpeechAction | None, *, consequence_enabled: bool = True) -> bool:
        delivered = bool(
            consequence_enabled
            and action == SpeechAction(intent="request", referent=FOOD_WORD)
            and self.visible_object == FOOD_WORD
        )
        if delivered:
            self.energy = min(1.0, self.energy + self.refill)
        return delivered


def _build_agent(seed: int, *, drive_weight: float, cue_weight: float) -> MergedNavConvAgent:
    return MergedNavConvAgent(
        seed=seed,
        vocab=VOCAB,
        co_resident_composer=True,
        co_resident_composer_kind="onebrain",
        co_resident_perception=True,
        co_resident_generalization=True,
        perception_grounding="gen_spikes",
        co_resident_drive=True,
        co_resident_grounded_speech=True,
        speech_drive_weight=drive_weight,
        speech_cue_weight=cue_weight,
        co_resident_command_route=False,
    )


def _csr_pre_post(bridge):
    csr = bridge.cp_connections
    indptr = np.asarray(to_host(csr.indptr), dtype=np.int64)
    post = np.asarray(to_host(csr.indices), dtype=np.int64)
    pre = np.empty(post.shape[0], dtype=np.int64)
    for row in range(int(csr.shape[0])):
        pre[indptr[row]:indptr[row + 1]] = row
    return pre, post


def _connection_mask(bridge, pre_indices, post_indices):
    pre, post = _csr_pre_post(bridge)
    mask = np.isin(pre, np.asarray(pre_indices)) & np.isin(post, np.asarray(post_indices))
    if not np.any(mask):
        raise RuntimeError("grounded speech pathway has no synapses")
    xp, _ = get_backend()
    return xp.asarray(mask), mask


def _percept_indices(agent: MergedNavConvAgent, obj_word: str) -> np.ndarray:
    gen = agent._handles["gen"]
    obj_idx = OBJECT_WORDS.index(obj_word)
    held_out = list(gen["gen_held_out"])
    local = np.asarray(gen["vis_sets"][held_out[obj_idx]], dtype=np.int64)
    return local + int(np.asarray(gen["perc_region"])[0])


def train_food_association(agent: MergedNavConvAgent, *, food_word: str = FOOD_WORD,
                           epochs: int = 16, train_steps: int = 60,
                           settle_steps: int = 30, teacher_pA: float = 600.0,
                           learning_rate: float = 0.02,
                           association_max_weight: float = 4.0):
    """Locally learn visual-feature -> food-cue support, then freeze it."""
    bridge = agent._merged_bridge
    cc = bridge.core_config
    xp, _ = get_backend()
    visual_idx = np.asarray(agent._handles["gen"]["perc_region"], dtype=np.int64)
    cue_idx = np.asarray(agent._handles["grounded_speech"][SPEECH_FOOD_CUE], dtype=np.int64)
    mask_x, mask_h = _connection_mask(bridge, visual_idx, cue_idx)
    initial = np.asarray(to_host(bridge.cp_connections.data[mask_x])).copy()

    if bridge.cp_plasticity_rate_gain is None:
        bridge.set_global_plasticity_gain(1.0)
    gain = bridge.cp_plasticity_rate_gain
    saved_gain = gain.copy()
    saved = (
        cc.enable_hebbian_learning,
        cc.enable_stdp,
        cc.enable_reward_modulation,
        cc.enable_ou_process,
        cc.hebbian_learning_rate,
        cc.hebbian_weight_decay,
    )
    threshold_before = bridge.cp_neuron_firing_thresholds.copy()
    activity_before = bridge.cp_neuron_activity_ema.copy()
    percept_idx = _percept_indices(agent, food_word)
    drive = xp.zeros(int(cc.num_neurons), dtype=xp.float32)
    drive[xp.asarray(percept_idx)] = np.float32(GEN_PERC_DRIVE_PA)
    drive[xp.asarray(cue_idx)] = np.float32(teacher_pA)

    gain[:] = 0.0
    gain[mask_x] = 1.0
    bridge.set_plasticity_gate(SPEECH_GROUNDING_GATE, 1.0)
    cc.enable_hebbian_learning = True
    cc.enable_stdp = False
    cc.enable_reward_modulation = False
    cc.enable_ou_process = False
    cc.hebbian_learning_rate = float(learning_rate)
    cc.hebbian_weight_decay = 0.0
    try:
        for _ in range(int(epochs)):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(int(settle_steps)):
                bridge._run_one_simulation_step()
            for _ in range(int(train_steps)):
                bridge.cp_external_input_current[:] = drive
                bridge._run_one_simulation_step()
    finally:
        bridge.cp_external_input_current[:] = 0.0
        bridge.set_plasticity_gate(SPEECH_GROUNDING_GATE, 0.0)
        gain[:] = saved_gain
        (
            cc.enable_hebbian_learning,
            cc.enable_stdp,
            cc.enable_reward_modulation,
            cc.enable_ou_process,
            cc.hebbian_learning_rate,
            cc.hebbian_weight_decay,
        ) = saved
        # The caregiver may change this association and nothing else. Restore
        # network-wide excitability so repeated apple pairing cannot fatigue
        # the visual concept or prime the speech pools through homeostasis.
        bridge.cp_neuron_firing_thresholds[:] = threshold_before
        bridge.cp_neuron_activity_ema[:] = activity_before

    trained = np.asarray(to_host(bridge.cp_connections.data[mask_x])).copy()
    trained = np.minimum(trained, float(association_max_weight)).astype(np.float32)
    bridge.cp_connections.data[mask_x] = xp.asarray(trained)
    return {
        "mask_device": mask_x,
        "mask_host": mask_h,
        "initial_weights": initial,
        "trained_weights": trained,
        "n_synapses": int(initial.size),
        "mean_initial": float(initial.mean()),
        "mean_trained": float(trained.mean()),
        "mean_delta": float(np.mean(trained - initial)),
        "max_delta": float(np.max(np.abs(trained - initial))),
    }


def _set_grounding_weights(agent: MergedNavConvAgent, learned, values) -> None:
    xp, _ = get_backend()
    agent._merged_bridge.cp_connections.data[learned["mask_device"]] = xp.asarray(
        np.asarray(values), dtype=xp.float32
    )


def set_request_threshold(agent: MergedNavConvAgent, threshold_mv: float):
    """Fix the request pool's neural operating point for the inference battery."""
    xp, _ = get_backend()
    idx = xp.asarray(agent._handles["grounded_speech"][SPEECH_REQUEST])
    before = np.asarray(to_host(agent._merged_bridge.cp_neuron_firing_thresholds[idx])).copy()
    agent._merged_bridge.cp_neuron_firing_thresholds[idx] = np.float32(threshold_mv)
    return {"before_mean_mv": float(before.mean()), "fixed_mv": float(threshold_mv)}


_TRIAL_STATE_PREFIXES = (
    "cp_membrane_",
    "cp_recovery_",
    "cp_gating_",
    "cp_conductance_",
    "cp_firing_",
    "cp_prev_firing_",
    "cp_refractory_",
    "cp_synapse_pulse_",
    "cp_stp_",
    "cp_hh_",
    "cp_adex_",
)
_TRIAL_STATE_NAMES = {
    "cp_external_input_current",
    "cp_neuron_firing_thresholds",
    "cp_neuron_activity_ema",
    "cp_ou_current",
    "cp_input_mean_ema",
    "cp_dendritic_source_activity",
    "cp_hebb_coactivity_trace",
}


def _snapshot_trial_state(agent: MergedNavConvAgent):
    """Capture mutable neural state so causal branches share one baseline."""
    bridge = agent._merged_bridge
    state = {}
    for name, value in vars(bridge).items():
        if value is None or not hasattr(value, "copy"):
            continue
        if name in _TRIAL_STATE_NAMES or name.startswith(_TRIAL_STATE_PREFIXES):
            state[name] = value.copy()
    return state


def _restore_trial_state(agent: MergedNavConvAgent, state) -> None:
    bridge = agent._merged_bridge
    for name, saved in state.items():
        current = getattr(bridge, name)
        if current is None or current.shape != saved.shape:
            raise RuntimeError(f"cannot restore trial state array {name}")
        current[...] = saved


def settle_after_training(agent: MergedNavConvAgent, *, steps: int = 1000) -> None:
    """Remove transient teacher activity while preserving learned weights."""
    bridge = agent._merged_bridge
    cc = bridge.core_config
    saved = (
        cc.enable_stdp,
        cc.enable_reward_modulation,
        cc.enable_hebbian_learning,
        cc.homeostasis_threshold_adapt_rate,
        cc.enable_ou_process,
    )
    cc.enable_stdp = False
    cc.enable_reward_modulation = False
    cc.enable_hebbian_learning = False
    cc.homeostasis_threshold_adapt_rate = 0.0
    cc.enable_ou_process = False
    bridge.cp_external_input_current[:] = 0.0
    try:
        for _ in range(int(steps)):
            bridge._run_one_simulation_step()
    finally:
        bridge.cp_external_input_current[:] = 0.0
        (
            cc.enable_stdp,
            cc.enable_reward_modulation,
            cc.enable_hebbian_learning,
            cc.homeostasis_threshold_adapt_rate,
            cc.enable_ou_process,
        ) = saved


def decide_speech_action(agent: MergedNavConvAgent, *, obj_word: str, energy: float,
                         lesion_drive: bool = False, lesion_perception: bool = False,
                         washout_steps: int = 160, decision_steps: int = 140,
                         drive_i_scale: float = 300.0,
                         silence_tonic_pA: float = 210.0):
    """Present the unchanged scene/body state and decode the neural speech action."""
    bridge = agent._merged_bridge
    xp, _ = get_backend()
    speech = agent._handles["grounded_speech"]
    rm = bridge.region_manager
    agrp = np.asarray(list(rm.indices("drive_agrp")), dtype=np.int64)
    pomc = np.asarray(list(rm.indices("drive_pomc")), dtype=np.int64)
    cue = np.asarray(speech[SPEECH_FOOD_CUE], dtype=np.int64)
    concept = np.asarray(agent._handles["gen"]["conc_region"], dtype=np.int64)
    request = np.asarray(speech[SPEECH_REQUEST], dtype=np.int64)
    silence = np.asarray(speech[SPEECH_SILENCE], dtype=np.int64)
    fs = np.asarray(speech[SPEECH_WTA_FS], dtype=np.int64)

    cc = bridge.core_config
    saved_learning = (
        cc.enable_stdp,
        cc.enable_reward_modulation,
        cc.enable_hebbian_learning,
        cc.homeostasis_threshold_adapt_rate,
        cc.enable_ou_process,
        cc.ou_std_current_pA,
    )
    cc.enable_stdp = False
    cc.enable_reward_modulation = False
    cc.enable_hebbian_learning = False
    cc.homeostasis_threshold_adapt_rate = 0.0
    cc.enable_ou_process = True
    cc.ou_std_current_pA = 20.0
    # Paired causal conditions share one OU realization.  The neural circuit
    # remains stochastic, but hunger/satiety and lesion comparisons cannot win
    # merely because they drew different background noise.
    xp.random.seed(int(agent.seed * 1009 + 13))
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(int(washout_steps)):
        bridge._run_one_simulation_step()

    deficit = float(np.clip(1.0 - float(energy), 0.0, 1.0))
    current = xp.zeros(int(bridge.core_config.num_neurons), dtype=xp.float32)
    if not lesion_perception:
        current[xp.asarray(_percept_indices(agent, obj_word))] = np.float32(GEN_PERC_DRIVE_PA)
    current[xp.asarray(agrp)] = np.float32(0.0 if lesion_drive else drive_i_scale * deficit)
    current[xp.asarray(pomc)] = np.float32(drive_i_scale * (1.0 - deficit))
    current[xp.asarray(silence)] += np.float32(silence_tonic_pA)

    counts = {"agrp": 0.0, "pomc": 0.0, "gen_concept": 0.0, "food_cue": 0.0,
              "request": 0.0, "silence": 0.0, "wta_fs": 0.0}
    index = {"agrp": agrp, "pomc": pomc, "gen_concept": concept, "food_cue": cue,
             "request": request, "silence": silence, "wta_fs": fs}
    try:
        for _ in range(int(decision_steps)):
            bridge.cp_external_input_current[:] = current
            bridge._run_one_simulation_step()
            firing = np.asarray(to_host(bridge.cp_firing_states))
            for name, idx in index.items():
                counts[name] += float(firing[idx].sum())
    finally:
        bridge.cp_external_input_current[:] = 0.0
        (
            cc.enable_stdp,
            cc.enable_reward_modulation,
            cc.enable_hebbian_learning,
            cc.homeostasis_threshold_adapt_rate,
            cc.enable_ou_process,
            cc.ou_std_current_pA,
        ) = saved_learning

    margin = counts["request"] - counts["silence"]
    action = SpeechAction(intent="request", referent=FOOD_WORD) if margin > 0.0 else None
    return {
        "scene": obj_word,
        "energy": float(energy),
        "deficit": deficit,
        "action": None if action is None else asdict(action),
        "decision": "request" if action is not None else "silence",
        "spikes": counts,
        "request_margin": float(margin),
        "lesion_drive": bool(lesion_drive),
        "lesion_perception": bool(lesion_perception),
    }, action


def _verdict(row):
    deficits = np.asarray([d["deficit"] for d in row["drive_dose_response"]], dtype=float)
    margins = np.asarray([d["request_margin"] for d in row["drive_dose_response"]], dtype=float)
    dose_corr = float(np.corrcoef(deficits, margins)[0, 1])
    checks = {
        "learned_route_changed": row["learning"]["mean_delta"] > 1e-4,
        "hungry_food_requests": row["hungry_food"]["decision"] == "request",
        "consequence_changes_body": row["food_delivered"] and row["energy_after"] > row["energy_before"],
        "request_precedes_delivery": row["causal_events"][:2] == [
            "brain_decision:request_apple", "world_consequence:food_delivered"],
        "same_scene_sated_silence": (
            row["hungry_food"]["scene"] == row["sated_same_scene"]["scene"]
            and row["sated_same_scene"]["decision"] == "silence"
        ),
        "deficit_tracks_request_margin": bool(
            np.isfinite(dose_corr)
            and dose_corr >= 0.90
            and row["drive_dose_response"][0]["decision"] == "silence"
            and row["drive_dose_response"][-1]["decision"] == "request"
        ),
        "drive_lesion_collapses": row["drive_lesion"]["decision"] == "silence",
        "perception_lesion_collapses": row["perception_lesion"]["decision"] == "silence",
        "wrong_object_silent": row["wrong_object"]["decision"] == "silence",
        "route_lesion_collapses": row["route_lesion"]["decision"] == "silence",
        "untrained_route_collapses": row["untrained_route"]["decision"] == "silence",
        "no_consequence_repeats_request": (
            not row["no_consequence"]["delivered"]
            and row["no_consequence"]["energy_after"] == row["no_consequence"]["energy_before"]
            and row["no_consequence"]["first"]["decision"] == "request"
            and row["no_consequence"]["second"]["decision"] == "request"
        ),
        "one_shared_bridge": row["one_shared_bridge"],
    }
    return {"go": bool(all(checks.values())), "checks": checks, "dose_correlation": dose_corr}


def run_seed(seed: int, *, drive_weight: float = 8.0, cue_weight: float = 60.0,
             train_epochs: int = 16, train_steps: int = 60,
             association_max_weight: float = 4.0,
             decision_steps: int = 140, silence_tonic_pA: float = 210.0,
             request_threshold_mv: float = -35.0,
             post_train_settle_steps: int = 1000,
             verbose: bool = True):
    agent = _build_agent(seed, drive_weight=drive_weight, cue_weight=cue_weight)
    learned = train_food_association(
        agent, epochs=train_epochs, train_steps=train_steps,
        association_max_weight=association_max_weight)
    threshold = set_request_threshold(agent, request_threshold_mv)
    settle_after_training(agent, steps=post_train_settle_steps)
    branch_origin = _snapshot_trial_state(agent)
    world = SocialFoodWorld()
    energy_before = world.energy

    _restore_trial_state(agent, branch_origin)
    hungry, hungry_action = decide_speech_action(
        agent, obj_word=world.visible_object, energy=world.energy, decision_steps=decision_steps,
        silence_tonic_pA=silence_tonic_pA)
    causal_events = [
        "brain_decision:request_apple" if hungry_action is not None else "brain_decision:silence"
    ]
    delivered = world.apply(hungry_action)
    causal_events.append(
        "world_consequence:food_delivered" if delivered else "world_consequence:none")
    energy_after = world.energy
    sated, _ = decide_speech_action(
        agent, obj_word=world.visible_object, energy=world.energy, decision_steps=decision_steps,
        silence_tonic_pA=silence_tonic_pA)

    dose_response = []
    for dose_energy in (1.0, 0.75, 0.50, 0.25):
        _restore_trial_state(agent, branch_origin)
        dose, _ = decide_speech_action(
            agent, obj_word=FOOD_WORD, energy=dose_energy,
            decision_steps=decision_steps, silence_tonic_pA=silence_tonic_pA)
        dose_response.append(dose)

    _restore_trial_state(agent, branch_origin)
    drive_lesion, _ = decide_speech_action(
        agent, obj_word=FOOD_WORD, energy=energy_before, lesion_drive=True,
        decision_steps=decision_steps, silence_tonic_pA=silence_tonic_pA)
    _restore_trial_state(agent, branch_origin)
    perception_lesion, _ = decide_speech_action(
        agent, obj_word=FOOD_WORD, energy=energy_before, lesion_perception=True,
        decision_steps=decision_steps, silence_tonic_pA=silence_tonic_pA)
    _restore_trial_state(agent, branch_origin)
    wrong_object, _ = decide_speech_action(
        agent, obj_word="river", energy=energy_before, decision_steps=decision_steps,
        silence_tonic_pA=silence_tonic_pA)

    _restore_trial_state(agent, branch_origin)
    _set_grounding_weights(agent, learned, np.zeros_like(learned["trained_weights"]))
    route_lesion, _ = decide_speech_action(
        agent, obj_word=FOOD_WORD, energy=energy_before, decision_steps=decision_steps,
        silence_tonic_pA=silence_tonic_pA)
    _restore_trial_state(agent, branch_origin)
    _set_grounding_weights(agent, learned, learned["initial_weights"])
    untrained, _ = decide_speech_action(
        agent, obj_word=FOOD_WORD, energy=energy_before, decision_steps=decision_steps,
        silence_tonic_pA=silence_tonic_pA)
    _set_grounding_weights(agent, learned, learned["trained_weights"])

    _restore_trial_state(agent, branch_origin)
    no_consequence_world = SocialFoodWorld()
    nc_first, nc_action = decide_speech_action(
        agent, obj_word=FOOD_WORD, energy=no_consequence_world.energy,
        decision_steps=decision_steps, silence_tonic_pA=silence_tonic_pA)
    nc_before = no_consequence_world.energy
    nc_delivered = no_consequence_world.apply(nc_action, consequence_enabled=False)
    nc_second, _ = decide_speech_action(
        agent, obj_word=FOOD_WORD, energy=no_consequence_world.energy,
        decision_steps=decision_steps, silence_tonic_pA=silence_tonic_pA)

    bridge = agent._merged_bridge
    speech_regions = set(agent._handles["grounded_speech"])
    region_names = set(bridge.region_manager.region_indices_dict())
    row = {
        "seed": int(seed),
        "learning": {k: v for k, v in learned.items() if not k.endswith("weights") and not k.endswith("device")
                     and not k.endswith("host")},
        "request_threshold": threshold,
        "energy_before": float(energy_before),
        "energy_after": float(energy_after),
        "food_delivered": bool(delivered),
        "causal_events": causal_events,
        "hungry_food": hungry,
        "sated_same_scene": sated,
        "drive_dose_response": dose_response,
        "drive_lesion": drive_lesion,
        "perception_lesion": perception_lesion,
        "wrong_object": wrong_object,
        "route_lesion": route_lesion,
        "untrained_route": untrained,
        "no_consequence": {
            "first": nc_first,
            "second": nc_second,
            "delivered": bool(nc_delivered),
            "energy_before": float(nc_before),
            "energy_after": float(no_consequence_world.energy),
        },
        "one_shared_bridge": bool(
            speech_regions.issubset(region_names)
            and {"gen_perception", "gen_concept", "drive_agrp", "drive_pomc"}.issubset(region_names)
        ),
        "scaffolds": [
            "explicit caregiver pairing of apple percept with food-cue population",
            "fixed need/cue coincidence and request-vs-silence wiring",
            "fixed host decoder from request population to conceptual request apple action",
            "host world renders sensory/interoceptive input and applies the social food consequence",
            "counterfactual lesion branches restore one shared post-training neural-state snapshot",
        ],
    }
    row["verdict"] = _verdict(row)
    if verbose:
        c = row["verdict"]["checks"]
        print(
            f"[seed {seed}] hungry={hungry['decision']} margin={hungry['request_margin']:.0f} -> "
            f"energy {energy_before:.2f}->{energy_after:.2f} -> same-scene {sated['decision']} "
            f"margin={sated['request_margin']:.0f} | lesions drive={drive_lesion['decision']} "
            f"percept={perception_lesion['decision']} route={route_lesion['decision']} "
            f"untrained={untrained['decision']} wrong={wrong_object['decision']} | "
            f"no-consequence={nc_first['decision']}/{nc_second['decision']} | "
            f"{'GO' if row['verdict']['go'] else 'NO'} {c}",
            flush=True,
        )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--drive-weight", type=float, default=8.0)
    ap.add_argument("--cue-weight", type=float, default=60.0)
    ap.add_argument("--association-max-weight", type=float, default=4.0)
    ap.add_argument("--silence-tonic-pA", type=float, default=210.0)
    ap.add_argument("--request-threshold-mv", type=float, default=-35.0)
    ap.add_argument("--train-epochs", type=int, default=16)
    ap.add_argument("--train-steps", type=int, default=60)
    ap.add_argument("--decision-steps", type=int, default=140)
    ap.add_argument("--post-train-settle-steps", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="research/findings/raw/grounded_speech_action_loop_6seed.json")
    a = ap.parse_args()

    seeds = a.seeds[:1] if a.smoke else a.seeds
    rows = [run_seed(
        seed,
        drive_weight=a.drive_weight,
        cue_weight=a.cue_weight,
        train_epochs=a.train_epochs,
        train_steps=a.train_steps,
        association_max_weight=a.association_max_weight,
        decision_steps=a.decision_steps,
        silence_tonic_pA=a.silence_tonic_pA,
        request_threshold_mv=a.request_threshold_mv,
        post_train_settle_steps=a.post_train_settle_steps,
    ) for seed in seeds]
    n_go = sum(r["verdict"]["go"] for r in rows)
    from tools.verdict import Verdict

    earned = Verdict("minimal grounded speech-action loop")
    earned.require("six independent seeds", len(rows), expect=lambda n: n >= 6)
    earned.require(
        "learned route changed on every seed",
        all(r["verdict"]["checks"]["learned_route_changed"] for r in rows), expect=True)
    earned.require(
        "all faculties occupy one shared bridge",
        all(r["one_shared_bridge"] for r in rows), expect=True)
    earned.require(
        "graded drive measurement is finite on every seed",
        all(np.isfinite(r["verdict"]["dose_correlation"]) for r in rows), expect=True)
    earned.disabled(
        "STDP / Hebbian / reward modulation / threshold adaptation during inference",
        why="the first rung isolates whether the trained food association and fixed need/cue circuit cause the action",
    )
    decided = earned.decide(go=bool(rows) and n_go == len(rows) and len(rows) >= 6, verbose=False)
    artifact = {
        "claim": "hunger plus grounded food perception causes a neural request whose consequence creates same-scene silence",
        "seeds": seeds,
        "n_go": int(n_go),
        "n_seeds": len(rows),
        "verdict": decided["status"],
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "config": {
            "drive_weight": a.drive_weight,
            "cue_weight": a.cue_weight,
            "association_max_weight": a.association_max_weight,
            "silence_tonic_pA": a.silence_tonic_pA,
            "request_threshold_mv": a.request_threshold_mv,
            "train_epochs": a.train_epochs,
            "train_steps": a.train_steps,
            "decision_steps": a.decision_steps,
            "post_train_settle_steps": a.post_train_settle_steps,
            "renderer": "none; conceptual speech action only",
        },
        "rows": rows,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"[saved] {out} | {artifact['verdict']} {n_go}/{len(rows)}", flush=True)
    return 0 if rows and n_go == len(rows) and (a.smoke or artifact["verdict"] == "GO") else 1


if __name__ == "__main__":
    raise SystemExit(main())
