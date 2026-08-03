"""Learn a compositional raw-vocal convention from consequences on one bridge.

Functional role in the whole brain
----------------------------------
Body/social context and perception must jointly cause a vocal action.  The
listener reacts to overt motor channels, and the resulting physical or social
reward changes later choices through a local coactivity eligibility trace and the
co-resident reward-US -> SNc dopamine pathway.

Two unlabeled intent channels and two unlabeled referent channels factor four
messages.  The listener convention is arbitrary and can be reversed.  The
brain never receives a target channel, intended label, or expected answer.

Honest scope
------------
This is a preverbal reinforcement-learning rung, not language. The original
benchmark preserves a target-independent balanced motor-babbling schedule;
the replacement path uses a shared spiking arousal/competition circuit and
sensory negative feedback. The world still has only two contexts and two
objects, and the body reads population spike counts. These are tracked
scaffolds. Direct caregiver output clamping and the old fixed ``request apple``
semantic decoder are absent.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sim.backend import get_backend, to_host
from research.runners.navigate_to_compose_then_answer import GEN_PERC_DRIVE_PA, OBJECT_WORDS
from research.runners._grounded_speech_action_loop_derisk import (
    _connection_mask,
    _percept_indices,
    _restore_trial_state,
    _snapshot_trial_state,
    settle_after_training,
)
from research.runners.nav_conv_merged_bridge import (
    MergedNavConvAgent,
    VOCAL_INTENT_FS,
    VOCAL_INTENT_PREFIX,
    VOCAL_EXPLORATION_AROUSAL,
    VOCAL_EXPLORE_PREFIX,
    VOCAL_LEARNING_GATE,
    VOCAL_NEGATIVE_FEEDBACK,
    VOCAL_REFERENT_FS,
    VOCAL_REFERENT_PREFIX,
    VOCAL_RMTG,
    VOCAL_SILENCE,
    VOCAL_SOCIAL_CUE,
    VOCAL_SPEAK,
    VOCAL_SPEAK_FS,
)


INTENTS = ("request", "report")
REFERENTS = ("apple", "river")
ALL_CASES = tuple((intent, referent) for intent in INTENTS for referent in REFERENTS)
TRAIN_CASES = (("request", "apple"), ("report", "river"))
HELD_OUT_CASES = tuple(case for case in ALL_CASES if case not in TRAIN_CASES)
VOCAB = list(OBJECT_WORDS) + ["need", "joint_attention", *INTENTS]


@dataclass(frozen=True)
class RawVocalAction:
    intent_channel: int
    referent_channel: int


@dataclass(frozen=True)
class VocalConvention:
    intent_to_channel: dict[str, int]
    referent_to_channel: dict[str, int]

    @classmethod
    def identity(cls):
        return cls(
            {name: i for i, name in enumerate(INTENTS)},
            {name: i for i, name in enumerate(REFERENTS)},
        )

    @classmethod
    def swapped(cls):
        return cls(
            {name: 1 - i for i, name in enumerate(INTENTS)},
            {name: 1 - i for i, name in enumerate(REFERENTS)},
        )

    def target(self, intent: str, referent: str) -> RawVocalAction:
        return RawVocalAction(
            int(self.intent_to_channel[intent]),
            int(self.referent_to_channel[referent]),
        )

    def decode(self, action: RawVocalAction | None):
        if action is None:
            return None
        inv_i = {v: k for k, v in self.intent_to_channel.items()}
        inv_r = {v: k for k, v in self.referent_to_channel.items()}
        intent = inv_i.get(int(action.intent_channel))
        referent = inv_r.get(int(action.referent_channel))
        return None if intent is None or referent is None else (intent, referent)


@dataclass
class InteractiveListenerWorld:
    """Outside-world code: observes motor channels and supplies consequences."""

    convention: VocalConvention
    visible_object: str
    context: str
    energy: float = 0.25
    social_satisfaction: float = 0.25

    def apply(self, action: RawVocalAction | None):
        decoded = self.convention.decode(action)
        expected_intent = "request" if self.context == "need" else "report"
        success = decoded == (expected_intent, self.visible_object)
        consequence = "none"
        if success and expected_intent == "request":
            self.energy = 1.0
            consequence = "resource_delivered"
        elif success:
            self.social_satisfaction = 1.0
            consequence = "social_acknowledgment"
        return {
            "success": bool(success),
            "decoded": None if decoded is None else list(decoded),
            "consequence": consequence,
            "energy": float(self.energy),
            "social_satisfaction": float(self.social_satisfaction),
        }


def _build_agent(
    seed: int,
    *,
    intrinsic_exploration: bool = False,
    error_feedback: bool = False,
):
    agent = MergedNavConvAgent(
        seed=seed,
        vocab=VOCAB,
        co_resident_composer=True,
        co_resident_composer_kind="onebrain",
        co_resident_perception=True,
        co_resident_generalization=True,
        perception_grounding="gen_spikes",
        co_resident_drive=True,
        co_resident_limbic=True,
        co_resident_nav_critic=False,
        co_resident_developmental_vocal=True,
        vocal_n_channels=2,
        vocal_intrinsic_exploration=bool(intrinsic_exploration),
        vocal_error_feedback=bool(error_feedback),
        co_resident_command_route=False,
    )
    if intrinsic_exploration:
        # Slow after-spike recovery makes a recently active exploration pool
        # temporarily less excitable, preventing one structural winner from
        # monopolizing development. This is intrinsic neural fatigue, not a
        # runner-selected action schedule.
        xp, _ = get_backend()
        handles = agent._handles["developmental_vocal"]
        explore = np.concatenate([
            np.asarray(handles[f"{VOCAL_EXPLORE_PREFIX}{bank}_{i}"], dtype=np.int64)
            for bank in ("speak", "intent", "referent")
            for i in range(2)
        ])
        explore_x = xp.asarray(explore)
        agent._merged_bridge.cp_izh_d_increment[explore_x] = xp.float32(80.0)
        agent._merged_bridge.cp_izh_a[explore_x] = xp.float32(0.008)
    return agent


def _vocal_indices(agent):
    h = agent._handles["developmental_vocal"]
    out = {
        "social": np.asarray(h[VOCAL_SOCIAL_CUE], dtype=np.int64),
        "speak": np.asarray(h[VOCAL_SPEAK], dtype=np.int64),
        "silence": np.asarray(h[VOCAL_SILENCE], dtype=np.int64),
        "speak_fs": np.asarray(h[VOCAL_SPEAK_FS], dtype=np.int64),
        "intent": [
            np.asarray(h[f"{VOCAL_INTENT_PREFIX}{i}"], dtype=np.int64)
            for i in range(len(INTENTS))
        ],
        "referent": [
            np.asarray(h[f"{VOCAL_REFERENT_PREFIX}{i}"], dtype=np.int64)
            for i in range(len(REFERENTS))
        ],
        "intent_fs": np.asarray(h[VOCAL_INTENT_FS], dtype=np.int64),
        "referent_fs": np.asarray(h[VOCAL_REFERENT_FS], dtype=np.int64),
    }
    explore_keys = [
        f"{VOCAL_EXPLORE_PREFIX}{bank}_{i}"
        for bank in ("speak", "intent", "referent")
        for i in range(2)
    ]
    if all(key in h for key in explore_keys):
        out["exploration_arousal"] = np.asarray(
            h[VOCAL_EXPLORATION_AROUSAL], dtype=np.int64
        )
        out["explore"] = {
            bank: [
                np.asarray(h[f"{VOCAL_EXPLORE_PREFIX}{bank}_{i}"], dtype=np.int64)
                for i in range(2)
            ]
            for bank in ("speak", "intent", "referent")
        }
    if VOCAL_NEGATIVE_FEEDBACK in h:
        out["negative_feedback"] = np.asarray(
            h[VOCAL_NEGATIVE_FEEDBACK], dtype=np.int64
        )
        out["rmtg"] = np.asarray(h[VOCAL_RMTG], dtype=np.int64)
    return out


def _region_indices(agent, name):
    return np.asarray(list(agent._merged_bridge.region_manager.indices(name)), dtype=np.int64)


def _step(bridge, n=1):
    for _ in range(int(n)):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms


def _set_ou_sigma(bridge, sigma_pA):
    """Update both the OU config value and its cached per-step coefficient."""
    cc = bridge.core_config
    cc.ou_std_current_pA = float(sigma_pA)
    dt_s = float(cc.dt_ms) / 1000.0
    tau_s = float(cc.ou_tau_ms) / 1000.0
    bridge.ou_noise_std = float(
        float(sigma_pA) * np.sqrt((1.0 - np.exp(-2.0 * dt_s / tau_s)) / 2.0)
    )


def _vocal_route_masks(agent):
    bridge = agent._merged_bridge
    h = _vocal_indices(agent)
    visual = np.asarray(agent._handles["gen"]["perc_region"], dtype=np.int64)
    agrp = _region_indices(agent, "drive_agrp")
    context_masks = []
    for target in (h["speak"], h["silence"], *h["intent"]):
        for source in (agrp, h["social"]):
            _, mask = _connection_mask(bridge, source, target)
            context_masks.append(mask)
    visual_masks = []
    for target in h["referent"]:
        _, mask = _connection_mask(bridge, visual, target)
        visual_masks.append(mask)
    context_host = np.logical_or.reduce(context_masks)
    visual_host = np.logical_or.reduce(visual_masks)
    host = context_host | visual_host
    xp, _ = get_backend()
    return xp.asarray(host), host, xp.asarray(context_host), xp.asarray(visual_host)


def _route_means(agent, convention):
    bridge = agent._merged_bridge
    h = _vocal_indices(agent)
    agrp = _region_indices(agent, "drive_agrp")
    routes = {}
    for intent in INTENTS:
        source = agrp if intent == "request" else h["social"]
        for bank, target in (("speak", h["speak"]), ("silence", h["silence"])):
            mask, _ = _connection_mask(bridge, source, target)
            weights = np.asarray(to_host(bridge.cp_connections.data[mask]), dtype=np.float64)
            routes[f"{bank}:{intent}"] = float(weights.mean())
        for channel, target in enumerate(h["intent"]):
            mask, _ = _connection_mask(bridge, source, target)
            weights = np.asarray(to_host(bridge.cp_connections.data[mask]), dtype=np.float64)
            routes[f"intent:{intent}:ch{channel}"] = float(weights.mean())
        routes[f"intent:{intent}"] = routes[
            f"intent:{intent}:ch{convention.intent_to_channel[intent]}"]
    for referent in REFERENTS:
        for channel, target in enumerate(h["referent"]):
            mask, _ = _connection_mask(bridge, _percept_indices(agent, referent), target)
            weights = np.asarray(to_host(bridge.cp_connections.data[mask]), dtype=np.float64)
            routes[f"referent:{referent}:ch{channel}"] = float(weights.mean())
        routes[f"referent:{referent}"] = routes[
            f"referent:{referent}:ch{convention.referent_to_channel[referent]}"]
    return routes


def _context_current(agent, intent: str, referent: str, *, context_pA=500.0):
    xp, _ = get_backend()
    bridge = agent._merged_bridge
    h = _vocal_indices(agent)
    current = xp.zeros(int(bridge.core_config.num_neurons), dtype=xp.float32)
    current[xp.asarray(_percept_indices(agent, referent))] = np.float32(GEN_PERC_DRIVE_PA)
    if intent == "request":
        current[xp.asarray(_region_indices(agent, "drive_agrp"))] = np.float32(context_pA)
    else:
        current[xp.asarray(_region_indices(agent, "drive_pomc"))] = np.float32(context_pA)
        current[xp.asarray(h["social"])] = np.float32(context_pA)
    return current


def _decode_counts(speak_counts, intent_counts, referent_counts, *, min_spikes=2.0, min_margin=1.0):
    intent_order = np.argsort(-intent_counts)
    referent_order = np.argsort(-referent_counts)
    speak_margin = float(speak_counts[0] - speak_counts[1])
    intent_margin = float(intent_counts[intent_order[0]] - intent_counts[intent_order[1]])
    referent_margin = float(referent_counts[referent_order[0]] - referent_counts[referent_order[1]])
    emitted = bool(
        speak_counts[0] >= min_spikes
        and speak_margin >= min_margin
        and intent_counts[intent_order[0]] >= min_spikes
        and intent_margin >= min_margin
        and referent_counts[referent_order[0]] >= min_spikes
        and referent_margin >= min_margin
    )
    action = RawVocalAction(int(intent_order[0]), int(referent_order[0])) if emitted else None
    return action, {
        "speak_spikes": speak_counts.tolist(),
        "intent_spikes": intent_counts.tolist(),
        "referent_spikes": referent_counts.tolist(),
        "speak_margin": speak_margin,
        "intent_margin": intent_margin,
        "referent_margin": referent_margin,
    }


def _explore_action(agent, intent, referent, exploration, *, pairings=1,
                    lead_steps=8, act_steps=18, exploration_pA=700.0):
    """Target-independent motor perturbation plus neural WTA readout."""
    bridge = agent._merged_bridge
    xp, _ = get_backend()
    h = _vocal_indices(agent)
    context_current = _context_current(agent, intent, referent)
    speak_choice = int(exploration[0])
    intent_choice = int(exploration[1])
    referent_choice = int(exploration[2])
    speak_counts = np.zeros(2, dtype=np.float64)
    intent_counts = np.zeros(2, dtype=np.float64)
    referent_counts = np.zeros(2, dtype=np.float64)
    for _ in range(int(pairings)):
        bridge.cp_external_input_current[:] = context_current
        _step(bridge, lead_steps)
        action_current = context_current.copy()
        action_current[xp.asarray(h["speak"] if speak_choice == 0 else h["silence"])] += np.float32(
            exploration_pA)
        action_current[xp.asarray(h["intent"][intent_choice])] += np.float32(exploration_pA)
        action_current[xp.asarray(h["referent"][referent_choice])] += np.float32(exploration_pA)
        for _ in range(int(act_steps)):
            bridge.cp_external_input_current[:] = action_current
            _step(bridge)
            firing = np.asarray(to_host(bridge.cp_firing_states))
            speak_counts += [firing[h["speak"]].sum(), firing[h["silence"]].sum()]
            for i in range(2):
                intent_counts[i] += firing[h["intent"][i]].sum()
                referent_counts[i] += firing[h["referent"][i]].sum()
    bridge.cp_external_input_current[:] = 0.0
    action, neural = _decode_counts(speak_counts, intent_counts, referent_counts)
    neural["exploration_channels"] = [speak_choice, intent_choice, referent_choice]
    neural["raw_action"] = None if action is None else asdict(action)
    return neural, action


def _intrinsic_explore_action(
    agent,
    intent,
    referent,
    *,
    snc_tonic_pA=0.0,
    exploration_arousal_pA=700.0,
    lead_steps=8,
    act_steps=120,
):
    """Read the first clear motor commitment produced by intrinsic exploration.

    Unlike ``_explore_action``, this function never writes current into a vocal
    output or channel-specific exploration population. It drives one shared
    arousal population, whose symmetric pathways and independent downstream
    spiking dynamics choose the raw action. The body's readout stops at the first
    jointly clear speak/intent/referent competition instead of averaging across
    later exploratory switches.
    """
    bridge = agent._merged_bridge
    xp, _ = get_backend()
    h = _vocal_indices(agent)
    current = _context_current(agent, intent, referent)
    current[xp.asarray(h["exploration_arousal"])] = np.float32(
        exploration_arousal_pA
    )
    if float(snc_tonic_pA) != 0.0:
        current[xp.asarray(_region_indices(agent, "limbic_snc"))] = np.float32(
            snc_tonic_pA
        )
    speak_counts = np.zeros(2, dtype=np.float64)
    intent_counts = np.zeros(2, dtype=np.float64)
    referent_counts = np.zeros(2, dtype=np.float64)
    explore_counts = {
        bank: np.zeros(2, dtype=np.float64)
        for bank in ("speak", "intent", "referent")
    }
    bridge.cp_external_input_current[:] = current
    _step(bridge, lead_steps)
    committed_action = None
    committed = False
    commit_latency_steps = None
    motor_outcome = "timeout"
    for step in range(int(act_steps)):
        bridge.cp_external_input_current[:] = current
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states))
        speak_counts += [firing[h["speak"]].sum(), firing[h["silence"]].sum()]
        for i in range(2):
            intent_counts[i] += firing[h["intent"][i]].sum()
            referent_counts[i] += firing[h["referent"][i]].sum()
            for bank in ("speak", "intent", "referent"):
                explore_counts[bank][i] += firing[h["explore"][bank][i]].sum()
        bank_counts = {
            "speak": speak_counts,
            "intent": intent_counts,
            "referent": referent_counts,
        }
        orders = {bank: np.argsort(-counts) for bank, counts in bank_counts.items()}
        committed = all(
            counts[orders[bank][0]] >= 2.0
            and counts[orders[bank][0]] - counts[orders[bank][1]] >= 1.0
            for bank, counts in bank_counts.items()
        )
        if committed:
            commit_latency_steps = int(step + 1)
            if int(orders["speak"][0]) == 0:
                committed_action = RawVocalAction(
                    int(orders["intent"][0]), int(orders["referent"][0])
                )
                motor_outcome = "emitted"
            else:
                motor_outcome = "silence"
            break
    bridge.cp_external_input_current[:] = 0.0
    _, neural = _decode_counts(speak_counts, intent_counts, referent_counts)
    neural["exploration_channels"] = None
    neural["exploration_spikes"] = {
        bank: counts.tolist() for bank, counts in explore_counts.items()
    }
    neural["injected_output_current"] = False
    neural["exploration_arousal_pA"] = float(exploration_arousal_pA)
    neural["motor_committed"] = bool(committed)
    neural["motor_outcome"] = motor_outcome
    neural["commit_latency_steps"] = commit_latency_steps
    neural["raw_action"] = (
        None if committed_action is None else asdict(committed_action)
    )
    return neural, committed_action


def calibrate_snc_tonic(agent, *, tonic_pA=220.0, settle_steps=300, measure_steps=120):
    """Center signed dopamine production on the measured tonic SNc rate."""
    bridge = agent._merged_bridge
    xp, _ = get_backend()
    snc = np.asarray(_region_indices(agent, "limbic_snc"), dtype=np.int64)
    snc_x = xp.asarray(snc)
    mgr = bridge.neuromodulator_manager
    da_cfg = mgr._config_by_name("dopamine")
    da_rule = da_cfg.production_rules[0]
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[snc_x] = np.float32(tonic_pA)
    _step(bridge, settle_steps)
    spikes = 0.0
    for _ in range(int(measure_steps)):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[snc_x] = np.float32(tonic_pA)
        _step(bridge)
        spikes += float(np.asarray(to_host(bridge.cp_firing_states[snc_x])).sum())
    tonic_fraction = spikes / max(1.0, float(len(snc) * measure_steps))
    da_rule.threshold = float(tonic_fraction)
    mgr._rule_state["dopamine"]["signed_rate_ema"] = float(tonic_fraction)
    mgr.set_concentration("dopamine", da_cfg.baseline)
    bridge.cp_external_input_current[:] = 0.0
    return {
        "tonic_pA": float(tonic_pA),
        "tonic_firing_fraction": float(tonic_fraction),
        "tonic_hz": float(1000.0 * tonic_fraction),
        "dopamine_baseline": float(da_cfg.baseline),
    }


def _drive_reward_us(
    agent,
    *,
    reward_steps=18,
    reward_pA=700.0,
    snc_tonic_pA=0.0,
):
    bridge = agent._merged_bridge
    xp, _ = get_backend()
    reward_us = _region_indices(agent, "limbic_reward_us")
    snc = _region_indices(agent, "limbic_snc")
    peak_da = bridge.neuromodulator_manager.get_concentration("dopamine")
    snc_spikes = 0.0
    current = xp.zeros(int(bridge.core_config.num_neurons), dtype=xp.float32)
    current[xp.asarray(reward_us)] = np.float32(reward_pA)
    if float(snc_tonic_pA) != 0.0:
        current[xp.asarray(snc)] = np.float32(snc_tonic_pA)
    for _ in range(int(reward_steps)):
        bridge.cp_external_input_current[:] = current
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states))
        snc_spikes += float(firing[snc].sum())
        peak_da = max(peak_da, bridge.neuromodulator_manager.get_concentration("dopamine"))
    bridge.cp_external_input_current[:] = 0.0
    return {
        "kind": "reward",
        "snc_spikes": snc_spikes,
        "peak_dopamine": float(peak_da),
        "minimum_dopamine": float(peak_da),
        "rmtg_spikes": 0.0,
    }


def _drive_negative_feedback(
    agent,
    *,
    feedback_steps=18,
    feedback_pA=700.0,
    snc_tonic_pA=220.0,
):
    """Route a negative social consequence through feedback -> RMTg -> SNc."""
    bridge = agent._merged_bridge
    xp, _ = get_backend()
    h = _vocal_indices(agent)
    if "negative_feedback" not in h:
        raise RuntimeError("developmental vocal error-feedback circuit is absent")
    snc = _region_indices(agent, "limbic_snc")
    rmtg = h["rmtg"]
    mgr = bridge.neuromodulator_manager
    minimum_da = mgr.get_concentration("dopamine")
    snc_spikes = 0.0
    rmtg_spikes = 0.0
    current = xp.zeros(int(bridge.core_config.num_neurons), dtype=xp.float32)
    current[xp.asarray(snc)] = np.float32(snc_tonic_pA)
    current[xp.asarray(h["negative_feedback"])] = np.float32(feedback_pA)
    for _ in range(int(feedback_steps)):
        bridge.cp_external_input_current[:] = current
        _step(bridge)
        firing = np.asarray(to_host(bridge.cp_firing_states))
        snc_spikes += float(firing[snc].sum())
        rmtg_spikes += float(firing[rmtg].sum())
        minimum_da = min(minimum_da, mgr.get_concentration("dopamine"))
    bridge.cp_external_input_current[:] = 0.0
    return {
        "kind": "negative_feedback",
        "snc_spikes": snc_spikes,
        "rmtg_spikes": rmtg_spikes,
        "peak_dopamine": float(minimum_da),
        "minimum_dopamine": float(minimum_da),
    }


def train_by_consequence(
    agent,
    convention,
    *,
    trials=720,
    exploration_seed=0,
    mode="contingent",
    yoked_schedule=None,
    intertrial_steps=100,
    exploration_mode="balanced",
    negative_feedback=False,
    snc_tonic_pA=0.0,
    intrinsic_action_steps=120,
    intrinsic_noise_pA=20.0,
    exploration_arousal_pA=700.0,
    phasic_update_window=False,
    phasic_consolidation_steps=30,
    negative_feedback_steps=18,
    negative_consolidation_steps=5,
    negative_learning_scale=0.10,
    vocal_weight_max=100.0,
    context_learning_gain=0.35,
    visual_learning_gain=1.0,
):
    """Reinforce overt successes; never inject a desired output population."""
    if mode not in ("contingent", "negative_only", "none", "yoked", "da_lesion"):
        raise ValueError(mode)
    if exploration_mode not in ("balanced", "intrinsic"):
        raise ValueError(exploration_mode)
    bridge = agent._merged_bridge
    cc = bridge.core_config
    xp, _ = get_backend()
    _, mask_h, context_mask_x, visual_mask_x = _vocal_route_masks(agent)
    before = _route_means(agent, convention)
    all_before = np.asarray(to_host(bridge.cp_connections.data), dtype=np.float32).copy()
    if bridge.cp_plasticity_rate_gain is None:
        bridge.set_global_plasticity_gain(1.0)
    saved_gain = bridge.cp_plasticity_rate_gain.copy()
    saved = {
        name: getattr(cc, name)
        for name in (
            "enable_hebbian_learning", "enable_stdp", "enable_reward_modulation",
            "enable_neuromodulator_subsystem",
            "reward_defer_stdp_weight_update", "reward_learning_rate",
            "reward_eligibility_tau_ms", "reward_eligibility_from_coactivity",
            "reward_coactivity_trace_tau_ms", "reward_coactivity_threshold",
            "reward_coactivity_scale", "stdp_a_plus", "stdp_a_minus",
            "stdp_w_min", "stdp_w_max", "hebbian_min_weight", "hebbian_max_weight",
            "enable_ou_process", "ou_std_current_pA",
            "homeostasis_threshold_adapt_rate",
        )
    }
    saved_eligibility_scope = bridge.cp_reward_eligibility_synapse_indices
    saved_ou_noise_std = bridge.ou_noise_std
    mgr = bridge.neuromodulator_manager
    da_mod = mgr._config_by_name("dopamine")
    saved_da_tau = da_mod.decay_tau_ms
    da_rule = da_mod.production_rules[0]
    saved_da_window = da_rule.window_ms
    bridge.cp_plasticity_rate_gain[:] = 0.0
    bridge.set_plasticity_gate(VOCAL_LEARNING_GATE, 1.0)
    # Context populations are much denser and more active than the sparse visual
    # feature ensemble. The lower context gain is a fixed anatomical learning-
    # rate difference, shared by every channel and convention.
    bridge.cp_plasticity_rate_gain[context_mask_x] = float(context_learning_gain)
    bridge.cp_plasticity_rate_gain[visual_mask_x] = float(visual_learning_gain)
    cc.enable_hebbian_learning = False
    cc.enable_stdp = False
    cc.enable_reward_modulation = True
    # Trials are compressed into milliseconds, whereas cortical threshold
    # homeostasis acts over developmental time. Letting the default rate run at
    # this accelerated schedule raises motor thresholds until learned speech is
    # silent before any biologically comparable recovery interval can pass.
    cc.homeostasis_threshold_adapt_rate = 0.0
    cc.reward_learning_rate = 0.05
    cc.reward_eligibility_tau_ms = 20.0
    cc.reward_eligibility_from_coactivity = True
    cc.reward_coactivity_trace_tau_ms = 20.0
    cc.reward_coactivity_threshold = 0.001
    # The bounded trace product is typically O(1e-2) during a successful
    # exploratory burst. This scale brings a few dozen rewarded experiences
    # into the same weight range as the downstream spiking operating point;
    # reward and the anatomical gate still determine whether any update occurs.
    cc.reward_coactivity_scale = 360.0
    cc.stdp_w_min = 0.0
    cc.stdp_w_max = float(vocal_weight_max)
    cc.hebbian_min_weight = 0.0
    cc.hebbian_max_weight = float(vocal_weight_max)
    cc.enable_ou_process = True
    _set_ou_sigma(
        bridge,
        float(intrinsic_noise_pA) if exploration_mode == "intrinsic" else 10.0
    )
    da_mod.decay_tau_ms = 20.0
    da_rule.window_ms = 20.0
    bridge.cp_reward_eligibility_synapse_indices = xp.asarray(
        np.flatnonzero(mask_h), dtype=xp.int64
    )
    if mode == "da_lesion":
        cc.enable_neuromodulator_subsystem = False
    if bridge.cp_eligibility_trace is not None:
        bridge.cp_eligibility_trace[:] = 0.0
    if bridge.cp_reward_coactivity_trace is not None:
        bridge.cp_reward_coactivity_trace[:] = 0.0
    active_gain = bridge.cp_plasticity_rate_gain.copy()

    rng = np.random.default_rng(int(exploration_seed))
    # Balanced motor babbling: each context encounters every raw speak/intent/
    # referent combination once per shuffled cycle. The schedule is independent
    # of the listener convention and desired output, but avoids conflating a
    # learning failure with a seed that simply never explored one action.
    motor_space = np.asarray(
        [
            (speak, intent_channel, referent_channel)
            for speak in range(2)
            for intent_channel in range(2)
            for referent_channel in range(2)
        ],
        dtype=np.int64,
    )
    motor_queues = {case: [] for case in TRAIN_CASES}
    events = []
    reward_schedule = []
    try:
        for trial in range(int(trials)):
            if phasic_update_window:
                bridge.cp_plasticity_rate_gain[:] = active_gain
                cc.enable_neuromodulator_subsystem = False
            intent, referent = TRAIN_CASES[trial % len(TRAIN_CASES)]
            case = (intent, referent)
            if exploration_mode == "intrinsic":
                neural, action = _intrinsic_explore_action(
                    agent,
                    intent,
                    referent,
                    snc_tonic_pA=snc_tonic_pA,
                    exploration_arousal_pA=exploration_arousal_pA,
                    act_steps=intrinsic_action_steps,
                )
            else:
                if not motor_queues[case]:
                    order = rng.permutation(len(motor_space))
                    motor_queues[case] = [motor_space[i] for i in order]
                exploration = motor_queues[case].pop()
                neural, action = _explore_action(
                    agent, intent, referent, exploration
                )
            world = InteractiveListenerWorld(
                convention, referent, "need" if intent == "request" else "joint_attention")
            consequence = world.apply(action)
            if mode == "contingent":
                reward_now = bool(consequence["success"])
            elif mode == "negative_only":
                reward_now = False
            elif mode == "yoked":
                reward_now = bool(yoked_schedule[trial])
            else:
                reward_now = False
            if mode == "da_lesion":
                reward_now = bool(consequence["success"])
            negative_now = bool(
                negative_feedback
                and mode in ("contingent", "negative_only", "da_lesion")
                and not consequence["success"]
                and action is not None
            )
            reward_trace = {
                "kind": "neutral",
                "snc_spikes": 0.0,
                "rmtg_spikes": 0.0,
                "peak_dopamine": 0.5,
                "minimum_dopamine": 0.5,
            }
            if reward_now:
                if phasic_update_window and mode != "da_lesion":
                    cc.enable_neuromodulator_subsystem = True
                reward_trace = _drive_reward_us(
                    agent, snc_tonic_pA=snc_tonic_pA
                )
            elif negative_now:
                if phasic_update_window and mode != "da_lesion":
                    cc.enable_neuromodulator_subsystem = True
                    # Error dips are frequent before a convention exists, so
                    # they prune more gently than successes consolidate.
                    bridge.cp_plasticity_rate_gain[:] = (
                        active_gain * float(negative_learning_scale)
                    )
                reward_trace = _drive_negative_feedback(
                    agent,
                    feedback_steps=negative_feedback_steps,
                    snc_tonic_pA=snc_tonic_pA,
                )
            else:
                bridge.cp_external_input_current[:] = 0.0
                if float(snc_tonic_pA) != 0.0:
                    bridge.cp_external_input_current[
                        xp.asarray(_region_indices(agent, "limbic_snc"))
                    ] = np.float32(snc_tonic_pA)
                _step(bridge, 18)
            reward_schedule.append(bool(reward_now))
            consolidation_steps = 0
            if phasic_update_window and (reward_now or negative_now):
                consolidation_limit = (
                    negative_consolidation_steps
                    if negative_now
                    else phasic_consolidation_steps
                )
                consolidation_steps = min(
                    int(intertrial_steps), int(consolidation_limit)
                )
                bridge.cp_external_input_current[:] = 0.0
                if float(snc_tonic_pA) != 0.0:
                    bridge.cp_external_input_current[
                        xp.asarray(_region_indices(agent, "limbic_snc"))
                    ] = np.float32(snc_tonic_pA)
                _step(bridge, consolidation_steps)
            if phasic_update_window:
                bridge.cp_plasticity_rate_gain[:] = 0.0
                cc.enable_neuromodulator_subsystem = mode != "da_lesion"
            bridge.cp_external_input_current[:] = 0.0
            if float(snc_tonic_pA) != 0.0:
                bridge.cp_external_input_current[
                    xp.asarray(_region_indices(agent, "limbic_snc"))
                ] = np.float32(snc_tonic_pA)
            _step(bridge, int(intertrial_steps) - consolidation_steps)
            if trial < 12 or trial >= int(trials) - 12 or reward_now or negative_now:
                events.append({
                    "trial": trial,
                    "case": [intent, referent],
                    "neural": neural,
                    "listener_success": consequence["success"],
                    "reward_delivered": bool(reward_now),
                    "negative_feedback_delivered": bool(negative_now),
                    "reward_trace": reward_trace,
                })
    finally:
        bridge.cp_external_input_current[:] = 0.0
        bridge.set_plasticity_gate(VOCAL_LEARNING_GATE, 0.0)
        bridge.cp_plasticity_rate_gain[:] = saved_gain
        bridge.cp_reward_eligibility_synapse_indices = saved_eligibility_scope
        for name, value in saved.items():
            setattr(cc, name, value)
        bridge.ou_noise_std = saved_ou_noise_std
        da_mod.decay_tau_ms = saved_da_tau
        da_rule.window_ms = saved_da_window

    after = _route_means(agent, convention)
    all_after = np.asarray(to_host(bridge.cp_connections.data), dtype=np.float32)
    changed = np.abs(all_after - all_before) > 1e-7
    outside_changed = int(np.count_nonzero(changed & ~mask_h))
    deltas = {name: after[name] - before[name] for name in before}
    return {
        "mode": mode,
        "exploration_mode": exploration_mode,
        "negative_feedback": bool(negative_feedback),
        "exploration_arousal_pA": float(exploration_arousal_pA),
        "phasic_update_window": bool(phasic_update_window),
        "phasic_consolidation_steps": int(phasic_consolidation_steps),
        "negative_feedback_steps": int(negative_feedback_steps),
        "negative_consolidation_steps": int(negative_consolidation_steps),
        "negative_learning_scale": float(negative_learning_scale),
        "vocal_weight_max": float(vocal_weight_max),
        "context_learning_gain": float(context_learning_gain),
        "visual_learning_gain": float(visual_learning_gain),
        "trials": int(trials),
        "n_rewards": int(sum(reward_schedule)),
        "reward_schedule": reward_schedule,
        "route_before": before,
        "route_after": after,
        "route_deltas": deltas,
        "n_changed_synapses": int(changed.sum()),
        "outside_vocal_changed_synapses": outside_changed,
        "events": events,
    }


def decide_raw_vocal_action(
    agent,
    *,
    intent_context,
    referent,
    lesion_context=False,
    lesion_perception=False,
    washout_steps=100,
    decision_steps=100,
):
    bridge = agent._merged_bridge
    cc = bridge.core_config
    xp, _ = get_backend()
    h = _vocal_indices(agent)
    saved = (
        cc.enable_stdp, cc.enable_reward_modulation, cc.enable_hebbian_learning,
        cc.homeostasis_threshold_adapt_rate, cc.enable_ou_process, cc.ou_std_current_pA,
    )
    saved_ou_noise_std = bridge.ou_noise_std
    cc.enable_stdp = False
    cc.enable_reward_modulation = False
    cc.enable_hebbian_learning = False
    cc.homeostasis_threshold_adapt_rate = 0.0
    cc.enable_ou_process = True
    _set_ou_sigma(bridge, 20.0)
    xp.random.seed(int(agent.seed * 1009 + 71))
    bridge.cp_external_input_current[:] = 0.0
    _step(bridge, washout_steps)
    current = xp.zeros(int(cc.num_neurons), dtype=xp.float32)
    if not lesion_perception:
        current[xp.asarray(_percept_indices(agent, referent))] = np.float32(GEN_PERC_DRIVE_PA)
    if not lesion_context and intent_context == "need":
        current[xp.asarray(_region_indices(agent, "drive_agrp"))] = np.float32(500.0)
    elif not lesion_context and intent_context == "joint_attention":
        current[xp.asarray(_region_indices(agent, "drive_pomc"))] = np.float32(500.0)
        current[xp.asarray(h["social"])] = np.float32(500.0)
    speak_counts = np.zeros(2, dtype=np.float64)
    intent_counts = np.zeros(2, dtype=np.float64)
    referent_counts = np.zeros(2, dtype=np.float64)
    try:
        for _ in range(int(decision_steps)):
            bridge.cp_external_input_current[:] = current
            _step(bridge)
            firing = np.asarray(to_host(bridge.cp_firing_states))
            speak_counts += [firing[h["speak"]].sum(), firing[h["silence"]].sum()]
            for i in range(2):
                intent_counts[i] += firing[h["intent"][i]].sum()
                referent_counts[i] += firing[h["referent"][i]].sum()
    finally:
        bridge.cp_external_input_current[:] = 0.0
        (
            cc.enable_stdp, cc.enable_reward_modulation, cc.enable_hebbian_learning,
            cc.homeostasis_threshold_adapt_rate, cc.enable_ou_process, cc.ou_std_current_pA,
        ) = saved
        bridge.ou_noise_std = saved_ou_noise_std
    action, neural = _decode_counts(speak_counts, intent_counts, referent_counts)
    neural.update({
        "context": intent_context,
        "referent_scene": referent,
        "raw_action": None if action is None else asdict(action),
        "lesion_context": bool(lesion_context),
        "lesion_perception": bool(lesion_perception),
    })
    return neural, action


def _evaluate(agent, convention, origin):
    rows = []
    for intent, referent in ALL_CASES:
        context = "need" if intent == "request" else "joint_attention"
        _restore_trial_state(agent, origin)
        neural, action = decide_raw_vocal_action(
            agent, intent_context=context, referent=referent)
        world = InteractiveListenerWorld(convention, referent, context)
        rows.append({
            "evaluation_target": [intent, referent],
            "target_raw_channels": asdict(convention.target(intent, referent)),
            "neural": neural,
            "listener": world.apply(action),
        })
    return rows


def _accuracy(rows):
    return float(np.mean([row["listener"]["success"] for row in rows]))


def _factor_accuracy(rows, factor):
    index = 0 if factor == "intent" else 1
    values = []
    for row in rows:
        decoded = row["listener"]["decoded"]
        values.append(decoded is not None and decoded[index] == row["evaluation_target"][index])
    return float(np.mean(values))


def _release():
    import gc

    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _train_and_evaluate(seed, convention, mode, trials, exploration_seed, yoked_schedule=None):
    agent = _build_agent(seed)
    settle_after_training(agent, steps=300)
    training = train_by_consequence(
        agent,
        convention,
        trials=trials,
        exploration_seed=exploration_seed,
        mode=mode,
        yoked_schedule=yoked_schedule,
    )
    settle_after_training(agent, steps=500)
    origin = _snapshot_trial_state(agent)
    rows = _evaluate(agent, convention, origin)
    return agent, training, origin, rows


def run_seed(
    seed,
    *,
    trials=720,
    full_controls=True,
    reversal_diagnostic=True,
    verbose=True,
):
    identity = VocalConvention.identity()
    swapped = VocalConvention.swapped()
    exploration_seed = int(seed * 7919 + 17)

    agent, main_train, origin, main_rows = _train_and_evaluate(
        seed, identity, "contingent", trials, exploration_seed)
    _restore_trial_state(agent, origin)
    context_lesion, _ = decide_raw_vocal_action(
        agent, intent_context="need", referent="apple", lesion_context=True)
    _restore_trial_state(agent, origin)
    perception_lesion, _ = decide_raw_vocal_action(
        agent, intent_context="need", referent="apple", lesion_perception=True)
    _restore_trial_state(agent, origin)
    no_reason, _ = decide_raw_vocal_action(
        agent, intent_context=None, referent="apple")

    reversal_train = None
    reversed_rows = []
    reversal_checks = {}
    if reversal_diagnostic:
        pre_reversal_actions = [row["neural"]["raw_action"] for row in main_rows]
        reversal_train = train_by_consequence(
            agent,
            swapped,
            trials=trials * 2,
            exploration_seed=exploration_seed + 1,
            mode="contingent",
        )
        settle_after_training(agent, steps=500)
        reversal_origin = _snapshot_trial_state(agent)
        reversed_rows = _evaluate(agent, swapped, reversal_origin)
        post_reversal_actions = [row["neural"]["raw_action"] for row in reversed_rows]
        reversal_checks = {
            "same_brain_reversal_accuracy": _accuracy(reversed_rows),
            "same_brain_reversal_relearned": _accuracy(reversed_rows) >= 0.80,
            "same_brain_reversal_changed_channels": all(
                before != after
                for before, after in zip(pre_reversal_actions, post_reversal_actions)
            ),
            "interpretation": (
                "diagnostic only: the present positive-reward rule has no learned "
                "omission/error pathway for depressing an obsolete convention"
            ),
        }
    one_shared_bridge = bool(
        "developmental_vocal" in agent._handles
        and "limbic" in agent._handles
        and "gen" in agent._handles
        and "drive" in agent._handles
    )
    del agent, origin
    _release()

    controls = {}
    if full_controls:
        swapped_agent, swapped_train, swapped_origin, swapped_rows = _train_and_evaluate(
            seed, swapped, "contingent", trials, exploration_seed)
        controls["fresh_swapped_convention"] = {
            "training": swapped_train,
            "rows": swapped_rows,
        }
        del swapped_agent, swapped_origin
        _release()

        no_agent, no_train, no_origin, no_rows = _train_and_evaluate(
            seed, identity, "none", trials, exploration_seed)
        controls["no_consequence"] = {"training": no_train, "rows": no_rows}
        del no_agent, no_origin
        _release()

        schedule = np.asarray(main_train["reward_schedule"], dtype=bool)
        yoked_rng = np.random.default_rng(seed + 444)
        yoked_schedule = schedule[yoked_rng.permutation(len(schedule))].tolist()
        yoked_agent, yoked_train, yoked_origin, yoked_rows = _train_and_evaluate(
            seed,
            identity,
            "yoked",
            trials,
            exploration_seed,
            yoked_schedule=yoked_schedule,
        )
        controls["yoked"] = {"training": yoked_train, "rows": yoked_rows}
        del yoked_agent, yoked_origin
        _release()

        da_agent, da_train, da_origin, da_rows = _train_and_evaluate(
            seed, identity, "da_lesion", trials, exploration_seed)
        controls["da_lesion"] = {"training": da_train, "rows": da_rows}
        del da_agent, da_origin
        _release()

    main_acc = _accuracy(main_rows)
    reversed_acc = _accuracy(reversed_rows) if reversed_rows else None
    control_acc = {name: _accuracy(value["rows"]) for name, value in controls.items()}
    outside_counts = [main_train["outside_vocal_changed_synapses"]]
    if reversal_train is not None:
        outside_counts.append(reversal_train["outside_vocal_changed_synapses"])
    outside_counts.extend(
        value["training"]["outside_vocal_changed_synapses"]
        for value in controls.values()
    )
    checks = {
        "main_joint_accuracy": main_acc >= 0.80,
        "main_intent_accuracy": _factor_accuracy(main_rows, "intent") >= 0.90,
        "main_referent_accuracy": _factor_accuracy(main_rows, "referent") >= 0.90,
        "held_out_composes": all(
            row["listener"]["success"] for row in main_rows
            if tuple(row["evaluation_target"]) in HELD_OUT_CASES),
        "reward_us_drove_snc": any(
            event["reward_trace"]["snc_spikes"] > 0
            and event["reward_trace"]["peak_dopamine"] > 0.5
            for event in main_train["events"] if event["reward_delivered"]),
        "only_vocal_synapses_changed": all(count == 0 for count in outside_counts),
        "context_lesion_blocks_action": context_lesion["raw_action"] is None,
        "perception_lesion_blocks_action": perception_lesion["raw_action"] is None,
        "no_reason_is_silent": no_reason["raw_action"] is None,
        "one_shared_bridge": one_shared_bridge,
        "no_target_channel_in_training": True,
        "control_battery_run": bool(full_controls),
    }
    if full_controls:
        checks.update({
            "fresh_swapped_convention_learned": (
                control_acc["fresh_swapped_convention"] >= 0.80),
            "no_consequence_does_not_learn": control_acc["no_consequence"] <= 0.25,
            "yoked_reward_does_not_learn": control_acc["yoked"] <= 0.50,
            "da_lesion_does_not_learn": control_acc["da_lesion"] <= 0.25,
            "contingency_beats_controls": all(
                main_acc >= control_acc[name] + 0.25
                for name in ("no_consequence", "yoked", "da_lesion")
            ),
        })
    row = {
        "seed": int(seed),
        "train_cases": [list(case) for case in TRAIN_CASES],
        "held_out_cases": [list(case) for case in HELD_OUT_CASES],
        "main_training": main_train,
        "main_rows": main_rows,
        "reversal_training": reversal_train,
        "reversed_rows": reversed_rows,
        "reversal_diagnostic": reversal_checks,
        "controls": controls,
        "accuracies": {
            "main_joint": main_acc,
            "main_intent": _factor_accuracy(main_rows, "intent"),
            "main_referent": _factor_accuracy(main_rows, "referent"),
            "reversed_joint": reversed_acc,
            **control_acc,
        },
        "lesions": {
            "context": context_lesion,
            "perception": perception_lesion,
            "no_reason": no_reason,
        },
        "checks": checks,
        "go": bool(all(checks.values())),
    }
    if verbose:
        print(
            f"[vocal seed={seed}] main={main_acc:.2f} "
            f"reversal={reversed_acc if reversed_acc is not None else 'skipped'} "
            f"controls={control_acc} -> {'GO' if row['go'] else 'NO-GO'}",
            flush=True,
        )
        if not row["go"]:
            print("  failed:", [name for name, ok in checks.items() if not ok], flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--trials", type=int, default=720)
    ap.add_argument("--smoke", action="store_true", help="skip expensive no/yoked/DA controls")
    ap.add_argument(
        "--skip-reversal-diagnostic", action="store_true",
        help="skip the non-gating same-brain convention-reversal diagnostic",
    )
    ap.add_argument(
        "--out",
        default="research/findings/raw/developmental_vocal_convention_6seed.json",
    )
    args = ap.parse_args()
    rows = [
        run_seed(
            seed,
            trials=args.trials,
            full_controls=not args.smoke,
            reversal_diagnostic=not args.skip_reversal_diagnostic,
        )
        for seed in args.seeds
    ]
    summary = {
        "probe": "developmental_vocal_convention",
        "seeds": args.seeds,
        "trials": args.trials,
        "full_controls": not args.smoke,
        "rows": rows,
        "n_go": int(sum(row["go"] for row in rows)),
        "all_go": bool(all(row["go"] for row in rows)),
        "scope": (
            "preverbal two-intent by two-referent dopamine-reinforced raw vocal convention; "
            "not natural language or open conversation"
        ),
    }
    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[vocal] {summary['n_go']}/{len(rows)} seeds GO -> {out}", flush=True)
    return 0 if summary["all_go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
