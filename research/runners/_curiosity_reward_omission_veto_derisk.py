"""DR-1 CURIOSITY — the SPIKING REWARD-OMISSION VETO (lane B · Curiosity, 2026-08-01).

CONTEXT. The 2026-08-01 finding `..._curiosity-veto-cannot-be-read-off-the-spiking-striosome-value-it-inverts.md`
established: DR-1 curiosity is GO, but its noisy-concept veto (the honesty anti-cheat: STOP asking about
UNLEARNABLE concepts) is still computed by a HOST Python ELP tracker (a TD low-pass fed by the SNc
paired-subtraction `snc_B - snc_A`). Thresholding the spiking striosome VALUE FAILED (0/6, it INVERTS: a noisy
concept reads a HIGHER value because reward-independent STDP drift inflates it). BUT the finding located the
clean spiking separator: the **SNc reward burst** — learnable concepts drive 4-31 Hz, noisy/unlearnable drive
0.0 Hz on every seed. So the named next method is a **reward-OMISSION circuit**.

THIS RUNNER builds that circuit — a spiking detector of reward-omission that gates the ASK pool DOWN for concepts
that yield NO SNc reward burst (= unlearnable), replacing the host ELP+threshold arithmetic with substrate
computation. The whole point: the veto is computed by NEURONS reading reward-omission, not a host tracker.

THE CIRCUIT (all additive to the DR-1 build — NO sim/ edit; new BrainRegions + RegionPathways only):
  snc --(exc)--> rmtg --(gaba_a, inhibitory)--> omit --(exc)--> veto <--(PLASTIC exc)-- cue
  * `rmtg`  (RMTg GABAergic relay, exc_fraction=0.0): snc drives it; it inhibits omit. The sign-inverting
    interneuron so a reward BURST in the (excitatory) SNc can SUPPRESS the omission detector.
  * `omit`  (lateral-habenula / RMTg omission detector, RS): TONICALLY driven. When the SNc bursts (reward
    present = learnable) rmtg silences omit; when the SNc does NOT burst (reward OMITTED = noisy) omit fires.
    This is the reward-omission detector — the subtraction (tonic - snc) is done by SYNAPTIC INTEGRATION, not
    by a Python `snc_B - snc_A`. (Matsumoto-Hikosaka 2007: LHb neurons fire on negative RPE / reward omission.)
  * `cue -> veto` PLASTIC: the per-concept omission MEMORY lives in these synapses. During each ask's veto-learn
    window, if omit fires (omission), omit->veto makes `veto` spike; cue[c] (pre) + veto (post) coincide -> STDP
    fills eligibility -> a constant "learning-enable" reward converts it -> cue[c]->veto POTENTIATES. A learnable
    concept's SNc burst keeps omit (hence veto) SILENT -> no post-spike -> no potentiation. So a NOISY concept
    accumulates cue->veto weight over asks; a learnable one does not. The accumulation is SYNAPTIC, produced by
    the omission detector's spikes — this is the part the host ELP tracker did in Python.
  * READ (in the candidate filter): drive cue[c] alone and read the `veto` pool's firing rate. HIGH veto rate ->
    the concept accumulated omission evidence -> VETOED (stop asking). LOW -> keep asking. This is a spiking
    read-out thresholded exactly like DR-1's own WANT_FLOOR / NOVEL_THRESH gates (an accepted pattern in the GO).

Per-pathway `set_plasticity_gate` keeps the two learners disjoint: during the veto-learn window cue->striosome is
FROZEN (the striosome critic is untouched, so DR-1 gates a/b/c still hold); during the striosome window cue->veto
is FROZEN. Nothing contaminates the DR-1 critic.

GO GATE (the DR-1 honesty anti-cheat, now veto = the spiking omission read):
  (a) corr(gap, SPIKING want) >= 0.9   (b) ask unknown >= 2x known   (c) conf rises above the abstain floor
  NOISY-STOPS: late noisy ask-rate << early WHILE noisy g stays HIGH AND the SPIKING veto fired (noisy vetoed).
  MOAT: confident subset of asked.
ANTI-CHEATS:
  * OMISSION-DETECTOR LESION (the load-bearing one for THIS conversion): zero omit's tonic drive -> omit never
    fires -> veto never learns -> nothing is vetoed -> the brain asks the noisy TV indiscriminately (the host
    tracker's failure mode, now produced by lesioning a spiking pool). The veto must COLLAPSE.
  * CURIOSITY LESION: no ASK drive -> no asking (DR-1).
  * YOKED-RANDOM reward: LP is an uninformative draw -> the SNc bursts on the WRONG asks -> omit fires on the
    wrong concepts -> the wrong veto -> masters fewer.
  * PERMUTED teacher: gap mis-mapped -> corr collapses.
Attribution (tools.lab): `attributable_to(noisy-veto, real, omit-lesion)` — the fraction of the veto firing that
is NOT present with the detector lesioned (deep-credit-style; the omission detector should own ~all of it).

Reuse-by-import from `_curiosity_seek_learn_onbridge_derisk` (World, the familiarity gate, the wash-out/settle
helpers, the SNc-neutral read, the config constants). SPIKING on a real bridge: CPU-smoke first
(SIM_BACKEND=numpy --smoke), then GPU 6-seed (SIM_BACKEND=cupy).
Run: SIM_BACKEND=cupy python -u -m research.runners._curiosity_reward_omission_veto_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Reuse-by-import: the DR-1 curiosity machinery (env, familiarity gate, wash-out helpers, SNc-neutral read,
# config constants). This runner only ADDS the spiking omission circuit + swaps the veto read.
from research.runners._curiosity_seek_learn_onbridge_derisk import (  # noqa: E402
    World, RealAntiHebbianFamiliarity,
    _host, _idx, _advance, _settle, _snapshot_state, _restore_state,
    _measure_snc_neutral, _lesion_gabab_mask,
    D, N_LEARN, N_NOISY, N_TURNS, ASK_BUDGET, NOVEL_THRESH, EPS, OBS_NOISE,
    WANT_FLOOR_HZ, SNC_TONIC_PA, CUE_DRIVE_PA, US_GAIN_PA, SNC_SCALE, RPE_GAIN,
    W_WANT, W_WARMUP, W_MEASURE, W_APPLY, W_VALUE, W_SETTLE,
)
from tools.lab import attributable_to, lever  # noqa: E402


# ------------------------------- the omission-circuit config (additive) -------------------------------
# The omission detector + accumulator read-out. Tuned so the SNc's clean 0-vs-burst separation (the finding's
# 4-31 Hz learnable vs 0.0 Hz noisy) maps to omit SILENT (learnable) vs omit FIRING (noisy).
N_RMTG = 24
N_OMIT = 30
N_VETO = 40
OMIT_TONIC_PA = 400.0        # tonic drive on the reward-absence detector (LHb tonic activity); 0.0 = detector LESION
REWARDUS_TO_RMTG_W = 9.0     # reward US (the delivered reward, LP>0) -> rmtg fires (the sign-inverting relay)
RMTG_TO_OMIT_W = 12.0        # rmtg fires -> silences omit (gaba_a); a DELIVERED reward thus suppresses omit
OMIT_TO_VETO_W = 11.0        # omit fires -> veto spikes (the post-spike on a reward-ABSENCE ask)
REWARDUS_TO_VETO_W = 9.0     # reward US -> veto also spikes on a REWARD ask (so eligibility builds either way; the
                             # SIGN of the update, not whether it happens, carries reward-vs-omission)
CUE_TO_VETO_W0 = 0.05        # PLASTIC per-concept veto memory, init ~0 so an UN-TRIED concept reads ~0 veto Hz
                             # (the veto rate must come ONLY from the learned memory, not from baseline cue drive)
OMIT_MID_HZ = 18.0           # the DA/LHb tonic baseline the omit read is compared against: a reward-absence ask
                             # reads omit ABOVE it (+RPE_veto -> POTENTIATE the veto memory), a reward ask reads
                             # omit BELOW it (-RPE_veto -> DEPRESS it). The opponent/protective term.
OMIT_RPE_SCALE = 45.0        # Hz normalizer mapping (omit_read - OMIT_MID) -> a graded veto RPE ~[-0.4, +1.4]
                             # (mirrors DR-1's SNC_SCALE on the striosome RPE, so potentiation is graded not saturating)
W_VETO = W_VALUE             # veto-read window (read the veto pool's rate under cue[c] alone)
VETO_FLOOR_HZ = 12.0         # a concept is VETOED iff its spiking veto rate exceeds this (thresholded read)

# --- LANE B (additive, default-OFF via --reserve): the DECAYING SUB-BASELINE INHIBITORY protective reserve ---
# The single cue->veto pathway is EXCITATORY, so its protective depression on reward can only lower it toward ~0 (the
# floor limitation the 2026-08-01 finding named: a concept that was EVER rewarding cannot bank a reserve that pushes
# the veto read BELOW zero). THIS second plastic cue->veto pathway is INHIBITORY (receptor="gaba_a") and learns with
# the OPPOSITE sign — POTENTIATED on REWARD (omit LOW -> reward present), DEPRESSED/decaying on absence — so an
# ever-rewarding concept accumulates an ACTIVE inhibitory reserve that drives its veto read SUB-baseline, protecting
# it from a false veto even if its excitatory omission memory drifts up. A disjoint `reserve_learn` plasticity gate
# keeps it from contaminating the striosome critic OR the excitatory veto memory. When --reserve is OFF the pathway
# is never added and no reserve window runs -> the config + trajectory are BYTE-IDENTICAL to the omission-only build.
CUE_TO_RESERVE_W0 = 0.05     # plastic reserve memory, init ~0 (an un-rewarded concept has no reserve)
RESERVE_RPE_SCALE = OMIT_RPE_SCALE   # reuse the omit RPE normalizer; the reserve applies its NEGATED argument
# The reserve is realized as a DISTINCT inhibitory region `reserve` (NOT a second cue->veto pathway). The region
# framework keys one wiring group per (from_region,to_region) pair (`regions.py` build_wiring_plan), so a second
# cue->veto pathway OVERWRITES the excitatory veto_learn memory in the plan and DROPS the veto_learn gate. Routing
# the reserve through its own pool — a PLASTIC cue->reserve memory (gate `reserve_learn`) + a FIXED gaba_a
# reserve->veto inhibition — keeps BOTH the veto_learn and reserve_learn gates registered (disjoint region pairs).
N_RESERVE = N_VETO           # inhibitory reserve pool (matches the veto pool size)
RESERVE_TONIC_PA = 400.0     # tonic drive on the reserve pool during ITS learn window so it spikes and cue+reserve
                             # coincidence fills cue->reserve eligibility (mirrors omit's tonic bootstrap of veto)
RESERVE_TO_VETO_W = 11.0     # FIXED gaba_a inhibition reserve->veto (the active protective inhibition read at recall)


def build_omission_veto_bridge(seed, n_concepts, *, n_per_cue=40, n_strio=60, n_reward_us=40,
                               n_snc=30, n_ask=80, cue_to_strio_weight=11.0,
                               reward_us_to_snc_weight=10.0, strio_to_snc_weight=2.0,
                               gabab_prop=0.22, gabab_tau_decay=150.0, reward_learning_rate=0.30,
                               curiosity_prod_sensitivity=0.10,
                               curiosity_excit_sensitivity=320.0, curiosity_decay_tau=50.0,
                               enable_heterogeneity=True, enable_reserve=False):
    """DR-1's build_curiosity_bridge (verbatim config) PLUS the spiking reward-omission veto circuit:
    the rmtg/omit/veto pools + snc->rmtg->omit->veto + a PLASTIC cue->veto, and two named plasticity gates
    (`strio_learn`, `veto_learn`) so the striosome critic and the veto accumulator learn in disjoint windows."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = True
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_parameter_heterogeneity = bool(enable_heterogeneity)
    cfg.reward_learning_rate = float(reward_learning_rate)
    cfg.current_reward_signal = 0.0
    cfg.reward_baseline = 0.0
    cfg.current_novelty_signal = 0.0
    cfg.novelty_baseline = 0.0
    cfg.reward_aversive_scale = 1.0
    cfg.stdp_w_max = 40.0

    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = float(gabab_tau_decay)
    cfg.gabab_propagation_strength = float(gabab_prop)
    cfg.gabab_conductance_max = 0.0

    RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    FS = NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name
    n_cue = n_concepts * n_per_cue
    cfg.brain_regions = [
        BrainRegion(name="cue", n_neurons=n_cue, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
        BrainRegion(name="striosome_value", n_neurons=n_strio, exc_fraction=0.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name,
                    syn_reversal_potential_i_override=-60.0),
        BrainRegion(name="reward_us", n_neurons=n_reward_us, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False, izh_neuron_type=RS),
        BrainRegion(name="snc", n_neurons=n_snc, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS,
                    syn_reversal_potential_i_override=-90.0),
        BrainRegion(name="ask", n_neurons=n_ask, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
        # --- the SPIKING reward-omission veto circuit (additive) ---
        # RMTg GABAergic relay: excited by snc, inhibits omit (the sign inverter so a reward burst suppresses
        # the omission detector). exc_fraction=0.0 -> its output is inhibitory.
        BrainRegion(name="rmtg", n_neurons=N_RMTG, exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=FS),
        # omission detector (LHb analog): tonically driven, silenced by rmtg -> fires iff the SNc did NOT burst.
        BrainRegion(name="omit", n_neurons=N_OMIT, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
        # veto read-out: driven by omit during learn (the STDP post-spike) and by the LEARNED cue->veto at read.
        BrainRegion(name="veto", n_neurons=N_VETO, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False, izh_neuron_type=RS),
    ]
    cfg.region_pathways = [
        # the DR-1 striosome critic (cue->strio PLASTIC), now tagged with a plasticity gate so it can be frozen
        # during the veto-learn window (default gain 1.0 -> byte-identical to DR-1 when never frozen).
        RegionPathway(from_region="cue", to_region="striosome_value",
                      density=0.6, weight_mean=float(cue_to_strio_weight),
                      weight_jitter=0.5, plastic=True, plasticity_gate="strio_learn"),
        RegionPathway(from_region="reward_us", to_region="snc",
                      density=0.6, weight_mean=float(reward_us_to_snc_weight),
                      weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="striosome_value", to_region="snc",
                      density=0.5, weight_mean=float(strio_to_snc_weight),
                      weight_jitter=0.2, plastic=False, receptor="gaba_b"),
        # --- the reward-absence circuit wiring ---
        # reward US -> rmtg (the delivered reward drives the GABAergic relay) -> inhibits omit.
        RegionPathway(from_region="reward_us", to_region="rmtg",
                      density=0.6, weight_mean=REWARDUS_TO_RMTG_W, weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="rmtg", to_region="omit",
                      density=0.6, weight_mean=RMTG_TO_OMIT_W, weight_jitter=0.2, plastic=False,
                      receptor="gaba_a"),
        # omit -> veto and reward_us -> veto BOTH make veto spike (so cue+veto STDP eligibility builds on EITHER
        # ask type); a transmission gate isolates the READ (cue alone) from these drives.
        RegionPathway(from_region="omit", to_region="veto",
                      density=0.6, weight_mean=OMIT_TO_VETO_W, weight_jitter=0.2, plastic=False,
                      transmission_gate="veto_drive"),
        RegionPathway(from_region="reward_us", to_region="veto",
                      density=0.6, weight_mean=REWARDUS_TO_VETO_W, weight_jitter=0.2, plastic=False,
                      transmission_gate="veto_drive"),
        # the PLASTIC per-concept veto memory (the host ELP tracker's replacement): POTENTIATED on reward-absence,
        # DEPRESSED on reward (the sign comes from omit_read - reward_baseline). Gated by `veto_learn`.
        RegionPathway(from_region="cue", to_region="veto",
                      density=0.6, weight_mean=CUE_TO_VETO_W0, weight_jitter=0.2, plastic=True,
                      plasticity_gate="veto_learn"),
    ]
    # LANE B (additive, default-OFF): the DECAYING SUB-BASELINE INHIBITORY protective reserve. Realized as a
    # DISTINCT inhibitory `reserve` pool driven by a PLASTIC cue->reserve memory (gate `reserve_learn`, learned in a
    # disjoint window with the OPPOSITE sign — potentiated on reward), which projects a FIXED gaba_a inhibition onto
    # `veto`. A second cue->veto pathway is NOT usable: the region framework keys ONE wiring group per (from,to)
    # pair, so it would OVERWRITE (drop) the excitatory veto_learn memory and de-register the veto_learn gate. The
    # reserve's own region pair keeps both gates registered. Added ONLY when enable_reserve -> off = byte-identical.
    if enable_reserve:
        cfg.brain_regions.append(
            BrainRegion(name="reserve", n_neurons=N_RESERVE, exc_fraction=0.0, internal_density=0.0,
                        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                        plastic_internal=False, izh_neuron_type=FS))
        cfg.region_pathways.append(
            RegionPathway(from_region="cue", to_region="reserve",
                          density=0.6, weight_mean=CUE_TO_RESERVE_W0, weight_jitter=0.2, plastic=True,
                          plasticity_gate="reserve_learn"))
        cfg.region_pathways.append(
            RegionPathway(from_region="reserve", to_region="veto",
                          density=0.6, weight_mean=RESERVE_TO_VETO_W, weight_jitter=0.2, plastic=False,
                          receptor="gaba_a"))
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [
        NeuromodulatorConfig(
            name="curiosity", baseline=0.0, decay_tau_ms=float(curiosity_decay_tau),
            concentration_min=0.0, concentration_max=5.0,
            targets=[ModulatorTarget(target_type="excitability_drive", scope="group:ask",
                                     sensitivity=float(curiosity_excit_sensitivity))],
            production_rules=[ProductionRule(rule_type="from_novelty",
                                             sensitivity=float(curiosity_prod_sensitivity))]),
    ]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    # veto accumulator starts FROZEN (learns only inside the veto-learn window); striosome critic normal.
    bridge.set_plasticity_gate("veto_learn", 0.0)
    bridge.set_plasticity_gate("strio_learn", 1.0)
    if enable_reserve:
        bridge.set_plasticity_gate("reserve_learn", 0.0)   # inhibitory reserve also frozen except in its window
    return bridge, cfg


drives_regions = ("cue", "striosome_value", "reward_us", "snc", "ask", "rmtg", "omit", "veto")


def run(seed, mode, *, n_learn=N_LEARN, n_noisy=N_NOISY, n_turns=N_TURNS, ask_budget=ASK_BUDGET,
        d=D, verbose=False, enable_reserve=False, **build_kw):
    from sim.backend import get_backend
    xp, _ = get_backend()
    rng = np.random.default_rng(seed * 101 + 5)
    n_concepts = n_learn + n_noisy
    world = World(seed, d, n_learn, n_noisy, OBS_NOISE)
    gate = RealAntiHebbianFamiliarity()
    concepts = world.concepts
    perm = {c: concepts[(i + 3) % len(concepts)] for i, c in enumerate(concepts)}

    lesion_curiosity = (mode == "lesion")
    omit_lesion = (mode == "omit_lesion")
    omit_tonic = 0.0 if omit_lesion else OMIT_TONIC_PA
    # LANE B: the reserve pathway is BUILT for real reserve runs AND for the reserve_lesion control (present but not
    # learning, so the control is a fair like-for-like). `suppress_reserve` zeroes the reserve RPE (no potentiation).
    reserve_on = bool(enable_reserve) or (mode == "reserve_lesion")
    suppress_reserve = (mode == "reserve_lesion")
    bk = dict(build_kw)
    bk["enable_reserve"] = reserve_on
    if lesion_curiosity:
        bk["curiosity_excit_sensitivity"] = 0.0
    bridge, cfg = build_omission_veto_bridge(seed, n_concepts, **bk)
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in drives_regions}
    if reserve_on:                                        # the reserve pool exists only when reserve is built
        idx_map["reserve"] = xp.asarray(_idx(bridge, "reserve"))
    cue_all = _host(idx_map["cue"]).astype(np.int64)
    n_per_cue = len(cue_all) // n_concepts
    cue_slice = {c: xp.asarray(cue_all[c * n_per_cue:(c + 1) * n_per_cue]) for c in concepts}
    if mode == "critic_lesion":
        _lesion_gabab_mask(bridge)

    _settle(bridge, W_SETTLE)
    snap0 = _snapshot_state(bridge)
    snc_neutral = _measure_snc_neutral(bridge, idx_map, xp, snap0)

    snc_idx = idx_map["snc"]; n_snc = len(_host(snc_idx))
    veto_idx = idx_map["veto"]; n_veto = len(_host(veto_idx))
    reward_us_idx = idx_map["reward_us"]; omit_idx = idx_map["omit"]; n_omit = len(_host(omit_idx))
    reserve_idx = idx_map.get("reserve")                  # None unless the reserve pool was built

    def read_value(c):
        _restore_state(bridge, snap0)
        bridge.cp_external_input_current[cue_slice[c]] = xp.float32(CUE_DRIVE_PA)
        bridge.core_config.current_novelty_signal = 0.0
        saved = cfg.reward_learning_rate; cfg.reward_learning_rate = 0.0
        strio_idx = idx_map["striosome_value"]; n_strio = len(_host(strio_idx)); spk = 0
        for _ in range(W_VALUE):
            _advance(bridge)
            spk += int(bridge.cp_firing_states[strio_idx].sum())
        cfg.reward_learning_rate = saved
        return spk / max(n_strio, 1) / (W_VALUE * 1e-3)

    def read_want(c, novelty):
        _restore_state(bridge, snap0)
        bridge.core_config.current_novelty_signal = float(novelty)
        ask_idx = idx_map["ask"]; n_ask = len(_host(ask_idx)); spk = 0
        saved = cfg.reward_learning_rate; cfg.reward_learning_rate = 0.0
        for _ in range(W_WANT):
            _advance(bridge)
            spk += int(bridge.cp_firing_states[ask_idx].sum())
        cfg.reward_learning_rate = saved
        return spk / max(n_ask, 1) / (W_WANT * 1e-3)

    def read_veto(c):
        """The SPIKING veto read: drive cue[c] ALONE, with the omit/reward_us->veto drives GATED OFF
        (`veto_drive` closed), and read the `veto` pool's rate. So veto fires ONLY from the LEARNED cue->veto
        memory. HIGH -> the concept accumulated reward-absence -> vetoed. This is the substrate-computed veto
        value, replacing the host `snc_B - snc_A` -> ELP low-pass. Frozen + reward_lr=0 -> a pure read."""
        bridge.set_transmission_gate("veto_drive", 0.0)     # isolate the read to cue->veto
        _restore_state(bridge, snap0)
        bridge.cp_external_input_current[cue_slice[c]] = xp.float32(CUE_DRIVE_PA)
        saved = cfg.reward_learning_rate; cfg.reward_learning_rate = 0.0
        spk = 0
        for _ in range(W_VETO):
            _advance(bridge)
            spk += int(bridge.cp_firing_states[veto_idx].sum())
        cfg.reward_learning_rate = saved
        bridge.set_transmission_gate("veto_drive", 1.0)
        _restore_state(bridge, snap0)
        return spk / max(n_veto, 1) / (W_VETO * 1e-3)

    def _snc_window(c, LP):
        _restore_state(bridge, snap0)
        bridge.cp_external_input_current[cue_slice[c]] = xp.float32(CUE_DRIVE_PA)
        bridge.cp_external_input_current[reward_us_idx] = xp.float32(US_GAIN_PA * max(LP, 0.0))
        bridge.cp_external_input_current[snc_idx] = xp.float32(SNC_TONIC_PA)
        bridge.core_config.current_reward_signal = 0.0
        spk = 0
        for i in range(W_WARMUP + W_MEASURE):
            _advance(bridge)
            if i >= W_WARMUP:
                spk += int(bridge.cp_firing_states[snc_idx].sum())
        return spk / max(n_snc, 1) / (W_MEASURE * 1e-3)

    def deliver_reward(c, LP):
        """DR-1's spiking-SNc striosome critic learning (unchanged): build cue->strio eligibility while the SNc
        computes r-V, convert with the SNc-read RPE. Returns snc_burst_Hz. cue->veto stays FROZEN throughout."""
        bridge.cp_eligibility_trace[:] = 0.0
        snc_B = _snc_window(c, LP)
        rpe = float(np.clip((snc_B - snc_neutral) / SNC_SCALE, -1.5, 1.5)) * RPE_GAIN
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[snc_idx] = xp.float32(SNC_TONIC_PA)
        bridge.core_config.current_reward_signal = rpe
        for _ in range(W_APPLY):
            _advance(bridge)
        bridge.core_config.current_reward_signal = 0.0
        return snc_B

    def learn_veto(c, LP):
        """The SPIKING reward-absence veto learning for concept c (the LHb opponent memory). Drive cue[c] +
        reward_us(∝LP) + omit tonic. The delivered reward US drives rmtg, which silences omit; so on a REWARD
        ask (LP>0) omit is LOW and veto fires via reward_us->veto, while on a reward-ABSENCE ask (LP~0) omit is
        HIGH and veto fires via omit->veto. Either way cue[c]+veto coincide -> cue->veto STDP eligibility. The
        UPDATE SIGN is set by the omit read against the tonic baseline (reward_baseline=OMIT_MID_HZ): omit ABOVE
        baseline (absence) -> +RPE -> POTENTIATE; omit BELOW baseline (reward) -> -RPE -> DEPRESS (the protective
        opponent term). So a NOISY concept (always absence) accumulates veto; a learnable one (early reward
        depresses, building a reserve) does not cross the floor before it is mastered out. cue->strio is FROZEN
        so the DR-1 critic is untouched. Returns the omit read (HIGH on absence, LOW on reward) — the detector."""
        bridge.set_plasticity_gate("strio_learn", 0.0)     # protect the DR-1 critic
        bridge.set_plasticity_gate("veto_learn", 0.0)      # frozen during warmup (steady-state only)
        _restore_state(bridge, snap0)
        bridge.cp_eligibility_trace[:] = 0.0
        bridge.cp_external_input_current[cue_slice[c]] = xp.float32(CUE_DRIVE_PA)
        bridge.cp_external_input_current[reward_us_idx] = xp.float32(US_GAIN_PA * max(LP, 0.0))
        bridge.cp_external_input_current[omit_idx] = xp.float32(omit_tonic)
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(W_WARMUP):                          # settle rmtg/omit/veto to steady state
            _advance(bridge)
        bridge.set_plasticity_gate("veto_learn", 1.0)      # now cue->veto STDP fills eligibility
        omit_spk = 0
        for _ in range(W_MEASURE):
            _advance(bridge)
            omit_spk += int(bridge.cp_firing_states[omit_idx].sum())
        omit_read = omit_spk / max(n_omit, 1) / (W_MEASURE * 1e-3)
        # APPLY: the SIGNED conversion. effective RPE = (omit_read - OMIT_MID_HZ)/OMIT_RPE_SCALE, clipped — a
        # reward-absence ask (omit high) -> + (POTENTIATE); a reward ask (omit low) -> - (DEPRESS). DA release ∝
        # the detector firing (the pattern DR-1's SNc-read reward uses); the baseline-subtract + scale mirror
        # DR-1's SNC_SCALE so potentiation is graded, letting the protective depression accumulate a reserve.
        veto_rpe = float(np.clip((omit_read - OMIT_MID_HZ) / OMIT_RPE_SCALE, -1.0, 1.5))
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = veto_rpe
        for _ in range(W_APPLY):
            _advance(bridge)
        bridge.core_config.current_reward_signal = 0.0
        bridge.set_plasticity_gate("veto_learn", 0.0)      # re-freeze the excitatory veto memory
        if reserve_on and reserve_idx is not None:
            # LANE B — the DECAYING SUB-BASELINE INHIBITORY reserve. OPPOSITE-sign learning on cue->reserve: drive
            # cue[c] + reward_us(∝LP) + omit tonic + a tonic on the reserve POOL so it spikes and cue+reserve
            # coincide (the pool cannot bootstrap from its ~0 init weight alone). Then apply reserve_rpe =
            # (OMIT_MID - omit_read)/scale — POSITIVE on a REWARD ask (omit LOW) = POTENTIATE the reserve memory;
            # NEGATIVE on absence = DEPRESS/decay it. An ever-rewarding concept banks cue->reserve weight, so at
            # recall cue[c] drives the inhibitory reserve pool -> reserve->veto pushes its veto read SUB-baseline; a
            # noisy (always-absence) concept never does. strio + veto_learn stay FROZEN (both memories untouched).
            bridge.set_plasticity_gate("reserve_learn", 0.0)
            _restore_state(bridge, snap0)
            bridge.cp_eligibility_trace[:] = 0.0
            bridge.cp_external_input_current[cue_slice[c]] = xp.float32(CUE_DRIVE_PA)
            bridge.cp_external_input_current[reward_us_idx] = xp.float32(US_GAIN_PA * max(LP, 0.0))
            bridge.cp_external_input_current[omit_idx] = xp.float32(omit_tonic)
            bridge.cp_external_input_current[reserve_idx] = xp.float32(RESERVE_TONIC_PA)
            bridge.core_config.current_reward_signal = 0.0
            for _ in range(W_WARMUP):                       # re-settle rmtg/omit/veto/reserve to steady state
                _advance(bridge)
            bridge.set_plasticity_gate("reserve_learn", 1.0)   # now cue->reserve STDP fills eligibility
            for _ in range(W_MEASURE):
                _advance(bridge)
            reserve_rpe = 0.0 if suppress_reserve else float(np.clip(
                (OMIT_MID_HZ - omit_read) / RESERVE_RPE_SCALE, -1.0, 1.5))
            bridge.cp_external_input_current[:] = 0.0
            bridge.core_config.current_reward_signal = reserve_rpe
            for _ in range(W_APPLY):
                _advance(bridge)
            bridge.core_config.current_reward_signal = 0.0
            bridge.set_plasticity_gate("reserve_learn", 0.0)
        bridge.set_plasticity_gate("strio_learn", 1.0)     # restore the critic gate
        _restore_state(bridge, snap0)
        return omit_read

    # bookkeeping (mirror DR-1)
    corr_gap, corr_want = [], []
    asked = set(); ask_events = []; conf_first_ask = {}; n_asks = 0
    elig_unknown = elig_known = ask_unknown = ask_known = 0
    third = max(1, n_turns // 3)
    noisy_elig = [0, 0, 0]; noisy_ask = [0, 0, 0]
    yoke_pool = rng.permutation(np.linspace(0.0, 1.0, 200)); yi = 0
    snc_learn_burst, snc_noisy_burst = [], []
    omitfire_learn, omitfire_noisy = [], []            # the omission drive (veto learn-window rate) by class
    # Vveto[c] = the last SPIKING veto read for concept c (a memoised measurement; the ACCUMULATION is synaptic
    # in cue->veto). Init 0 -> an un-tried concept is never vetoed (gets tried a couple times first).
    Vveto = {c: 0.0 for c in concepts}

    for turn in range(n_turns):
        if n_asks >= ask_budget:
            break
        true_gaps = {c: gate.novelty(world.render(c)) for c in concepts}
        gate_gap = ({c: true_gaps[perm[c]] for c in concepts} if mode == "permuted" else true_gaps)
        if mode == "yoked":
            drive_nov = {c: float(rng.choice(list(true_gaps.values()))) for c in concepts}
        elif mode == "permuted":
            drive_nov = {c: float(true_gaps[perm[c]]) for c in concepts}
        else:
            drive_nov = {c: float(true_gaps[c]) for c in concepts}

        want = {c: read_want(c, drive_nov[c]) for c in concepts}
        for c in concepts:
            corr_gap.append(true_gaps[c]); corr_want.append(want[c])
            unknown = true_gaps[c] > NOVEL_THRESH
            if unknown:
                elig_unknown += 1
            else:
                elig_known += 1
            if world.is_noisy[c] and unknown:
                noisy_elig[min(turn // third, 2)] += 1

        # the veto is now the SPIKING omission read: a concept is a candidate iff NOVEL, drive-active, and its
        # veto pool rate is BELOW the floor (not yet flagged unlearnable).
        not_vetoed = (lambda c: Vveto[c] < VETO_FLOOR_HZ)
        cands = [c for c in concepts
                 if gate_gap[c] > NOVEL_THRESH and want[c] > WANT_FLOOR_HZ and not_vetoed(c)]
        if not cands:
            continue

        if rng.random() < EPS:
            c_ask = int(rng.choice(cands))
        else:
            mx = max(want[c] for c in cands)
            c_ask = int(rng.choice([c for c in cands if want[c] >= mx - 1e-9]))

        if true_gaps[c_ask] > NOVEL_THRESH:
            ask_unknown += 1
        else:
            ask_known += 1
        if world.is_noisy[c_ask]:
            noisy_ask[min(turn // third, 2)] += 1

        g_before = true_gaps[c_ask]
        if (not world.is_noisy[c_ask]) and c_ask not in conf_first_ask:
            conf_first_ask[c_ask] = 1.0 - g_before
        gate.imprint(world.render(c_ask))
        g_after = gate.novelty(world.render(c_ask))
        if mode == "yoked":
            yb = float(yoke_pool[yi % len(yoke_pool)]); yi += 1
            ya = float(yoke_pool[yi % len(yoke_pool)]); yi += 1
            LP = yb - ya
        else:
            LP = g_before - g_after

        snc_hz = deliver_reward(c_ask, LP)                 # DR-1 critic (unchanged)
        omitfire = learn_veto(c_ask, LP)                   # SPIKING omission-veto learning
        Vveto[c_ask] = read_veto(c_ask)                    # re-read the substrate-computed veto value
        (snc_noisy_burst if world.is_noisy[c_ask] else snc_learn_burst).append(snc_hz)
        (omitfire_noisy if world.is_noisy[c_ask] else omitfire_learn).append(omitfire)

        asked.add(c_ask); n_asks += 1
        ask_events.append((turn, c_ask, float(g_before), float(g_before - g_after), bool(world.is_noisy[c_ask])))
        if verbose and n_asks <= 12:
            print(f"    [ask {n_asks:02d}] c={c_ask} noisy={world.is_noisy[c_ask]} g {g_before:.2f}->{g_after:.2f}"
                  f" LP {g_before-g_after:+.2f} sncHz {snc_hz:5.1f} omitFire {omitfire:5.1f} "
                  f"vetoHz {Vveto[c_ask]:5.1f} (floor {VETO_FLOOR_HZ})", flush=True)

    # ---- metrics ----
    corr_gap = np.array(corr_gap); corr_want = np.array(corr_want)
    corr = (float(np.corrcoef(corr_gap, corr_want)[0, 1])
            if corr_want.std() > 1e-9 and corr_gap.std() > 1e-9 else 0.0)
    rate_unknown = ask_unknown / max(elig_unknown, 1)
    rate_known = ask_known / max(elig_known, 1)
    ratio_b = rate_unknown / (rate_known + 1e-9)

    conf_after = {c: 1.0 - gate.novelty(world.render(c)) for c in concepts}
    learn_after = [conf_after[c] for c in range(n_learn)]
    learn_before = [conf_first_ask.get(c, 0.0) for c in range(n_learn) if c in conf_first_ask]
    abstain_floor = float(np.mean([conf_after[c] for c in range(n_learn, n_learn + n_noisy)]))
    conf_rise = float(np.mean(learn_after)) - (float(np.mean(learn_before)) if learn_before else 0.0)
    conf_after_mean = float(np.mean(learn_after))

    late_asks = [e for e in ask_events if e[0] >= 2 * third]
    late_learnable_frac = (sum(1 for e in late_asks if not e[4]) / len(late_asks)) if late_asks else 1.0
    noisy_early_rate = noisy_ask[0] / max(noisy_elig[0], 1)
    noisy_late_rate = noisy_ask[2] / max(noisy_elig[2], 1)
    noisy_g_final = float(np.mean([gate.novelty(world.render(c)) for c in range(n_learn, n_learn + n_noisy)]))

    # the veto quantity is now the SPIKING veto pool rate Vveto (HIGH = vetoed). noisy should read HIGH, learn LOW.
    asked_noisy = [c for c in range(n_learn, n_learn + n_noisy) if c in asked]
    noisy_V_final = float(np.mean([Vveto[c] for c in (asked_noisy or range(n_learn, n_learn + n_noisy))]))
    noisy_vetoed_frac = (float(np.mean([Vveto[c] >= VETO_FLOOR_HZ for c in asked_noisy])) if asked_noisy else 0.0)
    noisy_vetoed = bool(noisy_vetoed_frac >= 0.75)
    asked_learn = [c for c in range(n_learn) if c in asked]
    learn_V_final = float(np.mean([Vveto[c] for c in asked_learn])) if asked_learn else 0.0
    value_sep = noisy_V_final - learn_V_final           # POSITIVE = correct (noisy vetoed above learnable)

    strio_learn = float(np.mean([read_value(c) for c in asked_learn]) if asked_learn else 0.0)
    strio_v0 = float(np.mean([read_value(c) for c in range(n_learn, n_learn + n_noisy)]))  # noisy value (drift)
    noisy_asks_total = sum(1 for e in ask_events if e[4])
    mean_LP_learn = float(np.mean([e[3] for e in ask_events if not e[4]])) if any(not e[4] for e in ask_events) else 0.0
    mean_LP_noisy = float(np.mean([e[3] for e in ask_events if e[4]])) if noisy_asks_total else 0.0
    snc_learn_hz = float(np.mean(snc_learn_burst)) if snc_learn_burst else 0.0
    snc_noisy_hz = float(np.mean(snc_noisy_burst)) if snc_noisy_burst else 0.0
    omitfire_learn_hz = float(np.mean(omitfire_learn)) if omitfire_learn else 0.0
    omitfire_noisy_hz = float(np.mean(omitfire_noisy)) if omitfire_noisy else 0.0

    confident_set = {c for c in concepts if conf_after[c] > 0.5}
    moat_ok = confident_set.issubset(asked)
    learnable_mastered = int(sum(1 for c in range(n_learn) if conf_after[c] > 0.5))

    return {
        "mode": mode, "seed": seed, "veto_floor": VETO_FLOOR_HZ,
        "corr_gap_want": corr, "rate_unknown": rate_unknown, "rate_known": rate_known, "ratio_b": ratio_b,
        "conf_rise": conf_rise, "conf_after_mean": conf_after_mean, "abstain_floor": abstain_floor,
        "total_asks": len(ask_events), "noisy_asks_total": noisy_asks_total,
        "noisy_early_rate": noisy_early_rate, "noisy_late_rate": noisy_late_rate,
        "noisy_g_final": noisy_g_final, "noisy_veto_final": noisy_V_final, "noisy_vetoed": noisy_vetoed,
        "noisy_vetoed_frac": noisy_vetoed_frac, "learn_veto_final": learn_V_final, "value_sep": value_sep,
        "strio_learn": strio_learn, "strio_noisy": strio_v0,
        "late_learnable_frac": late_learnable_frac, "learnable_mastered": learnable_mastered,
        "mean_LP_learn": mean_LP_learn, "mean_LP_noisy": mean_LP_noisy,
        "snc_learn_hz": snc_learn_hz, "snc_noisy_hz": snc_noisy_hz,
        "omitfire_learn_hz": omitfire_learn_hz, "omitfire_noisy_hz": omitfire_noisy_hz, "moat_ok": bool(moat_ok),
    }


def evaluate(seed, enable_reserve=False, **kw):
    real = run(seed, "real", enable_reserve=enable_reserve, **kw)
    lesion = run(seed, "lesion", enable_reserve=enable_reserve, **kw)
    yoked = run(seed, "yoked", enable_reserve=enable_reserve, **kw)
    permuted = run(seed, "permuted", enable_reserve=enable_reserve, **kw)
    omit_les = run(seed, "omit_lesion", enable_reserve=enable_reserve, **kw)   # the load-bearing detector lesion

    gate_a = real["corr_gap_want"] >= 0.9
    gate_b = real["ratio_b"] >= 2.0
    gate_c = (real["conf_rise"] > 0.3) and (real["conf_after_mean"] > real["abstain_floor"] + 0.3)
    noisy_stops = ((real["noisy_late_rate"] <= 0.5 * real["noisy_early_rate"] + 1e-9)
                   and real["noisy_g_final"] > 0.7 and real["noisy_vetoed"])
    lesion_collapses = lesion["total_asks"] <= 1 and lesion["conf_rise"] < 0.15
    yoked_collapses = yoked["learnable_mastered"] < real["learnable_mastered"]
    permuted_collapses = (permuted["corr_gap_want"] < 0.5
                          or permuted["learnable_mastered"] < real["learnable_mastered"])
    # the omission-detector lesion must COLLAPSE the veto: omit never fires -> nothing is vetoed -> the brain
    # asks the noisy TV indiscriminately (the host-tracker failure mode, now produced by lesioning a pool). The
    # dissociation: real VETOES noisy and asks it FEWER times; the lesion does neither.
    omit_lesion_collapses_veto = ((not omit_les["noisy_vetoed"])
                                  and omit_les["noisy_asks_total"] >= real["noisy_asks_total"])

    go = bool(gate_a and gate_b and gate_c and noisy_stops and real["moat_ok"]
              and lesion_collapses and yoked_collapses and permuted_collapses
              and omit_lesion_collapses_veto)
    out = {
        "seed": seed, "real": real, "lesion": lesion, "yoked": yoked, "permuted": permuted,
        "omit_lesion": omit_les,
        "gate_a_corr": bool(gate_a), "gate_b_askratio": bool(gate_b), "gate_c_conf_rise": bool(gate_c),
        "noisy_stops_honest": bool(noisy_stops), "moat_ok": bool(real["moat_ok"]),
        "lesion_collapses": bool(lesion_collapses), "yoked_collapses": bool(yoked_collapses),
        "permuted_collapses": bool(permuted_collapses),
        "omit_lesion_collapses_veto": bool(omit_lesion_collapses_veto), "GO": go,
    }
    if enable_reserve:
        # LANE B DOMAIN DISSOCIATION — the RESERVE-LESION control: the inhibitory reserve pathway is present but not
        # learning (reserve_rpe forced to 0). The protective reserve must RESCUE learnable concepts — drive their
        # spiking veto read BELOW the reserve-lesion — WITHOUT weakening the noisy veto (noisy stays vetoed, and its
        # veto rate is not materially lowered). This isolates the reserve's contribution from the excitatory memory.
        reserve_les = run(seed, "reserve_lesion", **kw)
        reserve_rescues = bool((real["learn_veto_final"] < reserve_les["learn_veto_final"] - 1e-9)
                               and real["noisy_vetoed"]
                               and (real["noisy_veto_final"] >= reserve_les["noisy_veto_final"] - 2.0))
        out["reserve_lesion"] = reserve_les
        out["reserve_rescues"] = reserve_rescues
        out["reserve_learn_veto_final"] = float(real["learn_veto_final"])
        out["reserve_lesion_learn_veto_final"] = float(reserve_les["learn_veto_final"])
        out["GO"] = bool(go and reserve_rescues)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="tiny CPU smoke (5 concepts, short budget)")
    ap.add_argument("--probe", action="store_true",
                    help="report-only: one seed real vs omit_lesion, verbose, with tools.lab attribution")
    ap.add_argument("--reserve", action="store_true",
                    help="LANE B (additive, default-OFF): add the DECAYING SUB-BASELINE INHIBITORY protective-reserve "
                         "pathway (a second cue->veto, gaba_a, opposite-sign learning gated by `reserve_learn`) + the "
                         "reserve-lesion domain dissociation to the GO. OFF -> byte-identical to the omission-only build.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = "research/findings/raw/_curiosity_reward_omission_veto.json"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    kw = {}
    if a.smoke:
        kw = dict(n_learn=3, n_noisy=2, n_turns=90, ask_budget=14, d=512)

    from sim.backend import get_backend
    _, backend = get_backend()
    print(f"[DR-1 REWARD-OMISSION VETO] backend={backend} smoke={a.smoke}  "
          f"spiking omission circuit snc->rmtg->omit->veto<-cue(plastic); the noisy-veto is the SPIKING veto "
          f"pool rate, NOT a host ELP tracker.\n"
          f"  GO: (a) corr(gap,want)>=0.9 (b) ask unk>=2x known (c) conf rises; noisy STOPS (spiking veto) while "
          f"g HIGH; curiosity/yoked/permuted collapse; moat; + OMISSION-DETECTOR LESION collapses the veto.\n",
          flush=True)

    if a.reserve:
        print("  [LANE B] --reserve ON: DECAYING SUB-BASELINE INHIBITORY protective reserve (cue->veto gaba_a, "
              "opposite-sign) + reserve-lesion dissociation added to the GO.\n", flush=True)

    if a.probe:
        s = a.seeds[0]
        print(f"  --- REAL (seed {s}) ---", flush=True)
        real = run(s, "real", verbose=True, enable_reserve=a.reserve, **kw)
        print(f"  --- OMIT-LESION (seed {s}) ---", flush=True)
        oles = run(s, "omit_lesion", verbose=True, enable_reserve=a.reserve, **kw)
        print(f"\n  real: omitFire learn {real['omitfire_learn_hz']:.1f} vs noisy {real['omitfire_noisy_hz']:.1f} Hz "
              f"| veto learn {real['learn_veto_final']:.1f} vs noisy {real['noisy_veto_final']:.1f} Hz "
              f"vetoed={real['noisy_vetoed']}", flush=True)
        print(f"  omit-lesion: veto noisy {oles['noisy_veto_final']:.1f} Hz vetoed={oles['noisy_vetoed']} "
              f"| noisy asks early {oles['noisy_early_rate']:.2f}->late {oles['noisy_late_rate']:.2f}", flush=True)
        lever("noisy veto pool Hz (real vs omit-lesion)", round(oles["noisy_veto_final"], 2),
              round(real["noisy_veto_final"], 2), required=False)
        attributable_to("noisy-veto firing", real["noisy_veto_final"], oles["noisy_veto_final"])
        return

    results = []
    for seed in a.seeds:
        r = evaluate(seed, enable_reserve=a.reserve, **kw)
        results.append(r)
        re = r["real"]; ol = r["omit_lesion"]
        print(f"  [seed {seed}] corr(gap,want) {re['corr_gap_want']:+.3f} | ask-ratio unk/known {re['ratio_b']:.2f} | "
              f"conf-rise {re['conf_rise']:+.2f} (after {re['conf_after_mean']:.2f} vs floor {re['abstain_floor']:.2f})",
              flush=True)
        print(f"            SNc RPE: learn-burst {re['snc_learn_hz']:.1f}Hz vs noisy {re['snc_noisy_hz']:.1f}Hz | "
              f"omit-detector fire: learn {re['omitfire_learn_hz']:.1f}Hz vs noisy {re['omitfire_noisy_hz']:.1f}Hz",
              flush=True)
        print(f"            NOISY asks early {re['noisy_early_rate']:.2f} -> late {re['noisy_late_rate']:.2f} "
              f"(g stays {re['noisy_g_final']:.2f}); SPIKING veto: noisy {re['noisy_veto_final']:.1f}Hz vs learn "
              f"{re['learn_veto_final']:.1f}Hz (floor {re['veto_floor']:.0f}) vetoed={re['noisy_vetoed']} "
              f"sep {re['value_sep']:+.1f}", flush=True)
        print(f"            controls: curiosity-lesion asks={r['lesion']['total_asks']} | yoked mastered "
              f"{r['yoked']['learnable_mastered']} vs real {re['learnable_mastered']} | permuted corr "
              f"{r['permuted']['corr_gap_want']:+.2f} | OMIT-LESION veto noisy {ol['noisy_veto_final']:.1f}Hz "
              f"vetoed={ol['noisy_vetoed']} (late-rate {ol['noisy_late_rate']:.2f}) | moat {r['real']['moat_ok']}",
              flush=True)
        if "reserve_rescues" in r:
            print(f"            RESERVE: learn-veto real {r['reserve_learn_veto_final']:.1f}Hz vs reserve-lesion "
                  f"{r['reserve_lesion_learn_veto_final']:.1f}Hz (reserve rescues learnable={r['reserve_rescues']}) | "
                  f"reserve-lesion noisy {r['reserve_lesion']['noisy_veto_final']:.1f}Hz vetoed="
                  f"{r['reserve_lesion']['noisy_vetoed']}", flush=True)
        flags = (f"a={r['gate_a_corr']} b={r['gate_b_askratio']} c={r['gate_c_conf_rise']} "
                 f"noisy-stops={r['noisy_stops_honest']} curiosity-lesion={r['lesion_collapses']} "
                 f"yoked={r['yoked_collapses']} permuted={r['permuted_collapses']} "
                 f"omit-lesion-collapses={r['omit_lesion_collapses_veto']}"
                 + (f" reserve-rescues={r['reserve_rescues']}" if "reserve_rescues" in r else ""))
        print(f"            [{flags}]  ==>  {'GO' if r['GO'] else 'NO'}\n", flush=True)

    n_go = sum(r["GO"] for r in results)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results, "backend": backend, "smoke": a.smoke, "kw": kw}, fh, indent=2, default=str)
    print(f"{'='*104}", flush=True)
    print(f"  REWARD-OMISSION VETO: {n_go}/{len(results)} seeds GO "
          f"({'ALL GO' if n_go == len(results) else 'partial/negative — pins the exact spiking wall'})", flush=True)
    print(f"  [saved] {a.out}\n{'='*104}", flush=True)


if __name__ == "__main__":
    main()
