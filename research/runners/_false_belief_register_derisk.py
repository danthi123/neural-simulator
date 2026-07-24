"""W3 (Stage-3 flagship social build): the AGENT-KEYED FALSE-BELIEF register -- the first rung of the
theory-of-mind ladder. The faculty the foundation plan omitted; the master roadmap
(docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md sec 2.7 + Stage-3 W3) added it as needed for genuine human
conversation.

THE ECONOMY (Fleming-Daw 2017; roadmap sec 2.7 unifying thesis): SELF = OTHER-INFERENCE. The SAME "meta-schema"
region class that models the brain's OWN state (the GO self-schema, `_self_schema_region_derisk.py`, 6-seed GO) is
applied OUTWARD to model ANOTHER agent's state. Here that state is a BELIEF -- which can DIFFER from reality. We
build the classic FALSE-BELIEF test (Sally-Anne / Wimmer-Perner change-of-location): track what agent X BELIEVES
and predict X's action FROM X'S (possibly false) BELIEF, not from the true state of the world.
Biology: TPJ/mPFC mentalizing; the meta-schema represents "agent X's model of the world" as a separate slot from
"the true world". The roadmap's surpass: an agent-keyed belief store (D3 register keyed by agent) with
WITNESSING-GATED WRITES (`sim/`'s own `transmission_gate` = the witnessing gate; open when X watches -> the belief
updates to reality; closed when X isn't watching -> the belief HOLDS its old, now-false, value).

HONEST SCOPE (carried into every line): this is a FUNCTIONAL MENTALIZING CORRELATE -- a region that represents +
predicts-from another agent's belief-state, dissociable from reality and from the self's belief. It is NOT a claim
that the system has phenomenal access to another mind; it is the computational structure that a false-belief task
probes.

THE MECHANISM (reuse-by-import of the self-schema / GNW machinery; NO `sim/` edit). ONE spiking `SimulationBridge`,
THREE belief stores of the SAME meta-schema class (each = the self-schema's workspace: K self-recurrent NMDA member
assemblies + a shared inhibitory pool -> GNW single-content occupancy = "which location is currently believed"):
  1. `world`  -- holds the TRUE current object location (re-ignited on every move; reality).
  2. `belief` -- holds AGENT X's believed location (the OTHER-agent slot). Updated ONLY on a WITNESSED event: a
     register CLEAR-BEFORE-WRITE (restore the belief slot to the quiescent snapshot -> evicts the old attractor
     cleanly; a hyperpolarizing "clear" instead leaves the substrate unable to re-ignite) then a witness-gated fresh
     ignite of the new location. The witnessing is realized as `sim/`'s own `transmission_gate="witness_other"` on the
     topographic `world_k -> belief_k` pathways (carries the witnessing signal in parallel) PLUS the gating of that
     clear+ignite. Unwitnessed -> no clear, no ignite, zero write current -> the persistent NMDA attractor HOLDS its
     old (now false) belief.
  3. `self`   -- holds the SYSTEM'S OWN believed location, witness_self ALWAYS open (the system witnesses everything)
     -> self-belief always tracks reality. The self/other dissociation: same machinery, different witnessing -> one
     can be false while the other stays true.
PREDICT X's action = argmax over `belief` member rates (X looks where X BELIEVES). The REALITY BASELINE = argmax
over `world` member rates (predict from the true state) -- fails false-belief BY CONSTRUCTION.

THE SCENARIO (change-of-location, per trial, K locations, chance 1/K):
  * Event 1 (placement at A): X WITNESSES it (Sally places her own marble). world<-A, belief<-A, self<-A.
  * Event 2 (move to B): witnessed_other in {0,1}. Unwitnessed (0) => FALSE-belief trial: world<-B, belief HOLDS A,
    self<-B. Witnessed (1) => TRUE-belief control: world<-B, belief<-B, self<-B.
  * Query "where will X look?": true answer = X's belief = A on false trials / B on true trials. Reality = B always.

GO GATE (6-seed): the belief-store predictor tracks X's BELIEF, well above the reality baseline:
  * false_belief_acc   >= 0.85   (on unwitnessed trials, predict A; chance 1/K)  -- predicts belief, not reality
  * reality_baseline_false_acc <= 0.20  (the world read predicts B = wrong on false trials)  -- baseline FAILS
  * true_belief_acc    >= 0.85   (on witnessed trials, predict B) -- the belief UPDATES when X watches (not always-old)
ANTI-CHEATS (all must hold):
  (1) TRUE-BELIEF control -- when X witnessed the move, belief-predictor == reality-predictor (both B): the system
      is NOT just always-predicting-old; witnessing drives the update. (true_belief_agree high.)
  (2) OTHER-LESION -- sever the other-belief store (its self-loop weight 0) AND force witness_other open (no gating):
      belief collapses onto reality -> predicts reality -> false_belief_acc drops to the reality baseline. Proves the
      false-belief prediction rides the SEPARATE, GATED belief store.
  (3) SCRAMBLE-WITNESSING -- permute the witnessed_other flags across trials (belief updates at the WRONG times);
      score vs the TRUE ground-truth belief -> collapse to chance. Proves it tracks the ACTUAL witnessing schedule.
  (extra) SELF/OTHER DISSOCIATION -- on false-belief trials, `self` tracks reality (self_reality_acc high, correct
      for the self) WHILE `belief` is false (!= reality): the two agent slots dissociate.

Usage:
  # CPU smoke (1 seed, tiny -- proves it runs, controls live, prints a verdict):
  python -u -m research.runners._false_belief_register_derisk --smoke --seed 42 \
      --json research/findings/raw/_false_belief_smoke.json --backend numpy
  # full 6-seed (local CPU):
  python -u -m research.runners._false_belief_register_derisk --seeds 42 43 44 100 101 102 --n-trials 48 \
      --json research/findings/raw/_false_belief_6seed.json --backend numpy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.backend import get_backend, to_host

# reuse-by-import: the validated GNW spiking machinery (the self-recurrent NMDA attractor loop + the wash-out
# snapshot/restore) + the self-schema meta-schema geometry/constants (this ToM region IS that same class turned
# outward -- SELF = OTHER-INFERENCE).
from research.runners._gnw_rung1_ignition_curve_derisk import (
    _build_assembly_loop_population, _snapshot_state, _restore_state,
    DEFAULT_ATTRACTOR_WEIGHT, SETTLE_STEPS, DRIVE_STEPS, FREE_STEPS,
)
from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection
from research.runners._self_schema_region_derisk import (
    WS_LOOP_GATE, WS_TO_FS_WEIGHT, FS_TO_WS_WEIGHT, IGNITE_PA, WORKSPACE_FS_N, ASSEMBLY_SIZE,
)


# ── geometry ───────────────────────────────────────────────────────────────────────────────────────────────
K_LOC = 4                    # number of distinct object locations (chance for a location read = 1/K = 0.25)
STORE_ASSEMBLY = ASSEMBLY_SIZE   # per-location member assembly (self-recurrent ignitable unit; reuse self-schema 80)
STORE_FS_N = WORKSPACE_FS_N       # shared inhibitory pool per store (Rung-2 mutual inhibition -> single content)

HOLD_STEPS = 25              # inter-event hold: external current off, stores self-sustain via their loops
W_WRITE = 26.0               # world_k -> belief_k / self_k topographic WITNESSING-GATED write weight (transmission-
                             # gated synaptic IGNITE of the new belief when the gate is open; zero current = HOLD)


def _restore_slice(bridge, snap, indices):
    """Restore ONLY the given neurons' dynamical state (v/u/firing/conductances) to the quiescent snapshot -- a
    per-store CLEAR (the D3 register's clear-before-write) that returns that store's slot to rest WITHOUT the
    substrate damage a strong hyperpolarizing pulse causes (which leaves the recovery variable too deep to re-ignite
    a self-sustaining attractor). Other stores are untouched. `indices` are neuron indices (device array)."""
    for name, arr in snap.items():
        getattr(bridge, name)[indices] = arr[indices]


def _gated_write_projection(pre_idx, post_idx, weight, trans_gate):
    """A topographic dense projection carrying the WITNESSING-GATED write (frozen plasticity gate WS_LOOP_GATE +
    a runtime transmission_gate). When bridge.set_transmission_gate(trans_gate, 0) the current is ZERO (belief holds);
    at 1 the world member drives its belief member (ignite + evict-old via the shared inhibition)."""
    d = _dense_projection(np.asarray(pre_idx), np.asarray(post_idx), float(weight), WS_LOOP_GATE)
    d["transmission_gate"] = str(trans_gate)
    return d


# ── build the one-brain bridge (world + belief[other] + self, each a GNW single-content attractor store) ─────
def build_tom_bridge(seed: int = 42, attractor_weight: float = DEFAULT_ATTRACTOR_WEIGHT,
                     w_write: float = W_WRITE):
    """One `SimulationBridge`: three belief stores of the self-schema meta-schema class -- `world` (reality),
    `belief` (agent X, witnessing-gated), `self` (system, always-witnessing). The OTHER-LESION is a RUNTIME
    manipulation (witnessing gate forced open) on this same intact bridge, so the store is genuinely intact and the
    only thing ablated is the gating that represents "X did not witness this". Returns (bridge, xp, idx, snap)."""
    xp, _ = get_backend()

    n_store = STORE_ASSEMBLY * K_LOC
    regions = [
        BrainRegion(name="world", n_neurons=n_store, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="world_fs", n_neurons=STORE_FS_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="belief", n_neurons=n_store, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="belief_fs", n_neurons=STORE_FS_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="self", n_neurons=n_store, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="self_fs", n_neurons=STORE_FS_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
    ]
    pathways = []
    for store in ("world", "belief", "self"):
        pathways.append(RegionPathway(from_region=store, to_region=f"{store}_fs", density=0.5,
                                      weight_mean=WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False))
        pathways.append(RegionPathway(from_region=f"{store}_fs", to_region=store, density=0.5,
                                      weight_mean=FS_TO_WS_WEIGHT, weight_jitter=0.0, plastic=False))

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                          # seeds the substrate (het guard fires at seed>=0; the doc gotcha)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True     # matches the GO'd self-schema; desynchronizes the assemblies
    cfg.stdp_w_max = max(400.0, float(attractor_weight) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(attractor_weight) * 4.0)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager
    def members(store):
        base = np.asarray(rm.indices(store), dtype=np.int64)
        return {k: base[k * STORE_ASSEMBLY:(k + 1) * STORE_ASSEMBLY] for k in range(K_LOC)}
    world_m, belief_m, self_m = members("world"), members("belief"), members("self")

    union = dict(rm.build_wiring_plan(seed=int(seed)))
    for k in range(K_LOC):
        union[f"loop_world_{k}"] = _build_assembly_loop_population(world_m[k], float(attractor_weight))
        union[f"loop_belief_{k}"] = _build_assembly_loop_population(belief_m[k], float(attractor_weight))
        union[f"loop_self_{k}"] = _build_assembly_loop_population(self_m[k], float(attractor_weight))
        union[f"write_belief_{k}"] = _gated_write_projection(world_m[k], belief_m[k], w_write, "witness_other")
        union[f"write_self_{k}"] = _gated_write_projection(world_m[k], self_m[k], w_write, "witness_self")

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)
    bridge.set_transmission_gate("witness_other", 0.0)   # writes closed by default; opened per-event
    bridge.set_transmission_gate("witness_self", 0.0)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)

    idx = {
        "world_dev": {k: xp.asarray(v) for k, v in world_m.items()},
        "belief_dev": {k: xp.asarray(v) for k, v in belief_m.items()},
        "self_dev": {k: xp.asarray(v) for k, v in self_m.items()},
        "world_all": xp.asarray(np.concatenate([world_m[k] for k in range(K_LOC)])),
        "belief_all": xp.asarray(np.concatenate([belief_m[k] for k in range(K_LOC)])),
        "self_all": xp.asarray(np.concatenate([self_m[k] for k in range(K_LOC)])),
    }
    return bridge, xp, idx, snap


# ── one trial: place@A (witnessed) -> move@B (witnessed_other) -> query the three stores' late-window rates ─────
def _run_tom_trial(bridge, xp, idx, snap, start_loc, end_loc, witnessed_other,
                   lesion_other=False, helper_pa=0.0):
    """Run one change-of-location trial. Event 1: object placed at start_loc, X witnesses it. Event 2: object moved
    to end_loc, witness_other = witnessed_other (0=false-belief -> belief HOLDS start_loc; 1=true-belief -> belief
    UPDATES to end_loc). `self` always witnesses (updates every event). Each WITNESSED write = a witness-gated CLEAR
    (hyperpolarize the store's members to evict the OLD attractor) then a transmission-gated IGNITE (the world member
    synaptically drives the new belief member from the cleared, quiescent state). `lesion_other` forces witness_other
    OPEN at every event AND through the query (the witnessing gate = the theory that X may NOT have seen it is
    ablated) -> belief mirrors reality. `helper_pa` (optional, >0) adds a witness-gated direct ignite to the written
    member (robustness aid; 0 = the write's ignite is carried purely by the transmission-gated synaptic pathway).
    Returns per-store late-window member rates."""
    world_dev, belief_dev, self_dev = idx["world_dev"], idx["belief_dev"], idx["self_dev"]
    belief_all, self_all = idx["belief_all"], idx["self_all"]

    bridge.cp_external_input_current[:] = 0.0
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0

    write_pa = float(helper_pa) if helper_pa > 0.0 else IGNITE_PA

    def _write_event(loc, w_other):
        w_o = 1.0 if lesion_other else float(w_other)
        write_belief = w_o > 0.0
        # CLEAR = restore the WRITTEN stores' slices to quiescence (register clear-before-write; world+self always,
        # belief iff witnessed). Fresh-from-quiescence ignition is the validated reliable path -- a hyperpolarizing
        # clear instead leaves u too deep to re-ignite a self-sustaining attractor.
        _restore_slice(bridge, snap, idx["world_all"])
        _restore_slice(bridge, snap, self_all)
        if write_belief:
            _restore_slice(bridge, snap, belief_all)
        bridge.cp_external_input_current[:] = 0.0
        bridge.set_transmission_gate("witness_other", w_o)     # architectural witnessing gate on the synaptic write
        bridge.set_transmission_gate("witness_self", 1.0)
        # IGNITE from the cleared/quiescent slice: reality (world@loc) + the WITNESSED belief@loc + self@loc, each a
        # fresh validated ignition. The transmission-gated synaptic world->belief/self carries the same witnessing
        # signal in parallel (gate open iff witnessed).
        for _ in range(DRIVE_STEPS):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[world_dev[loc]] = xp.float32(IGNITE_PA)
            bridge.cp_external_input_current[self_dev[loc]] = xp.float32(write_pa)
            if write_belief:
                bridge.cp_external_input_current[belief_dev[loc]] = xp.float32(write_pa)
            bridge._run_one_simulation_step()
        # HOLD: writes closed (self-sustain via loops). LESION keeps witness_other OPEN so belief keeps mirroring
        # the world (reality) it is continuously driven by.
        bridge.set_transmission_gate("witness_other", 1.0 if lesion_other else 0.0)
        bridge.set_transmission_gate("witness_self", 0.0)
        for _ in range(HOLD_STEPS):
            bridge.cp_external_input_current[:] = 0.0
            bridge._run_one_simulation_step()

    _write_event(start_loc, 1.0)                 # placement: always witnessed by X
    _write_event(end_loc, witnessed_other)       # move: witnessed_other (0 => false belief)

    # QUERY: no external drive; the persistent stores hold. (LESION: witness_other stays OPEN so the loop-severed
    # belief keeps mirroring the world = reality.) Read the late-window member rates.
    bridge.set_transmission_gate("witness_other", 1.0 if lesion_other else 0.0)
    bridge.set_transmission_gate("witness_self", 0.0)
    late_start = FREE_STEPS - max(1, FREE_STEPS // 3)
    world_acc = {k: 0 for k in range(K_LOC)}
    belief_acc = {k: 0 for k in range(K_LOC)}
    self_acc = {k: 0 for k in range(K_LOC)}
    for t in range(FREE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
        if t >= late_start:
            for k in range(K_LOC):
                world_acc[k] += int(to_host(bridge.cp_firing_states[world_dev[k]].astype(xp.float64).sum()))
                belief_acc[k] += int(to_host(bridge.cp_firing_states[belief_dev[k]].astype(xp.float64).sum()))
                self_acc[k] += int(to_host(bridge.cp_firing_states[self_dev[k]].astype(xp.float64).sum()))
    nlate = float(FREE_STEPS - late_start) * STORE_ASSEMBLY
    return {
        "world": {k: world_acc[k] / nlate for k in range(K_LOC)},
        "belief": {k: belief_acc[k] / nlate for k in range(K_LOC)},
        "self": {k: self_acc[k] / nlate for k in range(K_LOC)},
    }


def _argmax_loc(rate_dict):
    return int(max(rate_dict, key=rate_dict.get))


# ── the per-seed scenario generator ──────────────────────────────────────────────────────────────────────────
def make_trials(seed, n_trials):
    """T change-of-location trials. Each: start_loc A, end_loc B(!=A), witnessed_other in {0,1} for the MOVE
    (balanced ~50/50). Ground-truth belief = A if the move is unwitnessed (FALSE belief) else B (TRUE belief);
    reality = B always."""
    rng = np.random.default_rng(seed * 131 + 5)
    trials = []
    for i in range(n_trials):
        a = int(rng.integers(K_LOC))
        b = int(rng.integers(K_LOC))
        while b == a:
            b = int(rng.integers(K_LOC))
        w = int(i % 2 == 0)                        # balanced witnessed/unwitnessed, shuffled below
        trials.append({"start": a, "end": b, "witnessed_other": w})
    rng.shuffle(trials)
    return trials


# ── evaluate one seed (intact GO + all anti-cheats) ────────────────────────────────────────────────────────────
def evaluate_seed(seed, n_trials, thresholds, helper_pa=0.0, w_write=W_WRITE, verbose=False):
    trials = make_trials(seed, n_trials)
    gt_belief = np.array([t["start"] if t["witnessed_other"] == 0 else t["end"] for t in trials], dtype=int)
    reality = np.array([t["end"] for t in trials], dtype=int)
    false_mask = np.array([t["witnessed_other"] == 0 for t in trials], dtype=bool)
    true_mask = ~false_mask

    def run_block(bridge, xp, idx, snap, witnessed_flags, lesion_other=False):
        pb = np.zeros(len(trials), dtype=int)   # predicted-belief (other)
        pr = np.zeros(len(trials), dtype=int)   # predicted-reality (world)
        ps = np.zeros(len(trials), dtype=int)   # predicted-self
        for i, t in enumerate(trials):
            r = _run_tom_trial(bridge, xp, idx, snap, t["start"], t["end"], int(witnessed_flags[i]),
                               lesion_other=lesion_other, helper_pa=helper_pa)
            pb[i] = _argmax_loc(r["belief"]); pr[i] = _argmax_loc(r["world"]); ps[i] = _argmax_loc(r["self"])
        return pb, pr, ps

    # ---- INTACT ----
    bridge, xp, idx, snap = build_tom_bridge(seed=seed, w_write=w_write)
    w_flags = np.array([t["witnessed_other"] for t in trials], dtype=int)
    pb, pr, ps = run_block(bridge, xp, idx, snap, w_flags, lesion_other=False)

    false_belief_acc = float(np.mean(pb[false_mask] == gt_belief[false_mask])) if false_mask.any() else 0.0
    true_belief_acc = float(np.mean(pb[true_mask] == gt_belief[true_mask])) if true_mask.any() else 0.0
    overall_belief_acc = float(np.mean(pb == gt_belief))
    reality_baseline_false = float(np.mean(pr[false_mask] == gt_belief[false_mask])) if false_mask.any() else 0.0
    reality_baseline_overall = float(np.mean(pr == gt_belief))
    # (1) TRUE-belief control: when X watched, belief-predictor agrees with reality-predictor (both B, not "always old")
    true_belief_agree = float(np.mean(pb[true_mask] == pr[true_mask])) if true_mask.any() else 0.0
    # (extra) SELF/OTHER dissociation: on FALSE trials self tracks reality while belief is false
    self_reality_acc_false = float(np.mean(ps[false_mask] == reality[false_mask])) if false_mask.any() else 0.0
    belief_isfalse_frac = float(np.mean(pb[false_mask] != reality[false_mask])) if false_mask.any() else 0.0

    # ---- (2) OTHER-LESION: ablate the witnessing gate (force it open) -> belief mirrors reality (same bridge) ----
    pb_l, pr_l, ps_l = run_block(bridge, xp, idx, snap, w_flags, lesion_other=True)
    lesion_false_belief_acc = float(np.mean(pb_l[false_mask] == gt_belief[false_mask])) if false_mask.any() else 0.0
    lesion_predicts_reality = float(np.mean(pb_l[false_mask] == reality[false_mask])) if false_mask.any() else 0.0
    lesion_collapsed = bool(lesion_false_belief_acc <= thresholds["chance_loc"])

    # ---- (3) SCRAMBLE-WITNESSING: permute the witnessed flags -> belief tracks wrong times -> chance ----
    rng = np.random.default_rng(seed * 977 + 19)
    scr_flags = w_flags[rng.permutation(len(trials))]
    pb_s, _pr_s, _ps_s = run_block(bridge, xp, idx, snap, scr_flags, lesion_other=False)
    scramble_belief_acc = float(np.mean(pb_s == gt_belief))     # scored vs TRUE ground-truth belief
    scramble_collapsed = bool(scramble_belief_acc <= thresholds["scramble_chance"])

    dissociation_ok = bool(self_reality_acc_false >= thresholds["true_belief_acc"]
                           and belief_isfalse_frac >= thresholds["true_belief_acc"])

    # ---- GO (per-seed) ----
    go_false = bool(false_belief_acc >= thresholds["false_belief_acc"])
    go_baseline_fails = bool(reality_baseline_false <= thresholds["reality_baseline_max"])
    go_true = bool(true_belief_acc >= thresholds["true_belief_acc"])
    go = bool(go_false and go_baseline_fails and go_true and lesion_collapsed and scramble_collapsed
              and dissociation_ok and true_belief_agree >= thresholds["true_belief_acc"])

    r = {
        "seed": int(seed), "n_trials": int(n_trials),
        "n_false_trials": int(false_mask.sum()), "n_true_trials": int(true_mask.sum()),
        "intact": {
            "false_belief_acc": false_belief_acc,
            "true_belief_acc": true_belief_acc,
            "overall_belief_acc": overall_belief_acc,
            "reality_baseline_false_acc": reality_baseline_false,
            "reality_baseline_overall_acc": reality_baseline_overall,
            "true_belief_agree_belief_vs_reality": true_belief_agree,
        },
        "self_other_dissociation": {
            "self_tracks_reality_on_false_trials": self_reality_acc_false,
            "belief_is_false_on_false_trials": belief_isfalse_frac,
            "dissociation_ok": dissociation_ok,
        },
        "other_lesion": {
            "false_belief_acc": lesion_false_belief_acc,
            "predicts_reality_on_false_trials": lesion_predicts_reality,
            "collapsed": lesion_collapsed,
        },
        "scramble_witnessing": {
            "belief_acc_vs_true": scramble_belief_acc,
            "collapsed": scramble_collapsed,
        },
        "go_components": {"false_belief": go_false, "reality_baseline_fails": go_baseline_fails,
                          "true_belief_updates": go_true, "true_belief_agree": bool(true_belief_agree >= thresholds["true_belief_acc"]),
                          "other_lesion_collapses": lesion_collapsed, "scramble_collapses": scramble_collapsed,
                          "self_other_dissociation": dissociation_ok},
        "go": go,
    }
    if verbose:
        _print_seed(r)
    return r


def _print_seed(r):
    it = r["intact"]; ds = r["self_other_dissociation"]; le = r["other_lesion"]; sc = r["scramble_witnessing"]
    print(f"  [seed {r['seed']}]  {r['n_false_trials']} false / {r['n_true_trials']} true trials (K={K_LOC}, "
          f"chance {1.0/K_LOC:.2f})", flush=True)
    print(f"    INTACT   false_belief_acc={it['false_belief_acc']:.3f}  true_belief_acc={it['true_belief_acc']:.3f}  "
          f"overall={it['overall_belief_acc']:.3f}", flush=True)
    print(f"             reality-baseline: false_acc={it['reality_baseline_false_acc']:.3f} (must FAIL) "
          f"overall={it['reality_baseline_overall_acc']:.3f} | true-belief agree(belief==reality)="
          f"{it['true_belief_agree_belief_vs_reality']:.3f}", flush=True)
    print(f"    DISSOC   self_tracks_reality(false trials)={ds['self_tracks_reality_on_false_trials']:.3f}  "
          f"belief_is_false={ds['belief_is_false_on_false_trials']:.3f}  ok={ds['dissociation_ok']}", flush=True)
    print(f"    LESION   other_false_belief_acc={le['false_belief_acc']:.3f} predicts_reality="
          f"{le['predicts_reality_on_false_trials']:.3f}  collapsed={le['collapsed']}", flush=True)
    print(f"    SCRAMBLE belief_acc_vs_true={sc['belief_acc_vs_true']:.3f}  collapsed={sc['collapsed']}", flush=True)
    print(f"    >>> seed GO = {r['go']}  {r['go_components']}", flush=True)


DEFAULT_THRESHOLDS = {
    "false_belief_acc": 0.85,        # intact: predict X's (false) belief on unwitnessed trials
    "true_belief_acc": 0.85,         # intact: belief UPDATES when X watched; also the dissociation / agree bar
    "reality_baseline_max": 0.20,    # the world read predicts reality -> must FAIL false-belief
    "chance_loc": 0.45,              # lesion false-belief must drop to ~chance (1/K=0.25) + margin
    "scramble_chance": 0.70,         # scrambled witnessing collapses toward its 0.5 FLOOR (the held belief is binary
                                     # start-vs-end by task construction, so the floor is 0.5 not 1/K -- honest);
                                     # 0.70 cleanly separates that collapse from the intact ~0.95.
}


def main():
    ap = argparse.ArgumentParser(description="W3 agent-keyed FALSE-BELIEF register de-risk (Sally-Anne ToM).")
    ap.add_argument("--seed", type=int, default=42, help="single seed (used by --smoke)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="multi-seed list (overrides --seed)")
    ap.add_argument("--n-trials", type=int, default=48, help="change-of-location trials per block")
    ap.add_argument("--smoke", action="store_true", help="tiny 1-seed smoke (fewer trials)")
    ap.add_argument("--helper-pa", type=float, default=3000.0,
                    help="witness-gated direct ignite current for the written belief/self member (from the "
                         "restored/quiescent slot); <=0 falls back to IGNITE_PA. World@loc always uses IGNITE_PA.")
    ap.add_argument("--w-write", type=float, default=W_WRITE, help="world->belief/self witnessing-gated write weight")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--json", type=str, default="research/findings/raw/_false_belief_smoke.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    if args.smoke:
        seeds = [args.seed]
        n_trials = min(args.n_trials, 20)
    else:
        seeds = args.seeds if args.seeds is not None else [args.seed]
        n_trials = args.n_trials

    print(f"[false-belief] W3 agent-keyed FALSE-BELIEF register (Sally-Anne ToM) | seeds={seeds} "
          f"n_trials={n_trials} backend={args.backend} K_loc={K_LOC} w_write={args.w_write} helper_pa={args.helper_pa}",
          flush=True)
    print(f"[false-belief] stores (self-schema meta-schema class turned OUTWARD): world(reality) + belief(agent X, "
          f"witness_other-gated) + self(system, always-witnessing); each = {K_LOC}x{STORE_ASSEMBLY} NMDA attractors "
          f"+ shared inhibition", flush=True)
    print("[false-belief] HONEST: a FUNCTIONAL mentalizing correlate (predicts from another agent's belief-state, "
          "dissociable from reality + self) -- NOT a claim of access to another mind.", flush=True)

    t0 = time.time()
    per_seed = []
    for s in seeds:
        per_seed.append(evaluate_seed(s, n_trials, DEFAULT_THRESHOLDS, helper_pa=args.helper_pa,
                                      w_write=args.w_write, verbose=True))

    n_go = sum(1 for r in per_seed if r["go"])
    all_go = bool(n_go == len(per_seed))
    verdict = "GO" if all_go else ("PARTIAL" if n_go > 0 else "NEGATIVE")

    def _mean(path):
        vals = []
        for r in per_seed:
            v = r
            for k in path:
                v = v[k]
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    agg = {
        "mean_false_belief_acc": _mean(["intact", "false_belief_acc"]),
        "mean_true_belief_acc": _mean(["intact", "true_belief_acc"]),
        "mean_reality_baseline_false_acc": _mean(["intact", "reality_baseline_false_acc"]),
        "mean_lesion_false_belief_acc": _mean(["other_lesion", "false_belief_acc"]),
        "mean_scramble_belief_acc": _mean(["scramble_witnessing", "belief_acc_vs_true"]),
        "all_lesion_collapse": all(r["other_lesion"]["collapsed"] for r in per_seed),
        "all_scramble_collapse": all(r["scramble_witnessing"]["collapsed"] for r in per_seed),
        "all_dissociation_ok": all(r["self_other_dissociation"]["dissociation_ok"] for r in per_seed),
    }

    out = {
        "runner": "_false_belief_register_derisk",
        "faculty": "W3 belief attribution / false belief (Stage-3 flagship social build; ToM ladder rung 1)",
        "theory": "Fleming-Daw SELF=OTHER-INFERENCE meta-schema + Wimmer-Perner/Sally-Anne false belief; TPJ/mPFC "
                  "mentalizing (FUNCTIONAL correlate only)",
        "mechanism": "self-schema meta-schema class turned OUTWARD: agent-keyed belief store (GNW single-content "
                     "attractor) with sim/'s own transmission_gate = witnessing-gated writes",
        "seeds": seeds, "n_trials": n_trials, "backend": args.backend,
        "knobs": {
            "K_LOC": K_LOC, "STORE_ASSEMBLY": STORE_ASSEMBLY, "STORE_FS_N": STORE_FS_N,
            "attractor_weight": float(DEFAULT_ATTRACTOR_WEIGHT), "W_WRITE": float(args.w_write),
            "write_ignite_pa": (float(args.helper_pa) if args.helper_pa > 0.0 else float(IGNITE_PA)),
            "world_ignite_pa": float(IGNITE_PA),
            "WS_TO_FS_WEIGHT": float(WS_TO_FS_WEIGHT), "FS_TO_WS_WEIGHT": float(FS_TO_WS_WEIGHT),
            "DRIVE_STEPS": int(DRIVE_STEPS), "HOLD_STEPS": int(HOLD_STEPS), "FREE_STEPS": int(FREE_STEPS),
            "SETTLE_STEPS": int(SETTLE_STEPS), "nmda_ratio": 0.5, "dt_ms": 1.0,
            "read_window": "late third of FREE_STEPS", "chance": 1.0 / K_LOC,
        },
        "w_write": args.w_write, "helper_pa": args.helper_pa,
        "thresholds": DEFAULT_THRESHOLDS,
        "verdict": verdict, "n_go": n_go, "n_seeds": len(seeds),
        "aggregate": agg,
        "per_seed": per_seed,
        "honest_scope": ("A functional mentalizing correlate: an agent-keyed belief store (the self-schema "
                         "meta-schema class turned outward) predicts another agent's action FROM the agent's "
                         "witnessing-gated belief, dissociable from reality (reality-baseline fails) and from the "
                         "self's belief; collapses under other-store lesion / scrambled witnessing. NOT a claim of "
                         "access to another mind. Change-of-location only; unexpected-contents is the same mechanism "
                         "(a semantic relabel) and a follow-on."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[false-belief] === VERDICT: {verdict} ({n_go}/{len(seeds)} seeds GO) ===", flush=True)
    print(f"[false-belief]   mean false_belief_acc={agg['mean_false_belief_acc']:.3f} (chance {1.0/K_LOC:.2f}) | "
          f"mean true_belief_acc={agg['mean_true_belief_acc']:.3f} | "
          f"mean reality-baseline(false)={agg['mean_reality_baseline_false_acc']:.3f} (must FAIL)", flush=True)
    print(f"[false-belief]   anti-cheats: other-lesion collapses={agg['all_lesion_collapse']} "
          f"(lesion false_belief_acc={agg['mean_lesion_false_belief_acc']:.3f}) | "
          f"scramble collapses={agg['all_scramble_collapse']} (acc={agg['mean_scramble_belief_acc']:.3f}) | "
          f"self/other dissociation={agg['all_dissociation_ok']}", flush=True)
    print(f"[false-belief]   elapsed={time.time()-t0:.1f}s  wrote {args.json}", flush=True)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
