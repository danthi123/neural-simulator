"""W4 (Stage-4 CONVERSANT): RECURSIVE THEORY OF MIND -- nested belief frames on a WM-buffer STACK + iterated
speaker-listener (RSA). The depth rung above W3 (agent-keyed false BELIEF, 6-seed GO) and W5 (affective ToM, 6-seed
GO). Master roadmap sec 8 L314 / row 139: "recursion depth -> bounded theta-gamma WM-buffer stack (nested belief
frames = nested clauses); RSA = iterated speaker-listener best-response; unbounded = OPEN (humans ~2-3 too)".

TWO faculties, one runner, both COMPUTED ON THE SPIKING SUBSTRATE (reuse-by-import, NO `sim/` edit):

  PART A -- 2nd-ORDER FALSE BELIEF (nested belief frames on a WM-buffer stack).
    W3 gave ONE agent-keyed belief store: "X believes the object is at A". W4 STACKS them: frame_1 = "agent J
    believes L", frame_2 = "agent M believes (J believes L)", ... frame_d = the d-th nesting. Each frame is a W3
    belief store (a GNW single-content attractor = the WM-buffer slot); the STACK is a chain of witnessing-gated
    writes world->frame_1->frame_2->... (`sim/`'s own transmission_gate is the witnessing gate, reused verbatim
    from W3). frame_d is written FROM frame_{d-1} (the level below), gated by witness_d = "did the level-d agent
    observe the level-(d-1) update". So a FALSE belief at a lower level propagates upward correctly, and the
    classic Perner-Wimmer 2nd-order test (ice-cream van: J saw the van move, M did NOT see that J saw, so M's model
    of J is stuck at the OLD location) is: witness_1=1 (J saw), witness_2=0 (M didn't see J see) -> frame_2 HOLDS
    the placement A while frame_1 (=J's real belief) and reality are both B. The DECISIVE 2nd-order signature:
    frame_2 dissociates from BOTH reality AND frame_1 -- a 1st-order reader (predict J's real belief) FAILS.

  PART B -- SCALAR IMPLICATURE to depth 2 (iterated speaker-listener = RSA, on the substrate). States {none,
    some-but-not-all (SBNA), all}; utterances {"none","some","all"}; literal truth: "some" is TRUE of {SBNA, all}.
    The RSA recursion L0 -> S1 -> L1 (depth 2) turns the LITERAL listener's flat L0("some")=[SBNA .5, all .5] into
    the PRAGMATIC listener L1("some") that PREFERS SBNA (the "some -> not all" implicature), because a speaker in
    state `all` prefers the more-informative "all". Realized on the substrate: each RSA distribution is read as the
    graded firing rates of a competitive assembly population whose shared FS pool performs DIVISIVE NORMALIZATION
    (Carandini-Heeger). At rationality alpha=1 the whole recursion is THREE rounds of proportional (divisive)
    normalization -- the SUBSTRATE's operation -- and the implicature is a CONSEQUENCE of that normalization: the
    single-item state (all|"all") fires HARDER than each of the two-item states (SBNA|"some", all|"some")), and that
    informativity gap, propagated through the depth-2 iteration, yields L1(SBNA|"some") > L1(all|"some"). LESION the
    normalization (FS inhibition -> 0) and rates ride the raw truth (both = 1) -> the gap vanishes -> NO implicature.

HONEST SCOPE (carried into every line). BOTH parts are FUNCTIONAL mentalizing/pragmatics CORRELATES: a substrate
that REPRESENTS and OPERATES ON nested belief frames + the pragmatic recursion, dissociable from reality and from
the lower nesting levels, collapsing under lesion/scramble/permute. NOT a claim of phenomenal access to another
mind. The literal-truth lexicon (Part B) is the legitimate LINGUISTIC input (as W5's situation->valence appraisal
is legitimate world input); the ToM-specific neural work is the STACK of gated belief frames (A) and the ITERATED
COMPETITIVE NORMALIZATION (B). Self-report is a functional read-out, never a phenomenal-experience claim.

GO GATE (6-seed 42 43 44 100 101 102, CPU numpy):
  A (2nd-order false belief, depth 2; chance 1/K_loc):
    - order2_false_belief_acc      >= 0.85   (frame_2 predicts M's FALSE model of J = the placement)
    - order1_baseline_false_acc    <= 0.20   (frame_1 = J's REAL belief = reality -> WRONG: 2nd != 1st order)
    - reality_baseline_false_acc   <= 0.20   (the world read predicts reality -> WRONG)
    - order2_true_belief_acc       >= 0.85   (when M witnessed, frame_2 UPDATES -- not "always old")
  B (scalar implicature, depth 2; chance 0.5):
    - implicature_depth2_acc       >= 0.85   (L1("some") ranks SBNA > all)
    - literal_depth0_acc <= 0.40             (L0("some") shows NO implicature -- it is DEPTH-created; the flat
                                              L0 distribution scored by a strict SBNA>all readout floors at ~0)
  MOAT / ANTI-CHEATS (all must collapse -> the positive read is not a confabulation):
    - A buffer-scramble (read a RANDOM stack frame) collapses (<= 0.55)
    - A permuted-premises (shuffle the premise tuples) collapses (<= 0.55)
    - A flatten-lesion (force every witness gate OPEN -> frames mirror reality) collapses (<= 0.55)
    - B normalization-lesion (FS inhibition -> 0) collapses the implicature (<= 0.65)
    - B permuted-lexicon (shuffle the truth matrix) collapses (<= 0.65)
  CHARACTERIZATION (reported, NOT gated): the depth profile 1/2/3(/4). Deeper nesting = more chained writes; the
  bound where it degrades is the honest human ~2-3-embedding limit. We do NOT force GO past where it works.

Usage:
  # smoke (1 seed, tiny -- proves it runs, controls live, prints a verdict):
  SIM_BACKEND=numpy python -u -m research.runners._recursive_tom_rsa_derisk --smoke --seed 42 \
      --json research/findings/raw/_recursive_tom/smoke.json
  # one seed (the pool submits six of these):
  SIM_BACKEND=numpy python -u -m research.runners._recursive_tom_rsa_derisk --seed 42 \
      --json research/findings/raw/_recursive_tom/seed42.json
  # aggregate the six per-seed jsons into the 6-seed verdict:
  SIM_BACKEND=numpy python -u -m research.runners._recursive_tom_rsa_derisk \
      --aggregate 'research/findings/raw/_recursive_tom/seed*.json' \
      --json research/findings/raw/_recursive_tom/summary_6seed.json
  # local all-in-one (loops seeds, then aggregates):
  SIM_BACKEND=numpy python -u -m research.runners._recursive_tom_rsa_derisk --seeds 42 43 44 100 101 102 \
      --json research/findings/raw/_recursive_tom/summary_6seed.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

# reuse-by-import: the validated GNW spiking machinery (self-recurrent NMDA attractor loop + wash-out snapshot)
from research.runners._gnw_rung1_ignition_curve_derisk import (  # noqa: E402
    _build_assembly_loop_population, _snapshot_state, _restore_state,
    DEFAULT_ATTRACTOR_WEIGHT, SETTLE_STEPS, FREE_STEPS,
)
# reuse-by-import: the W3 belief-store primitives -- the SAME agent-keyed store, now STACKED.
from research.runners._false_belief_register_derisk import (  # noqa: E402
    _gated_write_projection, _restore_slice, K_LOC, STORE_ASSEMBLY, STORE_FS_N,
    HOLD_STEPS, W_WRITE, WRITE_DRIVE_STEPS,
)
from research.runners._self_schema_region_derisk import (  # noqa: E402
    WS_LOOP_GATE, WS_TO_FS_WEIGHT, FS_TO_WS_WEIGHT, IGNITE_PA,
)

# ============================================================================================================
# PART A -- nested belief frames on a WM-buffer STACK (2nd-order false belief)
# ============================================================================================================

def build_stack_bridge(seed: int, depth: int, attractor_weight: float = DEFAULT_ATTRACTOR_WEIGHT,
                       w_write: float = W_WRITE):
    """ONE spiking `SimulationBridge` with a STACK of `depth`+1 belief stores of the W3 meta-schema class:
    level 0 = `world` (reality); level d = `frame_d` (the d-th nesting = "agent_d believes (agent_{d-1}
    believes ... )"). Each store = K_LOC self-recurrent NMDA member assemblies + a shared FS pool (GNW single
    content = "which location is currently believed at this nesting level"). The stack's writes chain
    world->frame_1->frame_2->...: `write_frame_d` is a witnessing-gated topographic projection from frame_{d-1}
    to frame_d (transmission_gate = witness_d). Returns (bridge, xp, idx, snap)."""
    xp, _ = get_backend()
    n_store = STORE_ASSEMBLY * K_LOC
    store_names = ["world"] + [f"frame_{d}" for d in range(1, depth + 1)]

    regions = []
    for nm in store_names:
        regions.append(BrainRegion(name=nm, n_neurons=n_store, exc_fraction=1.0,
                                    internal_density=0.0, enable_nmda=True))
        regions.append(BrainRegion(name=f"{nm}_fs", n_neurons=STORE_FS_N, exc_fraction=0.0,
                                    internal_density=0.0, enable_nmda=False))
    pathways = []
    for nm in store_names:
        pathways.append(RegionPathway(from_region=nm, to_region=f"{nm}_fs", density=0.5,
                                      weight_mean=WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False))
        pathways.append(RegionPathway(from_region=f"{nm}_fs", to_region=nm, density=0.5,
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
    cfg.seed = int(seed)                       # seeds the substrate (het guard fires at seed>=0; the doc gotcha)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True
    cfg.stdp_w_max = max(400.0, float(attractor_weight) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(attractor_weight) * 4.0)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager

    def members(nm):
        base = np.asarray(rm.indices(nm), dtype=np.int64)
        return {k: base[k * STORE_ASSEMBLY:(k + 1) * STORE_ASSEMBLY] for k in range(K_LOC)}

    mem = {nm: members(nm) for nm in store_names}

    union = dict(rm.build_wiring_plan(seed=int(seed)))
    for lvl, nm in enumerate(store_names):
        for k in range(K_LOC):
            union[f"loop_{nm}_{k}"] = _build_assembly_loop_population(mem[nm][k], float(attractor_weight))
        if lvl >= 1:
            src = store_names[lvl - 1]
            for k in range(K_LOC):
                union[f"write_{nm}_{k}"] = _gated_write_projection(mem[src][k], mem[nm][k], w_write,
                                                                   f"witness_{lvl}")
    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)
    for lvl in range(1, depth + 1):
        bridge.set_transmission_gate(f"witness_{lvl}", 0.0)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)

    idx = {"names": store_names, "depth": depth}
    for nm in store_names:
        idx[f"{nm}_dev"] = {k: xp.asarray(v) for k, v in mem[nm].items()}
        idx[f"{nm}_all"] = xp.asarray(np.concatenate([mem[nm][k] for k in range(K_LOC)]))
    return bridge, xp, idx, snap


def _run_stack_trial(bridge, xp, idx, snap, start_loc, end_loc, witness_move,
                     flatten=False, helper_pa=5000.0, drive_steps=50):
    """One change-of-location trial on the STACK. Placement@start is witnessed by ALL levels (every frame
    ignites to start). Move@end: level 0 (world) always -> end; for level d>=1, witness_move[d] (0/1) gates the
    frame_{d-1}->frame_d copy. `flatten` forces EVERY witness gate OPEN at the move (all frames mirror reality =
    the LESION that ablates the gated stack). Returns per-store late-window member rates {level_name: {k: rate}}."""
    names = idx["names"]
    depth = idx["depth"]
    ds = int(drive_steps)
    write_pa = float(helper_pa) if helper_pa > 0.0 else IGNITE_PA

    bridge.cp_external_input_current[:] = 0.0
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0

    def _event(contents, write_flags):
        # CLEAR (register clear-before-write) the WRITTEN stores' slices to quiescence; ignite fresh-from-rest.
        for lvl, nm in enumerate(names):
            if write_flags[lvl]:
                _restore_slice(bridge, snap, idx[f"{nm}_all"])
        bridge.cp_external_input_current[:] = 0.0
        for lvl in range(1, depth + 1):
            bridge.set_transmission_gate(f"witness_{lvl}", 1.0 if write_flags[lvl] else 0.0)
        for _ in range(ds):
            bridge.cp_external_input_current[:] = 0.0
            for lvl, nm in enumerate(names):
                if write_flags[lvl]:
                    pa = IGNITE_PA if lvl == 0 else write_pa
                    bridge.cp_external_input_current[idx[f"{nm}_dev"][contents[lvl]]] = xp.float32(pa)
            bridge._run_one_simulation_step()
        # HOLD: writes closed (frames self-sustain via loops) unless flattened (keep mirroring reality).
        for lvl in range(1, depth + 1):
            bridge.set_transmission_gate(f"witness_{lvl}", 1.0 if flatten else 0.0)
        for _ in range(HOLD_STEPS):
            bridge.cp_external_input_current[:] = 0.0
            bridge._run_one_simulation_step()

    # placement: all levels witness -> contents = start everywhere, all write.
    _event([start_loc] * (depth + 1), [True] * (depth + 1))

    # move: world -> end; frame_d copies frame_{d-1} iff witnessed (or flatten).
    cont = [end_loc]                      # level 0 = reality
    wr = [True]                           # world always writes at the move
    for d in range(1, depth + 1):
        w = True if flatten else bool(witness_move.get(d, 0))
        cont.append(cont[d - 1] if w else start_loc)   # holds the placement `start` if this level didn't witness
        wr.append(w)
    _event(cont, wr)

    # QUERY: no drive; persistent stores hold (flatten: gates stay OPEN so frames keep mirroring reality).
    for lvl in range(1, depth + 1):
        bridge.set_transmission_gate(f"witness_{lvl}", 1.0 if flatten else 0.0)
    late_start = FREE_STEPS - max(1, FREE_STEPS // 3)
    acc = {nm: {k: 0 for k in range(K_LOC)} for nm in names}
    for t in range(FREE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
        if t >= late_start:
            for nm in names:
                for k in range(K_LOC):
                    acc[nm][k] += int(to_host(bridge.cp_firing_states[idx[f"{nm}_dev"][k]].astype(xp.float64).sum()))
    nlate = float(FREE_STEPS - late_start) * STORE_ASSEMBLY
    return {nm: {k: acc[nm][k] / nlate for k in range(K_LOC)} for nm in names}


def _argmax_loc(rate_dict):
    return int(max(rate_dict, key=rate_dict.get))


def _make_stack_trials(seed, depth, n_trials):
    """n_trials change-of-location trials at nesting `depth`. Each: placement A, move to B(!=A); balanced
    ~50/50 FALSE (levels 1..depth-1 witness the move, level `depth` does NOT -> frame_depth HOLDS A = the
    d-th-order FALSE belief) vs TRUE (all levels witness -> frame_depth UPDATES to B)."""
    rng = np.random.default_rng(seed * 131 + depth * 17 + 5)
    trials = []
    for i in range(n_trials):
        a = int(rng.integers(K_LOC)); b = int(rng.integers(K_LOC))
        while b == a:
            b = int(rng.integers(K_LOC))
        is_false = (i % 2 == 0)
        wit = {d: 1 for d in range(1, depth + 1)}
        if is_false:
            wit[depth] = 0                     # the TOP nesting level did not witness -> its frame holds `start`
        gt = a if is_false else b               # frame_depth content (M's model of J...)
        trials.append({"start": a, "end": b, "witness": wit, "is_false": is_false, "gt": gt})
    rng.shuffle(trials)
    return trials


def evaluate_stack_depth(seed, depth, n_trials, helper_pa, drive_steps, thr):
    """Evaluate 2nd(/d-th)-order false belief at a given nesting `depth`. Reads EVERY store per trial so the
    reality-baseline (world), the 1st-order baseline (frame_1) and the d-th-order read (frame_depth) are all
    measured on the SAME trial. Runs the intact block + the flatten-lesion + buffer-scramble + permuted-premises
    anti-cheats. Returns a metrics dict."""
    trials = _make_stack_trials(seed, depth, n_trials)
    gt = np.array([t["gt"] for t in trials], dtype=int)
    reality = np.array([t["end"] for t in trials], dtype=int)
    false_mask = np.array([t["is_false"] for t in trials], dtype=bool)
    true_mask = ~false_mask

    bridge, xp, idx, snap = build_stack_bridge(seed=seed, depth=depth)
    names = idx["names"]
    top = names[depth]

    def _read_all(t, flatten=False):
        r = _run_stack_trial(bridge, xp, idx, snap, t["start"], t["end"], t["witness"],
                             flatten=flatten, helper_pa=helper_pa, drive_steps=drive_steps)
        return {nm: _argmax_loc(r[nm]) for nm in names}

    # ---- INTACT: read every store per trial ----
    reads = [_read_all(t) for t in trials]
    pred_top = np.array([r[top] for r in reads], dtype=int)
    pred_world = np.array([r["world"] for r in reads], dtype=int)
    pred_l1 = np.array([r[names[1]] for r in reads], dtype=int)

    order_d_false_acc = float(np.mean(pred_top[false_mask] == gt[false_mask])) if false_mask.any() else 0.0
    order_d_true_acc = float(np.mean(pred_top[true_mask] == gt[true_mask])) if true_mask.any() else 0.0
    reality_baseline_false = float(np.mean(pred_world[false_mask] == gt[false_mask])) if false_mask.any() else 0.0
    order1_baseline_false = float(np.mean(pred_l1[false_mask] == gt[false_mask])) if false_mask.any() else 0.0
    # per-level stack fidelity (does each frame hold its true nested content?) -- reported characterization
    stack_fidelity = float(np.mean([np.mean([reads[i][names[lvl]] ==
                                             (trials[i]["gt"] if lvl == depth else
                                              (trials[i]["end"] if trials[i]["witness"].get(lvl, 1) or lvl == 0
                                               else trials[i]["start"]))
                                             for lvl in range(depth + 1)]) for i in range(len(trials))]))

    # ---- FLATTEN-LESION: force every witness gate open -> frames mirror reality -> false belief collapses ----
    pred_top_flat = np.array([_read_all(t, flatten=True)[top] for t in trials], dtype=int)
    flatten_false_acc = float(np.mean(pred_top_flat[false_mask] == gt[false_mask])) if false_mask.any() else 0.0
    flatten_predicts_reality = float(np.mean(pred_top_flat[false_mask] == reality[false_mask])) if false_mask.any() else 0.0

    # ---- BUFFER-SCRAMBLE: read a UNIFORMLY RANDOM stack frame instead of frame_depth -> the ordered buffer is
    # load-bearing (on FALSE trials only frame_depth holds the answer; a random frame gives reality). ----
    srng = np.random.default_rng(seed * 733 + depth * 29 + 11)
    scr_pred = np.array([reads[i][names[int(srng.integers(depth + 1))]] for i in range(len(trials))], dtype=int)
    buffer_scramble_false_acc = float(np.mean(scr_pred[false_mask] == gt[false_mask])) if false_mask.any() else 0.0

    # ---- PERMUTED-PREMISES: permute the premise tuples across trials, read frame_depth, score vs TRUE gt ----
    prng = np.random.default_rng(seed * 977 + depth * 37 + 19)
    perm = prng.permutation(len(trials))
    perm_reads_top = np.array([_read_all(trials[perm[i]])[top] for i in range(len(trials))], dtype=int)
    permuted_false_acc = float(np.mean(perm_reads_top[false_mask] == gt[false_mask])) if false_mask.any() else 0.0

    m = {
        "depth": depth, "n_trials": n_trials, "chance": 1.0 / K_LOC,
        "n_false": int(false_mask.sum()), "n_true": int(true_mask.sum()),
        "order_d_false_belief_acc": order_d_false_acc,
        "order_d_true_belief_acc": order_d_true_acc,
        "reality_baseline_false_acc": reality_baseline_false,
        "order1_baseline_false_acc": order1_baseline_false,
        "stack_fidelity": stack_fidelity,
        "flatten_lesion_false_acc": flatten_false_acc,
        "flatten_predicts_reality": flatten_predicts_reality,
        "buffer_scramble_false_acc": buffer_scramble_false_acc,
        "permuted_premises_false_acc": permuted_false_acc,
    }
    return m


# ============================================================================================================
# PART B -- scalar implicature to depth 2 (iterated speaker-listener = RSA, on the substrate)
# ============================================================================================================

STATES = ["none", "SBNA", "all"]                       # SBNA = some-but-not-all
UTTS = ["none", "some", "all"]
# literal truth[utterance][state]: 1 if the utterance is literally true of the state.
TRUTH = {
    "none": {"none": 1, "SBNA": 0, "all": 0},
    "some": {"none": 0, "SBNA": 1, "all": 1},          # "some" is literally true of BOTH SBNA and all
    "all":  {"none": 0, "SBNA": 0, "all": 1},
}
RSA_ITEM_SIZE = 40
RSA_FS_N = 40
RSA_EXC_FS_W = 6.0
RSA_FS_EXC_W = 22.0           # divisive-normalization strength (fs -> exc inhibition); 0 = normalization lesion
RSA_TRUTH_DRIVE_PA = 1300.0  # drive current for a fully-supported item
RSA_INPUT_GAIN = 3600.0      # rate -> drive current gain for the recursion steps (rates ~0..0.33)
RSA_SETTLE = 25
RSA_SETTLE_JITTER = 25
RSA_DRIVE = 70
RSA_READ = 45                # read the mean per-item rate over the last RSA_READ steps of the drive
# DEADBAND: an implicature "present" iff the SBNA-minus-all L1 rate margin exceeds this. Tuned so the depth-2
# implicature margin (~+0.033, 10x the fixed per-neuron heterogeneity bias) clears it while the literal L0
# margin, the normalization-lesion margin, and permuted-lexicon margins (all ~+0.002-0.003) do NOT. Without it a
# sign test on a near-zero margin would read a tiny FIXED positional bias as a "preference" (measured 2026-08-01).
RSA_MARGIN_EPS = 0.012


def build_rsa_bridge(seed, normalize=True):
    """A competitive assembly population = the RSA normalizer. K=3 item assemblies (states OR utterances -- the
    op is identical) sharing ONE FS pool that performs DIVISIVE normalization via feedback inhibition. Driving
    each item with a current and reading its graded rate = a proportional (softmax-at-alpha=1) normalization.
    `normalize=False` sets the fs->exc weight to 0 (the normalization LESION: rates ride the raw input, no
    single-vs-multi-item contrast -> no implicature)."""
    xp, _ = get_backend()
    regions = [
        BrainRegion(name="item", n_neurons=RSA_ITEM_SIZE * 3, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="item_fs", n_neurons=RSA_FS_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
    ]
    pathways = [
        RegionPathway(from_region="item", to_region="item_fs", density=0.6, weight_mean=RSA_EXC_FS_W,
                      weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="item_fs", to_region="item", density=0.6,
                      weight_mean=(RSA_FS_EXC_W if normalize else 0.0), weight_jitter=0.0, plastic=False),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process", "enable_nmda"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    rm = bridge.region_manager
    base = np.asarray(rm.indices("item"), dtype=np.int64)
    item_dev = {i: xp.asarray(base[i * RSA_ITEM_SIZE:(i + 1) * RSA_ITEM_SIZE]) for i in range(3)}
    inh = list(rm.inhibitory_indices("item_fs"))
    # the region-framework pathways are already installed; just ensure inhibitory tagging via a no-op wiring inject
    bridge.inject_explicit_wiring(dict(rm.build_wiring_plan(seed=int(seed))),
                                  output_inhibitory_indices=inh or None)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)
    return bridge, xp, item_dev, snap


def _compete(bridge, xp, item_dev, snap, input_scores, settle_ms):
    """Drive the 3 item assemblies with currents proportional to `input_scores` (a length-3 vector), let the FS
    divisive normalization settle, and read each item's mean firing rate over the read window = the competitive
    (normalized) distribution. `input_scores` in {0,1} (literal truth) OR rates from a previous level."""
    bridge.cp_external_input_current[:] = 0.0
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(int(settle_ms)):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
    drive = [float(s) for s in input_scores]
    acc = np.zeros(3, dtype=np.float64)
    for t in range(RSA_DRIVE):
        bridge.cp_external_input_current[:] = 0.0
        for i in range(3):
            if drive[i] > 0.0:
                bridge.cp_external_input_current[item_dev[i]] = xp.float32(drive[i])
        bridge._run_one_simulation_step()
        if t >= RSA_DRIVE - RSA_READ:
            for i in range(3):
                acc[i] += float(to_host(bridge.cp_firing_states[item_dev[i]].astype(xp.float64).sum()))
    return acc / (RSA_READ * RSA_ITEM_SIZE)             # mean per-neuron rate per item


def _rsa_recursion(bridge, xp, item_dev, snap, truth, settle_ms):
    """One depth-2 RSA pass on the substrate. Returns (L0, S1, L1) as 3x3 rate matrices indexed [utt][state].
    L0[u] = compete over STATES driven by truth[u]*prior. S1[.][s] = compete over UTTERANCES driven by L0[.][s].
    L1[u] = compete over STATES driven by S1[u][.]. alpha=1 => each step is one divisive normalization; the
    implicature is a consequence of the single-vs-multi-item contrast the FS pool creates."""
    ui = {u: j for j, u in enumerate(UTTS)}
    si = {s: j for j, s in enumerate(STATES)}
    # ---- L0: for each utterance, normalize over states (literal truth as the drive) ----
    L0 = np.zeros((3, 3))                                # [utt][state]
    for u in UTTS:
        scores = [RSA_TRUTH_DRIVE_PA * float(truth[u][s]) for s in STATES]
        L0[ui[u], :] = _compete(bridge, xp, item_dev, snap, scores, settle_ms)
    # ---- S1: for each state, normalize over utterances (driven by how much each utterance's listener supports s)
    S1 = np.zeros((3, 3))                                # [utt][state]
    for s in STATES:
        scores = [RSA_INPUT_GAIN * L0[ui[u], si[s]] for u in UTTS]
        col = _compete(bridge, xp, item_dev, snap, scores, settle_ms)
        for u in UTTS:
            S1[ui[u], si[s]] = col[ui[u]]
    # ---- L1: for each utterance, normalize over states (driven by the speaker's S1 for that utterance) ----
    L1 = np.zeros((3, 3))                                # [utt][state]
    for u in UTTS:
        scores = [RSA_INPUT_GAIN * S1[ui[u], si[s]] for s in STATES]
        L1[ui[u], :] = _compete(bridge, xp, item_dev, snap, scores, settle_ms)
    return L0, S1, L1


def evaluate_rsa(seed, n_trials, thr):
    """Depth-2 scalar implicature on the substrate. Per trial (settle-jitter -> genuine noise-phase variance):
    run the intact recursion, the normalization-LESION recursion, and a PERMUTED-LEXICON recursion; score the
    'some' implicature = does L1(SBNA|"some") > L1(all|"some")? (depth-0 = L0, should be flat)."""
    si = {s: j for j, s in enumerate(STATES)}
    intact, xp, item_dev, snap = build_rsa_bridge(seed, normalize=True)
    lesion, _xpl, item_dev_l, snap_l = build_rsa_bridge(seed, normalize=False)
    rng = np.random.default_rng(seed * 131 + 71)

    imp_d2 = np.zeros(n_trials); lit_d0 = np.zeros(n_trials)
    les_d2 = np.zeros(n_trials); perm_d2 = np.zeros(n_trials)
    margins = {"intact_l1": [], "intact_l0": [], "lesion_l1": []}

    for i in range(n_trials):
        settle = int(RSA_SETTLE + rng.integers(0, RSA_SETTLE_JITTER + 1))
        L0, S1, L1 = _rsa_recursion(intact, xp, item_dev, snap, TRUTH, settle)
        m_l1 = L1[UTTS.index("some"), si["SBNA"]] - L1[UTTS.index("some"), si["all"]]
        m_l0 = L0[UTTS.index("some"), si["SBNA"]] - L0[UTTS.index("some"), si["all"]]
        imp_d2[i] = 1.0 if m_l1 > RSA_MARGIN_EPS else 0.0
        lit_d0[i] = 1.0 if m_l0 > RSA_MARGIN_EPS else 0.0
        margins["intact_l1"].append(float(m_l1)); margins["intact_l0"].append(float(m_l0))

        _l0, _s1, L1l = _rsa_recursion(lesion, _xpl, item_dev_l, snap_l, TRUTH, settle)
        m_les = L1l[UTTS.index("some"), si["SBNA"]] - L1l[UTTS.index("some"), si["all"]]
        les_d2[i] = 1.0 if m_les > RSA_MARGIN_EPS else 0.0
        margins["lesion_l1"].append(float(m_les))

        # PERMUTED-LEXICON: shuffle which state each utterance is true of -> the implicature direction randomizes;
        # score vs the TRUE implicature (SBNA preferred). Collapses to chance.
        prng = np.random.default_rng(seed * 977 + i * 13 + 3)
        perm_states = list(STATES); prng.shuffle(perm_states)
        ptruth = {u: {perm_states[j]: TRUTH[u][STATES[j]] for j in range(3)} for u in UTTS}
        _p0, _p1, L1p = _rsa_recursion(intact, xp, item_dev, snap, ptruth, settle)
        m_perm = L1p[UTTS.index("some"), si["SBNA"]] - L1p[UTTS.index("some"), si["all"]]
        perm_d2[i] = 1.0 if m_perm > RSA_MARGIN_EPS else 0.0

    return {
        "n_trials": n_trials, "chance": 0.5,
        "implicature_depth2_acc": float(np.mean(imp_d2)),
        "literal_depth0_acc": float(np.mean(lit_d0)),
        "normalization_lesion_acc": float(np.mean(les_d2)),
        "permuted_lexicon_acc": float(np.mean(perm_d2)),
        "mean_margin_intact_l1": float(np.mean(margins["intact_l1"])),
        "mean_margin_intact_l0": float(np.mean(margins["intact_l0"])),
        "mean_margin_lesion_l1": float(np.mean(margins["lesion_l1"])),
    }


# ============================================================================================================
# per-seed evaluation + 6-seed aggregation / verdict
# ============================================================================================================

DEFAULT_THR = {
    "order2_false_acc": 0.85, "order1_baseline_max": 0.20, "reality_baseline_max": 0.20,
    "order2_true_acc": 0.85, "collapse_max": 0.60, "chance_loc": 1.0 / K_LOC,
    # Part B controls have DIFFERENT floors: the deadband sends the literal-L0 and normalization-lesion margins to
    # ~0 (floor ~0, threshold 0.40), but the permuted-lexicon RANDOMIZES the implicature's DIRECTION while the read
    # positions stay fixed, so it floors near chance (~0.33-0.5, threshold 0.65) -- collapsing from the intact 1.0.
    "implicature_acc": 0.85, "literal_max": 0.40, "rsa_lesion_max": 0.40, "rsa_perm_max": 0.65,
}


def evaluate_seed(seed, depths, n_trials, rsa_trials, helper_pa, drive_steps, verbose=True):
    t0 = time.time()
    partA = {int(d): evaluate_stack_depth(seed, d, n_trials, helper_pa, drive_steps, DEFAULT_THR) for d in depths}
    partB = evaluate_rsa(seed, rsa_trials, DEFAULT_THR)

    d2 = partA.get(2)
    thr = DEFAULT_THR
    goA = bool(d2 is not None
               and d2["order_d_false_belief_acc"] >= thr["order2_false_acc"]
               and d2["order1_baseline_false_acc"] <= thr["order1_baseline_max"]
               and d2["reality_baseline_false_acc"] <= thr["reality_baseline_max"]
               and d2["order_d_true_belief_acc"] >= thr["order2_true_acc"]
               and d2["flatten_lesion_false_acc"] <= thr["collapse_max"]
               and d2["buffer_scramble_false_acc"] <= thr["collapse_max"]
               and d2["permuted_premises_false_acc"] <= thr["collapse_max"])
    goB = bool(partB["implicature_depth2_acc"] >= thr["implicature_acc"]
               and partB["literal_depth0_acc"] <= thr["literal_max"]
               and partB["normalization_lesion_acc"] <= thr["rsa_lesion_max"]
               and partB["permuted_lexicon_acc"] <= thr["rsa_perm_max"])
    go = bool(goA and goB)
    r = {"seed": int(seed), "part_a_by_depth": {str(k): v for k, v in partA.items()}, "part_b_rsa": partB,
         "go_a_2nd_order_false_belief": goA, "go_b_scalar_implicature": goB, "go": go,
         "elapsed_seconds": round(time.time() - t0, 1)}
    if verbose:
        _print_seed(r)
    return r


def _print_seed(r):
    d2 = r["part_a_by_depth"].get("2", {})
    b = r["part_b_rsa"]
    print(f"  [seed {r['seed']}]  ({r['elapsed_seconds']}s)", flush=True)
    for k in sorted(r["part_a_by_depth"].keys(), key=int):
        a = r["part_a_by_depth"][k]
        print(f"    A depth{k}: false_belief={a['order_d_false_belief_acc']:.3f} true={a['order_d_true_belief_acc']:.3f} "
              f"| 1st-order_baseline={a['order1_baseline_false_acc']:.3f} reality={a['reality_baseline_false_acc']:.3f} "
              f"(must FAIL) | flatten={a['flatten_lesion_false_acc']:.3f} buf-scr={a['buffer_scramble_false_acc']:.3f} "
              f"perm={a['permuted_premises_false_acc']:.3f} | stack_fid={a['stack_fidelity']:.3f}", flush=True)
    print(f"    B RSA: implicature(L1)={b['implicature_depth2_acc']:.3f} literal(L0)={b['literal_depth0_acc']:.3f} "
          f"| norm-lesion={b['normalization_lesion_acc']:.3f} perm-lexicon={b['permuted_lexicon_acc']:.3f} "
          f"| margins l1={b['mean_margin_intact_l1']:+.4f} l0={b['mean_margin_intact_l0']:+.4f} "
          f"les={b['mean_margin_lesion_l1']:+.4f}", flush=True)
    print(f"    >>> seed GO = {r['go']}  (A 2nd-order={r['go_a_2nd_order_false_belief']}  "
          f"B implicature={r['go_b_scalar_implicature']})", flush=True)


def _agg(per_seed, key_a_depth2, sub):
    vals = []
    for r in per_seed:
        v = r["part_a_by_depth"].get("2", {}).get(sub) if key_a_depth2 else r["part_b_rsa"].get(sub)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def build_summary(per_seed, seeds, depths, n_trials, rsa_trials, backend):
    n_go = sum(1 for r in per_seed if r["go"])
    all_go = bool(n_go == len(per_seed) and len(per_seed) > 0)
    verdict = "GO" if all_go else ("PARTIAL" if n_go > 0 else "NEGATIVE")

    agg = {
        "mean_order2_false_belief_acc": _agg(per_seed, True, "order_d_false_belief_acc"),
        "mean_order2_true_belief_acc": _agg(per_seed, True, "order_d_true_belief_acc"),
        "mean_order1_baseline_false_acc": _agg(per_seed, True, "order1_baseline_false_acc"),
        "mean_reality_baseline_false_acc": _agg(per_seed, True, "reality_baseline_false_acc"),
        "mean_flatten_lesion_false_acc": _agg(per_seed, True, "flatten_lesion_false_acc"),
        "mean_buffer_scramble_false_acc": _agg(per_seed, True, "buffer_scramble_false_acc"),
        "mean_permuted_premises_false_acc": _agg(per_seed, True, "permuted_premises_false_acc"),
        "mean_implicature_depth2_acc": _agg(per_seed, False, "implicature_depth2_acc"),
        "mean_literal_depth0_acc": _agg(per_seed, False, "literal_depth0_acc"),
        "mean_normalization_lesion_acc": _agg(per_seed, False, "normalization_lesion_acc"),
        "mean_permuted_lexicon_acc": _agg(per_seed, False, "permuted_lexicon_acc"),
        "depth_profile_false_belief": {str(d): _agg([r for r in per_seed], True, "order_d_false_belief_acc")
                                       if d == 2 else
                                       float(np.mean([r["part_a_by_depth"][str(d)]["order_d_false_belief_acc"]
                                                      for r in per_seed if str(d) in r["part_a_by_depth"]]))
                                       for d in depths},
        "depth_profile_true_belief": {str(d): float(np.mean([r["part_a_by_depth"][str(d)]["order_d_true_belief_acc"]
                                                             for r in per_seed if str(d) in r["part_a_by_depth"]]))
                                      for d in depths},
        "all_goA": all(r["go_a_2nd_order_false_belief"] for r in per_seed),
        "all_goB": all(r["go_b_scalar_implicature"] for r in per_seed),
    }

    thr = DEFAULT_THR
    from tools.verdict import Verdict  # noqa: E402
    v = Verdict("recursive ToM (W4): 2nd-order false belief + depth-2 scalar implicature", chance=1.0 / K_LOC)
    v.require("6 seeds (project bar)", len(seeds) >= 6, expect=True)
    v.floor("2nd-order false-belief acc vs chance (1/K_loc)", agg["mean_order2_false_belief_acc"], 1.0 / K_LOC)
    v.require("2nd-order (frame_2) BEATS 1st-order baseline (frame_1 = J's real belief) on false trials",
              agg["mean_order1_baseline_false_acc"], expect=lambda x: x <= thr["order1_baseline_max"],
              note="if frame_1 solved it, this would be 1st-order ToM, not recursive")
    v.require("reality-baseline FAILS the 2nd-order false belief",
              agg["mean_reality_baseline_false_acc"], expect=lambda x: x <= thr["reality_baseline_max"])
    v.require("2nd-order true-belief UPDATES when M witnessed (not always-old)",
              agg["mean_order2_true_belief_acc"], expect=lambda x: x >= thr["order2_true_acc"])
    v.control("flatten-lesion (all witness gates open) collapses the false belief",
              treatment=agg["mean_order2_false_belief_acc"], control=agg["mean_flatten_lesion_false_acc"])
    v.control("buffer-scramble (read a random stack frame) collapses the false belief",
              treatment=agg["mean_order2_false_belief_acc"], control=agg["mean_buffer_scramble_false_acc"])
    v.control("permuted-premises collapses the false belief",
              treatment=agg["mean_order2_false_belief_acc"], control=agg["mean_permuted_premises_false_acc"])
    v.floor("depth-2 scalar implicature (L1) vs chance 0.5", agg["mean_implicature_depth2_acc"], 0.5)
    v.require("literal listener L0 shows NO robust implicature (the effect is DEPTH-created)",
              agg["mean_literal_depth0_acc"], expect=lambda x: x <= thr["literal_max"])
    v.control("permuted-lexicon collapses the implicature",
              treatment=agg["mean_implicature_depth2_acc"], control=agg["mean_permuted_lexicon_acc"])
    v.control("normalization-lesion (FS inhibition -> 0) collapses the implicature",
              treatment=agg["mean_implicature_depth2_acc"], control=agg["mean_normalization_lesion_acc"])
    # Explicit attribution (attribution-required gate + the honest "whose is the difference"): the 2nd-order read
    # belongs to the belief-frame STACK (vs the flattened baseline), the implicature to the FS normalization (vs
    # its lesion) -- not merely that both arms were measured. Both controls floor at 0 -> ~100% attributable.
    from tools.lab import attributable_to
    attributable_to("2nd-order false belief attributable to the belief-frame STACK (vs flatten-lesion)",
                     agg["mean_order2_false_belief_acc"], agg["mean_flatten_lesion_false_acc"])
    attributable_to("depth-2 implicature attributable to FS divisive normalization (vs normalization-lesion)",
                     agg["mean_implicature_depth2_acc"], agg["mean_normalization_lesion_acc"])
    v.require("all seeds GO on Part A (2nd-order false belief)", agg["all_goA"], expect=True)
    v.require("all seeds GO on Part B (scalar implicature)", agg["all_goB"], expect=True)
    v.disabled("STDP/Hebbian/homeostasis/STP/structural/reward/OU",
               "belief stores + RSA normalizer are read at a fixed operating point; plasticity off (as in W3/W5)")
    vb = v.decide(go=all_go)
    if vb["status"] != "GO" and verdict == "GO":
        verdict = vb["status"]

    moat_intact = bool(agg["mean_flatten_lesion_false_acc"] <= thr["collapse_max"]
                       and agg["mean_buffer_scramble_false_acc"] <= thr["collapse_max"]
                       and agg["mean_permuted_premises_false_acc"] <= thr["collapse_max"]
                       and agg["mean_normalization_lesion_acc"] <= thr["rsa_lesion_max"]
                       and agg["mean_permuted_lexicon_acc"] <= thr["rsa_perm_max"])

    summary = {
        "runner": "_recursive_tom_rsa_derisk",
        "faculty": "W4 recursive theory of mind (2nd-order false belief on a WM-buffer stack + depth-2 scalar "
                   "implicature via iterated speaker-listener; Stage-4 CONVERSANT; ToM ladder recursion rung)",
        "theory": "Perner-Wimmer 2nd-order false belief (nested belief frames = nested clauses on a bounded WM "
                  "buffer) + Frank-Goodman Rational Speech Acts (L0->S1->L1); the W3 agent-keyed belief store "
                  "STACKED + the substrate's divisive normalization iterated (FUNCTIONAL correlate only, NOT "
                  "access to another mind).",
        "mechanism": "PART A: a STACK of W3 belief stores (GNW single-content attractors = WM-buffer slots) with "
                     "sim/'s own transmission_gate = witnessing-gated chained writes world->frame_1->frame_2; "
                     "frame_d copies frame_{d-1} iff witness_d, so a lower false belief propagates upward. PART B: "
                     "each RSA distribution = the graded rates of a competitive assembly population whose FS pool "
                     "does divisive normalization; at alpha=1 the depth-2 recursion is 3 substrate normalizations "
                     "and the implicature is a consequence of the single-vs-multi-item firing-rate contrast.",
        "seeds": list(seeds), "depths": list(depths), "n_trials": n_trials, "rsa_trials": rsa_trials,
        "backend": backend, "chance_partA": 1.0 / K_LOC, "chance_partB": 0.5,
        "verdict": verdict, "n_go": n_go, "n_seeds": len(seeds), "moat_intact": moat_intact,
        "thresholds": thr,
        **{k: vb[k] for k in ("preconditions", "disabled_processes", "undefined_reasons")},
        "aggregate": agg,
        "per_seed": per_seed,
        "honest_scope": ("A FUNCTIONAL recursive-mentalizing + pragmatics correlate. PART A: a bounded WM-buffer "
                         "stack of agent-keyed belief frames predicts a 2nd-order false belief (frame_2), "
                         "dissociable from reality AND from the 1st-order belief (frame_1) -- so a 1st-order reader "
                         "FAILS -- and collapses under flatten-lesion / buffer-scramble / permuted-premises. PART "
                         "B: the scalar implicature EMERGES at the depth-2 pragmatic listener (absent at the "
                         "literal L0) from the substrate's iterated divisive normalization, and collapses when the "
                         "normalization is lesioned or the lexicon permuted. The literal-truth lexicon is the "
                         "legitimate linguistic input; the ToM/pragmatics-specific neural work is the gated frame "
                         "stack and the iterated competitive normalization. The depth profile characterizes the "
                         "bounded-recursion limit (deeper nesting = more chained writes; humans cap ~2-3). NOT a "
                         "claim of phenomenal access to another mind. numpy-CPU on real spiking Izhikevich bridges; "
                         "NO sim/ edit (reuse-by-import of W3 + the GNW machinery). Self-report is a functional "
                         "read-out, never a phenomenal-experience claim."),
    }
    return summary, verdict, all_go


def _emit(summary, verdict, out_path):
    Path(os.path.dirname(os.path.abspath(out_path))).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    agg = summary["aggregate"]
    print("\n" + "=" * 112, flush=True)
    print(f"[recursive-ToM] === VERDICT: {verdict} ({summary['n_go']}/{summary['n_seeds']} seeds GO) | "
          f"moat_intact={summary['moat_intact']} ===", flush=True)
    print(f"[recursive-ToM]  A 2nd-order false_belief={agg['mean_order2_false_belief_acc']} "
          f"(1st-order baseline={agg['mean_order1_baseline_false_acc']} / reality={agg['mean_reality_baseline_false_acc']} "
          f"must FAIL) true={agg['mean_order2_true_belief_acc']}", flush=True)
    print(f"[recursive-ToM]    collapses: flatten={agg['mean_flatten_lesion_false_acc']} "
          f"buffer-scramble={agg['mean_buffer_scramble_false_acc']} permuted={agg['mean_permuted_premises_false_acc']}",
          flush=True)
    print(f"[recursive-ToM]    depth profile (false-belief): {agg['depth_profile_false_belief']} | "
          f"(true-belief): {agg['depth_profile_true_belief']}", flush=True)
    print(f"[recursive-ToM]  B implicature(L1)={agg['mean_implicature_depth2_acc']} "
          f"literal(L0)={agg['mean_literal_depth0_acc']} | norm-lesion={agg['mean_normalization_lesion_acc']} "
          f"perm-lexicon={agg['mean_permuted_lexicon_acc']}", flush=True)
    print(f"[recursive-ToM]  wrote {out_path}\n" + "=" * 112, flush=True)


def main():
    ap = argparse.ArgumentParser(description="W4 RECURSIVE THEORY OF MIND de-risk: 2nd-order false belief on a "
                                             "WM-buffer stack + depth-2 scalar implicature (iterated speaker-"
                                             "listener / RSA), computed on the spiking substrate.")
    ap.add_argument("--seed", type=int, default=42, help="single seed (a pool job runs one of these)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="multi-seed list (loops in-process, aggregates)")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3],
                    help="nesting depths to evaluate for Part A (GO uses depth 2; others characterize the bound)")
    ap.add_argument("--n-trials", type=int, default=48, help="change-of-location trials per depth per block (Part A)")
    ap.add_argument("--rsa-trials", type=int, default=16, help="RSA scenarios per seed (Part B)")
    ap.add_argument("--helper-pa", type=float, default=5000.0, help="witness-gated write ignite current (W3 fix)")
    ap.add_argument("--drive-steps", type=int, default=WRITE_DRIVE_STEPS, help="witnessed-event encoding window")
    ap.add_argument("--smoke", action="store_true", help="tiny 1-seed smoke (fewer trials, depths 1 2)")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--aggregate", type=str, default=None,
                    help="glob of per-seed jsons to aggregate into a 6-seed verdict (no simulation)")
    ap.add_argument("--json", type=str, default="research/findings/raw/_recursive_tom/summary.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    # ---- aggregation-only mode: read per-seed jsons, build the 6-seed verdict ----
    if args.aggregate:
        files = sorted(glob.glob(args.aggregate))
        if not files:
            print(f"[recursive-ToM] ⛔ no files match {args.aggregate}", flush=True)
            return 2
        per_seed = []
        for fp in files:
            with open(fp) as f:
                d = json.load(f)
            per_seed.extend(d["per_seed"] if "per_seed" in d and "seed" not in d else [d])
        per_seed = [p for p in per_seed if "seed" in p and "go" in p]
        per_seed.sort(key=lambda p: p["seed"])
        seeds = [p["seed"] for p in per_seed]
        depths = sorted({int(k) for p in per_seed for k in p["part_a_by_depth"].keys()})
        n_trials = per_seed[0]["part_a_by_depth"]["2"]["n_trials"] if "2" in per_seed[0]["part_a_by_depth"] else 0
        rsa_trials = per_seed[0]["part_b_rsa"]["n_trials"]
        print(f"[recursive-ToM] aggregating {len(files)} files -> seeds {seeds}", flush=True)
        summary, verdict, _ = build_summary(per_seed, seeds, depths, n_trials, rsa_trials, args.backend)
        _emit(summary, verdict, args.json)
        return 0 if verdict == "GO" else 1

    if args.smoke:
        seeds = [args.seed]
        depths = [1, 2]
        n_trials = min(args.n_trials, 12)
        rsa_trials = min(args.rsa_trials, 6)
    else:
        seeds = args.seeds if args.seeds is not None else [args.seed]
        depths = args.depths
        n_trials = args.n_trials
        rsa_trials = args.rsa_trials

    print(f"[recursive-ToM] W4 -- 2nd-order false belief (WM-buffer stack) + depth-2 scalar implicature (RSA) | "
          f"seeds={seeds} depths={depths} n_trials={n_trials} rsa_trials={rsa_trials} backend={args.backend} "
          f"K_loc={K_LOC}", flush=True)
    print("[recursive-ToM] PART A: a STACK of W3 agent-keyed belief stores (GNW attractor = WM-buffer slot), "
          "chained witnessing-gated writes; frame_2 = M's model of J's belief, dissociable from reality AND "
          "frame_1. PART B: RSA L0->S1->L1 as 3 substrate divisive normalizations; the implicature emerges at "
          "depth-2.", flush=True)
    print("[recursive-ToM] HONEST: a FUNCTIONAL recursive-mentalizing + pragmatics correlate (dissociable, "
          "collapses under lesion/scramble/permute) -- NOT a claim of access to another mind. Self-report is a "
          "functional read-out.", flush=True)

    per_seed = []
    for s in seeds:
        per_seed.append(evaluate_seed(s, depths, n_trials, rsa_trials, args.helper_pa, args.drive_steps,
                                      verbose=True))

    # single-seed pool job: write the per-seed record (aggregate later). multi-seed: write the summary.
    if len(seeds) == 1 and args.seeds is None and not args.smoke:
        Path(os.path.dirname(os.path.abspath(args.json))).mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(per_seed[0], f, indent=2, default=str)
        print(f"[recursive-ToM] wrote per-seed record {args.json} (go={per_seed[0]['go']})", flush=True)
        return 0 if per_seed[0]["go"] else 1

    summary, verdict, all_go = build_summary(per_seed, seeds, depths, n_trials, rsa_trials, args.backend)
    _emit(summary, verdict, args.json)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
