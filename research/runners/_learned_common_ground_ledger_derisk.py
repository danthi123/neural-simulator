"""Lane · Conversational pragmatics — the COMMON-GROUND LEDGER driving AUDIENCE DESIGN, on ONE spiking substrate.

THE FACULTY (a conversation-DRIVING pragmatics primitive). A competent speaker keeps a running COMMON GROUND: the set
of referents that have been MUTUALLY ESTABLISHED so far this conversation (Clark & Brennan 1991 grounding; Clark &
Marshall 1981; Brennan-Clark 1996 conceptual pacts). That ledger DRIVES audience design: a referent already in common
ground is REDUCED/omitted ("it", a pronoun) while a NOT-yet-grounded referent is INTRODUCED with a full description
("the red ball on the table"). Change ONE thing about the ledger and the very NEXT thing the speaker says changes:
this is the load-bearing coupling that makes the faculty drive the conversation, not merely observe it.

BIOLOGY: common ground is hippocampal/declarative (Duff & Brown-Schmidt 2012 — amnesics fail to build it); it is a
PERSISTENT record that must HOLD across the whole conversation and be UPDATED per grounding act. The persistence is a
self-sustaining NMDA attractor (the same GNW ignition machinery the workspace / self-schema / false-belief builds GO'd
on); each referent's grounded bit is one bistable assembly, latched ON by a grounding act and held by its recurrence.
Audience design is a spiking BIASED-COMPETITION decision (Namburi-Tye opponent motif): a novelty PRIOR biases
INTRODUCE, and evidence that the queried referent is already grounded (read out of the substrate through a gated
synapse) biases REDUCE; the winner is the speaker's choice.

HONEST SCOPE (carried into every line): a FUNCTIONAL common-ground / audience-design correlate — a persistent per-
referent grounded-state ledger, read at speak-time to choose introduce-vs-reduce, dissociable from any fixed pattern
(it tracks the ACTUAL grounding history) and load-bearing (its lesion makes audience design go static). It is NOT a
claim of the full conceptual-pact machinery (lexical entrainment, partner-specificity) — those are named follow-ons.

THE MECHANISM (reuse-by-import of the validated GNW ignition + snapshot/restore + the biased-competition motif; ONE
numpy Izhikevich `SimulationBridge`, NO `sim/` edit):
  * LEDGER = K independent bistable referent stores `led0..led{K-1}`, each = a densely self-recurrent NMDA assembly
    (`_build_assembly_loop_population`, weight DEFAULT_ATTRACTOR_WEIGHT) + a per-store FS pool that STABILIZES it
    (led_k -> led_k_fs -> led_k, the ignition-bridge lateral inhibition). Independent (each its own FS) so MANY
    referents can be simultaneously grounded — this is NOT the single-content GNW workspace, it is a MULTI-SLOT
    ledger.
  * A GROUNDING ACT on referent k ignites led_k (external IGNITE_PA to its members for GROUND_DRIVE_STEPS = the world
    delivering the grounding event); the NMDA recurrence then HOLDS the grounded bit through the rest of the
    conversation (self-sustained across all later queries). Grounding a second referent leaves the first latched.
  * AUDIENCE DESIGN at speak-time (query on target t): a per-referent GATED read `led_k -> evidence`
    (transmission_gate "query_k"); opening ONLY query_t routes led_t's PERSISTENT FIRING into the shared `evidence`
    pool. `evidence -> reduce` (excitatory). A novelty PRIOR (tonic INTRO_PA to `introduce`) biases INTRODUCE.
    `reduce`/`introduce` mutually inhibit through their FS interneurons (Namburi-Tye biased competition). Winner:
    led_t grounded -> evidence fires -> REDUCE wins (omit/pronominalize); led_t ungrounded -> evidence silent ->
    the novelty prior wins -> INTRODUCE (full description). THE LEDGER READ IS A SUBSTRATE READ: evidence fires only
    because led_t's neurons are firing and drive it through a real synapse — NOT a host dict lookup.

GO GATE (6-seed {42 43 44 100 101 102}): audience design FOLLOWS the ledger + is lesion-load-bearing:
  * audience_design_acc >= 0.90   (chance 0.5; balanced grounded/ungrounded query targets) — REDUCE iff grounded
  * separation real - lesion >= 0.25 AND real - permute >= 0.25 (the two anti-cheat controls collapse the read)
  * substrate-read margin: evidence rate on grounded targets >> on ungrounded (the read rides real firing)

MANDATORY ANTI-CHEATS (all wired into the runner's OWN verdict; a GO whose anti-cheats fail is a NO-GO):
  (1) PERMUTED GROUNDING HISTORY -> audience design WRONG. Ground a PERMUTED referent set (same count, different
      referents), score audience design against the TRUE required design -> collapses to chance (~0.5). Proves
      audience design tracks WHICH referents were ACTUALLY grounded, not a fixed/base-rate pattern.
  (2) LESION THE LEDGER-UPDATE -> audience design goes STATIC. Cut the ledger's write-persistence: build the referent
      self-recurrent loops at weight 0 (the validated load-bearing-recurrence lesion). Grounding acts still ignite
      but nothing HOLDS -> by speak-time every store has decayed -> the ledger cannot record any grounding -> every
      query reads ungrounded -> audience design goes static (always INTRODUCE), IGNORING what was grounded.
      (frac_introduce_lesion ~ 1.0; acc collapses to the ungrounded base rate.)
  (3) THE LEDGER READ IS A SUBSTRATE READ, not a host dict. By construction `evidence` fires only via the gated
      synaptic pathway from led_t's spiking neurons; verified by evidence_rate_grounded >> evidence_rate_ungrounded
      (a dict lookup would leave evidence firing-independent). Reported + gated.

DISCIPLINE: SIM_BACKEND=numpy (CPU lane); reuse-by-import; NO `sim/` edit. cfg.seed set per-seed (SEEDS THE SUBSTRATE
— NOT actual_seed_used, the CLAUDE.md gotcha).

Run (smoke):  SIM_BACKEND=numpy python -u -m research.runners._learned_common_ground_ledger_derisk --smoke
Run (6-seed): SIM_BACKEND=numpy OMP_NUM_THREADS=4 python -u -m research.runners._learned_common_ground_ledger_derisk \
                  --seeds 42 43 44 100 101 102 --out research/findings/raw/four_day/_learned_common_ground_ledger_derisk_6seed.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

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

# reuse-by-import: the validated GNW spiking machinery (the self-recurrent NMDA attractor loop that holds a bistable
# bit + the wash-out snapshot/restore) + the self-schema / ignition constants. The common-ground ledger IS a bank of
# those same attractors, one per referent, read by a biased-competition decision.
from research.runners._gnw_rung1_ignition_curve_derisk import (
    _build_assembly_loop_population, _snapshot_state, _restore_state,
    DEFAULT_ATTRACTOR_WEIGHT, SETTLE_STEPS,
    ASSEMBLY_SIZE, WS_LOOP_GATE, WS_TO_FS_WEIGHT, FS_TO_WS_WEIGHT,
)
from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection
from research.runners._self_schema_region_derisk import IGNITE_PA
from tools.verdict import Verdict            # a verdict that carries its own preconditions into the artifact
from tools.lab import attributable_to        # force the treatment/control subtraction to be asked out loud

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_learned_common_ground_ledger_6seed.json"

# ── geometry / operating point ───────────────────────────────────────────────────────────────────────────────
K_REF = 6                       # distinct referents (chance for the binary introduce/reduce decision = 0.5)
LED_ASSEMBLY = ASSEMBLY_SIZE    # per-referent grounded-store assembly (reuse validated 80-neuron attractor)
LED_FS_N = 20                   # per-referent FS stabilizer (lateral inhibition; independent so slots don't compete)
ATTR_W = 55.0                   # per-referent self-recurrent attractor weight. Raised above the reused 30 so EVERY
                                # heterogeneity-adverse referent assembly (not just the lucky one) reliably latches
                                # into the self-sustaining basin — the K-store analogue of false-belief's write-
                                # reliability fix (a single global weight had to serve ALL K heterogeneous stores).
N_EVID = 40                     # shared grounded-evidence read-out pool
N_DEC = 40                      # per decision pool (reduce / introduce)
N_DEC_FS = 15                   # per decision-pool opponent interneuron pool

W_LED_EVID = 3.5                # gated led_k -> evidence (the SUBSTRATE read of the grounded bit; tuned so persistent
                                # led firing drives evidence supra-threshold, silence leaves it quiescent)
W_EVID_REDUCE = 24.0            # evidence -> reduce (grounded evidence pushes the REDUCE choice)
W_DEC_INH = 8.0                 # decision pool -> its opponent interneuron
W_INH_DEC = 30.0                # opponent interneuron -> the OTHER decision pool (cross-inhibition, gaba_a)

INTRO_PA = 220.0                # tonic novelty-PRIOR current to `introduce` during a query (biases INTRODUCE; the
                                # grounded-evidence drive to REDUCE must overcome it -> the ledger read is decisive)
GROUND_DRIVE_STEPS = 50         # grounding-act ignition window (world delivers the grounding event); a longer window
                                # reliably drives the fresh-from-quiescence ignition into the self-sustaining basin
HOLD_STEPS = 25                 # inter-act / pre-query hold: external current off, grounded stores self-sustain
QUERY_STEPS = 45                # speak-time read window (decision settles under biased competition)
QUERY_FLUSH = 30                # inter-query drain: the non-NMDA decision pools decay to rest; the NMDA ledger HOLDS


# ── build the one-brain bridge: K referent stores (bistable attractors) + the audience-design decision circuit ──
def build_cg_bridge(seed: int = 42, lesion_update: bool = False):
    """One `SimulationBridge`. K referent grounded-stores (`led_k`, each a self-recurrent NMDA attractor + its own
    FS stabilizer) + a shared `evidence` read pool + a `reduce`/`introduce` biased-competition decision. Gated
    `led_k -> evidence` pathways carry the per-referent SUBSTRATE read (transmission_gate query_k). `lesion_update`
    builds the referent self-loops at weight 0 (the load-bearing-recurrence lesion) -> the ledger cannot HOLD a
    grounded bit -> it goes static. Returns (bridge, xp, idx, snap)."""
    xp, _ = get_backend()

    regions = []
    for k in range(K_REF):
        regions.append(BrainRegion(name=f"led{k}", n_neurons=LED_ASSEMBLY, exc_fraction=1.0,
                                   internal_density=0.0, enable_nmda=True))
        regions.append(BrainRegion(name=f"led{k}_fs", n_neurons=LED_FS_N, exc_fraction=0.0,
                                   internal_density=0.0, enable_nmda=False))
    regions += [
        BrainRegion(name="evidence", n_neurons=N_EVID, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="reduce", n_neurons=N_DEC, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="introduce", n_neurons=N_DEC, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="reduce_fs", n_neurons=N_DEC_FS, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="introduce_fs", n_neurons=N_DEC_FS, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
    ]

    pathways = []
    # per-referent lateral inhibition (stabilize each attractor; independent so slots don't mutually compete)
    for k in range(K_REF):
        pathways.append(RegionPathway(from_region=f"led{k}", to_region=f"led{k}_fs", density=0.5,
                                      weight_mean=WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False))
        pathways.append(RegionPathway(from_region=f"led{k}_fs", to_region=f"led{k}", density=0.5,
                                      weight_mean=FS_TO_WS_WEIGHT, weight_jitter=0.0, plastic=False))
    # the audience-design decision: evidence -> reduce; biased-competition cross-inhibition reduce <-> introduce
    pathways.append(RegionPathway(from_region="evidence", to_region="reduce", density=0.7,
                                  weight_mean=W_EVID_REDUCE, weight_jitter=0.1, plastic=False))
    pathways.append(RegionPathway(from_region="reduce", to_region="reduce_fs", density=0.6,
                                  weight_mean=W_DEC_INH, weight_jitter=0.1, plastic=False))
    pathways.append(RegionPathway(from_region="reduce_fs", to_region="introduce", density=0.7,
                                  weight_mean=W_INH_DEC, weight_jitter=0.1, plastic=False, receptor="gaba_a"))
    pathways.append(RegionPathway(from_region="introduce", to_region="introduce_fs", density=0.6,
                                  weight_mean=W_DEC_INH, weight_jitter=0.1, plastic=False))
    pathways.append(RegionPathway(from_region="introduce_fs", to_region="reduce", density=0.7,
                                  weight_mean=W_INH_DEC, weight_jitter=0.1, plastic=False, receptor="gaba_a"))

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
    cfg.stdp_w_max = max(400.0, float(DEFAULT_ATTRACTOR_WEIGHT) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(DEFAULT_ATTRACTOR_WEIGHT) * 4.0)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager
    led_m = {k: np.asarray(rm.indices(f"led{k}"), dtype=np.int64) for k in range(K_REF)}
    evid_idx = np.asarray(rm.indices("evidence"), dtype=np.int64)
    reduce_idx = np.asarray(rm.indices("reduce"), dtype=np.int64)
    intro_idx = np.asarray(rm.indices("introduce"), dtype=np.int64)

    union = dict(rm.build_wiring_plan(seed=int(seed)))
    loop_w = 0.0 if lesion_update else float(ATTR_W)   # lesion = the ledger cannot HOLD (weight 0)
    for k in range(K_REF):
        union[f"loop_led{k}"] = _build_assembly_loop_population(led_m[k], loop_w)
        # gated per-referent SUBSTRATE read: led_k -> evidence, conducts only when query_k is open
        d = _dense_projection(led_m[k], evid_idx, float(W_LED_EVID), WS_LOOP_GATE)
        d["transmission_gate"] = f"query_{k}"
        union[f"read_led{k}"] = d

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)
    for k in range(K_REF):
        bridge.set_transmission_gate(f"query_{k}", 0.0)     # all reads closed by default; opened per query

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)

    idx = {
        "led_dev": {k: xp.asarray(v) for k, v in led_m.items()},
        "evid_dev": xp.asarray(evid_idx),
        "reduce_dev": xp.asarray(reduce_idx),
        "intro_dev": xp.asarray(intro_idx),
    }
    return bridge, xp, idx, snap


# ── one conversation: ground a set of referents, then read audience design for a set of query targets ──────────
def run_conversation(bridge, xp, idx, snap, grounded_set):
    """Reset to quiescence, run the grounding acts (ignite each referent in `grounded_set`, holding the ledger),
    then for EVERY referent read the audience-design decision (introduce vs reduce). Returns per-target
    (winner, reduce_rate, introduce_rate, evidence_rate)."""
    led_dev = idx["led_dev"]; evid_dev = idx["evid_dev"]
    reduce_dev = idx["reduce_dev"]; intro_dev = idx["intro_dev"]

    bridge.cp_external_input_current[:] = 0.0
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0

    # --- grounding phase: ignite each grounded referent in turn; previously-grounded stores keep self-sustaining ---
    for k in grounded_set:
        for _ in range(GROUND_DRIVE_STEPS):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[led_dev[k]] = xp.float32(IGNITE_PA)
            bridge._run_one_simulation_step()
        for _ in range(HOLD_STEPS):
            bridge.cp_external_input_current[:] = 0.0
            bridge._run_one_simulation_step()

    # settle the ledger before the first query (decision pools drain; NMDA stores hold)
    for _ in range(HOLD_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()

    # --- speak-time: for each target referent, read introduce-vs-reduce with only that query gate open ---
    out = {}
    late = QUERY_STEPS - max(1, QUERY_STEPS // 3)
    for t in range(K_REF):
        # drain the decision pools without disturbing the (self-sustaining) ledger
        for _ in range(QUERY_FLUSH):
            bridge.cp_external_input_current[:] = 0.0
            bridge._run_one_simulation_step()
        bridge.set_transmission_gate(f"query_{t}", 1.0)
        r_reduce = r_intro = r_evid = 0.0
        for step in range(QUERY_STEPS):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[intro_dev] = xp.float32(INTRO_PA)   # novelty prior -> INTRODUCE
            bridge._run_one_simulation_step()
            if step >= late:
                fs = bridge.cp_firing_states
                r_reduce += float(to_host(fs[reduce_dev].astype(xp.float64).sum()))
                r_intro += float(to_host(fs[intro_dev].astype(xp.float64).sum()))
                r_evid += float(to_host(fs[evid_dev].astype(xp.float64).sum()))
        bridge.set_transmission_gate(f"query_{t}", 0.0)
        nlate = float(QUERY_STEPS - late)
        rr = r_reduce / (nlate * N_DEC)
        ri = r_intro / (nlate * N_DEC)
        re = r_evid / (nlate * N_EVID)
        margin = rr - ri
        winner = "reduce" if margin > 1e-9 else ("introduce" if margin < -1e-9 else None)
        out[t] = {"winner": winner, "reduce_rate": rr, "introduce_rate": ri, "evidence_rate": re, "margin": margin}
    return out


def _make_grounded_set(seed, n_grounded):
    """Choose which referents are grounded this conversation (balanced ~half of K), as an ordered list."""
    rng = np.random.default_rng(seed * 131 + 7)
    order = rng.permutation(K_REF).tolist()
    return sorted(order[:n_grounded]), rng


def _audience_design_acc(reads, grounded_set, rng_tiebreak):
    """Ground truth: REDUCE iff the target is grounded, else INTRODUCE. A genuine tie (both pools silent, e.g. under
    lesion with the store decayed and no evidence, or a degenerate) is broken RANDOMLY so a no-signal control scores
    true chance, not a balanced-class artifact. Returns (acc, frac_introduce, per_target correctness list)."""
    gset = set(grounded_set)
    correct = []
    n_intro = 0
    for t in range(K_REF):
        w = reads[t]["winner"]
        if w is None:
            w = "reduce" if rng_tiebreak.random() < 0.5 else "introduce"
        n_intro += int(w == "introduce")
        want = "reduce" if t in gset else "introduce"
        correct.append(int(w == want))
    return float(np.mean(correct)), n_intro / K_REF, correct


# ── evaluate one seed: intact GO + permute anti-cheat + lesion anti-cheat + substrate-read check ────────────────
def evaluate_seed(seed, n_conv, thresholds, verbose=False):
    rng = np.random.default_rng(seed * 977 + 3)
    n_grounded = K_REF // 2                       # balanced: half the referents grounded per conversation

    # ---- INTACT: audience design must FOLLOW the actual ledger ----
    bridge, xp, idx, snap = build_cg_bridge(seed=seed, lesion_update=False)
    real_accs, real_evid_g, real_evid_u = [], [], []
    permute_accs = []
    conv_specs = []
    for _ in range(n_conv):
        gset, _r = _make_grounded_set(int(rng.integers(1 << 30)), n_grounded)
        conv_specs.append(gset)
        reads = run_conversation(bridge, xp, idx, snap, gset)
        acc, _fi, _c = _audience_design_acc(reads, gset, rng)
        real_accs.append(acc)
        gs = set(gset)
        real_evid_g += [reads[t]["evidence_rate"] for t in range(K_REF) if t in gs]
        real_evid_u += [reads[t]["evidence_rate"] for t in range(K_REF) if t not in gs]

        # ---- (1) PERMUTED GROUNDING HISTORY: ground a DIFFERENT (permuted) referent set, score vs the TRUE design ----
        perm = rng.permutation(K_REF).tolist()
        pset = sorted(perm[:n_grounded])
        reads_p = run_conversation(bridge, xp, idx, snap, pset)
        acc_p, _fip, _cp = _audience_design_acc(reads_p, gset, rng)   # scored against the ORIGINAL required design
        permute_accs.append(acc_p)

    real_acc = float(np.mean(real_accs))
    permute_acc = float(np.mean(permute_accs))
    evid_grounded = float(np.mean(real_evid_g)) if real_evid_g else 0.0
    evid_ungrounded = float(np.mean(real_evid_u)) if real_evid_u else 0.0

    # ---- (2) LESION THE LEDGER-UPDATE: recurrence weight 0 -> the ledger cannot HOLD -> audience design goes static ----
    lb, lxp, lidx, lsnap = build_cg_bridge(seed=seed, lesion_update=True)
    lesion_accs, lesion_frac_intro = [], []
    for gset in conv_specs:
        reads_l = run_conversation(lb, lxp, lidx, lsnap, gset)
        acc_l, fi_l, _cl = _audience_design_acc(reads_l, gset, rng)
        lesion_accs.append(acc_l); lesion_frac_intro.append(fi_l)
    lesion_acc = float(np.mean(lesion_accs))
    lesion_frac_introduce = float(np.mean(lesion_frac_intro))

    r = {
        "seed": int(seed), "n_conv": int(n_conv), "n_grounded": int(n_grounded), "K_ref": int(K_REF),
        "audience_design_acc": real_acc,
        "permute_acc": permute_acc,
        "lesion_acc": lesion_acc,
        "lesion_frac_introduce": lesion_frac_introduce,
        "evidence_rate_grounded": evid_grounded,
        "evidence_rate_ungrounded": evid_ungrounded,
        "real_minus_lesion": real_acc - lesion_acc,
        "real_minus_permute": real_acc - permute_acc,
    }
    # per-seed GO
    r["go"] = bool(
        real_acc >= thresholds["acc"]
        and (real_acc - lesion_acc) >= thresholds["separation"]
        and (real_acc - permute_acc) >= thresholds["separation"]
        and lesion_frac_introduce >= thresholds["static_frac"]
        and (evid_grounded - evid_ungrounded) >= thresholds["read_margin"]
    )
    if verbose:
        print(f"  [seed {seed}] audience-design acc={real_acc:.3f} (chance 0.5) | permute {permute_acc:.3f} | "
              f"lesion {lesion_acc:.3f} (frac-introduce {lesion_frac_introduce:.2f}) | evid grounded "
              f"{evid_grounded:.3f} vs ungrounded {evid_ungrounded:.3f} || GO={r['go']}", flush=True)
    return r


DEFAULT_THRESHOLDS = {
    "acc": 0.90,            # intact audience-design accuracy (chance 0.5)
    "separation": 0.25,     # real must beat BOTH controls by this margin
    "static_frac": 0.85,    # lesion goes static: nearly always INTRODUCE (ignores what was grounded)
    "read_margin": 0.05,    # evidence fires on grounded >> ungrounded targets (the read is a substrate read)
}


def _aggregate_verdict(rows, thresholds):
    def m(k):
        return float(np.mean([r[k] for r in rows]))
    real = m("audience_design_acc"); perm = m("permute_acc"); les = m("lesion_acc")
    static = m("lesion_frac_introduce")
    eg = m("evidence_rate_grounded"); eu = m("evidence_rate_ungrounded")
    n_go = sum(1 for r in rows if r["go"])
    all_go = bool(n_go == len(rows))

    means = {"audience_design_acc": real, "permute_acc": perm, "lesion_acc": les,
             "lesion_frac_introduce": static, "evidence_rate_grounded": eg, "evidence_rate_ungrounded": eu}

    v = Verdict("common-ground ledger -> audience design", chance=0.5)
    v.require("6 seeds (project bar)", len(rows) >= 6, expect=True)
    v.floor("audience-design acc beats chance", measured=real, floor=0.5)
    v.control("lesion the ledger-update (recurrence=0) collapses audience design",
              treatment=real, control=les, min_separation=thresholds["separation"])
    v.control("permuted grounding history collapses audience design (vs true required design)",
              treatment=real, control=perm, min_separation=thresholds["separation"])
    v.require("lesion goes STATIC (nearly always INTRODUCE -> ignores what was grounded)",
              static, expect=lambda x: x >= thresholds["static_frac"])
    v.reaches("the ledger read is a SUBSTRATE read (evidence fires on grounded >> ungrounded)",
              before=eu, after=eg)
    v.require("substrate-read margin (grounded - ungrounded evidence rate)",
              eg - eu, expect=lambda x: x >= thresholds["read_margin"])
    v.disabled("STDP / Hebbian / reward-mod / homeostasis / short-term & structural plasticity",
               why="fixed-structure ledger + biased-competition read is the scope; the LEARNED conceptual-pact / "
                   "lexical-entrainment / partner-specific ledger is the named follow-on")
    decided = v.decide(go=all_go, verbose=False)

    # force the treatment/control subtraction to be asked out loud
    attributable_to("the actual ledger contents (vs a lesioned, unheld ledger)", real, les)
    return all_go, n_go, means, decided


def main():
    ap = argparse.ArgumentParser(description="Common-ground ledger -> audience design de-risk (conversational "
                                             "pragmatics; one spiking substrate).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42, help="single seed (used by --smoke)")
    ap.add_argument("--n-conv", type=int, default=8, help="conversations per seed (each = a grounding set + queries)")
    ap.add_argument("--smoke", action="store_true", help="1 seed, fewer conversations — proves it RUNS + arms live")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if a.smoke:
        seeds = [a.seed]
        n_conv = min(a.n_conv, 4)
    else:
        seeds = a.seeds
        n_conv = a.n_conv

    print(f"[common-ground] COMMON-GROUND LEDGER -> AUDIENCE DESIGN (conversational pragmatics). seeds={seeds} "
          f"n_conv={n_conv} K_ref={K_REF} (chance 0.5)", flush=True)
    print(f"[common-ground] GATE: audience-design acc >= {DEFAULT_THRESHOLDS['acc']} | real-lesion & real-permute >= "
          f"{DEFAULT_THRESHOLDS['separation']} | lesion goes static | evidence read is a substrate read", flush=True)
    print("[common-ground] HONEST: a FUNCTIONAL common-ground/audience-design correlate (persistent per-referent "
          "grounded-state ledger, read at speak-time to choose introduce-vs-reduce). NOT full conceptual-pact ToM.",
          flush=True)

    t0 = time.time()
    rows = [evaluate_seed(s, n_conv, DEFAULT_THRESHOLDS, verbose=True) for s in seeds]
    all_go, n_go, means, decided = _aggregate_verdict(rows, DEFAULT_THRESHOLDS)
    verdict = "GO" if all_go else ("PARTIAL" if n_go > 0 else "NEGATIVE")
    if decided["status"] != "GO" and verdict == "GO":
        verdict = decided["status"]

    out = {
        "runner": "_learned_common_ground_ledger_derisk",
        "faculty": "common-ground ledger -> audience design (conversational pragmatics)",
        "theory": "Clark & Brennan 1991 grounding; Clark & Marshall 1981 common ground; Brennan-Clark 1996 conceptual "
                  "pacts; Duff & Brown-Schmidt 2012 (common ground is hippocampal/declarative) — FUNCTIONAL correlate",
        "mechanism": "K bistable referent stores (self-recurrent NMDA attractors, one per referent) latched by "
                     "grounding acts + held by recurrence; gated per-referent SUBSTRATE read (led_k -> evidence) into "
                     "a Namburi-Tye biased-competition reduce/introduce decision biased by a novelty prior",
        "seeds": seeds, "n_conv": n_conv, "backend": "numpy",
        "knobs": {"K_REF": K_REF, "LED_ASSEMBLY": LED_ASSEMBLY, "LED_FS_N": LED_FS_N, "N_EVID": N_EVID,
                  "N_DEC": N_DEC, "attractor_weight": float(DEFAULT_ATTRACTOR_WEIGHT), "W_LED_EVID": W_LED_EVID,
                  "W_EVID_REDUCE": W_EVID_REDUCE, "W_INH_DEC": W_INH_DEC, "INTRO_PA": INTRO_PA,
                  "GROUND_DRIVE_STEPS": GROUND_DRIVE_STEPS, "HOLD_STEPS": HOLD_STEPS, "QUERY_STEPS": QUERY_STEPS,
                  "QUERY_FLUSH": QUERY_FLUSH, "chance": 0.5},
        "thresholds": DEFAULT_THRESHOLDS,
        "verdict": verdict, "n_go": n_go, "n_seeds": len(seeds),
        "status": decided["status"], "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"], "undefined_reasons": decided["undefined_reasons"],
        "means": means,
        "per_seed": rows,
        "honest_scope": ("A functional common-ground / audience-design correlate: a persistent per-referent "
                         "grounded-state ledger (bistable NMDA attractors) read at speak-time through a gated synapse "
                         "to choose introduce-vs-reduce. Audience design FOLLOWS the actual grounding history "
                         "(permuting it collapses the read) and is load-bearing (lesioning the ledger's write-"
                         "persistence makes audience design go static). NOT the learned conceptual-pact / lexical-"
                         "entrainment / partner-specific machinery — those are named follow-ons."),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))

    print("\n" + "=" * 110, flush=True)
    print(f"[common-ground] === VERDICT: {verdict} ({n_go}/{len(seeds)} seeds GO) ===", flush=True)
    print(f"[common-ground]   audience-design acc={means['audience_design_acc']:.3f} (chance 0.5) | "
          f"permute {means['permute_acc']:.3f} | lesion {means['lesion_acc']:.3f} "
          f"(frac-introduce {means['lesion_frac_introduce']:.2f})", flush=True)
    print(f"[common-ground]   substrate-read: evidence grounded={means['evidence_rate_grounded']:.3f} >> "
          f"ungrounded={means['evidence_rate_ungrounded']:.3f}", flush=True)
    print(f"[common-ground]   elapsed={time.time()-t0:.1f}s  wrote {a.out}", flush=True)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
