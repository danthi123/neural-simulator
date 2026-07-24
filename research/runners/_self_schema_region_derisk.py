"""DR-3: the attention/agency SELF-SCHEMA region -- the first functional step toward self-awareness.

THE OWNER REFRAME (context): a genuinely-conversing, self-aware sim-brain that builds a MODEL OF ITSELF and can
REPORT it. This is DR-3 of docs/plans/2026-07-22-genuine-conversation-affective-self-aware-brain-plan.md (P1.1).
Biology/theory: Graziano's ATTENTION SCHEMA THEORY (the brain builds a simplified model of its own attention ->
what we call "awareness") + higher-order metacognition (a self-model that represents the brain's own knowing).

HONEST BAR (carried into every line): this is a FUNCTIONAL SELF-MODEL CORRELATE -- a region that represents +
reports three internal-state axes read from the brain's OWN signals. It is NOT, and makes no claim of, subjective
experience / phenomenal consciousness (OPEN, arguably untestable -- Chalmers hard problem). We build + measure the
correlate; we never assert the experience.

THE MECHANISM (reuse-by-import, NO `sim/` edit). ONE spiking `SimulationBridge`, three regions:
  1. `workspace` (exc, NMDA) -- K content MEMBER assemblies, each a dense self-recurrent attractor loop @ weight 30
     (GNW Rung-1 `_build_assembly_loop_population`), sharing ONE inhibitory `workspace_fs` pool (GNW Rung-2 mutual
     inhibition -> only ONE content ignites at a time = the GLOBAL-WORKSPACE OCCUPANCY, "what is currently
     broadcast"). This is the project's own validated spiking GNW (Dehaene-Changeux ignition + Baars one-spotlight).
  2. `workspace_fs` (inhib) -- the shared WTA inhibition.
  3. `self_schema` (exc) -- the SELF region. Three DISJOINT sub-blocks, each reading ONE internal signal:
       * `attend[k]` (K sub-blocks): a FIXED dense feed-forward projection from workspace member-k -> attend-k
         (GNW Rung-3b member->report). Which attend sub-block sustains = the schema's MODEL of "what I am
         attending to" (a DOWNSTREAM read of the real GNW occupancy, a separate population from the content).
       * `confid`: driven by graded external current proportional to the CONFIDENCE/FAMILIARITY scalar (in
         production the Bogacz-Brown familiarity gate; here a REAL graded novelty read of a per-trial cue via the
         reused `RealAntiHebbianFamiliarity`). Its firing RATE encodes "how sure I am".
       * `author`: driven by a binary current when the thought is SELF-generated (vs externally heard) = the
         AUTHORSHIP tag "did I author this".
The schema's late-window firing thus jointly REPRESENTS (attention, confidence, authorship); the read-outs REPORT
each -- "what I'm thinking about / how sure I am / whether I authored it".

WHY THIS IS NOT A TRIVIAL "inject X, read X". (a) The ATTENTION axis is read from a genuine SUSTAINED spiking GNW
attractor (the workspace latches one content; the attend block reads the sustained assembly -- a real occupancy
read, not a passthrough). (b) The value the self-model adds is the JOINT, DISSOCIABLE representation of three axes
read from the brain's own state, all recoverable at once, that COLLAPSES to chance when the schema's access is
severed (self-lesion) or when the fed internal signals are decorrelated from the true state (shuffle). The
anti-cheats below are what license the claim.

THE DE-RISK (single per-trial state = (content c, confidence q, authorship a), all sampled INDEPENDENTLY so
confidence/authorship are genuinely ORTHOGONAL axes to content -- you can think about the same thing surely or
unsurely). Per trial: ignite content c on the workspace; hold the confidence current (∝ q) + authorship current
tonically; free-run; read the schema's late-window sub-block rates; decode (c^, q^, a^); score vs the TRUE (c,q,a).

GO GATE (the schema's read-outs TRACK the ground-truth internal state, 6-seed, well above chance):
  * attention_acc     >= 0.85   (chance 1/K = 0.25) -- reports WHICH content is in the workspace
  * confidence_spearman >= 0.6 AND confidence_hilo_acc >= 0.85 -- MONOTONE in the true confidence
  * authorship_acc    >= 0.90   (chance 0.5)          -- reports self-vs-heard
ANTI-CHEATS (all must hold):
  (1) SELF-LESION -- sever the schema's ACCESS to the internal signals (member->attend weight 0 + no confid/author
      drive; the underlying workspace still ignites, so the BRAIN state is unchanged -- only the schema's READ is
      cut) -> ALL three read-outs drop to chance. Proves the report rides the REAL internal state.
  (2) SCHEMA-IS-ABOUT-STATE-NOT-CONTENT -- confidence/authorship are a SEPARATE axis from content: |corr(decoded
      confidence, content)| ~ 0 AND |corr(decoded authorship, content)| ~ 0, and confidence tracking holds uniformly
      across content identities (the schema reports HOW-SURE regardless of WHAT). Reports the dissociation.
  (3) SHUFFLED-INTERNAL-SIGNAL -- drive the schema with a permuted (c',q',a') decorrelated from the trial's true
      (c,q,a); score vs TRUE -> collapse to chance. Proves it tracks the ACTUAL signal, not a fixed pattern.
  (extra) FAMILIARITY-LESION -- lesion the reused familiarity gate -> the confidence scalar flattens -> the
      confidence read-out can no longer track true confidence. Grounds the confidence source in a real reused signal.

Usage:
  # CPU smoke (1 seed, tiny -- proves it runs, controls live, prints a verdict):
  python -u -m research.runners._self_schema_region_derisk --smoke --seed 42 \
      --json research/findings/raw/_self_schema_smoke.json --backend numpy
  # full 6-seed (local CPU):
  python -u -m research.runners._self_schema_region_derisk --seeds 42 43 44 100 101 102 --n-trials 96 \
      --json research/findings/raw/_self_schema_6seed.json --backend numpy
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

# reuse-by-import: the validated GNW spiking machinery + the Bogacz-Brown familiarity gate.
from research.runners._gnw_rung1_ignition_curve_derisk import (
    _build_assembly_loop_population, _snapshot_state, _restore_state,
    DEFAULT_ATTRACTOR_WEIGHT, SETTLE_STEPS, DRIVE_STEPS, FREE_STEPS,
)
from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection
from research.runners._phaseB_biologize_moat_streamcodes_derisk import RealAntiHebbianFamiliarity


# ── geometry ───────────────────────────────────────────────────────────────────────────────────────────────
K_CONTENTS = 4               # number of distinct workspace contents (chance for attention = 1/K = 0.25)
ASSEMBLY_SIZE = 80           # per-content workspace member assembly (self-recurrent ignitable unit; Rung-1)
WORKSPACE_FS_N = 50          # shared inhibitory pool (Rung-2 mutual inhibition)
ATTEND_SIZE = 50             # per-content self_schema attend sub-block (reads the workspace occupancy)
CONFID_SIZE = 60             # self_schema confidence sub-block (graded read of the familiarity scalar)
AUTHOR_SIZE = 60             # self_schema authorship sub-block (binary self-vs-heard tag)

WS_LOOP_GATE = "workspace_loop_fixed"
WS_TO_FS_WEIGHT = 6.0
FS_TO_WS_WEIGHT = 16.0        # shared mutual inhibition (Rung-2)
MEMBER_TO_ATTEND_W = 12.0     # workspace member -> its attend sub-block (Rung-3b MEMBER_TO_REPORT_W)

IGNITE_FRAC = 0.5
SOLO_PLATEAU = 1.0 / 3.0      # the Rung-1 ignited period-3 limit-cycle rate
IGNITE_PA = 2500.0            # workspace ignite pulse

# familiarity/confidence source geometry
CUE_D = 64                    # cue-code dimension for the RealAntiHebbianFamiliarity gate
N_ANCHORS = 8                 # imprinted anchor codes (the "known/familiar" span)


def _spearman(a, b):
    """Spearman rank correlation, robust to zero variance (returns 0.0 -- the collapse case)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 1e-12 else 0.0


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 1e-12 else 0.0


def _auc(scores, labels):
    """Tie-robust separation: P(score of a positive > score of a negative), ties = 0.5 (Mann-Whitney AUC). 0.5 =
    chance, 1.0 = perfect separation. Robust to the coarse quantization + ties a median-split accuracy mishandles."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


# ── build the one-brain bridge (workspace + shared inhibition + self_schema) ─────────────────────────────────
def build_self_schema_bridge(seed: int = 42, lesion_schema: bool = False,
                             attractor_weight: float = DEFAULT_ATTRACTOR_WEIGHT):
    """One `SimulationBridge`: `workspace` (K self-recurrent member assemblies + shared inhibition) + `self_schema`
    (attend[K] + confid + author sub-blocks). The attend sub-blocks read the workspace occupancy via a fixed dense
    member->attend projection (weight 0 if `lesion_schema` -> the self-lesion severs the schema's read of the
    workspace). Returns (bridge, xp, idx, snap)."""
    xp, _ = get_backend()

    n_ws = ASSEMBLY_SIZE * K_CONTENTS
    n_schema = ATTEND_SIZE * K_CONTENTS + CONFID_SIZE + AUTHOR_SIZE
    regions = [
        BrainRegion(name="workspace", n_neurons=n_ws, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="workspace_fs", n_neurons=WORKSPACE_FS_N, exc_fraction=0.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="self_schema", n_neurons=n_schema, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=False),
    ]
    pathways = [
        RegionPathway(from_region="workspace", to_region="workspace_fs", density=0.5,
                      weight_mean=WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="workspace_fs", to_region="workspace", density=0.5,
                      weight_mean=FS_TO_WS_WEIGHT, weight_jitter=0.0, plastic=False),
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
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process"):
        setattr(cfg, f, False)
    # Per-neuron HETEROGENEITY (static, deterministic given the seed) DESYNCHRONIZES the confid/author pools so
    # their POPULATION firing RATE is a smoothly-graded function of the injected current (a proper graded rate
    # code) instead of a coarse synchronous burst count. GNW Rung-2 validated that heterogeneity does not break the
    # workspace attractor's ignition. It is seeded from cfg.seed so the substrate is deterministic per seed.
    cfg.enable_parameter_heterogeneity = True
    cfg.stdp_w_max = max(400.0, float(attractor_weight) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(attractor_weight) * 4.0)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager
    ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
    ss = np.asarray(rm.indices("self_schema"), dtype=np.int64)
    member_idx = {k: ws[k * ASSEMBLY_SIZE:(k + 1) * ASSEMBLY_SIZE] for k in range(K_CONTENTS)}
    attend_idx = {k: ss[k * ATTEND_SIZE:(k + 1) * ATTEND_SIZE] for k in range(K_CONTENTS)}
    base = ATTEND_SIZE * K_CONTENTS
    confid_idx = ss[base:base + CONFID_SIZE]
    author_idx = ss[base + CONFID_SIZE:base + CONFID_SIZE + AUTHOR_SIZE]

    w_attend = 0.0 if lesion_schema else float(MEMBER_TO_ATTEND_W)
    union = dict(rm.build_wiring_plan(seed=int(seed)))
    for k in range(K_CONTENTS):
        union[f"loop_{k}"] = _build_assembly_loop_population(member_idx[k], float(attractor_weight))
        union[f"member{k}_to_attend"] = _dense_projection(member_idx[k], attend_idx[k], w_attend, WS_LOOP_GATE)

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    # settle to a true quiescent rest, snapshot (the EMERGE-61 / Rung-1 wash-out: each trial restores this).
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)

    idx = {
        "member_dev": {k: xp.asarray(v) for k, v in member_idx.items()},
        "attend_dev": {k: xp.asarray(v) for k, v in attend_idx.items()},
        "confid_dev": xp.asarray(confid_idx),
        "author_dev": xp.asarray(author_idx),
    }
    return bridge, xp, idx, snap


# ── one trial: ignite content c + tonic confidence/authorship currents -> read schema late-window rates ──────
def _run_trial(bridge, xp, idx, snap, content_k: int, conf_current: float, author_current: float,
               schema_access: bool = True):
    """Restore quiescence -> ignite content_k on the workspace (a brief pulse; it self-sustains) while holding the
    confidence current (∝ q) + authorship current tonically -> free-run -> read the late-window mean firing rate of
    every schema sub-block (attend[k], confid, author). `schema_access=False` (the self-lesion) zeroes the confid +
    author drives (the member->attend weight is already 0 in a lesion bridge) while STILL igniting the workspace, so
    the brain state is identical and only the schema's READ is severed. Returns dict of rates."""
    conf_c = conf_current if schema_access else 0.0
    author_c = author_current if schema_access else 0.0
    member_dev = idx["member_dev"][content_k]
    confid_dev = idx["confid_dev"]
    author_dev = idx["author_dev"]

    bridge.cp_external_input_current[:] = 0.0
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0

    def _set_tonic():
        # confidence + authorship are ongoing internal-state signals held through the whole trial.
        bridge.cp_external_input_current[confid_dev] = xp.float32(conf_c)
        bridge.cp_external_input_current[author_dev] = xp.float32(author_c)

    # (1) ignite the workspace member (pulse) + hold the tonic schema drives.
    for _ in range(DRIVE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[member_dev] = xp.float32(IGNITE_PA)
        _set_tonic()
        bridge._run_one_simulation_step()

    # (2) remove the ignite pulse (workspace self-sustains); keep the tonic schema drives; read the late window.
    late_start = FREE_STEPS - max(1, FREE_STEPS // 3)
    attend_acc = {k: 0 for k in range(K_CONTENTS)}
    confid_acc = 0
    author_acc = 0
    for t in range(FREE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        _set_tonic()
        bridge._run_one_simulation_step()
        if t >= late_start:
            for k in range(K_CONTENTS):
                attend_acc[k] += int(to_host(bridge.cp_firing_states[idx["attend_dev"][k]].astype(xp.float64).sum()))
            confid_acc += int(to_host(bridge.cp_firing_states[confid_dev].astype(xp.float64).sum()))
            author_acc += int(to_host(bridge.cp_firing_states[author_dev].astype(xp.float64).sum()))
    nlate = float(FREE_STEPS - late_start)
    return {
        "attend": {k: attend_acc[k] / (nlate * ATTEND_SIZE) for k in range(K_CONTENTS)},
        "confid": confid_acc / (nlate * CONFID_SIZE),
        "author": author_acc / (nlate * AUTHOR_SIZE),
    }


# ── the per-trial internal-state generator (confidence grounded in the reused familiarity gate) ─────────────
def make_trials(seed, n_trials):
    """Sample T independent internal states (content, confidence, authorship). Confidence is a GENUINE graded read
    of the reused Bogacz-Brown familiarity gate: imprint N_ANCHORS anchor codes (the "known/familiar" span); each
    trial draws a cue = mix(anchor, noise) at a random alpha (independent of content) -> novelty in [0,1] ->
    q = 1 - novelty (familiar cue -> confident; novel cue -> unsure). Content + authorship are independent draws, so
    confidence/authorship are ORTHOGONAL axes to content. Returns (contents, q_true, authorship, gate, cues,
    fam_check)."""
    rng = np.random.default_rng(seed * 101 + 7)
    gate = RealAntiHebbianFamiliarity()
    anchors = [_normalize(rng.standard_normal(CUE_D)) for _ in range(N_ANCHORS)]
    for a in anchors:
        gate.imprint(a)

    contents = rng.integers(0, K_CONTENTS, size=n_trials).astype(int)
    authorship = rng.integers(0, 2, size=n_trials).astype(int)     # 0 = heard, 1 = self
    alphas = rng.uniform(0.0, 1.0, size=n_trials)                   # familiarity mixing (independent of content)
    cues = []
    q_true = np.zeros(n_trials)
    for i in range(n_trials):
        a = anchors[int(rng.integers(N_ANCHORS))]
        cue = _normalize(alphas[i] * a + (1.0 - alphas[i]) * rng.standard_normal(CUE_D))
        cues.append(cue)
        nov = float(np.clip(gate.novelty(cue), 0.0, 1.0))
        q_true[i] = 1.0 - nov                                       # confident (familiar) -> 1; unsure (novel) -> 0
    # sanity: the familiarity gate really produces a graded confidence signal (alpha -> q monotone).
    fam_check = {"corr_alpha_q": _spearman(alphas, q_true),
                 "q_min": float(q_true.min()), "q_max": float(q_true.max()), "q_mean": float(q_true.mean())}
    return contents, q_true, authorship, gate, cues, fam_check


def _conf_current(q, conf_min_pa, conf_max_pa):
    return float(conf_min_pa) + float(np.clip(q, 0.0, 1.0)) * (float(conf_max_pa) - float(conf_min_pa))


# ── evaluate one seed (intact + all anti-cheats) ─────────────────────────────────────────────────────────────
def evaluate_seed(seed, n_trials, conf_min_pa, conf_max_pa, author_pa, thresholds, verbose=False):
    contents, q_true, authorship, gate, cues, fam_check = make_trials(seed, n_trials)

    # ---- INTACT: the schema reads the real internal state ----
    bridge, xp, idx, snap = build_self_schema_bridge(seed=seed, lesion_schema=False)

    def run_block(bridge, xp, idx, snap, drive_contents, drive_q, drive_author, schema_access=True):
        """Run T trials driving the schema with (drive_contents[i], drive_q[i], drive_author[i]); return decoded
        arrays (attention^, confid_rate, author_rate)."""
        dec_content = np.zeros(len(drive_contents), dtype=int)
        confid_rate = np.zeros(len(drive_contents))
        author_rate = np.zeros(len(drive_contents))
        for i in range(len(drive_contents)):
            r = _run_trial(bridge, xp, idx, snap, int(drive_contents[i]),
                           _conf_current(drive_q[i], conf_min_pa, conf_max_pa),
                           float(author_pa) * float(drive_author[i]), schema_access=schema_access)
            att = r["attend"]
            dec_content[i] = int(max(att, key=att.get))
            confid_rate[i] = r["confid"]
            author_rate[i] = r["author"]
        return dec_content, confid_rate, author_rate

    dec_c, conf_rate, auth_rate = run_block(bridge, xp, idx, snap, contents, q_true, authorship)

    # attention: argmax over attend sub-blocks == true content?
    attention_acc = float(np.mean(dec_c == contents))
    # confidence: monotone in true q (Spearman) + a tie-robust hi/lo separation (AUC of conf_rate vs high-q label).
    confidence_spearman = _spearman(q_true, conf_rate)
    q_hi = q_true >= np.median(q_true)
    confidence_auc = _auc(conf_rate, q_hi)
    # authorship: self vs heard from the author-block rate (midpoint threshold between the class means).
    self_mask = authorship == 1
    heard_mask = authorship == 0
    if self_mask.any() and heard_mask.any():
        auth_thr = 0.5 * (auth_rate[self_mask].mean() + auth_rate[heard_mask].mean())
    else:
        auth_thr = float(np.median(auth_rate))
    auth_pred = (auth_rate >= auth_thr).astype(int)
    authorship_acc = float(np.mean(auth_pred == authorship))

    # ---- ANTI-CHEAT (2) STATE-NOT-CONTENT: confidence/authorship are a SEPARATE axis from content ----
    corr_conf_content = _pearson(conf_rate, contents.astype(float))
    corr_auth_content = _pearson(auth_rate, contents.astype(float))
    corr_true_q_content = _pearson(q_true, contents.astype(float))   # ~0 by construction (independent draws)
    # confidence tracking holds uniformly across content identities (report how-sure regardless of what).
    per_content_spearman = {}
    for k in range(K_CONTENTS):
        m = contents == k
        per_content_spearman[k] = _spearman(q_true[m], conf_rate[m]) if m.sum() >= 3 else None
    valid_pcs = [v for v in per_content_spearman.values() if v is not None]
    dissociation_ok = bool(abs(corr_conf_content) <= thresholds["max_axis_corr"]
                           and abs(corr_auth_content) <= thresholds["max_axis_corr"])

    # ---- ANTI-CHEAT (1) SELF-LESION: sever the schema's access -> collapse ----
    bridge_l, xp_l, idx_l, snap_l = build_self_schema_bridge(seed=seed, lesion_schema=True)
    dec_c_l, conf_rate_l, auth_rate_l = run_block(bridge_l, xp_l, idx_l, snap_l, contents, q_true, authorship,
                                                  schema_access=False)
    les_attention_acc = float(np.mean(dec_c_l == contents))
    les_conf_spearman = _spearman(q_true, conf_rate_l)
    les_auth_pred = (auth_rate_l >= (auth_thr if np.isfinite(auth_thr) else 0.0)).astype(int)
    les_authorship_acc = float(np.mean(les_auth_pred == authorship))
    lesion_collapsed = bool(les_attention_acc <= thresholds["chance_attention"]
                            and abs(les_conf_spearman) <= thresholds["collapse_spearman"]
                            and les_authorship_acc <= thresholds["chance_authorship"])

    # ---- ANTI-CHEAT (3) SHUFFLED-INTERNAL-SIGNAL: drive the (intact) schema with a permuted state; score vs TRUE ----
    rng = np.random.default_rng(seed * 777 + 13)
    pc = rng.permutation(n_trials); pq = rng.permutation(n_trials); pa = rng.permutation(n_trials)
    dec_c_sh, conf_rate_sh, auth_rate_sh = run_block(
        bridge, xp, idx, snap, contents[pc], q_true[pq], authorship[pa], schema_access=True)
    sh_attention_acc = float(np.mean(dec_c_sh == contents))            # decoded (from c') vs TRUE c -> chance
    sh_conf_spearman = _spearman(q_true, conf_rate_sh)                 # rate (from q') vs TRUE q -> ~0
    sh_auth_pred = (auth_rate_sh >= auth_thr).astype(int)
    sh_authorship_acc = float(np.mean(sh_auth_pred == authorship))
    shuffle_collapsed = bool(sh_attention_acc <= thresholds["chance_attention"]
                             and abs(sh_conf_spearman) <= thresholds["collapse_spearman"]
                             and sh_authorship_acc <= thresholds["chance_authorship"])

    # ---- EXTRA: FAMILIARITY-GATE LESION flattens the confidence scalar (grounds the source in a real signal) ----
    gate.lesion()
    q_flat = np.array([1.0 - float(np.clip(gate.novelty(c), 0.0, 1.0)) for c in cues])
    fam_lesion_flat = bool(np.std(q_flat) <= 0.05)

    # ---- GO (per-seed) ----
    go_attention = bool(attention_acc >= thresholds["attention_acc"])
    go_confidence = bool(confidence_spearman >= thresholds["confidence_spearman"]
                         and confidence_auc >= thresholds["confidence_auc"])
    go_authorship = bool(authorship_acc >= thresholds["authorship_acc"])
    go = bool(go_attention and go_confidence and go_authorship
              and lesion_collapsed and shuffle_collapsed and dissociation_ok)

    r = {
        "seed": int(seed), "n_trials": int(n_trials),
        "familiarity_source": fam_check,
        "intact": {
            "attention_acc": attention_acc,
            "confidence_spearman": confidence_spearman,
            "confidence_auc": confidence_auc,
            "authorship_acc": authorship_acc,
            "conf_rate_range": [float(conf_rate.min()), float(conf_rate.max())],
            "auth_rate_self_mean": float(auth_rate[self_mask].mean()) if self_mask.any() else None,
            "auth_rate_heard_mean": float(auth_rate[heard_mask].mean()) if heard_mask.any() else None,
        },
        "dissociation": {
            "corr_decoded_confidence_vs_content": corr_conf_content,
            "corr_decoded_authorship_vs_content": corr_auth_content,
            "corr_true_confidence_vs_content": corr_true_q_content,
            "per_content_confidence_spearman": {str(k): v for k, v in per_content_spearman.items()},
            "min_per_content_spearman": (float(min(valid_pcs)) if valid_pcs else None),
            "dissociation_ok": dissociation_ok,
        },
        "self_lesion": {
            "attention_acc": les_attention_acc, "confidence_spearman": les_conf_spearman,
            "authorship_acc": les_authorship_acc, "collapsed": lesion_collapsed,
        },
        "shuffle_internal": {
            "attention_acc": sh_attention_acc, "confidence_spearman": sh_conf_spearman,
            "authorship_acc": sh_authorship_acc, "collapsed": shuffle_collapsed,
        },
        "familiarity_lesion_flattens_confidence": fam_lesion_flat,
        "go_components": {"attention": go_attention, "confidence": go_confidence, "authorship": go_authorship,
                          "self_lesion_collapses": lesion_collapsed, "shuffle_collapses": shuffle_collapsed,
                          "dissociation": dissociation_ok},
        "go": go,
    }
    if verbose:
        _print_seed(r)
    return r


def _print_seed(r):
    it = r["intact"]; ll = r["self_lesion"]; sh = r["shuffle_internal"]; ds = r["dissociation"]
    print(f"  [seed {r['seed']}] familiarity source: corr(alpha,q)={r['familiarity_source']['corr_alpha_q']:+.2f} "
          f"q in [{r['familiarity_source']['q_min']:.2f},{r['familiarity_source']['q_max']:.2f}]", flush=True)
    print(f"    INTACT   attention_acc={it['attention_acc']:.3f} (chance .25)  "
          f"confidence_spearman={it['confidence_spearman']:+.3f} auc={it['confidence_auc']:.3f}  "
          f"authorship_acc={it['authorship_acc']:.3f} (chance .5)", flush=True)
    print(f"             confid_rate self/heard author mean = "
          f"{it['auth_rate_self_mean']}/{it['auth_rate_heard_mean']}", flush=True)
    print(f"    DISSOC   corr(conf,content)={ds['corr_decoded_confidence_vs_content']:+.3f} "
          f"corr(author,content)={ds['corr_decoded_authorship_vs_content']:+.3f} "
          f"min_per_content_conf_spearman={ds['min_per_content_spearman']}  ok={ds['dissociation_ok']}", flush=True)
    print(f"    LESION   attention={ll['attention_acc']:.3f} conf_sp={ll['confidence_spearman']:+.3f} "
          f"author={ll['authorship_acc']:.3f}  collapsed={ll['collapsed']}", flush=True)
    print(f"    SHUFFLE  attention={sh['attention_acc']:.3f} conf_sp={sh['confidence_spearman']:+.3f} "
          f"author={sh['authorship_acc']:.3f}  collapsed={sh['collapsed']}", flush=True)
    print(f"    fam-lesion flattens confidence = {r['familiarity_lesion_flattens_confidence']}", flush=True)
    print(f"    >>> seed GO = {r['go']}  {r['go_components']}", flush=True)


DEFAULT_THRESHOLDS = {
    "attention_acc": 0.85, "confidence_spearman": 0.60, "confidence_auc": 0.85, "authorship_acc": 0.90,
    "max_axis_corr": 0.30,          # |corr(decoded confidence/authorship, content)| must be <= this (dissociation)
    "chance_attention": 0.45,       # lesion/shuffle attention must drop to ~chance (1/K = 0.25) + margin
    "chance_authorship": 0.65,      # lesion/shuffle authorship must drop to ~chance (0.5) + margin
    "collapse_spearman": 0.30,      # lesion/shuffle confidence spearman must collapse to ~0
}


def main():
    ap = argparse.ArgumentParser(description="DR-3 attention/agency SELF-SCHEMA region de-risk.")
    ap.add_argument("--seed", type=int, default=42, help="single seed (used by --smoke)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="multi-seed list (overrides --seed)")
    ap.add_argument("--n-trials", type=int, default=96, help="internal-state trials per block")
    ap.add_argument("--smoke", action="store_true", help="tiny 1-seed smoke (fewer trials)")
    ap.add_argument("--conf-min-pa", type=float, default=150.0, help="confidence current at q=0")
    ap.add_argument("--conf-max-pa", type=float, default=750.0, help="confidence current at q=1")
    ap.add_argument("--author-pa", type=float, default=650.0, help="authorship (self) drive current")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--json", type=str, default="research/findings/raw/_self_schema_smoke.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    if args.smoke:
        seeds = [args.seed]
        n_trials = min(args.n_trials, 40)
    else:
        seeds = args.seeds if args.seeds is not None else [args.seed]
        n_trials = args.n_trials

    print(f"[self-schema] DR-3 attention/agency SELF-SCHEMA region | seeds={seeds} n_trials={n_trials} "
          f"backend={args.backend} K_contents={K_CONTENTS} conf_pa[{args.conf_min_pa},{args.conf_max_pa}] "
          f"author_pa={args.author_pa}", flush=True)
    print(f"[self-schema] regions: workspace({ASSEMBLY_SIZE}x{K_CONTENTS} assemblies + NMDA attractors) + shared "
          f"inhibition + self_schema(attend[{K_CONTENTS}x{ATTEND_SIZE}] + confid[{CONFID_SIZE}] + author[{AUTHOR_SIZE}])",
          flush=True)
    print("[self-schema] HONEST: a FUNCTIONAL self-model correlate (represents+reports 3 internal-state axes read "
          "from the brain's OWN signals) -- NOT a claim of subjective experience.", flush=True)

    t0 = time.time()
    per_seed = []
    for s in seeds:
        per_seed.append(evaluate_seed(s, n_trials, args.conf_min_pa, args.conf_max_pa, args.author_pa,
                                      DEFAULT_THRESHOLDS, verbose=True))

    n_go = sum(1 for r in per_seed if r["go"])
    all_go = bool(n_go == len(per_seed))
    verdict = "GO" if all_go else ("PARTIAL" if n_go > 0 else "NEGATIVE")

    # aggregate means (reporting)
    def _mean(key_path):
        vals = []
        for r in per_seed:
            v = r
            for k in key_path:
                v = v[k]
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    agg = {
        "mean_attention_acc": _mean(["intact", "attention_acc"]),
        "mean_confidence_spearman": _mean(["intact", "confidence_spearman"]),
        "mean_confidence_auc": _mean(["intact", "confidence_auc"]),
        "mean_authorship_acc": _mean(["intact", "authorship_acc"]),
        "all_self_lesion_collapse": all(r["self_lesion"]["collapsed"] for r in per_seed),
        "all_shuffle_collapse": all(r["shuffle_internal"]["collapsed"] for r in per_seed),
        "all_dissociation_ok": all(r["dissociation"]["dissociation_ok"] for r in per_seed),
        "all_fam_lesion_flattens": all(r["familiarity_lesion_flattens_confidence"] for r in per_seed),
    }

    out = {
        "runner": "_self_schema_region_derisk",
        "faculty": "F4 self-model/metacognition (DR-3 attention/agency SELF-SCHEMA region)",
        "theory": "Graziano Attention Schema Theory + higher-order metacognition (FUNCTIONAL correlate only)",
        "seeds": seeds, "n_trials": n_trials, "backend": args.backend,
        "thresholds": DEFAULT_THRESHOLDS,
        "verdict": verdict, "n_go": n_go, "n_seeds": len(seeds),
        "aggregate": agg,
        "per_seed": per_seed,
        "honest_scope": ("A functional self-model correlate: a self_schema region reads the brain's OWN internal "
                         "signals (GNW workspace occupancy + a familiarity-gate confidence scalar + an authorship "
                         "tag) and jointly represents+reports them; the read-outs track ground-truth internal state "
                         "and collapse under self-lesion / shuffled-signal. NOT a claim of subjective experience "
                         "(phenomenal consciousness is OPEN, arguably untestable)."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[self-schema] === VERDICT: {verdict} ({n_go}/{len(seeds)} seeds GO) ===", flush=True)
    print(f"[self-schema]   mean attention_acc={agg['mean_attention_acc']:.3f} (chance .25) | "
          f"mean confidence_spearman={agg['mean_confidence_spearman']:+.3f} auc={agg['mean_confidence_auc']:.3f} | "
          f"mean authorship_acc={agg['mean_authorship_acc']:.3f} (chance .5)", flush=True)
    print(f"[self-schema]   anti-cheats: self-lesion collapses={agg['all_self_lesion_collapse']} | "
          f"shuffle collapses={agg['all_shuffle_collapse']} | dissociation(state!=content)={agg['all_dissociation_ok']} | "
          f"fam-lesion flattens confidence={agg['all_fam_lesion_flattens']}", flush=True)
    print(f"[self-schema]   elapsed={time.time()-t0:.1f}s  wrote {args.json}", flush=True)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
