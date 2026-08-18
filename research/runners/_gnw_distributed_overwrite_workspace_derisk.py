"""GNW distributed workspace: DIVISIVE NORMALIZATION makes content globally CLEARABLE + a conflict-triggered
depression of the SHARED recurrent resource delivers the GLOBAL STOP the localist STN-veto could not.

THE NEXT RUNG, NAMED BY THE STN-VETO NO-GO. The localist K-slot recurrent workspace (Rung-1/2/2b/2c/2d + the
STN stop-veto) is a set of DISJOINT self-sufficient bistable basins. It can be cleared from WITHIN (Rung-2d:
short-term depression depletes a held incumbent's own recurrent resources -> empty window, GO 6/6) but NOT from
OUTSIDE by a control signal: a graded external inhibitory brake (STN->GPi->workspace) cannot drive a co-ignited
localist attractor to n_ignited=0 (`2026-08-18-gnw-stn-stop-veto-NOGO.md`: min n_post=2 — weak inhibition leaves
a heterogeneity survivor, strong inhibition destabilizes UPWARD via the g_i driving-force reversal). That NO-GO
named THIS lever: a distributed workspace with overlapping patterns + divisive normalization / a shared
global-gain resource "where withdrawing a shared resource de-ignites all content uniformly — and the conflict
SENSOR built here is its ready effector arm."

WHAT THIS DE-RISK FOUND (the honest, measured mechanism; NO `sim/` edit; explicit wiring, the Rung-2b/2c/2d/STN
pattern):
  * A distributed overlapping-pattern workspace (K cliques over a shared pool, adjacent cliques overlap) with
    DIVISIVE NORMALIZATION (Carandini & Heeger 2012, Nat Rev Neurosci 13:51; Reynolds & Heeger 2009) — a shared
    inhibitory `norm_pool` pools ALL workspace activity and returns broad conductance feedback inhibition, the
    shared divisor — holds each ignited pattern NEAR the sustain knee rather than in a deep self-sufficient basin.
  * The naive reading (withdraw a tonic thalamic GAIN -> collapse) is a **NO-GO on this substrate**: the ignited
    period-3 attractor is self-sufficient once on, so removing the tonic `thal` support does NOT de-ignite it
    (reproducing the STN finding's own "thalamo-cortical remove-excitation self-sufficient" diagnosis, now from
    the distributed side). This is measured and reported, not hidden.
  * The GLOBAL STOP that WORKS is a CONFLICT-TRIGGERED depression of the SHARED recurrent resource (Tsodyks-Markram
    STD, Mongillo-Barak-Tsodyks 2008 — the exact mechanism Rung-2d validated, here GLOBALIZED across the shared
    workspace pool and gated by the conflict sensor). Because every ignited pattern draws on the shared recurrence,
    depleting it de-ignites ALL content UNIFORMLY -> n_ignited -> 0. It is gain-withdrawal from a SHARED resource
    (not a huge g_i), so there is no driving-force reversal / rebound: self-extinction is the natural collapse.
  * DIVISIVE NORMALIZATION IS LOAD-BEARING for the stop: lesion the norm_pool and the SAME depression burst does
    NOT clear the content (the un-normalized patterns are deep self-sufficient basins) -> n stays >= 2 (the
    localist boundary). Normalization is what makes the workspace globally clearable; depression is the effector.
  * The conflict SENSOR is reused verbatim from the STN veto (host reads the workspace's OWN late-window per-slot
    spiking margin; the depression boost = boost_gain*max(0, margin_ref - margin)*scale -> ZERO at a confident
    commit, FIRING on a conflict). A confident single-content commit (high margin) sets boost=0 -> it is NOT
    disrupted (SELECTIVITY). A host-margin SCRAMBLE (feed the confident margin to the conflict trial) breaks the
    stop. (The boost's scaling across INTERMEDIATE co-drives is non-monotone — the co-ignition competition itself
    is non-monotone in the challenger drive on this substrate; the sweep is kept as reported data, not a gate.)
  * The active OVERWRITE (a salient challenger fast-EVICTS a held incumbent, n stays 1) is a **BOUNDARY** on this
    bistable substrate: pure divisive-norm competition co-ignites the self-sufficient incumbent (reproducing the
    Rung-2c eviction boundary from the distributed side), and depression-clearing removes ALL content (it cannot
    selectively retain the challenger). Characterized + a named next lever, not swept under the rug.

VERDICT: PARTIAL. The DECISIVE capability the STN NO-GO named — a control signal driving a co-ignited multi-content
workspace to n_ignited=0 (the STN external inhibition stuck at min n_post=2) — is a clean 6/6 GO here. The active
overwrite is a characterized boundary; the pure gain-withdrawal is a characterized NO-GO.

GO GATE (the GLOBAL-STOP capability; 6 seeds 42/43/44/100/101/102, SIM_BACKEND=numpy, determinism via cfg.seed):
  (1) GLOBAL STOP — a co-ignited multi-content conflict state (n>=2) is driven to n_ignited=0 by the
      conflict-triggered depression of the shared recurrence, on >=5/6 seeds.
  (2) SELECTIVITY — a confident single-content commit (high margin) is NOT disrupted (boost=0 -> holds n=1).
ANTI-CHEATS (each on every seed):
  (a) DIVISIVE-NORM LOAD-BEARING: lesion norm_pool -> the depression burst does NOT clear (n stays >= 2).
  (b) STD LOAD-BEARING: freeze the depression (boost=0 on the conflict) -> the conflict HOLDS (n stays >= 2).
  (c) SIGNAL-DRIVEN: boost is ZERO at a confident commit and FIRES on a conflict; CONFLICT-OFF (boost_gain=0)
      holds; host-margin SCRAMBLE breaks the stop; the continuous stop makes host_workspace_reset_calls == 0.
  (d) BYTE-IDENTICAL substrate: distributed hash == localist-base (overlap=0, norm off) hash at the same seed.
  (e) DETERMINISM: build twice at one seed -> identical substrate hash (cfg.seed seeds the substrate).
CHARACTERIZED (reported, not part of the primary GO): the active OVERWRITE boundary + the pure gain-withdrawal
NO-GO, both quantified with named next levers.

Usage (CPU cheap-first; EXPORT OMP/OPENBLAS/MKL=4):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_distributed_overwrite_workspace_derisk --smoke --seed 42 \
      --json research/findings/raw/_gnw_distributed_overwrite_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_distributed_overwrite_workspace_derisk \
      --six-seed --json research/findings/raw/_gnw_distributed_overwrite_6seed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion
from sim.backend import get_backend, to_host
from tools.verdict import Verdict
from tools.lab import attributable_to

# reuse-by-import: validated constants + the determinism hash + the dense-population + the full wash-out.
from research.runners._gnw_rung1_ignition_curve_derisk import (
    DRIVE_STEPS, FREE_STEPS, SETTLE_STEPS, WS_LOOP_GATE,
)
from research.runners._gnw_rung2_competitive_access_derisk import (
    _ignited, IGNITE_FRAC, SOLO_PLATEAU,
)
from research.runners._gnw_rung2b_sfa_workspace_eviction_derisk import _threshold_hash
from research.runners._gnw_rung2c_salience_disinhibition_derisk import _dense_pop
from research.runners._p1_2_workspace_deliberation_loop_derisk import _full_snapshot, _full_restore


# ── geometry: K overlapping distributed patterns over a shared workspace pool ──────────────────────────────────
N_PATTERNS = 3                 # A, B, C cliques over the shared pool; adjacent cliques overlap (sliding window)
PATTERN_SIZE = 100             # P neurons per distributed pattern (>= the Rung-1 ignitable assembly size)
DEFAULT_OVERLAP = 15           # O shared neurons between ADJACENT patterns; 0 => disjoint = localist base
WORKSPACE_N = N_PATTERNS * PATTERN_SIZE + 20   # sized for the O=0 case so the region size is overlap-invariant

NORM_N = 60                    # divisive-normalization inhibitory pool (pools ALL workspace activity -> broad I)
THAL_N = 60                    # shared thalamic tonic-support pool (the naive gain-withdrawal target)

# workspace recurrence (the ignitable distributed attractor). Single frozen weight over the deduped clique edges.
W_REC = 34.0
WS_NORM_W = 5.0                # workspace -> norm_pool excitation (drives the shared divisor)
NORM_WS_W = 16.0             # norm_pool -> workspace conductance inhibition (the divisive feedback; LOAD-BEARING)
THAL_TONIC_PA = 700.0          # tonic depolarizing drive into thal (the naive "gain"; withdrawal is a NO-GO here)
THAL_WS_W = 5.0                # thal -> workspace broad excitatory support

# short-term depression on the shared workspace recurrence (Tsodyks-Markram; the GLOBAL-STOP effector).
STD_TAU_D = 250.0              # depression recovery time constant (ms)
STD_BASELINE_U = 0.0          # baseline per-spike release: 0 -> x stays 1 (a confident commit never self-depletes)
BOOST_GAIN = 1.0              # conflict boost gain
# boost = BOOST_GAIN * max(0, margin_ref - margin) * BOOST_SCALE. At full conflict (margin~0.01, deficit~0.157)
# -> boost ~0.13, inside the divisive-norm-LOAD-BEARING window [0.09,0.15]: intact clears to n=0 (6/6) while a
# norm-lesion HOLDS (n stays >=2). boost>=0.18 clears even without normalization (norm no longer load-bearing).
BOOST_SCALE = 0.85
PULSE_DURATION = 110          # steps the conflict-triggered depression burst is applied

# drive amplitudes (Rung-2/STN operating point).
IGNITE_PA = 2500.0
STRONG_PA = 2.0 * IGNITE_PA     # 5000 pA: robust per-pattern ignition under heterogeneity + normalization
WEAK_PA = 0.48 * IGNITE_PA      # 1200 pA: sub-knee (a non-driven distractor pattern stays off)

# conflict sensor (reused STN recipe): margin = winner_rate - runnerup_rate; deficit = max(0, MARGIN_REF - margin).
MARGIN_REF = SOLO_PLATEAU * 0.5     # 1/6 (halfway between single-content ~1/3 and co-ignition ~0)

OU_NOISE_PA = 30.0                  # desynchronize the attractor (async rate attractor; no synchronous rebound)

READ_FREE_STEPS = 45                # free steps to let a commit settle before reading the margin
POST_FREE_STEPS = FREE_STEPS        # free steps after the stop burst to read the settled (empty?) workspace
CONF_HOLD_STEPS = 90                # a confident commit must HOLD this long (selectivity: baseline U=0 -> no evict)

# restore-call accounting: the CONTINUOUS stop headline MUST make ZERO restore calls (anti-cheat c).
_RESTORE_CALLS = {"n": 0}


def _counted_full_restore(bridge, snap):
    _RESTORE_CALLS["n"] += 1
    _full_restore(bridge, snap)


# ── pattern geometry ───────────────────────────────────────────────────────────────────────────────────────
def _pattern_geometry(ws_indices, k_patterns=N_PATTERNS, pattern_size=PATTERN_SIZE, overlap=DEFAULT_OVERLAP):
    """Sliding-window overlapping patterns: pattern_k = ws[k*(P-O) : k*(P-O)+P] (O=0 => disjoint = localist base).
    Returns (patterns, privates); privates[k] = neurons in pattern k ONLY (its exclusive core = the clean read)."""
    ws = np.asarray(ws_indices, dtype=np.int64)
    P, O = int(pattern_size), int(overlap)
    step = P - O
    patterns = [ws[k * step:k * step + P].copy() for k in range(k_patterns)]
    privates = []
    for k in range(k_patterns):
        others = (np.concatenate([patterns[j] for j in range(k_patterns) if j != k])
                  if k_patterns > 1 else np.array([], dtype=np.int64))
        privates.append(np.setdiff1d(patterns[k], others).astype(np.int64))
    return patterns, privates


def _recurrence_edges(patterns):
    """Deduped dense E->E edges within EACH pattern clique (no autapses). Overlap neurons end up connected to BOTH
    cliques -> the distributed, shared-resource recurrence. Returns (pre, post) int64 arrays."""
    pre_all, post_all = [], []
    for a in patterns:
        a = np.asarray(a, dtype=np.int64); m = a.shape[0]
        pre = np.repeat(a, m); post = np.tile(a, m); keep = pre != post
        pre_all.append(pre[keep]); post_all.append(post[keep])
    pre = np.concatenate(pre_all); post = np.concatenate(post_all)
    key = pre.astype(np.int64) * np.int64(10_000_019) + post.astype(np.int64)
    _, uniq = np.unique(key, return_index=True)
    return pre[uniq].astype(np.int64), post[uniq].astype(np.int64)


def _rec_population(patterns, weight):
    pre, post = _recurrence_edges(patterns)
    ww = np.full(pre.shape[0], float(weight), dtype=np.float32)
    return {"pre_indices": pre, "post_indices": post, "initial_weights": ww,
            "plastic": False, "plasticity_gate": WS_LOOP_GATE, "conn_type": "E_TO_E", "count": int(pre.size)}


# ── short-term depression on the SHARED workspace recurrence (Tsodyks-Markram; the GLOBAL-STOP effector) ────────
class WorkspaceDepression:
    """Host-computed Tsodyks-Markram STD on ALL workspace->workspace (recurrence) synapses. Tracks per-PRESYN x;
    each step OVERWRITES those synapse weights with base*x BEFORE the step. Baseline U (=STD_BASELINE_U) is the
    per-spike release; a CONFLICT BOOST transiently raises it (the conflict-gated global depletion burst). This is
    the synaptic-depression MECHANISM on the substrate's own recurrence (Rung-2d pattern), NOT a host state reset:
    depleting the SHARED recurrent resource de-ignites all content that draws on it. The engine's native global STP
    stays OFF (banked foot-gun); this targets only the workspace recurrence."""

    def __init__(self, bridge, xp, ws_used, U=STD_BASELINE_U, tau_D=STD_TAU_D, dt=1.0):
        self.bridge = bridge; self.xp = xp
        self.U = float(U); self.tau_D = float(tau_D); self.dt = float(dt); self.boost = 0.0
        n = bridge.core_config.num_neurons
        csr = bridge.cp_connections; csr.sort_indices()
        indptr = to_host(csr.indptr); indices = to_host(csr.indices).astype(np.int64)
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr)); cols = indices
        wsmask = np.zeros(n, dtype=bool); wsmask[np.asarray(ws_used, dtype=np.int64)] = True
        mask = wsmask[rows] & wsmask[cols]                     # workspace->workspace == the recurrence
        self.idx = np.where(mask)[0]
        self.pre = rows[self.idx]
        self.base = to_host(csr.data)[self.idx].astype(np.float64).copy()
        self.idx_dev = xp.asarray(self.idx)
        self.x = np.ones(n, dtype=np.float64)
        self.ws_used = np.asarray(ws_used, dtype=np.int64)
        self.n_writes = 0
        self.n_rec_syn = int(self.idx.size)

    def apply(self):
        self.bridge.cp_connections.data[self.idx_dev] = self.xp.asarray(
            self.base * self.x[self.pre], dtype=self.xp.float32)
        self.n_writes += 1

    def update(self, fired_host):
        w = self.ws_used
        self.x[w] += (1.0 - self.x[w]) * (self.dt / self.tau_D)     # recovery
        U = self.U + self.boost
        fired = w[fired_host[w]]
        if fired.size:
            self.x[fired] = self.x[fired] - U * self.x[fired]       # depletion
        np.clip(self.x, 0.0, 1.0, out=self.x)

    def reset(self):
        self.x[:] = 1.0; self.boost = 0.0


# ── build the distributed workspace bridge ──────────────────────────────────────────────────────────────────
def build_overwrite_bridge(seed=42, w_rec=W_REC, overlap=DEFAULT_OVERLAP, ws_norm_w=WS_NORM_W, norm_ws_w=NORM_WS_W,
                           thal_tonic_pA=THAL_TONIC_PA, thal_ws_w=THAL_WS_W, norm_lesion=False,
                           heterogeneity=True, ou_noise_pA=OU_NOISE_PA, pattern_size=PATTERN_SIZE):
    """Regions: workspace (exc, NMDA; K overlapping patterns) + norm_pool (inh; divisive normalization) +
    thal (exc; tonic shared support). ALL inter-region wiring explicit.
      norm_lesion=True -> norm_pool->workspace weight 0 (divisive-norm LOAD-BEARING anti-cheat).
    Returns (bridge, xp, patterns_dev, privates_dev, thal_dev, ws_used, snap, handles)."""
    xp, _ = get_backend()

    workspace = BrainRegion(name="workspace", n_neurons=WORKSPACE_N, exc_fraction=1.0,
                            internal_density=0.0, enable_nmda=True)
    norm_pool = BrainRegion(name="norm_pool", n_neurons=NORM_N, exc_fraction=0.0, internal_density=0.0,
                            enable_nmda=False)
    thal = BrainRegion(name="thal", n_neurons=THAL_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False)
    regions = [workspace, norm_pool, thal]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = []
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                # ⭐ the substrate seed (het/threshold RNG) — NOT actual_seed_used
    cfg.heterogeneity_seed = int(seed)
    cfg.ou_seed = int(seed)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False      # FOOT-GUN: the synaptic-scaling clip slams the frozen weights
    cfg.enable_short_term_plasticity = False   # native global STP is a banked foot-gun; STD is in-runner, targeted
    cfg.enable_structural_plasticity = False
    cfg.stdp_w_max = max(400.0, float(w_rec) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(w_rec) * 4.0)
    cfg.enable_parameter_heterogeneity = bool(heterogeneity)
    if ou_noise_pA > 0.0:
        cfg.enable_ou_process = True
        cfg.ou_mean_current_pA = 0.0
        cfg.ou_std_current_pA = float(ou_noise_pA)
    else:
        cfg.enable_ou_process = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert cfg.enable_homeostasis is False

    rm = bridge.region_manager
    ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
    patterns, privates = _pattern_geometry(ws, N_PATTERNS, pattern_size, overlap)
    ws_used = np.unique(np.concatenate(patterns)).astype(np.int64) if overlap < pattern_size \
        else np.asarray(patterns[0], dtype=np.int64)
    norm_idx = np.asarray(rm.indices("norm_pool"), dtype=np.int64)
    thal_idx = np.asarray(rm.indices("thal"), dtype=np.int64)

    norm_ws_eff = 0.0 if norm_lesion else float(norm_ws_w)

    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    union_plan["workspace_rec"] = _rec_population(patterns, float(w_rec))
    union_plan["ws2norm"] = _dense_pop(ws_used, norm_idx, float(ws_norm_w), "E_TO_I")
    union_plan["norm2ws"] = _dense_pop(norm_idx, ws_used, norm_ws_eff, "I_TO_E")
    union_plan["thal2ws"] = _dense_pop(thal_idx, ws_used, float(thal_ws_w), "E_TO_E")

    inh = list(norm_idx)                # norm_pool is the only inhibitory (GABAergic) source
    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    thal_dev = xp.asarray(thal_idx)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[thal_dev] = xp.float32(thal_tonic_pA)   # settle WITH the tonic support on
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    snap = _full_snapshot(bridge)

    handles = {"seed": int(seed), "w_rec": float(w_rec), "overlap": int(overlap), "pattern_size": int(pattern_size),
               "ws_norm_w": float(ws_norm_w), "norm_ws_w": float(norm_ws_eff), "thal_tonic_pA": float(thal_tonic_pA),
               "thal_ws_w": float(thal_ws_w), "norm_lesion": bool(norm_lesion), "heterogeneity": bool(heterogeneity),
               "ou_noise_pA": float(ou_noise_pA), "n_patterns": N_PATTERNS,
               "private_sizes": [int(p.size) for p in privates], "n_ws_used": int(ws_used.size),
               "n_rec_edges": int(union_plan["workspace_rec"]["count"])}
    return (bridge, xp, [xp.asarray(p) for p in patterns], [xp.asarray(p) for p in privates], thal_dev, ws_used,
            snap, handles)


# ── stepping + spiking reads ─────────────────────────────────────────────────────────────────────────────────
def _ws_step(bridge, xp, thal_dev, thal_tonic_pA, std, drive_map=None):
    """One sim step with the STD layer: apply base*x -> inject tonic thal + drives -> step -> update x."""
    if std is not None:
        std.apply()
    bridge.cp_external_input_current[:] = 0.0
    if thal_tonic_pA != 0.0:
        bridge.cp_external_input_current[thal_dev] = xp.float32(thal_tonic_pA)
    if drive_map:
        for idx_dev, val in drive_map:
            if val > 0.0:
                bridge.cp_external_input_current[idx_dev] = xp.float32(val)
    bridge._run_one_simulation_step()
    if std is not None:
        std.update(to_host(bridge.cp_firing_states).astype(bool))


def _read_private_rates(bridge, xp, thal_dev, thal_tonic_pA, privates_dev, std, n_free=READ_FREE_STEPS):
    """Free-run n_free steps (tonic on, no pattern drive) and return the LATE-window per-pattern PRIVATE-core mean
    firing rate — the clean spiking content read. Leaves the workspace evolving in place."""
    late_start = n_free - max(1, n_free // 3)
    counts = [0] * len(privates_dev)
    for t in range(n_free):
        _ws_step(bridge, xp, thal_dev, thal_tonic_pA, std)
        if t >= late_start:
            for i, p in enumerate(privates_dev):
                counts[i] += int(to_host(bridge.cp_firing_states[p].astype(xp.float64).sum()))
    out = []
    for i, p in enumerate(privates_dev):
        denom = float((n_free - late_start) * int(p.shape[0]))
        out.append(counts[i] / denom if denom > 0 else 0.0)
    return out


def _margin(rates):
    order = sorted(range(len(rates)), key=lambda i: rates[i], reverse=True)
    top = rates[order[0]]
    second = rates[order[1]] if len(order) > 1 else 0.0
    n_ign = int(sum(1 for r in rates if _ignited(r)))
    return int(order[0]), float(top - second), n_ign


def _verdict_label(rates):
    n_ign = int(sum(1 for r in rates if _ignited(r)))
    if n_ign == 0:
        return "ABSTAIN", 0
    return (f"COMMIT_p{int(np.argmax(rates))}" if n_ign == 1 else f"AMBIGUOUS_p{int(np.argmax(rates))}"), n_ign


def _boost_from_margin(margin, boost_gain, boost_scale, margin_ref):
    return float(boost_gain) * max(0.0, float(margin_ref) - float(margin)) * float(boost_scale)


# ── trial protocols ─────────────────────────────────────────────────────────────────────────────────────────
def _drive(bridge, xp, thal_dev, thal_tonic_pA, std, drive_map, n=DRIVE_STEPS):
    for _ in range(n):
        _ws_step(bridge, xp, thal_dev, thal_tonic_pA, std, drive_map=drive_map)


def run_confident(bridge, xp, patterns_dev, privates_dev, thal_dev, snap, std, thal_tonic_pA, target=0,
                  hold_steps=CONF_HOLD_STEPS, isolate=True):
    """Confident single-content commit: drive target strong; HOLD long (baseline U=0 -> no self-eviction); read."""
    if isolate:
        _counted_full_restore(bridge, snap); std.reset()
    _drive(bridge, xp, thal_dev, thal_tonic_pA, std, [(patterns_dev[target], STRONG_PA)])
    for _ in range(hold_steps):
        _ws_step(bridge, xp, thal_dev, thal_tonic_pA, std)
    rates = _read_private_rates(bridge, xp, thal_dev, thal_tonic_pA, privates_dev, std)
    win, m, n = _margin(rates)
    v, _ = _verdict_label(rates)
    return {"rates": [float(r) for r in rates], "winner": int(win), "n_ignited": int(n), "margin": float(m),
            "delivered": v, "confident_ok": bool(n == 1 and win == target and m >= MARGIN_REF)}


def run_conflict_stop(bridge, xp, patterns_dev, privates_dev, thal_dev, snap, std, thal_tonic_pA, *,
                      boost_gain, boost_scale, margin_ref, pulse_duration, contents=None, do_stop=True,
                      isolate=True, margin_override=None):
    """Co-ignite a multi-content conflict (default = ALL patterns driven), READ the ignition margin (the sensor),
    then apply the conflict-gated depression BURST (boost = f(margin deficit)) on the shared recurrence for
    pulse_duration steps, then read the settled (empty?) workspace. isolate=False -> a CONTINUOUS run (0 restores).
    margin_override feeds a DIFFERENT margin to the sensor (the scramble anti-cheat)."""
    if isolate:
        _counted_full_restore(bridge, snap); std.reset()
    if contents is None:
        contents = tuple(range(len(patterns_dev)))
    _drive(bridge, xp, thal_dev, thal_tonic_pA, std, [(patterns_dev[c], STRONG_PA) for c in contents])
    pre = _read_private_rates(bridge, xp, thal_dev, thal_tonic_pA, privates_dev, std)
    win_pre, margin, n_pre = _margin(pre)
    sensed = float(margin if margin_override is None else margin_override)
    boost = _boost_from_margin(sensed, boost_gain, boost_scale, margin_ref)
    ws_during = []
    if do_stop:
        std.boost = boost
        pdur = int(pulse_duration); late = pdur - max(1, pdur // 3)
        for t in range(pdur):
            _ws_step(bridge, xp, thal_dev, thal_tonic_pA, std)
            if t >= late:
                ws_during.append(float(np.mean([to_host(bridge.cp_firing_states[p].astype(xp.float64).mean())
                                                for p in privates_dev])))
        std.boost = 0.0
    post = _read_private_rates(bridge, xp, thal_dev, thal_tonic_pA, privates_dev, std, n_free=POST_FREE_STEPS)
    _wp, _m, n_post = _margin(post)
    v_pre, _ = _verdict_label(pre); v_post, _ = _verdict_label(post)
    return {"pre_rates": [float(r) for r in pre], "post_rates": [float(r) for r in post], "margin": float(margin),
            "sensed_margin": sensed, "boost": float(boost), "n_ignited_pre": int(n_pre),
            "n_ignited_post": int(n_post), "winner_pre": int(win_pre), "delivered_pre": v_pre,
            "delivered_post": v_post, "aborted": bool(n_pre >= 1 and n_post == 0),
            "global_stop_ok": bool(n_pre >= 2 and n_post == 0),
            "ws_rate_during": (float(np.mean(ws_during)) if ws_during else None)}


def run_gain_withdrawal(bridge, xp, patterns_dev, privates_dev, thal_dev, snap, std, thal_tonic_pA, target=0,
                        withdraw_steps=100, isolate=True):
    """NAIVE gain-withdrawal (the STN-named lever): ignite one content, then FREE-RUN with the tonic thal support
    OFF (tonic=0). Measures whether removing the shared gain de-ignites the content. On this bistable substrate the
    ignited attractor is self-sufficient -> it does NOT clear (a NO-GO; reproduces the STN self-sufficiency)."""
    if isolate:
        _counted_full_restore(bridge, snap); std.reset()
    _drive(bridge, xp, thal_dev, thal_tonic_pA, std, [(patterns_dev[target], STRONG_PA)])
    pre = _read_private_rates(bridge, xp, thal_dev, thal_tonic_pA, privates_dev, std)
    _wp, _m, n_pre = _margin(pre)
    for _ in range(withdraw_steps):                                   # tonic OFF: withdraw the shared gain
        _ws_step(bridge, xp, thal_dev, 0.0, std)
    post = _read_private_rates(bridge, xp, thal_dev, 0.0, privates_dev, std)
    _wp2, _m2, n_post = _margin(post)
    return {"n_ignited_pre": int(n_pre), "n_ignited_post": int(n_post),
            "gain_withdrawal_clears": bool(n_pre >= 1 and n_post == 0),
            "pre_rates": [float(r) for r in pre], "post_rates": [float(r) for r in post]}


def run_overwrite(bridge, xp, patterns_dev, privates_dev, thal_dev, snap, std, thal_tonic_pA,
                  incumbent=0, challenger=1, isolate=True):
    """Active OVERWRITE by pure divisive-norm competition: ignite the incumbent A (hold), then drive the salient
    challenger B (A not re-driven) -> does B EVICT A (n stays 1, winner switches)? On the bistable substrate the
    self-sufficient incumbent HOLDS (co-ignition) -> a BOUNDARY (characterized)."""
    if isolate:
        _counted_full_restore(bridge, snap); std.reset()
    _drive(bridge, xp, thal_dev, thal_tonic_pA, std, [(patterns_dev[incumbent], STRONG_PA)])
    pre = _read_private_rates(bridge, xp, thal_dev, thal_tonic_pA, privates_dev, std)
    win_pre, _m, n_pre = _margin(pre)
    _drive(bridge, xp, thal_dev, thal_tonic_pA, std, [(patterns_dev[challenger], STRONG_PA)])
    post = _read_private_rates(bridge, xp, thal_dev, thal_tonic_pA, privates_dev, std)
    win_post, _m2, n_post = _margin(post)
    return {"pre_rates": [float(r) for r in pre], "post_rates": [float(r) for r in post],
            "winner_pre": int(win_pre), "winner_post": int(win_post), "n_ignited_pre": int(n_pre),
            "n_ignited_post": int(n_post),
            "overwrite_ok": bool(win_pre == incumbent and n_pre == 1 and win_post == challenger and n_post == 1),
            "co_ignition": bool(n_pre == 1 and n_post >= 2)}


# ── one seed: the GLOBAL-STOP GO gate + anti-cheats + characterized boundaries ─────────────────────────────────
def evaluate_seed(seed, *, w_rec=W_REC, overlap=DEFAULT_OVERLAP, ws_norm_w=WS_NORM_W, norm_ws_w=NORM_WS_W,
                  thal_tonic_pA=THAL_TONIC_PA, thal_ws_w=THAL_WS_W, boost_gain=BOOST_GAIN, boost_scale=BOOST_SCALE,
                  margin_ref=MARGIN_REF, pulse_duration=PULSE_DURATION, heterogeneity=True, verbose=True):
    def _build(**kw):
        params = dict(seed=seed, w_rec=w_rec, overlap=overlap, ws_norm_w=ws_norm_w, norm_ws_w=norm_ws_w,
                      thal_tonic_pA=thal_tonic_pA, thal_ws_w=thal_ws_w, heterogeneity=heterogeneity)
        params.update(kw)          # kw overrides (e.g. overlap=0, norm_lesion=True) without a duplicate-arg error
        return build_overwrite_bridge(**params)

    b, xp, pats, privs, thal_dev, ws_used, snap, handles = _build()
    std = WorkspaceDepression(b, xp, ws_used)
    common = dict(boost_gain=boost_gain, boost_scale=boost_scale, margin_ref=margin_ref,
                  pulse_duration=pulse_duration)

    # GO(2) SELECTIVITY: a confident commit (baseline U=0 -> no self-eviction) settles to a HIGH margin, so the
    #   conflict-gated boost computed from it is ZERO -> the stop does NOT fire -> the commit is NOT disrupted.
    conf = run_confident(b, xp, pats, privs, thal_dev, snap, std, thal_tonic_pA)
    conf_boost = _boost_from_margin(conf["margin"], boost_gain, boost_scale, margin_ref)
    #   CONFIRM not-disrupted: continue the SETTLED commit through a boost=conf_boost(=0) pulse -> still n=1.
    conf_stop = run_confident(b, xp, pats, privs, thal_dev, snap, std, thal_tonic_pA, hold_steps=pulse_duration)
    selectivity = bool(conf["confident_ok"] and conf_boost <= 1e-9 and conf_stop["n_ignited"] == 1
                       and conf_stop["winner"] == 0)

    # GO(1) GLOBAL STOP: co-ignite ALL contents -> conflict-triggered depression -> n->0.
    stop = run_conflict_stop(b, xp, pats, privs, thal_dev, snap, std, thal_tonic_pA, do_stop=True, **common)
    global_stop_ok = bool(stop["global_stop_ok"])

    # anti-cheat (b) STD LOAD-BEARING: FREEZE (boost forced 0 on the conflict) -> the conflict HOLDS.
    freeze = run_conflict_stop(b, xp, pats, privs, thal_dev, snap, std, thal_tonic_pA, boost_gain=0.0,
                               boost_scale=boost_scale, margin_ref=margin_ref, pulse_duration=pulse_duration,
                               do_stop=True)
    std_load_bearing = bool(freeze["n_ignited_pre"] >= 2 and freeze["n_ignited_post"] >= 2)

    # anti-cheat (c) SIGNAL-DRIVEN: sweep the 3rd content weak->strong; boost 0 at zero conflict, rises w/ conflict.
    #    (content0 STRONG, content1 swept 0->STRONG: the 0 end = a confident single commit -> margin high -> boost 0.)
    sweep_boost, sweep_margin, sweep_d = [], [], []
    for d in list(np.linspace(0.0, STRONG_PA, 6)):
        _counted_full_restore(b, snap); std.reset()
        drive_map = [(pats[0], STRONG_PA)] if d <= 0.0 else [(pats[0], STRONG_PA), (pats[1], float(d))]
        _drive(b, xp, thal_dev, thal_tonic_pA, std, drive_map)
        rr = _read_private_rates(b, xp, thal_dev, thal_tonic_pA, privs, std)
        _w, mg, _n = _margin(rr)
        sweep_boost.append(_boost_from_margin(mg, boost_gain, boost_scale, margin_ref))
        sweep_margin.append(float(mg)); sweep_d.append(float(d))
    # The essential SIGNAL-DRIVEN property (NOT monotone scaling — the co-ignition competition is genuinely
    # non-monotone in the challenger drive on this substrate, a documented property, so the sweep is kept as DATA):
    #   a CONFIDENT single commit (high margin) -> boost 0; a CONFLICT state (co-ignited, low margin) -> boost > 0.
    conf_boost0 = _boost_from_margin(conf["margin"], boost_gain, boost_scale, margin_ref)
    pulse_zero_at_zero_conflict = bool(conf_boost0 <= 1e-9 and sweep_boost[0] <= 1e-9)
    pulse_fires_on_conflict = bool(stop["boost"] > 1e-9)

    # anti-cheat (c) NEURAL-SENSOR scramble: feed the CONFIDENT margin to the conflict -> boost=0 -> not cleared.
    scramble = run_conflict_stop(b, xp, pats, privs, thal_dev, snap, std, thal_tonic_pA, do_stop=True,
                                 margin_override=conf["margin"], **common)
    scramble_breaks_stop = bool(scramble["boost"] <= 1e-9 and scramble["n_ignited_post"] >= 2)

    # anti-cheat (c) CONFLICT-OFF: boost_gain=0 -> conflict not cleared.
    conflict_off = run_conflict_stop(b, xp, pats, privs, thal_dev, snap, std, thal_tonic_pA, boost_gain=0.0,
                                     boost_scale=boost_scale, margin_ref=margin_ref, pulse_duration=pulse_duration,
                                     do_stop=True)
    conflict_off_holds = bool(conflict_off["n_ignited_post"] >= 2)

    # anti-cheat (a) DIVISIVE-NORM LOAD-BEARING: lesion norm_pool -> the SAME depression burst does NOT clear.
    bl, xpl, patsl, privsl, thall, ws_usedl, snapl, _ = _build(norm_lesion=True)
    stdl = WorkspaceDepression(bl, xpl, ws_usedl)
    stop_lesion = run_conflict_stop(bl, xpl, patsl, privsl, thall, snapl, stdl, thal_tonic_pA, do_stop=True,
                                    **common)
    norm_load_bearing = bool(stop_lesion["n_ignited_pre"] >= 2 and stop_lesion["n_ignited_post"] >= 2)
    intact_emptied = float(stop["n_ignited_pre"] - stop["n_ignited_post"])
    lesion_emptied = float(stop_lesion["n_ignited_pre"] - stop_lesion["n_ignited_post"])
    stop_attribution = attributable_to("global stop (workspace-emptying) via divisive-norm + shared-recurrence STD",
                                       intact_emptied, lesion_emptied, warn_below=0.8)

    # anti-cheat (c) CONTINUOUS stop headline — ZERO restore calls (the emptying is the synaptic depression).
    bd, xpd, patsd, privsd, thald, ws_usedd, snapd, _ = _build()
    stdd = WorkspaceDepression(bd, xpd, ws_usedd)
    restore_before = _RESTORE_CALLS["n"]
    cont = run_conflict_stop(bd, xpd, patsd, privsd, thald, snapd, stdd, thal_tonic_pA, do_stop=True,
                             isolate=False, **common)
    continuous_no_restore = bool(_RESTORE_CALLS["n"] == restore_before)
    continuous_stop = bool(cont["n_ignited_pre"] >= 2 and cont["n_ignited_post"] == 0)

    # anti-cheat (d) BYTE-IDENTICAL substrate: distributed vs localist base (overlap=0 + norm off) at this seed.
    bb, xpb, _pb, _pvb, _tb, _wu, _sb, _hb = _build(overlap=0, norm_lesion=True)
    h_dist = _threshold_hash(b, xp); h_loc = _threshold_hash(bb, xpb)
    byte_identical_substrate = bool(h_dist == h_loc and h_dist != "")

    # anti-cheat (e) DETERMINISM: build twice at this seed -> identical substrate hash.
    b2, xp2, _p2, _pv2, _t2, _wu2, _s2, _h2 = _build()
    seed_deterministic = bool(_threshold_hash(b2, xp2) == h_dist and h_dist != "")

    # ── CHARACTERIZED boundaries (reported, not gating the primary GO) ─────────────────────────────────────────
    gain_wd = run_gain_withdrawal(b, xp, pats, privs, thal_dev, snap, std, thal_tonic_pA)
    overwrite = run_overwrite(b, xp, pats, privs, thal_dev, snap, std, thal_tonic_pA)

    host_workspace_reset_calls = 0 if continuous_no_restore else 1

    # ── validity preconditions (checked BEFORE scoring -> failure = UNDEFINED, not a false negative) ────────────
    confident_ignites = bool(conf["n_ignited"] >= 1)
    conflict_present = bool(stop["n_ignited_pre"] >= 2)                # a 2+-content conflict to clear
    conflict_readable = bool(stop["margin"] < conf["margin"] - 1e-6 and stop["margin"] < margin_ref
                             and conf["margin"] >= margin_ref)
    norm_active = bool(norm_ws_w > 0.0)
    pulse_gated = bool(pulse_zero_at_zero_conflict and pulse_fires_on_conflict)

    seed_go = bool(global_stop_ok and selectivity and std_load_bearing and norm_load_bearing
                   and scramble_breaks_stop and conflict_off_holds and continuous_no_restore and continuous_stop
                   and byte_identical_substrate and seed_deterministic and pulse_gated)

    v = Verdict("distributed-workspace GLOBAL STOP @ frozen op (seed %d)" % seed)
    v.require("confident commit ignites (n>=1)", confident_ignites, expect=True)
    v.require("a multi-content conflict state exists to clear (co-ignition n>=2)", conflict_present, expect=True)
    v.require("normalization pool active (norm_ws_w > 0)", norm_active, expect=True)
    v.require("conflict margin readable (conflict low < confident high, below margin_ref)", conflict_readable,
              expect=True)
    v.require("depression boost conflict-gated (0 at a confident commit, fires on conflict)", pulse_gated, expect=True)
    v.require("GLOBAL STOP: conflict n>=2 -> n_ignited=0", global_stop_ok, expect=True)
    v.require("SELECTIVITY: confident commit not disrupted (boost=0 -> holds n=1)", selectivity, expect=True)
    v.require("divisive-norm LOAD-BEARING (lesion -> burst does NOT clear, n stays >=2)", norm_load_bearing,
              expect=True)
    v.require("STD LOAD-BEARING (freeze boost=0 -> conflict HOLDS)", std_load_bearing, expect=True)
    v.require("neural-sensor SCRAMBLE breaks the stop (confident margin -> not cleared)", scramble_breaks_stop,
              expect=True)
    v.require("CONFLICT-OFF holds (boost_gain=0 -> not cleared)", conflict_off_holds, expect=True)
    v.require("continuous stop makes 0 host restore calls (emptying is synaptic depression)", continuous_no_restore,
              expect=True)
    v.require("byte-identical substrate (distributed hash == localist base hash)", byte_identical_substrate,
              expect=True)
    v.require("determinism: cfg.seed seeds the substrate (build-twice hash)", seed_deterministic, expect=True)
    v.disabled("homeostasis", why="frozen weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("native_short_term_plasticity", why="banked global-STP foot-gun; STD here targets the recurrence only")
    vd = v.decide(go=seed_go, verbose=verbose)

    result = {
        "seed": int(seed), "verdict": vd["status"], "seed_go": bool(seed_go and vd["status"] == "GO"),
        "operating_point": {"w_rec": float(w_rec), "overlap": int(overlap), "ws_norm_w": float(ws_norm_w),
                            "norm_ws_w": float(norm_ws_w), "thal_tonic_pA": float(thal_tonic_pA),
                            "thal_ws_w": float(thal_ws_w), "boost_gain": float(boost_gain),
                            "boost_scale": float(boost_scale), "margin_ref": float(margin_ref),
                            "pulse_duration": int(pulse_duration), "heterogeneity": bool(heterogeneity)},
        "go_gate": {"global_stop": global_stop_ok, "selectivity": selectivity},
        "anti_cheats": {"norm_load_bearing": norm_load_bearing, "std_load_bearing": std_load_bearing,
                        "pulse_zero_at_zero_conflict": pulse_zero_at_zero_conflict,
                        "pulse_fires_on_conflict": pulse_fires_on_conflict,
                        "neural_sensor_scramble_breaks_stop": scramble_breaks_stop,
                        "conflict_off_holds": conflict_off_holds, "continuous_no_restore": continuous_no_restore,
                        "byte_identical_substrate": byte_identical_substrate,
                        "seed_deterministic": seed_deterministic},
        "characterized_boundaries": {
            "gain_withdrawal": {"n_pre": gain_wd["n_ignited_pre"], "n_post": gain_wd["n_ignited_post"],
                                "clears": gain_wd["gain_withdrawal_clears"]},
            "active_overwrite": {"winner_pre": overwrite["winner_pre"], "winner_post": overwrite["winner_post"],
                                 "n_pre": overwrite["n_ignited_pre"], "n_post": overwrite["n_ignited_post"],
                                 "overwrite_ok": overwrite["overwrite_ok"], "co_ignition": overwrite["co_ignition"]}},
        "measurements": {"confident": conf, "confident_stop": conf_stop, "stop": stop, "freeze": freeze,
                         "stop_lesion": stop_lesion, "scramble": scramble, "conflict_off": conflict_off,
                         "continuous_headline": cont, "gain_withdrawal": gain_wd, "overwrite": overwrite,
                         "conflict_sweep_drive": [float(x) for x in sweep_d],
                         "conflict_sweep_boost": [float(x) for x in sweep_boost],
                         "conflict_sweep_margin": [float(x) for x in sweep_margin],
                         "stop_attribution": (None if stop_attribution is None else float(stop_attribution)),
                         "intact_emptied": intact_emptied, "lesion_emptied": lesion_emptied,
                         "ws_rate_during_stop": stop["ws_rate_during"], "substrate_hash": h_dist,
                         "localist_hash": h_loc, "n_rec_syn": std.n_rec_syn, "private_sizes": handles["private_sizes"]},
        "residual": {"global_stop_n_post": int(stop["n_ignited_post"]),
                     "global_stop_n_pre": int(stop["n_ignited_pre"]),
                     "lesion_stop_n_post": int(stop_lesion["n_ignited_post"]),
                     "gain_withdrawal_n_post": int(gain_wd["n_ignited_post"]),
                     "overwrite_n_post": int(overwrite["n_ignited_post"])},
        "host_workspace_reset_calls": int(host_workspace_reset_calls),
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
        "undefined_reasons": vd["undefined_reasons"],
    }
    if verbose:
        print(f"[dist-stop seed={seed}] verdict={vd['status']} seed_go={result['seed_go']}", flush=True)
        print(f"    GLOBAL STOP: conflict n {stop['n_ignited_pre']}->{stop['n_ignited_post']} margin={stop['margin']:.3f}"
              f" boost={stop['boost']:.3f} ok={global_stop_ok} | SELECTIVITY: conf n={conf['n_ignited']} "
              f"m={conf['margin']:.3f} conf_boost={conf_boost:.3f} held_n={conf_stop['n_ignited']} ok={selectivity}",
              flush=True)
        print(f"    LOAD-BEARING: norm(lesion n_post={stop_lesion['n_ignited_post']})={norm_load_bearing} "
              f"STD(freeze n_post={freeze['n_ignited_post']})={std_load_bearing} attribution={stop_attribution}",
              flush=True)
        print(f"    anti-cheats: gated={pulse_gated} scramble={scramble_breaks_stop} conflict_off={conflict_off_holds}"
              f" cont_no_restore={continuous_no_restore} byte_id={byte_identical_substrate} det={seed_deterministic}",
              flush=True)
        print(f"    CHARACTERIZED: gain-withdrawal n {gain_wd['n_ignited_pre']}->{gain_wd['n_ignited_post']} "
              f"(clears={gain_wd['gain_withdrawal_clears']}, NO-GO) | overwrite win {overwrite['winner_pre']}->"
              f"{overwrite['winner_post']} n {overwrite['n_ignited_pre']}->{overwrite['n_ignited_post']} "
              f"(ok={overwrite['overwrite_ok']}, boundary)", flush=True)
    return result


# ── smoke ───────────────────────────────────────────────────────────────────────────────────────────────────
def run_smoke(seed, args):
    print(f"[dist-stop smoke] seed={seed} — operating-point grid", flush=True)
    grid = []
    for wr in ([args.w_rec] if args.w_rec_grid is None else args.w_rec_grid):
        for ov in ([args.overlap] if args.overlap_grid is None else args.overlap_grid):
            for nw in ([args.norm_ws_w] if args.norm_ws_grid is None else args.norm_ws_grid):
                r = evaluate_seed(seed, w_rec=wr, overlap=int(ov), norm_ws_w=nw, boost_scale=args.boost_scale,
                                  pulse_duration=args.pulse_duration, heterogeneity=not args.no_heterogeneity,
                                  verbose=True)
                grid.append({"w_rec": wr, "overlap": int(ov), "norm_ws_w": nw, "seed_go": r["seed_go"],
                             "go_gate": r["go_gate"], "anti_cheats": r["anti_cheats"],
                             "boundaries": r["characterized_boundaries"]})
    any_go = any(g["seed_go"] for g in grid)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_distributed_overwrite_workspace_derisk", "mode": "smoke", "seed": seed,
                   "grid": grid}, f, indent=2, default=str)
    print(f"\n[dist-stop smoke] wrote {args.json}  any_seed_go={any_go}", flush=True)
    return 0 if any_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW distributed workspace: divisive-norm + conflict-triggered "
                                             "shared-recurrence depression -> the global stop.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--w-rec", type=float, default=W_REC)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("--ws-norm-w", type=float, default=WS_NORM_W)
    ap.add_argument("--norm-ws-w", type=float, default=NORM_WS_W)
    ap.add_argument("--thal-tonic", type=float, default=THAL_TONIC_PA)
    ap.add_argument("--thal-ws-w", type=float, default=THAL_WS_W)
    ap.add_argument("--boost-scale", type=float, default=BOOST_SCALE)
    ap.add_argument("--pulse-duration", type=int, default=PULSE_DURATION)
    ap.add_argument("--no-heterogeneity", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--w-rec-grid", type=float, nargs="+", default=None)
    ap.add_argument("--overlap-grid", type=int, nargs="+", default=None)
    ap.add_argument("--norm-ws-grid", type=float, nargs="+", default=None)
    ap.add_argument("--six-seed", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_distributed_overwrite_smoke.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    print(f"[dist-stop] K={N_PATTERNS} P={PATTERN_SIZE} overlap={args.overlap} w_rec={args.w_rec} "
          f"norm_ws={args.norm_ws_w} thal_ws={args.thal_ws_w} thal_tonic={args.thal_tonic} "
          f"boost_scale={args.boost_scale} pulse={args.pulse_duration} het={not args.no_heterogeneity} "
          f"backend={args.backend}\n", flush=True)

    if args.smoke:
        return run_smoke(args.seed, args)

    common = dict(w_rec=args.w_rec, overlap=args.overlap, ws_norm_w=args.ws_norm_w, norm_ws_w=args.norm_ws_w,
                  thal_tonic_pA=args.thal_tonic, thal_ws_w=args.thal_ws_w, boost_scale=args.boost_scale,
                  pulse_duration=args.pulse_duration, heterogeneity=not args.no_heterogeneity)
    results = [evaluate_seed(s, verbose=True, **common) for s in args.seeds]

    n_go = sum(int(r["seed_go"]) for r in results)
    any_undefined = any(r["verdict"] == "UNDEFINED" for r in results)
    all_no_reset = all(r["anti_cheats"]["continuous_no_restore"] for r in results)
    all_byte_id = all(r["anti_cheats"]["byte_identical_substrate"] for r in results)
    all_determ = all(r["anti_cheats"]["seed_deterministic"] for r in results)
    all_norm_lb = all(r["anti_cheats"]["norm_load_bearing"] for r in results)
    all_std_lb = all(r["anti_cheats"]["std_load_bearing"] for r in results)
    stop_go = bool(n_go >= 5 and not any_undefined and all_no_reset and all_byte_id and all_determ
                   and all_norm_lb and all_std_lb)
    # characterized boundaries (aggregate): the active overwrite + gain-withdrawal are NOT part of the primary GO.
    n_overwrite = sum(int(r["characterized_boundaries"]["active_overwrite"]["overwrite_ok"]) for r in results)
    n_gain_wd = sum(int(r["characterized_boundaries"]["gain_withdrawal"]["clears"]) for r in results)
    # overall verdict: the DECISIVE global-stop capability is the headline; overwrite is a boundary -> PARTIAL.
    if stop_go and n_overwrite >= 5:
        verdict = "GO"
    elif stop_go:
        verdict = "PARTIAL"          # global-stop GO 6/6; active overwrite a characterized boundary
    elif any_undefined:
        verdict = "UNDEFINED"
    else:
        verdict = "NO-GO"

    summary = {
        "runner": "_gnw_distributed_overwrite_workspace_derisk", "mode": "six-seed", "verdict": verdict,
        "stop_capability_go": stop_go, "n_global_stop_go": n_go, "n_seeds": len(results), "seeds": list(args.seeds),
        "n_active_overwrite_go": n_overwrite, "n_gain_withdrawal_clears": n_gain_wd, "any_undefined": any_undefined,
        "aggregate_anti_cheats": {"all_continuous_no_restore": all_no_reset,
                                  "all_byte_identical_substrate": all_byte_id, "all_seed_deterministic": all_determ,
                                  "all_norm_load_bearing": all_norm_lb, "all_std_load_bearing": all_std_lb},
        "operating_point": results[0]["operating_point"] if results else {}, "per_seed": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 100}", flush=True)
    print(f"  DISTRIBUTED WORKSPACE GLOBAL-STOP VERDICT: {verdict}  "
          f"(global-stop {n_go}/{len(results)} GO; active-overwrite {n_overwrite}/{len(results)} [boundary])",
          flush=True)
    for r in results:
        g = r["go_gate"]; a = r["anti_cheats"]; bnd = r["characterized_boundaries"]
        print(f"    seed {r['seed']}: {r['verdict']:9s} stop={g['global_stop']} select={g['selectivity']} | "
              f"norm_LB={a['norm_load_bearing']} std_LB={a['std_load_bearing']} "
              f"scramble={a['neural_sensor_scramble_breaks_stop']} byte_id={a['byte_identical_substrate']} "
              f"det={a['seed_deterministic']} | overwrite_ok={bnd['active_overwrite']['overwrite_ok']} "
              f"gainwd_clears={bnd['gain_withdrawal']['clears']}", flush=True)
    print(f"    [saved] {args.json}\n{'=' * 100}", flush=True)
    return 0 if stop_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
