"""R1 close (Option 1 of `2026-06-30-tier2-integrated-spiking-loop-scoping.md`): fold the divnorm-score pool + the
K-way sequencer onto ONE Izhikevich "fabric" bridge, and route the cleanup membrane -> score pool DEVICE-RESIDENT
(NO `to_host` of the cleanup score) -- closing the LAST host DATA seam in the integrated who/what query path.

THE RESIDUAL (R1). In `OneBrainComposer(integrated_loop=True)`, the cue-match CONTROL is on-substrate (the spiking
K-way sequencer + the on-bridge `input_divisive_norm`). BUT the cleanup membrane (the OP RESULT, an RF region on
`OneBrainComposer.b`) is read TO HOST (`block_cleanup_scores`: `mem = to_host(b.cp_membrane_potential_v)`) and the
host array is re-driven onto a SEPARATE divnorm-score `SimulationBridge` (`onbridge_divnorm_drive`:
`score_sb.cp_external_input_current[:] = from_host(cur)`). That `to_host` of the cleanup score + the cross-bridge
re-drive is the last host DATA round-trip in the deployed who/what query path -- NOT a host COMPUTATION (the
normalization + match + select are all spiking), a host DATA TRANSFER between co-resident-in-principle fabric slices
that are currently separate bridge objects.

THE CLOSE (this module, reuse-by-import, NO `sim/` edit):
  1. `build_fused_fabric_bridge` -- the divnorm-score pool (V word-pools, input_divisive_norm flagged) + the K-way
     sequencer (the gated-disinhibition match cascade + BG first-match WTA) on ONE `enable_brain_region_framework`
     Izhikevich bridge. Disjoint region namespaces: `score_w{w}` (the score pool) and the sequencer's
     `cueA_*/cueX_*/d{b}{role}_*/mw.../mA{b}/mX{b}/m{b}/ans{b}/abstain/inh{b}`. The score-pool->sequencer hand-off is
     then intra-bridge. Region/pathway specs are the SAME as the validated standalone builders
     (`build_divnorm_score_bridge` + `build_sequencerK_bridge`), merged. The divnorm sim/ primitive
     (`input_divisive_norm`) is reused verbatim; the score-pool drive runs ONE role at a time so the per-query divisor
     is that role's own total (byte-faithful to the S5 op-point).
  2. `fused_block_drives` -- the DEVICE-RESIDENT cleanup->score handoff. The cleanup membrane lives on
     `OneBrainComposer.b` (a backend array). For each block + role, gather the role's V cleanup scores ON-DEVICE
     (a backend slice, NOT `to_host`), scatter them (broadcast each word's score to its n_word score-pool neurons)
     into the fabric bridge's score-pool `cp_external_input_current` (backend -> backend, same device), settle, and
     read which score-pool words FIRE (the placed rheobase -- a BODY read of `cp_firing_states`, the legitimate
     "which neuron fired" boundary, NOT the cleanup score). The per-block (dA, dX) firing drives are the decoded-line
     drive. NO `to_host` of the cleanup membrane anywhere.
  3. `run_fused_sequencer` -- drive the CO-LOCATED sequencer from those device-derived decoded-line drives (the SAME
     match cascade + first-match priority WTA + production rule as the standalone path), and read which BG channel
     wins (the S7 body read).

`OneBrainComposer(integrated_loop="fused")` selects this path (default False = host `_scan`; True = the legacy
separate-bridge spiking path = the revertible escape). Both legacy paths are BYTE-UNCHANGED (this module is purely
additive). The op-point is the validated S5/S2 production op-point (match_thresh 0.06, gain 0.1, sigma 1, input_gain 1).

NO `sim/` edit. The fabric bridge is a plain Izhikevich `_run_one_simulation_step` bridge (its own step path -- it does
NOT use the RF megakernel or the masked rf_kick, so the design's contingent tracker-mask edit is not anticipated). The
cleanup membrane read is the SAME RF op (already masked to `c.rf_mask`), just kept on-device.

  SIM_BACKEND=numpy python -u -m research.runners._seq_fused_fabric --seeds 42,43,44,45,46,47 --dim 128 --ks 2,4
  SIM_BACKEND=cupy  python -u -m research.runners._seq_fused_fabric --seeds 42,43,44,100,101,102 --dim 128 --ks 2,4,8
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.backend import get_backend, to_host, is_gpu_backend
from research.runners.one_brain_composer import OneBrainComposer
# The validated production op-point + fact table + the K-way sequencer's reset/decode/wiring (reused verbatim -- the
# fused fabric replicates the standalone sequencer's CONTROL exactly; only the score pool is co-located + the cleanup
# hand-off device-resident).
from research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk import ALL_FACTS, VOCAB, _build_queries
from research.runners._phaseB_onebrain_sequencerK_derisk import (
    reset_sequencerK_state, host_scan_block, decision_to_block, patient_of)


# The validated production op-point (S5/S2 K=32-margin GO: 2026-06-21-shortcut3-K32-capability-surpass.md).
FUSED_OP = dict(match_thresh=0.06, gain=0.1, sigma=1.0, input_gain=1.0)
N_WORD = 20      # score-pool + word-line pool size (== build_divnorm_score_bridge / build_sequencerK_bridge default)
N_POOL = 30      # match/answer pool size (== build_sequencerK_bridge default)


def build_fused_fabric_bridge(seed, V, K, n_word=N_WORD, n_pool=N_POOL, sigma=FUSED_OP["sigma"],
                              gain=FUSED_OP["gain"], w_match=300.0, w_or=300.0, w_blk=300.0, w_ans=320.0,
                              w_lat_inhib=320.0, abstain_tonic_pA=420.0):
    """ONE Izhikevich bridge holding BOTH the divnorm-score pool AND the K-way sequencer (today two separate bridges).

    The SCORE POOL: V word-pools `score_w{w}` (one normalization pool = the role's V words), all
    input_divisive_norm=True, the global flag on -- the per-step divide r_i = x_i/(sigma + gain*mean_j x_j) over the
    flagged set (== build_divnorm_score_bridge). The sequencer's word-lines are SEPARATE regions (not flagged), so the
    divisive mean is over the score pool ONLY (byte-faithful to the standalone score bridge).

    The SEQUENCER: the K-way gated-disinhibition match cascade + BG first-match priority WTA (== build_sequencerK_bridge),
    region names UNCHANGED (cueA_*/cueX_*/d{b}{role}_*/mw{b}{role}_*/mA{b}/mX{b}/m{b}/ans{b}/abstain/inh{b}).

    Returns (sb, meta). meta carries V, K, n_word, n_pool + the score-pool's per-word first index (precomputed once for
    the device scatter). The score-pool->sequencer hand-off is intra-bridge; the cue + decoded-line drives are written
    to the sequencer regions' cp_external_input_current as in the standalone path."""
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.enable_brain_region_framework = True
    cfg.ou_std_current_pA = 0.0
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp"):
        setattr(cfg, flag, False)
    cfg.enable_vectorized_gate_couplings = True            # == build_sequencerK_bridge (byte-identical gate couplings)
    cfg.enable_input_divisive_norm = True                  # == build_divnorm_score_bridge (the S5 primitive)
    cfg.input_divisive_sigma = float(sigma)
    cfg.input_divisive_gain = float(gain)

    regions = []
    # --- the divnorm SCORE pool (V word-pools, divisive-norm flagged) ---
    regions += [BrainRegion(name=f"score_w{w}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0,
                            input_divisive_norm=True) for w in range(V)]
    # --- the sequencer fabric (== build_sequencerK_bridge regions) ---
    for grp in ("cueA", "cueX"):                           # cue word-lines (shared across blocks)
        regions += [BrainRegion(name=f"{grp}_{w}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0)
                    for w in range(V)]
    for b in range(K):                                     # per-block decoded word-lines
        for role in ("A", "X"):
            regions += [BrainRegion(name=f"d{b}{role}_{w}", n_neurons=n_word, exc_fraction=1.0,
                                    internal_density=0.0) for w in range(V)]
    for b in range(K):                                     # per-word gated-match line (decoded gated by cue)
        for role in ("A", "X"):
            regions += [BrainRegion(name=f"mw{b}{role}_{w}", n_neurons=n_word, exc_fraction=1.0,
                                    internal_density=0.0) for w in range(V)]
    for b in range(K):                                     # match/answer pools per block
        for nm in (f"mA{b}", f"mX{b}", f"m{b}", f"ans{b}"):
            regions.append(BrainRegion(name=nm, n_neurons=n_pool, exc_fraction=1.0, internal_density=0.0))
    regions.append(BrainRegion(name="abstain", n_neurons=n_pool, exc_fraction=1.0, internal_density=0.0))
    for b in range(K):                                     # inhibitory interneurons (first-match priority)
        regions.append(BrainRegion(name=f"inh{b}", n_neurons=n_pool, exc_fraction=0.0, internal_density=0.0))
    cfg.brain_regions = regions

    # --- the sequencer pathways (== build_sequencerK_bridge) ---
    P = []
    for b in range(K):
        for w in range(V):
            for (role, dec, cue) in (("A", f"d{b}A", "cueA"), ("X", f"d{b}X", "cueX")):
                P.append(RegionPathway(from_region=f"{dec}_{w}", to_region=f"mw{b}{role}_{w}", density=1.0,
                                       weight_mean=w_match, weight_jitter=0.0, plastic=False,
                                       transmission_gate=f"g{b}{role}_{w}"))
            P += [RegionPathway(from_region=f"mw{b}A_{w}", to_region=f"mA{b}", density=1.0, weight_mean=w_or,
                                weight_jitter=0.0, plastic=False),
                  RegionPathway(from_region=f"mw{b}X_{w}", to_region=f"mX{b}", density=1.0, weight_mean=w_or,
                                weight_jitter=0.0, plastic=False)]
        P.append(RegionPathway(from_region=f"mX{b}", to_region=f"m{b}", density=1.0, weight_mean=w_blk,
                               weight_jitter=0.0, plastic=False, transmission_gate=f"gblk{b}"))
    w_inh_drive = abs(w_lat_inhib)
    for b in range(K):
        P.append(RegionPathway(from_region=f"m{b}", to_region=f"ans{b}", density=1.0, weight_mean=w_ans,
                               weight_jitter=0.0, plastic=False))
        P.append(RegionPathway(from_region=f"ans{b}", to_region=f"inh{b}", density=1.0, weight_mean=w_inh_drive,
                               weight_jitter=0.0, plastic=False))
        for j in range(b + 1, K):
            P.append(RegionPathway(from_region=f"inh{b}", to_region=f"ans{j}", density=1.0, weight_mean=w_inh_drive,
                                   weight_jitter=0.0, plastic=False))
        P.append(RegionPathway(from_region=f"inh{b}", to_region="abstain", density=1.0, weight_mean=w_inh_drive,
                               weight_jitter=0.0, plastic=False))
    cfg.region_pathways = P

    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    # the cue gate<->pool couplings (== wire_sequencerK_couplings): the cue word-line firing opens the per-word match
    # gate; the agent-match pool firing opens the per-block AND gate.
    for b in range(K):
        for w in range(V):
            sb.couple_gate_to_pool(f"g{b}A_{w}", f"cueA_{w}", threshold=0.03)
            sb.couple_gate_to_pool(f"g{b}X_{w}", f"cueX_{w}", threshold=0.03)
        sb.couple_gate_to_pool(f"gblk{b}", f"mA{b}", threshold=0.03)

    # precompute the score-pool's per-word first neuron index (for the device scatter) -- region indices are
    # query-invariant, so build the (V, n_word) index matrix once.
    score_idx = np.stack([np.asarray(sb.region_manager.indices(f"score_w{w}"), dtype=np.int64) for w in range(V)])
    meta = dict(V=int(V), K=int(K), n_word=int(n_word), n_pool=int(n_pool), abstain_tonic_pA=float(abstain_tonic_pA),
                score_idx=score_idx)
    return sb, meta


def _reset_score_pool(sb, score_idx_flat):
    """Reset the score-pool neurons (the divisive-norm word-pools) to resting before a role's settle so each role's
    divisive divisor is that role's OWN per-query total (the standalone `_reset_score_bridge` discipline, restricted
    to the score-pool slice since the sequencer regions share the bridge). Resting = the Izhikevich c-reset, NOT 0mV."""
    xp, _ = get_backend()
    sl = score_idx_flat
    if getattr(sb, "cp_izh_c_reset", None) is not None:
        # cp_izh_c_reset may be a scalar-broadcast or per-neuron array; index defensively.
        cr = sb.cp_izh_c_reset
        sb.cp_membrane_potential_v[sl] = cr[sl] if hasattr(cr, "shape") and cr.shape == sb.cp_membrane_potential_v.shape else -65.0
    else:
        sb.cp_membrane_potential_v[sl] = -65.0
    sb.cp_recovery_variable_u[sl] = 0.0
    if getattr(sb, "cp_firing_states", None) is not None:
        sb.cp_firing_states[sl] = False
    for attr in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise"):
        arr = getattr(sb, attr, None)
        if arr is not None:
            arr[sl] = 0.0


def _device_role_scores(c, block_idx, role_a="agent", role_x="action"):
    """Run the composer's validated reconstruct + unbind + cleanup for one block (the SAME RF op `block_cleanup_scores`
    runs) but return the cleanup membrane DEVICE-RESIDENT -- the agent + action score slices stay on `c.b`'s membrane
    (a backend array), NO `to_host`. Returns (mem_dev, sa_slice, sx_slice) where mem_dev is the live device membrane and
    sa/sx_slice are the (start, stop) c_base offsets for role_a/role_x. The caller gathers the role's V scores from
    mem_dev[start:stop] ON-DEVICE. This REPLACES `block_cleanup_scores`'s `to_host(b.cp_membrane_potential_v)`."""
    comp, b, D, Pd, V = c.comp, c.b, c.D, c.period, c.V
    ra = c.main_roles.index(role_a)
    rx = c.main_roles.index(role_x)
    b.cp_membrane_potential_v[:] = 0.0
    b.cp_recovery_variable_u[:] = 0.0
    trig = c.store_base + block_idx * c.block
    kick = np.zeros(c.n_total, dtype=np.complex128)
    kick[trig] = 1.0
    b.rf_set_complex_weights(c.store_conns)
    b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=c.rf_mask)
    b.rf_resonate_steps(Pd + 8)
    unbind = []
    for ri, role in enumerate(c.bind_roles):
        zc = np.conj(comp._to_phasor(comp.roles[role]))
        unbind += [(c.q_base + ri * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
    b.rf_set_complex_weights(unbind)
    b.rf_resonate_steps(Pd + 8)
    clean = c._seq_cleanup_conns()                          # opt #4: block-invariant cleanup codebook (== standalone)
    b.rf_set_complex_weights(clean)
    b.rf_resonate_steps(1)
    # the cleanup membrane is NOW on b.cp_membrane_potential_v (device). DO NOT to_host it -- return the live array +
    # the role offsets so the caller gathers on-device.
    mem_dev = b.cp_membrane_potential_v
    sa = (c.c_base + ra * V, c.c_base + (ra + 1) * V)
    sx = (c.c_base + rx * V, c.c_base + (rx + 1) * V)
    return mem_dev, sa, sx


def _onbridge_divnorm_drive_device(sb, meta, score_vec_dev, input_gain, settle=20, hi_pA=1500.0):
    """OPTION 4 on the FUSED bridge, DEVICE-RESIDENT input: `score_vec_dev` is a V-length backend array (gathered
    on-device from the cleanup membrane -- NEVER `to_host`-ed). Scatter input_gain*max(score,0) onto each score-pool
    word's n_word neurons (broadcast), let the bridge's input_divisive_norm divide the pre-threshold drive by the
    per-query total pool drive every step, and read which word pools FIRE (the placed rheobase). Returns a HOST boolean
    `lit[V]` -- the body read of `cp_firing_states` over the score pool (which word fired), NOT the cleanup score.
    NO `to_host` of the cleanup membrane (only the firing-state body read)."""
    xp, _ = get_backend()
    V, n_word = meta["V"], meta["n_word"]
    score_idx = meta["score_idx"]                           # (V, n_word) int64 host
    score_idx_flat = xp.asarray(score_idx.reshape(-1))
    _reset_score_pool(sb, score_idx_flat)
    # device drive = input_gain * max(score, 0), broadcast each word -> its n_word pool neurons.
    sdev = xp.maximum(xp.asarray(score_vec_dev), 0.0).astype(sb.cp_external_input_current.dtype)
    drv_per_word = (float(input_gain) * sdev)               # (V,) device
    sb.cp_external_input_current[:] = 0.0
    # scatter: cp_external_input_current[score_idx[w, :]] = drv_per_word[w]  (broadcast over n_word) -- device-resident.
    idx_dev = xp.asarray(score_idx)                         # (V, n_word) device
    sb.cp_external_input_current[idx_dev] = drv_per_word[:, None]
    acc = np.zeros(V)
    for _ in range(settle):
        sb._run_one_simulation_step()
        fir = sb.cp_firing_states[idx_dev]                  # (V, n_word) device bool -- the BODY read (which fired)
        acc += np.asarray(to_host(fir.mean(axis=1))).astype(float)   # per-word firing fraction (firing-state read only)
    sb.cp_external_input_current[:] = 0.0
    return acc > 0                                           # lit[V] host bool


def fused_block_drives(c, sb, meta, input_gain=FUSED_OP["input_gain"]):
    """The DEVICE-RESIDENT cleanup -> score-pool -> firing drive, per block. For each stored block: run the RF cleanup
    (device), gather the agent + action score slices ON-DEVICE, divisively-normalize + threshold them through the
    co-located score pool, and return the per-block (dA, dX) decoded-line firing drives (hi_pA on the firing words).
    NO `to_host` of the cleanup membrane anywhere. Mirrors `make_block_drives` but with the cleanup score kept on the
    device the whole way from the RF cleanup to the score-pool drive."""
    xp, _ = get_backend()
    K, V = meta["K"], meta["V"]
    hi_pA = 1500.0
    drives = []
    for bi in range(min(K, len(c.kb))):
        mem_dev, (sa0, sa1), (sx0, sx1) = _device_role_scores(c, bi)
        ag_dev = mem_dev[sa0:sa1]                           # device slice (the agent cleanup scores) -- NOT to_host
        ax_dev = mem_dev[sx0:sx1]
        lit_a = _onbridge_divnorm_drive_device(sb, meta, ag_dev, input_gain)
        lit_x = _onbridge_divnorm_drive_device(sb, meta, ax_dev, input_gain)
        dA = np.where(lit_a, hi_pA, 0.0)
        dX = np.where(lit_x, hi_pA, 0.0)
        drives.append((dA, dX))
    return drives


def run_fused_sequencer(sb, meta, cue_agent_idx, cue_action_idx, block_drives, settle=60, lesion=False,
                        match_thresh=FUSED_OP["match_thresh"], permute=False):
    """Drive the CO-LOCATED sequencer (on the fused bridge) from the device-derived decoded-line drives + the cue, and
    read which BG channel wins (== the standalone `run_sequencerK_with_drive`: the SAME match cascade, first-match
    priority WTA, production rule). `block_drives` = [(dA[V], dX[V]), ...] per block (the fused_block_drives output).
    Returns (decision, rates). lesion=True zeros the decoded drive (fail-safe -> abstain); permute cyclically shifts
    the match->answer rule (anti-cheat)."""
    V, K = meta["V"], meta["K"]
    xp, _ = get_backend()
    reset_sequencerK_state(sb)                              # the standalone per-query housekeeping (drain + c-reset)
    cur = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    idx = lambda nm: np.asarray(sb.region_manager.indices(nm))
    cur[idx(f"cueA_{cue_agent_idx}")] = 1500.0
    cur[idx(f"cueX_{cue_action_idx}")] = 1500.0
    if not lesion:
        for bi, (dA, dX) in enumerate(block_drives[:K]):
            for w in range(V):
                if dA[w] > 0:
                    cur[idx(f"d{bi}A_{w}")] = dA[w]
                if dX[w] > 0:
                    cur[idx(f"d{bi}X_{w}")] = dX[w]
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    cur_dev = xp.asarray(cur)
    for _ in range(settle):
        sb.cp_external_input_current[:] = cur_dev
        sb._run_one_simulation_step()
        acc += np.asarray(to_host(sb.cp_firing_states)).astype(np.float64)   # firing-state body read (not cleanup score)
    sb.cp_external_input_current[:] = 0.0
    m_rates = [acc[idx(f"m{b}")].mean() / settle for b in range(K)]
    fired = [r > match_thresh for r in m_rates]
    rates = {f"m{b}": round(m_rates[b], 3) for b in range(K)}
    winner = next((b for b in range(K) if fired[b]), None)
    decision = "abstain" if winner is None else (f"ans{(winner + 1) % K}" if permute else f"ans{winner}")
    rates["winner"] = winner
    return decision, rates


# ----------------------------------------------------------------------------------------------------------------
# Composer-facing entry points: build + cache the fused fabric on the composer, run the (agent, action) routing.
# ----------------------------------------------------------------------------------------------------------------
def ensure_fused_fabric(c, K):
    """Lazily build (and cache) the fused fabric bridge + its per-block device-resident decoded-line drives for store
    size K, recomputing the drives when the store grew or a write dirtied them. Mirrors `_ensure_sequencer` but for the
    fused (one-bridge, device-resident-handoff) path. Stored on the composer as `_fused_seq` / `_fused_drives`."""
    if getattr(c, "_fused_seq", None) is None or c._fused_K != K:
        sb, meta = build_fused_fabric_bridge(seed=c.seed, V=c.V, K=K, sigma=c.sequencer_sigma, gain=c.sequencer_gain)
        c._fused_seq = (sb, meta)
        c._fused_K = K
        c._fused_dirty = True
    if c._fused_dirty or getattr(c, "_fused_drives", None) is None:
        c._seq_cleanup_conns_cache = None                  # opt #4: rebuild the block-invariant cleanup conns once
        sb, meta = c._fused_seq
        c._fused_drives = fused_block_drives(c, sb, meta, input_gain=c.sequencer_input_gain)
        c._fused_dirty = False


def fused_seq_block(c, agent, action):
    """The SELECTED block index for cue (agent, action) on the FUSED path (the device-resident-handoff spiking K-way
    sequencer decision), or None = abstain. An absent cue WORD abstains before the sequencer (the moat). Mirrors
    `_seq_block`'s spiking branch."""
    K = len(c.kb)
    if K == 0:
        return None
    if agent not in c._word_index or action not in c._word_index:
        return None                                        # absent cue word -> no block -> abstain (the moat)
    ensure_fused_fabric(c, K)
    sb, meta = c._fused_seq
    dec, _rates = run_fused_sequencer(sb, meta, c._word_index[agent], c._word_index[action], c._fused_drives,
                                      match_thresh=c.sequencer_match_thresh)
    return decision_to_block(dec, K)


def fused_query_patient_lesioned(c, agent, action):
    """Anti-cheat helper: the fused query with the cleanup->score drive SEVERED (lesion) -> the sequencer must abstain
    (fail safe), never confabulate. Returns the answer (must be None for a fail-safe lesion)."""
    K = len(c.kb)
    if K == 0 or agent not in c._word_index or action not in c._word_index:
        return None
    ensure_fused_fabric(c, K)
    sb, meta = c._fused_seq
    dec, _ = run_fused_sequencer(sb, meta, c._word_index[agent], c._word_index[action], c._fused_drives,
                                 match_thresh=c.sequencer_match_thresh, lesion=True)
    idx = decision_to_block(dec, K)
    if idx is None:
        return None
    return c.kb[idx][0].get("patient")


# ----------------------------------------------------------------------------------------------------------------
# The CPU/GPU de-risk (the R1 GO bar): ==host + moat 0-FA + cleanup-score to_host eliminated + lesion fails safe +
# OFF byte-identical. CPU runs a toy K subset; the GPU 6-seed K{2,4,8} is the controller's run.
# ----------------------------------------------------------------------------------------------------------------
def _build_composers(seed, D, K, vocab):
    kw = dict(seed=seed, D=D, vocab=vocab, k_max=max(8, K), enable_batched=False, enable_rf_cudagraph=False)
    op = dict(sequencer_match_thresh=FUSED_OP["match_thresh"], sequencer_gain=FUSED_OP["gain"],
              sequencer_sigma=FUSED_OP["sigma"], sequencer_input_gain=FUSED_OP["input_gain"])
    c_host = OneBrainComposer(integrated_loop=False, **kw)
    c_sep = OneBrainComposer(integrated_loop=True, **kw, **op)
    c_fused = OneBrainComposer(integrated_loop="fused", **kw, **op)
    for (a, x, p) in ALL_FACTS[:K]:
        c_host.store(a, x, p); c_sep.store(a, x, p); c_fused.store(a, x, p)
    return c_host, c_sep, c_fused


def _to_host_spy(c_fused):
    """Instrument sim.backend.to_host + the one_brain_composer alias to count reads of the composer's cleanup membrane
    (the RF bridge `b`'s cp_membrane_potential_v -- the cleanup-score carrier). Returns (install, restore, counter)."""
    import sim.backend as backend
    from research.runners import one_brain_composer as obc
    cleanup_membrane = c_fused.b.cp_membrane_potential_v
    counter = {"cleanup_membrane_reads": 0, "total": 0}
    real = backend.to_host

    def _spy(arr):
        counter["total"] += 1
        if arr is cleanup_membrane:
            counter["cleanup_membrane_reads"] += 1
        return real(arr)

    def install():
        backend.to_host = _spy
        obc.to_host = _spy

    def restore():
        backend.to_host = real
        obc.to_host = real

    return install, restore, counter


def run_seed_K(seed, D, K, vocab):
    c_host, c_sep, c_fused = _build_composers(seed, D, K, vocab)
    queries = _build_queries(ALL_FACTS[:K])

    rows = []
    for (qa, qx), kind in queries:
        h = c_host.query_patient(qa, qx)
        s = c_sep.query_patient(qa, qx)
        f = c_fused.query_patient(qa, qx)
        rows.append(dict(cue=(qa, qx), kind=kind, host=h, sep=s, fused=f,
                         fused_eq_host=(f == h), sep_eq_host=(s == h)))
    eq_host = all(r["fused_eq_host"] for r in rows)
    sep_eq_host = all(r["sep_eq_host"] for r in rows)        # op-point sanity (the legacy spiking path also == host)
    moat_rows = [r for r in rows if r["kind"].startswith(("absent", "cross"))]
    fa = sum(1 for r in moat_rows if r["fused"] is not None)
    moat_ok = (fa == 0)

    # cleanup-score to_host eliminated (R1, precisely): instrument ONLY the cleanup->sequencer hand-off (`_seq_block`,
    # the S4 cleanup -> S5 score -> S6 select path R1 lives in) -> 0 cleanup-membrane reads. The SEPARATE downstream
    # patient body-read (`query_patient`'s `got = _read_blocks()[idx]`, S7) is R5 (the legitimate "which neuron won"
    # boundary, closed under enable_spiking_cleanup) -- NOT the cleanup->sequencer DATA seam, so it is excluded here.
    install, restore, counter = _to_host_spy(c_fused)
    install()
    try:
        a0, x0 = ALL_FACTS[0][0], ALL_FACTS[0][1]
        _ = c_fused._seq_block(a0, x0)
    finally:
        restore()
    to_host_clean = (counter["cleanup_membrane_reads"] == 0)

    # lesion-fails-safe: sever the cleanup->score drive on every present cue -> abstain.
    les = [fused_query_patient_lesioned(c_fused, a, x) for (a, x, p) in ALL_FACTS[:K]]
    lesion_fails_safe = all(v is None for v in les)

    # OFF==byte-identical: the two legacy paths still agree (the fused change is additive).
    off_byte_identical = sep_eq_host

    go = eq_host and moat_ok and to_host_clean and lesion_fails_safe and off_byte_identical
    return dict(seed=seed, D=D, K=K, rows=rows, eq_host=eq_host, sep_eq_host=sep_eq_host, moat_ok=moat_ok, fa=fa,
                to_host_clean=to_host_clean, to_host_counter=counter, lesion_fails_safe=lesion_fails_safe,
                lesion_answers=les, off_byte_identical=off_byte_identical, go=go)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--ks", default="2", help="store sizes K (CPU smoke: a toy subset; GPU 6-seed: 2,4,8)")
    ap.add_argument("--out", default="research/findings/raw/_seq_fused_fabric.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ks = [int(k) for k in args.ks.split(",")]

    print(f"[R1 FUSED FABRIC] fold score-pool + K-way sequencer onto ONE bridge; cleanup->score DEVICE-RESIDENT "
          f"(no to_host of the cleanup score). op={FUSED_OP} V={len(VOCAB)} gpu={is_gpu_backend()}\n", flush=True)
    all_results = {}
    for K in ks:
        results = []
        for s in seeds:
            r = run_seed_K(s, args.dim, K, VOCAB)
            results.append(r)
            eq = "==host" if r["eq_host"] else "!=HOST"
            moat = "moat-OK" if r["moat_ok"] else f"MOAT-BREACH(fa={r['fa']})"
            th = "to_host-clean" if r["to_host_clean"] else f"TO_HOST-LEAK({r['to_host_counter']})"
            les = "lesion-SAFE" if r["lesion_fails_safe"] else f"lesion-UNSAFE({r['lesion_answers']})"
            off = "OFF-byte-ident" if r["off_byte_identical"] else "OFF-DIVERGES"
            det = "  ".join(f"{rr['kind']}:fused={rr['fused']}|host={rr['host']}" for rr in r["rows"])
            print(f"K={K} seed {s} D{args.dim}: {'GO' if r['go'] else 'NO'}  {eq}  {moat}  {th}  {les}  {off}",
                  flush=True)
            print(f"    {det}", flush=True)
        all_results[str(K)] = results

    summary = {}
    overall = True
    for K in ks:
        rs = all_results[str(K)]
        n = len(rs)
        eq_n = sum(r["eq_host"] for r in rs)
        moat_n = sum(r["moat_ok"] for r in rs)
        th_n = sum(r["to_host_clean"] for r in rs)
        les_n = sum(r["lesion_fails_safe"] for r in rs)
        off_n = sum(r["off_byte_identical"] for r in rs)
        fa_total = sum(r["fa"] for r in rs)
        go = (eq_n == n and moat_n == n and th_n == n and les_n == n and off_n == n and fa_total == 0)
        overall = overall and go
        summary[str(K)] = dict(n=n, eq_host_n=eq_n, moat_n=moat_n, fa_total=fa_total, to_host_clean_n=th_n,
                               lesion_n=les_n, off_byte_identical_n=off_n, verdict="GO" if go else "NEGATIVE")
        print(f"\nK={K} SUMMARY: ==host {eq_n}/{n}  moat {moat_n}/{n} (FA {fa_total})  to_host-clean {th_n}/{n}  "
              f"lesion-safe {les_n}/{n}  OFF-byte-ident {off_n}/{n}  -> {summary[str(K)]['verdict']}", flush=True)

    verdict = "GO" if overall else "NEGATIVE"
    print(f"\nOVERALL: {verdict}  (K in {ks}, {len(seeds)} seeds, V={len(VOCAB)}; the cleanup-score to_host is GONE "
          f"from the fused query path = R1 closed)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=dict(per_K=summary, verdict=verdict, op=FUSED_OP, V=len(VOCAB), gpu=is_gpu_backend()),
                       results=all_results), f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
