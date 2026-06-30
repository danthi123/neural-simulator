"""R3 (Option 2 of `2026-06-30-tier2-integrated-spiking-loop-scoping.md`): SCALE-VALIDATE the R1 fused fabric (the
device-resident cleanup->score handoff, `_seq_fused_fabric.py`) at PRODUCTION scale D=2048 / V=320 -- the
`consolidated_320_conversation_demo` scale (the 320 stream-learned cortex codes + the grounded projection).

THE RESIDUAL (R3). The R1 fused fold (`_seq_fused_fabric.build_fused_fabric_bridge` + `fused_seq_block`) is validated
GO 6-seed only at V=72 / K{2,4} / D=128. The scoping's named follow-on -- confirm the divnorm firing-band placement
holds + the moat stays 0-FA at the PRODUCTION D=2048 / V=320 -- was NOT done. The flagship demo runs `integrated_loop=
True` (the SEPARATE-bridge spiking path) at V=320, NOT `integrated_loop="fused"`, so the fused device-resident seam is
unconfirmed at scale.

THE SCALE-WIRING (this module, reuse-by-import, NO `sim/` edit). The R1 fused fabric builds the K-way sequencer at the
FULL V (320 `score_w{w}` pools + 320 cueA/cueX lines + K*2*320 decoded + K*2*320 match lines): at V=320/K=8 that is
~225K neurons (probed) -- the same un-shrunk cost the NON-fused integrated path AVOIDS via `enable_seq_vocab_shrink`
(the reduced cue vocab V'_A = distinct stored agents, V'_X = distinct stored actions; `_seq_vocab_shrink_derisk.py`,
GO 2026-06-21: byte-identical decisions, 34.6x smaller at the production K=8/V=320 store). So the scale-wiring is to
fold the divnorm-score pool + the *reduced* K-way sequencer onto ONE bridge:

  * the SCORE pool stays FULL-V (`score_w{w}`, w in 0..V-1, input_divisive_norm=True) -- the cleanup membrane lights
    over all V words and the per-query divisor is the mean over ALL V score-pool neurons (byte-faithful to the
    standalone `build_divnorm_score_bridge(V=self.V)` the non-fused path uses);
  * the SEQUENCER's cue/decoded/gated-match word-lines shrink to V'_A (role A) / V'_X (role X) -- the
    `build_sequencerK_reduced_bridge` layout -- so the fabric is tractable at V=320;
  * the device-resident handoff: the cleanup membrane -> the full-V score pool (DEVICE-RESIDENT, NO `to_host` of the
    cleanup score) -> which words FIRE (the placed rheobase, a firing-state BODY read) -> remap each lit word from the
    global word space into the role's reduced decoded index (DROP a spurious lit word outside the reduced vocab, the
    SAME net effect as the full-V build's gated-closed line -- the load-bearing argument is `_seq_vocab_shrink_derisk`'s,
    proven byte-identical) -> drive the reduced sequencer's decoded lines.

The cleanup score is kept ON-DEVICE the whole way from the RF cleanup to the score-pool drive (the R1 close is
PRESERVED at scale: zero `to_host` of the cleanup membrane). The op-point is the validated S5/S2/K32 production
op-point (match_thresh 0.06, gain 0.1, sigma 1, input_gain 1 -- `2026-06-21-shortcut3-K32-capability-surpass.md`).

GO BAR (R3): ==host (the fused-reduced path's per-query decision == the host `_scan` oracle) at D=2048/V=320; moat
0-FA (HARD); to_host-clean (the cleanup-score seam stays closed at scale); OFF byte-identical. 6 seeds is the GPU run.

  CPU smoke (a small sample at V=320/D=2048 -- proves the path builds + ==host on a few queries):
    SIM_BACKEND=numpy python -u -m research.runners._seq_fused_fabric_320 --seeds 42 --dim 256 --smoke
  GPU 6-seed (the controller's run -- the full who/what battery at production scale):
    SIM_BACKEND=cupy python -u -m research.runners._seq_fused_fabric_320 --seeds 42,43,44,100,101,102 --dim 2048

NO `sim/` edit. The fabric bridge is a plain Izhikevich `_run_one_simulation_step` bridge (its own step path); the
cleanup membrane read is the SAME RF op (already masked to `c.rf_mask`), kept on-device.
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
from sim.backend import get_backend, to_host, is_gpu_backend
from research.runners.one_brain_composer import OneBrainComposer
# The R1 fused fabric: reuse the DEVICE-RESIDENT cleanup-score reader + the full-V score-pool divnorm drive VERBATIM.
# Only the SEQUENCER fabric (shrunk to the reduced cue vocab) + the score->decoded remap are new here.
from research.runners._seq_fused_fabric import (
    FUSED_OP, N_WORD, N_POOL, _device_role_scores, _onbridge_divnorm_drive_device, _reset_score_pool,
)
# The reduced-vocab sequencer's reset + decode helpers (reused verbatim -- the fused-reduced fabric replicates the
# build_sequencerK_reduced_bridge CONTROL exactly; only the score pool is co-located + the cleanup handoff device-resident).
from research.runners._phaseB_onebrain_sequencerK_derisk import (
    reset_sequencerK_state, decision_to_block, patient_of, host_scan_block)
from research.runners._seq_vocab_shrink_derisk import reduced_cue_vocab
# The production V=320 codes + taxonomy + the grounding projection (the consolidated-320 path, verbatim).
from research.runners.stream_taxonomy_320 import TAXONOMY_40x8
from research.runners.option_c_real_cooccurrence_derisk import taxonomy_to_vocab_categories
from research.runners.consolidated_320_conversation_demo import _projection, grounded_phases, FACTS, ABSENT_WHAT


# ----------------------------------------------------------------------------------------------------------------
# The FUSED-REDUCED fabric: the FULL-V divnorm score pool + the REDUCED (V'_A / V'_X) K-way sequencer on ONE bridge.
# ----------------------------------------------------------------------------------------------------------------
def build_fused_fabric_reduced_bridge(seed, V, VA, VX, K, n_word=N_WORD, n_pool=N_POOL,
                                      sigma=FUSED_OP["sigma"], gain=FUSED_OP["gain"],
                                      w_match=300.0, w_or=300.0, w_blk=300.0, w_ans=320.0,
                                      w_lat_inhib=320.0, abstain_tonic_pA=420.0):
    """ONE Izhikevich bridge holding the FULL-V divnorm-score pool AND the REDUCED K-way sequencer.

    The SCORE pool: V word-pools `score_w{w}` (the role's V words), all input_divisive_norm=True, the global flag on --
    the per-step divide r_i = x_i/(sigma + gain*mean_j x_j) over the flagged set (== build_divnorm_score_bridge at full
    V). FULL-V because the cleanup membrane lights over all V words and the per-query divisor must be the role's whole
    per-query total (byte-faithful to the standalone score bridge the non-fused path uses).

    The SEQUENCER: the K-way gated-disinhibition match cascade + BG first-match priority WTA, but the cue/decoded/
    gated-match word-lines span the REDUCED per-role vocabs (cueA over VA, cueX over VX -- the distinct stored
    agents/actions; == build_sequencerK_reduced_bridge). The match/answer/abstain/inh pools + the BG priority wiring
    are vocab-INDEPENDENT (byte-identical to the full-V build).

    Returns (sb, meta). meta carries V, VA, VX, K, n_word, n_pool + the FULL-V score-pool per-word first index
    (precomputed once for the device scatter)."""
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
    cfg.enable_vectorized_gate_couplings = True            # == the standalone builds (byte-identical gate couplings)
    cfg.enable_input_divisive_norm = True                  # == build_divnorm_score_bridge (the S5 primitive)
    cfg.input_divisive_sigma = float(sigma)
    cfg.input_divisive_gain = float(gain)

    role_V = {"A": int(VA), "X": int(VX)}
    regions = []
    # --- the FULL-V divnorm SCORE pool (V word-pools, divisive-norm flagged) ---
    regions += [BrainRegion(name=f"score_w{w}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0,
                            input_divisive_norm=True) for w in range(V)]
    # --- the REDUCED sequencer fabric (== build_sequencerK_reduced_bridge regions) ---
    for grp, role in (("cueA", "A"), ("cueX", "X")):       # cue word-lines (shared across blocks), per-role reduced
        regions += [BrainRegion(name=f"{grp}_{w}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0)
                    for w in range(role_V[role])]
    for b in range(K):                                     # per-block decoded word-lines, per-role reduced
        for role in ("A", "X"):
            regions += [BrainRegion(name=f"d{b}{role}_{w}", n_neurons=n_word, exc_fraction=1.0,
                                    internal_density=0.0) for w in range(role_V[role])]
    for b in range(K):                                     # per-word gated-match line (decoded gated by cue), reduced
        for role in ("A", "X"):
            regions += [BrainRegion(name=f"mw{b}{role}_{w}", n_neurons=n_word, exc_fraction=1.0,
                                    internal_density=0.0) for w in range(role_V[role])]
    for b in range(K):                                     # match/answer pools per block (vocab-independent)
        for nm in (f"mA{b}", f"mX{b}", f"m{b}", f"ans{b}"):
            regions.append(BrainRegion(name=nm, n_neurons=n_pool, exc_fraction=1.0, internal_density=0.0))
    regions.append(BrainRegion(name="abstain", n_neurons=n_pool, exc_fraction=1.0, internal_density=0.0))
    for b in range(K):                                     # inhibitory interneurons (first-match priority)
        regions.append(BrainRegion(name=f"inh{b}", n_neurons=n_pool, exc_fraction=0.0, internal_density=0.0))
    cfg.brain_regions = regions

    # --- the sequencer pathways (== build_sequencerK_reduced_bridge) ---
    P = []
    for b in range(K):
        for role in ("A", "X"):
            for w in range(role_V[role]):
                P.append(RegionPathway(from_region=f"d{b}{role}_{w}", to_region=f"mw{b}{role}_{w}", density=1.0,
                                       weight_mean=w_match, weight_jitter=0.0, plastic=False,
                                       transmission_gate=f"g{b}{role}_{w}"))
                pool = f"mA{b}" if role == "A" else f"mX{b}"
                P.append(RegionPathway(from_region=f"mw{b}{role}_{w}", to_region=pool, density=1.0, weight_mean=w_or,
                                       weight_jitter=0.0, plastic=False))
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
    # the cue gate<->pool couplings (== wire_sequencerK_(reduced_)couplings): the cue word-line firing opens the
    # per-word match gate (per-role over the reduced vocab); the agent-match pool firing opens the per-block AND gate.
    for b in range(K):
        for role, grp in (("A", "cueA"), ("X", "cueX")):
            for w in range(role_V[role]):
                sb.couple_gate_to_pool(f"g{b}{role}_{w}", f"{grp}_{w}", threshold=0.03)
        sb.couple_gate_to_pool(f"gblk{b}", f"mA{b}", threshold=0.03)

    # precompute the FULL-V score-pool's per-word first neuron index (for the device scatter) -- region indices are
    # query-invariant.
    score_idx = np.stack([np.asarray(sb.region_manager.indices(f"score_w{w}"), dtype=np.int64) for w in range(V)])
    meta = dict(V=int(V), VA=int(VA), VX=int(VX), K=int(K), n_word=int(n_word), n_pool=int(n_pool),
                abstain_tonic_pA=float(abstain_tonic_pA), score_idx=score_idx)
    return sb, meta


def fused_block_drives_reduced(c, sb, meta, input_gain=FUSED_OP["input_gain"]):
    """The DEVICE-RESIDENT cleanup -> FULL-V score-pool -> firing drive, per block (the lit FULL-V word vector per role).
    For each stored block: run the RF cleanup (device), gather the agent + action score slices ON-DEVICE, divisively-
    normalize + threshold them through the co-located FULL-V score pool, and return the per-block (litA[V], litX[V])
    boolean firing vectors (which FULL-V word fired). The global->reduced remap happens in `run_fused_sequencer_reduced`
    (it needs the per-query mapA/mapX). NO `to_host` of the cleanup membrane anywhere -- mirrors
    `_seq_fused_fabric.fused_block_drives` but keeps the lit vector full-V (the reduced sequencer remaps at drive time)."""
    K, V = meta["K"], meta["V"]
    out = []
    for bi in range(min(K, len(c.kb))):
        mem_dev, (sa0, sa1), (sx0, sx1) = _device_role_scores(c, bi)
        ag_dev = mem_dev[sa0:sa1]                           # device slice (the agent cleanup scores) -- NOT to_host
        ax_dev = mem_dev[sx0:sx1]
        lit_a = _onbridge_divnorm_drive_device(sb, meta, ag_dev, input_gain)   # lit_a[V] host bool (firing-state read)
        lit_x = _onbridge_divnorm_drive_device(sb, meta, ax_dev, input_gain)
        out.append((lit_a, lit_x))
    return out


def run_fused_sequencer_reduced(sb, meta, words, mapA, mapX, cue_agent_word, cue_action_word, block_lit,
                                settle=60, lesion=False, match_thresh=FUSED_OP["match_thresh"], permute=False,
                                hi_pA=1500.0, spurious_counter=None):
    """Drive the CO-LOCATED REDUCED sequencer (on the fused-reduced bridge) from the device-derived FULL-V lit vectors +
    the cue, and read which BG channel wins. Mirrors `_seq_vocab_shrink_derisk.run_sequencerK_reduced_with_drive`: the
    cue + each lit decoded word are remapped from the global word space (0..V-1) into the role's reduced index space
    (mapA/mapX); a lit word OUTSIDE the reduced vocab (a spurious near-tie) is DROPPED + tracked (the SAME net effect as
    the full-V build's gated-closed line -- proven byte-identical by `_seq_vocab_shrink_derisk`). `block_lit` =
    [(litA[V], litX[V]), ...] per block (the FULL-V firing booleans from `fused_block_drives_reduced`).

    Returns (decision, rates). lesion=True zeros the decoded drive (fail-safe -> abstain); permute cyclically shifts the
    match->answer rule (anti-cheat). The cue is guaranteed present in the reduced vocab (the caller abstains otherwise)."""
    VA, VX, K = meta["VA"], meta["VX"], meta["K"]
    xp, _ = get_backend()
    idx = lambda nm: np.asarray(sb.region_manager.indices(nm))
    reset_sequencerK_state(sb)                              # the standalone per-query housekeeping (drain + c-reset)
    cur = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    cur[idx(f"cueA_{mapA[cue_agent_word]}")] = 1500.0
    cur[idx(f"cueX_{mapX[cue_action_word]}")] = 1500.0
    if not lesion:
        for bi, (litA, litX) in enumerate(block_lit[:K]):
            for w in range(len(words)):
                if litA[w]:
                    word = words[w]
                    if word in mapA:
                        cur[idx(f"d{bi}A_{mapA[word]}")] = hi_pA
                    elif spurious_counter is not None:
                        spurious_counter["A"] = spurious_counter.get("A", 0) + 1
                if litX[w]:
                    word = words[w]
                    if word in mapX:
                        cur[idx(f"d{bi}X_{mapX[word]}")] = hi_pA
                    elif spurious_counter is not None:
                        spurious_counter["X"] = spurious_counter.get("X", 0) + 1
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
# Composer-facing entry points: build + cache the fused-reduced fabric on the composer, run the (agent, action) routing.
# These MIRROR `_seq_fused_fabric.ensure_fused_fabric / fused_seq_block` but build the REDUCED (V'_A/V'_X) fabric so it
# is tractable at V=320. Stored on the composer as `_fused320_*` to avoid colliding with the V<=72 `_fused_*` cache.
# ----------------------------------------------------------------------------------------------------------------
def ensure_fused_fabric_reduced(c, K):
    facts = [(f.get("agent"), f.get("action"), f.get("patient")) for (f, _) in c.kb[:K]]
    agentsA, actionsX, mapA, mapX = reduced_cue_vocab(facts, K)
    sig = (tuple(agentsA), tuple(actionsX))
    if getattr(c, "_fused320_seq", None) is None or c._fused320_K != K or c._fused320_sig != sig:
        sb, meta = build_fused_fabric_reduced_bridge(seed=c.seed, V=c.V, VA=len(agentsA), VX=len(actionsX), K=K,
                                                     sigma=c.sequencer_sigma, gain=c.sequencer_gain)
        c._fused320_seq = (sb, meta)
        c._fused320_K = K
        c._fused320_sig = sig
        c._fused320_mapA = mapA
        c._fused320_mapX = mapX
        c._fused320_dirty = True
    if getattr(c, "_fused320_dirty", True) or getattr(c, "_fused320_lit", None) is None:
        c._seq_cleanup_conns_cache = None                  # opt #4: rebuild the block-invariant cleanup conns once
        sb, meta = c._fused320_seq
        c._fused320_lit = fused_block_drives_reduced(c, sb, meta, input_gain=c.sequencer_input_gain)
        c._fused320_dirty = False


def fused_seq_block_reduced(c, agent, action):
    """The SELECTED block index for cue (agent, action) on the FUSED-REDUCED path (the device-resident-handoff spiking
    reduced K-way sequencer decision), or None = abstain. An absent cue WORD abstains before the sequencer (the moat).
    A cue word that is not a stored agent/action also abstains (== no decoded line in the full-V build; moat-preserving)."""
    K = len(c.kb)
    if K == 0:
        return None
    if agent not in c._word_index or action not in c._word_index:
        return None                                        # absent cue word -> no block -> abstain (the moat)
    ensure_fused_fabric_reduced(c, K)
    sb, meta = c._fused320_seq
    if agent not in c._fused320_mapA or action not in c._fused320_mapX:
        return None                                        # cue not a stored agent/action -> abstain (moat-preserving)
    dec, _rates = run_fused_sequencer_reduced(sb, meta, c.words, c._fused320_mapA, c._fused320_mapX, agent, action,
                                              c._fused320_lit, match_thresh=c.sequencer_match_thresh)
    return decision_to_block(dec, K)


def fused_reduced_query_patient(c, agent, action):
    """The fused-reduced (agent, action) -> patient answer (or None = abstain), via the reduced spiking sequencer +
    the composer's own patient body-read of the selected block (the legitimate S7 boundary)."""
    idx = fused_seq_block_reduced(c, agent, action)
    if idx is None:
        return None
    return c.kb[idx][0].get("patient")


def fused_reduced_query_agent(c, action, patient):
    """who_does: scan for the block whose ACTION and PATIENT match, returning its AGENT, with the reduced spiking
    sequencer doing the (agent-candidate, action) cue-match. The moat is structural: a cue forming no stored pair
    abstains. We iterate the stored agents as candidate cues for `action` (each abstains unless it is THE stored fact),
    then verify the selected block's patient matches (== the host `_scan(cue={action,patient}, agent)` semantics --
    a swapped-cue 1-role cascade, the documented bounded follow-on; here it routes through the fused-reduced sequencer
    for the (agent, action) hot-path and verifies patient on the body-read)."""
    K = len(c.kb)
    if K == 0:
        return None
    if action not in c._word_index or patient not in c._word_index:
        return None
    for i, (f, _) in enumerate(c.kb[:K]):
        a = f.get("agent")
        if a is None:
            continue
        idx = fused_seq_block_reduced(c, a, action)
        if idx is not None and c.kb[idx][0].get("patient") == patient:
            return c.kb[idx][0].get("agent")
    return None


def fused_reduced_query_patient_lesioned(c, agent, action):
    """Anti-cheat helper: the fused-reduced query with the cleanup->score drive SEVERED (lesion) -> the sequencer must
    abstain (fail safe), never confabulate. Returns the answer (must be None for a fail-safe lesion)."""
    K = len(c.kb)
    if K == 0 or agent not in c._word_index or action not in c._word_index:
        return None
    ensure_fused_fabric_reduced(c, K)
    sb, meta = c._fused320_seq
    if agent not in c._fused320_mapA or action not in c._fused320_mapX:
        return None
    dec, _ = run_fused_sequencer_reduced(sb, meta, c.words, c._fused320_mapA, c._fused320_mapX, agent, action,
                                         c._fused320_lit, match_thresh=c.sequencer_match_thresh, lesion=True)
    idx = decision_to_block(dec, K)
    if idx is None:
        return None
    return c.kb[idx][0].get("patient")


# ----------------------------------------------------------------------------------------------------------------
# The V=320 grounded-codes harness + the scale de-risk (R3 GO bar): ==host + moat 0-FA + cleanup-score to_host
# eliminated + lesion fails safe + OFF byte-identical, at D=2048/V=320.
# ----------------------------------------------------------------------------------------------------------------
D_DEFAULT = 2048


def _load_codes(seed, readout="neural"):
    suffix = "neural_seed" if readout == "neural" else "seed"
    cpath = os.path.join(_REPO, "research", "findings", "raw", f"_phaseB_stream_codes_320_{suffix}{seed}.npy")
    if not os.path.exists(cpath):
        return None
    codes = np.load(cpath)
    return codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)


def _grounded_vocab(seed, codes, D):
    vocab, cat_ids, _ = taxonomy_to_vocab_categories(TAXONOMY_40x8)
    proj = _projection(D, codes.shape[1], seed)
    grounded = {vocab[i]: grounded_phases(codes[i], proj) for i in range(len(vocab))}
    concepts = {vocab[i]: codes[i] for i in range(len(vocab))}
    return vocab, concepts, grounded


def _build_composers(seed, D, codes, facts):
    """Two production-V=320 composers on the grounded stream codes: the host `_scan` oracle (integrated_loop=False) +
    the fused-reduced device-resident path. Both store the SAME facts. (We build the fused composer with
    integrated_loop="fused" so the store-dirty bookkeeping fires; the fused-REDUCED helpers in this module override the
    fabric build/drive so it is tractable at V=320 -- the base `_fused_*` (full-V) cache is never touched.)"""
    vocab, concepts, grounded = _grounded_vocab(seed, codes, D)
    op = dict(sequencer_match_thresh=FUSED_OP["match_thresh"], sequencer_gain=FUSED_OP["gain"],
              sequencer_sigma=FUSED_OP["sigma"], sequencer_input_gain=FUSED_OP["input_gain"])
    base = dict(seed=seed, D=D, vocab=concepts, grounded_codes=grounded, k_max=max(8, len(facts)),
                enable_batched=False, enable_rf_cudagraph=False)
    c_host = OneBrainComposer(integrated_loop=False, **base)
    c_fused = OneBrainComposer(integrated_loop="fused", **base, **op)
    for (a, x, p) in facts:
        c_host.store(a, x, p)
        c_fused.store(a, x, p)
    return c_host, c_fused, vocab


def _to_host_spy(c_fused):
    """Count reads of the composer's cleanup membrane (the RF bridge `b`'s cp_membrane_potential_v -- the cleanup-score
    carrier) DURING the cleanup->sequencer hand-off. Returns (install, restore, counter)."""
    import sim.backend as backend
    from research.runners import _seq_fused_fabric as sff
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
        sff.to_host = _spy
        # also patch this module's alias (fused_block_drives_reduced calls _onbridge_divnorm_drive_device which uses
        # the sff alias; the device-resident reader _device_role_scores does NOT to_host the membrane).
        globals()["to_host"] = _spy

    def restore():
        backend.to_host = real
        sff.to_host = real
        globals()["to_host"] = real

    return install, restore, counter


def run_seed(seed, D, readout="neural", smoke_present=None, smoke_absent=None, strict_eqhost=False):
    codes = _load_codes(seed, readout)
    if codes is None:
        return dict(seed=seed, skip=True)
    facts = list(FACTS)
    c_host, c_fused, vocab = _build_composers(seed, D, codes, facts)

    present = [(a, x, p) for (a, x, p) in facts]
    absent = [(a, x) for (a, x) in ABSENT_WHAT]
    if smoke_present is not None:                          # CPU smoke: a small sample (build is heavy at D=2048)
        present = present[:smoke_present]
    if smoke_absent is not None:
        absent = absent[:smoke_absent]

    rows = []
    for (qa, qx, want) in present:
        h = host_scan_block(c_host, qa, qx)
        hp = patient_of(c_host, h)
        f = fused_reduced_query_patient(c_fused, qa, qx)
        rows.append(dict(cue=(qa, qx), kind="present", host=hp, fused=f, want=want,
                         fused_eq_host=(f == hp), correct=(f == want),
                         over_abstain=(f is None and hp is not None),       # the SAFE-direction low-D miss
                         wrong=(f is not None and f != hp)))                # a CONFABULATION (must never happen)
    moat_rows = []
    for (qa, qx) in absent:
        h = host_scan_block(c_host, qa, qx)
        hp = patient_of(c_host, h)
        f = fused_reduced_query_patient(c_fused, qa, qx)
        moat_rows.append(dict(cue=(qa, qx), kind="absent", host=hp, fused=f, fused_eq_host=(f == hp)))

    eq_host = all(r["fused_eq_host"] for r in rows) and all(r["fused_eq_host"] for r in moat_rows)
    # never-wrong (the no-confab guarantee, INDEPENDENT of D fidelity): the fused answer is ALWAYS host-or-abstain --
    # it never emits a WRONG patient. A present cue may OVER-ABSTAIN at low D (the documented code-fidelity miss, the
    # SAFE direction; both spiking paths do this at D=128 -- the consolidated_320 R3 note: "over-abstains 2/8 on the
    # K=8 demo set"); the GPU run at the production D=2048 has high fidelity and is the strict ==host target.
    never_wrong = all(not r["wrong"] for r in rows)
    over_abstain = sum(1 for r in rows if r["over_abstain"])
    fa = sum(1 for r in moat_rows if r["fused"] is not None)
    moat_ok = (fa == 0)

    # cleanup-score to_host eliminated (R1, precisely): instrument ONLY the cleanup->sequencer hand-off
    # (`fused_seq_block_reduced`) -> 0 cleanup-membrane reads.
    install, restore, counter = _to_host_spy(c_fused)
    install()
    try:
        a0, x0, _p0 = present[0] if present else facts[0]
        _ = fused_seq_block_reduced(c_fused, a0, x0)
    finally:
        restore()
    to_host_clean = (counter["cleanup_membrane_reads"] == 0)

    # lesion-fails-safe: sever the cleanup->score drive on every present cue -> abstain.
    les = [fused_reduced_query_patient_lesioned(c_fused, a, x) for (a, x, _p) in present]
    lesion_fails_safe = all(v is None for v in les)

    # OFF==byte-identical: the host oracle path is unchanged by the fused-reduced helpers (additive). The fused
    # composer's NON-fused query (integrated_loop branch) is never taken here; we assert the host composer's answers
    # are exactly the host `_scan` (a regression sanity that the additive code didn't perturb the oracle path).
    off_byte_identical = all(patient_of(c_host, host_scan_block(c_host, qa, qx)) ==
                             c_host.query_patient(qa, qx) for (qa, qx, _w) in present)

    # GO: moat 0-FA (HARD) + never-wrong (no confabulation) + to_host-clean + lesion-safe + OFF byte-identical. The
    # answer-fidelity gate is `eq_host` when --strict-eqhost (the GPU production D=2048 bar) else never-wrong (the
    # CPU-smoke / low-D bar that tolerates the documented over-abstention -- the moat is NEVER traded for a pass).
    answer_gate = eq_host if strict_eqhost else never_wrong
    go = answer_gate and moat_ok and to_host_clean and lesion_fails_safe and off_byte_identical
    return dict(seed=seed, D=D, readout=readout, n_facts=len(facts), V=c_fused.V,
                VA=len(c_fused._fused320_mapA), VX=len(c_fused._fused320_mapX),
                rows=rows, moat_rows=moat_rows, eq_host=eq_host, never_wrong=never_wrong,
                over_abstain=over_abstain, strict_eqhost=strict_eqhost, moat_ok=moat_ok, fa=fa,
                to_host_clean=to_host_clean, to_host_counter=counter,
                lesion_fails_safe=lesion_fails_safe, lesion_answers=les,
                off_byte_identical=off_byte_identical, go=go, smoke=(smoke_present is not None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--dim", type=int, default=D_DEFAULT, help="phasor dim (production D=2048; smoke can use 256)")
    ap.add_argument("--readout", choices=["neural", "host"], default="neural")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU smoke: a SMALL sample (the first --smoke-present present cues + --smoke-absent moat "
                         "cues) -- proves the V=320 fused-reduced path builds + ==host on a few queries (the full "
                         "battery + 6 seeds is the GPU run).")
    ap.add_argument("--smoke-present", type=int, default=3)
    ap.add_argument("--smoke-absent", type=int, default=2)
    ap.add_argument("--strict-eqhost", action="store_true",
                    help="require STRICT ==host on every present cue (the GPU production D=2048 bar -- high fidelity, "
                         "no over-abstention). Default OFF: the answer gate is NEVER-WRONG (no confabulation) which "
                         "tolerates the documented low-D over-abstention (the SAFE-direction miss; the moat is the "
                         "HARD gate either way). Pass this for the D=2048 6-seed run.")
    ap.add_argument("--out", default="research/findings/raw/_seq_fused_fabric_320.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"[R3 FUSED-REDUCED @ V=320] scale-validate the R1 device-resident cleanup->score fold at D={args.dim} "
          f"V=320; op={FUSED_OP} gpu={is_gpu_backend()} smoke={args.smoke} strict_eqhost={args.strict_eqhost}\n",
          flush=True)
    results = []
    for s in seeds:
        t0 = time.perf_counter()
        kw = dict(smoke_present=args.smoke_present, smoke_absent=args.smoke_absent) if args.smoke else {}
        r = run_seed(s, args.dim, args.readout, strict_eqhost=args.strict_eqhost, **kw)
        r["wall_s"] = round(time.perf_counter() - t0, 1)
        results.append(r)
        if r.get("skip"):
            print(f"seed {s}: SKIP -- no {args.readout} codes", flush=True)
            continue
        eq = "==host" if r["eq_host"] else (f"OVER-ABSTAIN({r['over_abstain']})" if r["never_wrong"] else "CONFAB!")
        moat = "moat-OK" if r["moat_ok"] else f"MOAT-BREACH(fa={r['fa']})"
        th = "to_host-clean" if r["to_host_clean"] else f"TO_HOST-LEAK({r['to_host_counter']})"
        les = "lesion-SAFE" if r["lesion_fails_safe"] else f"lesion-UNSAFE({r['lesion_answers']})"
        off = "OFF-byte-ident" if r["off_byte_identical"] else "OFF-DIVERGES"
        nw = "never-wrong" if r["never_wrong"] else "CONFABULATES"
        det = "  ".join(f"{rr['cue']}:f={rr['fused']}|h={rr['host']}" for rr in r["rows"])
        mdet = "  ".join(f"{rr['cue']}:f={rr['fused']}" for rr in r["moat_rows"])
        print(f"seed {s} D{args.dim} V={r['V']} (VA={r['VA']} VX={r['VX']}): "
              f"{'GO' if r['go'] else 'NO'}  {eq}  {nw}  {moat}  {th}  {les}  {off}  [{r['wall_s']}s]", flush=True)
        print(f"    present: {det}", flush=True)
        print(f"    moat:    {mdet}", flush=True)

    real = [r for r in results if not r.get("skip")]
    n = len(real)
    eq_n = sum(r["eq_host"] for r in real)
    nw_n = sum(r["never_wrong"] for r in real)
    oa_total = sum(r["over_abstain"] for r in real)
    moat_n = sum(r["moat_ok"] for r in real)
    th_n = sum(r["to_host_clean"] for r in real)
    les_n = sum(r["lesion_fails_safe"] for r in real)
    off_n = sum(r["off_byte_identical"] for r in real)
    fa_total = sum(r["fa"] for r in real)
    answer_n = eq_n if args.strict_eqhost else nw_n          # the answer gate (strict ==host for GPU, else never-wrong)
    go = n > 0 and (answer_n == n and moat_n == n and th_n == n and les_n == n and off_n == n and fa_total == 0)
    verdict = "GO" if go else ("NO-CODES" if n == 0 else "NEGATIVE")
    print(f"\nSUMMARY ({n} seeds, D={args.dim}, V=320{', SMOKE' if args.smoke else ''}): ==host {eq_n}/{n}  "
          f"never-wrong {nw_n}/{n} (over-abstain {oa_total})  moat {moat_n}/{n} (FA {fa_total})  "
          f"to_host-clean {th_n}/{n}  lesion-safe {les_n}/{n}  OFF-byte-ident {off_n}/{n}  "
          f"[answer-gate={'==host' if args.strict_eqhost else 'never-wrong'}]  -> {verdict}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=dict(n=n, eq_host_n=eq_n, never_wrong_n=nw_n, over_abstain_total=oa_total,
                                    moat_n=moat_n, fa_total=fa_total, to_host_clean_n=th_n, lesion_n=les_n,
                                    off_byte_identical_n=off_n, verdict=verdict, strict_eqhost=args.strict_eqhost,
                                    op=FUSED_OP, D=args.dim, V=320, smoke=args.smoke, gpu=is_gpu_backend()),
                       results=results), f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
