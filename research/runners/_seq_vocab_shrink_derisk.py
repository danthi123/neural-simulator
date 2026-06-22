"""Audit opportunity #2 (the cheapest sub-lever of the 2026-06-22 megakernel-revisit audit, §5 row #2 "shrink the
fabric"): build the K-way on-substrate SEQUENCER at a REDUCED CUE vocab (only the words that can actually appear as a
stored agent / action), not the composer's full V -- ANSWER-IDENTICAL, and far smaller.

WHY IT IS ANSWER-IDENTICAL (the load-bearing argument, verified empirically by this de-risk):
  The sequencer's per-word match gate g{b}A_{w} is opened ONLY by the CUE word-line cueA_{w} firing
  (`wire_sequencerK_couplings`: couple_gate_to_pool(f"g{b}A_{w}", f"cueA_{w}")), and a match requires BOTH the cue
  line AND the block's DECODED line on the SAME word w. The decoded agent of every block is one of the K stored
  agents; the decoded action one of the K stored actions. So:
    * a word that is NOT a stored agent can never be a block's decoded agent -> its agent match line never fires ->
      it is dead weight in the agent role;
    * likewise the action role and the stored actions.
  Therefore building the sequencer over only V'_A = {distinct stored agents} (role A) and V'_X = {distinct stored
  actions} (role X), with the cue index + the decoded-line drives REMAPPED from the global word index (0..V-1) into
  the reduced V'_A / V'_X spaces, yields the IDENTICAL spiking match cascade -- no gate that the full-V build would
  open is removed, and no match that the full-V build would make is lost. A query cue (agent, action): if `agent` is
  not in V'_A OR `action` is not in V'_X, NO block can match -> abstain immediately (a global no-op, == the full-V
  sequencer which would also find no decoded line for it -> abstain). Else run the shrunk sequencer with the cue
  remapped + the per-block decoded drive restricted to w in V'_A / V'_X (the cleanup score at every kept cue index is
  identical, so the match COMPARISON is byte-identical).

The reduction at production V=320, K=32 (8 distinct actions, 32 distinct agents): role-A word-lines 320 -> 32, role-X
320 -> 8, so the cue + decoded + gated-match fabric collapses from ~837K neurons to a tiny build (see the printed
neuron counts). NO `sim/` edit (reuse-by-import: the S0 K-way sequencer builder/wiring/reset/production-rule + the
composer cleanup); this de-risk ONLY adds a reduced-vocab BUILDER + a remapping RUNNER and asserts byte-identical
DECISIONS vs the full-V sequencer on the FULL battery (every present cue, three moat cues, the permuted anti-cheat,
the sequencer-lesion, and the degenerate per-block-priority store).

GO BAR (this de-risk): the block DECISION (`decision_to_block`) is IDENTICAL full-V vs shrunk for EVERY case
(present / absent-agent / absent-action / cross / lesion / permute / priority), multi-seed. A single divergence ->
HONEST NEGATIVE (report exactly which case + why; do NOT weaken any assertion). Report the neuron-count reduction +
a wall-clock timing of the 80-step run both ways at a representative scale.

  SIM_BACKEND=numpy  python -u -m research.runners._seq_vocab_shrink_derisk --seeds 42,43,44 --dim 64 --ks 8
  SIM_BACKEND=cupy CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u -m research.runners._seq_vocab_shrink_derisk \
      --seeds 42,43,44 --dim 128 --ks 8,32 --time
"""
from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.backend import to_host, from_host, is_gpu_backend
from research.runners.one_brain_composer import OneBrainComposer
# Reuse the op-result reader + the full-V sequencer (builder / reset / production-rule + the anti-cheat helpers +
# the canonical fact/vocab tables) VERBATIM -- this de-risk only ADDS the reduced-vocab build + remap and compares.
from research.runners._phaseB_onebrain_sequencer_derisk import block_cleanup_scores, scores_to_drive
from research.runners._phaseB_onebrain_sequencerK_derisk import (
    build_sequencerK_bridge, run_sequencerK, reset_sequencerK_state,
    host_scan_block, decision_to_block, patient_of,
    ALL_FACTS as ALL_FACTS_SMALL, VOCAB as VOCAB_SMALL,
)
# The S2 PRODUCTION-representative table: 32 distinct facts, V=72, 8 actions each shared by 4 agents (the maximal
# shared-action routing stress -- exactly the production K=32 / 8-distinct-action shape the audit names). Used for
# K>16 so the de-risk exercises the real production scale (the small S0 table only reaches K=16 / V=22).
from research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk import (
    ALL_FACTS as ALL_FACTS_BIG, VOCAB as VOCAB_BIG,
)


def _facts_vocab_for_K(K):
    """The small S0 table (16 facts, V=22) for K<=16; the S2 production table (32 facts, V=72, 8 shared actions) for
    K>16. The de-risk runs the larger table whenever it is needed to reach K, so K=32 is the true production scale."""
    if K <= 16:
        return ALL_FACTS_SMALL, VOCAB_SMALL
    return ALL_FACTS_BIG, VOCAB_BIG


def _build_queries(facts, vocab):
    """The query set for a K-fact store: every PRESENT cue (each answers ITS block, so the scan must reach the LAST
    block) + THREE moat cues (absent agent / absent action / cross = agent of fact0 + an action fact0 does NOT have,
    that forms no stored pair). Local copy of the S0/S2 helper, parameterized by `vocab` so it works for both tables."""
    queries = [((a, x), f"blk{i}-present") for i, (a, x, p) in enumerate(facts)]
    agents = {a for (a, x, p) in facts}
    actions = {x for (a, x, p) in facts}
    pairs = {(a, x) for (a, x, p) in facts}
    absent_agent = next((w for w in vocab if w not in agents), "zzz")
    absent_action = next((w for w in vocab if w not in actions), "zzz")
    a0, x0 = facts[0][0], facts[0][1]
    cross_action = next((x for (a, x, p) in facts if (a0, x) not in pairs), absent_action)
    queries += [((absent_agent, x0), "absent-agent"), ((a0, absent_action), "absent-action"),
                ((a0, cross_action), "cross-no-block")]
    return queries


# ----------------------------------------------------------------------------------------------------------------
# The REDUCED-vocab sequencer: identical CONTROL fabric to build_sequencerK_bridge, but the cue + decoded + gated-
# match word-lines span SEPARATE per-role reduced vocabs (VA for role A, VX for role X) instead of the shared full V.
# The match/answer/abstain/inh pools + the BG priority wiring are vocab-INDEPENDENT, so they are byte-identical.
# ----------------------------------------------------------------------------------------------------------------
def build_sequencerK_reduced_bridge(seed, VA, VX, K, n_word=20, n_pool=30,
                                    w_match=300.0, w_or=300.0, w_blk=300.0, w_ans=320.0, w_lat_inhib=320.0,
                                    abstain_tonic_pA=420.0):
    """build_sequencerK_bridge with role A over VA words and role X over VX words (the distinct stored agents/actions),
    everything else IDENTICAL. The per-role word loops use the role's own reduced size; the cue/decoded/gated-match
    indices for a kept word are remapped UPSTREAM by the caller (run_sequencerK_reduced)."""
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
    cfg.enable_vectorized_gate_couplings = True            # same perf flag as the full-V build (byte-identical)

    role_V = {"A": int(VA), "X": int(VX)}
    regions = []
    # cue word-lines (shared across blocks), per role over the role's reduced vocab
    for grp, role in (("cueA", "A"), ("cueX", "X")):
        regions += [BrainRegion(name=f"{grp}_{w}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0)
                    for w in range(role_V[role])]
    # per-block decoded word-lines, per role over the role's reduced vocab
    for b in range(K):
        for role in ("A", "X"):
            regions += [BrainRegion(name=f"d{b}{role}_{w}", n_neurons=n_word, exc_fraction=1.0,
                                    internal_density=0.0) for w in range(role_V[role])]
    # per-word gated-match line (decoded gated by cue), per block, per role over the role's reduced vocab
    for b in range(K):
        for role in ("A", "X"):
            regions += [BrainRegion(name=f"mw{b}{role}_{w}", n_neurons=n_word, exc_fraction=1.0,
                                    internal_density=0.0) for w in range(role_V[role])]
    # match/answer pools per block + the single abstain channel (vocab-independent -- identical to the full-V build)
    for b in range(K):
        for nm in (f"mA{b}", f"mX{b}", f"m{b}", f"ans{b}"):
            regions.append(BrainRegion(name=nm, n_neurons=n_pool, exc_fraction=1.0, internal_density=0.0))
    regions.append(BrainRegion(name="abstain", n_neurons=n_pool, exc_fraction=1.0, internal_density=0.0))
    for b in range(K):
        regions.append(BrainRegion(name=f"inh{b}", n_neurons=n_pool, exc_fraction=0.0, internal_density=0.0))
    cfg.brain_regions = regions

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
    # the gate<->cue couplings, per role over the reduced vocab (identical wiring to wire_sequencerK_couplings)
    for b in range(K):
        for role, grp in (("A", "cueA"), ("X", "cueX")):
            for w in range(role_V[role]):
                sb.couple_gate_to_pool(f"g{b}{role}_{w}", f"{grp}_{w}", threshold=0.03)
        sb.couple_gate_to_pool(f"gblk{b}", f"mA{b}", threshold=0.03)
    meta = dict(VA=int(VA), VX=int(VX), K=int(K), n_word=n_word, n_pool=n_pool, abstain_tonic_pA=abstain_tonic_pA,
                permute_default=False)
    return sb, meta


def reduced_cue_vocab(facts, K):
    """V'_A / V'_X: the distinct stored agents / actions across the first K facts, with a STABLE order (first
    appearance) so the reduced index is deterministic. Returns (agentsA, actionsX, mapA, mapX) where map* are
    word -> reduced-index dicts."""
    fk = facts[:K]
    agentsA, actionsX = [], []
    for (a, x, p) in fk:
        if a not in agentsA:
            agentsA.append(a)
        if x not in actionsX:
            actionsX.append(x)
    mapA = {w: i for i, w in enumerate(agentsA)}
    mapX = {w: i for i, w in enumerate(actionsX)}
    return agentsA, actionsX, mapA, mapX


def run_sequencerK_reduced(sb, meta, words, mapA, mapX, cue_agent_word, cue_action_word, blocks_scores,
                           settle=60, lesion=False, match_thresh=0.15, permute=False, drive_frac=0.9,
                           spurious_counter=None):
    """One who/what scan on the SHRUNK sequencer. Drives the decoded lines from the SAME host `scores_to_drive(...,
    drive_frac)` the full-V build uses (NOT a separately-recomputed argmax) -- the ONLY change vs S0's run_sequencerK
    is the word-line layout: the cue + each lit decoded word are remapped from the global word space (0..V-1) into the
    role's reduced index space (mapA / mapX).

    THE SPURIOUS-LIT SUBTLETY (why a lit decoded word can be absent from the reduced vocab): at imperfect fidelity
    (small D) a block's cleanup can light, within drive_frac of the peak, a NEAR-TIE word that is NOT this block's
    stored role-filler (e.g. a D=64 block-7 action cleanup lit `river` alongside `fly`). In the FULL-V build that
    spurious line is HARMLESS for every battery cue because its per-word gate g{b}{role}_{spur} is opened only by the
    CUE word cue{role}_{spur} firing -- and no battery cue's role-word IS that spurious word (the present cues use the
    stored fillers; the moat cues use words guaranteed not to pair). So the spurious line is gated CLOSED and never
    matches. In the REDUCED build that word simply has no line (it is not a stored agent/action), which is the SAME
    net effect (a closed/absent line on a word no battery cue drives). We therefore DROP a spurious lit word here
    (tracking it via `spurious_counter`) rather than crash -- and the load-bearing claim is then PROVEN EMPIRICALLY by
    the battery comparison: if dropping it ever flipped a decision vs the full-V build, the `identical` check is False
    and the de-risk reports NEGATIVE. The cue is guaranteed present in the reduced vocab (the caller abstains else).
    Channel semantics IDENTICAL to S0; match COMPARISON + production rule are S0's, verbatim."""
    VA, VX, K = meta["VA"], meta["VX"], meta["K"]
    idx = lambda nm: np.asarray(sb.region_manager.indices(nm))
    reset_sequencerK_state(sb)
    cur = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    cur[idx(f"cueA_{mapA[cue_agent_word]}")] = 1500.0
    cur[idx(f"cueX_{mapX[cue_action_word]}")] = 1500.0
    if not lesion:
        for bi, (ag, ax) in enumerate(blocks_scores[:K]):
            dA = scores_to_drive(ag, frac=drive_frac)     # IDENTICAL drive computation to the full-V run_sequencerK
            dX = scores_to_drive(ax, frac=drive_frac)
            for w in range(len(words)):
                if dA[w] > 0:
                    word = words[w]
                    if word in mapA:
                        cur[idx(f"d{bi}A_{mapA[word]}")] = dA[w]
                    elif spurious_counter is not None:
                        spurious_counter["A"] = spurious_counter.get("A", 0) + 1
                if dX[w] > 0:
                    word = words[w]
                    if word in mapX:
                        cur[idx(f"d{bi}X_{mapX[word]}")] = dX[w]
                    elif spurious_counter is not None:
                        spurious_counter["X"] = spurious_counter.get("X", 0) + 1
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    cur = from_host(cur)
    for _ in range(settle):
        sb.cp_external_input_current[:] = cur
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    sb.cp_external_input_current[:] = 0.0
    m_rates = [acc[idx(f"m{b}")].mean() / settle for b in range(K)]
    fired = [r > match_thresh for r in m_rates]
    rates = {f"m{b}": round(m_rates[b], 3) for b in range(K)}
    winner = next((b for b in range(K) if fired[b]), None)
    if winner is None:
        decision = "abstain"
    else:
        decision = f"ans{(winner + 1) % K}" if permute else f"ans{winner}"
    rates["winner"] = winner
    return decision, rates


def run_sequencerK_reduced_with_drive(sb, meta, words, mapA, mapX, cue_agent_word, cue_action_word, block_drives,
                                      settle=60, lesion=False, match_thresh=0.15, permute=False, spurious_counter=None):
    """STEP-2 PRODUCTION variant: the SHRUNK sequencer driven from the composer's ON-BRIDGE divnorm `block_drives`
    (the same `make_block_drives` output the full-V `run_sequencerK_with_drive` consumes), remapped global->reduced.
    Mirrors `_phaseB_onebrain_sequencerK_divnorm_derisk.run_sequencerK_with_drive` exactly, EXCEPT the word-line
    layout: the cue + each lit decoded word are remapped into the role's reduced index space (mapA/mapX), and a lit
    decoded word outside the reduced vocab (a spurious near-tie -- see run_sequencerK_reduced) is dropped + tracked.
    `block_drives` = [(dA[V], dX[V]), ...] per block (global word-indexed, the divnorm firing). NO `scores_to_drive`
    / `s.max()` here (== the divnorm drive path). Channel semantics IDENTICAL to the full-V divnorm runner. The cue is
    guaranteed present in the reduced vocab (the caller abstains otherwise)."""
    VA, VX, K = meta["VA"], meta["VX"], meta["K"]
    idx = lambda nm: np.asarray(sb.region_manager.indices(nm))
    reset_sequencerK_state(sb)
    cur = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    cur[idx(f"cueA_{mapA[cue_agent_word]}")] = 1500.0
    cur[idx(f"cueX_{mapX[cue_action_word]}")] = 1500.0
    if not lesion:
        for bi, (dA, dX) in enumerate(block_drives[:K]):
            for w in range(len(words)):
                if dA[w] > 0:
                    word = words[w]
                    if word in mapA:
                        cur[idx(f"d{bi}A_{mapA[word]}")] = dA[w]
                    elif spurious_counter is not None:
                        spurious_counter["A"] = spurious_counter.get("A", 0) + 1
                if dX[w] > 0:
                    word = words[w]
                    if word in mapX:
                        cur[idx(f"d{bi}X_{mapX[word]}")] = dX[w]
                    elif spurious_counter is not None:
                        spurious_counter["X"] = spurious_counter.get("X", 0) + 1
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    cur = from_host(cur)
    for _ in range(settle):
        sb.cp_external_input_current[:] = cur
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    sb.cp_external_input_current[:] = 0.0
    m_rates = [acc[idx(f"m{b}")].mean() / settle for b in range(K)]
    fired = [r > match_thresh for r in m_rates]
    rates = {f"m{b}": round(m_rates[b], 3) for b in range(K)}
    winner = next((b for b in range(K) if fired[b]), None)
    decision = "abstain" if winner is None else (f"ans{(winner + 1) % K}" if permute else f"ans{winner}")
    rates["winner"] = winner
    return decision, rates


def run_seed_K(seed, D, K, do_time=False):
    """Build the composer + K facts, compute each block's cleanup scores, then run the FULL battery on BOTH the full-V
    sequencer (S0's run_sequencerK with scores_to_drive) and the SHRUNK sequencer (run_sequencerK_reduced over the
    reduced cue vocab), and assert the block DECISION is identical for every case."""
    all_facts, vocab = _facts_vocab_for_K(K)
    facts = all_facts[:K]
    c = OneBrainComposer(seed=seed, D=D, vocab=vocab, k_max=max(8, K), enable_batched=False, enable_rf_cudagraph=False)
    for (a, x, p) in facts:
        c.store(a, x, p)
    V = c.V
    word_idx = {w: i for i, w in enumerate(c.words)}
    bscores = [block_cleanup_scores(c, b) for b in range(K)]    # the op RESULTS (cleanup scores per block)

    # full-V sequencer (the reference)
    sb_full, meta_full = build_sequencerK_bridge(seed=seed, V=V, K=K)
    # shrunk sequencer (reduced cue vocab)
    agentsA, actionsX, mapA, mapX = reduced_cue_vocab(facts, K)
    VA, VX = len(agentsA), len(actionsX)
    sb_red, meta_red = build_sequencerK_reduced_bridge(seed=seed, VA=VA, VX=VX, K=K)

    n_full = sb_full.core_config.num_neurons
    n_red = sb_red.core_config.num_neurons

    spurious = {}                                          # count decoded words dropped by the shrink (must not flip a decision)
    queries = _build_queries(facts, vocab)
    rows = []
    for (qa, qx), kind in queries:
        ca, cx = word_idx[qa], word_idx[qx]
        host_blk = host_scan_block(c, qa, qx)
        # full-V decision
        dec_f, rf = run_sequencerK(sb_full, meta_full, ca, cx, bscores)
        full_blk = decision_to_block(dec_f, K)
        # shrunk decision: an absent cue WORD in the reduced vocab -> abstain immediately (global no-op, == full-V)
        if qa not in mapA or qx not in mapX:
            dec_r, rr = "abstain", {"abstain_fast": True}
        else:
            dec_r, rr = run_sequencerK_reduced(sb_red, meta_red, c.words, mapA, mapX, qa, qx, bscores,
                                               spurious_counter=spurious)
        red_blk = decision_to_block(dec_r, K)
        rows.append(dict(cue=(qa, qx), kind=kind, host_block=host_blk, full_block=full_blk, red_block=red_blk,
                         full_decision=dec_f, red_decision=dec_r,
                         identical=(full_blk == red_blk),
                         full_eq_host=(full_blk == host_blk)))

    # --- LESION on every present cue (both builds must FAIL SAFE -> abstain, and identically)
    les_rows = []
    for (a, x, p) in facts:
        dec_fl, _ = run_sequencerK(sb_full, meta_full, word_idx[a], word_idx[x], bscores, lesion=True)
        dec_rl, _ = run_sequencerK_reduced(sb_red, meta_red, c.words, mapA, mapX, a, x, bscores, lesion=True)
        les_rows.append((dec_fl, dec_rl))
    lesion_identical = all(decision_to_block(f, K) == decision_to_block(r, K) for (f, r) in les_rows)
    lesion_both_safe = all(f == "abstain" and r == "abstain" for (f, r) in les_rows)

    # --- PERMUTE on every present cue (the cyclic-shift anti-cheat; both builds must follow the rule + match)
    perm_rows = []
    for i, (a, x, p) in enumerate(facts):
        dec_fp, _ = run_sequencerK(sb_full, meta_full, word_idx[a], word_idx[x], bscores, permute=True)
        dec_rp, _ = run_sequencerK_reduced(sb_red, meta_red, c.words, mapA, mapX, a, x, bscores, permute=True)
        perm_rows.append((dec_fp, dec_rp))
    permute_identical = all(f == r for (f, r) in perm_rows)
    permute_follows_rule = all(perm_rows[i][1] == f"ans{(i + 1) % K}" for i in range(len(facts)))

    decisions_identical = all(r["identical"] for r in rows) and lesion_identical and permute_identical

    timing = None
    if do_time:
        timing = _time_runs(sb_full, meta_full, sb_red, meta_red, c.words, mapA, mapX, facts, word_idx, bscores, V, K)

    return dict(seed=seed, D=D, K=K, V=V, VA=VA, VX=VX, n_full=n_full, n_red=n_red,
                n_reduction=round(1.0 - n_red / n_full, 4),
                rows=rows, decisions_identical=decisions_identical,
                lesion_identical=lesion_identical, lesion_both_safe=lesion_both_safe,
                permute_identical=permute_identical, permute_follows_rule=permute_follows_rule,
                spurious_dropped=spurious, full_eq_host=all(r["full_eq_host"] for r in rows), timing=timing)


def _time_runs(sb_full, meta_full, sb_red, meta_red, words, mapA, mapX, facts, word_idx, bscores, V, K, reps=3):
    """Wall-clock the 80-step (20 drain + 60 settle) run on a representative PRESENT cue, full-V vs shrunk. The reset
    inside each run is the 20-step drain; the settle is 60 -> 80 full steps. Warmup once (JIT), then median of `reps`."""
    a0, x0, p0 = facts[0]
    ca, cx = word_idx[a0], word_idx[x0]
    # warmup (first-call JIT / CUDA-graph capture)
    run_sequencerK(sb_full, meta_full, ca, cx, bscores)
    run_sequencerK_reduced(sb_red, meta_red, words, mapA, mapX, a0, x0, bscores)
    tf, tr = [], []
    for _ in range(reps):
        t0 = time.perf_counter(); run_sequencerK(sb_full, meta_full, ca, cx, bscores); tf.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); run_sequencerK_reduced(sb_red, meta_red, words, mapA, mapX, a0, x0, bscores); tr.append(time.perf_counter() - t0)
    mf, mr = float(np.median(tf)), float(np.median(tr))
    return dict(full_ms=round(mf * 1e3, 2), red_ms=round(mr * 1e3, 2),
                speedup=round(mf / mr, 2) if mr > 0 else None, reps=reps)


def run_priority_check(seed, D):
    """The degenerate per-block-priority anti-cheat on BOTH builds: two blocks share (dog, go); the LOWER block must
    win (== host first-match) on the full-V AND the shrunk sequencer, identically."""
    facts = [("dog", "go", "north"), ("dog", "go", "river"), ("cat", "run", "tree")]   # 0 and 1 share (dog, go)
    c = OneBrainComposer(seed=seed, D=D, vocab=VOCAB_SMALL, k_max=8, enable_batched=False, enable_rf_cudagraph=False)
    for (a, x, p) in facts:
        c.store(a, x, p)
    word_idx = {w: i for i, w in enumerate(c.words)}
    K = len(facts)
    bscores = [block_cleanup_scores(c, b) for b in range(K)]
    sb_full, meta_full = build_sequencerK_bridge(seed=seed, V=c.V, K=K)
    agentsA, actionsX, mapA, mapX = reduced_cue_vocab(facts, K)
    sb_red, meta_red = build_sequencerK_reduced_bridge(seed=seed, VA=len(agentsA), VX=len(actionsX), K=K)
    dec_f, _ = run_sequencerK(sb_full, meta_full, word_idx["dog"], word_idx["go"], bscores)
    dec_r, _ = run_sequencerK_reduced(sb_red, meta_red, c.words, mapA, mapX, "dog", "go", bscores)
    host_blk = host_scan_block(c, "dog", "go")
    full_blk = decision_to_block(dec_f, K); red_blk = decision_to_block(dec_r, K)
    return dict(seed=seed, D=D, host_block=host_blk, full_block=full_blk, red_block=red_blk,
                identical=(full_blk == red_blk),
                priority_ok=(full_blk == 0 and red_blk == 0 and host_blk == 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--ks", default="8", help="store sizes K to test")
    ap.add_argument("--time", action="store_true", help="wall-clock the 80-step run full-V vs shrunk (1 rep set/seed)")
    ap.add_argument("--out", default="research/findings/raw/_seq_vocab_shrink_derisk.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ks = [int(k) for k in args.ks.split(",")]

    all_results = {}
    overall_go = True
    for K in ks:
        results = []
        for s in seeds:
            r = run_seed_K(s, args.dim, K, do_time=args.time)
            results.append(r)
            ident = "IDENTICAL" if r["decisions_identical"] else "DIVERGES"
            les = "lesion-safe+ident" if (r["lesion_identical"] and r["lesion_both_safe"]) else "LESION-FAIL"
            perm = "perm-ident+rule" if (r["permute_identical"] and r["permute_follows_rule"]) else "PERM-FAIL"
            tinfo = ""
            if r["timing"]:
                tinfo = f"  time full={r['timing']['full_ms']}ms red={r['timing']['red_ms']}ms ({r['timing']['speedup']}x)"
            spur = r["spurious_dropped"] or {}
            sp = f"  spur-dropped={spur}" if spur else ""
            print(f"K={K} seed {s} D{args.dim}: decisions {ident}  {les}  {perm}  "
                  f"neurons {r['n_full']}->{r['n_red']} (-{r['n_reduction']*100:.1f}%) VA={r['VA']} VX={r['VX']}"
                  f"  full=={'host' if r['full_eq_host'] else 'NOThost'}{sp}{tinfo}", flush=True)
            # print any divergent row for forensics
            for row in r["rows"]:
                if not row["identical"]:
                    print(f"    DIVERGE {row['kind']} cue={row['cue']}: full_block={row['full_block']} "
                          f"red_block={row['red_block']} (full_dec={row['full_decision']} red_dec={row['red_decision']})",
                          flush=True)
        all_results[str(K)] = results

    prio = [run_priority_check(s, args.dim) for s in seeds]
    prio_n = sum(p["priority_ok"] and p["identical"] for p in prio)
    for p in prio:
        ok = "OK" if (p["priority_ok"] and p["identical"]) else "FAIL"
        print(f"PRIORITY seed {p['seed']}: {ok}  full_block={p['full_block']} red_block={p['red_block']} "
              f"host_block={p['host_block']}", flush=True)

    summary = {}
    for K in ks:
        rs = all_results[str(K)]
        n = len(rs)
        ident_n = sum(r["decisions_identical"] for r in rs)
        les_n = sum(r["lesion_identical"] and r["lesion_both_safe"] for r in rs)
        perm_n = sum(r["permute_identical"] and r["permute_follows_rule"] for r in rs)
        host_n = sum(r["full_eq_host"] for r in rs)
        go = (ident_n == n and les_n == n and perm_n == n)
        overall_go = overall_go and go
        red_pct = round(100.0 * np.mean([r["n_reduction"] for r in rs]), 1)
        summary[str(K)] = dict(n=n, identical_n=ident_n, lesion_n=les_n, permute_n=perm_n, full_eq_host_n=host_n,
                               mean_neuron_reduction_pct=red_pct, verdict="GO" if go else "NEGATIVE")
        print(f"\nK={K} SUMMARY: decisions-identical {ident_n}/{n}  lesion {les_n}/{n}  permute {perm_n}/{n}  "
              f"(full==host {host_n}/{n}, mean neuron reduction {red_pct}%)  -> {summary[str(K)]['verdict']}", flush=True)
    n_prio = len(prio)
    overall_go = overall_go and (prio_n == n_prio)
    print(f"PRIORITY SUMMARY: {prio_n}/{n_prio}  -> {'GO' if prio_n == n_prio else 'NEGATIVE'}", flush=True)
    verdict = "GO" if overall_go else "NEGATIVE"
    print(f"\nOVERALL: {verdict}  (vocab-shrink == full-V decisions byte-identical on every case; K in {ks}, "
          f"{len(seeds)} seeds, GPU={is_gpu_backend()})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=dict(per_K=summary, priority_n=prio_n, priority_total=n_prio, verdict=verdict,
                                    gpu=is_gpu_backend()),
                       results=all_results, priority_results=prio), f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
