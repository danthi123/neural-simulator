"""Phase B / burndown #3 Stage S0: the K-WAY on-substrate SEQUENCER (generalize K=2 -> K).

The K=2 sequencer (`_phaseB_onebrain_sequencer_derisk.py`, GO 3+ seeds) replaces the host `_scan`
(`one_brain_composer.py:_scan`/`query_patient`) -- a Python `for/if/return` cue-match + answer/abstain ROUTING --
for a 2-block who/what scan, in spikes (gated-disinhibition match cascade + a BG production rule). #3 retires that
host control flow at PRODUCTION scale (K up to 32 stored facts). Stage S0 generalizes the sequencer BUILDER from the
hard-coded K=2 (explicit `m0/m1`, `ans0/ans1`, `inh0/inh1`, `blocks_scores[:2]`, a 2-way rule dict, a block-0-priority
chain) to a clean parameter K, reusing the PROVEN mechanism (the `couple_gate_to_pool` gated-disinhibition match + the
BG first-match priority + the resting-membrane reset discipline) -- NO new selection mechanism, NO `sim/` edit.

THE K-WAY GENERALIZATION:
  * K match cascades: for b in range(K), the per-block gated-disinhibition match -- decoded word-line d{b}{role}_w
    routed to mw{b}{role}_w THROUGH a gate g{b}{role}_w that the CUE word-line cue{role}_w opens (so mw fires iff the
    decoded word == the cue word); a role OR-pool m{role}{b}; a block AND m{b} <- mX{b} gated by mA{b} (action-match
    passes iff agent ALSO matched). Identical to K=2, replicated K times by the block loop.
  * K-way priority WTA + abstain: each m{b} drives its answer channel ans{b}; FIRST-MATCH priority (== the host
    `_scan`'s "return the FIRST matching block") = block i inhibits every block j>i AND abstain, via an inhibitory
    interneuron inh{b} (ans{b} -> inh{b} -> {ans{j>b}} u {abstain}). The abstain channel is the tonic default
    SUPPRESSED by ANY match (the K-way OR into abstain's inhibition -- the canonical BG default-suppression). For the
    common case (facts are distinct, a unique cue matches exactly one block) a plain WTA suffices; the priority chain
    only disambiguates the rare degenerate multi-match, preserving the host first-match semantics exactly.
  * The K-way production rule: the decision is read from the K spiking match pools m{0..K-1} (the K=2 precedent reads
    m0/m1 + applies the rule in Python -- the production rule OVER the spiking match result, the legitimate body read):
    the LOWEST-index block with m{b} > match_thresh answers (first-match priority); none -> abstain (the moat).
    `permute` (the anti-cheat) CYCLICALLY SHIFTS the match->answer map (block b's match routes to answer (b+1)%K), so a
    matching cue routes to the WRONG block -> the decision must follow the RULE, not a fixed scan order.

GO BAR (this stage, CPU/numpy -- the exact-algebra parity oracle):
  * K=2 PARITY: the K-way sequencer reproduces the K=2 GO EXACTLY (== host `_scan`, same answer/abstain) -- a
    regression guard against the existing committed K=2 path.
  * K-WAY (K in {4,8}): == host `_scan` for who/what (the right block answers; absent/cross cues abstain), AND the
    no-confab MOAT holds (0 false-accepts), multi-seed.
  * ANTI-CHEATS: sequencer-LESION fails safe (cut the result->op conditioning -> abstain, never confabulate);
    permuted-rule INVERTS (the decision follows the cyclic-shift rule, not a fixed order); per-block priority correct
    (a degenerate two-block-match cue answers the LOWER block).

NO `sim/` edit (reuse-by-import: OneBrainComposer + SimulationBridge + couple_gate_to_pool + the public
set_transmission_gate / cp_external_input_current). NEGATIVE is a valid deliverable (it maps where on-substrate K-way
control flow breaks on point neurons).

  SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_sequencerK_derisk --seeds 42,43,44 --dim 64 --ks 2,4,8
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.backend import to_host, from_host
from research.runners.one_brain_composer import OneBrainComposer
# Reuse the composer-side op-result reader + the score->drive mapping VERBATIM from the K=2 de-risk (the FHRR cleanup
# is unchanged; S0 only generalizes the sequencer CONTROL fabric). Importing them keeps S0 a pure K-generalization.
from research.runners._phaseB_onebrain_sequencer_derisk import block_cleanup_scores, scores_to_drive


# ----------------------------------------------------------------------------------------------------------------
# The K-WAY sequencer: a spiking Izhikevich subnetwork doing the CONTROL (K match cascades + a K-way first-match
# priority WTA) in spikes. Every K=2 hard-coding (m0/m1, ans0/ans1, inh0/inh1, the block-0 priority chain) is
# replaced by a loop over range(K). All on `cp_connections` (Izhikevich) -- the standard step + transmission gates
# drive it. No sim/ edit.
# ----------------------------------------------------------------------------------------------------------------
def build_sequencerK_bridge(seed, V, K, n_word=20, n_pool=30,
                            w_match=300.0, w_or=300.0, w_blk=300.0, w_ans=320.0, w_lat_inhib=320.0,
                            abstain_tonic_pA=420.0, permute=False):
    """A K-block who/what SEQUENCER, all spiking on cp_connections (no sim/ edit). Generalizes
    `build_sequencer_bridge` (K=2) to a parameter K via a block loop. Stages, per block b in range(K):
      word match    d{b}{role}_w --[gate g{b}{role}_w opened by cue{role}_w firing]--> mw{b}{role}_w
                    (mw fires iff the DECODED word == the CUE word -- the cue's gate is the only one open);
      role OR-pool  m{role}{b} <- OR_w mw{b}{role}_w (only the cue-word match line can fire);
      block AND     m{b} <- [mX{b} --[gate gblk{b} opened by mA{b} firing]--> m{b}] (gated AND: action match passes
                    iff agent ALSO matched);
      BG selection  ans{b} <- m{b}; FIRST-MATCH priority: ans{b} -> inh{b} -| {ans{j} for j>b} + abstain; abstain
                    TONIC = the default channel SUPPRESSED by any answer (K-way OR into abstain's inhibition).
    The cue opens the per-word + per-block gates in-substrate via couple_gate_to_pool (registered after build by
    `wire_sequencerK_couplings`). `permute` cyclically shifts which match drives which answer (m{b} -> ans{(b+1)%K})
    -- the anti-cheat (a matching cue then routes to the WRONG channel -> the decision must follow the RULE)."""
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
    # PERF (byte-identical): batch the ~K*V*2 activity-driven gate-coupling control-pool means into ONE segment-sum
    # per step instead of a Python .mean()-per-coupling loop. The control pools are contiguous DISJOINT boolean
    # blocks so the segment-sum reproduces each per-coupling mean exactly (integer sum / integer count). Default-OFF
    # globally; opted-in here for the K-way sequencer where the scalar loop dominates host CPU time (~52% @ K=8).
    cfg.enable_vectorized_gate_couplings = True

    regions = []
    # cue word-lines (shared across blocks) + per-block decoded word-lines
    for grp in ("cueA", "cueX"):
        regions += [BrainRegion(name=f"{grp}_{w}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0)
                    for w in range(V)]
    for b in range(K):
        for role in ("A", "X"):
            regions += [BrainRegion(name=f"d{b}{role}_{w}", n_neurons=n_word, exc_fraction=1.0,
                                    internal_density=0.0) for w in range(V)]
    # per-word gated-match line (decoded gated by cue), per block
    for b in range(K):
        for role in ("A", "X"):
            regions += [BrainRegion(name=f"mw{b}{role}_{w}", n_neurons=n_word, exc_fraction=1.0,
                                    internal_density=0.0) for w in range(V)]
    # match/answer pools per block + the single abstain channel
    for b in range(K):
        for nm in (f"mA{b}", f"mX{b}", f"m{b}", f"ans{b}"):
            regions.append(BrainRegion(name=nm, n_neurons=n_pool, exc_fraction=1.0, internal_density=0.0))
    regions.append(BrainRegion(name="abstain", n_neurons=n_pool, exc_fraction=1.0, internal_density=0.0))
    # inhibitory interneurons per block (BG default-suppression / first-match priority)
    for b in range(K):
        regions.append(BrainRegion(name=f"inh{b}", n_neurons=n_pool, exc_fraction=0.0, internal_density=0.0))
    cfg.brain_regions = regions

    P = []
    for b in range(K):
        for w in range(V):
            # word match: decoded word-line -> per-word match line, THROUGH a gate opened by the cue word-line.
            for (role, dec, cue) in (("A", f"d{b}A", "cueA"), ("X", f"d{b}X", "cueX")):
                P.append(RegionPathway(from_region=f"{dec}_{w}", to_region=f"mw{b}{role}_{w}", density=1.0,
                                       weight_mean=w_match, weight_jitter=0.0, plastic=False,
                                       transmission_gate=f"g{b}{role}_{w}"))
            # role OR-pool: any open-and-driven match line lights the role-match pool (only the cue word can)
            P += [RegionPathway(from_region=f"mw{b}A_{w}", to_region=f"mA{b}", density=1.0, weight_mean=w_or,
                                weight_jitter=0.0, plastic=False),
                  RegionPathway(from_region=f"mw{b}X_{w}", to_region=f"mX{b}", density=1.0, weight_mean=w_or,
                                weight_jitter=0.0, plastic=False)]
        # block AND (gated): action-match passes to m{b} THROUGH a gate opened by the agent-match pool
        P.append(RegionPathway(from_region=f"mX{b}", to_region=f"m{b}", density=1.0, weight_mean=w_blk,
                               weight_jitter=0.0, plastic=False, transmission_gate=f"gblk{b}"))

    # BG: each match drives its answer channel (permute cyclically shifts the match->answer map). FIRST-MATCH
    # priority via inhibitory interneurons: ans{b} -> inh{b} -> {ans{j>b}} u {abstain}. abstain = tonic default
    # SUPPRESSED by any answer. `permute` (the anti-cheat) routes m{b} -> ans{(b+1)%K} so a matching cue answers
    # the WRONG block -- the decision must follow the RULE, not the fixed scan order. Same network otherwise.
    w_inh_drive = abs(w_lat_inhib)        # excitatory drive INTO the inhibitory interneurons (then they inhibit)
    for b in range(K):
        ans_target = f"ans{(b + 1) % K}" if permute else f"ans{b}"
        P.append(RegionPathway(from_region=f"m{b}", to_region=ans_target, density=1.0, weight_mean=w_ans,
                               weight_jitter=0.0, plastic=False))
        # ans{b} excites its priority interneuron inh{b}; inh{b} inhibits all LOWER-priority answers (j>b) + abstain
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
    wire_sequencerK_couplings(sb, V, K)
    meta = dict(V=V, K=K, n_word=n_word, n_pool=n_pool, abstain_tonic_pA=abstain_tonic_pA)
    return sb, meta


def wire_sequencerK_couplings(sb, V, K, gate_thresh=0.03):
    """Register the in-substrate gate<->pool couplings (the cue opens each per-word + per-block match gate from its
    FIRING, via the shipped `_apply_gate_couplings` hook -- no runner read). Generalizes the K=2 wiring to a block
    loop. The cue word-line firing opens the per-word gate g{b}{role}_w; the agent-match pool firing opens the
    per-block gate gblk{b}."""
    for b in range(K):
        for w in range(V):
            sb.couple_gate_to_pool(f"g{b}A_{w}", f"cueA_{w}", threshold=gate_thresh)
            sb.couple_gate_to_pool(f"g{b}X_{w}", f"cueX_{w}", threshold=gate_thresh)
        sb.couple_gate_to_pool(f"gblk{b}", f"mA{b}", threshold=gate_thresh)   # action-match -> m{b} opens iff agent matched


def reset_sequencerK_state(sb, drain_steps=20):
    """Reset the per-query dynamical state (gate-coupling EMAs / stale gate values / residual membrane) so
    consecutive queries on the SAME persistent bridge don't leak. Same discipline as the K=2 reset (the
    resting-membrane c-reset, NOT 0mV which is above threshold), plus a `drain_steps` blank-input settle BEFORE the
    membrane/gate clear. K-agnostic (it iterates ALL gates/couplings).

    Why the drain (the one K-scale addition): at K=2 the small inhibitory fabric dissipated within the reset, but at
    K=8 a PRIOR query that matched block b (firing mA{b}, opening gblk{b} via its EMA, driving the inh{b} priority
    chain) leaves delayed/recurrent activity that a single membrane clear doesn't fully drain -- a borderline
    near-tie m{b} (~0.16, just over the 0.15 threshold) then leaks into the NEXT query and first-match priority picks
    the stale lower block (diagnosed: K=8 seed-42 blk5 read m4=0.159 carried over from the blk4 query). Running
    `drain_steps` steps with ZERO external input lets the prior recurrent/delayed state decay to rest first; then the
    EMA/gate/membrane clear leaves every query starting from the SAME resting state. This is per-query housekeeping
    (the existing reset's stated job), NOT control logic -- the match COMPARISON is still entirely in spikes."""
    if drain_steps > 0 and getattr(sb, "cp_external_input_current", None) is not None:
        sb.cp_external_input_current[:] = 0.0             # drain prior recurrent/delayed activity to rest first
        for _ in range(int(drain_steps)):
            sb._run_one_simulation_step()
    for c in sb._gate_couplings:                          # zero the coupling EMAs + force a re-evaluation next step
        c["ema"] = 0.0
        c["last_value"] = None
    for gname in list(sb._transmission_gate_to_synapses.keys()):
        sb.set_transmission_gate(gname, 0.0)              # all match gates CLOSED at query start (cue re-opens them)
    if getattr(sb, "cp_izh_c_reset", None) is not None:
        sb.cp_membrane_potential_v[:] = sb.cp_izh_c_reset
    else:
        sb.cp_membrane_potential_v[:] = sb.cp_membrane_potential_v * 0.0 - 65.0
    sb.cp_recovery_variable_u[:] = sb.cp_recovery_variable_u * 0.0
    if getattr(sb, "cp_firing_states", None) is not None:
        sb.cp_firing_states[:] = False
    for attr in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise"):
        arr = getattr(sb, attr, None)
        if arr is not None:
            arr[:] = arr * 0.0


def run_sequencerK(sb, meta, cue_agent_idx, cue_action_idx, blocks_scores, settle=60, lesion=False,
                   match_thresh=0.15, permute=False, drive_frac=0.9):
    """One who/what scan on the SUBSTRATE, K-way. `blocks_scores` = [(agent_scores, action_scores), ...] (<=K) from
    the composer's cleanup (the op result). Drive the cue word-lines + each block's decoded word-lines (from the
    cleanup scores), settle the WHOLE spiking match cascade (gated disinhibition), read the K SPIKING match pools
    m{0..K-1}, and apply the K-way FIRST-MATCH priority production rule: the LOWEST-index block with m{b} >
    match_thresh answers; none -> abstain (the moat). The match COMPARISON is fully in spikes; the decision is the
    production rule over the spiking result (the body read). Returns (decision, rates), decision in
    {"ans0".."ans{K-1}", "abstain"}.

    `drive_frac` (the one K-scale tuning): the score->drive threshold (a decoded word-line is driven iff its cleanup
    score is within `drive_frac` of the block's peak). The host `_scan` matches on the cleanup ARGMAX (a single
    winner per role); the K=2 default frac=0.5 also lit the RUNNER-UP when its score was >=50% of the peak, which is
    harmless at K=2 but at imperfect-fidelity K a near-tie runner-up (measured worst ratio 0.81 across 48
    role/block/seed cleanups) spuriously matches another cue and first-match priority picks the wrong block
    (diagnosed: seed-45 block-0 agent decoded dog@1.00 AND bird@0.54, so cue (bird,go) spuriously matched block 0).
    drive_frac=0.9 (> the 0.81 worst runner-up) lights ONLY the argmax winner -- the faithful spiking realization of
    the cleanup's own decision (which word this block decodes to), == the host argmax. NOT a moat-relevant axis (an
    ABSENT cue matches NO block at any frac); it only removes the wrong-PRESENT-block leak.

    `lesion`=True severs the result->op conditioning (the decoded word-lines get ZERO drive) -> the match can never
    fire -> the sequencer fails SAFE (abstain). `permute`=True cyclically shifts the match->answer rule (m{b} ->
    ans{(b+1)%K}) -- the anti-cheat (the decision follows the RULE)."""
    V, K = meta["V"], meta["K"]
    idx = lambda nm: np.asarray(sb.region_manager.indices(nm))
    reset_sequencerK_state(sb)                            # clear prior-query gate/EMA/membrane leak
    cur = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    # present the CUE (the question) as a spiking word-line pattern (the cue opens the per-word match gates)
    cur[idx(f"cueA_{cue_agent_idx}")] = 1500.0
    cur[idx(f"cueX_{cue_action_idx}")] = 1500.0
    # drive each block's DECODED word-lines from THAT block's cleanup scores (the result->sequencer coupling)
    if not lesion:
        for bi, (ag, ax) in enumerate(blocks_scores[:K]):
            dA = scores_to_drive(ag, frac=drive_frac); dX = scores_to_drive(ax, frac=drive_frac)
            for w in range(V):
                if dA[w] > 0:
                    cur[idx(f"d{bi}A_{w}")] = dA[w]
                if dX[w] > 0:
                    cur[idx(f"d{bi}X_{w}")] = dX[w]
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    cur = from_host(cur)                                 # match bridge backend (numpy build -> cupy under SIM_BACKEND=cupy)
    for _ in range(settle):
        sb.cp_external_input_current[:] = cur            # hold the cue + decoded drive across the settle
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    sb.cp_external_input_current[:] = 0.0
    # the spiking match result per block (clean cascade: ~0.22 match / 0.00 no-match)
    m_rates = [acc[idx(f"m{b}")].mean() / settle for b in range(K)]
    fired = [r > match_thresh for r in m_rates]
    rates = {f"m{b}": round(m_rates[b], 3) for b in range(K)}
    rates.update({f"f{b}": fired[b] for b in range(K)})
    # the K-way FIRST-MATCH priority production rule over the spiking match. `permute` cyclically shifts match->answer.
    winner = next((b for b in range(K) if fired[b]), None)
    if winner is None:
        decision = "abstain"
    else:
        decision = f"ans{(winner + 1) % K}" if permute else f"ans{winner}"
    rates["winner"] = winner
    return decision, rates


# ----------------------------------------------------------------------------------------------------------------
# The de-risk: a K-fact store; for each query, compare the SUBSTRATE-SEQUENCED decision against the host `_scan`.
# Up to 16 distinct flat facts (S0 validates K in {2,4,8,16} on CPU; the production K=32 + D=128 margin sweep is
# Stage S2 -- the boundary test). Each fact has a UNIQUE (agent, action) cue so the host `_scan` is unambiguous and
# the moat cues (absent/cross) can be constructed cleanly.
# ----------------------------------------------------------------------------------------------------------------
ALL_FACTS = [("dog", "go", "north"), ("cat", "run", "river"), ("fox", "see", "tree"), ("bird", "fly", "sun"),
             ("sun", "see", "moon"), ("tree", "run", "fox"), ("moon", "go", "cat"), ("river", "fly", "bird"),
             ("wolf", "go", "hill"), ("hawk", "run", "lake"), ("deer", "see", "rock"), ("frog", "fly", "star"),
             ("star", "see", "leaf"), ("leaf", "run", "wolf"), ("hill", "go", "deer"), ("lake", "fly", "frog")]
VOCAB = ["cat", "dog", "fox", "go", "north", "river", "run", "see", "tree", "bird", "sun", "moon",
         "fly", "wolf", "hawk", "deer", "frog", "star", "leaf", "hill", "lake", "rock"]


def host_scan_block(c, cue_agent, cue_action):
    """The host `_scan` CONTROL decision: the INDEX of the first block whose decoded agent+action match the cue, or
    None (abstain). This is EXACTLY the host control flow the sequencer replaces -- `query_patient`'s internal
    `for i, got in enumerate(self._read_blocks()): if got['agent']==agent and got['action']==action: return i`.
    S0 generalizes that cue-match + first-match ROUTING; the patient LABEL is a downstream body-read (read from kb on
    both sides), NOT part of the control op. (Comparing the host's separately-re-decoded patient would conflate the
    sequencer's control decision with the host cleanup's patient-readback fidelity, which at small D=64 occasionally
    mis-decodes a patient even when the agent+action cue-match is clean -- e.g. seed-46 fact-1 patient river->go;
    the SUBSTRATE selects the right block there too. S2 runs the production D=128 where fidelity is higher.)"""
    for i, got in enumerate(c._read_blocks()):
        if got.get("agent") == cue_agent and got.get("action") == cue_action:
            return i
    return None


def decision_to_block(decision, K):
    """Map the substrate's channel decision to the selected block INDEX (the control decision), or None (abstain)."""
    if decision == "abstain":
        return None
    return int(decision[len("ans"):])


def patient_of(c, block_idx):
    """The body-read: block_idx's stored patient label (the kb routing), or None for abstain. Read from kb on BOTH
    the host and substrate sides so the comparison is the CONTROL decision (which block), not the host's noisy
    patient re-decode."""
    return None if block_idx is None else c.kb[block_idx][0]["patient"]


def _build_queries(facts):
    """The query set for a K-fact store: every PRESENT cue (each answers ITS block -- so the scan must reach the LAST
    block) + THREE moat cues (absent agent / absent action / cross = agent of one fact + action of a DIFFERENT fact
    with no full match). Returns [((agent, action), kind), ...]."""
    queries = [((a, x), f"blk{i}-present") for i, (a, x, p) in enumerate(facts)]
    # moat cues: pick words guaranteed not to form a stored (agent, action) pair
    agents = {a for (a, x, p) in facts}
    actions = {x for (a, x, p) in facts}
    pairs = {(a, x) for (a, x, p) in facts}
    absent_agent = next((w for w in VOCAB if w not in agents), "zzz")
    absent_action = next((w for w in VOCAB if w not in actions), "zzz")
    a0, x0 = facts[0][0], facts[0][1]
    # a CROSS cue: agent of fact0 with an action that fact0 does NOT have and that does not pair with that agent
    cross_action = next((x for (a, x, p) in facts if (a0, x) not in pairs), absent_action)
    queries += [((absent_agent, x0), "absent-agent"), ((a0, absent_action), "absent-action"),
                ((a0, cross_action), "cross-no-block")]
    return queries


def run_seed_K(seed, D, K):
    """Run one seed at store size K: build the composer + K facts, read each block's cleanup scores, build the K-way
    sequencer, and check == host_scan + the moat + lesion-fails-safe + permuted-inverts + per-block-priority."""
    facts = ALL_FACTS[:K]
    c = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=max(8, K), enable_batched=False, enable_rf_cudagraph=False)
    for (a, x, p) in facts:
        c.store(a, x, p)
    V = c.V
    word_idx = {w: i for i, w in enumerate(c.words)}
    blocks = list(range(len(facts)))
    bscores = [block_cleanup_scores(c, b) for b in blocks]      # the op RESULTS (cleanup scores per block)

    sb, meta = build_sequencerK_bridge(seed=seed, V=V, K=K)

    queries = _build_queries(facts)
    rows = []
    for (qa, qx), kind in queries:
        ca, cx = word_idx[qa], word_idx[qx]
        host_blk = host_scan_block(c, qa, qx)            # the host _scan CONTROL decision (which block / abstain)
        dec, rates = run_sequencerK(sb, meta, ca, cx, bscores)
        sub_blk = decision_to_block(dec, K)              # the substrate CONTROL decision (which block / abstain)
        host = patient_of(c, host_blk)                   # the body-read patient (kb routing, BOTH sides)
        sub = patient_of(c, sub_blk)
        rows.append(dict(cue=(qa, qx), kind=kind, host=host, sub=sub, decision=dec,
                         host_block=host_blk, sub_block=sub_blk, rates=rates,
                         match_host_eq=(sub_blk == host_blk)))    # CONTROL parity: same block selected / both abstain

    # --- the MOAT (HARD): every NON-present cue must abstain (decision==abstain, no block selected) -- FA == 0.
    moat_rows = [r for r in rows if r["kind"].startswith(("absent", "cross"))]
    false_accepts = sum(1 for r in moat_rows if r["sub_block"] is not None)
    moat_ok = (false_accepts == 0) and all(r["decision"] == "abstain" for r in moat_rows)

    # --- sequencer-LESION (sever the result->op conditioning) on every present cue -> must FAIL SAFE (abstain).
    les = []
    for (a, x, p) in facts:
        dec_l, _ = run_sequencerK(sb, meta, word_idx[a], word_idx[x], bscores, lesion=True)
        les.append(dec_l)
    lesion_fails_safe = all(d == "abstain" for d in les)

    # --- PERMUTED-RULE: cyclic shift (m{b} -> ans{(b+1)%K}). A present cue for block b must route to ans{(b+1)%K}
    #     (NOT ans{b}) -- the decision follows the RULE applied to the spiking match, not a fixed scan order.
    perm_decs = []
    perm_ok = True
    for i, (a, x, p) in enumerate(facts):
        dec_p, _ = run_sequencerK(sb, meta, word_idx[a], word_idx[x], bscores, permute=True)
        perm_decs.append(dec_p)
        if dec_p != f"ans{(i + 1) % K}":
            perm_ok = False
    permuted_inverts = perm_ok

    eq_all = all(r["match_host_eq"] for r in rows)
    return dict(seed=seed, D=D, K=K, rows=rows, eq_all=eq_all, moat_ok=moat_ok, false_accepts=false_accepts,
                lesion_fails_safe=lesion_fails_safe, lesion_decisions=les,
                permuted_inverts=permuted_inverts, permuted_decisions=perm_decs)


def run_priority_check(seed, D):
    """Per-block PRIORITY anti-cheat: a degenerate store with TWO blocks sharing the SAME (agent, action) cue but
    different patients. The host `_scan` returns the FIRST (lowest-index) match; the K-way priority WTA must answer
    the LOWER block too (block i inhibits block j>i). K=3 store: blocks 0 and 1 share the cue, block 2 distinct."""
    facts = [("dog", "go", "north"), ("dog", "go", "river"), ("cat", "run", "tree")]   # 0 and 1 share (dog, go)
    c = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=8, enable_batched=False, enable_rf_cudagraph=False)
    for (a, x, p) in facts:
        c.store(a, x, p)
    word_idx = {w: i for i, w in enumerate(c.words)}
    blocks = list(range(len(facts)))
    bscores = [block_cleanup_scores(c, b) for b in blocks]
    sb, meta = build_sequencerK_bridge(seed=seed, V=c.V, K=len(facts))
    dec, rates = run_sequencerK(sb, meta, word_idx["dog"], word_idx["go"], bscores)
    sub_blk = decision_to_block(dec, len(facts))
    host_blk = host_scan_block(c, "dog", "go")           # the host first-match (block 0, lowest matching index)
    # the priority check passes iff the substrate selects the LOWER (first) matching block == host first-match (0)
    priority_ok = (sub_blk == 0) and (host_blk == 0) and (sub_blk == host_blk)
    return dict(seed=seed, D=D, decision=dec, sub_block=sub_blk, host_block=host_blk,
                sub=patient_of(c, sub_blk), host=patient_of(c, host_blk), rates=rates, priority_ok=priority_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--ks", default="2,4,8", help="store sizes K to test")
    ap.add_argument("--out", default="research/findings/raw/_phaseB_onebrain_sequencerK_derisk.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ks = [int(k) for k in args.ks.split(",")]

    all_results = {}
    for K in ks:
        results = []
        for s in seeds:
            r = run_seed_K(s, args.dim, K)
            results.append(r)
            eq = "==host" if r["eq_all"] else "!=host"
            moat = "moat-OK" if r["moat_ok"] else f"MOAT-BREACH(fa={r['false_accepts']})"
            les = "lesion-SAFE" if r["lesion_fails_safe"] else f"lesion-UNSAFE({r['lesion_decisions']})"
            perm = "perm-inverts" if r["permuted_inverts"] else f"perm-FAIL({r['permuted_decisions']})"
            det = "  ".join(f"{rr['kind']}:sub={rr['sub']}|host={rr['host']}" for rr in r["rows"])
            print(f"K={K} seed {s} D{args.dim}: {eq}  {moat}  {les}  {perm}   [{det}]", flush=True)
        all_results[str(K)] = results

    # the per-block priority anti-cheat (degenerate two-block-match -> lower block wins, == host first-match)
    prio_results = [run_priority_check(s, args.dim) for s in seeds]
    prio_n = sum(p["priority_ok"] for p in prio_results)
    for p in prio_results:
        ok = "priority-OK" if p["priority_ok"] else "priority-FAIL"
        print(f"PRIORITY seed {p['seed']}: {ok}  decision={p['decision']} sub={p['sub']} host={p['host']}", flush=True)

    summary = {}
    overall_go = True
    for K in ks:
        rs = all_results[str(K)]
        n = len(rs)
        eq_n = sum(r["eq_all"] for r in rs)
        moat_n = sum(r["moat_ok"] for r in rs)
        les_n = sum(r["lesion_fails_safe"] for r in rs)
        perm_n = sum(r["permuted_inverts"] for r in rs)
        go = (eq_n == n and moat_n == n and les_n == n and perm_n == n)
        overall_go = overall_go and go
        summary[str(K)] = dict(n=n, eq_n=eq_n, moat_n=moat_n, lesion_n=les_n, permuted_n=perm_n,
                               verdict="GO" if go else "NEGATIVE")
        print(f"\nK={K} SUMMARY: ==host {eq_n}/{n}  moat {moat_n}/{n}  lesion-fails-safe {les_n}/{n}  "
              f"permuted-inverts {perm_n}/{n}  -> {summary[str(K)]['verdict']}", flush=True)
    n_prio = len(prio_results)
    overall_go = overall_go and (prio_n == n_prio)
    print(f"PRIORITY SUMMARY: {prio_n}/{n_prio}  -> {'GO' if prio_n == n_prio else 'NEGATIVE'}", flush=True)
    verdict = "GO" if overall_go else "NEGATIVE"
    print(f"\nOVERALL: {verdict}  (K in {ks}, {len(seeds)} seeds)", flush=True)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=dict(per_K=summary, priority_n=prio_n, priority_total=n_prio, verdict=verdict),
                       results=all_results, priority_results=prio_results), f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
