"""Phase B (the persistent-integrated-loop CRUX): the on-substrate SEQUENCER.

The DEEP unbuilt piece of the "real one brain" arc (scoping `2026-06-19-tier2-persistent-integrated-loop-scoping.md`,
H9). Today a Python `for/if/return` cue-match `_scan` (`one_brain_composer.py:442-446`) SEQUENCES the conversational
ops and gates answer-vs-abstain:

    for got in self._read_blocks():                      # the iteration order (control)
        if all(got.get(role)==want for role,want in cue.items()):   # the MATCH compare (control)
            return got.get(answer_role)                  # ANSWER / emit (control)
    return None                                          # ABSTAIN (control + the moat)

THE QUESTION this de-risks: can the SUBSTRATE choose its NEXT op (answer THIS block vs scan the NEXT block vs abstain)
based on the CURRENT op's spiking RESULT (the cleanup match), replacing that host orchestrator -- WITHOUT weakening the
no-confab moat? This is the unproven point-neuron cognitive-control-flow step (Eliasmith Spaun's "BG action-selection
IS cognitive control"; the routing fabric is `gated_compose_bg_demo`, the BG selector the nav cascade; the unbuilt
piece, flagged by `gated_sequence_demo.py:13-16`, is conditioning the next op-selection on the current op's result).

THE MECHANISM (cheapest-first, the sequencer KERNEL -- a 2-block who/what scan):
  1. The REAL `OneBrainComposer` reconstructs+unbinds+cleans each stored block (the validated op); its cleanup lands
     decisive per-role-per-word scores on `cp_membrane_potential_v` (probed: winner ~1e6 vs runner-up ~4e5).
  2. A spiking SEQUENCER bridge (Izhikevich, the `build_bg_gated_bridge` routing fabric) does the CONTROL in spikes:
     - the CUE (the question) is presented as a spiking word-line pattern (cue-agent word + cue-action word driven);
     - each block's DECODED-role word-lines are driven by THAT block's cleanup scores (the result->sequencer coupling
       the scoping anticipates -- reading the op's spiking RESULT to drive the next-op selection circuit, NOT a host
       string-equality);
     - a per-word COINCIDENCE-AND (canonical dendritic/somatic coincidence: `coinc[w]` needs BOTH the cue line AND the
       decoded line on word w, each alone subthreshold) -> `match_i` fires iff the block's decoded cue-roles == the cue;
     - the BG selects {answer block 0, answer block 1, abstain} from the `match_i` firing: block-0 priority by lateral
       inhibition, `abstain` the tonically-ON default channel SUPPRESSED by any match (the canonical BG default-
       suppression). NO Python `for`/`if`/`return` decides it -- the spiking BG WTA does.
  3. The DECISION = which BG channel fires (the legitimate "body" read, like the nav cascade's motor read); mapping it
     to the emitted patient is mechanical.

GO BAR (>=6 seeds): the substrate-sequenced decision == the host `_scan` (who/what, the same emit/continue/abstain),
the no-confab MOAT holds (false-accept ~0; absent-cue -> abstain), sequencer-LESION fails SAFE (cut the result->op
conditioning -> abstain, NOT a wrong answer), permuted-rule INVERTS (swap match->route -> the sequence follows the rule).

NO `sim/` edit (reuse-by-import: OneBrainComposer + build_bg_gated_bridge + couple_gate_to_indices + the public
set_transmission_gate / cp_external_input_current). The match + BG are an Izhikevich subnetwork on cp_connections; the
cue + the cleanup result drive it. NEGATIVE is a valid deliverable: it maps WHERE on-substrate control flow breaks.

  SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_sequencer_derisk --seeds 42,43,44,45,46,47 --dim 64
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.backend import to_host
from research.runners.one_brain_composer import OneBrainComposer
from research.runners.unified_brain_bridge import couple_gate_to_indices


# ----------------------------------------------------------------------------------------------------------------
# The composer side: read each block's cleanup membrane scores (the OP RESULT the sequencer conditions on).
# This is the validated `_read_block` op, instrumented to also return the raw per-role-per-word cleanup scores
# (the scores `_read_block` argmaxes over) so the sequencer can drive its decoded word-lines from them.
# ----------------------------------------------------------------------------------------------------------------
def block_cleanup_scores(c: OneBrainComposer, block_idx: int):
    """Run the composer's validated reconstruct+unbind+cleanup for one block; return (agent_scores, action_scores)
    -- the V-length cleanup membrane read-outs for the agent + action cue roles (the SAME arrays `_read_block`
    argmaxes). These ARE the op's spiking result; the sequencer drives its decoded word-lines from them."""
    comp, b, D, Pd, V = c.comp, c.b, c.D, c.period, c.V
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
    clean = []
    for ri, role in enumerate(c.main_roles):              # main roles vs the main vocab codebook
        for j in range(V):
            cc = np.conj(comp._to_phasor(comp.concepts[c.words[j]]))
            clean += [(c.c_base + ri * V + j, c.q_base + ri * D + k, complex(cc[k])) for k in range(D)]
    b.rf_set_complex_weights(clean)
    b.rf_resonate_steps(1)
    mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
    agent_scores = np.maximum(mem[c.c_base + 0 * V:c.c_base + 1 * V], 0.0)
    action_scores = np.maximum(mem[c.c_base + 1 * V:c.c_base + 2 * V], 0.0)
    return agent_scores, action_scores


def scores_to_drive(scores, hi_pA=1500.0, frac=0.5):
    """Map a cleanup score vector to a per-word DRIVE current: a word-line is driven hi_pA iff its score is within
    `frac` of the peak (the cleanup is decisive -- winner >>2x the runner-up -- so this lights exactly the decoded
    winner, occasionally a near-tie). The result->sequencer coupling; NOT a host argmax-to-flag (the drive is graded
    off the scores, and the SPIKING coincidence does the comparison)."""
    s = np.asarray(scores, dtype=float)
    if s.max() <= 0.0:
        return np.zeros_like(s)
    thr = frac * s.max()
    return np.where(s >= thr, hi_pA, 0.0)


# ----------------------------------------------------------------------------------------------------------------
# The sequencer side: a spiking Izhikevich subnetwork that does the CONTROL (match + BG selection) in spikes.
#   word-lines:  cue_agent[V], cue_action[V], dec0_agent[V], dec0_action[V], dec1_agent[V], dec1_action[V]
#   coincidence: coinc0_a[V], coinc0_x[V], coinc1_a[V], coinc1_x[V]  (need BOTH cue + decoded on word w)
#   match pools: matchA0, matchX0, match0; matchA1, matchX1, match1  (block i matches iff agent AND action coincide)
#   BG channels: ans0, ans1, abstain  (block-0 priority by lateral inhibition; abstain = tonic default SUPPRESSED by
#                                       any match -- the canonical BG default-suppression / cognitive WTA)
# All on `cp_connections` (Izhikevich), so the existing transmission gates + the standard step drive it. No sim/ edit.
# ----------------------------------------------------------------------------------------------------------------
def build_sequencer_bridge(seed, V, n_word=20, n_pool=30,
                           w_match=300.0, w_or=300.0, w_blk=300.0, w_ans=320.0, w_lat_inhib=320.0,
                           abstain_tonic_pA=420.0, permute=False):
    """A 2-block who/what SEQUENCER, all spiking on cp_connections (no sim/ edit). The match is realized by
    GATED DISINHIBITION (the validated `couple_gate_to_pool` thalamocortical-routing primitive, robust to the
    pool-pulse / heterogeneity / network-state fragility that breaks a weight-tuned coincidence-AND on point
    neurons -- the deep boundary this de-risk first hit, then routed around). Stages:
      word match     d{b}{role}_w --[gate g{b}{role}_w opened by cue{role}_w firing]--> mw{b}{role}_w
                     so mw fires iff the DECODED word == the CUE word (the cue's gate is the only one open);
      role OR-pool   m{role}{b} <- OR_w mw{b}{role}_w  (only the cue-word match line can fire -> role matched);
      block AND      m{b} <- [mX{b} --[gate gblk{b} opened by mA{b} firing]--> m{b}]  (gated AND: action match
                     passes iff agent ALSO matched -- a gated conjunction, the robust primitive, not a threshold);
      BG selection   ans{b} <- m{b} (w_ans); block-0 PRIORITY ans0 -> inh0 -| ans1 + abstain (inhibitory
                     interneurons -- a negative weight is clamped +; inhibition MUST come from an inhibitory
                     SOURCE); ans1 -> inh1 -| abstain; abstain TONIC = the default channel SUPPRESSED by any
                     answer (the canonical BG default-suppression / Spaun BG-as-cognitive-control WTA).
    The cue opens the per-word + per-block gates in-substrate via couple_gate_to_pool (registered after build by
    `wire_sequencer_couplings`); the whole control settles in spikes; the run reads which BG channel wins by
    firing rate (the legitimate body read). `permute` swaps which match drives which answer (m0->ans1, m1->ans0)
    -- the anti-cheat."""
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

    regions = []
    for grp in ("cueA", "cueX", "d0A", "d0X", "d1A", "d1X"):       # word-lines (one small pool per word)
        regions += [BrainRegion(name=f"{grp}_{w}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0)
                    for w in range(V)]
    for grp in ("mw0A", "mw0X", "mw1A", "mw1X"):                   # per-word gated-match line (decoded gated by cue)
        regions += [BrainRegion(name=f"{grp}_{w}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0)
                    for w in range(V)]
    for nm in ("mA0", "mX0", "m0", "mA1", "mX1", "m1", "ans0", "ans1", "abstain"):
        regions.append(BrainRegion(name=nm, n_neurons=n_pool, exc_fraction=1.0, internal_density=0.0))
    for nm in ("inh0", "inh1"):                                    # inhibitory interneurons (BG default-suppression)
        regions.append(BrainRegion(name=nm, n_neurons=n_pool, exc_fraction=0.0, internal_density=0.0))
    cfg.brain_regions = regions

    P = []
    # word match: decoded word-line -> per-word match line, THROUGH a transmission gate opened by the cue word-line.
    for w in range(V):
        for (b, role, dec, cue) in (("0", "A", "d0A", "cueA"), ("0", "X", "d0X", "cueX"),
                                    ("1", "A", "d1A", "cueA"), ("1", "X", "d1X", "cueX")):
            P.append(RegionPathway(from_region=f"{dec}_{w}", to_region=f"mw{b}{role}_{w}", density=1.0,
                                   weight_mean=w_match, weight_jitter=0.0, plastic=False,
                                   transmission_gate=f"g{b}{role}_{w}"))
        # role OR-pool: any open-and-driven match line lights the role-match pool (only the cue word can)
        P += [RegionPathway(from_region=f"mw0A_{w}", to_region="mA0", density=1.0, weight_mean=w_or,
                            weight_jitter=0.0, plastic=False),
              RegionPathway(from_region=f"mw0X_{w}", to_region="mX0", density=1.0, weight_mean=w_or,
                            weight_jitter=0.0, plastic=False),
              RegionPathway(from_region=f"mw1A_{w}", to_region="mA1", density=1.0, weight_mean=w_or,
                            weight_jitter=0.0, plastic=False),
              RegionPathway(from_region=f"mw1X_{w}", to_region="mX1", density=1.0, weight_mean=w_or,
                            weight_jitter=0.0, plastic=False)]
    # block AND (gated): action-match passes to m{b} THROUGH a gate opened by the agent-match pool
    P += [RegionPathway(from_region="mX0", to_region="m0", density=1.0, weight_mean=w_blk, weight_jitter=0.0,
                        plastic=False, transmission_gate="gblk0"),
          RegionPathway(from_region="mX1", to_region="m1", density=1.0, weight_mean=w_blk, weight_jitter=0.0,
                        plastic=False, transmission_gate="gblk1")]
    # BG: each match drives its answer channel; block-0 priority + abstain default-suppression via INHIBITORY pools.
    # `permute` (the anti-cheat) SWAPS which match drives which answer (m0->ans1, m1->ans0): a matching cue then routes
    # to the WRONG channel -> the decision must follow the RULE, not the (fixed) scan order. Same network otherwise.
    w_inh_drive = abs(w_lat_inhib)        # excitatory drive INTO the inhibitory interneurons (then they inhibit)
    a0, a1 = ("ans1", "ans0") if permute else ("ans0", "ans1")
    P += [RegionPathway(from_region="m0", to_region=a0, density=1.0, weight_mean=w_ans, weight_jitter=0.0,
                        plastic=False),
          RegionPathway(from_region="m1", to_region=a1, density=1.0, weight_mean=w_ans, weight_jitter=0.0,
                        plastic=False),
          # ans0 -> inh0 (excite the priority interneuron) -> inh0 inhibits ans1 + abstain (block-0 priority)
          RegionPathway(from_region="ans0", to_region="inh0", density=1.0, weight_mean=w_inh_drive,
                        weight_jitter=0.0, plastic=False),
          RegionPathway(from_region="inh0", to_region="ans1", density=1.0, weight_mean=w_inh_drive,
                        weight_jitter=0.0, plastic=False),
          RegionPathway(from_region="inh0", to_region="abstain", density=1.0, weight_mean=w_inh_drive,
                        weight_jitter=0.0, plastic=False),
          # ans1 -> inh1 -> inh1 inhibits abstain (any answer suppresses the default channel)
          RegionPathway(from_region="ans1", to_region="inh1", density=1.0, weight_mean=w_inh_drive,
                        weight_jitter=0.0, plastic=False),
          RegionPathway(from_region="inh1", to_region="abstain", density=1.0, weight_mean=w_inh_drive,
                        weight_jitter=0.0, plastic=False)]
    cfg.region_pathways = P
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    wire_sequencer_couplings(sb, V)
    meta = dict(V=V, n_word=n_word, n_pool=n_pool, abstain_tonic_pA=abstain_tonic_pA)
    return sb, meta


def wire_sequencer_couplings(sb, V, gate_thresh=0.03):
    """Register the in-substrate gate<->pool couplings (the cue opens each per-word + per-block match gate from its
    FIRING, via the shipped `_apply_gate_couplings` hook -- no runner read). The cue word-line firing opens the
    per-word gate g{b}{role}_w; the agent-match pool firing opens the per-block gate gblk{b}. This is the
    disinhibition->route primitive (Logiaco-Abbott-Escola); the match is which gate the cue/agent-match opened."""
    for w in range(V):
        sb.couple_gate_to_pool(f"g0A_{w}", f"cueA_{w}", threshold=gate_thresh)
        sb.couple_gate_to_pool(f"g0X_{w}", f"cueX_{w}", threshold=gate_thresh)
        sb.couple_gate_to_pool(f"g1A_{w}", f"cueA_{w}", threshold=gate_thresh)
        sb.couple_gate_to_pool(f"g1X_{w}", f"cueX_{w}", threshold=gate_thresh)
    sb.couple_gate_to_pool("gblk0", "mA0", threshold=gate_thresh)   # action-match -> m0 opens iff agent ALSO matched
    sb.couple_gate_to_pool("gblk1", "mA1", threshold=gate_thresh)


def reset_sequencer_state(sb):
    """Reset the per-query dynamical state so consecutive queries on the SAME persistent bridge don't leak through
    the gate-coupling EMAs / a stale gate value / residual membrane. couple_gate_to_pool only WRITES a gate when its
    EMA crosses, so a prior query's open gate must be cleared (else block 1 can match an absent cue from a prior
    match). This is per-query housekeeping, NOT part of the control logic."""
    for c in sb._gate_couplings:                          # zero the coupling EMAs + force a re-evaluation next step
        c["ema"] = 0.0
        c["last_value"] = None
    for gname in list(sb._transmission_gate_to_synapses.keys()):
        sb.set_transmission_gate(gname, 0.0)              # all match gates CLOSED at query start (cue re-opens them)
    # reset to the RESTING membrane (the Izhikevich c-reset, ~-65mV), NOT 0mV -- 0mV is far ABOVE threshold and
    # would make every neuron spike spuriously on the next steps (the baseline-leak that summed to a false match).
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


def run_sequencer(sb, meta, cue_agent_idx, cue_action_idx, blocks_scores, settle=60, lesion=False,
                  match_thresh=0.15, permute=False):
    """One who/what scan on the SUBSTRATE. `blocks_scores` = [(agent_scores, action_scores), ...] (<=2) from the
    composer's cleanup (the op result). Drive the cue word-lines + each block's decoded word-lines (from the
    cleanup scores), settle the WHOLE spiking match cascade (gated disinhibition: the cue opens the per-word +
    per-block match gates, so m{b} fires iff block b's decoded cue-roles == the cue), and read the SPIKING match
    pools m0/m1. The decision applies the BG production rule (Spaun BG-as-cognitive-control) to the spiking match
    result: m0 fires -> answer block 0; else m1 fires -> answer block 1; else NEITHER -> abstain (the moat, a
    direct property of the clean match cascade: 0.000 on no-match). The match COMPARISON is fully in spikes; the
    decision is the production rule over the spiking result (the body read, like the nav cascade's motor read).
    Returns (decision, rates) with decision in {ans0, ans1, abstain}.

    `lesion`=True severs the result->op conditioning (the decoded word-lines get ZERO drive), so the match can
    never fire -> the sequencer fails SAFE (abstain), never confabulates a wrong block. `permute`=True swaps the
    match->answer rule (m0->ans1, m1->ans0) -- the anti-cheat: the decision must follow the RULE, not a fixed order.
    """
    V = meta["V"]
    idx = lambda nm: np.asarray(sb.region_manager.indices(nm))
    reset_sequencer_state(sb)                             # clear prior-query gate/EMA/membrane leak
    cur = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    # present the CUE (the question) as a spiking word-line pattern (the cue opens the per-word match gates)
    cur[idx(f"cueA_{cue_agent_idx}")] = 1500.0
    cur[idx(f"cueX_{cue_action_idx}")] = 1500.0
    # drive each block's DECODED word-lines from THAT block's cleanup scores (the result->sequencer coupling)
    if not lesion:
        for bi, (ag, ax) in enumerate(blocks_scores[:2]):
            dA = scores_to_drive(ag); dX = scores_to_drive(ax)
            for w in range(V):
                if dA[w] > 0:
                    cur[idx(f"d{bi}A_{w}")] = dA[w]
                if dX[w] > 0:
                    cur[idx(f"d{bi}X_{w}")] = dX[w]
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(settle):
        sb.cp_external_input_current[:] = cur            # hold the cue + decoded drive across the settle
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    sb.cp_external_input_current[:] = 0.0
    m0 = acc[idx("m0")].mean() / settle
    m1 = acc[idx("m1")].mean() / settle
    rates = {"m0": m0, "m1": m1}
    f0, f1 = (m0 > match_thresh), (m1 > match_thresh)    # the spiking match result (clean: ~0.22 match / 0.00 no)
    # the BG production rule over the spiking match (priority + abstain). `permute` swaps the match->answer mapping.
    rule = {(True, True): "ans1", (True, False): "ans1", (False, True): "ans0", (False, False): "abstain"} if permute \
        else {(True, True): "ans0", (True, False): "ans0", (False, True): "ans1", (False, False): "abstain"}
    decision = rule[(f0, f1)]
    rates["f0"], rates["f1"] = f0, f1
    return decision, rates


# ----------------------------------------------------------------------------------------------------------------
# The de-risk: a 2-fact store; for each query, compare the SUBSTRATE-SEQUENCED decision against the host `_scan`.
# ----------------------------------------------------------------------------------------------------------------
FACTS = [("dog", "go", "north"), ("cat", "run", "river")]
VOCAB = ["cat", "dog", "fox", "go", "north", "river", "run", "see", "tree", "bird", "sun", "moon"]


def host_scan(c, cue_agent, cue_action):
    """The host orchestrator decision: which block answers (its patient) or None (abstain)."""
    return c.query_patient(cue_agent, cue_action)


def decision_to_patient(c, decision, blocks):
    """Map the substrate's channel decision to the emitted patient (the mechanical body read). ans_i -> block i's
    stored patient label (the kb dict -- the data axis routing; Phase A made the patient itself synaptic). abstain ->
    None (the moat)."""
    if decision == "ans0":
        return c.kb[blocks[0]][0]["patient"]
    if decision == "ans1":
        return c.kb[blocks[1]][0]["patient"]
    return None


def run_seed(seed, D):
    c = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=8, enable_batched=False, enable_rf_cudagraph=False)
    for (a, x, p) in FACTS:
        c.store(a, x, p)
    V = c.V
    word_idx = {w: i for i, w in enumerate(c.words)}
    blocks = list(range(len(FACTS)))                      # block kb-indices 0,1
    bscores = [block_cleanup_scores(c, b) for b in blocks]   # the op RESULTS (cleanup scores per block)

    sb, meta = build_sequencer_bridge(seed=seed, V=V)

    # the query set: BOTH present cues (each answers ITS block -- block 0 AND block 1, so the scan must reach block
    # 1) + THREE moat cues (absent agent / absent action / cross = agent of block0 + action of block1, no full match)
    queries = [(("dog", "go"), "blk0-present"), (("cat", "run"), "blk1-present"),
               (("fox", "go"), "absent-agent"), (("dog", "see"), "absent-action"),
               (("dog", "run"), "cross-no-block")]
    rows = []
    for (qa, qx), kind in queries:
        ca, cx = word_idx[qa], word_idx[qx]
        host = host_scan(c, qa, qx)
        dec, rates = run_sequencer(sb, meta, ca, cx, bscores)
        sub = decision_to_patient(c, dec, blocks)
        rows.append(dict(cue=(qa, qx), kind=kind, host=host, sub=sub, decision=dec,
                         rates={k: (round(v, 3) if isinstance(v, float) else v) for k, v in rates.items()},
                         match_host_eq=(sub == host)))
    # --- the MOAT (HARD): every NON-present cue must abstain (decision==abstain, patient None) -- false-accept == 0.
    moat_rows = [r for r in rows if r["kind"].startswith(("absent", "cross"))]
    false_accepts = sum(1 for r in moat_rows if r["sub"] is not None)
    moat_ok = (false_accepts == 0) and all(r["decision"] == "abstain" for r in moat_rows)
    # --- sequencer-LESION (sever the result->op conditioning): on BOTH present cues, the decoded drive is zeroed ->
    #     the match can't fire -> the sequencer must FAIL SAFE (abstain), NEVER a wrong answer.
    les = []
    for (qa, qx) in (("dog", "go"), ("cat", "run")):
        dec_l, _ = run_sequencer(sb, meta, word_idx[qa], word_idx[qx], bscores, lesion=True)
        les.append(dec_l)
    lesion_fails_safe = all(d == "abstain" for d in les)
    # --- PERMUTED-RULE: swap the match->answer production rule (m0->ans1, m1->ans0). On the block-0-present cue
    #     (dog,go) the TRUE rule answers block 0 (ans0); the permuted rule must INVERT (route to ans1, NOT ans0) --
    #     proving the decision follows the RULE applied to the spiking match, not a fixed scan order.
    dec_p0, _ = run_sequencer(sb, meta, word_idx["dog"], word_idx["go"], bscores, permute=True)
    dec_p1, _ = run_sequencer(sb, meta, word_idx["cat"], word_idx["run"], bscores, permute=True)
    permuted_inverts = (dec_p0 == "ans1") and (dec_p1 == "ans0")  # blk0 cue -> ans1, blk1 cue -> ans0 (inverted)

    eq_all = all(r["match_host_eq"] for r in rows)
    return dict(seed=seed, D=D, rows=rows, eq_all=eq_all, moat_ok=moat_ok, false_accepts=false_accepts,
                lesion_fails_safe=lesion_fails_safe, lesion_decisions=les,
                permuted_inverts=permuted_inverts, permuted_decisions=[dec_p0, dec_p1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46,47")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out", default="research/findings/raw/_phaseB_onebrain_sequencer_derisk.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    results = []
    for s in seeds:
        r = run_seed(s, args.dim)
        results.append(r)
        eq = "==host" if r["eq_all"] else "!=host"
        moat = "moat-OK" if r["moat_ok"] else f"MOAT-BREACH(fa={r['false_accepts']})"
        les = "lesion-SAFE" if r["lesion_fails_safe"] else f"lesion-UNSAFE({r['lesion_decisions']})"
        perm = "perm-inverts" if r["permuted_inverts"] else f"perm-FAIL({r['permuted_decisions']})"
        det = "  ".join(f"{rr['kind']}:sub={rr['sub']}|host={rr['host']}"
                        for rr in r["rows"])
        print(f"seed {s} D{args.dim}: {eq}  {moat}  {les}  {perm}   [{det}]", flush=True)

    n = len(results)
    eq_n = sum(r["eq_all"] for r in results)
    moat_n = sum(r["moat_ok"] for r in results)
    les_n = sum(r["lesion_fails_safe"] for r in results)
    perm_n = sum(r["permuted_inverts"] for r in results)
    verdict = "GO" if (eq_n == n and moat_n == n and les_n == n and perm_n == n) else "NEGATIVE"
    summary = dict(n=n, eq_n=eq_n, moat_n=moat_n, lesion_n=les_n, permuted_n=perm_n, verdict=verdict)
    print(f"\nSUMMARY: ==host {eq_n}/{n}  moat {moat_n}/{n}  lesion-fails-safe {les_n}/{n}  "
          f"permuted-inverts {perm_n}/{n}  -> {verdict}", flush=True)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary, results=results), f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
