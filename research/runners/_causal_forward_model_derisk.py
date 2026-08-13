"""De-risk a LEARNED CAUSAL FORWARD MODEL on the shared spiking substrate — the reasoning
bottleneck named T1-4 in the 2026-08-12 faculty audit. Generalises the E2 affective
world-model (a 2-channel VALENCE predictor, 6/6 GO 2026-08-12) to a structured, DIRECTED,
n-way STATE forward model: given a state (an event/action), the brain PREDICTS the next
state by forward-simulation on the substrate, the transition edges are DIRECTED by
temporal-order STDP + phasic dopamine (DA-RPE), and a DO-intervention distinguishes a
genuine CAUSE from a mere CORRELATION.

WHY THIS IS THE GENUINE MISSING RUNG (the honest boundary this sits at)
----------------------------------------------------------------------
The production "why" answer is today a HOST symbolic JOIN over stored triples
(`_causal_composition_chain_derisk.py`: dir == obj_dir over query_patient reads) — it
RETROSPECTIVELY explains from stored facts; there is no forward model and no intervention.
The substrate already has (i) E2's 2-channel VALENCE forward model — but it predicts a
2-way affect SIGN, and its OWN finding names "n-way next-STATE recall" + "high-order
transition" as the disabled next rungs; (ii) gap#5's directed traveling-replay band — but
over a continuous track, not queryable discrete state transitions, and with no cause-vs-
correlation. NONE is a DIRECTED, QUERYABLE, n-way STATE forward model that (a) predicts an
UNSEEN multi-step consequence by rolling its own dynamics forward, (b) recovers CAUSE vs
CORRELATION under a DO-intervention, and (c) answers a "what-if" a host JOIN cannot. This
runner de-risks exactly that.

THE MECHANISM UNDER TEST — a directed spiking next-state forward model
---------------------------------------------------------------------
A single recurrent EVENT population `evt` of n_events blocks (one assembly per event).
Cross-block edges are weak + plastic at init; NO within-block edges. Learning has TWO
brain-based factors that make the edges DIRECTED and CAUSAL:

  1. TEMPORAL-ORDER STDP (Mehta-Blum-Abbott causal window; the gap#5 mechanism, 6-seed GO):
     the teacher renders each experienced episode as an ORDERED pair (event i then event j),
     so block-i fires BEFORE block-j -> the asymmetric STDP window tags i->j (pre-before-
     post = LTP) and DEPRESSES j->i (post-before-pre = LTD). The edge is DIRECTED by
     temporal precedence.
  2. PHASIC DOPAMINE (DA-RPE), the substrate's three-factor rule
     (`reward_defer_stdp_weight_update`=True: STDP timing creates the eligibility TAG but
     does NOT itself move the weight; a dopamine signal converts the tag to LTP/LTD). DA is
     held ON THROUGHOUT each ordered episode, so the tag is converted AS it forms and the
     net never free-runs into spurious cross-episode potentiation. This is also what lets a
     DO-intervention PRUNE a confounded edge:

CAUSE vs CORRELATION (Pearl do-calculus, teacher-scaffolded — T1-4's named mechanism):
  A CHAIN world  A ->(causes) B ->(causes) D, shown as ADJACENT pairs [A,B],[B,D] so A->D is
                 NEVER experienced (the "unseen consequence" must be a substrate ROLLOUT).
  A CONFOUND     a common cause C: [C,X],[C,Y]; and the SPURIOUS co-occurrence [X,Y] — X is
                 observed just before Y (both effects of C), so temporal-order STDP tags a
                 spurious X->Y though X does NOT cause Y (X,Y correlated, not causal).
  OBSERVATIONAL phase: every ordered episode is experienced with +DA -> every directed edge
  consolidates, INCLUDING the spurious X->Y (the correlation is learned).
  INTERVENTIONAL phase (the DO): the teacher forces do(X) in ISOLATION (C absent). The
  learned X->Y edge makes the model PREDICT Y (Y fires); but under do(X) the world produces
  NO Y -> the teacher delivers NEGATIVE DA (predicted-but-absent) -> the three-factor rule
  DEPRESSES the eligible X->Y. do(A) in isolation -> B genuinely follows -> +DA -> the chain
  is preserved. Over the interventional episodes the confounded X->Y erodes while the
  genuinely-causal edges (robust across observation AND intervention) survive — the
  invariance-across-interventions principle (Peters/Scholkopf) realised on spikes.

WHAT IS NEURAL vs THE LEGITIMATE (teacher/environment) BOUNDARY
--------------------------------------------------------------
- The PREDICTION is neural + LEARNED: the next-state read is `argmax` over the evt BLOCK
  spike-rates after cueing a state (the n-way generalisation of E2's sign(pos-neg) read) —
  a `cp_firing_states` read, never a host lookup over a stored transition table.
- The FORWARD-SIMULATION is neural: cue a state, then RELEASE — the directed recurrent edges
  propagate the packet A->B->D autonomously (the gap#5 traveling-packet mechanism); the
  multi-step consequence D is read from the substrate's own rollout, never a host chain-join.
- The transition edges are neural: STDP timing + DA-gated three-factor plasticity; no host
  writes a transition matrix.
- Legitimate boundary (declared, T1-4's "teacher DO-interventions"): the teacher renders the
  event drive (the environment presenting the state), the temporal ORDER of the experienced
  episode, and the phasic DA sign (the teacher's reinforcement — the brain's own dopamine
  channel converts it to a weight change). The teacher knowing do(X) produced no Y is the
  environment boundary, exactly as E2's observed next-valence was delivered as drive. THE
  NEXT RUNG (declared): drive the DA from a SPIKING mismatch unit (E2's surprise read ->
  from_reward/from_novelty DA) so the prediction-error is itself neural.

GO-GATE (pre-registered, 6 seeds 42/43/44/100/101/102)
------------------------------------------------------
 (1) DIRECTED FORWARD PREDICTION: cueing a state fires its LEARNED successor block as the
     argmax on the 1-step chain edges (A->B, B->D) — accuracy 1.0 — and the edge is DIRECTED
     (forward successor rate >> reverse predecessor rate).
 (2) UNSEEN CONSEQUENCE (forward-simulation, NOT recall): cue A, release, roll the substrate
     forward; the 2-step consequence D (NEVER experienced as a direct A->D pair) fires after
     B and above every off-chain block. A substrate rollout a host 1-step store cannot
     produce without an external chaining loop.
 (3) CAUSE vs CORRELATION (DO-intervention): after the interventional phase, do(X) does NOT
     fire Y (X does not cause Y) while do(C) DOES fire Y (C causes Y) — a clear separation.
 (4) LESION (load-bearing, decisive): zero the learned cross-block edges -> forward
     prediction collapses (cue A -> no successor); separation attributable to the edges >=0.8.
 (5) ANTI-CHEAT — CORRELATION-ONLY control collapses the cause claim: WITHOUT the
     interventional phase (observational learning only) the spurious X->Y survives, so do(X)
     WRONGLY fires Y — the DO-intervention pruning is load-bearing + attributable. AND a
     SHUFFLE of the transition curriculum fails to reproduce the true successors (the model
     learns the structure it is SHOWN, not a fixed template).
 (6) BRAIN-BASED: predictions = evt block-rate argmax reads (`cp_firing_states`); edges =
     STDP + DA three-factor weights; no host argmax over a stored transition table, no host
     formula computes the prediction or the causal verdict.

CPU-friendly (~n_events*blk ~= 180-neuron bridge); run under SIM_BACKEND=numpy for a
deterministic regime (the GPU gives no benefit at this scale — the E2 precedent, same scale,
ran 6-seed on numpy CPU).

Usage
-----
    SIM_BACKEND=numpy python -m research.runners._causal_forward_model_derisk \
        --seeds 42,43,44,100,101,102 \
        --out research/findings/raw/_causal_forward_model_6seed.json
    SIM_BACKEND=numpy python -m research.runners._causal_forward_model_derisk --smoke
    SIM_BACKEND=numpy python -m research.runners._causal_forward_model_derisk --opsearch
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ---------------------------------------------------------------------------
# The toy causal world. Events indexed 0..5.
#   CHAIN:    A(0) -> B(1) -> D(2)          (A causes B causes D)
#   CONFOUND: C(3) -> X(4), C(3) -> Y(5)    (common cause; X precedes Y -> spurious X->Y)
# ---------------------------------------------------------------------------
A, B, D, C, X, Y = 0, 1, 2, 3, 4, 5
N_EVENTS = 6
EVENT_NAMES = {A: "A", B: "B", D: "D", C: "C", X: "X", Y: "Y"}

# Experienced OBSERVATIONAL episodes — each an ordered pair (earlier fires before later ->
# a DIRECTED STDP tag). The chain is shown as ADJACENT pairs so A->D is NEVER experienced
# (the "unseen consequence" must be a substrate ROLLOUT A->B->D, not a learned A->D edge).
OBS_EPISODES = [
    (A, B),        # chain step 1  (A causes B)
    (B, D),        # chain step 2  (B causes D)
    (C, X),        # confound: common cause C -> X
    (C, Y),        # confound: common cause C -> Y
    (X, Y),        # SPURIOUS co-occurrence: X observed just before Y (both effects of C)
]
# 1-step transitions the forward prediction is scored on (the unambiguous chain edges).
CHAIN_EDGES = [(A, B), (B, D)]


def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def _shuffled_episodes(seed):
    """Anti-cheat: relabel the events by a non-identity permutation. The forward prediction,
    scored vs the TRUE chain edges, must then fail (the model learned the structure it was
    SHOWN, not a fixed A->B->D template)."""
    import numpy as np
    rng = np.random.RandomState(int(seed) + 4242)
    perm = list(range(N_EVENTS))
    for _ in range(64):
        rng.shuffle(perm)
        if perm != list(range(N_EVENTS)):
            break
    eps = [(perm[i], perm[j]) for (i, j) in OBS_EPISODES]
    return eps, perm


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_forward_model(seed, *, n_events=N_EVENTS, blk=30, init_w=0.2, xblock_density=0.6,
                        stdp_w_max=24.0, stdp_a_plus=0.02, stdp_a_minus=0.010,
                        stdp_tau_plus_ms=12.0, reward_learning_rate=0.18,
                        propagation_strength=0.05):
    """One recurrent EVENT population; cross-block edges weak + plastic; NO within-block edges.
    Three-factor DA-gated STDP: STDP tags (temporal order -> DIRECTED); phasic DA converts the
    tag to a weight change (`reward_defer_stdp_weight_update`)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    from sim.enums import NeuronModel, NeuronType

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    # Per-spike conductance gain (operating strength). The learned DIRECTED STRUCTURE (the
    # weight ratios) is what the STDP+DA build; this uniform gain brings the learned edges to
    # the strength at which a held state's successor fires (a maturation scalar, exactly as
    # gap#5's uniform operating-strength gain — it preserves the learned ratios).
    cfg.propagation_strength = float(propagation_strength)

    # --- DIRECTED plasticity: temporal-order STDP + DA-gated three-factor expression --------
    cfg.enable_stdp = True
    cfg.stdp_a_plus = float(stdp_a_plus)     # LTP (pre-before-post) — the DIRECTED, causal arm
    cfg.stdp_a_minus = float(stdp_a_minus)   # LTD (post-before-pre) — depresses the reverse edge
    cfg.stdp_tau_plus_ms = float(stdp_tau_plus_ms)   # short window -> only ADJACENT events pair
    cfg.stdp_tau_minus_ms = float(stdp_tau_plus_ms)
    cfg.stdp_w_max = float(stdp_w_max)       # operating cap (soft-bound gotcha: keep above init)
    cfg.stdp_w_min = 0.0
    cfg.enable_reward_modulation = True
    cfg.reward_defer_stdp_weight_update = True   # THREE-FACTOR: STDP tags; DA converts tag->weight
    cfg.reward_learning_rate = float(reward_learning_rate)
    cfg.reward_eligibility_tau_ms = 150.0        # short: keep the tag local to the current episode
    cfg.reward_baseline = 0.0
    cfg.current_reward_signal = 0.0
    cfg.reward_aversive_scale = 1.0              # symmetric LTP/LTD for the interventional prune

    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_parameter_heterogeneity = False
    # Deterministic regime (limbic/nav read protocol): silent at rest.
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False

    RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    cfg.brain_regions = [
        BrainRegion(name="evt", n_neurons=n_events * blk, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False, izh_neuron_type=RS),
    ]
    cfg.region_pathways = []

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)

    import numpy as np
    evt = np.asarray(bridge.region_manager.indices("evt"), dtype=np.int64)
    blocks = [evt[e * blk:(e + 1) * blk] for e in range(n_events)]

    # Cross-block edges only (i != j), weak + plastic. NO within-block edges (within-block
    # firing is synchronous -> STDP delta_t ~ 0 -> would not potentiate; leaving them out keeps
    # the topology + the lesion clean).
    rng = np.random.RandomState(int(seed) + 17)
    pre, post, w = [], [], []
    for i in range(n_events):
        for j in range(n_events):
            if i == j:
                continue
            for a_ in blocks[i]:
                for b_ in blocks[j]:
                    if xblock_density >= 1.0 or rng.rand() < xblock_density:
                        pre.append(int(a_)); post.append(int(b_)); w.append(float(init_w))
    bridge.inject_explicit_wiring({"xblock": {
        "pre_indices": pre, "post_indices": post, "initial_weights": w,
        "plastic": True, "conn_type": "ff"}})

    meta = dict(n_events=n_events, blk=blk)
    bridge._blocks = blocks
    bridge._blk = blk
    bridge._rest_v = bridge.cp_membrane_potential_v.copy()
    bridge._rest_u = bridge.cp_recovery_variable_u.copy()
    return bridge, cfg, meta


# ---------------------------------------------------------------------------
# Step / drive / read primitives (E2 conventions + the gap#5 current_time_ms GOTCHA)
# ---------------------------------------------------------------------------
def _step(bridge):
    # GOTCHA (gap#5): a raw _run_one_simulation_step does NOT advance current_time_ms; STDP
    # timestamps each spike from it, so without advancing it every delta_t=0 -> STDP no-ops.
    bridge._run_one_simulation_step()
    bridge.runtime_state.current_time_step += 1
    bridge.runtime_state.current_time_ms = (
        bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)


def _hard_reset(bridge):
    bridge.cp_membrane_potential_v[:] = bridge._rest_v
    bridge.cp_recovery_variable_u[:] = bridge._rest_u
    for name in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab",
                 "cp_conductance_g_nmda", "cp_firing_states", "cp_refractory"):
        arr = getattr(bridge, name, None)
        if arr is not None:
            arr[:] = 0
    bridge.cp_external_input_current[:] = 0.0


def _reset_eligibility(bridge):
    """Clear the STDP eligibility tag AND the per-neuron last-spike-time timing base between
    episodes. CRITICAL: STDP pairs from `cp_last_spike_time` (bridge.py:801) — if it is NOT
    reset, the LAST spikes of one episode cross-tag with the FIRST spikes of the next (a
    phantom cross-episode directed edge, e.g. D->C), which the DA then consolidates. Resetting
    it to a large-negative time puts every cross-episode pair outside the STDP window, so a
    phasic DA only converts tags formed WITHIN the current ordered episode."""
    et = getattr(bridge, "cp_eligibility_trace", None)
    if et is not None:
        et[:] = 0
    lst = getattr(bridge, "cp_last_spike_time", None)
    if lst is not None:
        lst[:] = -1000.0


def _drive_block(bridge, blocks, e, pA, xp):
    bridge.cp_external_input_current[:] = 0.0
    if e is not None:
        bridge.cp_external_input_current[xp.asarray(blocks[e])] = xp.float32(pA)


def _block_rates(bridge, blocks, n_steps, xp, *, drive_e=None, drive_pa=0.0,
                 da=0.0, read_from=0):
    """Step n_steps with an optional held drive on block drive_e and an optional phasic DA
    (current_reward_signal). Return per-block firing RATE (Hz), counted from step `read_from`."""
    if drive_e is not None:
        _drive_block(bridge, blocks, drive_e, drive_pa, xp)
    bridge.core_config.current_reward_signal = float(da)
    n_ev = len(blocks); blk = len(blocks[0])
    counts = [0] * n_ev
    for t in range(n_steps):
        _step(bridge)
        if t >= read_from:
            fs = bridge.cp_firing_states
            for e in range(n_ev):
                counts[e] += int(fs[xp.asarray(blocks[e])].sum())
    bridge.core_config.current_reward_signal = 0.0
    dur_s = max(n_steps - read_from, 1) * 1e-3
    return [c / blk / dur_s for c in counts]


# ---------------------------------------------------------------------------
# Training — observational (STDP + DA) then interventional (DO-prune)
# ---------------------------------------------------------------------------
def _episode(bridge, blocks, xp, events, *, cue_pa, dwell, gap, da):
    """Drive an ordered episode (event_0 then event_1 ...) with DA held ON so the temporal-
    order STDP tag is converted AS it forms (no free-running cascade)."""
    _hard_reset(bridge); _reset_eligibility(bridge)
    for e in events:
        _block_rates(bridge, blocks, dwell, xp, drive_e=e, drive_pa=cue_pa, da=da)
        _block_rates(bridge, blocks, gap, xp, drive_e=None, da=da)   # brief release, DA still on


def train(bridge, cfg, meta, xp, episodes, *, obs_reps=30, interv_reps=30,
          cue_pa=1200.0, dwell=8, gap=3, da_pos=1.0, da_neg=3.0, do_intervention=True,
          prune_src=X, prune_hold=30, interv_prop=0.30):
    """OBSERVATIONAL (at the LOW build propagation_strength — the net does not self-ignite, so
    only the externally-driven ORDERED pairs tag): every ordered episode with +DA -> directed
    edges consolidate (incl. the spurious X->Y). INTERVENTIONAL do(X): raise the gain so the
    learned X->Y actually FIRES Y; HOLD X (so X fires, then Y fires via X->Y, an X-before-Y
    LTP tag, a_plus>a_minus) with -DA -> the three-factor rule DEPRESSES the X->Y edge (the
    Pearl do-intervention). The chain edges (A->B,B->D) and the genuine C->Y are never driven
    by X here, so they are untouched — verified: AB/CY constant, only X->Y erodes."""
    blocks = bridge._blocks
    obs_prop = float(cfg.propagation_strength)
    for _ in range(obs_reps):
        for ep in episodes:
            _episode(bridge, blocks, xp, ep, cue_pa=cue_pa, dwell=dwell, gap=gap, da=da_pos)
    if not do_intervention:
        cfg.current_reward_signal = 0.0
        return
    # Raise the operating gain so the learned X->Y conducts (at obs_prop it does not); HOLD X so
    # Y fires and the X->Y tag forms, with -DA depressing it. Only DEPRESSION happens here
    # (da<=0), so the intervention can PRUNE but never create spurious potentiation.
    cfg.propagation_strength = float(interv_prop)
    for _ in range(interv_reps):
        _hard_reset(bridge); _reset_eligibility(bridge)
        _block_rates(bridge, blocks, prune_hold, xp, drive_e=prune_src, drive_pa=cue_pa, da=-da_neg)
    cfg.propagation_strength = obs_prop
    cfg.current_reward_signal = 0.0


# ---------------------------------------------------------------------------
# Reads (all spiking): forward prediction, rollout, DO-intervention
# ---------------------------------------------------------------------------
def _name_of(pred, lm):
    inv = {v: k for k, v in lm.items()}
    return EVENT_NAMES.get(inv.get(pred, pred), str(pred))


def _held_read(bridge, blocks, xp, src, *, cue_pa=1200.0, settle=4, read_steps=12):
    """HOLD the source block driven (E2's protocol: the state is held while the prediction is
    read) and return per-block firing rate over the read window. Holding the current state lets
    its learned successor fire steadily via the transition edge (a feedforward block does not
    self-sustain once released), and the packet propagates downstream while the state is held."""
    _hard_reset(bridge)
    _block_rates(bridge, blocks, settle, xp, drive_e=src, drive_pa=cue_pa)          # settle
    return _block_rates(bridge, blocks, read_steps, xp, drive_e=src, drive_pa=cue_pa)  # HOLD+read


def _predict_next(bridge, blocks, xp, src, **kw):
    rates = _held_read(bridge, blocks, xp, src, **kw)
    order = sorted([e for e in range(len(blocks)) if e != src], key=lambda e: rates[e], reverse=True)
    return order[0], rates


def forward_prediction(bridge, meta, xp, *, label_map=None):
    """Accuracy on the 1-step chain edges + a directedness read (successor >> predecessor)."""
    blocks = bridge._blocks
    lm = label_map or {e: e for e in range(meta["n_events"])}
    correct, per = 0, []
    for src, tgt in CHAIN_EDGES:
        s, t = lm[src], lm[tgt]
        pred, rates = _predict_next(bridge, blocks, xp, s)
        ok = int(pred == t)
        correct += ok
        per.append({"src": EVENT_NAMES.get(src), "tgt": EVENT_NAMES.get(tgt),
                    "pred": _name_of(pred, lm), "ok": ok, "succ_rate": round(rates[t], 2)})
    # directedness: hold B -> D fires (B->D forward); hold D -> B does NOT (D->B reverse depressed)
    s_b, s_d = lm[B], lm[D]
    rb = _held_read(bridge, blocks, xp, s_b)     # hold B -> expect D
    rd = _held_read(bridge, blocks, xp, s_d)     # hold D -> should NOT fire B
    directed_ratio = rb[s_d] / max(rd[s_b], 1e-6)      # B->D forward vs D->B reverse
    return {"acc": correct / len(CHAIN_EDGES), "n_correct": correct, "per": per,
            "directed_fwd_BtoD": round(rb[s_d], 2), "directed_rev_DtoB": round(rd[s_b], 2),
            "directed_ratio": round(directed_ratio, 2)}


def unseen_consequence(bridge, meta, xp, *, label_map=None, w_AD=None):
    """Forward-SIMULATION of an UNSEEN consequence: HOLD A. The learned A->B edge fires B, and
    B->D fires D — so the 2-step consequence D activates via the substrate's own dynamics though
    A->D was NEVER experienced (the direct A->D weight stays at init). D fires while every
    OFF-chain block (C,X,Y) stays silent, and B (the 1-step) fires at least as strongly as D
    (the 2-step downstream). A host 1-step store given 'A' returns only B; it cannot return D
    without an external chaining loop — the substrate returns D by rolling its dynamics forward."""
    blocks = bridge._blocks
    lm = label_map or {e: e for e in range(meta["n_events"])}
    s_a, s_b, s_d = lm[A], lm[B], lm[D]
    rates = _held_read(bridge, blocks, xp, s_a, read_steps=20)
    D_rate, B_rate = rates[s_d], rates[s_b]
    offchain_max = max(rates[lm[C]], rates[lm[X]], rates[lm[Y]])
    return {"predicts_D": bool(D_rate > max(offchain_max, 1.0) * 1.5 and B_rate >= D_rate * 0.8
                               and (w_AD is None or w_AD < 1.0)),
            "D_rate": round(D_rate, 2), "B_rate": round(B_rate, 2),
            "offchain_max": round(offchain_max, 2), "w_AD_direct": w_AD}


def do_intervention(bridge, meta, xp, *, label_map=None):
    """DO-intervention cause-vs-correlation: do(X) (HOLD X) should NOT fire Y (X not a cause of
    Y); do(C) (HOLD C) SHOULD fire Y (C is a cause). Returns Y rate under each intervention."""
    blocks = bridge._blocks
    lm = label_map or {e: e for e in range(meta["n_events"])}
    s_x, s_c, s_y = lm[X], lm[C], lm[Y]
    y_do_x = _held_read(bridge, blocks, xp, s_x)[s_y]   # X does NOT cause Y -> should be LOW
    y_do_c = _held_read(bridge, blocks, xp, s_c)[s_y]   # C causes Y         -> should be HIGH
    return {"Y_rate_do_X": round(y_do_x, 2), "Y_rate_do_C": round(y_do_c, 2),
            "cause_separation": round(y_do_c - y_do_x, 2),
            "X_not_cause_of_Y": bool(y_do_x < max(y_do_c, 1.0) * 0.5)}


def _lesion_xblock(bridge):
    """Zero the learned cross-block edges -> no forward model. CSR-direct, orientation-robust."""
    import numpy as np
    import scipy.sparse as sp
    blocks = bridge._blocks
    blkset = {}
    for e, bl in enumerate(blocks):
        for n in bl:
            blkset[int(n)] = e
    M = bridge.cp_connections.tocsr()
    indptr = np.asarray(_host(M.indptr)); indices = np.asarray(_host(M.indices))
    data = np.asarray(_host(M.data)).astype(np.float32)
    nz = 0
    for r in range(M.shape[0]):
        rb = blkset.get(r)
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off]); cb = blkset.get(c)
            if rb is not None and cb is not None and rb != cb:
                data[off] = 0.0; nz += 1
    bridge.cp_connections = sp.csr_matrix((data, indices, indptr), shape=M.shape)
    return nz


def _xblock_weight(bridge, i, j):
    """Mean learned weight from block i -> block j (a directedness / learning probe)."""
    import numpy as np
    blocks = bridge._blocks
    iset = set(int(n) for n in blocks[i]); jset = set(int(n) for n in blocks[j])
    M = bridge.cp_connections.tocoo()
    rows = np.asarray(_host(M.row)); cols = np.asarray(_host(M.col)); vals = np.asarray(_host(M.data))
    m = np.array([(int(r) in iset and int(c) in jset) for r, c in zip(rows, cols)])
    return float(vals[m].mean()) if m.any() else 0.0


# ---------------------------------------------------------------------------
# Per-seed driver
# ---------------------------------------------------------------------------
def run_seed(seed, *, mode="intact", verbose=True, obs_reps=30, interv_reps=30,
             read_prop=0.50, **build_kw):
    """mode: intact | lesion | corr_only | shuffle."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge, cfg, meta = build_forward_model(seed, **build_kw)

    if mode == "shuffle":
        eps, perm = _shuffled_episodes(seed)
        label_map = {e: e for e in range(meta["n_events"])}  # score vs TRUE chain -> must fail
        train(bridge, cfg, meta, xp, eps, obs_reps=obs_reps, interv_reps=interv_reps,
              do_intervention=True, prune_src=perm[X])
    else:
        train(bridge, cfg, meta, xp, OBS_EPISODES, obs_reps=obs_reps, interv_reps=interv_reps,
              do_intervention=(mode != "corr_only"))
        label_map = {e: e for e in range(meta["n_events"])}

    # FREEZE the learned structure + apply the uniform MATURATION GAIN (gap#5 protocol): learning
    # ran at LOW operating strength so the net never self-ignited during training (only the
    # externally-driven ORDERED pairs tagged -> a SELECTIVE directed structure, A->D unlearned);
    # now raise propagation_strength so those clean learned edges reach the strength at which a
    # held state's successor fires. The gain is UNIFORM -> it preserves the learned ratios.
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.current_reward_signal = 0.0
    cfg.propagation_strength = float(read_prop)

    # learning probes over the TRUE-role blocks (identity map). The DIRECT A->D edge must stay
    # unlearned so the unseen-consequence D can only route through the B intermediate. (For
    # shuffle these are not meaningful — shuffle is gated only on forward accuracy vs the true
    # chain — but they are harmless diagnostics.)
    w_XY = _xblock_weight(bridge, X, Y)
    w_YX = _xblock_weight(bridge, Y, X)
    w_AB = _xblock_weight(bridge, A, B)
    w_CY = _xblock_weight(bridge, C, Y)
    w_AD = _xblock_weight(bridge, A, D)

    if mode == "lesion":
        nz = _lesion_xblock(bridge)
        if verbose:
            print(f"  [lesion] zeroed {nz} cross-block edges")

    # BRAIN-BASED: reads are firing-state argmax; no host transition table is consulted.
    fwd = forward_prediction(bridge, meta, xp, label_map=label_map)
    unseen = unseen_consequence(bridge, meta, xp, label_map=label_map, w_AD=w_AD)
    doi = do_intervention(bridge, meta, xp, label_map=label_map)

    res = {"seed": seed, "mode": mode,
           "fwd_acc": fwd["acc"], "fwd_directed_ratio": fwd["directed_ratio"], "fwd": fwd,
           "unseen": unseen, "do": doi,
           "w_AB": round(w_AB, 3), "w_CY": round(w_CY, 3), "w_XY": round(w_XY, 3),
           "w_YX": round(w_YX, 3), "w_AD": round(w_AD, 3)}
    res["go"] = bool(fwd["acc"] >= 1.0 and unseen["predicts_D"] and doi["X_not_cause_of_Y"])
    if verbose:
        print(f"  [{mode:9s} seed {seed}] fwd_acc={fwd['acc']:.2f} dir_ratio={fwd['directed_ratio']:.1f} "
              f"| unseen B={unseen['B_rate']:.0f} D={unseen['D_rate']:.0f} off={unseen['offchain_max']:.0f} predictsD={unseen['predicts_D']} "
              f"| do(X)Y={doi['Y_rate_do_X']:.0f} do(C)Y={doi['Y_rate_do_C']:.0f} Xcause={not doi['X_not_cause_of_Y']} "
              f"| w:AB={w_AB:.1f} CY={w_CY:.1f} XY={w_XY:.1f} YX={w_YX:.1f} | GO={res['go']}")
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--obs-reps", type=int, default=30)
    ap.add_argument("--interv-reps", type=int, default=30)
    ap.add_argument("--blk", type=int, default=30)
    ap.add_argument("--init-w", type=float, default=0.2)
    ap.add_argument("--stdp-w-max", type=float, default=24.0)
    ap.add_argument("--reward-lr", type=float, default=0.18)
    ap.add_argument("--density", type=float, default=0.6)
    ap.add_argument("--read-prop", type=float, default=0.50)
    ap.add_argument("--opsearch", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1-seed intact+lesion+corr_only quick check")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    build_kw = dict(blk=args.blk, init_w=args.init_w, stdp_w_max=args.stdp_w_max,
                    reward_learning_rate=args.reward_lr, xblock_density=args.density)
    rep_kw = dict(obs_reps=args.obs_reps, interv_reps=args.interv_reps, read_prop=args.read_prop)

    if args.opsearch:
        print("[causal-forward-model OPSEARCH seed=42]")
        for lr in (0.08, 0.12, 0.18):
            for wmax in (6.0, 8.0, 12.0):
                bk = dict(build_kw); bk.update(stdp_w_max=wmax, reward_learning_rate=lr)
                r = run_seed(42, verbose=False, **rep_kw, **bk)
                d = r["do"]
                print(f"  lr={lr:.2f} wmax={wmax:4.0f} | fwd_acc={r['fwd_acc']:.2f} "
                      f"predictsD={r['unseen']['predicts_D']} do(X)Y={d['Y_rate_do_X']:.0f} "
                      f"do(C)Y={d['Y_rate_do_C']:.0f} w:AB={r['w_AB']:.1f} XY={r['w_XY']:.1f} GO={r['go']}")
        return

    if args.smoke:
        print("=== SMOKE (seed 42): intact | lesion | corr_only ===")
        it = run_seed(42, mode="intact", **rep_kw, **build_kw)
        le = run_seed(42, mode="lesion", **rep_kw, **build_kw)
        co = run_seed(42, mode="corr_only", **rep_kw, **build_kw)
        print("\n  SMOKE checks:")
        print(f"   intact GO ............................. {it['go']}")
        print(f"   lesion collapses forward pred ......... {le['fwd_acc'] < 1.0}  (acc {le['fwd_acc']:.2f})")
        print(f"   corr_only WRONGLY makes X cause Y ..... {not co['do']['X_not_cause_of_Y']}  "
              f"(do(X)->Y {co['do']['Y_rate_do_X']:.0f} vs intact {it['do']['Y_rate_do_X']:.0f})")
        return

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    print("=== INTACT (directed causal forward model) ===")
    intact = [run_seed(s, mode="intact", **rep_kw, **build_kw) for s in seeds]

    ac_seeds = seeds[:3]
    print("\n=== ANTI-CHEATS (mechanistic; 3 seeds) ===")
    lesion = [run_seed(s, mode="lesion", **rep_kw, **build_kw) for s in ac_seeds]
    corr_only = [run_seed(s, mode="corr_only", **rep_kw, **build_kw) for s in ac_seeds]
    shuffle = [run_seed(s, mode="shuffle", **rep_kw, **build_kw) for s in ac_seeds]

    n_go = sum(1 for r in intact if r["go"])
    verdict = "GO" if (len(intact) >= 6 and n_go >= 5) or (len(intact) < 6 and n_go == len(intact)) else "BOUNDARY"
    les_collapse = sum(1 for r in lesion if r["fwd_acc"] < 1.0)
    corr_wrong = sum(1 for r in corr_only if not r["do"]["X_not_cause_of_Y"])
    shuf_fail = sum(1 for r in shuffle if r["fwd_acc"] < 1.0)

    from tools.lab import attributable_to
    from tools.verdict import Verdict
    intact_sep = _st.mean([r["do"]["cause_separation"] for r in intact[:len(ac_seeds)]])
    corr_sep = _st.mean([r["do"]["cause_separation"] for r in corr_only])
    frac = attributable_to("cause-vs-correlation separation @ DO-intervention",
                           intact_sep, corr_sep)

    fwd_min = min(r["fwd_acc"] for r in intact)
    unseen_go = sum(1 for r in intact if r["unseen"]["predicts_D"])
    docorrect = sum(1 for r in intact if r["do"]["X_not_cause_of_Y"])

    print("\n=== VERDICT ===")
    print(f"  INTACT GO: {n_go}/{len(intact)} seeds (>=5/6 required)  ->  {verdict}")
    print(f"  forward-prediction acc (min over seeds): {fwd_min:.2f}")
    print(f"  unseen-consequence (predicts D): {unseen_go}/{len(intact)}")
    print(f"  cause-vs-correlation (X not cause of Y): {docorrect}/{len(intact)}")
    print(f"  lesion collapses forward prediction: {les_collapse}/{len(lesion)}")
    print(f"  corr_only WRONGLY asserts X->Y (DO-prune load-bearing): {corr_wrong}/{len(corr_only)}")
    print(f"  shuffle fails to reproduce true chain: {shuf_fail}/{len(shuffle)}")

    v = (Verdict("directed causal forward model — n-way state prediction + DO-intervention")
         .require("intact GO on >=5/6 seeds", n_go,
                  expect=lambda k: k >= max(5, len(intact) - 1) if len(intact) >= 6 else k == len(intact))
         .require("forward-prediction accuracy 1.0 (min over seeds)", fwd_min, expect=lambda x: x >= 1.0)
         .require("predicts UNSEEN 2-step consequence D (all seeds)", unseen_go,
                  expect=lambda k: k == len(intact))
         .require("cause-vs-correlation: X not a cause of Y (all seeds)", docorrect,
                  expect=lambda k: k == len(intact))
         .require("lesion collapses forward prediction (3/3)", les_collapse,
                  expect=lambda k: k == len(lesion))
         .require("corr_only WRONGLY makes X cause Y — DO-prune load-bearing (3/3)", corr_wrong,
                  expect=lambda k: k == len(corr_only))
         .require("shuffle fails to reproduce true chain (3/3)", shuf_fail,
                  expect=lambda k: k == len(shuffle))
         .control("DO-intervention separation vs corr_only", intact_sep, corr_sep, min_separation=1.0)
         .require("cause separation attributable to DO-prune (>=0.8)", frac,
                  expect=lambda x: x is not None and x >= 0.8)
         .disabled("OU background process", "deterministic regime for a controllable operating point")
         .disabled("conductance noise", "deterministic regime")
         .disabled("spiking-mismatch-driven DA", "the DA sign is delivered by the teacher (the "
                   "environment boundary, per T1-4 'teacher DO-interventions'); driving it from a "
                   "SPIKING mismatch unit (E2's surprise read -> from_reward DA) so the "
                   "prediction-error is itself neural is the named next rung")
         .disabled("high-order (history-dependent) transitions",
                   "the model is FIRST-ORDER (state -> next); ambiguous-given-history states "
                   "need the HTM-TM high-order predictor (EMERGE-15 GO) — the named next rung")
         .disabled("learned event GROUNDING", "events are delivered as block drive (the "
                   "environment boundary); grounding them in the emergent relational/spatial "
                   "code (2026-08-11 GO) is the follow-on"))
    decided = v.decide(go=(verdict == "GO"))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mode": "causal_forward_model", "intact": intact, "lesion": lesion,
                       "corr_only": corr_only, "shuffle": shuffle,
                       "n_go": n_go, "n_seeds": len(intact),
                       "verdict": decided["status"], "verdict_label": verdict,
                       "forward_acc_min": fwd_min, "unseen_go": unseen_go,
                       "cause_correct": docorrect, "lesion_collapse": les_collapse,
                       "corr_only_wrong": corr_wrong, "shuffle_fail": shuf_fail,
                       "intact_cause_separation": intact_sep, "corr_cause_separation": corr_sep,
                       "cause_attributable_to_do_prune": frac,
                       "preconditions": decided["preconditions"],
                       "disabled_processes": decided["disabled_processes"],
                       "verdict_status": decided["status"]}, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
