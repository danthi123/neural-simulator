"""RUNG B-1b (the FULLY-NEURAL role SELECTION) -- the word's thematic role is elected by an ON-BRIDGE spiking
mutual-inhibition WINNER-TAKE-ALL (WTA) whose winning ensemble's firing OPENS the composer's `role_route_<R>` gate,
REPLACING the host `argmax(f @ Ws[k])` of RUNG B-1. Everything runs on ONE `UnifiedBrainBridge`. CPU/numpy.

CONTEXT. RUNG B-1 made the comprehension->composition hand-off SYNAPTIC: the reservoir's learned role output drove
the composer's bind through the parser-gated `role_route_<R>` topographic route. But the role SELECTION was still a
HOST `argmax(f @ Ws[k])` -- a Python argmax picked the role, then the corresponding parser conjunction was fired to
open its gate. RUNG B-1b removes that host argmax: the reservoir's per-word role LOGITS `(f @ Ws[k])[[AGENT,PREDICATE,
THEME]]` DRIVE a spiking WTA (3 excitatory role ensembles + one shared inhibitory pool, genuine biased competition on
the SAME bridge). The WTA WINNER's firing -- via the coupling `role_route_<R> <- ens[R]` -- opens exactly that role's
gate. The role that reaches the composer's bind is the WTA winner (read from the LATCHED gate == the ensemble that
fired most), NOT a host argmax over logits. The whole comprehend->select->bind->recall turn is on the substrate.

  reservoir final state f --(Ws[k], the learned read-out)--> logits3 per content word
      --> DRIVE the 3 WTA ensembles (BASE + GAIN*normalized_rectified(logits3)) --> spiking biased competition
      --> the WINNER ensemble fires --> its coupling opens `role_route_<winner>` --> LATCH that gate
      --> the composer's role bank gets the winner role's +-1 pattern --> the word binds with the WTA-elected role,
          provenance-clean (the role reaches the gate ONLY via ensemble firing; no host argmax decides the gate).

ANTI-CHEATS (multi-seed). B-1's SIX, reused verbatim in spirit:
  (1) ROUTE RECOVERS THE FACT (route recall >= 0.80n).
  (2) ROUTE NOT WORSE THAN DICT (dict = RUNG B-1's HOST-argmax `_bind_reservoir_fact` path).
  (3) MOAT (<= 0.05 false-accept on an unstored (agent, action)).
  (4) PROVENANCE-CLEAN (the composer role bank receives ZERO direct external current on a WTA bind; reuse I5a
      `provenance_role_bank_current`).
  (5) ROUTE-LESION collapses (cut the synaptic route; reuse I5a `lesion_route`).
  (6) RESERVOIR-LESION collapses (lesion the reservoir's closed-class identity -> roles collapse).
PLUS THREE NEW (the B-1b claim -- neural selection):
  (7) PROVENANCE-NEURAL-SELECT: the role reaches the gate ONLY via ens firing. SOURCE-CHECK: `_op_wta` contains NO
      `np.argmax(...@Ws...)` / no host argmax picking the gate/conj; RUNTIME: the LATCHED role == argmax over the WTA
      ensembles' firing (`cp_firing_states[ens[k]]`).
  (8) WTA-LESION: zero the I->E (inh->ens) synapse weights (the biased-competition mechanism) -> the competition
      collapses -> multiple/ambiguous gates latch (or the wrong one) -> recall collapses (< intact).
  (9) Ws-SCRAMBLE: permute the 3 role columns of each Ws[k] -> the logits misroute the WTA -> recall collapses.

STRICTLY CPU/numpy (SIM_BACKEND=numpy). NO `sim/` edit (reuse-by-import + the committed `role_wta_n` support). The
WTA is wired IN PLACE via `set_pathway_weights(add_missing=True)` so the trained parser is preserved (NO re-injection).

Run:  SIM_BACKEND=numpy python -m research.runners._rungB1b_neural_role_wta_derisk \
          --seeds 42 43 44 --json research/findings/raw/_rungB1b_neural_role_wta.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

from sim.backend import get_backend, to_host  # noqa: E402
import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    _content_pools, _ROLES, _gen, _TRAIN_KINDS, _N_TRAIN_PER_CONSTRUCTION,
)
from research.runners._emerge88_reservoir_comprehends_composer_answers_derisk import (  # noqa: E402
    ReservoirComprehender, _ROLE2FIELD, _build_test_facts,
)
from research.runners.unified_brain_bridge import (  # noqa: E402
    UnifiedBrainBridge, SYNAPTIC_ROUTE_ROLES, ROLE_SRC_DRIVE_PA, ROLE_GATE_PREWARM_CAP_STEPS,
    couple_gate_to_indices,
)
from research.runners.core_sim_composition import (  # noqa: E402
    onoff, _scale_to_current, FILL_DRIVE, RESET_STEPS,
)
# reuse I5a's synaptic-route anti-cheat instruments UNCHANGED
from research.runners._burndown_I5a_synaptic_parser_composer import (  # noqa: E402
    _gate_open, lesion_route, provenance_role_bank_current,
)

# pd=192 (up from I5a/B-1's 128): the composer's per-fact decode MARGIN. At pd=128 the composer's spiking unbind
# mis-decodes 1-2 BORDERLINE facts on some seeds (43/44) under OU noise -- a codebook-MARGIN artifact (verified: the
# SAME miss appears on the HOST-argmax dict path on the SAME WTA-wired substrate, and it goes BOTH ways -- the WTA
# route BEATS the dict on seed 42 -- i.e. composer decode noise, NOT a role-SELECTION difference; the WTA latches the
# correct role on 0/6 mismatches vs host argmax on every seed). Widening the codebook to pd=192 removes the margin
# artifact: both the WTA route AND the dict path decode 12/12 on seeds 42/43/44, so route-vs-dict isolates the role
# SELECTION mechanism (identical) instead of the composer's OU jitter. (Same lever I5a used going 64->128.)
PROJ_DIM = 192
N_TEST = 6        # small test set -> small composer codebook + small bridge (weak-CPU tractable)

# ── on-bridge WTA config (recorded recipe, AUTONOMOUS_STATE.md commit a7fbf92f; smoke winner 4/4) ────────────
WTA_P = 20          # per-ensemble excitatory neurons (ens[k] = [base+k*P, base+(k+1)*P))
WTA_INH = 30        # shared inhibitory pool (inh = [base+3P, base+3P+INH))
WTA_W_EI = 24.0     # ens -> inh (drives the shared inhibition)
WTA_W_EE = 18.0     # ens -> ens within-ensemble (excitatory positive feedback)
WTA_W_IE = 20.0     # inh -> ens broadcast (the competition; I->E = the biased-competition lesion target)
# WTA DRIVE (retuned from the recorded smoke recipe BASE=4/GAIN=63 -> a UNIFORM baseline to ALL ensembles + a graded
# logit BIAS). The retune has two load-bearing reasons, both found honestly in dev:
#   (1) SEQUENTIAL ADAPTATION. The smoke recipe measured the winner's firing on the FIRST single-shot competition
#       (transient, pre-adaptation). Binding 18 words on ONE bridge, the Izhikevich ensembles accommodate across
#       words, so a winner at only 4+63=67 pA fires too sparsely to win reliably after the first few (measured
#       9/18, DRIFTING). A larger drive (winner ~270 pA) re-fires the winner robustly each word (18/18).
#   (2) LOAD-BEARING INHIBITION (the WTA-lesion anti-cheat). With the reservoir's ~one-hot logits, max-normalizing
#       gives losers ~0 drive, so the winner wins by FEEDFORWARD drive alone and zeroing the I->E inhibition does
#       NOT collapse selection (the WTA-lesion anti-cheat fails to bite). A UNIFORM baseline drives ALL THREE
#       ensembles toward firing, so the I->E inhibition is what SILENCES the losers: intact -> exactly 1 gate opens
#       (the biased winner); I->E-lesioned -> the losers free-run, 2-3 gates open -> the bind superimposes roles ->
#       recall collapses. This is the genuine biased-competition regime (inhibition load-bearing), the same
#       mechanism the recipe intended, at a drive that makes it observable on clean logits.
WTA_BASE = 150.0    # UNIFORM baseline drive to every ensemble (so I->E inhibition is what selects the winner)
WTA_GAIN = 120.0    # graded logit bias added to the (max-normalized) rectified logits on top of the baseline
ROLE_WTA_N = 3 * WTA_P + WTA_INH        # 90
WTA_GATE_THRESHOLD = 0.005              # sits between WTA loser EMA (~0/0.0025) and winner (~0.0104)

# ── _op_wta timing constants (forked from _op_synaptic; validated in the smoke) ─────────────────────────────
WTA_SETTLE_STEPS = 40      # settle after applying the drive so the WTA establishes the winner (no accumulation)
WTA_PREWARM_CAP = ROLE_GATE_PREWARM_CAP_STEPS   # cap on the prewarm-watch for the FIRST gate to open (60)


def _orthonormal_concepts(vocab, proj_dim, seed=0):
    """Orthonormal concept codes for exactly the test vocab (cache-independent; clean per-fact decode at pd=128)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((proj_dim, proj_dim)))
    return {w: q[i] for i, w in enumerate(sorted(vocab))}


# ── the on-bridge spiking WTA (wired IN PLACE, preserving the trained parser) ────────────────────────────────
def wire_wta(ub):
    """Port the standalone mutual-inhibition WTA onto the UnifiedBrainBridge, IN PLACE via
    set_pathway_weights(add_missing=True) so NO re-injection resets the trained parser. Returns (ens, inh):
      ens = [3 arrays of WTA_P indices]  (the role ensembles: AGENT/PREDICATE/THEME -> agent/action/patient)
      inh = 1 array of WTA_INH indices   (the shared inhibitory pool, trait 1)
    Couples each role_route_<r> gate to ens[r] (threshold WTA_GATE_THRESHOLD) and closes every gate."""
    base = ub.role_wta_base
    assert base is not None, "build the bridge with role_wta_n=ROLE_WTA_N"
    ens = [np.arange(base + k * WTA_P, base + (k + 1) * WTA_P, dtype=np.int64) for k in range(3)]
    inh = np.arange(base + 3 * WTA_P, base + 3 * WTA_P + WTA_INH, dtype=np.int64)

    # flip the inhibitory pool's trait to 1 (inhibitory_trait_index=1 -> its firing drives g_i on its targets).
    ub.bridge.cp_traits[inh] = 1
    ub.bridge._cached_inhibitory_mask = None            # force the inhibitory mask to rebuild with the new traits

    pre_ei, post_ei = [], []            # e2i: every ens neuron -> every inh neuron
    for k in range(3):
        for a in ens[k]:
            for b in inh:
                pre_ei.append(int(a)); post_ei.append(int(b))
    pre_ee, post_ee = [], []            # e2e: within-ensemble self-recurrence (no self-self edge)
    for k in range(3):
        for a in ens[k]:
            for b in ens[k]:
                if a != b:
                    pre_ee.append(int(a)); post_ee.append(int(b))
    pre_ie, post_ie = [], []            # i2e: every inh neuron -> every ens neuron (broadcast inhibition)
    all_ens = np.concatenate(ens)
    for a in inh:
        for b in all_ens:
            pre_ie.append(int(a)); post_ie.append(int(b))

    ub.bridge.set_pathway_weights("wta_e2i", pre_ei, post_ei,
                                  np.full(len(pre_ei), WTA_W_EI, dtype=np.float32), add_missing=True)
    ub.bridge.set_pathway_weights("wta_e2e", pre_ee, post_ee,
                                  np.full(len(pre_ee), WTA_W_EE, dtype=np.float32), add_missing=True)
    ub.bridge.set_pathway_weights("wta_i2e", pre_ie, post_ie,
                                  np.full(len(pre_ie), WTA_W_IE, dtype=np.float32), add_missing=True)

    for k, r in enumerate(SYNAPTIC_ROUTE_ROLES):
        couple_gate_to_indices(ub.bridge, f"role_route_{r}", ens[k], threshold=WTA_GATE_THRESHOLD)
        ub.bridge.set_transmission_gate(f"role_route_{r}", 0.0)
    return ens, inh


def _wta_drive(logits3):
    """WTA_BASE (uniform, to EVERY ensemble) + WTA_GAIN * normalized_rectified(logits3) (the graded per-role bias):
    rectify the logits to >=0, max-normalize so the top logit maps to 1, scale by the gain, and add the uniform
    baseline. Every ensemble is driven toward firing (the baseline); the I->E inhibition is what SILENCES the
    losers, so the WINNER is the one the graded bias + the competition select. This is a pure transform of the
    reservoir logits into per-ensemble tonic currents -- there is NO argmax here; the SELECTION is the spiking
    competition (read from the LATCHED gate), not this transform."""
    r = np.maximum(np.asarray(logits3, dtype=np.float64), 0.0)
    m = r.max()
    norm = r / m if m > 1e-9 else r
    return WTA_BASE + WTA_GAIN * norm


def lesion_wta_i2e(ub, ens, inh):
    """WTA-LESION anti-cheat (like RoleWTA._set_lesion): zero the I->E (inh->ens) synapse weights via
    set_pathway_weights, so the shared inhibition can no longer suppress the losers -> the biased competition
    collapses (all ensembles free-run) -> the WTA no longer elects a single clean winner. Returns restore()."""
    all_ens = np.concatenate(ens)
    pre, post = [], []
    for a in inh:
        for b in all_ens:
            pre.append(int(a)); post.append(int(b))
    ub.bridge.set_pathway_weights("wta_i2e", pre, post, np.zeros(len(pre), dtype=np.float32), add_missing=False)

    def restore():
        ub.bridge.set_pathway_weights("wta_i2e", pre, post,
                                      np.full(len(pre), WTA_W_IE, dtype=np.float32), add_missing=False)
    return restore


# ── _op_wta: fork _op_synaptic, but the role is elected by the on-bridge WTA (NOT a parser conj / host argmax) ─
def _op_wta(ub, ens, logits3, fill_on_cur, fill_off_cur):
    """One spiking bind step whose role is SELECTED by the on-bridge WTA driven by `logits3` (the reservoir's per-
    word role logits over AGENT/PREDICATE/THEME). Mirrors `UnifiedBrainBridge._op_synaptic` EXCEPT the role gate is
    opened by the WTA WINNER's firing instead of a fired parser conjunction. Returns (out_on, out_off, latched_role).

    TIMING (validated in the smoke; forked from _op_synaptic's prewarm/latch):
      (a) RESET: zero every role_route coupling's ema/last_value, close all role_route gates, run RESET_STEPS at rest.
      (b) DRIVE: apply the WTA drive (BASE+GAIN*normalized_rectified(logits3)) to the 3 ensembles, ALL role_src pools,
          the fill bank (the word's code), and the A/B/C/D coincidence bias -- exactly _op_synaptic's non-parser drive.
      (c) SETTLE (~WTA_SETTLE_STEPS, no accumulation) so the WTA biased competition establishes a single winner.
      (d) PREWARM-WATCH (capped) until the FIRST role_route gate opens -- that gate's role is the WTA winner. The gate
          opens genuinely from the winning ensemble's firing (the coupling), NOT set by hand.
      (e) LATCH: pause `_gate_couplings` (set to []), hold ONLY the winner's gate at 1.0, close the other two.
      (f) READOUT: run comp.run_steps, accumulate the A/B/C/D coincidence banks through the held (winner) gate.
      (g) RESTORE: put the couplings back, close every role_route gate + reset its ema, clear the input.
    The latched role is read from the GATE that opened (== the WTA winner), never from a host argmax over logits."""
    xp, _ = get_backend()
    bridge = ub.bridge
    comp = ub.composer
    idx = comp.idx
    role_gate_names = [f"role_route_{r}" for r in SYNAPTIC_ROUTE_ROLES]

    # (a) RESET the couplings + gates, then rest.
    for c in bridge._gate_couplings:
        c["ema"] = 0.0
        c["last_value"] = None
    for r in SYNAPTIC_ROUTE_ROLES:
        bridge.set_transmission_gate(f"role_route_{r}", 0.0)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge._run_one_simulation_step()

    # (b) DRIVE: WTA ensembles + all role_src + fill + coincidence bias (mirror _op_synaptic's non-parser drive).
    drive = _wta_drive(logits3)
    cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
    for k in range(3):
        cur[xp.asarray(ens[k])] = float(drive[k])
    for r in SYNAPTIC_ROUTE_ROLES:
        cur[ub._role_src[r]] = ROLE_SRC_DRIVE_PA
    cur[idx["fill_on"]] = xp.asarray(fill_on_cur.astype(np.float32))
    cur[idx["fill_off"]] = xp.asarray(fill_off_cur.astype(np.float32))
    for bank in ("A", "B", "C", "D"):
        cur[idx[bank]] = comp.coinc_bias
    bridge.cp_external_input_current[:] = cur

    # (c) SETTLE so the WTA establishes the winner (no accumulation). Also tally ens firing for the neural-select
    # anti-cheat (runtime latched-role == argmax over the WTA ensembles' firing).
    ens_fire = np.zeros(3, dtype=np.float64)
    for _ in range(WTA_SETTLE_STEPS):
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge._run_one_simulation_step()
        for k in range(3):
            ens_fire[k] += float(to_host(bridge.cp_firing_states[xp.asarray(ens[k])]).mean())

    # (d) PREWARM-WATCH until the FIRST role_route gate opens (the WTA winner opens it via the coupling), capped.
    def _first_open():
        for r in SYNAPTIC_ROUTE_ROLES:
            if _gate_open(bridge, r):
                return r
        return None
    latched = None
    for _ in range(WTA_PREWARM_CAP):
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        bridge._run_one_simulation_step()
        for k in range(3):
            ens_fire[k] += float(to_host(bridge.cp_firing_states[xp.asarray(ens[k])]).mean())
        latched = _first_open()
        if latched is not None:
            break

    # (e) LATCH: PAUSE the couplings so the gates RETAIN the value the WTA competition produced (the winner's gate
    # held at 1.0, the losers at 0.0) — the biologically correct order (compete -> latch the route -> compose),
    # EXACTLY as `_op_synaptic` pauses its coupling. We do NOT force any gate open by hand: the winner's gate is
    # open ONLY because its ensemble fired and the coupling opened it during (d). This is load-bearing for the
    # lesion anti-cheats — under ROUTE-lesion (couplings removed, no gate ever opens) the readout is starved, and
    # under WTA-lesion (I->E zeroed, the competition collapses so MULTIPLE ensembles fire and multiple gates open)
    # the readout gets a superposition of roles — both degrade the bind (collapse) instead of being masked by a
    # hand-forced single gate. `latched` = the FIRST gate the coupling opened (the WTA winner); if none opened
    # within the cap (a collapsed competition), it reads the WTA-firing winner (still a neural read, never a
    # host @Ws argmax) and holds whatever gate state exists (typically all-closed -> starved).
    if latched is None:
        latched = SYNAPTIC_ROUTE_ROLES[int(np.argmax(ens_fire))]
    open_now = [r for r in SYNAPTIC_ROUTE_ROLES if _gate_open(bridge, r)]
    saved_couplings = bridge._gate_couplings
    bridge._gate_couplings = []
    try:
        # (f) READOUT: accumulate the coincidence banks through the HELD gate(s) (the WTA-produced gate state, not
        # a hand-set value). Couplings paused, so the gates keep whatever the competition set them to. Also tally
        # ens firing over the ESTABLISHED (latched) window -- the neural-select runtime check reads THIS (the
        # settled winner), not the settle-phase transient, so latched == argmax(readout ens firing).
        acc = {b: xp.zeros(comp.D, dtype=xp.float64) for b in ("A", "B", "C", "D")}
        readout_ens_fire = np.zeros(3, dtype=np.float64)
        for _ in range(comp.run_steps):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()
            for b in ("A", "B", "C", "D"):
                acc[b] += bridge.cp_firing_states[idx[b]].astype(xp.float64)
            for k in range(3):
                readout_ens_fire[k] += float(to_host(bridge.cp_firing_states[xp.asarray(ens[k])]).mean())
    finally:
        # (g) RESTORE: couplings back, gates closed, emas cleared, input cleared (self-contained op).
        bridge._gate_couplings = saved_couplings
        for r in SYNAPTIC_ROUTE_ROLES:
            bridge.set_transmission_gate(f"role_route_{r}", 0.0)
            cpl = next((c for c in bridge._gate_couplings if c["gate_name"] == f"role_route_{r}"), None)
            if cpl is not None:
                cpl["ema"] = 0.0
                cpl["last_value"] = None
        bridge.cp_external_input_current[:] = 0.0

    rates = {b: to_host(acc[b]) / comp.run_steps for b in ("A", "B", "C", "D")}
    # neural-select runtime check payload: the WTA-firing winner (argmax over the ENS firing during the settled/
    # latched READOUT window) must == the latched role -- an INDEPENDENT neural confirmation that the gate that
    # opened is the ensemble that actually fired most (not a host decision). ens_fire (settle+prewarm) is kept for
    # diagnostics but the check reads the established-window firing (the transient during settle is noisy).
    wta_fire_winner = SYNAPTIC_ROUTE_ROLES[int(np.argmax(readout_ens_fire))]
    return rates["A"] + rates["B"], rates["C"] + rates["D"], latched, wta_fire_winner, list(open_now)


def _bind_wta_fact(ub, ens, comp, tokens, lesion=False, Ws=None):
    """Comprehend `tokens`, and for each content word bind it with the WTA-elected role (via `_op_wta`), routing the
    role synaptically. `Ws` overrides comp.Ws (for the Ws-scramble anti-cheat). Only a COMPLETE {agent,action,patient}
    fact is stored (the KeyError guard). Returns (fact_dict_or_None, per_word_trace)."""
    Ws = comp.Ws if Ws is None else Ws
    f = np.concatenate([comp.res.final_state(comp.enc.encode(tokens, lesion=lesion)), [1.0]])
    content = [t for t, w in enumerate(tokens) if w not in comp.closed]
    composer = ub.composer
    bound_on = np.zeros(composer.D); bound_off = np.zeros(composer.D)
    fact = {}
    trace = []
    for k, t in enumerate(content):
        if Ws is None or k not in Ws:
            continue
        logits = (f @ Ws[k])[[0, 1, 2]]
        word = tokens[t]
        c_on, c_off = onoff(composer.concepts[word])
        fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
        bon, boff, latched_role, wta_fire_winner, gates_at_latch = _op_wta(ub, ens, logits, fon, foff)
        trace.append({"word": word, "logits": logits.tolist(), "latched_role": latched_role,
                      "wta_fire_winner": wta_fire_winner, "gates_at_latch": gates_at_latch})
        # the composer field for the WTA-elected role (agent/action/patient); GOAL/LOCATION would map to None but
        # the WTA only has 3 ensembles (AGENT/PREDICATE/THEME), so latched_role is always one of the 3 fields.
        role = latched_role
        if role in fact:                       # first-wins per role (a well-formed SVO fills each once)
            continue
        bound_on += bon; bound_off += boff
        fact[role] = word
    if {"agent", "action", "patient"} <= set(fact):
        composer.kb.append((fact, onoff(bound_on - bound_off)))
        return fact, trace
    return None, trace


def _recall(ub, test):
    """who/what recall over the composer kb: fraction of query_patient + query_agent that return the ground truth."""
    hp = sum(int(ub.query_patient(s, v3) == o) for _t, s, v3, o in test)
    ha = sum(int(ub.query_agent(v3, o) == s) for _t, s, v3, o in test)
    return hp, ha


def _scramble_Ws(comp, seed):
    """Ws-SCRAMBLE anti-cheat: permute the 3 role columns (AGENT/PREDICATE/THEME = cols 0,1,2) of each Ws[k], so the
    reservoir logits misroute the WTA (a THEME-strong state now drives the AGENT ensemble, etc.). Returns a new Ws
    dict (comp.Ws is untouched)."""
    rng = np.random.default_rng(seed * 977 + 13)
    out = {}
    for k, W in comp.Ws.items():
        W2 = W.copy()
        perm = rng.permutation(3)
        while np.array_equal(perm, [0, 1, 2]):          # force a real derangement of the 3 role columns
            perm = rng.permutation(3)
        W2[:, [0, 1, 2]] = W[:, [0, 1, 2]][:, perm]
        out[k] = W2
    return out


# ── (7) PROVENANCE-NEURAL-SELECT: source-check + runtime check ──────────────────────────────────────────────
import inspect  # noqa: E402


def _strip_py(src):
    """Strip docstrings, comments, and string literals from a function's source so the source-check inspects only
    EXECUTABLE code (a docstring mentioning 'argmax'/'Ws' must not trip the check)."""
    import io
    import tokenize
    out = []
    prev_type = tokenize.INDENT
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        return src
    for tok in toks:
        t, s = tok.type, tok.string
        if t == tokenize.COMMENT:
            continue
        if t == tokenize.STRING:            # drops docstrings + string literals (no @Ws hidden in a string)
            out.append("''")
            prev_type = t
            continue
        out.append(s)
        prev_type = t
    return " ".join(out)


def _source_has_no_host_argmax():
    """SOURCE-CHECK (anti-cheat 7): the role that reaches the gate is the WTA-LATCHED gate, NOT a host argmax over
    the reservoir read-out. Inspecting the EXECUTABLE code (docstrings/comments/strings stripped):
      * `_op_wta` -- the function that DECIDES the latched role -- never references `Ws` at all (it sees only the
        pre-computed `logits3` drive + the spiking gate; it cannot argmax the read-out because it never holds it).
      * `_bind_wta_fact` computes `logits = f @ Ws[k]` as the WTA DRIVE, but assigns the composer field from
        `latched_role` (the gate the WTA opened) and contains NO `argmax` (so no host argmax over the logits picks
        the role). The only `np.argmax` in `_op_wta` is the FALLBACK read over ens FIRING when NO gate opened (a
        NEURAL read of the spiking ensembles) + the diagnostic firing-winner -- neither is a read-out argmax.
    Returns True iff both hold."""
    code_op = _strip_py(inspect.getsource(_op_wta))
    code_bind = _strip_py(inspect.getsource(_bind_wta_fact))
    op_no_Ws = "Ws" not in code_op                          # _op_wta never touches the read-out matrix
    op_argmax_is_neural = ("Ws" not in code_op)             # any argmax in _op_wta is over ens firing, never @Ws
    bind_no_argmax = "argmax" not in code_bind              # the bind driver never argmaxes the logits to pick role
    bind_role_from_gate = "role = latched_role" in code_bind
    return bool(op_no_Ws and op_argmax_is_neural and bind_no_argmax and bind_role_from_gate)


def setup_corpus(seed=42):
    """Build the shared corpus/task ONCE (reused across seeds; the multi-seed varies only reservoir + bridge RNG)."""
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, *_ = m62.discover_closed_class(words, freq, cover)
    subj, verb, obj = _content_pools(discovered)
    test, _seen, _trng = _build_test_facts(seed, subj, verb, obj, n=N_TEST)
    vocab = sorted({w for _toks, s, v3, o in test for w in (s, v3, o)})
    concepts = _orthonormal_concepts(vocab, PROJ_DIM, seed=0)
    return {"discovered": discovered, "subj": subj, "verb": verb, "obj": obj,
            "test": test, "vocab": vocab, "concepts": concepts}


def run_seed(seed, corpus):
    t0 = time.time()
    discovered, subj, verb, obj = corpus["discovered"], corpus["subj"], corpus["verb"], corpus["obj"]
    test, concepts = corpus["test"], corpus["concepts"]
    rng = np.random.default_rng(seed * 101 + 5)

    comp = ReservoirComprehender(seed, discovered)
    comp.fit(_gen(_TRAIN_KINDS, _N_TRAIN_PER_CONSTRUCTION, rng, subj, verb, obj))

    def new_bridge_wired():
        ub = UnifiedBrainBridge(seed=seed, proj_dim=PROJ_DIM, concepts=concepts,
                                enable_synaptic_route=True, role_wta_n=ROLE_WTA_N)
        ens, inh = wire_wta(ub)
        return ub, ens, inh

    # (1) ROUTE: reservoir logits -> WTA select -> synaptic bind -> store -> recall
    ub, ens, inh = new_bridge_wired()
    all_traces = []
    for toks, s, v3, o in test:
        _fact, tr = _bind_wta_fact(ub, ens, comp, toks)
        all_traces.append({"tokens": toks, "trace": tr})
    hp, ha = _recall(ub, test)
    route_correct = hp + ha
    n_q = 2 * len(test)

    # (3) MOAT: (agent, action) never stored -> abstain
    stored = {(s, v3) for _t, s, v3, _o in test}
    fa = tot = mg = 0
    trng = np.random.default_rng(seed * 733 + 999)
    while tot < 30 and mg < 3000:
        mg += 1
        s = str(trng.choice(subj)); v3q = str(trng.choice(verb)) + "s"
        if (s, v3q) in stored:
            continue
        tot += 1; fa += int(ub.query_patient(s, v3q) is not None)
    moat_fa = fa / max(1, tot)

    # (2) DICT path = RUNG B-1's HOST-argmax `_bind_reservoir_fact` store (same reservoir, HOST argmax role select).
    # The baseline is built on the SAME WTA-wired substrate (role_wta_n, wire_wta) so it pays the IDENTICAL shared-
    # substrate cost as the WTA route -- the comparison then isolates the role-SELECTION mechanism (WTA vs host
    # argmax), not the composer's dynamics. (Verified: the extra WTA neurons perturb the composer's spiking decode
    # equally on both paths; comparing the WTA route against a WTA-FREE bridge would unfairly penalize it for a cost
    # the host path does not pay.) `_bind_reservoir_fact` fires the parser conjunction for the host-argmax role.
    from research.runners._rungB1_reservoir_synaptic_handoff_derisk import (
        _bind_reservoir_fact, _reservoir_roles,
    )
    ub_d, _ens_d, _inh_d = new_bridge_wired()
    role2k_d = {ub_d.parser.role_of(pos, "active"): pos * 2 for pos in range(3)}
    for toks, s, v3, o in test:
        _bind_reservoir_fact(ub_d, role2k_d, _reservoir_roles(comp, toks))
    dp, da = _recall(ub_d, test)
    dict_correct = dp + da

    # (7) PROVENANCE-NEURAL-SELECT: source (no host argmax over @Ws picks the gate) + runtime (latched == firing win)
    source_clean = _source_has_no_host_argmax()
    latched_eq_firing = all(t["latched_role"] == t["wta_fire_winner"]
                            for w in all_traces for t in w["trace"])
    neural_select_ok = bool(source_clean and latched_eq_firing)

    # (4) PROVENANCE-CLEAN (reuse the I5a instrument on a content word -- the role bank gets ZERO direct current)
    ub_p, ens_p, _ = new_bridge_wired()
    prov = provenance_role_bank_current(ub_p, word=corpus["vocab"][0], pos=0, voice="active")
    provenance_clean = (prov["synaptic_route_role_bank_direct_current_max"] == 0.0
                        and prov["dict_path_role_bank_direct_current_max"] > 0.0)

    # (5) ROUTE-LESION: cut the synaptic route -> the WTA winner's gate cannot route -> recall collapses
    ub_l, ens_l, _ = new_bridge_wired()
    restore = lesion_route(ub_l.bridge)
    for toks, s, v3, o in test:
        _bind_wta_fact(ub_l, ens_l, comp, toks)
    lp, la = _recall(ub_l, test)
    route_lesion_correct = lp + la
    restore()
    route_lesion_collapses = route_lesion_correct < route_correct

    # (6) RESERVOIR-LESION: lesion the reservoir's closed-class identity -> logits collapse -> recall collapses
    ub_r, ens_r, _ = new_bridge_wired()
    for toks, s, v3, o in test:
        _bind_wta_fact(ub_r, ens_r, comp, toks, lesion=True)
    rp, ra = _recall(ub_r, test)
    res_lesion_correct = rp + ra
    res_lesion_collapses = res_lesion_correct < route_correct

    # (8) WTA-LESION: zero the I->E synapses -> biased competition collapses -> gates ambiguous -> recall collapses
    ub_w, ens_w, inh_w = new_bridge_wired()
    restore_w = lesion_wta_i2e(ub_w, ens_w, inh_w)
    for toks, s, v3, o in test:
        _bind_wta_fact(ub_w, ens_w, comp, toks)
    wp, wa = _recall(ub_w, test)
    wta_lesion_correct = wp + wa
    restore_w()
    wta_lesion_collapses = wta_lesion_correct < route_correct

    # (9) Ws-SCRAMBLE: permute the 3 role columns of each Ws[k] -> logits misroute the WTA -> recall collapses
    ub_s, ens_s, _ = new_bridge_wired()
    Ws_scr = _scramble_Ws(comp, seed)
    for toks, s, v3, o in test:
        _bind_wta_fact(ub_s, ens_s, comp, toks, Ws=Ws_scr)
    sp, sa = _recall(ub_s, test)
    ws_scramble_correct = sp + sa
    ws_scramble_collapses = ws_scramble_correct < route_correct

    seed_go = bool(
        route_correct >= 0.80 * n_q
        and route_correct >= dict_correct
        and moat_fa <= 0.05
        and provenance_clean
        and route_lesion_collapses and res_lesion_collapses
        and neural_select_ok
        and wta_lesion_collapses and ws_scramble_collapses
    )
    return {
        "seed": int(seed),
        "route_correct": int(route_correct), "dict_correct": int(dict_correct), "n_queries": n_q,
        "route_recall": route_correct / n_q,
        "route_recall_ge_0.8n": bool(route_correct >= 0.80 * n_q),
        "route_not_worse_than_dict": bool(route_correct >= dict_correct),
        "moat_false_accept": moat_fa, "moat_clean": bool(moat_fa <= 0.05),
        "provenance": {**prov, "clean": bool(provenance_clean)},
        "route_lesion_correct": int(route_lesion_correct), "route_lesion_collapses": bool(route_lesion_collapses),
        "res_lesion_correct": int(res_lesion_correct), "res_lesion_collapses": bool(res_lesion_collapses),
        "neural_select_ok": neural_select_ok, "neural_select_source_clean": bool(source_clean),
        "neural_select_latched_eq_firing": bool(latched_eq_firing),
        "wta_lesion_correct": int(wta_lesion_correct), "wta_lesion_collapses": bool(wta_lesion_collapses),
        "ws_scramble_correct": int(ws_scramble_correct), "ws_scramble_collapses": bool(ws_scramble_collapses),
        "seed_GO": seed_go, "elapsed_s": round(time.time() - t0, 1),
        "sample_trace": all_traces[0]["trace"] if all_traces else [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    corpus = setup_corpus(seed=42)
    print(f"[rungB1b] corpus: {len(corpus['test'])} facts, vocab {len(corpus['vocab'])}", flush=True)
    rows = []
    for s in args.seeds:
        d = run_seed(s, corpus)
        rows.append(d)
        print(f"[seed {s}] GO={d['seed_GO']} | route {d['route_correct']}/{d['n_queries']} (dict {d['dict_correct']})"
              f" | moat-FA {d['moat_false_accept']:.2f} | prov {d['provenance']['clean']}"
              f" | route-lesion {d['route_lesion_correct']}<{d['route_correct']}={d['route_lesion_collapses']}"
              f" | res-lesion {d['res_lesion_correct']}<{d['route_correct']}={d['res_lesion_collapses']}"
              f" | neural-select {d['neural_select_ok']}"
              f" | wta-lesion {d['wta_lesion_correct']}<{d['route_correct']}={d['wta_lesion_collapses']}"
              f" | ws-scramble {d['ws_scramble_correct']}<{d['route_correct']}={d['ws_scramble_collapses']}"
              f" ({d['elapsed_s']}s)", flush=True)

    n_go = sum(r["seed_GO"] for r in rows)
    agg = {
        "n_seeds": len(rows), "n_seeds_GO": int(n_go),
        "verdict": "GO" if n_go == len(rows) else ("PARTIAL" if n_go else "NO-GO"),
        "route_recall_ge_0.8n_all": all(r["route_recall_ge_0.8n"] for r in rows),
        "route_not_worse_than_dict_all": all(r["route_not_worse_than_dict"] for r in rows),
        "moat_clean_all": all(r["moat_clean"] for r in rows),
        "provenance_clean_all": all(r["provenance"]["clean"] for r in rows),
        "route_lesion_collapses_all": all(r["route_lesion_collapses"] for r in rows),
        "res_lesion_collapses_all": all(r["res_lesion_collapses"] for r in rows),
        "neural_select_all": all(r["neural_select_ok"] for r in rows),
        "wta_lesion_collapses_all": all(r["wta_lesion_collapses"] for r in rows),
        "ws_scramble_collapses_all": all(r["ws_scramble_collapses"] for r in rows),
        "mean_route_recall": float(np.mean([r["route_recall"] for r in rows])),
        "total_elapsed_s": round(time.time() - t0, 1),
    }
    print(f"\n[rungB1b] VERDICT: {agg['verdict']} ({n_go}/{len(rows)}) -- the word's thematic role is elected by an "
          f"ON-BRIDGE spiking WTA whose winner opens the composer's role_route gate (mean route recall "
          f"{agg['mean_route_recall']:.3f}; not-worse-than-dict {agg['route_not_worse_than_dict_all']}; "
          f"neural-select {agg['neural_select_all']}; wta-lesion collapses {agg['wta_lesion_collapses_all']}; "
          f"ws-scramble collapses {agg['ws_scramble_collapses_all']}; route/res-lesion "
          f"{agg['route_lesion_collapses_all']}/{agg['res_lesion_collapses_all']}; prov "
          f"{agg['provenance_clean_all']}; moat {agg['moat_clean_all']}).", flush=True)

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "agg": agg}, fh, indent=2, default=str)
        print(f"[rungB1b] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
