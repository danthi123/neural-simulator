"""CAUSAL WHY / WHAT-IF ORGAN wired into the PRODUCTION conversational turn (Gate-B, T1-4, 2026-08-13).

A co-resident spiking CAUSAL FORWARD MODEL, grounded in the brain's REAL fact store, that answers a real
"what happens if <agent> <action>?" (forward-SIMULATION of an unseen consequence) and "why did <agent>
<action>?" (the directed cause that survives a Pearl DO-probe) — the reasoning rung a host triple-JOIN
cannot serve. It REUSES (does not reinvent) the adversarially-verified grounded de-risk
(`research/runners/_causal_forward_model_grounded_derisk.py`, 6/6 GO) + its toy primitives
(`_causal_forward_model_derisk.py`): a directed n-way spiking next-state forward model on the Izhikevich
bridge whose transition edges are DIRECTED + CAUSAL via temporal-order STDP (Mehta-Blum-Abbott) + phasic-DA
three-factor plasticity, with a DO-intervention that PRUNES a confounded correlation.

HOW IT MAPS ONTO A TURN
-----------------------
The canonical causal world the 6/6-GO de-risk validated (every state a real (agent,action)->patient fact):
  CHAIN     A=(dog,go,east) -> B=(dog,reach,river) -> D=(dog,drink,water)   taught as ADJACENT pairs, so the
            "what happens if the dog goes east?" consequence D is a substrate ROLLOUT (A->D is NEVER a taught edge).
  CONFOUND  C=(sun,rise,sky) is the COMMON CAUSE of X=(bird,sing,dawn) and Y=(dog,wake,morning); X precedes Y
            -> temporal-order STDP tags a SPURIOUS X->Y that the DO-intervention must prune, so "why did the dog
            wake?" reads the DO-surviving cause C ("because the sun rose"), never the correlate X ("the bird sang").

THE GROUNDING — READ-ONLY against the LIVE production composer (grounding-by-DERIVATION)
---------------------------------------------------------------------------------------
The organ is grounded in the SAME production RF-VSA fact store the live chat recall uses (`chat.inner.composer`,
the RFPhasorComposer whose `query_patient(agent,action)->patient` IS the no-confab moat). Three load-bearing
bindings, ALL read-only (the organ NEVER writes a fact — it grounds against WHATEVER the brain already learned):
  (1) EVENT SET DERIVED FROM THE COMPOSER — a canonical event exists ONLY if the live composer moat-recalls it
      (`enumerate_events`, the spiking RF unbind). No stored fact -> no event -> the organ cannot reason about it.
  (2) THE CAUSAL CURRICULUM IS GATED BY RECALL — a causal episode (fact_i then fact_j) is trained ONLY when BOTH
      endpoints are moat-recalled. A fact the brain never learned -> its event vanishes -> every edge touching it
      never forms -> the downstream why/what-if for it COLLAPSES to abstain (the GROUNDING lesion, driven by the
      live brain's real knowledge — this is real grounding, not a toy under a new label).
  (3) THE ANSWERS ARE REAL RECALLED FACTS, MOAT-SAFE — the what-if consequence and the why-cause are each mapped
      BACK to a fact and CONFIRMED by `query_patient`. A predicted consequence the composer cannot confirm is
      REJECTED (0 confabulation). The organ reads/notices; it NEVER manufactures a fact.

MOAT-SAFE + ADDITIVE: the organ ONLY answers a DISJOINT turn class (an explicit "what happens if <X>" /
"why did <X>" whose (agent,action) maps to a grounded canonical causal event) with a moat-confirmed fact, and
ABSTAINS to the honest `_honest_causal_answer` disclaimer otherwise (INTEGRATION #5 fallback). It never flips an
existing abstain into a fact, never changes WHICH answer the recall path produces, and never enters the recall
turn class ("what does <X> do") — a query it does not map to a grounded causal event falls through unchanged.

LESION-LOAD-BEARING: zeroing the learned cross-block forward edges (`BRAIN_CAUSAL_LESION=1`, the de-risk's
`_lesion_xblock`) removes the directed transition structure, so the forward-simulation cannot roll A->B->D
(what-if collapses) and the DO-probe predecessor of Y is no longer C (why collapses) -> BOTH why/what-if fall to
the honest abstain. The answers are therefore CAUSED by the learned SPIKING forward edges, not the host drive.

HONEST RESIDUALS (declared — the mission's named NEXT RUNGS, not caveats):
  * GROUNDING-BY-DERIVATION, not shared-SUBSTRATE-merge — the events are DERIVED from + gated by the composer's
    moat recall (and the answers re-confirmed by it), but the composer's unbind SPIKES do not yet directly DRIVE
    the forward-model event blocks in ONE merged bridge. The organ runs on its OWN co-resident forward-model
    bridge ALONGSIDE the recall composer (rides on the one-brain merge, burn-down #1), exactly as the affect /
    surprise / world-model organs do. Driving the event blocks from the composer's unbind spikes is the next rung.
  * THE DA SIGN IS TEACHER-DELIVERED — the temporal ORDER of each causal episode + the phasic-DA sign are the
    environment/teacher reinforcement (the brain's dopamine channel converts it to a weight change). Driving the
    DA from a spiking mismatch unit (E2's surprise read -> from_reward/from_novelty DA) is the named next rung.
  * FIRST-ORDER + FIXED CAUSAL CURRICULUM — the model is state->next (Markov-1; high-order needs the HTM-TM
    predictor, EMERGE-15 GO), and the canonical CHAIN/CONFOUND causal STRUCTURE (which fact causes which) is
    teacher-rendered (the environment boundary); learning WHICH facts causally connect from raw co-occurrence is
    the deeper relational-code arc. The wired SCOPE is the validated chain-source what-if (A->D) + confound why
    (Y<-C); a why/what-if outside that grounded structure abstains honestly.

NO `sim/` edit; reuse-by-import; process backend (cupy in production, numpy in tests). Default-ON;
`BRAIN_CAUSAL=0` -> fully skipped (byte-identical oracle).
"""
from __future__ import annotations

import os
import re

# Reuse the toy forward-model primitives (build/step/train/reads/lesion) verbatim — the SAME instrument the
# 6/6-GO grounded de-risk assembles; NO reimplementation.
from research.runners._causal_forward_model_derisk import (  # noqa: E402
    build_forward_model, train, unseen_consequence, do_intervention,
    _lesion_xblock, _xblock_weight, OBS_EPISODES, EVENT_NAMES, N_EVENTS,
)
# Reuse the grounded de-risk's canonical real-fact world + composer-grounding bindings (event set derived from
# the composer, recall-gated curriculum, moat-confirmed answers, the spiking DO-probe "why" read).
from research.runners._causal_forward_model_grounded_derisk import (  # noqa: E402
    A, B, D, C, X, Y, FACTS, FACT_ORDER,
    enumerate_events, moat_battery, why_cause, _recalled,
)

# ── the DISJOINT turn class this organ owns (kept narrow so it never hijacks a recall / follow-up turn) ──────
# what-if: "what happens if <agent> <action> ..." / "what would happen if ..." / "what if ..."
_WHATIF_RE = re.compile(
    r"\bwhat\s+(?:happens?|would\s+happen)\s+(?:if|when)\b|\bwhat\s+if\b", re.IGNORECASE)
# why: "why did/does/do/is/are/was/were <agent> <action> ..." — REQUIRES an auxiliary so a bare "why?" follow-up
# (handled by the rich discourse thread) is NOT intercepted.
_WHY_RE = re.compile(r"\bwhy\s+(?:did|does|do|is|are|was|were)\b", re.IGNORECASE)

# Surface->lemma verb normalisation for the canonical causal actions (go/reach/drink/rise/sing/wake). Explicit
# irregulars first, then a light suffix strip. Purely a host SENSORY-encoding boundary (surface-tagging the input,
# exactly the D4/B3 lemmatizer status); the DECISION is the moat read + the spiking forward model.
_VERB_IRREGULAR = {
    "goes": "go", "going": "go", "went": "go", "gone": "go",
    "wakes": "wake", "waking": "wake", "woke": "wake", "woken": "wake",
    "rises": "rise", "rising": "rise", "rose": "rise", "risen": "rise",
    "sings": "sing", "singing": "sing", "sang": "sing", "sung": "sing",
    "reaches": "reach", "reaching": "reach", "reached": "reach",
    "drinks": "drink", "drinking": "drink", "drank": "drink", "drunk": "drink",
}


def causal_enabled() -> bool:
    """Default-ON. `BRAIN_CAUSAL` in {0,false,no,off} -> the byte-identical oracle (fully disabled)."""
    v = os.environ.get("BRAIN_CAUSAL")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def causal_lesioned() -> bool:
    """`BRAIN_CAUSAL_LESION` in {1,true,yes,on} -> zero the learned forward edges (load-bearing)."""
    v = os.environ.get("BRAIN_CAUSAL_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def _norm_verb(tok: str) -> str:
    t = (tok or "").lower().strip(".,!?;:'\"")
    if t in _VERB_IRREGULAR:
        return _VERB_IRREGULAR[t]
    for suf in ("ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) > len(suf) + 1:
            return t[: -len(suf)]
    return t


def is_causal_query(text: str) -> str | None:
    """Returns 'what_if' | 'why' | None. The surface gate for the disjoint causal turn class."""
    if _WHATIF_RE.search(text or ""):
        return "what_if"
    if _WHY_RE.search(text or ""):
        return "why"
    return None


def extract_cue(text: str, agents=None):
    """Extract the (agent, action-lemma) the query is ABOUT and map it to the canonical causal event index (or
    None if it maps to no grounded canonical fact). Returns (event_index_or_None, agent, action_lemma). `agents`
    (the brain's real agent vocabulary, e.g. `chat.agents_set`) lets a KNOWN-but-non-canonical agent still be
    recognised so its causal query abstains HONESTLY (rather than falling through); the canonical-world agents are
    always included. Pure surface parse (a declared host sensory-encoding boundary; the DECISION is the moat read)."""
    toks = re.findall(r"[a-z]+", (text or "").lower())
    tokset = set(toks)
    canon_agents = {FACTS[e][0] for e in FACT_ORDER}
    known_agents = canon_agents | {str(a).lower() for a in (agents or set())}
    # the agent = the first known agent word present (canonical world first, then the brain's real vocab).
    agent = next((t for t in toks if t in canon_agents), None) or next((t for t in toks if t in known_agents), None)
    if agent is None:
        return None, None, None
    lemmas = {_norm_verb(t) for t in toks}
    # find the canonical event whose (agent,action) both appear (action matched by lemma) -> the grounded cue.
    for e in FACT_ORDER:
        fa, fv, _fp = FACTS[e]
        if fa == agent and (fv in lemmas or fv in tokset):
            return e, fa, fv
    # no canonical action matched: surface the agent + the best action lemma present (for the honest abstain).
    action = next((_norm_verb(t) for t in toks
                   if _norm_verb(t) not in known_agents and _norm_verb(t) != agent and len(_norm_verb(t)) > 1), None)
    return None, agent, action


# ── the verbatim moat-confirmed answers (identical wording to the 6/6-GO grounded de-risk `run_seed`) ────────
def what_if_readout() -> str:
    a, v, _p = FACTS[A]
    _, _, dp = FACTS[D]
    return (f"If the {a} {v}es {_p}, it will {FACTS[D][1]} {dp} — a consequence I rolled forward through "
            f"{FACTS[B][0]} {FACTS[B][1]}ing the {FACTS[B][2]}, and my no-confab moat confirms "
            f"({FACTS[D][0]},{FACTS[D][1]})->{dp} is a fact I stored.")


def why_readout() -> str:
    ya, yv, yp = FACTS[Y]
    ca, cv, cp = FACTS[C]
    return (f"The {ya} {yv}s ({yp}) because the {ca} {cv}s — that cause survives a DO-probe "
            f"(forcing the {ca} to {cv} makes the {ya} {yv}; forcing the {FACTS[X][0]} to {FACTS[X][1]} does "
            f"NOT), so it is a cause not a mere correlation, and ({ca},{cv})->{cp} is a fact I stored.")


class CausalWhatIfProductionOrgan:
    """A process/session-shared spiking causal forward model, grounded READ-ONLY in a live production composer.
    Built ONCE per composer (lazily): the ~180-neuron directed forward-model bridge, curriculum GATED by the
    composer's moat recall, TRAINED (temporal-order STDP + phasic-DA three-factor) then FROZEN + matured. Each
    turn: `what_if(composer)` rolls the substrate forward + moat-confirms the consequence; `why(composer)` reads
    the DO-surviving directed cause + moat-confirms it. A lesioned twin (forward edges zeroed) is built lazily.

    ONE-BRAIN WAVE-2 SINGLE-POOL (opt-in, byte-identical when `shared` is None): when a Wave-2 `MergedPool` is
    injected, the "evt" causal population is this organ's SLICE of the shared 9-organ spiking bridge instead of
    its own standalone bridge (the declared next rung named in this file's own docstring: "rides on the one-brain
    merge, burn-down #1"). The RECALL-GATED curriculum (bindings 1+2) is UNCHANGED — only the substrate the evt
    slice trains on changes. Two co-residence hazards, both handled: (1) TRAINING briefly needs
    `cp_plasticity_rate_gain=1` on evt-internal edges (`_freeze_non_evt`) — restored to the pool's own frozen
    array immediately after, so every OTHER organ's plasticity gate is untouched; (2) a LIVE `what_if`/`why` read
    needs `propagation_strength` raised to the organ's read-operating-point, but that is a GLOBAL bridge-wide
    scalar (`sim/bridge.py`'s per-spike conductance gain) — leaving it raised would rescale every co-resident
    organ's synaptic transmission, so `_shared_read_guard` raises it ONLY for the duration of one read, under
    `pool.read_isolation("causal_whatif")`, and restores the pool's baseline value immediately after. The lesion
    twin ALWAYS builds its own standalone bridge (never the shared pool — a diagnostic, not the production path).
    HONEST RESIDUAL: the shared pool is memoized PER SEED (not per composer/key), so two DIFFERENT composer
    sessions sharing one seed would retrain the SAME physical evt slice sequentially (the second overwrites the
    first's learned weights) — a genuine cross-session collision the standalone (default) path does not have;
    declared here, not solved (single-composer-per-seed is the smoke-verified operating point)."""

    def __init__(self, seed: int = 42, obs_reps: int = 30, interv_reps: int = 30, read_prop: float = 0.50,
                shared=None):
        self.seed = int(seed)
        self.obs_reps = int(obs_reps)
        self.interv_reps = int(interv_reps)
        self.read_prop = float(read_prop)
        # ONE-BRAIN MERGE (opt-in, byte-identical when None): see class docstring.
        self._shared = shared
        self._built = False
        self._st = None            # intact circuit state
        self._les = None           # lazily-built edge-lesioned twin (ALWAYS standalone, never shared)
        self._composer = None
        self.recall_status = None
        self.recalled_events = None
        self.battery_fa = None

    @staticmethod
    def _freeze_non_evt(b, evt_arr, xp):
        """Universal gain-0 freeze of every edge that is NOT evt-internal (both endpoints in evt), for the
        DURATION of this organ's own build-time train only (restored to the pool's own frozen array right after
        by the caller). Mirrors `onebrain_merge_framework._CausalReadOrgan._freeze_non_evt` exactly — tocoo()
        preserves `cp_connections.data` order, so the mask aligns with `cp_plasticity_rate_gain`."""
        import numpy as np
        from research.runners._spiking_expectation_rpe_derisk import _host
        coo = b.cp_connections.tocoo()
        row = np.asarray(_host(coo.row)); col = np.asarray(_host(coo.col))
        row_in = np.isin(row, evt_arr); col_in = np.isin(col, evt_arr)
        both = row_in & col_in
        ng = np.zeros(row.shape[0], dtype=np.float32)
        ng[both] = 1.0
        b.cp_plasticity_rate_gain = xp.asarray(ng, dtype=xp.float32)

    def _build_shared(self, episodes, do_interv, recall_status, recalled_events, xp) -> dict:
        """The evt slice's build+train on the SHARED Wave-2 pool bridge (see class docstring for the two
        co-residence hazards this handles). Returns the same `dict` shape `_build_one` returns (plus
        `shared=True`) so `what_if`/`why` need only branch on that one key."""
        import numpy as np
        pool = self._shared
        pool.ensure_built()
        b = pool.bridge
        cc = b.core_config
        blk, n_events = 30, N_EVENTS
        evt = np.asarray(b.region_manager.indices("evt"), dtype=np.int64)
        blocks = [evt[e * blk:(e + 1) * blk] for e in range(n_events)]
        b._blocks = blocks
        b._blk = blk
        snap = pool.snap or {}
        b._rest_v = (snap["cp_membrane_potential_v"].copy()
                    if "cp_membrane_potential_v" in snap else b.cp_membrane_potential_v.copy())
        b._rest_u = (snap["cp_recovery_variable_u"].copy()
                    if "cp_recovery_variable_u" in snap else b.cp_recovery_variable_u.copy())
        meta = dict(n_events=n_events, blk=blk)

        _cfg_keys = ("enable_stdp", "stdp_a_plus", "stdp_a_minus", "stdp_tau_plus_ms", "stdp_tau_minus_ms",
                    "stdp_w_max", "stdp_w_min", "enable_reward_modulation", "reward_defer_stdp_weight_update",
                    "reward_learning_rate", "reward_eligibility_tau_ms", "reward_baseline",
                    "current_reward_signal", "reward_aversive_scale", "propagation_strength")
        saved = {k: getattr(cc, k) for k in _cfg_keys}
        saved_tstep = b.runtime_state.current_time_step
        saved_tms = b.runtime_state.current_time_ms
        g0 = getattr(b, "cp_plasticity_rate_gain", None)
        saved_gain = g0.copy() if g0 is not None else None
        saved_elig = getattr(b, "cp_eligibility_trace", None)
        saved_lst = getattr(b, "cp_last_spike_time", None)
        nnz = int(b.cp_connections.nnz)
        n_all = int(b.cp_membrane_potential_v.shape[0])
        w_AD = None
        with pool.read_isolation("causal_whatif"):
            try:
                cc.enable_stdp = True
                cc.stdp_a_plus = 0.02; cc.stdp_a_minus = 0.010
                cc.stdp_tau_plus_ms = 12.0; cc.stdp_tau_minus_ms = 12.0
                cc.stdp_w_max = 24.0; cc.stdp_w_min = 0.0
                cc.enable_reward_modulation = True
                cc.reward_defer_stdp_weight_update = True
                cc.reward_learning_rate = 0.18; cc.reward_eligibility_tau_ms = 150.0
                cc.reward_baseline = 0.0; cc.current_reward_signal = 0.0
                cc.reward_aversive_scale = 1.0; cc.propagation_strength = 0.05
                if saved_elig is None:
                    b.cp_eligibility_trace = xp.zeros(nnz, dtype=xp.float32)
                if saved_lst is None:
                    b.cp_last_spike_time = xp.full(n_all, -1000.0, dtype=xp.float32)
                self._freeze_non_evt(b, evt, xp)
                train(b, cc, meta, xp, episodes, obs_reps=self.obs_reps, interv_reps=self.interv_reps,
                     do_intervention=do_interv, prune_src=X)
                w_AD = _xblock_weight(b, A, D)   # the direct A->D edge must stay unlearned (unseen-consequence guard)
            finally:
                for k, v in saved.items():
                    setattr(cc, k, v)
                b.runtime_state.current_time_step = saved_tstep
                b.runtime_state.current_time_ms = saved_tms
                if saved_gain is not None:
                    b.cp_plasticity_rate_gain[:] = saved_gain
                b.cp_eligibility_trace = saved_elig
                b.cp_last_spike_time = saved_lst
        return {"bridge": b, "cfg": cc, "meta": meta, "xp": xp, "w_AD": w_AD,
                "recall_status": recall_status, "recalled_events": recalled_events, "shared": True}

    def _shared_read_guard(self, st):
        """A LIVE `what_if`/`why` read needs `propagation_strength` raised to `self.read_prop` (the organ's
        maturation-gain read operating point), but on the SHARED pool that is a GLOBAL bridge-wide scalar (every
        co-resident organ's synaptic transmission is scaled by it) — so raise it ONLY for the duration of this one
        read, under `read_isolation`, and restore the pool's baseline immediately after. A no-op (nullcontext) for
        the standalone (non-shared) path, where the organ already parked `propagation_strength` at `read_prop`
        permanently at build (its own bridge; nothing else co-resides on it)."""
        import contextlib
        if not st.get("shared"):
            return contextlib.nullcontext()
        pool = self._shared
        cc = st["cfg"]

        @contextlib.contextmanager
        def _guard():
            with pool.read_isolation("causal_whatif"):
                saved_prop = cc.propagation_strength
                cc.propagation_strength = float(self.read_prop)
                try:
                    yield
                finally:
                    cc.propagation_strength = saved_prop
        return _guard()

    def _build_one(self, composer, lesion: bool = False) -> dict:
        from sim.backend import get_backend
        xp, _ = get_backend()
        # binding (1)+(2): enumerate events + gate the causal curriculum by the composer's moat recall.
        recalled_events, recall_status = enumerate_events(composer)
        episodes = [ep for ep in OBS_EPISODES if all(recall_status[e] for e in ep)]
        do_interv = bool(recall_status.get(X, False) and recall_status.get(Y, False))
        if self._shared is not None and not lesion:
            return self._build_shared(episodes, do_interv, recall_status, recalled_events, xp)
        bridge, cfg, meta = build_forward_model(self.seed)
        train(bridge, cfg, meta, xp, episodes, obs_reps=self.obs_reps, interv_reps=self.interv_reps,
              do_intervention=do_interv, prune_src=X)
        # freeze the learned structure + apply the uniform maturation gain (the gap#5 protocol; preserves ratios).
        cfg.enable_stdp = False
        cfg.enable_reward_modulation = False
        cfg.current_reward_signal = 0.0
        w_AD = _xblock_weight(bridge, A, D)          # the direct A->D edge must stay unlearned (unseen-consequence guard)
        cfg.propagation_strength = float(self.read_prop)
        if lesion:
            _lesion_xblock(bridge)                   # LOAD-BEARING: zero the learned forward edges
        return {"bridge": bridge, "cfg": cfg, "meta": meta, "xp": xp, "w_AD": w_AD,
                "recall_status": recall_status, "recalled_events": recalled_events, "shared": False}

    def ensure_built(self, composer):
        if self._built:
            return
        self._composer = composer
        self._st = self._build_one(composer, lesion=False)
        self.recall_status = self._st["recall_status"]
        self.recalled_events = self._st["recalled_events"]
        self.battery_fa = moat_battery(composer)     # no-false-accept moat battery (grounding sanity)
        self._built = True

    def _ensure_les(self, composer) -> dict:
        if self._les is None:
            self._les = self._build_one(composer, lesion=True)
        return self._les

    def what_if(self, composer, lesion: bool = False) -> dict:
        """'What happens if the dog goes east?' — HOLD A, roll the substrate forward (A->B->D via the learned
        directed edges though A->D was never taught), read the spiking successor, and MOAT-CONFIRM the consequence.
        Emits the real fact ONLY when confirmed; abstains (answer=None) otherwise (0 confabulation)."""
        self.ensure_built(composer)
        st = self._ensure_les(composer) if lesion else self._st
        b, xp, meta = st["bridge"], st["xp"], st["meta"]
        label_map = {e: e for e in range(meta["n_events"])}
        with self._shared_read_guard(st):
            unseen = unseen_consequence(b, meta, xp, label_map=label_map, w_AD=st["w_AD"])
        # binding (3): the consequence is a fact ONLY if the composer moat-confirms it.
        confirmed = bool(unseen["predicts_D"] and _recalled(composer, D))
        confab = bool(unseen["predicts_D"] and not _recalled(composer, D))   # predicted but NOT a real fact -> reject
        return {"on": True, "lesioned": bool(lesion), "kind": "what_if",
                "predicts_D": bool(unseen["predicts_D"]),
                "D_rate": float(unseen["D_rate"]), "B_rate": float(unseen["B_rate"]),
                "offchain_max": float(unseen["offchain_max"]), "w_AD_direct": st["w_AD"],
                "confirmed": confirmed, "confab": confab,
                "answer": (what_if_readout() if confirmed else None),
                "consequence_fact": (list(FACTS[D]) if confirmed else None)}

    def why(self, composer, lesion: bool = False) -> dict:
        """'Why did the dog wake?' — read the directed edge INTO Y as the argmax DO-probe predecessor, MOAT-CONFIRM
        it, and confirm it SURVIVES the DO-probe (do(cause) fires the effect; the spurious correlate does not).
        Emits the real cause fact ONLY when confirmed; abstains (answer=None) otherwise."""
        self.ensure_built(composer)
        st = self._ensure_les(composer) if lesion else self._st
        b, xp, meta = st["bridge"], st["xp"], st["meta"]
        label_map = {e: e for e in range(meta["n_events"])}
        with self._shared_read_guard(st):
            cause_evt, cause_rates = why_cause(b, xp, Y)
            doi = do_intervention(b, meta, xp, label_map=label_map)
        why_is_C = bool(cause_evt == C)
        # binding (3) + the DO-probe: the cause must be C, moat-confirmed, AND survive the DO-intervention.
        confirmed = bool(why_is_C and _recalled(composer, C) and doi["X_not_cause_of_Y"])
        return {"on": True, "lesioned": bool(lesion), "kind": "why",
                "cause": EVENT_NAMES.get(cause_evt), "why_is_C": why_is_C,
                "Y_rate_do_C": float(doi["Y_rate_do_C"]), "Y_rate_do_X": float(doi["Y_rate_do_X"]),
                "X_not_cause_of_Y": bool(doi["X_not_cause_of_Y"]),
                "why_target_rate_C": float(cause_rates.get(C, 0.0)),
                "why_target_rate_X": float(cause_rates.get(X, 0.0)),
                "confirmed": confirmed,
                "answer": (why_readout() if confirmed else None),
                "cause_fact": (list(FACTS[C]) if confirmed else None)}


# ── process/session-shared singleton (built once per live composer) ─────────────────────────────────────────
_ORGANS: dict = {}


def get_organ(key=None, seed: int = 42) -> CausalWhatIfProductionOrgan:
    """The causal organ for `key` (the ChatBrain cache key). Built once per key; grounded READ-ONLY against the
    live composer on `ensure_built`. Keyed (not a global singleton) because the grounding is per-brain composer.

    ONE-BRAIN WAVE-2 SINGLE-POOL (opt-in, `BRAIN_ONEBRAIN_WAVE2_POOL`, default-OFF) WINS when on: the organ's
    "evt" causal population is this organ's SLICE of the shared 9-organ Wave-2 `merge_organs` pool
    (`onebrain_wave2_pool_production.get_wave2_pool`). OFF (default) -> its own standalone bridge exactly as
    today (byte-identical) — mirrors the surprise/world-model/metacog/pragmatic single_pool branch. See the class
    docstring's declared residual: the pool is memoized per SEED, not per `key`, so multiple composer sessions at
    one seed would retrain the same shared evt slice sequentially."""
    org = _ORGANS.get(key)
    if org is None:
        from research.runners.onebrain_wave2_pool_production import wave2_pool_enabled, get_wave2_pool
        shared = get_wave2_pool(seed) if wave2_pool_enabled() else None
        org = CausalWhatIfProductionOrgan(seed=seed, shared=shared)
        _ORGANS[key] = org
    return org


def reset_organ(key=None):
    """Drop the cached organ for `key` (mirrors the D5/D6 per-session reset so a re-taught brain re-grounds)."""
    _ORGANS.pop(key, None)
