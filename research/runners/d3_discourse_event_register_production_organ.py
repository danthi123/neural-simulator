"""DISCOURSE EVENT REGISTER — "who was doing it BEFORE?" across a discourse connective, wired for the PRODUCTION turn
(D3, Gate-B shape, 2026-08-13).

The faculty: a discourse connective ("then"/"but"/"meanwhile") marks an EVENT BOUNDARY that SHIFTS the running event
into a PREVIOUS slot instead of overwriting it, so the brain holds a PAIR of composed events (a_curr,p_curr |
a_prev,p_prev) and can be ASKED about EITHER — "who is doing it now?" AND "who was doing it BEFORE?". A single-event
register STRUCTURALLY CANNOT answer the before-question (it overwrote the prior event); recency and "answer the current
agent" both fail. This is the deployed conversational payoff of the connectives arc.

It REUSES (does not reinvent) the validated spiking twin
(`research/runners/_d3_event_pair_agent_derisk.PairEventRegister`, spiking=True → the 6-seed GO
`2026-07-10-D3-event-connectives-ON-SPIKES-GO` and the deployed BEFORE-answer `2026-07-10-D3-event-pair-live-agent-
BEFORE-GO`): FOUR one-of-K FS-WTA attractor slots, EACH re-discretized by its own K-pool Izhikevich attractor bridge +
a shared FS inhibitory pool (`_d3_spiking_attractor_derisk.build_fswta_score_bridge`/`fswta_drive`). The four spiking
winners ARE the next state (per-slot host-agree 0.992). NOTHING here is reimplemented — the organ imports the class and
the fold logic mirrors `multi_turn_agent.MultiTurnAgent.hear` (:297-311) register branch verbatim.

WHAT IS SPIKING vs WHAT IS HOST (declared honestly — the honesty boundary is a deliverable, not a caveat):
  * SPIKING (load-bearing): the four event SLOTS. Each slot's next value is `argmax` over the accumulated per-pool
    `cp_firing_states` firing of a K-pool Izhikevich attractor under FS lateral inhibition, after real
    `_run_one_simulation_step()` settles (`fswta_drive`) — the SAME read-out-instrument class the affect/comprehension/
    metacog/d6 organs use over their spiking pools. The PRIOR event is HELD across arbitrarily many following clauses
    by rolling the spiking winners forward. Silencing the two prev slots (the LESION) collapses the before-answer.
  * HOST (declared residual scaffolds, ride existing burn-down items):
    - the transition δ (prev-slots + clause-code → per-slot scores) is a rate-learned RNN (`multislot_rnn`); ONLY the
      re-discretization (score → one-of-K winner) is on spikes — the same rung scope as every D3 spiking port.
    - the boundary DETECTION (a leading connective) + the referent/verb PARSE are host (the vocab-ceiling residual,
      the same class comprehension/d6 declare); the SHIFT the connective triggers IS executed on the substrate.
    - the register is CO-RESIDENT on its own four FS-WTA bridges alongside the recall composer, not merged onto the one
      recall bridge — rides the one-brain merge (burn-down #1), exactly as the affect/comprehension/d6 organs do.

THE HONESTY FLOOR is preserved BY CONSTRUCTION: this organ NEVER manufactures a fact, flips an abstain into an assert,
or changes WHICH answer the recall produces. `observe` SKIPS an unknown subject/patient (no write → no confabulation).
A before-answer is surfaced ONLY after a discourse connective actually opened a boundary THIS conversation
(`state['boundary_seen']`) — otherwise the held prev slot reads the identity (referents[0]) and the organ honestly
abstains ("no earlier event yet"), never a false "dog was". Holding a prior event does NOT cost the present: NOW stays
correct under the lesion. The moat is therefore only ever tightened, never loosened.

Scope / stateful note: like d5/d6 the register ACCUMULATES across a conversation (it holds the running event pair), so
its turn STATE (`boundary_seen`, `heard_any`) is CONVERSATION-SCOPED (one `new_state()` per cache_key, cleared on
reset). The register object itself is built ONCE with the agent (attached as `event_register`).

Latency residual (speed is SECONDARY, mission non-negotiable): the spiking twin trains the rate δ + builds four
FS-WTA bridges at load (~76s on numpy@K=6; comparable to the accepted ~180s onebrain composer build); a heard clause
runs four bridge settles (~28ms/clause on numpy). Both are one-time / per-clause, never per-token.

Additive, default-ON, `BRAIN_DISCOURSE_REGISTER=0` → the byte-identical host oracle (the register is built spiking=False
— TODAY's rate register — AND the endpoint hook is skipped). `BRAIN_DISCOURSE_REGISTER_LESION=1` → the load-bearing
prev-slot-silence spiking register. NO `sim/` edit; uses the process backend (cupy in production, numpy in tests) via
reuse-by-import.
"""
from __future__ import annotations

import os

# reuse-by-import: the validated spiking pair register (4 FS-WTA attractor slots). NO reimplementation.
from research.runners._d3_event_pair_agent_derisk import PairEventRegister

CONNECTIVES = ("then", "but", "meanwhile")


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Enable / lesion flags — the exact contract the other Gate-B organs use.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def discourse_register_enabled() -> bool:
    """Default-ON. `BRAIN_DISCOURSE_REGISTER` in {0,false,no,off} -> the byte-identical host oracle: the register is
    built spiking=False (TODAY's host-numpy rate register) AND the endpoint hook is skipped (no before/now reply)."""
    v = os.environ.get("BRAIN_DISCOURSE_REGISTER")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def discourse_register_lesioned() -> bool:
    """`BRAIN_DISCOURSE_REGISTER_LESION` in {1,true,yes,on} -> silence the SPIKING HOLD of the prior event (force the
    two prev slots to the identity every clause) -> the who-was-before read collapses; NOW is untouched (load-bearing)."""
    v = os.environ.get("BRAIN_DISCOURSE_REGISTER_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The LESION instrument (load-bearing): kill the spiking HOLD of the prior event. Thin override of the reused class,
# NOT a reimplementation of the mechanism — after each clause the two PREV slots (2,3) are forced to the identity, so
# nothing is carried on the substrate across the connective boundary. a_curr/p_curr (slots 0,1) are untouched, so the
# CURRENT event still tracks (NOW preserved) — the analogue of d6's recur=0 hold-kill.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class _PrevSilencePairRegister(PairEventRegister):
    def observe(self, subject_word, object_word):
        super().observe(subject_word, object_word)
        self.slots[2] = self.ident      # silence the held a_prev
        self.slots[3] = self.ident      # silence the held p_prev


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The production FACTORY (THE FLIP). Default -> the VALIDATED genuinely-SPIKING PairEventRegister. Disabled ->
# spiking=False (TODAY's host-numpy rate register, byte-identical). Lesioned -> the prev-slot-silence spiking register.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def make_discourse_register(referents, seed: int = 42, *, enabled: bool | None = None, lesion: bool | None = None):
    """Return the discourse event register for the current build. Reuse-by-import; NO reimplementation.

    Default (enabled, not lesioned): the four-FS-WTA-slot spiking twin — the who-was-before answer is read off
    `cp_firing_states`. `enabled=False`: `PairEventRegister(spiking=False)` — byte-identical to today's production.
    `lesion=True`: the `_PrevSilencePairRegister` (spiking hold of the prior event killed)."""
    en = discourse_register_enabled() if enabled is None else bool(enabled)
    les = discourse_register_lesioned() if lesion is None else bool(lesion)
    refs = list(referents)
    if not en:
        return PairEventRegister(refs, seed=seed, spiking=False)      # byte-identical to today's production register
    if les:
        return _PrevSilencePairRegister(refs, seed=seed, spiking=True)  # load-bearing: the held prior event is silenced
    return PairEventRegister(refs, seed=seed, spiking=True)           # the validated spiking twin (default-ON)


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Turn-class detection (host — the parse side, like is_referential / is_hold_query in the sibling organs). The SHIFT +
# the HELD read that follow are spiking.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Set-membership (mirrors brain_chat_tui.ChatBrain._discourse_turn :845/:850) — a small closed set of the disjoint
# before/now discourse queries. Robust to punctuation/whitespace; unambiguous (no normal turn matches).
_BEFORE_SET = {"who was doing it before", "who was before", "who did it before", "who was doing that before",
               "who was doing that", "who did that before"}
_NOW_SET = {"who is doing it now", "who is doing it", "who is now", "who did it now", "who is doing that now",
            "who is doing that"}


def _norm_q(text: str) -> str:
    return " ".join((text or "").strip().rstrip(".!?").lower().split())


def is_before_query(text: str) -> bool:
    """The disjoint 'who was doing it before?' turn class — no other organ handles it."""
    return _norm_q(text) in _BEFORE_SET


def is_now_query(text: str) -> bool:
    """'who is doing it now?' — read the CURRENT event's agent off the register."""
    return _norm_q(text) in _NOW_SET


def _strip_connective(text: str):
    """(had_connective, tokens_after_connective) for an SVO clause with an optional leading discourse connective."""
    w = (text or "").strip().rstrip(".!?").split()
    if w and w[0].lower() in CONNECTIVES and len(w) >= 4:
        return True, w[1:]
    return False, w


def is_discourse_clause(text: str, actions=None) -> bool:
    """An SVO discourse clause (optionally connective-led): 3 content tokens where the verb is a known action (or the
    canonical 'chase') and the subject is a referent-or-pronoun. `actions` = the agent's known verbs; when None only a
    connective-led clause qualifies (the safe, unambiguous discourse-continuation case)."""
    had_conn, w = _strip_connective(text)
    if len(w) != 3:
        return False
    verb = w[1].lower()
    verb_ok = (verb == "chase") or (actions is not None and verb in actions)
    if not verb_ok:
        return False
    if not had_conn and actions is None:
        return False                         # a plain SVO with no verb whitelist -> leave to the normal path
    return True


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Register access helper: accept either the MultiTurnAgent (carries `_event_register`) or a register directly.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _get_reg(agent_or_reg):
    return getattr(agent_or_reg, "_event_register", agent_or_reg)


def new_state() -> dict:
    """Fresh conversation-scoped discourse state. `boundary_seen` is the moat gate for the before-answer; `heard_any`
    for the now-answer."""
    return {"boundary_seen": False, "heard_any": False}


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# FOLD a discourse clause into the register (register-only — mirrors MultiTurnAgent.hear :297-311, WITHOUT re-storing
# the fact, so the normal assertion path stays the sole writer). Additive side-effect; NEVER changes the reply.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def note_turn(text: str, agent_or_reg, state: dict, actions=None) -> dict | None:
    """If `text` is an SVO discourse clause, fold it into the register (a connective marks the event boundary → SHIFT;
    the observe re-discretizes each slot on spikes) and update `state`. Returns a small info dict (or None when the turn
    is not a discourse clause). Additive: this ONLY updates the running event pair; it does not store a fact or reply."""
    reg = _get_reg(agent_or_reg)
    if reg is None or not is_discourse_clause(text, actions):
        return None
    had_conn, w = _strip_connective(text)
    if had_conn and hasattr(reg, "mark_boundary"):
        reg.mark_boundary()                  # the connective opens a NEW event; the ended one shifts into the prev slot
    reg.observe(w[0], w[2])                   # subject, object — observe handles corefs + skips unknown referents
    state["heard_any"] = True
    if had_conn:
        state["boundary_seen"] = True         # an earlier event now genuinely exists (the moat gate for who-was-before)
    return {"on": True, "kind": "clause", "had_connective": bool(had_conn),
            "now": _safe_now(reg), "boundary_seen": bool(state["boundary_seen"])}


def _safe_now(reg):
    try:
        return reg.who_agent()
    except Exception:
        return None


def _safe_before(reg):
    try:
        return reg.who_agent_prev() if hasattr(reg, "who_agent_prev") else None
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ANSWER the disjoint before/now query classes with an honest functional read-out + the moat abstain.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def answer_before(agent_or_reg, state: dict) -> dict:
    """'who was doing it before?' — the PRIOR event's agent, read off the held spiking slot. Abstains (moat) until a
    discourse connective actually opened a boundary this conversation, and if the register cannot hold a pair at all
    (a single-event register lacks `who_agent_prev` → structurally cannot answer)."""
    reg = _get_reg(agent_or_reg)
    if not hasattr(reg, "who_agent_prev"):
        return {"kind": "before", "answer": "I can't answer who was doing it before -- I hold only one event at a time.",
                "abstained": True, "structural": True}
    if not state.get("boundary_seen"):
        return {"kind": "before", "answer": "I don't know who was doing it before -- no earlier event yet.",
                "abstained": True, "no_boundary": True}
    a = _safe_before(reg)
    if a:
        return {"kind": "before", "answer": f"{a} was.", "abstained": False, "agent": a}
    return {"kind": "before", "answer": "I don't know who was doing it before.", "abstained": True}


def answer_now(agent_or_reg, state: dict) -> dict:
    """'who is doing it now?' — the CURRENT event's agent, read off the register. Abstains until a clause was heard."""
    reg = _get_reg(agent_or_reg)
    if not state.get("heard_any"):
        return {"kind": "now", "answer": "I don't know who is doing it now -- nothing said yet.",
                "abstained": True, "nothing_heard": True}
    a = _safe_now(reg)
    if a:
        return {"kind": "now", "answer": f"{a} is.", "abstained": False, "agent": a}
    return {"kind": "now", "answer": "I don't know who is doing it now.", "abstained": True}


def maybe_answer(text: str, agent_or_reg, state: dict) -> dict | None:
    """Production entry for the DISJOINT reply classes. Returns a reply dict for a before/now query, else None (the
    turn is out of scope for this organ → the caller leaves it byte-identical). Does NOT fold clauses — call
    `note_turn` for that (additive side-effect)."""
    if is_before_query(text):
        return answer_before(agent_or_reg, state)
    if is_now_query(text):
        return answer_now(agent_or_reg, state)
    return None
