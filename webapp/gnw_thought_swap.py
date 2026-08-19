"""GNW NEURAL THOUGHT-SWAP wired into the LIVE `/api/brain-chat` held-topic workspace — the swap's natural use-case.

WHAT THIS IS (board #77, INTEGRATION-TO-PRODUCTION). The GNW thought-swap is fully NEURAL end-to-end at the de-risk
level (each 6/6-seed GO): the recurrence-weaken swap EVICTS the incumbent coalition neurally (Rung-2d short-term
depression drains its own recurrent E->E loop below the sustain knee -> self-collapse); the neural vacancy gate ADMITS
the challenger neurally (occ occupancy interneurons -> gate_k disinhibition -> gate_k -> pattern_k); and the neural
swap-intention detector DECIDES when to swap (a spiking mismatch/salience comparator fires only for a salient proposal
that does NOT match the held content, and its firing sets the recurrence-depression boost). This module wires that
whole chain onto the LIVE conversational brain so the workspace holds the CURRENT CONVERSATIONAL TOPIC across turns and
SWAPS it when a new user turn is a genuine TOPIC CHANGE.

THE LIVE HELD-CONTENT REPRESENTATION. The already-wired GNW gates (`webapp/gnw_deliberation.py`,
`webapp/gnw_multistep_deliberation.py`) run a workspace that is RESTORED to a clean snapshot each turn — they do not
hold a thought ACROSS turns. This module adds the missing cross-turn held-content register: a per-session GNW swap
workspace (one ignited coalition = the current topic), reused-by-import from the swap de-risk. The held topic is the
"current thought"; a topic-change user turn is a salient MISMATCH between the new input and that held content, so the
swap machinery evicts the old topic and ignites the new one; a same-topic follow-up MATCHES (pred vetoes the mismatch
detector) so the current thought persists.

THE MAPPING (host boundary vs neural decision).
  * TOPIC (host comprehension of the world/teacher input — the SAME declared boundary the SVO question parser and the
    chase-marker detector occupy): the user message's topic = the FIRST GROUNDED concept token in the message (a known
    agent/patient in the brain's store). An anaphoric/no-new-concept follow-up ("what does it chase?", "tell me more")
    yields NO new topic -> the held thought persists. Restricting the topic to a KNOWN concept filters action verbs and
    unknown words, and keeps the swap workspace grounded (moat-consistent: only a concept the brain knows can become a
    held topic). This is host parsing of the environment input; the swap DECISION itself is 100% the substrate.
  * DECISION (the substrate): given the currently-held topic slot (incumbent) and the incoming topic slot (proposal),
    the reused `run_intention_swap` drives the proposal into the spiking mismatch/salience detector + the vacancy gate.
    A DIFFERENT incoming topic is a salient MISMATCH -> mm fires -> boost -> the incumbent self-evicts -> the vacancy
    gate admits the newcomer (a SWAP). The SAME incoming topic MATCHES -> the pred interneuron vetoes mm -> no boost ->
    the incumbent holds (NO swap). The swap-vs-hold verdict is the neurons', not a host `if`.

CONTRACT (additive, DEFAULT-OFF, reversible).
  * `BRAIN_GNW_SWAP` truthy (1/true/on/yes) -> ENABLED; unset/0/false/off/no -> DISABLED. Default OFF (pending owner
    review). When DISABLED the handler block is fully skipped: no workspace is built, no per-turn work runs, and the
    response carries NO `gnw_swap` key -> the turn is BYTE-IDENTICAL to pre-wiring.
  * When ENABLED the swap workspace is an ADDITIVE held-topic TRACKER: it NEVER changes the answer text (`answer`,
    `abstained`, `recalled_svo`, `verified`, ...). It only stashes a per-turn `gnw_swap` info block (held topic, whether
    this turn swapped, the spiking read of the post-swap workspace). So ordinary turns (recall/abstain/multi-step/self)
    are answer-unchanged with it enabled; only the additive key appears. This mirrors how `gnw_bus` is attached.
  * The workspace build (0.8s) is lazy on the first grounded-topic turn per session and kept warm; each subsequent
    topic turn runs one ~0.3s swap decision.

REUSE-BY-IMPORT (NO `sim/` edit). The swap substrate build, the `MultiLoopSTD` eviction effector, the neural vacancy
gate, the mismatch/salience detector and `run_intention_swap` come STRAIGHT from
`research/runners/_gnw_neural_swap_intention_derisk` (6/6-seed GO). This module adds only the production glue (the
per-session held-topic register + the grounded-topic extractor). `git diff sim/` is empty.

HONEST RESIDUAL (named, not claimed closed).
  1. The cross-turn CONTINUITY of the held thought is carried by a host label (`held_slot`) and RE-ESTABLISHED on the
     substrate each turn via `run_intention_swap(isolate=True)` (restore clean snapshot -> re-ignite the held topic ->
     present the proposal -> neural decision). The swap-vs-hold VERDICT is neural every turn; the between-turn persistence
     of "which coalition is held" is a host bookkeeping label (like "attending to A"), not the ignition literally
     surviving the HTTP gap. A truly continuous cross-turn ignition is the named next rung.
  2. The mm->boost COUPLING is host arithmetic (`eff_boost = gain * mm_rate`), a neuromodulator-like linear read-out of
     the salience population's firing to the loop's release-probability U — there is no engine primitive for
     "presynaptic firing raises U of other synapses". The DECISION (whether/when there is any boost) is fully the mm
     spikes; the read-out itself is a scaffold to burn down (unchanged from the de-risk). A functional correlate only.
  3. Content routing is per-pattern labeled-line (N_PATTERNS=3 held-topic slots; beyond that, least-recently-held
     reuse), inherited from the de-risk — the topic identity is the world's input, not a learned/composed code.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Optional

import numpy as np

# reuse-by-import the 6/6-seed-GO neural swap machinery (build + MultiLoopSTD + the mismatch/vacancy-gate swap) — NO sim/ edit.
from research.runners._gnw_neural_swap_intention_derisk import (
    build as _build_swap_substrate,
    run_intention_swap,
    MultiLoopSTD,
    SALIENT_PA,
    N_PATTERNS,
)

_DEFAULT_SEED = 42

# tokens that are never a conversational TOPIC on their own (question words / auxiliaries / articles / pronouns /
# prepositions / discourse fillers). The topic must additionally be a GROUNDED concept (see _extract_topic), so this
# set only trims the obvious function words before the grounded-concept match; it is not the sole filter.
_STOP = frozenset("""
what who whom whose where when why how which
do does did is are was were be been being am will would can could should shall must may might has have had
the a an this that these those there here
it its it's he she they them him her his their i you we me my your our us mine yours ours
to of in on at for with and or but about from by as into onto than then so if not no yes
tell more please thing things something anything someone anyone know knows knew say says said talk talks
ok okay hi hello hey thanks thank oh well just really very much any some all
""".split())


def swap_enabled() -> bool:
    """The master flag. DEFAULT-OFF (a reversible flag pending owner review): `BRAIN_GNW_SWAP` truthy (1/true/on/yes)
    -> ENABLED. Unset or 0/false/off/no -> DISABLED (the handler block is skipped, no `gnw_swap` key -> byte-identical)."""
    return os.environ.get("BRAIN_GNW_SWAP", "0").strip().lower() in ("1", "true", "on", "yes")


def _known_concepts(composer) -> set:
    """The set of GROUNDED concept tokens the brain knows = every agent/patient string in its store (lowercased). A
    topic must be one of these (so an action verb / unknown word is not a topic; the held workspace stays grounded)."""
    out: set = set()
    if composer is None:
        return out
    kb = getattr(composer, "kb", None)
    if kb is not None:
        try:
            for fact, _handle in kb:
                for role in ("agent", "patient"):
                    v = fact.get(role) if isinstance(fact, dict) else None
                    if isinstance(v, str) and v.strip():
                        out.add(v.strip().lower())
            return out
        except Exception:
            pass
    # fallback: role-unbind scan (mirrors gnw_deliberation.all_candidate_patients' faithful read)
    try:
        for fact, comp in composer._iter_facts():
            for role in ("agent", "patient"):
                try:
                    v = composer.unbind(comp, role)
                except Exception:
                    v = fact.get(role) if isinstance(fact, dict) else None
                if isinstance(v, str) and v.strip():
                    out.add(v.strip().lower())
    except Exception:
        pass
    return out


def _extract_topic(message: str, composer) -> Optional[str]:
    """Host comprehension of the world/teacher input (the declared boundary). The topic of the user's message = the
    FIRST GROUNDED concept token (a known agent/patient) in message order. No grounded concept (an anaphoric or
    no-new-topic follow-up: "what does it chase?", "tell me more") -> None -> the held thought persists. Read-only."""
    if not isinstance(message, str) or not message.strip():
        return None
    known = _known_concepts(composer)
    toks = re.findall(r"[a-zA-Z']+", message.lower())
    if known:
        for t in toks:
            if t in known:
                return t
        return None
    # no scannable store -> degrade to the first non-stopword content token (still host parsing of the input)
    for t in toks:
        if t not in _STOP and len(t) > 1:
            return t
    return None


class ThoughtSwapWorkspace:
    """A per-session GNW swap workspace holding the CURRENT CONVERSATIONAL TOPIC as one ignited coalition, reused-by-
    import from the swap de-risk. `observe(topic)` runs one neural swap DECISION: a DIFFERENT salient topic swaps the
    workspace (old evicted, new ignited); the SAME topic (or None) holds. The workspace build is lazy + warm."""

    def __init__(self, seed: int = _DEFAULT_SEED):
        self.seed = int(seed)
        self._S = None
        self._std = None
        self._lock = threading.Lock()
        self.held_slot: Optional[int] = None
        self.topic_to_slot: dict = {}
        self.slot_to_topic: dict = {}
        self._lru: list = []          # slot usage order (least-recently-held first) for reuse beyond N_PATTERNS
        self.n_turns = 0
        self._rng_state = None        # the swap's PRIVATE RNG timeline (the host process-global RNG is never advanced)

    def _isolated(self, fn):
        """Run `fn()` (the swap build + spiking sim) on the swap's PRIVATE RNG timeline, leaving the host process-global
        RNG (numpy + the sim backend) BYTE-UNTOUCHED. The swap substrate's build reseeds cfg.seed and its stepping draws
        OU noise off the SAME process-global RNG the rest of the pipeline shares — without this, enabling the swap would
        perturb the downstream RNG-dependent organs (curiosity/self-initiation) and break byte-identity. So: snapshot the
        host RNG, swap in the swap's own continuous timeline, run, capture the advanced private timeline, restore host."""
        xp = None
        try:
            from sim.backend import get_backend
            xp, _ = get_backend()
        except Exception:
            xp = None
        host_np = np.random.get_state()
        host_xp = None
        if xp is not None and xp is not np:
            try:
                host_xp = xp.random.get_random_state().get_state()
            except Exception:
                host_xp = None
        # swap IN the swap's private timeline (seed it on the first call; else continue it)
        if self._rng_state is None:
            np.random.seed(self.seed)
            if xp is not None and xp is not np:
                try:
                    xp.random.seed(self.seed)
                except Exception:
                    pass
        else:
            try:
                np.random.set_state(self._rng_state["np"])
            except Exception:
                pass
            if xp is not None and xp is not np and self._rng_state.get("xp") is not None:
                try:
                    xp.random.get_random_state().set_state(self._rng_state["xp"])
                except Exception:
                    pass
        try:
            return fn()
        finally:
            st = {"np": np.random.get_state(), "xp": None}
            if xp is not None and xp is not np:
                try:
                    st["xp"] = xp.random.get_random_state().get_state()
                except Exception:
                    st["xp"] = None
            self._rng_state = st
            # restore the host process-global RNG — the swap never advanced it
            try:
                np.random.set_state(host_np)
            except Exception:
                pass
            if host_xp is not None:
                try:
                    xp.random.get_random_state().set_state(host_xp)
                except Exception:
                    pass

    def _ensure(self):
        if self._S is None:
            self._S = _build_swap_substrate(seed=self.seed)
            self._std = MultiLoopSTD(self._S["bridge"], self._S["xp"], self._S["ws_used"], self._S["patterns_host"])

    def _slot_for(self, topic: str) -> int:
        """Labeled-line assignment: a distinct topic -> a distinct slot (0..N_PATTERNS-1); beyond N_PATTERNS, reuse the
        least-recently-held slot. The topic identity is the world's input (the de-risk's labeled-line limit)."""
        if topic in self.topic_to_slot:
            slot = self.topic_to_slot[topic]
        else:
            used = set(self.slot_to_topic)
            free = [k for k in range(N_PATTERNS) if k not in used]
            if free:
                slot = free[0]
            else:
                slot = self._lru[0] if self._lru else 0
                old = self.slot_to_topic.pop(slot, None)
                if old is not None:
                    self.topic_to_slot.pop(old, None)
            self.topic_to_slot[topic] = slot
            self.slot_to_topic[slot] = topic
        if slot in self._lru:
            self._lru.remove(slot)
        self._lru.append(slot)
        return slot

    def _held_topic(self) -> Optional[str]:
        return self.slot_to_topic.get(self.held_slot) if self.held_slot is not None else None

    def observe(self, topic: Optional[str], *, lesion: bool = False) -> dict:
        """Run one neural swap decision for the incoming `topic` against the currently-held topic. Returns a per-turn
        info dict (never raises out; the caller degrades to no-op). The swap-vs-hold VERDICT is the substrate's.

        `lesion=True` threads `trigger_lesion=True` into `run_intention_swap` -> the spiking mismatch/salience
        detector is given NO proposal drive (mm never fires -> the STD boost stays 0), so a salient TOPIC-CHANGE can
        no longer trigger a swap (the incumbent holds). This is the de-risk's own NEURAL lesion (the swap DECISION
        collapses at its source, the mismatch spikes), reused-by-import for the board-#85 load-bearing proof. Default
        False -> byte-identical to the #77 observer (which never passes it)."""
        with self._lock:
            self.n_turns += 1
            info = {"acted": False, "turn": self.n_turns, "topic": topic, "swapped": False,
                    "held_topic_before": self._held_topic(), "held_topic": self._held_topic(),
                    "evicted_topic": None, "reason": None, "seed": self.seed, "lesioned": bool(lesion)}
            if topic is None:
                info["reason"] = "no_topic_hold"
                return info
            self._isolated(self._ensure)
            slot = self._slot_for(topic)
            info["proposed_slot"] = int(slot)

            if self.held_slot is None:
                # FIRST THOUGHT: establish + hold the opening topic (a match probe: ignite the slot, pred vetoes its own
                # proposal -> it holds). Confirms the workspace ignites and holds a single coalition.
                r = self._isolated(lambda: run_intention_swap(self._S, self._std, incumbent=slot, proposed=slot,
                                                              proposal_pa=SALIENT_PA, trigger_lesion=bool(lesion),
                                                              isolate=True))
                self.held_slot = int(slot)
                info.update({"acted": True, "reason": "first_thought", "held_slot": int(slot),
                             "held_topic": topic, "held_topic_before": None,
                             "winner_slot": int(r["winner_post"]), "n_ignited_post": int(r["n_ignited_post"]),
                             "held_rate_post": float(r["new_rate_post"]), "mm_peak": float(r["mm_peak"]),
                             "boost_max": float(r["boost_max"])})
                return info

            # THERE IS A HELD TOPIC: present the incoming topic as a salient proposal; the substrate decides swap-vs-hold.
            incumbent = int(self.held_slot)
            r = self._isolated(lambda: run_intention_swap(self._S, self._std, incumbent=incumbent, proposed=int(slot),
                                                          proposal_pa=SALIENT_PA, trigger_lesion=bool(lesion),
                                                          isolate=True))
            swapped = bool(r["swapped"])
            evicted = None
            if swapped:
                evicted = self.slot_to_topic.get(incumbent)
                self.held_slot = int(r["winner_post"])   # == slot
            info.update({
                "acted": True, "swapped": swapped, "evicted_topic": evicted,
                "held_slot": int(self.held_slot), "held_topic": self._held_topic(),
                "incumbent_slot": incumbent, "winner_slot": int(r["winner_post"]),
                "n_ignited_post": int(r["n_ignited_post"]),
                "old_residual_post": float(r["old_residual_post"]), "new_rate_post": float(r["new_rate_post"]),
                "mm_peak": float(r["mm_peak"]), "boost_max": float(r["boost_max"]),
                "reason": ("topic_change_swap" if swapped else
                           ("same_topic_hold" if slot == incumbent else "mismatch_held_no_swap")),
            })
            return info


def get_swap_workspace(chat, *, seed: int = _DEFAULT_SEED) -> ThoughtSwapWorkspace:
    """Idempotently attach a per-session `ThoughtSwapWorkspace` to the cached ChatBrain (auto-cleared on session reset,
    which drops the ChatBrain). No `sim/` edit; the ChatBrain instance is a host scaffold."""
    ws = getattr(chat, "_gnw_swap_workspace", None)
    if ws is None:
        ws = ThoughtSwapWorkspace(seed=seed)
        chat._gnw_swap_workspace = ws
    return ws


def observe_turn(chat, message: str, *, seed: int = _DEFAULT_SEED, lesion: bool = False) -> dict:
    """The production entry point: extract this turn's grounded topic from the user message and run one neural swap
    decision on the per-session held-topic workspace. Returns the per-turn `gnw_swap` info (also stashed on
    `chat._last_gnw_swap`). Never raises out (on any error it returns an inert info dict so a turn can never crash).
    `lesion=True` silences the mismatch detector (the de-risk's neural swap lesion) -> a topic-change can no longer
    swap; default False -> byte-identical to the #77 observer."""
    try:
        composer = getattr(getattr(chat, "inner", None), "composer", None)
        topic = _extract_topic(message, composer)
        ws = get_swap_workspace(chat, seed=seed)
        info = ws.observe(topic, lesion=bool(lesion))
    except Exception as e:  # never let the swap tracker crash / change a turn
        info = {"acted": False, "reason": f"error:{type(e).__name__}: {e}", "swapped": False}
    chat._last_gnw_swap = info
    return info
