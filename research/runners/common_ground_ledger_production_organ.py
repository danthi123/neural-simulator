"""PRODUCTION ORGAN — the COMMON-GROUND LEDGER driving AUDIENCE DESIGN, on one spiking substrate, PERSISTENT across a
live conversation's turns.

This is the production wrap of the 6-seed-GO de-risk `research/runners/_learned_common_ground_ledger_derisk.py`
(commit 1a1478b76: audience-design acc=1.000, chance 0.5; permute -> 0.500; lesion -> 0.500 static frac-introduce 1.00;
substrate-read evidence grounded 0.113 >> ungrounded 0.000). The de-risk built the K bistable NMDA-attractor ledger +
the Namburi-Tye biased-competition reduce/introduce read and PROVED the read follows the actual grounding history and
is lesion-load-bearing. This organ makes that same substrate PERSISTENT PER SESSION so a live conversation's common
ground ACCUMULATES turn-over-turn, and exposes a single per-turn call the chat handler uses to make the served reply's
referring expression FOLLOW the ledger:

  * A referent NOT yet in this conversation's common ground (first mention) -> the substrate reads UNGROUNDED (its
    NMDA store was never ignited) -> the biased-competition novelty prior wins -> INTRODUCE (full description).
  * A referent ALREADY in common ground (mentioned earlier this conversation, its NMDA store latched + self-sustained)
    -> the gated substrate read routes that PERSISTENT FIRING into `evidence` -> REDUCE wins (pronominalize / reduce).

The decision is a SUBSTRATE read (evidence fires only because the ledger slot's neurons are firing and drive it through
a real synapse), NOT a host `if word in seen_set`. LESION (recurrence weight 0): the ledger cannot HOLD -> even a
re-mentioned referent's store has decayed by read-time -> every read is UNGROUNDED -> audience design goes STATIC
(always INTRODUCE). That collapse is what makes the wiring load-bearing rather than observe-only.

REUSE-BY-IMPORT (NO `sim/` edit). The bridge, the ignite/hold/query primitives and every operating-point constant come
straight from the de-risk module. This organ adds only: (a) persistence (build once, never restore between turns -> the
NMDA recurrence holds the accumulating common ground), (b) a word->slot map (the host language-comprehension boundary,
like the SVO parser / the swap-topic extractor), (c) a per-turn observe->decision->ground step, (d) numpy global-RNG
save/restore around every read so enabling the organ cannot perturb the other RNG-dependent chat faculties.

HONEST SCOPE: a FUNCTIONAL common-ground / audience-design correlate. The word->slot mapping and the surface string are
host (a comprehension boundary + a conditioned-articulation scaffold); the grounded-vs-ungrounded DECISION that drives
the reduce/introduce choice IS the spiking ledger read (lesion-proven). Learned conceptual pacts / lexical entrainment
/ partner-specificity are named follow-ons, not claimed here.
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from typing import Optional

import numpy as np

# reuse-by-import: the validated de-risk bridge builder + ignite/hold/query operating point. NO sim/ edit.
from research.runners._learned_common_ground_ledger_derisk import (
    build_cg_bridge, K_REF, N_DEC, N_EVID,
    GROUND_DRIVE_STEPS, HOLD_STEPS, QUERY_STEPS, QUERY_FLUSH, INTRO_PA,
)
from research.runners._self_schema_region_derisk import IGNITE_PA
from sim.backend import to_host

_DEFAULT_SEED = 42


class CommonGroundLedgerOrgan:
    """One persistent spiking common-ground ledger for a single conversation. K bistable referent slots latched by
    grounding acts (ignite + NMDA self-sustain) and read at speak-time by the biased-competition reduce/introduce
    decision. `lesion=True` builds the referent self-loops at weight 0 (the de-risk's load-bearing-recurrence lesion)
    -> the ledger cannot HOLD -> audience design goes static (always INTRODUCE). Lazy-builds the bridge on first use."""

    def __init__(self, seed: int = _DEFAULT_SEED, lesion: bool = False):
        self.seed = int(seed)
        self.lesion = bool(lesion)
        self._bridge = None
        self._xp = None
        self._idx = None
        # host comprehension boundary: referent word -> ledger slot index (assigned in first-seen order, K_REF slots).
        self._word_slot: dict[str, int] = {}
        # which slots have received a grounding act this conversation (host bookkeeping; the DECISION is the substrate).
        self._grounded_slots: set[int] = set()
        self._next_slot = 0
        self.n_turns = 0

    # ── lazy build ────────────────────────────────────────────────────────────────────────────────────────────
    def _ensure(self):
        if self._bridge is None:
            # build at quiescence; DO NOT restore the snapshot between turns -> the ledger accumulates + self-sustains.
            b, xp, idx, _snap = build_cg_bridge(seed=self.seed, lesion_update=self.lesion)
            self._bridge, self._xp, self._idx = b, xp, idx
        return self._bridge, self._xp, self._idx

    # ── one grounding act on a slot (ignite -> NMDA self-sustain) ───────────────────────────────────────────────
    def _ground(self, slot: int):
        b, xp, idx = self._ensure()
        led_dev = idx["led_dev"][slot]
        for _ in range(GROUND_DRIVE_STEPS):
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[led_dev] = xp.float32(IGNITE_PA)
            b._run_one_simulation_step()
        for _ in range(HOLD_STEPS):
            b.cp_external_input_current[:] = 0.0
            b._run_one_simulation_step()
        self._grounded_slots.add(int(slot))

    # ── read the audience-design decision for a slot (the SUBSTRATE read) ───────────────────────────────────────
    def _read(self, slot: int) -> dict:
        b, xp, idx = self._ensure()
        reduce_dev = idx["reduce_dev"]; intro_dev = idx["intro_dev"]; evid_dev = idx["evid_dev"]
        # drain the (non-NMDA) decision pools without disturbing the self-sustaining ledger
        for _ in range(QUERY_FLUSH):
            b.cp_external_input_current[:] = 0.0
            b._run_one_simulation_step()
        b.set_transmission_gate(f"query_{slot}", 1.0)
        late = QUERY_STEPS - max(1, QUERY_STEPS // 3)
        r_reduce = r_intro = r_evid = 0.0
        for step in range(QUERY_STEPS):
            b.cp_external_input_current[:] = 0.0
            b.cp_external_input_current[intro_dev] = xp.float32(INTRO_PA)   # novelty prior -> INTRODUCE
            b._run_one_simulation_step()
            if step >= late:
                fs = b.cp_firing_states
                r_reduce += float(to_host(fs[reduce_dev].astype(xp.float64).sum()))
                r_intro += float(to_host(fs[intro_dev].astype(xp.float64).sum()))
                r_evid += float(to_host(fs[evid_dev].astype(xp.float64).sum()))
        b.set_transmission_gate(f"query_{slot}", 0.0)
        nlate = float(QUERY_STEPS - late)
        rr = r_reduce / (nlate * N_DEC)
        ri = r_intro / (nlate * N_DEC)
        re = r_evid / (nlate * N_EVID)
        margin = rr - ri
        # tie (both silent) -> INTRODUCE: the novelty prior is the default for an un-established referent
        decision = "reduce" if margin > 1e-9 else "introduce"
        return {"decision": decision, "reduce_rate": rr, "introduce_rate": ri,
                "evidence_rate": re, "margin": margin}

    def _slot_for(self, word: str) -> tuple[int, bool]:
        """Map a referent word to a ledger slot. Returns (slot, is_new). New referents claim the next free slot;
        overflow past K_REF reuses the oldest slot (a bounded ledger)."""
        w = word.strip().lower()
        if w in self._word_slot:
            return self._word_slot[w], False
        if self._next_slot < K_REF:
            slot = self._next_slot
            self._next_slot += 1
        else:  # bounded ledger: recycle the lowest slot (rare in a short conversation)
            slot = self._next_slot % K_REF
            self._next_slot += 1
            # forget the word that previously held this slot
            for k, v in list(self._word_slot.items()):
                if v == slot:
                    self._word_slot.pop(k, None)
            self._grounded_slots.discard(slot)
        self._word_slot[w] = slot
        return slot, True

    # ── the production entry point ──────────────────────────────────────────────────────────────────────────────
    def observe_turn(self, topic: Optional[str]) -> dict:
        """Run ONE audience-design decision for this turn's referent, reading the ledger AS IT STANDS (before this
        turn's grounding act), then apply this turn's grounding act (ignite the slot so the referent enters / refreshes
        common ground). Returns {on, topic, slot, decision, in_common_ground, reduce_rate, introduce_rate,
        evidence_rate, margin, lesioned}. `topic=None` (no grounded referent this turn) -> inert no-decision info.
        Wraps every substrate step in a numpy global-RNG save/restore so enabling the organ leaves the other
        RNG-dependent chat faculties byte-identical."""
        self.n_turns += 1
        if not topic:
            return {"on": True, "topic": None, "decision": None, "reason": "no_grounded_topic"}
        rng_state = np.random.get_state()
        try:
            slot, is_new = self._slot_for(topic)
            was_grounded = slot in self._grounded_slots
            read = self._read(slot)               # the SUBSTRATE read of the current common-ground state
            decision = read["decision"]
            self._ground(slot)                    # this turn's grounding act: the referent is now in common ground
            out = {
                "on": True,
                "topic": topic.strip().lower(),
                "slot": int(slot),
                "decision": decision,             # 'reduce' (grounded) | 'introduce' (new)
                "in_common_ground": bool(was_grounded),
                "first_mention": bool(is_new),
                "reduce_rate": read["reduce_rate"],
                "introduce_rate": read["introduce_rate"],
                "evidence_rate": read["evidence_rate"],
                "margin": read["margin"],
                "lesioned": self.lesion,
            }
        except Exception as e:   # never let the ledger crash / change a turn -> inert no-decision info
            out = {"on": True, "topic": topic, "decision": None, "error": f"{type(e).__name__}: {e}"}
        finally:
            np.random.set_state(rng_state)
        return out


# process-wide per-session cache (mirrors the other production organs' _ORGANS pattern)
_ORGANS: dict = {}


def get_organ(cache_key, seed: int = _DEFAULT_SEED, lesion: bool = False) -> CommonGroundLedgerOrgan:
    """The per-session organ (lazy build). One ledger per conversation so common ground does not leak across
    sessions. Rebuilds if the lesion flag changed for this key (so a lesion probe gets a fresh lesioned ledger)."""
    org = _ORGANS.get(cache_key)
    if org is None or org.lesion != bool(lesion):
        org = CommonGroundLedgerOrgan(seed=seed, lesion=bool(lesion))
        _ORGANS[cache_key] = org
    return org


def reset_organ(cache_key):
    _ORGANS.pop(cache_key, None)
