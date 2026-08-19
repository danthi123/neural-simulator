"""GNW MULTI-STEP re-entrant deliberation for the production `brain_chat` gate — THE KEYSTONE'S DEFERRED RUNG, WIRED.

WHAT THIS IS (roadmap T1-1 rung d, the "deliberation-until-sure over a CHAIN of inferences" half that
`webapp/gnw_deliberation.py` explicitly named as DEFERRED). The already-wired single-hop deliberation gate decides
ONCE: ignite the competing stored answers -> read the workspace conflict -> commit-or-abstain. It closes the
"halt-if-unsure" half. But a genuinely multi-step problem (a transitive chase A>B, B>C => the terminal of the A-chain)
needs the workspace to CYCLE the partial answer back through itself, re-igniting, UNTIL the substrate's own spiking
read says it has converged (reached the leaf), and only THEN emit. In every prior wiring the number of re-entrant
cycles was a HOST CONSTANT (`query_chain(cue, actions)` runs `len(actions)` hops; `HOPS=2`). This module wires the
keystone de-risk's EMERGENT-count loop onto the LIVE `/api/brain-chat` recall path so the number of cycles is a read
of the substrate's OWN spikes (`n_ignited` off `cp_firing_states`), not a host counter — the brain works through a
multi-step inference whose DEPTH it discovers itself, LIVE.

MECHANISM (reuse-by-import; NO `sim/` edit). The workspace build + the ignition/NMDA read + the ACC conflict gate +
the theta self-calibration + the whole `confidence_gated_chase` re-entry loop come STRAIGHT from the keystone de-risk
runner (`_gnw_reentrant_metacog_gated_deliberation_derisk`, 6/6-seed GO). This module adds ONLY the production glue:

  * DETECT (environment/comprehension boundary, exactly like the existing SVO question parser): a small set of explicit
    "follow the relation to the end" markers on the RAW user utterance ("all the way", "to the end", "eventually",
    "keep going", "the whole chain", "transitively", "follow the ... chain"). No marker -> the wrapper is a PURE
    PASS-THROUGH (byte-identical to today's gate). This is host parsing of the TEACHER/WORLD input, the same declared
    boundary the single-hop "what does X <action>?" parser already occupies; the COGNITION (the multi-step inference
    itself + WHEN to stop) is 100% the substrate.
  * EXTRACT (a, action): strip the marker, run the UNCHANGED inner gate on the clean "what does <agent> <action>?" so
    the production pipeline extracts (agent, action) + its first hop + runs its side effects EXACTLY ONCE. We reuse
    that (agent, action) as the chase's (cue, relation) — no re-parse, no double side effects.
  * CHASE (the substrate): `confidence_gated_chase` drives the current concept's `query_patient(x, action)` candidate
    into the P1.2 GNW workspace, WTA-ignites one winner, reads (n_ignited, conf) off the SPIKES, and the keystone
    `acc_conflict_gate` decides ADVANCE (broadcast the winner back, next hop) vs COMMIT (n_ignited==0 at the leaf ->
    the terminal reached) vs ABSTAIN. The cycle count EMERGES from the spiking read; a generous H_cap is a pure
    SAFETY budget correct answers never reach (the de-risk proves the stop is the spikes, not the budget).

CONTRACT:
  * DEFAULT-ON via `BRAIN_GNW_MULTISTEP` (production-default flip 2026-08-19). The handler INSTALLS the gate by default
    (server `_GNW_MULTISTEP_DEFAULT_ON = True`, the production-integration anchor, mirroring the single-hop
    `_GNW_DELIBERATE_DEFAULT_ON`); the installed wrapper is ACTIVE unless `BRAIN_GNW_MULTISTEP=0` (or false/off/no), in
    which case it is a pure PASS-THROUGH -> the live turn is byte-identical to the pre-flip default (the reversible
    DISABLE override). (Was default-off through the 6/6-seed live GO; flipped to on-by-default after that GO, mirroring
    how the bus + single-hop deliberation were introduced default-off -> verified -> flipped.)
  * BYTE-IDENTICAL when there is no chase marker. Every reactive-panel turn (recall/abstain/learn/anaphora on a
    single-hop question) has no marker -> the wrapper returns the inner gate's decision UNTOUCHED. Only an explicit
    chase-form question can change the outcome.
  * MOAT-SAFE. It NEVER un-abstains (a bus/deliberation abstain stays an abstain); on an unstored cue or an over-run
    past the leaf the chase reads `n_ignited==0` and ABSTAINS rather than confabulating a chain end; it never invents
    a fact. The emitted terminal is only ever a concept the substrate ignited off a real stored `query_patient` hop.
  * LESION-LOAD-BEARING. `BRAIN_GNW_MULTISTEP_LESION=1` runs the chase on the workspace built with the assembly
    self-recurrence ZEROED -> a hop can no longer sustain ignition -> `n_ignited` collapses at hop 0 -> the loop
    cannot advance/converge appropriately -> the multi-step answer collapses (the terminal is NOT reached). The
    emergent stopping is the SPIKING competition, not a host `for _ in range(depth)`.

REUSE-BY-IMPORT. The warm workspace + theta caches are SHARED with the single-hop deliberation gate
(`webapp.gnw_deliberation._get_bridge` / `_get_theta`), so one P1.2 workspace serves both. `git diff sim/` is empty.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import numpy as np

# reuse-by-import the keystone spiking mechanism (the whole EMERGENT-count re-entry loop) — NO sim/ edit.
from research.runners._p1_2_workspace_deliberation_loop_derisk import K_SLOTS
from research.runners._gnw_reentrant_metacog_gated_deliberation_derisk import (
    confidence_gated_chase, NMDA_ATTR_DEFAULT,
)
# share the ONE warm P1.2 workspace + self-calibrated theta with the single-hop deliberation gate.
from webapp.gnw_deliberation import _get_bridge, _get_theta, _DEFAULT_SEED


# ── the DETECT boundary (comprehension of the teacher/world utterance — the same boundary the SVO parser occupies) ──
# Explicit "follow the relation to the end" cues. Matching one flips the single recall hop into the multi-step chase;
# no cue -> pure pass-through. Ordered longest-first so the strip removes the full phrase.
_CHASE_MARKERS = (
    "all the way to the end",
    "all the way through",
    "the whole way through",
    "to the very end",
    "to the end of the chain",
    "the whole chain",
    "the entire chain",
    "following the chain",
    "follow the chain",
    "down the chain",
    "along the chain",
    "all the way",
    "to the end",
    "transitively",
    "eventually",
    "keep going until",
    "keep chasing",
    "in the end",
)


def multistep_enabled() -> bool:
    """The master flag, DEFAULT-ON (production-default flip 2026-08-19, mirroring the single-hop deliberation gate).
    `BRAIN_GNW_MULTISTEP` unset / 1 / true / on / yes -> ON. Only an explicit `BRAIN_GNW_MULTISTEP=0` (or false/off/no)
    makes the installed wrapper a pure PASS-THROUGH -> the live turn is byte-identical to the pre-flip default: the
    reversible, byte-identical-provable DISABLE override. (Was default-off through the 6/6-seed live GO; flipped to
    on-by-default after that GO, mirroring how the bus + single-hop deliberation were introduced default-off ->
    verified -> flipped.)"""
    return os.environ.get("BRAIN_GNW_MULTISTEP", "1").strip().lower() not in ("0", "false", "off", "no")


def multistep_lesion_on() -> bool:
    """The load-bearing lesion lever. `BRAIN_GNW_MULTISTEP_LESION` truthy -> run the chase on the workspace with the
    assembly self-recurrence ZEROED -> a hop cannot sustain ignition -> `n_ignited` collapses -> the loop cannot
    converge appropriately -> the multi-step terminal is NOT reached (proving the spiking workspace does the deciding)."""
    return os.environ.get("BRAIN_GNW_MULTISTEP_LESION", "").strip().lower() in ("1", "true", "on", "yes")


def detect_chase(question) -> Optional[str]:
    """If the raw utterance carries an explicit chase marker, return the question with the marker(s) STRIPPED (so the
    unchanged inner parser sees a clean "what does <agent> <action>?"); else return None (no marker -> pass-through).
    Read-only host comprehension of the environment/teacher input — the declared boundary, not the cognition."""
    if not isinstance(question, str):
        return None
    low = question.lower()
    if not any(m in low for m in _CHASE_MARKERS):
        return None
    cleaned = question
    for m in _CHASE_MARKERS:                                  # strip case-insensitively, longest-first
        cleaned = re.sub(re.escape(m), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # tidy a trailing preposition left dangling by the strip ("... does dog chase along ?") + re-attach a '?'
    cleaned = re.sub(r"\b(along|down|through|until)\b\s*\??$", "", cleaned, flags=re.IGNORECASE).strip()
    if cleaned and not cleaned.endswith("?"):
        cleaned = cleaned + "?"
    return cleaned or None


def _all_concepts(composer) -> list:
    """The distinct concept tokens the chase can pick distractors from = every agent/patient string in the store
    (read straight off `composer.kb`, no substrate retrieval). Order-preserving-ish (sorted) for determinism."""
    seen = set()
    for fact, _handle in getattr(composer, "kb", []):
        for role in ("agent", "patient"):
            v = fact.get(role)
            if isinstance(v, str):
                seen.add(v)
    return sorted(seen)


def multistep_chase(chat, agent, action, *, seed: int = _DEFAULT_SEED, lesion: bool = False
                    ) -> Tuple[Optional[str], dict]:
    """Run the keystone EMERGENT-count re-entrant chase on the LIVE composer + the warm P1.2 workspace. The cycle
    count is set by the substrate's own `n_ignited` read (via `confidence_gated_chase` -> `acc_conflict_gate`), NOT a
    host counter. Returns (terminal_concept | None, meta) where meta carries the per-hop trace + resolved_hops."""
    composer = getattr(getattr(chat, "inner", None), "composer", None)
    if composer is None or not hasattr(composer, "query_patient"):
        return None, {"reason": "no_queryable_composer"}
    # the composer can LEARN new facts between turns -> drop the deterministic per-composer query_patient memo so a
    # stale None cannot survive a mid-conversation acquisition (a pure-read cache; clearing changes no result).
    try:
        composer._qp_cache = {}
    except Exception:
        pass
    all_concepts = _all_concepts(composer)
    theta_hi, theta_lo, _cal = _get_theta(seed)
    bridge, xp, slots, snap = _get_bridge(seed, lesion)
    rng = np.random.default_rng(int(seed) * 991 + 7)
    terminal, meta = confidence_gated_chase(
        bridge, xp, slots, snap, composer, agent, action, all_concepts, rng,
        theta_hi, theta_lo, nmda_attr=NMDA_ATTR_DEFAULT, return_trace=True)
    return terminal, meta


def deliberate_multistep(chat, base_svo, question, *, seed: int = _DEFAULT_SEED):
    """Given the inner gate's committed single-hop decision on the (marker-stripped) chase question, run the
    substrate multi-step chase from its (agent, action) and emit the CHAIN TERMINAL (or abstain). Returns
    (svo_out, info). Only a bus-authored covered-class recall on a chase-form question can change; everything else is
    returned UNCHANGED."""
    info = {"acted": False, "reason": None, "agent": None, "action": None, "first_patient": None,
            "terminal": None, "resolved_hops": None, "cycles": None, "halted_at_cap": None,
            "abstained": False, "lesion": multistep_lesion_on(), "n_concepts": None}

    if base_svo is None:                                     # moat: never un-abstain a halted/unstored turn
        info["reason"] = "already_abstained"
        return None, info
    if type(base_svo).__name__ == "HypothesisSVO":           # an open-ended guess is untouched
        info["reason"] = "open_ended_generation"
        return base_svo, info
    if not (isinstance(base_svo, (list, tuple)) and len(base_svo) == 3):
        info["reason"] = "not_svo"
        return base_svo, info

    agent, action = base_svo[0], base_svo[1]
    composer = getattr(getattr(chat, "inner", None), "composer", None)
    if composer is None or not hasattr(composer, "query_patient"):
        info["reason"] = "no_queryable_composer"
        return base_svo, info

    terminal, meta = multistep_chase(chat, agent, action, seed=seed, lesion=info["lesion"])
    info.update({
        "acted": True, "agent": agent, "action": action, "first_patient": base_svo[2],
        "terminal": terminal, "resolved_hops": meta.get("resolved_hops"), "cycles": meta.get("cycles"),
        "halted_at_cap": meta.get("halted_at_cap"), "trace": meta.get("trace"),
        "n_concepts": len(_all_concepts(composer)),
    })
    if terminal is None:                                     # moat: an unstored/over-run chase abstains, never confabulates
        info["abstained"] = True
        info["reason"] = "chase_abstained"
        return None, info
    # emit the chain TERMINAL as the answer to the transitive-chase question (the surface SVO is a rendering of
    # "following <action> from <agent>, the chain ends at <terminal>"; the substrate claim is the terminal + the
    # EMERGENT hop count in info.trace/resolved_hops).
    info["reason"] = "chain_terminal"
    out = list(base_svo)
    out[2] = terminal
    return out, info


def install_multistep_gate(chat, *, seed: int = _DEFAULT_SEED) -> bool:
    """Idempotently wrap `chat.gate` (already the bus + single-hop-deliberation gate) so an explicit chase-form
    question runs the substrate EMERGENT-count multi-step chase. Preserves the pre-multistep gate as
    `chat._gnw_premultistep_gate`; stashes the per-turn info on `chat._last_gnw_multistep`. When `BRAIN_GNW_MULTISTEP`
    is off the wrapper is a pure pass-through. On any exception the inner gate's decision is returned unchanged (a
    turn never crashes). Returns True if it installed (False if already installed). No `sim/` edit."""
    if getattr(chat, "_gnw_multistep_installed", False):
        return False
    inner_gate = chat.gate
    chat._gnw_premultistep_gate = inner_gate

    def _multistep_gate(question):
        if not multistep_enabled():                          # OFF -> pure pass-through (byte-identical)
            chat._last_gnw_multistep = {"acted": False, "reason": "disabled"}
            return inner_gate(question)
        cleaned = detect_chase(question)
        if cleaned is None:                                  # no chase marker -> pure pass-through (byte-identical)
            chat._last_gnw_multistep = {"acted": False, "reason": "not_chase_form"}
            return inner_gate(question)
        base_svo = inner_gate(cleaned)                        # production extracts (agent, action) + first hop ONCE
        try:
            out_svo, info = deliberate_multistep(chat, base_svo, cleaned, seed=seed)
        except Exception as e:                               # never let the chase crash / change a turn on error
            chat._last_gnw_multistep = {"acted": False, "reason": f"error:{type(e).__name__}: {e}"}
            return base_svo
        chat._last_gnw_multistep = info
        return out_svo

    chat.gate = _multistep_gate
    chat._gnw_multistep_installed = True
    return True
