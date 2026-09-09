"""SPIKING CA3 PATTERN-COMPLETION anaphor DETECTION as a per-session organ — the PRODUCTION WIRE-IN of the 6/6-seed-GO
mechanism de-risk (`research/runners/_spiking_anaphor_detection_derisk.py`, finding
`2026-09-09-spiking-anaphor-detection-CA3-pattern-completion-6seed-GO.md`, biology binding
`research/biology/spiking-closed-class-pattern-completion.md`), behind a DEFAULT-OFF flag.

WHAT THIS RETIRES. Two call sites gate the ALREADY-SPIKING referent resolution (`held_referent()` / the WTA
biased-competition read) on a bare host Python `set`-membership DETECTION step — "is the current token one of my known
closed-class pronouns at all?":

    anaphors = {"it","that","they","them","this"}; ... if tl in anaphors:      # brain_chat_tui.ChatBrain._resolve_anaphora
    if not (isinstance(word, str) and word.lower() in _ANAPHORS): return word  # multi_turn_agent.MultiTurnAgent._resolve

Per CLAUDE.md's brain-based-only standard this is a shortcut: a functional judgment ("do I know this closed-class
word?") computed by host bookkeeping, deciding whether the spiking substrate is consulted at all. The de-risk moved that
DECISION onto the substrate as CA3-style autoassociative pattern completion (Kandel PNS 6e: "a few cues are often
sufficient to retrieve a complex stored memory"; Marr's "reactivation of a subset ... would be sufficient to activate
the entire original neural ensemble ... referred to as pattern completion") and proved (6/6 seeds) it recognises a clean
cue, RECOVERS the correct classification from a cue sharing as little as 20% of a stored assembly's neurons (a
capability `x in {...}` structurally cannot have at ANY corruption level), rejects content-word cues, and collapses to
floor when the attractor weights are removed.

THE MECHANISM (reuse-by-import, NO `sim/` edit; VERBATIM the de-risk's circuit + decision). This organ imports the
de-risk's OWN constants + helpers (`ANAPHORS`, `PATTERN_SIZE`, `ATTRACTOR_WEIGHT`, `BUF_KW`, `_unused_pool`,
`_cortex_local_index`, `_fresh_probe`, `decide_pronoun`) so the wired mechanism is byte-for-byte the de-risked one:
`content_selection_spiking.SpikingLoopContextBuffer` — a cortico-PFC NMDA-bistable loop (`cortex_ctx` <-> `dlpfc_wm`)
with one Hebbian outer-product attractor installed per closed-class word. ONE scratch buffer is built per SESSION purely
to read off the deterministic pattern allocation (each anaphor's stored assembly neurons + the unused-neuron pool that no
assembly claims); every CLASSIFICATION then builds its OWN FRESH buffer (`_fresh_probe`) — the de-risk's OWN load-bearing
lesson: the NMDA-bistable assemblies LATCH permanently (persistent WM is this buffer's validated feature), so reusing one
buffer across calls makes every later probe read as already-ignited. A fresh, quiescent buffer per decision is the
correct experimental unit and the correct live unit (one "is this token a pronoun?" decision per token).

WHAT IS ON THE SUBSTRATE (and load-bearing), AND WHAT IS A DECLARED HOST SHORTCUT. The CLASSIFICATION DECISION — does a
driven cue COMPLETE to an ignited stored assembly above threshold (`decide_pronoun` reading `cp_firing_states`) — is on
the substrate, and the lesion proves it (with the attractor weights removed, even a full cue no longer self-sustains
through the silent read window, ignition collapses, and detection reverts). The STRING -> CUE encoding is host
bookkeeping, DECLARED exactly as the de-risk + biology binding already declare it ("the string-to-neuron-index encoding
a live deployment would need" — named there as the residual): `_cue_for_token` maps a live token STRING to a cue over
cortex_ctx neurons. It is deliberately EXACT-token (a known anaphor -> that anaphor's FULL stored assembly; any other
token -> a cue that would be drawn from the unused pool, which de-risk G3 established cannot ignite, so it is reported
not-an-anaphor without a build — equivalent and far cheaper on the common content-word path). Exact-token, NOT a host
string fuzzy-match, because a single-character edit tolerance over a 2-4-letter closed-class vocabulary collides with
common real words ("what"~"that", "the"/"then"~"them"/"they") — a specificity claim this wire-in does not attempt and the
de-risk never validated. So on CLEAN typed text the ON path recognises exactly the host set's tokens, but via the
substrate's ignition decision rather than a Python `in`; the DELIVERABLE is that the decision now lives on the neurons
(retiring the host test), and the substrate's PATTERN-COMPLETION SURPASS over exact-match — recovering a CORRUPTED cue an
`x in {...}` cannot — is the de-risked, proven capability that justifies the substrate detector and is exercised here
through `probe_corrupted_cue` (the same G2/G4 the de-risk gated on) and available for a future noisy-perception path.

CONTRACT (additive, reversible, DEFAULT-OFF). `BRAIN_SPIKING_ANAPHOR` truthy (1/true/on/yes) ARMS the spiking detection
at both call sites; UNSET or in {0,false,no,off,''} (the DEFAULT) leaves each call site's pre-existing host `set` test
UNCHANGED and this organ NEVER BUILT -> byte-identical to pre-wiring. Both call sites also fall back to the host `set`
test on ANY organ error (never raises out). The flip to default-ON is a SEPARATE step, gated on an integrated
`/api/brain-chat` no-regression soak (this wire-in lands default-OFF; see the finding).

LESION (the load-bearing proof). `BRAIN_SPIKING_ANAPHOR_LESION=1` builds every buffer with `attractor_weight=0.0` (the
de-risk's OWN G4 untrained-network lesion): with no recurrent CA3 completion a driven cue cannot self-sustain through the
silent read window, so NOTHING ignites, `is_anaphor` returns False for real anaphors, and the whole downstream
resolution reverts to pass-through. Distinct from any downstream resolution lesion.

RNG ISOLATION (the #77 footgun). Building a `SpikingLoopContextBuffer` reseeds the backend RNG (via `cfg.seed`) and the
stepping shares the process-global RNG the rest of the pipeline uses; enabling this organ must not perturb downstream
RNG-dependent organs. Every substrate build + read runs inside `_isolated`, which snapshots the host numpy + backend RNG,
runs on this organ's OWN private continuous timeline, and restores the host RNG byte-untouched (the same isolation
`DaModeDrivesWorkspace`/`AffectDrivesWorkspace` use). Cue construction uses a per-token LOCAL `np.random.default_rng`
(never the global RNG). When the flag is OFF the organ is never built, so byte-identity is unconditional.

FUNCTIONAL CORRELATE, NOT phenomenal. This reads + reports a spiking recognition CORRELATE; it makes no claim of
subjective familiarity or experience.
"""
from __future__ import annotations

import os
import hashlib
import threading
from typing import List, Optional

import numpy as np


def spiking_anaphor_enabled() -> bool:
    """The master flag, DEFAULT-OFF. `BRAIN_SPIKING_ANAPHOR` truthy (1/true/on/yes) arms the spiking CA3
    pattern-completion anaphor DETECTION at both call sites; UNSET (the default) or in {0,false,no,off,''} leaves each
    call site's pre-existing host `set` membership test unchanged and this organ never built -> byte-identical to
    pre-wiring. Mirrors `spiking_novelty_habituation_organ.spiking_novelty_enabled()`'s default-off semantics (a NEW
    retirement landing default-OFF; the flip to default-ON rides a separate integrated no-regression soak)."""
    return os.environ.get("BRAIN_SPIKING_ANAPHOR", "0").strip().lower() in ("1", "true", "on", "yes")


def spiking_anaphor_lesioned() -> bool:
    """`BRAIN_SPIKING_ANAPHOR_LESION` truthy -> build every buffer with `attractor_weight=0.0` (the de-risk's G4
    untrained-network lesion): with no recurrent CA3 completion a driven cue cannot self-sustain, nothing ignites, and
    detection reverts (real anaphors read as not-an-anaphor -> downstream resolution never fires). The load-bearing
    proof."""
    return os.environ.get("BRAIN_SPIKING_ANAPHOR_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


def _normalize(word: str) -> str:
    """The host token-normalization boundary: lowercase + strip surrounding punctuation, matching the two call sites'
    own `t.lower().strip('.,!?')` / `word.lower()` normalization so the trained-vocabulary comparison is like-for-like."""
    return str(word or "").lower().strip(".,!?;:'\"()[]")


class SpikingAnaphorDetectorOrgan:
    """A per-SESSION spiking CA3 pattern-completion anaphor detector (the detection is per-conversation, exactly like
    the referent WM it gates — so this is per-agent/per-ChatBrain, NOT a process singleton). Caches, per session, the
    deterministic pattern allocation of the de-risked circuit (each anaphor's stored assembly + the unused-neuron pool);
    `is_anaphor(word)` maps the token to a CUE (declared host encoding shortcut) and, when the cue could plausibly match
    a known anaphor, builds a FRESH buffer, drives the cue, and returns whether it COMPLETES to an ignited assembly
    above threshold (the substrate decision). Fresh-per-decision because the NMDA-bistable assemblies latch."""

    def __init__(self, seed: int = 42, lesion: bool = False):
        self.seed = int(seed)
        self.lesion = bool(lesion)
        self._built = False
        self._full_patterns: dict = {}       # anaphor -> its stored assembly's GLOBAL cortex_ctx neuron ids
        self._unused_global = None            # GLOBAL cortex_ctx neuron ids no assembly ever claims (content-word pool)
        self._anaphors: List[str] = []
        self._rng_state = None                # this organ's PRIVATE RNG timeline (host process-global RNG untouched)
        self._lock = threading.Lock()

    # ── RNG isolation (the #77 footgun; copied from DaModeDrivesWorkspace._isolated) ────────────────────────────────
    def _isolated(self, fn):
        """Run `fn()` (a substrate build + spiking read) on this organ's PRIVATE RNG timeline, leaving the host
        process-global RNG (numpy + the sim backend) byte-untouched. Snapshot host RNG, swap in the organ's own
        continuous timeline, run, capture the advanced private timeline, restore host."""
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
            try:
                np.random.set_state(host_np)
            except Exception:
                pass
            if host_xp is not None:
                try:
                    xp.random.get_random_state().set_state(host_xp)
                except Exception:
                    pass

    # ── lazy per-session build: read the deterministic pattern allocation off ONE scratch buffer ────────────────────
    def _ensure(self):
        if self._built:
            return
        # Import the de-risk's OWN constants + helpers so the wired mechanism is verbatim the de-risked one. Deferred
        # to here (not module top) so importing this organ does not eagerly run the de-risk module's SIM_BACKEND
        # setdefault before the caller has chosen a backend.
        from research.runners._spiking_anaphor_detection_derisk import (
            ANAPHORS, ATTRACTOR_WEIGHT, BUF_KW, SpikingLoopContextBuffer, _cortex_local_index, _unused_pool)

        def _build():
            aw = 0.0 if self.lesion else ATTRACTOR_WEIGHT
            scratch = SpikingLoopContextBuffer(ANAPHORS, attractor_weight=aw, seed=self.seed, **BUF_KW)
            cidx = _cortex_local_index(scratch)                       # GLOBAL cortex_ctx neuron ids, local-index order
            unused_local = _unused_pool(self.seed)                    # LOCAL positions no assembly claims
            unused_global = cidx[unused_local]
            assemblies = {c: np.asarray(scratch.B.to_host(scratch._cpat[c]), dtype=np.int64) for c in ANAPHORS}
            del scratch
            return assemblies, unused_global

        assemblies, unused_global = self._isolated(_build)
        self._anaphors = list(ANAPHORS)
        self._full_patterns = assemblies
        self._unused_global = unused_global
        self._built = True

    # ── the string -> cue encoding (DECLARED HOST SHORTCUT; exact-token, see the module docstring) ──────────────────
    def _cue_for_token(self, word: str):
        """Map a live token STRING to a CUE over cortex_ctx neurons. EXACT-token: a known anaphor -> its FULL stored
        assembly (a clean cue the substrate must sustain to ignite); any other token -> None (a cue that would be drawn
        entirely from the unused pool, which de-risk G3 established cannot ignite -> the caller reports not-an-anaphor
        without a build). Returns (cue_global_indices | None, matched_anaphor | None). The 'which anaphor' lookup is host
        bookkeeping; the ignition DECISION on the resulting cue is the substrate's job (and the lesion collapses even
        this full cue -> the substrate, not this lookup, is load-bearing)."""
        w = _normalize(word)
        if w in self._full_patterns:
            return np.asarray(self._full_patterns[w], dtype=np.int64), w
        return None, None

    def _probe(self, cue) -> tuple:
        """Build a FRESH quiescent buffer (attractor_weight=0 under lesion, else the de-risked ATTRACTOR_WEIGHT), drive
        `cue`, and return the de-risk's own `decide_pronoun` verdict (winner_concept | None, peak_rate). RNG-isolated."""
        from research.runners._spiking_anaphor_detection_derisk import ATTRACTOR_WEIGHT, _fresh_probe
        aw = 0.0 if self.lesion else ATTRACTOR_WEIGHT
        return self._isolated(lambda: _fresh_probe(self.seed, aw, np.asarray(cue, dtype=np.int64)))

    # ── the read both call sites use ────────────────────────────────────────────────────────────────────────────────
    def detect(self, word: str) -> dict:
        """Full record: the encoding + the substrate ignition decision. `is_anaphor` is the boolean the call sites use."""
        with self._lock:
            self._ensure()
            cue, matched = self._cue_for_token(word)
            if cue is None:                                          # no closed-class token -> cannot ignite (G3)
                return {"is_anaphor": False, "word": _normalize(word), "matched": None, "cue_built": False,
                        "winner": None, "peak": 0.0, "lesioned": self.lesion, "on": True}
            winner, peak = self._probe(cue)
            return {"is_anaphor": bool(winner is not None), "word": _normalize(word), "matched": matched,
                    "cue_built": True, "winner": winner, "peak": float(peak), "lesioned": self.lesion, "on": True}

    def is_anaphor(self, word: str) -> bool:
        """True iff the token's cue COMPLETES to an ignited stored assembly above threshold on a fresh substrate — the
        spiking replacement for `word.lower() in {...}`. Non-closed-class tokens short-circuit to False (their
        unused-pool cue cannot ignite; de-risk G3)."""
        return bool(self.detect(word)["is_anaphor"])

    # ── the de-risked SURPASS, exercised through the organ (G2/G4): recover a CORRUPTED cue an `x in {...}` cannot ────
    def corrupted_cue(self, anaphor: str, keep_frac: float, rng: Optional[np.random.Generator] = None):
        """Build a NOISY/PARTIAL cue for a known anaphor exactly as the de-risk's G2 does: keep `keep_frac` of its TRUE
        stored assembly neurons and replace the rest with RANDOM unused-region neurons that carry no stored assembly at
        all (an 80%-corrupted rendering at keep_frac=0.2). A cue an exact host string match cannot recognise AT ALL."""
        with self._lock:
            self._ensure()
        a = _normalize(anaphor)
        if a not in self._full_patterns:
            raise KeyError(f"{anaphor!r} is not a known anaphor {list(self._full_patterns)}")
        from research.runners._spiking_anaphor_detection_derisk import PATTERN_SIZE
        rng = rng if rng is not None else np.random.default_rng(
            int.from_bytes(hashlib.md5(f"{a}|{self.seed}|{keep_frac}".encode()).digest()[:4], "little"))
        assembly = np.asarray(self._full_patterns[a], dtype=np.int64)
        n_keep = int(np.clip(round(float(keep_frac) * PATTERN_SIZE), 1, PATTERN_SIZE))
        keep = rng.choice(assembly, size=n_keep, replace=False)
        if n_keep < PATTERN_SIZE:
            fill = rng.choice(self._unused_global, size=PATTERN_SIZE - n_keep, replace=False)
            return np.concatenate([keep, fill]).astype(np.int64)
        return np.asarray(keep, dtype=np.int64)

    def probe_corrupted_cue(self, anaphor: str, keep_frac: float = 0.20,
                            rng: Optional[np.random.Generator] = None) -> dict:
        """Present a corrupted cue for `anaphor` and report whether the substrate completes it to the correct assembly.
        Intact -> the recurrent attractor recovers the full assembly and ignites (winner == anaphor); lesioned
        (attractor_weight=0) -> no completion, no ignition (winner is None). The de-risk's G2 (intact) + G4 (lesion),
        exercised through the organ's own surface — the proof the SUBSTRATE, not the host encoding, does the recovery."""
        cue = self.corrupted_cue(anaphor, keep_frac, rng)
        winner, peak = self._probe(cue)
        return {"anaphor": _normalize(anaphor), "keep_frac": float(keep_frac), "winner": winner,
                "peak": float(peak), "recovered": bool(winner == _normalize(anaphor)), "lesioned": self.lesion}
