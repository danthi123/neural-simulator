"""CURIOSITY — an honest FOLLOW-UP QUESTION on a NOVEL topic, wired into the PRODUCTION turn (Gate-B, D3, 2026-08-12).

The owner's named CURIOSITY faculty: when the brain is asked about something it does NOT hold (the no-confab
moat ABSTAINS), it should not merely refuse — it should CRAVE to learn, and ASK. "Don't refuse when unsure;
seek to learn." This wires the DR-1 curiosity-inversion so a NOVEL topic (the brain's own epistemic gap = it
holds nothing) drives the brain to APPEND an honest FOLLOW-UP QUESTION about the gap, rather than a bare refusal.
The moat is INVERTED, not broken: the answer stays an abstain (never a confabulated fact); the ADDED text is
unambiguously a QUESTION.

It REUSES (does not reinvent) the adversarially-verified DR-1 crave-DRIVE
(`research/runners/_curiosity_seek_learn_onbridge_derisk.build_curiosity_bridge`, on-bridge 6-seed CPU GO;
crave-on-spikes also 6/6-SAFE in the Stage-A step-3 integration): the epistemic-gap scalar `current_novelty_signal`
(the SAME uncertainty signal that drives the no-confab moat) feeds the `curiosity` neuromodulator (the `from_novelty`
production rule) -> an `excitability_drive` on a spiking ASK pool (scope="group:ask") -> the ASK pool SPIKES. The
WANTING is read DIRECTLY from `cp_firing_states[ask]` (Hz). corr(gap, SPIKING-want) = +0.996 (reproduced numpy-CPU
2026-08-12); a HIGH novelty -> a HIGH ASK-pool firing rate -> the brain is CURIOUS. No host `if novel` flag decides
to ask — the DECISION is a threshold on the SPIKING ASK-pool rate (calibrated at build).

BRAIN-BASED: the curiosity signal = a `cp_firing_states[ask]` READ, driven by the `from_novelty` neuromodulator
(already committed additive + default-off in sim/neuromodulators.py + sim/config.py, byte-identical when unused ->
NO new sim/ edit). The only host boundary is the NOVELTY DERIVATION — the abstain (the brain's own memory read: it
holds no answer -> maximal epistemic gap) supplies the novelty scalar, exactly as the surprise organ's sensory
encoding and the metacog organ's role-decode confidence are declared host boundaries with the spiking read-back
load-bearing. The wh-FRAME of the emitted question is a fixed host language scaffold (like the body acting on motor
output); the topic CONTENT is the concept the user surfaced.

MOAT-SAFE + ADDITIVE: curiosity NEVER manufactures a fact, flips an abstain into an assert, or enters the certainty
band. It runs ONLY on an ABSTAIN (the brain already refused — there is no answer to corrupt), and it only APPENDS
an honest QUESTION. A FAMILIAR topic (a confident recall) is OUT OF SCOPE -> byte-identical (no follow-up).
Default-ON; `BRAIN_CURIOSITY=0` -> the byte-identical oracle (fully skipped).

LESION-LOAD-BEARING: the follow-up is caused by the SPIKING ASK-pool firing, not by a host abstain flag. The lesion
REMOVES the curiosity drive PATHWAY (`curiosity_excit_sensitivity=0` -> the `from_novelty` modulator no longer
drives the ASK pool), so even on a NOVEL topic the ASK pool stays at baseline -> the want collapses below threshold
-> NO follow-up. `BRAIN_CURIOSITY_LESION=1` reads the prediction-removed twin. Verified: the SAME novel abstain
craves a follow-up intact and is silent lesioned -> the follow-up is caused by the spiking drive, not the abstain.

HONEST RESIDUALS (declared — the mission's named next rungs, not faked):
  * ONLY the crave-DRIVE is wired (the 6-seed / 6/6-SAFE spiking part). The learning-progress SELECTOR (which of
    several concepts to ask) is a CPU-proxy host formula whose on-bridge memory is seed-fragile (1/6), and the
    noisy-TV VETO is a host ELP TD tracker (survives the critic lesion) — NEITHER is wired: a single-topic chat
    follow-up needs no multi-armed LP selection nor a noise-avoidance veto (there is one gap, the one the user
    surfaced). Those remain the named next rungs.
  * NOVELTY = the ABSTAIN (a binary epistemic gap: the brain holds the concept or it does not), a declared host
    boundary; a graded familiarity-gate novelty (Bogacz-Brown) is the next rung. Curiosity is scoped to ABSTAINS
    (the clearest novelty); a low-confidence RECALL is handled by the metacog hedge (E1) — curiosity on a
    low-confidence recall is a named next rung.
  * CO-RESIDENT on its own curiosity/ASK bridge, ALONGSIDE the recall composer — rides on the one-brain merge
    (burn-down #1), exactly as the affect/surprise/comprehension/metacog/world-model organs do.

FUNCTIONAL CORRELATE, NOT phenomenal: this measures + reports a curiosity CORRELATE (an ASK-pool drive that tracks
the epistemic gap). It makes NO claim of subjective wanting.

NO `sim/` edit; reuse-by-import; process backend (cupy in production, numpy in tests).
"""
from __future__ import annotations

import os
import re
import zlib

import numpy as np

from research.runners._curiosity_seek_learn_onbridge_derisk import (
    build_curiosity_bridge,
    _settle,
    _snapshot_state,
    _restore_state,
    _advance,
    _idx,
    drives_regions,
    W_WANT,
    W_SETTLE,
    WANT_FLOOR_HZ,
)
# reuse-by-import (scaffold-retirement backlog rank-10, 2026-09-05): the SAME Bogacz-Brown anti-Hebbian
# familiarity projector (catalog D.04) the v320 gate (2026-06-11-familiarity-gate-v320-GO.md) and INTEGRATION
# #7's burn-down #2 (2026-08-10, spiking-familiarity-gate-moat-fully-spiking-6seed.md) already use, and the SAME
# genuine time-stepped resonate-and-fire spike-phasor bind (`phase_sum_neuron`, Orchard Algorithm 1) that gate's
# spiking realization uses to render its cue. Neither is reinvented here.
from research.runners.cortex_learned_cleanup_derisk import AntiHebbianFamiliarity
from research.runners.spiking_phasor_fhrr import phases_to_spikes, spikes_to_phases, phase_sum_neuron

# ── the novelty operating points (the epistemic-gap scalar fed to the from_novelty drive) ────────────────────
NOVEL_SIGNAL = 0.95     # an ABSTAIN: the brain holds NO answer -> a maximal epistemic gap (novel)
FAMILIAR_SIGNAL = 0.0   # a held concept: no gap (the calibration low anchor)
N_CONCEPTS = 4          # a tiny ASK organ (only the ASK pool's crave read is load-bearing here)
N_READ_REPS = 4         # average the ASK-pool want over N reads (denoises the OU jitter; the read is drift-free)

# words stripped to expose the salient TOPIC the brain is curious about (a host language scaffold, like the
# surprise organ's assertion extractor; the DECISION to ask is the spiking read, not this).
_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and", "or", "that",
    "this", "these", "those", "it", "its", "they", "them", "he", "she", "his", "her",
    "my", "your", "our", "i", "you", "we", "me", "us", "him", "on", "in", "at", "by", "with",
    "for", "as", "so", "then", "now", "just", "please", "does", "do", "did", "can", "could",
    "would", "will", "should", "any", "some",
}
_WH = {"what", "who", "whom", "whose", "where", "when", "why", "how", "which"}
# meta / speech-act verbs that are not the topic (so "tell me about wombats" -> topic "wombats")
_META_VERBS = {
    "tell", "know", "knows", "knew", "think", "thinks", "say", "says", "said", "explain", "explains",
    "describe", "mean", "means", "meant", "about", "more", "something", "anything", "everything",
    "understand", "wonder", "curious",
}
_WORD_RE = re.compile(r"[a-zA-Z']+")


def curiosity_enabled() -> bool:
    """Default-ON. `BRAIN_CURIOSITY` in {0,false,no,off} -> the byte-identical oracle (fully disabled)."""
    v = os.environ.get("BRAIN_CURIOSITY")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def curiosity_lesioned() -> bool:
    """`BRAIN_CURIOSITY_LESION` in {1,true,yes,on} -> remove the curiosity drive pathway (load-bearing lesion)."""
    v = os.environ.get("BRAIN_CURIOSITY_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def extract_topic(text: str) -> str | None:
    """Return the salient TOPIC content word the brain is curious about (the first content token after stripping
    function / wh / meta-verb words), or None when nothing topical is present (garbage / bare function words ->
    no follow-up). A host language scaffold: it chooses WHICH word to frame, never WHETHER to ask (that is the
    spiking ASK-pool read)."""
    toks = [w.lower() for w in _WORD_RE.findall(text or "")]
    content = [t for t in toks
               if t not in _FUNCTION_WORDS and t not in _WH and t not in _META_VERBS and len(t) >= 2]
    if not content:
        return None
    return content[0]


# ── GRADED novelty for the topic (scaffold-retirement backlog rank-10, 2026-09-05), additive / default-OFF ────
# THE GAP THIS CLOSES. The production call (`webapp/server.py::_curiosity_followup`) always invokes
# `judge(novelty=NOVEL_SIGNAL, ...)` — the SAME host constant (0.95) on EVERY abstain, whatever the topic. That is
# a binary flag ("abstained -> maximally novel"), never a graded read of how novel THIS topic actually is. This
# section retires the constant with a genuine familiarity/mismatch circuit: `TopicNoveltyGate` renders a topic
# word's cue on the SAME genuine spike-phasor bind (`phase_sum_neuron`) the v320 gate uses, and reads its novelty
# through the SAME learned Bogacz-Brown anti-Hebbian projector (`AntiHebbianFamiliarity`) — reuse-by-import, not
# reinvented. A word the brain already holds (imprinted from `_brain_vocab(chat)`, the SAME known-vocabulary
# source the comprehension organ already reads) renders inside the learned span -> LOW novelty; an unrelated word
# renders far outside it -> HIGH novelty (near the old NOVEL_SIGNAL anchor); a noisy/partial cue of a known word
# reads in between — a genuinely CONTINUOUS, topic-dependent value, not a constant.
#
# LESION-LOAD-BEARING: `TopicNoveltyGate.lesion()` clears the learned projector (fresh `AntiHebbianFamiliarity`,
# W==0), so EVERY topic's cue renders at the SAME ceiling energy (‖x‖²=1 for the unit-normalized I/Q render,
# regardless of content) — the gradation collapses. The graded read rides the LEARNED weights, not an artifact of
# the render.
#
# FLIPPED DEFAULT-ON 2026-09-05 (rank-10 production-flip GO, `research/findings/2026-09-05-rank16-rank10-
# production-flip-GO.md`): `graded_novelty_enabled()` now defaults ON (`BRAIN_CURIOSITY_GRADED_NOVELTY` unset) --
# the production call feeds the GRADED per-topic read. `BRAIN_CURIOSITY_GRADED_NOVELTY` in {0,false,off,no}
# (explicit) is the BYTE-IDENTICAL ESCAPE back to the constant `NOVEL_SIGNAL`.
#
# HONEST SCOPE (declared, not hidden): the word->phase code is a FIXED per-word draw (a declared host boundary,
# exactly like the v320 gate's percept->phase projection and the curiosity organ's own wh-frame scaffold) — it
# carries no lexical-semantic structure of its own, so gradation among two DIFFERENT unrelated words is not
# claimed; the validated gradation is FIDELITY-graded (a clean vs. a noisy/partial cue of the SAME word), the
# same axis the v320 gate and the no-confab moat already validate on. The anti-Hebbian basis is also CAPACITY-
# BOUNDED at `2*D` orthogonal directions (D=256 -> 512): production-vocabulary scale is a named next rung, not
# claimed here.
TOPIC_GATE_D = 256   # SAME dimension as the v320 spiking conjunctive familiarity gate


_GRADED_NOVELTY_DEFAULT_ON = True   # FLIPPED 2026-09-05 (rank-10 production-flip GO, 6/6 no-regression)


def graded_novelty_enabled() -> bool:
    """`BRAIN_CURIOSITY_GRADED_NOVELTY` unset -> ON (FLIPPED DEFAULT-ON 2026-09-05, `_GRADED_NOVELTY_DEFAULT_ON`):
    feed the curiosity judge a GRADED per-topic novelty (this section) instead of the constant `NOVEL_SIGNAL`.
    `BRAIN_CURIOSITY_GRADED_NOVELTY` in {0,false,off,no} (explicit) is the BYTE-IDENTICAL ESCAPE back to the
    constant. Any of {1,true,yes,on} also arms it (identical branch to unset, by construction). Flip verify:
    `research/findings/2026-09-05-rank16-rank20-rank10-production-flip-GO.md`."""
    v = os.environ.get("BRAIN_CURIOSITY_GRADED_NOVELTY")
    if v is None:
        return _GRADED_NOVELTY_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "yes", "on")


def graded_novelty_lesioned() -> bool:
    """`BRAIN_CURIOSITY_GRADED_NOVELTY_LESION` truthy -> read the weights-cleared twin (every topic reads the
    same ceiling novelty, regardless of `known_vocab`) — the load-bearing anti-cheat: the graded dependence on
    the brain's OWN vocabulary vanishes. Distinct from `BRAIN_CURIOSITY_LESION` (removes the ASK-pool drive
    pathway) and `BRAIN_CURIOSITY_DA_LESION` (severs the DA->crave-threshold coupling)."""
    v = os.environ.get("BRAIN_CURIOSITY_GRADED_NOVELTY_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def _word_phase(word: str, seed: int, D: int = TOPIC_GATE_D) -> np.ndarray:
    """A FIXED per-word phase code: seeded by a stable hash of `(seed, lowercased word)` so the SAME word always
    renders the SAME clean prototype code within a process/seed, mirroring the project's established fixed-per-
    concept phase-code pattern (`SpikingConjunctiveFamiliarityGate.act_phase`)."""
    h = zlib.crc32(f"{int(seed)}:{str(word).strip().lower()}".encode("utf-8"))
    rng = np.random.default_rng(h)
    return rng.uniform(0.0, 1.0, int(D))


class TopicNoveltyGate:
    """A genuine Bogacz-Brown familiarity/mismatch read for an arbitrary chat TOPIC word. Reuse-by-import: the
    cue is bound via the SAME genuine resonate-and-fire spike-phasor neuron (`phase_sum_neuron`) the v320 gate's
    spiking realization uses; novelty is read by the SAME learned `AntiHebbianFamiliarity` projector (catalog
    D.04). `imprint`/`imprint_vocab` mark words the brain already holds as familiar; `novelty(word)` returns a
    graded [0,1] mismatch energy (0 == an exact imprinted cue, ~1 == unrelated to everything imprinted, or a
    noisy/partial cue of an imprinted word in between). `lesion()` clears the learned weights (collapses the
    gradation — the load-bearing anti-cheat)."""

    def __init__(self, seed: int = 42, D: int = TOPIC_GATE_D):
        self.seed = int(seed)
        self.D = int(D)
        # a FIXED bind partner (like a fixed per-action phase code) so a bare word cue still exercises the
        # genuine spike-phasor bind, exactly as the v320 gate binds (referent_phase, action_phase).
        self._topic_role = _word_phase("__topic_role__", self.seed, self.D)
        self.gate = AntiHebbianFamiliarity(self.D)
        self._imprinted: set[str] = set()

    def _cue(self, word: str, noise: float = 0.0) -> np.ndarray:
        wp = _word_phase(word, self.seed, self.D)
        if noise > 0.0:
            # a NOISY/partial perceptual draw of the word's own code (the project's established env.draw()-vs-
            # env.proto() pattern) — deterministic per (word, noise) for reproducibility across seeds/reads.
            jrng = np.random.default_rng(zlib.crc32(f"{self.seed}:{word}:jit:{noise}".encode("utf-8")))
            wp = np.mod(wp + jrng.normal(0.0, float(noise), self.D), 1.0)
        bound = phase_sum_neuron(phases_to_spikes(wp), phases_to_spikes(self._topic_role))
        return spikes_to_phases(bound)

    def imprint(self, word: str) -> bool:
        """Imprint `word`'s clean prototype cue (mark it FAMILIAR). Idempotent — a word already imprinted is a
        no-op (mirrors `AntiHebbianFamiliarity.imprint`'s own in-span guard). Returns True iff a NEW word was
        actually imprinted."""
        w = str(word).strip().lower()
        if not w or w in self._imprinted:
            return False
        self.gate.imprint(self._cue(w))
        self._imprinted.add(w)
        return True

    def imprint_vocab(self, words) -> int:
        """Imprint every NEW word in `words` (e.g. `_brain_vocab(chat)`). Returns the count newly imprinted."""
        return sum(1 for w in (words or ()) if self.imprint(w))

    def novelty(self, word: str, noise: float = 0.0) -> float:
        """The graded [0,1] mismatch energy for `word` (optionally a noisy/partial cue, `noise` = the per-
        dimension phase-jitter std). Never raises on a non-string `word` (str()-coerced)."""
        return float(self.gate.novelty(self._cue(str(word), noise)))

    def lesion(self) -> None:
        """FULLY clear the learned pool (a fresh `AntiHebbianFamiliarity`) so a subsequent imprint rebuilds W —
        matches `SpikingConjunctiveFamiliarityGate.lesion()`'s own discipline (a pure zero-W lesion would leave
        `_basis` populated and block a later re-imprint from actually rebuilding W)."""
        self.gate = AntiHebbianFamiliarity(self.D)
        self._imprinted = set()


_TOPIC_GATE: dict[int, TopicNoveltyGate] = {}
_TOPIC_GATE_LES: dict[int, TopicNoveltyGate] = {}


def get_topic_gate(seed: int = 42, lesion: bool = False) -> TopicNoveltyGate:
    """The process-shared `TopicNoveltyGate` for `seed` (built once; a SEPARATE, PERMANENTLY-empty lesioned twin
    under `lesion=True` — mirroring `CuriosityProductionOrgan`'s own bridge/les split — never imprinted, so its
    weights stay cleared for the life of the process)."""
    store = _TOPIC_GATE_LES if lesion else _TOPIC_GATE
    g = store.get(seed)
    if g is None:
        g = TopicNoveltyGate(seed=seed)
        store[seed] = g
    return g


def topic_novelty(topic: str | None, known_vocab=(), seed: int = 42, lesion: bool = False) -> float:
    """THE graded per-topic epistemic-gap read that replaces the `NOVEL_SIGNAL` constant. Imprints any word in
    `known_vocab` (e.g. `_brain_vocab(chat)`) not yet imprinted into the process-shared gate, then reads
    `topic`'s novelty. `topic` falsy (nothing extractable) -> the old constant `NOVEL_SIGNAL` (the declared
    fallback). `lesion=True` reads the PERMANENTLY-unimprinted twin (`known_vocab` is never applied to it) — every
    topic reads the same ceiling novelty, the load-bearing anti-cheat. Never raises (degrades to `NOVEL_SIGNAL` on
    any error, so a bad vocab entry can never crash a turn or corrupt the moat)."""
    if not topic:
        return NOVEL_SIGNAL
    try:
        g = get_topic_gate(seed=seed, lesion=lesion)
        if not lesion:
            g.imprint_vocab(known_vocab)
        return g.novelty(topic)
    except Exception:
        return NOVEL_SIGNAL


class CuriosityProductionOrgan:
    """A process-shared spiking curiosity (crave) organ. Built ONCE (lazily): the DR-1 curiosity bridge (the
    `from_novelty` -> ASK-pool excitability drive), plus a build-time calibration of the curious-vs-incurious
    ASK-pool firing threshold from a NOVEL vs FAMILIAR novelty battery. Each read maps the topic's novelty scalar
    to the from_novelty drive, settles the ASK pool, and reads the wanting off `cp_firing_states[ask]`."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._built = False
        self.bridge = self.cfg = self.xp = None
        self.idx_ask = None
        self.snap0 = None
        self.threshold = None
        self.calib = None
        self.les = None            # lazily-built lesioned twin (drive pathway removed)

    def _build_one(self, lesion: bool = False):
        from sim.backend import get_backend
        xp, _ = get_backend()
        bk = {}
        if lesion:
            bk["curiosity_excit_sensitivity"] = 0.0   # remove the from_novelty -> ASK drive (load-bearing lesion)
        bridge, cfg = build_curiosity_bridge(self.seed, N_CONCEPTS, **bk)
        idx_map = {n: xp.asarray(_idx(bridge, n)) for n in drives_regions}
        _settle(bridge, W_SETTLE)                      # clean post-init dynamic state (EMERGE-61 wash-out)
        snap0 = _snapshot_state(bridge)
        return bridge, cfg, xp, idx_map["ask"], snap0

    def ensure_built(self):
        if self._built:
            return
        self.bridge, self.cfg, self.xp, self.idx_ask, self.snap0 = self._build_one(lesion=False)
        # CALIBRATE the curious/incurious threshold from a NOVEL vs FAMILIAR want battery (the same ASK-pool read
        # the production turn uses). Place the threshold in the gap, biased toward the familiar side so a clearly
        # novel topic reliably reads curious; fall back to the de-risk's WANT_FLOOR_HZ if the two do not separate.
        want_novel = self._read_want_raw(NOVEL_SIGNAL, self.bridge, self.xp, self.idx_ask, self.snap0)
        want_fam = self._read_want_raw(FAMILIAR_SIGNAL, self.bridge, self.xp, self.idx_ask, self.snap0)
        if want_novel > want_fam + 1.0:
            self.threshold = float(0.5 * (want_novel + want_fam))
        else:
            self.threshold = float(WANT_FLOOR_HZ)
        self.calib = {"want_novel_hz": float(want_novel), "want_familiar_hz": float(want_fam),
                      "threshold_hz": float(self.threshold), "novel_signal": NOVEL_SIGNAL,
                      "familiar_signal": FAMILIAR_SIGNAL, "read_reps": N_READ_REPS}
        self._built = True

    def _ensure_les(self):
        if self.les is None:
            b, c, xp, idx_ask, snap0 = self._build_one(lesion=True)
            self.les = {"bridge": b, "cfg": c, "xp": xp, "idx_ask": idx_ask, "snap0": snap0}
        return self.les

    def _read_want_raw(self, novelty, bridge, xp, idx_ask, snap0) -> float:
        """The SPIKING ASK-pool wanting (Hz) at epistemic-gap `novelty`, averaged over N_READ_REPS drift-free reads.
        Each read restores the clean post-init state (+ resets the neuromodulator concentrations to baseline), sets
        `current_novelty_signal = novelty` (the from_novelty drive input), advances the ASK pool W_WANT steps, and
        reads `cp_firing_states[ask]`. Reward learning is frozen (a pure crave read)."""
        import numpy as _np
        n_ask = int(len(_idx(bridge, "ask")))
        saved = self.cfg.reward_learning_rate if bridge is self.bridge else bridge.core_config.reward_learning_rate
        vals = []
        for _ in range(N_READ_REPS):
            _restore_state(bridge, snap0)
            bridge.core_config.current_novelty_signal = float(novelty)
            bridge.core_config.reward_learning_rate = 0.0
            spk = 0
            for _ in range(W_WANT):
                _advance(bridge)
                spk += int(bridge.cp_firing_states[idx_ask].sum())
            vals.append(spk / max(n_ask, 1) / (W_WANT * 1e-3))
        bridge.core_config.reward_learning_rate = saved
        _restore_state(bridge, snap0)
        return float(_np.mean(vals))

    def judge(self, novelty: float = NOVEL_SIGNAL, lesion: bool = False) -> dict:
        """Read whether the brain is CURIOUS about a topic whose epistemic gap is `novelty`. Returns the spiking
        ASK-pool wanting (Hz), the calibrated threshold, and `curious` (want >= threshold). A HIGH want -> the honest
        follow-up. `lesion` reads the drive-removed twin (want collapses -> not curious)."""
        self.ensure_built()
        if lesion:
            st = self._ensure_les()
            want = self._read_want_raw(novelty, st["bridge"], st["xp"], st["idx_ask"], st["snap0"])
        else:
            want = self._read_want_raw(novelty, self.bridge, self.xp, self.idx_ask, self.snap0)
        return {"on": True, "lesioned": bool(lesion), "novelty": float(novelty),
                "want_hz": float(want), "threshold": float(self.threshold),
                "curious": bool(want >= self.threshold), "calib": self.calib}

    def salience_of(self, raw: float, lesion: bool = False) -> dict:
        """THE SHARED SPIKING NOVELTY/SALIENCE AFFERENT (scaffold-retirement backlog rank-4, 2026-09-05): the SAME
        ASK-pool spiking transduction `judge()` uses (`current_novelty_signal` -> `from_novelty` -> excitability_drive
        -> the ASK pool fires -> `cp_firing_states[ask]` read, corr(gap,want)=+0.996), generalized to an ARBITRARY
        continuous raw scalar in [0,1] (not just the two abstain-calibration anchors NOVEL_SIGNAL/FAMILIAR_SIGNAL),
        and reported as a NORMALIZED salience against those SAME anchors:
            normalized = (want_hz(raw) - want_hz(FAMILIAR_SIGNAL)) / (want_hz(NOVEL_SIGNAL) - want_hz(FAMILIAR_SIGNAL))
        so normalized ~= raw's position on the organ's own familiar<->novel spiking scale (0 at FAMILIAR, ~1 at NOVEL;
        an input above NOVEL_SIGNAL extrapolates slightly past 1).

        THIS IS THE ONE SHARED AFFERENT other production organs (da_mode_drives_chat's per-turn engagement,
        bg_action_selection_production_organ's SPEAK/STAY-SILENT salience, value_choice_production_organ's per-
        candidate engagement context) read INSTEAD OF computing their own separate host novelty/salience formula --
        REUSE of this already-6-seed-GO ASK-pool crave-drive, not a new mechanism (research/coordination/
        scaffold_retirement_backlog.md rank-4: "Both halves EXIST + are independently de-risked but have never been
        wired to each other or to the live turn -- this is INTEGRATION, not a new mechanism").

        `lesion=True` reads the drive-removed twin (`curiosity_excit_sensitivity=0`, `judge()`'s own lesion): the
        ASK-pool want COLLAPSES to its un-driven baseline regardless of `raw`, so `normalized` loses its dependence
        on the input -- the load-bearing lesion arm every consumer site's de-risk reuses verbatim."""
        self.ensure_built()
        r = float(max(0.0, min(1.0, raw)))
        if lesion:
            st = self._ensure_les()
            want = self._read_want_raw(r, st["bridge"], st["xp"], st["idx_ask"], st["snap0"])
        else:
            want = self._read_want_raw(r, self.bridge, self.xp, self.idx_ask, self.snap0)
        span = float(self.calib["want_novel_hz"] - self.calib["want_familiar_hz"])
        normalized = ((float(want) - self.calib["want_familiar_hz"]) / span) if abs(span) > 1e-9 else 0.0
        return {"raw": r, "want_hz": float(want), "normalized": float(normalized), "lesioned": bool(lesion),
                "calib": self.calib}


_ORGAN: CuriosityProductionOrgan | None = None


def get_organ(seed: int = 42) -> CuriosityProductionOrgan:
    """The process-shared curiosity organ (built once on first use)."""
    global _ORGAN
    if _ORGAN is None:
        _ORGAN = CuriosityProductionOrgan(seed=seed)
    return _ORGAN


def followup_question(topic: str | None) -> str:
    """The honest curiosity FOLLOW-UP QUESTION appended to an abstain when the spiking ASK pool craves. A
    FUNCTIONAL read of the spiking curiosity drive — unambiguously a QUESTION, never a confabulated fact. The
    wh-FRAME is a fixed host language scaffold; `topic` is the concept the user surfaced (None -> a generic ask)."""
    if topic:
        return (f" My curiosity is piqued — I haven't learned about {topic} yet: "
                f"what can you tell me about {topic}?")
    return " My curiosity is piqued — I haven't learned that yet: can you tell me more so I can learn?"
