"""PROSPECTIVE MEMORY — a spiking INTENTION-LATCH + BA10 cue-MONITOR wired into the PRODUCTION chat turn (Gate-B, 2026-08-13).

THE FACULTY (the missing conversational rung). A genuine conversant remembers to do something LATER: "remind me to
X when Y comes up." That is a deferred INTENTION held ACROSS intervening turns, RELEASED only when its CUE appears —
and NOT before, NOT on a wrong cue. This wires the de-risked GO end-to-end onto the DEFAULT /api/brain-chat turn as a
co-resident spiking organ, exactly as the affect / surprise / comprehension / world-model / curiosity / causal organs
ride the one-brain merge.

REUSE (does not reinvent). It imports the de-risked GO substrate verbatim: `SFANmdaProspectiveMemory` from
`research/runners/_pmem_sfa_nmda_amplifier_derisk.py` (finding 2026-08-13-prospective-sfa-nmda-amplifier-GO.md,
6/6 fire_on_cue, all silence clauses 6/6) — a validated spiking persistent-attractor PFC intention LATCH
(cortex_ctx<->dlpfc_wm per-concept outer-product attractors) + a BA10-style NMDA-recurrent CUE-MONITOR whose
release amplitude is closed by a per-pool intrinsic-plasticity homeostat + a supralinear POOL-GATED NMDA/dendritic-
plateau COINCIDENCE amplifier (Kandel 6e / Larkum 1999 / Schiller & Schiller 2000). Built once per session (seed 42);
the homeostat bias + the plateau threshold are process-cached per seed, so the SECOND session's build reuses the
calibration and only re-instantiates the bridge that HOLDS the intention.

THE PRODUCTION FLOW (per session, a persistent co-resident spiking bridge holds the intention across turns):
  * FORMATION turn — the host language scaffold detects "remind me to X when Y" / "when Y, remind me to X" /
    "when Y, do X"; the intention assembly is LATCHED (`encode_intention`, a self-sustaining cortex<->dlpfc attractor)
    and the cue-word set is stored host-side. A disjoint turn class: it short-circuits with an acknowledgement.
  * INTERVENING turns — each subsequent NON-cue turn ADVANCES the hold (`intervening_turn`, a distractor write = real
    competing WM load); the latch self-sustains (persistence) and the cue-monitor stays SILENT (no premature fire).
  * CUE turn — when the host cue-presence detector (a declared sensory boundary, like curiosity's novelty derivation)
    sees the cue words this turn, the cue assembly is DRIVEN (`present_cue`); the SPIKING coincidence of the HELD
    intention AND the cue ramps the rel accumulator over threshold — the intention FIRES. The DECISION to fire is a
    read off `cp_firing_states` (rel rate >= the frozen FIRE_THR), not a host string match; the reminder is surfaced
    (PREPENDED to whatever the normal turn produces) and the intention is CONSUMED (fires once).

BRAIN-BASED: the HOLD (attractor persistence), the cue-monitoring (NMDA-plateau coincidence integration) and the
RELEASE (the accumulator crossing threshold) are all done by spiking neurons + synapses; every fire/hold read is
`cp_firing_states`, the plateau reads `cp_conductance_g_nmda`. The load-bearing spiking property — the release is
gated by the HELD intention (the coincidence), not the cue alone — is proven by the lesion (zero the latch -> the
held assembly collapses -> the same cue does NOT fire -> the intention is forgotten).

CUE->ACTION BINDING — now LEARNED at formation (WIRED 2026-08-13, default-ON). The cue->action CONTENT binding is no
longer INSTALLED synaptically at build: it is LEARNED via a ONE-SHOT HEBBIAN potentiation at intention-formation
(Gollwitzer implementation-intention; de-risk GO 6/6, `2026-08-13-prospective-hebbian-binding-GO.md`,
`_pmem_hebbian_binding_derisk.HebbianBindingProspectiveMemory`, reuse-by-import). At build the canonical binding is
installed so the homeostat bias + plateau theta calibrate against it (a developmental operating-point tuning), then
it is ZEROED — none exists before the formation turn; `form_intention` relearns it one-shot from the coincident
cue+action+rel_X spiking (a saturating Hebbian outer product), load-bearing on the event. `BRAIN_PMEM_HEBBIAN=0`
reverts to the build-time install (byte-identical to the pre-wiring organ). The BINDING lesion
(`BRAIN_PMEM_HEBBIAN_LESION=1`) latches WITHOUT the event -> the binding stays absent -> the cue cannot fire.

HOST-SCAFFOLD, FLAGGED (the honest residual — declared, narrowed): the host still maps arbitrary intention/cue TEXT
onto the fixed slot-A assembly + derives cue-presence from the turn text (a language/sensory boundary, like the
surprise organ's assertion extraction and curiosity's wh-frame), provides the formation goal-activation drive to
rel_X, and calibrates the pool operating point against the canonical binding strength. The build-time synaptic
INSTALL of the content binding is RETIRED; the engine-native STDP realization of the same local rule is the further
step. So `wired: YES / on_by_default: YES` with the binding retirement rung CLOSED (the remaining scaffolds are the
declared text/sensory boundary + operating-point calibration).

FUNCTIONAL CORRELATE, NOT phenomenal: this measures + reports a prospective-memory CORRELATE (a held-intention x cue
coincidence release). It makes NO claim of subjective intending.

MOAT-SAFE + ADDITIVE: prospective memory never manufactures a fact, flips an abstain, or changes WHICH fact a normal
turn recalls — a FORMATION turn is a disjoint acknowledgement class; a CUE fire only PREPENDS an honest reminder. A
turn that is neither is byte-identical. Default-ON; `BRAIN_PMEM=0` -> the whole organ is skipped (byte-identical
oracle). `BRAIN_PMEM_LESION=1` -> the latch is zeroed after formation (the held assembly collapses) -> the cue does
NOT fire -> NO reminder (lesion-load-bearing: the fire is caused by the spiking latch, not the host cue match).

NO `sim/` edit; reuse-by-import; process backend (cupy in production, numpy in tests).
"""
from __future__ import annotations

import os
import re

# reuse-by-import: the de-risked GO substrate + the FROZEN release thresholds (imported, never re-typed).
from research.runners._pmem_sfa_nmda_amplifier_derisk import SFANmdaProspectiveMemory
from research.runners._pmem_intention_latch_derisk import FIRE_THR, SILENT_MAX

# The single production intention slot. The de-risk substrate carries two action slots (A/B) with their own cue
# assemblies; production uses slot A for the one active deferred intention and keeps B as an UNLATCHED slot the
# verify can drive to prove the cue-monitor is cue-SPECIFIC at the spiking level.
_SLOT = "A"
_OTHER_SLOT = "B"
_ACTIONS = ["A", "B"]
_DISTRACTORS = ["d0", "d1", "d2", "d3"]

# ── host language scaffold: intention FORMATION detection ('remind me to X when Y' / 'when Y, do X') ─────────────
# ACTION-first: "remind me to <ACTION> when <CUE>"  (optional 'i mention/i see/... ' filler inside the cue clause).
_RE_ACTION_FIRST = re.compile(
    r"\bremind me to\b\s+(?P<action>.+?)\s+\bwhen(?:ever)?\b\s+(?P<cue>.+?)\s*[.!]?\s*$", re.I)
# CUE-first with an explicit remind: "when <CUE>[ comes up][,] (please) remind me to <ACTION>".
_RE_CUE_FIRST_REMIND = re.compile(
    r"\bwhen(?:ever)?\b\s+(?P<cue>.+?)(?:\s+comes? up)?\s*,?\s*"
    r"(?:please\s+)?remind me to\s+(?P<action>.+?)\s*[.!]?\s*$", re.I)
# CUE-first imperative (no 'remind'): "when <CUE>[ comes up], <ACTION>" — an implementation-intention "when Y, do X".
# Guarded: must START with 'when', must have a comma-or-'comes up' boundary, must NOT be a question.
_RE_CUE_FIRST_IMPER = re.compile(
    r"^\s*when(?:ever)?\b\s+(?P<cue>.+?)(?:\s+comes? up)?\s*,\s*(?P<action>.+?)\s*[.!]?\s*$", re.I)

# words stripped when reducing a cue clause to its salient content keyword(s) (a host language scaffold, like the
# curiosity organ's topic extractor). The DECISION to fire is the spiking read, never this reduction.
_CUE_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and", "or", "that", "this",
    "these", "those", "it", "its", "they", "them", "i", "you", "we", "me", "us", "my", "your", "our",
    "on", "in", "at", "by", "with", "for", "as", "so", "then", "now", "just", "please", "up", "comes",
    "come", "next", "again", "mention", "mentions", "mentioned", "see", "sees", "saw", "talk", "talks",
    "talked", "about", "bring", "brings", "get", "gets", "hear", "hears", "heard", "there", "here",
}
_WORD_RE = re.compile(r"[a-zA-Z']+")


def pmem_enabled() -> bool:
    """Default-ON. `BRAIN_PMEM` in {0,false,no,off} -> the byte-identical oracle (the organ is fully skipped)."""
    v = os.environ.get("BRAIN_PMEM")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def pmem_lesioned() -> bool:
    """`BRAIN_PMEM_LESION` in {1,true,yes,on} -> zero the latch after formation (load-bearing lesion: the fire dies)."""
    v = os.environ.get("BRAIN_PMEM_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def pmem_hebbian_enabled() -> bool:
    """Default-ON. The cue->action CONTENT binding is LEARNED via a ONE-SHOT HEBBIAN potentiation at intention-
    formation (Gollwitzer implementation-intention; de-risk GO 6/6, `2026-08-13-prospective-hebbian-binding-GO.md`)
    instead of installed synaptically at build. `BRAIN_PMEM_HEBBIAN` in {0,false,no,off} -> the build-time install
    (byte-identical to the pre-wiring production organ)."""
    v = os.environ.get("BRAIN_PMEM_HEBBIAN")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def pmem_hebbian_lesioned() -> bool:
    """`BRAIN_PMEM_HEBBIAN_LESION` in {1,true,yes,on} -> latch the intention WITHOUT the one-shot Hebbian formation
    event -> the cue->action binding stays ABSENT -> the correct cue cannot fire. The BINDING lesion (load-bearing:
    the fire is caused by the learned formation event, not a residual install); DISTINCT from BRAIN_PMEM_LESION, which
    zeroes the HELD assembly (the latch). Only meaningful when the Hebbian binding is enabled."""
    v = os.environ.get("BRAIN_PMEM_HEBBIAN_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def _cue_keywords(cue_clause: str) -> list[str]:
    """Reduce a cue clause to its salient content keyword(s) (a host language scaffold). Falls back to the raw
    tokens when everything is a stop-word (so a bare cue still has something to match)."""
    toks = [w.lower() for w in _WORD_RE.findall(cue_clause or "")]
    content = [t for t in toks if t not in _CUE_STOP and len(t) >= 2]
    return content or [t for t in toks if len(t) >= 2]


def parse_intention(text: str):
    """Host language scaffold: detect an intention-FORMATION utterance and return
    {'action': <action text>, 'cue_clause': <cue text>, 'cue_keywords': [..]} or None.

    Never fires on a question (a trailing '?'), so a normal 'when did the dog sleep?' turn falls through untouched."""
    t = (text or "").strip()
    if not t or t.rstrip().endswith("?"):
        return None
    for rx in (_RE_ACTION_FIRST, _RE_CUE_FIRST_REMIND, _RE_CUE_FIRST_IMPER):
        m = rx.search(t)
        if not m:
            continue
        action = m.group("action").strip(" ,.!")
        cue_clause = m.group("cue").strip(" ,.!")
        kws = _cue_keywords(cue_clause)
        if not action or not cue_clause or not kws:
            continue
        return {"action": action, "cue_clause": cue_clause, "cue_keywords": kws}
    return None


def cue_present(text: str, cue_keywords) -> bool:
    """Host sensory boundary: do the stored cue keyword(s) appear as whole words in this turn's text? (The declared
    boundary — like curiosity's novelty derivation; given the cue is present the SPIKING coincidence decides the fire.)"""
    toks = {w.lower() for w in _WORD_RE.findall(text or "")}
    return any(kw in toks for kw in (cue_keywords or []))


class ProspectiveMemoryOrgan:
    """A PER-SESSION spiking prospective-memory organ: a persistent co-resident bridge (the de-risked GO
    `SFANmdaProspectiveMemory`) that HOLDS a deferred intention across turns, plus the host-side intention CONTENT
    (which arbitrary cue text releases which arbitrary action text — the flagged synaptic-install scaffold).

    State machine (per session):
      idle            -> no intention held.
      held            -> an intention is latched; each intervening turn advances the hold; a cue turn fires it.
    The latch persists in the bridge's recurrent activity between turns (one continuous spiking substrate)."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._pm = None                 # lazily-built prospective-memory substrate (the de-risked GO)
        self._hebbian = None            # is the cue->action binding LEARNED at formation (True) or installed (False)?
        self.action_text = None         # the deferred action (host content: what to remind of)
        self.cue_clause = None          # the cue phrase (host content: the trigger, for the reminder wording)
        self.cue_keywords = None        # the salient cue keyword(s) matched against later turns (host sensory)
        self.held = False               # is an intention currently latched?
        self.lesioned = False           # was this intention formed under the latch lesion (BRAIN_PMEM_LESION)?
        self.hebbian_lesioned = False   # was this intention formed under the binding lesion (BRAIN_PMEM_HEBBIAN_LESION)?
        self.calib = None               # the substrate's calibrated bias + plateau separation margin (debug)
        self.last_read = None           # the most recent read_turn result (held/rel/fired) — always current because
                                        # the handler's prospective block runs BEFORE any disjoint short-circuit, so
                                        # this reflects the hold even on a turn whose RESPONSE omits the debug key.
        self._turn_i = 0                 # intervening-turn counter -> a DETERMINISTIC distractor cycle (matches the
                                         # de-risk's `dists[i % len]`; a randomized hash-picked sequence can land on
                                         # a competing combination that collapses the latch — the GO used the cycle)

    def _ensure_pm(self):
        if self._pm is None:
            self._hebbian = pmem_hebbian_enabled()
            if self._hebbian:
                # LEARN the cue->action binding at formation (de-risk GO 6/6): reuse-by-import the validated
                # HebbianBindingProspectiveMemory subclass (LAZY import — keeps the de-risk runner's env
                # `setdefault`s off the default module-load path; only imported when the binding is learned). The GO
                # config is unchanged; the canonical binding is installed at build so the homeostat bias + plateau
                # theta CALIBRATE against it (a developmental operating-point tuning), then it is ZEROED — none
                # exists before the formation turn, and `form_intention` relearns it one-shot from real spikes.
                from research.runners._pmem_hebbian_binding_derisk import HebbianBindingProspectiveMemory
                self._pm = HebbianBindingProspectiveMemory(
                    _ACTIONS, list(_DISTRACTORS), seed=self.seed,
                    homeostat_on=True, sfa_on=True, plateau_on=True)   # the GO config (plateau owns the closure)
            else:
                # ESCAPE (`BRAIN_PMEM_HEBBIAN=0`): the cue->action binding is INSTALLED synaptically at build
                # (byte-identical to the pre-wiring production organ — the retired scaffold).
                self._pm = SFANmdaProspectiveMemory(
                    _ACTIONS, list(_DISTRACTORS), seed=self.seed,
                    homeostat_on=True, sfa_on=True, plateau_on=True)
            self.calib = {"bias_pA": dict(getattr(self._pm, "_bias_trace", {})),
                          "plateau_diag": dict(getattr(self._pm, "_diag", {})),
                          "fire_thr": float(FIRE_THR), "silent_max": float(SILENT_MAX)}
        return self._pm

    # ---- FORMATION: latch the deferred intention on slot A (LEARNING the cue->action binding one-shot) ----
    def form_intention(self, action_text: str, cue_clause: str, cue_keywords,
                       lesion: bool = False, hebbian_lesion: bool = False) -> dict:
        """Form a fresh deferred intention. With the Hebbian binding (default), a ONE-SHOT HEBBIAN event at formation
        (coincident cue+action+rel_X spiking -> a saturating potentiation of cue_Y->rel_X + act_X->rel_X, Gollwitzer
        implementation-intention) LEARNS the cue->action binding, then latches the self-sustaining cortex<->dlpfc
        attractor. Under the LATCH lesion (`lesion`) the held assembly is zeroed after encoding (the cue will not
        fire); under the BINDING lesion (`hebbian_lesion`) the intention is latched WITHOUT the Hebbian event so the
        binding stays absent (the cue cannot fire — the fire is caused by the learned formation event, not a residual
        install)."""
        pm = self._ensure_pm()
        pm._reset_dynamics()                 # clean substrate state for a fresh latch (calibration is preserved)
        binding_learned = None
        if self._hebbian:
            if hebbian_lesion:
                # BINDING LESION (load-bearing): latch WITHOUT the one-shot Hebbian event -> binding stays absent.
                pm.form_intention_no_hebbian(_SLOT)
            else:
                # LEARN the cue->action binding ONE-SHOT at formation, then latch (retires the build-time install).
                pm.form_intention_hebbian(_SLOT)
            binding_learned = dict(getattr(pm, "_last_form", {}) or {})
        else:
            pm.encode_intention(_SLOT)       # ESCAPE: SPIKING latch on the build-time-INSTALLED binding
        self.action_text = action_text
        self.cue_clause = cue_clause
        self.cue_keywords = list(cue_keywords or [])
        self.held = True
        self.lesioned = bool(lesion)
        self.hebbian_lesioned = bool(hebbian_lesion)
        self._turn_i = 0                     # fresh intervening-turn cycle for this intention
        held_after = None
        if lesion:
            # LOAD-BEARING LATCH LESION: destroy the latch at the substrate level (zero the attractor edges + reset
            # dynamics) so the held assembly collapses and the cue coincidence can no longer fire.
            pm.lesion_latch(_SLOT)
            held_after = float(pm._read(window=20, cue=None)["held"][_SLOT])
        out = {"held": True, "action": action_text, "cue_clause": cue_clause,
               "cue_keywords": self.cue_keywords, "lesioned": bool(lesion),
               "held_after_lesion": held_after, "calib": self.calib}
        if self._hebbian:
            out["hebbian"] = True
            out["hebbian_lesioned"] = bool(hebbian_lesion)
            out["binding_learned"] = binding_learned
        return out

    def clear(self):
        """Consume/forget the held intention (fired once, or a fresh conversation)."""
        self.held = False
        self.action_text = self.cue_clause = self.cue_keywords = None
        self.lesioned = False

    # ---- READ a subsequent turn: cue -> present_cue (fire read); else -> intervening_turn (advance the hold) ----
    def read_turn(self, text: str) -> dict:
        """On a turn AFTER formation: if the cue is present, present the cue and read the SPIKING coincidence fire;
        otherwise advance the hold with a distractor write (persistence) and read the SILENT cue-monitor. Returns
        {'is_cue', 'fired', 'rel', 'threshold', 'held', ...}. `fired` is a read off cp_firing_states (rel >= FIRE_THR)."""
        pm = self._ensure_pm()
        is_cue = cue_present(text, self.cue_keywords)
        if is_cue:
            read = pm.present_cue(_SLOT)         # SPIKING: drive the cue -> the held x cue coincidence ramps rel_A
            rel = float(read["rel"][_SLOT])
            held = float(read["held"][_SLOT])
            fired = bool(rel >= FIRE_THR)
            self.last_read = {"is_cue": True, "fired": fired, "rel": rel, "held": held,
                              "threshold": float(FIRE_THR), "silent_max": float(SILENT_MAX),
                              "lesioned": self.lesioned, "cue_clause": self.cue_clause, "action": self.action_text}
            return dict(self.last_read)
        # intervening / distractor turn: advance the hold (real competing WM load), read persistence + silence.
        # DETERMINISTIC distractor cycle (matches the de-risk's `dists[i % len]` that validated the hold 6/6).
        distractor = _DISTRACTORS[self._turn_i % len(_DISTRACTORS)]
        self._turn_i += 1
        read = pm.intervening_turn(distractor)
        rel = float(read["rel"][_SLOT])
        held = float(read["held"][_SLOT])
        self.last_read = {"is_cue": False, "fired": False, "rel": rel, "held": held,
                          "threshold": float(FIRE_THR), "silent_max": float(SILENT_MAX),
                          "lesioned": self.lesioned, "cue_clause": self.cue_clause, "action": self.action_text}
        return dict(self.last_read)

    def read_named_cue(self, slot: str) -> float:
        """VERIFY hook: drive an arbitrary registered cue slot's assembly and return rel_A's firing rate. Driving the
        UNLATCHED slot B while A is held reads the cue-monitor's SPECIFICITY directly (rel_A must stay silent)."""
        read = self._ensure_pm().present_cue(slot)
        return float(read["rel"][_SLOT])


def reminder_text(action_text: str, cue_clause: str) -> str:
    """The honest prospective REMINDER surfaced when the spiking cue-monitor fires (a host language scaffold framing
    a spiking release; unambiguously a reminder, never a fabricated fact). PREPENDED to the normal turn's answer."""
    cue = (cue_clause or "that").strip()
    act = (action_text or "your intention").strip()
    return f"(Reminder — you asked me to {act} when {cue} came up, and {cue} just came up.) "


def acknowledgement_text(action_text: str, cue_clause: str) -> str:
    """The FORMATION acknowledgement (the disjoint intention-formation turn class). Confirms the intention is held."""
    cue = (cue_clause or "that").strip()
    act = (action_text or "it").strip()
    return f"Okay — I'll hold onto that and remind you to {act} when {cue} comes up."
