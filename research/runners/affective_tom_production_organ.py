"""W5 AFFECTIVE THEORY OF MIND wired into the PRODUCTION conversational turn (board #W5, 2026-08-26).

This is the production-integration glue that makes the brain infer ANOTHER agent's EMOTION (valence) from THAT
agent's WITNESSED situation and lead its reply with an EMPATHIC expression -- default-OFF for now (the parent flips
it default-ON after the pool soak), moat-safe, lesion-load-bearing. It REUSES (does not reinvent) the
adversarially-verified 6/6-seed-GO W5 de-risk (`research/runners/_affective_tom_derisk.py`,
`research/findings/2026-08-01-W5-affective-theory-of-mind-6seed-GO.md`):

  * the OTHER-tagged affect model = a P0.3 `AffectStateBrain` (verbatim import from `_affect_state_region_derisk`,
    the opponent slow-NMDA attractor: affect_vplus/vminus + Namburi-Tye cross-inhibition). It is DISSOCIABLE from the
    system's OWN affect (the #84 `affect_drives_chat` ladder) by construction -- a SEPARATE region driven ONLY by the
    OTHER agent's situation valence, never by the system's mood. This is the "separate slot per agent" motif W3 uses
    for belief, turned onto the affect region (Shamay-Tsoory affective perspective-taking / cognitive empathy).
  * the tone read = the de-risk's EXACT `read_tone(brain, valence_sign, lesion, settle_ms)` (reuse-by-import): drive
    the OTHER model on the appraised OTHER-situation valence sign, then read the SYNAPTIC differential
    rate(recall_pos) - rate(recall_neg) off the ONE `affect_out` transmission gate. `tone_sign = +1` share-joy,
    `-1` comfort, `0` neutral (|differential| below `_NEUTRAL_TOL`). The differential is NEVER host-set: it is a
    difference of two SPIKE RATES from the OTHER-tagged region; you cannot get it without running that region.
  * the appraisal VALUE source = the SAME Gate-B DR-2 learned distributional valence the wired affect organs use
    (`affect_production_organ.appraise_text`). n_hits==0 (no strongly-affective word) -> no empathic trigger (the
    turn is byte-identical). This is the legitimate world/perceptual input (P0.3's interface, DR-2 learned-tag
    precedent); the ToM-specific NEURAL work is (a) the SEPARATE OTHER-tagged affect state and (b) the synaptic tone.

THE COUPLING (what makes it LOAD-BEARING, not observe-only). On a turn describing ANOTHER agent's affectively-charged
situation ("Maria is devastated", "Sam's team lost", "my friend won the award") or an explicit "how does X feel"
query, the reply LEADS with a graded empathic expression sourced from the OTHER model's neural tone:
`tone_sign=-1 -> "that sounds really hard for {agent} -- "`, `+1 -> "that's wonderful for {agent} -- "`, `0 ->` no
lead. The lead is an honest EXPRESSION of the inferred OTHER emotion (the empathic prosody a listener renders), NOT
content: the FACT after it is the SAME gate-matched, moat-verified answer, and the VERIFY re-parse is unchanged. So
empathy changes HOW the reply opens, never WHICH fact is true and never whether an unmatched cue abstains.

THE HONESTY FLOOR (preserved BY CONSTRUCTION, mirrors the #84 affect path). The moat / recall / abstain verdict runs
FIRST and unchanged; the empathic lead only DECORATES an already-matched surface. It never enters the certainty band,
never manufactures a fact, never flips an abstain into an assert. The content fields (`abstained`, `recalled_svo`,
`verified`) are BYTE-IDENTICAL with the coupling on or off; only the answer SURFACE (the empathic lead) and the
additive `affective_tom` trace change.

LESION (the load-bearing / brain-based proof). `BRAIN_AFFECTIVE_TOM_LESION=1` clamps the OTHER model's `affect_out`
gate to 0 (`set_affect_lesion(True)`) on every read -> the two recall pools receive NO bias from the OTHER affect
state -> the differential collapses to ~0 (verified: intact |differential| ~0.067, lesion == 0.0000 exactly) -> the
tone is neutral -> the empathic lead VANISHES and the reply reverts to the flag-off surface. So the surface change
RIDES the SPIKING OTHER-region read, not a host `if valence<0`: kill the neural OTHER read and the empathic tone
disappears even though the appraised valence is unchanged. This is the finding's egocentric|incongruent=0.000 vs
other|incongruent=1.000 dissociation, in production form: an INCONGRUENT case (the OTHER feels bad while the system's
own affect is neutral) collapses to neutral under the OTHER-region lesion while the content stays byte-identical.

CONTRACT (additive, reversible, byte-identical-off).
  * `affective_tom_enabled()` gates the whole block. DISABLED -> the handler skips it entirely: no OTHER model is
    built, no read runs, no `affective_tom` key is attached, no empathic lead is prepended -> the turn is
    BYTE-IDENTICAL to pre-wiring (orthogonal to the #84 self-affect path, which is untouched).
  * An ORDINARY turn (no other-agent situation, or no affective word) is byte-identical even with the flag ON:
    `observe_turn` returns an inert no-lead info and never builds/reads the bridge (so it cannot perturb the
    downstream RNG-dependent organs). The faculty changes only its TRIGGERED turns -- the soak-gate property.
  * The OTHER-model build + read run on a PRIVATE RNG timeline (the host process-global RNG -- numpy + the sim
    backend -- is snapshotted and restored around them; the #77 global-RNG footgun): a triggered turn cannot perturb
    the downstream organs, so the OTHER response fields stay byte-identical.

REUSE-BY-IMPORT (NO `sim/` edit). The OTHER-tagged region (`AffectStateBrain`), the tone read (`read_tone`) and the
read protocol come STRAIGHT from the W5 de-risk (6/6-seed GO). The appraisal comes from
`affect_production_organ.appraise_text` (the Gate-B host-comprehension boundary). This module adds only the
production glue (the other-agent trigger detector + the tone->empathic-expression map). `git diff sim/` is empty.

HONEST RESIDUALS (named, not claimed closed).
  1. The message->OTHER-situation valence APPRAISAL is host (a language-comprehension boundary, like the SVO parser).
     The tone READ (the OTHER-tagged affect state -> the synaptic recall differential) and its `affect_out`
     dependence ARE the neural W5 mechanism (lesion-proven).
  2. The other-agent DETECTION (who the third party is, and their display name) is a host regex boundary -- a
     language/comprehension boundary, like curiosity's wh-frame or the prospective-memory cue text.
  3. The tone_sign->EXPRESSION-MARKER string is a HOST conditioned-articulation scaffold (the "mouth"): the tone that
     DRIVES it is the neural OTHER read (load-bearing -- the lesion collapses the marker), but the surface STRING for
     a given sign is a host template (the sanctioned articulation-crutch pattern: scaffold-ok IF the faculty is
     load-bearing on the tone, which the lesion proves). A brain-native empathic mouth is the named next rung.
  4. Scoped to VALENCE (good/bad) -- matches the P0.3 bistable good/bad latch (QUALIFIED-GO/BOUNDARY). Fine discrete
     emotions need the SAME graded-circumplex surpass P0.3 named, NOT a new wall. A FUNCTIONAL affective-mentalizing
     correlate (a separate, other-driven, dissociable affect attribution), NOT a claim of access to another mind.
  5. This module reads its OWN co-resident OTHER-tagged bridge, run ALONGSIDE the recall composer, not merged onto the
     single recall bridge (the one-brain consolidation step, shared with the #84 / Gate-B affect burn-down).

Backend: uses the process backend (cupy in production, numpy in tests) -- NO global-backend flip. NO `sim/` edit;
additive; default-OFF anchor with the `BRAIN_AFFECTIVE_TOM` env escape.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Optional

import numpy as np

# reuse-by-import the W5 6/6-seed-GO de-risk organ (the OTHER-tagged region + its EXACT tone read) -- NO sim/ edit.
from research.runners._affect_state_region_derisk import AffectStateBrain
from research.runners._affective_tom_derisk import read_tone, SETTLE_BASE

_DEFAULT_SEED = 42

# ── read window: a FIXED settle (the read is deterministic since read_tone.reset() re-seeds cfg.seed each call). The
#    de-risk jitters settle across trials only to average genuine noise-phase variance; a single production read uses
#    a fixed settle in the same band (SETTLE_BASE + a small constant) -> the same intact separation, deterministically.
_READ_SETTLE_MS = SETTLE_BASE + 20   # = 60 ms; matches the de-risk's per-trial settle band (40..80)

# ── |differential| below this reads NEUTRAL (tone_sign 0 -> NO lead). Calibrated on the de-risk read: intact
#    |differential| ~0.067 (both signs), the OTHER-region lesion collapses it to 0.0000 EXACTLY (affect_out=0 removes
#    all bias from the two recall pools, which then fire only from the equal recall cue). 0.020 separates them with a
#    wide margin on every seed -> the lesion cleanly collapses the empathic tone to neutral.
_NEUTRAL_TOL = 0.020


def affective_tom_enabled() -> bool:
    """The master flag. `BRAIN_AFFECTIVE_TOM` truthy (1/true/on/yes) enables; anything else (incl. UNSET) disables.
    Default-OFF for now -- the parent flips the server anchor default-ON after the pool soak passes. Server combines
    this with its `_AFFECTIVE_TOM_DEFAULT_ON` anchor (mirroring the #84/#85 flags)."""
    return os.environ.get("BRAIN_AFFECTIVE_TOM", "0").strip().lower() in ("1", "true", "on", "yes")


def affective_tom_off() -> bool:
    """Explicit OFF (for a default-ON anchor once the parent flips it): `BRAIN_AFFECTIVE_TOM` in {0,false,no,off}."""
    v = os.environ.get("BRAIN_AFFECTIVE_TOM")
    return v is not None and v.strip().lower() in ("0", "false", "no", "off", "")


def affective_tom_lesioned() -> bool:
    """`BRAIN_AFFECTIVE_TOM_LESION` truthy -> clamp the OTHER model's `affect_out` gate to 0 on every read (the
    finding's other-output lesion): the recall differential collapses -> the empathic tone is neutral -> the lead
    VANISHES. The load-bearing proof (the tone rides the OTHER-region read, not the host appraisal)."""
    return os.environ.get("BRAIN_AFFECTIVE_TOM_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


def affective_tom_graded_enabled() -> bool:
    """The GRADED-CIRCUMPLEX upgrade flag (NEW, 2026-09-01, additive, default-OFF). When truthy
    (`BRAIN_AFFECTIVE_TOM_GRADED` in 1/true/on/yes), the OTHER-model read swaps from the bistable valence-SIGN
    (`AffectStateBrain` / `read_tone`, 3-state {-1,0,+1}) to the GRADED valence x arousal circumplex -- the SAME
    #81 Koulakov bistable-LADDER the SELF read already uses (`research.runners._affective_tom_graded_derisk`).
    2026-09-01 AUTO-FLIPPED default-ON (`BRAIN_AFFECTIVE_TOM_GRADED` unset -> ON; set =0 to opt out) per the
    auto-flip policy: organ 6/6 GO + full-brain HANDLER soak 6/6 GO (ordinary_identical + triggered_content_identical
    + lesion_collapsed all True, research/findings/raw/_affective_tom_graded/flip_soak_summary_6seed.json). Off/lesion
    stays byte-identical to the shipped bistable path (the graded module is not even imported when off), so the base
    `_AFFECTIVE_TOM_DEFAULT_ON=True` faculty is unchanged; when on, the OTHER-model read swaps to the 7-tier graded
    empathic lead."""
    return os.environ.get("BRAIN_AFFECTIVE_TOM_GRADED", "1").strip().lower() in ("1", "true", "on", "yes")


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# THE OTHER-AGENT TRIGGER DETECTOR (a host language-comprehension boundary; the ToM-specific neural work is the
# OTHER-tagged affect read below, NOT this parse). Two classes: (A) a third party's affectively-charged situation
# ("Maria is devastated", "Sam's team lost", "my friend won"), and (B) an explicit "how does X feel" query. First
# person (I/we/you) is EXCLUDED -- that is the self / the system's own affect (the #84 path / the feel-query path).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
_FIRST_PERSON = {"i", "we", "you", "me", "us", "myself", "ourselves", "yourself"}
# a proper-name subject followed by a state/predicate verb (Maria is devastated / Tom is furious / Sam won ...).
_NAME_SUBJ_RE = re.compile(
    r"\b([A-Z][a-z]+)\s+(?:is|was|are|were|'s|has|had|got|gets|feels?|felt|seems?|seemed|looks?|looked|"
    r"lost|loses|won|wins|failed|fails|passed|passes|died|survived)\b")
# a possessive proper name (Sam's team lost / Maria's project ...).
_NAME_POSS_RE = re.compile(r"\b([A-Z][a-z]+)'s\b")
# "my <relation>" (the user's third party): my friend / my sister / my colleague ... -> display "your <relation>".
_MY_RELATION_RE = re.compile(
    r"\bmy\s+(friend|friends|sister|brother|mother|father|mom|dad|mum|colleague|coworker|co-worker|"
    r"neighbou?r|partner|wife|husband|son|daughter|boss|teacher|classmate|cousin|aunt|uncle|"
    r"grandmother|grandfather|grandma|grandpa|parent|parents|child|kid|kids|roommate|teammate)\b",
    re.IGNORECASE)
# an explicit affective-ToM query: "how does X feel" / "how is X feeling" (X not first/second person).
_FEEL_QUERY_RE = re.compile(
    r"\bhow\s+(?:does|do|is|are|did|was|were)\s+([A-Za-z][A-Za-z']*)\s+feel(?:ing)?\b", re.IGNORECASE)


def detect_other_agent(text: str) -> Optional[str]:
    """Return a DISPLAY NAME for the third party this turn is about, or None. Host comprehension boundary. Excludes
    first/second person (that is the self / the system). Order: possessive name > name-subject > my-relation >
    explicit feel-query. Returns e.g. 'Sam', 'your friend', 'Maria'."""
    t = text or ""
    m = _NAME_POSS_RE.search(t)
    if m and m.group(1).lower() not in _FIRST_PERSON:
        return m.group(1)
    m = _NAME_SUBJ_RE.search(t)
    if m and m.group(1).lower() not in _FIRST_PERSON:
        return m.group(1)
    m = _MY_RELATION_RE.search(t)
    if m:
        return "your " + m.group(1).lower()
    m = _FEEL_QUERY_RE.search(t)
    if m and m.group(1).lower() not in _FIRST_PERSON:
        return m.group(1)
    return None


# ── the tone_sign -> empathic EXPRESSION marker (the host conditioned-articulation scaffold; DRIVEN by the neural
#    OTHER read -- the lesion collapses tone_sign to 0, so the lead VANISHES). Sign 0 (neutral / lesion) -> '' so the
#    surface is byte-identical. The FACT after the lead is unchanged (VERIFY re-parse intact).
def empathic_lead(tone_sign: int, agent: str) -> str:
    """The empathic lead for this turn's inferred OTHER emotion. tone_sign from the neural OTHER read: -1 comfort,
    +1 share-joy, 0 (neutral / lesion-collapsed) -> '' (no lead, byte-identical surface)."""
    a = str(agent or "them")
    if int(tone_sign) < 0:
        return "That sounds really hard for %s -- " % a
    if int(tone_sign) > 0:
        return "That's wonderful for %s -- " % a
    return ""


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# THE PROCESS-SHARED OTHER-TAGGED AFFECT ORGAN. Built ONCE (lazily); each read RESETS + drives the OTHER model on the
# appraised OTHER-situation valence sign and reads the synaptic recall differential through `affect_out`. Snapshot/
# restore-isolated (a read leaves the host process-global RNG BYTE-UNTOUCHED -- the #77 footgun).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class AffectiveToMOrgan:
    def __init__(self, seed: int = _DEFAULT_SEED):
        self.seed = int(seed)
        self._brain = None
        self._lock = threading.Lock()

    def _isolated(self, fn):
        """Run `fn()` (the OTHER-model build / read) leaving the host process-global RNG (numpy + the sim backend)
        BYTE-UNTOUCHED. `AffectStateBrain(...)` and `read_tone`'s `brain.reset()` both re-seed the backend global RNG
        from cfg.seed and step it (OU noise) off the SAME process-global RNG the rest of the pipeline shares --
        without this, a triggered turn would perturb the downstream RNG-dependent organs and break byte-identity.
        Snapshot the host RNG, run, restore. (Same #77 fix as affect_drives_chat._isolated; no private timeline is
        kept because read_tone.reset() re-seeds each read -> the read is deterministic regardless of incoming state.)"""
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
        try:
            return fn()
        finally:
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
        if self._brain is None:
            self._brain = AffectStateBrain(self.seed, nmda_on=True)

    def read_other_tone(self, valence_sign: int, *, lesion: bool = False,
                        settle_ms: int = _READ_SETTLE_MS) -> dict:
        """Drive the OTHER model on the appraised OTHER-situation valence sign (+1 good / -1 bad) and read the
        synaptic recall differential through `affect_out`. `lesion` clamps affect_out=0 -> the differential collapses.
        Returns the differential + the graded tone_sign (0 when |differential| < _NEUTRAL_TOL, incl. under lesion)."""
        with self._lock:
            self._isolated(self._ensure)
            ts, pos, neg, mood = self._isolated(
                lambda: read_tone(self._brain, int(valence_sign), lesion=bool(lesion), settle_ms=int(settle_ms)))
        diff = float(pos) - float(neg)
        tone_sign = 0 if abs(diff) < _NEUTRAL_TOL else (1 if diff > 0.0 else -1)
        return {"tone_sign": int(tone_sign), "differential": float(diff), "pos_rate": float(pos),
                "neg_rate": float(neg), "raw_tone": int(ts), "mood": float(mood),
                "valence_sign": int(valence_sign), "lesioned": bool(lesion)}


_ORGAN: Optional[AffectiveToMOrgan] = None


def get_organ(seed: int = _DEFAULT_SEED) -> AffectiveToMOrgan:
    """The process-shared OTHER-tagged affect organ (built once on first use)."""
    global _ORGAN
    if _ORGAN is None:
        _ORGAN = AffectiveToMOrgan(seed=seed)
    return _ORGAN


def observe_turn(chat, message: str, appraisal: Optional[dict] = None, *,
                 seed: int = _DEFAULT_SEED) -> dict:
    """The production entry point. Detect whether this turn is about ANOTHER agent (host comprehension boundary);
    if so, appraise the OTHER-situation valence (reuse the Gate-B `affect_production_organ` DR-2 appraisal), drive
    the OTHER-tagged affect model on that valence sign, read the neural tone, and build the empathic lead. Returns the
    per-turn `affective_tom` info (also stashed on `chat._last_affective_tom`). Never raises out (on any error it
    returns an inert no-lead info dict so a turn can never crash).

    An ORDINARY turn (no other-agent, or no affective word) returns an inert no-lead info WITHOUT building/reading the
    bridge -> byte-identical + no RNG perturbation. The faculty changes only its TRIGGERED turns."""
    info = {"acted": False, "lead": "", "agent": None, "tone_sign": 0, "reason": None, "seed": int(seed)}
    try:
        agent = detect_other_agent(message)
        if agent is None:
            info["reason"] = "no_other_agent"
            if chat is not None:
                chat._last_affective_tom = info
            return info
        if appraisal is None:
            from research.runners import affect_production_organ as _AO
            appraisal = _AO.appraise_text(message)
        n_hits = int(appraisal.get("n_hits", 0))
        valence = float(appraisal.get("valence", 0.0))
        info.update({"agent": agent, "valence": valence, "n_hits": n_hits})
        if n_hits <= 0 or valence == 0.0:
            info["reason"] = "no_affective_content"   # nothing to be empathic about -> no lead (byte-identical)
            if chat is not None:
                chat._last_affective_tom = info
            return info
        valence_sign = 1 if valence > 0.0 else -1
        lesion = affective_tom_lesioned()
        if affective_tom_graded_enabled():
            # THE GRADED-CIRCUMPLEX UPGRADE (additive, default-OFF): the OTHER-model read consumes the FULL
            # appraised (valence, arousal) MAGNITUDE through the SAME #81 graded ladder the SELF read uses, instead
            # of collapsing it to a bistable sign. Reuse-by-import; no change to the bistable path above/below.
            from research.runners._affective_tom_graded_derisk import get_graded_organ, empathic_lead_graded
            arousal = float(appraisal.get("arousal", 0.0))
            read = get_graded_organ(seed=seed).read_other_state(valence, arousal, lesion=lesion)
            tone_level = int(read["tone_level"])
            lead = empathic_lead_graded(tone_level, agent)
            info.update({"acted": True, "lead": lead, "tone_level": tone_level, "mood": read["mood"],
                         "felt_arousal": read["felt_arousal"], "valence_sign": valence_sign,
                         "lesioned": bool(lesion), "graded": True,
                         "reason": ("lesion_collapsed" if lesion else
                                    ("neutral" if tone_level == 0 else "empathic"))})
        else:
            read = get_organ(seed=seed).read_other_tone(valence_sign, lesion=lesion)
            tone_sign = int(read["tone_sign"])
            lead = empathic_lead(tone_sign, agent)
            info.update({"acted": True, "lead": lead, "tone_sign": tone_sign,
                         "differential": read["differential"], "pos_rate": read["pos_rate"],
                         "neg_rate": read["neg_rate"], "valence_sign": valence_sign, "lesioned": bool(lesion),
                         "reason": ("lesion_collapsed" if lesion else
                                    ("neutral" if tone_sign == 0 else "empathic"))})
    except Exception as e:   # never let the empathy read crash / change a turn
        info = {"acted": False, "lead": "", "agent": None, "tone_sign": 0,
                "reason": f"error:{type(e).__name__}: {e}", "seed": int(seed)}
    if chat is not None:
        chat._last_affective_tom = info
    return info
