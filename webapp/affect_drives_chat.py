"""The #81 GRADED-AFFECT ladder wired into the LIVE `/api/brain-chat` turn so the brain's felt valence x arousal is
LOAD-BEARING on what/how it responds -- NOT observe-only (board #84, INTEGRATION-TO-PRODUCTION).

WHAT THIS IS. Board #81 landed a 6/6-seed GO graded-affect substrate (`_graded_affect_attractor_derisk`): a
Koulakov-2002/Goldman-2003 robust-integrator LADDER of independently-latched bistable NMDA sub-pools reads the
brain's #49 interoceptive body-state as a SMOOTH valence x arousal (Pearson +0.97 / +0.95; the embodiment lesion --
cutting the interoceptive->ladder synapses -- collapses it to 0). That GO shipped as a DEFAULT-OFF de-risk RUNNER,
never in the live chat. This module wires that NEURAL read onto the live conversational brain and makes it CHANGE
the response: the felt affect state colors the AFFECTIVE EXPRESSION the reply leads with (a graded warmth/curtness
marker), and the forthcomingness the reply plans for. It is the anti-hollow-integration counterpart to the
observe-only faculties: the affect READ is neural AND it demonstrably shapes the surface.

THE READ (the #81 neural mechanism, reused-by-import; NO sim/ edit).
  * Each turn, the message's affective valence/arousal (the SAME host-comprehension boundary the SVO question parser
    and the Gate-B appraisal occupy -- `affect_production_organ.appraise_text`, DR-2 learned distributional valence)
    is EMA-folded into a persistent per-session BODY-STATE (h = comfort/homeostasis, a = bodily arousal). A neutral
    turn (no strongly-affective word) HOLDS the prior body-state -> cross-turn affect PERSISTENCE.
  * That body-state drives the #81 ladder through the interoceptive relays (`read_body`), and the FELT state is the
    ladder's OWN population read off `cp_firing_states`: mood = rate(V+ ladder) - rate(V- ladder) (graded valence),
    felt_arousal = rate(arousal ladder). NEVER a host formula. Positive body-state (h>0.5) latches more V+ sub-pools
    (mood -> +); negative (h<0.5) latches more V- (mood -> -); the set-point h=0.5 reads ~0.
  * The graded mood differential is binned into a Koulakov staircase LEVEL (-3..+3) from calibrated thresholds on
    the #81 mood scale (mood ranges ~[-0.08, +0.08]; the set-point band is neutral).

THE COUPLING (what makes it LOAD-BEARING, not observe-only).
  * VALENCE -> the affective EXPRESSION the reply leads with: a graded warmth/curtness discourse marker prepended to
    the answer surface ("Wonderful — <fact>", "Sure — <fact>", neutral -> no lead, "Honestly — <fact>",
    "Frankly — <fact>"). AROUSAL -> the marker's emphasis (high felt-arousal -> "! ", else " — "). The marker is an
    honest EXPRESSION of the read affect state (tone-of-voice / prosody the body renders), NOT content: the FACT
    after it is the SAME gate-matched, moat-verified SVO, and the VERIFY re-parse is unchanged. So affect changes
    HOW the reply sounds, never WHICH fact is true and never whether an unmatched cue abstains. (This is the single
    coupling this module wires; the Gate-B `BRAIN_AFFECT` path independently colors prose-manner + forthcomingness.)

THE HONESTY FLOOR (preserved BY CONSTRUCTION, mirrors the Gate-B affect path).
  * The moat / recall / abstain verdict runs FIRST and unchanged; the affect coupling only DECORATES an
    already-matched answer surface. It never enters the certainty band, never manufactures a fact, never flips an
    abstain into an assert. The content fields (`abstained`, `recalled_svo`, `verified`) are BYTE-IDENTICAL with the
    coupling on or off; only the answer SURFACE (tone) and the additive `affect_drives` trace change.

LESION (the load-bearing / brain-based proof). `BRAIN_AFFECT_DRIVES_LESION=1` cuts the interoceptive->ladder
synapses (`intero_out` gate=0, the #81 embodiment lesion) on every read -> the neural mood differential collapses
to ~0 -> the staircase level is 0 -> the affective lead VANISHES and the answer surface reverts to the neutral
(coupling-off) surface. So the surface change RIDES the SPIKING ladder read, not a host `if valence>0`: kill the
neural read and the tone-difference disappears.

CONTRACT (additive, reversible, byte-identical-off).
  * `affect_drives_enabled()` gates the whole block. When DISABLED the handler skips it entirely: no workspace is
    built, no read runs, no `affect_drives` key is attached, and NO affective lead is prepended -> the turn is
    BYTE-IDENTICAL to pre-wiring (the Gate-B affect path, if on, is untouched -- this module is orthogonal to it).
  * The ladder is run on the workspace's PRIVATE RNG timeline and the host process-global RNG (numpy + the sim
    backend) is restored around every read (the #77 global-RNG footgun): enabling this module cannot perturb the
    downstream RNG-dependent organs, so the OTHER response fields stay byte-identical.
  * The ladder build (~0.4s) is lazy on the first turn per session and kept warm; each turn runs one ~0.15s read.

REUSE-BY-IMPORT (NO `sim/` edit). The graded-affect ladder build (`GradedAffectBrain`), the neural body->felt read
(`read_body`) and the operating point come STRAIGHT from `_graded_affect_attractor_derisk` (board #81, 6/6-seed GO).
The appraisal comes from `affect_production_organ.appraise_text` (the Gate-B host-comprehension boundary). This
module adds only the production glue (the per-session body-state register + the level->expression map). `git diff
sim/` is empty.

HONEST RESIDUALS (named, not claimed closed).
  1. The message->valence APPRAISAL is host (a language-comprehension boundary, like the SVO parser). The felt READ
     (body-state -> graded valence x arousal off cp_firing_states) and its embodiment dependence ARE the #81 neural
     mechanism (lesion-proven). The body-state VARIABLES (h, a) are the standard body boundary.
  2. The level->EXPRESSION-MARKER SELECTION is now available as a SPIKING lateral-inhibition WTA circuit (board
     #86, 2026-08-28; `research/runners/_affect_marker_wta_derisk.py`), additive and DEFAULT-OFF pending owner
     review of the affect-path default (`BRAIN_AFFECT_MARKER_SPIKING=1`). Intact: the felt mood/arousal projects,
     as a topographic population code, onto 6 (resp. 2) small excitatory marker/arousal assemblies with their own
     cross-inhibiting FSI sub-pools (mutual lateral inhibition -- the SAME motif already 6-seed flip-soak GO'd at
     N=2 by the BG action selector); the assembly whose spiking rate clears the others by a dead margin, after the
     network settles, NAMES the marker -- the host renders the winner's fixed TOKEN, but the SELECTION is
     neurons/synapses, not `_LEAD_WORD[level]`. Verified byte-identical-OFF, load-bearing (mood sweep -> the
     marker changes, matching the register the host table would have picked, 6/6 seeds), lesioned (cutting the
     felt-state -> assembly projection collapses every pool to no clean winner -> the marker VANISHES, i.e. an
     honest no-lead turn, NOT a silent revert to the host template), and shuffle-anti-cheated (mis-routing which
     physical assembly receives which register's tuning drive changes the REPORTED marker, proving the identity is
     read off which assembly won, not re-derived from the raw mood float). See
     `research/findings/2026-08-28-affect-marker-spiking-wta-derisk.md`. `_LEAD_WORD` and the pre-existing
     host-dict `expression_lead` path remain the exact behavior when the flag is off (the default).
  3. This module reads its OWN co-resident #81 ladder bridge, run ALONGSIDE the recall composer, not merged onto the
     single recall bridge (the one-brain consolidation step, shared with the Gate-B affect burn-down).
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

# reuse-by-import the board-#81 6/6-seed-GO graded-affect ladder (build + the neural body->felt read) -- NO sim/ edit.
from research.runners._graded_affect_attractor_derisk import (
    GradedAffectBrain as _GradedAffectBrain,
    read_body as _read_body,
    I_BODY_PA as _I_BODY_PA,
)

_DEFAULT_SEED = int(os.environ.get("BRAIN_CHAT_SEED", "42"))  # research/seed-threading-lbf, 2026-09-20: reads the
# same env var as webapp/server.py._brain_chat_seed so this organ reseeds coherently with the rest of the tiny-
# demo brain. Unset -> 42, BYTE-IDENTICAL to the pre-existing hardcoded value.

# ── read windows (calibrated on the #81 ladder: (40,140,80) preserves the graded monotone mood(h) staircase at
#    ~0.13s/read vs the runner's (60,250,120)@0.16s -- the NMDA latches settle fast, so the shorter window is faithful).
_SETTLE_MS = 40
_ESTABLISH_MS = 140
_READ_MS = 80

# ── EMA body-state persistence: a strong induction turn dominates; a neutral turn (0 affective hits) HOLDS the prior
#    body-state (cross-turn affect persistence). Matches the Gate-B `_MOOD_EMA_DECAY` so the two affect paths agree.
_EMA_DECAY = 0.4

# ── mood -> graded Koulakov staircase LEVEL (-3..+3). Thresholds on the #81 mood scale (mood ~[-0.08,+0.08], the
#    set-point h=0.5 reads ~0). The neutral band keeps the answer surface byte-identical at a neutral mood.
_MOOD_NEUTRAL_TOL = 0.010    # |mood| below this -> level 0 (neutral: NO lead, surface unchanged)
_MOOD_L1 = 0.010
_MOOD_L2 = 0.045
_MOOD_L3 = 0.070
# ── bodily-arousal gain: the #81 arousal ladder only latches above body a~0.5 (felt ~0.04@a=0.5, ~0.065@a=0.7),
#    and appraised arousal (~0.6 for an affective word) is EMA-diluted, so map body a = clip(gain * ema_arousal) to
#    reach the felt-responsive band as affective arousal ACCUMULATES across turns (a single mild turn stays low).
_AROUSAL_GAIN = 1.5
# ── felt-arousal -> emphasis. felt_arousal ranges ~[0,0.075]; above this the affective marker is emphatic ("! ").
_AROUSAL_HIGH = 0.050

# ── the level -> affective EXPRESSION marker (the host conditioned-articulation scaffold; DRIVEN by the neural read).
_LEAD_WORD = {3: "Wonderful", 2: "Gladly", 1: "Sure", -1: "Hm", -2: "Honestly", -3: "Frankly"}


def affect_drives_enabled() -> bool:
    """The master flag. `BRAIN_AFFECT_DRIVES` truthy (1/true/on/yes) enables; 0/false/off/no disables. The default
    when the env var is UNSET follows the production-integration anchor `_AFFECT_DRIVES_DEFAULT_ON` in server.py --
    this reads only the explicit env override (server.py combines it with the anchor, mirroring the Gate-B flags)."""
    return os.environ.get("BRAIN_AFFECT_DRIVES", "0").strip().lower() in ("1", "true", "on", "yes")


def affect_drives_off() -> bool:
    """Explicit OFF (for a default-ON anchor): `BRAIN_AFFECT_DRIVES` in {0,false,no,off}."""
    v = os.environ.get("BRAIN_AFFECT_DRIVES")
    return v is not None and v.strip().lower() in ("0", "false", "no", "off", "")


def affect_drives_lesioned() -> bool:
    """`BRAIN_AFFECT_DRIVES_LESION` truthy -> cut the interoceptive->ladder synapses on every read (the #81
    embodiment lesion): the neural mood collapses to ~0, so the affective lead VANISHES. The load-bearing proof."""
    return os.environ.get("BRAIN_AFFECT_DRIVES_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


# Board #86 AUTO-FLIP (2026-09-01, GAP_CLOSURE_MISSION.md auto-flip policy: validated-GO + load-bearing +
# moat-safe + byte-identical-off + no-regression -> default-ON, owner-gating removed). 6-seed re-verify
# (research/runners/_affect_marker_wta_verify.py, seeds 42/43/44/100/101/102) reproduced GO fresh this cycle:
# (A) byte-identical-OFF (now the explicit `BRAIN_AFFECT_MARKER_SPIKING=0` escape, see `marker_selection_
# spiking_off()`), (B) load-bearing (mood sweep selects the matching register, 36/36 rows), (C) lesion collapses
# to the honest no-marker fallback (36/36 rows), (D) shuffle anti-cheat (30/36 differ from intact, ~1/6 expected
# fixed-point rate), (E) attribution (100% of the winner-vs-runner-up separation rides the felt-state->assembly
# drive, 0% in the lesioned control). research/findings/2026-08-28-affect-marker-spiking-wta-derisk.md.
_AFFECT_MARKER_SPIKING_DEFAULT_ON = True


def marker_selection_spiking_off() -> bool:
    """Explicit OFF (for the default-ON anchor above): `BRAIN_AFFECT_MARKER_SPIKING` in {0,false,no,off,''}."""
    v = os.environ.get("BRAIN_AFFECT_MARKER_SPIKING")
    return v is not None and v.strip().lower() in ("0", "false", "no", "off", "")


def marker_selection_spiking_enabled() -> bool:
    """Board #86 (2026-08-28, DEFAULT-ON 2026-09-01 -- see `_AFFECT_MARKER_SPIKING_DEFAULT_ON` above).
    `BRAIN_AFFECT_MARKER_SPIKING` truthy -> the level/mood -> expression-MARKER SELECTION step routes through the
    spiking lateral-inhibition WTA circuit (`research.runners._affect_marker_wta_derisk`) instead of the host
    `_LEAD_WORD[level]` dict lookup. `BRAIN_AFFECT_MARKER_SPIKING=0` (or false/no/off/'') is the BYTE-IDENTICAL
    escape back to the exact pre-existing host-template behavior."""
    if _AFFECT_MARKER_SPIKING_DEFAULT_ON:
        return not marker_selection_spiking_off()
    return os.environ.get("BRAIN_AFFECT_MARKER_SPIKING", "0").strip().lower() in ("1", "true", "on", "yes")


def marker_selection_lesioned() -> bool:
    """`BRAIN_AFFECT_MARKER_SPIKING_LESION` truthy -> (only meaningful when the spiking selector is enabled) cut
    the felt-state -> marker-assembly topographic PROJECTION on every read: every assembly receives the SAME
    baseline current, so the lateral-inhibition competition has no differentiating signal to resolve. The
    documented fallback is an HONEST NO-LEAD turn ('' -- the same safe fallback the circuit uses whenever it
    cannot find a clean winner), NOT a silent revert to the host `_LEAD_WORD` template. The load-bearing proof for
    the SELECTION step itself (distinct from `BRAIN_AFFECT_DRIVES_LESION`, which collapses the FELT STATE the
    ladder reads, upstream of this circuit)."""
    return os.environ.get("BRAIN_AFFECT_MARKER_SPIKING_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


def marker_selection_shuffled() -> bool:
    """`BRAIN_AFFECT_MARKER_SPIKING_SHUFFLE` truthy -> the anti-cheat control: mis-route which physical marker
    assembly receives which register's topographic tuning drive (a fixed random permutation). Verification-only
    (never set in normal operation) -- proves the reported marker identity tracks WHICH ASSEMBLY actually won the
    spiking competition, not a fixed host formula re-derived from the raw mood float."""
    return os.environ.get("BRAIN_AFFECT_MARKER_SPIKING_SHUFFLE", "0").strip().lower() in ("1", "true", "on", "yes")


# ── AFFECT-MARKER SURFACE RETIREMENT (2026-09-25, owner decision on branch research/retire-affect-marker-word).
#    Owner, verbatim: "It would be weird for the brain's replies to just be adding 'wonderful!' randomly. Its
#    speech should be influenced by its feelings, not just have a feeling-related word thrown in randomly."
#    Approved option A: STOP prepending the affect-marker word to the answer surface, but KEEP computing +
#    recording it (`affect_drives.lead` -- the felt-state read, the #86 spiking WTA selection and the A2
#    congruence gate are ALL unchanged; only whether the host string is glued onto `resp['answer']` is gated
#    here). Default OFF: the marker is an internal record only. `BRAIN_AFFECT_MARKER_SURFACE=1` restores the
#    pre-2026-09-25 production behavior byte-identically (the SAME recorded lead, now also prepended) -- see
#    `tests/test_webapp_server.py::test_brain_chat_affect_marker_surface_default_off_records_but_does_not_surface`
#    for the hash-style ON==OFF+lead check. See research/findings/2026-09-24-affect-marker-settle-flip-criteria-
#    AMENDMENT-PREREG.md (dated 2026-09-25 addendum: SETTLE's A3 timing run is CANCELLED by this decision) and
#    docs/PRODUCTION_INTEGRATION_LEDGER.yaml (affect-drives-response / affect-marker-spiking-wta rows).
def affect_marker_surface_enabled() -> bool:
    """`BRAIN_AFFECT_MARKER_SURFACE` truthy (1/true/on/yes) -> PREPEND the already-selected+congruence-gated
    marker to the answer surface (today's byte-identical pre-2026-09-25 behavior). Default OFF (unset, or any
    other value): the marker stays a computed+recorded internal field (`affect_drives.lead`) but is never glued
    onto `resp['answer']`. This flag touches ONLY the host string-concat step in webapp/server.py -- the neural
    read, the WTA selection and the congruence gate all run exactly as before regardless of this flag."""
    return os.environ.get("BRAIN_AFFECT_MARKER_SURFACE", "0").strip().lower() in ("1", "true", "on", "yes")


# ── A2 ABSTENTION-CONGRUENCE GATE (2026-09-25 amendment; research/findings/2026-09-24-affect-marker-settle-
#    flip-criteria-AMENDMENT-PREREG.md, "Amendment 2"; biology: research/biology/affective-marker-abstention-
#    congruence-gate.md). WHY: the prior mechanism (`research.runners._affect_marker_settle_congruence.apply_policy`)
#    was a host string-edit on an ALREADY-COMPOSED reply, unreachable from `/api/brain-chat` at all (named shortcut
#    S6 in the amendment) -- a prefix-only strip left the marker in the reply whenever a later stage prepended its
#    own text. This gate instead checks BEFORE the marker is ever prepended, using two reads the brain has ALREADY
#    computed this turn: the moat/BG speak-vs-abstain decision (`abstained`) and the Gate-B spiking affect organ's
#    OWN independent valence read (`gateb_valence_sign`, from `affect_production_organ.read_differential` -- a
#    DIFFERENT circuit from the #81 ladder that selected this marker). Both inputs are neural; only the register->
#    sign lookup and the withhold/surface branch are host control flow, the same pattern every other Gate-B-driven
#    surface coupling in webapp/server.py already uses (metacog hedge / curiosity follow-up / surprise prefix, each
#    gating a string operation on a spiking read's boolean) -- see the biology entry's "Declared host step".
CONGRUENCE_ENV = "BRAIN_AFFECT_MARKER_CONGRUENCE"
# the SAME fixed word->sign mapping `_LEAD_WORD` already inverts to select a word from a level -- re-read here,
# not re-derived (research.runners._affect_marker_settle_congruence carries the identical table for its own,
# unreachable, research-only scorer; duplicated here rather than imported so this production module has no
# import-time dependency on a `research/runners/*` research script -- reuse-by-VALUE of a fixed constant, not
# reuse-by-import of behavior).
_POS_REGISTERS = frozenset({"Wonderful", "Gladly", "Sure"})
_NEG_REGISTERS = frozenset({"Hm", "Honestly", "Frankly"})
_SIGN_MAP = {"+": 1, "-": -1, "0": 0}


def congruence_gate_enabled() -> bool:
    """`BRAIN_AFFECT_MARKER_CONGRUENCE` truthy -> the A2 gate runs. Default OFF -> `congruence_gate()` below is a
    no-op passthrough (byte-identical: same lead, no trace attached) -- verified in
    tests/test_affect_marker_congruence_gate.py."""
    return os.environ.get(CONGRUENCE_ENV, "0").strip().lower() in ("1", "true", "on", "yes")


def _register_word(lead: str) -> str:
    """The marker WORD a lead surfaces (strip the trailing '! '/' — ' emphasis punctuation), or '' for no lead."""
    w = (lead or "").strip()
    for suffix in ("! ", " — ", "!", "—"):
        if w.endswith(suffix):
            w = w[: -len(suffix)].strip()
            break
    return w


def _register_sign(word: str) -> int:
    if word in _POS_REGISTERS:
        return 1
    if word in _NEG_REGISTERS:
        return -1
    return 0


def congruence_gate(lead: str, *, abstained: bool, gateb_affect_info: Optional[dict] = None) -> tuple:
    """The A2 production gate. Returns (lead_out, trace_or_None). `lead` is this turn's ALREADY-SELECTED affective
    marker (from `expression_lead`, upstream and unchanged); `abstained` is the moat/BG decision already recorded
    on `resp["abstained"]`; `gateb_affect_info` is `resp["affect"]` (Gate-B's dict, carrying `valence_sign` in
    {"+","-","0"} or None when Gate-B is off).

    Default OFF, or no lead to check -> passthrough: (lead, None), byte-identical, no trace attached (mirrors
    every other additive coupling's "no key when disabled" contract).

    On, with a lead: an ABSTENTION CONFLICT (a marker on a turn the brain declined to answer) or a VALENCE
    CONFLICT (the marker's register disagrees with Gate-B's independently-computed sign, when both are non-zero)
    withholds the lead ('' -- an honest no-lead turn, the SAME documented fallback `expression_lead` itself uses
    on a reader exception or a lesioned WTA, never a silent revert to a different marker). Congruent -> the lead
    passes through unchanged. Never raises: any lookup failure degrades to passthrough (an honest no-op), mirroring
    `expression_lead`'s own never-crash contract."""
    if not lead or not congruence_gate_enabled():
        return lead, None
    try:
        word = _register_word(lead)
        rsign = _register_sign(word)
        vsign_raw = (gateb_affect_info or {}).get("valence_sign")
        vsign = _SIGN_MAP.get(vsign_raw)
        abstention_conflict = bool(abstained)
        valence_conflict = bool(vsign is not None and rsign != 0 and vsign != 0 and rsign != vsign)
        incongruent = abstention_conflict or valence_conflict
        trace = {"on": True, "checked_lead": lead, "register_word": word, "register_sign": rsign,
                 "abstained": bool(abstained), "gateb_valence_sign_raw": vsign_raw, "gateb_valence_sign": vsign,
                 "abstention_conflict": abstention_conflict, "valence_conflict": valence_conflict,
                 "incongruent": incongruent, "suppressed": incongruent}
        if incongruent:
            trace["reason"] = "abstention" if abstention_conflict else "valence_mismatch"
            return "", trace
        return lead, trace
    except Exception as e:            # never let the congruence check crash or silently mutate a turn
        return lead, {"on": True, "error": f"{type(e).__name__}: {e}"}


def _valence_to_body(valence: float, arousal: float) -> tuple:
    """Map the appraised message affect to the #81 body-state. valence in [-1,1] -> comfort/homeostasis h in [0,1]
    (h = 0.5 + 0.5*valence: valence 0 -> the neutral set-point h=0.5; +1 -> comfort; -1 -> discomfort). arousal in
    [0,1] -> bodily arousal a. This is the body boundary; the felt READ off the ladder is the neural part."""
    v = float(np.clip(valence, -1.0, 1.0))
    a = float(np.clip(_AROUSAL_GAIN * arousal, 0.0, 1.0))
    return 0.5 + 0.5 * v, a


def mood_to_level(mood: float) -> int:
    """The graded valence LEVEL (-3..+3) from the neural ladder mood differential (the Koulakov staircase)."""
    m = float(mood)
    s = 1 if m > 0 else -1
    am = abs(m)
    if am < _MOOD_NEUTRAL_TOL:
        return 0
    if am >= _MOOD_L3:
        return 3 * s
    if am >= _MOOD_L2:
        return 2 * s
    return 1 * s


def expression_lead(level: int, high_arousal: bool, *,
                    mood: Optional[float] = None, felt_arousal: Optional[float] = None,
                    seed: int = _DEFAULT_SEED) -> str:
    """The affective EXPRESSION marker for this turn's felt state. Level 0 (neutral) -> '' so the surface is
    byte-identical, REGARDLESS of which selection path is active (the neutral gate is checked first, before any
    spiking circuit would even be invoked, so a neutral turn never pays for or depends on it).

    Non-neutral: two selection paths.
      * DEFAULT (2026-09-01 auto-flip; `marker_selection_spiking_enabled()` is True unless `BRAIN_AFFECT_
        MARKER_SPIKING` is an explicit off) -- board #86: the level/word and the emphasis are each SELECTED by a
        spiking lateral-inhibition WTA circuit (`research.runners._affect_marker_wta_derisk.AffectMarkerWTA`)
        reading the CONTINUOUS felt mood/arousal as a topographic population code, instead of a host dict lookup
        on the pre-binned `level`/`high_arousal`. `BRAIN_AFFECT_MARKER_SPIKING_LESION=1` cuts the felt-state->
        assembly projection (documented fallback: '' -- an honest no-lead turn, not a silent revert to the host
        table). Any internal failure (import/build error) ALSO degrades to '' -- never raises, never silently
        falls back to the host template (mirrors the `mouth_tone_marker` fail-safe convention elsewhere in this
        repo) -- so this path can only ever REMOVE or CHANGE a lead, never crash a turn.
      * `BRAIN_AFFECT_MARKER_SPIKING=0` (or false/no/off/'') -- the BYTE-IDENTICAL escape to the ORIGINAL host
        conditioned-articulation scaffold: `_LEAD_WORD[level]` + '! '/' — ' by `high_arousal`."""
    if int(level) == 0:
        return ""
    if marker_selection_spiking_enabled() and mood is not None:
        try:
            from research.runners._affect_marker_wta_derisk import get_reader, marker_from_level
            reader = get_reader(seed=seed)
            lesion = marker_selection_lesioned()
            shuffle = marker_selection_shuffled()
            sel_level, _rates, _meta = reader.select_valence(float(mood), lesion=lesion, shuffle=shuffle)
            word = marker_from_level(sel_level)
            if not word:
                return ""
            fa = float(felt_arousal) if felt_arousal is not None else (0.075 if high_arousal else 0.0)
            high, _r2, _m2 = reader.select_arousal(fa, lesion=lesion, shuffle=shuffle)
            emphatic = bool(high) if high is not None else bool(high_arousal)
            return (word + "! ") if emphatic else (word + " — ")
        except Exception:
            return ""            # never raise, never silently revert to the host template -- an honest no-lead turn
    word = _LEAD_WORD.get(int(level))
    if not word:
        return ""
    return (word + "! ") if high_arousal else (word + " — ")


class AffectDrivesWorkspace:
    """A per-session graded-affect workspace: a persistent #81 ladder + an EMA body-state. `observe(valence, arousal,
    n_hits)` folds the appraisal into the body-state (a neutral turn HOLDS it), runs one neural ladder read, and
    returns the felt state + the graded level + the affective lead. The ladder build + read run on the workspace's
    PRIVATE RNG timeline (the host process-global RNG is restored around them -- the #77 footgun)."""

    def __init__(self, seed: int = _DEFAULT_SEED):
        self.seed = int(seed)
        self._brain = None
        self._lock = threading.Lock()
        self.h = 0.5           # persistent body-state: comfort/homeostasis (set-point 0.5)
        self.a = 0.0           # persistent body-state: bodily arousal
        self.ema_valence = 0.0
        self.ema_arousal = 0.0
        self.n_turns = 0
        self._rng_state = None  # the ladder's PRIVATE RNG timeline (the host process-global RNG is never advanced)

    def _isolated(self, fn):
        """Run `fn()` (the ladder build + spiking read) on the workspace's PRIVATE RNG timeline, leaving the host
        process-global RNG (numpy + the sim backend) BYTE-UNTOUCHED. The #81 build reseeds cfg.seed and its stepping
        draws OU noise off the SAME process-global RNG the rest of the pipeline shares -- without this, enabling this
        module would perturb the downstream RNG-dependent organs and break byte-identity. Snapshot the host RNG, swap
        in this workspace's own continuous timeline, run, capture the advanced private timeline, restore host. (Copied
        from gnw_thought_swap.ThoughtSwapWorkspace._isolated -- the same #77 fix.)"""
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

    def _ensure(self):
        if self._brain is None:
            self._brain = _GradedAffectBrain(self.seed)

    def observe(self, valence: float, arousal: float, n_hits: int, *,
                lesion: bool = False,
                valence_override: Optional[float] = None,
                arousal_override: Optional[float] = None) -> dict:
        """Fold the appraisal into the persistent body-state, run one neural ladder read, and return the felt state +
        graded level + the affective lead. n_hits==0 (a neutral turn) HOLDS the prior body-state (persistence). A
        `valence_override` / `arousal_override` sets the body-state directly (a mood INDUCTION, for the (B) proof:
        vary the affect state with the message fixed). `lesion` cuts the interoceptive->ladder synapses so the neural
        mood collapses (the load-bearing lesion). Never raises out (the caller degrades to no-lead)."""
        with self._lock:
            self.n_turns += 1
            if valence_override is not None or arousal_override is not None:
                if valence_override is not None:
                    self.ema_valence = float(valence_override)
                if arousal_override is not None:
                    self.ema_arousal = float(arousal_override)
            elif int(n_hits) > 0:
                d = _EMA_DECAY
                self.ema_valence = d * self.ema_valence + (1.0 - d) * float(valence)
                self.ema_arousal = d * self.ema_arousal + (1.0 - d) * float(arousal)
            # else: neutral turn -> HOLD the prior EMA (cross-turn persistence)
            self.h, self.a = _valence_to_body(self.ema_valence, self.ema_arousal)

            info = {"acted": False, "turn": self.n_turns, "lesioned": bool(lesion),
                    "ema_valence": float(self.ema_valence), "ema_arousal": float(self.ema_arousal),
                    "body_h": float(self.h), "body_a": float(self.a),
                    "mood": 0.0, "felt_arousal": 0.0, "level": 0, "high_arousal": False, "lead": "",
                    "reason": None, "seed": self.seed}
            try:
                self._isolated(self._ensure)
                r = self._isolated(lambda: _read_body(self._brain, self.h, self.a, _I_BODY_PA,
                                                      settle=_SETTLE_MS, establish=_ESTABLISH_MS, read=_READ_MS,
                                                      lesion_gate=bool(lesion)))
                mood = float(r["mood"])
                felt = float(r["felt_arousal"])
                level = mood_to_level(mood)
                high = bool(felt > _AROUSAL_HIGH)
                lead = expression_lead(level, high, mood=mood, felt_arousal=felt, seed=self.seed)
                info.update({"acted": True, "mood": mood, "felt_arousal": felt, "level": int(level),
                             "high_arousal": high, "lead": lead,
                             "vplus_rate": float(r.get("vplus_rate", 0.0)),
                             "vminus_rate": float(r.get("vminus_rate", 0.0)),
                             "reason": ("lesion_collapsed" if lesion else
                                        ("neutral_hold" if level == 0 else "graded_affect"))})
            except Exception as e:   # never let the affect read crash / change a turn
                info["reason"] = f"error:{type(e).__name__}: {e}"
            return info


    def relax_idle(self, relax: float, neutral: float = 0.0) -> dict:
        """ONE IDLE-TICK relaxation step (board #91, 2026-08-26) -- the continuous-engine companion to `observe()`:
        decay the persistent EMA body-state toward the neutral set-point (the SAME homeostatic relaxation formula
        `continuous_engine.tick_session` already applies to the legacy Gate-B mood, `v1 = NEUTRAL + (v0-NEUTRAL)*RELAX`
        -- reused verbatim, not reinvented), recompute the body-state (h, a), and re-run ONE neural #81 ladder READ at
        the DECAYED point. So the felt mood this coupling reports between turns is a genuine spiking read AT THE
        RELAXED body-state -- never a host time-based formula computing the level/lead directly. Does NOT increment
        `n_turns` (an idle tick is not a conversational turn) and does NOT touch the induction/hold branching
        `observe()` uses -- it is a distinct, idempotent decay step the idle loop calls once per tick, applied
        directly to the SAME `ema_valence`/`ema_arousal`/`h`/`a` state a live `observe()` turn reads and writes next.

        Returns the same-shaped record `observe()` returns (mood/felt_arousal/level/lead/...) plus `relaxed: True`,
        so a caller can log an inner-life note ("my felt warmth is fading toward neutral"). Never raises out (mirrors
        `observe()`'s never-crash contract) -- on any error the level/lead stay at the inert default (0/'')."""
        with self._lock:
            self.ema_valence = float(neutral) + (float(self.ema_valence) - float(neutral)) * float(relax)
            self.ema_arousal = float(self.ema_arousal) * float(relax)
            self.h, self.a = _valence_to_body(self.ema_valence, self.ema_arousal)

            info = {"acted": False, "relaxed": True, "turn": self.n_turns, "lesioned": False,
                    "ema_valence": float(self.ema_valence), "ema_arousal": float(self.ema_arousal),
                    "body_h": float(self.h), "body_a": float(self.a),
                    "mood": 0.0, "felt_arousal": 0.0, "level": 0, "high_arousal": False, "lead": "",
                    "reason": None, "seed": self.seed}
            try:
                self._isolated(self._ensure)
                r = self._isolated(lambda: _read_body(self._brain, self.h, self.a, _I_BODY_PA,
                                                      settle=_SETTLE_MS, establish=_ESTABLISH_MS, read=_READ_MS,
                                                      lesion_gate=False))
                mood = float(r["mood"])
                felt = float(r["felt_arousal"])
                level = mood_to_level(mood)
                high = bool(felt > _AROUSAL_HIGH)
                lead = expression_lead(level, high, mood=mood, felt_arousal=felt, seed=self.seed)
                info.update({"acted": True, "mood": mood, "felt_arousal": felt, "level": int(level),
                             "high_arousal": high, "lead": lead,
                             "vplus_rate": float(r.get("vplus_rate", 0.0)),
                             "vminus_rate": float(r.get("vminus_rate", 0.0)),
                             "reason": "idle_relax"})
            except Exception as e:   # never let the idle-relax read crash the tick loop
                info["reason"] = f"error:{type(e).__name__}: {e}"
            return info


def get_workspace(chat, *, seed: int = _DEFAULT_SEED) -> AffectDrivesWorkspace:
    """Idempotently attach a per-session `AffectDrivesWorkspace` to the cached ChatBrain (auto-cleared on session
    reset, which drops the ChatBrain). No `sim/` edit; the ChatBrain instance is a host scaffold."""
    ws = getattr(chat, "_affect_drives_workspace", None)
    if ws is None:
        ws = AffectDrivesWorkspace(seed=seed)
        chat._affect_drives_workspace = ws
    return ws


def relax_idle(chat, relax: float, neutral: float = 0.0) -> Optional[dict]:
    """The IDLE-TICK entry point (board #91): relax THIS session's #84 affect-drives EMA toward neutral and re-read
    the neural ladder at the decayed point. `continuous_engine`'s headline mechanism -- "the felt mood keeps
    evolving while idle" -- previously reached ONLY the legacy Gate-B affect path (`_SESSION_MOOD` +
    `_get_affect_organ().read_differential`); this extends the SAME idle-relax idea to the flagship, default-ON,
    most user-visible affect->tone coupling (the #84 lead marker this module drives), closing an observe-vs-drive
    gap: tell the brain something emotionally charged, wait idle, then send a neutral follow-up -- BEFORE this, the
    #84 lead on return was IDENTICAL to zero idle time, because the thing that decays (`_SESSION_MOOD`) never fed
    #84.

    Returns None (a clean no-op) when this session has no `_affect_drives_workspace` yet -- i.e. #84 was never
    triggered on a live turn -- so a session that never had an affect-drives turn is BYTE-IDENTICAL: idling it
    does nothing new, exactly like today. Never raises (delegates to `AffectDrivesWorkspace.relax_idle`, itself
    never-raising)."""
    ws = getattr(chat, "_affect_drives_workspace", None)
    if ws is None:
        return None
    try:
        return ws.relax_idle(relax, neutral=neutral)
    except Exception as e:
        return {"acted": False, "relaxed": True, "reason": f"error:{type(e).__name__}: {e}", "lead": "", "level": 0}


def observe_turn(chat, message: str, appraisal: Optional[dict] = None, *,
                 seed: int = _DEFAULT_SEED,
                 valence_override: Optional[float] = None,
                 arousal_override: Optional[float] = None) -> dict:
    """The production entry point: appraise this turn's message (reuse the Gate-B `affect_production_organ` DR-2
    appraisal unless one is passed in), fold it into the per-session body-state, run one neural #81 ladder read, and
    return the per-turn `affect_drives` info (also stashed on `chat._last_affect_drives`). Never raises out (on any
    error it returns an inert no-lead info dict so a turn can never crash).

    MOOD-INDUCTION affordance (for the (B) load-bearing proof + a live mood-set): `BRAIN_AFFECT_DRIVES_INDUCE="v"`
    or `"v,a"` sets the body-state directly (valence v in [-1,1], arousal a in [0,1]) so the affect state can be
    varied with the MESSAGE HELD FIXED (a mood induction, exactly the (B) design) -- the neural ladder read still
    runs on that induced body-state and the lesion still collapses it. An explicit override arg takes precedence."""
    try:
        if valence_override is None and arousal_override is None:
            _ind = os.environ.get("BRAIN_AFFECT_DRIVES_INDUCE")
            if _ind:
                try:
                    parts = [float(x) for x in str(_ind).split(",")]
                    valence_override = parts[0]
                    if len(parts) > 1:
                        arousal_override = parts[1]
                except Exception:
                    pass
        if appraisal is None:
            from research.runners import affect_production_organ as _AO
            appraisal = _AO.appraise_text(message)
        ws = get_workspace(chat, seed=seed)
        info = ws.observe(float(appraisal.get("valence", 0.0)), float(appraisal.get("arousal", 0.0)),
                          int(appraisal.get("n_hits", 0)), lesion=affect_drives_lesioned(),
                          valence_override=valence_override, arousal_override=arousal_override)
    except Exception as e:
        info = {"acted": False, "reason": f"error:{type(e).__name__}: {e}", "lead": "", "level": 0}
    chat._last_affect_drives = info
    return info
