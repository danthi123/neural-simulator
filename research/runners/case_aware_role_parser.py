"""CaseAwareRoleParser -- production drop-in role-assigner that EXTENDS the multi-cue role-COMPETITION with a
CASE cue, so a FREE-word-order CASE-MARKED sentence (Japanese-style ga/wo, Korean i/ga.eul/leul) reads thematic
roles by the case PARTICLE where word-position cannot.

This is the production wire-in of the Phase-2 spiking de-risk
(`research/runners/_phaseB_case_cue_crosslanguage_derisk.py`, GO -- finding `2026-06-19-case-cue-crosslanguage-derisk.md`):
the role-COMPETITION + the reliability-weighted ACCUMULATION + the WINNER are real spiking neurons on a
`SimulationBridge` (the same `SpikingRoleCompetition` Wong-Wang/Rutishauser WTA over thematic ROLES + plastic
cue->role projections the Phase-1 `MultiCueRoleParser` uses), now with a fifth CUE -- `case` -- whose signed
vote (nominative-particle -> +1 agent, accusative-particle -> -1 patient) joins the position+animacy+verb-fit
competition. The cue VALIDITIES are the validated INSTALL-path weights (case is the dominant cue in a case
language; position is low). The drop-in exposes the SAME `parse(words, voice) -> {agent, action, patient}` shape
`BridgeParser.parse` / `MultiCueRoleParser.parse` return, so it slots straight into
`BrainConversationalAgent.hear()` behind a default-OFF flag (the same wire-in pattern as `enable_multicue_competition`).

WHY A SEPARATE CLASS (not a flag on `MultiCueRoleParser`): the case cue is a FIFTH cue, so the spiking competition
bridge it builds has an extra cue population + plastic projection. The Phase-1 `SpikingRoleCompetition` reads the
module-level `CUES`/`SEMANTIC_CUES` of `_phaseB_multicue_competition_spiking_derisk` at construction AND at
inference (e.g. `cue_weights`, `set_cue_weight`, `_semantic_contrast` all iterate them). To build + run the
case-aware competition WITHOUT permanently mutating those globals (which would break a co-resident plain
`MultiCueRoleParser` -- its moat `_semantic_contrast` would KeyError on a missing `ev["case"]`), every public
method here TRANSIENTLY swaps the case-extended `CUES`/`_CUE_ID`/`SEMANTIC_CUES` in for the duration of the call
and restores the originals afterward (`_case_cue_context`). The plain Phase-1 path is left byte-identical when not
inside a case-parser call.

WHAT IS WIRED vs DEFERRED (honest, per the BRAIN-BASED-ONLY directive + the de-risk):
  * WIRED (this drop-in): the validated SPIKING role-competition INFERENCE WITH THE CASE CUE -- the cue
    populations (now incl. case) -> plastic cue->role projections -> Wong-Wang role accumulators in mutual
    inhibition; the WINNER is the spiking WTA settle. The cue VALIDITIES are the validated INSTALL-path
    case-language weights (`INSTALLED_CASE_WEIGHTS`: case dominant, position low).
  * DEFERRED (documented follow-ons, NOT blockers for the wire-in): (a) continual ON-SUBSTRATE cue-validity
    LEARNING -- the three-factor rule that learns the validity spread (and produces the cross-linguistic
    dissociation, English w_case->floor / Japanese w_case->top) is seed-variable in robustness at the spiking
    scale (Tier 1 item 2, the firm-the-learning follow-on); (b) neuralizing that learner's reward.
  * HOST front-end (the legitimate token-level lexical boundary, identical to the verb/animacy/verb-fit lexicons
    of `MultiCueRoleParser`): the CASE MARKERS are a host-supplied lexicon -- which particle TOKENS are
    nominative vs accusative (e.g. {"ga": "nom", "wo": "acc"}). This is the ISOLATING-particle case (a
    set-membership check on the particle token, NO morphological segmentation). FUSED/portmanteau case
    (Russian -a/-u, Latin -us/-um, which need sub-word morphology) is Phase 3 -- DEFERRED, NOT built here. The
    case lexicon is the lexical front-end, flagged for the eventual learned/neural front-end; the role
    COMPETITION + the install-path validities are the brain-based win.

The no-confab MOAT is preserved: an UNMARKED ambiguous transitive (two animate nouns + a symmetric verb, no
case particles) leaves the agent/patient decision undecided (case silent + animacy ties + verb symmetric -> no
decisive content cue); `parse_decisive` reports that so the caller ABSTAINS rather than confabulate.

Reuse-by-import; NO `sim/` edit.
"""
from __future__ import annotations

import contextlib

import research.runners._phaseB_multicue_competition_spiking_derisk as P1
from research.runners._phaseB_multicue_competition_spiking_derisk import (
    SpikingRoleCompetition,
)

# ---------------------------------------------------------------------------
# The case-extended cue set + the case-language INSTALL validities (the validated Phase-2 magnitudes).
# `case` sits adjacent to the semantic cues; the chance distractor `lexbias` stays last (matches the de-risk's
# ordering so `_CUE_ID` indices are identical, keeping any learned-path noise keys consistent).
# ---------------------------------------------------------------------------
CASE_CUES = ("position", "animacy", "verbfit", "case", "lexbias")
CASE_CUE_ID = {c: i for i, c in enumerate(CASE_CUES)}
CASE_SEMANTIC_CUES = ("animacy", "verbfit", "case")  # case is the dominant CONTENT cue in a case language

# validated case-language cue->role validities at the spiking operating scale (case dominant, position low,
# distractor low) -- mirrors `_phaseB_case_cue_crosslanguage_derisk.INSTALLED_CASE_WEIGHTS`.
INSTALLED_CASE_WEIGHTS = {"position": 6.0, "animacy": 14.0, "verbfit": 14.0, "case": 22.0, "lexbias": 2.0}

# default isolating-particle case lexicon (Japanese-style). The caller may override with any token->{nom,acc} map
# (e.g. Korean {"i": "nom", "ga": "nom", "eul": "acc", "leul": "acc"}). A particle token NOT in the map is treated
# as a plain word (no case vote) -- the same as an unmarked noun.
DEFAULT_CASE_LEXICON = {"ga": "nom", "wo": "acc", "o": "acc"}  # ga = nominative (agent), wo/o = accusative (patient)


@contextlib.contextmanager
def _case_cue_context():
    """Transiently install the case-extended cue set onto the Phase-1 module + class so the reused
    `SpikingRoleCompetition` builds/reads the CASE cue, then restore the originals. Keeps a co-resident plain
    `MultiCueRoleParser` byte-identical outside any case-parser call (its moat would otherwise KeyError on a
    missing `ev["case"]`)."""
    saved_cues = P1.CUES
    saved_cue_id = P1._CUE_ID
    saved_sem = SpikingRoleCompetition.SEMANTIC_CUES
    P1.CUES = CASE_CUES
    P1._CUE_ID = CASE_CUE_ID
    SpikingRoleCompetition.SEMANTIC_CUES = CASE_SEMANTIC_CUES
    try:
        yield
    finally:
        P1.CUES = saved_cues
        P1._CUE_ID = saved_cue_id
        SpikingRoleCompetition.SEMANTIC_CUES = saved_sem


class CaseAwareRoleParser:
    """Spiking multi-cue role-competition WITH A CASE CUE as a production role-assigner. `parse(words, voice)`
    returns `{agent, action, patient}` for a transitive sentence in ANY word order, assigning the AGENT/PATIENT
    nouns by the validated spiking competition (position + animacy + verb-fit + CASE cues, case-language install
    validities) and identifying the verb lexically from `known_verbs`. The case marker of each noun is read from
    the surface tokens via `case_lexicon` (an isolating-particle that immediately FOLLOWS its noun) OR supplied
    explicitly per-noun via `markers=`.

    The competition's installed cue validities are placed once at construction (the validated case-language arm);
    the plasticity gates are frozen so inference does not drift them. `parse_decisive` exposes the moat content
    gate (case silent + animacy ties + verb symmetric -> abstain)."""

    def __init__(self, known_verbs, case_lexicon=None, seed=42, abstain_margin=None, read_steps=60, comp_kw=None):
        """`known_verbs` = the caller's verb set (the lexical front-end identifies the sentence's verb from it).
        `case_lexicon` = a {particle_token: 'nom'|'acc'} map (default the Japanese-style ga/wo). A particle token
        immediately following a noun in the surface string marks that noun's case. `abstain_margin` (optional)
        overrides the moat content-gate margin. `comp_kw` = extra kwargs forwarded to `SpikingRoleCompetition`."""
        self.known_verbs = set(known_verbs)
        self.case_lexicon = dict(case_lexicon) if case_lexicon is not None else dict(DEFAULT_CASE_LEXICON)
        # normalize lexicon values to {'nom','acc'}
        for tok, role in list(self.case_lexicon.items()):
            assert role in ("nom", "acc"), f"case_lexicon[{tok!r}] must be 'nom' or 'acc', got {role!r}"
        self.seed = int(seed)
        self.read_steps = int(read_steps)
        self._abstain_margin = abstain_margin
        comp_kw = dict(comp_kw or {})
        # build the case-aware spiking role-competition bridge ONCE (case cue present) + install the validated
        # case-language cue validities + freeze plasticity. The context manager keeps the Phase-1 globals pristine.
        with _case_cue_context():
            self.comp = SpikingRoleCompetition(seed=seed, **comp_kw)
            for c, w in INSTALLED_CASE_WEIGHTS.items():
                self.comp.set_cue_weight(c, w)
            self.comp.freeze_all_cue_plasticity()

    # --- the lexical front-end (HOST boundary): find the verb; pull the case PARTICLES out of the surface tokens ---
    def _split_verb_nouns_markers(self, words, markers=None):
        """Return (verb, [noun words], [per-noun case marker in {'nom','acc',None}]). The verb is the (first) word
        in `known_verbs`. A surface token that is a case particle (in `case_lexicon`) marks the IMMEDIATELY
        PRECEDING noun and is itself consumed (it is a clitic, not a noun). If `markers` is given (a per-noun list
        aligned to the nouns AFTER particle removal), it overrides the surface-extracted markers."""
        verb = None
        noun_words = []
        noun_markers = []
        for w in words:
            if w in self.case_lexicon:
                # a case particle: mark the preceding noun (if any); the particle is consumed, not a noun
                if noun_markers:
                    noun_markers[-1] = self.case_lexicon[w]
                continue
            if verb is None and w in self.known_verbs:
                verb = w
            else:
                noun_words.append(w)
                noun_markers.append(None)
        if markers is not None:
            # explicit per-noun markers override (aligned to the extracted nouns); normalize None/strings
            for i in range(min(len(noun_markers), len(markers))):
                m = markers[i]
                noun_markers[i] = m if m in ("nom", "acc") else None
        return verb, noun_words, noun_markers

    def _case_vote(self, marker):
        """Isolating-particle case vote: nominative -> +1 (agent), accusative -> -1 (patient), else 0."""
        if marker == "nom":
            return +1.0
        if marker == "acc":
            return -1.0
        return 0.0

    def _evidence_for_nouns(self, noun_words, noun_markers, verb):
        """Cue evidence for each noun = the Phase-1 base cues (position/animacy/verbfit/lexbias) PLUS the CASE cue
        (this noun's case-particle vote). `clean_cues=True` reads noise-free cue values (deterministic production
        parse; the per-cue label-noise is a TRAINING construct, not an inference signal). The case cue's
        reliability is 1 iff the noun is marked."""
        n = len(noun_words)
        evs = []
        for ni, noun in enumerate(noun_words):
            ev = P1.cue_evidence(noun, ni, n, verb, sent_id=0, clean_cues=True)  # base cues (Phase-1, unchanged)
            raw = self._case_vote(noun_markers[ni] if ni < len(noun_markers) else None)
            ev["case"] = (float(raw), 1.0 if raw != 0.0 else 0.0)               # the additive CASE cue
            evs.append(ev)
        return evs

    def parse(self, words, voice="active", markers=None):
        """Comprehend a (possibly FREE-word-order, case-marked) transitive sentence -> {agent, action, patient}.
        The verb is identified lexically; each noun's case particle is pulled from the surface tokens (or supplied
        via `markers`); the two nouns' AGENT/PATIENT roles are assigned by the spiking multi-cue competition (with
        the case cue). Returns the SAME {role: word} dict shape `BridgeParser.parse` returns. `voice` is accepted
        for signature-compatibility; the competition reads roles from content/case, not a declared voice frame."""
        words = list(words) if not isinstance(words, str) else words.split()
        with _case_cue_context():
            verb, noun_words, noun_markers = self._split_verb_nouns_markers(words, markers=markers)
            if len(noun_words) != 2:
                # not a 2-noun transitive -> surface-order fallback (the de-risk scope is the 2-noun transitive).
                roles = {}
                order = ["agent", "action", "patient"]
                for i, w in enumerate(noun_words):
                    roles[order[i] if i < 3 else "patient"] = w
                if verb is not None:
                    roles["action"] = verb
                return roles
            evs = self._evidence_for_nouns(noun_words, noun_markers, verb)
            assignment, _decisive, _dbg = self.comp.assign_roles(noun_words, evs, read_steps=self.read_steps)
            roles = {"action": verb}
            for ni, w in enumerate(noun_words):
                roles[assignment.get(ni, "agent" if ni == 0 else "patient")] = w
            return roles

    def parse_decisive(self, words, voice="active", markers=None):
        """Like `parse`, but also returns whether the content (case + semantics) decisively determined the roles
        (the no-confab moat content gate). Returns (roles, decisive). `decisive=False` => the nouns are content-
        ambiguous (e.g. two animate nouns + a symmetric verb + NO case particles) so a role assignment would be a
        guess; the caller should ABSTAIN rather than store a confabulated fact."""
        words = list(words) if not isinstance(words, str) else words.split()
        with _case_cue_context():
            verb, noun_words, noun_markers = self._split_verb_nouns_markers(words, markers=markers)
            if len(noun_words) != 2:
                return self.parse(words, voice, markers=markers), True
            evs = self._evidence_for_nouns(noun_words, noun_markers, verb)
            margin = self._abstain_margin if self._abstain_margin is not None else self._default_margin()
            assignment, decisive, _dbg = self.comp.assign_roles(noun_words, evs, abstain_margin=margin,
                                                                read_steps=self.read_steps)
            roles = {"action": verb}
            for ni, w in enumerate(noun_words):
                roles[assignment.get(ni, "agent" if ni == 0 else "patient")] = w
            return roles, bool(decisive)

    def cue_weights(self):
        """The installed cue->role validities (case dominant, position low) -- read inside the case context so the
        case cue is included."""
        with _case_cue_context():
            return self.comp.cue_weights()

    def _default_margin(self):
        """A conservative moat content-gate margin from the installed case-language cue weights: half the magnitude
        a single decisive CONTENT cue (case, the dominant one) would contribute. An ambiguous UNMARKED sentence
        produces a near-zero content contrast (case silent + animacy ties + symmetric verb), well below this.
        (Called inside `_case_cue_context`; reads the case-aware SEMANTIC_CUES.)"""
        w = self.comp.cue_weights()
        # the dominant content cue in a case language is `case`; gate at half its installed weight.
        w_content = max(w.get("case", 0.0), 0.5 * (w.get("animacy", 0.0) + w.get("verbfit", 0.0)))
        return 0.5 * w_content
