"""MultiCueRoleParser -- production drop-in role-assigner that routes a transitive sentence's AGENT/PATIENT
decision through the validated SPIKING multi-cue role-competition, robust to DEGRADED English (object-fronted /
scrambled word order) where a position-only parser collapses.

This is the production wire-in of the spiking de-risk
(`research/runners/_phaseB_multicue_competition_spiking_derisk.py`, GO -- finding
`2026-06-19-multicue-competition-spiking-derisk.md`): the role-COMPETITION + the reliability-weighted
ACCUMULATION + the WINNER are real spiking neurons on a `SimulationBridge` (the re-pointed
`biased_competition_buffer.py` Wong-Wang/Rutishauser WTA over thematic ROLES + plastic cue->role projections), and
the cue VALIDITIES are the validated INSTALL-path weights (position < semantic, distractor low). The drop-in
exposes the SAME `parse(words, voice) -> {agent, action, patient}` shape `BridgeParser.parse` returns, so it slots
straight into `BrainConversationalAgent.hear()` / the composer `store()` path behind a default-OFF flag.

WHAT IS WIRED vs DEFERRED (honest, per the BRAIN-BASED-ONLY directive + the finding):
  * WIRED (this drop-in): the validated SPIKING role-competition INFERENCE -- the cue populations -> plastic
    cue->role projections -> Wong-Wang role accumulators (sel_agent/sel_patient) in mutual inhibition; the WINNER
    is the spiking WTA settle. The cue VALIDITIES are the validated INSTALL-path weights (the robust 5/6-seed GO
    arm; an installed learned parameter, like a pre-trained weight).
  * DEFERRED (documented follow-ons, NOT blockers for the wire-in): (a) continual ON-SUBSTRATE cue-validity
    LEARNING -- the three-factor rule that learns the validity spread on the substrate is seed-variable in
    robustness at the spiking scale (an honest boundary on the *learning*, not the mechanism); (b) neuralizing the
    reward signal in that learner.
  * HOST front-end (the legitimate lexical boundary, identical to `FrameParser` + the buffer's
    `content_bias_target`): the verb (action) is identified from the caller's known-verb set, and the feature
    LEXICONS (animacy, verb-selectional-fit) supply each cue's VALUE for a word. They do NOT supply the role
    decision -- that is the learned-weight spiking competition. The conversion target is a learned lexical-feature
    map (the de-risk's documented boundary).

The no-confab MOAT is preserved: an all-ambiguous transitive (two animate nouns + a symmetric verb, scrambled ->
no decisive content cue) leaves the agent/patient decision undecided; `parse_decisive` reports that so the caller
can ABSTAIN rather than confabulate a role assignment.

Reuse-by-import; NO `sim/` edit. Build is lazy + cached on the bridge cost.
"""
from __future__ import annotations

from research.runners._phaseB_multicue_competition_spiking_derisk import (
    ANIMACY,
    VERB_SELECTS,
    INSTALLED_CUE_WEIGHTS,
    CUES,
    SpikingRoleCompetition,
    cue_evidence,
)


class MultiCueRoleParser:
    """Spiking multi-cue role-competition as a production role-assigner. `parse(words, voice)` returns
    `{agent, action, patient}` for a transitive sentence in ANY word order (SVO / OSV / scrambled), assigning the
    AGENT/PATIENT nouns by the validated spiking competition (position + animacy + verb-fit cues, install-path
    validities) and identifying the verb lexically from `known_verbs`.

    The competition's installed cue validities are placed once at construction (the validated 5/6-seed GO arm); the
    plasticity gates are frozen so inference does not drift them. `parse_decisive` exposes the moat content gate."""

    def __init__(self, known_verbs, seed=42, abstain_margin=None, read_steps=60, comp_kw=None):
        """`known_verbs` = the caller's verb set (the lexical front-end identifies the sentence's verb from it).
        `abstain_margin` (optional) overrides the moat content-gate margin; if None, a small default is used (the
        learned-weight semantic-contrast scale). `comp_kw` = extra kwargs forwarded to `SpikingRoleCompetition`."""
        self.known_verbs = set(known_verbs)
        self.seed = int(seed)
        self.read_steps = int(read_steps)
        self._abstain_margin = abstain_margin
        comp_kw = dict(comp_kw or {})
        # build the spiking role-competition bridge ONCE; install the validated cue validities + freeze plasticity
        self.comp = SpikingRoleCompetition(seed=seed, **comp_kw)
        for c, w in INSTALLED_CUE_WEIGHTS.items():
            self.comp.set_cue_weight(c, w)
        self.comp.freeze_all_cue_plasticity()

    # --- the lexical front-end (HOST boundary, identical to FrameParser): find the verb, leave the nouns ---
    def _split_verb_nouns(self, words):
        """Return (verb, [noun surface positions], [noun words]). The verb is the (first) word in `known_verbs`;
        every other word is a noun in its surface order. The nouns' SURFACE indices feed the position cue."""
        verb = None
        noun_pos, noun_words = [], []
        for i, w in enumerate(words):
            if verb is None and w in self.known_verbs:
                verb = w
            else:
                noun_pos.append(i)
                noun_words.append(w)
        return verb, noun_pos, noun_words

    def _evidence_for_nouns(self, noun_words, verb):
        """Cue evidence for each noun. The POSITION cue uses the noun's index AMONG THE NOUNS (0=first noun ->
        agent-vote, 1=second noun -> patient-vote) -- so a canonical 'dog eat apple' gives dog the agent-position
        vote, and an object-fronted 'apple eat dog' gives apple the agent-position vote (which the semantic cues
        then OVERRIDE: apple is inanimate + the patient of 'eat' -> patient). `sent_id=0`+`clean_cues=True` reads
        the noise-free cue values (the production parse is deterministic; the per-cue label-noise is a TRAINING
        construct in the de-risk, not an inference-time signal)."""
        n = len(noun_words)
        return [cue_evidence(noun, ni, n, verb, sent_id=0, clean_cues=True) for ni, noun in enumerate(noun_words)]

    def parse(self, words, voice="active"):
        """Comprehend a transitive sentence -> {agent, action, patient}. The verb is identified lexically; the two
        nouns' AGENT/PATIENT roles are assigned by the spiking multi-cue competition (robust to degraded order).

        Returns the SAME {role: word} dict shape `BridgeParser.parse` returns (so it drops straight into the agent
        `hear()` / composer `store()` path). `voice` is accepted for signature-compatibility with `BridgeParser`;
        the multi-cue competition does not need it (it reads roles from content, not a declared voice frame)."""
        words = list(words) if not isinstance(words, str) else words.split()
        verb, noun_pos, noun_words = self._split_verb_nouns(words)
        if len(noun_words) != 2:
            # not a 2-noun transitive -> fall back to a position read (the de-risk scope is 2-noun transitive). A
            # 1- or 3+-noun input is out of this drop-in's validated scope; assign by surface order so the caller
            # still gets a dict (the no-confab moat in the agent's Q&A still abstains on any unstored fact).
            roles = {}
            order = ["agent", "action", "patient"]
            for i, w in enumerate(words):
                roles[order[i] if i < 3 else "patient"] = w
            if verb is not None:
                roles["action"] = verb
            return roles
        evs = self._evidence_for_nouns(noun_words, verb)
        assignment, _decisive, _dbg = self.comp.assign_roles(noun_words, evs, read_steps=self.read_steps)
        roles = {"action": verb}
        for ni, w in enumerate(noun_words):
            roles[assignment.get(ni, "agent" if ni == 0 else "patient")] = w
        return roles

    def parse_decisive(self, words, voice="active"):
        """Like `parse`, but also returns whether the SEMANTIC content decisively determined the roles (the
        no-confab moat content gate). Returns (roles, decisive). `decisive=False` => the two nouns are content-
        ambiguous (e.g. two animate nouns + a symmetric verb) so a role assignment would be a guess; the caller
        should ABSTAIN rather than store a confabulated fact."""
        words = list(words) if not isinstance(words, str) else words.split()
        verb, noun_pos, noun_words = self._split_verb_nouns(words)
        if len(noun_words) != 2:
            return self.parse(words, voice), True
        evs = self._evidence_for_nouns(noun_words, verb)
        margin = self._abstain_margin if self._abstain_margin is not None else self._default_margin(evs)
        assignment, decisive, _dbg = self.comp.assign_roles(noun_words, evs, abstain_margin=margin,
                                                            read_steps=self.read_steps)
        roles = {"action": verb}
        for ni, w in enumerate(noun_words):
            roles[assignment.get(ni, "agent" if ni == 0 else "patient")] = w
        return roles, bool(decisive)

    def _default_margin(self, evs):
        """A conservative moat content-gate margin from the installed semantic-cue weights: half the magnitude a
        single decisive semantic cue would contribute. Two content-ambiguous nouns produce a near-zero semantic
        contrast (both animate -> animacy votes cancel; symmetric verb -> verb-fit silent), well below this."""
        w = self.comp.cue_weights()
        w_sem = 0.5 * (w.get("animacy", 0.0) + w.get("verbfit", 0.0))
        return 0.5 * w_sem
