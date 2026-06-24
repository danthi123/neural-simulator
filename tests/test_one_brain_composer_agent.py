"""CI GUARD (roadmap phase 2, the real "one brain"): BrainConversationalAgent(composer_kind="onebrain") must keep
answering the core who/what/yes-no/moat matrix on the production OneBrainComposer -- the WHOLE pipeline (comprehend ->
store -> query -> abstain) on ONE persistent co-resident bridge, the agent delegating comprehension to the composer's
on-bridge parser (one parser on the one brain).

Why this test exists: the OneBrainComposer is the integrated one-brain conversational composer (2026-06-18-one-brain-
composer-A3-GO.md). Without a guard it silently bit-rots as the agent / composer / bridge code evolves. This pins the
core capability + the no-confab moat.

HONEST SCOPE: affirmative facts (who / what / affirmative yes-no + abstention). Negation (a bound polarity tag = a 4th
role) + the richer caps (describe / reason_chain / elaborate) are documented follow-ons, NOT asserted here.

GPU-only (the on-bridge parser trains on the CuPy substrate); skips gracefully without GPU / when the concept cache is
absent (like the other on-brain agent tests).
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim.backend import is_gpu_backend  # noqa: E402

pytestmark = pytest.mark.skipif(not is_gpu_backend(),
                                reason="the OneBrainComposer's on-bridge parser needs the CuPy/GPU substrate")

VOCAB = ["dog", "cat", "bird", "river", "apple", "go", "come", "look", "stop", "swim",
         "north", "east", "south", "west", "home"]


def _build(seed):
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    a = BrainConversationalAgent(seed=seed, composer_kind="onebrain", concepts={w: None for w in VOCAB})
    a.hear("dog go north", polarity="AFFIRM")
    a.hear("cat come east", polarity="AFFIRM")
    a.hear("bird look south", polarity="AFFIRM")
    a.hear("west stop river", voice="passive", polarity="AFFIRM")   # passive frame -> agent=river (voice-invariant)
    return a


def test_onebrain_agent_matrix_and_moat():
    try:
        a = _build(42)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")

    # who / what on the persistent on-bridge store
    assert a.what_does("dog", "go") == "north"
    assert a.who_does("go", "north") == "dog"
    assert a.what_does("cat", "come") == "east"
    # voice-invariant comprehension: the passively-heard "west stop river" stores (agent=river, action=stop,
    # patient=west) -- the passive frame flips 1st<->3rd -- so it queries back as river-stop-west
    assert a.what_does("river", "stop") == "west"
    assert a.who_does("stop", "west") == "river"
    # affirmative yes/no
    assert a.is_it_true("dog", "go", "north") == "yes"
    assert a.is_it_true("bird", "look", "south") == "yes"

    # the no-confab moat: an unheard cue abstains (what_does -> None), an unheard fact abstains (is_it_true -> unknown)
    assert a.what_does("apple", "stop") is None, "moat breach: unstored cue not abstained"
    assert a.is_it_true("cat", "go", "west") in ("unknown", "no"), "moat breach: unstored fact not abstained"


def test_onebrain_negation_yes_no():
    """Negation: a fact heard with polarity='NEGATE' (a bound 4th polarity role) -> is_it_true 'no'; an affirmative
    fact -> 'yes'; an unstored fact -> 'unknown' (the moat). who/what read the stored subject-verb-object regardless of
    polarity (only the yes/no answer flips), matching the rf composer's semantics."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    try:
        a = BrainConversationalAgent(seed=42, composer_kind="onebrain", concepts={w: None for w in VOCAB})
        a.hear("dog go north", polarity="AFFIRM")
        a.hear("cat come east", polarity="NEGATE")     # asserts: cat does NOT come east
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    assert a.is_it_true("dog", "go", "north") == "yes", "affirmative fact must answer yes"
    assert a.is_it_true("cat", "come", "east") == "no", "negated fact must answer no"
    assert a.is_it_true("dog", "go", "south") == "unknown", "moat breach: unstored fact not abstained"
    # who/what still read the stored SVO of the negated fact (only the polarity/yes-no flips)
    assert a.what_does("cat", "come") == "east"


def test_onebrain_describe_and_reason():
    """The richer caps via the agent: `describe` (generation -- render the stored fact for an agent, None on an unknown
    agent = no confabulation) and `reason_chain` (multi-hop -- each action's patient becomes the next hop's agent,
    abstaining the moment a hop has no fact)."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    try:
        a = BrainConversationalAgent(seed=42, composer_kind="onebrain", concepts={w: None for w in VOCAB})
        a.hear("dog go cat")        # dog -go-> cat
        a.hear("cat go north")      # cat -go-> north
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    assert a.describe("dog") == "dog go cat", "describe must render the stored fact"
    assert a.describe("bird") is None, "moat breach: describe must not confabulate an unknown agent"
    assert a.reason_chain("dog", ["go", "go"]) == "north", "multi-hop: dog -go-> cat -go-> north"
    assert a.reason_chain("dog", ["go", "come"]) is None, "moat: no (cat, come) fact -> abstain at hop 2"


def test_onebrain_clause_parity_with_rf_oracle():
    """Recursive embedded clause: a fact whose patient is an SVO clause ('dog go (cat look south)') stores + decodes on
    the OneBrainComposer == the RFPhasorComposer numpy oracle == ground truth, via BOTH query_patient (the decoded
    inner clause sentence) AND render_fact (the outer fact with the inner clause filling the patient slot). This brings
    the rf composer's recursive-clause feature to parity on the one-brain path (toward retiring the legacy numpy
    production runtime while keeping numpy as the oracle). The on-bridge decode is a chained register->register unbind
    (outer patient -> a Q register -> the 3 clause roles -> cleanup)."""
    from research.runners.one_brain_composer import OneBrainComposer
    from research.runners.rf_phasor_composer import RFPhasorComposer, Clause
    clause = Clause(agent="cat", action="look", patient="south")   # all of dog/go/cat/look/south are in VOCAB
    try:
        c = OneBrainComposer(seed=42, D=64, vocab=VOCAB)
        oracle = RFPhasorComposer(seed=42, D=64, vocab=VOCAB)       # same seed/D/period -> identical codes
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    c.store("dog", "go", clause)
    oracle.store("dog", "go", clause)
    # query_patient: the decoded inner clause sentence
    got = c.query_patient("dog", "go")
    assert got == oracle.query_patient("dog", "go") == "cat look south", \
        f"clause query_patient {got!r} != oracle {oracle.query_patient('dog', 'go')!r} != truth 'cat look south'"
    # render_fact: the outer fact with the clause in the patient slot
    gotr = c.render_fact("dog")
    assert gotr == oracle.render_fact("dog") == "dog go cat look south", \
        f"clause render_fact {gotr!r} != oracle {oracle.render_fact('dog')!r} != truth 'dog go cat look south'"
    # the no-confab moat still holds for an unstored cue (abstain before any clause decode)
    assert c.query_patient("apple", "stop") is None, "moat breach: unstored cue not abstained"


def test_onebrain_agent_clause_fact():
    """The agent path: hear_clause_fact stores an embedded-clause fact on the OneBrainComposer; what_does decodes the
    inner clause + describe renders the outer fact; an unknown agent still abstains (the moat). Uses the agent's own
    core_sim_composition.Clause (a DISTINCT namedtuple from the rf module's -- the duck-typed _is_clause spans both)."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    from research.runners.core_sim_composition import Clause
    try:
        a = BrainConversationalAgent(seed=42, composer_kind="onebrain", concepts={w: None for w in VOCAB})
        a.hear_clause_fact("dog", "go", Clause(agent="cat", action="look", patient="south"))
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    assert a.what_does("dog", "go") == "cat look south", "agent must decode the embedded clause patient"
    assert a.describe("dog") == "dog go cat look south", "agent must render the outer fact with the inner clause"
    assert a.what_does("bird", "go") is None, "moat breach: unknown agent not abstained"


def test_onebrain_reconsolidation_parity():
    """Reconsolidation (prediction-error-gated in-place fact update) on the OneBrainComposer == the RFPhasorComposer
    numpy oracle. A corrective utterance reactivates the cued fact and -- only above the labilization gate --
    REWRITES the patient in place (no contradictory duplicate); a re-statement restabilizes; a never-stored cue
    abstains (the no-confab moat). The in-place rewrite re-composes the fact and overwrites the same store block.
    Brings the rf composer's reconsolidation to parity on the one-brain path (toward retiring the numpy runtime)."""
    from research.runners.one_brain_composer import OneBrainComposer
    from research.runners.rf_phasor_composer import RFPhasorComposer
    facts = [("dog", "go", "north"), ("cat", "come", "east")]
    try:
        c = OneBrainComposer(seed=42, D=64, vocab=VOCAB)
        oracle = RFPhasorComposer(seed=42, D=64, vocab=VOCAB)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    for (a, v, p) in facts:
        c.store(a, v, p); oracle.store(a, v, p)
    # (1) a CORRECTION ('actually, dog go south') -> rewrite in place (== oracle), no duplicate
    r = c.update_on_mismatch("dog", "go", "south")
    ro = oracle.update_on_mismatch("dog", "go", "south")
    assert r["action"] == ro["action"] == "rewrite", f"correction must rewrite: onebrain {r} vs oracle {ro}"
    assert c.query_patient("dog", "go") == "south", "rewritten fact must read the new patient"
    assert c.count_facts("dog", "go") == 1, "rewrite must not append a contradictory duplicate"
    # (2) a RE-STATEMENT ('cat come east' again) -> PE below the gate -> restabilize unchanged
    r2 = c.update_on_mismatch("cat", "come", "east")
    assert r2["action"] == "restabilize", f"a re-statement must restabilize, not rewrite: {r2}"
    assert c.query_patient("cat", "come") == "east" and c.count_facts("cat", "come") == 1
    # (3) the moat: a NEVER-stored cue abstains (no fabricated trace)
    rm = c.update_on_mismatch("bird", "go", "west")
    assert rm["action"] == "abstain" and c.count_facts("bird", "go") == 0, "moat breach: unstored cue not abstained"


def test_onebrain_grounded_codes_passthrough():
    """Production drop-in: OneBrainComposer(grounded_codes=...) uses the LEARNED-from-conversation concept codes (the
    same path the rf composer takes), not fresh random ones -- so onebrain is a true drop-in for the production
    conversational pipeline (e.g. the 320 stream-learned cortex codes). Gate: the overridden codes land in the
    cleanup codebook AND a fact stored with them queries == the RFPhasorComposer oracle built with the same codes."""
    from research.runners.one_brain_composer import OneBrainComposer
    from research.runners.rf_phasor_composer import RFPhasorComposer
    rng = np.random.default_rng(7)
    grounded = {w: rng.uniform(0.0, 1.0, 64) for w in ("dog", "go", "north")}    # learned codes for a few words
    try:
        c = OneBrainComposer(seed=42, D=64, vocab=VOCAB, grounded_codes=grounded)
        oracle = RFPhasorComposer(seed=42, D=64, vocab=VOCAB, grounded_codes=grounded)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    for w in grounded:                                                            # the learned codes propagate
        assert np.allclose(c.comp.concepts[w], grounded[w]), f"grounded code for {w!r} did not propagate"
    c.store("dog", "go", "north"); oracle.store("dog", "go", "north")            # a fact built on the learned codes
    assert c.query_patient("dog", "go") == oracle.query_patient("dog", "go") == "north", \
        "a fact stored on grounded codes must query == the oracle == truth"


def test_onebrain_multiturn_correction():
    """AGENT-LEVEL reconsolidation on the onebrain path: MultiTurnAgentV2(composer_kind="onebrain") parses a
    correction turn through the ONE on-bridge parser (the agent's own parser is None on this path -> the
    parser-agnostic agent.parse falls back to the composer's), resolves a pronoun agent from the discourse buffer,
    and rewrites the cued fact IN PLACE; a never-stored correction abstains (the no-confab moat). Validates the
    correction + pronoun + moat wiring end-to-end on the one brain (toward making onebrain the agent default)."""
    from research.runners.multi_turn_agent_v2 import MultiTurnAgentV2
    cvocab = ["dog", "cat", "bird", "elephant", "go", "run", "fly", "north", "south", "east", "west"]
    referents = ["dog", "cat", "bird", "elephant"]
    try:
        a = MultiTurnAgentV2(referent_concepts=referents, concepts={w: None for w in cvocab},
                             seed=42, composer_kind="onebrain")
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    a.hear("dog go north")                            # foregrounds 'dog' (north is not a referent)
    res = a.correct("actually it go south")           # 'it' -> dog (the discourse buffer) -> rewrite in place
    assert res["wrote"] is True and res["action"] == "rewrite", f"pronoun correction must resolve + rewrite: {res}"
    assert a.what_does("dog", "go") == "south", "the corrected fact must read the new patient"
    assert a.agent.composer.count_facts("dog", "go") == 1, "no contradictory duplicate"
    rm = a.correct("actually elephant fly west")      # a never-stored subject -> abstain (the moat)
    assert rm["action"] == "abstain" and rm["wrote"] is False, "moat breach: never-stored correction not abstained"


def test_onebrain_multiturn_anaphora():
    """AGENT-LEVEL multi-turn anaphora on the onebrain path: MultiTurnAgent(composer_kind="onebrain") writes a
    discourse referent on turn 1, and a turn-2 pronoun ('it') resolves to it for the onebrain query (the cross-turn
    spiking WM + the onebrain composer together). The empty-WM moat is a WM property independent of composer kind
    (it abstains BEFORE any composer query), so it is covered by the rf MultiTurnAgent test, not re-run here."""
    from research.runners.multi_turn_agent import MultiTurnAgent
    nouns = ["dog", "cat", "fish", "bird"]
    cvocab = nouns + ["chase", "eat"]
    try:
        a = MultiTurnAgent(referent_concepts=nouns, concepts={w: None for w in cvocab}, seed=42,
                           composer_kind="onebrain")
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    a.agent.composer.store("cat", "eat", "fish")      # the fact the turn-2 answer needs
    a.hear("dog chase cat")                           # turn 1: writes the object referent 'cat' to the WM
    assert a.what_does("it", "eat") == "fish", "turn-2 'it' -> cat -> (cat eat fish)"


def test_onebrain_confidence_gate_preserves_intact():
    """The familiarity/confidence gate (the graceful-degradation fix, 2026-06-18-emergent-graceful-degradation-derisk):
    a `confidence_gate > 0` blanks a NOISE-DOMINATED (low cleanup-margin) block so a damaged store abstains instead of
    confabulating. The SAFETY property pinned here: on an INTACT store the gate must NOT change the answers (intact
    reconstructions have a high margin, well above the gate) -- recall + the no-confab moat are preserved -- so enabling
    the gate is safe. (The damaged-store abstention is exercised by the de-risk runner.)"""
    from research.runners.one_brain_composer import OneBrainComposer
    facts = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south")]
    try:
        c = OneBrainComposer(seed=42, D=64, vocab=VOCAB, confidence_gate=0.15)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    for (a, v, p) in facts:
        c.store(a, v, p)
    for (a, v, p) in facts:                                            # intact reads are confident -> NOT blanked
        assert c.query_patient(a, v) == p, f"the gate must not blank a confident intact read for {(a, v)}"
        assert c.ask_yes_no(a, v, p) == "yes"
    assert c.query_patient("apple", "stop") is None, "moat intact under the gate: an unstored cue still abstains"
    assert c.ask_yes_no("cat", "go", "west") in ("unknown", "no")


def test_agent_attributed_comprehension():
    """Richer-syntax #1 PRODUCTION integration: BrainConversationalAgent(enable_attributed=True) comprehends an
    attributed-entity sentence ('dog eat big red apple') via the NEURAL attributed parser (parse-in-spikes) and
    routes the (adjs, noun) to the composer's ready attribute roles, so what_does('dog','eat') -> 'big red apple'.
    Also asserts default-off is byte-identical (enable_attributed=False -> no attributed parser)."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    avocab = ["dog", "cat", "apple", "river", "eat", "see", "big", "red", "small", "hot"]
    try:
        a = BrainConversationalAgent(seed=42, composer_kind="rf", concepts={w: None for w in avocab},
                                     enable_attributed=True)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    a.hear_attributed("dog eat big red apple")            # neural attributed parse -> composer.store((adjs,noun))
    assert a.what_does("dog", "eat") == "big red apple", "attributed patient must round-trip through the agent"
    a.hear_attributed("cat see small apple")              # 1-adjective attributed
    assert a.what_does("cat", "see") == "small apple"
    assert a.what_does("river", "eat") is None, "moat: an unheard cue abstains"
    # explicit opt-OUT (enable_attributed is now DEFAULT-ON after the 2026-06-19 default-on consolidation): with the
    # flag explicitly off, no attributed parser is built -> the byte-identical opt-out path still works.
    b = BrainConversationalAgent(seed=42, composer_kind="rf", concepts={w: None for w in avocab},
                                 enable_attributed=False)
    assert b._attr_parser is None, "enable_attributed=False opt-out: no attributed parser (byte-identical)"


def test_onebrain_batched_equals_per_block():
    """A5 lever 1: the BATCHED read (default, read all blocks in 3 windows) == the per-block oracle (enable_batched
    toggled off) on the production OneBrainComposer -- answer-identical, just faster (the de-risk: 7.3x)."""
    from research.runners.one_brain_composer import OneBrainComposer
    facts = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south")]
    try:
        c = OneBrainComposer(seed=42, D=64, vocab=VOCAB)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    for (a, v, p) in facts:
        c.store(a, v, p)                                  # store() resolves roles directly (no parser needed)
    for (a, v, p) in facts:
        c.enable_batched = True
        bat = (c.query_patient(a, v), c.query_agent(v, p), c.ask_yes_no(a, v, p))
        c.enable_batched = False
        per = (c.query_patient(a, v), c.query_agent(v, p), c.ask_yes_no(a, v, p))
        assert bat == per == (p, a, "yes"), f"batched {bat} != per-block {per} != truth for {(a, v, p)}"
    # moat parity (absent cue)
    c.enable_batched = True
    assert c.query_patient("apple", "stop") is None
    c.enable_batched = False
    assert c.query_patient("apple", "stop") is None


def test_onebrain_default_path_unaffected():
    """The additive wiring must not change the default ('rf') agent: it has no `hear` on its composer, so it builds the
    agent's own parser and uses parse+store (the byte-unchanged path). A construction smoke (no GPU run needed)."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    import inspect
    src = inspect.getsource(BrainConversationalAgent.hear)
    assert "self.composer.hear" in src and "_ensure_parser" in src and "self.composer.store" in src, \
        "hear() must keep BOTH the delegation path (onebrain: composer.hear) and the parse+store path " \
        "(rf/rate default: _ensure_parser().parse -> composer.store; _ensure_parser is the lazy-built parser)"


def test_onebrain_encoding_gain_default_off_byte_identical():
    """(Tier-2 #6, Route B mirror on the one-brain path) DEFAULT-OFF GUARD: encoding_gain_fn=None must write the SAME
    persistent store weights as before (the byte-identical unit-magnitude write) -- so wiring the dopamine encoding-gain
    hook changes nothing unless a gain fn is supplied. A constant-1.0 gain fn must ALSO be byte-identical (g=1.0 == the
    unit write). No GPU run needed (only the store-weight construction is exercised)."""
    from research.runners.one_brain_composer import OneBrainComposer
    facts = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south")]
    try:
        base = OneBrainComposer(seed=42, D=64, vocab=VOCAB)                          # default (no gain hook)
        unit = OneBrainComposer(seed=42, D=64, vocab=VOCAB, encoding_gain_fn=lambda: 1.0)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    assert base.encoding_gain_fn is None, "default must be the byte-identical unit write (encoding_gain_fn None)"
    for (a, v, p) in facts:
        base.store(a, v, p); unit.store(a, v, p)
    # the persistent store weights (cp_rf_w_re/im carry these) must be IDENTICAL: same (post, pre) edges, same complex w
    bc = [(int(po), int(pr), complex(w)) for (po, pr, w) in base.store_conns]
    uc = [(int(po), int(pr), complex(w)) for (po, pr, w) in unit.store_conns]
    assert len(bc) == len(uc) == len(facts) * 64, "store_conns length must match (3 facts x D=64)"
    assert bc == uc, "g=1.0 (and None) must produce the BYTE-IDENTICAL unit-magnitude store write"


def test_onebrain_encoding_gain_lifts_recall_moat_intact():
    """(Tier-2 #6, Route B mirror) A constant encoding_gain_fn (g>1) must (1) WRITE a higher-magnitude store block (the
    salient/rewarded fact stored more strongly -- the only thing the gain changes) while (2) PRESERVING the no-confab
    moat: every who/what answer still correct, and an unstored cue / unstored fact still ABSTAIN (is None / unknown).
    The HARD load-bearing constraint: DA-modulated encoding must NEVER produce a false-accept. The recall lift itself
    is the magnitude differential under the RF read floor (sim/bridge.py:5589); here we pin that the stored block's
    magnitude is scaled by g AND the moat is byte-intact (0 false-accepts)."""
    from research.runners.one_brain_composer import OneBrainComposer
    facts = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south")]
    try:
        c = OneBrainComposer(seed=42, D=64, vocab=VOCAB, encoding_gain_fn=lambda: 1.5)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    for (a, v, p) in facts:
        c.store(a, v, p)
    # (1) the WRITE: every stored block's per-edge magnitude is g x the unit composite (|zc|==1 -> |g*zc|==g)
    mags = np.abs([complex(w) for (_po, _pr, w) in c.store_conns])
    assert np.allclose(mags, 1.5, atol=1e-9), f"a g=1.5 fact must write |w|==1.5 (got mean {mags.mean():.4f})"
    # (2) the moat HARD gate -- recall correct AND 0 false-accepts under the gain
    for (a, v, p) in facts:
        assert c.query_patient(a, v) == p, f"recall must stay correct under the encoding gain for {(a, v)}"
        assert c.query_agent(v, p) == a
        assert c.ask_yes_no(a, v, p) == "yes"
    assert c.query_patient("apple", "stop") is None, "MOAT BREACH: unstored cue must abstain under the gain"
    assert c.query_agent("swim", "home") is None, "MOAT BREACH: unstored cue must abstain under the gain"
    assert c.ask_yes_no("cat", "go", "west") in ("unknown", "no"), "MOAT BREACH: unstored fact must abstain"


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_onebrain_cleanup_codebook_local_conj_byte_identical(seed):
    """FHRR-B cleanup-codebook residual on the PRODUCTION one-brain path: with local_reciprocal_unbind ON, the 7
    cleanup-codebook conj sites (comp.concepts[...] / comp.pol_words[...]) derive the matched-filter codebook from the
    concept codes by the per-component quadrature-flip (_cleanup_conj) instead of the host np.conj. It must give
    answers BYTE-IDENTICAL to the host-conj default on the who/what matrix + the no-confab moat, AND a FULL store+query
    build must issue ZERO np.conj calls TOTAL (combined with Mechanism 1's unbind rule -> the whole one-brain
    bind+cleanup structure host-free). De-risk: research/findings/2026-06-20-FHRR-B-cleanup-codebook-local-conj.md."""
    from research.runners.one_brain_composer import OneBrainComposer
    try:
        cn = OneBrainComposer(seed=seed, D=64, vocab=VOCAB, enable_batched=False, local_reciprocal_unbind=False)
        cl = OneBrainComposer(seed=seed, D=64, vocab=VOCAB, enable_batched=False, local_reciprocal_unbind=True)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")

    # (a) the cleanup codebook helper is bit-for-bit conj(concept) for every concept (main vocab + polarity tags).
    for w in list(cn.words) + cn.pol_words:
        legacy = np.conj(cn.comp._to_phasor(cn.comp.concepts[w]))
        local = cl._cleanup_conj(w)
        assert np.array_equal(legacy, local), f"local cleanup rule != conj(concept) for '{w}'"

    # (b) the who/what matrix + abstentions byte-identical OFF vs ON (GPU run).
    for c in (cn, cl):
        c.store("dog", "go", "north"); c.store("cat", "come", "east"); c.store("bird", "look", "south")
    assert cl.query_agent("go", "north") == cn.query_agent("go", "north") == "dog"
    assert cl.query_patient("cat", "come") == cn.query_patient("cat", "come") == "east"
    assert cl.query_patient("bird", "look") == cn.query_patient("bird", "look") == "south"
    # the no-confab moat: identical abstentions
    assert cl.query_agent("go", "south") is None and cn.query_agent("go", "south") is None
    assert cl.query_patient("apple", "stop") is None and cn.query_patient("apple", "stop") is None

    # (c) substrate-purity: with the flag ON, a FULL store+query build issues ZERO np.conj calls TOTAL.
    cp = OneBrainComposer(seed=seed, D=48, vocab=VOCAB, enable_batched=False, local_reciprocal_unbind=True)
    cp.store("dog", "go", "north"); cp.store("cat", "come", "east")
    n_conj = {"n": 0}
    orig = np.conj
    np.conj = lambda x, _c=n_conj, _o=orig: (_c.__setitem__("n", _c["n"] + 1), _o(x))[1]
    try:
        cp.query_agent("go", "north")
        cp.query_patient("cat", "come")
        cp.query_agent("go", "south")        # the moat (abstains)
    finally:
        np.conj = orig
    assert n_conj["n"] == 0, "flag ON must not call np.conj in a full one-brain store+query build"
