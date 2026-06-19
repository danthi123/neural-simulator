"""THE GATEKEEPER for the default-on consolidation (owner directive 2026-06-19, AUTONOMOUS_STATE CYCLE 266):
validate that the conversational capability flags WORK TOGETHER and -- the HARD gate -- the no-confab moat holds
0-breach under the COMBINED config, before flipping their constructor defaults to ON.

The capability flags and their substrate:
  * enable_attributed         -- a NEURAL AttributedBridgeParser (parse-in-spikes). GPU-validated.
  * enable_multiframe         -- a NEURAL FrameParser (verb-position -> frame -> role). GPU-validated.
  * enable_neural_render      -- a spiking competitive-queuing serial-order renderer. GPU-validated.
  * enable_biased_competition -- (MultiTurnAgent) WTA biased competition for multi-referent pronouns. CPU+GPU.
  * enable_multicue_competition -- the spiking multi-cue role competition for degraded word order. CPU+GPU,
                                   but REQUIRES `multicue_verbs` (a hand-curated VERB_SELECTS/animacy lexicon) ->
                                   it CANNOT be sensibly defaulted (the agent's vocab is just {word: code}, no
                                   part-of-speech / selectional-restriction info). It also re-routes ALL of hear()
                                   through the 2-noun-transitive content competition. Hence it is the documented
                                   OPT-IN carve-out and is NOT folded into the always-on combined config here; it is
                                   tested SEPARATELY (one agent at a time) for moat + capability, since enabling it
                                   replaces -- rather than composes with -- the position/frame parser path.

What the combined-config build covers (all four always-on capabilities ON together, multi-seed where it matters):
  1. each capability still WORKS with the others on;
  2. the documented-FRAGILE interaction (attribute + embedded clause together) -- exercised, classified honestly;
  3. the no-confab MOAT holds (0 breaches) -- the HARD gate;
  4. clean canonical who/what Q&A is un-regressed.

GPU-required (the attributed / frame / render bridges are GPU-validated Hebbian parsers); skips gracefully on a
non-GPU backend (like the other on-brain agent tests). The two CPU-runnable flags (biased_competition, multicue)
ALSO have their own numpy-backend CI guards (test_multireferent_biased_competition.py, test_multicue_competition_
agent.py); this file is the COMBINED-config gate on the production substrate.
"""
import os

os.environ.setdefault("SIM_BACKEND", "cupy")

import pytest

from sim.backend import is_gpu_backend  # noqa: E402

pytestmark = pytest.mark.skipif(
    not is_gpu_backend(),
    reason="the attributed / frame / neural-render bridges are GPU-validated Hebbian parsers")

# rf composer with an explicit vocab -> no denoise64 cache needed. Animate agents + an inanimate object so the
# attributed / multiframe / canonical sentences all read cleanly, plus adjectives for the attributed path.
NOUNS = ["dog", "cat", "bird", "apple", "ball", "river"]
VERBS = ["eat", "see", "go", "run", "chase", "look", "come"]
ADJS = ["big", "red", "small", "hot"]
DIRS = ["north", "south", "east", "west"]
VOCAB = {w: None for w in NOUNS + VERBS + ADJS + DIRS}
SEEDS = (42, 43, 44)   # multi-seed where it matters (>=3 per the directive)


def _combined_agent(seed):
    """A BrainConversationalAgent with ALL four always-on capabilities ON together (attributed + multiframe +
    neural_render are the combined-config trio on the agent; biased_competition lives on MultiTurnAgent and is
    exercised in its own test below). rf composer, explicit vocab -> no cache."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    return BrainConversationalAgent(seed=seed, composer_kind="rf", concepts=VOCAB,
                                    enable_attributed=True, enable_multiframe=True, enable_neural_render=True)


# ----------------------------------------------------------------------------------------------------------------
# 1. each capability WORKS with the others on (the combined config), multi-seed; 4. canonical Q&A un-regressed.
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_combined_canonical_qa_and_describe(seed):
    """Clean canonical who/what + neural-rendered describe, with attributed+multiframe ALSO enabled (they must not
    perturb the native SVO path). The no-confab moat abstains on an unstored cue."""
    try:
        a = _combined_agent(seed)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    a.hear("dog go north")
    a.hear("cat come south")            # a second canonical SVO (all words in-vocab)
    assert a.what_does("dog", "go") == "north"
    assert a.who_does("go", "north") == "dog"
    # neural-rendered generation (enable_neural_render ON): the word ORDER is the spiking CQ read-out, identical to
    # the f-string for a flat fact.
    assert a.describe("dog") == "dog go north"
    # HARD gate -- the moat under the combined config:
    assert a.what_does("river", "look") is None, "moat breach: unstored cue not abstained"
    assert a.describe("river") is None, "moat breach: describe must not confabulate an unknown agent"


@pytest.mark.parametrize("seed", SEEDS)
def test_combined_attributed_with_other_flags_on(seed):
    """enable_attributed WORKS with multiframe + neural_render also on: an attributed-entity sentence
    ('dog eat big red apple') round-trips through the neural attributed parser -> the composer's attribute roles."""
    try:
        a = _combined_agent(seed)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    a.hear_attributed("dog eat big red apple")
    assert a.what_does("dog", "eat") == "big red apple", "attributed patient must round-trip (combined config)"
    a.hear_attributed("cat see small apple")               # 1-adjective attributed
    assert a.what_does("cat", "see") == "small apple"
    assert a.what_does("river", "eat") is None, "moat: an unheard attributed cue abstains"


@pytest.mark.parametrize("seed", SEEDS)
def test_combined_multiframe_with_other_flags_on(seed):
    """enable_multiframe WORKS with attributed + neural_render also on: a non-SVO frame is auto-selected and
    comprehended -- VSO ('run dog north') and OSV ('north dog run') answer who/what like canonical SVO."""
    try:
        a = _combined_agent(seed)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    a.hear_multiframe("run dog north", VERBS)              # VSO (verb at position 0)
    assert a.what_does("dog", "run") == "north", "VSO comprehension (combined config)"
    assert a.who_does("run", "north") == "dog"
    a.hear_multiframe("south cat go", VERBS)               # OSV (verb at position 2)
    assert a.what_does("cat", "go") == "south", "OSV comprehension (combined config)"
    assert a.what_does("river", "go") is None, "moat: an unheard multiframe cue abstains"


# ----------------------------------------------------------------------------------------------------------------
# 2. the documented-FRAGILE interaction: attribute + embedded clause together. Classify honestly.
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_combined_attribute_and_clause_interaction(seed):
    """CLAUDE.md flags the attribute(5-role) + embedded-clause combination as tipping on the noise margin at small
    D. Exercise it under the combined config and classify: store an attributed fact AND a clause fact in the SAME
    agent, and assert (a) each reads back AND (b) -- the load-bearing part -- the moat holds (no confabulation),
    even if the harder of the two degrades. The rf composer's attribute role is the single-attribute path, which
    HOLDS; the clause is a separate role binding. We assert both round-trip at this D and seed; if a future D/seed
    shows the documented fragility, this test localizes it (and the moat assertion is the non-negotiable part)."""
    from research.runners.core_sim_composition import Clause
    try:
        a = _combined_agent(seed)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    a.hear_attributed("dog eat big red apple")                       # 5-role attributed fact
    a.hear_clause_fact("bird", "look", Clause("cat", "go", "south"))  # embedded-clause fact in the SAME agent
    # both round-trip (the attribute path is single-attribute -> holds; the clause is a distinct role binding):
    assert a.what_does("dog", "eat") == "big red apple", "attributed fact must hold alongside a clause fact"
    assert a.what_does("bird", "look") == "cat go south", "clause fact must hold alongside an attributed fact"
    # the HARD gate -- the moat under the attribute+clause combination:
    assert a.what_does("river", "look") is None, "moat breach: unstored cue not abstained (attr+clause config)"
    assert a.what_does("ball", "eat") is None, "moat breach: unstored attributed cue not abstained"


# ----------------------------------------------------------------------------------------------------------------
# biased_competition (MultiTurnAgent) -- the 4th always-on capability, on its own agent (it lives on MultiTurnAgent).
# CPU+GPU; here we run it on whatever backend is active (the dedicated numpy guard is test_multireferent_*).
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_combined_biased_competition_multireferent(seed):
    """enable_biased_competition WORKS: a pronoun over >=2 held referents of opposing content features resolves to
    the content-favored one; the moat abstains on empty WM / content-silent verb. (MultiTurnAgent also turns the
    inner BrainConversationalAgent's neural_render on, so this exercises render + biased competition together.)"""
    from research.runners.multi_turn_agent import MultiTurnAgent
    bc_nouns = ["dog", "cat", "fish", "bird", "worm", "ball"]
    bc_vocab = bc_nouns + ["chase", "eat"]
    try:
        a = MultiTurnAgent(referent_concepts=bc_nouns, concepts={w: None for w in bc_vocab}, seed=seed,
                           enable_biased_competition=True, enable_neural_render=True)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    a.agent.composer.store("cat", "eat", "fish")           # if 'it'->cat (correct for 'eat'), answer = fish
    a.agent.composer.store("ball", "eat", "worm")          # if 'it'->ball (wrong for 'eat'), answer = worm
    a._write_referent("cat"); a._write_referent("ball")
    assert a._resolve_biased("eat") == "cat", "content (animate) must win for 'eat'"
    assert a.what_does("it", "eat") == "fish", "the turn answers via the content-resolved referent"
    # HARD gate -- the moat:
    b = MultiTurnAgent(referent_concepts=bc_nouns, concepts={w: None for w in bc_vocab}, seed=seed,
                       enable_biased_competition=True, enable_neural_render=True)
    assert b._resolve_biased("eat") is None, "moat: empty WM -> abstain"


# ----------------------------------------------------------------------------------------------------------------
# multicue (the OPT-IN carve-out): validated SEPARATELY (it re-routes hear() and cannot be defaulted). We pin the
# capability + moat once more under the combined-config spirit -- one agent with multicue ON, exercised alongside a
# fresh canonical agent -- to confirm enabling it does not weaken the moat.
# ----------------------------------------------------------------------------------------------------------------
def test_multicue_optin_capability_and_moat():
    """The carve-out flag, validated for capability + moat (its dedicated guard is test_multicue_competition_
    agent.py). Object-fronted 'apple eat dog' assigns dog=agent/apple=patient; the moat abstains on an unstored
    cue and reports decisive=False for an all-ambiguous transitive. Confirms enabling multicue keeps the moat."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    try:
        a = BrainConversationalAgent(seed=42, composer_kind="rf", concepts=VOCAB,
                                     enable_multicue_competition=True, multicue_verbs=["eat", "chase"])
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    roles = a.hear("apple eat dog")                        # object-fronted; content overrides position
    assert roles["agent"] == "dog" and roles["patient"] == "apple"
    assert a.who_does("eat", "apple") == "dog"
    assert a.what_does("dog", "go") is None, "moat: an unstored fact abstains under multicue"
    parser = a._ensure_multicue_parser()
    _r, decisive = parser.parse_decisive(["dog", "chase", "cat"])    # two animate + symmetric verb
    assert decisive is False, "moat: content cannot break the tie -> decisive False (no confabulation)"
