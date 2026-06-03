"""Tests for the integrated conversation loop (comprehend -> content-selection Control -> produce).

The loop unifies three validated abilities: comprehend (SVO parse) + decide-what-to-say (the content-
selection Control) + produce (generate-by-composition). These tests cover binding, factual Q&A, and the
NEW Control-driven topic elaboration (the dialogue-planning capability the factual-Q&A loop lacked).
"""

from research.runners.integrated_conversation_loop import ConversationalAgent, WORDS, ROLES3


def _teach(agent, *facts):
    for f in facts:
        agent.hear(f)


def test_statement_binds_into_kb():
    a = ConversationalAgent(seed=42)
    resp = a.hear("dog chase cat")
    assert len(a.kb) == 1
    assert a.kb[0] == {"agent": WORDS.index("dog"), "action": WORDS.index("chase"),
                       "patient": WORDS.index("cat")}
    assert "learned" in resp


def test_factual_question_what_produces_sentence():
    a = ConversationalAgent(seed=42)
    a.hear("dog chase cat")
    resp = a.hear("what does dog chase")
    assert resp.split() == ["dog", "chase", "cat"]      # produced full sentence, in order


def test_factual_question_who():
    a = ConversationalAgent(seed=42)
    a.hear("dog chase cat")
    resp = a.hear("who chase cat")
    assert resp.split() == ["dog", "chase", "cat"]


def test_topic_elaboration_produces_on_topic_fact():
    a = ConversationalAgent(seed=42)
    _teach(a, "dog chase cat", "dog eat apple")
    resp = a.hear("dog")                                # raise a topic (not a question)
    # the agent elaborates: produces a real dog fact as a sentence
    assert "dog" in resp.split()
    assert resp.split() in (["dog", "chase", "cat"], ["dog", "eat", "apple"])


def test_more_progresses_non_repeating():
    a = ConversationalAgent(seed=42)
    _teach(a, "dog chase cat", "dog eat apple")
    r1 = a.hear("dog")
    r2 = a.hear("more")
    assert r1.split() != r2.split()                     # two DISTINCT dog facts (non-repetition)
    assert {tuple(r1.split()), tuple(r2.split())} == {("dog", "chase", "cat"), ("dog", "eat", "apple")}


def test_elaboration_exhausts_then_says_so():
    a = ConversationalAgent(seed=42)
    _teach(a, "dog chase cat", "dog eat apple")
    a.hear("dog"); a.hear("more")                       # both dog facts elaborated
    resp = a.hear("more")
    assert "that's all" in resp.lower() or "don't know" in resp.lower()


def test_topic_shift_elaborates_new_topic():
    a = ConversationalAgent(seed=42)
    _teach(a, "dog chase cat", "child hold ball")
    a.hear("dog")
    resp = a.hear("child")                              # shift topic
    assert "child" in resp.split()                      # now elaborates a child fact
    assert resp.split() == ["child", "hold", "ball"]


def test_unknown_topic_is_honest():
    a = ConversationalAgent(seed=42)
    a.hear("dog chase cat")
    resp = a.hear("river")                              # known word but no facts about it
    assert "don't know" in resp.lower() or "that's all" in resp.lower()


def test_unparseable_input():
    a = ConversationalAgent(seed=42)
    assert "didn't understand" in a.hear("").lower() or a.hear("") == "(i didn't understand)"
