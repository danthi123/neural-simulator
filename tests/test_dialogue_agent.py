"""Tests for the interactive coherent-dialogue agent (content-selection Control applied)."""

from research.runners.dialogue_agent import DialogueAgent, run_conversation


def _graph():
    # Two topics (weather, food), each with within-topic associations; no cross edges.
    return {
        "rain": {"cloud": 2.0, "storm": 1.5},
        "cloud": {"sky": 2.0, "rain": 2.0},
        "storm": {"wind": 1.8, "rain": 1.5},
        "apple": {"fruit": 2.0, "tree": 1.5},
        "fruit": {"sweet": 1.8, "apple": 2.0},
        "tree": {"leaf": 1.6, "apple": 1.5},
    }


def test_agent_responds_with_strongest_associate_of_topic():
    agent = DialogueAgent(_graph())
    assert agent.respond("rain") == "cloud"        # rain's strongest associate


def test_follow_up_continues_same_topic_without_repeating():
    agent = DialogueAgent(_graph())
    r1 = agent.respond("rain")                      # cloud
    r2 = agent.respond("more")                      # next coherent associate of rain, not cloud again
    r3 = agent.respond("more")
    assert r1 == "cloud"
    assert len({r1, r2, r3}) == 3                   # three distinct responses, no repeats


def test_topic_shift_changes_focus():
    agent = DialogueAgent(_graph())
    agent.respond("rain")
    agent.respond("more")
    shifted = agent.respond("apple")                # new concept -> focus shifts to apple
    assert shifted in _graph()["apple"]             # responds within apple's associations


def test_no_repeats_across_a_whole_conversation():
    convo = run_conversation(_graph(), ["rain", "more", "apple", "more", "more"])
    responses = [r for _, r in convo if r is not None]
    assert len(set(responses)) == len(responses)    # coherence carries across topic shifts, no repeats


def test_no_focus_yet_returns_none():
    agent = DialogueAgent(_graph())
    assert agent.respond("more") is None            # 'more' before any topic -> nothing to say


def test_yes_no_question_associated():
    agent = DialogueAgent(_graph())
    ans = agent.respond("is rain related to storm?")   # rain-storm edge exists
    assert ans.startswith("Yes")
    assert "rain" in ans and "storm" in ans


def test_yes_no_question_not_associated():
    agent = DialogueAgent(_graph())
    ans = agent.respond("is rain related to apple?")   # different topics, no edge
    assert ans.startswith("No")


def test_common_link_question():
    # cloud associates {sky, rain}; storm associates {wind, rain} -> common = rain
    g = {"cloud": {"sky": 2.0, "rain": 2.0}, "storm": {"wind": 1.8, "rain": 1.5}}
    agent = DialogueAgent(g)
    ans = agent.respond("what links cloud and storm?")
    assert "rain" in ans and "both associated with" in ans


def test_unknown_input():
    agent = DialogueAgent(_graph())
    assert agent.respond("xyzzy") == "I don't know about that."


def test_describe_multi_fact():
    agent = DialogueAgent(_graph())
    ans = agent.respond("tell me about rain")          # rain -> cloud (2.0), storm (1.5)
    assert "rain is associated with" in ans
    assert "cloud" in ans and "storm" in ans


def test_negation_when_not_associated():
    agent = DialogueAgent(_graph())
    ans = agent.respond("is rain not related to apple?")   # different topics, no edge
    assert ans.startswith("Right")


def test_negation_when_associated():
    agent = DialogueAgent(_graph())
    ans = agent.respond("is rain not related to storm?")   # they ARE associated -> correct the negation
    assert ans.startswith("Actually")


def test_agent_accepts_injected_controller():
    # An injected controller (same .turn / .ctx.update interface) is used instead of the default --
    # this is how the faithful SpikingSpreadingController is dropped in behind the same dialogue agent.
    class _StubCtx:
        def update(self, concepts):
            pass

    class _StubController:
        def __init__(self):
            self.ctx = _StubCtx()

        def turn(self, concepts):
            return "stub:" + concepts[0]

    agent = DialogueAgent(_graph(), controller=_StubController())
    assert agent.respond("rain") == "stub:rain"            # the injected controller's turn() is used
