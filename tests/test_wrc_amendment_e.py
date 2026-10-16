"""Failing-first test for Amendment E (RULING wrc-prereg-go-1016b), word-role-flip prereg.

RED before the persistence lands: the identity runners do not persist a word-role parse-route readout (the
`for` field is NOT accepted as the readout), so the off-target precondition could not be evaluated per gate
seed. GREEN once both runners persist `wr_route` per turn via research.runners.wr_route.
"""
import inspect

from research.runners.wr_route import wr_route


def test_wr_route_semantics():
    assert wr_route(None) == "not-reached"
    assert wr_route({}) == "not-reached"
    assert wr_route({"route": "word_role", "agent": "cat", "action": "eats"}) == "engaged"
    assert wr_route({"route": "word_role_commit", "agent": "cat", "action": "eats"}) == "engaged"
    assert wr_route({"route": "word_role", "agent": None, "action": None}) == "declined"
    assert wr_route({"route": "word_role_commit", "agent": "cat", "action": None}) == "declined"
    assert wr_route({"route": "word_role", "agent": "cat", "action": "eats", "error": None}) == "engaged"
    assert wr_route({"error": "ValueError: swallowed organ boom"}) == "declined"


def test_identity_runners_persist_wr_route():
    """Both gate-line identity runners persist the route per turn (Amendment E)."""
    from research.runners import iw1_default_identity, iw2_default_identity

    for mod in (iw1_default_identity, iw2_default_identity):
        src = inspect.getsource(mod)
        assert '"wr_route"' in src, "%s does not persist wr_route per turn" % mod.__name__
        assert 'word_role_trace' in src, "%s does not read the per-turn word_role_trace" % mod.__name__
