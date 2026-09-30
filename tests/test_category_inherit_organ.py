"""Category-inheritance wiring for Stage-1 C1 (research/runners/category_inherit_organ.py,
webapp/category_inherit_chat.py, research/runners/stage1_c1_category.py). Prereg:
research/findings/2026-09-30-stage1-c1-category-inheritance-PREREGISTRATION.md.

Two layers: (1) the gate wrapper's routing logic against a scripted fake chat (no brain); (2) the spiking organ on the
Stage-1 C1 fact sequence at a DEV seed (numpy, ~40 s): the held-out canary reaches `worm`, follows a permuted
exposure to `bug`, abstains under the read lesion and without the learned frame-frame association, keeps the moat,
and is deterministic per seed."""
import os

import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")
os.environ.setdefault("SIM_BACKEND", "numpy")

from webapp import category_inherit_chat as CI  # noqa: E402

C1_FACTS = [("robin", "eat", "worm"), ("trout", "eat", "bug"), ("robin", "fly", "sky"), ("trout", "swim", "lake"),
            ("sparrow", "eat", "worm"), ("salmon", "eat", "bug"), ("sparrow", "fly", "sky"),
            ("salmon", "swim", "lake"), ("finch", "eat", "worm"), ("finch", "fly", "sky")]
DEV_SEED = 3


# ── (1) the wrapper, no brain ────────────────────────────────────────────────────────────────────────────────────
class _StubOrgan:
    def __init__(self, patient="worm"):
        self.observed, self.asked, self.patient = [], [], patient

    def observe(self, a, v, p):
        self.observed.append((a, v, p))
        return {"observed": True}

    def infer(self, a, v):
        self.asked.append((a, v))
        return {"patient": self.patient, "reason": "reached" if self.patient else "below_floor", "admitted": []}


class _FakeChat:
    """gate_extract returns the scripted tuple; gate() calls gate_extract (as every production bus does) and returns
    the scripted verdict."""

    def __init__(self):
        self.script = {}

    def gate_extract(self, q):
        return self.script[q][0]

    def gate(self, q):
        self.gate_extract(q)
        return self.script[q][1]


def _chat(monkeypatch, patient="worm"):
    chat = _FakeChat()
    chat.script = {
        "the canary flies in the sky": (("done", ["canary", "fly", "sky"]), ["canary", "fly", "sky"]),
        "the canary does not swim": (("done", ["canary", "swim", "lake"]), ["canary", "swim", "lake"]),
        "what does the canary eat": (("route", "q", "canary", "eat", False), None),
        "what does it eat": (("route", "q", "canary", "eat", True), None),
        "what does the robin eat": (("route", "q", "robin", "eat", False), ["robin", "eat", "worm"]),
        "who are you": (("router", "q"), None),
    }
    org = _StubOrgan(patient)
    monkeypatch.setattr(CI, "get_organ", lambda c, seed: org)
    chat._category_inherit_organ = org
    monkeypatch.setattr(CI, "_affirm_polarity", lambda q: "not" not in q)
    CI.install(chat, seed=7)
    return chat, org


def test_default_is_off():
    assert CI._CATEGORY_INHERIT_DEFAULT_ON is False
    old = os.environ.pop("BRAIN_CATEGORY_INHERIT", None)
    try:
        assert CI.category_inherit_enabled() is False
    finally:
        if old is not None:
            os.environ["BRAIN_CATEGORY_INHERIT"] = old


def test_wrapper_acts_only_on_a_direct_routable_miss(monkeypatch):
    monkeypatch.setenv("BRAIN_CATEGORY_INHERIT", "1")
    chat, org = _chat(monkeypatch)
    assert chat.gate("the canary flies in the sky") == ["canary", "fly", "sky"]
    assert org.observed == [("canary", "fly", "sky")]
    assert chat.gate("the canary does not swim") == ["canary", "swim", "lake"]
    assert org.observed == [("canary", "fly", "sky")], "a negated assertion must not teach the organ"
    got = chat.gate("what does the canary eat")
    from research.runners.brain_chat_tui import HypothesisSVO
    assert isinstance(got, HypothesisSVO) and list(got) == ["canary", "eat", "worm"], got
    assert chat.gate("what does it eat") is None, "an anaphora-resolved miss is untouched"
    assert chat.gate("what does the robin eat") == ["robin", "eat", "worm"], "a committed answer is untouched"
    assert chat.gate("who are you") is None
    assert org.asked == [("canary", "eat")]
    assert CI.turn_trace(chat)["read"] is None          # the last turn (router) left no read


def test_wrapper_is_inert_when_the_flag_is_off_at_runtime(monkeypatch):
    monkeypatch.setenv("BRAIN_CATEGORY_INHERIT", "1")
    chat, org = _chat(monkeypatch)
    monkeypatch.setenv("BRAIN_CATEGORY_INHERIT", "0")
    assert chat.gate("the canary flies in the sky") == ["canary", "fly", "sky"]
    assert chat.gate("what does the canary eat") is None
    assert org.observed == [] and org.asked == []


def test_an_organ_abstain_keeps_the_abstain(monkeypatch):
    monkeypatch.setenv("BRAIN_CATEGORY_INHERIT", "1")
    chat, _org = _chat(monkeypatch, patient=None)
    assert chat.gate("what does the canary eat") is None
    assert CI.turn_trace(chat)["read"]["reason"] == "below_floor"


def test_lesion_mode_parse(monkeypatch):
    from research.runners.category_inherit_organ import category_inherit_lesion_mode as m
    for v, want in (("", None), ("0", None), ("off", None), ("1", "all"), ("all", "all"), ("on", "all"),
                    ("lateral", "lateral"), ("lateral_learn", "lateral_learn")):
        monkeypatch.setenv("BRAIN_CATEGORY_INHERIT_LESION", v)
        assert m() == want, (v, m())


def test_server_installs_only_behind_the_flag():
    src = open(os.path.join(os.path.dirname(__file__), "..", "webapp", "server.py")).read()
    i = src.find("_ci_mod.install(chat")
    assert i > 0 and "if _category_inherit_on():" in src[i - 400:i]
    import re
    assert len(re.findall(r'(?<![_\w])resp\["category_inherit"\]', src)) == 1      # the rich path
    assert src.count('_resp["category_inherit"]') == 1                                 # the single-fact path


# ── (2) the spiking organ at a dev seed ──────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def taught():
    from research.runners.category_inherit_organ import CategoryInheritOrgan
    orgs = {}
    for name, canary_frame, lesion in (("intact", ("fly", "sky"), None), ("intact2", ("fly", "sky"), None),
                                       ("permuted", ("swim", "lake"), None),
                                       ("lateral_learn", ("fly", "sky"), "lateral_learn")):
        o = CategoryInheritOrgan(seed=DEV_SEED, lesion=lesion)
        for f in C1_FACTS + [("canary",) + canary_frame]:
            assert o.observe(*f)["observed"]
        orgs[name] = o
    return orgs


def test_heldout_canary_inherits_worm(taught):
    r = taught["intact"].infer("canary", "eat")
    assert r["patient"] == "worm", r
    assert dict(r["admitted"])["bug"] == 0.0, r
    assert r["attribution"]["learned_pathways"] == 1.0, r["attribution"]   # every winning spike is the pathways'


def test_permuted_exposure_is_followed(taught):
    assert taught["permuted"].infer("canary", "eat")["patient"] == "bug"


def test_read_lesion_abstains_and_restores(taught):
    o = taught["intact"]
    o.lesion = "all"
    try:
        r = o.infer("canary", "eat")
        assert r["patient"] is None and r["reason"] == "below_floor", r
        assert r["lesion_held"] is True, "docs/TERMS.md: the cut must still hold at the end of the read"
    finally:
        o.lesion = None
    assert o.infer("canary", "eat")["patient"] == "worm", "the lesion must restore the weights after the read"


def test_the_learned_frame_association_is_necessary(taught):
    assert taught["lateral_learn"].infer("canary", "eat")["patient"] is None


def test_moat(taught):
    o = taught["intact"]
    assert o.infer("canary", "swim")["patient"] is None       # never taught, not in the bird frames
    assert o.infer("zebra", "eat")["reason"] == "agent_never_observed"


def test_deterministic_per_seed(taught):
    assert taught["intact"].read("canary") == taught["intact2"].read("canary")


def test_weight_audit_floor_before_selective_after(taught):
    a = taught["intact"].weight_audit()
    assert a["member_frame"]["canary"]["fly sky"] > 10 and a["member_frame"]["canary"]["swim lake"] < 0.1
    assert a["lateral"]["fly sky"]["eat worm"] > 10 and a["lateral"]["fly sky"]["eat bug"] < 0.1


def test_runner_selftest():
    from research.runners import stage1_c1_category as R
    ok, checks = R.selftest(verbose=False)
    assert ok, {k: v for k, v in checks.items() if not v}


def _write_pass_shards(td, mutate=None):
    import json
    from research.runners import stage1_c1_category as R
    for s in R.GATE_SEEDS:
        for arm in R.ARMS:
            d = R._synthetic("pass", arm)
            if mutate is not None:
                mutate(s, arm, d)
            p = R.shard_path(td, s, arm)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                json.dump(d, f)


@pytest.mark.parametrize("where", ["observe", "read", "trace"])
def test_a_swallowed_organ_error_is_undefined_never_a_capability_fail(tmp_path, where):
    """The wiring swallows every organ exception (a turn never crashes), so the exception survives only in the
    `category_inherit` trace. A deterministic error is identical in intact_a and intact_b, so the null stays clean.
    Prereg section 4 counts a seed only with no errored turn and the wiring applied: an organ error in a counted arm
    must read UNDEFINED ("not applied"), never FAIL "capability absent"."""
    from research.runners import stage1_c1_category as R

    def mutate(s, arm, d):
        if not (s == 42 and arm in ("intact_a", "intact_b")):   # a deterministic error: both same-seed builds
            return
        r = d["responses"]
        r["p_canary_eat"] = R._fake("I don't know", abstained=True, trace=True)   # the error's reply-level shape
        if where == "observe":
            r["t_canary_fly"]["category_inherit"]["observe"] = {"error": "MemoryError: "}
        elif where == "read":
            r["p_canary_eat"]["category_inherit"]["read"] = {"agent": "canary", "action": "eat", "patient": None,
                                                            "reason": "error: RuntimeError: boom"}
        else:
            r["p_canary_eat"]["category_inherit"] = {"error": "KeyError: 'x'"}

    _write_pass_shards(str(tmp_path), mutate)
    agg = R.aggregate(str(tmp_path))
    assert agg["verdict"].startswith("UNDEFINED"), (agg["verdict"], agg["why"])
    assert any("not applied" in u for u in agg["rows"][42]["undefined_reasons"]), agg["rows"][42]["undefined_reasons"]
