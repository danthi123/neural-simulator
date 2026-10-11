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
    # RULING c1-stamp-fix: ONE write of the stamp (in the helper), called before EVERY return path.
    import re
    assert src.count('resp["category_inherit"] =') == 1                     # the helper's single write
    assert len(re.findall(r"(?m)^\s*_attach_category_inherit\(payload, chat\)", src)) == 1  # abstain early return
    assert len(re.findall(r"(?m)^\s*_attach_category_inherit\(resp, chat\)", src)) == 1     # the rich path
    assert len(re.findall(r"(?m)^\s*_attach_category_inherit\(_resp, chat\)", src)) == 1    # single-fact path


def test_category_inherit_stamp_both_paths():
    """RULING c1-stamp-fix: the stamp is present exactly when the organ fired this turn (an observe
    or read record exists) and absent otherwise — the invariant that lets a missing stamp be read
    as 'the pathway never fired'. The three server call sites are pinned in the test above."""
    from types import SimpleNamespace

    chat = SimpleNamespace()
    assert CI.turn_stamp(chat) is None                       # nothing fired this turn -> nothing to stamp
    chat._last_category_inherit = {"patient": "worm", "reason": "reached"}
    st = CI.turn_stamp(chat)                                 # the read fired -> the stamp carries the decision
    assert st is not None and (st.get("read") or {}).get("patient") == "worm", st
    chat2 = SimpleNamespace(_ci_last_observe={"observed": True})   # a teach turn fired the observe leg
    assert (CI.turn_stamp(chat2) or {}).get("observe") == {"observed": True}


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


def test_gate_seed_shard_hands_the_gate_flag_to_its_worker(monkeypatch, tmp_path):
    """Pre-data fix 2026-09-30 (pool canary): a pool line runs `--shard --seed 42 --allow-gate-seed --prereg <p>`;
    run_shard spawns `--worker` in a fresh process and main()'s gate-seed guard applies to --worker too, so the
    worker argv must carry the flag, or every one of the 54 gate jobs dies rc=2 before building a brain (measured on
    pool42 at bb6705de6 through the pool sandbox: 'GATE seed 42 refused' -> 'worker FAILED rc=2')."""
    import json
    import sys
    from research.runners import stage1_c1_category as R
    seen = {}

    class _P:
        returncode = 0

    def _fake_worker(seed, arm, out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"responses": {}, "seed": seed}, f)
        return 0

    def _fake_run(cmd, env=None, cwd=None):
        seen["cmd"] = list(cmd)
        assert cmd[1:4] == ["-u", "-m", "research.runners.stage1_c1_category"], cmd
        monkeypatch.setattr(sys, "argv", ["stage1_c1_category"] + list(cmd[4:]))
        with pytest.raises(SystemExit) as e:
            R.main()
        p = _P()
        p.returncode = e.value.code
        seen["worker_rc"] = e.value.code
        return p

    monkeypatch.setattr(R, "run_worker", _fake_worker)
    monkeypatch.setattr(R.subprocess, "run", _fake_run)
    out = str(tmp_path / "s42" / "cat_disc__intact_a.json")
    monkeypatch.setattr(sys, "argv", ["stage1_c1_category", "--shard", "--seed", "42", "--arm", "intact_a",
                                      "--allow-gate-seed", "--prereg", R.PREREG, "--out", out])
    with pytest.raises(SystemExit) as outer:
        R.main()
    assert seen.get("worker_rc") == 0, seen
    assert outer.value.code == 0 and os.path.exists(out)
    assert seen["cmd"][seen["cmd"].index("--prereg") + 1] == R.PREREG
    # a dev seed never needs (or gets) the flag
    monkeypatch.setattr(sys, "argv", ["stage1_c1_category", "--shard", "--seed", "7", "--arm", "intact_a",
                                      "--out", str(tmp_path / "s7.json")])
    with pytest.raises(SystemExit) as dev:
        R.main()
    assert dev.value.code == 0 and "--allow-gate-seed" not in seen["cmd"]


# ── (8) RULING tree-c1-observe (a): a teach turn observes the organ at the STORE WRITE (default OFF) ──────────
_TEACH_VOCAB = ["robin", "wolf", "dog", "bird", "isa", "flies", "sky", "bites", "apple",
                "does", "not", "eat", "grass", "glorp", "makes", "everyone", "wonderful", "happy"]
_TEACH_ACTIONS = {"isa", "flies", "bites", "eat", "does", "makes"}


def _teach_chat(monkeypatch, observe_flag=True):
    """A real ChatBrain, faculty ON, teach-observe flag per argument, NO bus installed — so the teach turn
    exercises the `_maybe_acquire` store-write wire directly (the path where no gate_extract combiner runs)."""
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer, DEFAULT_SELF_ALIASES
    from research.runners.multi_turn_agent import MultiTurnAgent
    monkeypatch.setenv("BRAIN_CATEGORY_INHERIT", "1")
    if observe_flag:
        monkeypatch.setenv("BRAIN_CATEGORY_INHERIT_TEACH_OBSERVE", "1")
    else:
        monkeypatch.delenv("BRAIN_CATEGORY_INHERIT_TEACH_OBSERVE", raising=False)
    referents = [w for w in _TEACH_VOCAB if w not in _TEACH_ACTIONS]
    agent = MultiTurnAgent(referent_concepts=referents,
                           concepts={w: None for w in _TEACH_VOCAB}, seed=42,
                           enable_neural_render=False, composer_kind="rf",
                           enable_biased_competition=False, defer_planner=True, event_register=None)
    chat = ChatBrain(agent, self_aliases=DEFAULT_SELF_ALIASES, renderer=StubRenderer())
    CI.install(chat, seed=7)
    return chat


def test_teach_observe_told_teaches_the_organ(monkeypatch):
    """RULING tree-c1-observe (a): after ONE told teach turn the organ exists and has observed that agent."""
    chat = _teach_chat(monkeypatch)
    svo = chat.gate("the robin isa bird")
    assert svo is not None                        # the store write happened (acquired)
    org = getattr(chat, "_category_inherit_organ", None)
    assert org is not None, "the told teach turn must build the organ via the store-write observe wire"
    assert "robin" in org.member_slot             # ...and has observed THAT agent
    assert (getattr(chat, "_ci_last_observe", None) or {}).get("svo") == ["robin", "isa", "bird"]


def test_teach_observe_lived_teaches_the_organ(monkeypatch):
    """RULING tree-c1-observe (a): after ONE lived teach turn the organ exists and has observed that agent."""
    chat = _teach_chat(monkeypatch)
    svo = chat.gate("the wolf bites the apple")
    assert svo is not None
    org = getattr(chat, "_category_inherit_organ", None)
    assert org is not None and "wolf" in org.member_slot


def test_teach_observe_default_off_is_inert(monkeypatch):
    """Default OFF: the same teach turn stores the fact but never builds an organ — production unchanged."""
    chat = _teach_chat(monkeypatch, observe_flag=False)
    assert chat.gate("the robin isa bird") is not None
    assert getattr(chat, "_category_inherit_organ", None) is None
    assert getattr(chat, "_ci_last_observe", None) is None


def test_teach_observe_exactly_once_when_gate_extract_runs(monkeypatch):
    """The teach wire and `_ci_gate_extract` must not double-observe one turn: exactly once."""
    chat = _teach_chat(monkeypatch)
    res = chat.gate_extract("the robin isa bird")   # orig_extract runs _maybe_acquire -> the wire fires first
    assert res[0] == "done"
    org = getattr(chat, "_category_inherit_organ", None)
    assert org is not None and org.n_observed == 1  # the wire observed; the wrapper's new guard then skipped


def test_teach_observe_skips_negation(monkeypatch):
    """AFFIRM-only: a NEGATE store never teaches the organ (the organ's declared input class)."""
    chat = _teach_chat(monkeypatch)
    chat._ci_teach_observe("dog", "eat", "grass", "NEGATE")
    assert getattr(chat, "_category_inherit_organ", None) is None
    assert getattr(chat, "_ci_last_observe", None) is None


def test_val_novel_exposure_writes_no_store(monkeypatch):
    """RULING tree-c1-observe (a) RESIDUAL — reported to Claude (NEEDS), not silently passed: the val_novel
    expose sentence is 5 content words, the B3 extractor declines it, so NO store write exists to hook and no
    store-write observe wire can build the organ from an exposure (0/8 flipdev cells ever observed; kb 5 -> 5
    across 12 exposures). The exposure->observe site is a design question for Claude."""
    chat = _teach_chat(monkeypatch)
    svo = chat.gate("the glorp makes everyone wonderful and happy")
    assert svo is None                            # declined: never acquired -> no store write happens
    assert getattr(chat, "_category_inherit_organ", None) is None
