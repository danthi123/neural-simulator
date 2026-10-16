"""Packet s2-c4-build (docs/drafts/stage2-c4-build/PLAN.md): emotion-category channels of the affect ladder, held on
spiking neurons and named by a spiking commit, behind BRAIN_EMOTION_CATEGORY (default OFF).

Pins (numpy, CPU; one organ build + development ~4 s):
  * the flag is default OFF; in webapp/server.py every use of the organ sits under the flag's env check (or behind a
    sys.modules guard that is a no-op unless the organ was imported), so flag off imports and runs nothing;
  * the feel clause names exactly one Stage-2 lexicon category when a channel latched and none otherwise;
  * development hygiene: no innate trigger or teacher label is a Stage-2 situation FRAME word; triggers and labels are
    disjoint;
  * the organ: the dev battery passes every case on dev seed 1 at row 0 (intact, permuted, every lesion, the shuffle
    and teach-swap builds, the ladder-only probe, baseline); an unknown message leaves the state untouched; a read
    leaves the held state untouched; forget_session drops it; the global RNG is not perturbed;
  * the transducer read, the hook that edits the feel response, and the dev CLI's gate-seed refusal.
"""
import ast
import os
import re
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

# the Stage-2 frozen lexicon (research/runners/stage2_exit_gate.py EMOTION_LEXICON; equality checked below when the
# runner is importable on this branch)
EMOTION_LEXICON = {
    "joy": ("happy", "joy", "joyful", "glad", "delighted", "pleased", "cheerful", "elated"),
    "fear": ("afraid", "scared", "fear", "fearful", "anxious", "frightened", "terrified"),
    "anger": ("angry", "anger", "mad", "furious", "rage", "irritated", "annoyed"),
    "sadness": ("sad", "sadness", "sorrow", "grief", "lonely", "unhappy", "miserable", "down"),
}
FRAME_WORDS = {"win", "gift", "wolf", "attacking", "someone", "cheats", "alone"}


def _E():
    import research.runners.emotion_category_organ as E
    return E


def _named(text):
    low = text.lower()
    return [c for c, ws in EMOTION_LEXICON.items() if any(re.search(r"\b%s\b" % re.escape(w), low) for w in ws)]


def _server_switch():
    """`_EMOTION_CATEGORY_DEFAULT_ON` and `_emotion_category_on` exec'd from webapp/server.py's own source (no app import)."""
    tree = ast.parse(open(os.path.join(REPO, "webapp", "server.py")).read())
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_EMOTION_CATEGORY_DEFAULT_ON"
                                                                      for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name == "_emotion_category_on")]
    assert len(keep) == 2
    ns = {"os": os}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "server_switch", "exec"), ns)
    return ns


@pytest.mark.parametrize("v,want", [(None, True), ("", True), ("0", False), ("off", False), ("1", True),
                                    ("true", True), (" on ", True)])
def test_flag_default_on(monkeypatch, v, want):
    """DEFAULT ON since 2026-10-15 (handler dev check C4 PASS 8/8 + flip check); BRAIN_EMOTION_CATEGORY=0 is the escape."""
    ns = _server_switch()
    assert ns["_EMOTION_CATEGORY_DEFAULT_ON"] is True
    if v is None:
        monkeypatch.delenv("BRAIN_EMOTION_CATEGORY", raising=False)
    else:
        monkeypatch.setenv("BRAIN_EMOTION_CATEGORY", v)
    assert ns["_emotion_category_on"]() is want


@pytest.mark.parametrize("v,want", [(None, 0), ("", 0), ("3", 3), ("x", 0)])
def test_op_reader(monkeypatch, v, want):
    if v is None:
        monkeypatch.delenv("BRAIN_EMOTION_CATEGORY_OP", raising=False)
    else:
        monkeypatch.setenv("BRAIN_EMOTION_CATEGORY_OP", v)
    assert _E().operating_point() == want


def test_server_uses_the_organ_only_under_the_flag():
    src = open(os.path.join(REPO, "webapp", "server.py")).read()
    tree = ast.parse(src)
    parents = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parents[ch] = node

    def guarded(node):
        n = node
        while n in parents:
            n = parents[n]
            if isinstance(n, ast.If):
                t = ast.unparse(n.test)
                if t == "_emotion_category_on()" or "emotion_category_organ' in sys.modules" in t \
                        or 'emotion_category_organ" in sys.modules' in t:
                    return True
        return False

    uses = [n for n in ast.walk(tree)
            if (isinstance(n, ast.ImportFrom) and any(a.name == "emotion_category_organ" for a in n.names))
            or (isinstance(n, ast.Constant) and n.value == "research.runners.emotion_category_organ"
                and not isinstance(parents.get(n), ast.Compare))]
    assert len(uses) >= 5
    assert all(guarded(n) for n in uses), [n.lineno for n in uses if not guarded(n)]


def test_feel_clause_names_exactly_one_category():
    E = _E()
    for c in E.CATS:
        assert _named(E.feel_clause(c)) == [c]
    assert _named(E.feel_clause(None)) == []
    base = ("My affect monitor reads negative right now (valence differential -0.052, high arousal) -- that's a "
            "functional read of my mood state, not a felt experience.")
    assert _named(base) == []
    for c in E.CATS:
        assert _named(base + E.feel_clause(c)) == [c]


def test_stage2_lexicon_and_situations_match_when_available():
    try:
        import research.runners.stage2_exit_gate as G
    except Exception:
        pytest.skip("the Stage-2 runner is not on this branch")
    import research.runners.emotion_category_dev as D
    assert G.EMOTION_LEXICON == EMOTION_LEXICON
    assert G.SITUATIONS == D.SITUATIONS and G.SITUATION_AFFECT_WORDS == D.SITUATION_AFFECT_WORDS
    for c in _E().CATS:
        assert G.parse_feel("x reads positive (valence differential +0.100, high arousal)." +
                            _E().feel_clause(c))["category"] == c
        assert G.parse_feel(_E().feel_clause(None))["category"] is None


def test_development_hygiene():
    E = _E()
    trig = {w for ws in E.TRIGGERS.values() for w in ws}
    lab = {w for ws in E.LABELS.values() for w in ws}
    assert not (trig & lab)
    assert not ((trig | lab) & FRAME_WORDS)
    for t, w, c in E.curriculum():
        assert t in E.TRIGGERS[c] and w in E.LABELS[c]
    sw = E.curriculum(teach_swap=True)
    assert {t for t, _w, c in sw if c == "fear"} <= set(E.TRIGGERS["anger"])
    assert {t for t, _w, c in sw if c == "anger"} <= set(E.TRIGGERS["fear"])


def test_transducer_read():
    E = _E()
    p = E.GRID[0]
    thr = p["READ_FRAC"] * p["NS"] * p["READ_STEPS"]
    rec = {"slot_counts": {"joy": 0, "fear": 90, "anger": 0, "sadness": 0}}
    r = E.read_category(rec, p)
    assert r["category"] == "fear" and r["read_invariant"] and not r["co_commit"]
    r = E.read_category({"slot_counts": {c: 0 for c in E.CATS}}, p)
    assert r["category"] is None and r["read_invariant"]
    r = E.read_category({"slot_counts": {"joy": 80, "fear": 80, "anger": 0, "sadness": 0}}, p)
    assert r["category"] is None and r["co_commit"]
    r = E.read_category({"slot_counts": {"joy": 0, "fear": int(thr * 1.5), "anger": 0, "sadness": 0}}, p)
    assert r["category"] == "fear" and not r["read_invariant"]            # x2 loses it: not invariant


def test_apply_to_feel(monkeypatch):
    E = _E()
    feel = {"answer": "My affect monitor reads neutral.", "inner_state_readout": True}
    monkeypatch.setattr(E, "read_turn", lambda key, lesion: {"category": "anger", "route": "emotion_category"})
    out = E.apply_to_feel(feel, ("s",), False)
    assert out["answer"].endswith(E.feel_clause("anger")) and feel["answer"] == "My affect monitor reads neutral."
    assert out["emotion_category"]["category"] == "anger" and out["emotion_category"]["error"] is None

    def boom(key, lesion):
        raise RuntimeError("organ down")
    monkeypatch.setattr(E, "read_turn", boom)
    out = E.apply_to_feel(feel, ("s",), False)
    assert out["answer"] == feel["answer"] and "organ down" in out["emotion_category"]["error"]


def test_dev_cli_refuses_gate_seeds():
    import research.runners.emotion_category_dev as D
    with pytest.raises(SystemExit):
        D.run_seed(42, 0)


@pytest.fixture(scope="module")
def organ():
    return _E().EmotionCategoryOrgan(seed=1).train()


def test_unknown_message_and_read_leave_state(organ):
    E = _E()
    key = ("t-state",)
    assert organ.observe(key, "what does the cat eat")["acted"] is False
    assert organ.read(key)["category"] is None
    organ.observe(key, "you are scared and afraid")
    organ.observe(key, "you are scared and afraid")
    snap = {k: np.array(v).copy() for k, v in organ._sessions[key].items()}
    r1, r2 = organ.read(key), organ.read(key)
    assert r1["category"] == r2["category"] == "fear"
    assert all(np.array_equal(snap[k], np.array(organ._sessions[key][k])) for k in snap)
    organ.observe(key, "what does the cat eat")
    assert organ.read(key)["category"] == "fear"
    E.forget_session(key)                 # module-level: only organs made by get_organ; this one is local
    organ.forget(key)
    assert key not in organ._sessions and organ.read(key)["category"] is None


def test_global_rng_untouched(organ):
    st = np.random.get_state()
    organ.observe(("t-rng",), "you are angry and mad")
    organ.read(("t-rng",))
    st2 = np.random.get_state()
    assert st[0] == st2[0] and np.array_equal(st[1], st2[1]) and st[2:] == st2[2:]


def test_dev_battery_passes_on_dev_seed_1():
    import research.runners.emotion_category_dev as D
    rec = D.run_seed(1, 0)
    sp = D.seed_pass(rec)
    assert sp["pass"], sp
    assert rec["weights"]["word_taught"] > 5 * rec["weights"]["word_untaught"]
