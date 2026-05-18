"""Q2R --tiny smoke + structural guards. Tiny verdict NOT propagated."""
import json, subprocess, sys, os, importlib, re


def test_kb_is_96plus_genuinely_distinct():
    g = importlib.import_module("research.runners.q2r_gate")
    kb = g._Q2R_GROUNDED
    assert len(kb) >= 96, "KB must have >=96 props, got %d" % len(kb)

    def content(s):
        fw = importlib.import_module(
            "research.runners.generator_g_core").FUNCTION_WORDS
        toks = [re.sub(r"[^\w]", "", w.lower()) for w in str(s).split()]
        return frozenset(t for t in toks if t and t not in fw)
    sets = [content(v) for v in kb.values()]
    assert all(len(s) >= 3 for s in sets), "a prop has <3 content words"
    assert len(set(sets)) == len(sets), "duplicate content-word sets "\
        "(templated/padded KB)"


def test_byte_unmodified_imports():
    g = importlib.import_module("research.runners.q2r_gate")
    from research.runners.constrained_decode_gate import _GroundedConstrainedLM
    from research.runners.constrained_decode_core import cdc_verdict
    assert g._GroundedConstrainedLM is _GroundedConstrainedLM
    assert g.cdc_verdict is cdc_verdict


def test_no_new_training_in_net_new():
    import inspect
    g = importlib.import_module("research.runners.q2r_gate")
    src = inspect.getsource(g)
    assert "backward(" not in src and ".step()" not in src
    assert "optimizer" not in src.lower() and "loss" not in src.lower()


def test_tiny_smoke_runs(tmp_path):
    out = tmp_path / "q2r_smoke.json"
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.q2r_gate",
         "--tiny", "--seeds", "42", "43", "44", "--out", str(out)],
        capture_output=True, text=True, timeout=3600, env={**os.environ})
    assert r.returncode == 0, r.stderr[-3000:]
    d = json.loads(out.read_text())
    assert d["tiny"] is True and d["note"].startswith("TINY")
    assert len(d["ladder"]) == 1
    ps = d["ladder"][0]["verdict"]["per_seed"]
    assert all(v["abstain_on_ungrounded_rate"]
               >= v["bare_moat_abstain_rate"] - 1e-9 for v in ps.values())
