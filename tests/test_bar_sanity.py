"""tools/bar_sanity.py (the library + CLI) and tools/gates/bar_sanity_line.py (CLASS BAR), 2026-10-06.

Layers: the library on toy specs (UNREACHABLE / TOOTHLESS / OK / INVALID / override rules / reproducibility); the gate's
selftest and its judging functions; the gate in a scratch repo through `check()`; the two real demonstrations
(Stage-1 C4 old and constrained control) when the scorer they import is present, and the committed records."""
from __future__ import annotations

import json
import os
import subprocess
import textwrap

import pytest

import tools.bar_sanity as bs
import tools.gates as gates
import tools.gates.bar_sanity_line as gate

_REPO = bs._ROOT


def _spec(tmp_path, body, name="spec.py"):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


_TOY = textwrap.dedent("""
    import numpy as np
    def decide(x):
        return x > 0.5           # a bar: the draw must exceed 0.5
    def perfect(rng):
        return 0.6 + 0.3 * rng.random()
    def null(rng):
        return 0.4 * rng.random()
""")


def test_ok_toy_spec(tmp_path):
    rec = bs.run(_spec(tmp_path, _TOY), n=200, seed=1, root=str(tmp_path))
    assert rec["verdict"] == "OK" and rec["pass_rate"] == {"null": 0.0, "perfect": 1.0}
    assert rec["thresholds"]["perfect_min"] == 0.95 and rec["thresholds"]["null_max"] == 0.05
    assert rec["spec"] == "spec.py" and len(rec["spec_sha256"]) == 64 and len(rec["scorer_sha256"]) == 64


def test_unreachable_when_a_perfect_brain_fails(tmp_path):
    spec = _spec(tmp_path, _TOY.replace("0.6 + 0.3", "0.3 + 0.3"))     # perfect lands above 0.5 half the time
    rec = bs.run(spec, n=400, seed=1, root=str(tmp_path))
    assert rec["verdict"] == "UNREACHABLE" and 0.2 < rec["pass_rate"]["perfect"] < 0.8
    assert any("UNREACHABLE: perfect" in f for f in rec["failures"])


def test_toothless_when_a_null_brain_passes(tmp_path):
    rec = bs.run(_spec(tmp_path, _TOY.replace("0.4 * rng", "0.9 * rng")), n=400, seed=1, root=str(tmp_path))
    assert rec["verdict"] == "TOOTHLESS" and rec["pass_rate"]["null"] > 0.05


def test_unreachable_wins_when_both_fail(tmp_path):
    spec = _spec(tmp_path, _TOY.replace("0.6 + 0.3", "0.3 + 0.3").replace("0.4 * rng", "0.9 * rng"))
    rec = bs.run(spec, n=300, seed=1, root=str(tmp_path))
    assert rec["verdict"] == "UNREACHABLE" and len(rec["failures"]) == 2


def test_extra_generators_are_thresholded_by_kind_and_info_is_not(tmp_path):
    spec = _spec(tmp_path, _TOY + "\ndef perfect_hard(rng):\n    return 0.3 + 0.3 * rng.random()\n"
                 "def info_weak(rng):\n    return 0.9 * rng.random()\n")
    rec = bs.run(spec, n=300, seed=1, root=str(tmp_path))
    assert rec["kinds"]["perfect_hard"] == "perfect" and rec["kinds"]["info_weak"] == "info"
    assert rec["verdict"] == "UNREACHABLE" and not any("info_weak" in f for f in rec["failures"])
    spec2 = _spec(tmp_path, _TOY + "\ndef info_weak(rng):\n    return 0.9 * rng.random()\n", "s2.py")
    assert bs.run(spec2, n=300, seed=1, root=str(tmp_path))["verdict"] == "OK"     # info never moves the verdict


def test_a_spec_needs_both_a_perfect_and_a_null_brain(tmp_path):
    rec = bs.run(_spec(tmp_path, "def decide(x):\n    return True\ndef perfect(rng):\n    return 1\n"), n=10, root=str(tmp_path))
    assert rec["verdict"] == "INVALID"


def test_override_needs_a_reason(tmp_path):
    with pytest.raises(ValueError, match="reason"):
        bs.run(_spec(tmp_path, _TOY + "\nOVERRIDES = {'perfect_min': 0.8}\n"), n=10, root=str(tmp_path))
    with pytest.raises(ValueError, match="reason"):
        bs.run(_spec(tmp_path, _TOY + "\nOVERRIDES = {'null_max': 0.2, 'reason': 'too short'}\n", "s2.py"),
               n=10, root=str(tmp_path))
    ok = bs.run(_spec(tmp_path, _TOY + "\nOVERRIDES = {'perfect_min': 0.8, 'reason': 'a sanity floor; the claim is made elsewhere'}\n",
                      "s3.py"), n=50, root=str(tmp_path))
    assert ok["thresholds"]["perfect_min"] == 0.8 and ok["thresholds"]["override_reason"]
    tighter = bs.run(_spec(tmp_path, _TOY + "\nOVERRIDES = {'perfect_min': 0.99}\n", "s4.py"), n=50, root=str(tmp_path))
    assert tighter["thresholds"]["override_reason"] is None


def test_records_reproduce_exactly_from_the_seed(tmp_path):
    spec = _spec(tmp_path, _TOY)
    a, b = bs.run(spec, n=100, seed=7, root=str(tmp_path)), bs.run(spec, n=100, seed=7, root=str(tmp_path))
    for k in ("pass_rate", "n_pass", "spec_sha256", "scorer_sha256"):
        assert a[k] == b[k]


def test_a_crashing_decide_is_an_error_not_a_fail(tmp_path):
    with pytest.raises(RuntimeError, match="decide"):
        bs.run(_spec(tmp_path, "def decide(x):\n    raise KeyError('x')\ndef perfect(r):\n    return 1\ndef null(r):\n    return 0\n"),
               n=5, root=str(tmp_path))


def test_cli_exit_code_and_record(tmp_path, capsys):
    out = tmp_path / "rec.json"
    assert bs.main([_spec(tmp_path, _TOY), "--n", "50", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"] == "OK"
    assert bs.main([_spec(tmp_path, _TOY.replace("0.4 * rng", "0.9 * rng"), "t.py"), "--n", "50"]) == 1


def test_wilson_interval():
    lo, hi = bs.wilson(943, 1000)
    assert 0.925 < lo < 0.93 and 0.955 < hi < 0.96 and bs.wilson(0, 10)[0] == 0.0


# ---------------------------------------------------------------- the gate
def test_gate_selftest_is_clean_and_the_registry_runs_it():
    assert gate.selftest() == []
    names = [n for n, mod, err in gates.discover() if err is None]
    assert "bar_sanity_line" in names


def test_gate_matches_the_files_it_should():
    for n in ("2026-09-29-x-PREREG.md", "STAGE-PREREGISTRATION.md", "AMENDMENT-A1.md", "x-AMENDMENT-2.md"):
        assert gate._TARGET_RE.match(n), n
    for n in ("RESULT.md", "notes-on-amendments.md", "prereg.txt", "HANDOFF.md"):
        assert not gate._TARGET_RE.match(n), n


def _repo(tmp_path):
    td = str(tmp_path)

    def g(*a):
        r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"] + list(a), cwd=td,
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        return r.stdout

    g("init", "-q", "-b", "main")
    g("config", "user.email", "t@example.invalid")
    g("config", "user.name", "t")
    return td, g


def test_gate_in_a_scratch_repo_grandfathers_old_files_and_blocks_new_and_modified(tmp_path):
    td, g = _repo(tmp_path)
    (tmp_path / "A-PREREG.md").write_text("old prereg\n")
    g("add", "-A")
    g("commit", "-q", "-m", "base")
    assert gate.check([], root=td, invocation="commit") == []
    (tmp_path / "A-PREREG.md").write_text("old prereg\n\n## AMENDMENT 1\nbar r < .2\n")
    g("add", "-A")
    assert any("A-PREREG.md" in p for p in gate.check([], root=td, invocation="commit"))
    (tmp_path / "A-PREREG.md").write_text("old prereg\n\nbar-sanity: N/A this amendment records a recount, no bar\n")
    g("add", "-A")
    assert gate.check([], root=td, invocation="commit") == []
    assert gate.check([], root=td, invocation="merge") == []            # a clean auto-merge is not checked


def test_gate_refuses_a_stale_record_and_accepts_a_fresh_one(tmp_path):
    td, g = _repo(tmp_path)
    (tmp_path / "scorer.py").write_text("x = 1\n")
    (tmp_path / "spec.py").write_text(_TOY)
    g("add", "-A")
    g("commit", "-q", "-m", "base")
    rec = bs.run(str(tmp_path / "spec.py"), n=200, seed=1, root=td)
    rec["scorer_files"] = {"scorer.py": bs.sha256_file(str(tmp_path / "scorer.py"))}
    (tmp_path / "rec.json").write_text(json.dumps(rec))
    (tmp_path / "X-AMENDMENT-1.md").write_text("bar-sanity: rec.json\n")
    g("add", "-A")
    assert gate.check([], root=td, invocation="commit") == []
    (tmp_path / "scorer.py").write_text("x = 2\n")                       # the scorer moves after the measurement
    g("add", "-A")
    assert any("STALE" in p for p in gate.check([], root=td, invocation="commit"))


# ------------------------------------------------- the two real demonstrations (need the scorer they import)
_C4 = os.path.join(_REPO, "docs", "drafts", "stage1-exit")
_T1 = os.path.join(_REPO, "docs", "drafts", "long-delay-credit")


def _c4():
    pytest.importorskip("research.runners.stage1_exit_gate")
    sys_path = __import__("sys").path
    if _C4 not in sys_path:
        sys_path.insert(0, _C4)
    import bar_sanity_c4_common as C
    return C


def _pass_rate(C, draw, n, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    return sum(1 for _ in range(n) if C.decide(C.make_data(rng, draw))) / n


def test_c4_old_control_is_unreachable_for_a_perfect_brain():
    C = _c4()
    rate = _pass_rate(C, C.derangement_old, 120)
    assert 0.25 < rate < 0.65, rate           # the review's estimate is 0.44: 0.87**6 = 0.43


def test_c4_constrained_control_is_far_more_reachable():
    C = _c4()
    if not hasattr(C.G, "_scrambled_valence_draw"):
        pytest.skip("research/runners/stage1_exit_gate.py predates RULING s1-c4-control (bc8f804b9)")
    assert _pass_rate(C, C.derangement_constrained, 120) > 0.85


def test_c4_scorer_still_fails_a_brain_that_ignores_the_control():
    import numpy as np
    C = _c4()
    rng = np.random.default_rng(0)
    assert not any(C.decide(C.make_data(rng, C.derangement_old, control="ignores")) for _ in range(10))
    assert not any(C.decide(C.make_data(rng, C.derangement_old, lesion="inert")) for _ in range(10))


def test_t1_amended_bar_is_reachable_and_the_registered_one_is_not():
    if not os.path.isfile(os.path.join(_REPO, "research", "runners", "_ldc_omission_pilot.py")):
        pytest.skip("the long-delay-credit runner lives on research/draft-ldc-outcome-timing, not on main")
    import sys
    if _T1 not in sys.path:
        sys.path.insert(0, _T1)
    import numpy as np
    import bar_sanity_t1_common as T
    rng = np.random.default_rng(0)
    assert all(T.decide_reward_window(T.make_run(rng, "silent_in_window", T.TIMING_4A)) for _ in range(30))
    assert not any(T.decide_reward_window(T.make_run(rng, "none", T.TIMING_4A)) for _ in range(30))
    assert not any(T.decide_whole_window(T.make_run(rng, "silent_in_window", T.TIMING_4A)) for _ in range(30))
    assert not any(T.decide_whole_window(T.make_run(rng, "silent_from_onset", T.TIMING_4A)) for _ in range(30))


def test_committed_records_say_what_they_demonstrate():
    want = {"bar_sanity_record_c4_old_control.json": ("UNREACHABLE", 0.3, 0.6),
            "bar_sanity_record_c4_constrained_0p10.json": ("UNREACHABLE", 0.90, 0.95),
            "bar_sanity_record_c4_constrained_0p05.json": ("OK", 0.95, 1.0)}
    for name, (verdict, lo, hi) in want.items():
        full = os.path.join(_C4, name)
        assert os.path.exists(full), name
        rec = json.load(open(full))
        assert gate._verdict(rec) == rec["verdict"] == verdict, name
        assert lo <= rec["pass_rate"]["perfect"] <= hi, name
        assert rec["pass_rate"]["null_control_ignored"] == 0.0 and rec["backend"] == "numpy"
