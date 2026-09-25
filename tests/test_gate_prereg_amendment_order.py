"""tests for tools/gates/prereg_amendment_order.py (CLASS PRA).

Imports and exercises the REAL gate module the pre-commit registry calls -- same convention as
tests/test_gate_finding_mechanism_on_main.py (`import tools.gates.X as X_gate`), so these tests cannot drift
from what the hook actually runs. The module's own `selftest()` is the registry's own trust mechanism (a gate
whose selftest does not fail in the failing direction is treated as BROKEN); this file adds pytest-level, real-git
coverage on top of it (both parts of this gate need an ACTUAL staged git index / real commit history to exercise
their real wiring, not just their pure helper functions, which is what `selftest()` itself already checks).
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

import tools.gates.prereg_amendment_order as pra_gate

_ENV = pra_gate._git_env()


def _git(root, *args):
    r = subprocess.run(["git", *args], cwd=root, env=_ENV, capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, "git %s failed: %s" % (" ".join(args), r.stderr)
    return r.stdout


def _write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p


@pytest.fixture()
def repo(tmp_path):
    root = str(tmp_path)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    return root


def test_registry_selftest_passes():
    problems = pra_gate.selftest()
    assert problems == [], "gate selftest reported problems: %r" % problems


def test_registry_discovers_this_gate_with_the_expected_contract():
    from tools.gates import discover
    hits = [t for t in discover() if t[0] == pra_gate.NAME]
    assert len(hits) == 1, "gates/__init__.discover() did not find exactly one %r module" % pra_gate.NAME
    name, mod, err = hits[0]
    assert err is None, "the registry reports this gate as broken: %s" % err
    assert mod.CLASS_ID == "PRA"
    assert mod.BLOCKING is True


# ---------------------------------------------------------------------------------------------------------
# PART (1): mechanical same-commit rule, against a REAL staged git index
# ---------------------------------------------------------------------------------------------------------
PREREG_REL = "research/findings/2026-01-01-x-PREREGISTRATION.md"


def _commit_base_prereg(root):
    _write(root, PREREG_REL, "# prereg\n\nbase text, no amendments yet.\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base prereg")


def test_amendment_and_raw_artifact_in_the_same_commit_is_blocked(repo):
    _commit_base_prereg(repo)
    with open(os.path.join(repo, PREREG_REL), "a", encoding="utf-8") as fh:
        fh.write("\n## AMENDMENT 1 (2026-01-02, before any run) -- new thresholds\n")
    _write(repo, "research/findings/raw/x/s42.json", '{"score": 1}')
    _git(repo, "add", "-A")
    paths = [PREREG_REL, "research/findings/raw/x/s42.json"]
    problems = pra_gate.check(paths, root=repo)
    assert len(problems) == 1
    assert "AMENDMENT 1" in problems[0]
    assert "s42.json" in problems[0]


def test_the_amendment_same_commit_escape_clears_it(repo):
    _commit_base_prereg(repo)
    with open(os.path.join(repo, PREREG_REL), "a", encoding="utf-8") as fh:
        fh.write("\n## AMENDMENT 1 (2026-01-02, before any run) -- new thresholds\n"
                 "- amendment-same-commit: these artifacts are integrity smokes no gate in the prereg reads\n")
    _write(repo, "research/findings/raw/x/s42.json", '{"score": 1}')
    _git(repo, "add", "-A")
    paths = [PREREG_REL, "research/findings/raw/x/s42.json"]
    assert pra_gate.check(paths, root=repo) == []


def test_modifying_a_prereg_with_no_new_amendment_heading_beside_a_raw_artifact_passes(repo):
    _commit_base_prereg(repo)
    with open(os.path.join(repo, PREREG_REL), "a", encoding="utf-8") as fh:
        fh.write("\nsome clarifying prose, not a new amendment heading.\n")
    _write(repo, "research/findings/raw/x/s42.json", '{"score": 1}')
    _git(repo, "add", "-A")
    paths = [PREREG_REL, "research/findings/raw/x/s42.json"]
    assert pra_gate.check(paths, root=repo) == []


def test_a_new_amendment_with_no_raw_artifact_staged_passes(repo):
    _commit_base_prereg(repo)
    with open(os.path.join(repo, PREREG_REL), "a", encoding="utf-8") as fh:
        fh.write("\n## AMENDMENT 1 (2026-01-02) -- new thresholds, no run yet\n")
    _git(repo, "add", "-A")
    assert pra_gate.check([PREREG_REL], root=repo) == []


def test_a_brand_new_added_prereg_with_an_amendment_heading_is_not_this_gates_job(repo):
    """An ADDED (not modified) prereg is gates/prereg_before_run's territory (CLASS PR); this gate only
    fires on a MODIFIED prereg (status M), so a same-commit ADDED prereg+artifact must not double-fire here."""
    _write(repo, PREREG_REL, "# prereg\n\n## AMENDMENT 1 (2026-01-02) -- new thresholds\n")
    _write(repo, "research/findings/raw/x/s42.json", '{"score": 1}')
    _git(repo, "add", "-A")
    assert pra_gate.check([PREREG_REL, "research/findings/raw/x/s42.json"], root=repo) == []


def test_no_staged_paths_is_a_no_op():
    assert pra_gate.check([]) == []
    assert pra_gate.check(None) == []


# ---------------------------------------------------------------------------------------------------------
# PART (2): provenance form -- an artifact's recorded git_sha vs. the amendment's introducing commit
# ---------------------------------------------------------------------------------------------------------
ART_REL = "research/findings/raw/y/s7.json"


def _commit_amendment_citing(root, artifact_rel):
    _write(root, PREREG_REL, "# prereg\n\nbase text.\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base prereg")
    with open(os.path.join(root, PREREG_REL), "a", encoding="utf-8") as fh:
        fh.write("\n## AMENDMENT 1 (2026-01-02, unique-pra-test-marker) -- registered before any run\n\n"
                 "governs `%s`.\n" % artifact_rel)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add amendment 1")
    return _git(root, "rev-parse", "HEAD").strip()


def test_an_artifact_whose_git_sha_predates_the_amendment_it_cites_is_blocked(repo):
    # capture the base commit BEFORE the amendment is added
    _write(repo, PREREG_REL, "# prereg\n\nbase text.\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base prereg")
    base_sha = _git(repo, "rev-parse", "HEAD").strip()
    with open(os.path.join(repo, PREREG_REL), "a", encoding="utf-8") as fh:
        fh.write("\n## AMENDMENT 1 (2026-01-02, unique-pra-test-marker) -- registered before any run\n\n"
                 "governs `%s`.\n" % ART_REL)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add amendment 1")

    _write(repo, ART_REL, '{"score": 1}')
    _write(repo, ART_REL + ".prov.json", json.dumps({"git_sha": base_sha}))
    _git(repo, "add", "-A")
    problems = pra_gate.check([ART_REL], root=repo)
    assert len(problems) == 1
    assert "predates the amendment" in problems[0]
    assert ART_REL in problems[0]


def test_an_artifact_whose_git_sha_postdates_the_amendment_it_cites_passes(repo):
    amend_sha = _commit_amendment_citing(repo, ART_REL)
    _write(repo, "later.txt", "later\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "later commit")
    after_sha = _git(repo, "rev-parse", "HEAD").strip()
    assert after_sha != amend_sha

    _write(repo, ART_REL, '{"score": 1}')
    _write(repo, ART_REL + ".prov.json", json.dumps({"git_sha": after_sha}))
    _git(repo, "add", "-A")
    assert pra_gate.check([ART_REL], root=repo) == []


def test_an_artifact_with_no_prov_sidecar_is_not_blocked(repo):
    _commit_amendment_citing(repo, ART_REL)
    _write(repo, ART_REL, '{"score": 1}')
    _git(repo, "add", "-A")
    assert pra_gate.check([ART_REL], root=repo) == []


def test_an_artifact_no_amendment_cites_is_not_blocked(repo):
    _commit_amendment_citing(repo, ART_REL)
    other_rel = "research/findings/raw/y/never_cited.json"
    _write(repo, other_rel, '{"score": 1}')
    _write(repo, other_rel + ".prov.json", json.dumps({"git_sha": "0" * 40}))
    _git(repo, "add", "-A")
    assert pra_gate.check([other_rel], root=repo) == []
