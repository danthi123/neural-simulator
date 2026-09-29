"""pytest.ini's `testpaths = tests` (2026-09-29 re-review, M1): without it, a bare `pytest` run from the repo
root walks the WHOLE tree looking for conftest.py files to import -- pytest's own collection semantics load a
conftest.py based on directory structure alone, regardless of whether that directory holds any test files. An
agent-authored GPU/pool job can legitimately write into its own writable output subdirectory,
research/findings/raw/_agent_jobs/<job-id>/ -- a conftest.py planted there would be imported (arbitrary code
execution) the next time anyone runs a bare `pytest` from the repo root.

Hermetic: builds its own SCRATCH project (a temp dir with its own tests/, its own research/findings/raw/
_agent_jobs/<id>/conftest.py, and its own pytest.ini) -- never touches or plants anything in the real repo.
"""
import os
import subprocess
import sys
import tempfile


def _scaffold(tmp, with_testpaths):
    os.makedirs(os.path.join(tmp, "tests"), exist_ok=True)
    with open(os.path.join(tmp, "tests", "test_x.py"), "w") as f:
        f.write("def test_x():\n    assert True\n")

    job_dir = os.path.join(tmp, "research", "findings", "raw", "_agent_jobs", "evil-job")
    os.makedirs(job_dir, exist_ok=True)
    marker = os.path.join(tmp, "IMPORTED_MARKER")
    with open(os.path.join(job_dir, "conftest.py"), "w") as f:
        f.write("open(%r, 'w').write('pwned')\n" % marker)

    ini_text = "[pytest]\ntestpaths = tests\n" if with_testpaths else "[pytest]\n"
    with open(os.path.join(tmp, "pytest.ini"), "w") as f:
        f.write(ini_text)
    return marker


def _run_bare_pytest(tmp):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=tmp, capture_output=True, text=True, timeout=60,
    )


def test_testpaths_prevents_a_planted_conftest_under_research_from_being_imported():
    with tempfile.TemporaryDirectory() as tmp:
        marker = _scaffold(tmp, with_testpaths=True)
        res = _run_bare_pytest(tmp)
        assert res.returncode == 0, res.stdout + res.stderr
        assert not os.path.exists(marker), (
            "a conftest.py planted under research/findings/raw/_agent_jobs/ was imported by a bare pytest run "
            "-- testpaths=tests failed to confine collection")


def test_without_testpaths_the_planted_conftest_would_have_been_imported():
    """Failing-first proof: WITHOUT testpaths, the same layout DOES get the conftest imported -- confirming the
    vulnerability this fix closes was real, not hypothetical."""
    with tempfile.TemporaryDirectory() as tmp:
        marker = _scaffold(tmp, with_testpaths=False)
        res = _run_bare_pytest(tmp)
        assert res.returncode == 0, res.stdout + res.stderr
        assert os.path.exists(marker), (
            "expected the OLD (no testpaths) behaviour to import the planted conftest.py -- if this now fails, "
            "either pytest's own default behaviour changed or this test's assumption needs revisiting")


def test_real_repo_pytest_ini_declares_testpaths_tests():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ini = os.path.join(root, "pytest.ini")
    assert os.path.exists(ini), "pytest.ini must exist at the repo root"
    text = open(ini).read()
    assert "testpaths" in text and "tests" in text
