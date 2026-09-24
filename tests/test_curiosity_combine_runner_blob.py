"""--combine compares the runner FILE at each input's commit, not SHA strings (2026-09-23 re-review)."""
import subprocess

from research.runners._curiosity_metacog_neuromod_gain_derisk import runner_code_mismatch

REL = "research/runners/_curiosity_metacog_neuromod_gain_derisk.py"


def _sha(rev):
    return subprocess.run(["git", "rev-parse", rev], capture_output=True, text=True, check=True).stdout.strip()


def _first_commit_with_runner():
    out = subprocess.run(["git", "log", "--format=%H", "--diff-filter=A", "--", REL], capture_output=True, text=True,
                         check=True).stdout.split()
    return out[-1]


def test_short_and_full_sha_of_one_commit_agree():
    full = _first_commit_with_runner()
    assert runner_code_mismatch({"a.json": full[:9], "b.json": full}) is None


def test_missing_or_unknown_sha_refuses():
    full = _first_commit_with_runner()
    assert runner_code_mismatch({"a.json": full, "b.json": None})
    assert runner_code_mismatch({"a.json": full, "b.json": "unknown"})
    assert runner_code_mismatch({"a.json": "0" * 40})


def test_different_runner_code_refuses():
    first = _first_commit_with_runner()
    head_blob_differs = subprocess.run(["git", "rev-parse", f"{first}:{REL}"], capture_output=True, text=True).stdout != \
        subprocess.run(["git", "rev-parse", f"HEAD:{REL}"], capture_output=True, text=True).stdout
    if head_blob_differs:
        assert runner_code_mismatch({"a.json": first, "b.json": _sha("HEAD")})
