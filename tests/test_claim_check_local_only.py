"""tools/claim_check.py local_only(): a cited run record that is git-ignored and untracked exists in one checkout
only, so a finding citing it must fail the check (2026-10-06: the ca3 `parts/` records blocked every branch sync)."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import claim_check as cc  # noqa: E402


def _repo(tmp_path):
    r = tmp_path / "repo"
    (r / "research/findings/raw/_x/parts").mkdir(parents=True)
    (r / "research/findings/raw/_x/kept").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    (r / ".gitignore").write_text("parts/\n")
    (r / "research/findings/raw/_x/parts/s1.json").write_text('{"v": 0.123456}')
    (r / "research/findings/raw/_x/kept/s1.json").write_text('{"v": 0.654321}')
    doc = r / "research/findings/2026-10-06-x.md"
    doc.write_text("value 0.123456 from `research/findings/raw/_x/parts/s1.json`\n"
                   "value 0.654321 from `research/findings/raw/_x/kept/s1.json`\n")
    return r, str(doc)


def test_ignored_untracked_record_fails_and_force_added_passes(tmp_path, monkeypatch):
    r, doc = _repo(tmp_path)
    monkeypatch.setattr(cc, "ROOT", str(r))
    ign = "research/findings/raw/_x/parts/s1.json"
    assert cc.local_only([str(r / ign), str(r / "research/findings/raw/_x/kept/s1.json")]) == [ign]
    assert cc.check(doc, verbose=False) == 1
    subprocess.run(["git", "-C", str(r), "add", "-f", ign], check=True)    # staged = travels with the finding
    assert cc.local_only([str(r / ign)]) == []
    assert cc.check(doc, verbose=False) == 0


def test_outside_a_git_checkout_reports_nothing(tmp_path, monkeypatch):
    d = tmp_path / "plain/research/findings/raw/_x/parts"
    d.mkdir(parents=True)
    (d / "s1.json").write_text("{}")
    monkeypatch.setattr(cc, "ROOT", str(tmp_path / "plain"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert cc.local_only([str(d / "s1.json")]) == []
