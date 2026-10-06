"""tools/claude_brief.py on a fixture repo and runbook: open NEEDS lines with age, branch (named / inferred from the key
/ inferred from a cited sha), ahead/behind, diffstat, worktree dirtiness, fast-forward from the base, merge conflicts and
cited paths. Read-only."""
import datetime
import importlib.util
import os
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("claude_brief", os.path.join(_ROOT, "tools", "claude_brief.py"))
CB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CB)

NOW = datetime.datetime(2026, 10, 6, 12, 0)


def g(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                          capture_output=True, text=True, check=True).stdout.strip()


def commit(repo, name, text, msg):
    p = repo / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    g(repo, "add", "-A")
    g(repo, "commit", "-m", msg)
    return g(repo, "rev-parse", "--short", "HEAD")


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    g(repo, "init", "-b", "main")
    commit(repo, "base.txt", "base\n", "init")
    commit(repo, "docs/exists.md", "x\n", "docs")
    # research/alpha: two commits, a worktree that is clean
    g(repo, "worktree", "add", "-b", "research/alpha", str(tmp_path / "wt-alpha"))
    wa = tmp_path / "wt-alpha"
    commit(wa, "tools/a.py", "a = 1\n" * 5, "alpha 1")
    sha_alpha = commit(wa, "tools/b.py", "b = 1\n", "alpha 2")
    # claude-review/alpha on top of research/alpha, DIRTY worktree
    g(repo, "worktree", "add", "-b", "claude-review/alpha", str(tmp_path / "wt-cr-alpha"), "research/alpha")
    wc = tmp_path / "wt-cr-alpha"
    commit(wc, "tools/c.py", "c\n", "review")
    (wc / "stray.txt").write_text("leftover\n")
    # main moves on after research/beta forks, and both touch conflict.txt
    g(repo, "branch", "research/beta")
    commit(repo, "docs/main_only.md", "m\n", "main only doc")
    commit(repo, "conflict.txt", "main side\n", "main conflict")
    g(repo, "worktree", "add", str(tmp_path / "wt-beta"), "research/beta")
    commit(tmp_path / "wt-beta", "conflict.txt", "beta side\n", "beta conflict")
    # research/gamma: no worktree, only a cited sha
    g(repo, "branch", "research/gamma", "main")
    g(repo, "update-ref", "refs/heads/research/gamma", g(repo, "rev-parse", "main"))
    return repo, sha_alpha


RUNBOOK = """# rb
## 6. RULINGS

- alpha-rule: [alpha] something. (2026-10-05)

## NEEDS CLAUDE

Format (one line, at most 300 characters, newest last):

- 2026-10-06 00:00 [open-items] INFO (seeded): other judgment items stay in each HANDOFF
- 2026-10-05 12:00 [alpha] RESULT: built; merge claude-review/alpha; see docs/exists.md and docs/main_only.md and docs/gone.md and tools/c.py
- 2026-10-06 06:30 [beta] ASK #2: please confirm; evidence tools/zzz.py
- 2026-10-06 11:00 [mystery] STOP: cannot find the data, tried; blocks T9
- 2026-10-06 11:30 [no-branch] RESULT: done at SHA_PLACEHOLDER on research/alpha
"""


def test_parse_needs_fields_and_age(tmp_path):
    rows = CB.parse_needs(RUNBOOK)
    assert [r["key"] for r in rows] == ["open-items", "alpha", "beta", "mystery", "no-branch"]
    assert rows[2]["kind"] == "ASK" and rows[2]["n"] == "#2" and rows[0]["n"] == "(seeded)"
    assert CB.parse_needs("no section here") == []


def test_brief_names_branch_ff_dirty_conflict_and_missing_paths(tmp_path):
    repo, sha = make_repo(tmp_path)
    out = CB.brief(RUNBOOK.replace("SHA_PLACEHOLDER", sha), str(repo), NOW)
    alpha = out.split("[2] alpha")[1].split("[3] beta")[0]
    assert "age 24.0 h" in alpha and "branch: claude-review/alpha (named)" in alpha
    assert "ahead 3 / behind 2 main" in alpha                                  # alpha 1, alpha 2, review; main moved on
    assert "fast-forward from base research/alpha: yes" in alpha
    assert "is DIRTY (1 path(s))" in alpha and "stray.txt" in alpha
    assert "cites docs/exists.md: on claude-review/alpha" in alpha and "cites docs/main_only.md: on main" in alpha and "cites docs/gone.md: MISSING" in alpha
    assert "cites tools/c.py: on claude-review/alpha" in alpha
    assert "diffstat vs main: 3 file(s), +7 -0" in alpha and "+5" in alpha and "tools/a.py" in alpha
    assert "section-6 rulings naming this key: alpha-rule" in alpha
    assert "cites claude-review/alpha" not in alpha                                 # a branch name is not a cited path
    beta = out.split("[3] beta")[1].split("[4] mystery")[0]
    assert "branch: research/beta (inferred from key)" in beta and "ahead 1 / behind 2 main" in beta
    assert "CONFLICTS in 1 file(s): conflict.txt" in beta and "tools/zzz.py: MISSING" in beta
    assert "worktree" in beta and "clean" in beta
    assert "branch: none named or inferable" in out.split("[4] mystery")[1].split("[5]")[0]


def test_branch_inferred_from_a_cited_sha_and_main_only_sha_is_ignored(tmp_path):
    repo, sha = make_repo(tmp_path)
    out = CB.brief(RUNBOOK.replace("SHA_PLACEHOLDER", sha), str(repo), NOW)
    last = out.split("[5] no-branch")[1]
    assert "(named)" in last                                                      # research/alpha is named in the text
    out2 = CB.brief(RUNBOOK.replace("SHA_PLACEHOLDER", sha).replace(" on research/alpha", ""), str(repo), NOW)
    assert "branch: research/alpha (inferred from sha %s)" % sha in out2.split("[5] no-branch")[1]
    main_sha = g(repo, "rev-parse", "--short", "main")
    out3 = CB.brief(RUNBOOK.replace("SHA_PLACEHOLDER", main_sha).replace(" on research/alpha", ""), str(repo), NOW)
    assert "none named or inferable" in out3.split("[5] no-branch")[1]


def test_read_only_and_key_filter_and_cli(tmp_path, capsys):
    repo, sha = make_repo(tmp_path)
    rb = tmp_path / "rb.md"
    rb.write_text(RUNBOOK.replace("SHA_PLACEHOLDER", sha))
    before = g(repo, "for-each-ref") + g(tmp_path / "wt-cr-alpha", "status", "--porcelain")
    assert CB.main(["--runbook", str(rb), "--repo", str(repo), "--now", "2026-10-06 12:00", "--key", "beta"]) == 0
    text = capsys.readouterr().out
    assert "1 open NEEDS CLAUDE line(s)" in text and "[1] beta" in text and "alpha" not in text.split("[1] beta")[0]
    assert g(repo, "for-each-ref") + g(tmp_path / "wt-cr-alpha", "status", "--porcelain") == before
    assert CB.main(["--runbook", str(tmp_path / "missing.md")]) == 2
