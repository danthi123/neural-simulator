"""Smoke-first (RULING smoke-first-1018, owner-approved 2026-10-18): no FLIP prereg is filed, and none of its gate-seed
lines queued, without a committed dev-seed `tools/answer_rate.py paired` record. Covers the shared validator
(tools/smoke_first.py), the commit-time gate (tools/gates/flip_smoke_first.py, real `git diff --cached` in scratch
repos) and the queue-time check (tools/gate_preflight.py check). Fixtures only; no brain, no GPU, no network."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import answer_rate as AR  # noqa: E402
import smoke_first as SF  # noqa: E402
from tools.gates import flip_smoke_first as GATE  # noqa: E402
from tools.gates import prereg_amendment_order as PRA  # noqa: E402

FLIP = "research/findings/2026-10-19-newthing-flip-PREREGISTRATION.md"
OLD_FLIP = "research/findings/2026-10-01-oldthing-flip-PREREGISTRATION.md"
REC = "research/findings/raw/_smoke/newthing_answer_rate.json"
RUNBOOK = SF.RUNBOOK_REL
RULINGS = ("# runbook\n\n## 5. Idle\n\n- idle-only-key: a section-5 bullet is not a ruling\n\n"
           "## 6. RULINGS (Claude writes `- <KEY>: <ruling> (<date>)`)\n\n"
           "- old-style-key: the ruling text (2026-10-01)\n"
           "- 2026-10-18 RULING dated-style-key (Claude; owner-approved): the ruling text\n\n## NEEDS CLAUDE\n\n"
           "- not-a-ruling-key: this is under NEEDS\n")


def _side(answered, n):
    return {"n": n, "answered": answered, "abstained": n - answered, "empty": 0, "text_only": 0,
            "answer_rate": (answered / n) if n else None}


def rec(seeds=(7, 8), hurt=(), mode="paired", unmeasured=()):
    """The JSON `answer_rate.py paired --json` writes, built by hand so each test varies one thing."""
    per = {}
    for s in seeds:
        if s in unmeasured:
            per[str(s)] = {"on": _side(0, 0), "off": _side(4, 4), "status": "UNMEASURED"}
        elif s in hurt:
            per[str(s)] = {"on": _side(2, 4), "off": _side(4, 4), "status": SF.HURTS}
        else:
            per[str(s)] = {"on": _side(4, 4), "off": _side(4, 4), "status": "ok"}
    out = {"check": "answer_rate", "label": "fac: flip vs off", "per_seed": per, "n_seeds": len(per),
           "flags": [SF.HURTS] if hurt else []}
    if mode:
        out["mode"] = mode
    return out


def prereg(extra="", flip="flip: yes\n"):
    return "---\ntype: preregistration\n%s---\n# a flip\n\n%s" % (flip, extra)


def blobs(**files):
    return {k: (json.dumps(v) if not isinstance(v, (str, bytes)) else v) for k, v in files.items()}


def judge(text, files, name="x-flip-PREREGISTRATION.md"):
    b = {k: v.encode() if isinstance(v, str) else v for k, v in blobs(**files).items()}
    return SF.judge_prereg(name, text, b.get, RULINGS)


# ---------------------------------------------------------------- the detector
@pytest.mark.parametrize("name,text,want", [
    ("2026-10-19-x-flip-PREREGISTRATION.md", "# t\n", True),                       # fallback: a `flip` token in the name
    ("2026-10-19-x-PREREGISTRATION.md", "flip: yes\n", True),                      # explicit
    ("2026-10-19-x-PREREGISTRATION.md", "---\nflip: BRAIN_A, BRAIN_B\n---\n", True),
    ("2026-10-19-x-PREREGISTRATION.md", "- **flip:** default-on\n", True),
    ("2026-10-19-x-PREREGISTRATION.md", "| flag | old default | new default |\n|---|---|---|\n", True),
    ("2026-10-19-x-flip-PREREGISTRATION.md", "flip: no\n", True),                  # a bare `flip: no` is the drive-by
    ("2026-10-19-x-flip-PREREGISTRATION.md", "flip: no (this registers a battery and flips no default)\n", False),
    ("2026-10-19-x-PREREGISTRATION.md", "flip: no\n", False),
    ("2026-10-19-x-PREREGISTRATION.md", "# battery\nvalidates default flips; becomes the production default\n", False),
    ("2026-10-12-validation-battery-prereg.md", "# The one validation battery for the default flips\n", False),
    ("2026-09-29-stage1-exit-gate-PREREGISTRATION.md", "# t\n", False),
    ("2026-10-19-flipper-PREREGISTRATION.md", "# t\n", False),                      # `flipper` is no `flip` token
    ("2026-10-19-x-flip-criteria-AMENDMENT-PREREG.md", "# t\n", False),             # an amendment inherits its parent's
    ("2026-10-19-x-flip-criteria-AMENDMENT-PREREG.md", "flip: yes\n", True),        # ... and may opt in
    ("2026-10-19-x-PREREGISTRATION.md", "flip: maybe later\n", False),              # an unrecognised value is ignored
])
def test_detector(name, text, want):
    assert SF.is_flip(name, text)[0] is want


def test_detector_has_no_false_positive_on_the_real_corpus():
    """Every flip the detector finds in the tree is a real flip prereg (the name rule found 10, the table rule only
    the word-role one), and none of the known non-flip preregs is among them."""
    found = {rel for rel, _, _ in SF._scan(str(ROOT))}
    assert "research/findings/2026-10-16-word-role-flip-PREREGISTRATION.md" in found
    for rel in ("research/findings/2026-10-12-validation-battery-prereg.md",
                "research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md",
                "research/findings/2026-09-24-production-chat-phase-timing-PREREGISTRATION.md",
                "research/findings/2026-09-24-production-default-battery-B2b-PREREGISTRATION.md",
                "research/findings/2026-09-23-d6-learn-through-use-v3-PREREGISTRATION-capability-gate.md",
                "research/findings/2026-09-29-iw1-wave-battery-PREREGISTRATION.md"):
        assert rel not in found, rel


# ---------------------------------------------------------------- the rulings
def test_ruling_keys_read_section_6_only_and_both_entry_styles():
    assert SF.ruling_keys(RULINGS) == {"old-style-key", "dated-style-key"}
    real = (ROOT / RUNBOOK).read_text(encoding="utf-8", errors="replace")
    keys = SF.ruling_keys(real)
    assert {"smoke-first-1018", "stall-alarm-1018"} <= keys                           # the dated style, in the real file
    assert "push-credential" in keys                                                  # the old style, in the real file


# ---------------------------------------------------------------- one prereg (the required scenarios)
def test_flip_prereg_without_smoke_record_blocks():
    probs = judge(prereg(), {})
    assert len(probs) == 1 and "no `smoke-record:` line" in probs[0] and "answer_rate.py paired" in probs[0]


def test_flip_prereg_with_a_valid_record_passes():
    assert judge(prereg("smoke-record: %s\n" % REC), {REC: rec()}) == []
    assert judge(prereg("- **smoke-record:** `%s`\n" % REC), {REC: rec()}) == []     # bullet / bold / backticks
    assert judge(prereg("smoke-record: %s\nsmoke-record: %s\n" % (REC, REC)), {REC: rec(seeds=(7, 8, 9))}) == []


def test_record_must_be_committed():
    probs = judge(prereg("smoke-record: %s\n" % REC), {})
    assert len(probs) == 1 and "not committed" in probs[0]


def test_a_gate_seed_record_blocks():
    probs = judge(prereg("smoke-record: %s\n" % REC), {REC: rec(seeds=(42, 43))})
    assert any("GATE seed" in p for p in probs)
    probs = judge(prereg("smoke-record: %s\n" % REC), {REC: rec(seeds=(7, 102))})   # one gate seed among dev seeds
    assert any("seed 102 is a GATE seed" in p for p in probs)
    probs = judge(prereg("smoke-record: %s\n" % REC), {REC: rec(seeds=(7, 16))})    # 16 is neither a dev nor a gate seed
    assert any("not a dev seed" in p for p in probs)


def test_a_hurts_record_blocks_without_the_ruling_and_passes_with_a_real_ruling_key():
    r = {REC: rec(hurt=(8,))}
    p = judge(prereg("smoke-record: %s\n" % REC), r)
    assert len(p) == 1 and SF.HURTS in p[0] and "ASK" in p[0]
    assert judge(prereg("smoke-record: %s\nsmoke-hurts-ruling: old-style-key\n" % REC), r) == []
    assert judge(prereg("smoke-record: %s\nsmoke-hurts-ruling: dated-style-key\n" % REC), r) == []
    for fake in ("made-up-key", "idle-only-key", "not-a-ruling-key"):                 # invented / wrong section
        p = judge(prereg("smoke-record: %s\nsmoke-hurts-ruling: %s\n" % (REC, fake)), r)
        assert len(p) == 1 and "not a RULING in section 6" in p[0], fake


def test_a_ruling_key_does_not_excuse_a_record_with_other_defects():
    p = judge(prereg("smoke-record: %s\nsmoke-hurts-ruling: old-style-key\n" % REC), {REC: rec(seeds=(42, 43), hurt=(42,))})
    assert any("GATE seed" in x for x in p)


def test_record_shape_rules():
    one = judge(prereg("smoke-record: %s\n" % REC), {REC: rec(seeds=(7,))})
    assert any("at least 2" in p for p in one)
    prefixed = judge(prereg("smoke-record: %s\n" % REC), {REC: rec(mode="prefixed")})
    assert any("not paired" in p for p in prefixed)
    nomode = judge(prereg("smoke-record: %s\n" % REC), {REC: rec(mode=None)})
    assert any("lacks `mode`" in p for p in nomode)
    unmeasured = judge(prereg("smoke-record: %s\n" % REC), {REC: rec(seeds=(7, 8, 9), unmeasured=(9,))})
    assert any("UNMEASURED" in p for p in unmeasured)
    notjson = judge(prereg("smoke-record: %s\n" % REC), {REC: "{nope"})
    assert any("not JSON" in p for p in notjson)
    other = judge(prereg("smoke-record: %s\n" % REC), {REC: {"check": "bar_sanity", "mode": "paired"}})
    assert any("not an answer_rate report" in p for p in other)
    for bad in ("/etc/passwd", "../outside.json"):
        assert any("inside the repository" in p for p in judge(prereg("smoke-record: %s\n" % bad), {}))


def test_a_hand_edited_record_is_refused():
    r = rec(hurt=(8,))
    r["per_seed"]["8"]["status"] = "ok"                      # counts say 2/4 vs 4/4; the edit says ok
    r["flags"] = []
    assert any("hand-edited" in p for p in judge(prereg("smoke-record: %s\n" % REC), {REC: r}))
    r2 = rec()
    r2["flags"] = [SF.HURTS]                                  # counts say no hurt; the edit adds the flag
    assert any("hand-edited" in p for p in judge(prereg("smoke-record: %s\n" % REC), {REC: r2}))


def test_non_flip_prereg_is_ignored():
    flip, why, probs = SF.evaluate("research/findings/2026-10-19-battery-PREREGISTRATION.md", "# a battery\n", {}.get,
                                   RULINGS)
    assert (flip, probs) == (False, [])
    flip, _, probs = SF.evaluate("research/findings/2026-10-19-x-PREREGISTRATION.md", "flip: no\n", {}.get, RULINGS)
    assert (flip, probs) == (False, [])


# ---------------------------------------------------------------- the producer/consumer contract
def _arm(d, name, probe):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(str(d), name), "w") as f:
        json.dump({"x_ask": probe}, f)


def test_a_record_written_by_answer_rate_itself_validates(tmp_path, capsys):
    ans, ab = {"answer": "The dog chases cat.", "abstained": False}, {"answer": "I don't know about that.", "abstained": True}
    for s in (7, 8):
        for tag, probe in (("flip", ans), ("off", ans)):
            _arm(tmp_path / tag / ("s%d" % s) / "fac", "intact_a_q.json", probe)
    argv = ["paired", "--base", str(tmp_path), "--on-tag", "flip", "--off-tag", "off", "--faculty", "fac", "--seeds",
            "7,8", "--json"]
    assert AR.main(argv) == 0
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert doc["mode"] == "paired" and doc["n_measured"] == 2
    assert SF.judge_record("r.json", out) == ([], [])
    _arm(tmp_path / "flip" / "s8" / "fac", "intact_a_q.json", ab)                     # now the flip abstains at seed 8
    assert AR.main(argv) == 0
    probs, hurts = SF.judge_record("r.json", capsys.readouterr().out)
    assert probs == [] and hurts == [8]
    assert AR.main(argv[:-3] + ["--json"]) == 0                                        # default seeds = the six gate seeds
    probs, _ = SF.judge_record("r.json", capsys.readouterr().out)
    assert any("GATE seed" in p for p in probs)


def test_report_without_a_mode_is_byte_identical_to_before():
    """Additive: the `mode` key exists only when a caller passes it (tools/iw2_score.py passes none)."""
    r = AR.report({"7": {"on": [{"answer": "a", "abstained": False}], "off": [{"answer": "a", "abstained": False}]}})
    assert "mode" not in r and r["check"] == "answer_rate"
    assert AR.report({}, mode="paired")["mode"] == "paired"


# ---------------------------------------------------------------- commit time: real git, scratch repos
def _git(td, env, *a):
    r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *a], cwd=td, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    env = PRA._stripped_env()
    env.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@example.invalid")

    class R:
        td = str(tmp_path)

        def w(self, rel, text):
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text if isinstance(text, str) else json.dumps(text), encoding="utf-8")

        def git(self, *a, date=None):
            e = dict(env)
            if date:
                e["GIT_COMMITTER_DATE"] = e["GIT_AUTHOR_DATE"] = date + "T12:00:00+00:00"
            return _git(self.td, e, *a)

        def commit(self, msg, date=None):
            self.git("add", "-A")
            self.git("commit", "-q", "-m", msg, date=date)
            return self.git("rev-parse", "HEAD")

        def check(self, kind="commit"):
            return GATE.check([], root=self.td, invocation=kind)

    r = R()
    r.git("init", "-q", "-b", "main")
    r.w(RUNBOOK, RULINGS)
    r.w(OLD_FLIP, "# an old flip prereg, written before the gate\n")
    r.commit("base", date="2026-10-01")
    return r


def test_gate_blocks_an_added_flip_prereg_with_no_smoke_record(repo):
    repo.w(FLIP, "# new flip\n")
    repo.git("add", "-A")
    probs = repo.check()
    assert len(probs) == 1 and "no `smoke-record:` line" in probs[0]


def test_gate_passes_with_a_valid_record_staged_in_the_same_commit_or_committed_before(repo):
    repo.w(FLIP, "# new flip\nsmoke-record: %s\n" % REC)
    repo.w(REC, rec())
    repo.git("add", "-A")
    assert repo.check() == []                                                          # staged together
    repo.git("reset", "-q")
    repo.git("add", REC)
    repo.git("commit", "-q", "-m", "the smoke")
    repo.git("add", "-A")
    assert repo.check() == []                                                          # committed before


def test_gate_blocks_an_uncommitted_gate_seed_or_hurts_record(repo):
    repo.w(FLIP, "# new flip\nsmoke-record: %s\n" % REC)
    repo.w(REC, rec())
    repo.git("add", FLIP)                                                              # the record is on disk, not staged
    assert any("not committed" in p for p in repo.check())
    repo.w(REC, rec(seeds=(42, 43)))
    repo.git("add", "-A")
    assert any("GATE seed" in p for p in repo.check())
    repo.w(REC, rec(hurt=(7,)))
    repo.git("add", "-A")
    assert any(SF.HURTS in p for p in repo.check())
    repo.w(FLIP, "# new flip\nsmoke-record: %s\nsmoke-hurts-ruling: old-style-key\n" % REC)
    repo.git("add", "-A")
    assert repo.check() == []                                                          # the ruling is in the staged runbook
    repo.w(FLIP, "# new flip\nsmoke-record: %s\nsmoke-hurts-ruling: invented\n" % REC)
    repo.git("add", "-A")
    assert any("not a RULING" in p for p in repo.check())


def test_gate_ignores_a_non_flip_prereg_and_non_prereg_files(repo):
    repo.w("research/findings/2026-10-19-some-battery-PREREGISTRATION.md", "# a battery; it flips nothing\n")
    repo.w("research/findings/2026-10-19-newthing-flip-RESULT.md", "# a result named flip is not a prereg\n")
    repo.w("research/findings/raw/x/2026-10-19-flip-PREREG.md", "# nested: not a pre-registration location\n")
    repo.git("add", "-A")
    assert repo.check() == []


def test_gate_grandfathers_existing_flip_preregs(repo):
    assert repo.check() == []                                                          # nothing staged
    repo.w(OLD_FLIP, "# an old flip prereg\n\n## AMENDMENT 1\nanother bar\n")
    repo.git("add", "-A")
    assert repo.check() == []                                                          # MODIFIED, not ADDED
    repo.git("reset", "-q")
    repo.git("checkout", "--", OLD_FLIP)
    repo.git("checkout", "--", OLD_FLIP)
    repo.git("mv", OLD_FLIP, "research/findings/2026-10-02-renamed-flip-PREREGISTRATION.md")
    assert repo.check() == []                                                          # a rename is not ADDED content


def test_gate_does_not_check_a_clean_auto_merge(repo):
    repo.w(FLIP, "# new flip\n")
    repo.git("add", "-A")
    assert repo.check("commit") and repo.check("merge") == []


def test_whole_current_tree_passes_the_gate_and_the_scan():
    """Grandfathering, proven on the real tree: nothing staged means nothing to judge, and every existing flip prereg
    predates the cutoff, so none would be checked at queue time."""
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=str(ROOT), capture_output=True, text=True)
    if staged.stdout.strip():
        pytest.skip("something is staged in this checkout; the gate would rightly judge it")
    assert GATE.check([], root=str(ROOT), invocation="commit") == []
    rows = SF._scan(str(ROOT))
    assert rows, "the tree has flip preregs; the scan must find them"
    for rel, _why, date in rows:
        assert date is not None and date < SF.GRANDFATHER_DATE, rel
        assert SF.is_grandfathered(str(ROOT), "HEAD", rel), rel
    assert GATE.selftest() == []


# ---------------------------------------------------------------- queue time
def _queue_repo(repo, date, with_record=False, extra=""):
    repo.w(FLIP, "# new flip\n" + (("smoke-record: %s\n" % REC) if with_record else "") + extra)
    if with_record:
        repo.w(REC, rec())
    return repo.commit("the flip prereg", date=date)


def test_queue_time_refuses_a_new_flip_prereg_without_a_record_and_accepts_with_one(repo):
    sha = _queue_repo(repo, "2026-10-19")
    ok, why = SF.check_queue(FLIP, sha, repo.td)
    assert not ok and "no valid dev-seed smoke" in why and "smoke-first-1018" in why
    sha2 = _queue_repo(repo, "2026-10-19", with_record=True)
    ok, why = SF.check_queue(FLIP, sha2, repo.td)
    assert ok, why
    repo.w(REC, rec(seeds=(42, 43)))
    sha3 = repo.commit("a gate-seed record", date="2026-10-19")
    ok, why = SF.check_queue(FLIP, sha3, repo.td)
    assert not ok and "GATE seed" in why
    ok, why = SF.check_queue(FLIP, sha, repo.td)                                       # the pinned revision decides
    assert not ok


def test_queue_time_grandfathers_a_flip_prereg_added_before_the_cutoff(repo):
    ok, why = SF.check_queue(OLD_FLIP, "HEAD", repo.td)
    assert ok and "grandfathered" in why
    sha = _queue_repo(repo, "2026-10-17")                                              # file name is dated 10-19, git says 10-17
    ok, why = SF.check_queue(FLIP, sha, repo.td)
    assert ok and "grandfathered" in why                                               # git history is the primary source
    assert SF.added_date(repo.td, sha, FLIP) == "2026-10-17"


def test_queue_time_is_silent_on_a_non_flip_prereg_and_on_an_unreadable_one(repo):
    repo.w("research/findings/2026-10-19-battery-PREREGISTRATION.md", "# a battery\n")
    sha = repo.commit("battery", date="2026-10-19")
    ok, why = SF.check_queue("research/findings/2026-10-19-battery-PREREGISTRATION.md", sha, repo.td)
    assert ok and "not a flip prereg" in why
    ok, why = SF.check_queue("research/findings/nope-flip-PREREGISTRATION.md", sha, repo.td)
    assert ok and "not checked" in why


def test_gate_preflight_check_refuses_through_the_real_cli(tmp_path):
    """tools/gate_preflight.py check = the call pool_queue.sh makes for a gate-seed line: the smoke-first refusal comes
    first, with its own message; an old flip prereg and a valid new one fall through to the pre-flight rows."""
    td = tmp_path / "repo"
    (td / "tools").mkdir(parents=True)
    for f in ("gate_preflight.py", "smoke_first.py"):
        shutil.copy(ROOT / "tools" / f, td / "tools" / f)
    env = PRA._stripped_env()
    env.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@example.invalid", GATE_PREFLIGHT_PATH=str(tmp_path / "preflight.tsv"))
    _git(str(td), env, "init", "-q", "-b", "main")

    def dated(day):
        return dict(env, GIT_COMMITTER_DATE=day + "T12:00:00+00:00", GIT_AUTHOR_DATE=day + "T12:00:00+00:00")

    for day, files in (("2026-10-01", ((RUNBOOK, RULINGS), (OLD_FLIP, "# old flip\n"))),
                       ("2026-10-19", ((FLIP, "# new flip\n"),))):
        for rel, text in files:
            p = td / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        _git(str(td), dated(day), "add", "-A")
        _git(str(td), dated(day), "commit", "-q", "-m", "files of " + day)
    e = dated("2026-10-19")
    sha = _git(str(td), e, "rev-parse", "HEAD")
    (tmp_path / "preflight.tsv").write_text("2026-10-19 00:00\t%s\t%s\tCLEAN\tok\t\t\n" % (FLIP, sha[:9]))

    def run(prereg):
        return subprocess.run([sys.executable, str(td / "tools" / "gate_preflight.py"), "check", "--prereg", prereg,
                               "--revision", sha], cwd=str(td), env=env, capture_output=True, text=True, timeout=60)

    r = run(FLIP)                                                                     # new, no record: smoke-first refuses
    assert r.returncode == 2 and "smoke" in r.stderr and "no `smoke-record:`" in r.stderr, r.stdout + r.stderr
    (td / REC).parent.mkdir(parents=True, exist_ok=True)
    (td / REC).write_text(json.dumps(rec()))
    (td / FLIP).write_text("# new flip\nsmoke-record: %s\n" % REC)
    _git(str(td), e, "add", "-A")
    _git(str(td), e, "commit", "-q", "-m", "record")
    sha = _git(str(td), e, "rev-parse", "HEAD")
    (tmp_path / "preflight.tsv").write_text("2026-10-19 00:00\t%s\t%s\tCLEAN\tok\t\t\n" % (FLIP, sha[:9]))
    r = subprocess.run([sys.executable, str(td / "tools" / "gate_preflight.py"), "check", "--prereg", FLIP,
                        "--revision", sha], cwd=str(td), env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr                                      # smoke ok, pre-flight row CLEAN
    r = subprocess.run([sys.executable, str(td / "tools" / "gate_preflight.py"), "check", "--prereg", OLD_FLIP,
                        "--revision", sha], cwd=str(td), env=env, capture_output=True, text=True, timeout=60)
    assert "smoke" not in r.stderr and "smoke-record" not in r.stderr                  # grandfathered: only the missing pre-flight row
    assert r.returncode == 2 and "pre-flight" in r.stderr
