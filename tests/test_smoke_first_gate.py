"""Smoke-first (RULING smoke-first-1018, owner-approved 2026-10-18): no FLIP prereg is filed, and none of its gate-seed
lines queued, without committed dev-seed `tools/answer_rate.py paired` records. Covers the shared validator
(tools/smoke_first.py), the commit-time gate (tools/gates/flip_smoke_first.py, real `git diff --cached` in scratch
repos) and the queue-time check (tools/gate_preflight.py check). Fixtures only; no brain, no GPU, no network.

Reviewed 2026-10-18 (two reviewers): a record the CLI writes must pass the real commit gates; a record is more than a
shape (two records, seeds 7 and 8, evidence, the faculty named in the prereg); every NEW prereg declares; an old prereg
edited or renamed into a flip is judged; a ruling must be Claude-signed, on main, and about THIS prereg; the grandfather
clock is a constant; a crash of the check opens nothing; and the tree test does not turn red when the gate works."""
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
from tools.gates import artifact_provenance as AP  # noqa: E402
from tools.gates import device_and_cost as DC  # noqa: E402
from tools.gates import flip_smoke_first as GATE  # noqa: E402
from tools.gates import prereg_amendment_order as PRA  # noqa: E402

FLIP = "research/findings/2026-10-19-newthing-flip-PREREGISTRATION.md"
OLD_FLIP = "research/findings/2026-10-01-oldthing-flip-PREREGISTRATION.md"
OLD_PLAIN = "research/findings/2026-10-01-oldplain-PREREGISTRATION.md"
REC_A = "research/findings/smoke/newthing_a.json"
REC_B = "research/findings/smoke/newthing_b.json"
RUNBOOK = SF.RUNBOOK_REL
RULINGS = ("# runbook\n\n## 5. Idle\n\n- idle-only-key: a section-5 bullet is not a ruling (Claude)\n\n"
           "## 6. RULINGS (Claude writes `- <KEY>: <ruling> (<date>)`)\n\n"
           "- old-style-key: the ruling about 2026-10-19-newthing-flip-PREREGISTRATION.md, continued\n"
           "  on an indented line (Claude 2026-10-01)\n"
           "- 2026-10-18 RULING dated-style-key (Claude; owner-approved): about x-flip-PREREGISTRATION.md\n"
           "- unsigned-key: about x-flip-PREREGISTRATION.md, written by the driver (Hermes 2026-10-01)\n"
           "- other-key: signed, but about another-file-PREREGISTRATION.md (Claude 2026-10-01)\n"
           "- record-key: about research/findings/smoke/newthing_b.json only (Claude 2026-10-01)\n\n## NEEDS CLAUDE\n\n"
           "- not-a-ruling-key: this is under NEEDS (Claude)\n")


def rec(tag="a", **kw):
    """The JSON `answer_rate.py paired --json` writes (the gate selftest's builder), so each test varies one thing."""
    return GATE._rec(tag, **kw)


def prereg(extra="", flip="flip: yes\n"):
    return "---\ntype: preregistration\n%s---\n# a flip for fac-x\n\n%s" % (flip, extra)


TWO = "smoke-record: %s\nsmoke-record: %s\n" % (REC_A, REC_B)


def good(**over):
    return {REC_A: rec("a", **over.get("a", {})), REC_B: rec("b", **over.get("b", {}))}


def judge(text, files, name="x-flip-PREREGISTRATION.md", runbook=RULINGS, read_blob=None):
    b = {k: (v if isinstance(v, bytes) else (v if isinstance(v, str) else json.dumps(v)).encode()) for k, v in files.items()}
    return SF.judge_prereg(name, text, read_blob or b.get, runbook)


# ---------------------------------------------------------------- the detector and the declaration
@pytest.mark.parametrize("name,text,want", [
    ("2026-10-19-x-flip-PREREGISTRATION.md", "# t\n", "flip"),                      # fallback: a `flip` token in the name
    ("2026-10-19-x-flips-PREREGISTRATION.md", "# t\n", "flip"),                     # ... also the plural
    ("2026-10-19-x-PREREGISTRATION.md", "flip: yes\n", "flip"),                     # explicit
    ("2026-10-19-x-PREREGISTRATION.md", "---\nflip: BRAIN_A, BRAIN_B\n---\n", "flip"),
    ("2026-10-19-x-PREREGISTRATION.md", "- **flip:** default-on\n", "flip"),
    ("2026-10-19-x-PREREGISTRATION.md", "| flag | old default | new default |\n|---|---|---|\n", "flip"),
    ("2026-10-19-x-flip-PREREGISTRATION.md", "flip: no\n", "flip"),                 # a bare `flip: no` is the drive-by
    ("2026-10-19-x-flip-PREREGISTRATION.md", "flip: no this prereg only registers a measurement honestly\n", "flip"),
    ("2026-10-19-x-PREREGISTRATION.md", "flip: no this registers a battery and flips no default\n", "nonflip"),
    ("2026-10-19-x-PREREGISTRATION.md", "flip: no\n", "undeclared"),                # a non-flip needs its reason
    ("2026-10-19-x-PREREGISTRATION.md", "flip: no too short\n", "undeclared"),
    ("2026-10-19-x-PREREGISTRATION.md", "# battery\nvalidates default flips; becomes the production default\n", "undeclared"),
    ("2026-10-19-d6-default-on-PREREGISTRATION.md", "# Flip BRAIN_D6_HEBBIAN_WRITE to default ON\n", "undeclared"),
    ("2026-10-12-validation-battery-prereg.md", "# The one validation battery for the default flips\n", "undeclared"),
    ("2026-10-19-flipper-PREREGISTRATION.md", "# t\n", "undeclared"),                # `flipper` is no `flip` token
    ("2026-10-19-x-flip-criteria-AMENDMENT-PREREG.md", "# t\n", "undeclared"),      # an amendment is exempt from the NAME rule
    ("2026-10-19-x-flip-criteria-AMENDMENT-PREREG.md", "flip: yes\n", "flip"),      # ... and may opt in
    ("2026-10-19-x-flip-criteria-AMENDMENT-PREREG.md", "flip: no amends a grandfathered prereg, adds a bar\n", "nonflip"),
    ("2026-10-19-x-PREREGISTRATION.md", "flip: maybe later\n", "undeclared"),       # an unrecognised value is ignored
])
def test_classification(name, text, want):
    assert SF.classify(name, text)[0] == want


def test_classification_on_the_real_corpus_finds_the_known_flips_and_no_battery():
    rows = SF._scan(str(ROOT), with_state=False)
    found = {r[0] for r in rows}
    assert "research/findings/2026-10-16-word-role-flip-PREREGISTRATION.md" in found
    assert len(found) >= 10
    for rel in ("research/findings/2026-10-12-validation-battery-prereg.md",
                "research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md",
                "research/findings/2026-09-24-production-chat-phase-timing-PREREGISTRATION.md",
                "research/findings/2026-09-24-production-default-battery-B2b-PREREGISTRATION.md",
                "research/findings/2026-09-23-d6-learn-through-use-v3-PREREGISTRATION-capability-gate.md",
                "research/findings/2026-09-29-iw1-wave-battery-PREREGISTRATION.md"):
        assert rel not in found, rel


# ---------------------------------------------------------------- the rulings
def test_ruling_entries_are_claude_signed_section_6_only_with_continuation_lines():
    e = SF.ruling_entries(RULINGS)
    assert set(e) == {"old-style-key", "dated-style-key", "other-key", "record-key"}
    assert "continued" in e["old-style-key"] and "indented line" in e["old-style-key"]       # the continuation is the entry
    real = (ROOT / RUNBOOK).read_text(encoding="utf-8", errors="replace")
    keys = SF.ruling_keys(real)
    assert {"smoke-first-1018", "stall-alarm-1018"} <= keys                                   # the dated style, signed
    assert "wrc-hold-1018" in keys                                                            # ... and another one


# ---------------------------------------------------------------- one prereg (the required scenarios)
def test_flip_prereg_without_smoke_record_blocks():
    probs = judge(prereg(), {})
    assert len(probs) == 1 and "no `smoke-record:` line" in probs[0] and "answer_rate.py paired" in probs[0]
    assert "iw1/iw2 battery" in probs[0] and "smoke-exempt-ruling" in probs[0]


def test_flip_prereg_with_two_valid_records_passes():
    assert judge(prereg(TWO), good()) == []
    assert judge(prereg("- **smoke-record:** `%s` faculty=fac-x\nsmoke-record: %s\n" % (REC_A, REC_B)), good()) == []


def test_the_ruling_asks_for_two_surfaces_so_one_record_or_a_repeat_blocks():
    one = judge(prereg("smoke-record: %s\n" % REC_A), good())
    assert len(one) == 1 and "at least 2 distinct records" in one[0]
    twice = judge(prereg("smoke-record: %s\nsmoke-record: %s\n" % (REC_A, REC_A)), good())
    assert any("at least 2 distinct records" in p for p in twice)
    same = {REC_A: rec("a"), REC_B: rec("a")}                                                  # two files, one smoke
    probs = judge(prereg(TWO), same)
    assert any("share the label" in p for p in probs) and any("read the same probe file" in p for p in probs)


def test_record_must_be_committed():
    probs = judge(prereg(TWO), {REC_A: rec("a")})
    assert len(probs) == 1 and "not committed" in probs[0] and REC_B in probs[0]


def test_gate_seed_and_other_seed_rules():
    probs = judge(prereg(TWO), {REC_A: rec("a"), REC_B: rec("b", seeds=(42, 43))})
    assert any("GATE seed" in p for p in probs)
    probs = judge(prereg(TWO), {REC_A: rec("a"), REC_B: rec("b", seeds=(7, 8, 102))})
    assert any("seed 102 is a GATE seed" in p for p in probs)
    probs = judge(prereg(TWO), {REC_A: rec("a"), REC_B: rec("b", seeds=(7, 8, 16))})
    assert any("not a dev seed" in p for p in probs)
    for seeds in ((1, 2), (7, 9), (8, 9)):                                  # the ruling names 7 AND 8
        probs = judge(prereg(TWO), {REC_A: rec("a"), REC_B: rec("b", seeds=seeds)})
        assert any("not measured; the smoke runs at dev seeds 7 and 8" in p for p in probs), seeds
    assert judge(prereg(TWO), {REC_A: rec("a"), REC_B: rec("b", seeds=(7, 8, 9, 10))}) == []   # more dev seeds are fine


def test_a_hurts_record_blocks_without_a_signed_ruling_about_this_decision():
    files = {REC_A: rec("a"), REC_B: rec("b", hurt=(8,))}
    p = judge(prereg(TWO), files)
    assert len(p) == 1 and SF.HURTS in p[0] and "ASK" in p[0]
    name = "2026-10-19-newthing-flip-PREREGISTRATION.md"
    assert judge(prereg(TWO + "smoke-hurts-ruling: old-style-key\n"), files, name=name) == []   # cites this file name
    assert judge(prereg(TWO + "smoke-hurts-ruling: record-key\n"), files, name=name) == []      # cites the record path
    cases = {"unsigned-key": "not a Claude-signed RULING", "idle-only-key": "not a Claude-signed RULING",
             "not-a-ruling-key": "not a Claude-signed RULING", "made-up-key": "not a Claude-signed RULING",
             "other-key": "does not cite this prereg"}
    for key, why in cases.items():
        p = judge(prereg(TWO + "smoke-hurts-ruling: %s\n" % key), files, name=name)
        assert len(p) == 1 and why in p[0], (key, p)
    p = judge(prereg(TWO + "smoke-hurts-ruling: dated-style-key\n"), files, name=name)          # about x-flip-..., not this one
    assert len(p) == 1 and "does not cite this prereg" in p[0]
    assert judge(prereg(TWO + "smoke-hurts-ruling: dated-style-key\n"), files, name="x-flip-PREREGISTRATION.md") == []


def test_a_ruling_key_does_not_excuse_a_record_with_other_defects():
    files = {REC_A: rec("a"), REC_B: rec("b", seeds=(42, 43), hurt=(42,))}
    p = judge(prereg(TWO + "smoke-hurts-ruling: old-style-key\n"), files, name="2026-10-19-newthing-flip-PREREGISTRATION.md")
    assert any("GATE seed" in x for x in p)


def test_exempt_ruling_replaces_the_records_but_only_with_a_real_one():
    name = "2026-10-19-newthing-flip-PREREGISTRATION.md"
    assert judge(prereg("smoke-exempt-ruling: old-style-key\n"), {}, name=name) == []
    p = judge(prereg("smoke-exempt-ruling: made-up-key\n"), {}, name=name)
    assert len(p) == 1 and "not a Claude-signed RULING" in p[0]
    p = judge(prereg("smoke-exempt-ruling: other-key\n"), {}, name=name)
    assert len(p) == 1 and "does not cite this prereg" in p[0]


def test_record_shape_rules():
    def two_with(bad):
        return judge(prereg(TWO), {REC_A: rec("a"), REC_B: bad})
    assert any("measures" in p or "not measured" in p for p in two_with(rec("b", seeds=(7,))))
    assert any("not paired" in p for p in two_with(rec("b", mode="prefixed")))
    assert any("lacks `mode`" in p for p in two_with(rec("b", mode=None)))
    assert any("UNMEASURED" in p for p in two_with(rec("b", seeds=(7, 8, 9), unmeasured=(9,))))
    assert any("not JSON" in p for p in two_with("{nope"))
    assert any("not an answer_rate report" in p for p in two_with({"check": "bar_sanity", "mode": "paired"}))
    for bad in ("/etc/passwd", "../outside.json"):
        assert any("inside the repository" in p for p in judge(prereg("smoke-record: %s\n%s" % (bad, TWO)), good()))


def test_a_hand_typed_or_borrowed_record_is_refused():
    typed = {"check": "answer_rate", "mode": "paired", "flags": [], "label": "x", "per_seed": {
        "7": {"on": {"n": 1, "answered": 1}, "off": {"n": 1, "answered": 1}, "status": "ok"},
        "8": {"on": {"n": 1, "answered": 1}, "off": {"n": 1, "answered": 1}, "status": "ok"}}}
    p = judge(prereg(TWO), {REC_A: rec("a"), REC_B: typed})
    assert any("not written by answer_rate.py" in x for x in p) and any("evidence" in x for x in p)
    p = judge(prereg(TWO), {REC_A: rec("a"), REC_B: rec("b", faculty="another-flip")})             # another flip's record
    assert any("faculty 'another-flip', which this prereg never names" in x for x in p)
    p = judge(prereg(TWO), {REC_A: rec("a"), REC_B: dict(rec("b"), off_tag="flip-b")})            # compared with itself
    assert any("compared with itself" in x for x in p)
    p = judge(prereg(TWO), {REC_A: rec("a"), REC_B: {k: v for k, v in rec("b").items() if k != "script"}})
    assert any("no `script` / `argv`" in x for x in p)
    one_probe = rec("b")
    for s in ("7", "8"):                                                                            # a one-probe smoke
        one_probe["per_seed"][s] = {"on": {"n": 1, "answered": 1}, "off": {"n": 1, "answered": 1}, "status": "ok"}
        one_probe["evidence"][s] = {side: one_probe["evidence"][s][side][:1] for side in ("on", "off")}
    assert any("needs at least 2 per side" in x for x in judge(prereg(TWO), {REC_A: rec("a"), REC_B: one_probe}))


def test_evidence_hash_is_checked_when_the_file_is_committed():
    import hashlib
    r = rec("b")
    r["evidence"]["7"]["on"][0] = {"file": "ev/probe.json", "sha256": hashlib.sha256(b"the real probe").hexdigest()}
    files = {REC_A: json.dumps(rec("a")).encode(), REC_B: json.dumps(r).encode()}
    ok = judge(prereg(TWO), files, read_blob=dict(files, **{"ev/probe.json": b"the real probe"}).get)
    assert ok == []
    forged = judge(prereg(TWO), files, read_blob=dict(files, **{"ev/probe.json": b"a different file"}).get)
    assert any("differs from the sha256 the record states" in x for x in forged)
    assert judge(prereg(TWO), files) == []                                               # not committed: not verifiable


@pytest.mark.parametrize("junk", [
    {"per_seed": {"²": {}}}, {"per_seed": {"7": [1], "8": "x"}}, {"flags": 5}, {"per_seed": {"7": {"on": [1], "off": {}}}},
    {"per_seed": {"7": {"on": {"n": "4", "answered": 4}, "off": {"n": 4, "answered": 4}, "status": "ok"}}},
    {"evidence": "nope"}, {"evidence": {"7": {"on": "x", "off": 3}}}, {"per_seed": []}, {"label": 5}])
def test_a_malformed_record_is_a_finding_never_a_crash(junk):
    probs = judge(prereg(TWO), {REC_A: rec("a"), REC_B: dict(rec("b"), **junk)})
    assert probs and all(isinstance(p, str) for p in probs)
    assert SF.judge_record("r.json", json.dumps(dict(rec("b"), **junk)))[0]


def test_a_probe_file_listed_twice_in_one_record_is_refused():
    r = rec("b")
    r["evidence"]["7"]["off"][0] = dict(r["evidence"]["7"]["on"][0])               # the same probe file on both sides
    probs = judge(prereg(TWO), {REC_A: rec("a"), REC_B: r})
    assert any("listed twice" in p for p in probs)


def test_the_validator_has_a_safety_net_for_a_crash_it_did_not_foresee(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("unforeseen")
    monkeypatch.setattr(SF, "_judge_record_inner", boom)
    probs, hurts = SF.judge_record("r.json", json.dumps(rec("a")))
    assert hurts == [] and len(probs) == 1 and "malformed record (RuntimeError: unforeseen)" in probs[0]
    assert judge(prereg(TWO), good())                                              # a problem, not a traceback


def test_a_hand_edited_record_is_refused_even_when_only_the_status_is_wrong():
    r = rec("b", hurt=(8,))
    r["per_seed"]["8"]["status"] = "ok"                      # counts say 2/4 vs 4/4; the edit says ok
    r["flags"] = []
    assert any("hand-edited" in p for p in judge(prereg(TWO), {REC_A: rec("a"), REC_B: r}))
    r2 = rec("b")
    r2["flags"] = [SF.HURTS]                                  # counts say no hurt; the edit adds the flag
    assert any("hand-edited" in p for p in judge(prereg(TWO), {REC_A: rec("a"), REC_B: r2}))
    r3 = rec("b")
    r3["per_seed"]["8"]["status"] = SF.HURTS                  # ONLY the status is wrong; flags and counts agree (no hurt)
    assert any("says status 'FLIP-HURTS-ANSWERS' but its own counts" in p for p in judge(prereg(TWO), {REC_A: rec("a"), REC_B: r3}))


def test_a_declared_non_flip_is_skipped_and_a_silent_new_prereg_is_judged():
    action, why, probs = SF.decide(str(ROOT), "HEAD", "research/findings/2026-10-19-battery-PREREGISTRATION.md",
                                   "flip: no it registers a battery and flips no default\n", {}.get, RULINGS, force_new=True)
    assert (action, probs) == ("skip", [])
    action, why, probs = SF.decide(str(ROOT), "HEAD", "research/findings/2026-10-19-battery-PREREGISTRATION.md", "# silent\n",
                                   {}.get, RULINGS, force_new=True)
    assert action == "judged" and len(probs) == 1 and "no flip declaration" in probs[0] and "flip: yes" in probs[0]


# ---------------------------------------------------------------- the producer/consumer contract
def _arm(d, name, probe):
    os.makedirs(str(d), exist_ok=True)
    with open(os.path.join(str(d), name), "w") as f:
        json.dump({"x_ask": probe}, f)


ANS = {"answer": "The dog chases cat.", "abstained": False}
ABS = {"answer": "I don't know about that.", "abstained": True}


def _cli_record(tmp_path, capsys, tag, on_probe=ANS, off_probe=ANS, faculty="fac-x", seeds="7,8", n_probes=2, extra=()):
    for s in seeds.split(","):
        for t, probe in (("flip-" + tag, on_probe), ("off-" + tag, off_probe)):
            for i in range(n_probes):
                _arm(tmp_path / t / ("s%s" % s) / faculty, "intact_q%d.json" % i, probe)
    argv = ["paired", "--base", str(tmp_path), "--on-tag", "flip-" + tag, "--off-tag", "off-" + tag, "--faculty", faculty,
            "--seeds", seeds, "--json", *extra]
    capsys.readouterr()
    assert AR.main(argv) == 0
    return capsys.readouterr().out


def test_a_record_written_by_answer_rate_itself_validates_and_passes_the_real_commit_gates(tmp_path, capsys):
    out_a = _cli_record(tmp_path, capsys, "a")
    out_b = _cli_record(tmp_path, capsys, "b")
    doc = json.loads(out_a)
    assert doc["mode"] == "paired" and doc["n_measured"] == 2 and doc["faculty"] == "fac-x"
    assert doc["on_tag"] == "flip-a" and doc["off_tag"] == "off-a" and doc["backend"] == "numpy" and doc["device"] == "cpu"
    assert doc["script"] == "tools/answer_rate.py" and doc["argv"][0] == "paired" and len(doc["evidence"]["7"]["on"]) == 2
    assert SF.judge_record("r.json", out_a) == ([], [])
    assert judge(prereg(TWO), {REC_A: out_a, REC_B: out_b}) == []
    # the real commit gates over the CLI's own output: device-and-cost (DC) and artifact-provenance (AP), no sidecar
    for name, text in (("smoke_a.json", out_a), ("smoke_b.json", out_b)):
        p = tmp_path / name
        p.write_text(text)
        assert DC._check_one(str(p), "research/findings/smoke/" + name) == [], name
        assert AP._has_provenance(str(p))[0], name
    # a flip that abstains at seed 8 is a HURTS, still a valid record
    out_h = _cli_record(tmp_path, capsys, "h", on_probe=ABS)
    probs, hurts = SF.judge_record("r.json", out_h)
    assert probs == [] and hurts == [7, 8]
    # default seeds = the six gate seeds: refused as a smoke
    argv = ["paired", "--base", str(tmp_path), "--on-tag", "flip-a", "--off-tag", "off-a", "--faculty", "fac-x", "--json"]
    assert AR.main(argv) == 0
    probs, _ = SF.judge_record("r.json", capsys.readouterr().out)
    assert any("GATE seed" in p for p in probs)


def test_answer_rate_refuses_to_compare_a_tag_with_itself(tmp_path, capsys):
    _cli_record(tmp_path, capsys, "a")
    argv = ["paired", "--base", str(tmp_path), "--on-tag", "flip-a", "--off-tag", "flip-a", "--faculty", "fac-x",
            "--seeds", "7,8", "--json"]
    assert AR.main(argv) == 2
    assert "compared with itself" in capsys.readouterr().err


def test_report_without_a_mode_is_byte_identical_to_before():
    """Additive: the `mode` key exists only when a caller passes it, and the library report carries none of the CLI's stamps
    (tools/iw2_score.py calls report() directly)."""
    r = AR.report({"7": {"on": [{"answer": "a", "abstained": False}], "off": [{"answer": "a", "abstained": False}]}})
    assert "mode" not in r and r["check"] == "answer_rate"
    assert not {"evidence", "backend", "script", "argv", "faculty"} & set(r)
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

        def two_records(self):
            self.w(REC_A, rec("a"))
            self.w(REC_B, rec("b"))

    r = R()
    r.git("init", "-q", "-b", "main")
    r.w(RUNBOOK, RULINGS)
    r.w(OLD_FLIP, "# an old flip prereg for fac-x, written before the gate\n")
    r.w(OLD_PLAIN, "# an old prereg that flips nothing, written before the gate\n")
    r.commit("base", date="2026-10-01")
    return r


def test_gate_blocks_an_added_flip_prereg_with_no_smoke_record(repo):
    repo.w(FLIP, "# new flip for fac-x\n")
    repo.git("add", "-A")
    probs = repo.check()
    assert len(probs) == 1 and "no `smoke-record:` line" in probs[0]


def test_gate_blocks_a_new_prereg_that_says_nothing(repo):
    repo.w("research/findings/2026-10-19-quiet-PREREGISTRATION.md", "# a prereg about d6 hebbian write, default ON\n")
    repo.git("add", "-A")
    probs = repo.check()
    assert len(probs) == 1 and "no flip declaration" in probs[0]
    repo.w("research/findings/2026-10-19-quiet-PREREGISTRATION.md", "# t\nflip: no it registers a battery and flips no default\n")
    repo.git("add", "-A")
    assert repo.check() == []


def test_a_new_amendment_named_file_must_still_declare(repo):
    name = "research/findings/2026-10-19-newflip-flip-AMENDMENT-PREREGISTRATION.md"
    repo.w(name, "# an amendment\n")
    repo.git("add", "-A")
    assert any("no flip declaration" in p for p in repo.check())
    repo.w(name, "# an amendment\nflip: no it amends the grandfathered word-role prereg and adds a bar only\n")
    repo.git("add", "-A")
    assert repo.check() == []


def test_gate_passes_with_two_valid_records_staged_in_the_same_commit_or_committed_before(repo):
    repo.w(FLIP, "# new flip for fac-x\n" + TWO)
    repo.two_records()
    repo.git("add", "-A")
    assert repo.check() == []                                                          # staged together
    repo.git("reset", "-q")
    repo.git("add", REC_A, REC_B)
    repo.git("commit", "-q", "-m", "the smoke")
    repo.git("add", "-A")
    assert repo.check() == []                                                          # committed before


def test_gate_blocks_uncommitted_gate_seed_and_hurts_records(repo):
    repo.w(FLIP, "# new flip for fac-x\n" + TWO)
    repo.two_records()
    repo.git("add", FLIP)                                                              # the records are on disk, not staged
    assert any("not committed" in p for p in repo.check())
    repo.w(REC_A, rec("a", seeds=(42, 43)))
    repo.git("add", "-A")
    assert any("GATE seed" in p for p in repo.check())
    repo.w(REC_A, rec("a", hurt=(7,)))
    repo.git("add", "-A")
    assert any(SF.HURTS in p for p in repo.check())
    repo.w(FLIP, "# new flip for fac-x\n" + TWO + "smoke-hurts-ruling: old-style-key\n")    # signed, on main, cites this file
    repo.git("add", "-A")
    assert repo.check() == []
    repo.w(FLIP, "# new flip for fac-x\n" + TWO + "smoke-hurts-ruling: invented\n")
    repo.git("add", "-A")
    assert any("not a Claude-signed RULING" in p for p in repo.check())


def test_a_ruling_staged_in_the_same_commit_does_not_count_only_main_does(repo):
    """Review 2026-10-18: a commit could carry its own approval. The runbook is read from main, not the index."""
    repo.w(REC_A, rec("a", hurt=(7,)))
    repo.w(REC_B, rec("b"))
    repo.w(FLIP, "# new flip for fac-x\n" + TWO + "smoke-hurts-ruling: self-made\n")
    repo.w(RUNBOOK, RULINGS.replace("## NEEDS CLAUDE", "- 2026-10-19 RULING self-made (Claude): fine, "
                                    "2026-10-19-newthing-flip-PREREGISTRATION.md\n\n## NEEDS CLAUDE"))
    repo.git("add", "-A")
    assert any("not a Claude-signed RULING" in p for p in repo.check())
    # the same ruling LANDED on main in an earlier commit: now it counts
    repo.git("reset", "-q")
    repo.git("add", RUNBOOK)
    repo.git("commit", "-q", "-m", "Claude's ruling lands on main", date="2026-10-19")
    repo.git("add", "-A")
    assert repo.check() == []


def test_gate_ignores_non_prereg_files_and_declared_non_flips(repo):
    repo.w("research/findings/2026-10-19-some-battery-PREREGISTRATION.md", "# a battery\nflip: no it registers a battery only\n")
    repo.w("research/findings/2026-10-19-newthing-flip-RESULT.md", "# a result named flip is not a prereg\n")
    repo.w("research/findings/raw/x/2026-10-19-flip-PREREG.md", "# nested: not a pre-registration location\n")
    repo.git("add", "-A")
    assert repo.check() == []


def test_gate_grandfathers_old_flip_preregs_but_judges_one_edited_or_renamed_INTO_a_flip(repo):
    assert repo.check() == []                                                          # nothing staged
    repo.w(OLD_FLIP, "# an old flip prereg\n\n## AMENDMENT 1\nanother bar\n")
    repo.git("add", "-A")
    assert repo.check() == []                                                          # MODIFIED, already a flip: grandfathered
    repo.git("reset", "-q")
    repo.git("checkout", "--", OLD_FLIP)
    repo.git("mv", OLD_FLIP, "research/findings/2026-10-02-renamed-flip-PREREGISTRATION.md")
    assert repo.check() == []                                                          # renamed, still the same old flip
    repo.git("mv", "research/findings/2026-10-02-renamed-flip-PREREGISTRATION.md", OLD_FLIP)
    repo.git("mv", OLD_FLIP, "research/findings/2026-10-19-renamed-flip-PREREGISTRATION.md")
    assert any("no `smoke-record:` line" in p for p in repo.check())                  # ... but a post-cutoff DATE is new
    repo.git("mv", "research/findings/2026-10-19-renamed-flip-PREREGISTRATION.md", OLD_FLIP)
    assert repo.check() == []
    # C2: an old prereg that was no flip is edited in place to declare itself one
    repo.w(OLD_PLAIN, "# an old prereg\nflip: yes\n")
    repo.git("add", "-A")
    probs = repo.check()
    assert len(probs) == 1 and "no `smoke-record:` line" in probs[0]
    repo.git("reset", "-q")
    repo.git("checkout", "--", OLD_PLAIN)
    # C1: ... or renamed into a flip name
    repo.git("mv", OLD_PLAIN, "research/findings/2026-10-02-sneaky-flip-PREREGISTRATION.md")
    assert any("no `smoke-record:` line" in p for p in repo.check())
    repo.git("mv", "research/findings/2026-10-02-sneaky-flip-PREREGISTRATION.md", OLD_PLAIN)
    repo.w(OLD_PLAIN, "# an old prereg, edited but still no flip\n")
    repo.git("add", "-A")
    assert repo.check() == []                                                          # a silent old prereg is not asked to declare


def test_a_new_flip_prereg_stays_judged_on_every_later_commit_that_touches_it(repo):
    repo.w(FLIP, "# new flip for fac-x\n" + TWO)
    repo.two_records()
    repo.commit("the flip prereg with its records", date="2026-10-19")
    repo.w(FLIP, "# new flip for fac-x\n" + TWO + "\n## AMENDMENT 1\nanother bar\n")
    repo.git("add", "-A")
    assert repo.check() == []                                                          # an amendment appended: still valid
    repo.w(FLIP, "# new flip for fac-x\n\n## AMENDMENT 1\nthe smoke lines were dropped\n")
    repo.git("add", "-A")
    assert any("no `smoke-record:` line" in p for p in repo.check())


def test_gate_does_not_check_a_clean_auto_merge(repo):
    repo.w(FLIP, "# new flip for fac-x\n")
    repo.git("add", "-A")
    assert repo.check("commit") and repo.check("merge") == []


def test_the_tree_test_is_not_a_time_bomb():
    """Grandfathering proven on the real tree without going red when the gate works: every flip prereg dated before the
    cutoff is grandfathered at HEAD; one dated after it must carry valid smoke (that is the gate doing its job)."""
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=str(ROOT), capture_output=True, text=True)
    if staged.stdout.strip():
        pytest.skip("something is staged in this checkout; the gate would rightly judge it")
    assert GATE.check([], root=str(ROOT), invocation="commit") == []
    rows = SF._scan(str(ROOT), with_state=False)
    assert rows, "the tree has flip preregs; the scan must find them"
    runbook = SF.read_runbook(str(ROOT))
    probe = [r for r in rows if r[2] is not None and r[2] < SF.GRANDFATHER_DATE][:3]       # history is ~1 s per file
    for rel, _why, date, _s in probe:
        assert SF.history_state(str(ROOT), "HEAD", rel)[0] == "old", rel
        assert SF.decide(str(ROOT), "HEAD", rel, (ROOT / rel).read_text(encoding="utf-8", errors="replace"),
                         lambda r: SF.blob_at(str(ROOT), "HEAD", r), runbook)[0] == "skip", rel
    for rel, _why, date, _s in rows:
        if date is not None and date < SF.GRANDFATHER_DATE:
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")           # added after the gate: it must pass it
        probs = SF.decide(str(ROOT), "HEAD", rel, text, lambda r: SF.blob_at(str(ROOT), "HEAD", r), runbook)[2]
        assert probs == [], (rel, probs)
    assert GATE.selftest() == []


# ---------------------------------------------------------------- queue time
def _queue_repo(repo, date, with_records=False, extra=""):
    repo.w(FLIP, "# new flip for fac-x\n" + (TWO if with_records else "") + extra)
    if with_records:
        repo.two_records()
    return repo.commit("the flip prereg", date=date)


def test_queue_time_refuses_a_new_flip_prereg_without_records_and_accepts_with_them(repo):
    sha = _queue_repo(repo, "2026-10-19")
    ok, why = SF.check_queue(FLIP, sha, repo.td)
    assert not ok and "smoke-first-1018" in why and "no `smoke-record:` line" in why
    sha2 = _queue_repo(repo, "2026-10-19", with_records=True)
    ok, why = SF.check_queue(FLIP, sha2, repo.td)
    assert ok, why
    repo.w(REC_B, rec("b", seeds=(42, 43)))
    sha3 = repo.commit("a gate-seed record", date="2026-10-19")
    ok, why = SF.check_queue(FLIP, sha3, repo.td)
    assert not ok and "GATE seed" in why
    ok, why = SF.check_queue(FLIP, sha, repo.td)                                       # the pinned revision decides
    assert not ok


def test_queue_time_grandfathers_by_the_later_of_git_and_file_name_date(repo):
    ok, why = SF.check_queue(OLD_FLIP, "HEAD", repo.td)
    assert ok and "grandfathered" in why
    # a forged committer date does not move a file whose NAME is dated after the cutoff
    sha = _queue_repo(repo, "2026-10-02")
    ok, why = SF.check_queue(FLIP, sha, repo.td)
    assert not ok and "no `smoke-record:` line" in why
    # ... and an honest old commit of a file with an old name stays grandfathered
    repo.w("research/findings/2026-10-03-honest-flip-PREREGISTRATION.md", "# an honest old flip\n")
    sha = repo.commit("old", date="2026-10-03")
    ok, why = SF.check_queue("research/findings/2026-10-03-honest-flip-PREREGISTRATION.md", sha, repo.td)
    assert ok and "grandfathered" in why


def test_the_cutoff_is_a_constant_not_an_environment_variable(repo):
    r = subprocess.run([sys.executable, "-c", "import smoke_first as s; print(s.GRANDFATHER_DATE)"],
                       cwd=str(ROOT / "tools"), capture_output=True, text=True,
                       env=dict(os.environ, SMOKE_FIRST_GRANDFATHER_DATE="2099-01-01"))
    assert r.stdout.strip() == "2026-10-18", r.stderr
    sha = _queue_repo(repo, "2026-10-19")
    old = SF.GRANDFATHER_DATE
    try:
        SF.GRANDFATHER_DATE = "2099-01-01"                                              # the way a TEST moves it
        assert SF.check_queue(FLIP, sha, repo.td)[0] is True
    finally:
        SF.GRANDFATHER_DATE = old
    assert SF.check_queue(FLIP, sha, repo.td)[0] is False


def test_queue_time_judges_an_old_prereg_edited_or_renamed_into_a_flip(repo):
    repo.w(OLD_PLAIN, "# an old prereg\nflip: yes\n")
    sha = repo.commit("edited into a flip", date="2026-10-19")
    ok, why = SF.check_queue(OLD_PLAIN, sha, repo.td)
    assert not ok and "became a flip" in why
    new = "research/findings/2026-10-19-moved-flip-PREREGISTRATION.md"
    repo.git("mv", OLD_FLIP, new)
    sha = repo.commit("moved", date="2026-10-19")
    ok, why = SF.check_queue(new, sha, repo.td)                  # an old flip renamed to a post-cutoff DATE: counts as new
    assert not ok and "no `smoke-record:` line" in why
    same_age = "research/findings/2026-10-02-moved-flip-PREREGISTRATION.md"
    repo.git("mv", new, same_age)
    sha = repo.commit("moved again", date="2026-10-19")
    ok, why = SF.check_queue(same_age, sha, repo.td)             # ... but with its old date it is the same old flip
    assert ok and "grandfathered" in why


def test_queue_time_is_silent_on_a_non_flip_prereg_and_on_an_unreadable_one(repo):
    repo.w("research/findings/2026-10-19-battery-PREREGISTRATION.md", "# a battery\nflip: no it registers a battery only\n")
    sha = repo.commit("battery", date="2026-10-19")
    ok, why = SF.check_queue("research/findings/2026-10-19-battery-PREREGISTRATION.md", sha, repo.td)
    assert ok and "no" in why
    ok, why = SF.check_queue("research/findings/nope-flip-PREREGISTRATION.md", sha, repo.td)
    assert ok and "not checked" in why


def _preflight_repo(tmp_path):
    """tools/ copy + a scratch repo with an old flip and a new one, for the real gate_preflight CLI."""
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
                       ("2026-10-19", ((FLIP, "# new flip for fac-x\n"),))):
        for rel, text in files:
            p = td / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        _git(str(td), dated(day), "add", "-A")
        _git(str(td), dated(day), "commit", "-q", "-m", "files of " + day)
    return td, env, dated("2026-10-19")


def _preflight(td, env, prereg, sha):
    return subprocess.run([sys.executable, str(td / "tools" / "gate_preflight.py"), "check", "--prereg", prereg,
                           "--revision", sha], cwd=str(td), env=env, capture_output=True, text=True, timeout=60)


def test_gate_preflight_check_refuses_through_the_real_cli(tmp_path):
    """tools/gate_preflight.py check = the call pool_queue.sh makes for a gate-seed line: the smoke-first refusal comes
    first, with its own message; an old flip prereg and a valid new one fall through to the pre-flight rows."""
    td, env, e = _preflight_repo(tmp_path)
    sha = _git(str(td), e, "rev-parse", "HEAD")
    (tmp_path / "preflight.tsv").write_text("2026-10-19 00:00\t%s\t%s\tCLEAN\tok\t\t\n" % (FLIP, sha[:9]))
    r = _preflight(td, env, FLIP, sha)                                      # new, no record: smoke-first refuses
    assert r.returncode == 2 and "smoke" in r.stderr and "no `smoke-record:`" in r.stderr, r.stdout + r.stderr
    for rel, tag in ((REC_A, "a"), (REC_B, "b")):
        (td / rel).parent.mkdir(parents=True, exist_ok=True)
        (td / rel).write_text(json.dumps(rec(tag)))
    (td / FLIP).write_text("# new flip for fac-x\n" + TWO)
    _git(str(td), e, "add", "-A")
    _git(str(td), e, "commit", "-q", "-m", "records")
    sha = _git(str(td), e, "rev-parse", "HEAD")
    (tmp_path / "preflight.tsv").write_text("2026-10-19 00:00\t%s\t%s\tCLEAN\tok\t\t\n" % (FLIP, sha[:9]))
    r = _preflight(td, env, FLIP, sha)
    assert r.returncode == 0, r.stdout + r.stderr                           # smoke ok, pre-flight row CLEAN
    r = _preflight(td, env, OLD_FLIP, sha)
    assert "smoke" not in r.stderr and "smoke-record" not in r.stderr       # grandfathered: only the missing pre-flight row
    assert r.returncode == 2 and "pre-flight" in r.stderr


def test_gate_preflight_fails_closed_when_the_smoke_check_crashes_and_open_only_when_it_is_missing(tmp_path):
    """Review 2026-10-18: a validator that raised opened the queue. Only a missing module skips the check."""
    td, env, e = _preflight_repo(tmp_path)
    sha = _git(str(td), e, "rev-parse", "HEAD")
    (tmp_path / "preflight.tsv").write_text("2026-10-19 00:00\t%s\t%s\tCLEAN\tok\t\t\n" % (FLIP, sha[:9]))
    (td / "tools" / "smoke_first.py").write_text("def check_queue(*a, **k):\n    raise RuntimeError('boom')\n")
    r = _preflight(td, env, FLIP, sha)
    assert r.returncode == 2 and "crashed (RuntimeError: boom)" in r.stderr, r.stdout + r.stderr
    (td / "tools" / "smoke_first.py").write_text("this is not python (\n")
    r = _preflight(td, env, FLIP, sha)
    assert r.returncode == 2 and "could not be loaded (SyntaxError" in r.stderr, r.stdout + r.stderr
    (td / "tools" / "smoke_first.py").unlink()
    r = _preflight(td, env, FLIP, sha)
    assert r.returncode == 0 and "smoke-first check skipped" in r.stderr, r.stdout + r.stderr   # an older checkout


def test_python38_syntax_for_the_modules_the_servers_system_python_runs():
    import ast
    for rel in ("tools/smoke_first.py", "tools/gate_preflight.py", "tools/stall_watch.py"):
        ast.parse((ROOT / rel).read_text(encoding="utf-8"), feature_version=(3, 8))
