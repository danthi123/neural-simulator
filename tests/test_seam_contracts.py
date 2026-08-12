"""INTEGRATION-SEAM CONTRACTS: the two halves of a handoff must agree on format and location.

WHY THIS FILE EXISTS. Four defects landed on 2026-07-31, and not one of them was visible from inside either
component, because **both halves were individually correct**. Every unit test of either side passed throughout.

  1. `research/runners/__init__.py` (the automatic provenance door) writes `<artifact>.prov.json`, while
     `tools/gates/artifact_provenance.py` accepted only `.cmd.json` / `.provenance.json` -- so every artifact
     the door stamped was reported UNPROVENANCED. Fixed once by adding the suffix, then a second time because
     the two halves also disagreed on how the suffix ATTACHES: the door APPENDS to the full path
     (`x.json` -> `x.json.prov.json`), the hand-written sidecars REPLACE the extension (`x.json` -> `x.cmd.json`).
  2. `tools/lane_check.py` read `research/queue/gpu.queue` only, so every job staged to the mini-PC pool in
     `research/queue/pool.queue` was invisible and the tool reported starvation that had already been fixed.
  3. `tools/gates/lane_starvation.py` counted local `ps` only. Work DISPATCHED to a pool node leaves the local
     queue and runs remotely, so three lanes read as STARVED while their jobs were running on pool40/41.
  4. `tools/pool_queue.sh` appends a `#checked:<reason>` token that `tools/pool_autodispatch.sh` must STRIP
     before executing. It once did not: the exit-status wrapper puts the job in a brace group, `{ $JOB; }`, and
     an unstripped `#` comments out the closing `; }` -- unterminated brace group, syntax error, six jobs
     dispatched and silently killed while the log said "dispatched" six times.

WHAT A CONTRACT TEST HERE MUST DO. Assert that BOTH halves agree, so a UNILATERAL change to either one fails.
Every assertion below is anchored to one of:
  * the producer's ACTUAL emitted bytes (its `printf`/`echo` executed, or its writer function called), and
  * the consumer's ACTUAL parsing code (its `awk`/split/strip lines extracted from source and executed).
Nothing here re-implements a format it then tests against itself -- that is the tautology this whole class of
test exists to avoid.

POWER CONTROLS. Each seam carries at least one assertion in the FAILING direction (a wrong suffix must be
REJECTED; a stale dispatch log must FIRE starvation; an unstripped token must BREAK `bash -n`). A contract test
whose every assertion is "the good case passes" cannot distinguish a working seam from a check that accepts
everything.

INSTRUMENT VERIFICATION (do not trust a green contract suite that has never been shown to go red). Verified by
MUTATION, not by inspection: a shadow copy of the repo is built in a temp dir, ONE half of ONE seam is broken,
and the suite must fail. **25 of 25 mutations caught** (both halves of every seam: writer suffix, gate suffix
list, each attachment mode separately, each queue path, each strip, each delimiter, each brace group).
Harness + log: `scratchpad/mutate.py`, `scratchpad/mutation_run.log` (session scratch; regenerate by copying
the listed sources into a temp tree and re-running). The harness refuses to score a mutation whose pattern did
not apply -- a no-op mutation reports MISSED and is indistinguishable from a real gap.

That check found TWO real defects in this file, both of the "assertion that cannot fail" kind:
  * `assert "{ $JOB; }" in src` passed with the brace-group wrapper DELETED, because a comment four lines above
    the code contains the identical string. Every source-presence assertion is now scoped to `_code_lines()`.
  * `assert parsed.endswith("--seeds 42")` passed with the log delimiter changed, because a `sed` that matches
    nothing returns the line UNCHANGED -- and the unparsed line also ends with those characters. Parses are now
    compared by EXACT equality against the string the producer emitted.

WHAT THIS FILE CANNOT CATCH.
  * Whether the exchanged content is TRUE (a `.prov.json` naming the wrong runner passes).
  * Runtime/environment failures: ssh reachability, systemd unit state, whether a dispatched job actually ran.
  * Seams not listed above. This file is scoped to the four measured defects, deliberately: a contract test
    suite that fails on unrelated legacy drift gets deleted, which is worse than not having it.
  * Producer-side shell scripts are NOT executed end-to-end (`pool_queue.sh add` writes to the LIVE queue at a
    hard-coded absolute path and would inject a job into the running dispatcher). Its emitted FORMAT is
    extracted from source and executed; its argparse/record gates are out of scope here.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The provenance door registers an atexit hook that walks the REAL research/findings/raw/ and stamps every file
# touched since import. Under pytest that would litter the working tree with sidecars for unrelated artifacts.
# Disable it for the import only (`_ENABLED` is read once, at import time); the module's functions -- which are
# what this file tests -- are unaffected.
_PREV_NOPROV = os.environ.get("SIM_NO_PROVENANCE")
os.environ["SIM_NO_PROVENANCE"] = "1"
import research.runners as prov_door  # noqa: E402  (the WRITER half of seam 1)
if _PREV_NOPROV is None:
    os.environ.pop("SIM_NO_PROVENANCE", None)
else:
    os.environ["SIM_NO_PROVENANCE"] = _PREV_NOPROV

import tools.gates.artifact_provenance as prov_gate      # noqa: E402  (the READER half of seam 1)
import tools.lane_check as lane_check                    # noqa: E402  (the READER half of seam 2)
import tools.gates.lane_starvation as lane_gate          # noqa: E402  (the READER half of seams 2 + 3)

assert not prov_door._ENABLED, "provenance door must be import-disabled inside the test session"


# ----------------------------------------------------------------------------------------------------------
# helpers: read the real sources, and execute real shell fragments taken from them
# ----------------------------------------------------------------------------------------------------------
def _src(rel: str) -> str:
    p = ROOT / rel
    assert p.exists(), "seam source vanished: %s" % rel
    return p.read_text(encoding="utf-8")


def _unescape(fmt: str) -> str:
    """A shell printf format literal ('%s\\t%s\\n' as it appears in the file) -> the real control chars."""
    return fmt.replace("\\t", "\t").replace("\\n", "\n")


def _printf_format(rel: str, must_contain: str, redirect: str) -> str:
    """The single-quoted format string of the `printf ... >> <redirect>` line that emits `must_contain`."""
    hits = [l for l in _src(rel).splitlines()
            if "printf" in l and must_contain in l and redirect in l]
    assert len(hits) == 1, "expected exactly 1 emitting printf in %s, found %d" % (rel, len(hits))
    m = re.search(r"printf\s+'([^']+)'", hits[0])
    assert m, "could not read the printf format out of: %s" % hits[0].strip()
    return m.group(1)


def _code_lines(rel: str) -> list:
    """Source lines with comment-ONLY lines removed.

    Earned during this file's own mutation check: `assert "{ $JOB; }" in src` passed even after the brace-group
    wrapper was deleted from the executing line, because a COMMENT four lines above explains the hazard using
    the identical string. A source assertion satisfied by prose is a check that cannot fail -- exactly the class
    of defect this file exists to pin. Every source-presence assertion below is scoped to code lines.
    """
    return [l for l in _src(rel).splitlines() if not l.lstrip().startswith("#")]


def _in_code(rel: str, needle: str) -> list:
    return [l.strip() for l in _code_lines(rel) if needle in l]


def _line_starting(rel: str, prefix: str, contains: str | None = None) -> str:
    """Exactly one matching line, or fail. Ambiguity here means the extraction (not the seam) is unsound, and
    an extraction that silently picks the first of several matches is how a contract test starts asserting
    something other than what it names."""
    hits = [l.strip() for l in _code_lines(rel)
            if l.strip().startswith(prefix) and (contains is None or contains in l)]
    assert len(hits) == 1, "expected exactly 1 line starting %r%s in %s, found %d" % (
        prefix, "" if contains is None else " containing %r" % contains, rel, len(hits))
    return hits[0]


def _bash(script: str, cwd: Path | None = None):
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          cwd=str(cwd) if cwd else None, timeout=60)


def _systemd_redirect_targets(unit_text: str) -> list:
    """Where a systemd unit appends its stdout/stderr. Split out of the test body ON PURPOSE: the unit lives
    outside the repo, so the shadow-repo mutation harness cannot exercise that assertion, and a parser that is
    only ever fed the one good input is a check of unknown power. This function can be mutation-tested on
    text alone."""
    return [t.strip() for t in re.findall(r"^Standard(?:Output|Error)=append:(.+)$", unit_text, re.M)]


class _FakePs:
    """Stands in for the `subprocess` module inside a gate, so a real local `ps` cannot mask the failing
    direction: a genuine `research.runners` process running on this machine while the suite runs would serve a
    lane and make the starvation control silently vacuous."""

    def __init__(self, stdout: str = ""):
        self._stdout = stdout

    def run(self, *a, **k):
        class R:
            pass
        r = R()
        r.stdout = self._stdout
        r.returncode = 0
        return r


def _raw(tmp_path: Path) -> Path:
    """A temp dir whose path contains `research/findings/raw/` -- required by the gate's _is_artifact()."""
    d = tmp_path / "research" / "findings" / "raw"
    d.mkdir(parents=True)
    return d


# ==========================================================================================================
# SEAM 1 — provenance sidecar: research/runners/__init__.py (writer) <-> tools/gates/artifact_provenance.py
# ==========================================================================================================
class TestSeam1ProvenanceSidecarNaming:

    @staticmethod
    def _stamp(tmp_path: Path, monkeypatch, artifact_bodies: dict):
        """Run the REAL writer over a temp raw/ dir. Returns (raw_dir, {name: path}, sidecars_created)."""
        raw = _raw(tmp_path)
        monkeypatch.setattr(prov_door, "_RAW_DIR", str(raw))
        monkeypatch.setattr(prov_door, "_PROV_DIR", str(raw / "_provenance"))
        monkeypatch.setattr(prov_door, "_START", time.time() - 5.0)
        monkeypatch.setenv("SIM_RUN_ID", "")            # _record_start() sets it; monkeypatch restores it
        paths = {}
        for name, body in artifact_bodies.items():
            p = raw / name
            p.write_text(body, encoding="utf-8")
            paths[name] = p
        rec = prov_door._record_start()
        made = prov_door._stamp_outputs(rec)
        return raw, paths, made

    def test_writer_output_is_accepted_by_the_gate_end_to_end(self, tmp_path, monkeypatch):
        """The exact bytes the door writes must satisfy the gate. Includes its own before/after control: the
        artifact MUST be flagged before stamping, or the 'accepted' half proves nothing."""
        raw = _raw(tmp_path)
        art = raw / "result.json"
        art.write_text('{"final_score": 0.91, "n": 6}', encoding="utf-8")   # deliberately provenance-FREE
        before = prov_gate.check([str(art)])
        assert before, "control failed: a provenance-free artifact was not flagged, so this test has no power"

        monkeypatch.setattr(prov_door, "_RAW_DIR", str(raw))
        monkeypatch.setattr(prov_door, "_PROV_DIR", str(raw / "_provenance"))
        monkeypatch.setattr(prov_door, "_START", time.time() - 5.0)
        monkeypatch.setenv("SIM_RUN_ID", "")
        made = prov_door._stamp_outputs(prov_door._record_start())

        assert str(art) in made, "the door did not stamp the artifact it created: %s" % made
        after = prov_gate.check([str(art)])
        assert after == [], (
            "SEAM BROKEN: the provenance door stamped %s but the gate still reports it unprovenanced: %s"
            % (art.name, after))

    def test_writer_appends_the_suffix_and_the_gate_accepts_that_attachment_mode(self, tmp_path, monkeypatch):
        """Two disagreements happened here: the SUFFIX, then how it ATTACHES. Pin both, empirically."""
        raw, paths, made = self._stamp(tmp_path, monkeypatch, {"a.json": '{"score": 1}'})
        art = paths["a.json"]
        sidecars = sorted(p.name for p in raw.iterdir() if p.name != art.name and p.is_file())
        assert sidecars == ["a.json.prov.json"], (
            "the door's sidecar naming changed; the gate must be updated in the same commit. Got: %s" % sidecars)

        sidecar = raw / sidecars[0]
        suffix = sidecar.name[len(art.name):]
        assert suffix in prov_gate._SIDECARS, (
            "SEAM BROKEN: writer emits suffix %r, gate accepts only %r" % (suffix, prov_gate._SIDECARS))
        assert sidecar.name == art.name + suffix, "writer must APPEND to the full path, not replace .json"
        assert not (raw / (art.stem + suffix)).exists(), "writer is not using the replace form"

        # ...and the gate must accept it in the APPEND position specifically (the second regression).
        ok, why = prov_gate._has_provenance(str(art))
        assert ok, "gate rejects the appended sidecar it claims to support: %s" % why
        assert json.loads(sidecar.read_text())["argv"], "sidecar carries no argv -- provenance without content"

    def test_gate_accepts_the_replace_form_the_hand_written_sidecars_use(self, tmp_path):
        """The OTHER convention, still live: g11 runners write Path(out).with_suffix('.cmd.json'). Both
        attachment modes must be accepted, because both are in the tree."""
        for rel in ("research/runners/g11_bg_runner.py", "research/runners/g11_bg_replicated_runner.py"):
            assert _in_code(rel, 'with_suffix(".cmd.json")'), (
                "%s no longer builds its sidecar with with_suffix('.cmd.json') -- if it switched to the append "
                "form the gate is already fine, but this contract must be re-derived deliberately" % rel)

        raw = _raw(tmp_path)
        art = raw / "g11_seed42.json"
        art.write_text('{"score": 1}', encoding="utf-8")
        assert prov_gate.check([str(art)]), "control failed: bare artifact not flagged"
        Path(art).with_suffix(".cmd.json").write_text('{"cmd": ["python", "-m", "x"]}', encoding="utf-8")
        assert prov_gate.check([str(art)]) == [], "SEAM BROKEN: gate rejects the replace-form .cmd.json sidecar"

    def test_neither_sidecar_form_is_itself_treated_as_an_artifact(self, tmp_path, monkeypatch):
        """A sidecar is a .json under raw/. If the gate demanded provenance FOR it, every stamped run would
        block the commit -- and the door skips .prov.json when walking, so it would never be satisfied."""
        raw, paths, _made = self._stamp(tmp_path, monkeypatch, {"b.json": '{"score": 1}'})
        appended = raw / "b.json.prov.json"
        replaced = raw / "b.cmd.json"
        replaced.write_text('{"cmd": ["x"]}', encoding="utf-8")
        for p in (appended, replaced):
            assert p.exists()
            assert prov_gate._is_artifact(str(p)) is False, "%s is treated as an artifact needing provenance" % p.name
            assert prov_gate.check([str(p)]) == [], "gate flagged the sidecar %s itself" % p.name

    def test_writer_never_stamps_a_sidecar_of_a_sidecar(self, tmp_path, monkeypatch):
        """The door's skip rule (`fn.endswith('.prov.json')`) must cover the name the door itself produces."""
        raw, paths, _ = self._stamp(tmp_path, monkeypatch, {"c.json": '{"score": 1}'})
        prov_door._stamp_outputs(prov_door._record_start())      # second run, sidecars now exist and are fresh
        names = sorted(p.name for p in raw.iterdir() if p.is_file())
        assert names == ["c.json", "c.json.prov.json"], "recursive stamping: %s" % names

    def test_an_unrecognised_sidecar_suffix_is_rejected(self, tmp_path):
        """POWER CONTROL for this whole class: acceptance above is agreement on a NAME, not blanket acceptance
        of any sibling file. If this passes, the seam-1 tests are tautological."""
        raw = _raw(tmp_path)
        art = raw / "d.json"
        art.write_text('{"score": 1}', encoding="utf-8")
        (raw / "d.json.meta.json").write_text('{"note": "not a sanctioned sidecar"}', encoding="utf-8")
        (raw / "d.sidecar.json").write_text('{"note": "nor is this"}', encoding="utf-8")
        assert prov_gate.check([str(art)]), "gate accepts ANY sibling .json as provenance -- it is not checking a name"


# ==========================================================================================================
# SEAM 2 — queue files: tools/pool_queue.sh + tools/queue_add.sh (writers) <-> lane_check / lane_starvation
# ==========================================================================================================
def _producer_queue_paths() -> set:
    """Every research/queue/*.queue file a producer can write, derived from the producers themselves.

    Scoped to CODE lines (the shell comments in these scripts name queues in prose, and a prose mention is not
    a write). `${LANE}` templates are expanded from the script's own usage line, so a new lane added to a
    producer surfaces here rather than silently escaping both consumers. The caller pins the resulting set
    exactly, so ANY drift -- a new queue, a renamed one -- fails loudly instead of quietly widening.
    """
    paths = set()
    for rel in ("tools/pool_queue.sh", "tools/queue_add.sh", "tools/lane_dispatch.sh"):
        src = "\n".join(_code_lines(rel))
        for m in re.finditer(r"research/queue/([A-Za-z0-9_{}$]+)\.queue", src):
            token = m.group(1)
            if token.startswith("${"):                                   # a ${LANE} template
                lanes = re.search(r"usage: \S+ <([a-z|]+)>", src)
                assert lanes, "%s has a ${LANE} queue template but no <a|b> usage line to expand it" % rel
                paths.update("research/queue/%s.queue" % ln for ln in lanes.group(1).split("|"))
            else:
                paths.add("research/queue/%s.queue" % token)
    assert paths, "no queue producers found -- the extraction, not the seam, is broken"
    return paths


class TestSeam2BothQueuesAreRead:

    def test_every_queue_a_producer_writes_is_read_by_lane_check(self, tmp_path, monkeypatch):
        expected = _producer_queue_paths()
        assert expected == {"research/queue/gpu.queue", "research/queue/pool.queue"}, \
            "the set of producer queues changed to %s -- update BOTH consumers deliberately" % sorted(expected)

        markers = {}
        for rel in sorted(expected):
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            markers[rel] = "python -m research.runners.marker_%s" % Path(rel).stem
            p.write_text(markers[rel] + "\n", encoding="utf-8")

        monkeypatch.setattr(lane_check, "ROOT", str(tmp_path))
        jobs = lane_check._queue_jobs()
        for rel, marker in markers.items():
            assert marker in jobs, "SEAM BROKEN: lane_check does not read %s (jobs=%s)" % (rel, jobs)

    def test_every_queue_a_producer_writes_is_read_by_the_lane_starvation_gate(self, tmp_path, monkeypatch):
        """The same seam, second consumer. lane_starvation BLOCKS commits; reading one queue of two would
        block on starvation that was already fixed."""
        expected = _producer_queue_paths()
        for rel in sorted(expected):
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("python -m research.runners.marker_%s\n" % Path(rel).stem, encoding="utf-8")
        monkeypatch.setattr(lane_gate, "_ROOT", str(tmp_path))
        monkeypatch.setattr(lane_gate, "subprocess", _FakePs(""))
        lines = "\n".join(lane_gate._work_lines())
        for rel in sorted(expected):
            assert "marker_%s" % Path(rel).stem in lines, \
                "SEAM BROKEN: lane_starvation does not read %s" % rel

    def test_the_pool_queue_line_format_the_producer_emits_parses_to_the_bare_command(self, tmp_path, monkeypatch):
        """pool_queue.sh emits '<epoch>\\t<cmd>  #checked:<reason>'. lane_check must return <cmd> alone."""
        fmt = _unescape(_printf_format("tools/pool_queue.sh", "#checked:", '>> "$Q"'))
        cmd = "SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._emerge72_construction_registry_derisk --seeds 42"
        line = fmt % (str(int(time.time())), cmd, "corpus: nothing covers this axis")
        assert line.endswith("\n") and "\t" in line, "producer format lost its tab/newline: %r" % line

        q = tmp_path / "research" / "queue" / "pool.queue"
        q.parent.mkdir(parents=True)
        q.write_text(line, encoding="utf-8")
        monkeypatch.setattr(lane_check, "ROOT", str(tmp_path))
        jobs = lane_check._queue_jobs()
        assert jobs == [cmd], "SEAM BROKEN: pool.queue line parsed to %r, expected %r" % (jobs, [cmd])

    def test_the_gpu_queue_line_format_the_producer_emits_parses_to_the_bare_command(self, tmp_path, monkeypatch):
        """queue_add.sh emits '<cmd>  #checked:<reason>' -- NO timestamp. The same reader must handle both."""
        fmt = _unescape(_printf_format("tools/queue_add.sh", "#checked:", '>> "$Q"'))
        cmd = ".venv/bin/python -m research.runners._gap4_deep_credit_derisk --seeds 42"
        line = fmt % (cmd, "new-config")
        assert "\t" not in line, "queue_add.sh gained a tab field; lane_check's split changes meaning"

        q = tmp_path / "research" / "queue" / "gpu.queue"
        q.parent.mkdir(parents=True)
        q.write_text(line, encoding="utf-8")
        monkeypatch.setattr(lane_check, "ROOT", str(tmp_path))
        assert lane_check._queue_jobs() == [cmd], "SEAM BROKEN: gpu.queue line parsed to %r" % lane_check._queue_jobs()

    def test_a_pool_queued_job_classifies_into_its_roadmap_lane(self, tmp_path, monkeypatch):
        """THE ACTUAL 2026-07-31 DEFECT: work staged to the pool read as 'no CPU lane served'."""
        fmt = _unescape(_printf_format("tools/pool_queue.sh", "#checked:", '>> "$Q"'))
        q = tmp_path / "research" / "queue" / "pool.queue"
        q.parent.mkdir(parents=True)
        q.write_text(fmt % (str(int(time.time())),
                            "SIM_BACKEND=numpy .venv/bin/python -u -m research.runners."
                            "_emerge72_construction_registry_derisk --seeds 42 43 44", "banked: none"),
                     encoding="utf-8")
        monkeypatch.setattr(lane_check, "ROOT", str(tmp_path))
        jobs = lane_check._queue_jobs()
        assert jobs, "no job read back"
        assert lane_check.classify(jobs[0]) == ["E · Language"], \
            "pool-queued lane-E work classified as %s" % lane_check.classify(jobs[0])

    def test_dropping_the_pool_queue_hides_the_job(self, tmp_path, monkeypatch):
        """POWER CONTROL: proves the pool.queue assertions above are load-bearing rather than satisfied by the
        gpu.queue read. This is the exact blindness that shipped."""
        base = tmp_path / "research" / "queue"
        base.mkdir(parents=True)
        (base / "gpu.queue").write_text("python -m research.runners.gpu_marker\n", encoding="utf-8")
        monkeypatch.setattr(lane_check, "ROOT", str(tmp_path))
        assert "python -m research.runners.pool_marker" not in lane_check._queue_jobs()
        (base / "pool.queue").write_text("%d\tpython -m research.runners.pool_marker  #checked:x\n"
                                         % int(time.time()), encoding="utf-8")
        assert "python -m research.runners.pool_marker" in lane_check._queue_jobs()


# ==========================================================================================================
# SEAM 3 — dispatch.log: tools/pool_autodispatch.sh (writer, via systemd) <-> tools/gates/lane_starvation.py
# ==========================================================================================================
class TestSeam3DispatchLogFormatAndLocation:

    DISPATCH_REL = "research/queue/dispatch.log"

    def _producer_line(self, node: str, job: str) -> str:
        """Execute the dispatcher's OWN echo line to produce a log entry -- no re-implementation."""
        # The dispatcher prints three "[pool-dispatch]" lines (refusal, startup, dispatch); only the one
        # carrying "$NODE <- $JOB" is the record the consumers parse.
        echo = _line_starting("tools/pool_autodispatch.sh", 'echo "[pool-dispatch]', contains="<- $JOB")
        r = _bash('NODE=%s; JOB=%s; %s' % (node, json.dumps(job), echo))
        assert r.returncode == 0, r.stderr
        return r.stdout

    def test_the_consumers_split_on_the_delimiter_the_producer_emits(self):
        job = "SIM_BACKEND=numpy python -m research.runners._affect_x --seeds 42"
        produced = self._producer_line("pool41", job)

        # CONSUMER 1: the starvation gate. Take the delimiter from ITS OWN code, do not hard-code it here --
        # a hard-coded copy would keep passing while the consumer drifted (M19 in the mutation check).
        split_line = [l for l in _in_code("tools/gates/lane_starvation.py", "split(") if "for l in tail" in l]
        assert len(split_line) == 1, "lane_starvation's dispatch-log split changed: %s" % split_line
        delim = re.search(r'split\("([^"]+)"', split_line[0])
        assert delim, "could not extract the delimiter from: %s" % split_line[0]
        delim = delim.group(1)
        assert produced.count(delim) == 1, \
            "SEAM BROKEN: the gate splits on %r, which appears %d times in the producer's line %r" % (
                delim, produced.count(delim), produced)
        # EXACT equality, not endswith(). `endswith` cannot tell "the delimiter was stripped" from "nothing was
        # stripped, so the whole log line came back" -- which is precisely how a delimiter mutation slipped
        # through this assertion the first time it was written.
        assert produced.split(delim, 1)[1].strip() == job, \
            "SEAM BROKEN: gate parse yields %r, producer dispatched %r" % (produced.split(delim, 1)[1].strip(), job)

        # CONSUMER 2: the puller, which re-attaches provenance to pulled pool artifacts. Execute ITS OWN sed
        # expression rather than a copy of it -- a hard-coded copy here would test this file against itself.
        # Three lines mention dispatch.log; two are JSON strings inside the embedded python heredoc. The one
        # that PARSES the log is the one that also pipes through sed.
        pull_line = [l for l in _in_code("tools/pull_pool_results.sh", "dispatch.log") if "sed " in l]
        assert len(pull_line) == 1, "pull_pool_results.sh's dispatch.log parse changed: %s" % pull_line
        sed_expr = re.search(r"sed\s+'([^']+)'", pull_line[0])
        assert sed_expr, "could not extract the puller's sed expression from: %s" % pull_line[0]
        r = _bash("printf '%%s' %s | sed %s" % (json.dumps(produced.rstrip("\n")),
                                                json.dumps(sed_expr.group(1))))
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == job, (
            "SEAM BROKEN: the puller's own sed (%r) yields %r from the producer's line, expected %r. A sed that "
            "matches nothing returns the line UNCHANGED, so only exact equality detects a delimiter change."
            % (sed_expr.group(1), r.stdout.strip(), job))

    def test_the_log_location_agrees_between_producer_and_consumer(self, tmp_path, monkeypatch):
        """Consumer half, behaviourally: put the file exactly where the producer's redirect points and the
        gate must see it. A rename on either side fails here."""
        d = tmp_path / "research" / "queue"
        d.mkdir(parents=True)
        (d / "dispatch.log").write_text(self._producer_line("pool40", "python -m research.runners._affect_x"),
                                        encoding="utf-8")
        monkeypatch.setattr(lane_gate, "_ROOT", str(tmp_path))
        monkeypatch.setattr(lane_gate, "subprocess", _FakePs(""))
        assert any("_affect_x" in l for l in lane_gate._work_lines()), \
            "SEAM BROKEN: gate does not read %s" % self.DISPATCH_REL

    def test_the_systemd_redirect_points_at_the_file_the_gate_reads(self):
        """Producer half. The redirect lives in a unit file OUTSIDE the repo, which is precisely why it can
        drift unnoticed -- nothing in a `git diff` shows it."""
        unit = Path.home() / ".config" / "systemd" / "user" / "pool-dispatch.service"
        if not unit.exists():
            pytest.skip("pool-dispatch.service is not installed on this machine")
        text = unit.read_text(encoding="utf-8")
        assert "tools/pool_autodispatch.sh" in text, "the unit no longer runs the dispatcher this seam assumes"
        targets = _systemd_redirect_targets(text)
        assert targets, "the unit does not append its stdout anywhere; dispatch.log would never be written"
        expected = os.path.join(lane_gate._ROOT, "research", "queue", "dispatch.log")
        assert all(t == expected for t in targets), \
            "SEAM BROKEN: unit writes %s, gate reads %s" % (targets, expected)

    def test_the_systemd_redirect_parser_can_fail(self):
        """POWER CONTROL for the assertion above, which the shadow-repo mutation harness cannot reach (the
        unit file lives outside the repo, so a shadow copy's paths never match it by construction)."""
        good = ("[Service]\nExecStart=/usr/bin/bash %s/tools/pool_autodispatch.sh\n"
                "StandardOutput=append:%s/research/queue/dispatch.log\n"
                "StandardError=append:%s/research/queue/dispatch.log\n"
                % (lane_gate._ROOT, lane_gate._ROOT, lane_gate._ROOT))
        expected = os.path.join(lane_gate._ROOT, "research", "queue", "dispatch.log")
        assert _systemd_redirect_targets(good) == [expected, expected]
        renamed = good.replace("dispatch.log", "dispatch2.log")
        assert _systemd_redirect_targets(renamed) != [expected, expected], "a renamed target parsed as identical"
        assert _systemd_redirect_targets(good.replace("StandardOutput=append:", "StandardOutput=journal\n#")) \
            != [expected, expected], "a unit that stopped appending still parsed as appending"
        assert _systemd_redirect_targets("[Service]\nExecStart=/bin/true\n") == [], \
            "a unit with no append redirect must parse as no targets, not silently pass"

    def test_pool_dispatched_work_serves_its_lane_with_no_local_process(self, tmp_path, monkeypatch):
        """THE DEFECT: a dispatched job leaves the queue and runs on a remote node, so local `ps` sees nothing.
        A FRESH dispatch record must count as serving its lane."""
        d = tmp_path / "research" / "queue"
        d.mkdir(parents=True)
        runners = ["_affect_state_region_derisk", "_curiosity_seek_learn_onbridge_derisk",
                   "_self_schema_region_derisk", "_b1_v1_selforg_onbridge_derisk",
                   "_emerge72_construction_registry_derisk"]
        log = "".join(self._producer_line("pool4%d" % (i % 3),
                                          "SIM_BACKEND=numpy python -u -m research.runners.%s --seeds 42" % r)
                      for i, r in enumerate(runners))
        (d / "dispatch.log").write_text(log, encoding="utf-8")

        monkeypatch.setattr(lane_gate, "_ROOT", str(tmp_path))
        monkeypatch.setattr(lane_gate, "_WAIVER", str(tmp_path / "no-such-waiver"))
        monkeypatch.setattr(lane_gate, "subprocess", _FakePs(""))          # nothing running locally
        assert lane_gate.check([]) == [], \
            "SEAM BROKEN: five lanes dispatched to the pool still read as starved: %s" % lane_gate.check([])

        # POWER CONTROL A -- the same log, aged past the freshness window, MUST fire.
        stale = time.time() - (lane_gate.RECENT_DISPATCH_MIN + 10) * 60
        os.utime(d / "dispatch.log", (stale, stale))
        assert lane_gate.check([]), "a dispatch log older than RECENT_DISPATCH_MIN did not read as starvation"

        # POWER CONTROL B -- no log at all MUST fire, or the gate cannot detect anything.
        (d / "dispatch.log").unlink()
        assert lane_gate.check([]), "with no queue, no ps and no dispatch log the gate still passed"


# ==========================================================================================================
# SEAM 4 — the `#checked:` token: tools/pool_queue.sh (emits) <-> tools/pool_autodispatch.sh (requires+strips)
# ==========================================================================================================
class TestSeam4CheckedTokenStripping:

    TOKEN = "#checked:"

    def test_the_producer_emits_the_token_both_consumers_require(self):
        fmt = _printf_format("tools/pool_queue.sh", self.TOKEN, '>> "$Q"')
        assert self.TOKEN in fmt, "pool_queue.sh no longer emits %r; every dispatch would be BLOCKED" % self.TOKEN
        for rel in ("tools/pool_autodispatch.sh", "tools/lane_dispatch.sh"):
            assert _in_code(rel, '*"%s"*)' % self.TOKEN), \
                "%s no longer gates on %r in CODE -- producer and consumer disagree" % (rel, self.TOKEN)

    def test_the_consumers_own_extraction_recovers_the_command_without_the_token(self, tmp_path):
        """Run pool_autodispatch.sh's ACTUAL awk + parameter-expansion lines against a queue file written in
        pool_queue.sh's ACTUAL printf format. Neither side is re-implemented here."""
        fmt = _unescape(_printf_format("tools/pool_queue.sh", self.TOKEN, '>> "$Q"'))
        cmd = "SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._b1_v1_selforg_onbridge_derisk --seeds 42 43"
        q = tmp_path / "pool.queue"
        q.write_text(fmt % (str(int(time.time())), cmd, "corpus: no prior run at these seeds"), encoding="utf-8")

        awk_line = _line_starting("tools/pool_autodispatch.sh", "job=$(awk")
        strip_line = _line_starting("tools/pool_autodispatch.sh", 'job="${job%%')
        raw_r = _bash('set -uo pipefail\nQUEUE=%s\ncutoff=0\n%s\nprintf "%%s" "$job"'
                      % (json.dumps(str(q)), awk_line))
        assert raw_r.returncode == 0, raw_r.stderr
        assert self.TOKEN in raw_r.stdout, \
            "the consumer's awk did not even retrieve the token -- the guard could never fire: %r" % raw_r.stdout

        r = _bash('set -uo pipefail\nQUEUE=%s\ncutoff=0\n%s\n%s\nprintf "%%s" "$job"'
                  % (json.dumps(str(q)), awk_line, strip_line))
        assert r.returncode == 0, r.stderr
        assert self.TOKEN not in r.stdout, "SEAM BROKEN: the token survived into the executed command: %r" % r.stdout
        assert r.stdout.strip() == cmd, "SEAM BROKEN: recovered %r, producer wrote %r" % (r.stdout.strip(), cmd)

    def test_an_unstripped_token_breaks_the_brace_group_wrapper(self, tmp_path):
        """POWER CONTROL + the actual failure mode. The exit-status wrapper runs the job inside `{ $JOB; }`; a
        surviving `#` comments out the closing `; }`, giving an unterminated brace group -- a SYNTAX error, so
        neither the job nor the status printf ever runs. Six jobs died in exactly this way while the dispatch
        log reported them launched."""
        # Scoped to CODE and to the line that actually executes: an identical string sits in a comment four
        # lines above, and a whole-file `in` check passed with the wrapper deleted (found by mutating it).
        wrapper = _in_code("tools/pool_autodispatch.sh", "{ $JOB; }")
        assert len(wrapper) == 1 and "autodispatch.out" in wrapper[0], (
            "the dispatcher no longer executes the job inside a brace group, so the hazard this strip defends "
            "against has changed -- re-derive the contract instead of deleting it. Found: %s" % wrapper)

        cmd = "python -m research.runners._b1_v1_selforg_onbridge_derisk --seeds 42"
        unstripped = "%s  %s%s" % (cmd, self.TOKEN, "corpus: nothing prior")
        bad = _bash("bash -n -c %s" % json.dumps("{ %s; } > /dev/null 2>&1" % unstripped))
        good = _bash("bash -n -c %s" % json.dumps("{ %s; } > /dev/null 2>&1" % cmd))
        assert bad.returncode != 0, \
            "the unstripped token no longer breaks the brace group; this control has lost its power"
        assert "unexpected end of file" in bad.stderr or "syntax error" in bad.stderr, bad.stderr
        assert good.returncode == 0, "the STRIPPED command does not parse inside the wrapper: %s" % good.stderr

    def test_the_gpu_dispatcher_legitimately_does_not_strip_because_it_uses_eval(self):
        """The two consumers differ ON PURPOSE, and the difference is load-bearing: lane_dispatch.sh runs
        `eval "$LINE"` with nothing after it, so the token degrades to an inert trailing comment. If that ever
        becomes a brace group without a strip, this fails -- which is the whole point of pinning it."""
        assert _in_code("tools/lane_dispatch.sh", 'eval "$LINE"'), \
            "lane_dispatch.sh no longer evals the raw line in code; it must now STRIP the token itself"
        assert not _in_code("tools/lane_dispatch.sh", "{ $LINE; }"), \
            "lane_dispatch.sh now brace-wraps the line but never strips %r -- the seam-4 defect, second site" % self.TOKEN
        r = _bash('LINE=%s; eval "$LINE"' % json.dumps("echo dispatched  %sreason" % self.TOKEN))
        assert r.returncode == 0 and r.stdout.strip() == "dispatched", \
            "the trailing token is NOT inert under eval (rc=%d, out=%r, err=%r)" % (r.returncode, r.stdout, r.stderr)
