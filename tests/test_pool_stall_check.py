"""tools/pool_stall_check.py -- catches a live-but-stalled pool (2026-09-25: 7 of 17 D6 processes were
DUPLICATES of cells whose output had already landed at the same pinned revision, running 7-26h, while the
heartbeat read SATURATED). Pure-logic tests need no ssh; the end-to-end tests fake `ssh` on PATH the same way
tests/test_pool_autodispatch_workflow.py already does for the shell dispatcher, never touching a real node.
"""
from __future__ import annotations

import base64
import json
import os
import re
import stat
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import pool_stall_check as psc  # noqa: E402


# --------------------------------------------------------------------------------------------------- node list

def test_get_pool_nodes_default(monkeypatch):
    monkeypatch.delenv("POOL_NODES", raising=False)
    monkeypatch.setenv("POOL_EXTRA_NODES_FILE", "/does/not/exist")
    assert psc.get_pool_nodes() == ["pool40", "pool41", "pool42"]


def test_get_pool_nodes_honours_env_override(monkeypatch):
    monkeypatch.setenv("POOL_NODES", "poolA poolB")
    monkeypatch.setenv("POOL_EXTRA_NODES_FILE", "/does/not/exist")
    assert psc.get_pool_nodes() == ["poolA", "poolB"]


def test_get_pool_nodes_merges_extra_nodes_file_deduped(tmp_path, monkeypatch):
    extra = tmp_path / "extra_nodes"
    extra.write_text("# comment\npool1\npool40\n\n")
    monkeypatch.setenv("POOL_NODES", "pool40 pool41")
    monkeypatch.setenv("POOL_EXTRA_NODES_FILE", str(extra))
    assert psc.get_pool_nodes() == ["pool40", "pool41", "pool1"]   # pool40 not duplicated


# ------------------------------------------------------------------------------------------- signature parsing

def test_job_signature_extracts_module_and_key_args():
    text = ("cd ~/derisk-pool/revisions/abc1234 && POOL_CHECKED_REASON=x python3 -m "
            "research.runners.d6_capacity_curve --n-facts 500 --family colors --out research/findings/raw/x.json")
    assert psc.job_signature(text) == ("d6_capacity_curve", (("family", "colors"), ("n-facts", "500")))


def test_job_signature_none_without_a_runner_module():
    assert psc.job_signature("cd ~/derisk-pool/sim && echo hi") is None
    assert psc.job_signature("") is None
    assert psc.job_signature(None) is None


def test_job_signature_ignores_surrounding_wrapper_text():
    # The SAME logical job appears three ways in this codebase: a raw claims-file line, a JOB_B64-decoded
    # running command, and a completed job_status.log v2 record -- all must resolve to the identical signature.
    bare = "python3 -m research.runners.foo --n-facts 10"
    claims_line = "cd ~/derisk-pool/sim && python3 -m research.runners.foo --n-facts 10  #checked:reason"
    running = "POOL_CHECKED_REASON=x cd ~/derisk-pool/revisions/deadbeef && python3 -m research.runners.foo --n-facts 10"
    sig = psc.job_signature(bare)
    assert psc.job_signature(claims_line) == sig
    assert psc.job_signature(running) == sig


def test_pinned_revision():
    assert psc.pinned_revision("cd ~/derisk-pool/revisions/abc1234 && x") == "abc1234"
    assert psc.pinned_revision("cd ~/derisk-pool/sim && x") is None
    assert psc.pinned_revision(None) is None


def test_declared_out_path_space_and_equals_forms():
    assert psc.declared_out_path("... --out research/findings/raw/x.json ...") == "research/findings/raw/x.json"
    assert psc.declared_out_path("... --json=research/findings/raw/y.json ...") == "research/findings/raw/y.json"
    assert psc.declared_out_path("... --output 'research/findings/raw/z.json' ...") == "research/findings/raw/z.json"
    assert psc.declared_out_path("no output flag here") is None


# ------------------------------------------------------------------------------------------- parse_probe_output

def test_parse_probe_output_groups_duplicate_pids_by_job_id_taking_max_etimes():
    text_b64 = base64.b64encode(b"python3 -m research.runners.foo").decode()
    text = "\n".join([
        "RUN\t100\tjobA\t50\t" + text_b64,
        "RUN\t101\tjobA\t55\t" + text_b64,   # same job, a later-forked descendant -> smaller/larger etimes
        "===STATUS===",
        "v2\t1000\t0\t" + text_b64,
        "not-a-v2-line",
        "v2\tnotanumber\t0\t" + text_b64,     # malformed epoch -- must be skipped, not raise
    ])
    running, completions = psc.parse_probe_output("pool40", text)
    assert set(running.keys()) == {"jobA"}
    rec = running["jobA"]
    assert rec["pids"] == {"100", "101"}
    assert rec["max_etimes"] == 55
    assert rec["job_text"] == "python3 -m research.runners.foo"
    assert completions == [(1000, 0, "python3 -m research.runners.foo")]


def test_parse_probe_output_empty_and_malformed_text_never_raises():
    running, completions = psc.parse_probe_output("pool40", "")
    assert running == {} and completions == []
    running, completions = psc.parse_probe_output("pool40", "garbage\n\x00\x01\nRUN\tonly-two-fields")
    assert running == {} and completions == []


# ------------------------------------------------------------------------------------------ historical duration

def test_historical_durations_pairs_completion_with_nearest_prior_claim():
    now = 1_000_000
    text = "cd ~/derisk-pool/sim && python3 -m research.runners.slow --n-facts 5"
    claims = [(now - 500, text + "  #checked:x"), (now - 200, text + "  #checked:x")]
    completions = [(now, 0, text)]
    durations = psc.historical_durations(completions, claims)
    sig = psc.job_signature(text)
    # nearest prior claim is (now-200) -> duration 200, NOT the earlier (now-500) one
    assert durations[sig] == [200]


def test_historical_durations_ignores_a_claim_that_postdates_the_completion():
    now = 1_000_000
    text = "python3 -m research.runners.slow --n-facts 5"
    claims = [(now + 50, text)]   # claim is AFTER the completion -- cannot be its start
    completions = [(now, 0, text)]
    assert psc.historical_durations(completions, claims) == {}


def test_overdue_verdict_unknown_without_history_never_silently_ok():
    sig = ("some_runner", ())
    assert psc.overdue_verdict(999999, sig, {}) == "UNKNOWN"
    assert psc.overdue_verdict(999999, None, {"x": [10]}) == "UNKNOWN"


def test_overdue_verdict_flags_more_than_3x_median_and_passes_under_it():
    sig = ("r", ())
    durations = {sig: [100, 100, 100]}
    assert psc.overdue_verdict(150, sig, durations) == "OK"
    assert psc.overdue_verdict(301, sig, durations) == "OVERDUE"


# ------------------------------------------------------------------------------------------- DUP-OF-LANDED

def _write_prov(path, git_sha):
    with open(path + ".prov.json", "w") as f:
        json.dump({"git_sha": git_sha}, f)


def test_dup_of_landed_true_when_short_local_sha_is_a_prefix_of_the_pinned_full_sha(tmp_path):
    out_rel = "research/findings/raw/x.json"
    full = os.path.join(str(tmp_path), out_rel)
    os.makedirs(os.path.dirname(full))
    open(full, "w").close()
    _write_prov(full, "abc1234")           # short-form git_sha, as runners/__init__.py records by default
    pinned = "abc1234" + "d" * 33          # a 40-char sha this short one is a genuine prefix of
    assert psc.check_dup_of_landed(str(tmp_path), out_rel, pinned) is True


def test_dup_of_landed_false_when_shas_disagree(tmp_path):
    out_rel = "research/findings/raw/x.json"
    full = os.path.join(str(tmp_path), out_rel)
    os.makedirs(os.path.dirname(full))
    open(full, "w").close()
    _write_prov(full, "abc1234")
    assert psc.check_dup_of_landed(str(tmp_path), out_rel, "f" * 40) is False


def test_dup_of_landed_false_when_artifact_or_sidecar_missing(tmp_path):
    assert psc.check_dup_of_landed(str(tmp_path), "research/findings/raw/missing.json", "a" * 40) is False


def test_dup_of_landed_none_when_not_applicable():
    assert psc.check_dup_of_landed("/repo", None, "a" * 40) is None       # no --out on the command
    assert psc.check_dup_of_landed("/repo", "research/findings/raw/x.json", None) is None   # unpinned job


def test_dup_of_landed_none_on_path_escaping_repo_root(tmp_path):
    assert psc.check_dup_of_landed(str(tmp_path), "../../etc/passwd", "a" * 40) is None


def test_kill_command_lists_all_pids_sorted():
    cmd = psc.kill_command("pool40", {"200", "5", "100"})
    assert cmd == 'ssh -n pool40 "kill -TERM 5 100 200"'


# ---------------------------------------------------------------------------------------------- end-to-end (fake ssh)

def _b64(text):
    return base64.b64encode(text.encode()).decode()


def _build_probe_text(running=(), completions=()):
    lines = []
    for pid, jid, etimes, text in running:
        lines.append("RUN\t%s\t%s\t%s\t%s" % (pid, jid, etimes, _b64(text)))
    lines.append("===STATUS===")
    for epoch, rc, text in completions:
        lines.append("v2\t%s\t%s\t%s" % (epoch, rc, _b64(text)))
    return "\n".join(lines) + "\n"


def _write_fake_ssh(tmp_path, node_outputs, unreachable=()):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fixdir = tmp_path / "ssh_out"
    fixdir.mkdir(exist_ok=True)
    for node, text in node_outputs.items():
        (fixdir / node).write_text(text)
    unreach = tmp_path / "ssh_unreachable.txt"
    unreach.write_text("\n".join(unreachable) + "\n")
    stub = bin_dir / "ssh"
    # The node is always the SECOND-TO-LAST argv element for every call this module makes (script is last) --
    # `-n [-F cfg] -o BatchMode=yes -o ConnectTimeout=N <node> <script>`.
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'node="${@: -2:1}"\n'
        'if grep -qxF "$node" "%s" 2>/dev/null; then exit 255; fi\n'
        'f="%s/$node"\n'
        'if [ -f "$f" ]; then cat "$f"; fi\n'
        "exit 0\n" % (unreach, fixdir)
    )
    st = stub.stat()
    stub.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


def test_check_all_end_to_end_flags_dup_and_overdue_and_reports_unreachable(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "research" / "findings" / "raw").mkdir(parents=True)
    out_rel = "research/findings/raw/d6_result.json"
    (repo / out_rel).write_text("{}")
    sha_full = "a" * 40
    _write_prov(str(repo / out_rel), sha_full[:9])   # short local git_sha, prefix of the job's pinned full sha

    now = int(time.time())
    dup_text = ("cd ~/derisk-pool/revisions/%s && POOL_CHECKED_REASON=x python3 -m "
                "research.runners.d6_capacity_curve --n-facts 500 --family colors --out %s" % (sha_full, out_rel))
    overdue_text = ("cd ~/derisk-pool/sim && python3 -m research.runners.slow_runner --n-facts 10 "
                    "--out research/findings/raw/other.json")
    ok_text = ("cd ~/derisk-pool/sim && python3 -m research.runners.fast_runner --n-facts 10 "
               "--out research/findings/raw/fast.json")
    unknown_text = "cd ~/derisk-pool/sim && python3 -m research.runners.never_seen_before --n-facts 1"

    pool40_out = _build_probe_text(running=[("111", "jobA", 3600, dup_text), ("444", "jobD", 10, unknown_text)])
    pool41_out = _build_probe_text(
        running=[("222", "jobB", 5000, overdue_text), ("333", "jobC", 50, ok_text)],
        completions=[(now - 100000, 0, overdue_text), (now - 90000, 0, ok_text)],
    )
    bin_dir = _write_fake_ssh(tmp_path, {"pool40": pool40_out, "pool41": pool41_out}, unreachable=["pool99"])

    claims_path = tmp_path / "pool.queue.claims"
    claims_path.write_text(
        "%d\t%s  #checked:x\n%d\t%s  #checked:x\n" % (now - 100300, overdue_text, now - 90100, ok_text)
    )

    monkeypatch.setenv("PATH", "%s:%s" % (bin_dir, os.environ.get("PATH", "")))
    report = psc.check_all(nodes=["pool40", "pool41", "pool99"], root=str(repo),
                            claims_path=str(claims_path), now=now)

    assert report["unreachable"] == ["pool99"]
    assert report["n_running"] == 4
    assert report["n_dup"] == 1
    assert report["n_overdue"] == 1
    # jobA (no completion history for its signature) AND jobD (never-seen signature) both read UNKNOWN on the
    # OVERDUE axis -- DUP-OF-LANDED and OVERDUE are independent signals, never conflated.
    assert report["n_unknown_overdue"] == 2

    by_id = {r["job_id"]: r for r in report["running"]}
    assert by_id["jobA"]["dup_of_landed"] is True
    assert by_id["jobA"]["overdue"] == "UNKNOWN"
    assert by_id["jobA"]["pinned_sha"] == sha_full
    assert by_id["jobB"]["overdue"] == "OVERDUE"     # 5000s vs history ~300s (3x -> 900s threshold)
    assert by_id["jobC"]["overdue"] == "OK"          # 50s vs history ~100s (3x -> 300s threshold)
    assert by_id["jobD"]["overdue"] == "UNKNOWN"

    flagged_ids = {r["job_id"] for r in report["flagged"]}
    assert flagged_ids == {"jobA", "jobB"}
    assert "DUP-OF-LANDED" in report["summary_line"]
    assert "OVERDUE" in report["summary_line"]
    assert "kill -TERM 111" in by_id["jobA"]["kill_cmd"]
    assert "pool40" in by_id["jobA"]["kill_cmd"]


def test_check_all_all_nodes_unreachable_reports_cleanly_never_raises(tmp_path, monkeypatch):
    bin_dir = _write_fake_ssh(tmp_path, {}, unreachable=["pool40", "pool41"])
    monkeypatch.setenv("PATH", "%s:%s" % (bin_dir, os.environ.get("PATH", "")))
    report = psc.check_all(nodes=["pool40", "pool41"], root=str(tmp_path),
                            claims_path=str(tmp_path / "no-such-claims"))
    assert report["unreachable"] == ["pool40", "pool41"]
    assert report["running"] == []
    assert report["flagged"] == []
    assert "clean" in report["summary_line"]


def test_format_row_includes_kill_command():
    row = {
        "node": "pool40", "job_id": "j1", "pids": ["1", "2"], "elapsed_s": 42, "module": "foo",
        "pinned_sha": "abc", "out_path": "x.json", "dup_of_landed": True, "overdue": "OK",
        "kill_cmd": 'ssh -n pool40 "kill -TERM 1 2"',
    }
    line = psc.format_row(row)
    assert "DUP-OF-LANDED" in line
    assert 'kill -TERM 1 2' in line


# =================================================================================== queue-line checks (UNRUNNABLE)
# 2026-09-25 addition: six revision-pinned queue lines sat 7.5h because the revision was never provisioned where
# it could fit, and nothing outside the dispatcher's own per-cycle log line ever surfaced it.

# --------------------------------------------------------------------------------------------------- job_est_gb

def test_job_est_gb_reads_explicit_mem_gb_first():
    assert psc.job_est_gb("cd ~/derisk-pool/sim && mem_gb=8 python3 -m research.runners.foo") == 8


def test_job_est_gb_falls_back_to_memcap_wrapper(monkeypatch):
    monkeypatch.delenv("POOL_RUNNER_MEM_PATH", raising=False)
    text = "bash tools/memcap.sh 12 -- python3 -m research.runners.foo"
    assert psc.job_est_gb(text) == 12


def test_job_est_gb_prefers_mem_gb_over_memcap_when_both_present():
    text = "mem_gb=8 bash tools/memcap.sh 12 -- python3 -m research.runners.foo"
    assert psc.job_est_gb(text) == 8


def test_job_est_gb_falls_back_to_runner_mem_table(tmp_path):
    tsv = tmp_path / "pool_runner_mem.tsv"
    tsv.write_text("# comment\nload_bearing_fraction\t6\nother_runner\t3\n")
    text = "cd ~/derisk-pool/sim && python3 -m research.runners.load_bearing_fraction --n-facts 5"
    assert psc.job_est_gb(text, runner_mem_path=str(tsv)) == 6


def test_job_est_gb_runner_table_env_override(tmp_path, monkeypatch):
    tsv = tmp_path / "pool_runner_mem.tsv"
    tsv.write_text("some_runner\t9\n")
    monkeypatch.setenv("POOL_RUNNER_MEM_PATH", str(tsv))
    text = "cd ~/derisk-pool/sim && python3 -m research.runners.some_runner"
    assert psc.job_est_gb(text) == 9


def test_job_est_gb_default_when_nothing_matches(monkeypatch, tmp_path):
    monkeypatch.setenv("POOL_RUNNER_MEM_PATH", str(tmp_path / "no-such-table.tsv"))
    monkeypatch.delenv("POOL_JOB_EST_GB", raising=False)
    assert psc.job_est_gb("cd ~/derisk-pool/sim && python3 -m research.runners.never_seen") == 1


def test_job_est_gb_default_honours_env_override(monkeypatch):
    monkeypatch.setenv("POOL_JOB_EST_GB", "3")
    assert psc.job_est_gb("no runner module here at all") == 3


def test_job_est_gb_empty_text():
    assert psc.job_est_gb(None) == 1
    assert psc.job_est_gb("") == 1


# --------------------------------------------------------------------------------------------------- load_queue

def test_load_queue_parses_valid_lines_and_skips_malformed(tmp_path):
    q = tmp_path / "pool.queue"
    q.write_text(
        "# a comment\n"
        "\n"
        "1000\tcd ~/derisk-pool/sim && python3 -m research.runners.foo  #checked:x\n"
        "not-a-number\tbad line\n"
        "2000\n"                       # no tab / no command -> malformed
        "3000\tpython3 -m research.runners.bar\n"
    )
    entries = psc.load_queue(str(q))
    assert entries == [
        (1000, "cd ~/derisk-pool/sim && python3 -m research.runners.foo  #checked:x"),
        (3000, "python3 -m research.runners.bar"),
    ]


def test_load_queue_missing_file_returns_empty(tmp_path):
    assert psc.load_queue(str(tmp_path / "no-such-queue")) == []


def test_load_claims_and_load_queue_share_parsing_but_different_files(tmp_path):
    # Regression guard for the load_claims/load_queue refactor: each reads its OWN path, not the other's.
    claims = tmp_path / "pool.queue.claims"
    queue = tmp_path / "pool.queue"
    claims.write_text("111\tclaims-line\n")
    queue.write_text("222\tqueue-line\n")
    assert psc.load_claims(str(claims)) == [(111, "claims-line")]
    assert psc.load_queue(str(queue)) == [(222, "queue-line")]


# ---------------------------------------------------------------------------------------- fix_provision_command

def test_fix_provision_command_format():
    cmd = psc.fix_provision_command("abc1234", ["pool1", "pool2"])
    assert cmd == "bash tools/pool_provision.sh --revision abc1234 --isolated pool1 pool2"


# ------------------------------------------------------------------------ provisioned-marker filename drift guard

def test_provisioned_marker_filename_matches_bash_source():
    # tools/pool_revision_marker.sh is the ONE place the bash dispatcher/queue scripts get this predicate from;
    # this Python module cannot `source` bash, so POOL_REVISION_MARKER_FILE is a duplicated literal -- this test
    # is what stops the two from drifting apart the way pool_autodispatch.sh's revision_available() and
    # pool_queue.sh's `add` once did over the SAME question (see that file's own docstring).
    marker_path = os.path.join(ROOT, "tools", "pool_revision_marker.sh")
    with open(marker_path) as f:
        text = f.read()
    m = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("POOL_REVISION_MARKER_FILE="):
            m = line.split("=", 1)[1].strip('"').strip("'")
            break
    assert m is not None, "tools/pool_revision_marker.sh no longer defines POOL_REVISION_MARKER_FILE"
    assert m == psc.POOL_REVISION_MARKER_FILE


def test_min_avail_gb_default_matches_bash_source():
    # Same drift guard as above, for DEFAULT_MIN_AVAIL_GB (LOW-1 fix): pool_autodispatch.sh's own
    # POOL_MIN_AVAIL_GB default (node_is_idle's MemAvailable floor) must not silently diverge from what this
    # module assumes when computing a line's "capable" ceiling.
    path = os.path.join(ROOT, "tools", "pool_autodispatch.sh")
    with open(path) as f:
        text = f.read()
    m = re.search(r"POOL_MIN_AVAIL_GB:-(\d+)", text)
    assert m is not None, "tools/pool_autodispatch.sh no longer defines a POOL_MIN_AVAIL_GB default"
    assert int(m.group(1)) == psc.DEFAULT_MIN_AVAIL_GB


def test_pool_job_max_age_default_matches_bash_source():
    # Same drift guard, for DEFAULT_POOL_JOB_MAX_AGE_S (MEDIUM-3 fix): every `${POOL_JOB_MAX_AGE:-N}` default in
    # pool_autodispatch.sh must agree with each other AND with this module's own duplicated constant.
    path = os.path.join(ROOT, "tools", "pool_autodispatch.sh")
    with open(path) as f:
        text = f.read()
    matches = set(re.findall(r"POOL_JOB_MAX_AGE:-(\d+)", text))
    assert matches, "tools/pool_autodispatch.sh no longer defines a POOL_JOB_MAX_AGE default"
    assert matches == {str(psc.DEFAULT_POOL_JOB_MAX_AGE_S)}, (
        "pool_autodispatch.sh's POOL_JOB_MAX_AGE default(s) %r disagree with each other or with "
        "pool_stall_check.DEFAULT_POOL_JOB_MAX_AGE_S=%r" % (matches, psc.DEFAULT_POOL_JOB_MAX_AGE_S))


# --------------------------------------------------------------------------- probe_mem_total_gb / check_provisioned
# Unit-level (monkeypatched subprocess.run), not a fake-ssh binary -- these two functions make ONE simple ssh
# call each, and testing the exact returncode/stdout branches directly is more mutation-resistant than routing
# through a shell script that could itself hide a bug the same way.

class _FakeCompleted:
    def __init__(self, returncode, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def test_probe_mem_total_gb_parses_stdout(monkeypatch):
    monkeypatch.setattr(psc.subprocess, "run", lambda *a, **k: _FakeCompleted(0, "15\n"))
    assert psc.probe_mem_total_gb("pool40") == 15


def test_probe_mem_total_gb_nonzero_returncode_is_none(monkeypatch):
    monkeypatch.setattr(psc.subprocess, "run", lambda *a, **k: _FakeCompleted(255, ""))
    assert psc.probe_mem_total_gb("pool40") is None


def test_probe_mem_total_gb_unparseable_stdout_is_none(monkeypatch):
    monkeypatch.setattr(psc.subprocess, "run", lambda *a, **k: _FakeCompleted(0, "garbage\n"))
    assert psc.probe_mem_total_gb("pool40") is None


def test_probe_mem_total_gb_timeout_is_none(monkeypatch):
    def boom(*a, **k):
        raise psc.subprocess.TimeoutExpired(cmd="ssh", timeout=10)
    monkeypatch.setattr(psc.subprocess, "run", boom)
    assert psc.probe_mem_total_gb("pool40") is None


def test_check_provisioned_marker_present_is_true(monkeypatch):
    monkeypatch.setattr(psc.subprocess, "run", lambda *a, **k: _FakeCompleted(0, ""))
    assert psc.check_provisioned("pool40", "abc1234") is True


def test_check_provisioned_marker_absent_is_false(monkeypatch):
    monkeypatch.setattr(psc.subprocess, "run", lambda *a, **k: _FakeCompleted(1, ""))
    assert psc.check_provisioned("pool40", "abc1234") is False


def test_check_provisioned_unreachable_is_none_not_false(monkeypatch):
    # A confident "not provisioned" (False) must never be produced for a node the probe could not even reach --
    # callers (check_queue) treat False as evidence, None as "cannot say".
    monkeypatch.setattr(psc.subprocess, "run", lambda *a, **k: _FakeCompleted(255, ""))
    assert psc.check_provisioned("pool40", "abc1234") is None


def test_check_provisioned_timeout_is_none(monkeypatch):
    def boom(*a, **k):
        raise psc.subprocess.TimeoutExpired(cmd="ssh", timeout=10)
    monkeypatch.setattr(psc.subprocess, "run", boom)
    assert psc.check_provisioned("pool40", "abc1234") is None


# ------------------------------------------------------------------------------------- probe_live_node_budget_gb
# (HIGH-1/MEDIUM fix, 2026-09-25 review -- the capacity_stalled evidence check's live-budget probe)

def _write_fake_autodispatch_script(root):
    tools_dir = root / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / "pool_autodispatch.sh").write_text("#!/usr/bin/env bash\n")


def test_probe_live_node_budget_gb_parses_idle_output(monkeypatch, tmp_path):
    _write_fake_autodispatch_script(tmp_path)
    monkeypatch.setattr(psc.subprocess, "run", lambda *a, **k: _FakeCompleted(0, "idle budget=7GB\n"))
    assert psc.probe_live_node_budget_gb("pool40", root=str(tmp_path)) == 7


def test_probe_live_node_budget_gb_parses_busy_output(monkeypatch, tmp_path):
    _write_fake_autodispatch_script(tmp_path)
    monkeypatch.setattr(psc.subprocess, "run", lambda *a, **k: _FakeCompleted(0, "busy/unreachable (budget=0GB)\n"))
    assert psc.probe_live_node_budget_gb("pool40", root=str(tmp_path)) == 0


def test_probe_live_node_budget_gb_unparseable_stdout_is_none(monkeypatch, tmp_path):
    _write_fake_autodispatch_script(tmp_path)
    monkeypatch.setattr(psc.subprocess, "run", lambda *a, **k: _FakeCompleted(0, "garbage\n"))
    assert psc.probe_live_node_budget_gb("pool40", root=str(tmp_path)) is None


def test_probe_live_node_budget_gb_timeout_is_none(monkeypatch, tmp_path):
    _write_fake_autodispatch_script(tmp_path)

    def boom(*a, **k):
        raise psc.subprocess.TimeoutExpired(cmd="bash", timeout=20)
    monkeypatch.setattr(psc.subprocess, "run", boom)
    assert psc.probe_live_node_budget_gb("pool40", root=str(tmp_path)) is None


def test_probe_live_node_budget_gb_missing_script_is_none(tmp_path):
    # root has no tools/pool_autodispatch.sh at all -- must degrade to None, never raise or shell out.
    assert psc.probe_live_node_budget_gb("pool40", root=str(tmp_path / "no-tools-dir")) is None


def test_all_mem_totals_splits_reachable_and_unreachable(monkeypatch):
    def fake_probe(node, timeout=10, connect_timeout=6):
        return {"pool40": 15, "pool41": 8}.get(node)
    monkeypatch.setattr(psc, "probe_mem_total_gb", fake_probe)
    totals, unreachable = psc.all_mem_totals(["pool40", "pool41", "pool99"])
    assert totals == {"pool40": 15, "pool41": 8}
    assert unreachable == ["pool99"]


# ------------------------------------------------------------------------------------------ check_queue: pure logic
# (mem_totals / provisioned status injected directly via monkeypatch -- no ssh at all)

def test_check_queue_flags_unrunnable_when_no_capable_node_has_the_marker(tmp_path, monkeypatch):
    now = int(time.time())
    sha = "a" * 40
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.settle_a2 "
                 "--out x.json  #checked:x\n" % (now - 3600, sha))
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6:
                         ({"pool1": 32, "pool2": 32, "pool41": 15, "pool42": 15}, []))
    monkeypatch.setattr(psc, "check_provisioned", lambda node, s, timeout=10, connect_timeout=6: False)
    report = psc.check_queue(nodes=["pool1", "pool2", "pool41", "pool42"], queue_path_=str(q), now=now)
    assert report["n_queued"] == 1
    assert len(report["unrunnable"]) == 1
    row = report["unrunnable"][0]
    assert row["pinned_sha"] == sha
    assert row["mem_gb"] == 8
    assert set(row["capable_nodes"]) == {"pool1", "pool2", "pool41", "pool42"}   # all 4 fit 8+2<=15
    assert "pool_provision.sh --revision %s --isolated" % sha in row["fix_cmd"]
    assert report["memory_budget_stalled"] == []
    assert "UNRUNNABLE" in report["summary_line"]


def test_check_queue_not_flagged_unrunnable_when_one_capable_node_has_the_marker(tmp_path, monkeypatch):
    now = int(time.time())
    sha = "b" * 40
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.settle_a2\n"
                 % (now - 3600, sha))
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6:
                         ({"pool1": 32, "pool41": 15}, []))
    monkeypatch.setattr(psc, "check_provisioned",
                         lambda node, s, timeout=10, connect_timeout=6: node == "pool1")
    # HIGH-1 fix (2026-09-25 review): capacity_stalled now requires EVIDENCE the line is actually being skipped
    # -- fake a live --node-budget confirmation (real headroom below what this line needs) rather than letting a
    # real ssh/subprocess call happen from a pure-logic test.
    monkeypatch.setattr(psc, "probe_live_node_budget_gb", lambda node, **k: 0)
    report = psc.check_queue(nodes=["pool1", "pool41"], queue_path_=str(q),
                              claims_path_=str(tmp_path / "no-such-claims"), now=now)
    assert report["unrunnable"] == []
    # HIGH-2 fix (2026-09-25 review): a capable+provisioned node no longer silences the line unconditionally as
    # "clean" -- at 1h old (>= the default membudget_min_age_h) something OTHER than provisioning must be
    # keeping it queued, so it is reported as capacity_stalled instead (the exact "revision on the mini-PCs,
    # 7.5h old" incident state the review replayed, which used to read clean).
    assert len(report["capacity_stalled"]) == 1
    row = report["capacity_stalled"][0]
    assert row["provisioned_node"] == "pool1"
    # MEDIUM fix (2026-09-25 review): once aged into capacity_stalled, the OTHER capable node (pool41, confirmed
    # NOT to have the revision) must be named too -- the old code stopped probing at the first True and never
    # learned this, so the real fix (provision pool41) was never suggested.
    assert row["missing_on_nodes"] == ["pool41"]
    assert "pool41" in row["fix_cmd"]
    assert "stalled on live capacity" in report["summary_line"]


def test_check_queue_not_flagged_before_min_age(tmp_path, monkeypatch):
    now = int(time.time())
    sha = "c" * 40
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.settle_a2\n"
                 % (now - 60, sha))   # 1 minute old, well under the default 30 min
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool1": 32}, []))
    monkeypatch.setattr(psc, "check_provisioned", lambda node, s, timeout=10, connect_timeout=6: False)
    report = psc.check_queue(nodes=["pool1"], queue_path_=str(q), now=now)
    assert report["unrunnable"] == []


def test_check_queue_custom_age_threshold_flags_a_younger_line(tmp_path, monkeypatch):
    now = int(time.time())
    sha = "d" * 40
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.settle_a2\n"
                 % (now - 300, sha))   # 5 minutes old
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool1": 32}, []))
    monkeypatch.setattr(psc, "check_provisioned", lambda node, s, timeout=10, connect_timeout=6: False)
    report = psc.check_queue(nodes=["pool1"], queue_path_=str(q), now=now, unrunnable_min_age_min=2)
    assert len(report["unrunnable"]) == 1


def test_check_queue_memory_budget_stalled_when_no_node_could_ever_fit(tmp_path, monkeypatch):
    now = int(time.time())
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/sim && mem_gb=40 python3 -m research.runners.huge_job\n" % (now - 7200,))
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6:
                         ({"pool40": 15, "pool41": 15}, []))
    report = psc.check_queue(nodes=["pool40", "pool41"], queue_path_=str(q), now=now)
    assert report["unrunnable"] == []
    assert len(report["memory_budget_stalled"]) == 1
    row = report["memory_budget_stalled"][0]
    assert row["mem_gb"] == 40
    # margin = max(reserve(2), min_avail(3)+1) = 4 (LOW-1 fix, 2026-09-25 review) -> ceiling = 15 - 4 = 11
    assert row["max_known_ceiling_gb"] == 11
    assert "over every known node's memory ceiling" in report["summary_line"]


def test_check_queue_capability_respects_os_reserve_gb(tmp_path, monkeypatch):
    # Boundary case: a 15 GB node minus the default margin (max(reserve=2, min_avail=3+1)=4) leaves 11 GB -- a
    # line declaring mem_gb=12 must NOT be counted "capable" here (mutation-caught: a version that checked raw
    # MemTotal without subtracting the margin passed every OTHER check_queue test unchanged, since none of them
    # sat exactly on this boundary).
    now = int(time.time())
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/sim && mem_gb=12 python3 -m research.runners.tight_fit\n" % (now - 7200,))
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool41": 15}, []))
    report = psc.check_queue(nodes=["pool41"], queue_path_=str(q), now=now)
    assert len(report["memory_budget_stalled"]) == 1   # 15 - max(2, 4) = 11 < 12 -> no capable node
    assert report["memory_budget_stalled"][0]["max_known_ceiling_gb"] == 11

    # Overriding BOTH knobs down (reserve=0, min_avail=-1 -> margin=max(0,0)=0, ceiling=15) makes it capable.
    report2 = psc.check_queue(nodes=["pool41"], queue_path_=str(q), now=now, os_reserve_gb=0, min_avail_gb=-1)
    assert report2["memory_budget_stalled"] == []


def test_check_queue_capability_reflects_dispatchers_min_avail_floor(tmp_path, monkeypatch):
    # LOW-1 fix (2026-09-25 review, replay G): the dispatcher's REAL budget also subtracts POOL_MIN_AVAIL_GB
    # (default 3) from MemAvailable, so mem_gb=12 on a bare 15 GB node can NEVER actually be dispatched --
    # under the OLD gb-reserve(2)-only ceiling (13) this read "capable" and the check read clean.
    now = int(time.time())
    sha = "1" * 40
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=12 python3 -m research.runners.settle_a2\n"
                 % (now - 7200, sha))
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool41": 15}, []))
    monkeypatch.setattr(psc, "check_provisioned", lambda *a, **k: False)
    report = psc.check_queue(nodes=["pool41"], queue_path_=str(q), now=now)
    assert report["unrunnable"] == []          # never "capable" in the first place, so not a provisioning stall
    assert len(report["memory_budget_stalled"]) == 1
    assert report["memory_budget_stalled"][0]["mem_gb"] == 12


def test_check_queue_capable_boundary_equality_counts_as_capable(tmp_path, monkeypatch):
    # LOW-2 fix: gb - margin == mem_gb (exact equality) must still count as capable (the dispatcher's own check
    # is `-ge`, not strictly greater) -- mutation-caught: flipping >= to > in the capable-list comprehension
    # passed every other check_queue test, since none of them sat exactly on this boundary.
    now = int(time.time())
    sha = "2" * 40
    q = tmp_path / "pool.queue"
    # default margin = max(2, 3+1) = 4 -> a 15 GB node's ceiling is exactly 11
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=11 python3 -m research.runners.settle_a2\n"
                 % (now - 7200, sha))
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool41": 15}, []))
    monkeypatch.setattr(psc, "check_provisioned", lambda *a, **k: False)
    report = psc.check_queue(nodes=["pool41"], queue_path_=str(q), now=now)
    assert report["memory_budget_stalled"] == []             # exactly-at-boundary must NOT read "too big"
    assert len(report["unrunnable"]) == 1                     # capable, and confirmed not provisioned -> flagged
    assert report["unrunnable"][0]["capable_nodes"] == ["pool41"]


def test_check_queue_age_gate_boundary_equality_is_old_enough(tmp_path, monkeypatch):
    # LOW-2 fix: age_s == unrunnable_min_age_s (exact equality) must be treated as "old enough" (the gate is
    # `age_s < threshold: continue`, not `<=`) -- mutation-caught: flipping < to <= passed every other test.
    now = int(time.time())
    sha = "3" * 40
    min_age_min = 30
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.settle_a2\n"
                 % (now - min_age_min * 60, sha))   # exactly at the age boundary, not one second past it
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool1": 32}, []))
    monkeypatch.setattr(psc, "check_provisioned", lambda *a, **k: False)
    report = psc.check_queue(nodes=["pool1"], queue_path_=str(q), now=now, unrunnable_min_age_min=min_age_min)
    assert len(report["unrunnable"]) == 1


def test_check_queue_memory_budget_not_flagged_before_its_own_age_threshold(tmp_path, monkeypatch):
    now = int(time.time())
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/sim && mem_gb=40 python3 -m research.runners.huge_job\n" % (now - 300,))
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool40": 15}, []))
    report = psc.check_queue(nodes=["pool40"], queue_path_=str(q), now=now)
    assert report["memory_budget_stalled"] == []


def test_check_queue_unpinned_line_never_flagged_unrunnable(tmp_path, monkeypatch):
    # The ~/derisk-pool/sim compatibility path (no revision pin) is always "provisioned" -- not this check's job.
    now = int(time.time())
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/sim && mem_gb=8 python3 -m research.runners.foo\n" % (now - 7200,))
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool40": 15}, []))
    monkeypatch.setattr(psc, "check_provisioned", lambda *a, **k: False)
    report = psc.check_queue(nodes=["pool40"], queue_path_=str(q), now=now)
    assert report["unrunnable"] == []
    assert report["memory_budget_stalled"] == []


def test_check_queue_probes_each_node_sha_pair_once_even_across_several_lines(tmp_path, monkeypatch):
    now = int(time.time())
    sha = "e" * 40
    q = tmp_path / "pool.queue"
    q.write_text(
        "%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.a --out a.json\n"
        "%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.b --out b.json\n"
        % (now - 3600, sha, now - 3600, sha)
    )
    calls = []

    def fake_check_provisioned(node, s, timeout=10, connect_timeout=6):
        calls.append((node, s))
        return False
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool1": 32}, []))
    monkeypatch.setattr(psc, "check_provisioned", fake_check_provisioned)
    report = psc.check_queue(nodes=["pool1"], queue_path_=str(q), now=now)
    assert len(report["unrunnable"]) == 2
    assert calls == [("pool1", sha)]   # one probe reused for both lines, not two


def test_check_queue_stops_probing_at_first_true_when_too_young_for_capacity_check(tmp_path, monkeypatch):
    # LOW-3 fix: probing must stop the instant a capable node reads True, WHILE the line is not yet old enough to
    # need the capacity_stalled evidence check (MEDIUM fix, 2026-09-25 review -- see the companion test below,
    # which pins the opposite: once a line IS old enough, the remaining capable nodes ARE probed). 40 minutes
    # clears the default unrunnable_min_age (30 min, so the line reaches the provisioning check at all) but stays
    # under the default membudget_min_age (60 min, so capacity_stalled is never evaluated).
    now = int(time.time())
    sha = "5" * 40
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=4 python3 -m research.runners.x\n" % (now - 2400, sha))
    calls = []

    def fake_check_provisioned(node, s, timeout=10, connect_timeout=6):
        calls.append(node)
        return node == "pool1"   # the FIRST capable node (dict insertion order) already has it
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6:
                         ({"pool1": 32, "pool41": 15, "pool42": 15}, []))
    monkeypatch.setattr(psc, "check_provisioned", fake_check_provisioned)
    report = psc.check_queue(nodes=["pool1", "pool41", "pool42"], queue_path_=str(q), now=now)
    assert calls == ["pool1"]   # pool41/pool42 never probed once pool1 read True
    assert report["capacity_stalled"] == []   # too young for the capacity check to even run


def test_check_queue_capacity_stalled_probes_remaining_capable_nodes_once_aged(tmp_path, monkeypatch):
    # MEDIUM fix (2026-09-25 review): once a line has aged into a capacity_stalled INCIDENT worth reporting, the
    # remaining capable nodes ARE probed too (the (node, sha) cache bounds this to nodes x shas) so the report
    # can name every node CONFIRMED missing the revision -- the real fix -- instead of repeating the one node
    # that happens to have it (the "revision on the mini-PCs, never probed pool1/pool2" incident state).
    now = int(time.time())
    sha = "6" * 40
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=4 python3 -m research.runners.x\n" % (now - 7200, sha))
    calls = []

    def fake_check_provisioned(node, s, timeout=10, connect_timeout=6):
        calls.append(node)
        return node == "pool1"   # only pool1 has it; pool41/pool42 are confirmed missing
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6:
                         ({"pool1": 32, "pool41": 15, "pool42": 15}, []))
    monkeypatch.setattr(psc, "check_provisioned", fake_check_provisioned)
    monkeypatch.setattr(psc, "probe_live_node_budget_gb", lambda node, **k: 0)   # evidence: live budget too small
    report = psc.check_queue(nodes=["pool1", "pool41", "pool42"], queue_path_=str(q),
                              claims_path_=str(tmp_path / "no-such-claims"), now=now)
    assert set(calls) == {"pool1", "pool41", "pool42"}   # ALL capable nodes probed, not just the first True
    assert len(report["capacity_stalled"]) == 1
    row = report["capacity_stalled"][0]
    assert row["missing_on_nodes"] == ["pool41", "pool42"]
    assert "pool41" in row["fix_cmd"] and "pool42" in row["fix_cmd"]


def test_check_queue_capacity_stalled_not_reported_without_evidence(tmp_path, monkeypatch):
    # HIGH-1 fix (2026-09-25 review): a capable+provisioned node with enough elapsed time is ALSO exactly what a
    # perfectly healthy backlog looks like (replay: 47 clean lines flagged this way inside 30 minutes). With
    # NEITHER evidence check confirming anything (no live budget shortfall, no later-dispatched claim), the line
    # must not be reported as capacity_stalled at all.
    now = int(time.time())
    sha = "d" * 40
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.settle_a2\n"
                 % (now - 7200, sha))
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool1": 32}, []))
    monkeypatch.setattr(psc, "check_provisioned", lambda node, s, timeout=10, connect_timeout=6: True)
    monkeypatch.setattr(psc, "probe_live_node_budget_gb", lambda *a, **k: None)   # no live confirmation
    report = psc.check_queue(nodes=["pool1"], queue_path_=str(q),
                              claims_path_=str(tmp_path / "no-such-claims"), now=now)
    assert report["capacity_stalled"] == []
    assert "clean" in report["summary_line"]


def test_check_queue_capacity_stalled_evidence_from_later_dispatched_claim(tmp_path, monkeypatch):
    # HIGH-1 fix, second evidence path: no live --node-budget confirmation, but pool.queue.claims shows a job of
    # this size or larger was actually DISPATCHED after this line was already queued -- direct proof the
    # dispatcher had room for a job this size in that window and picked something else over this one.
    now = int(time.time())
    sha = "c" * 40
    epoch = now - 7200
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.settle_a2\n"
                 % (epoch, sha))
    claims = tmp_path / "pool.queue.claims"
    claims.write_text("%d\tcd ~/derisk-pool/sim && mem_gb=10 python3 -m research.runners.other_job\n"
                       % (epoch + 100,))   # dispatched AFTER this line was already queued, size >= 8
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool1": 32}, []))
    monkeypatch.setattr(psc, "check_provisioned", lambda node, s, timeout=10, connect_timeout=6: True)
    monkeypatch.setattr(psc, "probe_live_node_budget_gb", lambda *a, **k: None)   # no live confirmation
    report = psc.check_queue(nodes=["pool1"], queue_path_=str(q), claims_path_=str(claims), now=now)
    assert len(report["capacity_stalled"]) == 1
    assert "later-queued" in report["capacity_stalled"][0]["evidence"]


def test_check_queue_mem_stalled_becomes_unknown_when_a_node_is_unreachable(tmp_path, monkeypatch):
    # MEDIUM-1 fix: an unreachable node's true capacity is UNKNOWN, not confidently "too small". Before this
    # fix, EVERY aged line with no capable REACHABLE node was reported memory_budget_stalled even when a bigger
    # node might have fit it but simply could not be probed (replays C/D in the review).
    now = int(time.time())
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/sim && mem_gb=20 python3 -m research.runners.big_job\n" % (now - 7200,))
    # pool40 (15GB) genuinely too small; pool_aws (the one that might fit 20GB) is unreachable.
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6:
                         ({"pool40": 15}, ["pool_aws"]))
    report = psc.check_queue(nodes=["pool40", "pool_aws"], queue_path_=str(q), now=now)
    assert report["memory_budget_stalled"] == []
    assert len(report["unknown"]) == 1
    assert report["unknown"][0]["mem_gb"] == 20
    assert "UNKNOWN" in report["summary_line"]


def test_check_queue_still_mem_stalled_when_every_node_reachable(tmp_path, monkeypatch):
    # Companion to the above: when mem_unreachable is EMPTY (every known node was actually probed), the old
    # confident memory_budget_stalled verdict is still correct and must not be diluted into "unknown".
    now = int(time.time())
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/sim && mem_gb=20 python3 -m research.runners.big_job\n" % (now - 7200,))
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool40": 15}, []))
    report = psc.check_queue(nodes=["pool40"], queue_path_=str(q), now=now)
    assert report["unknown"] == []
    assert len(report["memory_budget_stalled"]) == 1


def test_check_queue_unrunnable_becomes_unknown_when_every_capable_probe_fails(tmp_path, monkeypatch):
    # MEDIUM-2 fix: a None marker result (probe failed) must never be treated as "confirmed missing". Before
    # this fix, an all-None status (every capable node's revision-marker probe failed) still read UNRUNNABLE,
    # contradicting check_provisioned's own None-means-"cannot say" contract.
    now = int(time.time())
    sha = "6" * 40
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.settle_a2\n"
                 % (now - 3600, sha))
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool1": 32}, []))
    monkeypatch.setattr(psc, "check_provisioned", lambda *a, **k: None)   # every probe fails
    report = psc.check_queue(nodes=["pool1"], queue_path_=str(q), now=now)
    assert report["unrunnable"] == []
    assert len(report["unknown"]) == 1
    assert "UNKNOWN" in report["summary_line"]


def test_check_queue_still_unrunnable_when_at_least_one_probe_confirms_false(tmp_path, monkeypatch):
    # Companion to the above: a MIX of False (confirmed absent) and None (probe failed) among capable nodes must
    # still flag UNRUNNABLE -- at least one genuine confirmation is enough; only an ALL-None status is unknown.
    now = int(time.time())
    sha = "7" * 40
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.settle_a2\n"
                 % (now - 3600, sha))
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6:
                         ({"pool1": 32, "pool2": 32}, []))

    def fake_check_provisioned(node, s, timeout=10, connect_timeout=6):
        return False if node == "pool1" else None
    monkeypatch.setattr(psc, "check_provisioned", fake_check_provisioned)
    report = psc.check_queue(nodes=["pool1", "pool2"], queue_path_=str(q), now=now)
    assert report["unknown"] == []
    assert len(report["unrunnable"]) == 1
    # LOW fix (2026-09-25 review): missing_nodes must carry only the CONFIRMED-False node; a probe-FAILED (None)
    # node belongs in unprobed_nodes instead, and must not appear in the provisioning fix command -- provisioning
    # an unreachable node was never confirmed useful.
    row = report["unrunnable"][0]
    assert row["missing_nodes"] == ["pool1"]
    assert row["unprobed_nodes"] == ["pool2"]
    assert "pool1" in row["fix_cmd"]
    assert "pool2" not in row["fix_cmd"]


def test_check_queue_expired_line_never_dispatchable_reports_separately(tmp_path, monkeypatch):
    # MEDIUM-3 fix: a line older than the dispatcher's own POOL_JOB_MAX_AGE (12h default) is skipped forever by
    # pop_job's own candidate-selection filter -- classifying it as UNRUNNABLE/memory-stalled would be moot
    # (its fix command could never make it run) AND misleading ("revision missing" implies provisioning helps).
    now = int(time.time())
    sha = "8" * 40
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.settle_a2\n"
                 % (now - 43200 - 60, sha))   # just past the 12h default cutoff
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool1": 32}, []))
    monkeypatch.setattr(psc, "check_provisioned", lambda *a, **k: False)
    report = psc.check_queue(nodes=["pool1"], queue_path_=str(q), now=now)
    assert report["unrunnable"] == []
    assert report["memory_budget_stalled"] == []
    assert len(report["expired"]) == 1
    assert "EXPIRED" in report["summary_line"]


def test_check_queue_not_yet_expired_line_still_classified_normally(tmp_path, monkeypatch):
    # Companion boundary: a line one second SHORT of the cutoff must still go through the normal UNRUNNABLE path
    # (not silently swallowed by an off-by-one in the expiry check).
    now = int(time.time())
    sha = "9" * 40
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.settle_a2\n"
                 % (now - 43200 + 60, sha))   # one minute short of the 12h default cutoff
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool1": 32}, []))
    monkeypatch.setattr(psc, "check_provisioned", lambda *a, **k: False)
    report = psc.check_queue(nodes=["pool1"], queue_path_=str(q), now=now)
    assert report["expired"] == []
    assert len(report["unrunnable"]) == 1


def test_check_queue_expired_boundary_equality_is_still_a_dispatcher_candidate(tmp_path, monkeypatch):
    # LOW fix (2026-09-25 review): pop_job's own awk filter selects `$1+0 >= cutoff`, i.e. age_s == job_max_age
    # EXACTLY is STILL a candidate the dispatcher will pick up next cycle -- the old `age_s >= job_max_age` here
    # reported this boundary EXPIRED (permanently undispatchable), a false and misleading verdict one cycle early.
    now = int(time.time())
    sha = "f0" * 20
    max_age = 43200
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.settle_a2\n"
                 % (now - max_age, sha))   # age_s == job_max_age exactly
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool1": 32}, []))
    monkeypatch.setattr(psc, "check_provisioned", lambda *a, **k: False)
    report = psc.check_queue(nodes=["pool1"], queue_path_=str(q), now=now, job_max_age_s=max_age)
    assert report["expired"] == []
    assert len(report["unrunnable"]) == 1


def test_check_queue_membudget_age_gate_boundary_equality_counts_as_old_enough(tmp_path, monkeypatch):
    # Review note (2026-09-25): "neither this mutation nor '>=' to '>' at :608 (the memory-budget age gate) is
    # caught by the suite" -- pins the CURRENT, intentional behavior of that internal reporting gate (unlike the
    # dispatcher-mirrored expired cutoff above, this age gate is this tool's own choice, not derived from an
    # external source) so a future flip of its boundary operator is no longer silent.
    now = int(time.time())
    sha = "f1" * 20
    membudget_min_age_h = 1
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/sim && mem_gb=40 python3 -m research.runners.huge_job\n"
                 % (now - int(membudget_min_age_h * 3600),))   # age_s == membudget_min_age_s exactly
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool40": 15}, []))
    report = psc.check_queue(nodes=["pool40"], queue_path_=str(q), now=now, membudget_min_age_h=membudget_min_age_h)
    assert len(report["memory_budget_stalled"]) == 1


def test_check_queue_no_entries_reports_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(psc, "all_mem_totals", lambda nodes, timeout=10, connect_timeout=6: ({"pool40": 15}, []))
    report = psc.check_queue(nodes=["pool40"], queue_path_=str(tmp_path / "no-queue"))
    assert report["n_queued"] == 0
    assert report["unrunnable"] == []
    assert report["memory_budget_stalled"] == []
    assert "clean" in report["summary_line"]


def test_check_queue_never_raises_when_all_mem_totals_blows_up(tmp_path, monkeypatch):
    q = tmp_path / "pool.queue"
    q.write_text("%d\tpython3 -m research.runners.foo\n" % int(time.time()))

    def boom(*a, **k):
        raise RuntimeError("ssh exploded")
    monkeypatch.setattr(psc, "all_mem_totals", boom)
    report = psc.check_queue(nodes=["pool40"], queue_path_=str(q))   # must not raise
    assert report["mem_totals"] == {}
    assert report["mem_unreachable"] == ["pool40"]


# --------------------------------------------------------------------------------------- check_queue: end-to-end
# (fake ssh on PATH, branching on the REMOTE SCRIPT this time -- not just the node, since check_queue makes two
# DIFFERENT kinds of probe per node: MemTotal and the .provisioned_ok marker)

def _write_fake_ssh_probes(tmp_path, mem_totals=None, markers=(), unreachable=()):
    bin_dir = tmp_path / "bin_probes"
    bin_dir.mkdir(exist_ok=True)
    memdir = tmp_path / "memtotals"
    memdir.mkdir(exist_ok=True)
    for node, gb in (mem_totals or {}).items():
        (memdir / node).write_text(str(gb))
    markers_file = tmp_path / "markers.txt"
    markers_file.write_text("\n".join("%s %s" % (n, s) for n, s in markers) + "\n")
    unreach_file = tmp_path / "unreachable_probes.txt"
    unreach_file.write_text("\n".join(unreachable) + "\n")
    stub = bin_dir / "ssh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'node="${@: -2:1}"\n'
        'script="${@: -1}"\n'
        'if grep -qxF "$node" "%s" 2>/dev/null; then exit 255; fi\n'
        'case "$script" in\n'
        '  *MemTotal*)\n'
        '    f="%s/$node"\n'
        '    if [ -f "$f" ]; then cat "$f"; exit 0; else exit 1; fi\n'
        '    ;;\n'
        "  *provisioned_ok*)\n"
        "    sha=$(grep -oE 'revisions/[0-9a-f]+' <<< \"$script\" | cut -d/ -f2)\n"
        '    if grep -qxF "$node $sha" "%s" 2>/dev/null; then exit 0; else exit 1; fi\n'
        '    ;;\n'
        '  *) exit 0 ;;\n'
        "esac\n"
        % (unreach_file, memdir, markers_file)
    )
    st = stub.stat()
    stub.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


def test_check_queue_end_to_end_with_fake_ssh(tmp_path, monkeypatch):
    now = int(time.time())
    sha = "f" * 40
    q = tmp_path / "pool.queue"
    q.write_text(
        # UNRUNNABLE: pinned, 2h old, fits pool41 (15GB) and pool1 (32GB), marker present on NEITHER
        "%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.settle_a2 --out a.json\n"
        # memory_budget_stalled: 2h old, mem_gb 40 exceeds every node
        "%d\tcd ~/derisk-pool/sim && mem_gb=40 python3 -m research.runners.huge_job\n"
        # clean: pinned, marker present on pool1
        "%d\tcd ~/derisk-pool/revisions/%s && mem_gb=4 python3 -m research.runners.ok_job\n"
        % (now - 7200, sha, now - 7200, now - 7200, "1" * 40)
    )
    bin_dir = _write_fake_ssh_probes(
        tmp_path,
        mem_totals={"pool1": 32, "pool41": 15},
        markers=[("pool1", "1" * 40)],   # only the third line's revision is provisioned, only on pool1
        unreachable=[],
    )
    monkeypatch.setenv("PATH", "%s:%s" % (bin_dir, os.environ.get("PATH", "")))
    # HIGH-1 fix (2026-09-25 review): stub the live-budget evidence probe so the "clean" third line's
    # capacity_stalled evaluation stays hermetic (no real `bash tools/pool_autodispatch.sh --node-budget ...`
    # subprocess, which would itself shell out to `ssh`) -- with no evidence, it must stay clean (asserted below).
    monkeypatch.setattr(psc, "probe_live_node_budget_gb", lambda *a, **k: None)
    report = psc.check_queue(nodes=["pool1", "pool41"], queue_path_=str(q),
                              claims_path_=str(tmp_path / "no-such-claims"), now=now)

    assert report["mem_totals"] == {"pool1": 32, "pool41": 15}
    assert report["n_queued"] == 3
    assert len(report["unrunnable"]) == 1
    assert report["unrunnable"][0]["pinned_sha"] == sha
    assert set(report["unrunnable"][0]["capable_nodes"]) == {"pool1", "pool41"}
    assert len(report["memory_budget_stalled"]) == 1
    assert report["capacity_stalled"] == []   # the clean, provisioned third line has no evidence -> not reported
    assert report["memory_budget_stalled"][0]["mem_gb"] == 40


def test_check_queue_end_to_end_unreachable_node_excluded_from_capability(tmp_path, monkeypatch):
    now = int(time.time())
    sha = "9" * 40
    q = tmp_path / "pool.queue"
    q.write_text("%d\tcd ~/derisk-pool/revisions/%s && mem_gb=8 python3 -m research.runners.x\n"
                 % (now - 7200, sha))
    bin_dir = _write_fake_ssh_probes(tmp_path, mem_totals={"pool41": 15}, markers=(), unreachable=["pool99"])
    monkeypatch.setenv("PATH", "%s:%s" % (bin_dir, os.environ.get("PATH", "")))
    report = psc.check_queue(nodes=["pool41", "pool99"], queue_path_=str(q), now=now)
    assert report["mem_unreachable"] == ["pool99"]
    assert len(report["unrunnable"]) == 1
    assert report["unrunnable"][0]["capable_nodes"] == ["pool41"]   # pool99 excluded, not "capable but missing"


# ------------------------------------------------------------------------------------- row formatting / main() CLI

def test_format_unrunnable_row_names_missing_nodes_and_fix_cmd():
    row = {
        "age_s": 7200, "module": "settle_a2", "pinned_sha": "a" * 40, "mem_gb": 8,
        "capable_nodes": ["pool1", "pool41"],
        "node_status": {"pool1": False, "pool41": None},
        "fix_cmd": "bash tools/pool_provision.sh --revision %s --isolated pool1 pool41" % ("a" * 40),
    }
    line = psc.format_unrunnable_row(row)
    assert "settle_a2" in line
    assert "pool1,pool41" in line or ("pool1" in line and "pool41" in line)
    assert "fix: bash tools/pool_provision.sh" in line


def test_format_membudget_row_names_ceiling():
    row = {"age_s": 7200, "module": "huge_job", "mem_gb": 40, "max_known_ceiling_gb": 13}
    line = psc.format_membudget_row(row)
    assert "huge_job" in line
    assert "40" in line
    assert "13" in line


def test_format_capacity_stalled_row_names_provisioned_node():
    row = {"age_s": 3600, "module": "settle_a2", "pinned_sha": "b" * 40, "mem_gb": 8,
           "provisioned_node": "pool1", "capable_nodes": ["pool1", "pool41"]}
    line = psc.format_capacity_stalled_row(row)
    assert "settle_a2" in line
    assert "pool1" in line
    assert "--node-budget pool1" in line


def test_format_capacity_stalled_row_names_evidence_and_missing_nodes():
    # HIGH-1/MEDIUM fix (2026-09-25 review): the row now carries WHY it was flagged (evidence) and WHICH other
    # capable nodes are confirmed missing the revision (missing_on_nodes/fix_cmd) -- both must render.
    row = {"age_s": 7200, "module": "settle_a2", "pinned_sha": "b" * 40, "mem_gb": 8,
           "provisioned_node": "pool1", "capable_nodes": ["pool1", "pool41"],
           "evidence": "live --node-budget pool1 reads 0GB < mem_gb=8",
           "missing_on_nodes": ["pool41"],
           "fix_cmd": "bash tools/pool_provision.sh --revision %s --isolated pool41" % ("b" * 40)}
    line = psc.format_capacity_stalled_row(row)
    assert "reads 0GB < mem_gb=8" in line
    assert "ALSO missing on pool41" in line
    assert "bash tools/pool_provision.sh" in line


def test_format_expired_row_names_reissue_command():
    row = {"age_s": 43260, "module": "settle_a2"}
    line = psc.format_expired_row(row)
    assert "settle_a2" in line
    assert "pool_queue.sh add" in line


def test_format_expired_row_names_delete_under_lock_with_no_generic_readd_only(tmp_path):
    # LOW fix (2026-09-25 review): pool_queue.sh has no remove command, so "re-add via ..." ALONE (the old text)
    # leaves the stale line alerting forever -- the message must also say to delete the original line first.
    row = {"age_s": 43260, "module": "settle_a2", "epoch": 1234567890}
    line = psc.format_expired_row(row)
    assert "delete it from pool.queue" in line
    assert "1234567890" in line


def test_format_unknown_row_carries_reason():
    row = {"age_s": 7200, "module": "big_job", "reason": "no reachable node fits mem_gb=20"}
    line = psc.format_unknown_row(row)
    assert "big_job" in line
    assert "no reachable node fits mem_gb=20" in line


def test_main_json_includes_queue_report(monkeypatch, capsys):
    monkeypatch.setattr(psc, "check_all", lambda timeout=12: {
        "unreachable": [], "running": [], "flagged": [], "n_running": 0, "n_dup": 0, "n_overdue": 0,
        "n_unknown_overdue": 0, "summary_line": "POOL STALL CHECK: clean (of 0 running across 0 node(s))",
    })
    monkeypatch.setattr(psc, "check_queue", lambda timeout=12: {
        "nodes": [], "mem_totals": {}, "mem_unreachable": [], "n_queued": 0, "unrunnable": [],
        "memory_budget_stalled": [], "summary_line": "POOL QUEUE CHECK: clean (of 0 queued line(s))",
    })
    rc = psc.main(["--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "queue" in out
    assert out["queue"]["summary_line"].startswith("POOL QUEUE CHECK")


def test_main_skip_queue_omits_queue_report(monkeypatch, capsys):
    monkeypatch.setattr(psc, "check_all", lambda timeout=12: {
        "unreachable": [], "running": [], "flagged": [], "n_running": 0, "n_dup": 0, "n_overdue": 0,
        "n_unknown_overdue": 0, "summary_line": "POOL STALL CHECK: clean (of 0 running across 0 node(s))",
    })
    called = []
    monkeypatch.setattr(psc, "check_queue", lambda timeout=12: called.append(1))
    rc = psc.main(["--json", "--skip-queue"])
    assert rc == 0
    assert called == []
    out = json.loads(capsys.readouterr().out)
    assert "queue" not in out


def test_main_human_readable_prints_queue_flags(monkeypatch, capsys):
    monkeypatch.setattr(psc, "check_all", lambda timeout=12: {
        "unreachable": [], "running": [], "flagged": [], "n_running": 0, "n_dup": 0, "n_overdue": 0,
        "n_unknown_overdue": 0, "summary_line": "POOL STALL CHECK: clean (of 0 running across 0 node(s))",
    })
    unrunnable_row = {
        "age_s": 7200, "module": "settle_a2", "pinned_sha": "a" * 40, "mem_gb": 8,
        "capable_nodes": ["pool1"], "node_status": {"pool1": False},
        "fix_cmd": "bash tools/pool_provision.sh --revision %s --isolated pool1" % ("a" * 40),
    }
    monkeypatch.setattr(psc, "check_queue", lambda timeout=12: {
        "nodes": ["pool1"], "mem_totals": {"pool1": 32}, "mem_unreachable": [], "n_queued": 1,
        "unrunnable": [unrunnable_row], "memory_budget_stalled": [],
        "summary_line": "POOL QUEUE CHECK: 1 UNRUNNABLE (of 1 queued line(s), 0 node(s) mem-unreachable)",
    })
    rc = psc.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "POOL QUEUE CHECK" in out
    assert "settle_a2" in out
    assert "fix: bash tools/pool_provision.sh" in out
