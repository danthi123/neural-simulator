"""tools/pool_stall_check.py -- catches a live-but-stalled pool (2026-09-25: 7 of 17 D6 processes were
DUPLICATES of cells whose output had already landed at the same pinned revision, running 7-26h, while the
heartbeat read SATURATED). Pure-logic tests need no ssh; the end-to-end tests fake `ssh` on PATH the same way
tests/test_pool_autodispatch_workflow.py already does for the shell dispatcher, never touching a real node.
"""
from __future__ import annotations

import base64
import json
import os
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
