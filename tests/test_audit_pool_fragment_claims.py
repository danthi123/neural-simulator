"""tools/audit_pool_fragment_claims.py must find a mid-line fragment claim, map it to its node / executed text /
swallowing probe / exit status, name the full line it came from and that line's later real run -- and must NOT
flag a complete line. Synthetic queue files only; nothing live is read.

Fix round r2 (2026-09-25, review of the dispatcher-fragment audit) added:
  - a label-prefixed complete claim (the SETTLE A2 shape) that must land in `nonstandard_start_complete_lines`,
    so a mutation of the `if ts and ts >= since and not NORMAL_START.match(job):` gate (or of `output_paths`)
    cannot pass silently;
  - a BLOCKED-quarantine dispatch.log sequence that must land in `blocked_unchecked_since`;
  - a `--root`-overridden shard-tag scan against a small fixture tree with one passing and one failing cell,
    reaching `scan_shard_tag`'s LIVE_ROOT-dependent code for the first time in this suite;
  - direct unit coverage of `output_paths` and of the `&&`/`;`/`|` word-boundary fix in `first_command_word`.
Each addition was mutation-verified against the exact code path it targets (see that round's commit message).

Fix round r3 (2026-09-25, review of research/dispatcher-fragment-audit) added, each mutation-verified against the
exact code path it targets (see this round's commit message):
  - `.job_status.tsv` coverage in `load_node_status` (review HIGH item: the committed evidence's real extension,
    `.log`, is gitignored, so a prior round's claim that it was committed was false; reverting the glob to
    `*.job_status.log` only makes the new tests fail);
  - the `--parent-status-archive` merge (review MEDIUM item): fills a parent occurrence's node_status only when a
    live `--node-status` fetch left it empty, and never overrides a live-resolved one;
  - the fail-open-turned-loud warnings (review LOW item): an empty/absent `--node-status` match, a fragment or
    nonstandard-start line still unresolved after `--node-status` was given, and a `--scan-shards` tag matching
    zero cells, each now append to `warnings` and flip the exit code non-zero;
  - real coverage of the `--node-outputs` loop, of `local_cell_report` (a fragment whose OWN swallowed text still
    carries its `--out` flag, matched against a `--root` fixture cell), of `unpinned_tree_sidecars`, and of
    `is_mid_line_fragment` -- all four previously reached no code because the fixtures never satisfied their
    preconditions, so mutating any of them left the suite fully green (the exact previous-round docstring claim
    this round found false: "the suite now reaches local_cell_report" -- it did not).
"""
import base64
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "tools", "audit_pool_fragment_claims.py")

_spec = importlib.util.spec_from_file_location("audit_pool_fragment_claims", SCRIPT)
audit_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_mod)

FULL = ("cd ~/derisk-pool/revisions/" + "a" * 40 + " && env SIM_BACKEND=numpy .venv/bin/python -u -m "
        "research.runners.some_runner --seed 42 --out research/findings/raw/_x/s42/out.json  #checked:prereg X")
OTHER = "SIM_BACKEND=numpy .venv/bin/python -u -m research.runners.other --seed 1  #checked:reason Y"
FRAG = FULL[FULL.index("s42/out.json"):]          # "s42/out.json  #checked:prereg X" -- a mid-line tail
# A SETTLE-A2-shaped complete line: a LABEL precedes the assignment, so bash's first word is "A2", not an
# assignment or a recognised command start -- NORMAL_START must not match it.
A2_JOB = ("A2 wiring seed 42: mem_gb=8 && cd ~/derisk-pool/revisions/" + "d" * 40 + " && .venv/bin/python -u -m "
          "research.runners.settle_affect_marker_wiring --run-wiring --seed 42  #checked:SETTLE wiring reason Z")


def _executed(job):
    body = job.split("#checked:", 1)[0].rstrip()
    reason = job.split("#checked:", 1)[1]
    return "POOL_CHECKED_REASON=%s %s" % (reason.replace(" ", "\\ "), body)


def _write(qd, t0):
    claims = [(t0, OTHER), (t0 + 100, FRAG), (t0 + 900, FULL), (t0 + 300, A2_JOB)]
    with open(os.path.join(qd, "pool.queue.claims"), "w") as fh:
        for t, j in claims:
            fh.write("%d\t%s\n" % (t, j))
    open(os.path.join(qd, "pool.queue"), "w").close()
    nodes = ("poolA", "poolB", "poolB", "poolD")
    with open(os.path.join(qd, "pool.running"), "w") as fh:
        for (t, j), node in zip(claims, nodes):
            fh.write("%s\t%s\t%s\n" % (dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"), node,
                                         _executed(j)))
    with open(os.path.join(qd, "dispatch.log"), "w") as fh:
        fh.write("[pool-dispatch] started 00:00:01 | queue=x\n")
        fh.write("[pool-dispatch] 00:00:02 poolA <- %s\n" % _executed(OTHER))
        fh.write("[pool-dispatch] revision %s not provisioned on poolB -- job(s) pinned to it stay queued\n" % ("b" * 40))
        fh.write("[pool-dispatch] 00:01:42 poolB <- %s\n" % _executed(FRAG))
        # A BLOCKED-quarantine sequence: a probe line, then the dispatcher's own "BLOCKED unchecked job" marker,
        # then the (truncated-to-96-char) head of the mid-`#checked:`-cut line it refused to run.
        fh.write("[pool-dispatch] revision %s not provisioned on poolB -- job(s) pinned to it stay queued\n" % ("e" * 40))
        fh.write("[pool-dispatch] BLOCKED unchecked job -- requeue via: bash tools/pool_queue.sh add '<cmd>' "
                  "--checked '<what the record says>'\n")
        fh.write("synthetic blocked head text for the fixture -- no #checked: reason survived the mid-line cut\n")
    st = os.path.join(qd, "status")
    os.makedirs(st)
    with open(os.path.join(st, "poolB.job_status.log"), "w") as fh:
        fh.write("v2\t%d\t127\t%s\n" % (t0 + 101, base64.b64encode(_executed(FRAG).encode()).decode()))
        fh.write("v2\t%d\t0\t%s\n" % (t0 + 950, base64.b64encode(_executed(FULL).encode()).decode()))
    with open(os.path.join(st, "poolD.job_status.log"), "w") as fh:
        fh.write("v2\t%d\t127\t%s\n" % (t0 + 301, base64.b64encode(_executed(A2_JOB).encode()).decode()))
    return st


def test_fragment_detected_mapped_and_complete_lines_not_flagged(tmp_path):
    qd = str(tmp_path)
    t0 = int(dt.datetime(2026, 9, 24, 12, 0, 0).timestamp())
    st = _write(qd, t0)
    out = os.path.join(qd, "report.json")
    r = subprocess.run([sys.executable, SCRIPT, "--queue-dir", qd, "--since", str(t0 - 10), "--node-status", st,
                        "--json", out], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    rep = json.load(open(out))
    assert rep["n_fragments"] == 1, rep
    f = rep["fragments"][0]
    assert f["claim_line"] == 2 and f["fragment"] == FRAG
    assert f["node"] == "poolB" and f["pinned"] is False
    assert f["swallowing_probe"]["node"] == "poolB"
    assert f["classification"]["expect"].startswith("command-not-found")
    assert [s["rc"] for s in f["node_status"]] == [127]
    (parent,) = f["parents"]
    later = parent["occurrences_at_or_after_fragment"]
    assert [(o["line"], o["node"], [s["rc"] for s in o["node_status"]]) for o in later] == [(3, "poolB", [0])]

    # -- SETTLE-A2 shape: a label-prefixed COMPLETE line must be flagged as nonstandard-start, not silently
    # accepted as ordinary or (worse) misdetected as a fragment. Mutating the `if ts and ts >= since and not
    # NORMAL_START.match(job):` gate to `if False:` empties this list; this assertion then fails.
    assert len(rep["nonstandard_start_complete_lines"]) == 1, rep["nonstandard_start_complete_lines"]
    a2 = rep["nonstandard_start_complete_lines"][0]
    assert a2["claim_line"] == 4 and a2["node"] == "poolD"
    assert a2["job_head"].startswith("A2 wiring seed 42:")
    assert [s["rc"] for s in a2["node_status"]] == [127]

    # -- BLOCKED quarantine: a mid-`#checked:`-cut line dropped by the record-check gate must be reported, with
    # its preceding probe attributed. Removing/short-circuiting the "BLOCKED unchecked job" scan empties this.
    assert len(rep["blocked_unchecked_since"]) == 1, rep["blocked_unchecked_since"]
    b = rep["blocked_unchecked_since"][0]
    assert b["head96"] == "synthetic blocked head text for the fixture -- no #checked: reason survived the mid-line cut"
    assert b["preceded_by_probe"] == {"sha": "e" * 40, "node": "poolB"}


def test_output_paths_extracts_every_out_flag_value():
    """Direct coverage of output_paths (line ~128): the review noted that disabling its `outs.append` call left
    the integration test above silently unaffected (FRAG/FULL never carry a --out flag inside the fragment's own
    tail). This calls it directly so a disabled append fails here instead."""
    text = "some_runner.py --seed 1 --out path/to/one.json --other x --json path/to/two.json --out-dir a/b"
    assert audit_mod.output_paths(text) == ["path/to/one.json", "path/to/two.json", "a/b"]
    assert audit_mod.output_paths("--out only.json") == ["only.json"]
    assert audit_mod.output_paths("nothing here") == []


def test_first_command_word_treats_shell_operators_as_word_boundaries():
    """The bug the review found: `X=1 && cmd` used to read first_word='&&' (misclassified command-not-found, rc
    127 expected) because `&&` was returned as if it were the command itself. A NAME=value assignment following
    an operator must still be recognised as a fresh assignment, not the command word."""
    assert audit_mod.first_command_word("X=1 && cmd --flag") == ("cmd", None)
    assert audit_mod.first_command_word("X=1 && Y=2 && .venv/bin/python foo.py") == (".venv/bin/python", None)
    assert audit_mod.first_command_word("cmd ; other") == ("cmd", None)
    assert audit_mod.first_command_word("a | b") == ("a", None)
    # a bare `&` is NOT skipped here -- classify()'s own special case (backgrounded assignment) depends on
    # seeing it come back as the word.
    assert audit_mod.first_command_word("X=1 & rest")[0] == "&"
    assert audit_mod.classify("X=1 && .venv/bin/python foo.py")["expect"] == "RUNS"


def test_scan_shards_root_override_with_fixture_cells(tmp_path):
    """LIVE_ROOT used to be hardcoded, so --scan-shards / --node-outputs' pin-rule code (scan_shard_tag,
    local_cell_report) could only ever read the real checkout and had NO test coverage. --root redirects it to a
    throwaway fixture tree with one cell that passes the pin rule and one that fails it."""
    root = tmp_path / "fixture_root"
    pin = "f" * 40
    base = root / "research" / "findings" / "raw" / "_load_bearing" / "_shards" / "fx_tag"
    pass_cell = base / "s1" / "fac_pass"
    fail_cell = base / "s2" / "fac_fail"
    pass_cell.mkdir(parents=True)
    fail_cell.mkdir(parents=True)

    def _sidecar(cell, sha):
        json.dump({"per_faculty": []}, open(cell / "lb.json", "w"))
        json.dump({
            "git_sha": sha, "source_kind": "git_archive", "source_manifest_verified_at_start": True,
            "source_manifest_verified_at_exit": True, "env": {"SIM_BACKEND": "numpy"},
        }, open(cell / "lb.json.prov.json", "w"))

    _sidecar(pass_cell, pin)          # matches the pin -> clean
    _sidecar(fail_cell, "0" * 40)     # wrong git_sha -> pin-rule failure

    qd = tmp_path / "queue"
    qd.mkdir()
    open(qd / "pool.queue.claims", "w").close()
    open(qd / "pool.running", "w").close()
    with open(qd / "dispatch.log", "w") as fh:
        fh.write("[pool-dispatch] started 00:00:01 | queue=x\n")

    out = tmp_path / "report.json"
    r = subprocess.run([sys.executable, SCRIPT, "--queue-dir", str(qd), "--since", "0", "--root", str(root),
                        "--scan-shards", "fx_tag=%s" % pin, "--json", str(out)],
                        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    rep = json.load(open(out))
    (scan,) = rep["shard_scans"]
    assert scan["n_cells"] == 2, scan
    assert list(scan["pin_rule_failing_cells"].keys()) == ["s2/fac_fail"], scan["pin_rule_failing_cells"]
    assert "s1/fac_pass" not in scan["pin_rule_failing_cells"]
    assert any("git_sha" in f for f in scan["pin_rule_failing_cells"]["s2/fac_fail"]["lb.json.prov.json"])


# ---------------------------------------------------------------------------------------------------------------
# Fix round r3 (2026-09-25, review of research/dispatcher-fragment-audit)
# ---------------------------------------------------------------------------------------------------------------

def test_load_node_status_reads_both_log_and_tsv_extensions(tmp_path):
    """Review HIGH item: the committed evidence's real name (`.job_status.log`) is silently dropped by
    `.gitignore`'s `*.log` rule, so a durable copy must use `.tsv` instead -- and the loader must actually read it.
    Reverting the glob to `*.job_status.log` only makes n_files/n_records read 1/1 instead of 2/2."""
    d = tmp_path / "status"
    d.mkdir()
    b64_a = base64.b64encode(b"cmd-a").decode()
    b64_b = base64.b64encode(b"cmd-b").decode()
    (d / "poolA.job_status.log").write_text("v2\t100\t0\t%s\n" % b64_a)
    (d / "poolB.job_status.tsv").write_text("v2\t200\t127\t%s\n" % b64_b)
    status, n_files, n_records = audit_mod.load_node_status(str(d))
    assert n_files == 2, n_files
    assert n_records == 2, n_records
    assert status[b64_a] == [{"node": "poolA", "ts": 100, "rc": 0}]
    assert status[b64_b] == [{"node": "poolB", "ts": 200, "rc": 127}]


def test_fragment_resolves_from_tsv_only_node_status_dir(tmp_path):
    """Review HIGH item, reproduced end-to-end exactly as the real evidence directory is laid out (ONLY `.tsv`
    files, no `.log`): research/findings/raw/_dispatcher_fragment_audit/node_evidence/. Reverting
    `load_node_status`'s glob to `*.job_status.log` only makes every assertion below fail (rc=? everywhere)."""
    qd = str(tmp_path / "queue")
    os.makedirs(qd)
    t0 = int(dt.datetime(2026, 9, 24, 12, 0, 0).timestamp())
    st_log = _write(qd, t0)
    st_tsv = str(tmp_path / "status_tsv")
    os.makedirs(st_tsv)
    for name in os.listdir(st_log):
        content = open(os.path.join(st_log, name)).read()
        open(os.path.join(st_tsv, name.replace(".job_status.log", ".job_status.tsv")), "w").write(content)

    out = os.path.join(qd, "report.json")
    r = subprocess.run([sys.executable, SCRIPT, "--queue-dir", qd, "--since", str(t0 - 10), "--node-status",
                        st_tsv, "--json", out], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    rep = json.load(open(out))
    assert not rep["warnings"], rep["warnings"]
    f = rep["fragments"][0]
    assert [s["rc"] for s in f["node_status"]] == [127]
    (parent,) = f["parents"]
    later = parent["occurrences_at_or_after_fragment"]
    assert [s["rc"] for s in later[0]["node_status"]] == [0]


def _archive_fixture(qd, t0):
    """One fragment (`frag`, claim_line 1) whose parent full line (`full`, claim_line 2) has ITS OWN pool.running
    dispatch recorded -- so only its job_status (rc) coverage is ever in question, matching the real MEDIUM defect
    exactly (the parent's dispatch was always known; only its exit status was never fetched)."""
    full = ("cd ~/derisk-pool/revisions/" + "9" * 40 + " && .venv/bin/python -u -m research.runners.rx --seed 9 "
            "--out research/findings/raw/_x/s9/outx.json  #checked:reason G")
    frag = full[full.index("s9/outx.json"):]
    with open(os.path.join(qd, "pool.queue.claims"), "w") as fh:
        fh.write("%d\t%s\n" % (t0, frag))
        fh.write("%d\t%s\n" % (t0 + 500, full))
    open(os.path.join(qd, "pool.queue"), "w").close()
    with open(os.path.join(qd, "pool.running"), "w") as fh:
        fh.write("%s\t%s\t%s\n" % (dt.datetime.fromtimestamp(t0 + 1).strftime("%Y-%m-%d %H:%M:%S"), "poolFRAG",
                                     _executed(frag)))
        fh.write("%s\t%s\t%s\n" % (dt.datetime.fromtimestamp(t0 + 500).strftime("%Y-%m-%d %H:%M:%S"), "poolLIVE",
                                     _executed(full)))
    with open(os.path.join(qd, "dispatch.log"), "w") as fh:
        fh.write("[pool-dispatch] started 00:00:01 | queue=x\n")
    return full, frag


def _archive_status_dir(tmp_path, frag, full=None, full_rc=None):
    st = tmp_path / "archive_status"
    st.mkdir()
    lines = ["v2\t0\t127\t%s\n" % base64.b64encode(_executed(frag).encode()).decode()]
    if full is not None:
        lines.append("v2\t0\t%d\t%s\n" % (full_rc, base64.b64encode(_executed(full).encode()).decode()))
    (st / "poolFRAG.job_status.log").write_text("".join(lines))
    return str(st)


def test_parent_status_archive_fills_gap_left_by_live_fetch(tmp_path):
    """Review MEDIUM item: a regeneration that fetched only the fragments' own job_status records (never the
    PARENT full-line reruns') silently lost every parent rc. `--parent-status-archive` restores one from a prior,
    already-committed resolution when the live `--node-status` fetch left it empty. Removing the archive-fallback
    branch (or the CLI flag) makes `occ["node_status"]` read None here."""
    qd = str(tmp_path / "queue")
    os.makedirs(qd)
    t0 = int(dt.datetime(2026, 9, 24, 12, 0, 0).timestamp())
    full, frag = _archive_fixture(qd, t0)
    st = _archive_status_dir(tmp_path, frag)  # no FULL coverage here -- exactly the gap the review found
    archive_path = tmp_path / "archive.json"
    parent_outputs = audit_mod.output_paths(audit_mod.strip_checked(full))
    entry = {
        "claim_line": 1, "parent_outputs": parent_outputs, "occurrence_src": "claims", "occurrence_line": 2,
        "occurrence_time": audit_mod._fmt(t0 + 500),
        "node_status": [{"node": "poolARCHIVE", "ts": 999999, "rc": 0}],
    }
    json.dump({"entries": [entry]}, open(archive_path, "w"))

    out = os.path.join(qd, "report.json")
    r = subprocess.run([sys.executable, SCRIPT, "--queue-dir", qd, "--since", str(t0 - 10), "--node-status", st,
                        "--parent-status-archive", str(archive_path), "--json", out],
                        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    rep = json.load(open(out))
    (f,) = rep["fragments"]
    assert f["claim_line"] == 1
    (parent,) = f["parents"]
    (occ,) = parent["occurrences_at_or_after_fragment"]
    assert occ["node_status"] == [{"node": "poolARCHIVE", "ts": 999999, "rc": 0}], occ
    assert occ["node_status_source"] == "archive:%s" % str(archive_path), occ


def test_parent_status_archive_never_overrides_live_node_status(tmp_path):
    """Review MEDIUM item, the other direction: a live `--node-status` fetch that DOES cover a parent occurrence
    must win over the archive every time, even when the archive disagrees. Removing the `if not entry["node_status"]`
    guard (always applying the archive) makes this read the archive's deliberately-wrong rc 55."""
    qd = str(tmp_path / "queue")
    os.makedirs(qd)
    t0 = int(dt.datetime(2026, 9, 24, 12, 0, 0).timestamp())
    full, frag = _archive_fixture(qd, t0)
    st = _archive_status_dir(tmp_path, frag, full=full, full_rc=0)
    archive_path = tmp_path / "archive.json"
    parent_outputs = audit_mod.output_paths(audit_mod.strip_checked(full))
    entry = {
        "claim_line": 1, "parent_outputs": parent_outputs, "occurrence_src": "claims", "occurrence_line": 2,
        "occurrence_time": audit_mod._fmt(t0 + 500),
        "node_status": [{"node": "poolWRONG", "ts": 1, "rc": 55}],
    }
    json.dump({"entries": [entry]}, open(archive_path, "w"))

    out = os.path.join(qd, "report.json")
    r = subprocess.run([sys.executable, SCRIPT, "--queue-dir", qd, "--since", str(t0 - 10), "--node-status", st,
                        "--parent-status-archive", str(archive_path), "--json", out],
                        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    rep = json.load(open(out))
    (f,) = rep["fragments"]
    (parent,) = f["parents"]
    (occ,) = parent["occurrences_at_or_after_fragment"]
    assert [s["rc"] for s in occ["node_status"]] == [0], occ
    assert occ["node"] == "poolLIVE"
    assert "node_status_source" not in occ, occ


def test_warns_and_exits_nonzero_when_node_status_dir_has_no_matching_files(tmp_path):
    """Review LOW item: this is exactly how the missing job_status evidence went unnoticed the first time --
    --node-status pointed at a dir with nothing it recognises used to print a clean-looking rc=? everywhere with no
    signal that the flag was effectively a no-op. Reverting the warning append makes the exit code 0 here."""
    qd = str(tmp_path / "queue")
    os.makedirs(qd)
    t0 = int(dt.datetime(2026, 9, 24, 12, 0, 0).timestamp())
    _write(qd, t0)
    empty_status = tmp_path / "empty_status"
    empty_status.mkdir()
    out = os.path.join(qd, "report.json")
    r = subprocess.run([sys.executable, SCRIPT, "--queue-dir", qd, "--since", str(t0 - 10), "--node-status",
                        str(empty_status), "--json", out], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 2, r
    assert "WARNING" in r.stderr
    rep = json.load(open(out))
    assert any("0 files" in w or "0 v2 records" in w for w in rep["warnings"]), rep["warnings"]


def test_warns_when_a_fragment_stays_unresolved_after_node_status_given(tmp_path):
    """Review LOW item: --node-status given AND non-empty, but not covering THIS fragment, must warn -- not just
    print an easy-to-miss rc=? in the per-line dump."""
    qd = str(tmp_path / "queue")
    os.makedirs(qd)
    t0 = int(dt.datetime(2026, 9, 24, 12, 0, 0).timestamp())
    _write(qd, t0)
    st = tmp_path / "partial_status"
    st.mkdir()
    (st / "irrelevant.job_status.log").write_text(
        "v2\t1\t0\t%s\n" % base64.b64encode(b"some-other-command").decode())
    out = os.path.join(qd, "report.json")
    r = subprocess.run([sys.executable, SCRIPT, "--queue-dir", qd, "--since", str(t0 - 10), "--node-status",
                        str(st), "--json", out], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 2, r
    rep = json.load(open(out))
    assert not rep["fragments"][0].get("node_status")
    assert any("still unresolved" in w for w in rep["warnings"]), rep["warnings"]


def test_warns_when_scan_shard_tag_matches_zero_cells(tmp_path):
    """Review LOW item: `scan_shard_tag` on a mistyped/missing tag used to print "cells=0 pin-rule-failing=0",
    which reads as a CLEAN scan rather than one that found nothing to check."""
    root = tmp_path / "empty_root"
    root.mkdir()
    qd = tmp_path / "queue"
    qd.mkdir()
    open(qd / "pool.queue.claims", "w").close()
    open(qd / "pool.running", "w").close()
    with open(qd / "dispatch.log", "w") as fh:
        fh.write("[pool-dispatch] started 00:00:01 | queue=x\n")
    out = tmp_path / "report.json"
    r = subprocess.run([sys.executable, SCRIPT, "--queue-dir", str(qd), "--since", "0", "--root", str(root),
                        "--scan-shards", "nonexistent_tag=%s" % ("0" * 40), "--json", str(out)],
                        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 2, r
    rep = json.load(open(out))
    assert rep["shard_scans"][0]["n_cells"] == 0
    assert any("matched 0 cells" in w for w in rep["warnings"]), rep["warnings"]


def test_node_outputs_loop_runs_pin_rule_on_a_node_copied_cell(tmp_path):
    """Review LOW item: --node-outputs had zero test coverage, so disabling its whole loop (`if a.node_outputs:`
    -> `if False:`) left the suite fully green."""
    node_outputs = tmp_path / "node_outputs"
    cell = (node_outputs / "pool9" / "research" / "findings" / "raw" / "_load_bearing" / "_shards" / "fx_tag_no" /
            "s1" / "fac1")
    cell.mkdir(parents=True)
    json.dump({"per_faculty": [{"faculty": "fac1", "verdict": "regressed", "load_bearing": True}]},
               open(cell / "lb.json", "w"))
    json.dump({
        "run_id": "r-node-9", "argv": ["/home/ubuntu/derisk-pool/sim/tools/x.py"], "git_sha": "a" * 40,
        "source_kind": "git_archive", "started": "2026-09-24T00:00:00", "env": {"SIM_BACKEND": "numpy"},
        "source_manifest_verified_at_start": True, "source_manifest_verified_at_exit": True,
    }, open(cell / "lb.json.prov.json", "w"))

    root = tmp_path / "fixture_root"
    local_cell = (root / "research" / "findings" / "raw" / "_load_bearing" / "_shards" / "fx_tag_no" / "s2" /
                  "fac2")
    local_cell.mkdir(parents=True)
    json.dump({"per_faculty": []}, open(local_cell / "lb.json", "w"))
    json.dump({
        "git_sha": "a" * 40, "source_kind": "git_archive", "source_manifest_verified_at_start": True,
        "source_manifest_verified_at_exit": True, "env": {"SIM_BACKEND": "numpy"},
    }, open(local_cell / "lb.json.prov.json", "w"))

    qd = tmp_path / "queue"
    qd.mkdir()
    open(qd / "pool.queue.claims", "w").close()
    open(qd / "pool.running", "w").close()
    with open(qd / "dispatch.log", "w") as fh:
        fh.write("[pool-dispatch] started 00:00:01 | queue=x\n")

    out = tmp_path / "report.json"
    r = subprocess.run([sys.executable, SCRIPT, "--queue-dir", str(qd), "--since", "0", "--root", str(root),
                        "--scan-shards", "fx_tag_no=%s" % ("a" * 40), "--node-outputs", str(node_outputs),
                        "--json", str(out)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    rep = json.load(open(out))
    (c,) = rep["node_output_cells"]
    assert c["node"] == "pool9"
    assert c["cell"] == "fx_tag_no/s1/fac1"
    assert c["run_id"] == "r-node-9"
    assert c["git_sha"] == "a" * 40
    assert c["per_faculty"] == [["fac1", "regressed", True]]
    assert c["pin_rule_fails"] == {}, c


def test_local_cell_report_reached_when_fragment_output_matches_root_fixture_cell(tmp_path):
    """Review LOW item: `local_cell_report` only runs when a fragment's OWN swallowed text still carries its
    `--out` flag (the swallow point landed BEFORE the flag, not after, as it does for FRAG/FULL above) -- no
    fixture in this suite satisfied that, so `rec["local_cells"]` was always []. Mutating `local_cell_report` to
    always `return None` left the suite fully green."""
    qd = str(tmp_path)
    t0 = int(dt.datetime(2026, 9, 24, 12, 0, 0).timestamp())
    full = ("cd ~/derisk-pool/revisions/" + "7" * 40 + " && FOO=1 BAR=2 .venv/bin/python -u -m "
            "research.runners.load_bearing_fraction --seed 3 --out "
            "research/findings/raw/_load_bearing/_shards/fx_tag_lcr/s3/fac3/lb.json  #checked:reason LCR")
    frag = full[full.index("BAR=2"):]  # swallow point BEFORE --out: the fragment keeps its own --out flag

    with open(os.path.join(qd, "pool.queue.claims"), "w") as fh:
        fh.write("%d\t%s\n" % (t0, frag))
        fh.write("%d\t%s\n" % (t0 + 500, full))
    open(os.path.join(qd, "pool.queue"), "w").close()
    with open(os.path.join(qd, "pool.running"), "w") as fh:
        fh.write("%s\t%s\t%s\n" % (dt.datetime.fromtimestamp(t0 + 1).strftime("%Y-%m-%d %H:%M:%S"), "poolLCR",
                                     _executed(frag)))
    with open(os.path.join(qd, "dispatch.log"), "w") as fh:
        fh.write("[pool-dispatch] started 00:00:01 | queue=x\n")

    root = tmp_path / "fixture_root"
    cell = root / "research" / "findings" / "raw" / "_load_bearing" / "_shards" / "fx_tag_lcr" / "s3" / "fac3"
    cell.mkdir(parents=True)
    json.dump({"per_faculty": []}, open(cell / "lb.json", "w"))
    json.dump({
        "git_sha": "7" * 40, "source_kind": "git_archive", "source_manifest_verified_at_start": True,
        "source_manifest_verified_at_exit": True, "env": {"SIM_BACKEND": "numpy"},
    }, open(cell / "lb.json.prov.json", "w"))

    out = os.path.join(qd, "report.json")
    r = subprocess.run([sys.executable, SCRIPT, "--queue-dir", qd, "--since", str(t0 - 10), "--root", str(root),
                        "--json", out], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    rep = json.load(open(out))
    (f,) = rep["fragments"]
    assert f["outputs"] == ["research/findings/raw/_load_bearing/_shards/fx_tag_lcr/s3/fac3/lb.json"], f
    (cell_report,) = f["local_cells"]
    assert cell_report["exists"] is True, cell_report
    assert cell_report["cell"] == "research/findings/raw/_load_bearing/_shards/fx_tag_lcr/s3/fac3"
    assert cell_report["lb_prov"]["git_sha"] == "7" * 40
    assert cell_report["pin_rule_fails"] == {}, cell_report


def test_unpinned_tree_sidecar_flagged_in_scan_shard_tag(tmp_path):
    """Review LOW item: the existing --root fixture never adds a bare `.prov.json` whose argv0 lives under an
    UNPINNED node tree, so `unpinned_tree_sidecars` -- the exact fingerprint the finding uses to say no
    fragment-produced cell survives locally -- was never populated by any test."""
    root = tmp_path / "fixture_root"
    pin = "f" * 40
    base = root / "research" / "findings" / "raw" / "_load_bearing" / "_shards" / "fx_tag_up"
    cell = base / "s1" / "fac1"
    cell.mkdir(parents=True)
    json.dump({"per_faculty": []}, open(cell / "lb.json", "w"))
    json.dump({
        "git_sha": pin, "source_kind": "git_archive", "source_manifest_verified_at_start": True,
        "source_manifest_verified_at_exit": True, "env": {"SIM_BACKEND": "numpy"},
        "argv": ["/home/ubuntu/derisk-pool/revisions/%s/tools/lb_runner.py" % pin],
    }, open(cell / "lb.json.prov.json", "w"))
    json.dump({"argv": ["/home/ubuntu/derisk-pool/sim/tools/some_runner.py"]},
               open(cell / "oed_something.prov.json", "w"))

    qd = tmp_path / "queue"
    qd.mkdir()
    open(qd / "pool.queue.claims", "w").close()
    open(qd / "pool.running", "w").close()
    with open(qd / "dispatch.log", "w") as fh:
        fh.write("[pool-dispatch] started 00:00:01 | queue=x\n")

    out = tmp_path / "report.json"
    r = subprocess.run([sys.executable, SCRIPT, "--queue-dir", str(qd), "--since", "0", "--root", str(root),
                        "--scan-shards", "fx_tag_up=%s" % pin, "--json", str(out)],
                        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    rep = json.load(open(out))
    (scan,) = rep["shard_scans"]
    assert scan["n_cells"] == 1, scan
    assert scan["unpinned_tree_sidecars"] == [
        {"sidecar": "s1/fac1/oed_something.prov.json", "argv0": "/home/ubuntu/derisk-pool/sim/tools/some_runner.py"}
    ], scan


def test_blocked_quarantine_flags_is_mid_line_fragment_when_head_matches_a_known_line(tmp_path):
    """Review LOW item: the earlier BLOCKED-quarantine fixture's head text is never a substring of any known queue
    line, so `hosts` is always empty and `is_mid_line_fragment`/`n_host_lines` are dead code in every prior test --
    mutating `bool(hosts)` to always False left the suite fully green. This head IS a genuine, non-prefix
    substring of a known claimed line."""
    qd = str(tmp_path)
    t0 = int(dt.datetime(2026, 9, 24, 12, 0, 0).timestamp())
    known_line = ("cd ~/derisk-pool/revisions/" + "1" * 40 + " && .venv/bin/python -u -m research.runners.z "
                  "--seed 5 --out research/findings/raw/_x/s5/z.json  #checked:reason Z")
    head = "research.runners.z --seed 5 --out research/findings/raw/_x/s5/z.json  #checked:reason Z"
    assert head in known_line and not known_line.startswith(head)  # sanity on the fixture itself

    with open(os.path.join(qd, "pool.queue.claims"), "w") as fh:
        fh.write("%d\t%s\n" % (t0, known_line))
    open(os.path.join(qd, "pool.queue"), "w").close()
    open(os.path.join(qd, "pool.running"), "w").close()
    with open(os.path.join(qd, "dispatch.log"), "w") as fh:
        fh.write("[pool-dispatch] started 00:00:01 | queue=x\n")
        fh.write("[pool-dispatch] revision %s not provisioned on poolZ -- job(s) pinned to it stay queued\n"
                  % ("f" * 40))
        fh.write("[pool-dispatch] BLOCKED unchecked job -- requeue via: bash tools/pool_queue.sh add '<cmd>' "
                  "--checked '<what the record says>'\n")
        fh.write("%s\n" % head)

    out = os.path.join(qd, "report.json")
    r = subprocess.run([sys.executable, SCRIPT, "--queue-dir", qd, "--since", str(t0 - 10), "--json", out],
                        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    rep = json.load(open(out))
    (b,) = rep["blocked_unchecked_since"]
    assert b["head96"] == head
    assert b["is_mid_line_fragment"] is True, b
    assert b["n_host_lines"] == 1, b
