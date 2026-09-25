"""tools/audit_pool_fragment_claims.py must find a mid-line fragment claim, map it to its node / executed text /
swallowing probe / exit status, name the full line it came from and that line's later real run -- and must NOT
flag a complete line. Synthetic queue files only; nothing live is read.

Fix round r2 (2026-09-25, review of the dispatcher-fragment audit) added:
  - a label-prefixed complete claim (the SETTLE A2 shape) that must land in `nonstandard_start_complete_lines`,
    so a mutation of the `if ts and ts >= since and not NORMAL_START.match(job):` gate (or of `output_paths`)
    cannot pass silently;
  - a BLOCKED-quarantine dispatch.log sequence that must land in `blocked_unchecked_since`;
  - a `--root`-overridden shard-tag scan against a small fixture tree with one passing and one failing cell,
    reaching `scan_shard_tag`/`local_cell_report`'s LIVE_ROOT-dependent code for the first time in this suite;
  - direct unit coverage of `output_paths` and of the `&&`/`;`/`|` word-boundary fix in `first_command_word`.
Each addition was mutation-verified against the exact code path it targets (see the fix round's commit message).
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
