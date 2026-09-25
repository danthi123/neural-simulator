"""tools/apply_parent_status_archive.py must fill only the parent occurrences a prior report left unresolved, never
touch a fragment's own node_status, never override an occurrence that already has a live-resolved one, and record
what it did in report["restorations"]. Fix round r3 (2026-09-25, review MEDIUM item): this exists because a
worktree-isolated fix round has no research/queue/ access to re-run the audit tool end to end -- see the script's
own docstring."""
import importlib.util
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "tools", "apply_parent_status_archive.py")

_spec = importlib.util.spec_from_file_location("apply_parent_status_archive", SCRIPT)
apply_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(apply_mod)


def _report():
    return {
        "fragments": [{
            "claim_line": 1,
            "node_status": [{"node": "poolFRAG", "ts": 1, "rc": 127}],  # never touched by this script
            "parents": [{
                "outputs": ["out/x.json"],
                "occurrences_at_or_after_fragment": [
                    {"src": "claims", "line": 2, "time": "2026-01-01 00:00:00", "node_status": None},
                    {"src": "claims", "line": 3, "time": "2026-01-01 00:05:00",
                     "node_status": [{"node": "poolLIVE", "ts": 2, "rc": 0}]},
                    {"src": "claims", "line": 4, "time": "2026-01-01 00:10:00", "node_status": None},
                ],
            }],
        }],
    }


def _archive():
    return {"entries": [
        {"claim_line": 1, "parent_outputs": ["out/x.json"], "occurrence_src": "claims", "occurrence_line": 2,
         "occurrence_time": "2026-01-01 00:00:00", "node_status": [{"node": "poolARCHIVE", "ts": 9, "rc": 0}]},
        # a conflicting record for the ALREADY-live occurrence -- must never be applied
        {"claim_line": 1, "parent_outputs": ["out/x.json"], "occurrence_src": "claims", "occurrence_line": 3,
         "occurrence_time": "2026-01-01 00:05:00", "node_status": [{"node": "poolWRONG", "ts": 9, "rc": 99}]},
        # no entry at all for line 4 -- stays unresolved
    ]}


def test_apply_archive_fills_only_the_unresolved_gap_and_never_overrides_live(tmp_path):
    report = _report()
    archive_path = tmp_path / "archive.json"
    json.dump(_archive(), open(archive_path, "w"))
    archive = apply_mod.load_archive(str(archive_path))
    filled, already_live, no_match = apply_mod.apply_archive(report, archive, "the/archive.json")
    assert filled == 1 and already_live == 1 and no_match == 1

    occs = report["fragments"][0]["parents"][0]["occurrences_at_or_after_fragment"]
    assert occs[0]["node_status"] == [{"node": "poolARCHIVE", "ts": 9, "rc": 0}]
    assert occs[0]["node_status_source"] == "archive:the/archive.json"
    # the already-live occurrence is untouched -- no archive tag, no rc 99
    assert occs[1]["node_status"] == [{"node": "poolLIVE", "ts": 2, "rc": 0}]
    assert "node_status_source" not in occs[1]
    # no matching archive entry -- stays unresolved, no crash
    assert occs[2]["node_status"] is None
    assert "node_status_source" not in occs[2]
    # the fragment's OWN node_status is never touched by this script
    assert report["fragments"][0]["node_status"] == [{"node": "poolFRAG", "ts": 1, "rc": 127}]


def test_cli_end_to_end_writes_restorations_block(tmp_path):
    report_path = tmp_path / "report.json"
    archive_path = tmp_path / "archive.json"
    json.dump(_report(), open(report_path, "w"))
    json.dump(_archive(), open(archive_path, "w"))

    r = subprocess.run([sys.executable, SCRIPT, "--report", str(report_path), "--archive", str(archive_path)],
                        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    patched = json.load(open(report_path))
    (restoration,) = patched["restorations"]
    assert restoration["filled"] == 1 and restoration["already_live"] == 1 and restoration["no_match"] == 1
    assert restoration["archive"] == os.path.relpath(str(archive_path), REPO)
