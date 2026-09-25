"""CLASS LBP — a load-bearing battery's aggregate.json is committed while its own per-cell provenance was never
checked against the tag's registered pin.

THE FAILURE (research/findings/2026-09-25-production-default-battery-B2a-FAIL.md). `tools/lb_shard.py aggregate`
computed `robust_core_n`/`union_n`/`mean_fraction` straight from each shard's OWN `null_control_clean`/`UNRELIABLE`
fields and had no per-cell provenance rule at all, so 23 of 28 coverable faculties' seed-102 cell — run from a
dirty, non-pinned local worktree instead of the pre-registered `git_archive` revision — folded into a
clean-looking aggregate with `git_dirty` reading false-negative-clean (research/FAILURE_LOG.md's row for this
declared it "NOT-GATEABLE as a repo-wide pre-commit check: whether a battery's per-cell provenance must be pinned
to one revision is battery-specific" — true for a GENERIC diff scanner, but `lb_shard.py aggregate --pin` now
carries the per-cell check itself and stamps its own verdict into the artifact this gate reads).

THE GATE. A newly-ADDED `aggregate.json` under `_load_bearing/.../<tag>/aggregate.json` that carries load-bearing
faculty data (`per_faculty` has an entry with `seeds_present`) must show `provenance.status == "verified"` (i.e.
it was produced by `lb_shard.py aggregate --pin <sha>`, which excludes and reports any cell that does not match),
UNLESS it carries an explicit top-level `"provenance_unverified_accepted": "<reason>"` for a deliberately
exploratory/dev aggregate that nobody will read as a verdict.

WHAT THIS GATE CANNOT CATCH. Whether the recorded `provenance.pin` is the RIGHT revision for this tag (that is a
battery-specific pre-registration fact, same limit `artifact_provenance` documents: presence of a checked field,
not truth of its content). A modified (not newly-added) aggregate.json never fires, same scoping as every other
content gate here (`--diff-filter=A`) — a battery re-aggregated in place is not re-flagged each time.
"""
from __future__ import annotations

import json
import os

NAME = "lb-battery-provenance"
CLASS_ID = "LBP"
BLOCKING = True


def _is_target(path: str) -> bool:
    p = path.replace("\\", "/")
    return os.path.basename(p) == "aggregate.json" and "_load_bearing" in p


def _has_coverable_data(data) -> bool:
    per_fac = data.get("per_faculty")
    if isinstance(per_fac, dict):
        return any(v.get("seeds_present") for v in per_fac.values() if isinstance(v, dict))
    if isinstance(per_fac, list):  # tolerate a future list-shaped per_faculty
        return any(v.get("seeds_present") for v in per_fac if isinstance(v, dict))
    return False


def check(paths) -> list:
    if not paths:
        return []
    problems = []
    for path in paths:
        if not _is_target(path) or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as e:
            problems.append("CLASS LBP unreadable aggregate.json: %s (%s: %s)" % (path, type(e).__name__, e))
            continue
        if not isinstance(data, dict) or not _has_coverable_data(data):
            continue  # nothing load-bearing rides on this file yet
        accepted = data.get("provenance_unverified_accepted")
        if isinstance(accepted, str) and accepted:
            continue
        prov = data.get("provenance")
        status = prov.get("status") if isinstance(prov, dict) else None
        if status == "verified":
            continue
        problems.append(
            "CLASS LBP unverified battery aggregate: %s carries load-bearing faculty data but "
            "provenance.status=%r (want 'verified'). A coverable cell may have run off the tag's registered "
            "pin without detection (2026-09-25 B2a: 23/28 coverable faculties' seed-102 cell did exactly this). "
            "Fix: re-run `lb_shard.py aggregate --tag <tag> --pin <full sha>` for this tag, or add a top-level "
            '"provenance_unverified_accepted": "<reason>" for a deliberately exploratory aggregate.'
            % (path, status))
    return problems


def selftest() -> list:
    import tempfile
    bad = []
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "research", "findings", "raw", "_load_bearing", "_shards")
        os.makedirs(raw, exist_ok=True)

        def w(tag, obj):
            d = os.path.join(raw, tag)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, "aggregate.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(obj, fh)
            return p

        coverable = {"seeds_present": [42, 43]}
        # --- THE FAILING DIRECTION FIRST: cases the gate MUST catch ---
        must_catch = {
            "no provenance key at all": w("t1", {"per_faculty": {"fac-a": coverable}}),
            "provenance status unverified": w("t2", {"per_faculty": {"fac-a": coverable},
                                                       "provenance": {"status": "unverified"}}),
            "provenance malformed (not a dict)": w("t3", {"per_faculty": {"fac-a": coverable},
                                                            "provenance": "verified"}),
            "unreadable JSON": None,
        }
        bad_json_path = os.path.join(raw, "t4", "aggregate.json")
        os.makedirs(os.path.dirname(bad_json_path), exist_ok=True)
        with open(bad_json_path, "w") as fh:
            fh.write("{not json")
        must_catch["unreadable JSON"] = bad_json_path
        for label, p in must_catch.items():
            if not check([p]):
                bad.append("GATE CANNOT FAIL: %s (%s) produced no problem" % (label, p))

        # --- only then the passing direction: cases it must NOT cry wolf on ---
        must_pass = {
            "status verified": w("t5", {"per_faculty": {"fac-a": coverable},
                                          "provenance": {"status": "verified", "pin": "a" * 40, "n_invalid": 0}}),
            "explicit unverified-accepted waiver": w("t6", {"per_faculty": {"fac-a": coverable},
                                                              "provenance": {"status": "unverified"},
                                                              "provenance_unverified_accepted": "dev smoke, not a verdict"}),
            "no coverable data yet (empty per_faculty)": w("t7", {"per_faculty": {}}),
            "no coverable data (seeds_present empty)": w("t8", {"per_faculty": {"fac-a": {"seeds_present": []}}}),
            "not an aggregate.json": os.path.join(raw, "t9", "lb.json"),
            "not under _load_bearing": None,
            "file does not exist": os.path.join(raw, "nope", "aggregate.json"),
        }
        os.makedirs(os.path.dirname(must_pass["not an aggregate.json"]), exist_ok=True)
        with open(must_pass["not an aggregate.json"], "w") as fh:
            json.dump({"per_faculty": {"fac-a": coverable}}, fh)
        elsewhere = os.path.join(td, "elsewhere", "aggregate.json")
        os.makedirs(os.path.dirname(elsewhere), exist_ok=True)
        with open(elsewhere, "w") as fh:
            json.dump({"per_faculty": {"fac-a": coverable}}, fh)  # no provenance -- but outside _load_bearing
        must_pass["not under _load_bearing"] = elsewhere
        for label, p in must_pass.items():
            probs = check([p])
            if probs:
                bad.append("FALSE POSITIVE: %s flagged -- %s" % (label, probs[0][:120]))

        if check([]):
            bad.append("check([]) returned problems; an empty staged list must never block")
    return bad


if __name__ == "__main__":
    print("class LBP lb-battery-provenance gate -- run selftest() for a self-check")
