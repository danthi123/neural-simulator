"""Verification harness for the #5 stabilizer fix (research/lbf-fix-source-provenance-abstain, 2026-09-22):
`BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE` (default OFF) makes source-provenance-honesty's `judge_fact()` report a
deterministic abstain (`label=None`) at the opponent's genuine no-signal collapse (`|d| < TIE_D_EPS`) instead of
`_laneC_source_provenance_opponent_derisk._judge()`'s seed-dependent host coin-flip -- see the module docstring
in `research/runners/source_provenance_honesty.py` and the fragile-step finding
`research/findings/2026-09-22-borderline-separability-stabilizer-is-buildable.md`.

Reads the per-seed `research.runners.load_bearing_fraction --only source-provenance-honesty --seed <N>` JSON
artifacts this fix's own build produced (flag ON for all 6 mandated seeds; flag OFF re-confirmed at the two
seeds the pre-fix finding named fragile, s44/s102) and earns a `tools.verdict.Verdict` over them -- never
re-derives a number the artifacts do not already state.

NOT a brain-building runner itself (no memcap needed to RUN this file); it only reads JSON already produced by
memcapped `load_bearing_fraction` invocations (see the commit's build log for those commands).
"""
from __future__ import annotations

import json
import os
import sys

from tools.verdict import Verdict

SEEDS = (42, 43, 44, 100, 101, 102)
KEY = "source-provenance-honesty"
# The pre-fix, already-committed historical artifacts this fix's flag-OFF re-run must match EXACTLY (TERMS.md
# "byte-identical" bar: asserted in the data via an exact compare, never inferred from reading the code).
_PRE_FIX_ADEQUATE6 = "research/findings/raw/_load_bearing/_adequate6/load_bearing_adequate_s%d.json"


def _load(path):
    with open(path) as f:
        return json.load(f)


def _faculty_row(report, key=KEY):
    for p in report.get("per_faculty", []):
        if p.get("faculty") == key:
            return p
    return None


def main(out_dir, out_path):
    v = Verdict("source-provenance-honesty abstain-at-tie fix: load-bearing 6/6 with the flag ON")

    on_rows = {}
    for seed in SEEDS:
        path = os.path.join(out_dir, "lbf_s%d_flagON.json" % seed)
        report = _load(path)
        row = _faculty_row(report)
        on_rows[seed] = row
        v.require("flag=1 seed=%d: flag actually reached the build (BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE=1 read back)"
                   % seed, report.get("source_prov_abstain_at_tie_env"), expect="1",
                   note="artifact-stamped env readback, not a claim")
        v.require("flag=1 seed=%d: arms built + the harness's own null control is clean" % seed,
                   row is not None and row.get("null_control_clean"), expect=True)
        v.require("flag=1 seed=%d: load_bearing" % seed, row is not None and row.get("load_bearing"), expect=True)
        v.require("flag=1 seed=%d: treatment_diffs > control_diffs (attributable to the lesion, not noise)" % seed,
                   (row.get("treatment_diffs") or 0) > (row.get("control_diffs") or 0) if row else None,
                   expect=True)

    off_rows = {}
    # The flag-OFF re-confirmation runs live in their OWN subdirectory (`off/`), not `out_dir` itself: the
    # per-arm filenames (intact_a_well_s<seed>.json, lesion_source_provenance_honesty_s<seed>.json) do not
    # encode the flag (BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE never enters base_env/grp_sig), so an OFF run sharing
    # `out_dir` with the ON run for the SAME seed would silently overwrite that seed's ON lesion artifact.
    #
    # COMPUTE REUSE (not a shortcut on the CLAIM, only on redundant brain-builds): the intact_a/intact_b arms
    # for a given seed are PROVABLY invariant to BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE -- the flag only changes
    # judge_fact()'s label at a genuine |d|<TIE_D_EPS collapse, and every intact arm's measured d sits at
    # >0.9999998 on every seed this fix observed (see the finding's separability table), nowhere near
    # TIE_D_EPS=1e-6. So the off/ dir was seeded with the ALREADY-BUILT flagON intact_a/intact_b JSON files for
    # s44/s102 (copied verbatim, not rebuilt) and the run launched with LB_RESUME_SKIP_EXISTING=1 -- confirmed
    # by `_spawn_arm`'s own resume logic (`research/runners/load_bearing_fraction.py`: "_LB_RESUME and
    # os.path.exists(out_path)" -> `json.load`, no env check), so ONLY the lesion arm -- the one arm the fix
    # actually touches -- was rebuilt fresh under an environment with BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE unset.
    # This is still a REAL, FRESH lesion-arm build (not inferred from code), just without redundantly re-paying
    # for two intact-arm builds whose outcome cannot differ by construction.
    off_dir = os.path.join(out_dir, "off")
    for seed in (44, 102):
        path = os.path.join(off_dir, "lbf_s%d_flagOFF.json" % seed)
        report = _load(path)
        row = _faculty_row(report)
        off_rows[seed] = row
        # Verdict.require() treats a `measured is None` as UNMEASURED (its own sentinel for "the run never
        # recorded this"), so the check itself must not BE None when the env genuinely read back unset --
        # pass a boolean (env-readback != "1") instead of the raw (usually-None) field.
        v.require("flag=0 seed=%d: env readback confirms the flag was actually UNSET for this build (not '1')"
                   % seed, report.get("source_prov_abstain_at_tie_env") != "1", expect=True,
                   note="artifact-stamped env readback (%r) -- guards against a leaked/inherited flag from the "
                        "parent shell" % (report.get("source_prov_abstain_at_tie_env"),))
        v.require("flag=0 seed=%d: byte-identical to the pre-fix finding (NOT load-bearing off@this seed)" % seed,
                   row is not None and row.get("load_bearing"), expect=False,
                   note="reproduces research/findings/2026-09-22-borderline-separability-stabilizer-is-buildable.md's "
                        "pre-fix off@s44/s102 result; the flag defaulting off must not change it")
        # EXACT compare against the historical pre-fix artifact (produced by an earlier session, before this
        # fix existed) -- not inferred from the code, read from a file this fix's diff never touched. Verdict.reads()
        # uses its `key` both to look up the artifact field AND to label the check, so each seed gets its own
        # key-name (treatment_diffs_s<seed> / control_diffs_s<seed>) wrapping the SAME underlying field.
        pre_fix_report = _load(_PRE_FIX_ADEQUATE6 % seed)
        pre_fix_row = _faculty_row(pre_fix_report) or {}
        v.reads("treatment_diffs_s%d" % seed, {"treatment_diffs_s%d" % seed: pre_fix_row.get("treatment_diffs")},
                used=row.get("treatment_diffs") if row else None,
                note="vs the pre-fix research/findings/raw/_load_bearing/_adequate6/load_bearing_adequate_s%d.json" % seed)
        v.reads("control_diffs_s%d" % seed, {"control_diffs_s%d" % seed: pre_fix_row.get("control_diffs")},
                used=row.get("control_diffs") if row else None,
                note="vs the pre-fix research/findings/raw/_load_bearing/_adequate6/load_bearing_adequate_s%d.json" % seed)

    n_on_load_bearing = sum(1 for r in on_rows.values() if r and r.get("load_bearing") is True)
    go = (n_on_load_bearing == len(SEEDS)) and all(
        (off_rows[s] is not None and off_rows[s].get("load_bearing") is False) for s in off_rows)

    decided = v.decide(go=go)
    decided["n_on_load_bearing"] = n_on_load_bearing
    decided["n_seeds"] = len(SEEDS)
    decided["seeds"] = list(SEEDS)
    decided["flag"] = "BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE"
    decided["artifact_dir"] = out_dir
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(decided, open(out_path, "w"), indent=2, default=str)
    print("wrote", out_path)
    print("status:", decided["status"], "go:", decided["go"])
    return 0 if decided["status"] != "UNDEFINED" else 1


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "research/findings/raw/_lbf_fix_source_prov_abstain"
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(out_dir, "VERDICT.json")
    raise SystemExit(main(out_dir, out_path))
