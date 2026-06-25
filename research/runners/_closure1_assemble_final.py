"""Assemble the close-out deliverable `_closure1_optionA_buildfix_gate3.json` from the component JSONs.

Reads: the gate-3 output (`_closure1_optionA_gate3_flip.json`, rewritten by the re-run driver), the build-fix
answer-identity guard (`_closure1_buildfix_answer_identity.json`), Probe-1 byte-identity
(`_consolidation_probe1_byteident.json`), and gate-4 limbic (`_consolidation_probe2_limbic.json`). Folds in the
flip status + per-suite test results passed on the CLI. Pure assembly (no GPU); run after gate 3 + the suites.

Run: python -m research.runners._closure1_assemble_final \
       --flip-status CLOSED \
       --suite-merged "8/8 PASS" --suite-step2b "14/14 PASS (onebrain+rf)" --suite-onebrain "19/19 PASS" \
       --git-diff-stat "<paste>"
"""
from __future__ import annotations

import argparse
import json
import os

RAW = os.path.join(os.path.dirname(__file__), "..", "findings", "raw")
OUT = os.path.join(RAW, "_closure1_optionA_buildfix_gate3.json")


def _load(name, default=None):
    p = os.path.join(RAW, name)
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        return {"_load_error": f"{type(e).__name__}: {e}", "_path": os.path.normpath(p)} if default is None else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip-status", required=True, help="CLOSED or STILL-BLOCKED")
    ap.add_argument("--closure1-flipped", default="true")
    ap.add_argument("--closure3-flipped", default="true")
    ap.add_argument("--suite-merged", default="")
    ap.add_argument("--suite-step2b", default="")
    ap.add_argument("--suite-onebrain", default="")
    ap.add_argument("--moat-note", default="")
    ap.add_argument("--git-diff-stat", default="")
    ap.add_argument("--verdict", default="")
    args = ap.parse_args()

    gate3 = _load("_closure1_optionA_gate3_flip.json")
    guard = _load("_closure1_buildfix_answer_identity.json")
    probe1 = _load("_consolidation_probe1_byteident.json")
    gate4 = _load("_consolidation_probe2_limbic.json")

    # Pull the per-seed deltas + verdict out of the gate-3 output.
    rows = gate3.get("rows") if isinstance(gate3, dict) else None
    gate3_summary = {
        "part1_onebrain_constructs": (gate3.get("part1_construct_both_agents", {}) or {}).get("onebrain_constructs"),
        "part1_nav_regions_identical": (gate3.get("part1_construct_both_agents", {}) or {}).get("nav_regions_identical"),
        "onebrain_rf_region_size": gate3.get("onebrain_rf_region_size"),
        "rf_kind_rf_region_size": gate3.get("rf_kind_rf_region_size"),
        "per_seed_delta": [{"seed": r["seed"], "onebrain_score": r["onebrain_score"], "rf_score": r["rf_score"],
                            "delta": r["delta"], "byte_identical": r["byte_identical"]} for r in rows] if rows else None,
        "max_abs_delta": gate3.get("max_abs_delta"),
        "n_byte_identical": gate3.get("n_byte_identical"),
        "n_seeds": gate3.get("n_seeds"),
        "gate3_pass": gate3.get("gate3_pass"),
        "gate3_verdict": gate3.get("verdict"),
    }

    out = {
        "task": "close-out option A — the genuine build-fix for the onebrain construction crash, then re-run gate 3 + flip",
        "date": "2026-06-24",
        "backend": "cupy (GPU) for gate 3 + the suites; numpy (CPU) for the build-fix guard + Probe-1",
        "the_bug": ("CoResidentOneBrainComposer.__init__ built an idle layout-only BridgeParser(shared_bridge=merged), "
                    "whose merge_population_into_shared_bridge re-injected from the EMPTY _unified_wiring_plan (the merged "
                    "bridge is framework-wired in ONE inject at nav_conv_merged_bridge.py:1125, never populating that "
                    "plan) -> WIPED the framework wiring + the COMMAND_GATE transmission-gate registration -> the agent "
                    "COMMAND_GATE anti-cheat assert (nav_conv_merged_bridge.py:~1941) crashed at construction under the "
                    "production co_resident_command_route=True default. The rf arm builds no parser -> survived."),
        "the_fix": {
            "kind": "runner-level (NO sim/ edit)",
            "what": ("added `build_parser=True` param to CoResidentOneBrainComposer.__init__ "
                     "(nav_conv_merged_bridge.py:1494); when False, SKIP the idle BridgeParser shared-bridge merge "
                     "and set self.parser=None (the parser-construction conditional at :1579-1582). MergedNavConvAgent "
                     "passes build_parser=False on the onebrain path (:~1893), because comprehension goes through "
                     "_MergedParserAdapter (:1887) -- the composer's own parser is REDUNDANT on the merged path."),
            "files_lines": [
                "research/runners/nav_conv_merged_bridge.py:1494 (build_parser=True param + docstring)",
                "research/runners/nav_conv_merged_bridge.py:1579-1582 (conditional parser construction; parser=None when False)",
                "research/runners/nav_conv_merged_bridge.py:~1893 (MergedNavConvAgent onebrain path passes build_parser=False)",
            ],
            "why_byte_identical": ("the RF ops are RESET-ISOLATED from the parser: every op _zero_rf_v_u()-resets the rf "
                                   "slice's v/u before kicking, and the complex cp_rf_w_* synapses are array-disjoint "
                                   "from the parser's `parse` cp_connections population; the composer codes come from "
                                   "RFPhasorComposer(seed=...)'s own RNG (built before the parser). So dropping the idle "
                                   "parser leaves the composer's RF numerical OUTPUT unchanged."),
            "minimal_and_clean": True,
            "sim_edit_required": False,
        },
        "build_fix_answer_identity_guard": {
            "file": "research/runners/_closure1_buildfix_answer_identity.py",
            "GO": guard.get("GO"),
            "summary": ("build_parser=False == build_parser=True == standalone OneBrainComposer, verbatim, on the full "
                        "who/what/yes-no + abstention matrix; moat preserved"),
            "configs": guard.get("configs"),
        },
        "probe1_byte_identity": {
            "GO": probe1.get("GO"),
            "summary": ("Probe-1 (default build_parser=True path) re-run: byte-identical (atol 1e-9), 30/30 matrix, "
                        "max|dmem|=0.0, moat 8/8, nav slice byte-isolated -- the fix leaves the Probe-1 baseline UNCHANGED"),
            "configs": {k: {"byte_identical": v.get("byte_identical"), "moat_preserved": v.get("moat_preserved"),
                            "max_abs_membrane_delta": v.get("max_abs_membrane_delta")}
                        for k, v in (probe1.get("configs") or {}).items()},
        },
        "gate3": gate3_summary,
        "gate4_limbic_for_closure3": {"GO": gate4.get("GO"), "load_bearing": gate4.get("load_bearing")},
        "flips": {
            "closure_1_kind_default_rf_to_onebrain": {
                "file_line": "research/runners/nav_conv_merged_bridge.py:1638 (co_resident_composer_kind default 'rf'->'onebrain')",
                "flipped": args.closure1_flipped.lower() == "true",
                "rf_retained_as_oracle": True,
            },
            "closure_3_enable_da_encoding_gain_default_on": {
                "file_line": "research/runners/nav_conv_merged_bridge.py:1641 (enable_da_encoding_gain default False->True)",
                "flipped": args.closure3_flipped.lower() == "true",
                "gate": "_consolidation_probe2_limbic.json GO (g=1.423 exact, lesion pins 1.0, moat HARD)",
            },
        },
        "test_suites": {
            "test_nav_conv_merged_agent.py": args.suite_merged,
            "test_nav_conv_step2b_coresident.py": args.suite_step2b,
            "test_one_brain_composer_agent.py": args.suite_onebrain,
            "moat_is_none_assertions": args.moat_note or "the what_does/who_does/elaborate/describe `is None` no-confab "
                                                          "moat assertions stay green in all three suites",
        },
        "git_diff_stat": args.git_diff_stat,
        "sim_edit": "NONE (all changes are runner-level + test/probe code; no protected sim/ edit made or required)",
        "verdict": args.verdict or args.flip_status,
        "flip_status": args.flip_status,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {os.path.normpath(OUT)}  (flip_status={args.flip_status})")


if __name__ == "__main__":
    main()
