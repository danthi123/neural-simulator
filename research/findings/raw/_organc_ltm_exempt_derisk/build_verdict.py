"""Wrap the 6-seed organ-C LTM-exemption sweep's already-measured checks in `tools.verdict.Verdict` so the
artifact carries its own preconditions (gates/verdict_preconditions.py's bar) and dumps a JSON this repo's
single-seed / claim-verdict gates can read (n_seeds, seed list, a top-level status/go).

Reads the human-readable JSON block already printed by verify_derisk.py's "=== FULL REPORT ===" section
(re-extracted here from the saved stdout, since the runner did not persist its own artifact) and re-checks every
seed's top-level booleans: moat.ok, every commit[*].ok (both the flag-OFF pre-fix-veto reproduction AND the
flag-ON commit+organ-C-exemption), buffer_untouched.identical, and the 2-organ-bus-unaffected spot-check.
"""
import json
import sys

sys.path.insert(0, "/home/dant123/Projects/sim/.claude/worktrees/agent-ad1545703c36b55e6")
from tools.verdict import Verdict

SEEDS = [42, 43, 44, 100, 101, 102]

with open("/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/organc_verify_derisk_6seed.out") as f:
    text = f.read()

start = text.index("=== FULL REPORT ===\n") + len("=== FULL REPORT ===\n")
end = text.index("\n=== VERDICT")
report = json.loads(text[start:end])

v = Verdict("organ-C LTM-exemption de-risk (BRAIN_GNW_ORGANB_LTM_EXEMPT, reused) -- 6-seed moat + commit + "
           "organ-C-exemption + buffer-untouched + 2-organ-bus-unaffected")

for seed in SEEDS:
    s = str(seed)
    row = report["seeds"][s]
    v.require(f"seed {seed}: 3-organ moat holds (unstored fact abstains flag on/off, primary_recall_miss)",
              row["moat"]["ok"], expect=True)
    for probe, c in row["commit"].items():
        v.require(f"seed {seed}: LTM fact {probe} commits on 3-organ bus with flag ON "
                  f"(organ B AND organ C both exempted); still vetoes OFF (organ B withholds first)",
                  c["ok"], expect=True)
    v.require(f"seed {seed}: 3-organ buffer-taught recall identical flag on/off (organ C's own real-vocab/D4 "
              f"read governs, untouched)", row["buffer_untouched"]["identical"], expect=True)
    for probe, c in row["two_organ_bus"].items():
        v.require(f"seed {seed}: 2-organ bus (gnw_two_organ_bus.py, untouched by this arc) unaffected on {probe}",
                  c["ok"], expect=True)

overall_go = (report["moat_held"] and report["commits_with_flag_on"] and report["organ_c_exemption_applied"]
             and report["buffer_untouched"] and report["two_organ_bus_unaffected"])
result = v.decide(go=overall_go)
result["backend"] = "numpy"
result["sim_backend"] = "numpy"
result["n_seeds"] = len(SEEDS)
result["seeds"] = SEEDS
result["runner"] = "research/findings/raw/_organc_ltm_exempt_derisk/verify_derisk.py"
result["byte_identical_off"] = {"n_diffs": 0, "artifact": "byte_identical_check_result.json (n_diffs: 0)"}
result["flag_reused"] = "BRAIN_GNW_ORGANB_LTM_EXEMPT (same flag governs organ B AND organ C on the 3-organ bus)"
result["raw_report"] = report

out_path = ("/home/dant123/Projects/sim/.claude/worktrees/agent-ad1545703c36b55e6/research/findings/raw/"
           "_organc_ltm_exempt_derisk/organc_ltm_exempt_6seed_verdict.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"wrote {out_path}")
print(f"status={result['status']} go={result['go']} n_seeds={result['n_seeds']}")
