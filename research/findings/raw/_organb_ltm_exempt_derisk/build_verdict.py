"""Wrap the 6-seed organ-B LTM-exemption sweep's already-measured checks in `tools.verdict.Verdict` so the
artifact carries its own preconditions (gates/verdict_preconditions.py's bar) and dumps a JSON this repo's
single-seed / claim-verdict gates can read (n_seeds, seed list, a top-level status/go).

Reads the human-readable JSON block already printed by verify_derisk.py's "=== FULL REPORT ===" section
(re-extracted here from the saved stdout, since the runner did not persist its own artifact) and re-checks
every seed's three top-level booleans (moat.ok, all commit[*].ok, buffer_untouched.identical) plus the
already-observed 3-organ organ-B-exempts-but-organ-C-blocks pattern (reported, not gated -- a SEPARATE,
un-fixed gap named in the finding).
"""
import json
import sys

sys.path.insert(0, "/home/dant123/Projects/sim/.claude/worktrees/agent-af06a18d790f070b8")
from tools.verdict import Verdict

SEEDS = [42, 43, 44, 100, 101, 102]

with open("/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/verify_derisk_6seed.out") as f:
    text = f.read()

start = text.index("=== FULL REPORT ===\n") + len("=== FULL REPORT ===\n")
end = text.index("\n=== VERDICT")
report = json.loads(text[start:end])

v = Verdict("organ-B LTM-exemption de-risk (BRAIN_GNW_ORGANB_LTM_EXEMPT) -- 6-seed moat + commit + buffer-untouched")

for seed in SEEDS:
    s = str(seed)
    row = report["seeds"][s]
    v.require(f"seed {seed}: moat holds (unstored fact abstains both flag on/off)", row["moat"]["ok"], expect=True)
    for probe, c in row["commit"].items():
        v.require(f"seed {seed}: LTM fact {probe} commits with flag ON (was vetoed OFF)", c["ok"], expect=True)
    v.require(f"seed {seed}: buffer-taught recall identical flag on/off", row["buffer_untouched"]["identical"],
              expect=True)
    # 3-organ bus: organ B's exemption reaches the 3-organ combine too (organ_b_confirmed True on every LTM
    # probe with the flag on); organ C's SEPARATE block is recorded as an observation, not gated here (it is
    # NOT this de-risk's mechanism -- see the finding's "organ C blocks LTM for a different reason" section).
    for probe, c in row["three_organ"].items():
        if probe.startswith("_"):
            continue
        v.require(f"seed {seed}: 3-organ organ-B exemption applied on {probe}", c["organb_ltm_exempt_applied"],
                  expect=True)
    v.require(f"seed {seed}: 3-organ moat holds on unstored fact",
              row["three_organ"]["_moat_unstored"]["committed"] is None, expect=True)

overall_go = (report["moat_held"] and report["commits_with_flag_on"] and report["buffer_untouched"])
result = v.decide(go=overall_go)
result["backend"] = "numpy"
result["sim_backend"] = "numpy"
result["n_seeds"] = len(SEEDS)
result["seeds"] = SEEDS
result["runner"] = "research/findings/raw/_organb_ltm_exempt_derisk/verify_derisk.py"
result["byte_identical_off"] = {"n_diffs": 0, "artifact": "byte_identical_check_result.json (n_diffs: 0)"}
result["three_organ_organ_c_blocks_separately"] = True
result["raw_report"] = report

out_path = "/home/dant123/Projects/sim/.claude/worktrees/agent-af06a18d790f070b8/research/findings/raw/_organb_ltm_exempt_derisk/organb_ltm_exempt_6seed_verdict.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"wrote {out_path}")
print(f"status={result['status']} go={result['go']} n_seeds={result['n_seeds']}")
