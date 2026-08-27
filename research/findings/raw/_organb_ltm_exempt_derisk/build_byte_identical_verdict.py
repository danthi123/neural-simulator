"""Wrap byte_identical_check.py's already-measured 0-diff result in tools.verdict.Verdict so the artifact
carries its own preconditions (gates/verdict_preconditions.py) + backend/device + provenance
(gates/device_and_cost.py, gates/artifact_provenance.py).
"""
import json
import sys

sys.path.insert(0, "/home/dant123/Projects/sim/.claude/worktrees/agent-af06a18d790f070b8")
from tools.verdict import Verdict

SRC = "/home/dant123/Projects/sim/.claude/worktrees/agent-af06a18d790f070b8/research/findings/raw/_organb_ltm_exempt_derisk/byte_identical_check_result.json"
with open(SRC) as f:
    raw = json.load(f)

v = Verdict("BRAIN_GNW_ORGANB_LTM_EXEMPT byte-identical-when-off (flag=False vs pre-patch code at HEAD)")
for row in raw["two_organ_rows"]:
    v.require(f"two_organ_combine flag-off == pre-patch HEAD, probe {row['probe']}", row["match"], expect=True)
for row in raw["three_organ_rows"]:
    v.require(f"three_organ_combine flag-off == pre-patch HEAD, probe {row['probe']}", row["match"], expect=True)
for row in raw["gate_rows"]:
    v.require(f"chat.gate() flag-off == pre-patch HEAD, q={row['q']!r}", row["match"], expect=True)

result = v.decide(go=(raw["n_diffs"] == 0))
result["backend"] = "numpy"
result["sim_backend"] = "numpy"
result["seed"] = 42
result["runner"] = "research/findings/raw/_organb_ltm_exempt_derisk/byte_identical_check.py"
result["n_diffs"] = raw["n_diffs"]
result["raw"] = raw

with open(SRC, "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"rewrote {SRC}: status={result['status']} go={result['go']}")
