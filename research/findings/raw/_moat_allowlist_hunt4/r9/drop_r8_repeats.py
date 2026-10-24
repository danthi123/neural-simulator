"""Round 9: drop the cases whose sentence repeats a round-8 sentence word for word (the brief forbids repeats of rounds 3-8; the
skeptic could not read the round-8 case file, which was not in its tree, so its own check covered rounds 3-7 and the round-8 sentences
quoted in the spec). A round-8 reply's single sentences count as round-8 sentences.
Reads the skeptic's output (/home/dant123/Projects/sim-rev/moat-allow7/hunt9/cases.json, written by build_cases.py, unchanged) and
../r8/cases.json; writes cases_skeptic.json (a verbatim copy of the skeptic's output) and cases.json (the filtered set) beside this file.
usage (cwd = worktree, as a module so the provenance door stamps both outputs):
  python -m research.findings.raw._moat_allowlist_hunt4.r9.drop_r8_repeats"""
import research.runners  # noqa: F401  (provenance door)
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SKEPTIC = "/home/dant123/Projects/sim-rev/moat-allow7/hunt9/cases.json"
raw = open(SKEPTIC, encoding="utf-8").read()
r9 = json.loads(raw)
with open(os.path.join(HERE, "cases_skeptic.json"), "w", encoding="utf-8") as f:
    f.write(raw)
r8 = json.load(open(os.path.join(HERE, "..", "r8", "cases.json"), encoding="utf-8"))
seen = {c["sentence"] for c in r8} | {s for c in r8 for s in c.get("sentences", [])}
keep = [c for c in r9 if c["sentence"] not in seen]
dropped = [c["id"] for c in r9 if c["sentence"] in seen]
with open(os.path.join(HERE, "cases.json"), "w", encoding="utf-8") as f:
    json.dump(keep, f, indent=1, ensure_ascii=False)
print("dropped", dropped, "kept", len(keep))
