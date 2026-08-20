"""FINALIZE the observe-vs-drive audit artifact: correct two normalize bugs from audit_fix.py + fold in affect-coloring.

BUG 1: the CORE faculties (semantic-recall/content-selection/moat-verify/in-loop-learning/anaphora-wm) were recorded
with classification='DRIVER' directly (no `answer_changed` field), so the generic normalize downgraded them to
NOT_CLEANLY_TESTABLE. They are DRIVERS (recall IS the answer; the moat abstains; a taught fact changes recall; 'it'
resolves) — shown by their own probes.
BUG 2: discourse-register was labeled DEAD_OBSERVER, but intact==lesion == 'no earlier event yet' means the register
NEVER POPULATED on the tiny-demo clauses (note_turn did not fold my two clauses into events) — a TRIGGER miss (same
family as episodic/causal), NOT a computed-verdict-going-nowhere. The ledger's own real-handler verify shows the
who-was-before answer FLIPPING cat->dog under the lesion (a DRIVER); this audit just used the wrong clause format ->
NOT_CLEANLY_TESTABLE here, ledger-cited.
"""
import json

_ART = "research/findings/raw/_observe_vs_drive/audit.json"
_AFC = "research/findings/raw/_observe_vs_drive/audit_affect_coloring.json"

CORE_DRIVERS = {"semantic-recall", "content-selection", "moat-verify", "in-loop-learning", "anaphora-wm"}

data = json.load(open(_ART))

# fold in affect-coloring (the omitted ledger row #13) if not already present
keys = {f.get("key") for f in data["faculties"]}
if "affect-coloring" not in keys:
    data["faculties"].append(json.load(open(_AFC)))

for f in data["faculties"]:
    k = f.get("key")
    if k in CORE_DRIVERS:
        f["final_class"] = "DRIVER"
        f["final_rationale"] = ("the recall IS the answer text (recall present vs honest abstain; a taught fact flips "
                                "recall; 'it' resolves to the referent) — load-bearing on the reply; its isolated "
                                "lesion is the internal composer.query_patient / _substrate_recall monkeypatch in "
                                "_production_lesion_probe, not an env flag")
    elif k == "discourse-register":
        f["final_class"] = "NOT_CLEANLY_TESTABLE"
        f["final_rationale"] = ("the register NEVER POPULATED on the tiny-demo clauses (intact==lesion=='no earlier "
                                "event yet') -> a TRIGGER miss (note_turn did not fold my clauses into events), NOT a "
                                "dead observer. Ledger real-handler verify shows the who-was-before answer FLIPPING "
                                "cat->dog under BRAIN_DISCOURSE_REGISTER_LESION -> a DRIVER on its own clause format")
    elif k == "affect-coloring":
        # already normalized in its own runner (NOT_CLEANLY_TESTABLE on numpy)
        f.setdefault("final_class", f.get("classification", "NOT_CLEANLY_TESTABLE"))

# recompute tally
tally = {}
for f in data["faculties"]:
    tally[f.get("final_class", "?")] = tally.get(f.get("final_class", "?"), 0) + 1
data["tally_final"] = tally
data["n_faculties_final"] = len(data["faculties"])
json.dump(data, open(_ART, "w"), indent=2, default=str)

print("FINAL TALLY:", tally, " n =", len(data["faculties"]))
for f in sorted(data["faculties"], key=lambda x: x.get("final_class", "")):
    print("  %-28s -> %s" % (f.get("key"), f.get("final_class")))
