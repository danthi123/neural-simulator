"""Record the organ's reads from the UNMODIFIED code (main 011fa0036) at dev seed 1, for the byte-identity pin."""
import os, sys, json
os.environ["SIM_BACKEND"] = "numpy"; os.environ["SIM_NO_PROVENANCE"] = "1"; os.environ["PYTHONHASHSEED"] = "0"
sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd())
from research.runners import emotion_category_organ as E
from research.runners import emotion_category_dev as D
org = E.EmotionCategoryOrgan(seed=1).train()
out = {}
for c in E.CATS:
    for _ in range(2):
        o = org.observe(("g",), D.situation_text(c))
    r = org.read(("g",))
    out[c] = {k: r[k] for k in ("category", "slot_counts", "readout_counts", "gate_counts", "co_commit", "read_invariant")}
    out[c]["obs_appraisal"] = o["appraisal_counts"]
print(json.dumps(out, sort_keys=True))
json.dump(out, open("/home/dant123/Projects/sim-rev/emobc_scratch/golden_main_seed1.json", "w"), indent=1, sort_keys=True)
