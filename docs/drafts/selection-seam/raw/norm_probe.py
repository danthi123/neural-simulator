import os, sys, json, dataclasses, time
os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, os.path.dirname(__file__))
from research.runners._vocal_action_selector_gate import selector_config
import _nch_sel as S
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
out = {}
for xw in (10.0,):
    cfg = dataclasses.replace(selector_config("v2"), commit_fsi_cross_weight=xw)
    t0 = time.time()
    for cond in ("intact", "no_arousal"):
        for vh in (-0.9, -0.45, 0.0, 0.45, 0.9):
            r = S.run_condition(seed, vh, condition=cond, trials=12, config=cfg)
            print(json.dumps({"seed": seed, "xw": xw, "cond": cond, "vh": vh, "commit": r["commit_rate"],
                              "wc": r["winner_counts"], "argmin": r["host_argmin"]}), flush=True)
    print("elapsed", round(time.time() - t0, 1))
