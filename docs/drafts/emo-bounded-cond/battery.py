"""Organ-level 10-case battery (research.runners.emotion_category_dev.run_seed) from the emo-bounded-cond worktree, with the
flag taken from the ENVIRONMENT exactly as production reads it. Adds (driver-side, no repo edit): a census of every step of
every bridge the seed builds (warm-up and training included) for v < -110 / v > 45, and a log of every read's engagement
fields. Usage: battery.py <flag 0|1> <tag> seeds...   -> SCRATCH/battery/<tag>.json"""
import os, sys, json, time
WT = "/home/dant123/Projects/sim-rev/wt-emobc"
flag = sys.argv[1]
os.environ["BRAIN_EMOTION_CATEGORY_BOUNDED"] = flag
os.environ["SIM_BACKEND"] = "numpy"; os.environ["PYTHONHASHSEED"] = "0"; os.environ["SIM_NO_PROVENANCE"] = "1"
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
os.chdir(WT)
sys.path.insert(0, WT)
import numpy as np
from sim import SimulationBridge
from research.runners import emotion_category_organ as E
from research.runners import emotion_category_dev as D
assert E.__file__.startswith(WT), E.__file__

CENS = {"per_bridge": {}, "vmin": 0.0, "vmax": 0.0}
_orig_step = SimulationBridge._run_one_simulation_step
def _step(self, *a, **k):
    r = _orig_step(self, *a, **k)
    v = self.cp_membrane_potential_v
    lo, hi = float(v.min()), float(v.max())
    if lo < CENS["vmin"]: CENS["vmin"] = lo
    if hi > CENS["vmax"]: CENS["vmax"] = hi
    if lo < -110.0 or hi > 45.0:
        CENS["per_bridge"].setdefault(id(self), set()).update(np.flatnonzero((v < -110.0) | (v > 45.0)).tolist())
    return r
SimulationBridge._run_one_simulation_step = _step

ENG = []
_orig_read = E.EmotionCategoryOrgan.read
def _read(self, *a, **k):
    r = _orig_read(self, *a, **k)
    ENG.append({x: r[x] for x in ("bounded_flag", "bounded_env", "bounded_readback", "oor_count", "oor_count_session")})
    return r
E.EmotionCategoryOrgan.read = _read

if __name__ == "__main__":
    tag = sys.argv[2]; seeds = [int(x) for x in sys.argv[3:]]
    recs = []
    out = "/home/dant123/Projects/sim-rev/emobc_scratch/battery/%s.json" % tag
    for s in seeds:
        t = time.time()
        CENS["per_bridge"] = {}; CENS["vmin"] = 0.0; CENS["vmax"] = 0.0; ENG.clear()
        r = D.run_seed(s, 0)
        sp = D.seed_pass(r)
        r["pass"] = bool(sp["pass"]); r["case_pass"] = sp["cases"]
        r["census"] = {"n_oor_neurons_all_steps": int(sum(len(v) for v in CENS["per_bridge"].values())),
                       "vmin": round(CENS["vmin"], 2), "vmax": round(CENS["vmax"], 2), "n_bridges_with_oor": len(CENS["per_bridge"])}
        r["engagement"] = {"n_reads": len(ENG), "bounded_flag": sorted({e["bounded_flag"] for e in ENG}),
                           "bounded_env": sorted({str(e["bounded_env"]) for e in ENG}),
                           "bounded_readback": sorted({e["bounded_readback"] for e in ENG}),
                           "oor_count_max": max(e["oor_count"] for e in ENG),
                           "oor_count_session_max": max(e["oor_count_session"] for e in ENG)}
        recs.append(r)
        print("[flag=%s] seed %d %s failing=%s census=%s eng=%s %.0fs" % (flag, s, "PASS" if sp["pass"] else "FAIL", {k: v for k, v in sp["cases"].items() if not v},
              r["census"], r["engagement"], time.time() - t), flush=True)
        json.dump({"flag": flag, "seeds": seeds, "records": recs}, open(out, "w"), default=str)
    v = D.battery_verdict(recs, tuple(seeds))
    print("[flag=%s] battery n_pass %d / %d" % (flag, v["n_pass"], v["n_seeds"]), flush=True)
