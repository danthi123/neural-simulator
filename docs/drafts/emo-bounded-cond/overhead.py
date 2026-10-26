"""Cost of the out-of-range census (organ._step) against a bare simulation step, organ seed 1, numpy CPU. Writes SCRATCH/overhead.json."""
import os, sys, time, json
os.environ["SIM_BACKEND"] = "numpy"; os.environ["SIM_NO_PROVENANCE"] = "1"
sys.path.insert(0, os.getcwd())
from research.runners import emotion_category_organ as E
org = E.EmotionCategoryOrgan(seed=1).train(); b = org._bridge
reps = []
for rep in range(5):
    t = time.perf_counter()
    for _ in range(300): b._run_one_simulation_step()
    t_plain = (time.perf_counter() - t) / 300
    s = set(); t = time.perf_counter()
    for _ in range(300): org._step(s)
    t_cens = (time.perf_counter() - t) / 300
    reps.append({"step_plain_us": round(t_plain * 1e6, 1), "step_plus_census_us": round(t_cens * 1e6, 1),
                 "census_overhead_pct": round(100 * (t_cens - t_plain) / t_plain, 2)})
    print(reps[-1])
json.dump({"what": "census cost per step, organ seed 1 (920 neurons), numpy CPU, 300 steps x 5 reps, workstation under load", "reps": reps,
           "median_overhead_us": sorted(r["step_plus_census_us"] - r["step_plain_us"] for r in reps)[2]}, open("/home/dant123/Projects/sim-rev/emobc_scratch/overhead.json", "w"), indent=1)
