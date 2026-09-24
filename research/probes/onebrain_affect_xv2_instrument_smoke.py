"""INSTRUMENT SMOKE (not a gate run): the v2 arm-X battery on a 2-organ pool (surprise + affect ladder + the
arousal->surprise edge), seed 7 (a non-gate seed). Checks the code path runs and reports X0 / S* / flips."""
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
from research.runners import _onebrain_affect_pool_verify as V  # noqa: E402
from research.runners.onebrain_affect_pool import AFFECT_DESCRIPTOR, AROUSAL_XEDGE, affect_descriptors  # noqa: E402
from research.runners.onebrain_merge_framework import merge_organs  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
out = sys.argv[2]
t0 = time.time()
descs = affect_descriptors()
sur = [d for d in descs if d.key == "surprise"][0]
pool = merge_organs([sur, AFFECT_DESCRIPTOR], seed, config_descriptors=descs, wire=True, cross_edges=[AROUSAL_XEDGE])
sorg, lad = V._surprise_and_ladder(pool, seed)
sorg.ensure_built()
print(f"built N={int(pool.bridge.cp_membrane_potential_v.shape[0])} thr={sorg.threshold:.3f} "
      f"{time.time() - t0:.0f}s", flush=True)
prod = V.production_path_battery(pool, sorg)
base = V.arousal_surprise_battery(pool, lad, sorg, 0.0)
s_star = V.select_marginal_strength(base)
eq = V._verdicts_equal(base, prod, V.ASSERT_GRID) and V._verdicts_equal(base, prod, (600.0,), "confirm")
dhz = max(abs(a - b) for S in V.ASSERT_GRID for a, b in zip(base["contradict"][f"{S:g}"]["hz"],
                                                           prod["contradict"][f"{S:g}"]["hz"]))
print(f"X0 verdicts equal={eq} max|dHz|={dhz:.4f} S*={s_star} ({time.time() - t0:.0f}s)", flush=True)
print("a=0  frac:", {k: v["frac"] for k, v in base["contradict"].items()}, flush=True)
print("prod frac:", {k: v["frac"] for k, v in prod["contradict"].items()}, flush=True)
pos = V.arousal_surprise_battery(pool, lad, sorg, 1.0)
print("a=+1 frac:", {k: v["frac"] for k, v in pos["contradict"].items()}, "arousal Hz", pos["arousal_rung_hz"],
      flush=True)
print("mean Hz a0:", {k: round(v["mean_hz"], 3) for k, v in base["contradict"].items()}, flush=True)
print("mean Hz a1:", {k: round(v["mean_hz"], 3) for k, v in pos["contradict"].items()}, flush=True)
fl = V._newly(base["contradict"][f"{s_star:g}"], pos["contradict"][f"{s_star:g}"]) if s_star else None
print(f"flips at S*: {fl}; confirm FA 600: a0={base['confirm']['600']['n_surprised']} "
      f"a1={pos['confirm']['600']['n_surprised']} ({time.time() - t0:.0f}s)", flush=True)
json.dump({"seed": seed, "instrument_smoke_not_gate": True, "prod": prod, "base": base, "pos": pos,
           "s_star": s_star, "x0_equal": eq, "x0_max_dhz": dhz}, open(out, "w"), indent=1, default=float)
print("wrote", out)
