import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, "/home/dant123/Projects/sim-worktrees/readfix-29runner-audit")

from research.runners.comprehension_production_organ import ComprehensionProductionOrgan, build_battery
from research.runners._spiking_comprehension_monitor_derisk import _build_comp

import inspect
print("enable_ou_process check via _build_comp source snippet:")
src = inspect.getsource(_build_comp)
for line in src.splitlines():
    if "ou" in line.lower():
        print(" ", line)
