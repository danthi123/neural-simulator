import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, "/home/dant123/Projects/sim-worktrees/readfix-29runner-audit")
import numpy as np

from research.runners._replay_dg_pattern_separation_gate import (
    smoke_config, build_bridge, _input_patterns, _answer_assemblies, _probe, _dg_engram,
    _replay_consolidate,
)

seed = 42
cfg = smoke_config()
bridge, handles = build_bridge(seed, cfg)
regions = handles["regions"]
inputs = _input_patterns(seed, cfg, "similar")
answers = _answer_assemblies(seed, cfg, regions["answer"])
memories = {"m0": {"input": inputs["m0"], "answer": answers["m0"]},
            "m1": {"input": inputs["m1"], "answer": answers["m1"]}}

_ = _replay_consolidate(bridge, cfg, regions, memories, True, seed)

# repeat _probe on the SAME target, no intervening trial
r1 = _probe(bridge, cfg, regions, memories, "m0", True, seed)
r2 = _probe(bridge, cfg, regions, memories, "m0", True, seed)
identical = (r1 == r2)

# order dependence: probe m1 in between
r3 = _probe(bridge, cfg, regions, memories, "m0", True, seed)
_ = _probe(bridge, cfg, regions, memories, "m1", True, seed)
r4 = _probe(bridge, cfg, regions, memories, "m0", True, seed)
order_identical = (r3 == r4)

print(json.dumps({
    "probe_repeat": {"r1": r1, "r2": r2, "identical": identical,
                      "margin_delta": abs(r1["margin"] - r2["margin"])},
    "probe_order_dep": {"r3": r3, "r4": r4, "identical": order_identical,
                         "margin_delta": abs(r3["margin"] - r4["margin"])},
}, indent=2))
