import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, "/home/dant123/Projects/sim-worktrees/readfix-29runner-audit")
import numpy as np

from research.runners._navsc_merged_opcheck import (
    _build_merged_with_sc, _present, MERGED_OP, F1_CASES,
)

seed = 42
bridge = _build_merged_with_sc(seed, MERGED_OP)
agent, goal = F1_CASES[0]

pk1, mn1, cc1, rus1 = _present(bridge, agent, goal, MERGED_OP)
pk2, mn2, cc2, rus2 = _present(bridge, agent, goal, MERGED_OP)
# intervening different case then re-read the first
_ = _present(bridge, *F1_CASES[3], op=MERGED_OP)
pk3, mn3, cc3, rus3 = _present(bridge, agent, goal, MERGED_OP)

print(json.dumps({
    "repeat_read": {"peak": [pk1, pk2], "mean": [mn1, mn2], "cc": [cc1, cc2], "rus": [rus1, rus2],
                     "identical": (pk1 == pk2 and mn1 == mn2 and cc1 == cc2 and rus1 == rus2)},
    "order_dependence": {"peak": [pk1, pk3], "cc": [cc1, cc3], "rus": [rus1, rus3],
                          "identical": (pk1 == pk3 and cc1 == cc3 and rus1 == rus3)},
}, indent=2))
