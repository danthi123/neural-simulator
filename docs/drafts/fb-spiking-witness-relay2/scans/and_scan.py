"""The AND gate with the right metric (dev seed): on a WITNESSED move to loc 1, target relay spikes vs the mean of
the off-target relays (presence alone drives those); on an UNWITNESSED move, the target relay (world alone)."""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan import O
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
for bias, wp, ww in itertools.product([-150.0, -250.0], [1.5, 3.0], [0.5, 1.0, 2.0, 4.0]):
    g = vars(O)
    for k, v in O._BASE_PARAMS.items(): g[k] = v
    g.update(dict(O._SHARED, MODE="relay_nmda", RELAY_BIAS_PA=bias, W_WORLD_RELAY=ww, W_PRES_RELAY=wp,
                  W_RELAY_BELIEF=48.0, W_RELAY_EVICT=8.0, W_EVICT_BELIEF=100.0))
    tb = O.run_case(seed, ["A", "0", "1"])["writes"][1]["relay_by_loc"]
    fb = O.run_case(seed, ["A", "0", "L", "1"])["writes"][1]["relay_by_loc"]
    off = (tb[0] + tb[2] + tb[3]) / 3.0
    print(f"bias={bias} wp={wp} ww={ww} | witnessed target {tb[1]} off-target mean {off:.1f} | unwitnessed target {fb[1]}", flush=True)
