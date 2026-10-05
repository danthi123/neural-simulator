import sys, itertools
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from scan import battery, O
seed = int(sys.argv[1])
for ww, wp in itertools.product([0.25, 0.5, 0.8], [1.5, 3.0, 6.0]):
    p = dict(O._SHARED, MODE="relay_nmda", RELAY_BIAS_PA=0.0, W_WORLD_RELAY=ww, W_PRES_RELAY=wp,
             W_RELAY_BELIEF=48.0, W_RELAY_EVICT=8.0, W_EVICT_BELIEF=100.0)
    r = battery(seed, p, cases=("false_belief", "true_belief"))
    fb, tb = r["false_belief"], r["true_belief"]
    print(f"ww={ww} wp={wp} | unwitnessed relay={fb[2][1][2]} witnessed relay={tb[2][1][2]} place={tb[2][0][2]} | FB {fb[0]} {fb[1]} | TB {tb[0]} {tb[1]}", flush=True)
