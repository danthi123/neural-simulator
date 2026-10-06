import sys, itertools
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from scan import battery, O
seed = int(sys.argv[1])
O.CASES.append(("place_only", ["A", "0"], {}, 0))
for rr, wb in [("ampa", 300.0), ("ampa", 1000.0), ("ampa", 3000.0), ("nmda_slow", 50.0), ("nmda_slow", 150.0), ("nmda_slow", 400.0)]:
    p = dict(O._SHARED, MODE="relay_nmda", RELAY_BIAS_PA=0.0, W_WORLD_RELAY=0.25, W_PRES_RELAY=3.0,
             W_RELAY_BELIEF=wb, W_RELAY_EVICT=8.0, W_EVICT_BELIEF=100.0, EVICT_RECEPTOR="gaba_b", RELAY_BELIEF_RECEPTOR=rr)
    r = battery(seed, p, cases=("place_only",))
    print(f"{rr} wb={wb} | place {r['place_only'][0]} {r['place_only'][1]} relay {r['place_only'][2]}", flush=True)
