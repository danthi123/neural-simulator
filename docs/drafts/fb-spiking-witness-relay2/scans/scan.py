import sys, json, time, itertools, os
sys.path.insert(0, __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../../..")))
os.environ.setdefault("SIM_NO_PROVENANCE", "1")
import research.runners.tom_spiking_witness_organ as O

def battery(seed, params, cases=None):
    g = vars(O)
    for k, v in O._BASE_PARAMS.items(): g[k] = v
    for k, v in params.items(): g[k] = v
    out = {}
    for name, events, kw, want in O.CASES:
        if cases and name not in cases: continue
        c = O.run_case(seed, events, **kw)
        ok = c["belief_loc"] == want and O.store_read_ok(c["belief_rates"], want)
        w = [(x.get("loc"), x.get("presence_spikes"), x.get("relay_spikes")) for x in c["writes"]]
        out[name] = (ok, c["belief_rates"], w)
    return out

if __name__ == "__main__":
    base = dict(O._SHARED, MODE="relay_nmda", RELAY_BIAS_PA=0.0, W_WORLD_RELAY=float(sys.argv[2]),
                W_PRES_RELAY=float(sys.argv[3]), W_RELAY_BELIEF=48.0, W_RELAY_EVICT=8.0, W_EVICT_BELIEF=100.0)
    if len(sys.argv) > 4: base["RELAY_BIAS_PA"] = float(sys.argv[4])
    t = time.time()
    r = battery(int(sys.argv[1]), base, cases=("false_belief", "true_belief"))
    for k, v in r.items(): print(k, v[0], v[1], v[2])
    print("secs", round(time.time() - t, 1))
