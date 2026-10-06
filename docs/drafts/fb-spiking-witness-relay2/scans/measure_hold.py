"""Item 1 of the DIAGNOSIS 'Next measurement' (dev seeds only): the belief slots' per-step spike count and mean
membrane voltage (mV) from write onset through HOLD_STEPS, plus the relay / eviction / presence activity, next to the
same trace in read-out mode -- at the defect-1 operating point (RELAY_BIAS_PA -250, W_PRES 3.0, W_WORLD 2.0).

Run: SIM_BACKEND=numpy python3 docs/drafts/fb-spiking-witness-relay2/scans/measure_hold.py <seed> <mode> [wb] [loc]
  mode = relay_nmda (default) | readout ; wb = W_RELAY_BELIEF (default 48) ; loc = written slot (default 0)
Full per-step rows go to scans/out/<mode>_wb<wb>_seed<seed>_loc<loc>.jsonl; a summary table prints to stdout.

A measurement, not a lever: nothing in the organ is edited here (the scan scripts' own contract, DIAGNOSIS line 3).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scan import O
from sim.backend import to_host

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
mode = sys.argv[2] if len(sys.argv) > 2 else "relay_nmda"
wb = float(sys.argv[3]) if len(sys.argv) > 3 else 48.0
loc = int(sys.argv[4]) if len(sys.argv) > 4 else 0

g = vars(O)
for k, v in O._BASE_PARAMS.items():
    g[k] = v
if mode == "readout":
    g.update(dict(O._SHARED, MODE="readout", PRES_MIN=0.03, PRES_READ_STEPS=20))
else:
    g.update(dict(O._SHARED, MODE=mode, RELAY_BIAS_PA=-250.0, W_WORLD_RELAY=2.0, W_PRES_RELAY=3.0,
                  W_RELAY_BELIEF=wb, W_RELAY_EVICT=8.0, W_EVICT_BELIEF=100.0))

t0 = time.time()
o = O.SpikingWitnessFalseBeliefOrgan(seed)
o.perceive_arrive()
o._ensure()
b, idx = o._bridge, o._idx
K = O.K_LOC

belief_ix = [idx["belief_dev"][k] for k in range(K)]
relay_ix = {k: idx["relay_dev"][k] for k in idx.get("relay_dev", {})}
evict_key = next((k for k in idx if "evict" in str(k)), None)
if evict_key is None:
    print("NOTE idx keys (no evict key):", sorted(map(str, idx)))
evict_ix = idx[evict_key] if evict_key else {}
pres_ix = idx["presence"]
slot0 = belief_ix[loc]

cur = {"p": "idle", "i": 0}
phases = []                      # phase label per o._run call during observe
orig_run = o._run
def run_wrap(steps, drive, lesion, silence=False, count=()):
    if phases:
        cur["p"] = phases.pop(0)
        cur["i"] = 0
    return orig_run(steps, drive, lesion, silence=silence, count=count)
o._run = run_wrap

rows = []
orig_step = b._run_one_simulation_step
def step():
    orig_step()
    v = np.asarray(to_host(b.cp_membrane_potential_v))
    gi = np.asarray(to_host(b.cp_conductance_g_i))
    ge = np.asarray(to_host(b.cp_conductance_g_e))
    f = np.asarray(to_host(b.cp_firing_states))
    r = {"p": cur["p"], "s": cur["i"],
         "bspk": [int(f[ix].sum()) for ix in belief_ix],
         "bv": [round(float(v[ix].mean()), 2) for ix in belief_ix],
         "rspk": [int(f[ix].sum()) for ix in (relay_ix[k] for k in range(K))] if relay_ix else None,
         "evspk": [int(f[ix].sum()) for ix in (evict_ix[k] for k in range(K))] if evict_ix else None,
         "pres": int(f[pres_ix].sum()),
         "gi": round(float(gi[slot0].mean()), 3), "ge": round(float(ge[slot0].mean()), 3),
         "v0": round(float(v[slot0].mean()), 2)}
    rows.append(r)
    cur["i"] += 1
b._run_one_simulation_step = step

phases[:] = (["pres_read", "write", "hold"] if mode == "readout" else ["write", "hold"])
o.observe_event(loc)
cur["p"] = "query"                      # the query's own steps must not be counted as hold
q = o.query()
secs = round(time.time() - t0, 1)

# ── summary table ────────────────────────────────────────────────────────────────────────────────────────────────
def phase_rows(p):
    return [r for r in rows if r["p"] == p]

print(f"mode={mode} seed={seed} wb={wb} loc={loc} steps={len(rows)} secs={secs} "
      f"HOLD_STEPS={O.HOLD_STEPS} WRITE_STEPS={O.WRITE_STEPS}")
for p in ("pres_read", "write", "hold"):
    rs = phase_rows(p)
    if not rs:
        continue
    tot = [sum(r["bspk"][k] for r in rs) for k in range(K)]
    rtot = [sum((r["rspk"][k] if r["rspk"] else 0) for r in rs) for k in range(K)] if relay_ix else None
    evtot = [sum((r["evspk"][k] if r["evspk"] else 0) for r in rs) for k in range(K)] if evict_ix else None
    v_first, v_last = rs[0]["bv"][loc], rs[-1]["bv"][loc]
    v_max = max(r["bv"][loc] for r in rs)
    pres = sum(r["pres"] for r in rs)
    print(f"  {p:9s} n={len(rs):3d} belief_spk={tot} relay_spk={rtot} evict_spk={evtot} "
          f"pres_spk={pres} | slot{loc} v first {v_first} -> max {v_max} -> last {v_last} mV")
h = phase_rows("hold")
if h:
    qs = max(1, len(h) // 4)
    early = sum(r["bspk"][loc] for r in h[:qs])
    late = sum(r["bspk"][loc] for r in h[-qs:])
    per10 = [sum(r["bspk"][loc] for r in h[i:i + 10]) for i in range(0, len(h), 10)]
    print(f"  hold slot{loc}: spikes first-quarter {early}, last-quarter {late}, per-10-step {per10}")
w = phase_rows("write")
if w:
    act = [k for k in range(K) if any(r["bspk"][k] for r in w)]
    print(f"  write ignited slots: {act}")
print(f"  final query: belief_loc={q['belief_loc']} want={loc} rates={ {k: round(v,4) for k,v in q['belief_rates'].items()} }")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)
tag = f"{mode}_wb{int(wb)}_seed{seed}_loc{loc}".replace(".", "p")
path = os.path.join(out, tag + ".jsonl")
with open(path, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print("rows ->", path)