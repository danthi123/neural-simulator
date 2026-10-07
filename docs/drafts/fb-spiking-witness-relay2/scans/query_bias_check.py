"""Relay-2 follow-up (Claude review 2026-10-06, dev seeds only; a measurement, nothing in the organ is edited).

The organ's `_run` adds RELAY_BIAS_PA to every relay cell on every step ("a cell property, the same every step"), but
`_query` steps the bridge in its own loop with the external current zeroed and never adds it. With the bias gone the
relays fire on presence alone, the off-target relays drive their eviction pools, and those inhibit the written slot.
This script runs the same write + query with the organ's query as is (AS-IS) and with the bias applied inside the
query loop the way `_run` applies it (BIAS-IN-QUERY), and prints the belief read and the off-target relay spikes.

Run: SIM_BACKEND=numpy python3 docs/drafts/fb-spiking-witness-relay2/scans/query_bias_check.py [seeds] [battery_seed]
  seeds = comma list (default 1-8); battery_seed = also run the 10-case battery on that seed both ways (default 7).
Writes nothing.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan import O, battery
from sim.backend import to_host

OP = dict(O._SHARED, MODE="relay_nmda", RELAY_BIAS_PA=-250.0, W_WORLD_RELAY=2.0, W_PRES_RELAY=3.0,
          W_RELAY_BELIEF=48.0, W_RELAY_EVICT=8.0, W_EVICT_BELIEF=100.0)
_orig_query = O.SpikingWitnessFalseBeliefOrgan._query


def _query_with_bias(self, lesion):
    b, idx, xp = self._bridge, self._idx, self._xp
    orig_step = b._run_one_simulation_step

    def step():
        b.cp_external_input_current[idx["relay_all"]] += xp.float32(O.RELAY_BIAS_PA)   # after _query zeroes it
        orig_step()
    b._run_one_simulation_step = step
    try:
        return _orig_query(self, lesion)
    finally:
        b._run_one_simulation_step = orig_step


def _apply_op():
    g = vars(O)
    for k, v in O._BASE_PARAMS.items():
        g[k] = v
    g.update(OP)


def one(seed, fixed):
    _apply_op()
    O.SpikingWitnessFalseBeliefOrgan._query = _query_with_bias if fixed else _orig_query
    o = O.SpikingWitnessFalseBeliefOrgan(seed)
    o.perceive_arrive()
    o.observe_event(0)
    b, idx = o._bridge, o._idx
    rel = {k: 0 for k in range(O.K_LOC)}
    orig_step = b._run_one_simulation_step

    def count_step():
        orig_step()
        f = to_host(b.cp_firing_states)
        for k in range(O.K_LOC):
            rel[k] += int(f[idx["relay_dev"][k]].sum())
    b._run_one_simulation_step = count_step
    q = o.query()
    b._run_one_simulation_step = orig_step
    O.SpikingWitnessFalseBeliefOrgan._query = _orig_query
    return q, rel


if __name__ == "__main__":
    seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "1,2,3,4,5,6,7,8").split(",")]
    bseed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    for s in seeds:
        for fixed in (False, True):
            q, rel = one(s, fixed)
            r = {k: round(v, 4) for k, v in q["belief_rates"].items()}
            print(f"seed {s} {'BIAS-IN-QUERY' if fixed else 'AS-IS        '} belief_loc={q['belief_loc']} "
                  f"rates={r} query relay spk by loc={[rel[k] for k in range(O.K_LOC)]}", flush=True)
    if bseed >= 0:
        for fixed in (False, True):
            O.SpikingWitnessFalseBeliefOrgan._query = _query_with_bias if fixed else _orig_query
            res = battery(bseed, OP)
            O.SpikingWitnessFalseBeliefOrgan._query = _orig_query
            n = sum(1 for v in res.values() if v[0])
            print(f"battery seed {bseed} {'BIAS-IN-QUERY' if fixed else 'AS-IS'}: {n}/{len(res)} pass "
                  + " ".join(f"{k}={'P' if v[0] else 'F'}" for k, v in res.items()), flush=True)
