"""tom_spiking_witness_organ: QUERY_RELAY_BIAS (opt-in, Claude 2026-10-06, relay2 review). `_run` applies
RELAY_BIAS_PA to every relay cell on every step, but `_query` zeroes the input in its own loop, so at query time the
relays lost their bias, fired on presence alone, and the off-target eviction pools silenced the written belief slot
(docs/drafts/fb-spiking-witness-relay2/WORKLOG.md, 2026-10-06 review). The default keeps the registered GRID behavior."""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import pytest

import research.runners.tom_spiking_witness_organ as O
from sim.backend import to_host

_OP = dict(O._SHARED, MODE="relay_nmda", RELAY_BIAS_PA=-250.0, W_WORLD_RELAY=2.0, W_PRES_RELAY=3.0,
           W_RELAY_BELIEF=48.0, W_RELAY_EVICT=8.0, W_EVICT_BELIEF=100.0)


def test_default_is_the_registered_behavior():
    assert O.QUERY_RELAY_BIAS is False


def _write_then_query(query_bias):
    g = vars(O)
    saved = {k: g[k] for k in list(O._BASE_PARAMS) + ["QUERY_RELAY_BIAS"]}
    try:
        for k, v in O._BASE_PARAMS.items():
            g[k] = v
        g.update(_OP, QUERY_RELAY_BIAS=query_bias)
        o = O.SpikingWitnessFalseBeliefOrgan(7)
        o.perceive_arrive()
        o.observe_event(0)
        b, idx = o._bridge, o._idx
        off = [0]
        step = b._run_one_simulation_step

        def counting():
            step()
            f = to_host(b.cp_firing_states)
            off[0] += sum(int(f[idx["relay_dev"][k]].sum()) for k in range(1, O.K_LOC))
        b._run_one_simulation_step = counting
        q = o.query()
        b._run_one_simulation_step = step
        return q["belief_rates"], off[0]
    finally:
        g.update(saved)


@pytest.mark.parametrize("query_bias", [False, True])
def test_query_bias_keeps_off_target_relays_silent(query_bias):
    rates, off = _write_then_query(query_bias)
    if query_bias:
        assert off == 0 and rates[0] > 0.0 and all(rates[k] == 0.0 for k in range(1, O.K_LOC))
    else:   # the registered behavior, kept: the bias vanishes at query, the relays fire, the slot is silenced
        assert off > 0 and rates[0] == 0.0
