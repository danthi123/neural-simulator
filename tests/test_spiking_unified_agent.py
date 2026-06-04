"""Stage-2a guard: the spiking unified agent reproduces the unified-agent benchmark's FLAT robust core
(flat / who / abstain) in genuine spikes, at the benchmark's 320-concept vocabulary.

Run: SIM_BACKEND=numpy python -m pytest tests/test_spiking_unified_agent.py -q
"""
from research.runners.spiking_unified_agent import run_core_benchmark, SpikingUnifiedAgent


def test_spiking_core_reproduces_benchmark():
    """flat / one-attribute / two-attribute / depth-1 clause / who / abstain all perfect in spikes at seed 42
    (N_dim=2048 -- the F=3 two-attribute resonator needs the dimension) — the spiking agent reproduces every
    benchmark category the numpy agent does, including embedded clauses (recursive decode) and the
    no-confabulation moat (abstention)."""
    res, wrong = run_core_benchmark(n_dim=2048, seed=42)
    assert res["flat"] == [8, 8], f"flat regressed: {res['flat']}  ({wrong})"
    assert res["1-attribute"] == [6, 6], f"one-attribute regressed: {res['1-attribute']}  ({wrong})"
    assert res["2-attribute"] == [5, 5], f"two-attribute regressed: {res['2-attribute']}  ({wrong})"
    assert res["clause-depth1"] == [5, 5], f"clause regressed: {res['clause-depth1']}  ({wrong})"
    assert res["who-query"] == [6, 6], f"who regressed: {res['who-query']}  ({wrong})"
    assert res["abstain"] == [6, 6], f"abstain (no-confabulation) regressed: {res['abstain']}  ({wrong})"


def test_spiking_agent_abstains_on_unknown_pair():
    """A direct no-confabulation check: a stored agent + a stored action that were never paired -> abstain."""
    agent = SpikingUnifiedAgent(["dog", "cat", "bird"], ["chase", "see"], n_dim=512, seed=42)
    agent.learn("dog", "chase", "cat")
    assert agent.query_patient("dog", "chase") == "cat"      # stored fact recovers
    assert agent.query_patient("dog", "see") is None         # never stored -> abstain (no confabulation)
    assert agent.query_patient("bird", "chase") is None      # never stored -> abstain
