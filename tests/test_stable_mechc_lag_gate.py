"""research/runners/stable_mechc_lag_gate.py: the output parser and the three pass rules (no brain is built)."""
from research.runners import stable_mechc_lag_gate as G

PRINTED = """  [recurrent: mean=0.471 max=41.177 nnz=1349617]
=== LearnedAssocGraph (seed=1) ===
learned-graph edges recovered: 20/24 oracle co-occurrence edges
learned top-associate in oracle: 3/9 concepts
"""


def test_parse_g1_reads_the_modules_printed_lines():
    assert G.parse_g1(PRINTED) == (20, 24, 3, 9)
    assert G.parse_g1("nothing here") is None


def test_g1_pass_needs_every_oracle_edge_and_every_top_associate():
    ok = {"edges_recovered": 24, "edges_oracle": 24, "top_match": 9, "top_total": 9}
    assert G.g1_pass(ok)
    assert not G.g1_pass({**ok, "edges_recovered": 23})
    assert not G.g1_pass({**ok, "top_match": 8})
    assert not G.g1_pass(None)


def test_g2_pass_needs_two_of_two_after_encode_and_after_swr():
    assert G.g2_pass({"n_pairs": 2, "post_encode_top1": 2, "post_swr_top1": 2})
    assert not G.g2_pass({"n_pairs": 2, "post_encode_top1": 1, "post_swr_top1": 2})
    assert not G.g2_pass({"n_pairs": 2, "post_encode_top1": 2, "post_swr_top1": 1})


def test_g3_pass_applies_the_derisks_own_per_seed_bars():
    d = {"main_order": 1.0, "main_exact": 1.0, "perm_order": 0.3, "perm_exact": 0.1, "nolearn_order": 0.4,
         "main_grammatical": 1.0, "ablate_grammatical": 0.0, "moat_calls_on_abstain": 0, "moat_produced_any": False,
         "answer_produced": True}
    assert G.g3_pass(d)
    assert not G.g3_pass({**d, "main_exact": 0.89})
    assert not G.g3_pass({**d, "main_order": 0.5, "main_exact": 0.5})
    assert not G.g3_pass({**d, "perm_order": 0.85})
    assert not G.g3_pass({**d, "main_grammatical": 0.9})
    assert not G.g3_pass({**d, "moat_calls_on_abstain": 1})
