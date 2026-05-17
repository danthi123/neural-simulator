import numpy as np
from research.runners.subword_lm_gate_core import (
    perplexity, shuffled_token_control, distinct_ngram_ratio,
    verbatim_copy_fraction, gs_verdict, gs_aggregate_multiseed,
    _GS_PPL_MARGIN, _GS_GENERALIZATION_MAX, _GS_DISTINCT_MIN,
    _GS_COPY_MAX, _GS_MIN_SEEDS,
)

def test_perplexity_is_exp_mean_nll():
    assert abs(perplexity([0.0, 0.0]) - 1.0) < 1e-9
    assert abs(perplexity([np.log(4)]) - 4.0) < 1e-6
    assert perplexity([]) == float("inf")

def test_shuffled_control_is_a_permutation_not_identity():
    ids = list(range(50))
    out = shuffled_token_control(ids, np.random.default_rng(1))
    assert sorted(out) == sorted(ids) and out != ids

def test_distinct_and_copy_metrics():
    assert distinct_ngram_ratio([1,2,3,1,2,3], n=3) == 3/4
    g = [1,2,3,4]; tr = [9,1,2,3,8]
    assert abs(verbatim_copy_fraction(g, tr, n=3) - 0.5) < 1e-9

def test_verdict_passes_only_when_all_fixed_bars_met():
    v = gs_verdict(heldout_ppl=10.0, shuffled_ppl=20.0, train_ppl=8.0,
                   distinct=0.7, copy_frac=0.05, has_shuffled_control=True)
    assert v["GATE"] == "PASS"

def test_no_shuffled_control_is_fail_even_if_perfect():
    v = gs_verdict(1.0, 1e9, 1.0, 1.0, 0.0, has_shuffled_control=False)
    assert v["GATE"] == "FAIL"

def test_memorization_is_fail():
    v = gs_verdict(heldout_ppl=100.0, shuffled_ppl=1e9, train_ppl=5.0,
                   distinct=0.9, copy_frac=0.0, has_shuffled_control=True)
    assert v["GATE"] == "FAIL"

def test_degenerate_or_copy_generation_is_fail():
    assert gs_verdict(5,99,5,0.10,0.0,True)["GATE"] == "FAIL"
    assert gs_verdict(5,99,5,0.9,0.80,True)["GATE"] == "FAIL"

def test_results_cannot_move_the_fixed_bars():
    assert (_GS_PPL_MARGIN, _GS_GENERALIZATION_MAX, _GS_DISTINCT_MIN,
            _GS_COPY_MAX, _GS_MIN_SEEDS) == (0.20, 1.5, 0.5, 0.20, 3)
    b = gs_verdict(1e-9, 1e9, 1e-9, 1.0, 0.0, True)
    assert b["bars"] == {"ppl_margin":0.20,"generalization_max":1.5,
                         "distinct_min":0.5,"copy_max":0.20}

def test_multiseed_requires_3_and_all_pass():
    P = {"GATE":"PASS"}; F = {"GATE":"FAIL"}
    assert gs_aggregate_multiseed([P,P,P])["GATE"] == "PASS"
    assert gs_aggregate_multiseed([P,F,P])["GATE"] == "FAIL"
    assert gs_aggregate_multiseed([P,P])["GATE"] == "FAIL"
    assert gs_aggregate_multiseed([])["GATE"] == "FAIL"
