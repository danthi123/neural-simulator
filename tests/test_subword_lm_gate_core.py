import numpy as np
from research.runners.subword_lm_gate_core import (
    perplexity, shuffled_token_control, distinct_ngram_ratio,
    verbatim_copy_fraction, gs_verdict, gs_aggregate_multiseed,
    _GS_PPL_MARGIN, _GS_GENERALIZATION_MAX, _GS_DISTINCT_MIN,
    _GS_COPY_MAX, _GS_MIN_SEEDS, _GS_ABS_COMPETENCE_PPL_RATIO,
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
                   distinct=0.7, copy_frac=0.05, has_shuffled_control=True,
                   uniform_ppl=1000.0)
    assert v["GATE"] == "PASS"

def test_no_shuffled_control_is_fail_even_if_perfect():
    v = gs_verdict(1.0, 1e9, 1.0, 1.0, 0.0, has_shuffled_control=False,
                   uniform_ppl=1e9)
    assert v["GATE"] == "FAIL"

def test_memorization_is_fail():
    v = gs_verdict(heldout_ppl=100.0, shuffled_ppl=1e9, train_ppl=5.0,
                   distinct=0.9, copy_frac=0.0, has_shuffled_control=True,
                   uniform_ppl=1e9)
    assert v["GATE"] == "FAIL"

def test_degenerate_or_copy_generation_is_fail():
    assert gs_verdict(5,99,5,0.10,0.0,True,uniform_ppl=1e9)["GATE"] == "FAIL"
    assert gs_verdict(5,99,5,0.9,0.80,True,uniform_ppl=1e9)["GATE"] == "FAIL"

def test_results_cannot_move_the_fixed_bars():
    assert (_GS_PPL_MARGIN, _GS_GENERALIZATION_MAX, _GS_DISTINCT_MIN,
            _GS_COPY_MAX, _GS_MIN_SEEDS) == (0.20, 1.5, 0.5, 0.20, 3)
    assert _GS_ABS_COMPETENCE_PPL_RATIO == 1.0
    b = gs_verdict(1e-9, 1e9, 1e-9, 1.0, 0.0, True, uniform_ppl=1.0)
    assert b["bars"] == {"ppl_margin":0.20,"generalization_max":1.5,
                         "distinct_min":0.5,"copy_max":0.20,
                         "abs_competence_ppl_ratio":1.0}

def test_multiseed_requires_3_and_all_pass():
    P = {"GATE":"PASS"}; F = {"GATE":"FAIL"}
    assert gs_aggregate_multiseed([P,P,P])["GATE"] == "PASS"
    assert gs_aggregate_multiseed([P,F,P])["GATE"] == "FAIL"
    assert gs_aggregate_multiseed([P,P])["GATE"] == "FAIL"
    assert gs_aggregate_multiseed([])["GATE"] == "FAIL"

def test_infinite_shuffled_control_cannot_manufacture_pass():
    # an inf shuffled-control perplexity would make the load-bearing
    # margin check vacuous (heldout <= 0.8*inf) -> must be FAIL.
    v = gs_verdict(heldout_ppl=50.0, shuffled_ppl=float("inf"),
                   train_ppl=40.0, distinct=0.9, copy_frac=0.05,
                   has_shuffled_control=True, uniform_ppl=1e9)
    assert v["GATE"] == "FAIL" and v["real_structure_vs_shuffled"] is False
    # nan/inf heldout or train ppl also never pass
    assert gs_verdict(float("nan"), 100.0, 50.0, 0.9, 0.0,
                      True, uniform_ppl=1e9)["GATE"] == "FAIL"
    assert gs_verdict(50.0, 100.0, float("inf"), 0.9, 0.0,
                      True, uniform_ppl=1e9)["GATE"] == "FAIL"

def test_minseeds_floor_is_unbypassable():
    P = {"GATE": "PASS"}
    # a caller trying to weaken the >=3 floor cannot: 1 seed stays FAIL
    r = gs_aggregate_multiseed([P], min_seeds=1)
    assert r["GATE"] == "FAIL"
    assert r["min_seeds"] == 3 and r["min_seeds_requested"] == 1
    # can only be STRENGTHENED (3 PASS but caller demands 5 -> FAIL)
    assert gs_aggregate_multiseed([P, P, P], min_seeds=5)["GATE"] == "FAIL"
    # default 3-seed all-pass still PASS (no regression)
    assert gs_aggregate_multiseed([P, P, P])["GATE"] == "PASS"

def test_shuffled_control_on_degenerate_input_does_not_false_pass():
    import numpy as np
    # all-identical tokens: control degenerates to identity, so it
    # cannot beat itself by 20% -> the gate must NOT PASS off it.
    out = shuffled_token_control([5, 5, 5], np.random.default_rng(0))
    assert sorted(out) == [5, 5, 5]            # same multiset
    assert shuffled_token_control([], np.random.default_rng(0)) == []

def test_noise_model_worse_than_random_cannot_pass_even_if_relative_bars_met():
    # THE Generator-S hole: real/control/train ALL astronomically bad
    # -> relative bars vacuously satisfied, but held-out ppl >> uniform
    # -> MUST FAIL on the absolute-competence floor.
    v = gs_verdict(heldout_ppl=117716.0, shuffled_ppl=224392.0,
                   train_ppl=133303.0, distinct=1.0, copy_frac=0.0,
                   has_shuffled_control=True, uniform_ppl=512.0)
    assert v["GATE"] == "FAIL"
    assert v["absolute_competence_beats_random"] is False
    # the relative bars WERE vacuously "met" -- proving the floor is
    # what catches it:
    assert v["real_structure_vs_shuffled"] is True
    assert v["generalizes_not_memorizes"] is True

def test_missing_uniform_baseline_is_fail_closed():
    # no uniform baseline supplied -> cannot certify competence -> FAIL
    v = gs_verdict(1.0, 1e9, 1.0, 1.0, 0.0, True)  # no uniform_ppl
    assert v["GATE"] == "FAIL" and v["absolute_competence_beats_random"] is False
    v2 = gs_verdict(1.0, 1e9, 1.0, 1.0, 0.0, True, uniform_ppl=float("nan"))
    assert v2["GATE"] == "FAIL"
    v3 = gs_verdict(1.0, 1e9, 1.0, 1.0, 0.0, True, uniform_ppl=0.0)
    assert v3["GATE"] == "FAIL"

def test_competent_model_still_passes_when_all_bars_met():
    # a genuinely-good model (held-out ppl 8 << uniform 512, beats
    # control by >20%, generalizes, non-degenerate, low copy) PASSES.
    v = gs_verdict(heldout_ppl=8.0, shuffled_ppl=40.0, train_ppl=6.0,
                   distinct=0.8, copy_frac=0.03,
                   has_shuffled_control=True, uniform_ppl=512.0)
    assert v["GATE"] == "PASS" and v["absolute_competence_beats_random"] is True
