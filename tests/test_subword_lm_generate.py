import inspect, numpy as np
from research.runners.subword_lm_generate import sample_next, generate

def test_sample_next_argmax_when_temp_zero():
    lg = np.array([0.1, 5.0, 0.2, 5.0])
    # temp 0 -> deterministic argmax, stable FIRST max on ties
    assert sample_next(lg, np.random.default_rng(0), temperature=0.0) == 1
    assert sample_next(lg, np.random.default_rng(9), temperature=0.0) == 1

def test_sample_next_temp_in_range_and_seed_reproducible():
    lg = np.array([1.0, 2.0, 3.0, 0.5])
    a = sample_next(lg, np.random.default_rng(42), temperature=1.0)
    b = sample_next(lg, np.random.default_rng(42), temperature=1.0)
    assert a == b and 0 <= a < 4                  # reproducible + in-range

def test_sample_next_degenerate_inputs_never_raise():
    assert 0 <= sample_next(np.array([0.0,0.0,0.0]),
                             np.random.default_rng(1), 1.0) < 3
    assert sample_next(np.array([np.inf, np.nan, -np.inf]),
                       np.random.default_rng(1), 1.0) in (0,1,2)
    assert sample_next(np.array([]), np.random.default_rng(1), 1.0) == 0

def test_generate_signature():
    p = inspect.signature(generate).parameters
    for k in ("layers","tok","prompt","n_tokens","T","xp","rng",
              "temperature"):
        assert k in p
