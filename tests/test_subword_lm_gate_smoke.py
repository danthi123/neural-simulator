import subprocess, sys

def test_import_and_helpers():
    import research.runners.subword_lm_gate as g
    assert callable(g.main)
    # BPE-invariant word-shuffle: same multiset of words, order changed
    import numpy as np
    s = g._word_shuffle("a b c d e f g h", np.random.default_rng(1))
    assert sorted(s.split()) == list("abcdefgh")

def test_fewer_than_3_seeds_is_not_runnable_exit_2():
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.subword_lm_gate",
         "--seeds", "42,43"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 2
    assert "NOT RUNNABLE" in r.stdout

def test_help_runs():
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.subword_lm_gate",
         "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "MULTI-SEED" in r.stdout
