import subprocess, sys, inspect

def test_import_and_passes_uniform_ppl_and_no_bar_redef():
    import research.runners.generator_d_gate as g
    src = inspect.getsource(g)
    assert "uniform_ppl=" in src             # MUST pass the floor baseline
    assert "song_g1_core" not in src         # no g1 ref
    # no bar redefinition in the runner (bars live ONLY in gate_core)
    assert "_GS_PPL_MARGIN =" not in src
    assert "_GS_ABS_COMPETENCE_PPL_RATIO =" not in src
    # BPE-invariant word-shuffle control helper present
    import numpy as np
    s = g._word_shuffle("a b c d e f g h", np.random.default_rng(1))
    assert sorted(s.split()) == list("abcdefgh")

def test_fewer_than_3_seeds_exit_2():
    r = subprocess.run([sys.executable, "-m",
        "research.runners.generator_d_gate", "--seeds", "42,43"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "NOT RUNNABLE" in r.stdout

def test_help():
    r = subprocess.run([sys.executable, "-m",
        "research.runners.generator_d_gate", "--help"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "MULTI-SEED" in r.stdout
