import subprocess, sys, inspect

def test_import_passes_uniform_ppl_no_bar_no_g1_no_gpu():
    import research.runners.generator_e_gate as g
    src = inspect.getsource(g)
    assert "uniform_ppl=" in src
    assert "song_g1_core" not in src
    assert "_GS_PPL_MARGIN =" not in src
    assert "_GS_ABS_COMPETENCE_PPL_RATIO =" not in src
    assert "import cupy" not in src and "_get_backend" not in src
    import numpy as np
    s = g._word_shuffle("a b c d e f g h", np.random.default_rng(1))
    assert sorted(s.split()) == list("abcdefgh")

def test_fewer_than_3_seeds_exit_2():
    r = subprocess.run([sys.executable, "-m",
        "research.runners.generator_e_gate", "--seeds", "42,43"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "NOT RUNNABLE" in r.stdout

def test_help():
    r = subprocess.run([sys.executable, "-m",
        "research.runners.generator_e_gate", "--help"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "MULTI-SEED" in r.stdout
