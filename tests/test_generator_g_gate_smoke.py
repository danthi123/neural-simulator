import subprocess, sys, inspect


def test_import_no_bar_no_g1_uses_validated_moat_and_is_answered():
    import research.runners.generator_g_gate as g
    src = inspect.getsource(g)
    assert "abstention_gate" in src
    assert "song_g1_core" not in src
    assert "subword_lm_gate_core" not in src
    assert "_GG_UNGROUNDED_ENTITY_MAX =" not in src
    assert "grounded_decode" in src
    assert "generator_g_core" in src
    assert "is_answered" in src


def test_fewer_than_3_seeds_exit_2():
    r = subprocess.run([sys.executable, "-m",
        "research.runners.generator_g_gate", "--seeds", "42,43"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "NOT RUNNABLE" in r.stdout


def test_help():
    r = subprocess.run([sys.executable, "-m",
        "research.runners.generator_g_gate", "--help"],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
