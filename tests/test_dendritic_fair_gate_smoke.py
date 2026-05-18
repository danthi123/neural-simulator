"""Import/signature + <3-seeds->exit2 + tiny-synth pipeline-turns
smoke (the project pattern; the decisive run is the controller's
Task 5). Also asserts the no-autograd invariant transitively."""
import subprocess, sys, json, inspect


def test_module_imports_and_shape():
    import research.runners.dendritic_fair_gate as m
    assert hasattr(m, "main") and callable(m.main)
    src = inspect.getsource(m)
    assert "torch" not in src and "autograd" not in src
    import sim.dendritic_mlp as dmm
    assert "torch" not in inspect.getsource(dmm)
    assert "autograd" not in inspect.getsource(dmm)


def test_fewer_than_three_seeds_exits_2():
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.dendritic_fair_gate",
         "--seeds", "42,43", "--tiny-synth"],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "NOT RUNNABLE" in r.stdout


def test_tiny_synth_pipeline_turns(tmp_path):
    out = str(tmp_path / "d.json")
    r = subprocess.run(
        [sys.executable, "-m", "research.runners.dendritic_fair_gate",
         "--seeds", "42,43,44", "--tiny-synth", "--out", out,
         "--ckpt-dir", str(tmp_path / "ck")],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = json.loads(open(out, encoding="utf-8").read())
    assert d["n_seeds"] == 3 and d["GATE"] in ("VOID","PASS","FAIL")
    for s in d["per_seed"]:
        assert s["verdict"]["GATE"] in ("VOID","PASS","FAIL")
