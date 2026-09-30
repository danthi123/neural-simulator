"""BRAIN_BG_CREDIT is default-off, and with it off the production BG organ is byte-identical to the pinned
pre-change SHA f52c6d246 (compared IN DATA: the same races, spike for spike, from the old and the new module)."""
import importlib.util
import os
import subprocess
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
import pytest

PINNED = "f52c6d246"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = "research/runners/bg_action_selection_production_organ.py"


def _pinned_module(tmp_path):
    src = subprocess.check_output(["git", "-C", ROOT, "show", "%s:%s" % (PINNED, REL)])
    path = tmp_path / "organ_pinned.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("organ_pinned", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _races(mod, n=4):
    org = mod.BGActionSelector(seed=42)
    out = []
    for k in range(n):
        r = org.select_once(0.0, 1.0) if k % 2 == 0 else org.select_once(1.0, 0.0)
        out.append((r["winner"], r["decision_step"], tuple(r["motor_spikes"]), r["loser_ratio"]))
    w = np.asarray(org._bridge.cp_connections.data).tobytes()
    v = np.asarray(org._bridge.cp_membrane_potential_v).tobytes()
    return out, w, v


def test_flag_default_off(monkeypatch):
    import research.runners.bg_action_selection_production_organ as organ
    monkeypatch.delenv("BRAIN_BG_CREDIT", raising=False)
    assert organ.bg_credit_enabled() is False
    with pytest.raises(RuntimeError):
        organ.build_credit_organ(42)
    for v in ("0", "false", "off", ""):
        monkeypatch.setenv("BRAIN_BG_CREDIT", v)
        assert organ.bg_credit_enabled() is False


def test_off_path_byte_identical_to_pinned_sha(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAIN_BG_CREDIT", raising=False)
    import research.runners.bg_action_selection_production_organ as organ
    organ.reset_organs()
    new = _races(organ)
    old = _races(_pinned_module(tmp_path))
    assert new[0] == old[0]
    assert new[1] == old[1] and new[2] == old[2]


def test_default_path_does_not_import_the_extension():
    env = dict(os.environ)
    env.pop("BRAIN_BG_CREDIT", None)
    env["SIM_BACKEND"] = "numpy"
    code = ("import sys, research.runners.bg_action_selection_production_organ as o; "
            "print('research.runners.bg_corticostriatal_credit' in sys.modules)")
    out = subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, env=env, text=True).strip().splitlines()
    assert out[-1] == "False"
