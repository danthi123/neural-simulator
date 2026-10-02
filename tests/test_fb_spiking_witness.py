"""Packet fb-spiking-witness: the spiking presence flip-flop decides who saw the move (default OFF)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SIM_BACKEND", "numpy")
import webapp.false_belief_chat as F  # noqa: E402
# imported at COLLECTION time (a later file that puts tools/ on sys.path breaks a first `import sim`)
import research.runners.tom_spiking_witness_organ as M  # noqa: E402

FB = ("Sally puts the marble in the basket. Sally leaves the room. Anne moves the marble to the box. "
      "Where will Sally look for the marble?")


def test_anchor_reads_false_and_flag_unset_is_off(monkeypatch):
    assert F._FALSE_BELIEF_SPIKING_WITNESS_DEFAULT_ON is False
    monkeypatch.delenv("BRAIN_FALSE_BELIEF_SPIKING_WITNESS", raising=False)
    assert F.spiking_witness_enabled() is False
    monkeypatch.setenv("BRAIN_FALSE_BELIEF_SPIKING_WITNESS", "1")
    assert F.spiking_witness_enabled() is True


def test_flag_off_builds_the_w3_organ_with_host_witnessing(monkeypatch):
    """Flag off: the original organ, the host `present` dict decides, and observe_event gets witnessed=..."""
    monkeypatch.delenv("BRAIN_FALSE_BELIEF_SPIKING_WITNESS", raising=False)
    calls = []

    class _W3:
        def __init__(self, seed=42):
            pass

        def observe_event(self, loc, witnessed, lesion=False):
            calls.append((loc, witnessed))
            return loc

        def query(self, lesion=False):
            return {"belief_loc": 0, "world_loc": 1, "belief_rates": {0: 0.3, 1: 0.0}, "world_rates": {}, "n_events": 2}

    import research.runners.tom_false_belief_chat_organ as W
    monkeypatch.setattr(W, "FalseBeliefChatOrgan", _W3)
    key = "s2p-fb-off"
    F.reset_session(key)
    info = F.observe_turn(key, FB, seed=7, lesion=False)
    F.reset_session(key)
    assert calls == [(0, True), (1, False)] and info["acted"] and "spiking_witness" not in info


def test_flag_on_routes_perception_to_the_organ_and_never_passes_witnessed(monkeypatch):
    monkeypatch.setenv("BRAIN_FALSE_BELIEF_SPIKING_WITNESS", "1")
    monkeypatch.setenv("BRAIN_FALSE_BELIEF_WITNESS_OP", "4")
    log = []

    class _SW:
        def __init__(self, seed=42, lesion=False):
            log.append(("build", lesion))
            self.last_write = {}

        def perceive_arrive(self):
            log.append("A")

        def perceive_leave(self):
            log.append("L")

        def observe_event(self, loc):
            log.append(loc)
            self.last_write = {"witnessed": loc == 0}
            return loc

        def query(self):
            return {"belief_loc": 0, "world_loc": 1, "belief_rates": {0: 0.05, 1: 0.0}, "world_rates": {},
                    "n_events": 2, "presence_rate": 0.0}

    monkeypatch.setattr(M, "SpikingWitnessFalseBeliefOrgan", _SW)
    monkeypatch.setattr(M, "set_operating_point", lambda i: log.append(("op", i)))
    key = "s2p-fb-on"
    F.reset_session(key)
    info = F.observe_turn(key, FB, seed=7, lesion=False)
    F.reset_session(key)
    assert log == [("op", 4), ("build", False), "A", 0, "L", 1]
    assert info["answer"] == "Sally will look in the basket for the marble." and info["spiking_witness"]["op"] == 4


def test_row_module_shape():
    from research.runners.lbf_rows import fb_spiking_witness as R
    st = R.selftest()
    assert st["ok"], st["checks"]


def test_grid_and_operating_point_reset():
    p = M.set_operating_point(4)
    assert p["MODE"] == "readout" and M.MODE == "readout"
    M.set_operating_point(0)
    assert M.MODE == "relay" and M.W_RELAY_BELIEF == M.GRID[0]["W_RELAY_BELIEF"]
    assert all(g["BOUNDED_CONDUCTANCE"] for g in M.GRID)


def test_cli_refuses_gate_seed_without_prereg():
    r = subprocess.run([sys.executable, "-m", "research.runners.tom_spiking_witness_organ", "--seed", "42",
                        "--op", "4", "--out", os.devnull], cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ, SIM_BACKEND="numpy", SIM_NO_PROVENANCE="1"))
    assert r.returncode == 2 and "GATE seed refused" in r.stderr


def test_dev_seed_7_readout_false_belief_and_lesion():
    """Two battery cases on dev seed 7 at the readout point (the organ itself; about 10 s)."""
    M.set_operating_point(4)
    fb = M.run_case(7, ["A", "0", "L", "1"])
    les = M.run_case(7, ["A", "0", "L", "1"], lesion=True)
    assert fb["belief_loc"] == 0 and M.store_read_ok(fb["belief_rates"], 0)
    assert les["belief_loc"] == 1 and M.store_read_ok(les["belief_rates"], 1)
