"""Membrane-validity census (research/runners/membrane_census.py): the observer records an out-of-range membrane on a
tiny real bridge, leaves the run it observes untouched, and removes itself afterwards. No brain build (the live arms are
run by the census itself)."""
import hashlib
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from research.runners import membrane_census as MC  # noqa: E402
from sim import (CoreSimConfig, GPUConfig, NeuronModel, NeuronType, RuntimeState, SimulationBridge,  # noqa: E402
                 VisualizationConfig)
from sim.backend import to_host  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402

_RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL
_FS = NeuronType.IZH2007_FS_CORTICAL_INTERNEURON


def _region(name, n, exc_fraction, neuron_type):
    return BrainRegion(name=name, n_neurons=int(n), exc_fraction=float(exc_fraction), internal_density=0.0,
                       exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                       izh_neuron_type=neuron_type.name, enable_homeostasis=False)


def _cfg(seed, bounded=False):
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.enable_brain_region_framework = True
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp"):
        setattr(cfg, flag, False)
    cfg.enable_bounded_conductance_integration = bool(bounded)
    return cfg


def _finish(cfg, regions, pathways, drive_regions=(), drive_pA=0.0):
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                         gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    for name in drive_regions:
        idx = np.asarray(b.region_manager.indices(name), dtype=np.int64)
        b.cp_external_input_current[idx] = np.float32(drive_pA)
    return b


def _tiny_bridge(seed=5, bounded=False):
    """Two weakly coupled regions (4 RS 'cortex' + 3 FS 'inh'), no input: v sits near rest."""
    pw = [RegionPathway(from_region="cortex", to_region="inh", density=1.0, weight_mean=0.01, weight_jitter=0.0,
                        plastic=False)]
    return _finish(_cfg(seed, bounded), [_region("cortex", 4, 1.0, _RS), _region("inh", 3, 0.0, _FS)], pw)


def _overshoot_bridge(seed, bounded, inh_weight=400.0):
    """The 2026-09-29 defect's literal reproducer (tests/test_bounded_conductance_integration.py): four driven single-neuron
    GABA-A pools converging on a 'commit' RS pool with no excitation."""
    cfg = _cfg(seed, bounded)
    regions, pathways, drive = [_region("commit", 1, 1.0, _RS)], [], []
    for i in range(4):
        name = "inh_src_%d" % i
        drive.append(name)
        regions.append(_region(name, 1, 0.0, _FS))
        pathways.append(RegionPathway(from_region=name, to_region="commit", density=1.0, weight_mean=float(inh_weight),
                                      weight_jitter=0.0, plastic=False))
    return _finish(cfg, regions, pathways, drive, 3000.0)


def _run(bridge, steps):
    for _ in range(steps):
        bridge._run_one_simulation_step()


def _inject_after_real_step(monkeypatch, lows=(), highs=(), nans=()):
    """Replace the class step with 'the real step, then write an out-of-range v' so the census (installed after) sees
    what a runaway integrator would leave behind. Returns the stub that the census must restore."""
    real = SimulationBridge.__dict__["_run_one_simulation_step"]

    def stub(self, *a, **k):
        real(self, *a, **k)
        v = self.cp_membrane_potential_v
        for i, x in lows:
            v[i] = np.float32(x)
        for i, x in highs:
            v[i] = np.float32(x)
        for i in nans:
            v[i] = np.float32("nan")

    monkeypatch.setattr(SimulationBridge, "_run_one_simulation_step", stub)
    return stub


def test_injected_out_of_range_v_is_recorded_with_its_parameters(monkeypatch):
    b = _tiny_bridge()
    stub = _inject_after_real_step(monkeypatch, lows=[(2, -500.0)], highs=[(5, 80.0)])
    with MC.Census() as census:
        _run(b, 4)
    rep = census.report()
    assert rep["observer_errors"] == 0, rep["first_observer_error"]
    (br,) = rep["bridges"]
    assert br["n_neurons"] == 7 and br["n_steps"] == 4 and br["neuron_model"] == "IZHIKEVICH"
    assert br["n_oor_neurons"] == 2 and br["n_low_neurons"] == 1 and br["n_high_neurons"] == 1
    rows = {r["idx"]: r for r in br["oor_neurons"]}
    assert set(rows) == {2, 5}
    lo, hi = rows[2], rows[5]
    assert lo["region"] == "cortex" and hi["region"] == "inh"
    assert lo["extreme_low_mV"] == -500.0 and lo["extreme_high_mV"] is None
    assert hi["extreme_high_mV"] == 80.0 and hi["extreme_low_mV"] is None
    assert lo["first_step"] == 1 and lo["n_steps_low"] == 4 and hi["n_steps_high"] == 4
    assert "refractory" in lo and isinstance(lo["refractory_post"], bool) and lo["g_i"] is not None   # first-step context
    # the recorded parameters ARE the bridge's drawn parameters
    for key, attr in (("C", "cp_izh_C"), ("a", "cp_izh_a"), ("b", "cp_izh_b"), ("d", "cp_izh_d_increment"),
                      ("vpeak", "cp_izh_vpeak"), ("k", "cp_izh_k")):
        assert lo[key] == pytest.approx(float(np.asarray(to_host(getattr(b, attr)))[2]), abs=1e-3), key
    # per-region extremes and counts
    assert br["regions"]["cortex"]["v_min"] == -500.0 and br["regions"]["cortex"]["n_low_neurons"] == 1
    assert br["regions"]["cortex"]["steps_low"] == 4 and br["regions"]["inh"]["steps_high"] == 4
    assert br["regions"]["inh"]["v_max"] == 80.0 and br["regions"]["inh"]["n_low_neurons"] == 0
    assert br["v_min"] == -500.0 and br["v_min_region"] == "cortex"
    cmin = float(np.asarray(to_host(b.cp_izh_C))[:4].min())
    assert br["regions"]["cortex"]["c_min"] == pytest.approx(cmin, abs=1e-3)
    assert br["regions"]["cortex"]["c_min_idx"] in range(4)
    assert br["bounded_conductance_integration"] is False
    # the wrapper is gone and the thing it wrapped is back
    assert SimulationBridge.__dict__["_run_one_simulation_step"] is stub
    assert not getattr(SimulationBridge.__dict__["_run_one_simulation_step"], "_membrane_census_wrapper", False)
    assert not getattr(SimulationBridge.__dict__["__init__"], "_membrane_census_wrapper", False)


def test_wrapper_is_removed_after_the_run_even_when_the_workload_raises():
    orig_step = SimulationBridge.__dict__["_run_one_simulation_step"]
    orig_init = SimulationBridge.__dict__["__init__"]
    c = MC.Census()
    with pytest.raises(RuntimeError, match="boom"):
        with c:
            assert getattr(SimulationBridge.__dict__["_run_one_simulation_step"], "_membrane_census_wrapper", False)
            raise RuntimeError("boom")
    assert SimulationBridge.__dict__["_run_one_simulation_step"] is orig_step
    assert SimulationBridge.__dict__["__init__"] is orig_init
    c.uninstall()                                   # idempotent
    assert SimulationBridge.__dict__["_run_one_simulation_step"] is orig_step
    with MC.Census():
        with pytest.raises(RuntimeError, match="already installed"):
            MC.Census().install()
    assert SimulationBridge.__dict__["_run_one_simulation_step"] is orig_step


def test_a_clean_run_records_no_out_of_range_neuron_and_reports_minima_and_bounded_flag():
    with MC.Census() as census:
        b = _tiny_bridge(seed=3, bounded=True)
        _run(b, 6)
    (br,) = census.report()["bridges"]
    assert br["n_oor_neurons"] == 0 and br["oor_neurons"] == [] and br["n_steps_with_oor"] == 0
    assert br["bounded_conductance_integration"] is True and br["bounded_values_seen"] == [True]
    assert -110.0 < br["v_min"] <= br["v_max"] < 35.0
    assert br["regions"]["cortex"]["c_min"] > 0 and br["regions"]["inh"]["c_min"] > 0
    assert br["organ"] and "test_membrane_census" in br["organ"]      # the builder is identified from the call chain
    (row,) = census.report()["organs"]
    assert row["n_neurons"] == 7 and row["n_oor_neurons"] == 0


def test_nan_membrane_is_recorded_as_nonfinite(monkeypatch):
    b = _tiny_bridge()
    _inject_after_real_step(monkeypatch, nans=[1])
    with MC.Census() as census:
        _run(b, 3)
    (br,) = census.report()["bridges"]
    assert br["n_nonfinite_neurons"] == 1 and br["n_oor_neurons"] == 1
    (row,) = br["oor_neurons"]
    assert row["idx"] == 1 and row["n_steps_nonfinite"] == 3 and row["extreme_low_mV"] is None


def test_the_real_euler_overshoot_is_detected_and_bounded_conductance_clears_it():
    """The defect this census exists for, on the literal reproducer: unbounded explicit Euler drives the commit pool far
    below -110 mV; the same network with bounded conductance integration (read back from the bridge's config) stays in range."""
    with MC.Census() as c_off:
        _run(_overshoot_bridge(42, bounded=False), 60)
    with MC.Census() as c_on:
        _run(_overshoot_bridge(42, bounded=True), 60)
    off = [b for b in c_off.report()["bridges"]][0]
    on = [b for b in c_on.report()["bridges"]][0]
    assert off["bounded_conductance_integration"] is False and on["bounded_conductance_integration"] is True
    assert off["n_low_neurons"] >= 1 and off["v_min"] < MC.V_LOW_MV, off["v_min"]
    assert off["regions"]["commit"]["n_low_neurons"] == 1
    assert off["regions"]["commit"]["max_gi_dt_over_C"] > 2.0          # the explicit-step hazard index tripped
    assert off["regions"]["commit"]["max_gtot_dt_over_C"] >= off["regions"]["commit"]["max_gi_dt_over_C"]
    (row,) = off["oor_neurons"]
    assert row["region"] == "commit" and row["inhibitory"] is False and row["g_i"] is not None and row["g_i"] > 0
    assert on["n_oor_neurons"] == 0 and on["v_min"] >= MC.V_LOW_MV, on["v_min"]


def test_observer_does_not_change_the_run():
    """Observation is read-only: the final membranes and the raster of the same seeded build are byte-identical with and
    without the census installed."""
    def digest(census):
        if census:
            with MC.Census():
                b = _overshoot_bridge(7, bounded=False)
                _run(b, 40)
        else:
            b = _overshoot_bridge(7, bounded=False)
            _run(b, 40)
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(np.asarray(to_host(b.cp_membrane_potential_v))).tobytes())
        h.update(np.ascontiguousarray(np.asarray(to_host(b.cp_recovery_variable_u))).tobytes())
        return h.hexdigest()
    assert digest(False) == digest(True)


def test_two_bridges_are_recorded_separately():
    with MC.Census() as census:
        a = _tiny_bridge(seed=1)
        b = _overshoot_bridge(42, bounded=False)
        _run(a, 3)
        _run(b, 60)
    reps = census.report()["bridges"]
    assert len(reps) == 2 and reps[0]["n_neurons"] == 7 and reps[1]["n_neurons"] == 5
    assert reps[0]["n_oor_neurons"] == 0 and reps[1]["n_low_neurons"] >= 1


def test_pool_lines_have_the_stage2_shape_for_dev_seeds_1_to_12_and_five_groups():
    from research.runners import stage2_exit_gate as SG
    lines = MC.pool_lines()
    assert len(lines) == 12 * 5 == 60 and len(set(lines)) == 60
    for ln in lines:
        assert "\t" not in ln and ln.startswith("cd ~/derisk-pool/revisions/<sha> && bash tools/mem_ok.sh 9 1 && mkdir -p ")
        assert "bash tools/memcap.sh 10 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1" in ln
        assert ".venv/bin/python -u -m research.runners.membrane_census --shard --seed " in ln
        assert "--allow-gate-seed" not in ln
    seeds = {int(ln.split("--seed ")[1].split()[0]) for ln in lines}
    groups = {ln.split("--group ")[1].split()[0] for ln in lines}
    assert seeds == set(range(1, 13)) and not (seeds & set(SG.GATE_SEEDS))
    assert groups == set(SG.PRIMARY_GROUPS) == {"conv", "infer", "ask", "emo", "inner"}
    assert MC.pool_lines(sha="abc123")[0].startswith("cd ~/derisk-pool/revisions/abc123 && ")


def test_gate_seed_is_refused_without_the_flag():
    import subprocess
    import sys
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    p = subprocess.run([sys.executable, "-m", "research.runners.membrane_census", "--shard", "--seed", "42", "--group",
                        "conv", "--arm", "intact_a", "--out", os.path.join(os.environ.get("TMPDIR", "/tmp"), "x.json")],
                       cwd=root, capture_output=True, text=True, env=dict(os.environ, SIM_NO_PROVENANCE="1"), timeout=120)
    assert p.returncode == 2 and "GATE seed 42 refused" in p.stderr


def test_a_high_membrane_inside_the_refractory_window_is_marked_refractory():
    """A cell driven hard keeps integrating through its refractory window (the crossing is withheld, nothing resets it):
    the census records the high membrane and that the cell was refractory when the step decided."""
    b = _tiny_bridge(seed=11)
    b.cp_external_input_current[0] = np.float32(20000.0)
    with MC.Census() as census:
        _run(b, 30)
    (br,) = census.report()["bridges"]
    rows = {r["idx"]: r for r in br["oor_neurons"]}
    assert 0 in rows and rows[0]["n_steps_high"] >= 1 and rows[0]["extreme_high_mV"] > 35.0 + 5.0
    assert rows[0]["refractory"] is True, rows[0]
    assert br["regions"]["cortex"]["n_high_neurons"] >= 1


def test_relabel_names_the_stepper_and_keeps_the_builder():
    bs = [{"owner_chain": ["pkg.builder:Build.make:12", "pkg.top:run:3"],
           "step_callers": [{"site": "pkg.organ:Organ.step_once:88", "steps": 90}, {"site": "pkg.other:f:1", "steps": 10}]},
          {"owner_chain": ["pkg.builder:Build.make:12"], "step_callers": [{"site": "pkg.fn:driver:5", "steps": 3}]},
          {"owner_chain": [], "step_callers": []}]
    MC.relabel(bs)
    assert bs[0]["organ"] == "pkg.organ:Organ" and bs[0]["built_by"] == "pkg.builder:Build.make"
    assert bs[1]["organ"] == "pkg.fn:driver"
    assert bs[2]["organ"] == "?" and bs[2]["built_by"] is None
    assert MC.relabel(bs) is bs and bs[0]["organ"] == "pkg.organ:Organ"           # idempotent


def test_parameters_are_captured_at_detection_so_a_transient_bridge_still_reports_them(monkeypatch):
    import gc
    b = _tiny_bridge(seed=5)
    _inject_after_real_step(monkeypatch, lows=[(1, -300.0)])
    with MC.Census() as census:
        _run(b, 3)
        want_C = float(np.asarray(to_host(b.cp_izh_C))[1])
        del b
        gc.collect()
    (br,) = census.report()["bridges"]
    assert br["alive_at_end"] is False
    (row,) = br["oor_neurons"]
    assert row["idx"] == 1 and row["C"] == pytest.approx(want_C, abs=1e-3) and row["a"] is not None and row["type"]


def test_spikes_from_cells_after_they_left_the_range_are_counted():
    with MC.Census() as census:
        _run(_overshoot_bridge(42, bounded=False), 60)
    (br,) = census.report()["bridges"]
    assert br["n_spikes_total"] > 0
    assert 0 < br["n_spikes_from_oor_cells"] <= br["n_spikes_total"], (br["n_spikes_from_oor_cells"], br["n_spikes_total"])
    (row,) = br["oor_neurons"]
    assert row["n_spikes_after_first"] == br["n_spikes_from_oor_cells"] > 0
