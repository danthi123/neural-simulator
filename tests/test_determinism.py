"""
Determinism and Reproducibility Tests

Tests that simulations with the same seed produce identical results.
This ensures the simulator is deterministic for scientific reproducibility.

Run with: pytest tests/test_determinism.py -v
"""

import sys
import os
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cupy as cp
import pytest

# Import from neural-simulator.py (using importlib to handle hyphen)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "neural_simulator",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "neural-simulator.py")
)
neural_simulator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neural_simulator)

SimulationBridge = neural_simulator.SimulationBridge
CoreSimConfig = neural_simulator.CoreSimConfig
VisualizationConfig = neural_simulator.VisualizationConfig
RuntimeState = neural_simulator.RuntimeState
GPUConfig = neural_simulator.GPUConfig
NeuronModel = neural_simulator.NeuronModel


class TestDeterministicSpikes:
    """Test that spike trains are deterministic given same seed."""
    
    def test_izhikevich_deterministic_spikes(self):
        """Two Izhikevich runs with same seed produce identical spike trains."""
        config = CoreSimConfig(
            num_neurons=100,
            connections_per_neuron=50,
            seed=42,
            neuron_model_type=NeuronModel.IZHIKEVICH.name,
            dt_ms=1.0,
            enable_hebbian_learning=False,  # Disable for strict determinism
            enable_short_term_plasticity=False,
            enable_homeostasis=False
        )
        
        gpu_config = GPUConfig(enable_profiling=False)
        
        # Run 1
        sim1 = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim1._initialize_simulation_data()
        
        spikes1 = []
        for _ in range(100):
            sim1._run_one_simulation_step()
            spikes1.append(cp.asnumpy(sim1.cp_firing_states).copy())
        
        sim1.clear_simulation_state_and_gpu_memory()
        
        # Run 2
        sim2 = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim2._initialize_simulation_data()
        
        spikes2 = []
        for _ in range(100):
            sim2._run_one_simulation_step()
            spikes2.append(cp.asnumpy(sim2.cp_firing_states).copy())
        
        sim2.clear_simulation_state_and_gpu_memory()
        
        # Compare
        for i, (s1, s2) in enumerate(zip(spikes1, spikes2)):
            assert np.array_equal(s1, s2), f"Step {i}: Izhikevich spike trains differ"
        
        print(f"✓ Izhikevich deterministic: {len(spikes1)} steps matched")
    
    def test_hodgkin_huxley_deterministic_spikes(self):
        """Two HH runs with same seed produce identical spike trains."""
        config = CoreSimConfig(
            num_neurons=50,  # HH is slower, use fewer neurons
            connections_per_neuron=25,
            seed=123,
            neuron_model_type=NeuronModel.HODGKIN_HUXLEY.name,
            dt_ms=0.025,  # HH needs smaller dt
            enable_hebbian_learning=False,
            enable_short_term_plasticity=False,
            enable_homeostasis=False
        )
        
        gpu_config = GPUConfig(enable_profiling=False)
        
        # Run 1
        sim1 = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim1._initialize_simulation_data()
        
        spikes1 = []
        for _ in range(50):  # Fewer steps due to smaller dt
            sim1._run_one_simulation_step()
            spikes1.append(cp.asnumpy(sim1.cp_firing_states).copy())
        
        sim1.clear_simulation_state_and_gpu_memory()
        
        # Run 2
        sim2 = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim2._initialize_simulation_data()
        
        spikes2 = []
        for _ in range(50):
            sim2._run_one_simulation_step()
            spikes2.append(cp.asnumpy(sim2.cp_firing_states).copy())
        
        sim2.clear_simulation_state_and_gpu_memory()
        
        # Compare
        for i, (s1, s2) in enumerate(zip(spikes1, spikes2)):
            assert np.array_equal(s1, s2), f"Step {i}: HH spike trains differ"
        
        print(f"✓ Hodgkin-Huxley deterministic: {len(spikes1)} steps matched")
    
    def test_adex_deterministic_spikes(self):
        """Two AdEx runs with same seed produce identical spike trains."""
        config = CoreSimConfig(
            num_neurons=100,
            connections_per_neuron=50,
            seed=456,
            neuron_model_type=NeuronModel.ADEX.name,
            dt_ms=0.1,  # AdEx benefits from smaller dt
            enable_hebbian_learning=False,
            enable_short_term_plasticity=False,
            enable_homeostasis=False
        )
        
        gpu_config = GPUConfig(enable_profiling=False)
        
        # Run 1
        sim1 = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim1._initialize_simulation_data()
        
        spikes1 = []
        for _ in range(100):
            sim1._run_one_simulation_step()
            spikes1.append(cp.asnumpy(sim1.cp_firing_states).copy())
        
        sim1.clear_simulation_state_and_gpu_memory()
        
        # Run 2
        sim2 = SimulationBridge(
            core_config=config,
            viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(),
            gpu_config=gpu_config
        )
        sim2._initialize_simulation_data()
        
        spikes2 = []
        for _ in range(100):
            sim2._run_one_simulation_step()
            spikes2.append(cp.asnumpy(sim2.cp_firing_states).copy())
        
        sim2.clear_simulation_state_and_gpu_memory()
        
        # Compare
        for i, (s1, s2) in enumerate(zip(spikes1, spikes2)):
            assert np.array_equal(s1, s2), f"Step {i}: AdEx spike trains differ"
        
        print(f"✓ AdEx deterministic: {len(spikes1)} steps matched")


class TestDeterministicMembranePotential:
    """Test that membrane potential traces are deterministic."""
    
    def test_izhikevich_membrane_potential(self):
        """Two runs produce identical membrane potential traces."""
        config = CoreSimConfig(
            num_neurons=50,
            connections_per_neuron=25,
            seed=789,
            neuron_model_type=NeuronModel.IZHIKEVICH.name,
            enable_hebbian_learning=False,
            enable_short_term_plasticity=False
        )
        
        gpu_config = GPUConfig(enable_profiling=False)
        
        # Run 1
        sim1 = SimulationBridge(core_config=config, gpu_config=gpu_config)
        sim1._initialize_simulation_data()
        
        v_traces1 = []
        for _ in range(50):
            sim1._run_one_simulation_step()
            v_traces1.append(cp.asnumpy(sim1.cp_membrane_potential_v).copy())
        
        sim1.clear_simulation_state_and_gpu_memory()
        
        # Run 2
        sim2 = SimulationBridge(core_config=config, gpu_config=gpu_config)
        sim2._initialize_simulation_data()
        
        v_traces2 = []
        for _ in range(50):
            sim2._run_one_simulation_step()
            v_traces2.append(cp.asnumpy(sim2.cp_membrane_potential_v).copy())
        
        sim2.clear_simulation_state_and_gpu_memory()
        
        # Compare with tolerance for floating-point arithmetic
        for i, (v1, v2) in enumerate(zip(v_traces1, v_traces2)):
            assert np.allclose(v1, v2, rtol=1e-6, atol=1e-6), \
                f"Step {i}: Membrane potential traces differ"
        
        print(f"✓ Membrane potential deterministic: {len(v_traces1)} steps matched")


class TestDeterministicConnectivity:
    """Test that connectivity generation is deterministic."""
    
    def test_connectivity_generation(self):
        """Same seed produces identical connectivity matrices."""
        config = CoreSimConfig(
            num_neurons=200,
            connections_per_neuron=50,
            seed=999,
            neuron_model_type=NeuronModel.IZHIKEVICH.name
        )
        
        gpu_config = GPUConfig(enable_profiling=False)
        
        # Generate 1
        sim1 = SimulationBridge(core_config=config, gpu_config=gpu_config)
        sim1._initialize_simulation_data()
        
        conn1_data = cp.asnumpy(sim1.cp_connections.data).copy()
        conn1_indices = cp.asnumpy(sim1.cp_connections.indices).copy()
        conn1_indptr = cp.asnumpy(sim1.cp_connections.indptr).copy()
        
        sim1.clear_simulation_state_and_gpu_memory()
        
        # Generate 2
        sim2 = SimulationBridge(core_config=config, gpu_config=gpu_config)
        sim2._initialize_simulation_data()
        
        conn2_data = cp.asnumpy(sim2.cp_connections.data).copy()
        conn2_indices = cp.asnumpy(sim2.cp_connections.indices).copy()
        conn2_indptr = cp.asnumpy(sim2.cp_connections.indptr).copy()
        
        sim2.clear_simulation_state_and_gpu_memory()
        
        # Compare connectivity
        assert np.allclose(conn1_data, conn2_data, rtol=1e-6), "Connection weights differ"
        assert np.array_equal(conn1_indices, conn2_indices), "Connection indices differ"
        assert np.array_equal(conn1_indptr, conn2_indptr), "Connection indptr differ"
        
        print(f"✓ Connectivity generation deterministic: {len(conn1_data)} synapses matched")


class TestSeedTracking:
    """Test that actual seed used is tracked correctly."""
    
    def test_explicit_seed_tracked(self):
        """Explicit seed is stored in runtime_state."""
        config = CoreSimConfig(seed=12345)
        sim = SimulationBridge(core_config=config)
        sim._initialize_simulation_data()
        
        assert sim.runtime_state.actual_seed_used == 12345, \
            "Explicit seed not tracked correctly"
        
        sim.clear_simulation_state_and_gpu_memory()
        print(f"✓ Explicit seed tracked: {sim.runtime_state.actual_seed_used}")
    
    def test_random_seed_generated(self):
        """Random seed (-1) generates and stores a seed."""
        config = CoreSimConfig(seed=-1)
        sim = SimulationBridge(core_config=config)
        sim._initialize_simulation_data()
        
        assert sim.runtime_state.actual_seed_used != -1, \
            "Random seed was not generated"
        assert sim.runtime_state.actual_seed_used >= 0, \
            "Generated seed is negative"
        
        sim.clear_simulation_state_and_gpu_memory()
        print(f"✓ Random seed generated and tracked: {sim.runtime_state.actual_seed_used}")


if __name__ == "__main__":
    # Can run directly without pytest
    print("Running determinism tests...")
    
    test_spikes = TestDeterministicSpikes()
    test_spikes.test_izhikevich_deterministic_spikes()
    test_spikes.test_hodgkin_huxley_deterministic_spikes()
    test_spikes.test_adex_deterministic_spikes()
    
    test_v = TestDeterministicMembranePotential()
    test_v.test_izhikevich_membrane_potential()
    
    test_conn = TestDeterministicConnectivity()
    test_conn.test_connectivity_generation()
    

class TestSubstrateActuallySeeded:
    """THE TEST THAT WOULD HAVE CAUGHT THE 2026-07-17 BUG -- and the gap it closes.

    Every other test in this file seeds the CONSTRUCTOR (`CoreSimConfig(..., seed=42, ...)`) and therefore passes.
    `TestSeedTracking` asserts `runtime_state.actual_seed_used == 12345` -- i.e. that the REPORTING FIELD IS SET.
    Neither shape can catch a caller that builds `CoreSimConfig()` BARE and then sets `actual_seed_used`, which is
    exactly what 8 research runners did:

        cfg = CoreSimConfig()            # cfg.seed stays -1
        cfg.actual_seed_used = int(seed) # a REPORTING field -- the bridge never reads it

    The bridge seeds heterogeneity from `cfg.seed` (bridge.py:2136):
        het_seed = cfg.heterogeneity_seed if cfg.heterogeneity_seed >= 0 else cfg.seed
        if het_seed >= 0: cp.random.seed(het_seed)
    Both default to -1 => the guard never fires => the per-neuron firing thresholds (bridge.py:1508,
    `cp.random.uniform`) come from the UNSEEDED GLOBAL RNG. MEASURED CONSEQUENCE: two builds at the same seed got
    DIFFERENT NEURONS (18.4 mV apart), which silently confounded an entire arc's same-seed comparisons -- the
    confound was ~3x the effect being measured.

    THE GENERAL LESSON THIS ENCODES: a test that asserts a FIELD IS SET is not a test that the field DOES ANYTHING.
    Assert the PROPERTY (same seed => same substrate), not the bookkeeping.
    """

    def test_same_seed_gives_identical_neurons_within_a_process(self):
        """Two bridges built back-to-back with the same seed must have identical neurons. This is the one that fails
        loudest under the bug: each build advances the global RNG, so the 2nd differs from the 1st."""
        import numpy as np
        from sim.config import CoreSimConfig, GPUConfig, VisualizationConfig, RuntimeState
        from sim.bridge import SimulationBridge

        def build():
            cfg = CoreSimConfig()
            cfg.num_neurons = 64
            cfg.dt_ms = 1.0
            cfg.seed = 4242                 # <- the field the bridge ACTUALLY reads. Omit it and this test fails.
            cfg.actual_seed_used = 4242     # reporting only; asserted here to prove it is NOT what makes it pass
            br = SimulationBridge(core_config=cfg, gpu_config=GPUConfig(),
                                  viz_config=VisualizationConfig(), runtime_state=RuntimeState())
            br._initialize_simulation_data()
            t = br.cp_neuron_firing_thresholds
            return np.asarray(t.get() if hasattr(t, "get") else t)

        a, b = build(), build()
        assert a.shape == b.shape and a.size > 0, "no thresholds allocated -- test cannot discriminate"
        np.testing.assert_array_equal(
            a, b,
            err_msg=("same seed produced DIFFERENT neurons. cfg.seed is not reaching the bridge's heterogeneity "
                     "seeding (bridge.py:2136) -- every same-seed comparison built on this is confounded."),
        )

    def test_the_reporting_field_alone_does_NOT_seed(self):
        """Pins the trap itself: setting ONLY `actual_seed_used` must NOT be mistaken for seeding. If this ever
        starts producing identical neurons, the engine changed (actual_seed_used became load-bearing) and the
        guidance in CLAUDE.md + the runner comments must be revisited."""
        import numpy as np
        from sim.config import CoreSimConfig, GPUConfig, VisualizationConfig, RuntimeState
        from sim.bridge import SimulationBridge

        def build_unseeded():
            cfg = CoreSimConfig()
            cfg.num_neurons = 64
            cfg.dt_ms = 1.0
            cfg.actual_seed_used = 4242     # ONLY the reporting field -- cfg.seed left at its -1 default
            br = SimulationBridge(core_config=cfg, gpu_config=GPUConfig(),
                                  viz_config=VisualizationConfig(), runtime_state=RuntimeState())
            br._initialize_simulation_data()
            t = br.cp_neuron_firing_thresholds
            return np.asarray(t.get() if hasattr(t, "get") else t)

        a, b = build_unseeded(), build_unseeded()
        assert not np.array_equal(a, b), (
            "setting ONLY actual_seed_used now yields identical neurons. Either the engine changed (the field became "
            "load-bearing) or this test no longer discriminates. Both require revisiting the 2026-07-17 finding."
        )


class TestGap5StoreByteReproducible:
    """Pins the 2026-08-25 gap#5 store reproducibility fix — and the ROOT CAUSE the fix corrects.

    THE FINDING'S FIRST GUESS WAS WRONG, so record what was actually measured. The connectivity +
    threshold DRAW was ALWAYS correctly cfg.seed-seeded: `_prepare_sequence(seed, ..., do_encode=False)`
    is byte-identical across fresh processes. What made the gap#5 store non-reproducible at a FIXED
    seed (two same-seed builds differing in cp_neuron_firing_thresholds AND cp_connections, readout
    forward_frac flipping 1.0<->0.0) was NOT an unseeded RNG. It was the per-step synaptic-current
    SpMV: cupyx/cuSPARSE Wᵀ@spikes is BIT-NON-reproducible run-to-run (atomic FP accumulation — the
    IDENTICAL SpMV returns distinct results call-to-call), and the chaotic spiking + BTSP plasticity
    amplify that per-step jitter into an entirely different store. The fix routes every per-step
    transpose SpMV through an explicit `add.reduceat` segmented reduction
    (sim.bridge._deterministic_csr_matvec, no atomics) under `deterministic_transpose_matvec`, which
    the gap#5 store builder (research/runners/_riii_ca3_coincidence_completion_derisk._build) now sets.

    THE GENERAL LESSON: "seeded" is necessary but NOT sufficient for reproducibility on a GPU. A
    correctly-seeded substrate whose dynamics run through a non-deterministic library SpMV is still
    non-reproducible. Assert the PROPERTY (same seed => byte-identical store), on the GPU backend
    where the non-determinism actually lives.
    """

    def test_deterministic_csr_matvec_is_reproducible_and_correct(self):
        """The reduceat SpMV that replaces the cuSPARSE one must be (a) byte-identical across repeated
        identical calls at the REAL gap#5 connectivity scale (the property the chaotic substrate needs;
        a regression to `csr @ v` would VARY here — that is exactly the measured bug), and (b) a correct
        float32 SpMV vs an exact float64 reference."""
        import numpy as np
        from sim.backend import get_backend, is_gpu_backend, to_host
        from sim.bridge import _deterministic_csr_matvec
        if not is_gpu_backend():
            import pytest as _pytest
            _pytest.skip("cuSPARSE SpMV non-determinism is a GPU/cupy concern; numpy SpMV is already deterministic")
        cp, _ = get_backend()
        from research.runners._gap5_sequence_replay_derisk import _prepare_sequence
        from research.runners._gap5_decoupled_store_bistable_readout_derisk import DECOUPLED_CFG
        # Real gap#5 connectivity at full CA3 scale (do_encode=False => fast, no BTSP loop) so a
        # regression of the helper back to cuSPARSE `@` would be caught (it is non-deterministic here).
        cfg = {**DECOUPLED_CFG, "n_ca3": 2000, "n_mem": 3, "freeze_between_refresh": True}
        W = _prepare_sequence(42, cfg, do_encode=False)["bridge"].cp_connections
        WT = W.T.tocsr()
        cp.random.seed(0)
        v = (cp.random.rand(W.shape[0]) > 0.9).astype(cp.float32)

        def _hash(a):
            return hashlib.sha1(np.ascontiguousarray(np.asarray(to_host(a))).tobytes()).hexdigest()

        hs = {_hash(_deterministic_csr_matvec(WT, v)) for _ in range(6)}
        assert len(hs) == 1, (
            "the deterministic SpMV is NOT reproducible — 6 identical calls gave "
            f"{len(hs)} distinct results. It has regressed to a non-deterministic (atomic) SpMV; the "
            "gap#5 store will again be non-reproducible at a fixed seed."
        )
        # Correctness vs an EXACT float64 reference (any correct float32 SpMV lands within f32 rounding).
        WT_h = WT.get() if hasattr(WT, "get") else WT
        ref = (WT_h.astype(np.float64) @ np.asarray(to_host(v)).astype(np.float64))
        mine = np.asarray(to_host(_deterministic_csr_matvec(WT, v))).astype(np.float64)
        rel = np.max(np.abs(mine - ref)) / (np.max(np.abs(ref)) + 1e-9)
        assert rel < 1e-3, f"deterministic SpMV is not a correct float32 matvec (rel err {rel:.2e})"

    @pytest.mark.slow
    def test_prepare_sequence_gives_byte_identical_store_at_fixed_seed(self):
        """END-TO-END property (what the finding's probe checks): two same-seed gap#5 store builds IN
        ONE PROCESS — through the full BTSP encode that drives the chaotic spiking — must be byte-
        identical in BOTH thresholds AND connectivity, and a DIFFERENT seed must give a DIFFERENT store
        (guards against over-seeding to a constant). Fails without the fix: at this scale the cuSPARSE
        SpMV jitter compounds into distinct stores (measured e45500ac vs 96d987f5 on one seed). Slow
        (~90s: two full BTSP encodes) but it is the load-bearing reproducibility property."""
        import numpy as np
        from sim.backend import is_gpu_backend, to_host
        if not is_gpu_backend():
            pytest.skip("reproducibility of the GPU store is a cupy concern; the numpy SpMV is already deterministic")
        from research.runners._gap5_sequence_replay_derisk import _prepare_sequence
        from research.runners._gap5_decoupled_store_bistable_readout_derisk import DECOUPLED_CFG
        cfg = {**DECOUPLED_CFG, "n_ca3": 600, "n_mem": 3, "freeze_between_refresh": True}

        def _store(seed):
            br = _prepare_sequence(seed, cfg, do_encode=True)["bridge"]
            def _h(a):
                return hashlib.sha1(np.ascontiguousarray(np.asarray(to_host(a))).tobytes()).hexdigest()
            return _h(br.cp_neuron_firing_thresholds), _h(br.cp_connections.data)

        a = _store(42)
        b = _store(42)
        assert a == b, (
            "same-seed gap#5 store build is NOT byte-reproducible (thresh/conn "
            f"{a} != {b}). The deterministic-SpMV path has regressed; every quantitative gap#5 readout "
            "metric built on this is confounded (only within-run control contrasts survive)."
        )
        c = _store(43)
        assert c[1] != a[1], (
            "a DIFFERENT seed gives the SAME connectivity — the store is over-seeded to a constant, "
            "not merely made reproducible."
        )


if __name__ == "__main__":
    test_seed = TestSeedTracking()
    test_seed.test_explicit_seed_tracked()
    test_seed.test_random_seed_generated()

    print("\n✅ All determinism tests passed!")
