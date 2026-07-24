# Contributing to GPU-Accelerated Neural Network Simulator

Thank you for considering contributing to this project! This document provides guidelines and instructions for contributors.

This is a GPU-accelerated simulator for biologically realistic spiking neural
networks, with a real-time 3-D view of the neurons firing. It is an active
research codebase — some capabilities are shipped and robust, others are
research-stage; the docs try to be clear about which is which.

## Getting oriented

Before diving in, skim these:

- [`README.md`](README.md) — what the project is, in plain language.
- [`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`](docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md)
  — **the current primary development plan.** As of the 2026-07-23 direction
  pivot the north-star is a sim-brain that **converses genuinely** — reasons to
  its own conclusions and has an affective world-model, emotion, self-awareness,
  and curiosity — not fact-recall/RAG and not large-language-model plausible
  text. The master roadmap lays out the complete faculty map, the one-brain
  architecture, the six developmental stages, and a 14-wall ledger (each wall
  paired with a biological surpass). The earlier "5-gap closure" framing is
  subsumed as a sub-view (the faculty-map + walls-ledger) and remains valid.
  Its foundation is
  [`docs/plans/2026-07-22-genuine-conversation-affective-self-aware-brain-plan.md`](docs/plans/2026-07-22-genuine-conversation-affective-self-aware-brain-plan.md).
- [`ROADMAP.md`](ROADMAP.md) — the plain-language **status** source of truth
  (what's done / in progress / open) that pairs with the master plan above.
- [`docs/INDEX.md`](docs/INDEX.md) — the full documentation map.
- [`docs/CURRENT-STATE.md`](docs/CURRENT-STATE.md) — what works today, with technical detail.
- [`docs/diagrams/brain_architecture_current.md`](docs/diagrams/brain_architecture_current.md)
  and [`brain_architecture_detailed.md`](docs/diagrams/brain_architecture_detailed.md) —
  live Mermaid flowcharts of the brain architecture (render on GitHub).
- [`CLAUDE.md`](CLAUDE.md) — deep technical gotchas (STDP soft bounds, Hebbian
  decay, NMDA configuration, seeding) worth knowing before you touch the engine.

## Development Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.x or 12.x (optional — the engine also runs on
  CPU via NumPy; set `SIM_BACKEND=numpy`)
- CUDA Toolkit installed (only needed for the GPU path)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/danthi123/neural-simulator.git
cd neural-simulator
```

2. Install dependencies:
```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x

# Other dependencies
pip install numpy h5py dearpygui PyOpenGL PyOpenGL-accelerate

# Development dependencies
pip install pytest pytest-cov
```

3. Verify installation:
```bash
python -c "import cupy; print(f'CuPy version: {cupy.__version__}')"
python -c "import cupy; print(f'GPU devices: {cupy.cuda.runtime.getDeviceCount()}')"
```

## Development Workflow

### Branching Strategy

- `main`: Stable, tested code
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `perf/*`: Performance improvements

### Making Changes

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes, following the coding style below

3. Run tests:
```bash
pytest tests/ -v
```

4. Run benchmarks (if performance-related):
```bash
python benchmark.py --quick
```

5. Commit with descriptive messages:
```bash
git commit -m "Add feature X: brief description

- Detailed point 1
- Detailed point 2"
```

6. Push and create pull request

## Coding Standards

### Python Style

- Follow PEP 8 style guide
- Use descriptive variable names
- Maximum line length: 120 characters
- Use type hints where helpful

### GPU Code

- Minimize CPU↔GPU transfers
- Use fused kernels for multiple operations
- Profile GPU memory usage for new features
- Document any CuPy kernel magic

### Documentation

- Add docstrings to all public methods
- Use NumPy-style docstrings
- Update README.md for user-facing features
- Comment complex algorithms

### Example:
```python
def update_network_activity(self, bridge: SimulationBridge) -> dict:
    """Monitor and update network activity based on firing states.

    Args:
        bridge: The SimulationBridge instance managing simulation

    Returns:
        Dictionary with activity statistics

    Notes:
        This method accesses GPU arrays directly via SimulationBridge
        to compute network metrics without CPU-GPU transfers.
    """
    # Access GPU firing states directly
    firing_states = bridge.cp_firing_states  # Boolean array (num_neurons,)
    spike_counts = cp.count_nonzero(firing_states)

    # Run a simulation step and retrieve results
    bridge._run_one_simulation_step()
    membrane_potentials = bridge.cp_membrane_potential_v

    return {
        'spike_count': int(spike_counts),
        'mean_voltage': float(cp.mean(membrane_potentials)),
        'firing_rate': float(spike_counts / bridge.core_config.num_neurons)
    }
```

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_determinism.py -v

# Specific test
pytest tests/test_determinism.py::TestDeterministicSpikes::test_izhikevich_deterministic_spikes -v

# With coverage
pytest tests/ --cov=sim --cov=experiment --cov-report=html
```

### Writing Tests

When adding new features:

1. **Determinism tests** - If feature involves randomness:
```python
def test_new_feature_deterministic(self):
    """Same seed produces same results."""
    config = CoreSimConfig(seed=42, ...)
    sim1 = SimulationBridge(core_config=config)
    # Run and capture results
    
    sim2 = SimulationBridge(core_config=config)
    # Run and compare
    assert results1 == results2
```

2. **Correctness tests** - Validate expected behavior:
```python
def test_new_feature_correctness(self):
    """Feature produces expected output."""
    sim = SimulationBridge(...)
    result = sim.new_feature()
    assert result meets_expected_criteria
```

3. **Performance tests** - For optimization work:
```python
def test_new_feature_performance(self):
    """Feature meets performance target."""
    import time
    start = time.time()
    # Run feature
    elapsed = time.time() - start
    assert elapsed < TARGET_TIME
```

### Test Guidelines

- Keep tests fast (<30s each)
- Use small network sizes for unit tests
- Clean up GPU memory after tests
- Make tests reproducible (fixed seeds)

## Performance Contributions

### Before Making Changes

1. Run baseline benchmarks:
```bash
python benchmark.py --output benchmarks/baseline_before.json
```

2. Profile if needed:
```python
gpu_config = GPUConfig(enable_profiling=True, profiling_detailed=True)
sim = SimulationBridge(gpu_config=gpu_config)
# ... run simulation ...
sim.export_profiling_report("profile_before.json")
```

### After Making Changes

1. Run benchmarks again:
```bash
python benchmark.py --output benchmarks/after_optimization.json
```

2. Compare:
```bash
python benchmark.py --compare benchmarks/baseline_before.json
```

3. Document performance improvements in PR:
   - What was optimized
   - Performance gains (%, absolute time)
   - Any trade-offs made
   - Benchmark results

### Performance Guidelines

- Profile before optimizing
- Measure actual improvements
- Don't sacrifice correctness for speed
- Document any precision trade-offs
- Test on multiple GPU architectures if possible

## Pull Request Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Benchmarks run (for performance changes)
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Code follows style guidelines
- [ ] No debugging print statements left

### PR Description Template

```markdown
## Description
Brief summary of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Breaking change

## Testing
Describe testing performed

## Performance Impact
For performance-related changes:
- Benchmark results
- Memory usage changes
- Any trade-offs

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Follows coding standards
```

### Review Process

1. Automated checks run (tests, linting)
2. Code review by maintainer
3. Address feedback
4. Merge after approval

## Areas for Contribution

### High Priority

- Additional neuron models (LIF, multi-compartment)
- Network analysis tools
- Export formats (SONATA, NeuroML)
- Performance optimizations
- Test coverage improvements

### Medium Priority

- UI improvements
- Additional plasticity rules
- Documentation enhancements
- Example notebooks
- Tutorial content

### Advanced

- Multi-GPU support
- AMD ROCm/HIP port
- Mixed precision training
- Differentiable simulation modes

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Features**: Open a GitHub Issue describing the use case
- **Code**: Tag maintainer in PR comments

## Code of Conduct

- Be respectful and constructive
- Focus on the code, not the person
- Help others learn
- Assume good intentions
- Report harassment to maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Quick Reference

### Common Commands

```bash
# Run all tests
pytest tests/ -v

# Quick benchmark
python benchmark.py --quick

# Full benchmark
python benchmark.py --output benchmarks/results.json

# Run a navigation research runner (basal-ganglia action selection)
python -m research.runners.g11_bg_runner --moving-goal --seed 42 --n-steps 1800

# Run a static probe of the action-selection circuit
python -m research.runners.g11_bg_runner --probe-action W

# Check GPU memory
nvidia-smi

# Format code (if using black)
black neural-simulator.py sim/ experiment/ tests/
```

### File Structure

```
neural-simulator.py            # GUI host + main entry point (~2.2K lines)
sim/                           # Core engine package (43 modules, ~15.9K lines of code)
  __init__.py                  # public API: SimulationBridge, configs, enums
  bridge.py                    # SimulationBridge — GPU state + step loop (largest module)
  config.py                    # @dataclass configs (CoreSimConfig etc.)
  enums.py                     # NeuronType, NeuronModel, preset managers
  connectivity.py              # spatial / small-world / motif generators (CuPy/NumPy)
  kernels.py                   # fused Izhikevich/HH/AdEx + plasticity kernels
  profiles.py                  # NEURAL_STRUCTURE_PROFILES dict (region presets)
  regions.py                   # BrainRegion, RegionPathway, RegionManager
  neuromodulators.py           # declarative dopamine / norepinephrine / serotonin subsystem
  backend.py                   # CuPy/NumPy backend abstraction (SIM_BACKEND)
  data_bus.py                  # DataChannel pub/sub
  replicas.py                  # replicated wiring (multi-network support)
  text_embeddings.py           # token embeddings for language regions
  visual_cortex.py             # Gabor/V1 receptive fields + retina rendering
  lineage.py                   # continuous-learning state across sessions
  progress.py                  # universal [PROGRESS] {json} event format
  # ... plus ~28 more modules (synapse storage, tokenizers, learning rules, etc.)
viz/                           # OpenGL renderer / camera / picker / overlays
ui/                            # DearPyGUI panels / callbacks / layout / plots
experiment/                    # ExperimentEngine + StimulusManager + Readout + Training
experiments/                   # YAML configs for headless parameter sweeps
research/
  runners/                     # ~1,270 headless experiment scripts (navigation, conversation/chat, memory consolidation, the word-meaning-learning experiments, the language generator, …); each runs via `python -m research.runners.<name>`
  findings/                    # session-by-session research notes (~1,780 markdown docs, incl. negative results)
  findings/raw/                # raw JSON output per experiment run
  datasets/                    # synthetic datasets (e.g. tiny_patterns.npz)
  experiment_runner.py         # YAML-driven sweep orchestrator
  result_aggregator.py         # cross-condition rollup + pass/fail summary
docs/
  INDEX.md                     # documentation map — start here to navigate the docs
  CURRENT-STATE.md             # what works today, technical details
  SCIENCE_ROADMAP.md           # long-term scientific direction + results table
  biology.md                   # neuroscience tour in plain language
  diagrams/                    # live Mermaid architecture flowcharts (current + detailed)
  plans/                       # per-feature design docs (paired with findings); current primary plan: 2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md
webapp/                        # FastAPI dashboard (server.py + static/)
tests/                         # 472 test files (mostly CPU-only)
  test_determinism.py          # RNG determinism (init + step)
  test_kernels_cpu.py          # CPU validation of fused kernels
  test_experiment_system.py    # experiment engine + stimulus manager
  test_neuromodulators.py      # neuromodulator subsystem
  test_regions.py              # brain-region framework + plasticity gates
  test_data_bus.py             # data-bus pub/sub
  test_backend.py              # CuPy/NumPy backend parity
  test_g{1,2,3,5,6,8,9}_runner_smoke.py  # per-runner smoke tests
  test_g11_bg_runner_flags.py  # navigation runner flag tests
  test_plastic_mask.py         # per-synapse plastic freeze
  test_plastic_mask_checkpoint.py  # plastic mask survives checkpoints
  test_progress.py             # universal [PROGRESS] event format
  test_experiment_runner.py    # YAML sweep runner
  test_result_aggregator.py    # cross-condition aggregator
  ...
benchmark.py                   # GPU throughput benchmark runner
viz_benchmark.py               # visualization performance benchmark
run_benchmarks.py              # biological validation suite (STDP timing, E/I balance, STP, gamma)
run_experiment_headless.py     # run a built-in experiment preset without GUI
run_parameter_sweep.py         # grid/zip parameter sweep with t-test + Cohen's d
simulation_profiles/           # 47 brain-region JSON profiles + auto-tune cache
simulation_checkpoints_h5/     # saved simulation state
simulation_recordings_h5/      # frame-by-frame recordings
```

### Common Imports

The engine exposes its public API through `sim/__init__.py`:

```python
from sim import (
    SimulationBridge, CoreSimConfig, VisualizationConfig,
    RuntimeState, GPUConfig, NeuronModel, NeuronType,
)
```

For research runners and brain-region work:

```python
from sim.regions import BrainRegion, RegionPathway
from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule
from sim.enums import DefaultIzhikevichParamsManager, DefaultHodgkinHuxleyParams
```

Thank you for contributing! 🚀
