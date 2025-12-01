# Contributing to GPU-Accelerated Neural Network Simulator

Thank you for considering contributing to this project! This document provides guidelines and instructions for contributors.

## Development Setup

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA 11.x or 12.x
- CUDA Toolkit installed
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
def process_spikes(self, firing_states: cp.ndarray) -> cp.ndarray:
    """Process spike events and update synaptic conductances.
    
    Args:
        firing_states: Boolean array of neuron firing states (N,)
        
    Returns:
        Updated conductance array (N,)
        
    Notes:
        This method uses GPU-accelerated sparse matrix operations
        to propagate spikes through the network efficiently.
    """
    # Implementation
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
pytest tests/ --cov=neural_simulator --cov-report=html
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

# Export profiling
python -c "from neural_simulator import *; sim = ...; sim.export_profiling_report('prof.json')"

# Check GPU memory
nvidia-smi

# Format code (if using black)
black neural-simulator.py benchmark.py tests/
```

### File Structure

```
sim/
├── neural-simulator.py      # Main simulator code
├── benchmark.py             # Benchmark runner
├── tests/                   # Test suite
│   ├── test_determinism.py # Determinism tests
│   └── README.md           # Test documentation
├── benchmarks/             # Benchmark results
├── simulation_profiles/    # Saved configurations
├── simulation_checkpoints_h5/  # Saved states
└── simulation_recordings_h5/   # Recorded simulations
```

Thank you for contributing! 🚀
