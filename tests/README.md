# Simulator Tests

This directory contains test suites for validating the neural network simulator's correctness, determinism, and performance.

## Test Suites

### Determinism Tests (`test_determinism.py`)
Validates that simulations are deterministic and reproducible:
- Same seed → identical spike trains
- Same seed → identical membrane potential traces
- Same seed → identical connectivity
- Seed tracking works correctly

**Run with pytest:**
```bash
pytest tests/test_determinism.py -v
```

**Run directly:**
```bash
python tests/test_determinism.py
```

### Regression Tests (`test_regression.py`)
*Coming soon* - Will compare key metrics against stored baselines to detect unintended behavioral changes.

## Requirements

Tests require:
- `pytest` (for test framework)
- All simulator dependencies (cupy, numpy, etc.)

Install pytest:
```bash
pip install pytest
```

## Running All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=neural_simulator --cov-report=html

# Run specific test class
pytest tests/test_determinism.py::TestDeterministicSpikes -v

# Run specific test method
pytest tests/test_determinism.py::TestDeterministicSpikes::test_izhikevich_deterministic_spikes -v
```

## Test Organization

Tests are organized into classes by functionality:
- `TestDeterministicSpikes`: Spike train reproducibility
- `TestDeterministicMembranePotential`: Voltage trace reproducibility
- `TestDeterministicConnectivity`: Network structure reproducibility
- `TestSeedTracking`: RNG seed management

## Writing New Tests

When adding new features:
1. Add determinism tests if feature involves randomness
2. Add regression tests for key metrics
3. Ensure tests are fast (< 30s each ideally)
4. Use small network sizes for unit tests
5. Document what the test validates

## Continuous Integration

These tests should be run:
- Before merging any pull request
- After any changes to core simulation logic
- After performance optimizations (to ensure correctness preserved)
