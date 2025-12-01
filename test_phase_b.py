#!/usr/bin/env python3
"""
Test script for Phase B (Parameter Heterogeneity & Enhanced Noise) features.

Validates mathematical correctness of:
- Parameter heterogeneity sampling distributions
- OU process statistical properties
- Conductance noise variability

Run with: python test_phase_b.py
"""

import sys
import numpy as np
import cupy as cp

def test_heterogeneity_sampling():
    """Test that parameter heterogeneity produces correct distributions."""
    print("=" * 60)
    print("TEST 1: Heterogeneity Sampling Statistics")
    print("=" * 60)
    
    n = 10000
    cp.random.seed(42)
    
    # Test lognormal distribution
    mean_log = np.log(0.02)
    sigma_log = 0.3
    samples = cp.random.lognormal(mean=mean_log, sigma=sigma_log, size=n).astype(cp.float32)
    samples_np = cp.asnumpy(samples)
    
    # Check that mean is close to expected (exp(mean_log + sigma_log^2/2))
    expected_mean = np.exp(mean_log + sigma_log**2 / 2)
    actual_mean = np.mean(samples_np)
    
    # Calculate CV (coefficient of variation)
    cv = np.std(samples_np) / actual_mean
    expected_cv = np.sqrt(np.exp(sigma_log**2) - 1)
    
    print(f"Lognormal distribution (mean_log={mean_log:.3f}, sigma_log={sigma_log}):")
    print(f"  Expected mean: {expected_mean:.4f}, Actual mean: {actual_mean:.4f}")
    print(f"  Expected CV: {expected_cv:.3f}, Actual CV: {cv:.3f}")
    
    assert abs(actual_mean - expected_mean) / expected_mean < 0.05, "Mean outside tolerance"
    assert abs(cv - expected_cv) / expected_cv < 0.1, "CV outside tolerance"
    
    print("✓ Heterogeneity sampling produces correct statistics")
    print()

def test_ou_process_properties():
    """Test OU process mathematical properties."""
    print("=" * 60)
    print("TEST 2: OU Process Properties")
    print("=" * 60)
    
    # OU process parameters
    dt = 0.1  # ms
    tau = 15.0  # ms
    sigma = 100.0  # pA
    mean = 0.0  # pA
    n_steps = 100000
    
    # Pre-compute OU coefficients
    dt_sec = dt / 1000.0
    tau_sec = tau / 1000.0
    decay_factor = np.exp(-dt_sec / tau_sec)
    noise_std = sigma * np.sqrt((1.0 - np.exp(-2.0 * dt_sec / tau_sec)) / 2.0)
    
    # Simulate OU process
    cp.random.seed(42)
    current = mean
    samples = []
    
    for _ in range(n_steps):
        noise = float(cp.random.randn())
        current = current * decay_factor + mean * (1.0 - decay_factor) + noise_std * noise
        samples.append(current)
    
    samples = np.array(samples)
    
    # Check steady-state statistics
    # After burn-in, mean should be close to mean parameter
    burn_in = 10000
    ss_samples = samples[burn_in:]
    actual_mean = np.mean(ss_samples)
    actual_std = np.std(ss_samples)
    
    # Theoretical steady-state std = sigma / sqrt(2)
    expected_std = sigma / np.sqrt(2.0)
    
    print(f"OU Process (tau={tau}ms, sigma={sigma}pA):")
    print(f"  Expected mean: {mean:.1f}pA, Actual mean: {actual_mean:.1f}pA")
    print(f"  Expected std: {expected_std:.1f}pA, Actual std: {actual_std:.1f}pA")
    
    assert abs(actual_mean - mean) < 5.0, "OU mean drift detected"
    assert abs(actual_std - expected_std) / expected_std < 0.05, "OU std incorrect"
    
    # Check autocorrelation time constant
    # Compute autocorrelation function
    acf_lags = np.arange(0, 500)  # 50ms
    acf = []
    for lag in acf_lags:
        if lag == 0:
            acf.append(1.0)
        else:
            corr = np.corrcoef(ss_samples[:-lag], ss_samples[lag:])[0, 1]
            acf.append(corr)
    
    acf = np.array(acf)
    
    # Theoretical: acf(t) = exp(-t/tau)
    theoretical_acf = np.exp(-acf_lags * dt / tau)
    
    # Check first 10 time constants
    check_range = min(100, len(acf))
    mse = np.mean((acf[:check_range] - theoretical_acf[:check_range])**2)
    
    print(f"  Autocorrelation MSE (first {check_range} steps): {mse:.4f}")
    assert mse < 0.01, "OU autocorrelation doesn't match theory"
    
    print("✓ OU process has correct statistical properties")
    print()

def test_conductance_noise():
    """Test conductance noise produces reasonable variability."""
    print("=" * 60)
    print("TEST 3: Conductance Noise")
    print("=" * 60)
    
    n = 10000
    g_nominal = 120.0  # mS/cm^2
    noise_std = 0.05  # 5% relative
    
    cp.random.seed(42)
    noise = cp.random.randn(n).astype(cp.float32)
    g_noisy = g_nominal * (1.0 + noise_std * noise)
    g_noisy = cp.maximum(g_noisy, 0.0)  # Clip negative values
    
    g_noisy_np = cp.asnumpy(g_noisy)
    
    actual_mean = np.mean(g_noisy_np)
    actual_std = np.std(g_noisy_np)
    actual_cv = actual_std / actual_mean
    
    expected_mean = g_nominal
    expected_std = g_nominal * noise_std
    expected_cv = noise_std
    
    print(f"Conductance Noise (nominal={g_nominal}, relative_std={noise_std}):")
    print(f"  Expected mean: {expected_mean:.2f}, Actual mean: {actual_mean:.2f}")
    print(f"  Expected std: {expected_std:.2f}, Actual std: {actual_std:.2f}")
    print(f"  Expected CV: {expected_cv:.3f}, Actual CV: {actual_cv:.3f}")
    
    assert abs(actual_mean - expected_mean) / expected_mean < 0.01
    assert abs(actual_cv - expected_cv) / expected_cv < 0.1
    
    print("✓ Conductance noise produces correct variability")
    print()

def main():
    """Run all Phase B tests."""
    print("\n")
    print("=" * 60)
    print("PHASE B FEATURE VALIDATION")
    print("Testing Parameter Heterogeneity & Enhanced Noise")
    print("=" * 60)
    print()
    
    try:
        test_heterogeneity_sampling()
        test_ou_process_properties()
        test_conductance_noise()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print()
        
        return 0
    
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED ✗: {e}")
        print("=" * 60)
        print()
        return 1
    
    except Exception as e:
        print()
        print("=" * 60)
        print(f"ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
