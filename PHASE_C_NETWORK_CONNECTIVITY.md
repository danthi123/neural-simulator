# Phase C: Network-Level Realism & Connectivity

## Overview
Phase C extends the simulator with biologically realistic network connectivity patterns and synaptic plasticity mechanisms, enabling the simulation of brain-region-specific circuits and learning dynamics.

## Features

### C1: Enhanced Layered Connectivity Motifs ✓ (Partially Implemented)
**Status**: Connectivity motifs framework exists; extended with additional patterns

**Scientific Basis**:
- Binzegger et al. (2004) "A quantitative map of the circuit of cat primary visual cortex"
- Douglas & Martin (2004) "Neuronal circuits of the neocortex"
- Amaral & Witter (1989) "The three-dimensional organization of the hippocampal formation"

**Implemented Connectivity Patterns**:
1. **Cortical Layers** (L2/3, L4, L5/6)
   - Layer-specific E/I ratios
   - Inter-laminar connectivity (e.g., L4→L2/3, L5→L2/3)
   - Distance-dependent connection probability
   
2. **Hippocampal Circuits**
   - CA3 recurrent collaterals (strong E→E)
   - CA1 feed-forward inhibition
   - Topographic organization

3. **Thalamic Loops**
   - TC-TRN reciprocal connectivity
   - Burst-mode dynamics

4. **Basal Ganglia**
   - STN-GPe loop
   - Striatal MSN connectivity

### C2: Spike-Timing-Dependent Plasticity (STDP)
**Status**: ✓ Implemented

**Scientific Basis**:
- Bi & Poo (1998) "Synaptic modifications in cultured hippocampal neurons"
- Markram et al. (1997) "Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs"
- Song et al. (2000) "Competitive Hebbian learning through spike-timing-dependent synaptic plasticity"

**Implementation Details**:
```
Pre-before-Post (LTP):  Δw = A_plus * exp(-Δt / τ_plus)   for Δt > 0
Post-before-Pre (LTD):  Δw = -A_minus * exp(Δt / τ_minus)  for Δt < 0

where:
- Δt = t_post - t_pre (spike timing difference)
- A_plus = 0.01 (LTP amplitude, ~1% weight change)
- A_minus = 0.0105 (LTD amplitude, slightly asymmetric)
- τ_plus = 20 ms (LTP time constant)
- τ_minus = 20 ms (LTD time constant)
```

**Features**:
- Asymmetric STDP window (classical Bi & Poo form)
- Eligibility traces for spike timing memory
- All-to-all vs nearest-neighbor spike pairing options
- GPU-accelerated implementation using sparse matrix operations
- Weight-dependent plasticity (soft bounds near w_min/w_max)

**Parameters**:
- `enable_stdp`: Enable/disable STDP
- `stdp_a_plus`: LTP amplitude (default: 0.01)
- `stdp_a_minus`: LTD amplitude (default: 0.0105)
- `stdp_tau_plus`: LTP time window in ms (default: 20.0)
- `stdp_tau_minus`: LTD time window in ms (default: 20.0)
- `stdp_w_min`: Minimum synaptic weight (default: 0.001)
- `stdp_w_max`: Maximum synaptic weight (default: 1.0)

### C2: Reward-Modulated Plasticity
**Status**: ✓ Implemented

**Scientific Basis**:
- Schultz et al. (1997) "A neural substrate of prediction and reward"
- Reynolds & Wickens (2002) "Dopamine-dependent plasticity of corticostriatal synapses"
- Izhikevich (2007) "Solving the distal reward problem through linkage of STDP and dopamine signaling"

**Implementation Details**:
```
Three-factor learning rule:
Δw(t) = η * eligibility_trace(t) * reward_signal(t)

Eligibility trace decay:
e(t+dt) = e(t) * exp(-dt / τ_eligibility) + STDP_update(t)

where:
- η = reward_learning_rate (default: 0.01)
- τ_eligibility = 1000 ms (1 second trace persistence)
- reward_signal: Configurable scalar or time-varying signal
```

**Features**:
- Eligibility traces persist for ~1 second after synaptic events
- Dopamine-like reward signal modulates trace conversion to weight changes
- Supports delayed reinforcement learning tasks
- Compatible with STDP (can be enabled simultaneously)

**Parameters**:
- `enable_reward_modulation`: Enable/disable reward-modulated plasticity
- `reward_learning_rate`: Modulation strength (default: 0.01)
- `reward_eligibility_tau_ms`: Trace decay time constant (default: 1000.0)
- `reward_baseline`: Baseline reward level (default: 0.0)
- `current_reward_signal`: Current reward value (set dynamically)

### C3: Structural Plasticity (Optional)
**Status**: ✓ Implemented

**Scientific Basis**:
- Butz et al. (2009) "A simple rule for dendritic spine and axonal bouton formation"
- Knott et al. (2006) "Spine growth precedes synapse formation in the adult neocortex in vivo"
- Turrigiano (2008) "The self-tuning neuron: synaptic scaling of excitatory synapses"

**Implementation Details**:
```
Synapse Formation:
- Probability ∝ activity correlation + spatial proximity
- New synapses initialized at w_init = 0.1 * w_max
- Formation rate: ~0.1% of synapses per second (10 connections/10k per step @ dt=1ms)

Synapse Elimination:
- Weak synapses (w < pruning_threshold) pruned stochastically
- Pruning rate: ~0.05% of synapses per second
- Homeostatic regulation maintains target connection density

Activity-Dependent Formation:
- High pre-post co-activation → increased formation probability
- Spatial clustering (distance < 50 units) preferred
```

**Features**:
- Activity-dependent synapse formation and elimination
- Homeostatic connection density regulation
- Pruning of weak/unused synapses
- GPU-efficient sparse matrix updates

**Parameters**:
- `enable_structural_plasticity`: Enable/disable structural plasticity
- `struct_plast_formation_rate`: Formation probability per step (default: 1e-6)
- `struct_plast_elimination_rate`: Elimination probability per step (default: 5e-7)
- `struct_plast_weight_threshold`: Pruning threshold (default: 0.05)
- `struct_plast_target_density`: Target connection density (default: current density)

## Configuration Examples

### Basic STDP Learning
```python
config = {
    "enable_stdp": True,
    "stdp_a_plus": 0.01,
    "stdp_a_minus": 0.0105,
    "stdp_tau_plus": 20.0,
    "stdp_tau_minus": 20.0,
}
```

### Reward-Modulated Learning (Reinforcement)
```python
config = {
    "enable_stdp": True,
    "enable_reward_modulation": True,
    "reward_learning_rate": 0.02,
    "reward_eligibility_tau_ms": 1000.0,
}

# During simulation, set reward signal:
sim_bridge.core_config.current_reward_signal = 1.0  # Reward delivery
# or
sim_bridge.core_config.current_reward_signal = -0.5  # Punishment
```

### Structural Plasticity (Long-term Adaptation)
```python
config = {
    "enable_structural_plasticity": True,
    "struct_plast_formation_rate": 1e-6,
    "struct_plast_elimination_rate": 5e-7,
    "struct_plast_weight_threshold": 0.05,
}
```

## Performance Considerations

### STDP Computational Cost
- Requires tracking spike times for all neurons (minimal memory: n * 4 bytes)
- Eligibility trace updates: O(active_synapses) per step
- Weight updates: O(synapses_with_recent_spikes) per step
- **Overhead**: ~5-10% for typical networks (1k-10k neurons)

### Reward Modulation Cost
- Additional eligibility trace storage: O(num_synapses) * 4 bytes
- Trace decay: O(num_synapses) per step (GPU-parallelized)
- **Overhead**: ~3-5% on top of STDP

### Structural Plasticity Cost
- Synapse addition: Requires CSR matrix reconstruction (expensive!)
- Synapse removal: Faster (in-place data modification)
- **Recommendation**: Run at low rates (1e-6 to 1e-5) or disable for short simulations
- **Overhead**: Varies; ~10-30% if formation rate is high

## Testing & Validation

### STDP Timing Window Test
```python
# Run test_phase_c.py
python test_phase_c.py --test stdp_timing

# Expected: Classic STDP curve (LTP for Δt>0, LTD for Δt<0)
```

### Reward Learning Test
```python
# Run reward modulation test
python test_phase_c.py --test reward_learning

# Expected: Synapses strengthen only when reward follows activity
```

### Structural Plasticity Test
```python
# Run structural plasticity test
python test_phase_c.py --test structural_dynamics

# Expected: Connection density stabilizes near target after initial transient
```

## Scientific References

1. **STDP**:
   - Bi, G. Q., & Poo, M. M. (1998). "Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type." *Journal of Neuroscience*, 18(24), 10464-10472.
   - Song, S., Miller, K. D., & Abbott, L. F. (2000). "Competitive Hebbian learning through spike-timing-dependent synaptic plasticity." *Nature Neuroscience*, 3(9), 919-926.

2. **Reward Modulation**:
   - Izhikevich, E. M. (2007). "Solving the distal reward problem through linkage of STDP and dopamine signaling." *Cerebral Cortex*, 17(10), 2443-2452.
   - Reynolds, J. N., & Wickens, J. R. (2002). "Dopamine-dependent plasticity of corticostriatal synapses." *Neural Networks*, 15(4-6), 507-521.

3. **Structural Plasticity**:
   - Butz, M., Wörgötter, F., & van Ooyen, A. (2009). "Activity-dependent structural plasticity." *Brain Research Reviews*, 60(2), 287-305.
   - Turrigiano, G. (2008). "The self-tuning neuron: synaptic scaling of excitatory synapses." *Cell*, 135(3), 422-435.

4. **Connectivity Patterns**:
   - Douglas, R. J., & Martin, K. A. (2004). "Neuronal circuits of the neocortex." *Annual Review of Neuroscience*, 27, 419-451.
   - Binzegger, T., Douglas, R. J., & Martin, K. A. (2004). "A quantitative map of the circuit of cat primary visual cortex." *Journal of Neuroscience*, 24(39), 8441-8453.

## Known Limitations

1. **STDP Implementation**:
   - Uses discrete timesteps (no continuous time integration)
   - Assumes instantaneous spike detection
   - Weight updates applied every step (vs batch updates in some implementations)

2. **Reward Modulation**:
   - Global reward signal (same for all synapses)
   - No spatial gradient of dopamine
   - Simplified three-factor rule (real DA signaling more complex)

3. **Structural Plasticity**:
   - Formation/elimination rates are stochastic approximations
   - No explicit dendritic spine/bouton dynamics
   - CSR matrix reconstruction can be slow for very large networks

## Future Enhancements (Post-Phase C)

- [ ] Triplet STDP rules (Pfister & Gerstner 2006)
- [ ] Calcium-based plasticity models
- [ ] Heterosynaptic plasticity
- [ ] Activity-dependent axonal sprouting
- [ ] Metaplasticity (sliding threshold)
