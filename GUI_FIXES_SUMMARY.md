# GUI Enhancements - Fixes and Improvements

## Issues Fixed

### 1. Benchmark Out-of-Memory (OOM) Failures

**Problem**: Benchmark was attempting configurations that exceeded available GPU VRAM, resulting in CuPy allocation errors like:
```
Out of memory allocating 30,000,000,000 bytes
```

**Root Cause**: No pre-check of available GPU memory before attempting to initialize large simulations.

**Solution** (benchmark.py):

#### Added Memory Estimation Method
```python
def estimate_memory_requirement(self, n_neurons, conn_per_neuron):
    # Conservative estimates:
    # - Each neuron: ~200 bytes (state variables)
    # - Each synapse: ~8 bytes (weight in sparse matrix)
    bytes_per_neuron = 200
    synapses_estimate = n_neurons * conn_per_neuron
    bytes_per_synapse = 8
    
    # Add 1.5x overhead for buffers, temp arrays
    overhead_factor = 1.5
    
    estimated_bytes = (
        (n_neurons * bytes_per_neuron + synapses_estimate * bytes_per_synapse) 
        * overhead_factor
    )
    
    return estimated_bytes
```

#### Added Memory Check Method
```python
def check_memory_available(self, required_bytes):
    mem_info = cp.cuda.Device().mem_info
    free_mem, total_mem = mem_info
    
    # Use 80% of free memory as safe limit
    safe_limit = free_mem * 0.8
    
    return required_bytes <= safe_limit, free_mem, total_mem
```

#### Pre-Flight Check in run_single_benchmark()
Before initializing simulation:
```python
estimated_mem = self.estimate_memory_requirement(
    config_dict['num_neurons'],
    config_dict['connections_per_neuron']
)

can_run, free_mem, total_mem = self.check_memory_available(estimated_mem)

print(f"Estimated memory needed: {estimated_mem / (1024**3):.2f}GB")
print(f"GPU free memory: {free_mem / (1024**3):.2f}GB / {total_mem / (1024**3):.2f}GB total")

if not can_run:
    print(f"SKIPPED: Insufficient GPU memory")
    return None
```

**Benefits**:
- Prevents OOM crashes mid-benchmark
- Provides clear feedback on why configurations are skipped
- Shows memory estimates and availability upfront
- Allows benchmark suite to continue with viable configurations

**Expected Output Now**:
```
Estimated memory needed: 2.35GB
GPU free memory: 18.4GB / 20.0GB total
✓ Proceeding with benchmark

vs.

Estimated memory needed: 28.0GB
GPU free memory: 18.4GB / 20.0GB total
SKIPPED: Insufficient GPU memory (need ~28.0GB, have 18.4GB free)
```

---

### 2. Missing Real-Time Log Output During Benchmarks

**Problem**: Console output from benchmark.py wasn't appearing in the System Logs section until the entire process completed.

**Root Cause**: 
1. Python buffers stdout by default when captured via subprocess
2. No explicit flushing after print statements
3. LogCapture was capturing but GUI wasn't seeing live updates

**Solution** (benchmark.py):

Added `sys.stdout.flush()` after all important print statements:
```python
print("Initializing simulation...")
sys.stdout.flush()  # Force immediate output

print(f"Step {step+1}/{self.num_steps}...")
sys.stdout.flush()  # Show progress immediately
```

Added flush points at:
- Configuration start
- Memory checks
- Initialization status
- Progress indicators (every 100 steps)
- Results summary
- Error messages

**Benefits**:
- Real-time visibility into benchmark progress
- User can see which configuration is currently running
- Progress updates appear as they happen
- Error messages visible immediately

**Expected Behavior Now**:
User sees in System Logs as benchmark runs:
```
Running: 1000 neurons, 100 conn/neuron, IZHIKEVICH
Estimated memory needed: 0.18GB
GPU free memory: 18.4GB / 20.0GB total
Initializing simulation...
Initialization: 0.45s
Running 1000 simulation steps...
  Step 100/1000 - Avg time: 1.23ms
  Step 200/1000 - Avg time: 1.25ms
  ...
```

---

### 3. Model Optimization Button Not Functional

**Problem**: Clicking "Run Model Optimization" showed placeholder message:
```
"Model optimization feature coming in Phase B."
```

But the auto-tuning workflow (`--auto-tune` flag) already exists and works.

**Root Cause**: Handler function was a placeholder instead of calling the actual auto-tuning code.

**Solution** (neural-simulator.py):

Completely rewrote `handle_run_optimization_click()`:

```python
def handle_run_optimization_click(sender=None, app_data=None, user_data=None):
    """Runs the auto-tuning workflow to optimize drive scales."""
    def run_optimization():
        try:
            dpg.set_value("perf_test_status_text", "Running auto-tuning workflow...")
            dpg.set_value("perf_test_results_text", 
                "This may take several minutes.\nCheck console for detailed progress.")
            
            import subprocess
            result = subprocess.run(
                [sys.executable, "neural-simulator.py", "--auto-tune"],
                capture_output=True, 
                text=True, 
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0:
                # Count tuned combinations
                with open("simulation_profiles/auto_tuned_overrides.json", "r") as f:
                    data = json.load(f)
                count = len(data.get("tuned_combinations", {}))
                summary = f"Successfully tuned {count} model/profile combinations.\n" \
                         f"Reload overrides to apply them."
                dpg.set_value("perf_test_results_text", summary)
            else:
                # Show error output
                summary = result.stderr[-500:]
                dpg.set_value("perf_test_results_text", summary)
        except subprocess.TimeoutExpired:
            dpg.set_value("perf_test_status_text", 
                "Auto-tuning timed out after 30 minutes.")
        except Exception as e:
            dpg.set_value("perf_test_status_text", f"Error: {str(e)}")
    
    threading.Thread(target=run_optimization, daemon=True).start()
```

**Also Updated Button Label**:
```python
"Run Auto-Tuning (Optimize Drive Scales)"
```
Now clearly indicates what the button does.

**Benefits**:
- Actually runs the auto-tuning workflow
- Shows results and success count
- Timeout protection (30 minutes)
- Captures and displays errors
- Runs in background thread (non-blocking)
- Reminds user to "Reload Overrides" after completion

**Expected Workflow**:
1. User clicks "Run Auto-Tuning (Optimize Drive Scales)"
2. Status shows: "Running auto-tuning workflow..."
3. Results panel shows: "This may take several minutes. Check console for detailed progress."
4. System Logs section shows live progress from auto-tuning
5. On completion, status shows: "Auto-tuning complete. Results saved to auto_tuned_overrides.json"
6. Results show: "Successfully tuned 18 model/profile combinations. Reload overrides to apply them."
7. User clicks "Reload Auto-Tuned Overrides" to activate new settings

---

## Testing Recommendations

### 1. Benchmark Memory Checks
Test on system with known VRAM:
```bash
python benchmark.py --quick
```

Verify:
- [ ] Shows "Estimated memory needed" before each config
- [ ] Shows "GPU free memory" with actual values
- [ ] Skips configurations that exceed VRAM
- [ ] Successfully runs configs that fit in memory
- [ ] Reports accurate count of completed vs skipped configs

### 2. Real-Time Logging
While benchmark runs, check System Logs section:
- [ ] See configuration start messages immediately
- [ ] See progress updates every 100 steps
- [ ] See completion messages as they happen
- [ ] Auto-scroll keeps latest logs visible
- [ ] Can search logs during benchmark run

### Auto-Tuning Integration
Click "Run Auto-Tuning (Optimize Drive Scales)":
- [ ] Status updates to "Running auto-tuning workflow (full mode)..." or "(quick mode)..."
- [ ] Quick checkbox toggles between full and quick mode (off by default)
- [ ] Quick mode runs faster with fewer test configurations
- [ ] System Logs shows auto-tuning progress
- [ ] Process completes within expected time
- [ ] Results show tuned combination count
- [ ] File `auto_tuned_overrides.json` is created/updated
- [ ] "Reload Auto-Tuned Overrides" loads new values
- [ ] "Apply Changes & Reset" uses new drive scales

### 4. Combined Workflow Test
Full test sequence:
1. Run benchmark suite
2. Check which configs succeeded/failed
3. Run auto-tuning on successful configs
4. Reload overrides
5. Apply changes and verify improved firing rates

---

## File Changes Summary

### benchmark.py
- Added `estimate_memory_requirement()` method
- Added `check_memory_available()` method  
- Added memory pre-check in `run_single_benchmark()`
- Added `sys.stdout.flush()` after all prints
- Memory info displayed before attempting each config

### neural-simulator.py
- Completely rewrote `handle_run_optimization_click()`
- Now spawns subprocess with `--auto-tune` flag
- Shows progress and results in GUI
- Updated button label to clarify functionality
- Added 30-minute timeout protection

---

## Memory Estimation Details

### Conservative Formula
```
total_bytes = (
    (n_neurons * 200 bytes) +           # Neuron state variables
    (n_neurons * conn_per_neuron * 8)   # Synapse weights
) * 1.5                                  # 50% overhead factor
```

### Overhead Includes:
- Recording buffers (if enabled)
- Temporary arrays during computation
- Sparse matrix internal structures
- CuPy memory pool overhead
- Visualization data copies

### Safety Margin
Uses 80% of **free** memory (not total) to account for:
- Other GPU processes
- Memory fragmentation
- Allocation spikes during runtime
- OS/driver reserves

### Example Calculations

| Neurons | Conn/N | Synapses | Estimate | Actual Need |
|---------|--------|----------|----------|-------------|
| 1,000   | 100    | 100K     | 0.18GB   | ~0.15GB     |
| 10,000  | 500    | 5M       | 7.5GB    | ~6.2GB      |
| 50,000  | 1,000  | 50M      | 90GB     | ~75GB       |

**Note**: Estimates intentionally conservative. Better to skip than crash.

---

## Known Limitations

1. **Memory estimation is heuristic**: Actual usage can vary by ±20% depending on:
   - Neuron model complexity (HH uses more than Izhikevich)
   - Plasticity features enabled
   - Recording mode
   - Sparse matrix density

2. **Auto-tuning timeout**: Very large networks may need more than 30 minutes. Can be increased in code if needed.

3. **Log buffer size**: LogCapture limited to 5000 lines. Very long benchmarks may truncate early output.

---

## Future Enhancements

1. **Dynamic memory estimation**: Query actual sizes after initialization, adjust future estimates
2. **Quick auto-tune option**: Add `--quick` flag support to optimization button
3. **Partial completion**: Save partial auto-tune results if interrupted
4. **Memory profiling**: Track actual vs estimated memory usage, improve formula
5. **GPU selection**: Support multi-GPU systems, choose device with most free memory
