# research/findings/raw/direction_Q_protocol.py
"""Direction Q Wang 2002 delayed-response protocol functions.

Pure functions that implement the canonical Wang 2002 working-memory
test on a Q test bridge (built by direction_Q_bridge_builder):

    baseline (no stim)
       -> cue (current injection into a subset of dlpfc_wm exc neurons)
       -> delay (no stim; measure persistence over time bins)

NMDA-driven persistent activity (Wang 2002 §"Working Memory Performance")
should manifest as a delay-period firing rate that stays elevated above
baseline for seconds after cue offset. This module exposes the three
primitive measurement functions; the multi-seed runner (Task 4)
composes them into the full protocol and feeds results to the
pre-registered verdict module (Task 3).

The bridge step machinery (NMDA decay, conductance update, neuron
dynamics, fast-spike-reset) is invoked unchanged; this module only
reads cp_firing_states and writes cp_external_input_current. The
existing fused NMDA kernel (fused_nmda_update_and_current) is what
actually decides whether persistence emerges or not.
"""
from __future__ import annotations
from typing import List


def _dlpfc_exc_indices(bridge) -> list:
    """Return indices of dlpfc_wm EXCITATORY neurons (first exc_fraction
    of the region, by construction in BrainRegion / RegionManager).

    Inhibitory neurons are excluded so that cue injection drives the
    pyramidal population that carries the persistent-activity signature.
    """
    all_idx = bridge.region_manager.indices("dlpfc_wm")
    inh_idx = set(bridge.region_manager.inhibitory_indices("dlpfc_wm"))
    exc = [i for i in all_idx if i not in inh_idx]
    return exc


def _step_and_accumulate(bridge, n_steps: int, exc_indices: list) -> int:
    """Step the bridge n_steps times; accumulate dlpfc_wm exc spike total.

    Reads bridge.cp_firing_states (boolean per-neuron mask True iff the
    neuron spiked on that step). Returns total spike count summed across
    n_steps and across exc_indices (a python int).
    """
    from sim.backend import get_backend
    xp, _ = get_backend()
    idx_arr = xp.asarray(exc_indices, dtype=xp.int64)
    total_spikes = 0
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[idx_arr]
        total_spikes += int(fired.sum())
    return total_spikes


def _spikes_to_rate(total_spikes: int, n_neurons: int,
                    duration_ms: float) -> float:
    """Convert raw spike count to mean population firing rate (Hz).

    rate = total_spikes / n_neurons / (duration_s)
    """
    if n_neurons <= 0 or duration_ms <= 0.0:
        return 0.0
    duration_s = duration_ms / 1000.0
    return float(total_spikes) / float(n_neurons) / float(duration_s)


def run_baseline_period(bridge, duration_ms: float) -> float:
    """Step bridge with zero external drive for duration_ms; return mean
    dlpfc_wm exc firing rate (Hz) over the period.

    Used to compute the pre-cue baseline rate against which delay-period
    persistence is normalized in the verdict (rate_ratio = delay/baseline).
    """
    from sim.backend import get_backend
    xp, _ = get_backend()

    dt_ms = float(bridge.core_config.dt_ms)
    n_steps = int(duration_ms / dt_ms)
    if n_steps <= 0:
        return 0.0

    exc = _dlpfc_exc_indices(bridge)
    n_exc = len(exc)

    # Zero external drive for baseline.
    bridge.cp_external_input_current[:] = 0.0

    total = _step_and_accumulate(bridge, n_steps, exc)
    actual_duration_ms = n_steps * dt_ms
    return _spikes_to_rate(total, n_exc, actual_duration_ms)


def apply_cue_stimulus(bridge, cue_amplitude_pA: float,
                          duration_ms: float,
                          cue_fraction: float) -> float:
    """Inject cue_amplitude_pA into the first cue_fraction of dlpfc_wm
    exc neurons for duration_ms; return mean dlpfc_wm exc firing rate (Hz)
    during the cue window.

    After the cue window completes, cp_external_input_current is zeroed
    so the immediately-following delay-period measurement starts clean
    (no residual external drive — only intrinsic + synaptic dynamics).
    """
    from sim.backend import get_backend
    xp, _ = get_backend()

    dt_ms = float(bridge.core_config.dt_ms)
    n_steps = int(duration_ms / dt_ms)
    if n_steps <= 0:
        return 0.0

    exc = _dlpfc_exc_indices(bridge)
    n_exc = len(exc)
    n_cue = max(1, int(cue_fraction * n_exc))
    cue_neurons = exc[:n_cue]

    # Set up cue drive (zero everything first, then set cue indices).
    bridge.cp_external_input_current[:] = 0.0
    cue_idx_arr = xp.asarray(cue_neurons, dtype=xp.int64)
    bridge.cp_external_input_current[cue_idx_arr] = \
        xp.asarray(cue_amplitude_pA, dtype=xp.float32)

    total = _step_and_accumulate(bridge, n_steps, exc)

    # Clear drive so delay period starts unstimulated.
    bridge.cp_external_input_current[:] = 0.0

    actual_duration_ms = n_steps * dt_ms
    return _spikes_to_rate(total, n_exc, actual_duration_ms)


def measure_delay_period(bridge, duration_ms: float,
                          bin_ms: float) -> List[float]:
    """Step bridge with zero external drive for duration_ms; return list
    of mean dlpfc_wm exc firing rates (Hz), one per bin_ms-wide bin.

    The trajectory characterizes whether NMDA-driven persistent activity
    decays cleanly to baseline (no persistence) or stays elevated for
    seconds (Wang 2002 working-memory bistable attractor).
    """
    from sim.backend import get_backend
    xp, _ = get_backend()

    dt_ms = float(bridge.core_config.dt_ms)
    n_bins = int(duration_ms / bin_ms)
    if n_bins <= 0:
        return []
    n_steps_per_bin = int(bin_ms / dt_ms)
    if n_steps_per_bin <= 0:
        return [0.0] * n_bins

    exc = _dlpfc_exc_indices(bridge)
    n_exc = len(exc)

    # Zero drive for delay period.
    bridge.cp_external_input_current[:] = 0.0

    rates: List[float] = []
    actual_bin_ms = n_steps_per_bin * dt_ms
    for _bin in range(n_bins):
        total = _step_and_accumulate(bridge, n_steps_per_bin, exc)
        rates.append(_spikes_to_rate(total, n_exc, actual_bin_ms))
    return rates
