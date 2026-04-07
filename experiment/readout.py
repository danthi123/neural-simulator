"""Readout engine for measuring network responses.

Provides real-time population firing rate, spike counts, synchrony metrics,
and optional spectral analysis (PSD, band power). All computations stay on
GPU where possible to minimize transfer overhead.
"""

import numpy as np

from sim.config import ReadoutConfig
from sim.enums import NeuronGroupRole


class ReadoutEngine:
    """Measures and logs network responses per neuron group.

    Provides real-time population firing rate, spike counts,
    and optional spectral analysis. All computations stay on GPU
    where possible to minimize transfer overhead.
    """

    def __init__(self, n_neurons, dt_ms):
        self.n_neurons = n_neurons
        self.dt_ms = dt_ms
        self.config = ReadoutConfig()
        self.group_manager = None

        # Rate estimation buffers (circular buffers on GPU)
        self._rate_buffers = {}          # group_name -> circular buffer of spike counts
        self._rate_buffer_idx = 0
        self._rate_buffer_size = 0

        # Spike count accumulators
        self._spike_counts = {}          # group_name -> int accumulator
        self._spike_count_window_steps = 0
        self._spike_count_step = 0

        # PSD buffers
        self._psd_buffers = {}           # group_name -> voltage history buffer
        self._psd_buffer_idx = 0

        # Current readout values (CPU, for UI display and logging)
        self.current_rates = {}          # group_name -> float (Hz)
        self.current_spike_counts = {}   # group_name -> int
        self.current_psd = {}            # group_name -> dict with freqs, power

        # Synchrony metrics (computed per readout window)
        self._sync_spike_counts = {}     # group_name -> list of per-step spike fractions
        self.current_synchrony = {}      # group_name -> synchrony index (0-1)

        # Trial-level metrics
        self.trial_metrics = []          # List of per-trial measurement dicts

    def initialize(self, config, group_manager, cp_module):
        """Set up readout buffers.

        Args:
            config: ReadoutConfig
            group_manager: NeuronGroupManager
            cp_module: CuPy module reference
        """
        self.config = config
        self.group_manager = group_manager

        # Rate buffer: store spike counts per step for sliding window
        self._rate_buffer_size = max(1, int(config.rate_window_ms / self.dt_ms))
        self._rate_buffer_idx = 0

        groups_to_track = config.rate_group_names
        if not groups_to_track:
            # Default: track all output groups
            groups_to_track = [g.name for g in group_manager.get_groups_by_role(NeuronGroupRole.OUTPUT.name)]
            # Also track input groups for comparison
            groups_to_track += [g.name for g in group_manager.get_groups_by_role(NeuronGroupRole.INPUT.name)]

        for gname in groups_to_track:
            group = group_manager.get_group(gname)
            if group and group.neuron_indices:
                self._rate_buffers[gname] = cp_module.zeros(self._rate_buffer_size, dtype=cp_module.float32)
                self._spike_counts[gname] = 0
                self.current_rates[gname] = 0.0
                self.current_spike_counts[gname] = 0

        # Spike count window
        self._spike_count_window_steps = max(1, int(config.spike_count_window_ms / self.dt_ms))
        self._spike_count_step = 0

        # Synchrony tracking: store per-step spike fractions within the rate window
        for gname in groups_to_track:
            self._sync_spike_counts[gname] = []
            self.current_synchrony[gname] = 0.0

        # PSD buffer
        if config.enable_psd:
            psd_steps = max(1, int(config.psd_window_ms / self.dt_ms))
            for gname in groups_to_track:
                self._psd_buffers[gname] = cp_module.zeros(psd_steps, dtype=cp_module.float32)
            self._psd_buffer_idx = 0

    def update(self, cp_firing_states, cp_membrane_potential_v, cp_module):
        """Update readout measurements for the current timestep.

        Args:
            cp_firing_states: GPU bool array [n_neurons] of current spikes
            cp_membrane_potential_v: GPU float32 array [n_neurons] of membrane voltages
            cp_module: CuPy module reference
        """
        for gname, buffer in self._rate_buffers.items():
            group = self.group_manager.get_group(gname)
            if group is None or not group.neuron_indices:
                continue

            # Count spikes in this group this step
            group_indices = cp_module.array(group.neuron_indices, dtype=cp_module.int32)
            group_spikes = cp_firing_states[group_indices]
            n_spikes = float(cp_module.sum(group_spikes).get())
            n_neurons_in_group = len(group.neuron_indices)

            # Update circular rate buffer
            buffer[self._rate_buffer_idx % self._rate_buffer_size] = n_spikes

            # Compute instantaneous population rate (Hz)
            total_spikes_in_window = float(cp_module.sum(buffer).get())
            window_duration_s = self._rate_buffer_size * self.dt_ms / 1000.0
            if n_neurons_in_group > 0 and window_duration_s > 0:
                self.current_rates[gname] = total_spikes_in_window / (n_neurons_in_group * window_duration_s)

            # Update spike count accumulator
            self._spike_counts[gname] = self._spike_counts.get(gname, 0) + int(n_spikes)

            # Track per-step spike fraction for synchrony computation
            spike_frac = n_spikes / max(n_neurons_in_group, 1)
            sync_list = self._sync_spike_counts.get(gname)
            if sync_list is not None:
                sync_list.append(spike_frac)
                # Keep only the last rate_window worth of steps
                if len(sync_list) > self._rate_buffer_size:
                    sync_list.pop(0)
                # Synchrony index: variance of spike fractions normalized by mean.
                # High synchrony = neurons fire together (high variance in fraction).
                # Fano factor of population spike count: Var(count) / Mean(count).
                # Ranges from ~0 (asynchronous, Poisson) to >>1 (synchronous bursting).
                if len(sync_list) >= 2:
                    arr = np.array(sync_list)
                    mean_f = arr.mean()
                    if mean_f > 1e-9:
                        # Fano factor of spike count = Var(N) / E[N]
                        # N = fraction * n_neurons, so Fano = Var(frac) * n / mean(frac)
                        fano = float(arr.var() * n_neurons_in_group / mean_f)
                        self.current_synchrony[gname] = round(fano, 4)
                    else:
                        self.current_synchrony[gname] = 0.0

        # Advance circular buffer index
        self._rate_buffer_idx += 1

        # Spike count window reset
        self._spike_count_step += 1
        if self._spike_count_step >= self._spike_count_window_steps:
            for gname in self._spike_counts:
                self.current_spike_counts[gname] = self._spike_counts[gname]
                self._spike_counts[gname] = 0
            self._spike_count_step = 0

        # PSD buffer update
        if self.config.enable_psd:
            psd_size = len(next(iter(self._psd_buffers.values()))) if self._psd_buffers else 0
            for gname, psd_buf in self._psd_buffers.items():
                group = self.group_manager.get_group(gname)
                if group and group.neuron_indices:
                    group_indices = cp_module.array(group.neuron_indices, dtype=cp_module.int32)
                    mean_v = float(cp_module.mean(cp_membrane_potential_v[group_indices]).get())
                    psd_buf[self._psd_buffer_idx % psd_size] = mean_v
            self._psd_buffer_idx += 1

    def compute_psd(self, group_name, cp_module):
        """Compute power spectral density for a group.

        Returns dict with 'frequencies_hz' and 'power' arrays (numpy).
        """
        psd_buf = self._psd_buffers.get(group_name)
        if psd_buf is None:
            return None

        signal = psd_buf.get()  # Transfer to CPU

        # FFT
        n = len(signal)
        if n < 2:
            return None

        fft_vals = np.fft.rfft(signal - np.mean(signal))
        power = np.abs(fft_vals) ** 2 / n
        freqs = np.fft.rfftfreq(n, d=self.dt_ms / 1000.0)

        # Filter to requested range
        f_min, f_max = self.config.psd_freq_range_hz
        mask = (freqs >= f_min) & (freqs <= f_max)

        return {
            'frequencies_hz': freqs[mask],
            'power': power[mask],
        }

    def compute_band_power(self, group_name, cp_module):
        """Compute power in standard frequency bands for a group.

        Returns dict mapping band names to power values, or None if PSD
        is not enabled or buffer is insufficient.

        Bands (Hz): delta 1-4, theta 4-8, alpha 8-13, beta 13-30,
                    gamma 30-80, high_gamma 80-150.
        """
        psd = self.compute_psd(group_name, cp_module)
        if psd is None:
            return None

        freqs = psd['frequencies_hz']
        power = psd['power']

        bands = {
            'delta': (1.0, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 80.0),
            'high_gamma': (80.0, 150.0),
        }

        band_power = {}
        total_power = float(np.sum(power)) if len(power) > 0 else 1e-12
        for band_name, (f_lo, f_hi) in bands.items():
            mask = (freqs >= f_lo) & (freqs < f_hi)
            bp = float(np.sum(power[mask])) if np.any(mask) else 0.0
            band_power[band_name] = round(bp, 6)
        band_power['total'] = round(total_power, 6)

        # Relative power (fraction of total)
        for band_name in bands:
            band_power[f'{band_name}_rel'] = (
                round(band_power[band_name] / total_power, 4)
                if total_power > 0 else 0.0
            )

        # Dominant band
        band_powers_abs = {k: band_power[k] for k in bands}
        band_power['dominant_band'] = max(band_powers_abs, key=band_powers_abs.get)

        return band_power

    def get_trial_snapshot(self):
        """Get current readout state for trial logging."""
        return {
            'rates': dict(self.current_rates),
            'spike_counts': dict(self.current_spike_counts),
        }

    def cleanup(self):
        """Release GPU memory."""
        self._rate_buffers.clear()
        self._psd_buffers.clear()
        self._spike_counts.clear()
