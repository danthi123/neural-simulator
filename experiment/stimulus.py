"""Stimulus current generation for experiments.

Generates GPU current arrays from stimulus channel definitions.
Called once per simulation step to compute the total stimulus current
for all active channels.
"""

import math

from sim.enums import StimulusPatternType


class StimulusManager:
    """Generates GPU current arrays from stimulus channel definitions.

    Called once per simulation step to compute the total stimulus current
    for all active channels. The result is a CuPy array of shape [n_neurons]
    that gets added to the neuron dynamics input current.
    """

    def __init__(self, n_neurons, dt_ms):
        self.n_neurons = n_neurons
        self.dt_ms = dt_ms
        self.channels = []               # List[StimulusChannel]
        self.cp_stimulus_current = None  # GPU array [n_neurons], float32
        self._channel_target_masks = {}  # channel_name -> GPU bool array
        self._poisson_active = {}        # channel_name -> GPU bool array for active spikes
        self._poisson_timers = {}        # channel_name -> GPU float32 for spike duration countdown
        self._poisson_rate_vectors = {}  # channel_name -> GPU float32 dense per-neuron rate
        self._rng = None

    def initialize(self, channels, group_manager, cp_module):
        """Set up channels with resolved neuron targets.

        Args:
            channels: List[StimulusChannel] definitions
            group_manager: NeuronGroupManager for resolving group names
            cp_module: CuPy module reference
        """
        self.channels = [ch for ch in channels if ch.enabled]
        self.cp_stimulus_current = cp_module.zeros(self.n_neurons, dtype=cp_module.float32)
        self._rng = cp_module.random

        for ch in self.channels:
            # Resolve target neuron indices
            indices = self._resolve_targets(ch, group_manager)

            # Validate rate_vector_hz length up front so errors are clear.
            if ch.pattern.pattern_type == StimulusPatternType.RATE_VECTOR_POISSON.name:
                if len(ch.pattern.rate_vector_hz) != len(indices):
                    raise ValueError(
                        f"RATE_VECTOR_POISSON rate_vector_hz length "
                        f"({len(ch.pattern.rate_vector_hz)}) must equal number of "
                        f"target neurons ({len(indices)}) for channel '{ch.name}'"
                    )

            mask = cp_module.zeros(self.n_neurons, dtype=cp_module.bool_)
            if len(indices) > 0:
                mask[cp_module.array(indices, dtype=cp_module.int32)] = True
            self._channel_target_masks[ch.name] = mask

            # Initialize Poisson state if needed
            if ch.pattern.pattern_type == StimulusPatternType.POISSON_SPIKE_TRAIN.name:
                self._poisson_active[ch.name] = cp_module.zeros(self.n_neurons, dtype=cp_module.bool_)
                self._poisson_timers[ch.name] = cp_module.zeros(self.n_neurons, dtype=cp_module.float32)
            elif ch.pattern.pattern_type == StimulusPatternType.RATE_VECTOR_POISSON.name:
                # Build a dense n_neurons-sized rate array (0 on non-target neurons).
                rate_full = cp_module.zeros(self.n_neurons, dtype=cp_module.float32)
                if len(indices) > 0:
                    idx_arr = cp_module.array(indices, dtype=cp_module.int32)
                    rate_full[idx_arr] = cp_module.array(
                        ch.pattern.rate_vector_hz, dtype=cp_module.float32
                    )
                self._poisson_rate_vectors[ch.name] = rate_full
                self._poisson_active[ch.name] = cp_module.zeros(self.n_neurons, dtype=cp_module.bool_)
                self._poisson_timers[ch.name] = cp_module.zeros(self.n_neurons, dtype=cp_module.float32)

    def _resolve_targets(self, channel, group_manager):
        """Resolve a channel's target specification to neuron indices."""
        if channel.target_neuron_indices:
            indices = channel.target_neuron_indices
        elif channel.target_group_name and group_manager:
            group = group_manager.get_group(channel.target_group_name)
            if group:
                indices = group.neuron_indices
            else:
                indices = list(range(self.n_neurons))
        elif channel.target_trait_index >= 0:
            # Will be resolved later when trait info is available
            indices = list(range(self.n_neurons))
        else:
            indices = list(range(self.n_neurons))

        # Apply fraction sampling
        if channel.target_fraction < 1.0 and len(indices) > 0:
            n_select = max(1, int(len(indices) * channel.target_fraction))
            import random as py_random
            indices = sorted(py_random.sample(indices, n_select))

        return indices

    def compute_step_current(self, current_time_ms, phase_start_ms, cp_module):
        """Compute total stimulus current for the current timestep.

        Args:
            current_time_ms: Absolute simulation time
            phase_start_ms: Start time of current experiment phase
            cp_module: CuPy module reference

        Returns:
            cp array of shape [n_neurons] with stimulus current in pA
        """
        self.cp_stimulus_current[:] = 0.0

        for ch in self.channels:
            if not ch.enabled:
                continue  # Skip channels disabled by current experiment phase

            mask = self._channel_target_masks.get(ch.name)
            if mask is None:
                continue

            # Check timing (relative to phase start, with optional trial repetition)
            t_rel = current_time_ms - phase_start_ms
            if ch.repeat_period_ms > 0:
                # Wrap time within trial period for repeating stimuli
                t_rel = t_rel % ch.repeat_period_ms
            if t_rel < ch.onset_ms or t_rel >= (ch.onset_ms + ch.duration_ms):
                continue

            t_in_stim = t_rel - ch.onset_ms  # Time since stimulus onset

            # Generate current based on pattern type
            current = self._compute_pattern(ch, t_in_stim, mask, cp_module)

            # Apply to target neurons
            self.cp_stimulus_current += current * mask.astype(cp_module.float32)

        return self.cp_stimulus_current

    def _compute_pattern(self, channel, t_ms, mask, cp_module):
        """Compute current value for a single channel at time t_ms."""
        p = channel.pattern

        if p.pattern_type == StimulusPatternType.CONSTANT.name:
            return cp_module.float32(p.amplitude_pA)

        elif p.pattern_type == StimulusPatternType.PULSE_TRAIN.name:
            period_ms = 1000.0 / max(p.pulse_frequency_hz, 0.01)
            t_in_period = t_ms % period_ms
            is_on = t_in_period < p.pulse_duration_ms
            return cp_module.float32(p.amplitude_pA * float(is_on))

        elif p.pattern_type == StimulusPatternType.SINUSOIDAL.name:
            phase = 2.0 * math.pi * p.frequency_hz * t_ms / 1000.0 + p.phase_offset_rad
            value = p.amplitude_pA * math.sin(phase) + p.dc_offset_pA
            return cp_module.float32(value)

        elif p.pattern_type == StimulusPatternType.RAMP.name:
            fraction = min(1.0, t_ms / max(channel.duration_ms, 0.001))
            value = p.start_amplitude_pA + fraction * (p.end_amplitude_pA - p.start_amplitude_pA)
            return cp_module.float32(value)

        elif p.pattern_type == StimulusPatternType.POISSON_SPIKE_TRAIN.name:
            # Poisson process: probability of spike in dt
            p_spike = p.poisson_rate_hz * self.dt_ms / 1000.0
            n_targets = int(cp_module.sum(mask).get())

            # Decrement active spike timers
            timers = self._poisson_timers.get(channel.name)
            if timers is not None:
                timers -= self.dt_ms
                timers_clipped = cp_module.maximum(timers, cp_module.float32(0.0))
                self._poisson_timers[channel.name] = timers_clipped

                # New spikes where timer has expired
                new_spikes = (self._rng.random(self.n_neurons) < p_spike) & mask & (timers_clipped <= 0)
                self._poisson_timers[channel.name] = cp_module.where(
                    new_spikes, cp_module.float32(p.spike_duration_ms), timers_clipped
                )

                # Current is applied where timer > 0
                is_active = self._poisson_timers[channel.name] > 0
                return cp_module.where(is_active, cp_module.float32(p.spike_current_pA), cp_module.float32(0.0))

            return cp_module.float32(0.0)

        elif p.pattern_type == StimulusPatternType.RATE_VECTOR_POISSON.name:
            # Per-neuron Poisson rate. Each target neuron draws Bernoulli(rate * dt/1000).
            rate_vec = self._poisson_rate_vectors.get(channel.name)
            timers = self._poisson_timers.get(channel.name)
            if rate_vec is None or timers is None:
                return cp_module.float32(0.0)

            # Decrement timers, clamp to 0
            timers = timers - self.dt_ms
            timers_clipped = cp_module.maximum(timers, cp_module.float32(0.0))

            # Per-neuron spike probability for this dt
            p_spike = rate_vec * (self.dt_ms / 1000.0)
            draws = self._rng.random(self.n_neurons)
            new_spikes = (draws < p_spike) & mask & (timers_clipped <= 0)

            # Load new spike duration where a spike just fired
            timers_next = cp_module.where(
                new_spikes, cp_module.float32(p.spike_duration_ms), timers_clipped
            )
            self._poisson_timers[channel.name] = timers_next

            is_active = timers_next > 0
            return cp_module.where(
                is_active, cp_module.float32(p.spike_current_pA), cp_module.float32(0.0)
            )

        elif p.pattern_type == StimulusPatternType.GAUSSIAN_NOISE.name:
            noise = self._rng.randn(self.n_neurons).astype(cp_module.float32) * p.noise_std_pA + p.noise_mean_pA
            return noise

        elif p.pattern_type == StimulusPatternType.CUSTOM_WAVEFORM.name:
            if len(p.custom_waveform_times_ms) < 2:
                return cp_module.float32(0.0)
            # Linear interpolation of custom waveform
            import numpy as np_interp_helper
            value = float(np_interp_helper.interp(t_ms, p.custom_waveform_times_ms, p.custom_waveform_values_pA))
            return cp_module.float32(value)

        return cp_module.float32(0.0)

    def cleanup(self):
        """Release GPU memory."""
        self.cp_stimulus_current = None
        self._channel_target_masks.clear()
        self._poisson_active.clear()
        self._poisson_timers.clear()
        self._poisson_rate_vectors.clear()
