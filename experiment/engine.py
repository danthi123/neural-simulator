"""Experiment engine - top-level orchestrator for multi-phase experiments.

Manages phase transitions, stimulus current generation, response measurement,
training protocol execution, and experiment logging.
"""

import json
import dataclasses
from enum import Enum

from sim.config import (ExperimentConfig, ReadoutConfig, TrainingConfig,
                        StimulusPattern, StimulusChannel, NeuronGroup,
                        ExperimentPhase)
from sim.enums import (ExperimentPhaseType, NeuronGroupRole, TrainingMode,
                       StimulusPatternType)

from experiment.groups import NeuronGroupManager
from experiment.stimulus import StimulusManager
from experiment.readout import ReadoutEngine
from experiment.training import TrainingProtocolEngine


class ExperimentEngine:
    """Orchestrates multi-phase experiments with stimulus, training, and readout.

    The engine is called once per simulation step by SimulationBridge.
    It manages:
    1. Phase transitions (baseline -> stimulus -> training -> testing -> rest)
    2. Stimulus current generation via StimulusManager
    3. Response measurement via ReadoutEngine
    4. Training protocol execution via TrainingProtocolEngine
    5. Experiment logging

    Usage:
        engine = ExperimentEngine(n_neurons, dt_ms)
        engine.load_experiment(experiment_config)
        engine.initialize(cp_traits, cp_module)

        # In simulation loop:
        stimulus_current = engine.step(current_time_ms, cp_firing_states, cp_v, sim_bridge, cp)
    """

    def __init__(self, n_neurons, dt_ms):
        self.n_neurons = n_neurons
        self.dt_ms = dt_ms

        self.config = None                # ExperimentConfig
        self.stimulus_manager = StimulusManager(n_neurons, dt_ms)
        self.group_manager = NeuronGroupManager(n_neurons)
        self.readout = ReadoutEngine(n_neurons, dt_ms)
        self.training = TrainingProtocolEngine(dt_ms)

        # Phase management
        self.phases = []                  # List[ExperimentPhase]
        self.current_phase_idx = 0
        self.phase_start_ms = 0.0
        self.phase_repetition = 0
        self.is_experiment_running = False
        self.is_experiment_complete = False

        # Phase-gated plasticity flag (checked by simulation step)
        self.plasticity_enabled_this_phase = True

        # Experiment log
        self.log = []                     # List of timestamped event dicts
        self._log_interval_steps = 100    # Log readout every N steps
        self._step_counter = 0

        # Active stimulus channels for current phase
        self._current_phase_channels = []

    def load_experiment(self, config):
        """Load an experiment configuration.

        Args:
            config: ExperimentConfig dataclass
        """
        self.config = config
        self.phases = list(config.phases)
        self.current_phase_idx = 0
        self.phase_repetition = 0
        self.is_experiment_running = False
        self.is_experiment_complete = False
        self.log = []

    def initialize(self, cp_traits=None, cp_module=None):
        """Initialize all subsystems with GPU arrays.

        Args:
            cp_traits: GPU array of neuron trait indices
            cp_module: CuPy module reference
        """
        if self.config is None:
            return

        # Initialize neuron groups
        self.group_manager = NeuronGroupManager(self.n_neurons)
        self.group_manager.initialize(self.config.neuron_groups, cp_traits, cp_module)

        # Initialize stimulus manager
        self.stimulus_manager = StimulusManager(self.n_neurons, self.dt_ms)
        self.stimulus_manager.initialize(self.config.stimulus_channels, self.group_manager, cp_module)

        # Initialize readout
        self.readout = ReadoutEngine(self.n_neurons, self.dt_ms)
        self.readout.initialize(self.config.readout, self.group_manager, cp_module)

        # Log initialization
        self.log.append({
            "event": "experiment_initialized",
            "groups": self.group_manager.get_summary(),
            "channels": len(self.config.stimulus_channels),
            "phases": len(self.phases),
        })

    def start(self, current_time_ms, sim_bridge_ref=None):
        """Begin experiment execution.

        Args:
            current_time_ms: Absolute simulation time
            sim_bridge_ref: Optional SimulationBridge for applying config overrides
        """
        self.is_experiment_running = True
        self.is_experiment_complete = False
        self.current_phase_idx = 0
        self.phase_repetition = 0
        self.phase_start_ms = current_time_ms
        self._step_counter = 0

        # Apply experiment-level simulation overrides (e.g. boosted propagation_strength)
        self._saved_overrides = {}
        if sim_bridge_ref is not None and self.config is not None:
            cfg = sim_bridge_ref.core_config
            if self.config.override_propagation_strength > 0:
                self._saved_overrides['propagation_strength'] = cfg.propagation_strength
                cfg.propagation_strength = self.config.override_propagation_strength
            if self.config.override_inhibitory_prop_strength > 0:
                self._saved_overrides['inhibitory_propagation_strength'] = cfg.inhibitory_propagation_strength
                cfg.inhibitory_propagation_strength = self.config.override_inhibitory_prop_strength
        self._sim_bridge_for_overrides = sim_bridge_ref

        if self.phases:
            self._enter_phase(self.phases[0], current_time_ms)

        self.log.append({"event": "experiment_started", "time_ms": current_time_ms})

    def stop(self):
        """Stop experiment execution and restore any config overrides."""
        self.is_experiment_running = False

        # Restore overridden config values
        if hasattr(self, '_saved_overrides') and self._saved_overrides:
            bridge = getattr(self, '_sim_bridge_for_overrides', None)
            if bridge is not None:
                for key, val in self._saved_overrides.items():
                    setattr(bridge.core_config, key, val)
            self._saved_overrides = {}

        self.log.append({"event": "experiment_stopped"})

    def step(self, current_time_ms, cp_firing_states, cp_membrane_potential_v, sim_bridge_ref, cp_module):
        """Execute one experiment step.

        Called every simulation step. Returns stimulus current array.

        Args:
            current_time_ms: Absolute simulation time
            cp_firing_states: GPU bool array [n_neurons]
            cp_membrane_potential_v: GPU float32 array [n_neurons]
            sim_bridge_ref: Reference to SimulationBridge
            cp_module: CuPy module

        Returns:
            cp array [n_neurons] with stimulus current in pA (zeros if no stimulus)
        """
        if not self.is_experiment_running or self.is_experiment_complete:
            return cp_module.zeros(self.n_neurons, dtype=cp_module.float32)

        # Store refs for use in phase transitions (weight diagnostics)
        self._sim_bridge_ref = sim_bridge_ref
        self._cp_module = cp_module

        # Check phase transition
        self._check_phase_transition(current_time_ms)

        # Update readout
        self.readout.update(cp_firing_states, cp_membrane_potential_v, cp_module)

        # Update training protocol
        if self.phases and self.current_phase_idx < len(self.phases):
            current_phase = self.phases[self.current_phase_idx]
            if current_phase.phase_type == ExperimentPhaseType.TRAINING.name:
                self.training.update(current_time_ms, sim_bridge_ref)

        # Compute stimulus current
        stimulus_current = self.stimulus_manager.compute_step_current(
            current_time_ms, self.phase_start_ms, cp_module
        )

        # Periodic logging
        self._step_counter += 1
        if self._step_counter % self._log_interval_steps == 0:
            self._log_step(current_time_ms)

        return stimulus_current

    def _check_phase_transition(self, current_time_ms):
        """Check if current phase has ended and transition to next."""
        if not self.phases or self.current_phase_idx >= len(self.phases):
            self.is_experiment_complete = True
            self.is_experiment_running = False
            self.log.append({"event": "experiment_complete", "time_ms": current_time_ms})
            return

        current_phase = self.phases[self.current_phase_idx]
        elapsed = current_time_ms - self.phase_start_ms

        if elapsed >= current_phase.duration_ms:
            self.phase_repetition += 1

            if self.phase_repetition < current_phase.num_repetitions:
                # Repeat current phase
                self.phase_start_ms = current_time_ms
                self.log.append({
                    "event": "phase_repeat",
                    "phase": current_phase.name,
                    "repetition": self.phase_repetition,
                    "time_ms": current_time_ms,
                })
            else:
                # Move to next phase
                self.current_phase_idx += 1
                self.phase_repetition = 0
                self.phase_start_ms = current_time_ms

                if self.current_phase_idx < len(self.phases):
                    self._enter_phase(self.phases[self.current_phase_idx], current_time_ms)
                else:
                    self.is_experiment_complete = True
                    self.is_experiment_running = False
                    self.log.append({"event": "experiment_complete", "time_ms": current_time_ms})

    def _log_intergroup_weights(self, sim_bridge_ref, cp_module, label=""):
        """Log mean weight of inter-group (INPUT->OUTPUT) connections for diagnostics."""
        try:
            input_groups = self.group_manager.get_groups_by_role(NeuronGroupRole.INPUT.name)
            output_groups = self.group_manager.get_groups_by_role(NeuronGroupRole.OUTPUT.name)
            if not input_groups or not output_groups or sim_bridge_ref is None:
                return
            coo = sim_bridge_ref._get_cached_coo()
            if coo is None:
                return
            for in_grp in input_groups:
                for out_grp in output_groups:
                    in_mask = (coo.row >= in_grp.index_start) & (coo.row < in_grp.index_end)
                    out_mask = (coo.col >= out_grp.index_start) & (coo.col < out_grp.index_end)
                    inter_mask = in_mask & out_mask
                    n_inter = int(cp_module.sum(inter_mask))
                    if n_inter > 0:
                        weights = sim_bridge_ref.cp_connections.data[
                            cp_module.where(inter_mask)[0]  # need global indices in data array
                        ] if hasattr(coo, 'row') else None
                        # For COO, the indices map directly to data array
                        inter_indices = cp_module.where(inter_mask)[0]
                        inter_weights = coo.data[inter_indices]
                        mean_w = float(cp_module.mean(inter_weights))
                        std_w = float(cp_module.std(inter_weights))
                        max_w = float(cp_module.max(inter_weights))
                        min_w = float(cp_module.min(inter_weights))
                        self.log.append({
                            "event": "intergroup_weights",
                            "label": label,
                            "from_group": in_grp.name,
                            "to_group": out_grp.name,
                            "n_connections": n_inter,
                            "mean_weight": round(mean_w, 6),
                            "std_weight": round(std_w, 6),
                            "min_weight": round(min_w, 6),
                            "max_weight": round(max_w, 6),
                        })
        except Exception as e:
            self.log.append({"event": "intergroup_weights_error", "error": str(e)})

    def _log_band_power(self, label, current_time_ms):
        """Log spectral band power for all readout groups if PSD is enabled."""
        if not self.readout.config.enable_psd:
            return
        if not hasattr(self, '_cp_module') or self._cp_module is None:
            return
        try:
            for gname in self.readout.config.rate_group_names:
                bp = self.readout.compute_band_power(gname, self._cp_module)
                if bp is not None:
                    self.log.append({
                        "event": "band_power",
                        "label": label,
                        "group": gname,
                        "time_ms": current_time_ms,
                        **bp,
                    })
        except Exception as e:
            self.log.append({"event": "band_power_error", "error": str(e)})

    def _enter_phase(self, phase, current_time_ms):
        """Set up a new experiment phase."""
        self.log.append({
            "event": "phase_entered",
            "phase": phase.name,
            "type": phase.phase_type,
            "time_ms": current_time_ms,
        })

        # Log inter-group weights at phase transitions for learning diagnostics
        if hasattr(self, '_sim_bridge_ref') and hasattr(self, '_cp_module'):
            self._log_intergroup_weights(self._sim_bridge_ref, self._cp_module,
                                          label=f"entering_{phase.name}")
            # Log spectral band power if PSD is enabled
            self._log_band_power(f"entering_{phase.name}", current_time_ms)

        # Gate plasticity based on phase setting (checked by simulation step)
        self.plasticity_enabled_this_phase = phase.enable_plasticity

        # Configure active stimulus channels.
        # None = all channels enabled (default); [] = no channels (baseline/rest).
        for ch in self.stimulus_manager.channels:
            if phase.active_channels is None:
                ch.enabled = True
            else:
                ch.enabled = (ch.name in phase.active_channels)

        # Configure training if this is a training phase
        if phase.phase_type == ExperimentPhaseType.TRAINING.name:
            self.training.initialize(
                phase.training_config, self.readout, self.group_manager
            )
            self.training.trial_phase = "idle"
            self.training.current_trial = 0
            self.training.trial_start_ms = current_time_ms

    def _log_step(self, current_time_ms):
        """Log periodic readout data."""
        if not self.config or not self.config.save_experiment_log:
            return

        entry = {
            "event": "readout",
            "time_ms": current_time_ms,
            "rates": dict(self.readout.current_rates),
            "spike_counts": dict(self.readout.current_spike_counts),
            "synchrony": dict(self.readout.current_synchrony),
        }

        if self.phases and self.current_phase_idx < len(self.phases):
            entry["phase"] = self.phases[self.current_phase_idx].name
            entry["phase_type"] = self.phases[self.current_phase_idx].phase_type

        training_state = self.training.get_training_summary()
        if training_state["mode"] != TrainingMode.NONE.name:
            entry["training"] = training_state

        self.log.append(entry)

    def get_experiment_status(self):
        """Get current experiment status for UI display."""
        status = {
            "is_running": self.is_experiment_running,
            "is_complete": self.is_experiment_complete,
            "current_phase_idx": self.current_phase_idx,
            "total_phases": len(self.phases),
            "readout_rates": dict(self.readout.current_rates),
            "readout_spike_counts": dict(self.readout.current_spike_counts),
        }

        if self.phases and self.current_phase_idx < len(self.phases):
            phase = self.phases[self.current_phase_idx]
            status["current_phase_name"] = phase.name
            status["current_phase_type"] = phase.phase_type
            status["phase_repetition"] = self.phase_repetition

        training_state = self.training.get_training_summary()
        if training_state["mode"] != TrainingMode.NONE.name:
            status["training"] = training_state

        return status

    def save_log(self, filepath):
        """Save experiment log to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                "experiment_name": self.config.name if self.config else "Unknown",
                "description": self.config.description if self.config else "",
                "groups": self.group_manager.get_summary() if self.group_manager else {},
                "training_summary": self.training.get_training_summary(),
                "log_entries": self.log,
                "trial_data": self.training.trials_data,
            }, f, indent=2, default=str)

    def ensure_inter_group_connectivity(self, sim_bridge, cp_module, min_connection_prob=0.95):
        """Ensure sufficient synaptic connections exist between INPUT and OUTPUT groups.

        For associative conditioning to work via STDP, there must be dense direct synaptic
        paths from CS (input) to US (output) neurons. In noise-dominated networks (where OU
        noise sigma ~ 80 pA dwarfs individual synaptic currents of ~3 pA), learning-induced weight
        changes are only detectable if enough connections exist. With propagation_strength=0.05,
        each synapse at weight=1.0 contributes ~3 pA of driving current. To produce a ~25 pA
        shift (detectable above noise), ~80 connections per output neuron are needed from a
        100-neuron input group (hence default 80% connection probability).

        Args:
            sim_bridge: SimulationBridge instance (for cp_connections access)
            cp_module: CuPy module reference
            min_connection_prob: Minimum connection probability between groups (default 80%)
        """
        import cupyx.scipy.sparse as csp_local

        input_groups = self.group_manager.get_groups_by_role(NeuronGroupRole.INPUT.name)
        output_groups = self.group_manager.get_groups_by_role(NeuronGroupRole.OUTPUT.name)

        if not input_groups or not output_groups:
            return 0

        total_added = 0
        for in_grp in input_groups:
            for out_grp in output_groups:
                in_indices = cp_module.arange(in_grp.index_start, in_grp.index_end)
                out_indices = cp_module.arange(out_grp.index_start, out_grp.index_end)

                # Count existing connections from input to output
                coo = sim_bridge._get_cached_coo()
                if coo is not None:
                    in_set = set(range(in_grp.index_start, in_grp.index_end))
                    out_set = set(range(out_grp.index_start, out_grp.index_end))
                    existing = sum(1 for r, c in zip(
                        cp_module.asnumpy(coo.row), cp_module.asnumpy(coo.col)
                    ) if int(r) in in_set and int(c) in out_set)
                else:
                    existing = 0

                n_in = in_grp.index_end - in_grp.index_start
                n_out = out_grp.index_end - out_grp.index_start
                target_connections = int(n_in * n_out * min_connection_prob)

                if existing >= target_connections:
                    continue  # Already have enough connections

                n_to_add = target_connections - existing

                # Filter to excitatory presynaptic neurons only: inhibitory CS neurons
                # with inhibitory_propagation_strength (0.105, 2.1x excitatory) create
                # strong opposing currents that cancel excitatory CS->US drive. With 20%
                # inhibitory neurons in a cortical profile, STDP potentiates both exc and
                # inh CS->US connections equally, but the inh pathways suppress the
                # conditioned response in post-test. Restricting to excitatory pre-synaptic
                # neurons ensures the learned pathway is purely excitatory.
                exc_neuron_ids = None
                if sim_bridge.cp_traits is not None:
                    inhibitory_idx = getattr(sim_bridge.core_config, 'inhibitory_trait_index', -1)
                    if inhibitory_idx >= 0:
                        in_range_traits = sim_bridge.cp_traits[in_grp.index_start:in_grp.index_end]
                        exc_mask = (in_range_traits != inhibitory_idx)
                        exc_neuron_ids = cp_module.arange(in_grp.index_start, in_grp.index_end)[exc_mask]
                        n_exc = int(exc_neuron_ids.size)
                        if n_exc > 0:
                            # Adjust target for excitatory-only source pool
                            target_connections = int(n_exc * n_out * min_connection_prob)
                            n_to_add = max(0, target_connections - existing)

                # Generate random input->output connections
                if exc_neuron_ids is not None and exc_neuron_ids.size > 0:
                    # Sample from excitatory neurons only
                    rand_indices = cp_module.random.randint(0, exc_neuron_ids.size,
                                                            size=n_to_add * 3, dtype=cp_module.int32)
                    pre_idx = exc_neuron_ids[rand_indices].astype(cp_module.int32)
                else:
                    pre_idx = cp_module.random.randint(in_grp.index_start, in_grp.index_end,
                                                        size=n_to_add * 3, dtype=cp_module.int32)
                post_idx = cp_module.random.randint(out_grp.index_start, out_grp.index_end,
                                                     size=n_to_add * 3, dtype=cp_module.int32)

                # Remove duplicates
                pair_ids = pre_idx.astype(cp_module.int64) * self.n_neurons + post_idx.astype(cp_module.int64)
                unique_ids, unique_indices = cp_module.unique(pair_ids, return_index=True)
                if unique_ids.size > n_to_add:
                    unique_indices = unique_indices[:n_to_add]
                    unique_ids = unique_ids[:n_to_add]

                new_pre = (unique_ids // self.n_neurons).astype(cp_module.int32)
                new_post = (unique_ids % self.n_neurons).astype(cp_module.int32)

                if new_pre.size > 0:
                    # Initial weight 0.1: low enough that STDP potentiation to ~0.99 produces
                    # a ~10x increase in synaptic drive (strongly detectable rate change).
                    # Must stay above structural plasticity pruning threshold.
                    initial_weights = cp_module.full(new_pre.size, 0.1, dtype=cp_module.float32)

                    new_matrix = csp_local.csr_matrix(
                        (initial_weights, (new_pre, new_post)),
                        shape=sim_bridge.cp_connections.shape,
                        dtype=cp_module.float32
                    )

                    nnz_before = sim_bridge.cp_connections.nnz
                    sim_bridge.cp_connections = sim_bridge.cp_connections + new_matrix
                    actual_new = sim_bridge.cp_connections.nnz - nnz_before

                    if actual_new > 0:
                        sim_bridge._invalidate_coo_cache()
                        sim_bridge._grow_synapse_arrays_if_needed(actual_new, sim_bridge.core_config)
                        sim_bridge._add_synapses_to_arrays(actual_new, sim_bridge.core_config)
                        sim_bridge._synapse_count = sim_bridge.cp_connections.nnz
                        total_added += actual_new

        return total_added

    def cleanup(self):
        """Release all GPU resources."""
        self.stimulus_manager.cleanup()
        self.readout.cleanup()
        self.is_experiment_running = False


# --- JSON Serialization for Experiment Configs ---

def experiment_config_to_dict(config):
    """Serialize an ExperimentConfig to a JSON-safe dictionary."""

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            d = {}
            for f in dataclasses.fields(obj):
                val = getattr(obj, f.name)
                d[f.name] = _to_dict(val)
            return d
        elif isinstance(obj, list):
            return [_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, Enum):
            return obj.name
        else:
            return obj

    return _to_dict(config)


def experiment_config_from_dict(d):
    """Deserialize an ExperimentConfig from a dictionary."""

    def _build_pattern(pd):
        if pd is None:
            return StimulusPattern()
        return StimulusPattern(**{k: v for k, v in pd.items()})

    def _build_channel(cd):
        cd = dict(cd)
        if 'pattern' in cd:
            cd['pattern'] = _build_pattern(cd['pattern'])
        return StimulusChannel(**cd)

    def _build_group(gd):
        return NeuronGroup(**gd)

    def _build_readout(rd):
        return ReadoutConfig(**rd)

    def _build_training(td):
        return TrainingConfig(**td)

    def _build_phase(pd):
        pd = dict(pd)
        if 'training_config' in pd:
            pd['training_config'] = _build_training(pd['training_config'])
        return ExperimentPhase(**pd)

    d = dict(d)
    if 'neuron_groups' in d:
        d['neuron_groups'] = [_build_group(g) for g in d['neuron_groups']]
    if 'stimulus_channels' in d:
        d['stimulus_channels'] = [_build_channel(c) for c in d['stimulus_channels']]
    if 'phases' in d:
        d['phases'] = [_build_phase(p) for p in d['phases']]
    if 'readout' in d:
        d['readout'] = _build_readout(d['readout'])

    return ExperimentConfig(**d)
