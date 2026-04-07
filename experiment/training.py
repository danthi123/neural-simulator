"""Training protocol engine for experiments.

Executes training protocols: associative pairing, RL, supervised, reservoir.
Coordinates stimulus timing, response measurement, and weight modification
signals across trials.
"""

import numpy as np

from sim.config import TrainingConfig
from sim.enums import TrainingMode, NeuronGroupRole


class TrainingProtocolEngine:
    """Executes training protocols: associative pairing, RL, supervised, reservoir.

    Coordinates stimulus timing, response measurement, and weight modification
    signals across trials. Works with the existing reward modulation and STDP
    systems rather than replacing them.
    """

    def __init__(self, dt_ms):
        self.dt_ms = dt_ms
        self.config = TrainingConfig()
        self.readout = None
        self.group_manager = None

        # Trial state
        self.current_trial = 0
        self.trial_start_ms = 0.0
        self.trial_phase = "idle"       # idle, stimulus, eval, reward, iti
        self.trials_data = []            # Per-trial performance metrics

        # Reservoir readout weights (CPU numpy for simplicity)
        self._readout_weights = None     # [n_output, n_reservoir]
        self._readout_bias = None        # [n_output]

        # Performance tracking
        self.recent_accuracy = 0.0
        self.is_converged = False

    def initialize(self, config, readout, group_manager):
        """Set up training protocol.

        Args:
            config: TrainingConfig
            readout: ReadoutEngine
            group_manager: NeuronGroupManager
        """
        self.config = config
        self.readout = readout
        self.group_manager = group_manager
        self.current_trial = 0
        self.trial_start_ms = 0.0
        self.trial_phase = "idle"
        self.trials_data = []
        self.recent_accuracy = 0.0
        self.is_converged = False

        # Initialize reservoir readout weights if needed
        if config.mode == TrainingMode.RESERVOIR_READOUT.name:
            output_groups = group_manager.get_groups_by_role(NeuronGroupRole.OUTPUT.name)
            hidden_groups = group_manager.get_groups_by_role(NeuronGroupRole.HIDDEN.name)

            n_output = sum(len(g.neuron_indices) for g in output_groups)
            n_reservoir = sum(len(g.neuron_indices) for g in hidden_groups)

            if n_output > 0 and n_reservoir > 0:
                self._readout_weights = np.zeros((n_output, n_reservoir), dtype=np.float32)
                self._readout_bias = np.zeros(n_output, dtype=np.float32)

    def update(self, current_time_ms, sim_bridge_ref):
        """Per-step training protocol update.

        Called every simulation step. Manages trial state machine and
        generates reward/error signals at appropriate times.

        Args:
            current_time_ms: Absolute simulation time
            sim_bridge_ref: Reference to SimulationBridge for setting reward signal

        Returns:
            dict with training state info for logging/UI
        """
        if self.config.mode == TrainingMode.NONE.name:
            return {"mode": "none"}

        if self.is_converged:
            return {"mode": self.config.mode, "converged": True, "trial": self.current_trial}

        if self.current_trial >= self.config.num_trials:
            return {"mode": self.config.mode, "completed": True, "trial": self.current_trial}

        t_in_trial = current_time_ms - self.trial_start_ms
        trial_total_ms = self.config.trial_duration_ms + self.config.inter_trial_interval_ms

        # Trial state machine
        if self.trial_phase == "idle":
            self.trial_phase = "stimulus"
            self.trial_start_ms = current_time_ms
            t_in_trial = 0.0

        if t_in_trial >= trial_total_ms:
            # Trial complete — advance to next trial
            self._end_trial(current_time_ms, sim_bridge_ref)
            self.current_trial += 1
            self.trial_start_ms = current_time_ms
            self.trial_phase = "stimulus"
            t_in_trial = 0.0

            # Check convergence
            if len(self.trials_data) >= 10:
                recent = self.trials_data[-10:]
                self.recent_accuracy = sum(1 for t in recent if t.get("success", False)) / len(recent)
                if self.recent_accuracy >= self.config.success_threshold:
                    self.is_converged = True

        # Evaluation window
        if (t_in_trial >= self.config.eval_delay_ms and
            t_in_trial < self.config.eval_delay_ms + self.config.eval_window_ms):
            self.trial_phase = "eval"

        # Reward delivery (for RL mode)
        if (self.config.mode == TrainingMode.REINFORCEMENT_LEARNING.name and
            self.trial_phase == "eval" and
            t_in_trial >= self.config.eval_delay_ms + self.config.eval_window_ms):
            self._deliver_reward(sim_bridge_ref)
            self.trial_phase = "iti"

        # Supervised error signal (continuous during stimulus)
        if (self.config.mode == TrainingMode.SUPERVISED_TARGET.name and
            t_in_trial < self.config.trial_duration_ms):
            self._apply_supervised_error(sim_bridge_ref)

        # ITI: clear reward signal
        if t_in_trial >= self.config.trial_duration_ms:
            if hasattr(sim_bridge_ref, 'core_config') and sim_bridge_ref.core_config is not None:
                sim_bridge_ref.core_config.current_reward_signal = 0.0

        return {
            "mode": self.config.mode,
            "trial": self.current_trial,
            "total_trials": self.config.num_trials,
            "phase": self.trial_phase,
            "accuracy": self.recent_accuracy,
            "t_in_trial": t_in_trial,
        }

    def _end_trial(self, current_time_ms, sim_bridge_ref):
        """Record trial outcome."""
        snapshot = self.readout.get_trial_snapshot() if self.readout else {}

        trial_data = {
            "trial": self.current_trial,
            "time_ms": current_time_ms,
            "rates": snapshot.get("rates", {}),
            "spike_counts": snapshot.get("spike_counts", {}),
        }

        # Evaluate success based on training mode
        if self.config.mode == TrainingMode.REINFORCEMENT_LEARNING.name:
            target_group = self.config.target_output_group
            rate = snapshot.get("rates", {}).get(target_group, 0.0)
            success = self.config.target_min_rate_hz <= rate <= self.config.target_max_rate_hz
            trial_data["success"] = success
            trial_data["output_rate"] = rate
        elif self.config.mode == TrainingMode.ASSOCIATIVE_PAIRING.name:
            # For associative conditioning: success = US output group rate exceeds CR threshold.
            # NOTE: _end_trial() is called at the end of each trial (during ITI), so the
            # readout rate reflects the post-ITI baseline, NOT the CS-driven response.
            # Per-trial accuracy during training is therefore not meaningful for associative
            # conditioning — the true learning metric is the pre_test vs post_test comparison
            # of output rates during CS-alone presentation. This metric is preserved here for
            # API consistency and may show success in later trials if STDP strengthens CS→US
            # pathways enough that residual activity during ITI exceeds the threshold.
            us_group = self.config.us_channel_name.replace("us", "us_output") if self.config.us_channel_name else "us_output"
            output_rate = 0.0
            for grp_name, rate in snapshot.get("rates", {}).items():
                if "output" in grp_name.lower() or "us" in grp_name.lower():
                    output_rate = rate
                    break
            cr_threshold = getattr(self.config, 'cr_threshold_hz', 8.0)
            trial_data["success"] = output_rate > cr_threshold
            trial_data["output_rate"] = output_rate

        self.trials_data.append(trial_data)

    def _deliver_reward(self, sim_bridge_ref):
        """Deliver reward or punishment based on output group activity."""
        if not hasattr(sim_bridge_ref, 'core_config') or sim_bridge_ref.core_config is None:
            return

        target_group = self.config.target_output_group
        rate = self.readout.current_rates.get(target_group, 0.0) if self.readout else 0.0

        if self.config.target_min_rate_hz <= rate <= self.config.target_max_rate_hz:
            sim_bridge_ref.core_config.current_reward_signal = self.config.reward_magnitude
        else:
            sim_bridge_ref.core_config.current_reward_signal = self.config.punishment_magnitude

    def _apply_supervised_error(self, sim_bridge_ref):
        """Apply supervised error signal as reward modulation.

        Uses the existing reward signal mechanism as an error channel.
        Error = (target_rate - actual_rate) * gain
        """
        if not hasattr(sim_bridge_ref, 'core_config') or sim_bridge_ref.core_config is None:
            return

        total_error = 0.0
        n_groups = 0

        for group_name, target_rate in self.config.target_rates_per_group.items():
            actual_rate = self.readout.current_rates.get(group_name, 0.0) if self.readout else 0.0
            error = target_rate - actual_rate
            total_error += error
            n_groups += 1

        if n_groups > 0:
            mean_error = total_error / n_groups
            sim_bridge_ref.core_config.current_reward_signal = mean_error * self.config.supervised_error_gain

    def get_training_summary(self):
        """Get summary of training progress."""
        return {
            "mode": self.config.mode,
            "trials_completed": self.current_trial,
            "total_trials": self.config.num_trials,
            "recent_accuracy": self.recent_accuracy,
            "is_converged": self.is_converged,
            "trials_data_count": len(self.trials_data),
        }
