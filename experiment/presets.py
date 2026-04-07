"""Experiment preset definitions.

Pre-built experiment configurations for common neuroscience paradigms.
"""

import math

from sim.config import (ExperimentConfig, ExperimentPhase, StimulusChannel,
                        StimulusPattern, NeuronGroup, ReadoutConfig, TrainingConfig)
from sim.enums import (StimulusPatternType, NeuronGroupRole, ExperimentPhaseType,
                       TrainingMode)


class ExperimentPresets:
    """Factory for common experiment configurations.

    Each preset returns a fully configured ExperimentConfig that can be
    loaded directly or customized before use.
    """

    @staticmethod
    def basic_stimulus_response(input_amplitude_pA=150.0, stimulus_duration_ms=500.0,
                                 num_trials=20, input_group_size=100, output_group_size=100):
        """Basic stimulus-response: inject current into input group, measure output.

        Good for characterizing network transfer functions and I/O mapping.
        """
        return ExperimentConfig(
            name="Basic Stimulus-Response",
            description="Inject constant current into input group, measure output group firing rate.",
            neuron_groups=[
                NeuronGroup(name="input", role=NeuronGroupRole.INPUT.name,
                           index_start=0, index_end=input_group_size,
                           highlight_color=[0.0, 1.0, 0.0, 1.0]),
                NeuronGroup(name="output", role=NeuronGroupRole.OUTPUT.name,
                           index_start=input_group_size, index_end=input_group_size + output_group_size,
                           highlight_color=[1.0, 0.0, 0.0, 1.0]),
            ],
            stimulus_channels=[
                StimulusChannel(
                    name="input_drive",
                    pattern=StimulusPattern(
                        pattern_type=StimulusPatternType.CONSTANT.name,
                        amplitude_pA=input_amplitude_pA,
                    ),
                    target_group_name="input",
                    onset_ms=100.0,
                    duration_ms=stimulus_duration_ms,
                ),
            ],
            phases=[
                ExperimentPhase(name="baseline", phase_type=ExperimentPhaseType.BASELINE.name,
                               duration_ms=2000.0, active_channels=[]),
                ExperimentPhase(name="stimulus", phase_type=ExperimentPhaseType.STIMULUS.name,
                               duration_ms=stimulus_duration_ms + 200.0,
                               active_channels=["input_drive"],
                               num_repetitions=num_trials),
                ExperimentPhase(name="post", phase_type=ExperimentPhaseType.BASELINE.name,
                               duration_ms=2000.0, active_channels=[]),
            ],
            readout=ReadoutConfig(
                rate_window_ms=50.0,
                rate_group_names=["input", "output"],
                spike_count_window_ms=100.0,
            ),
            enabled=True,
        )

    @staticmethod
    def associative_conditioning(cs_amplitude_pA=500.0, us_amplitude_pA=500.0,
                                  cs_us_delay_ms=100.0, num_trials=100,
                                  input_group_size=100, output_group_size=100):
        """Classical conditioning: pair CS (input) with US (output), test if CS alone evokes response.

        Based on Pavlovian conditioning with STDP as the learning mechanism.
        The CS-US delay determines the temporal window for STDP potentiation.

        High stimulus amplitudes (500 pA) ensure CS neurons fire at 30-40 Hz,
        needed because each synapse at propagation_strength=0.05 provides only
        ~3 pA per spike. Dense connectivity (80% via ensure_inter_group_connectivity)
        gives ~80 CS->US paths per output neuron for adequate signal vs OU noise.
        """
        return ExperimentConfig(
            name="Associative Conditioning (CS-US Pairing)",
            description="Pavlovian conditioning: repeated CS-US pairing followed by CS-alone testing.",
            # Boost propagation_strength during experiment: default 0.05 gives only
            # ~3 pA per synapse against OU noise sigma=100 pA (SNR ~ 0.27). Doubling
            # to 0.10 yields ~6 pA/synapse and ~54 pA total CS->US drive, producing
            # a detectable ~9 Hz conditioned response in post-test.
            override_propagation_strength=0.10,
            override_inhibitory_prop_strength=0.21,
            neuron_groups=[
                NeuronGroup(name="cs_input", role=NeuronGroupRole.INPUT.name,
                           index_start=0, index_end=input_group_size,
                           highlight_color=[0.0, 1.0, 0.0, 1.0]),
                NeuronGroup(name="us_output", role=NeuronGroupRole.OUTPUT.name,
                           index_start=input_group_size, index_end=input_group_size + output_group_size,
                           highlight_color=[1.0, 0.0, 0.0, 1.0]),
            ],
            stimulus_channels=[
                StimulusChannel(
                    name="cs",
                    pattern=StimulusPattern(
                        pattern_type=StimulusPatternType.PULSE_TRAIN.name,
                        amplitude_pA=cs_amplitude_pA,
                        pulse_frequency_hz=40.0,
                        pulse_duration_ms=5.0,
                    ),
                    target_group_name="cs_input",
                    onset_ms=0.0,
                    duration_ms=200.0,
                    repeat_period_ms=500.0,  # Repeat per trial (400ms stim + 100ms ITI)
                ),
                StimulusChannel(
                    name="us",
                    pattern=StimulusPattern(
                        pattern_type=StimulusPatternType.CONSTANT.name,
                        amplitude_pA=us_amplitude_pA,
                    ),
                    target_group_name="us_output",
                    onset_ms=cs_us_delay_ms,
                    duration_ms=100.0,
                    repeat_period_ms=500.0,  # Repeat per trial
                ),
            ],
            phases=[
                # Pre-training baseline: CS alone (5 presentations)
                ExperimentPhase(name="pre_test", phase_type=ExperimentPhaseType.TESTING.name,
                               duration_ms=500.0, active_channels=["cs"],
                               enable_plasticity=False, num_repetitions=5),
                # Training: CS + US paired — single long phase, trial engine manages repetitions
                ExperimentPhase(name="training", phase_type=ExperimentPhaseType.TRAINING.name,
                               duration_ms=num_trials * 500.0,  # 500ms per trial (400 stim + 100 ITI)
                               active_channels=["cs", "us"],
                               training_config=TrainingConfig(
                                   mode=TrainingMode.ASSOCIATIVE_PAIRING.name,
                                   num_trials=num_trials,
                                   trial_duration_ms=400.0,
                                   inter_trial_interval_ms=100.0,
                                   cs_channel_name="cs",
                                   us_channel_name="us",
                                   cs_us_delay_ms=cs_us_delay_ms,
                               ),
                               num_repetitions=1),
                # Post-training test: CS alone (US disabled, 10 presentations)
                ExperimentPhase(name="post_test", phase_type=ExperimentPhaseType.TESTING.name,
                               duration_ms=500.0, active_channels=["cs"],
                               enable_plasticity=False, num_repetitions=10),
            ],
            readout=ReadoutConfig(
                rate_window_ms=50.0,
                rate_group_names=["cs_input", "us_output"],
            ),
            enabled=True,
        )

    @staticmethod
    def reinforcement_learning(stimulus_amplitude_pA=400.0, num_trials=200,
                                input_group_size=100, output_group_size=50):
        """Reward-modulated STDP training: stimulus -> response -> reward/punishment.

        Based on three-factor learning rule (Izhikevich 2007, Fremaux et al. 2013).
        Uses the existing eligibility trace and reward modulation infrastructure.

        The target window must be achievable from spontaneous baseline (~5-8 Hz)
        so that random fluctuations occasionally reach the rewarded zone, allowing
        the RL mechanism to bootstrap learning (operant conditioning analogy:
        the animal must sometimes perform the desired behavior by chance).
        """
        return ExperimentConfig(
            name="Reinforcement Learning (R-STDP)",
            description="Three-factor learning: stimulus evokes response, reward/punishment shapes connections.",
            # Boost propagation for experiment signal-to-noise (same rationale as associative)
            override_propagation_strength=0.10,
            override_inhibitory_prop_strength=0.21,
            neuron_groups=[
                NeuronGroup(name="stimulus", role=NeuronGroupRole.INPUT.name,
                           index_start=0, index_end=input_group_size,
                           highlight_color=[0.0, 1.0, 0.5, 1.0]),
                NeuronGroup(name="response", role=NeuronGroupRole.OUTPUT.name,
                           index_start=input_group_size, index_end=input_group_size + output_group_size,
                           highlight_color=[1.0, 0.5, 0.0, 1.0]),
            ],
            stimulus_channels=[
                StimulusChannel(
                    name="input_pattern",
                    pattern=StimulusPattern(
                        pattern_type=StimulusPatternType.POISSON_SPIKE_TRAIN.name,
                        poisson_rate_hz=50.0,
                        spike_current_pA=stimulus_amplitude_pA,
                        spike_duration_ms=1.0,
                    ),
                    target_group_name="stimulus",
                    onset_ms=0.0,
                    duration_ms=300.0,
                    repeat_period_ms=600.0,  # Repeat per trial (400ms stim + 200ms ITI)
                ),
            ],
            phases=[
                ExperimentPhase(name="baseline", phase_type=ExperimentPhaseType.BASELINE.name,
                               duration_ms=3000.0),
                ExperimentPhase(name="rl_training", phase_type=ExperimentPhaseType.TRAINING.name,
                               duration_ms=num_trials * 600.0,  # 600ms per trial (400 stim + 200 ITI)
                               active_channels=["input_pattern"],
                               training_config=TrainingConfig(
                                   mode=TrainingMode.REINFORCEMENT_LEARNING.name,
                                   num_trials=num_trials,
                                   trial_duration_ms=400.0,
                                   inter_trial_interval_ms=200.0,
                                   reward_delay_ms=50.0,
                                   reward_magnitude=1.0,
                                   # No punishment: dopaminergic RPE is asymmetric
                                   # (Schultz 2002) — tonic DA maintains connections,
                                   # phasic dips are weaker than phasic bursts. Pure
                                   # punishment creates a negative spiral where failed
                                   # trials weaken pathways, making future success harder.
                                   punishment_magnitude=0.0,
                                   target_output_group="response",
                                   # Target window: must overlap the upper tail of
                                   # spontaneous fluctuations (~5-8 Hz +/- 2-3 Hz) so
                                   # ~10-20% of trials succeed by chance, bootstrapping
                                   # the reward signal for three-factor learning.
                                   target_min_rate_hz=8.0,
                                   target_max_rate_hz=30.0,
                                   eval_delay_ms=100.0,
                                   eval_window_ms=200.0,
                               ),
                               num_repetitions=1),
                ExperimentPhase(name="post_test", phase_type=ExperimentPhaseType.TESTING.name,
                               duration_ms=600.0,
                               active_channels=["input_pattern"],
                               enable_plasticity=False,
                               num_repetitions=20),
            ],
            readout=ReadoutConfig(
                rate_window_ms=50.0,
                rate_group_names=["stimulus", "response"],
            ),
            enabled=True,
        )

    @staticmethod
    def frequency_response_characterization(freq_start_hz=1.0, freq_end_hz=100.0,
                                             num_frequencies=20, duration_per_freq_ms=2000.0,
                                             amplitude_pA=100.0, input_group_size=200):
        """Characterize network frequency response with sinusoidal stimulation.

        Sweeps through frequencies to measure how the network filters/transforms
        oscillatory input — reveals resonance frequencies and bandpass properties.
        """
        channels = []
        phases = [
            ExperimentPhase(name="baseline", phase_type=ExperimentPhaseType.BASELINE.name,
                           duration_ms=3000.0, active_channels=[]),
        ]

        # Generate log-spaced frequencies
        log_start = math.log10(max(freq_start_hz, 0.1))
        log_end = math.log10(max(freq_end_hz, 1.0))

        for i in range(num_frequencies):
            frac = i / max(num_frequencies - 1, 1)
            freq = 10 ** (log_start + frac * (log_end - log_start))

            ch_name = f"sin_{freq:.1f}hz"
            channels.append(StimulusChannel(
                name=ch_name,
                pattern=StimulusPattern(
                    pattern_type=StimulusPatternType.SINUSOIDAL.name,
                    amplitude_pA=amplitude_pA,
                    frequency_hz=freq,
                    dc_offset_pA=amplitude_pA * 0.5,  # Ensure positive current
                ),
                target_group_name="input",
                onset_ms=100.0,
                duration_ms=duration_per_freq_ms - 200.0,
            ))

            phases.append(ExperimentPhase(
                name=f"freq_{freq:.1f}hz",
                phase_type=ExperimentPhaseType.STIMULUS.name,
                duration_ms=duration_per_freq_ms,
                active_channels=[ch_name],
                enable_plasticity=False,
            ))

        phases.append(ExperimentPhase(name="post", phase_type=ExperimentPhaseType.BASELINE.name,
                                     duration_ms=2000.0))

        return ExperimentConfig(
            name="Frequency Response Characterization",
            description=f"Sinusoidal sweep {freq_start_hz}-{freq_end_hz} Hz to characterize network filtering.",
            neuron_groups=[
                NeuronGroup(name="input", role=NeuronGroupRole.INPUT.name,
                           index_start=0, index_end=input_group_size,
                           highlight_color=[0.0, 0.8, 1.0, 1.0]),
                NeuronGroup(name="network", role=NeuronGroupRole.OUTPUT.name,
                           index_start=input_group_size, index_end=input_group_size * 3,
                           highlight_color=[1.0, 0.8, 0.0, 1.0]),
            ],
            stimulus_channels=channels,
            phases=phases,
            readout=ReadoutConfig(
                rate_window_ms=100.0,
                rate_group_names=["input", "network"],
                enable_psd=True,
                psd_window_ms=1000.0,
            ),
            enabled=True,
        )

    @staticmethod
    def get_preset_names():
        """Return list of available preset names."""
        return [
            "Basic Stimulus-Response",
            "Associative Conditioning (CS-US)",
            "Reinforcement Learning (R-STDP)",
            "Frequency Response Characterization",
        ]

    @staticmethod
    def get_preset(name, **kwargs):
        """Get a preset by name."""
        presets = {
            "Basic Stimulus-Response": ExperimentPresets.basic_stimulus_response,
            "Associative Conditioning (CS-US)": ExperimentPresets.associative_conditioning,
            "Reinforcement Learning (R-STDP)": ExperimentPresets.reinforcement_learning,
            "Frequency Response Characterization": ExperimentPresets.frequency_response_characterization,
        }
        factory = presets.get(name)
        if factory:
            return factory(**kwargs)
        return None
