"""Configuration dataclasses for the neural simulator."""

from dataclasses import dataclass, field, fields, asdict
from typing import List, Dict

from sim.enums import (NeuronModel, NeuronType, DefaultHodgkinHuxleyParams,
                        StimulusPatternType, NeuronGroupRole, ExperimentPhaseType,
                        TrainingMode)


# DefaultIzhikevichParamsManager is defined in neural-simulator.py and not extracted here.
# CoreSimConfig references it for default field values, so we import it lazily
# by deferring to a module-level reference that neural-simulator.py will set.
# For now, we hardcode the default values directly to avoid circular imports.

# Default Izhikevich RS Cortical Pyramidal parameters (from DefaultIzhikevichParamsManager)
_IZH_RS_DEFAULTS = {
    "C": 100.0, "k": 0.7, "vr": -60.0, "vt": -40.0, "vpeak": 35.0,
    "a": 0.03, "b": -2.0, "c_reset": -50.0, "d_increment": 100.0
}

# Default HH L5 Cortical Pyramidal RS parameters
_HH_L5_DEFAULTS = DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]


@dataclass
class CoreSimConfig:
    """Holds parameters essential for the simulation's logic and reproducibility."""
    total_simulation_time_ms: float = 60000.0
    dt_ms: float = 1.000
    num_neurons: int = 1000
    connections_per_neuron: int = 100
    num_traits: int = 5
    seed: int = -1
    neuron_model_type: str = NeuronModel.IZHIKEVICH.name
    default_neuron_type_izh: str = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    default_neuron_type_hh: str = NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name
    neural_profile_name: str = "GENERIC_UNSTRUCTURED"  # High-level structural preset (brain region / mode)
    inhibitory_trait_indices: List[int] = field(default_factory=list)  # Optional multi-trait inhibitory set
    hardware_performance_note: str = ""  # Note about hardware realtime capacity (populated by viz_benchmark.py)

    # Izhikevich - initialized from default type values
    izh_C_val: float = field(default_factory=lambda: _IZH_RS_DEFAULTS["C"])
    izh_k_val: float = field(default_factory=lambda: _IZH_RS_DEFAULTS["k"])
    izh_vr_val: float = field(default_factory=lambda: _IZH_RS_DEFAULTS["vr"])
    izh_vt_val: float = field(default_factory=lambda: _IZH_RS_DEFAULTS["vt"])
    izh_vpeak_val: float = field(default_factory=lambda: _IZH_RS_DEFAULTS["vpeak"])
    izh_a_val: float = field(default_factory=lambda: _IZH_RS_DEFAULTS["a"])
    izh_b_val: float = field(default_factory=lambda: _IZH_RS_DEFAULTS["b"])
    izh_c_val: float = field(default_factory=lambda: _IZH_RS_DEFAULTS["c_reset"])
    izh_d_val: float = field(default_factory=lambda: _IZH_RS_DEFAULTS["d_increment"])

    # Hodgkin-Huxley - initialized from default type values
    hh_C_m: float = field(default_factory=lambda: _HH_L5_DEFAULTS["C_m"])
    hh_g_Na_max: float = field(default_factory=lambda: _HH_L5_DEFAULTS["g_Na_max"])
    hh_g_K_max: float = field(default_factory=lambda: _HH_L5_DEFAULTS["g_K_max"])
    hh_g_L: float = field(default_factory=lambda: _HH_L5_DEFAULTS["g_L"])
    hh_E_Na: float = field(default_factory=lambda: _HH_L5_DEFAULTS["E_Na"])
    hh_E_K: float = field(default_factory=lambda: _HH_L5_DEFAULTS["E_K"])
    hh_E_L: float = field(default_factory=lambda: _HH_L5_DEFAULTS["E_L"])
    hh_v_rest_init: float = field(default_factory=lambda: _HH_L5_DEFAULTS["v_rest_hh"])
    hh_v_peak: float = field(default_factory=lambda: _HH_L5_DEFAULTS["v_peak_hh"])
    hh_m_init: float = field(default_factory=lambda: _HH_L5_DEFAULTS["m_init"])
    hh_h_init: float = field(default_factory=lambda: _HH_L5_DEFAULTS["h_init"])
    hh_n_init: float = field(default_factory=lambda: _HH_L5_DEFAULTS["n_init"])
    hh_temperature_celsius: float = 37.0
    hh_q10_factor: float = 3.0  # Legacy uniform Q10 — kept as fallback for
                                 # per-gate Q10 fields below when those are
                                 # set to <=0 (meaning "use legacy Q10").
    # Per-gate Q10 (preferred over hh_q10_factor at biological temps).
    # Default values produce real APs at 37°C; uniform Q10=3 over-compressed
    # dynamics so the cell tonically depolarized without firing — see
    # research/findings/2026-04-25-hh-temperature-bug.md.
    hh_q10_m: float = 3.0   # Activation (m-gate): fast — Mainen & Sejnowski 1996
    hh_q10_h: float = 1.5   # Inactivation (h-gate): slower — preserves AP width
    hh_q10_n: float = 1.5   # Recovery (n-gate): slower — preserves AP duration
    # Optional extended HH currents. Zero conductance disables each one.
    hh_g_M_max: float = 0.0
    hh_m_current_tau_ms: float = 100.0
    hh_g_CaT_max: float = 0.0
    hh_E_CaT: float = 120.0
    hh_g_h_max: float = 0.0
    hh_E_h: float = -30.0
    hh_g_NaP_max: float = 0.0

    # AdEx parameters. Default: Brette & Gerstner 2005 RS pyramidal.
    # Override via cfg.default_neuron_type_adex ∈ {ADEX_RS_CORTICAL_PYRAMIDAL,
    # ADEX_FS_CORTICAL_INTERNEURON, ADEX_IB_BURSTING, ADEX_CH_CHATTERING,
    # ADEX_LTS_LOW_THRESHOLD, ADEX_STRIATAL_MSN, ADEX_DOPAMINE}. The bridge
    # init reads this enum and overlays preset values onto the cfg.adex_*
    # fields below.
    default_neuron_type_adex: str = "ADEX_RS_CORTICAL_PYRAMIDAL"
    adex_C: float = 281.0          # pF
    adex_g_L: float = 30.0         # nS
    adex_E_L: float = -70.6        # mV
    adex_V_T: float = -50.4        # mV
    adex_Delta_T: float = 2.0      # mV
    adex_a: float = 4.0            # nS
    adex_tau_w: float = 144.0      # ms
    adex_b: float = 80.5           # pA
    adex_V_r: float = -70.6        # mV (reset voltage)
    adex_V_peak: float = -40.0     # mV (spike detection threshold)

    # Per-model external drive scaling (tuned per combination; 1.0 = baseline range)
    hh_external_drive_scale: float = 1.0
    adex_external_drive_scale: float = 1.0

    # B2: Parameter Heterogeneity (Marder & Goaillard 2006, Tripathy et al. 2013)
    enable_parameter_heterogeneity: bool = True  # Enabled by default for biological realism
    heterogeneity_seed: int = -1  # Separate from main seed for reproducibility (-1 = use main seed)
    # Distribution specifications: {"param_name": {"type": "lognormal"|"gaussian", "mean_log"|"mean": X, "sigma_log"|"std": Y}}
    heterogeneity_distributions: dict = field(default_factory=dict)  # Empty by default, populated on demand

    # B4: Enhanced Channel Noise (White et al. 2000, Destexhe & Rudolph-Lilith 2012)
    # Conductance noise (multiplicative, applied to HH channels)
    enable_conductance_noise: bool = True  # Enabled by default for HH model biological realism
    conductance_noise_relative_std: float = 0.05  # 5% relative noise (conservative estimate)

    # Ornstein-Uhlenbeck process for background synaptic drive
    enable_ou_process: bool = True  # Enabled by default for biological realism
    ou_mean_current_pA: float = 0.0           # Mean background current (pA)
    ou_std_current_pA: float = 100.0          # Fluctuation amplitude (50-200 pA typical, produces 2-5mV Vm fluctuations)
    ou_tau_ms: float = 15.0                   # Correlation time (10-20 ms, matches synaptic time constants)
    ou_seed: int = -1                         # Separate seed for noise (-1 = use main seed)

    # Synapse & Plasticity
    refractory_period_steps: int = 2
    syn_reversal_potential_e: float = 0.0
    syn_reversal_potential_i: float = -75.0  # GABA-A chloride reversal (was -70; -75 matches Cl- Nernst at 37C)
    syn_tau_g_e: float = 5.0
    syn_tau_g_i: float = 10.0
    # NMDA conductance with voltage-dependent Mg2+ block (Jahr & Stevens 1990)
    enable_nmda: bool = False
    nmda_ratio: float = 0.4           # NMDA:AMPA conductance ratio (0 = no NMDA, 1 = equal)
    nmda_tau_decay: float = 100.0     # NMDA decay time constant (ms) -- slow compared to AMPA
    nmda_tau_rise: float = 3.0        # NMDA rise time constant (ms)
    nmda_mg_concentration: float = 1.0  # Extracellular [Mg2+] in mM
    # Per-pathway slow-NMDA-dominant recurrent routing (2026-06-09; Wang 2001/2002
    # graded persistent attractor). When True AND a pathway sets exc_receptor=
    # "nmda_slow", that pathway's excitatory synapses feed a SEPARATE slow-NMDA
    # conductance (its own tau, Mg2+-block self-limiting) and their fast-AMPA g_e
    # component is SUPPRESSED -- so a CA3 recurrent can reverberate gradedly (no
    # synchronous AMPA volley) while the mossy detonator stays fast AMPA. This is
    # the EXCITATORY mirror of the GABA_B `receptor=` per-pathway routing (a second
    # conductance + a per-synapse routing mask), differing only in that the AMPA
    # component is suppressed for routed synapses (GABA_B is additive on top of
    # GABA_A; nmda_slow REPLACES AMPA with slow NMDA, per the Wang recurrent=NMDA /
    # feedforward=AMPA architecture). Default False => no pathway is routed, the new
    # increment block is unreached, the g_e matvec is unmasked, and
    # total_input_current_pA is byte-identical to today (mirrors enable_gabab /
    # enable_nmda). See research/findings/2026-06-09-learned-graded-ca3-design.md.
    enable_nmda_recurrent: bool = False
    nmda_recurrent_ratio: float = 1.0          # recurrent slow-NMDA increment scale (AMPA component suppressed for nmda_slow pathways)
    nmda_recurrent_tau_decay_ms: float = 100.0 # slow NR2B decay (Wang); >> AMPA ~5ms
    nmda_recurrent_tau_rise_ms: float = 2.0    # NMDA rise (ms)
    nmda_recurrent_propagation_strength: float = 0.05  # per-spike increment scale (mirrors propagation_strength for the excitatory path)
    # Per-pathway dendritic COINCIDENCE detection (2026-06-09; Major-Larkum-Schiller "Decade of the
    # Dendritic NMDA Spike" + Poirazi-Mel 2003 two-layer subunit). When True AND a pathway sets
    # coincidence_detector=True, each postsynaptic neuron forms a dendritic SUBUNIT over that pathway's
    # synapses: if >= coincidence_k_threshold of its routed inputs fire in the SAME step (a clustered/
    # synchronous volley), a regenerative ALL-OR-NONE plateau current (saturating, Mg2+-self-limiting via
    # the reused Jahr-Stevens block, slow ~80ms decay) is injected -- so a SPARSE-distinct ensemble (each
    # cell <0.2 spk/step) can fire the cell by COINCIDENCE where the linear rate-weighted matvec cannot
    # (the point-neuron rate-coding wall, 2026-06-09-learned-graded-ca3-derisk-RESULT.md). This is the
    # COINCIDENCE sibling of enable_nmda_recurrent (a second per-neuron conductance + a per-synapse
    # routing mask + the reused Mg-block kernel), differing in that the fast-AMPA g_e component is KEPT
    # (the plateau rides ADDITIVELY on top of the AMPA EPSP, matching the NMDA spike riding the EPSP --
    # UNLIKE nmda_slow, which REPLACES AMPA). Default False => no pathway is routed, the new per-neuron
    # coincidence block is unreached, the g_e matvec is unmasked, and total_input_current_pA is byte-
    # identical to today (mirrors enable_nmda_recurrent / enable_gabab). The K/gain operating point is
    # calibrated by research/runners/coincidence_wall_probe.py (Step 0). See
    # research/findings/2026-06-09-coincidence-substrate-upgrade-design.md.
    enable_coincidence_detection: bool = False
    coincidence_k_threshold: float = 6.0        # # of synchronous clustered inputs to trigger the plateau (biology: 10-50 synapses; scaled to fan-in)
    # Opt-in RF resonate megakernel fast path (perf, default OFF -> byte-identical to the per-step loop). One custom
    # CUDA kernel does the whole resonate step (complex sparse matvec + dynamics), collapsing ~15-20 CuPy kernels/
    # step into 1 -- the launch-bound conversational-latency fix. GPU-only; numpy backend + masked bridges fall back
    # to the loop. See docs/plans/2026-06-17-resonate-cudagraph-refactor-design.md.
    enable_rf_cudagraph: bool = False
    coincidence_gain: float = 2.0               # sigmoid slope of the all-or-none switch (~0.88/0.12 at K+/-1)
    coincidence_plateau_strength: float = 80.0  # per-step plateau conductance increment scale (the regenerative NMDA-spike drive)
    coincidence_tau_decay_ms: float = 80.0      # plateau duration (Major-Larkum-Schiller NMDA spike 50-100ms)
    coincidence_tau_rise_ms: float = 2.0        # plateau rise (ms)
    # Poirazi-Mel 2003 WEIGHTED-subunit refinement (2026-06-09). The count form above switches the plateau
    # on the bare COUNT of coincident routed inputs (c_i = mask^T @ prev_fired), so it is WEIGHT-BLIND: a
    # sparse-distinct ensemble that fires >= K cells everywhere triggers the same plateau regardless of the
    # learned synaptic weight, and a value readout cannot GRADE (e.g. the N9 place->value critic fires near
    # AND far the goal). The faithful Poirazi-Brannon-Mel subunit fires on the WEIGHTED local depolarization
    # Sum_j (w_eff_j * x_j), NOT a count. When this flag is True the per-neuron switch variable becomes the
    # per-step WEIGHTED coincident sum c_w (effective_connections_matrix.data masked to the routed synapses
    # @ prev_fired) so a strongly-weighted coincident ensemble (high learned value) crosses the supralinear
    # switch while a weakly-weighted one does not -> the plateau GRADES with synaptic value. c_w is still a
    # per-step sum vs prev_firing, so temporal jitter that desynchronizes the ensemble drops it below
    # threshold and collapses the plateau (the coincidence/anti-rate property of the count form is kept --
    # verified by the jitter anti-cheat). coincidence_k_threshold is then read in WEIGHT units (the runner
    # re-tunes it). Default False => the validated COUNT form runs unchanged (byte-identical to today).
    coincidence_weighted_drive: bool = False
    # GRADED dendritic-plateau READ-OUT (Stage 1, 2026-06-20; the SMOOTH/non-saturating sibling of the
    # all-or-none enable_coincidence_detection above). De-risk A (Stage 0, GO 6/6 seeds,
    # 2026-06-20-dendrite-derisk-A-graded-plateau-readout.md) proved the dendrite's ONE genuine unlock
    # is the GRADED ANALOG read-out of a distributed code (Mikulasch-Priesemann): the nav value-critic
    # delta=r-V needs a graded value V the point-neuron soma provably cannot express (LINEAR=sub-rheobase
    # 0 Hz flat delta~1.0; all-or-none PLATEAU over-clamps delta~0.0). When True AND a pathway sets
    # coincidence_detector=True (the SAME per-synapse routing mask the coincidence block uses -- NO new
    # wiring), each routed postsynaptic neuron forms a GRADED dendritic plateau: its per-step WEIGHTED
    # coincident drive c_w = Sum_j (w_eff_j * x_j) (the learned place->value synaptic value, the Poirazi-
    # Brannon-Mel weighted subunit) passes through a SMOOTH, CENTERED, non-saturating logistic
    # V = 1/(1+exp(-slope*(c_w - center))) scaled to a regenerative Mg2+-self-limiting plateau current --
    # so the plateau GRADES with the learned value (V(near) > V(mid) > V(far): the continuum the
    # all-or-none switch snaps to 0/1). This is the on-substrate realization of the Stage-0 numpy
    # sigmoid((v_basal-theta)/slope) value read-out, produced by the spiking bridge dendrite. It is the
    # GRADED-transfer sibling of fused_coincidence_plateau (same dual-exp Mg-block kinetics, same routing
    # mask, same E_e reversal), differing ONLY in the transfer function (gentle logistic vs the steep
    # all-or-none switch). MUTUALLY EXCLUSIVE with enable_coincidence_detection in practice (the runner
    # sets one). Default False => no graded plateau conductance is allocated, the new per-neuron graded
    # block is unreached, the new kernel is never called, and total_input_current_pA is BYTE-IDENTICAL to
    # today (mirrors enable_coincidence_detection / enable_gabab / enable_dendritic_divisive_gain exactly).
    # The center/slope/strength operating point matches the Stage-0 de-risk (dend_theta=8, dend_slope=3).
    enable_graded_dendritic_plateau: bool = False
    graded_plateau_center: float = 8.0          # the logistic center c_w (WEIGHT units) -- the half-max value point (Stage-0 dend_theta)
    graded_plateau_slope: float = 0.33          # the SMOOTH logistic slope (~1/dend_slope; gentle => non-saturating across the active range, the graded middle)
    graded_plateau_strength: float = 80.0       # per-step plateau conductance increment scale at V->1 (the regenerative NMDA-spike drive; matches coincidence_plateau_strength)
    graded_plateau_tau_decay_ms: float = 80.0   # plateau duration (Major-Larkum-Schiller NMDA spike 50-100ms; matches coincidence_tau_decay_ms)
    graded_plateau_tau_rise_ms: float = 2.0     # plateau rise (ms)
    # GABA_B -> GIRK slow K+ inhibitory conductance (metabotropic). Default OFF;
    # byte-identical when disabled. E_gabab is the POTASSIUM reversal (~-90 mV),
    # independent of the chloride gradient, so it strongly hyperpolarizes KCC2-lacking
    # DA cells where GABA_A (E_GABA ~ -55 mV) is weak/shunting. See catalog B.15, J.11.
    enable_gabab: bool = False
    gabab_reversal_potential: float = -90.0   # E_K (GIRK), mV
    gabab_tau_decay: float = 150.0            # slow decay (ms); GABA_B/GIRK >> GABA_A's 10 ms
    gabab_propagation_strength: float = 0.105 # per-spike conductance increment scale (mirrors inhibitory_propagation_strength)
    # TD value-DERIVATIVE channel (B-2, 2026-06-10 N9 TD cue-shift design §2.2/§3). The
    # bootstrap γ·V(s') − V(s) delivered to the SNc as the temporal DERIVATIVE of the GABA_B
    # value channel, computed AT THE MEMBRANE in conductance (NOT from the critic's firing
    # density — that is what made the B-3 zero-edit route net-depressing). Mechanism: maintain
    # one slow leaky-EMA copy of g_gabab (decaying with td_slow_tau_ms > gabab_tau_decay); the
    # band-passed difference (g_gabab − g_gabab_slow) is the value derivative. Delivered as a
    # DEPOLARIZING current via E_exc (= syn_reversal_potential_e = 0 mV): a value RISE
    # (g_gabab − g_gabab_slow > 0, with E_exc − V > 0) depolarizes the SNc → a burst at the cue;
    # a value FALL hyperpolarizes → the omission dip; flat value → ~0. ADDITIVE on top of the
    # existing −V GABA_B subtraction. Default OFF; byte-identical when disabled (these fields are
    # read ONLY inside `if enable_td_value_derivative`, which is False here, so the whole block is
    # unreached and total_input_current_pA is bit-identical). Mirrors enable_gabab exactly.
    enable_td_value_derivative: bool = False
    td_slow_tau_ms: float = 400.0             # slow-EMA decay of g_gabab (ms); > gabab_tau_decay (~150) so the difference is a derivative
    td_derivative_gain: float = 1.0           # scales the conductance-derivative current onto the SNc
    # DENDRITIC per-presynaptic-source DIVISIVE GAIN (D2 Phase 1, 2026-06-14). The on-substrate
    # realization of the de-risked (D1/D1.5/D1.6/D1.7) per-input divisive normalization a dendritic
    # compartment performs per branch: each presynaptic source's spike is scaled by a bounded gain
    # g_i = sigma / (sigma + a_i), where a_i is that source's own firing-rate EMA (cp_dendritic_source_
    # activity, a dedicated slow EMA decoupled from homeostasis). A high-frequency COMMON source
    # (large a_i) is suppressed toward 0; a rare INFORMATIVE source (a_i ~ 0) passes near 1 -- the
    # PPMI-marginal / Carandini-Heeger divisive-normalization the point-neuron soma provably cannot do
    # (a single global gain), realized at the synaptic input. The gain is in (0,1] (pure suppression,
    # never amplifies -> cannot destabilize the hot path). Applied to the presynaptic firing vector
    # BEFORE the excitatory/inhibitory conductance matvec. Default OFF; byte-identical when disabled
    # (cp_dendritic_source_activity stays None, so the gain block AND the EMA-update block are unreached
    # and the step is bit-identical). Mirrors enable_gabab / enable_coincidence_detection exactly.
    enable_dendritic_divisive_gain: bool = False
    dendritic_divisive_sigma: float = 0.05    # the (0,1] gain's semi-saturation: g = sigma/(sigma + activity_ema); ~ a typical firing-rate EMA
    dendritic_gain_ema_alpha: float = 0.01    # per-source activity EMA rate (tau ~100 steps; the slow marginal the gain normalizes by)
    propagation_strength: float = 0.05
    inhibitory_propagation_strength: float = 0.105  # Scaled for E_inh=-75mV (was 0.15 at E_inh=-70mV)
    max_synaptic_delay_ms: float = 20.0
    enable_inhibitory_neurons: bool = True
    inhibitory_trait_index: int = 1
    enable_hebbian_learning: bool = True
    hebbian_learning_rate: float = 0.0005
    hebbian_weight_decay: float = 0.00001
    hebbian_min_weight: float = 0.05
    hebbian_max_weight: float = 1.0
    enable_short_term_plasticity: bool = True
    stp_U: float = 0.15          # Global fallback U (used when per-type not available)
    stp_tau_d: float = 200.0     # Global fallback tau_d (ms)
    stp_tau_f: float = 50.0      # Global fallback tau_f (ms)
    # Per-connection-type STP parameters [E->E, E->I, I->E, I->I]
    # When enable_per_type_stp is True, these override the global values.
    enable_per_type_stp: bool = True
    stp_U_per_type: list = None       # [U_ee, U_ei, U_ie, U_ii] -- set in __post_init__
    stp_tau_d_per_type: list = None   # [tau_d_ee, tau_d_ei, tau_d_ie, tau_d_ii] (ms)
    stp_tau_f_per_type: list = None   # [tau_f_ee, tau_f_ei, tau_f_ie, tau_f_ii] (ms)
    enable_homeostasis: bool = True
    homeostasis_target_rate: float = 0.02
    homeostasis_threshold_adapt_rate: float = 0.0005  # Slower: ~0.5 mV/sec at max error (was 0.015)
    homeostasis_ema_alpha: float = 0.0002  # tau_ema ~5000 steps = 5s at dt=1ms (was 0.01 = 100ms)
    homeostasis_threshold_min: float = -55.0
    homeostasis_threshold_max: float = -30.0
    # Synaptic scaling (Turrigiano 2008): multiplicatively scales excitatory weights
    # toward target rate. Works across all neuron models, biologically grounded.
    enable_synaptic_scaling: bool = False
    synaptic_scaling_rate: float = 0.001  # Slow scaling rate (operates on seconds timescale)
    # Deterministic transpose SpMV (2026-06-10): the per-step synaptic drive is Wᵀ@fired, a
    # TRANSPOSE sparse mat-vec; on CuPy `csr.T`=csc → cusparse.spmv(transa=True) uses an atomic
    # scatter that is bit-NON-reproducible run-to-run (CUBLAS_WORKSPACE_CONFIG pins cuBLAS only).
    # When True, the bridge materializes the transpose as a CSR (a deterministic NON-transpose
    # SpMV, numerically allclose). Default False ⇒ the matvec is byte-identical to before. Costs a
    # per-step `.tocsr()`, so callers toggle it ON only where reproducibility matters (e.g. the N9
    # place-code self-org). See research/findings/2026-06-10-N9-placecode-reproducibility-*.
    deterministic_transpose_matvec: bool = False
    # GABA_B / GIRK conductance saturation cap (2026-06-10): finite GIRK K+ channels => the slow
    # GABA_B conductance cannot grow without bound. A HOT presynaptic source (e.g. an over-firing
    # value critic, ~125 Hz) otherwise over-accumulates g_gabab and FULLY CLAMPS the downstream
    # cell (the critic-rate-dependent over-clamp that makes the online δ=r−V binary on hot draws —
    # nav A/B 2026-06-10-N9-nav-deployment-stageB-PASS-seed42.md). Capping g_gabab bounds the
    # subtraction => a GRADED (Eshel arithmetic) shift at ANY presynaptic rate. 0.0 = no cap
    # (default) => byte-identical.
    gabab_conductance_max: float = 0.0
    enable_watts_strogatz: bool = True
    connectivity_k: int = 10
    connectivity_p_rewire: float = 0.1

    # C2: STDP (Spike-Timing-Dependent Plasticity) - Bi & Poo 1998, Caporale & Dan 2008
    enable_stdp: bool = True  # Enabled by default for biologically realistic learning
    stdp_a_plus: float = 0.012             # LTP amplitude (typical: 0.005-0.02, biased > A- for net potentiation)
    stdp_a_minus: float = 0.01             # LTD amplitude (typical: slightly < A+, net LTP bias per Song et al. 2000)
    stdp_tau_plus_ms: float = 20.0         # LTP time constant (ms, typical: 15-25ms)
    stdp_tau_minus_ms: float = 20.0        # LTD time constant (ms, typical: 15-25ms)
    stdp_w_min: float = 0.0                # Minimum synaptic weight
    stdp_w_max: float = 2.0                # Maximum synaptic weight
    stdp_only_nearest_spike: bool = True   # Use only nearest spike pairs (more efficient)
    # Performance: fast spike-reset path that avoids the GPU-CPU sync at
    # `if fired_indices.size > 0`. Uses cp.where masked-update instead of
    # fancy-index assignment. Numerically equivalent for the Izhikevich
    # path; biggest win on small networks where launch overhead dominates.
    # Default False so existing runs are bit-identical; opt-in via this flag.
    fast_spike_reset: bool = False

    # Performance: VECTORIZED activity-driven transmission-gate couplings
    # (couple_gate_to_pool / _apply_gate_couplings). The per-step hook reads
    # each registered coupling's control-pool firing fraction
    # (cp_firing_states[control_idx].mean()) in a Python for-loop. For the
    # K-way on-substrate sequencer (#3) the coupling count is ~K*V*2 (K=32,
    # V=72 -> ~4640 couplings), so the loop does ~4640 separate .mean()
    # reductions per step and dominates the host CPU time (profiled ~52% of a
    # K=8 query). When this flag is True, _apply_gate_couplings batches all
    # control-pool means into ONE segment-sum (cp.add.reduceat over a cached
    # flat concat of the control indices, divided by per-segment counts).
    #
    # BYTE-IDENTICAL: each control region is a contiguous DISJOINT index block
    # and the firing states are boolean, so a control pool's mean is an EXACT
    # integer sum / integer count -- identical whether computed per-slice or by
    # segment-sum (no float reassociation). The EMA update + gate-write loop
    # then runs in the SAME coupling order. Default False = the scalar loop
    # (the reference path stays available); the flag is opt-in (the #3 K-way
    # sequencer runner sets it). Guarded: with no couplings the hook is a no-op
    # under either path. See tests/test_gate_coupling_vectorized.py.
    enable_vectorized_gate_couplings: bool = False

    # Performance: FP16 mixed-precision for synaptic state (eligibility
    # traces + connection weights). Voltages and recovery vars stay FP32
    # because spiking integration needs higher precision near threshold.
    #
    # Honest expected gain: 1.05-1.15x. Sparse mat-vec (CSR SpMV) is
    # bandwidth-bound, so halving the bytes per element gives ~1.5x on
    # that step. But integration kernels dominate total time and they
    # stay FP32. Real gains come from more parallel processes (use VRAM
    # headroom) more than from FP16 — but FP16 stacks with parallelism.
    #
    # Numerical risk: weight precision drops from 7 decimal digits (FP32)
    # to 3 (FP16). For STDP at typical step sizes (1e-3 to 1e-2), this
    # accumulates noise of ~0.5% per 1000 events. Acceptable for our
    # task (final accuracy stable at ±2% normally) but validate via
    # tests/test_fp16_drift.py before relying on results.
    fp16_synapse_state: bool = False

    # C2: Reward-Modulated Plasticity (Three-factor learning rule) - Izhikevich 2007
    enable_reward_modulation: bool = True  # Enabled by default for reinforcement learning
    reward_learning_rate: float = 0.01     # Modulation strength (typical: 0.001-0.05)
    reward_eligibility_tau_ms: float = 1000.0  # Eligibility trace decay (ms, typical: 500-2000ms)
    reward_baseline: float = 0.0           # Expected reward (for prediction error)
    current_reward_signal: float = 0.0     # Current reward value (updated externally or via task)
    # NAMING NOTE (2026-04-29 catalog): `current_reward_signal` is a SIGNED SCALAR
    # that conflates two biologically distinct DA responses (Schultz98/16):
    #   (1) phasic activation above tonic DA (positive value); LTP via D1
    #   (2) phasic depression below tonic DA (negative value); LTD via D1 / LTP via D2
    # The receptor-level effect of (2) is NOT a sign-flipped (1). R2.4
    # added `reward_aversive_scale` to handle the magnitude asymmetry, and
    # Cluster C v1 (`--enable-tonic-da`) replaces this scalar with a real
    # `dopamine` neuromodulator concentration (tonic + phasic deviations).
    # The legacy scalar path remains for backward compatibility when the
    # DA modulator isn't registered.
    # Cluster C v2 (2026-04-29): last action selected by agent. Read by
    # `from_action_specific_reward` production rule so per-action DA
    # modulators only fire when the matching action is selected. -1 means
    # "no action" (between trials, runner hasn't reported yet). Runners
    # set this after each action selection.
    last_selected_action: int = -1
    # R2.4 (2026-04-29): aversive-vs-appetitive magnitude asymmetry.
    # Schultz98/Schultz16/Fiorillo 2013: phasic DA "activations" to aversive
    # stimuli largely reflect physical-impact artifacts; the underlying
    # valence response is a *depression* below tonic, smaller in magnitude
    # than appetitive activations. Models this asymmetry by scaling
    # negative reward_prediction_error by this factor (0.0 = aversive
    # rewards do nothing; 1.0 = symmetric; default 0.5 reflects the
    # ~50% magnitude observed in DA recordings).
    reward_aversive_scale: float = 0.5

    # C2b: Neuromodulator subsystem (Session E.1)
    # Opt-in framework subsuming the legacy current_reward_signal path. When
    # enabled, neuromodulators in `neuromodulators` are managed by a
    # NeuromodulatorManager with concentration dynamics, production rules,
    # and configurable receptor effects on synaptic gain / plasticity rate
    # / excitability drive. Default OFF for full backward compatibility.
    # See sim/neuromodulators.py and
    # docs/plans/2026-04-24-neuromodulator-subsystem.md
    enable_neuromodulator_subsystem: bool = False
    neuromodulators: list = field(default_factory=list)  # List[NeuromodulatorConfig]

    # C2c: Brain-region framework (Session E.2)
    # Opt-in declarative framework for multiple cortical / subcortical
    # populations sharing one bridge. When enabled, brain_regions defines
    # contiguous index slices and internal connectivity per region;
    # region_pathways defines directed cross-region projections with
    # optional neuromodulator-gated plasticity. Bridge auto-sets
    # num_neurons from RegionManager.total_neurons() when a region list
    # is non-empty.
    # See sim/regions.py and docs/plans/2026-04-24-brain-region-framework.md
    enable_brain_region_framework: bool = False
    brain_regions: list = field(default_factory=list)  # List[BrainRegion]
    region_pathways: list = field(default_factory=list)  # List[RegionPathway]

    # ─── Graded LGN decorrelation stage (2026-06-06) ───────────────
    # The biology-faithful, on-substrate realization of the validated
    # whitening rule (research/findings/2026-06-06-option1-local-learning-
    # whitening-VALIDATED-6seed.md): a per-region GRADED pairwise lateral
    # inhibition operating on SUB-THRESHOLD analog activity (NOT spikes),
    # pre-spike, where the retina/LGN does variance equalization. Default
    # OFF for full backward compatibility — when off, the Izhikevich/HH/
    # AdEx step paths are byte-unchanged (the new code is a guarded no-op
    # reached only when this flag AND a region's BrainRegion.graded_lateral
    # are both set). Requires enable_brain_region_framework=True.
    # Design: docs/plans/2026-06-06-graded-lgn-decorrelation-design.md
    enable_graded_lateral: bool = False
    # Learning rate η for the anti-Hebbian co-activity update ΔM ∝ ⟨a aᵀ⟩ - I.
    graded_lateral_lr: float = 0.02
    # The -λM synaptic weight-decay (the rate-model regularizer that settles a
    # GENTLE, bounded fixed point ≈C^-1/3 instead of over-whitening). Typically
    # set λ≈η. This is a DEDICATED knob (not cfg.hebbian_weight_decay) so the
    # graded lateral is fully isolated from the global Hebbian plumbing.
    graded_lateral_lambda: float = 0.01
    # Scales the graded inhibitory current -(M @ a) into pA before it is added
    # to the region's input current.
    graded_lateral_gain_pA: float = 300.0
    # Normalizer for the sub-threshold analog activity a = relu((v - v_rest)/scale).
    # Sets the operating point of a (roughly the mV of depolarization that maps
    # to a~1). ~10-20 mV is the sub-threshold dynamic range above rest.
    graded_lateral_act_scale: float = 15.0
    # EMA decay for the running co-activity estimate ⟨a aᵀ⟩ (0 = instantaneous
    # a aᵀ each step; >0 averages over ~1/(1-decay) steps for a smoother target).
    graded_lateral_coact_ema: float = 0.0

    # ─── Slow per-hub INPUT-MEAN adaptation (axis-0 per-feature centering, 2026-06-15) ──────
    # The SEPARABLE diagonal/DC half of whitening (per-feature mean-centering = subtractive
    # spike-frequency adaptation / point-neuron predictive coding; Lee, Dora, Mejias, Bohte &
    # Pennartz 2024, PMC11045951). Each FLAGGED neuron maintains a SLOW EMA m of its OWN pre-
    # threshold input drive and subtracts gain*m from that drive BEFORE the threshold:
    #   adapted = raw_drive - gain*m;  m <- (1-alpha)*m + alpha*raw_drive   (causal: subtract
    # the CURRENT m, THEN update m from raw_drive). raw_drive = the neuron's own synaptic +
    # external input current (on-substrate, NOT a host-precomputed x-mean -- BRAIN-BASED-ONLY).
    # This is the per-FEATURE (axis-0) centering the L1 learned cortex needs (the common-mode
    # POOL does the WRONG axis-1 per-concept removal; CYCLE-69). The numpy mechanism is 6-seed
    # GO (+0.311; clean-mean +0.311, spiking-mean +0.298). Default OFF -> when no region sets
    # BrainRegion.input_mean_adapt=True, cp_input_mean_ema / cp_input_mean_adapt_mask stay None,
    # the new step block is unreached, and total_input_current_pA is byte-identical to today.
    # Requires enable_brain_region_framework=True (the mask is built from region membership).
    # See research/findings/2026-06-15-slow-perhub-mean-primitive-deep-research.md (Option A) +
    # docs/plans/2026-06-15-analog-substrate-learned-cortex-build-plan.md (Phase 2).
    enable_input_mean_adapt: bool = False
    # PER-STEP EMA rate alpha. SLOW: the mean must span many concept PRESENTATIONS (not steps).
    # The runner sets this from the presentation length (e.g. alpha_per_presentation ~0.02-0.05
    # divided by steps-per-presentation -> a tiny per-step value). 0.0 (default) freezes the EMA
    # (m never moves) -- the runner uses this to FREEZE adaptation before the read-out phase.
    input_mean_adapt_alpha: float = 0.0
    # Subtraction strength: the effective input current of a flagged neuron becomes
    # raw_drive - gain*m. 1.0 = subtract the full running mean (the validated numpy op).
    input_mean_adapt_gain: float = 1.0

    # ─── Per-concept DIVISIVE input normalization (Carandini-Heeger, 2026-06-15) ───
    # For each FLAGGED neuron, divide its pre-threshold input by (sigma + gain*MEAN input over the
    # flagged set): r_i = x_i / (sigma + gain*mean_j x_j). A feedforward divisive-gain circuit that
    # realizes PPMI's per-concept (row-marginal) normalization -- the canonical cortical normalization
    # by total pool drive. Composes with input_mean_adapt (per-hub subtractive) + the neuron's log-ish
    # f-I to approximate PPMI on the bridge (validated load-bearing in numpy: +0.158 -> +0.339 on the
    # real corpus). GUARDED: unless enable_input_divisive_norm AND a region sets
    # BrainRegion.input_divisive_norm=True, cp_input_divisive_mask stays None and the per-step block is
    # unreached, so total_input_current_pA is byte-identical to today.
    enable_input_divisive_norm: bool = False
    # Semi-saturation constant sigma (keeps the divisor bounded away from 0 when the pool is quiet).
    input_divisive_sigma: float = 1.0
    # Divisive strength on the mean term.
    input_divisive_gain: float = 1.0

    # ─── SECOND, INDEPENDENT divisive-norm pool (Cascade-accumulator FIX A, 2026-06-20) ───
    # A byte-identical clone of the divisive primitive above, keyed by BrainRegion.input_divisive_norm_2
    # and its own sigma_2/gain_2, so a SECOND group of regions can be common-mode-normalized as a
    # SEPARATE pool (its divisor is sigma_2 + gain_2*mean over the SECOND flagged set only). Motivation:
    # the #6 SC popvector read-out already flags the four cortex_X pools input_divisive_norm=True (pool 1
    # = the bump-mass normalization, at the cortex current scale). The cascade North-bias FIX A needs an
    # INDEPENDENT divisive normalization at the four sel_X selection accumulators (a DIFFERENT current
    # scale) to divide out the common N+E+S+W drive BEFORE the Wang-2002 accumulators amplify it
    # (research/findings/2026-06-20-cascade-accumulator-Nbias-scoping.md FIX A). One shared pool would
    # mix cortex_X + sel_X currents into one mean and corrupt both; a second pool keeps them separate.
    # GUARDED: unless enable_input_divisive_norm_2 AND a region sets BrainRegion.input_divisive_norm_2,
    # cp_input_divisive_mask_2 stays None and the second per-step block is unreached -> byte-identical to
    # today (no existing config sets these). Default OFF.
    enable_input_divisive_norm_2: bool = False
    input_divisive_sigma_2: float = 1.0
    input_divisive_gain_2: float = 1.0

    # ─── Synapse tiering (Phase 3 Strategy B, 2026-05-11) ──────────
    # Activity-tracked TieredSynapseStore mirrors the per-pathway CSRs
    # alongside the monolithic cp_connections. Foundation for Phase 4
    # auto-tiering (SSD paging). Opt-in. Requires
    # enable_brain_region_framework=True (uses RegionPathway names).
    # Design doc: docs/plans/2026-05-11-tiering-phase3-part2-bridge-integration-design.md
    enable_synapse_tiering: bool = False
    # Eviction policy (only effective if Strategy A's per-pathway compute
    # is ever shipped; for Strategy B these are tracking thresholds only)
    synapse_tiering_evict_idle_steps: int = 1000
    synapse_tiering_grace_pagein_steps: int = 100
    # Phase 4 memory-pressure eviction. 0 = disabled (idle-counter only).
    # Non-zero: evict longest-idle pathway when in-RAM bytes exceed budget.
    # Useful when individual pathways are large; the idle counter alone
    # may not free memory fast enough under hot workloads.
    synapse_tiering_ram_budget_bytes: int = 0
    # Optional storage root override (default: bridges/synapse_shards/active)
    synapse_tiering_root: str = ""

    # C3: Structural Plasticity (Synapse Formation/Elimination) - Butz et al. 2009
    enable_structural_plasticity: bool = True  # Enabled by default for dynamic network adaptation
    struct_plast_formation_rate: float = 1e-6     # Probability per timestep per neuron pair
    struct_plast_elimination_rate: float = 5e-7   # Probability per timestep per synapse
    struct_plast_weight_threshold: float = 0.05   # Eliminate synapses below this weight
    struct_plast_target_density: float = 0.1      # Target connection density (fraction)
    struct_plast_distance_kernel: str = "exp_decay"  # "uniform", "exp_decay", "gaussian"
    struct_plast_distance_scale: float = 20.0     # Spatial scale for distance-dependent formation
    struct_plast_update_interval_steps: int = 100  # Update interval (for efficiency)
    struct_plast_activity_bias: float = 0.5  # Weight of co-activity vs random in formation [0=random, 1=fully activity-driven]

    # ─── Structural plasticity (2026-04-28) ──────────────────────────
    # Cheat #5 closure attempt #5 (option 1 of the post-v4 plan, see
    # docs/plans/2026-04-28-structural-plasticity-design.md). Adds
    # experience-dependent synapse pruning: synapses with negative
    # survival score AND low weight get permanently eliminated.
    enable_structural_pruning: bool = False
    pruning_alpha: float = 0.001
    pruning_threshold: float = -1.0
    pruning_weight_floor: float = 1.0

    # ─── Cluster B.1 (2026-04-28): D1/D2 plasticity asymmetry ─────────
    # D1 MSNs LTP under +DA / LTD under -DA; D2 MSNs invert both signs.
    # Implements via per-synapse sign multiplier on the reward-modulated
    # weight update. See docs/plans/2026-04-28-cluster-b1-d1d2-asymmetry-
    # implementation.md.
    enable_d1_d2_asymmetry: bool = False

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Initialize per-type STP defaults if not provided
        # Defaults: cortical-style depression for E->E/E->I, weaker for I->E/I->I
        if self.stp_U_per_type is None:
            self.stp_U_per_type = [0.5, 0.5, 0.25, 0.25]       # E->E, E->I, I->E, I->I
        if self.stp_tau_d_per_type is None:
            self.stp_tau_d_per_type = [200.0, 200.0, 100.0, 100.0]  # ms
        if self.stp_tau_f_per_type is None:
            self.stp_tau_f_per_type = [20.0, 20.0, 50.0, 50.0]      # ms

        errors = []

        # Time parameters
        if self.dt_ms <= 0:
            errors.append(f"dt_ms must be positive, got {self.dt_ms}")
        if self.dt_ms > 0.1 and self.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
            errors.append(f"dt_ms={self.dt_ms}ms is UNSAFE for Hodgkin-Huxley (max 0.1ms for stability). "
                          f"HH gating kinetics have time constants ~0.1-1ms at 37C; dt must resolve these.")
        if self.total_simulation_time_ms <= 0:
            errors.append(f"total_simulation_time_ms must be positive, got {self.total_simulation_time_ms}")

        # Network parameters
        if self.num_neurons <= 0:
            errors.append(f"num_neurons must be positive, got {self.num_neurons}")
        if self.connections_per_neuron < 0:
            errors.append(f"connections_per_neuron cannot be negative, got {self.connections_per_neuron}")
        if self.num_traits <= 0:
            errors.append(f"num_traits must be positive, got {self.num_traits}")

        # Learning rate validations
        if self.hebbian_learning_rate < 0:
            errors.append(f"hebbian_learning_rate cannot be negative, got {self.hebbian_learning_rate}")
        if self.reward_learning_rate < 0:
            errors.append(f"reward_learning_rate cannot be negative, got {self.reward_learning_rate}")
        if self.stdp_a_plus < 0:
            errors.append(f"stdp_a_plus cannot be negative, got {self.stdp_a_plus}")
        if self.stdp_a_minus < 0:
            errors.append(f"stdp_a_minus cannot be negative, got {self.stdp_a_minus}")

        # Weight bounds
        if self.hebbian_min_weight > self.hebbian_max_weight:
            errors.append(f"hebbian_min_weight ({self.hebbian_min_weight}) > hebbian_max_weight ({self.hebbian_max_weight})")
        if self.stdp_w_min > self.stdp_w_max:
            errors.append(f"stdp_w_min ({self.stdp_w_min}) > stdp_w_max ({self.stdp_w_max})")

        # Plasticity parameters
        if self.stp_U < 0 or self.stp_U > 1:
            errors.append(f"stp_U must be in [0, 1], got {self.stp_U}")
        if self.stp_tau_d <= 0:
            errors.append(f"stp_tau_d must be positive, got {self.stp_tau_d}")
        if self.stp_tau_f <= 0:
            errors.append(f"stp_tau_f must be positive, got {self.stp_tau_f}")

        # Structural plasticity
        if self.struct_plast_target_density < 0 or self.struct_plast_target_density > 1:
            errors.append(f"struct_plast_target_density must be in [0, 1], got {self.struct_plast_target_density}")

        # Raise all errors together
        if errors:
            raise ValueError("CoreSimConfig validation failed:\n  - " + "\n  - ".join(errors))

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class VisualizationConfig:
    """Holds parameters for visualization, such as camera and volume."""
    volume_min_x: float = -50.0; volume_max_x: float = 50.0
    volume_min_y: float = -50.0; volume_max_y: float = 50.0
    volume_min_z: float = -50.0; volume_max_z: float = 50.0
    camera_center_x: float = 0.0; camera_center_y: float = 0.0; camera_center_z: float = 0.0
    camera_radius: float = 150.0
    camera_azimuth_angle: float = 0.0
    camera_elevation_angle: float = 0.0
    camera_up_x: float = 0.0; camera_up_y: float = 1.0; camera_up_z: float = 0.0
    camera_fov: float = 60.0
    camera_near_clip: float = 0.1
    camera_far_clip: float = 1000.0
    mouse_last_x: int = 0; mouse_last_y: int = 0
    mouse_left_button_down: bool = False
    mouse_right_button_down: bool = False
    viz_update_interval_steps: int = 17  # Update visualization every N steps (~60fps at dt=1.0ms)


@dataclass
class RuntimeState:
    """Holds the dynamic state of the simulation run. Not typically saved in profiles."""
    current_time_ms: float = 0.0
    current_time_step: int = 0
    is_running: bool = False
    is_paused: bool = False
    simulation_speed_factor: float = 1.0
    neuron_positions_x: List[float] = field(default_factory=list)
    neuron_positions_y: List[float] = field(default_factory=list)
    neuron_types_list_for_viz: List[str] = field(default_factory=list)
    max_delay_steps: int = 200
    actual_seed_used: int = -1  # Actual RNG seed used (for reproducibility)


@dataclass
class GPUConfig:
    """GPU-specific performance and memory features."""
    # Recording modes
    enable_gpu_buffered_recording: bool = True
    recording_mode: str = "gpu_buffered"  # "gpu_buffered", "streaming", "disabled"
    max_recording_memory_fraction: float = 0.6  # Fraction of free GPU memory for recording
    recording_compression: str = "lz4"  # "lz4", "gzip", "none" - LZ4 is 5-10x faster
    recording_compression_level: int = 1  # 1-9 for gzip (lower=faster), ignored for lz4
    enable_parallel_compression: bool = True  # Use ThreadPoolExecutor for batch writes
    parallel_compression_workers: int = 4  # Number of worker threads for compression
    enable_delta_encoding: bool = False  # Store only changed values (experimental)
    delta_keyframe_interval: int = 100  # Full frame every N frames when delta encoding
    delta_threshold: float = 0.001  # Values must change by this much to store in delta

    # Large-scale recording options (for 100K+ neuron simulations)
    recording_skip_synaptic_data: bool = False  # Skip connection weights and STP arrays (16x smaller frames)
    recording_frame_skip: int = 1  # Record every Nth frame (1 = every frame, 10 = every 10th)
    streaming_write_batch_size: int = 10  # Write frames in batches when streaming
    streaming_async_write: bool = True  # Use background thread for async disk writes

    # Recording memory safety
    recording_memory_check_interval: int = 50  # Check memory every N frames during recording
    recording_gpu_memory_limit: float = 0.85  # Auto-pause when GPU usage exceeds this
    recording_cpu_memory_limit: float = 0.90  # Auto-pause when CPU RAM usage exceeds this
    recording_auto_pause_on_memory: bool = True  # Auto-pause simulation when memory critical

    # Playback modes
    enable_gpu_buffered_playback: bool = True
    playback_mode: str = "gpu_cached"  # "gpu_cached", "streaming", "auto"
    playback_cache_chunk_size: int = 100  # Frames per batch when loading cache
    enable_playback_prefetch: bool = True  # Prefetch next N frames during streaming
    playback_prefetch_count: int = 10  # Number of frames to prefetch ahead

    # Rendering performance
    render_vbo_update_skip: int = 2  # Update VBOs every Nth render frame (1=every, 2=every other, etc.)

    # CUDA-OpenGL interop
    enable_cuda_gl_interop: bool = True
    cuda_gl_fallback_on_error: bool = True

    # Memory management
    memory_pool_limit_fraction: float = 0.8  # Max fraction of GPU memory for mempool
    enable_adaptive_quality: bool = True  # Reduce quality under memory pressure
    memory_pressure_threshold: float = 0.9  # Trigger cleanup above this usage
    memory_warning_threshold: float = 0.8  # Log warning above this usage

    # GPU connection generation (future)
    enable_gpu_connectivity_generation: bool = False  # Placeholder for future work
    enable_gpu_synapse_filtering: bool = False  # Placeholder for future work

    # Performance profiling
    enable_profiling: bool = False  # Disabled by default for production
    profiling_window_size: int = 100  # Number of steps to keep in timing deques
    profiling_detailed: bool = False  # Log per-kernel timings

    # Performance tuning
    stats_sync_interval_steps: int = 17  # Sync GPU stats every N steps (default ~60Hz at dt=1ms)
    max_steps_per_batch: int = 60  # Max simulation steps before yielding to UI
    data_update_interval_steps: int = 1  # Steps between GUI data updates

    # Debug mode
    enable_debug_checks: bool = False  # Enable inf/nan checking (performance impact)
    enable_step_profiler: bool = False  # Log per-section timing for performance analysis

    # Structural plasticity optimization
    struct_plast_compaction_interval: int = 1000  # Steps between CSR compaction
    synapse_capacity_growth_factor: float = 1.5  # Pre-allocation growth factor

    def __post_init__(self):
        """Validate GPU configuration parameters."""
        errors = []

        # Memory fractions must be in valid range
        if not 0 < self.memory_pool_limit_fraction <= 1:
            errors.append(f"memory_pool_limit_fraction must be in (0, 1], got {self.memory_pool_limit_fraction}")
        if not 0 < self.max_recording_memory_fraction <= 1:
            errors.append(f"max_recording_memory_fraction must be in (0, 1], got {self.max_recording_memory_fraction}")
        if not 0 < self.memory_pressure_threshold <= 1:
            errors.append(f"memory_pressure_threshold must be in (0, 1], got {self.memory_pressure_threshold}")
        if not 0 < self.memory_warning_threshold <= 1:
            errors.append(f"memory_warning_threshold must be in (0, 1], got {self.memory_warning_threshold}")

        # Validate recording memory safety limits
        if not 0 < self.recording_gpu_memory_limit <= 1:
            errors.append(f"recording_gpu_memory_limit must be in (0, 1], got {self.recording_gpu_memory_limit}")
        if not 0 < self.recording_cpu_memory_limit <= 1:
            errors.append(f"recording_cpu_memory_limit must be in (0, 1], got {self.recording_cpu_memory_limit}")
        if self.recording_memory_check_interval < 1:
            errors.append(f"recording_memory_check_interval must be >= 1, got {self.recording_memory_check_interval}")

        # Validate intervals
        if self.stats_sync_interval_steps < 1:
            errors.append(f"stats_sync_interval_steps must be >= 1, got {self.stats_sync_interval_steps}")
        if self.max_steps_per_batch < 1:
            errors.append(f"max_steps_per_batch must be >= 1, got {self.max_steps_per_batch}")
        if self.struct_plast_compaction_interval < 1:
            errors.append(f"struct_plast_compaction_interval must be >= 1, got {self.struct_plast_compaction_interval}")

        # Validate recording/playback modes
        valid_recording_modes = {"gpu_buffered", "streaming", "disabled"}
        if self.recording_mode not in valid_recording_modes:
            errors.append(f"recording_mode must be one of {valid_recording_modes}, got '{self.recording_mode}'")
        valid_playback_modes = {"gpu_cached", "streaming", "auto"}
        if self.playback_mode not in valid_playback_modes:
            errors.append(f"playback_mode must be one of {valid_playback_modes}, got '{self.playback_mode}'")

        if errors:
            raise ValueError("GPUConfig validation failed:\n  - " + "\n  - ".join(errors))


# --- Experiment System Dataclasses ---

@dataclass
class StimulusPattern:
    """Defines a single stimulus waveform.

    All amplitudes are in picoamperes (pA), consistent with simulator units.
    """
    pattern_type: str = StimulusPatternType.CONSTANT.name
    amplitude_pA: float = 100.0       # Peak amplitude

    # Pulse train parameters
    pulse_frequency_hz: float = 20.0  # Pulse repetition rate
    pulse_duration_ms: float = 2.0    # Each pulse width

    # Sinusoidal parameters
    frequency_hz: float = 10.0        # Oscillation frequency
    phase_offset_rad: float = 0.0     # Phase offset
    dc_offset_pA: float = 0.0         # DC baseline offset

    # Ramp parameters
    start_amplitude_pA: float = 0.0   # Ramp start
    end_amplitude_pA: float = 200.0   # Ramp end

    # Poisson spike train parameters
    poisson_rate_hz: float = 50.0     # Mean firing rate of Poisson process
    spike_current_pA: float = 200.0   # Current per spike event
    spike_duration_ms: float = 1.0    # Duration of each spike current pulse

    # Gaussian noise parameters
    noise_mean_pA: float = 0.0
    noise_std_pA: float = 50.0

    # Custom waveform (time_ms, amplitude_pA pairs -- interpolated)
    custom_waveform_times_ms: List[float] = field(default_factory=list)
    custom_waveform_values_pA: List[float] = field(default_factory=list)

    # Per-neuron Poisson rate vector (for RATE_VECTOR_POISSON pattern).
    # Length must equal the number of target neurons in the channel.
    # Each target neuron fires Poisson with its own rate (same order as
    # target_neuron_indices).
    rate_vector_hz: List[float] = field(default_factory=list)


@dataclass
class StimulusChannel:
    """Maps a StimulusPattern to target neurons with timing.

    Multiple channels can be active simultaneously, targeting different
    neuron groups with different patterns (e.g., CS to input, US to output).
    """
    name: str = "channel_0"
    pattern: StimulusPattern = field(default_factory=StimulusPattern)

    # Targeting
    target_group_name: str = ""              # NeuronGroup name (preferred)
    target_neuron_indices: List[int] = field(default_factory=list)  # Direct indices (override)
    target_trait_index: int = -1             # Target by trait (-1 = all)
    target_fraction: float = 1.0            # Fraction of target group to stimulate (0-1)

    # Timing
    onset_ms: float = 0.0                   # Start time relative to phase/trial start
    duration_ms: float = 1000.0             # How long the stimulus is active
    repeat_period_ms: float = 0.0           # If > 0, stimulus repeats with this period (for trial-based phases)

    # Noise overlay
    add_trial_noise: bool = False           # Add per-trial amplitude jitter
    trial_noise_std_fraction: float = 0.1   # Fraction of amplitude as noise std

    enabled: bool = True


@dataclass
class NeuronGroup:
    """A designated population of neurons with a functional role.

    Groups are defined by their indices into the network's neuron array.
    The role determines how the group interacts with stimulus/readout systems.
    """
    name: str = "group_0"
    role: str = NeuronGroupRole.HIDDEN.name
    neuron_indices: List[int] = field(default_factory=list)

    # Auto-population rules (used when indices not specified directly)
    trait_index: int = -1              # Populate from trait (-1 = manual)
    index_start: int = 0              # Range-based population
    index_end: int = 0
    fraction_of_trait: float = 1.0    # Use only a fraction of the trait

    # Visual distinction
    highlight_color: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.0, 1.0])  # RGBA


@dataclass
class ReadoutConfig:
    """Configuration for network response measurement."""
    # Firing rate readout
    rate_window_ms: float = 50.0           # Sliding window for rate estimation
    rate_group_names: List[str] = field(default_factory=list)  # Groups to measure

    # Spike count readout
    spike_count_window_ms: float = 100.0   # Window for spike counting

    # Power spectral density
    enable_psd: bool = False
    psd_window_ms: float = 500.0           # FFT window
    psd_freq_range_hz: List[float] = field(default_factory=lambda: [1.0, 200.0])

    # Cross-correlation
    enable_cross_correlation: bool = False
    correlation_max_lag_ms: float = 50.0
    correlation_group_pairs: List[List[str]] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Configuration for training protocols.

    Scientific grounding:
    - Associative: Rescorla-Wagner 1972, Bi & Poo 1998 (STDP timing rules)
    - R-STDP: Izhikevich 2007 Ch.7, Fremaux et al. 2013
    - Supervised: Pfister et al. 2006 (target rate learning)
    - Reservoir: Maass et al. 2002, Jaeger & Haas 2004
    """
    mode: str = TrainingMode.NONE.name

    # Trial structure
    num_trials: int = 100
    trial_duration_ms: float = 500.0       # Single trial length
    inter_trial_interval_ms: float = 200.0 # Rest between trials

    # Associative pairing (CS-US)
    cs_channel_name: str = ""              # Conditioned stimulus channel
    us_channel_name: str = ""              # Unconditioned stimulus channel
    cs_us_delay_ms: float = 100.0          # Delay between CS onset and US onset
    cr_threshold_hz: float = 8.0           # Conditioned response detection threshold (Hz)

    # Reinforcement learning
    reward_delay_ms: float = 50.0          # Delay after response to deliver reward
    reward_magnitude: float = 1.0          # Reward signal strength
    punishment_magnitude: float = -0.5     # Punishment signal strength
    target_output_group: str = ""          # Output group to evaluate
    target_min_rate_hz: float = 10.0       # Min rate for "correct" response
    target_max_rate_hz: float = 50.0       # Max rate for "correct" response

    # Supervised target matching
    target_rates_per_group: Dict[str, float] = field(default_factory=dict)  # {group_name: target_hz}
    supervised_error_gain: float = 0.01    # Error signal scaling

    # Reservoir computing
    reservoir_freeze_weights: bool = True  # Freeze recurrent weights
    readout_learning_rate: float = 0.01    # Readout weight update rate
    readout_regularization: float = 1e-4   # L2 regularization

    # Evaluation
    eval_window_ms: float = 100.0          # Response evaluation window
    eval_delay_ms: float = 50.0            # Delay after stimulus onset before evaluation
    success_threshold: float = 0.9         # Fraction of correct trials for convergence


@dataclass
class ExperimentPhase:
    """A single phase in a multi-phase experiment."""
    name: str = "phase_0"
    phase_type: str = ExperimentPhaseType.BASELINE.name
    duration_ms: float = 5000.0

    # Which stimulus channels are active during this phase
    active_channels: List[str] = field(default_factory=list)

    # Training config for TRAINING phases
    training_config: TrainingConfig = field(default_factory=TrainingConfig)

    # Phase-specific overrides
    enable_plasticity: bool = True         # Allow weight changes
    record_data: bool = True               # Log readout data

    # Repeat control (for trial-based phases)
    num_repetitions: int = 1               # Repeat this phase N times


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    An experiment consists of:
    1. Neuron group definitions (input/output/hidden populations)
    2. Stimulus channels (patterns mapped to groups with timing)
    3. Phases (ordered sequence of baseline/stimulus/training/testing)
    4. Readout configuration (what to measure)
    """
    name: str = "Untitled Experiment"
    description: str = ""

    # Component definitions
    neuron_groups: List[NeuronGroup] = field(default_factory=list)
    stimulus_channels: List[StimulusChannel] = field(default_factory=list)
    phases: List[ExperimentPhase] = field(default_factory=list)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)

    # Global settings
    random_seed: int = -1                   # Experiment RNG seed (-1 = random)
    save_experiment_log: bool = True
    log_trial_details: bool = True          # Log per-trial metrics

    # Experiment-level simulation overrides (restored when experiment stops).
    # These allow experiments to temporarily adjust network parameters for
    # adequate signal-to-noise without permanently modifying global config.
    override_propagation_strength: float = -1.0     # -1 = use global default
    override_inhibitory_prop_strength: float = -1.0  # -1 = use global default

    enabled: bool = False                   # Master enable for experiment system


# --- Config Helper Functions ---

def _create_config_from_dict(config_cls, data_dict):
    """Helper to create a dataclass instance from a dictionary, ignoring extra keys."""
    if not data_dict:
        return config_cls()

    # Get the field names defined in the dataclass
    class_fields = {f.name for f in fields(config_cls)}

    # Filter the input dictionary to only include keys that are fields in the class
    filtered_data = {k: v for k, v in data_dict.items() if k in class_fields}

    return config_cls(**filtered_data)


def _get_full_config_dict(core_cfg, viz_cfg, runtime_state, gpu_cfg=None):
    """Helper to combine all config objects into a single dictionary for saving."""
    result = {
        "core_config": asdict(core_cfg),
        "viz_config": asdict(viz_cfg),
        "runtime_state": asdict(runtime_state)
    }
    if gpu_cfg is not None:
        result["gpu_config"] = asdict(gpu_cfg)
    return result
