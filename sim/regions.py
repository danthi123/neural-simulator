"""Brain-region framework (Session E.2).

A first-class framework for declaring multiple cortical / subcortical
populations that share a single SimulationBridge. Each `BrainRegion`
owns a contiguous slice of the neuron-index space; each `RegionPathway`
declares cross-region projections with optional neuromodulator gating.

Default OFF: when CoreSimConfig.brain_regions is empty (which is the
default), the bridge runs as a single population — today's behavior
unchanged.

Composes with sim/neuromodulators.py from Session E.1: pathways can
declare `neuromodulator_gates=["dopamine"]` to make their plasticity
rate depend on a specific neuromodulator's concentration. Regions
register themselves as neuron groups with the NeuromodulatorManager
so target scope `group:NAME` resolves naturally.

See:
- docs/plans/2026-04-24-brain-region-framework.md
- sim/neuromodulators.py (E.1, composes with this)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class BrainRegion:
    """One brain region: a population of neurons with local connectivity.

    name:
        Unique identifier. Also registered as a neuron-group name with
        the experiment engine and neuromodulator manager so target
        scopes like `group:PFC` resolve here.

    n_neurons:
        Number of neurons. Allocated as a contiguous slice of the
        global neuron-index space; concatenation order matches the
        order in core_config.brain_regions.

    exc_fraction:
        Fraction excitatory (rest inhibitory). 0.8 matches cortical
        layer 2/3 (Markram et al. 2015).

    internal_density:
        Fraction of all-pairs internal connections that exist
        (sparse Erdős-Rényi within the region).

    exc_weight_mean, inh_weight_mean:
        Mean weight of internal excitatory / inhibitory connections.

    weight_jitter:
        Relative std of normal noise around the means (0.2 = 20%).

    plastic_internal:
        Whether internal synapses are plastic (subject to STDP and
        reward modulation). False (reservoir style) for sensorimotor
        regions; True for cortical learning regions like PFC working
        memory.

    nm_outputs:
        List[str] of neuromodulator names this region produces. Used
        by future `from_region_activity` production rules. Currently
        informational; integrates with neuromodulator subsystem
        production rules in a later task.
    """

    name: str
    n_neurons: int
    exc_fraction: float = 0.8
    internal_density: float = 0.1
    exc_weight_mean: float = 0.3
    inh_weight_mean: float = 0.8
    weight_jitter: float = 0.2
    plastic_internal: bool = False
    nm_outputs: List[str] = field(default_factory=list)
    # Per-region neuron type override. If set, the bridge uses this
    # NeuronType enum name when initializing neurons in this region's
    # index slice. Allows e.g. striatum_D1 region to use IZH2007_STRIATAL_MSN_D1
    # while motor region uses IZH2007_RS_CORTICAL_PYRAMIDAL.
    # Falls back to cfg.default_neuron_type_izh / _hh / _adex if None.
    # 2026-04-25: required for Phase B (BG action selection module).
    izh_neuron_type: str = None
    hh_neuron_type: str = None
    adex_neuron_type: str = None

    # Optional population-scoped overrides for the HH spike-generating and
    # passive membrane parameters. None preserves the selected HH preset.
    # Conductances use mS/cm^2, capacitance uses uF/cm^2, and reversals use mV.
    hh_C_m_override: Optional[float] = None
    hh_g_Na_max_override: Optional[float] = None
    hh_g_K_max_override: Optional[float] = None
    hh_g_L_override: Optional[float] = None
    hh_E_Na_override: Optional[float] = None
    hh_E_K_override: Optional[float] = None
    hh_E_L_override: Optional[float] = None

    # Construction-time effective intrinsic drive in pA. This reduced-model
    # field represents unresolved cell-autonomous conductances; it is not a
    # sensory stimulus and is currently supported only by Izhikevich regions.
    intrinsic_current_pA: float = 0.0

    # Optional SNr pacemaker conductance bundle. Maxima use the HH path's
    # conductance-density convention (mS/cm^2). All-zero is strictly disabled;
    # runtime state is introduced only by the later bridge/kernel slice.
    snr_g_nalcn_max: float = 0.0
    snr_g_nap_max: float = 0.0
    snr_g_ca_max: float = 0.0
    snr_g_sk_max: float = 0.0
    snr_g_h_max: float = 0.0

    # Authenticated packet mode stores references only. Paths are canonical
    # POSIX-relative paths below the explicit simulation source root; the
    # digest pins the exact packet bytes. Legacy maxima and packet mode are
    # mutually exclusive so mixed-region simulations cannot silently blend
    # two authorities inside one region.
    snr_executable_packet_path: Optional[str] = None
    snr_executable_packet_sha256: Optional[str] = None

    # Per-region GABA_A reversal potential override in mV. None = use global
    # cfg.syn_reversal_potential_i. Used to model regions with different
    # chloride homeostasis (e.g., striatal MSNs ~−60 mV per PBR-160 ch 6;
    # SNc DA ~−55 mV per ch 11). MSNs lack the deep negative ECl seen in
    # cortical pyramidals: gramicidin perforated patch measurements give
    # ~-60 mV, producing shunting (depolarizing-near-rest, hyperpolarizing-
    # near-threshold) inhibition. SNc DA neurons lack KCC2 entirely.
    syn_reversal_potential_i_override: Optional[float] = None

    # Cluster G v2 (2026-05-01): per-region NMDA enable. When True, this
    # region's neurons participate in NMDA-mediated dynamics (Wang 2002
    # bistability). When False (default), this region's neurons do NOT get
    # NMDA conductance even if cfg.enable_nmda is True globally.
    #
    # Motivation: cfg.enable_nmda + cfg.nmda_ratio are global. v1 turning
    # them on for "PFC working memory" actually applies to all regions,
    # destabilizing hippocampus (D v1, D v2 SWR) and other recurrent
    # circuits (~11x worse cheat-5 results when D + global NMDA stacked).
    # Per-region NMDA: PFC has it on, hippocampus + cerebellum + cortex
    # do not. Biology source: Wang 2002 says PFC has elevated NMDA-NR2B
    # specifically; other cortical areas have less.
    enable_nmda: bool = False

    # Per-region homeostasis enable (2026-06-08). Mirrors enable_nmda. When
    # True, this region's neurons use the adapted cp_neuron_firing_thresholds
    # (intrinsic homeostatic plasticity / excitability homeostasis, an EMA
    # threshold update — deterministic, no randomness) as their spike
    # threshold AND have their thresholds adapted each step, EVEN WHEN the
    # global cfg.enable_homeostasis is False. When False (default), this
    # region's neurons use the fixed cp_izh_vpeak threshold (unless global
    # homeostasis is on, in which case the global path applies as before).
    #
    # Motivation: the deterministic-nav regime sets cfg.enable_homeostasis=False
    # (g11_bg_runner.py:3340), which makes an under-active MSN-D1 value critic
    # unable to fire from its place afferent — its KIR2-clamped rest-to-threshold
    # gap (vr=-80, vt=-25) is unreachable through the afferent without a way to
    # operate the cell in a firing range. Per-region homeostasis ON the critic
    # ONLY restores firing (forensic: fire + learn + place-graded) while keeping
    # the global determinism the nav eval relies on. Biology source: intrinsic
    # homeostatic plasticity (Desai 1999; Turrigiano) is a real, deterministic,
    # cell-autonomous mechanism letting a neuron operate in its firing range.
    # See research/findings/2026-06-08-navfaithful-derisk-FAIL-homeostasis-confound.md.
    enable_homeostasis: bool = False

    # Per-region parameter HETEROGENEITY enable (2026-06-18). Mirrors
    # enable_homeostasis / enable_nmda exactly. When True, this region's
    # neurons receive per-neuron jittered parameter samples (the
    # cfg.heterogeneity_distributions draws — the het-ON graded band) EVEN
    # WHEN the global cfg.enable_parameter_heterogeneity is False. When None
    # (default), this region follows the global flag: global ON => jittered
    # like every other neuron (legacy); global OFF => this region's neurons
    # keep their deterministic per-region preset values (set by
    # _apply_per_region_neuron_types). enable_heterogeneity=True is therefore
    # the per-region complement of the global flag, used to give ONE critic
    # region the het-ON graded operating band without perturbing the het-OFF
    # determinism that the merged nav/conv eval relies on.
    #
    # Motivation (merged-bridge TD cue-shift consolidation, roadmap #3): the
    # merged het-OFF config (5a stdp_w_max=400 conv-weight clip + per-region
    # homeostasis low threshold) forces the td_striosome MSN-D1 critic ~6x
    # hotter than the standalone's het-ON config -> V SATURATES instead of
    # grading -> the TD peak stays stuck @ reward (migration r=-0.43, not the
    # r<-0.7 GO bar). The --global-het-test diagnostic confirmed het-ON
    # restores the graded critic + value-growth + reward-shrink + dip. A
    # per-region heterogeneity mask gives the critic the het-ON graded band
    # WITHOUT perturbing the het-OFF nav/conv determinism. Biology source:
    # cell-type-specific intrinsic parameter heterogeneity (Marder & Goaillard
    # 2006; Tripathy 2013) is a real, deterministic-per-seed property; applying
    # it to one cell population while another stays homogeneous is biologically
    # legitimate. See research/findings/2026-06-18-merged-TD-cueshift-
    # consolidation-BOUNDARY.md.
    enable_heterogeneity: Optional[bool] = None

    # Cluster C v2 (2026-04-29): per-action DA compartmentalization.
    # When a region is action-specific (cortex_X, str_D1_X, str_D2_X,
    # gpi_X, thal_X, motor_X, etc), this is the action index in [0, N-1]
    # corresponding to the action channel. None for global / non-action-
    # specific regions (sensory, place_cells, stn, dopamine, hippocampus,
    # PFC, etc.).
    #
    # Used by inject_explicit_wiring() to populate cp_synapse_action_tag
    # so per-action DA modulators can target only synapses with their
    # action_index. See docs/plans/2026-04-29-cluster-c-v2-compartmentalized-
    # da-design.md.
    action_index: Optional[int] = None

    # Cluster E v1 (2026-04-29): topographic maps + distance-dependent
    # connection probability. When coordinate_dim > 0, this region's neurons
    # are deterministically assigned coordinates uniformly in coordinate_extent.
    # Coordinates are then used by RegionPathway.distance_sigma to sample
    # connections with Gaussian-weighted probability.
    #
    # coordinate_dim:
    #   0 (default) = no coordinates, no topography. Backward-compatible —
    #     pathways into/out of this region use uniform Bernoulli connectivity.
    #   1 = 1D layout (e.g., a tonotopic axis or motor-strip line)
    #   2 = 2D layout (e.g., retinotopic / somatotopic / motor-map sheet)
    #
    # coordinate_extent:
    #   None (default) = unit extent (1.0,) or (1.0, 1.0) inferred from
    #     coordinate_dim. Otherwise a tuple of length coordinate_dim giving
    #     the per-axis extent. Coordinates are sampled uniformly from
    #     [0, extent_k] on each axis k.
    #
    # See docs/plans/2026-04-29-cluster-e-topographic-maps-design.md.
    coordinate_dim: int = 0
    coordinate_extent: Optional[Tuple[float, ...]] = None
    # Cluster E v1: optional explicit center for placing all neurons in this
    # region at a single point in coordinate space. Useful when a region
    # represents a discrete location on a larger map (e.g., cortex_N at the
    # north corner of a unit square). When set, all neurons in this region
    # share these coordinates. coordinate_dim must equal len(coordinate_center).
    coordinate_center: Optional[Tuple[float, ...]] = None

    # Graded LGN decorrelation (2026-06-06): opt this region into the GRADED
    # pairwise lateral inhibition stage (the analog, pre-spike whitening the
    # retina/LGN does). When True AND cfg.enable_graded_lateral is True, the
    # bridge allocates a per-region dense plastic lateral matrix M (n_neurons x
    # n_neurons) and, BEFORE the spike threshold each step, adds the graded
    # recurrent inhibition -(M @ a) to this region's input current, where
    # a = relu((v - v_rest) / act_scale) is the region's SUB-THRESHOLD analog
    # activity (NOT spikes). M learns ΔM ∝ ⟨a aᵀ⟩ - I - λM (anti-Hebbian on
    # graded co-activity + identity target + weight-decay). Default False:
    # zero effect on every existing region/run (the new code is a guarded
    # no-op unless BOTH this flag and the global cfg.enable_graded_lateral
    # are set). See docs/plans/2026-06-06-graded-lgn-decorrelation-design.md.
    graded_lateral: bool = False

    # Slow per-hub INPUT-MEAN adaptation (axis-0 per-feature centering, 2026-06-15). When True
    # AND cfg.enable_input_mean_adapt is True, this region's neurons each subtract a SLOW running
    # mean of their OWN pre-threshold input drive (synaptic + external current) from that drive,
    # BEFORE the spike threshold: adapted = raw_drive - gain*m; m <- (1-alpha)*m + alpha*raw_drive
    # (causal -- subtract the current m, then update from raw_drive). This is the SEPARABLE
    # diagonal/DC half of whitening (per-FEATURE mean-centering = subtractive spike-frequency
    # adaptation / point-neuron predictive coding; Lee/Pennartz 2024, PMC11045951) -- the per-
    # feature centering the L1 learned cortex needs (a common-mode pool does the WRONG axis-1
    # per-concept removal). Mirrors BrainRegion.enable_nmda / enable_homeostasis: the bridge
    # builds a per-neuron boolean mask cp_input_mean_adapt_mask from the regions that set this,
    # and only those neurons adapt. Default False: zero effect on every existing region/run (the
    # new code is a guarded no-op -- cp_input_mean_ema stays None -- unless BOTH this flag and the
    # global cfg.enable_input_mean_adapt are set). See
    # research/findings/2026-06-15-slow-perhub-mean-primitive-deep-research.md (Option A) +
    # docs/plans/2026-06-15-analog-substrate-learned-cortex-build-plan.md (Phase 2).
    input_mean_adapt: bool = False
    # Per-concept DIVISIVE input normalization (Carandini-Heeger, 2026-06-15): when True AND
    # cfg.enable_input_divisive_norm, this region's neurons divide their pre-threshold input by
    # (sigma + gain*mean input over the flagged set) -- PPMI's per-concept normalization as a
    # feedforward divisive-gain circuit. Guarded no-op unless BOTH this flag and the global
    # cfg.enable_input_divisive_norm are set (cp_input_divisive_mask stays None otherwise).
    input_divisive_norm: bool = False
    # SECOND, INDEPENDENT divisive-norm pool (Cascade-accumulator FIX A, 2026-06-20): a byte-identical
    # clone of input_divisive_norm keyed by cfg.enable_input_divisive_norm_2 + sigma_2/gain_2, so a
    # SEPARATE set of regions (the four sel_X selection accumulators) is normalized as its OWN pool,
    # independent of the cortex_X bump-mass pool used by the #6 SC popvector read-out. Guarded no-op
    # unless BOTH this flag and cfg.enable_input_divisive_norm_2 are set (cp_input_divisive_mask_2 stays
    # None otherwise). See research/findings/2026-06-20-shortcut6-FIXA-divnorm-accumulator.md.
    input_divisive_norm_2: bool = False

    def __post_init__(self) -> None:
        positive_hh_fields = (
            "hh_C_m_override",
            "hh_g_Na_max_override",
            "hh_g_K_max_override",
            "hh_g_L_override",
        )
        for field_name in positive_hh_fields:
            value = getattr(self, field_name)
            if value is None:
                continue
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{field_name} must be finite and positive, got {value}"
                )

        for field_name in (
            "hh_E_Na_override",
            "hh_E_K_override",
            "hh_E_L_override",
        ):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite, got {value}")

        for field_name in (
            "snr_g_nalcn_max",
            "snr_g_nap_max",
            "snr_g_ca_max",
            "snr_g_sk_max",
            "snr_g_h_max",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{field_name} must be finite and nonnegative, got {value}"
                )

        packet_path = self.snr_executable_packet_path
        packet_sha256 = self.snr_executable_packet_sha256
        if (packet_path is None) != (packet_sha256 is None):
            raise ValueError(
                "snr_executable_packet_path and sha256 must be set together"
            )
        if packet_path is not None:
            if (
                not isinstance(packet_path, str)
                or not packet_path
                or packet_path != packet_path.strip()
                or "\\" in packet_path
                or "\x00" in packet_path
                or any(ord(character) > 127 for character in packet_path)
            ):
                raise ValueError(
                    "snr_executable_packet_path must be trimmed ASCII POSIX text"
                )
            parsed = PurePosixPath(packet_path)
            if (
                parsed.is_absolute()
                or str(parsed) != packet_path
                or any(part in {"", ".", ".."} for part in parsed.parts)
            ):
                raise ValueError(
                    "snr_executable_packet_path must be canonical and relative"
                )
            if (
                not isinstance(packet_sha256, str)
                or len(packet_sha256) != 64
                or any(character not in "0123456789abcdef" for character in packet_sha256)
            ):
                raise ValueError(
                    "snr_executable_packet_sha256 must be a lowercase SHA-256 digest"
                )
            if self.snr_conductance_bundle_enabled:
                raise ValueError(
                    "SNr packet references cannot be combined with legacy conductance maxima"
                )

    @property
    def snr_executable_packet_enabled(self) -> bool:
        return self.snr_executable_packet_path is not None

    @property
    def snr_conductance_bundle_enabled(self) -> bool:
        """Whether this region requests any part of the SNr channel bundle."""
        return any(
            value > 0.0
            for value in (
                self.snr_g_nalcn_max,
                self.snr_g_nap_max,
                self.snr_g_ca_max,
                self.snr_g_sk_max,
                self.snr_g_h_max,
            )
        )


@dataclass
class RegionPathway:
    """Directed projection from one region to another.

    from_region, to_region:
        BrainRegion.name strings. Both must exist in
        core_config.brain_regions.

    density:
        Fraction of pre-post pairs that have a synapse.

    weight_mean, weight_jitter:
        Mean weight + relative std (default 0.2 = 20%) of pathway
        synapses.

    plastic:
        Whether pathway synapses are plastic (subject to STDP +
        reward modulation). Cross-region projections default True
        so learning rules can shape them.

    neuromodulator_gates:
        List[str] of neuromodulator names that gate this pathway's
        plasticity rate. Each named modulator's
        `compute_plasticity_rate_multiplier()` contribution is
        multiplied with the global rate. Empty = no gating.

        Biological analogue: D1 corticostriatal LTP is gated by
        phasic dopamine; cortical LTP can be gated by acetylcholine
        attention signals. This field implements that as a config
        knob.

    plasticity_gate:
        Optional name for a runtime-controllable plasticity gate. When
        set, all synapses in this pathway share a per-synapse plasticity
        gain that defaults to 1.0 (full plasticity) and can be modified
        at runtime via `bridge.set_plasticity_gate(name, value)`. Setting
        the gain to 0.0 freezes the pathway (no STDP, no eligibility
        accumulation, no reward-driven updates). Setting it back to 1.0
        thaws.

        Biological analogue: developmental staging (sensory cortex
        matures before association cortex), critical periods (visual
        cortex ocular dominance plasticity closes via PV interneuron
        maturation), and neuromodulator-gated plasticity windows. The
        gate is the abstraction; what controls it (a fixed schedule, a
        neuromodulator concentration, a developmental clock) is up to
        the runner / experiment configuration.

        None = always-on (current behavior, not added to any gate).
    """

    from_region: str
    to_region: str
    density: float = 0.5
    weight_mean: float = 1.0
    weight_jitter: float = 0.2
    plastic: bool = True
    neuromodulator_gates: List[str] = field(default_factory=list)
    plasticity_gate: str = None
    # transmission_gate (2026-06-03): optional name for a runtime-controllable MULTIPLICATIVE
    # TRANSMISSION gate. Unlike plasticity_gate (which freezes weight UPDATES only, leaving synaptic
    # CURRENT flowing), this scales the pathway's effective synaptic CURRENT in [0,1] via
    # bridge.set_transmission_gate(name, value). Use it to pre-wire a route with a fixed weight, hold it
    # normally CLOSED (gate=0, no current, no STDP cold-start), and OPEN it on command -- thalamocortical
    # dynamical gating: binding = which gate is open, not which weight grew (Logiaco-Abbott-Escola 2021).
    # None = always-on transmission (current behavior, not added to any transmission gate).
    transmission_gate: str = None

    # receptor (2026-06-08): which inhibitory receptor an inhibitory pathway's synapses use.
    #   "gaba_a" (default) — fast ionotropic Cl- current via the single g_i conductance,
    #     reversal = the post neuron's E_GABA (current behavior, byte-identical routing).
    #   "gaba_b" — slow metabotropic GABA_B -> GIRK K+ current via a SEPARATE conductance
    #     g_gabab (E_K ~ -90 mV, tau ~150 ms), independent of the chloride gradient, so it
    #     strongly hyperpolarizes KCC2-lacking DA cells where GABA_A is weak/shunting.
    #     Requires cfg.enable_gabab=True; the pathway's synapses are added to the per-synapse
    #     GABA_B mask and the post neurons' E_gabab is set to the configured K+ reversal.
    # See catalog J.11 (the previously-missing slow inhibitory channel) and the
    # GABA_B/GIRK conductance design doc (2026-06-08).
    receptor: str = "gaba_a"

    # exc_receptor (2026-06-09): which EXCITATORY receptor an excitatory pathway uses.
    #   "ampa" (default) -- fast ionotropic g_e (current behavior, byte-identical routing).
    #   "nmda_slow" -- slow NR2B-NMDA-dominant: the pathway's synapses feed a SEPARATE
    #     slow-NMDA conductance (Mg2+-block self-limiting, tau ~100ms) and their fast-AMPA
    #     g_e component is SUPPRESSED, so a recurrent can hold a graded reverberatory
    #     attractor (Wang 2001/2002) without the fast-AMPA synchronous runaway, while the
    #     feedforward detonator stays AMPA. Requires cfg.enable_nmda_recurrent=True; the
    #     pathway's synapses are added to the per-synapse nmda-recurrent routing mask.
    # The EXCITATORY mirror of the GABA_B/`receptor=` precedent above (the inhibitory one),
    # differing only in that the AMPA component is suppressed for routed synapses (GABA_B is
    # additive; nmda_slow replaces AMPA). See 2026-06-09-learned-graded-ca3-design.md.
    exc_receptor: str = "ampa"

    # coincidence_detector (2026-06-09): when True, this EXCITATORY pathway is a dendritic-COINCIDENCE
    # afferent -- each postsynaptic neuron forms an NMDA-spike SUBUNIT (Poirazi-Mel 2003; Major-Larkum-
    # Schiller 2013) over this pathway's synapses, firing a regenerative all-or-none plateau current when
    # >= cfg.coincidence_k_threshold of its routed inputs COINCIDE in one step. Lets a sparse-distinct
    # ensemble drive the target by coincidence, not rate (the point-neuron rate-coding wall). Requires
    # cfg.enable_coincidence_detection=True; the pathway's synapses are added to the per-synapse
    # coincidence-routing mask. The fast-AMPA g_e component is KEPT (unlike nmda_slow, which suppresses
    # it) -- the plateau is ADDITIVE on top, matching the NMDA spike riding the AMPA EPSP. Default False
    # = byte-identical routing. This is the coincidence sibling of the exc_receptor/nmda_slow precedent
    # above. See 2026-06-09-coincidence-substrate-upgrade-design.md.
    coincidence_detector: bool = False

    # graded (2026-06-15): when True, this pathway transmits with GRADED (analog, non-spiking) release --
    # the retina's horizontal/bipolar-cell mechanism. The per-step conductance increment on the target uses
    # the SOURCE neuron's CONTINUOUS activity -- a saturating sub-threshold readout of its membrane potential,
    # a_cont = clip((v - rest)/scale, 0, 1) (the same analog signal the graded_lateral mechanism uses; the
    # cm pool's depolarization tracks the population mean it must subtract) -- INSTEAD of its binary
    # cp_firing_states, so the source drives the target in proportion to its graded membrane state, bypassing
    # the spike threshold. The E/I routing is unchanged (an inhibitory source's graded drive feeds g_i, an
    # excitatory source's feeds g_e), with the same propagation scaling as the spike path. Motivation: a
    # SPIKING inhibitory pool cannot linearly track the population mean (depolarization block makes its spikes
    # anti-track the mean), so spike-mediated inhibition cannot do the common-mode removal (whitening) a
    # learned cortex needs -- but the source's analog membrane DOES track the mean. Requires no config flag: the
    # per-synapse graded mask (cp_graded_synapse_mask) is built iff at least one pathway sets graded=True;
    # otherwise it is None and the new step block is unreached -> byte-identical routing. The graded-routed
    # synapses are REMOVED from the spike matvec (they transmit gradedly, not on spikes), mirroring the
    # exc_receptor=="nmda_slow" AMPA-suppression precedent. Default False. See catalog E.05 (center-surround)
    # + Kandel retina (horizontal/bipolar graded potentials) + 2026-06-15-analog-substrate-learned-cortex
    # -build-plan.md (Phase 1).
    graded: bool = False

    # stp_disabled (2026-07-21, gap#5 mossy detonator): when True, this pathway's synapses SKIP
    # Tsodyks-Markram short-term depression -- their effective STP factor (stp_u*stp_x) is forced to
    # 1.0 (full base weight) in the step while ALL OTHER synapses keep STP. The global toggle
    # cfg.enable_short_term_plasticity is only all-or-nothing; some circuits need OPPOSITE STP states
    # at once -- e.g. a mossy dg->ca3 DETONATOR that must not depress (STP-off) co-resident with a
    # ca3->ca3 recurrent that MUST keep STP to avoid an avalanche. The STP STATE (u/x) still evolves for
    # these synapses; only the effective multiplier is overridden. Realized via a per-synapse boolean
    # mask cp_stp_disabled_mask built iff >=1 pathway sets stp_disabled=True (no config flag -- the
    # pathway flag alone is the opt-in, like the transmission_gate/graded precedents). Default False =>
    # the mask is None and the STP factor is computed exactly as before -> byte-identical. See
    # research/runners/_gap5_emergent_dg_selection_derisk.py.
    stp_disabled: bool = False

    # Cluster E v1 (2026-04-29): distance-dependent connection probability.
    # When set AND both source and target regions have coordinate_dim > 0,
    # connections are sampled with Gaussian-weighted probability:
    #     p(i, j) = density * exp(-||c_i - c_j||² / (2 * sigma²))
    # where c_i, c_j are the source and target neurons' coordinates.
    # When None (default) or either region lacks coordinates, falls back
    # to uniform Bernoulli sampling with `density` (current behavior,
    # backward compatible).
    #
    # See docs/plans/2026-04-29-cluster-e-topographic-maps-design.md.
    distance_sigma: Optional[float] = None


class RegionManager:
    """Owns per-region neuron-index allocation, inhibitory cell selection,
    and (later) wiring-plan generation.

    Lifecycle:
        mgr = RegionManager(regions, pathways)
        mgr.initialize(seed=42)              # allocate index ranges + inh
        plan = mgr.build_wiring_plan(rng=...)  # used by bridge.inject_explicit_wiring (Task 3+)
        mgr.region_indices_dict()             # for nm_mgr.set_group_indices

    Backward compat: an empty regions list yields total_neurons() == 0
    and an empty wiring plan, so the bridge falls through to the legacy
    single-population path.
    """

    def __init__(self,
                 regions: Sequence[BrainRegion],
                 pathways: Sequence[RegionPathway]):
        self._regions: List[BrainRegion] = list(regions)
        self._pathways: List[RegionPathway] = list(pathways)
        self._indices: Dict[str, List[int]] = {}
        self._inhibitory: Dict[str, List[int]] = {}
        self._total_neurons: int = 0
        # Cluster E v1: per-region neuron coordinates, list of tuples.
        # Empty when coordinate_dim == 0 for that region.
        self._coordinates: Dict[str, List[Tuple[float, ...]]] = {}

    def initialize(self, seed: int = 0) -> None:
        """Allocate contiguous index ranges for each region and pick
        inhibitory cells deterministically from `seed`. Also assigns
        topographic coordinates when coordinate_dim > 0 (Cluster E v1)."""
        rng = random.Random(seed)
        # Use a separate RNG for coordinates so adding/removing topography
        # doesn't perturb inhibitory selection.
        coord_rng = random.Random(seed ^ 0xC00D)
        cursor = 0
        self._indices = {}
        self._inhibitory = {}
        self._coordinates = {}
        for region in self._regions:
            start = cursor
            end = cursor + int(region.n_neurons)
            idx_list = list(range(start, end))
            self._indices[region.name] = idx_list

            # Pick inhibitory subset deterministically
            n_inh = int(round((1.0 - region.exc_fraction) * region.n_neurons))
            n_inh = max(0, min(region.n_neurons, n_inh))
            inh_chosen = sorted(rng.sample(idx_list, n_inh)) if n_inh > 0 else []
            self._inhibitory[region.name] = inh_chosen

            # Cluster E v1: assign coordinates if coordinate_dim > 0
            if region.coordinate_dim and region.coordinate_dim > 0:
                self._coordinates[region.name] = self._assign_coords(
                    region, coord_rng,
                )
            else:
                self._coordinates[region.name] = []

            cursor = end
        self._total_neurons = cursor

    @staticmethod
    def _assign_coords(region: BrainRegion,
                        coord_rng: random.Random) -> List[Tuple[float, ...]]:
        """Assign coordinates to all neurons in this region.

        - If `coordinate_center` is set, all neurons share that point
          (e.g., cortex_N pinned to the north corner of a unit square).
        - Otherwise, neurons get coordinates sampled uniformly from
          [0, extent_k] on each axis k. Default extent is 1.0 per axis.
        """
        k = int(region.coordinate_dim)
        if k <= 0:
            return []

        if region.coordinate_center is not None:
            center = tuple(float(c) for c in region.coordinate_center)
            if len(center) != k:
                raise ValueError(
                    f"region {region.name!r}: coordinate_center has "
                    f"length {len(center)} but coordinate_dim={k}"
                )
            return [center for _ in range(int(region.n_neurons))]

        if region.coordinate_extent is None:
            extent = tuple(1.0 for _ in range(k))
        else:
            extent = tuple(float(e) for e in region.coordinate_extent)
            if len(extent) != k:
                raise ValueError(
                    f"region {region.name!r}: coordinate_extent has "
                    f"length {len(extent)} but coordinate_dim={k}"
                )

        coords = []
        for _ in range(int(region.n_neurons)):
            pt = tuple(coord_rng.uniform(0.0, extent[ax]) for ax in range(k))
            coords.append(pt)
        return coords

    def total_neurons(self) -> int:
        return self._total_neurons

    # Deprecated region-name aliases. Old name on the LEFT, canonical on the
    # RIGHT. When a caller looks up a region by an old name, it's silently
    # translated to the canonical form with a one-time DeprecationWarning.
    # Useful for loading old sidecar JSONs that hard-coded region names.
    _DEPRECATED_REGION_NAMES = {
        # 2026-04-29 Wave-1 rename #3: "dopamine" was the modeled-region name
        # (the project's A9-equivalent — SNc dopaminergic neurons). The
        # transmitter modulator stays named "dopamine" (correct); the BG
        # region is now named "snc". Per Cluster A.16 + glossary §SNc.
        "dopamine": "snc",
        # 2026-04-29 Wave-1 rename #9: striatal FS regions are PV-FSI specifically
        # (one of eight distinct striatal GABAergic interneuron classes per
        # Tepper-2018: PV-FSI, NPY-(P)LTS, NPY-NGF, CR, TH/THIN, FAI, SABI).
        # Old name "str_FS_X" suggested cortical-FS biology; new name
        # "str_PV_FSI_X" disambiguates from cortex_FS_X (PV+ basket).
        "str_FS_N": "str_PV_FSI_N",
        "str_FS_E": "str_PV_FSI_E",
        "str_FS_S": "str_PV_FSI_S",
        "str_FS_W": "str_PV_FSI_W",
        # 2026-04-29 Wave-1 rename #2: "pfc" was the whole prefrontal cortex
        # claim; we model only dlPFC working-memory persistent activity
        # (catalog G.06 / G.08). Renamed to "dlpfc_wm".
        "pfc": "dlpfc_wm",
        # 2026-04-29 Wave-1 renames #5/#6: legacy --hippocampus regions are
        # sensor-driven readout abstractions, not canonical hippocampus.
        # Per glossary: place_cells are not allocentric per O'Keefe-Nadel
        # 1978 criteria (sensor-driven); goal_cells are anatomically PPC-like.
        "place_cells": "sensor_place_readout",
        "goal_cells": "ppc_goal_input",
        # 2026-04-29 Wave-2 rename #22: DG basket cells are PV+ specifically
        # (Kandel ch 54). Old name "dg_fs" suggested generic fast-spiking
        # but the Cluster D DG inhibitory pool is canonically PV+ basket.
        "dg_fs": "dg_pv_basket",
        # 2026-04-29 Wave-3 rename #31: cosmetic. Both "patch" and "striosome"
        # are accepted in modern literature (Bolam 2000, PBR-160 ch 9).
        # Renaming to canonical scientific term while keeping "patch" as
        # the legacy alias for sidecar JSON compatibility.
        "str_patch_N": "str_striosome_N",
        "str_patch_E": "str_striosome_E",
        "str_patch_S": "str_striosome_S",
        "str_patch_W": "str_striosome_W",
    }

    def _canonicalize_region_name(self, region_name: str) -> str:
        canonical = self._DEPRECATED_REGION_NAMES.get(region_name)
        if canonical is None or canonical not in self._indices:
            return region_name
        if not hasattr(self, "_warned_deprecated_regions"):
            self._warned_deprecated_regions = set()
        if region_name not in self._warned_deprecated_regions:
            import warnings
            warnings.warn(
                f"Region name '{region_name}' is deprecated; use '{canonical}' instead. "
                f"Old name will be removed in a future release.",
                DeprecationWarning,
                stacklevel=3,
            )
            self._warned_deprecated_regions.add(region_name)
        return canonical

    def indices(self, region_name: str) -> List[int]:
        region_name = self._canonicalize_region_name(region_name)
        if region_name not in self._indices:
            raise KeyError(region_name)
        return list(self._indices[region_name])

    def inhibitory_indices(self, region_name: str) -> List[int]:
        region_name = self._canonicalize_region_name(region_name)
        if region_name not in self._inhibitory:
            raise KeyError(region_name)
        return list(self._inhibitory[region_name])

    def region_indices_dict(self) -> Dict[str, List[int]]:
        """Returns {name: indices} suitable for
        sim.neuromodulators.NeuromodulatorManager.set_group_indices().
        """
        return {name: list(idx) for name, idx in self._indices.items()}

    def coordinates(self, region_name: str) -> List[Tuple[float, ...]]:
        """Returns per-neuron coordinates for a region (Cluster E v1).

        Empty list if the region has coordinate_dim == 0. Otherwise returns
        a list of tuples, one per neuron in this region (in the order they
        appear in `indices(region_name)`).
        """
        if region_name not in self._coordinates:
            raise KeyError(region_name)
        return list(self._coordinates[region_name])

    def max_coordinate_dim(self) -> int:
        """Returns the maximum coordinate_dim across all regions, or 0 if
        no region has topographic coordinates assigned (Cluster E v1).

        The bridge uses this to size cp_neuron_coords. Regions with smaller
        or zero coordinate_dim get NaN-padded entries.
        """
        if not self._regions:
            return 0
        return max(int(r.coordinate_dim or 0) for r in self._regions)

    def regions(self) -> List[BrainRegion]:
        return list(self._regions)

    def pathways(self) -> List[RegionPathway]:
        return list(self._pathways)

    def build_wiring_plan(self, seed: int = 0) -> Dict[str, dict]:
        """Build a `wiring_plan` dict in the format consumed by
        bridge.inject_explicit_wiring().

        Each entry is one population of synapses with shape:
            {
                "pre_indices": [int, ...],
                "post_indices": [int, ...],
                "initial_weights": [float, ...],
                "plastic": bool,
                "conn_type": str,
                "count": int,
            }

        Population names:
            "{region}_internal"           — sparse internal connectivity
            "pathway_{from}_to_{to}"      — cross-region projection

        Determinism: rng seeded from `seed`. Independent of initialize()'s
        seed so the same RegionManager can re-build with different seeds.
        """
        if self._total_neurons == 0:
            return {}

        rng = random.Random(seed)
        plan: Dict[str, dict] = {}

        # ----- Internal connectivity per region -----
        for region in self._regions:
            entry = self._build_region_internal(region, rng)
            if entry is None:
                continue
            plan[f"{region.name}_internal"] = entry

        # ----- Cross-region pathways -----
        for pw in self._pathways:
            if pw.from_region not in self._indices:
                raise KeyError(pw.from_region)
            if pw.to_region not in self._indices:
                raise KeyError(pw.to_region)
            entry = self._build_pathway(pw, rng)
            if entry is None:
                continue
            plan[f"pathway_{pw.from_region}_to_{pw.to_region}"] = entry

        return plan

    def _build_region_internal(self, region: BrainRegion,
                                rng: random.Random) -> dict:
        """Sparse Erdős-Rényi internal connectivity for a region.

        Each ordered (pre, post) pair (pre != post) within the region is
        included with probability `region.internal_density`.
        """
        if region.n_neurons <= 1 or region.internal_density <= 0:
            return None

        idx = self._indices[region.name]
        inh = set(self._inhibitory[region.name])
        density = region.internal_density

        pre_list: List[int] = []
        post_list: List[int] = []
        weights: List[float] = []
        for pre in idx:
            base_w = region.inh_weight_mean if pre in inh else region.exc_weight_mean
            jitter = region.weight_jitter
            for post in idx:
                if pre == post:
                    continue
                if rng.random() < density:
                    pre_list.append(int(pre))
                    post_list.append(int(post))
                    if jitter > 0:
                        w = base_w * (1.0 + rng.gauss(0.0, jitter))
                    else:
                        w = base_w
                    # Clamp to a reasonable positive minimum
                    weights.append(max(0.01, float(w)))

        if not pre_list:
            return None

        return {
            "pre_indices": pre_list,
            "post_indices": post_list,
            "initial_weights": weights,
            "plastic": bool(region.plastic_internal),
            "conn_type": "MIXED",
            "count": len(pre_list),
        }

    def _build_pathway(self, pw: RegionPathway, rng: random.Random) -> dict:
        """Sparse Erdős-Rényi connectivity for a directed cross-region pathway.

        When `distance_sigma` is set AND both regions have topographic
        coordinates (coordinate_dim > 0), connections are sampled with
        Gaussian-weighted probability:
            p(i, j) = density * exp(-||c_i - c_j||² / (2 * sigma²))
        Otherwise falls back to uniform Bernoulli with `density`.
        """
        pre_idx = self._indices[pw.from_region]
        post_idx = self._indices[pw.to_region]
        if pw.density <= 0 or not pre_idx or not post_idx:
            return None

        # Cluster E v1: distance-weighted sampling when sigma is set AND
        # both regions have topographic coordinates.
        use_dist = (
            pw.distance_sigma is not None
            and pw.distance_sigma > 0
            and self._has_coords(pw.from_region)
            and self._has_coords(pw.to_region)
        )

        pre_coords = self._coordinates.get(pw.from_region, []) if use_dist else []
        post_coords = self._coordinates.get(pw.to_region, []) if use_dist else []
        sigma = float(pw.distance_sigma) if use_dist else 0.0
        two_sigma_sq = 2.0 * sigma * sigma if use_dist else 1.0

        pre_list: List[int] = []
        post_list: List[int] = []
        weights: List[float] = []
        for pre_local, pre in enumerate(pre_idx):
            for post_local, post in enumerate(post_idx):
                if use_dist:
                    c_i = pre_coords[pre_local]
                    c_j = post_coords[post_local]
                    d2 = 0.0
                    for ax in range(len(c_i)):
                        delta = c_i[ax] - c_j[ax]
                        d2 += delta * delta
                    p = pw.density * math.exp(-d2 / two_sigma_sq)
                else:
                    p = pw.density
                if rng.random() < p:
                    pre_list.append(int(pre))
                    post_list.append(int(post))
                    if pw.weight_jitter > 0:
                        w = pw.weight_mean * (1.0 + rng.gauss(0.0, pw.weight_jitter))
                    else:
                        w = pw.weight_mean
                    weights.append(max(0.01, float(w)))

        if not pre_list:
            return None

        return {
            "pre_indices": pre_list,
            "post_indices": post_list,
            "initial_weights": weights,
            "plastic": bool(pw.plastic),
            "conn_type": "E_TO_MIX",
            "count": len(pre_list),
            # Pathway-specific metadata used in Task 8 for plasticity gating
            "neuromodulator_gates": list(pw.neuromodulator_gates),
            # Per-pathway plasticity gate name (runtime-controllable). None = always-on.
            "plasticity_gate": pw.plasticity_gate,
            # Per-pathway transmission gate name (runtime-controllable; scales synaptic CURRENT). None = always-on.
            "transmission_gate": pw.transmission_gate,
            # Per-pathway inhibitory receptor: "gaba_a" (default) | "gaba_b" (slow GIRK, E_K=-90mV).
            "receptor": getattr(pw, "receptor", "gaba_a"),
            # Per-pathway excitatory receptor: "ampa" (default) | "nmda_slow" (slow NR2B
            # recurrent; feeds a separate slow-NMDA conductance, AMPA component suppressed).
            "exc_receptor": getattr(pw, "exc_receptor", "ampa"),
            # Per-pathway dendritic-coincidence flag (2026-06-09): True => this pathway's synapses feed a
            # per-postsynaptic-neuron NMDA-spike subunit (a supralinear plateau on >=K coincident inputs);
            # AMPA component KEPT (plateau is additive). Default False = no coincidence routing.
            "coincidence_detector": bool(getattr(pw, "coincidence_detector", False)),
            # Per-pathway GRADED (analog, non-spiking) transmission flag (2026-06-15): True => this pathway's
            # synapses transmit from the SOURCE's continuous MEMBRANE potential, a_cont = clip((v-rest)/scale,0,1),
            # not its spikes (the retina's horizontal-cell graded release). Default False = spike-mediated (byte-identical).
            "graded": bool(getattr(pw, "graded", False)),
            # Per-pathway STP-disable flag (2026-07-21, gap#5): True => this pathway's synapses skip
            # short-term depression (effective STP factor forced to 1.0). Default False = STP-gated (byte-identical).
            "stp_disabled": bool(getattr(pw, "stp_disabled", False)),
        }

    def _has_coords(self, region_name: str) -> bool:
        """Returns True if the named region has any topographic coordinates."""
        coords = self._coordinates.get(region_name)
        return bool(coords)
