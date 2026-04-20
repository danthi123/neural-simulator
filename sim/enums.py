"""Enum definitions and related constants for the neural simulator."""

import math
from enum import Enum


class NeuronModel(Enum):
    IZHIKEVICH = "IZHIKEVICH"
    HODGKIN_HUXLEY = "HODGKIN_HUXLEY"
    ADEX = "ADEX"  # Adaptive Exponential Integrate-and-Fire


class NeuronType(Enum):
    IZH2007_RS_CORTICAL_PYRAMIDAL = "IZH2007_RS_CORTICAL_PYRAMIDAL"
    IZH2007_FS_CORTICAL_INTERNEURON = "IZH2007_FS_CORTICAL_INTERNEURON"
    HH_L5_CORTICAL_PYRAMIDAL_RS = "HH_L5_CORTICAL_PYRAMIDAL_RS"
    HH_THALAMIC_RELAY_TBURST = "HH_THALAMIC_RELAY_TBURST"
    HH_CA1_PYRAMIDAL_BURST = "HH_CA1_PYRAMIDAL_BURST"
    HH_STRIATAL_MSN = "HH_STRIATAL_MSN"
    HH_TRN_BURST_INHIB = "HH_TRN_BURST_INHIB"
    HH_CA3_PYRAMIDAL_BURST = "HH_CA3_PYRAMIDAL_BURST"
    HH_STN_BURST = "HH_STN_BURST"
    HH_GPE_PACEMAKER = "HH_GPE_PACEMAKER"
    HH_CEREBELLAR_PURKINJE = "HH_CEREBELLAR_PURKINJE"
    HH_CEREBELLAR_GRANULE = "HH_CEREBELLAR_GRANULE"
    HH_SPINAL_MOTOR = "HH_SPINAL_MOTOR"
    HH_SPINAL_INTERNEURON = "HH_SPINAL_INTERNEURON"
    HH_PFC_PYRAMIDAL = "HH_PFC_PYRAMIDAL"
    HH_OLFACTORY_MITRAL = "HH_OLFACTORY_MITRAL"
    HH_DOPAMINE_SNC = "HH_DOPAMINE_SNC"
    HH_CORTICAL_FS_INTERNEURON = "HH_CORTICAL_FS_INTERNEURON"
    HH_INFERIOR_OLIVE = "HH_INFERIOR_OLIVE"
    RS_EXCITATORY_LEGACY = "RS_EXCITATORY_LEGACY"
    FS_INHIBITORY_LEGACY = "FS_INHIBITORY_LEGACY"
    IB_EXCITATORY_LEGACY = "IB_EXCITATORY_LEGACY"
    CH_EXCITATORY_LEGACY = "CH_EXCITATORY_LEGACY"
    LTS_INHIBITORY_LEGACY = "LTS_INHIBITORY_LEGACY"
    HH_EXCITATORY_DEFAULT_LEGACY = "HH_EXCITATORY_DEFAULT_LEGACY"


class DefaultHodgkinHuxleyParams:
    # Parameters for a more realistic Layer 5 Pyramidal Neuron (Regular Spiking) at 37 C
    # Adapted from literature, may require tuning for specific behaviors.
    # Key sources: Mainen & Sejnowski (1996), Pospischil et al. (2008) for general cortical neuron models.
    REALISTIC_L5_PYRAMIDAL_RS_37C = {
        "C_m": 1.0,       # Membrane capacitance (uF/cm^2) - Common value
        "g_Na_max": 50.0, # Max Na conductance (mS/cm^2) - Can vary (e.g., 50-120)
        "g_K_max": 5.0,   # Max K_DR conductance (mS/cm^2) - For delayed rectifier (e.g., 5-30)
        "g_L": 0.1,       # Leak conductance (mS/cm^2) - (e.g., 0.02-0.1)
        "E_Na": 50.0,     # Na reversal potential (mV) - (e.g., +50 to +60)
        "E_K": -85.0,     # K reversal potential (mV) - (e.g., -80 to -90 for K_DR)
        "E_L": -70.0,     # Leak reversal potential (mV) - (e.g., -65 to -75, often near V_rest)
        "v_rest_hh": -65.0, # Resting potential for HH model initialization (mV)
        "v_peak_hh": 40.0,  # Spike peak for HH model (mV) - for spike detection logic
        # Initial gating variable values (approximate for v_rest_hh = -65mV)
        "m_init": 0.0529, # Calculated from alpha_m / (alpha_m + beta_m) at -65mV for original HH
        "h_init": 0.5961, # Calculated from alpha_h / (alpha_h + beta_h) at -65mV for original HH
        "n_init": 0.3177, # Calculated from alpha_n / (alpha_n + beta_n) at -65mV for original HH
        # Extended current defaults (all off by default)
        "g_M_max": 0.0,
        "g_CaT_max": 0.0,
        "E_CaT": 120.0,
        "g_h_max": 0.0,
        "E_h": -30.0,
        "g_NaP_max": 0.0,
    }
    @staticmethod
    def compute_hh_gating_steady_state(V_rest, temperature_celsius=37.0, q10_factor=3.0):
        """Compute HH gating variable steady-state values at a given resting potential.

        This ensures gating variables start at equilibrium regardless of V_rest,
        avoiding transient artifacts at simulation onset. Uses original HH alpha/beta
        rate functions with Q10 temperature correction.

        Args:
            V_rest: Resting membrane potential (mV)
            temperature_celsius: Simulation temperature (°C)
            q10_factor: Q10 temperature coefficient (default 3.0)

        Returns:
            dict with 'm_init', 'h_init', 'n_init' at steady-state for V_rest
        """
        V = V_rest
        BASE_T = 6.3  # Original HH kinetics temperature
        phi = q10_factor ** ((temperature_celsius - BASE_T) / 10.0)

        # Alpha/beta rate functions (original HH, Hodgkin & Huxley 1952)
        v40 = V + 40.0
        if abs(v40) < 1e-6:
            alpha_m = 1.0  # L'Hôpital limit
        else:
            alpha_m = -0.1 * v40 / (math.exp(-v40 / 10.0) - 1.0)
        beta_m = 4.0 * math.exp(-(V + 65.0) / 18.0)

        alpha_h = 0.07 * math.exp(-(V + 65.0) / 20.0)
        beta_h = 1.0 / (math.exp(-(V + 35.0) / 10.0) + 1.0)

        v55 = V + 55.0
        if abs(v55) < 1e-6:
            alpha_n = 0.1 * 0.01 * 10.0  # L'Hôpital limit
        else:
            alpha_n = -0.01 * v55 / (math.exp(-v55 / 10.0) - 1.0)
        beta_n = 0.125 * math.exp(-(V + 65.0) / 80.0)

        # phi cancels in inf = alpha/(alpha+beta) since both scale by phi
        m_inf = alpha_m / (alpha_m + beta_m) if (alpha_m + beta_m) > 0 else 0.0
        h_inf = alpha_h / (alpha_h + beta_h) if (alpha_h + beta_h) > 0 else 1.0
        n_inf = alpha_n / (alpha_n + beta_n) if (alpha_n + beta_n) > 0 else 0.0

        return {"m_init": round(m_inf, 6), "h_init": round(h_inf, 6), "n_init": round(n_inf, 6)}

    # Original Hodgkin-Huxley parameters (Squid Giant Axon at 6.3 C)
    ORIGINAL_HH_PARAMS = {
        "C_m": 1.0, "g_Na_max": 120.0, "g_K_max": 36.0, "g_L": 0.3,
        "E_Na": 50.0, "E_K": -77.0, "E_L": -54.387, # Note: E_L adjusted for V_rest = -65mV in original model
        "v_rest_hh": -65.0, "v_peak_hh": 40.0,
        "m_init": 0.0529, "h_init": 0.5961, "n_init": 0.3177,
        # Extended current defaults (all off by default)
        "g_M_max": 0.0,
        "g_CaT_max": 0.0,
        "E_CaT": 120.0,
        "g_h_max": 0.0,
        "E_h": -30.0,
        "g_NaP_max": 0.0,
    }

    # Region-specific HH presets (single-compartment, point-neuron approximations)
    THALAMIC_RELAY_TBURST = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    THALAMIC_RELAY_TBURST.update({
        # Strong low-threshold CaT and Ih for bursty thalamic relay cells
        "g_CaT_max": 2.0,
        "E_CaT": 120.0,
        "g_h_max": 0.5,
        "E_h": -40.0,
        "g_M_max": 0.0,
        "g_NaP_max": 0.0,
    })

    CA1_PYRAMIDAL_BURST = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    CA1_PYRAMIDAL_BURST.update({
        # Moderate CaT, Ih, M-current and NaP to support burst firing and adaptation
        "g_Na_max": 60.0,
        "g_K_max": 6.0,
        "g_CaT_max": 1.0,
        "E_CaT": 120.0,
        "g_h_max": 0.2,
        "E_h": -40.0,
        "g_M_max": 0.8,
        "g_NaP_max": 0.5,
    })

    STRIATAL_MSN = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    STRIATAL_MSN.update({
        # Strong M-current and modest Ih to approximate down-state stability and slow ramping
        "g_Na_max": 45.0,
        "g_K_max": 4.0,
        "g_M_max": 1.2,
        "g_CaT_max": 0.0,
        "g_h_max": 0.3,
        "E_h": -35.0,
        "g_NaP_max": 0.0,
    })

    # Thalamic reticular nucleus (TRN) bursting inhibitory cell
    TRN_BURST_INHIB = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    TRN_BURST_INHIB.update({
        # Strong CaT and Ih, plus some M-current for burst–tonic transitions
        "g_Na_max": 50.0,
        "g_K_max": 5.0,
        "g_CaT_max": 2.5,
        "E_CaT": 120.0,
        "g_h_max": 0.4,
        "E_h": -40.0,
        "g_M_max": 0.5,
        "g_NaP_max": 0.0,
    })

    # Hippocampal CA3 pyramidal bursting cell
    CA3_PYRAMIDAL_BURST = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    CA3_PYRAMIDAL_BURST.update({
        # Slightly stronger Na/K and bursting currents than CA1
        "g_Na_max": 65.0,
        "g_K_max": 7.0,
        "g_CaT_max": 1.2,
        "E_CaT": 120.0,
        "g_h_max": 0.25,
        "E_h": -40.0,
        "g_M_max": 1.0,
        "g_NaP_max": 0.7,
    })

    # Subthalamic nucleus (STN) bursting cell
    STN_BURST = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    STN_BURST.update({
        # CaT- and NaP-mediated bursting with some Ih and M-current
        "g_Na_max": 55.0,
        "g_K_max": 6.0,
        "g_CaT_max": 1.5,
        "E_CaT": 120.0,
        "g_h_max": 0.3,
        "E_h": -40.0,
        "g_M_max": 0.5,
        "g_NaP_max": 0.8,
    })

    # Globus pallidus externus (GPe) pacemaking neuron
    GPE_PACEMAKER = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    GPE_PACEMAKER.update({
        # Strong M and NaP for tonic spiking, with modest Ih
        "g_Na_max": 55.0,
        "g_K_max": 5.5,
        "g_CaT_max": 0.0,
        "g_h_max": 0.2,
        "E_h": -35.0,
        "g_M_max": 1.0,
        "g_NaP_max": 0.8,
    })

    # Cerebellar Purkinje cell (Khaliq et al. 2003, De Schutter & Bower 1994)
    CEREBELLAR_PURKINJE = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    CEREBELLAR_PURKINJE.update({
        "C_m": 1.2,           # Larger soma + dendrite
        "g_Na_max": 75.0,     # High Na for fast tonic spiking
        "g_K_max": 8.0,       # Strong repolarization
        "g_CaT_max": 1.8,     # Strong T-type Ca for complex spikes / dendritic bursts
        "E_CaT": 120.0,
        "g_h_max": 0.0,       # Purkinje cells lack Ih
        "g_M_max": 1.5,       # Strong AHP via M-current (BK-like)
        "g_NaP_max": 0.3,     # Modest persistent Na supports tonic firing
        "E_L": -68.0,         # Slightly depolarized for spontaneous activity
        "v_rest_hh": -62.0,
    })

    # Cerebellar granule cell (D'Angelo et al. 2001)
    CEREBELLAR_GRANULE = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    CEREBELLAR_GRANULE.update({
        "C_m": 0.8,           # Small cells
        "g_Na_max": 40.0,     # Moderate Na
        "g_K_max": 4.0,       # Moderate K
        "g_L": 0.08,          # High input resistance (lower leak)
        "g_CaT_max": 0.0,     # Minimal CaT
        "g_h_max": 0.15,      # Small Ih for resonance
        "E_h": -30.0,
        "g_M_max": 0.3,       # Mild adaptation
        "g_NaP_max": 0.2,     # Small persistent Na
        "E_L": -72.0,
        "v_rest_hh": -68.0,
    })

    # Spinal motor neuron (Powers & Binder 2001, Heckman & Enoka 2012)
    SPINAL_MOTOR = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    SPINAL_MOTOR.update({
        "C_m": 1.5,           # Large alpha motor neuron soma
        "g_Na_max": 70.0,     # Strong Na for reliable spiking
        "g_K_max": 7.0,       # Strong repolarization
        "g_CaT_max": 1.2,     # CaT for plateau potentials / bistability
        "E_CaT": 120.0,
        "g_h_max": 0.3,       # Ih contributes to resting conductance
        "E_h": -30.0,
        "g_M_max": 1.0,       # M-current for adaptation and AHP
        "g_NaP_max": 0.6,     # Persistent Na for input amplification
        "E_L": -70.0,
        "v_rest_hh": -65.0,
    })

    # Spinal inhibitory interneuron (Renshaw / Ia inhibitory, Jankowska 2001)
    SPINAL_INTERNEURON = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    SPINAL_INTERNEURON.update({
        "C_m": 0.9,           # Moderate soma size
        "g_Na_max": 55.0,     # Moderate Na
        "g_K_max": 6.0,       # Strong K for fast repolarization
        "g_CaT_max": 0.8,     # CaT for rebound bursting
        "E_CaT": 120.0,
        "g_h_max": 0.15,      # Small Ih
        "E_h": -30.0,
        "g_M_max": 0.4,       # Mild adaptation
        "g_NaP_max": 0.0,     # No persistent Na
        "E_L": -72.0,
        "v_rest_hh": -68.0,
    })

    # Prefrontal Cortex pyramidal neuron (Wang 2001, Durstewitz et al. 2000)
    PFC_PYRAMIDAL = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    PFC_PYRAMIDAL.update({
        "C_m": 1.0,           # Standard pyramidal capacitance
        "g_Na_max": 50.0,     # Moderate Na (PFC pyramidals fire slower than L5 PT)
        "g_K_max": 5.0,       # Standard delayed rectifier
        "g_CaT_max": 0.5,     # Moderate Ca for UP-state calcium signaling
        "E_CaT": 120.0,
        "g_h_max": 0.25,      # Moderate Ih for subthreshold resonance
        "E_h": -30.0,
        "g_M_max": 0.8,       # Moderate M-current for spike frequency adaptation
        "g_NaP_max": 0.5,     # STRONG persistent Na — enables bistable persistent activity
        "E_L": -70.0,
        "v_rest_hh": -68.0,
    })

    # Olfactory bulb mitral cell (Migliore et al. 2005, Davison et al. 2003)
    OLFACTORY_MITRAL = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    OLFACTORY_MITRAL.update({
        "C_m": 1.0,           # Standard capacitance
        "g_Na_max": 65.0,     # High Na for fast, reliable spikes
        "g_K_max": 8.0,       # Strong K for fast repolarization
        "g_CaT_max": 0.3,     # Small CaT for calcium signaling
        "E_CaT": 120.0,
        "g_h_max": 0.1,       # Small Ih
        "E_h": -30.0,
        "g_M_max": 0.2,       # Minimal adaptation — mitral cells sustain high rates
        "g_NaP_max": 0.3,     # Moderate persistent Na for subthreshold oscillations
        "E_L": -65.0,
        "v_rest_hh": -62.0,
    })

    # Substantia nigra pars compacta / VTA dopamine neuron (Drion et al. 2011, Putzier et al. 2009)
    DOPAMINE_SNC = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    DOPAMINE_SNC.update({
        "C_m": 1.2,           # Moderate soma size
        "g_Na_max": 35.0,     # LOW Na — DA neurons have sparse Na channels
        "g_K_max": 4.0,       # Moderate K
        "g_CaT_max": 2.0,     # STRONG Ca — L-type Ca proxy, primary pacemaker driver
        "E_CaT": 120.0,
        "g_h_max": 0.4,       # Moderate-strong Ih, contributes to pacemaking rebound
        "E_h": -30.0,
        "g_M_max": 0.3,       # Mild M-current (SK channel analog for AHP)
        "g_NaP_max": 0.2,     # Small persistent Na
        "E_L": -55.0,         # Depolarized rest — key for autonomous firing
        "v_rest_hh": -52.0,
    })

    # Cortical fast-spiking PV+ interneuron (Erisir et al. 1999, Wang & Buzsaki 1996)
    CORTICAL_FS_INTERNEURON = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    CORTICAL_FS_INTERNEURON.update({
        "C_m": 0.8,           # Smaller soma than pyramidals
        "g_Na_max": 80.0,     # VERY HIGH Na — enables fast, narrow APs
        "g_K_max": 15.0,      # VERY HIGH K — fast repolarization, narrow spike
        "g_CaT_max": 0.0,     # No CaT
        "g_h_max": 0.0,       # No Ih
        "g_M_max": 0.0,       # NO adaptation — defining feature of FS interneurons
        "g_NaP_max": 0.0,     # No persistent Na
        "E_L": -70.0,
        "v_rest_hh": -68.0,
    })

    # Inferior olivary neuron (Llinas & Yarom 1981, De Gruijl et al. 2012)
    INFERIOR_OLIVE = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    INFERIOR_OLIVE.update({
        "C_m": 1.0,           # Standard capacitance
        "g_Na_max": 40.0,     # Moderate Na
        "g_K_max": 5.0,       # Standard K
        "g_CaT_max": 1.5,     # STRONG CaT — drives subthreshold oscillations
        "E_CaT": 120.0,
        "g_h_max": 0.5,       # STRONG Ih — rebound from inhibition, oscillation partner
        "E_h": -30.0,
        "g_M_max": 0.3,       # Mild M-current
        "g_NaP_max": 0.3,     # Moderate persistent Na for oscillation support
        "E_L": -65.0,
        "v_rest_hh": -60.0,
    })

    PARAMS = {
        NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS: REALISTIC_L5_PYRAMIDAL_RS_37C.copy(),
        NeuronType.HH_EXCITATORY_DEFAULT_LEGACY: ORIGINAL_HH_PARAMS.copy(), # Legacy can map to original HH
        NeuronType.HH_THALAMIC_RELAY_TBURST: THALAMIC_RELAY_TBURST.copy(),
        NeuronType.HH_CA1_PYRAMIDAL_BURST: CA1_PYRAMIDAL_BURST.copy(),
        NeuronType.HH_STRIATAL_MSN: STRIATAL_MSN.copy(),
        NeuronType.HH_TRN_BURST_INHIB: TRN_BURST_INHIB.copy(),
        NeuronType.HH_CA3_PYRAMIDAL_BURST: CA3_PYRAMIDAL_BURST.copy(),
        NeuronType.HH_STN_BURST: STN_BURST.copy(),
        NeuronType.HH_GPE_PACEMAKER: GPE_PACEMAKER.copy(),
        NeuronType.HH_CEREBELLAR_PURKINJE: CEREBELLAR_PURKINJE.copy(),
        NeuronType.HH_CEREBELLAR_GRANULE: CEREBELLAR_GRANULE.copy(),
        NeuronType.HH_SPINAL_MOTOR: SPINAL_MOTOR.copy(),
        NeuronType.HH_SPINAL_INTERNEURON: SPINAL_INTERNEURON.copy(),
        NeuronType.HH_PFC_PYRAMIDAL: PFC_PYRAMIDAL.copy(),
        NeuronType.HH_OLFACTORY_MITRAL: OLFACTORY_MITRAL.copy(),
        NeuronType.HH_DOPAMINE_SNC: DOPAMINE_SNC.copy(),
        NeuronType.HH_CORTICAL_FS_INTERNEURON: CORTICAL_FS_INTERNEURON.copy(),
        NeuronType.HH_INFERIOR_OLIVE: INFERIOR_OLIVE.copy(),
    }
    FALLBACK = PARAMS[NeuronType.HH_EXCITATORY_DEFAULT_LEGACY].copy()

    @staticmethod
    def get_params(neuron_type_enum):
        return DefaultHodgkinHuxleyParams.PARAMS.get(neuron_type_enum, DefaultHodgkinHuxleyParams.FALLBACK).copy()


class StimulusPatternType(Enum):
    """Available stimulus waveform types."""
    CONSTANT = "CONSTANT"                   # DC current step
    PULSE_TRAIN = "PULSE_TRAIN"             # Repeated brief pulses
    SINUSOIDAL = "SINUSOIDAL"               # AC sinusoidal current
    RAMP = "RAMP"                           # Linearly increasing/decreasing
    POISSON_SPIKE_TRAIN = "POISSON_SPIKE_TRAIN"  # Poisson-distributed brief pulses
    GAUSSIAN_NOISE = "GAUSSIAN_NOISE"       # White noise injection
    CUSTOM_WAVEFORM = "CUSTOM_WAVEFORM"     # Arbitrary time series
    RATE_VECTOR_POISSON = "RATE_VECTOR_POISSON"  # Per-neuron Poisson rate vector


class NeuronGroupRole(Enum):
    """Role designation for neuron populations."""
    INPUT = "INPUT"       # Receives external stimuli
    OUTPUT = "OUTPUT"     # Activity decoded as network response
    HIDDEN = "HIDDEN"     # Internal processing (default)


class TrainingMode(Enum):
    """Available training paradigms."""
    NONE = "NONE"
    ASSOCIATIVE_PAIRING = "ASSOCIATIVE_PAIRING"         # CS-US Pavlovian conditioning
    REINFORCEMENT_LEARNING = "REINFORCEMENT_LEARNING"   # R-STDP with reward signal
    SUPERVISED_TARGET = "SUPERVISED_TARGET"               # Target rate matching
    RESERVOIR_READOUT = "RESERVOIR_READOUT"               # Fixed recurrent, train readout


class ExperimentPhaseType(Enum):
    """Types of experiment phases."""
    BASELINE = "BASELINE"           # Record baseline activity (no stimulus)
    STIMULUS = "STIMULUS"           # Present stimulus, record response
    TRAINING = "TRAINING"           # Active learning with stimulus + feedback
    TESTING = "TESTING"             # Test learned responses (no weight updates)
    REST = "REST"                   # Inter-trial interval (no stimulus)


# --- Izhikevich Parameter Defaults ---
class DefaultIzhikevichParamsManager:
    PARAMS = {
        NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL: {
            "C": 100.0, "k": 0.7, "vr": -60.0, "vt": -40.0, "vpeak": 35.0,
            "a": 0.03, "b": -2.0, "c_reset": -50.0, "d_increment": 100.0
        },
        NeuronType.IZH2007_FS_CORTICAL_INTERNEURON: {
            "C": 20.0, "k": 1.0, "vr": -55.0, "vt": -40.0, "vpeak": 25.0,
            "a": 0.2, "b": -2.0, "c_reset": -45.0, "d_increment": 25.0
            # d_increment must be POSITIVE for FS interneurons (Izhikevich 2007, Table 2).
            # Positive d drives post-spike recovery variable u upward -> stronger AHP -> faster return to rest.
            # Negative d would paradoxically cause post-spike depolarization (excitation after inhibition).
            # Value of 25 pA gives the characteristic non-adapting, high-frequency firing pattern of PV+ basket cells.
        },
        NeuronType.RS_EXCITATORY_LEGACY: {"a": 0.02, "b": 0.2, "c_reset": -65.0, "d_increment": 8.0, "vpeak": 30.0},
        NeuronType.FS_INHIBITORY_LEGACY: {"a": 0.1, "b": 0.2, "c_reset": -65.0, "d_increment": 2.0, "vpeak": 30.0},
        NeuronType.IB_EXCITATORY_LEGACY: {"a": 0.02, "b": 0.2, "c_reset": -55.0, "d_increment": 4.0, "vpeak": 50.0},
        NeuronType.CH_EXCITATORY_LEGACY: {"a": 0.02, "b": 0.2, "c_reset": -50.0, "d_increment": 2.0, "vpeak": 35.0},
        NeuronType.LTS_INHIBITORY_LEGACY: {"a": 0.02, "b": 0.25, "c_reset": -65.0, "d_increment": 2.0, "vpeak": 30.0}
    }
    FALLBACK_2007 = PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL].copy()
    FALLBACK_LEGACY = PARAMS[NeuronType.RS_EXCITATORY_LEGACY].copy()

    @staticmethod
    def get_params(neuron_type_enum, use_2007_formulation=True):
        if use_2007_formulation:
            if neuron_type_enum in [NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL, NeuronType.IZH2007_FS_CORTICAL_INTERNEURON]:
                 return DefaultIzhikevichParamsManager.PARAMS.get(neuron_type_enum, DefaultIzhikevichParamsManager.FALLBACK_2007).copy()
            print(f"Warning: Requested legacy type {neuron_type_enum} for 2007 formulation. Using RS_CORTICAL_PYRAMIDAL fallback.")
            return DefaultIzhikevichParamsManager.FALLBACK_2007.copy()
        else: # Legacy formulation
            if 'LEGACY' in neuron_type_enum.name:
                 return DefaultIzhikevichParamsManager.PARAMS.get(neuron_type_enum, DefaultIzhikevichParamsManager.FALLBACK_LEGACY).copy()
            print(f"Warning: Requested 2007 type {neuron_type_enum} for legacy formulation. Using RS_EXCITATORY_LEGACY fallback.")
            return DefaultIzhikevichParamsManager.FALLBACK_LEGACY.copy()


# --- Performance Optimization: Neuron Type ID Mapper ---
class NeuronTypeIDMapper:
    """Maps NeuronType enums to integer IDs for GPU-efficient operations.

    This eliminates string comparisons on CPU by using integer type IDs
    that can be processed directly on the GPU.
    """
    def __init__(self):
        self.type_to_id = {}
        self.id_to_type = {}
        self.id_to_display_name = {}
        self._build_mappings()

    def _build_mappings(self):
        """Build bidirectional mappings between NeuronType enums and integer IDs."""
        # Izhikevich types
        izh_types = [nt for nt in NeuronType if "IZH2007" in nt.name and nt in DefaultIzhikevichParamsManager.PARAMS]
        for idx, ntype in enumerate(izh_types):
            self.type_to_id[ntype] = idx
            self.id_to_type[idx] = ntype
            self.id_to_display_name[idx] = f"Izh2007_{ntype.name.replace('IZH2007_', '')}"

        # Hodgkin-Huxley types (offset by max Izh type ID)
        hh_types = [nt for nt in NeuronType if "HH_" in nt.name and nt in DefaultHodgkinHuxleyParams.PARAMS]
        hh_offset = len(izh_types)
        for idx, ntype in enumerate(hh_types):
            type_id = hh_offset + idx
            self.type_to_id[ntype] = type_id
            self.id_to_type[type_id] = ntype
            self.id_to_display_name[type_id] = f"HH_{ntype.name.replace('HH_', '')}"

    def get_id(self, neuron_type_enum):
        """Get integer ID for a NeuronType enum."""
        return self.type_to_id.get(neuron_type_enum, 0)  # Default to 0 if not found

    def get_type(self, type_id):
        """Get NeuronType enum for an integer ID."""
        return self.id_to_type.get(type_id, list(self.id_to_type.values())[0] if self.id_to_type else NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL)

    def get_display_name(self, type_id):
        """Get display name string for an integer type ID."""
        return self.id_to_display_name.get(type_id, "Unknown")

    def get_all_display_names_for_model(self, model_name):
        """Get list of display names for a specific model type."""
        if model_name == NeuronModel.IZHIKEVICH.name:
            return [self.id_to_display_name[i] for i in sorted(self.id_to_display_name.keys())
                    if "Izh" in self.id_to_display_name[i]]
        elif model_name == NeuronModel.HODGKIN_HUXLEY.name:
            return [self.id_to_display_name[i] for i in sorted(self.id_to_display_name.keys())
                    if "HH" in self.id_to_display_name[i]]
        return []

    def get_id_from_display_name(self, display_name):
        """Get type ID from display name string."""
        for type_id, name in self.id_to_display_name.items():
            if name == display_name:
                return type_id
        return 0  # Default


# Global type mapper instance (initialized after all required classes are defined)
NEURON_TYPE_MAPPER = NeuronTypeIDMapper()
