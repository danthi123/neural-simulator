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
    # Phase A → B addition: Izhikevich 2007 presets for additional cell types.
    # All work cleanly at 37°C (unlike HH presets — see HH temperature bug).
    # Sources: Izhikevich 2003 IEEE TNN Table II + 2007 book parameters.
    IZH2007_STRIATAL_MSN = "IZH2007_STRIATAL_MSN"  # Medium spiny neuron, BG input
    IZH2007_THALAMIC_RELAY = "IZH2007_THALAMIC_RELAY"  # TC neurons (RS in tonic mode)
    IZH2007_THALAMIC_RETICULAR = "IZH2007_THALAMIC_RETICULAR"  # TRN, LTS-like bursting
    IZH2007_GPE_PACEMAKER = "IZH2007_GPE_PACEMAKER"  # Globus pallidus externus
    IZH2007_GPI_OUTPUT = "IZH2007_GPI_OUTPUT"  # Globus pallidus internus / SNr
    IZH2007_STN_BURST = "IZH2007_STN_BURST"  # Subthalamic nucleus
    IZH2007_HIPPO_PYRAMIDAL = "IZH2007_HIPPO_PYRAMIDAL"  # CA1/CA3 pyramidal (IB-like)
    IZH2007_DOPAMINE = "IZH2007_DOPAMINE"  # VTA/SNc DA neurons
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
    # BG completion (Phase A → B): missing presets identified during audit.
    HH_GPI_OUTPUT = "HH_GPI_OUTPUT"             # BG output gate, distinct from GPe
    HH_STRIATAL_MSN_D1 = "HH_STRIATAL_MSN_D1"   # Direct pathway MSN (DA D1+ sensitive)
    HH_STRIATAL_MSN_D2 = "HH_STRIATAL_MSN_D2"   # Indirect pathway MSN (DA D2- sensitive)
    HH_STRIATAL_TAN = "HH_STRIATAL_TAN"          # Tonically Active Cholinergic Interneuron
    IZH2007_STRIATAL_MSN_D1 = "IZH2007_STRIATAL_MSN_D1"
    IZH2007_STRIATAL_MSN_D2 = "IZH2007_STRIATAL_MSN_D2"
    IZH2007_STRIATAL_TAN = "IZH2007_STRIATAL_TAN"
    RS_EXCITATORY_LEGACY = "RS_EXCITATORY_LEGACY"
    FS_INHIBITORY_LEGACY = "FS_INHIBITORY_LEGACY"
    IB_EXCITATORY_LEGACY = "IB_EXCITATORY_LEGACY"
    CH_EXCITATORY_LEGACY = "CH_EXCITATORY_LEGACY"
    LTS_INHIBITORY_LEGACY = "LTS_INHIBITORY_LEGACY"
    HH_EXCITATORY_DEFAULT_LEGACY = "HH_EXCITATORY_DEFAULT_LEGACY"
    # AdEx (Adaptive Exponential Integrate-and-Fire) presets.
    # Brette & Gerstner 2005 "Adaptive Exponential Integrate-and-Fire model"
    # JNeuro 94(5):3637 — five canonical phenotypes via parameter tuning.
    ADEX_RS_CORTICAL_PYRAMIDAL = "ADEX_RS_CORTICAL_PYRAMIDAL"
    ADEX_FS_CORTICAL_INTERNEURON = "ADEX_FS_CORTICAL_INTERNEURON"
    ADEX_IB_BURSTING = "ADEX_IB_BURSTING"             # Intrinsic bursting
    ADEX_CH_CHATTERING = "ADEX_CH_CHATTERING"         # Chattering (high-rate gamma drivers)
    ADEX_LTS_LOW_THRESHOLD = "ADEX_LTS_LOW_THRESHOLD"  # Low-threshold spiking interneuron
    ADEX_STRIATAL_MSN = "ADEX_STRIATAL_MSN"           # Down-state stable MSN
    ADEX_DOPAMINE = "ADEX_DOPAMINE"                   # Slow tonic + phasic burst


class DefaultHodgkinHuxleyParams:
    # Parameters for a more realistic Layer 5 Pyramidal Neuron (Regular Spiking) at 37 C
    # Adapted from literature, may require tuning for specific behaviors.
    # Key sources: Mainen & Sejnowski (1996), Pospischil et al. (2008) for general cortical neuron models.
    REALISTIC_L5_PYRAMIDAL_RS_37C = {
        "C_m": 1.0,       # Membrane capacitance (uF/cm^2) - Common value
        "g_Na_max": 50.0, # Max Na conductance (mS/cm^2) - Can vary (e.g., 50-120)
        # NOTE: g_K bumped from 5→12 (2026-04-25 preset audit fix). Original
        # g_K=5 was way too low — caused depolarization block at moderate
        # input current. Real cortical RS pyramidals have g_K_DR ~15-30
        # mS/cm². Higher K allows faster repolarization and sustained
        # firing rates of 30+ Hz instead of getting stuck at 2 Hz.
        "g_K_max": 12.0,  # Max K_DR conductance (mS/cm^2) - For delayed rectifier (e.g., 5-30)
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
        # Extended currents.
        # 2026-04-25: Added g_M=0.6 (M-current / Kv7) to base. Real cortical
        # RS pyramidals have substantial M-current providing slow K-channel
        # activation above -55 mV. Without it, sustained input causes
        # depolarization block (cell fires once then locks). M-current
        # provides spike-frequency adaptation AND keeps V from staying at
        # plateau — required for proper tonic firing.
        # Yamada et al. 1989, Storm 1990 give g_M_density ~0.3-1.0 mS/cm².
        "g_M_max": 0.6,
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
        # Re-tuned 2026-04-25: g_NaP 0.5→0.15 (rest was -58 mV, now closer to -65)
        # g_K 6→12 (allow sustained firing instead of depolarization block)
        "g_Na_max": 60.0,
        "g_K_max": 12.0,
        "g_CaT_max": 1.0,
        "E_CaT": 120.0,
        "g_h_max": 0.2,
        "E_h": -40.0,
        "g_M_max": 0.8,
        "g_NaP_max": 0.15,
    })

    STRIATAL_MSN = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    STRIATAL_MSN.update({
        # Strong M-current and modest Ih to approximate down-state stability and slow ramping.
        # Re-tuned 2026-04-25: g_K 4→14 (was getting stuck at 2 Hz). Real MSNs
        # in up-state fire 5-30 Hz; with higher g_K we now allow proper firing rates.
        "g_Na_max": 45.0,
        "g_K_max": 14.0,
        "g_M_max": 1.2,
        "g_CaT_max": 0.0,
        "g_h_max": 0.3,
        "E_h": -35.0,
        "g_NaP_max": 0.0,
    })

    # Thalamic reticular nucleus (TRN) bursting inhibitory cell
    TRN_BURST_INHIB = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    TRN_BURST_INHIB.update({
        # Strong CaT and Ih, plus some M-current for burst–tonic transitions.
        # Re-tuned 2026-04-25: g_K 5→14 for proper firing rates (TRN can do
        # 100+ Hz tonic between bursts).
        "g_Na_max": 50.0,
        "g_K_max": 14.0,
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
        # Slightly stronger Na/K and bursting currents than CA1.
        # Re-tuned 2026-04-25: g_NaP 0.7→0.2 (rest was -56 mV, now closer to -65).
        # g_K 7→14 (allow sustained firing).
        "g_Na_max": 65.0,
        "g_K_max": 14.0,
        "g_CaT_max": 1.2,
        "E_CaT": 120.0,
        "g_h_max": 0.25,
        "E_h": -40.0,
        "g_M_max": 1.0,
        "g_NaP_max": 0.2,
    })

    # Subthalamic nucleus (STN) bursting cell.
    # Re-tuned (2026-04-25): original g_NaP=0.8 was 5-10x too high vs. real
    # biology (Bevan & Wilson 1999: g_NaP_density ~0.05-0.15 mS/cm²).
    # Excessive NaP pulled rest to -34 mV → categorically unfireable.
    # Now: g_NaP=0.15, g_K=12, E_L=-68 → biologically realistic resting
    # range and capable of pacemaking + rebound bursts.
    STN_BURST = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    STN_BURST.update({
        "g_Na_max": 70.0,     # was 55 — higher Na for fast spike upstroke
        "g_K_max": 12.0,      # was 6 — stronger K for repolarization
        "g_CaT_max": 1.5,     # T-type Ca for rebound bursting (kept)
        "E_CaT": 120.0,
        "g_h_max": 0.2,       # was 0.3 — modest Ih for sag
        "E_h": -40.0,
        "g_M_max": 0.3,       # was 0.5 — modest AHP
        "g_NaP_max": 0.15,    # was 0.8 — real biophysical density
        "E_L": -68.0,         # was -70 inherited — slightly depolarized for
                               # spontaneous activity, but not at threshold
        "v_rest_hh": -62.0,
    })

    # Globus pallidus externus (GPe) pacemaking neuron.
    # Re-tuned (2026-04-25): same NaP issue as STN. Real GPe has lower NaP
    # than original preset suggested. Cooper & Stanford 2000: GPe has high
    # tonic rate (30-60 Hz) supported by g_Na (~80) + g_K (~15) balance,
    # not by extreme NaP.
    GPE_PACEMAKER = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    GPE_PACEMAKER.update({
        "g_Na_max": 80.0,     # was 55 — high Na for fast tonic firing
        "g_K_max": 15.0,      # was 5.5 — strong K for high-rate repolarization
        "g_CaT_max": 0.0,
        "g_h_max": 0.1,       # was 0.2 — modest Ih (some sag)
        "E_h": -35.0,
        "g_M_max": 0.5,       # was 1.0 — moderate AHP (allows high rates)
        "g_NaP_max": 0.1,     # was 0.8 — real biophysical density
        "E_L": -65.0,         # was -70 — slightly depolarized to support 30 Hz tonic
        "v_rest_hh": -60.0,
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
        # Re-tuned 2026-04-25: g_K 4→14 (D'Angelo 2001 grain cells fire
        # >100 Hz; real biophysical g_K_DR is high). Was capped at 66 Hz
        # but should be capable of much higher.
        "C_m": 0.8,           # Small cells
        "g_Na_max": 40.0,     # Moderate Na
        "g_K_max": 14.0,
        "g_L": 0.08,          # High input resistance (lower leak)
        "g_CaT_max": 0.0,     # Minimal CaT
        "g_h_max": 0.05,      # was 0.15 — reduced to avoid mild rest depolarization
        "E_h": -30.0,
        "g_M_max": 0.3,       # Mild adaptation
        "g_NaP_max": 0.05,    # was 0.2 — reduced to keep rest near labeled -68
        "E_L": -72.0,
        "v_rest_hh": -68.0,
    })

    # Spinal motor neuron (Powers & Binder 2001, Heckman & Enoka 2012)
    SPINAL_MOTOR = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    SPINAL_MOTOR.update({
        # Re-tuned 2026-04-25: g_K 7→14 (allows higher firing rates),
        # g_NaP 0.6→0.15 (was pulling rest to -55 mV; biology has lower NaP).
        "C_m": 1.5,           # Large alpha motor neuron soma
        "g_Na_max": 70.0,     # Strong Na for reliable spiking
        "g_K_max": 14.0,
        "g_CaT_max": 1.2,     # CaT for plateau potentials / bistability
        "E_CaT": 120.0,
        "g_h_max": 0.3,       # Ih contributes to resting conductance
        "E_h": -30.0,
        "g_M_max": 1.0,       # M-current for adaptation and AHP
        "g_NaP_max": 0.15,    # was 0.6 — biological density
        "E_L": -70.0,
        "v_rest_hh": -65.0,
    })

    # Spinal inhibitory interneuron (Renshaw / Ia inhibitory, Jankowska 2001)
    SPINAL_INTERNEURON = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    SPINAL_INTERNEURON.update({
        # Re-tuned 2026-04-25: g_K 6→14 for high firing rates.
        "C_m": 0.9,           # Moderate soma size
        "g_Na_max": 55.0,     # Moderate Na
        "g_K_max": 14.0,
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
        # Re-tuned 2026-04-25: g_K 5→12 (was getting depolarization block),
        # g_NaP 0.5→0.15 (rest was -52 mV due to overly strong NaP).
        # Note: 0.15 is still a "moderate" NaP — enough to support persistent
        # activity in a network context (Wang 2001 needs ~0.1-0.2 g_NaP) but
        # not so strong it dominates rest in isolated cell tests.
        "C_m": 1.0,           # Standard pyramidal capacitance
        "g_Na_max": 50.0,     # Moderate Na (PFC pyramidals fire slower than L5 PT)
        "g_K_max": 12.0,
        "g_CaT_max": 0.3,     # was 0.5 — slightly lower CaT
        "E_CaT": 120.0,
        "g_h_max": 0.15,      # was 0.25 — modest Ih (still allows resonance)
        "E_h": -30.0,
        "g_M_max": 0.8,       # Moderate M-current for spike frequency adaptation
        "g_NaP_max": 0.15,
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
        # Re-tuned 2026-04-25 (v3): Earlier retune over-corrected — moved
        # E_L too far negative (-60) and the cell stopped firing entirely.
        # Real DA neurons NEED depolarized rest (-55 mV range) to support
        # the slow autonomous pacemaking via Cav1 (L-type Ca) at threshold.
        # Restored E_L=-55 with the reduced (but non-zero) CaT/NaP.
        "C_m": 1.2,
        "g_Na_max": 40.0,     # was 35 — slightly higher for spike upstroke
        "g_K_max": 8.0,       # was 4 — still allows slow firing
        "g_CaT_max": 1.5,     # was 2.0 then 1.0 — middle ground
        "E_CaT": 120.0,
        "g_h_max": 0.3,
        "E_h": -30.0,
        "g_M_max": 0.3,       # Mild M (SK analog for AHP)
        "g_NaP_max": 0.15,    # Small persistent Na
        "E_L": -55.0,         # Depolarized rest — autonomous firing
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
        # Re-tuned 2026-04-25: g_K 5→10. IO has slow STO dynamics so g_K
        # doesn't need to be very high, but 5 was causing stuck-at-2-Hz.
        "C_m": 1.0,           # Standard capacitance
        "g_Na_max": 40.0,     # Moderate Na
        "g_K_max": 10.0,
        "g_CaT_max": 1.5,     # STRONG CaT — drives subthreshold oscillations
        "E_CaT": 120.0,
        "g_h_max": 0.5,       # STRONG Ih — rebound from inhibition, oscillation partner
        "E_h": -30.0,
        "g_M_max": 0.3,       # Mild M-current
        "g_NaP_max": 0.3,     # Moderate persistent Na for oscillation support
        "E_L": -65.0,
        "v_rest_hh": -60.0,
    })

    # BG output gate: GPi (and SNr, which is functionally similar — primary BG
    # output to thalamus, suppressing motor activity at rest, releasing it on
    # action selection via direct-pathway disinhibition). Higher tonic firing
    # than GPe, modest g_NaP.
    GPI_OUTPUT = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    GPI_OUTPUT.update({
        # Bevan & Wilson 1999, Hashimoto 2003: GPi tonic 60-80 Hz at rest.
        # R3.8 (2026-04-29): tuned NaP + Ih + SK-equivalent (M-current) per
        # PBR-160 ch 9 Deniau pp 157-158. SNr/GPi 40-80 Hz autonomous
        # pacemaker rests on (1) slowly-inactivating TTX-sensitive NaP,
        # (2) some Ih (slow Ca spikes below -60 mV), (3) SK channels
        # coupled to Cav2.2 — apamin reduces firing precision. Our
        # framework has no explicit SK; we use the M-current (g_M) as
        # an AHP proxy. Earlier values were too conservative for the
        # biology — apamin-sensitive AHP is large in SNr.
        "g_Na_max": 80.0,
        "g_K_max": 18.0,    # Stronger K than GPe — allows higher tonic rate
        "g_CaT_max": 0.0,
        "g_h_max": 0.15,    # was 0.05 — Ih supports slow spikes per Deniau
        "E_h": -35.0,
        "g_M_max": 1.0,     # was 0.4 — SK-equivalent AHP (firing precision)
        "g_NaP_max": 0.4,   # was 0.12 — strong NaP pacemaker drive
        "E_L": -64.0,       # Slightly more depolarized than GPe (higher tonic rate)
        "v_rest_hh": -60.0,
    })

    # Striatal MSN — Direct pathway (D1 receptor expressing).
    # Functionally enhanced by DA via D1 receptors (DA → cAMP → enhanced response).
    # Biophysics: similar to base MSN but slightly higher AHP, supports up-state
    # ramping and bimodal firing pattern. Wilson & Kawaguchi 1996, Mahon 2003.
    STRIATAL_MSN_D1 = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    STRIATAL_MSN_D1.update({
        "g_Na_max": 45.0,
        "g_K_max": 14.0,
        "g_M_max": 1.0,    # Slightly less than D2 (D1 cells more prone to up-state)
        "g_CaT_max": 0.0,
        "g_h_max": 0.3,
        "E_h": -35.0,
        "g_NaP_max": 0.0,
        "E_L": -78.0,      # Strongly hyperpolarized rest (down-state)
        "v_rest_hh": -75.0,
    })

    # Striatal MSN — Indirect pathway (D2 receptor expressing).
    # Functionally suppressed by DA via D2 receptors. Slightly more KIR
    # inward rectifier than D1 (modeled by stronger M-current) so harder to
    # drive into up-state without DA-mediated D2 suppression.
    STRIATAL_MSN_D2 = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    STRIATAL_MSN_D2.update({
        "g_Na_max": 45.0,
        "g_K_max": 14.0,
        "g_M_max": 1.4,    # Stronger M-current = stronger AHP, harder to fire
        "g_CaT_max": 0.0,
        "g_h_max": 0.25,   # Slightly less Ih
        "E_h": -35.0,
        "g_NaP_max": 0.0,
        "E_L": -78.0,
        "v_rest_hh": -75.0,
    })

    # Striatal Tonically Active Neuron (TAN) — Cholinergic interneuron (~1-3% of
    # striatal cells, but functionally critical: modulates DA gain, enables
    # learning windows). Spontaneously firing 2-10 Hz, strong AHP gives long ISI.
    # Bennett & Wilson 1999, Reynolds et al. 2004.
    STRIATAL_TAN = REALISTIC_L5_PYRAMIDAL_RS_37C.copy()
    STRIATAL_TAN.update({
        "g_Na_max": 60.0,
        "g_K_max": 12.0,
        "g_M_max": 0.8,    # Long after-hyperpolarization for slow tonic
        "g_CaT_max": 0.0,
        "g_h_max": 0.3,    # Ih supports slow autonomous oscillation
        "E_h": -40.0,
        "g_NaP_max": 0.1,  # Modest NaP for tonic firing
        "E_L": -60.0,      # Depolarized rest enables spontaneous firing
        "v_rest_hh": -57.0,
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
        NeuronType.HH_GPI_OUTPUT: GPI_OUTPUT.copy(),
        NeuronType.HH_STRIATAL_MSN_D1: STRIATAL_MSN_D1.copy(),
        NeuronType.HH_STRIATAL_MSN_D2: STRIATAL_MSN_D2.copy(),
        NeuronType.HH_STRIATAL_TAN: STRIATAL_TAN.copy(),
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
        # ---- Basal Ganglia + Thalamus (Phase A → B Phase) ----
        # Izhikevich 2003 Table II, "Simple Model of Spiking Neurons" IEEE TNN 14(6).
        NeuronType.IZH2007_STRIATAL_MSN: {
            # Medium spiny neuron — D1/D2 striatal projection neurons.
            # Down-state stable, ramping with cortical input. Up-state firing
            # rate moderate (~5-30 Hz). Wilson & Kawaguchi 1996, Mahon 2003.
            #
            # R3.9 (2026-04-29) catalog note (PBR-160 ch 6 Wilson):
            # The biological MSN bistability rests on TWO voltage-dependent
            # K+ currents — KIR2 (clamps RMP -80 to -95 mV, IR ~20-60 MOhm)
            # and Kv-1.2/Kv-2.1 (deactivates ~-60 mV). Both deactivate near
            # -60 mV → input resistance PEAKS 6× higher (~150-300 MOhm) at
            # -60 mV, making the dendrite electrotonically compact at that
            # potential. KIR2 is developmentally late (P25-P28 in rat).
            # The negative b=-20 below approximates KIR2's contribution
            # (subthreshold u tracks -(V-vr), pulls toward rest), but the
            # explicit IR peak at -60 mV is not captured by Izhikevich.
            # A faithful implementation requires a new fused kernel that
            # blends KIR2 + Kv2 voltage-gated leaks; out of scope for this
            # remediation pass. Documented in catalog-remediation-pass.md.
            "C": 50.0, "k": 1.0, "vr": -80.0, "vt": -25.0, "vpeak": 40.0,
            "a": 0.01, "b": -20.0, "c_reset": -55.0, "d_increment": 150.0,
        },
        NeuronType.IZH2007_THALAMIC_RELAY: {
            # Thalamocortical relay neuron in tonic mode (no LTS bursting here;
            # bursting requires conditional T-current activation which Izh
            # doesn't natively model — use HH_THALAMIC_RELAY_TBURST for that).
            # In tonic firing mode this is RS-like at higher rate.
            "C": 200.0, "k": 1.6, "vr": -60.0, "vt": -50.0, "vpeak": 35.0,
            "a": 0.01, "b": 15.0, "c_reset": -60.0, "d_increment": 10.0,
        },
        NeuronType.IZH2007_THALAMIC_RETICULAR: {
            # TRN — bursting LTS-like inhibitory cell. Strong adaptation,
            # low threshold. Destexhe 1996, Wilson & Kawaguchi.
            "C": 40.0, "k": 0.25, "vr": -65.0, "vt": -45.0, "vpeak": 0.0,
            "a": 0.015, "b": 10.0, "c_reset": -55.0, "d_increment": 50.0,
        },
        NeuronType.IZH2007_GPE_PACEMAKER: {
            # GPe — autonomous pacemaker firing 30-60 Hz. Cooper & Stanford 2000,
            # Bevan et al. 2002.
            "C": 60.0, "k": 1.0, "vr": -65.0, "vt": -50.0, "vpeak": 25.0,
            "a": 0.05, "b": 1.0, "c_reset": -50.0, "d_increment": 20.0,
        },
        NeuronType.IZH2007_GPI_OUTPUT: {
            # GPi / SNr output — high tonic rate (60-80 Hz at rest), inhibitory.
            # Acts as the BG output gate (disinhibits thalamus on action selection).
            "C": 60.0, "k": 1.0, "vr": -65.0, "vt": -50.0, "vpeak": 25.0,
            "a": 0.05, "b": 2.0, "c_reset": -50.0, "d_increment": 25.0,
        },
        NeuronType.IZH2007_STN_BURST: {
            # STN — bursty pacemaker. Bevan & Wilson 1999. Strong rebound burst
            # after inhibition release. Uses negative b for low-threshold dynamics.
            "C": 80.0, "k": 1.5, "vr": -60.0, "vt": -50.0, "vpeak": 30.0,
            "a": 0.005, "b": -1.0, "c_reset": -45.0, "d_increment": 75.0,
        },
        NeuronType.IZH2007_HIPPO_PYRAMIDAL: {
            # Hippocampal CA1/CA3 pyramidal cell. Intrinsically bursting (IB)-like
            # phenotype with mild adaptation. Mason & Larkman 1990.
            "C": 100.0, "k": 0.7, "vr": -65.0, "vt": -40.0, "vpeak": 35.0,
            "a": 0.01, "b": 5.0, "c_reset": -55.0, "d_increment": 50.0,
        },
        NeuronType.IZH2007_DOPAMINE: {
            # VTA/SNc dopaminergic neuron. Slow tonic firing 1-5 Hz spontaneously,
            # bursts (>15 Hz) in response to phasic input. Grace & Bunney 1984.
            "C": 100.0, "k": 0.9, "vr": -65.0, "vt": -45.0, "vpeak": 40.0,
            "a": 0.01, "b": 1.0, "c_reset": -55.0, "d_increment": 5.0,
        },
        NeuronType.IZH2007_STRIATAL_MSN_D1: {
            # Direct-pathway striatal MSN (D1+ receptor). Same biophysics as
            # base MSN, semantic distinction is for DA modulation routing.
            "C": 50.0, "k": 1.0, "vr": -80.0, "vt": -25.0, "vpeak": 40.0,
            "a": 0.01, "b": -20.0, "c_reset": -55.0, "d_increment": 150.0,
        },
        NeuronType.IZH2007_STRIATAL_MSN_D2: {
            # Indirect-pathway MSN (D2 receptor). Slightly stiffer (more KIR-like
            # behavior via stronger b) — harder to enter up-state without DA
            # suppression. Identical to D1 except `b` (recovery slope).
            "C": 50.0, "k": 1.0, "vr": -80.0, "vt": -25.0, "vpeak": 40.0,
            "a": 0.01, "b": -25.0, "c_reset": -55.0, "d_increment": 180.0,
        },
        NeuronType.IZH2007_STRIATAL_TAN: {
            # Tonically Active Neuron (cholinergic interneuron). Spontaneously
            # firing 2-10 Hz with strong AHP between spikes. Bennett & Wilson 1999.
            "C": 80.0, "k": 0.5, "vr": -60.0, "vt": -45.0, "vpeak": 40.0,
            "a": 0.05, "b": 0.5, "c_reset": -50.0, "d_increment": 30.0,
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
            # Accept any IZH2007_* enum name (Phase A→B added BG/thal/HC/DA presets)
            if neuron_type_enum.name.startswith("IZH2007_") and neuron_type_enum in DefaultIzhikevichParamsManager.PARAMS:
                return DefaultIzhikevichParamsManager.PARAMS[neuron_type_enum].copy()
            print(f"Warning: Requested legacy type {neuron_type_enum} for 2007 formulation. Using RS_CORTICAL_PYRAMIDAL fallback.")
            return DefaultIzhikevichParamsManager.FALLBACK_2007.copy()
        else: # Legacy formulation
            if 'LEGACY' in neuron_type_enum.name:
                 return DefaultIzhikevichParamsManager.PARAMS.get(neuron_type_enum, DefaultIzhikevichParamsManager.FALLBACK_LEGACY).copy()
            print(f"Warning: Requested 2007 type {neuron_type_enum} for legacy formulation. Using RS_EXCITATORY_LEGACY fallback.")
            return DefaultIzhikevichParamsManager.FALLBACK_LEGACY.copy()


# --- AdEx (Adaptive Exponential Integrate-and-Fire) Parameter Defaults ---
# Brette & Gerstner 2005 J Neurophysiol "Adaptive Exponential Integrate-and-Fire
# model as an effective description of neuronal activity" — Table 1 phenotypes.
# Parameters: C (pF), g_L (nS), E_L (mV), V_T (mV), Delta_T (mV),
#             a (nS), tau_w (ms), b (pA), V_r (mV), V_peak (mV).
class DefaultAdExParamsManager:
    PARAMS = {
        NeuronType.ADEX_RS_CORTICAL_PYRAMIDAL: {
            # Cortical RS pyramidal — moderate adaptation. Brette & Gerstner 2005 default.
            "C": 281.0, "g_L": 30.0, "E_L": -70.6, "V_T": -50.4, "Delta_T": 2.0,
            "a": 4.0, "tau_w": 144.0, "b": 80.5, "V_r": -70.6, "V_peak": -40.0,
        },
        NeuronType.ADEX_FS_CORTICAL_INTERNEURON: {
            # PV+ fast-spiking — minimal adaptation, fast kinetics.
            "C": 200.0, "g_L": 10.0, "E_L": -65.0, "V_T": -50.0, "Delta_T": 2.0,
            "a": 0.001, "tau_w": 20.0, "b": 0.0, "V_r": -65.0, "V_peak": -40.0,
        },
        NeuronType.ADEX_IB_BURSTING: {
            # Intrinsically bursting cortical (e.g. some L5 PT cells).
            "C": 200.0, "g_L": 10.0, "E_L": -58.0, "V_T": -50.0, "Delta_T": 2.0,
            "a": 0.001, "tau_w": 720.0, "b": 120.0, "V_r": -46.0, "V_peak": -40.0,
        },
        NeuronType.ADEX_CH_CHATTERING: {
            # Chattering — high-rate gamma drivers (some L2/3 cells).
            "C": 200.0, "g_L": 10.0, "E_L": -65.0, "V_T": -50.0, "Delta_T": 2.0,
            "a": 4.0, "tau_w": 20.0, "b": 400.0, "V_r": -55.0, "V_peak": -40.0,
        },
        NeuronType.ADEX_LTS_LOW_THRESHOLD: {
            # Low-threshold spiking interneuron (somatostatin+ Martinotti cells).
            "C": 200.0, "g_L": 10.0, "E_L": -56.0, "V_T": -50.0, "Delta_T": 2.0,
            "a": 20.0, "tau_w": 20.0, "b": 0.0, "V_r": -65.0, "V_peak": -40.0,
        },
        NeuronType.ADEX_STRIATAL_MSN: {
            # Down-state stable MSN — hyperpolarized rest, ramping in response.
            # Wilson & Kawaguchi 1996; Naud & Gerstner 2008 AdEx fit.
            "C": 100.0, "g_L": 10.0, "E_L": -78.0, "V_T": -45.0, "Delta_T": 2.5,
            "a": 0.0, "tau_w": 100.0, "b": 200.0, "V_r": -55.0, "V_peak": -40.0,
        },
        NeuronType.ADEX_DOPAMINE: {
            # VTA/SNc DA neuron — slow tonic 1-5 Hz, can burst on phasic input.
            # Drion 2011, Putzier 2009.
            "C": 150.0, "g_L": 5.0, "E_L": -55.0, "V_T": -45.0, "Delta_T": 2.0,
            "a": 1.0, "tau_w": 200.0, "b": 60.0, "V_r": -55.0, "V_peak": -40.0,
        },
    }
    FALLBACK = PARAMS[NeuronType.ADEX_RS_CORTICAL_PYRAMIDAL].copy()

    @staticmethod
    def get_params(neuron_type_enum):
        return DefaultAdExParamsManager.PARAMS.get(
            neuron_type_enum, DefaultAdExParamsManager.FALLBACK
        ).copy()


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
