"""Neural structure profiles, connectivity motifs, and profile/neuron-type helpers."""

from sim.enums import NeuronModel, NeuronType, DefaultHodgkinHuxleyParams

# ---------------------------------------------------------------------------
# Neural Structure Profiles
# ---------------------------------------------------------------------------

NEURAL_STRUCTURE_PROFILES = {
    "GENERIC_UNSTRUCTURED": {
        "display_name": "Generic Unstructured Network",
        "description": "Random traits with no specific brain-region structure.",
        "recommended_neuron_model": NeuronModel.IZHIKEVICH.name,
        "trait_definitions": [],  # Falls back to existing random trait logic
        "default_core_overrides": {}
    },
    "CORTEX_L23_RS_FS": {
        "display_name": "Neocortex L2/3 RS/FS",
        "description": "Cortical microcircuit with ~80% excitatory RS pyramidal cells and ~20% FS interneurons.",
        "recommended_neuron_model": NeuronModel.IZHIKEVICH.name,
        # Optional laminar-like cortical motif when enabled
        "connectivity_motif": "CORTEX_L23_RS_FS",
        # When using HH, approximate L2/3 excitatory cells with the L5 RS HH preset
        "default_hh_neuron_type": NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Excitatory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.8},
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.2},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 1,
            "connectivity_k": 10,
            "connectivity_p_rewire": 0.1,
        },
    },
    "HIPPOCAMPUS_CA1_RS_FS": {
        "display_name": "Hippocampus CA1 RS/FS",
        "description": "Hippocampal CA1-like network with pyramidal cells and diverse interneurons (modeled as FS).",
        "recommended_neuron_model": NeuronModel.IZHIKEVICH.name,
        # Use CA1-specific HH bursting preset when HH model is active
        "default_hh_neuron_type": NeuronType.HH_CA1_PYRAMIDAL_BURST.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Excitatory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.7},
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.3},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 1,
            "connectivity_k": 8,
            "connectivity_p_rewire": 0.15,
        },
    },
    "CEREBELLAR_CORTEX_SIMPLE": {
        "display_name": "Cerebellar Cortex (simplified)",
        "description": "Simplified cerebellar cortex: granule-like excitatory cells and Purkinje / interneuron inhibition.",
        "recommended_neuron_model": NeuronModel.HODGKIN_HUXLEY.name,
        # Dedicated cerebellar HH preset: Purkinje-like for excitatory trait, granule-like available
        "default_hh_neuron_type": NeuronType.HH_CEREBELLAR_PURKINJE.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Excitatory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.75},
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.25},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 1,
            "connectivity_k": 6,
            "connectivity_p_rewire": 0.05,
            "syn_tau_g_e": 2.0,
            "syn_tau_g_i": 8.0,
        },
    },
    "SPINAL_CORD_SEGMENT": {
        "display_name": "Spinal Cord Segment",
        "description": "Segment with excitatory motor/interneurons and strong recurrent inhibition.",
        "recommended_neuron_model": NeuronModel.HODGKIN_HUXLEY.name,
        # Dedicated spinal HH preset: motor neuron for excitatory trait, interneuron available
        "default_hh_neuron_type": NeuronType.HH_SPINAL_MOTOR.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Excitatory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.6},
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.4},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 1,
            "connectivity_k": 12,
            "connectivity_p_rewire": 0.05,
        },
    },
    "BASAL_GANGLIA_STRIATUM": {
        "display_name": "Basal Ganglia Striatum",
        "description": "Striatal network with ~95% inhibitory MSNs and a small FS interneuron population.",
        "recommended_neuron_model": NeuronModel.IZHIKEVICH.name,
        # Use MSN-specific HH preset when HH model is active
        "default_hh_neuron_type": NeuronType.HH_STRIATAL_MSN.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.95},  # MSNs modeled as RS-like inhibitory
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.05},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 0,
            "connectivity_k": 20,
            "connectivity_p_rewire": 0.2,
        },
    },
    "THALAMUS_TC_TRN": {
        "display_name": "Thalamus TC-TRN Loop",
        "description": "Thalamic relay (TC) and reticular (TRN) network with excitatory-inhibitory recurrence and bursting.",
        "recommended_neuron_model": NeuronModel.IZHIKEVICH.name,
        "connectivity_motif": "THALAMUS_TC_TRN",
        # Use thalamic relay HH preset when HH model is active
        "default_hh_neuron_type": NeuronType.HH_THALAMIC_RELAY_TBURST.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Excitatory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.6},
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.4},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 1,
            "connectivity_k": 8,
            "connectivity_p_rewire": 0.1,
        },
    },
    "HIPPOCAMPUS_CA3_RECURRENT": {
        "display_name": "Hippocampus CA3 Recurrent",
        "description": "Hippocampal CA3-like network with strong recurrent excitation and interneuron-mediated inhibition.",
        "recommended_neuron_model": NeuronModel.IZHIKEVICH.name,
        "connectivity_motif": "HIPPOCAMPUS_CA3_RECURRENT",
        # Use CA3-specific HH bursting preset when HH model is active
        "default_hh_neuron_type": NeuronType.HH_CA3_PYRAMIDAL_BURST.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Excitatory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.7},
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.3},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 1,
            "connectivity_k": 10,
            "connectivity_p_rewire": 0.15,
        },
    },
    "CORTEX_L4_INPUT_LAYER": {
        "display_name": "Cortex L4 Input Layer",
        "description": "Sensory cortical input layer with spiny stellate-like excitatory cells and FS interneurons.",
        "recommended_neuron_model": NeuronModel.IZHIKEVICH.name,
        "connectivity_motif": "CORTEX_L4_INPUT_LAYER",
        # Use generic cortical RS HH preset as a proxy for L4 spiny stellate cells
        "default_hh_neuron_type": NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Excitatory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.8},
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.2},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 1,
            "connectivity_k": 10,
            "connectivity_p_rewire": 0.1,
        },
    },
    "BASAL_GANGLIA_STN_GPE": {
        "display_name": "Basal Ganglia STN-GPe",
        "description": "Subthalamic nucleus (STN) and globus pallidus externus (GPe) excitatory-inhibitory loop.",
        "recommended_neuron_model": NeuronModel.IZHIKEVICH.name,
        "connectivity_motif": "BASAL_GANGLIA_STN_GPE",
        # Use STN bursting HH preset when HH model is active
        "default_hh_neuron_type": NeuronType.HH_STN_BURST.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Excitatory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.3},
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.7},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 1,
            "connectivity_k": 14,
            "connectivity_p_rewire": 0.15,
        },
    },
    "CORTEX_L5_DEEP_OUTPUT": {
        "display_name": "Neocortex L5 Deep Output",
        "description": "Layer 5 corticofugal output circuit: thick-tufted pyramidal tract (PT) neurons with burst-firing properties.",
        "recommended_neuron_model": NeuronModel.HODGKIN_HUXLEY.name,
        "default_hh_neuron_type": NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Excitatory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.80},
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.20},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 1,
            "connectivity_k": 12,
            "connectivity_p_rewire": 0.15,
            "syn_tau_g_e": 5.0,
            "syn_tau_g_i": 10.0,
        },
    },
    "PREFRONTAL_CORTEX_WM": {
        "display_name": "Prefrontal Cortex (Working Memory)",
        "description": "PFC persistent activity network: strong NMDA recurrence enables working memory-like sustained firing.",
        "recommended_neuron_model": NeuronModel.HODGKIN_HUXLEY.name,
        "default_hh_neuron_type": NeuronType.HH_PFC_PYRAMIDAL.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Excitatory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.75},
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.25},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 1,
            "connectivity_k": 15,
            "connectivity_p_rewire": 0.2,
            "syn_tau_g_e": 5.0,
            "syn_tau_g_i": 10.0,
        },
    },
    "OLFACTORY_BULB": {
        "display_name": "Olfactory Bulb",
        "description": "Mitral/tufted cells with strong granule cell inhibition. High E/I ratio drives gamma/theta oscillations.",
        "recommended_neuron_model": NeuronModel.HODGKIN_HUXLEY.name,
        "default_hh_neuron_type": NeuronType.HH_OLFACTORY_MITRAL.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Excitatory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.50},
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.50},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 1,
            "connectivity_k": 10,
            "connectivity_p_rewire": 0.1,
            "syn_tau_g_e": 3.0,
            "syn_tau_g_i": 15.0,
        },
    },
    "DOPAMINERGIC_MIDBRAIN": {
        "display_name": "Dopaminergic Midbrain (SNc/VTA)",
        "description": "Midbrain dopamine circuit: autonomous pacemaker DA neurons (65%) with GABAergic interneurons (35%).",
        "recommended_neuron_model": NeuronModel.HODGKIN_HUXLEY.name,
        "default_hh_neuron_type": NeuronType.HH_DOPAMINE_SNC.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Excitatory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.65},
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.35},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 1,
            "connectivity_k": 8,
            "connectivity_p_rewire": 0.1,
            "syn_tau_g_e": 4.0,
            "syn_tau_g_i": 10.0,
        },
    },
    "CORTEX_GAMMA_FS_NETWORK": {
        "display_name": "Cortical Gamma Oscillation Network",
        "description": "Inhibition-dominated network for studying gamma (30-80 Hz) oscillations driven by PV+ FS interneurons.",
        "recommended_neuron_model": NeuronModel.HODGKIN_HUXLEY.name,
        "default_hh_neuron_type": NeuronType.HH_CORTICAL_FS_INTERNEURON.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Excitatory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.40},
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.60},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 1,
            "connectivity_k": 20,
            "connectivity_p_rewire": 0.15,
            "syn_tau_g_e": 3.0,
            "syn_tau_g_i": 5.0,
        },
    },
    "INFERIOR_OLIVE": {
        "display_name": "Inferior Olive",
        "description": "Olivary neurons with CaT/Ih-driven subthreshold oscillations. Note: gap junctions not modeled.",
        "recommended_neuron_model": NeuronModel.HODGKIN_HUXLEY.name,
        "default_hh_neuron_type": NeuronType.HH_INFERIOR_OLIVE.name,
        "trait_definitions": [
            {"trait_index": 0, "role": "Excitatory", "neuron_type": NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name, "fraction": 0.90},
            {"trait_index": 1, "role": "Inhibitory", "neuron_type": NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name, "fraction": 0.10},
        ],
        "default_core_overrides": {
            "num_traits": 2,
            "enable_inhibitory_neurons": True,
            "inhibitory_trait_index": 1,
            "connectivity_k": 10,
            "connectivity_p_rewire": 0.3,
            "syn_tau_g_e": 4.0,
            "syn_tau_g_i": 8.0,
        },
    },
}

# --- Profile / neuron-type compatibility helpers (realism-focused) ---

def get_profile_default_hh_type_name(profile_name):
    """Returns the default HH neuron type name for a profile, if any."""
    profile_def = NEURAL_STRUCTURE_PROFILES.get(profile_name)
    if not profile_def:
        return None
    return profile_def.get("default_hh_neuron_type")


def get_compatible_hh_type_names_for_profile(profile_name):
    """Returns the list of HH neuron type names considered realistic for a profile.

    If a profile defines a default HH preset (and optionally an explicit
    allowed_hh_neuron_types list), we restrict to those. Generic/unstructured
    profiles fall back to all defined HH presets.
    """
    profile_def = NEURAL_STRUCTURE_PROFILES.get(profile_name)
    if not profile_def:
        # Unknown profile: allow all defined HH presets
        return [nt.name for nt in NeuronType if nt in DefaultHodgkinHuxleyParams.PARAMS]

    allowed = []

    # Optional explicit list of compatible HH types
    explicit_list = profile_def.get("allowed_hh_neuron_types") or []
    for name in explicit_list:
        if name in NeuronType.__members__ and NeuronType[name] in DefaultHodgkinHuxleyParams.PARAMS:
            if name not in allowed:
                allowed.append(name)

    # Always include the profile's default HH preset if present
    default_hh = profile_def.get("default_hh_neuron_type")
    if default_hh and default_hh in NeuronType.__members__ and NeuronType[default_hh] in DefaultHodgkinHuxleyParams.PARAMS:
        if default_hh not in allowed:
            allowed.append(default_hh)

    # If nothing explicit was configured, fall back to all HH presets
    if not allowed:
        return [nt.name for nt in NeuronType if nt in DefaultHodgkinHuxleyParams.PARAMS]

    return allowed


def enforce_profile_neuron_type_compatibility(core_cfg):
    """Clamp core_cfg to a realistic (profile, neuron model, HH preset) combo.

    Currently this enforces that when running the HH model with a structured
    profile, the default HH neuron type is the profile-appropriate preset.
    """
    try:
        profile_name = getattr(core_cfg, "neural_profile_name", "GENERIC_UNSTRUCTURED")
        model_name = getattr(core_cfg, "neuron_model_type", NeuronModel.IZHIKEVICH.name)

        if model_name != NeuronModel.HODGKIN_HUXLEY.name:
            return

        allowed_hh = get_compatible_hh_type_names_for_profile(profile_name)
        if not allowed_hh:
            return

        current_hh = getattr(core_cfg, "default_neuron_type_hh", allowed_hh[0])
        if current_hh not in allowed_hh:
            new_hh = allowed_hh[0]
            print(
                f"[PROFILE_COMPAT] Profile '{profile_name}' does not support HH preset '{current_hh}'. "
                f"Using '{new_hh}' instead for realism."
            )
            core_cfg.default_neuron_type_hh = new_hh
    except Exception as e:
        print(f"Warning: failed to enforce profile/neuron-type compatibility: {e}")

# --- Connectivity Motifs Registry ---
# Each motif describes high-level population connectivity based on trait indices.
# The rules are approximate and designed to be GPU-friendly while capturing
# canonical motif structure (e.g., TC-TRN loops, CA3 recurrence, STN-GPe loops).
CONNECTIVITY_MOTIFS = {
    "CORTEX_L23_RS_FS": {
        "description": "Canonical L2/3 cortical microcircuit with RS excitatory and FS inhibitory neurons.",
        # Trait 0: excitatory RS, Trait 1: inhibitory FS
        "rules": [
            # Excitatory sources
            {"source_traits": [0], "target_traits": [0], "k_fraction": 0.6, "weight_scale": 1.0},  # E->E
            {"source_traits": [0], "target_traits": [1], "k_fraction": 0.4, "weight_scale": 1.0},  # E->I
            # Inhibitory sources
            {"source_traits": [1], "target_traits": [0], "k_fraction": 0.8, "weight_scale": 1.0},  # I->E
            {"source_traits": [1], "target_traits": [1], "k_fraction": 0.2, "weight_scale": 0.8},  # I->I
        ],
    },
    "THALAMUS_TC_TRN": {
        "description": "Thalamic relay (TC) and reticular (TRN) loop.",
        # Trait 0: TC excitatory, Trait 1: TRN inhibitory
        "rules": [
            {"source_traits": [0], "target_traits": [1], "k_fraction": 0.7, "weight_scale": 1.0},  # TC->TRN
            {"source_traits": [0], "target_traits": [0], "k_fraction": 0.3, "weight_scale": 0.7},  # TC->TC
            {"source_traits": [1], "target_traits": [0], "k_fraction": 0.7, "weight_scale": 1.0},  # TRN->TC
            {"source_traits": [1], "target_traits": [1], "k_fraction": 0.3, "weight_scale": 0.7},  # TRN->TRN
        ],
    },
    "HIPPOCAMPUS_CA3_RECURRENT": {
        "description": "Hippocampal CA3 recurrent excitatory network with inhibitory feedback.",
        # Trait 0: CA3 pyramidal, Trait 1: interneurons
        "rules": [
            {"source_traits": [0], "target_traits": [0], "k_fraction": 0.7, "weight_scale": 1.0},  # E->E strong recurrence
            {"source_traits": [0], "target_traits": [1], "k_fraction": 0.3, "weight_scale": 0.8},  # E->I
            {"source_traits": [1], "target_traits": [0], "k_fraction": 0.8, "weight_scale": 1.0},  # I->E
            {"source_traits": [1], "target_traits": [1], "k_fraction": 0.2, "weight_scale": 0.7},  # I->I
        ],
    },
    "CORTEX_L4_INPUT_LAYER": {
        "description": "Cortical L4 input layer with spiny stellate-like excitatory neurons and FS interneurons.",
        # Trait 0: excitatory, Trait 1: inhibitory
        "rules": [
            {"source_traits": [0], "target_traits": [0], "k_fraction": 0.5, "weight_scale": 1.0},  # E->E
            {"source_traits": [0], "target_traits": [1], "k_fraction": 0.5, "weight_scale": 1.0},  # E->I
            {"source_traits": [1], "target_traits": [0], "k_fraction": 0.8, "weight_scale": 1.0},  # I->E
            {"source_traits": [1], "target_traits": [1], "k_fraction": 0.2, "weight_scale": 0.8},  # I->I
        ],
    },
    "BASAL_GANGLIA_STN_GPE": {
        "description": "STN-GPe excitatory-inhibitory loop in basal ganglia.",
        # Trait 0: STN-like excitatory, Trait 1: GPe-like inhibitory
        "rules": [
            {"source_traits": [0], "target_traits": [1], "k_fraction": 0.9, "weight_scale": 1.0},  # STN->GPe
            {"source_traits": [0], "target_traits": [0], "k_fraction": 0.1, "weight_scale": 0.7},  # STN->STN (weak)
            {"source_traits": [1], "target_traits": [0], "k_fraction": 0.6, "weight_scale": 1.0},  # GPe->STN
            {"source_traits": [1], "target_traits": [1], "k_fraction": 0.4, "weight_scale": 0.8},  # GPe->GPe
        ],
    },
}
