"""SimulationBridge - core simulation engine for the neural simulator.

Manages all GPU state arrays (CuPy), simulation stepping, dynamics updates,
recording/playback to HDF5, checkpoint save/restore, and profiling.
"""

import os
import json
import time
import sys
import math
import random
import queue
import threading
import numpy as np
import h5py
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Dict, List, Optional

# Route through the backend abstraction so this module is forward-
# compatible with the NumPy backend (Phase 2 of the tiering design).
# For Phase 1, the abstraction is additive — current CuPy behavior is
# preserved exactly when sim.backend resolves to "cupy" (the default).
# GPU-specific calls (cp.cuda.*, cp.get_default_memory_pool()) remain
# unmodified in this file and will only work on the CuPy backend;
# Phase 2 work refactors those behind is_gpu_backend() guards.
try:
    from sim.backend import (
        get_backend, get_sparse_module, fuse, is_gpu_backend,
        synchronize as _backend_synchronize,
        set_device as _backend_set_device,
        get_device_mem_info as _backend_get_device_mem_info,
        get_device_properties as _backend_get_device_properties,
        get_memory_pool as _backend_get_memory_pool,
        get_pinned_memory_pool as _backend_get_pinned_memory_pool,
        to_host as _backend_to_host,
        from_host as _backend_from_host,
        get_random_state as _backend_get_random_state,
        set_random_state as _backend_set_random_state,
    )
    cp, _backend_name = get_backend()
    csp = get_sparse_module()
except ImportError:
    # Defensive bootstrap fallback (very early import contexts only).
    import cupy as cp
    try:
        import cupy.sparse as csp
    except (ImportError, ModuleNotFoundError):
        import cupyx.scipy.sparse as csp
    fuse = cp.fuse
    is_gpu_backend = lambda: True
    _backend_synchronize = lambda: cp.cuda.Stream.null.synchronize()
    _backend_set_device = lambda dev_id=0: cp.cuda.Device(dev_id).use()
    _backend_get_device_mem_info = lambda: cp.cuda.Device().mem_info
    _backend_get_device_properties = lambda dev_id=0: cp.cuda.runtime.getDeviceProperties(dev_id)
    _backend_get_memory_pool = lambda: cp.get_default_memory_pool()
    _backend_get_pinned_memory_pool = lambda: cp.get_default_pinned_memory_pool()
    _backend_to_host = lambda arr: arr.get() if hasattr(arr, "get") else arr
    _backend_from_host = lambda arr, dtype=None: cp.asarray(arr, dtype=dtype)
    _backend_get_random_state = lambda: cp.random.get_random_state()
    _backend_set_random_state = lambda s: cp.random.set_random_state(s)
    _backend_name = "cupy"

from sim.enums import (NeuronModel, NeuronType, DefaultHodgkinHuxleyParams,
                        DefaultIzhikevichParamsManager, NEURON_TYPE_MAPPER)
from sim.config import (CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
                         _create_config_from_dict, _get_full_config_dict)
from sim.profiles import (NEURAL_STRUCTURE_PROFILES, CONNECTIVITY_MOTIFS,
                          enforce_profile_neuron_type_compatibility)
from sim.connectivity import (generate_spatial_connections_gpu,
                              generate_spatial_connections_chunked,
                              generate_spatial_connections_binned,
                              generate_random_connections_large,
                              generate_spatial_connections_3d,
                              generate_watts_strogatz_3d,
                              generate_motif_connections_3d,
                              _calculate_distances_3d_gpu)
from sim.kernels import (fused_izhikevich_legacy_dynamics_update,
                         fused_izhikevich2007_dynamics_update,
                         fused_hodgkin_huxley_dynamics_update,
                         fused_hh_m_current_update,
                         fused_hh_CaT_current_update,
                         fused_hh_h_current_update,
                         fused_hh_NaP_current_update,
                         fused_adex_dynamics_update,
                         fused_conductance_decay_and_current,
                         fused_nmda_update_and_current,
                         fused_stp_decay_recovery,
                         fused_homeostasis_update,
                         fused_stdp_weight_update,
                         fused_eligibility_trace_decay)
from experiment import ExperimentEngine
from experiment.engine import experiment_config_from_dict, experiment_config_to_dict


# --- Optional dependencies ---
try:
    import hdf5plugin
    HAS_HDF5PLUGIN = True
except ImportError:
    HAS_HDF5PLUGIN = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# --- OpenGL availability check ---
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

# Recording format version (must match neural-simulator.py)
RECORDING_FORMAT_VERSION = "1.1.0-h5"

# --- Auto-tuned override support ---
AUTO_TUNED_OVERRIDES_PATH = os.path.join("simulation_profiles", "auto_tuned_overrides.json")
AUTO_TUNED_OVERRIDES = None  # Lazy-loaded mapping from combo key -> overrides dict


def _load_auto_tuned_overrides_if_needed():
    """Lazily loads auto-tuned overrides from JSON if present."""
    global AUTO_TUNED_OVERRIDES
    if AUTO_TUNED_OVERRIDES is not None:
        return

    if not os.path.exists(AUTO_TUNED_OVERRIDES_PATH):
        AUTO_TUNED_OVERRIDES = {}
        return

    try:
        with open(AUTO_TUNED_OVERRIDES_PATH, "r") as f:
            data = json.load(f)
        tuned_map = data.get("tuned_combinations", {})
        if isinstance(tuned_map, dict):
            AUTO_TUNED_OVERRIDES = tuned_map
        else:
            AUTO_TUNED_OVERRIDES = {}
        print(f"Loaded {len(AUTO_TUNED_OVERRIDES)} auto-tuned combinations from {AUTO_TUNED_OVERRIDES_PATH}.")
    except Exception as e:
        print(f"Warning: Failed to load auto-tuned overrides from {AUTO_TUNED_OVERRIDES_PATH}: {e}")
        AUTO_TUNED_OVERRIDES = {}


def get_auto_tuned_overrides_for_combo(neuron_model_type_str, profile_name_str, default_hh_type_str=None):
    """Returns auto-tuned overrides dict for a given (model, profile, HH preset) combo, if available."""
    _load_auto_tuned_overrides_if_needed()
    if not AUTO_TUNED_OVERRIDES:
        return None

    key_full = f"{neuron_model_type_str}|{profile_name_str}|{default_hh_type_str or 'NONE'}"
    entry = AUTO_TUNED_OVERRIDES.get(key_full)

    if entry is None and neuron_model_type_str != NeuronModel.HODGKIN_HUXLEY.name:
        key_model_profile = f"{neuron_model_type_str}|{profile_name_str}|NONE"
        entry = AUTO_TUNED_OVERRIDES.get(key_model_profile)

    return entry

# Module-level reference to OpenGL visualization config dict.
# Set by the main module after import to avoid circular dependencies.
opengl_viz_config = {}


# --- HDF5 Helper Functions ---
def save_dict_to_hdf5_attrs(h5_group_or_file, data_dict):
    """Saves dictionary items as attributes to an HDF5 group or file."""
    for key, value in data_dict.items():
        try:
            if value is None:
                h5_group_or_file.attrs[key] = "NoneType"
            elif isinstance(value, (list, tuple, dict)):
                h5_group_or_file.attrs[key] = json.dumps(value)
            else:
                h5_group_or_file.attrs[key] = value
        except TypeError as e:
            print(f"Warning: Could not save attribute '{key}' (value: {value}, type: {type(value)}): {e}. Storing as string.")
            try:
                h5_group_or_file.attrs[key] = str(value)
            except Exception as e_str:
                print(f"ERROR: Failed to store attribute '{key}' even as string: {e_str}")


def load_dict_from_hdf5_attrs(h5_group_or_file):
    """Loads attributes from an HDF5 group or file into a dictionary."""
    data_dict = {}
    for key, value in h5_group_or_file.attrs.items():
        if isinstance(value, str):
            if value == "NoneType":
                data_dict[key] = None
            else:
                try:
                    data_dict[key] = json.loads(value)
                except json.JSONDecodeError:
                    data_dict[key] = value
        else:
            data_dict[key] = value
    return data_dict


class SimulationBridge:
    def __init__(self, sim_core_ref=None, core_config=None, viz_config=None, runtime_state=None, gpu_config=None, ui_queue=None):
        """Initialize SimulationBridge with optional config objects.

        Args:
            sim_core_ref: Legacy parameter, not used with threading
            core_config: CoreSimConfig instance (creates default if None)
            viz_config: VisualizationConfig instance (creates default if None)
            runtime_state: RuntimeState instance (creates default if None)
            gpu_config: GPUConfig instance (creates default if None)
            ui_queue: Queue for sending data/status to UI thread (None for headless)
        """
        self.core_config = core_config if core_config is not None else CoreSimConfig()
        self.viz_config = viz_config if viz_config is not None else VisualizationConfig()
        self.runtime_state = runtime_state if runtime_state is not None else RuntimeState()
        self.gpu_config = gpu_config if gpu_config is not None else GPUConfig()
        self.ui_queue = ui_queue  # Reference to the queue for sending data/status to UI

        # --- CuPy Arrays for Simulation State ---
        self.cp_membrane_potential_v = None 
        self.cp_recovery_variable_u = None  
        self.cp_conductance_g_e = None
        self.cp_conductance_g_i = None
        self.cp_conductance_g_nmda = None
        # Cluster G v2 (2026-05-01): per-neuron NMDA mask (1.0 for neurons
        # in regions with BrainRegion.enable_nmda=True, 0.0 otherwise).
        # When None, NMDA applies globally per cfg.enable_nmda — backward
        # compatible. Set in _build_per_neuron_nmda_mask after region init.
        self.cp_nmda_neuron_mask = None
        self.cp_conductance_g_nmda_rise = None
        self.cp_external_input_current = None 
        self.cp_firing_states = None        
        self.cp_prev_firing_states = None   
        self.cp_traits = None               
        self.cp_neuron_type_ids = None      # Integer type IDs for GPU-efficient filtering
        self.cp_neuron_positions_3d = None  
        self.cp_refractory_timers = None    
        self.cp_viz_activity_timers = None

        self.cp_izh_C = None; self.cp_izh_k = None; self.cp_izh_vr = None; self.cp_izh_vt = None
        self.cp_izh_vpeak = None; self.cp_izh_a = None; self.cp_izh_b = None
        self.cp_izh_c_reset = None; self.cp_izh_d_increment = None

        self.cp_izh_legacy_a = None; self.cp_izh_legacy_b = None
        self.cp_izh_legacy_c_reset = None; self.cp_izh_legacy_d_increment = None
        self.cp_izh_legacy_vpeak = None

        # AdEx adaptation variable (w); membrane potential reuses cp_membrane_potential_v
        self.cp_adex_w = None

        self.cp_gating_variable_m = None 
        self.cp_gating_variable_h = None 
        self.cp_gating_variable_n = None 
        # Optional extended HH current state (slow K+ M-current activation and additional gates)
        self.cp_hh_m_current_activation = None
        self.cp_hh_CaT_m = None
        self.cp_hh_CaT_h = None
        self.cp_hh_h_current_q = None
        self.cp_hh_NaP_activation = None
 
        self.cp_hh_C_m = None; self.cp_hh_g_Na_max = None; self.cp_hh_g_K_max = None; self.cp_hh_g_L = None
        self.cp_hh_E_Na = None; self.cp_hh_E_K = None; self.cp_hh_E_L = None; self.cp_hh_v_peak = None

        self.cp_neuron_firing_thresholds = None
        self.cp_neuron_activity_ema = None

        # Per-neuron GABA_A reversal potential (mV). Defaults to a uniform
        # cfg.syn_reversal_potential_i; regions may override via
        # BrainRegion.syn_reversal_potential_i_override (e.g., striatal MSNs
        # use −60 mV per PBR-160 ch 6; SNc DA uses −55 mV per ch 11).
        # Allocated by _initialize_simulation_data once num_neurons is known.
        self.cp_syn_reversal_potential_i_per_neuron = None

        self.cp_connections = None

        self.cp_stp_u = None 
        self.cp_stp_x = None 

        self.cp_synapse_pulse_timers = None
        self.cp_synapse_pulse_progress = None

        # Optional per-synapse plastic mask. When not None, STDP (and other
        # plasticity paths that opt-in) only write back where this is True.
        # Set by inject_explicit_wiring when any population has plastic=False.
        self.cp_synapse_plastic_mask = None

        # Per-pathway plasticity gating (Stage 1, 2026-04-27).
        # cp_plasticity_rate_gain: per-synapse float multiplier (1.0=full plasticity,
        #   0.0=frozen). Default None when no pathway uses plasticity_gate.
        # _plasticity_gate_to_synapses: gate_name → list of synapse indices.
        # _plasticity_gate_indices_gpu: gate_name → cp.ndarray of indices.
        # _plasticity_gate_values: gate_name → current Python float value.
        # Used by curriculum, developmental staging, and future
        # neuromodulator-gated learning windows. See sim/regions.py
        # RegionPathway.plasticity_gate docstring for biology.
        self.cp_plasticity_rate_gain = None
        self._plasticity_gate_to_synapses = {}
        self._plasticity_gate_indices_gpu = {}
        self._plasticity_gate_values = {}

        # Per-pathway TRANSMISSION gating (2026-06-03; thalamocortical dynamical gating).
        # cp_transmission_gain: per-synapse float multiplier (1.0=full current, 0.0=closed).
        #   Scales effective synaptic CURRENT (complement of cp_plasticity_rate_gain, which gates only
        #   weight updates). Default None when no pathway uses transmission_gate.
        # set via set_transmission_gate(name, value). See RegionPathway.transmission_gate.
        self.cp_transmission_gain = None
        self._transmission_gate_to_synapses = {}
        self._transmission_gate_indices_gpu = {}
        self._transmission_gate_values = {}

        # Activity-driven gate couplings (bridge-internal thalamocortical loop, 2026-06-03): a transmission
        # gate can be OPENED by the firing of a control population (a thalamic gate pool) rather than an
        # external command -- thalamic activity opens the cortical route gate, in-substrate. Each entry:
        # {gate_name, control_idx (gpu), threshold, alpha (EMA), open_value, ema, last_value}.
        # Empty by default -> zero overhead in the step.
        self._gate_couplings = []

        # Cluster B.1 (2026-04-28): D1/D2 plasticity asymmetry.
        # Per-synapse sign multiplier on the reward-modulated weight update.
        # +1 for D1-targeting (and everything else), -1 for D2-targeting.
        # None when enable_d1_d2_asymmetry is False (default).
        # See docs/plans/2026-04-28-cluster-b1-d1d2-asymmetry-implementation.md.
        self.cp_d1_d2_sign = None
        # Per-synapse reward signal override (E.3 batched-replica framework).
        # When None (default), reward modulation uses the scalar
        # cfg.current_reward_signal globally. When set to a cp.float32 array
        # of shape (nnz,), each synapse's reward update uses its own value
        # instead — lets a replicated runner give each replica's synapse
        # block its own per-step reward. Composes with cp_d1_d2_sign and
        # cp_plasticity_rate_gain (those still multiply afterward).
        self.cp_per_synapse_reward_override = None

        # Cluster C v2 (2026-04-29): per-synapse action tag for compartmentalized DA.
        # int32 array of length nnz; tag[i] = action_index of synapse i's
        # POST region (∈ [0, N_ACTIONS-1]) or -1 for global / non-action-
        # specific synapses. Populated in inject_explicit_wiring() based on
        # BrainRegion.action_index of the post region.
        # See docs/plans/2026-04-29-cluster-c-v2-compartmentalized-da-design.md.
        self.cp_synapse_action_tag = None

        # Cluster E v1 (2026-04-29): topographic neuron coordinates.
        # cp_neuron_coords: cp.ndarray[float32, (n, k_dim)] where k_dim is
        # max(coordinate_dim) across all regions. None if no region declares
        # coordinates (default — backward compatible). Neurons in regions with
        # smaller coordinate_dim or coordinate_dim=0 get NaN-padded entries.
        # Populated by _initialize_simulation_data after the region manager
        # is set up. Currently informational — pathway gen uses host-side
        # numpy coordinates from RegionManager directly. Future GPU consumers
        # can read this array.
        # See docs/plans/2026-04-29-cluster-e-topographic-maps-design.md.
        self.cp_neuron_coords = None

        self.is_initialized = False

        self._mock_total_plasticity_events = 0
        self._mock_network_avg_firing_rate_hz = 0.0
        self._mock_num_spikes_this_step = 0

        # GPU-side statistics accumulators (avoid frequent GPU-CPU sync)
        self._stats_sync_counter = 0  # Counter for stats sync interval
        self._accumulated_spikes_gpu = None  # GPU-side spike accumulator
        self._last_synced_spike_count = 0  # Last synced value

        # COO matrix cache (avoid repeated conversions)
        self._cached_coo_matrix = None
        self._coo_cache_valid = False
        # Per-step derived-data caches (invalidated with COO cache on connectivity change)
        self._cached_stp_per_type = None  # (tau_f_full, tau_d_full, U_per_syn)
        self._cached_inhibitory_mask = None  # Boolean mask of inhibitory neurons

        # Structural plasticity optimization
        self._compaction_counter = 0  # Counter for deferred CSR compaction
        self._pending_eliminations = False  # Flag for pending zero-weight synapses
        self._synapse_capacity = 0  # Pre-allocated capacity for synapse arrays

        # Eligibility trace for STDP/reward
        self.cp_eligibility_trace = None

        # Neuromodulator subsystem (Session E.1, opt-in).
        # When core_config.enable_neuromodulator_subsystem is True, this is
        # populated by _initialize_simulation_data with a
        # sim.neuromodulators.NeuromodulatorManager that owns per-modulator
        # concentrations and applies receptor effects each step. Default
        # None means legacy reward path is used unchanged.
        self.neuromodulator_manager = None

        # Brain-region framework (Session E.2, opt-in).
        # When core_config.enable_brain_region_framework is True and
        # brain_regions is non-empty, this is populated with a
        # sim.regions.RegionManager that owns per-region index slices,
        # inhibitory-cell selection, and the region wiring plan. Default
        # None means legacy single-population path runs.
        self.region_manager = None

        # Synapse tiering (Phase 3 Strategy B, 2026-05-11)
        # Opt-in via cfg.enable_synapse_tiering. Mirrors per-pathway
        # CSRs and tracks activity each simulation step. Foundation
        # for Phase 4 auto-tiering.
        self.synapse_store = None

        # Experiment & stimulus system
        self.experiment_engine = None
        self.experiment_config = None  # ExperimentConfig dataclass

        # Performance profiling - now controlled by gpu_config
        self._profile_timings = {
            "step_total": deque(maxlen=self.gpu_config.profiling_window_size),
            "connectivity": deque(maxlen=self.gpu_config.profiling_window_size),
            "dynamics": deque(maxlen=self.gpu_config.profiling_window_size),
            "gpu_sync": deque(maxlen=self.gpu_config.profiling_window_size),
            "neuron_update": deque(maxlen=self.gpu_config.profiling_window_size),
            "synapse_update": deque(maxlen=self.gpu_config.profiling_window_size),
            "plasticity_update": deque(maxlen=self.gpu_config.profiling_window_size),
            "recording": deque(maxlen=self.gpu_config.profiling_window_size),
            "gpu_memory_ops": deque(maxlen=self.gpu_config.profiling_window_size)
        }

        self.PROFILE_DIR = "simulation_profiles/" 
        self.CHECKPOINT_DIR = "simulation_checkpoints_h5/" 
        self.RECORDING_DIR = "simulation_recordings_h5/"   

        self.recording_file_handle = None 
        self.recording_filepath = None    
        self.current_frame_count_for_h5 = 0
        
        # GPU-buffered recording/playback: store frames in VRAM (controlled by gpu_config)
        self.gpu_frame_buffer = {}  # Dict of frame_idx -> dict of CuPy arrays
        self.cpu_frame_buffer = {}  # Dict of frame_idx -> dict of NumPy arrays (overflow when GPU full)
        self.recording_overflow_to_cpu = False  # Flag: True when GPU is full, storing to CPU RAM
        self.gpu_recording_max_frames = 0  # Maximum frames we can buffer
        self.gpu_playback_cache = {}  # Dict of frame_idx -> dict of CuPy arrays

        # Streaming playback prefetch buffer (for non-cached playback mode)
        self.prefetch_buffer = {}  # Dict of frame_idx -> NumPy frame data (not GPU)
        self.prefetch_lock = threading.Lock()
        self.prefetch_executor = None  # ThreadPoolExecutor for background prefetching
        self.prefetch_pending = set()  # Frame indices currently being prefetched

        # Async streaming recording writer (for large-scale simulations)
        self.streaming_write_queue = queue.Queue()  # Queue of (frame_idx, frame_data_np) to write
        self.streaming_writer_thread = None  # Background thread for async disk writes
        self.streaming_writer_stop_event = threading.Event()  # Signal to stop writer thread
        self.streaming_frames_written = 0  # Counter for frames successfully written to disk
        self.streaming_frames_queued = 0  # Counter for frames queued for writing

        # Data bus for pub/sub streaming to UI (set externally or left None for legacy queue mode)
        self.data_bus = None

        for dir_path in [self.PROFILE_DIR, self.CHECKPOINT_DIR, self.RECORDING_DIR]:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)
                    self._log_console(f"Created directory: {dir_path}", "info")
                except OSError as e:
                    self._log_console(f"Error creating directory {dir_path}: {e}", "error")
        try:
             _backend_set_device(0)

             # Configure memory pool for better performance (controlled by
             # gpu_config). On NumPy backend, mempool is None — skip pool
             # configuration (no pool concept on CPU).
             mempool = _backend_get_memory_pool()
             pinned_mempool = _backend_get_pinned_memory_pool()

             dev_props = _backend_get_device_properties(0)
             total_mem = dev_props['totalGlobalMem']
             if mempool is not None:
                 mempool.set_limit(size=int(total_mem * self.gpu_config.memory_pool_limit_fraction))

             gpu_name = dev_props.get('name', b'Unknown').decode()
             if is_gpu_backend():
                 self._log_console(
                     f"CuPy using GPU: {gpu_name} ({total_mem / 1024**3:.1f} GB), "
                     f"mempool limit: {self.gpu_config.memory_pool_limit_fraction*100:.0f}%",
                     "info"
                 )
             else:
                 self._log_console(
                     f"NumPy backend ({gpu_name}, {total_mem / 1024**3:.1f} GB RAM available)",
                     "info"
                 )
        except Exception as e:
             self._log_console(f"Error setting device: {e}", "critical")

    def _log_console(self, message, level="info"):
        """Logs a message to the console (standard output)."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{timestamp}][{level.upper()}] SIM_BRIDGE: {message}")

    def _log_to_ui(self, message, level="info", color=None):
        """Sends a log message to the UI thread via the queue for display in the status bar."""
        if color is None:
            if level == "error" or level == "critical": color = [255, 0, 0]
            elif level == "warning": color = [255, 165, 0]
            elif level == "info": color = [200, 200, 200] 
            elif level == "success": color = [0, 200, 0]
            else: color = [200, 200, 200]
        
        if self.ui_queue:
            try:
                self.ui_queue.put_nowait({
                    "type": "STATUS_UPDATE",
                    "text": message,
                    "color": color,
                    "level": level
                })
            except queue.Full:
                self._log_console("UI queue full. Could not send status message.", "warning")
        self._log_console(message, level)
    
    def _get_gpu_memory_info(self):
        """Returns current GPU memory usage statistics."""
        mem_info = _backend_get_device_mem_info()
        free_memory, total_memory = mem_info
        used_memory = total_memory - free_memory
        
        return {
            "total_gb": total_memory / 1e9,
            "used_gb": used_memory / 1e9,
            "free_gb": free_memory / 1e9,
            "usage_percent": (used_memory / total_memory) * 100
        }
    
    def _check_gpu_memory_pressure(self):
        """Checks if GPU memory is under pressure and suggests cleanup (thresholds from gpu_config)."""
        mem_stats = self._get_gpu_memory_info()
        usage_fraction = mem_stats["usage_percent"] / 100.0
        
        if usage_fraction > self.gpu_config.memory_pressure_threshold:
            self._log_to_ui(
                f"WARNING: GPU memory usage at {mem_stats['usage_percent']:.1f}% ({mem_stats['used_gb']:.1f}GB/{mem_stats['total_gb']:.1f}GB)",
                "warning"
            )
            # Note: avoid free_all_blocks() here — it causes a GPU sync stall
            # (50-200ms) during simulation. CuPy's pool reuses freed blocks naturally.
            # Only free during cleanup/shutdown (clear_simulation_state_and_gpu_memory).
            return True
        elif usage_fraction > self.gpu_config.memory_warning_threshold:
            self._log_console(f"GPU memory high: {mem_stats['usage_percent']:.1f}%")
            return False

        return False

    def _get_cached_coo(self):
        """Returns cached COO representation of connectivity matrix.

        Avoids repeated tocoo() conversions within a simulation step.
        Cache is invalidated when connectivity changes (synapse formation/elimination).
        """
        if self.cp_connections is None or self.cp_connections.nnz == 0:
            return None

        if not self._coo_cache_valid or self._cached_coo_matrix is None:
            self._cached_coo_matrix = self.cp_connections.tocoo(copy=False)
            self._coo_cache_valid = True

        return self._cached_coo_matrix

    def _invalidate_coo_cache(self):
        """Invalidates COO cache and derived caches when connectivity changes."""
        self._coo_cache_valid = False
        self._cached_coo_matrix = None
        self._cached_stp_per_type = None  # STP per-synapse params depend on connectivity
        self._cached_inhibitory_mask = None  # Inhibitory mask depends on traits

    def _init_synapse_arrays_with_capacity(self, num_synapses, cfg):
        """Initializes synapse-indexed arrays with pre-allocated capacity for growth.

        Pre-allocates extra space to avoid frequent reallocations during structural plasticity.
        Uses gpu_config.synapse_capacity_growth_factor to determine extra capacity.
        """
        growth_factor = self.gpu_config.synapse_capacity_growth_factor
        capacity = int(num_synapses * growth_factor) if num_synapses > 0 else 100

        self._synapse_count = num_synapses
        self._synapse_capacity = capacity

        # STP arrays
        if cfg.enable_short_term_plasticity and num_synapses > 0:
            self._log_console(f"Initializing STP state for {num_synapses} synapses (capacity: {capacity})...")
            self.cp_stp_x = cp.ones(capacity, dtype=cp.float32)
            self.cp_stp_u = cp.full(capacity, cfg.stp_U, dtype=cp.float32)
        else:
            self.cp_stp_x = None
            self.cp_stp_u = None

        # Eligibility traces for reward modulation. dtype=float16 when
        # cfg.fp16_synapse_state to save bandwidth on the synapse-side
        # plasticity ops. Compute auto-promotes to fp32 in cupy operators
        # (any fp16 op vs fp32 scalar promotes), so kernels still run at
        # fp32 — only storage shrinks.
        synapse_dtype = cp.float16 if cfg.fp16_synapse_state else cp.float32
        if cfg.enable_reward_modulation and num_synapses > 0:
            self._log_console(
                f"Initializing eligibility traces for {num_synapses} synapses "
                f"(capacity: {capacity}, dtype: {synapse_dtype.__name__})..."
            )
            self.cp_eligibility_trace = cp.zeros(capacity, dtype=synapse_dtype)
        else:
            self.cp_eligibility_trace = None

        # Structural pruning (2026-04-28, cheat-5 option-1).
        # Per-synapse alive mask + survival score for axon pruning. See
        # docs/plans/2026-04-28-structural-plasticity-design.md. Default OFF
        # for full backward compatibility with the flagship config.
        if getattr(cfg, "enable_structural_pruning", False) and num_synapses > 0:
            self.cp_synapse_alive = cp.ones(num_synapses, dtype=cp.bool_)
            self.cp_synapse_survival = cp.zeros(num_synapses, dtype=cp.float32)
        else:
            self.cp_synapse_alive = None
            self.cp_synapse_survival = None

        # Neuromodulator subsystem (Session E.1, opt-in).
        # When `enable_neuromodulator_subsystem` is True, allocate a
        # NeuromodulatorManager per the user's `neuromodulators` configs.
        # When False (default), legacy reward modulation path runs unchanged.
        if getattr(cfg, "enable_neuromodulator_subsystem", False) and getattr(cfg, "neuromodulators", None):
            from sim.neuromodulators import NeuromodulatorManager
            self.neuromodulator_manager = NeuromodulatorManager(
                cfg.neuromodulators, cfg.dt_ms,
            )
            self.neuromodulator_manager.initialize(cfg.num_neurons, cp)
            # E.2 Task 7: if a brain-region framework is also active,
            # auto-register region indices as neuromodulator groups so
            # ModulatorTarget(scope='group:PFC') resolves natively.
            if self.region_manager is not None:
                self.neuromodulator_manager.set_group_indices(
                    self.region_manager.region_indices_dict()
                )
            self._log_console(
                f"Initialized neuromodulator subsystem with "
                f"{len(cfg.neuromodulators)} modulators: "
                f"{self.neuromodulator_manager.modulator_names()}"
            )
        else:
            self.neuromodulator_manager = None

        # Visualization arrays
        if OPENGL_AVAILABLE and num_synapses > 0:
            self.cp_synapse_pulse_timers = cp.zeros(capacity, dtype=cp.int32)
            self.cp_synapse_pulse_progress = cp.zeros(capacity, dtype=cp.float32)
        else:
            self.cp_synapse_pulse_timers = None
            self.cp_synapse_pulse_progress = None

    def _build_synapse_conn_type_array(self, cfg):
        """Build per-synapse connection type array: 0=E->E, 1=E->I, 2=I->E, 3=I->I.

        Uses the COO representation to look up pre/post neuron traits and classify
        each synapse. Falls back to all-zeros (E->E) if trait information is unavailable.
        """
        if self.cp_connections is None or self.cp_connections.nnz == 0:
            self.cp_synapse_conn_type = None
            return

        nnz = self.cp_connections.nnz
        capacity = self._synapse_capacity if hasattr(self, '_synapse_capacity') else nnz

        # Default: all E->E (type 0)
        conn_types = cp.zeros(max(capacity, nnz), dtype=cp.int8)

        inh_indices = getattr(cfg, 'inhibitory_trait_indices', [])
        if self.cp_traits is not None and len(inh_indices) > 0:
            coo = self._get_cached_coo()
            if coo is None:
                coo = self.cp_connections.tocoo(copy=False)

            pre_traits = self.cp_traits[coo.row]
            post_traits = self.cp_traits[coo.col]

            # Build inhibitory neuron mask from trait indices
            pre_is_inh = cp.zeros(coo.row.shape, dtype=cp.bool_)
            post_is_inh = cp.zeros(coo.col.shape, dtype=cp.bool_)
            for idx in inh_indices:
                pre_is_inh |= (pre_traits == idx)
                post_is_inh |= (post_traits == idx)

            # Classify: 0=E->E, 1=E->I, 2=I->E, 3=I->I
            conn_types[:nnz] = (pre_is_inh.astype(cp.int8) * 2) + post_is_inh.astype(cp.int8)

            type_counts = [int((conn_types[:nnz] == t).sum()) for t in range(4)]
            self._log_console(f"Per-synapse STP types: E->E={type_counts[0]}, E->I={type_counts[1]}, "
                              f"I->E={type_counts[2]}, I->I={type_counts[3]}")
        else:
            self._log_console("No trait info available; all synapses default to E->E STP type.", "warning")

        self.cp_synapse_conn_type = conn_types

    def _grow_synapse_arrays_if_needed(self, new_synapse_count, cfg):
        """Grows synapse arrays if new_synapse_count exceeds current capacity.

        Returns True if reallocation occurred, False if existing capacity was sufficient.
        """
        total_needed = self._synapse_count + new_synapse_count

        if total_needed <= self._synapse_capacity:
            return False  # Existing capacity is sufficient

        # Need to grow - calculate new capacity
        growth_factor = self.gpu_config.synapse_capacity_growth_factor
        new_capacity = int(total_needed * growth_factor)

        self._log_console(f"Growing synapse arrays: {self._synapse_capacity} -> {new_capacity}")

        # Grow STP arrays
        if cfg.enable_short_term_plasticity and self.cp_stp_x is not None:
            new_stp_x = cp.ones(new_capacity, dtype=cp.float32)
            new_stp_u = cp.full(new_capacity, cfg.stp_U, dtype=cp.float32)
            new_stp_x[:self._synapse_count] = self.cp_stp_x[:self._synapse_count]
            new_stp_u[:self._synapse_count] = self.cp_stp_u[:self._synapse_count]
            self.cp_stp_x = new_stp_x
            self.cp_stp_u = new_stp_u

        # Grow eligibility traces
        if cfg.enable_reward_modulation and self.cp_eligibility_trace is not None:
            new_traces = cp.zeros(new_capacity, dtype=cp.float32)
            new_traces[:self._synapse_count] = self.cp_eligibility_trace[:self._synapse_count]
            self.cp_eligibility_trace = new_traces

        # Grow connection type array for per-type STP
        if self.cp_synapse_conn_type is not None:
            new_conn_types = cp.zeros(new_capacity, dtype=cp.int8)
            new_conn_types[:self._synapse_count] = self.cp_synapse_conn_type[:self._synapse_count]
            self.cp_synapse_conn_type = new_conn_types

        # Grow visualization arrays
        if self.cp_synapse_pulse_timers is not None:
            new_timers = cp.zeros(new_capacity, dtype=cp.int32)
            new_progress = cp.zeros(new_capacity, dtype=cp.float32)
            new_timers[:self._synapse_count] = self.cp_synapse_pulse_timers[:self._synapse_count]
            new_progress[:self._synapse_count] = self.cp_synapse_pulse_progress[:self._synapse_count]
            self.cp_synapse_pulse_timers = new_timers
            self.cp_synapse_pulse_progress = new_progress

        self._synapse_capacity = new_capacity
        return True

    def _add_synapses_to_arrays(self, new_count, cfg):
        """Adds new synapses to pre-allocated arrays at the current synapse_count position.

        Assumes _grow_synapse_arrays_if_needed was called first to ensure capacity.
        Updates _synapse_count after adding.
        """
        start_idx = self._synapse_count

        # Initialize new STP entries
        if cfg.enable_short_term_plasticity and self.cp_stp_x is not None:
            self.cp_stp_x[start_idx:start_idx + new_count] = 1.0
            self.cp_stp_u[start_idx:start_idx + new_count] = cfg.stp_U

        # Initialize new eligibility traces
        if cfg.enable_reward_modulation and self.cp_eligibility_trace is not None:
            self.cp_eligibility_trace[start_idx:start_idx + new_count] = 0.0

        # Initialize new visualization entries
        if self.cp_synapse_pulse_timers is not None:
            self.cp_synapse_pulse_timers[start_idx:start_idx + new_count] = 0
            self.cp_synapse_pulse_progress[start_idx:start_idx + new_count] = 0.0

        self._synapse_count += new_count

    def _compact_synapse_arrays(self, keep_mask):
        """Compacts synapse arrays by removing eliminated synapses.

        Called when deferred CSR compaction occurs.
        keep_mask: boolean array indicating which synapses to keep.
        """
        if self.cp_stp_x is not None:
            # Extract kept values
            kept_x = self.cp_stp_x[:self._synapse_count][keep_mask]
            kept_u = self.cp_stp_u[:self._synapse_count][keep_mask]
            new_count = kept_x.size

            # Write back to beginning of arrays
            self.cp_stp_x[:new_count] = kept_x
            self.cp_stp_u[:new_count] = kept_u

        if self.cp_eligibility_trace is not None:
            kept_traces = self.cp_eligibility_trace[:self._synapse_count][keep_mask]
            self.cp_eligibility_trace[:kept_traces.size] = kept_traces

        if self.cp_synapse_pulse_timers is not None:
            kept_timers = self.cp_synapse_pulse_timers[:self._synapse_count][keep_mask]
            kept_progress = self.cp_synapse_pulse_progress[:self._synapse_count][keep_mask]
            self.cp_synapse_pulse_timers[:kept_timers.size] = kept_timers
            self.cp_synapse_pulse_progress[:kept_progress.size] = kept_progress

        self._synapse_count = int(cp.sum(keep_mask))  # int() works on cupy 0-d + numpy scalar

    def get_profiling_stats(self):
        """Returns summary statistics for profiling timings.
        
        Returns:
            Dict with keys for each timing category, each containing:
            - mean: average time in seconds
            - std: standard deviation
            - p50: median (50th percentile)
            - p95: 95th percentile
            - p99: 99th percentile
            - count: number of samples
        """
        if not self.gpu_config.enable_profiling:
            return {"profiling_disabled": True}
        
        stats = {}
        for category, timings in self._profile_timings.items():
            if len(timings) == 0:
                stats[category] = {
                    "mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0
                }
                continue
            
            timings_array = np.array(list(timings))
            stats[category] = {
                "mean": float(np.mean(timings_array)),
                "std": float(np.std(timings_array)),
                "p50": float(np.percentile(timings_array, 50)),
                "p95": float(np.percentile(timings_array, 95)),
                "p99": float(np.percentile(timings_array, 99)),
                "count": len(timings)
            }
        
        return stats
    
    def export_profiling_report(self, filepath):
        """Exports profiling statistics to a JSON file.
        
        Args:
            filepath: Path to save the JSON report
            
        Returns:
            True if successful, False otherwise
        """
        if not self.gpu_config.enable_profiling:
            self._log_to_ui("Profiling is disabled. Enable it in GPUConfig first.", "warning")
            return False
        
        try:
            stats = self.get_profiling_stats()
            
            # Add metadata
            report = {
                "profiling_report_version": "1.0",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "config": {
                    "neuron_model": self.core_config.neuron_model_type,
                    "num_neurons": self.core_config.num_neurons,
                    "dt_ms": self.core_config.dt_ms,
                    "enable_hebbian": self.core_config.enable_hebbian_learning,
                    "enable_stp": self.core_config.enable_short_term_plasticity,
                    "enable_homeostasis": self.core_config.enable_homeostasis,
                    "profiling_window_size": self.gpu_config.profiling_window_size,
                    "profiling_detailed": self.gpu_config.profiling_detailed
                },
                "gpu_info": self._get_gpu_memory_info(),
                "statistics": stats
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            self._log_to_ui(f"Profiling report exported to {filepath}", "success")
            return True
            
        except Exception as e:
            self._log_to_ui(f"Error exporting profiling report: {e}", "error")
            return False

    def _initialize_rng(self, seed):
        """Centralized RNG initialization for reproducibility.
        
        Args:
            seed: Random seed (-1 for random initialization based on time)
            
        Returns:
            The actual seed used (for reproducibility tracking)
        """
        if seed == -1:
            # Generate random seed from current time
            seed = int(time.time() * 1000) % (2**31)
        
        # Initialize all RNG sources
        cp.random.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Store the actual seed used
        self.runtime_state.actual_seed_used = seed
        self._log_console(f"RNG initialized with seed: {seed}")
        
        return seed

    def _initialize_simulation_data(self, called_from_playback_init=False):
        """Initializes or re-initializes all CuPy arrays and simulation state variables."""
        self._log_console(f"Initializing simulation data for model: {self.core_config.neuron_model_type} (3D)... (playback_init: {called_from_playback_init})")

        if not called_from_playback_init:
            # These global_gui_state checks are for context; actual state changes are UI-driven.
            # Sim thread should not directly modify global_gui_state.
            pass # UI thread manages stopping recording/playback before commanding re-init.

        try:
            cfg = self.core_config

            # Brain-region framework (Session E.2, opt-in): allocate the
            # RegionManager BEFORE anything that depends on num_neurons.
            # If brain_regions is non-empty, RegionManager.total_neurons()
            # determines the global neuron count.
            self.region_manager = None
            if (getattr(cfg, "enable_brain_region_framework", False)
                    and getattr(cfg, "brain_regions", None)):
                from sim.regions import RegionManager
                self.region_manager = RegionManager(
                    cfg.brain_regions,
                    getattr(cfg, "region_pathways", []) or [],
                )
                # Use main seed (or 0 default) for deterministic
                # inhibitory-cell selection.
                seed_val = cfg.seed if cfg.seed >= 0 else 0
                self.region_manager.initialize(seed=seed_val)
                # Override num_neurons to match the regions.
                cfg.num_neurons = self.region_manager.total_neurons()
                self._log_console(
                    f"Brain-region framework: {len(cfg.brain_regions)} regions, "
                    f"{cfg.num_neurons} total neurons, "
                    f"{len(getattr(cfg, 'region_pathways', []) or [])} pathways."
                )

            n = self.core_config.num_neurons
            if n <= 0:
                self._log_console(f"Number of neurons ({n}) must be positive. Initialization failed.", "warning")
                self.is_initialized = False; return

            # Use centralized RNG initialization
            self._initialize_rng(cfg.seed)

            # Initialize external input current
            # HH and AdEx neurons generally need some baseline drive to spike; Izhikevich can be spontaneous.
            if cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                # HH model expects current density in µA/cm²
                # For spiking: need ~5–20 µA/cm² (converted to pA for consistency)
                # 10 µA/cm² = 10,000,000 pA (when divided by 1e-6 later = 10 µA/cm²)
                drive_scale = getattr(cfg, "hh_external_drive_scale", 1.0)
                base_min, base_max = 5e6, 20e6
                self.cp_external_input_current = cp.random.uniform(base_min * drive_scale,
                                                                    base_max * drive_scale,
                                                                    n).astype(cp.float32)
            elif cfg.neuron_model_type == NeuronModel.ADEX.name:
                # AdEx uses current in pA directly; give a modest heterogeneous DC drive
                # so networks can spike even with sparse connectivity.
                drive_scale = getattr(cfg, "adex_external_drive_scale", 1.0)
                base_min, base_max = 50.0, 250.0
                self.cp_external_input_current = cp.random.uniform(base_min * drive_scale,
                                                                    base_max * drive_scale,
                                                                    n).astype(cp.float32)
            else:
                # Izhikevich and other models default to zero external drive unless overridden
                self.cp_external_input_current = cp.zeros(n, dtype=cp.float32)
            self.cp_firing_states = cp.zeros(n, dtype=bool)
            self.cp_prev_firing_states = cp.zeros(n, dtype=bool)
            # Start with a generic random trait assignment
            self.cp_traits = cp.random.randint(0, max(1, cfg.num_traits), (n,), dtype=cp.int32) if n > 0 else cp.array([], dtype=cp.int32)

            # If a structured neural profile is selected, override trait distribution on host
            profile_name = getattr(cfg, "neural_profile_name", "GENERIC_UNSTRUCTURED")
            profile_def = NEURAL_STRUCTURE_PROFILES.get(profile_name)

            # If running HH model and the profile defines a default HH preset, use it
            # unless the user has explicitly selected a non-default HH type.
            if cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name and profile_def:
                profile_hh_type = profile_def.get("default_hh_neuron_type")
                if profile_hh_type:
                    try:
                        # Only auto-override when HH type is still the global default preset
                        if cfg.default_neuron_type_hh == NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name:
                            # Validate that the profile's HH type exists
                            _ = NeuronType[profile_hh_type]
                            cfg.default_neuron_type_hh = profile_hh_type
                            self._log_console(f"Profile {profile_name}: using HH preset {profile_hh_type} as default.")
                    except Exception as e:
                        self._log_console(f"Warning: profile {profile_name} specifies invalid default_hh_neuron_type={profile_hh_type}: {e}", "warning")

            if profile_def and profile_def.get("trait_definitions") and n > 0:
                trait_defs = profile_def["trait_definitions"]
                # Extract and normalize fractions
                fractions = [max(0.0, float(td.get("fraction", 0.0))) for td in trait_defs]
                total_frac = sum(fractions)
                if total_frac <= 0.0:
                    fractions = [1.0 / len(trait_defs)] * len(trait_defs)
                else:
                    fractions = [f / total_frac for f in fractions]
                # Convert fractions to integer counts, then adjust to sum exactly to n
                counts = [int(round(f * n)) for f in fractions]
                diff = n - sum(counts)
                idx = 0
                while diff != 0 and len(counts) > 0:
                    j = idx % len(counts)
                    if diff > 0:
                        counts[j] += 1; diff -= 1
                    else:
                        if counts[j] > 0:
                            counts[j] -= 1; diff += 1
                    idx += 1
                np_traits = np.empty(n, dtype=np.int32)
                start = 0
                for td, c in zip(trait_defs, counts):
                    end = start + max(0, c)
                    if end > start:
                        np_traits[start:end] = int(td["trait_index"])
                    start = end
                # If rounding caused fewer than n assignments, fill the remainder with the first trait index
                if start < n and trait_defs:
                    np_traits[start:n] = int(trait_defs[0]["trait_index"])
                if n > 1:
                    np.random.shuffle(np_traits)
                self.cp_traits = cp.asarray(np_traits, dtype=cp.int32)
                # Ensure num_traits is at least large enough to index all configured traits
                max_trait_idx = max(td["trait_index"] for td in trait_defs)
                if cfg.num_traits <= max_trait_idx:
                    cfg.num_traits = max_trait_idx + 1

            self.cp_neuron_type_ids = cp.zeros(n, dtype=cp.int32) if n > 0 else cp.array([], dtype=cp.int32)  # Will be populated per neuron
            self.cp_conductance_g_e = cp.zeros(n, dtype=cp.float32)
            self.cp_conductance_g_i = cp.zeros(n, dtype=cp.float32)

            # Per-neuron inhibitory reversal potential. Defaults to global config
            # value; regions can override via BrainRegion.syn_reversal_potential_i_override
            # (e.g., striatal MSNs use −60 mV per PBR-160 ch 6; SNc DA uses
            # −55 mV per ch 11). The fused conductance kernel broadcasts this
            # array element-wise against per-neuron membrane potential.
            if n > 0:
                self.cp_syn_reversal_potential_i_per_neuron = cp.full(
                    n, cfg.syn_reversal_potential_i, dtype=cp.float32
                )
                if self.region_manager is not None:
                    for region in self.region_manager.regions():
                        override = getattr(region, "syn_reversal_potential_i_override", None)
                        if override is None:
                            continue
                        idx_list = self.region_manager.indices(region.name)
                        if not idx_list:
                            continue
                        idx_arr = cp.asarray(idx_list, dtype=cp.int32)
                        self.cp_syn_reversal_potential_i_per_neuron[idx_arr] = float(override)
            else:
                self.cp_syn_reversal_potential_i_per_neuron = cp.array([], dtype=cp.float32)

            # Cluster E v1 (2026-04-29): allocate cp_neuron_coords if any region
            # declares coordinate_dim > 0. NaN-padded for non-coordinate regions
            # so consumers can detect "no coords" via cp.isnan check.
            self.cp_neuron_coords = None
            if n > 0 and self.region_manager is not None:
                k_dim = self.region_manager.max_coordinate_dim()
                if k_dim > 0:
                    coords = np.full((n, k_dim), np.nan, dtype=np.float32)
                    for region in self.region_manager.regions():
                        if not region.coordinate_dim or region.coordinate_dim <= 0:
                            continue
                        region_coords = self.region_manager.coordinates(region.name)
                        if not region_coords:
                            continue
                        idx_list = self.region_manager.indices(region.name)
                        for local_i, global_i in enumerate(idx_list):
                            pt = region_coords[local_i]
                            for ax in range(min(len(pt), k_dim)):
                                coords[global_i, ax] = float(pt[ax])
                    self.cp_neuron_coords = cp.asarray(coords, dtype=cp.float32)

            # NMDA conductance (dual-exponential: g_nmda_slow - g_nmda_rise)
            self.cp_conductance_g_nmda = cp.zeros(n, dtype=cp.float32)
            self.cp_conductance_g_nmda_rise = cp.zeros(n, dtype=cp.float32)
            # Cluster G v2 (2026-05-01): per-neuron NMDA mask. 1.0 for
            # neurons in regions with BrainRegion.enable_nmda=True, 0.0
            # for all others. When the region_manager doesn't tag any
            # region with enable_nmda=True, mask is left as None and NMDA
            # applies globally per cfg.enable_nmda (v1 backward compat).
            if self.region_manager is not None:
                nmda_regions = [r for r in self.region_manager.regions()
                                if getattr(r, "enable_nmda", False)]
                if nmda_regions:
                    mask = cp.zeros(n, dtype=cp.float32)
                    for r in nmda_regions:
                        idx = list(self.region_manager.indices(r.name))
                        if idx:
                            mask[cp.asarray(idx, dtype=cp.int64)] = 1.0
                    self.cp_nmda_neuron_mask = mask
                    self._log_console(
                        f"NMDA per-region mask: {len(nmda_regions)} regions enabled "
                        f"({sum(int(r.n_neurons) for r in nmda_regions)} neurons)",
                    )
            self.cp_refractory_timers = cp.zeros(n, dtype=cp.int32)
            self.cp_neuron_activity_ema = cp.zeros(n, dtype=cp.float32) 
            self.cp_viz_activity_timers = cp.zeros(n, dtype=cp.int32) 

            self.cp_synapse_pulse_timers = cp.array([], dtype=cp.int32)
            self.cp_synapse_pulse_progress = cp.array([], dtype=cp.float32)

            self.runtime_state.neuron_types_list_for_viz = [""] * n

            if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                self._log_console(f"Initializing Izhikevich model specifics for {n} neurons...")
                self.cp_izh_C = cp.zeros(n, dtype=cp.float32); self.cp_izh_k = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_vr = cp.zeros(n, dtype=cp.float32); self.cp_izh_vt = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_vpeak = cp.zeros(n, dtype=cp.float32); self.cp_izh_a = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_b = cp.zeros(n, dtype=cp.float32); self.cp_izh_c_reset = cp.zeros(n, dtype=cp.float32)
                self.cp_izh_d_increment = cp.zeros(n, dtype=cp.float32)
                
                self.cp_membrane_potential_v = cp.zeros(n, dtype=cp.float32)
                self.cp_recovery_variable_u = cp.zeros(n, dtype=cp.float32)

                thresh_base = (cfg.homeostasis_threshold_min + cfg.homeostasis_threshold_max) / 2.0
                thresh_var = (cfg.homeostasis_threshold_max - cfg.homeostasis_threshold_min) / 2.0
                if thresh_var < 0: thresh_var = 1.0 
                self.cp_neuron_firing_thresholds = cp.random.uniform(
                    thresh_base - thresh_var, thresh_base + thresh_var, n
                ).astype(cp.float32) if n > 0 else cp.array([], dtype=cp.float32)
                if n > 0:
                    cp.clip(self.cp_neuron_firing_thresholds,
                            cfg.homeostasis_threshold_min, cfg.homeostasis_threshold_max,
                            out=self.cp_neuron_firing_thresholds)

                np_traits_host = _backend_to_host(self.cp_traits)
                defined_izh2007_types = [
                    ntype for ntype in NeuronType
                    if "IZH2007" in ntype.name and ntype in DefaultIzhikevichParamsManager.PARAMS
                ]
                num_defined_izh_variants = len(defined_izh2007_types)

                # Vectorized initialization: build arrays on CPU, transfer once to GPU
                # Pre-fetch all parameter sets
                param_sets = []
                type_names = []
                for ntype in defined_izh2007_types:
                    params = DefaultIzhikevichParamsManager.get_params(ntype, use_2007_formulation=True)
                    param_sets.append(params)
                    type_names.append(f"Izh2007_{ntype.name.replace('IZH2007_', '')}")

                # Build CPU arrays
                np_C = np.zeros(n, dtype=np.float32)
                np_k = np.zeros(n, dtype=np.float32)
                np_vr = np.zeros(n, dtype=np.float32)
                np_vt = np.zeros(n, dtype=np.float32)
                np_vpeak = np.zeros(n, dtype=np.float32)
                np_a = np.zeros(n, dtype=np.float32)
                np_b = np.zeros(n, dtype=np.float32)
                np_c_reset = np.zeros(n, dtype=np.float32)
                np_d_increment = np.zeros(n, dtype=np.float32)
                np_type_ids = np.zeros(n, dtype=np.int32)

                default_type_enum = NeuronType[cfg.default_neuron_type_izh]
                default_params = DefaultIzhikevichParamsManager.get_params(default_type_enum, use_2007_formulation=True)
                default_type_id = NEURON_TYPE_MAPPER.get_id(default_type_enum)

                # Trait-based multi-type assignment is opt-in: only happens if there
                # are >1 IZH2007 variants AND the config requests >1 traits.
                # This makes single-type configs (cfg.num_traits=1) use
                # cfg.default_neuron_type_izh for ALL neurons — fixes the bug
                # where adding new IZH2007 presets silently changed the modulo
                # math and reassigned existing populations.
                use_trait_based = (num_defined_izh_variants > 1
                                    and cfg.num_traits > 1)
                if use_trait_based:
                    # Vectorized type selection based on traits
                    type_indices = np_traits_host % num_defined_izh_variants
                    for type_idx, params in enumerate(param_sets):
                        mask = (type_indices == type_idx)
                        np_C[mask] = params["C"]
                        np_k[mask] = params["k"]
                        np_vr[mask] = params["vr"]
                        np_vt[mask] = params["vt"]
                        np_vpeak[mask] = params["vpeak"]
                        np_a[mask] = params["a"]
                        np_b[mask] = params["b"]
                        np_c_reset[mask] = params["c_reset"]
                        np_d_increment[mask] = params["d_increment"]
                        np_type_ids[mask] = NEURON_TYPE_MAPPER.get_id(defined_izh2007_types[type_idx])
                    # Build viz labels
                    self.runtime_state.neuron_types_list_for_viz = [type_names[type_indices[i]] for i in range(n)]
                else:
                    # All neurons use default type
                    np_C[:] = default_params["C"]
                    np_k[:] = default_params["k"]
                    np_vr[:] = default_params["vr"]
                    np_vt[:] = default_params["vt"]
                    np_vpeak[:] = default_params["vpeak"]
                    np_a[:] = default_params["a"]
                    np_b[:] = default_params["b"]
                    np_c_reset[:] = default_params["c_reset"]
                    np_d_increment[:] = default_params["d_increment"]
                    np_type_ids[:] = default_type_id
                    self.runtime_state.neuron_types_list_for_viz = [f"Izh2007_{default_type_enum.name.replace('IZH2007_', '')}"] * n

                # Single GPU transfer for all parameter arrays
                self.cp_izh_C = cp.asarray(np_C)
                self.cp_izh_k = cp.asarray(np_k)
                self.cp_izh_vr = cp.asarray(np_vr)
                self.cp_izh_vt = cp.asarray(np_vt)
                self.cp_izh_vpeak = cp.asarray(np_vpeak)
                self.cp_izh_a = cp.asarray(np_a)
                self.cp_izh_b = cp.asarray(np_b)
                self.cp_izh_c_reset = cp.asarray(np_c_reset)
                self.cp_izh_d_increment = cp.asarray(np_d_increment)
                self.cp_neuron_type_ids = cp.asarray(np_type_ids)

                # Initialize membrane potential and recovery variable
                self.cp_membrane_potential_v = cp.asarray(np_vr)
                self.cp_recovery_variable_u = self.cp_izh_b * (self.cp_membrane_potential_v - self.cp_izh_vr)

            elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                self._log_console(f"Initializing Hodgkin-Huxley model specifics for {n} neurons...")
                self.cp_hh_C_m = cp.zeros(n, dtype=cp.float32); self.cp_hh_g_Na_max = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_g_K_max = cp.zeros(n, dtype=cp.float32); self.cp_hh_g_L = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_E_Na = cp.zeros(n, dtype=cp.float32); self.cp_hh_E_K = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_E_L = cp.zeros(n, dtype=cp.float32); self.cp_hh_v_peak = cp.zeros(n, dtype=cp.float32)
                
                # Initialize membrane and gating variables
                self.cp_membrane_potential_v = cp.zeros(n, dtype=cp.float32)
                self.cp_gating_variable_m = cp.zeros(n, dtype=cp.float32)
                self.cp_gating_variable_h = cp.zeros(n, dtype=cp.float32)
                self.cp_gating_variable_n = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_m_current_activation = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_CaT_m = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_CaT_h = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_h_current_q = cp.zeros(n, dtype=cp.float32)
                self.cp_hh_NaP_activation = cp.zeros(n, dtype=cp.float32)
                self.cp_neuron_firing_thresholds = None 

                # Use default HH neuron type to populate extended current config defaults (if defined)
                try:
                    default_hh_type_enum = NeuronType[cfg.default_neuron_type_hh]
                    hh_base_params_for_ext = DefaultHodgkinHuxleyParams.get_params(default_hh_type_enum)
                    cfg.hh_g_M_max = hh_base_params_for_ext.get("g_M_max", cfg.hh_g_M_max)
                    cfg.hh_g_CaT_max = hh_base_params_for_ext.get("g_CaT_max", cfg.hh_g_CaT_max)
                    cfg.hh_E_CaT = hh_base_params_for_ext.get("E_CaT", cfg.hh_E_CaT)
                    cfg.hh_g_h_max = hh_base_params_for_ext.get("g_h_max", cfg.hh_g_h_max)
                    cfg.hh_E_h = hh_base_params_for_ext.get("E_h", cfg.hh_E_h)
                    cfg.hh_g_NaP_max = hh_base_params_for_ext.get("g_NaP_max", cfg.hh_g_NaP_max)
                except Exception as e:
                    self._log_console(f"Warning: Failed to derive extended HH defaults from {cfg.default_neuron_type_hh}: {e}", "warning")

                # Vectorized HH initialization: all neurons use same type, use cp.full() for broadcast
                default_hh_type_enum = NeuronType[cfg.default_neuron_type_hh]
                params = DefaultHodgkinHuxleyParams.get_params(default_hh_type_enum)
                type_id = NEURON_TYPE_MAPPER.get_id(default_hh_type_enum)

                # Single GPU transfer using cp.full() for uniform values
                self.cp_neuron_type_ids = cp.full(n, type_id, dtype=cp.int32)
                self.cp_hh_C_m = cp.full(n, params["C_m"], dtype=cp.float32)
                self.cp_hh_g_Na_max = cp.full(n, params["g_Na_max"], dtype=cp.float32)
                self.cp_hh_g_K_max = cp.full(n, params["g_K_max"], dtype=cp.float32)
                self.cp_hh_g_L = cp.full(n, params["g_L"], dtype=cp.float32)
                self.cp_hh_E_Na = cp.full(n, params["E_Na"], dtype=cp.float32)
                self.cp_hh_E_K = cp.full(n, params["E_K"], dtype=cp.float32)
                self.cp_hh_E_L = cp.full(n, params["E_L"], dtype=cp.float32)
                self.cp_hh_v_peak = cp.full(n, params["v_peak_hh"], dtype=cp.float32)
                self.cp_membrane_potential_v = cp.full(n, params["v_rest_hh"], dtype=cp.float32)
                self.cp_gating_variable_m = cp.full(n, params["m_init"], dtype=cp.float32)
                self.cp_gating_variable_h = cp.full(n, params["h_init"], dtype=cp.float32)
                self.cp_gating_variable_n = cp.full(n, params["n_init"], dtype=cp.float32)

                # Vectorized viz label assignment
                viz_label = f"HH_{default_hh_type_enum.name.replace('HH_', '')}"
                self.runtime_state.neuron_types_list_for_viz = [viz_label] * n

            elif cfg.neuron_model_type == NeuronModel.ADEX.name:
                self._log_console(f"Initializing AdEx model specifics for {n} neurons...")
                # Overlay AdEx preset params onto cfg.adex_* fields if a
                # preset is configured. This lets users select e.g.
                # ADEX_FS_CORTICAL_INTERNEURON without manually setting
                # all 10 parameters.
                preset_name = getattr(cfg, "default_neuron_type_adex", None)
                if preset_name:
                    try:
                        preset_enum = NeuronType[preset_name]
                        from sim.enums import DefaultAdExParamsManager
                        preset_params = DefaultAdExParamsManager.get_params(preset_enum)
                        cfg.adex_C = float(preset_params["C"])
                        cfg.adex_g_L = float(preset_params["g_L"])
                        cfg.adex_E_L = float(preset_params["E_L"])
                        cfg.adex_V_T = float(preset_params["V_T"])
                        cfg.adex_Delta_T = float(preset_params["Delta_T"])
                        cfg.adex_a = float(preset_params["a"])
                        cfg.adex_tau_w = float(preset_params["tau_w"])
                        cfg.adex_b = float(preset_params["b"])
                        cfg.adex_V_r = float(preset_params["V_r"])
                        cfg.adex_V_peak = float(preset_params["V_peak"])
                        self._log_console(
                            f"AdEx preset '{preset_name}' loaded: "
                            f"C={cfg.adex_C} g_L={cfg.adex_g_L} a={cfg.adex_a} "
                            f"tau_w={cfg.adex_tau_w} b={cfg.adex_b}",
                        )
                    except (KeyError, AttributeError) as e:
                        self._log_console(
                            f"Failed to load AdEx preset '{preset_name}': {e}. "
                            f"Using default cfg.adex_* fields.", "warning",
                        )
                self.cp_membrane_potential_v = cp.full(n, cfg.adex_E_L, dtype=cp.float32)
                self.cp_adex_w = cp.zeros(n, dtype=cp.float32)
                self.cp_neuron_firing_thresholds = None  # AdEx uses adex_V_peak from config
                # Vectorized viz label assignment
                viz_label = preset_name.replace("ADEX_", "AdEx_") if preset_name else "AdEx_RS"
                self.runtime_state.neuron_types_list_for_viz = [viz_label] * n
            
            # Per-region neuron type override (Phase B). After all neurons
            # are initialized with the default type, walk each region and
            # override the params for neurons in that region's slice using
            # the region's izh/hh/adex_neuron_type if specified. This lets
            # e.g. str_D1_X regions use IZH2007_STRIATAL_MSN_D1 while
            # motor regions use IZH2007_RS_CORTICAL_PYRAMIDAL (a modeling
            # shortcut — biologically motor neurons are α-motoneurons; we
            # use cortical pyramidals as a stand-in until spinal CPGs land).
            if self.region_manager is not None:
                self._apply_per_region_neuron_types(cfg, n)

            # B2: Apply parameter heterogeneity if enabled
            if cfg.enable_parameter_heterogeneity and n > 0:
                self._apply_parameter_heterogeneity(cfg, n)
            
            # B4: Initialize OU process state if enabled
            if cfg.enable_ou_process and n > 0:
                self._initialize_ou_process_state(cfg, n)
            else:
                self.cp_ou_current = None
                self.ou_decay_factor = None
                self.ou_noise_std = None
            
            self._log_console(f"Generating 3D neuron positions for {n} neurons...")
            if n > 0:
                np_positions_3d = np.random.uniform(
                    low=[self.viz_config.volume_min_x, self.viz_config.volume_min_y, self.viz_config.volume_min_z],
                    high=[self.viz_config.volume_max_x, self.viz_config.volume_max_y, self.viz_config.volume_max_z],
                    size=(n,3)).astype(np.float32)
                self.cp_neuron_positions_3d = cp.asarray(np_positions_3d)
                self.runtime_state.neuron_positions_x = np_positions_3d[:,0].tolist()
                self.runtime_state.neuron_positions_y = np_positions_3d[:,1].tolist()
            else: 
                self.cp_neuron_positions_3d = cp.array([], dtype=cp.float32).reshape(0,3)
                self.runtime_state.neuron_positions_x = []; self.runtime_state.neuron_positions_y = []

            if not called_from_playback_init:
                # Brain-region framework path: build wiring from RegionManager
                # plan and inject via existing inject_explicit_wiring API. Skip
                # the legacy motif/WS/spatial generators entirely.
                if self.region_manager is not None:
                    self._log_console("Generating connections (brain-region framework)...")
                    seed_val = cfg.seed if cfg.seed >= 0 else 0
                    plan = self.region_manager.build_wiring_plan(seed=seed_val)
                    # inject_explicit_wiring wires + sets self.cp_connections,
                    # the plastic mask, and updates _synapse_count.
                    inh_indices_concat = []
                    for region in self.region_manager.regions():
                        inh_indices_concat.extend(self.region_manager.inhibitory_indices(region.name))
                    self.inject_explicit_wiring(
                        plan,
                        output_inhibitory_indices=inh_indices_concat or None,
                    )
                else:
                    self._log_console("Generating connections (3D)...")
                    profile_name_for_conn = getattr(cfg, "neural_profile_name", "GENERIC_UNSTRUCTURED")
                    profile_def_for_conn = NEURAL_STRUCTURE_PROFILES.get(profile_name_for_conn)
                    motif_name = profile_def_for_conn.get("connectivity_motif") if profile_def_for_conn else None

                    if motif_name:
                        self.cp_connections = self._generate_motif_connections_3d(n, self.cp_neuron_positions_3d, self.cp_traits, cfg, motif_name)
                    elif cfg.enable_watts_strogatz:
                        self.cp_connections = self._generate_watts_strogatz_connections_3d(n, cfg.connectivity_k, cfg.connectivity_p_rewire, cfg)
                    elif cfg.connections_per_neuron == 0:
                        # Explicit "no connections" signal — caller plans to use
                        # inject_explicit_wiring afterwards (G9 runners since
                        # Session B). Skip the legacy spatial generator (which
                        # has a known bug at large N when called with cpn=0)
                        # and start with an empty CSR.
                        self.cp_connections = csp.csr_matrix((n, n), dtype=cp.float32)
                    else:
                        self.cp_connections = self._generate_spatial_connections_3d(n, cfg.connections_per_neuron, self.cp_neuron_positions_3d, self.cp_traits, cfg)

                # Defensive fallback: if no synapses were generated, fall back
                # to spatial generator. SKIP this fallback when caller explicitly
                # set connections_per_neuron=0 (they plan to inject wiring).
                if cfg.connections_per_neuron != 0 and (
                    self.cp_connections is None
                    or (hasattr(self.cp_connections, 'nnz') and self.cp_connections.nnz == 0 and n > 1)
                ):
                    self._log_console(
                        f"No synapses generated for profile '{profile_name_for_conn}' (motif={motif_name}). Falling back to spatial generator.",
                        "warning",
                    )
                    self.cp_connections = self._generate_spatial_connections_3d(
                        n,
                        cfg.connections_per_neuron,
                        self.cp_neuron_positions_3d,
                        self.cp_traits,
                        cfg,
                    )

                if self.cp_connections is None:
                    self._log_console("Connection generation resulted in None. Initializing as empty matrix.", "warning")
                    self.cp_connections = csp.csr_matrix((n,n), dtype=cp.float32)
            elif self.cp_connections is None: 
                 self._log_console("Connections are None during playback init before _apply_recorded_arrays. Initializing empty.", "warning")
                 self.cp_connections = csp.csr_matrix((n,n), dtype=cp.float32)

            num_synapses = self.cp_connections.nnz if self.cp_connections is not None else 0
            if num_synapses > 0:
                self.cp_synapse_pulse_timers = cp.zeros(num_synapses, dtype=cp.int32)
                self.cp_synapse_pulse_progress = cp.zeros(num_synapses, dtype=cp.float32)
            else:
                self.cp_synapse_pulse_timers = cp.array([], dtype=cp.int32)
                self.cp_synapse_pulse_progress = cp.array([], dtype=cp.float32)

            # If a structured neural profile is configured, populate inhibitory_trait_indices
            profile_name = getattr(cfg, "neural_profile_name", "GENERIC_UNSTRUCTURED")
            profile_def = NEURAL_STRUCTURE_PROFILES.get(profile_name)
            if profile_def and profile_def.get("trait_definitions"):
                inhibitory_indices = [td["trait_index"] for td in profile_def["trait_definitions"] if td.get("role", "").lower().startswith("inhib")]
                if inhibitory_indices:
                    cfg.inhibitory_trait_indices = inhibitory_indices

            # Initialize synapse-indexed arrays with pre-allocated capacity for structural plasticity
            self._init_synapse_arrays_with_capacity(num_synapses, cfg)

            # Build per-synapse connection type array for per-type STP
            # Types: 0=E->E, 1=E->I, 2=I->E, 3=I->I
            self.cp_synapse_conn_type = None
            if cfg.enable_per_type_stp and cfg.enable_short_term_plasticity and num_synapses > 0:
                self._build_synapse_conn_type_array(cfg)

            # C2: Initialize STDP state arrays
            if cfg.enable_stdp and n > 0:
                self._log_console(f"Initializing STDP state for {n} neurons...")
                # Track last spike time for each neuron (ms, initialized to large negative value)
                self.cp_last_spike_time = cp.full(n, -1000.0, dtype=cp.float32)
            else:
                self.cp_last_spike_time = None
            
            # C3: Initialize structural plasticity state
            if cfg.enable_structural_plasticity:
                self._log_console("Initializing structural plasticity state...")
                self.cp_struct_plast_step_counter = 0  # Track steps for update interval
            else:
                self.cp_struct_plast_step_counter = None

            # Pre-compute step-invariant constants (avoids redundant exp/pow per step)
            self._cached_decay_e = float(cp.exp(-cfg.dt_ms / cfg.syn_tau_g_e)) if cfg.syn_tau_g_e > 0 else 0.0
            self._cached_decay_i = float(cp.exp(-cfg.dt_ms / cfg.syn_tau_g_i)) if cfg.syn_tau_g_i > 0 else 0.0
            self._cached_decay_nmda = float(cp.exp(-cfg.dt_ms / cfg.nmda_tau_decay)) if cfg.nmda_tau_decay > 0 else 0.0
            self._cached_decay_nmda_rise = float(cp.exp(-cfg.dt_ms / cfg.nmda_tau_rise)) if cfg.nmda_tau_rise > 0 else 0.0
            _BASE_HH_TEMP = 6.3
            self._cached_hh_phi = cfg.hh_q10_factor ** ((cfg.hh_temperature_celsius - _BASE_HH_TEMP) / 10.0)
            # Per-gate phi values (Session "fix-bugs" — see HH temperature bug findings)
            _temp_delta_div_10 = (cfg.hh_temperature_celsius - _BASE_HH_TEMP) / 10.0
            self._cached_hh_phi_m = cfg.hh_q10_m ** _temp_delta_div_10
            self._cached_hh_phi_h = cfg.hh_q10_h ** _temp_delta_div_10
            self._cached_hh_phi_n = cfg.hh_q10_n ** _temp_delta_div_10

            self.is_initialized = True
            conn_count = self.cp_connections.nnz if self.cp_connections is not None else 0

            # Log GPU memory usage after initialization
            mem_stats = self._get_gpu_memory_info()
            self._log_console(
                f"Simulation data initialized for {n} neurons (3D). Synapses: {conn_count}. "
                f"GPU memory: {mem_stats['used_gb']:.1f}GB/{mem_stats['total_gb']:.1f}GB ({mem_stats['usage_percent']:.1f}%)"
            )
            self._check_gpu_memory_pressure()

            # ── Tiering Phase 3 Strategy B: synapse store mirror ──
            # Opt-in via cfg.enable_synapse_tiering. Requires brain
            # region framework so we can identify per-pathway slices.
            # The store mirrors the monolithic cp_connections as
            # per-pathway CSRs; inference still uses the monolithic
            # path. Activity tracked each step in _run_one_simulation_step.
            if (cfg.enable_synapse_tiering
                    and self.region_manager is not None
                    and self.cp_connections is not None):
                self._initialize_synapse_store(cfg)
        except Exception as e:
            self._log_console(f"Error during simulation data initialization (3D): {e}","critical")
            import traceback; traceback.print_exc()
            self.is_initialized = False
            if is_gpu_backend() and 'cupy' in sys.modules:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

    def _apply_per_region_neuron_types(self, cfg, n):
        """Override per-neuron parameters based on region.izh_neuron_type /
        region.hh_neuron_type / region.adex_neuron_type fields.

        Phase B addition (2026-04-25): the brain-region framework
        previously assigned all neurons the same type (cfg.default_neuron_type_*).
        For BG action-selection circuits, each region needs its own
        neuron type (str_D1_X uses MSN_D1, GPe uses GPE_PACEMAKER, etc.).

        This method walks each region in cfg.brain_regions, looks up its
        index range from region_manager, and overrides cp_izh_* / cp_hh_*
        arrays for those indices using the region's per-type preset.
        """
        import cupy as cp_local  # local alias avoids confusion with self attrs
        for region in cfg.brain_regions:
            indices = self.region_manager.indices(region.name)
            if not indices:
                continue
            idx_arr = cp.asarray(list(indices), dtype=cp.int64)

            if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name and region.izh_neuron_type:
                try:
                    type_enum = NeuronType[region.izh_neuron_type]
                    params = DefaultIzhikevichParamsManager.get_params(
                        type_enum, use_2007_formulation=True,
                    )
                    self.cp_izh_C[idx_arr] = cp.float32(params["C"])
                    self.cp_izh_k[idx_arr] = cp.float32(params["k"])
                    self.cp_izh_vr[idx_arr] = cp.float32(params["vr"])
                    self.cp_izh_vt[idx_arr] = cp.float32(params["vt"])
                    self.cp_izh_vpeak[idx_arr] = cp.float32(params["vpeak"])
                    self.cp_izh_a[idx_arr] = cp.float32(params["a"])
                    self.cp_izh_b[idx_arr] = cp.float32(params["b"])
                    self.cp_izh_c_reset[idx_arr] = cp.float32(params["c_reset"])
                    self.cp_izh_d_increment[idx_arr] = cp.float32(params["d_increment"])
                    # Reset Vm + recovery to the new vr / b
                    self.cp_membrane_potential_v[idx_arr] = cp.float32(params["vr"])
                    self.cp_recovery_variable_u[idx_arr] = (
                        cp.float32(params["b"]) *
                        (cp.float32(params["vr"]) - cp.float32(params["vr"]))  # 0 at rest
                    )
                    self._log_console(
                        f"Region '{region.name}' ({len(indices)} neurons): "
                        f"using Izh type {region.izh_neuron_type}"
                    )
                except (KeyError, AttributeError) as e:
                    self._log_console(
                        f"Region '{region.name}' Izh type override failed: {e}", "warning",
                    )
            # HH per-region override is more complex (uses scalar cfg fields,
            # not per-neuron arrays for many params) — defer until Phase B
            # actually needs HH-mode regions.
            # AdEx per-region override likewise deferred.

    def _apply_parameter_heterogeneity(self, cfg, n):
        """Applies parameter heterogeneity by drawing per-neuron values from distributions.
        
        Uses scientifically-grounded distributions based on:
        - Marder & Goaillard (2006) Nature Reviews Neuroscience
        - Tripathy et al. (2013) PNAS
        - Golowasch et al. (2002) Neural Computation
        
        Args:
            cfg: CoreSimConfig with heterogeneity_distributions dict
            n: Number of neurons
        """
        if not cfg.heterogeneity_distributions:
            # Use scientifically-validated defaults if no custom distributions specified
            cfg.heterogeneity_distributions = self._get_default_heterogeneity_distributions(cfg)
        
        self._log_console("Applying parameter heterogeneity to neuron parameters...")
        
        # Set separate RNG state for heterogeneity (deterministic if seed provided)
        het_seed = cfg.heterogeneity_seed if cfg.heterogeneity_seed >= 0 else cfg.seed
        if het_seed >= 0:
            rng_state = _backend_get_random_state()
            cp.random.seed(het_seed)
        
        # Map parameter names to CuPy arrays
        param_map = {
            # Izhikevich parameters
            "izh_C_val": getattr(self, 'cp_izh_C', None),
            "izh_a_val": getattr(self, 'cp_izh_a', None),
            "izh_b_val": getattr(self, 'cp_izh_b', None),
            "izh_d_val": getattr(self, 'cp_izh_d_increment', None),
            # Hodgkin-Huxley parameters
            "hh_C_m": getattr(self, 'cp_hh_C_m', None),
            "hh_g_Na_max": getattr(self, 'cp_hh_g_Na_max', None),
            "hh_g_K_max": getattr(self, 'cp_hh_g_K_max', None),
            "hh_g_L": getattr(self, 'cp_hh_g_L', None),
        }
        
        applied_count = 0
        for param_name, dist_spec in cfg.heterogeneity_distributions.items():
            target_array = param_map.get(param_name)
            if target_array is None or target_array.size != n:
                continue
            
            dist_type = dist_spec.get("type")
            if dist_type == "lognormal":
                # CuPy lognormal takes mean and sigma of underlying normal distribution
                samples = cp.random.lognormal(
                    mean=dist_spec["mean_log"],
                    sigma=dist_spec["sigma_log"],
                    size=n
                ).astype(cp.float32)
            elif dist_type == "gaussian":
                samples = cp.random.normal(
                    loc=dist_spec["mean"],
                    scale=dist_spec["std"],
                    size=n
                ).astype(cp.float32)
                # Clip to prevent non-physical values (~0.1x to 3x magnitude from mean)
                mean_val = dist_spec["mean"]
                if mean_val > 0:
                    samples = cp.clip(samples, mean_val * 0.1, mean_val * 3.0)
                elif mean_val < 0:
                    # For negative parameters (e.g., izh_b = -2.0 nS): clip symmetrically around mean
                    samples = cp.clip(samples, mean_val * 3.0, mean_val * 0.1)
                # else mean == 0: no clipping (allow both positive and negative)
            else:
                self._log_console(f"Unknown distribution type '{dist_type}' for {param_name}", "warning")
                continue
            
            # Apply heterogeneity
            target_array[:] = samples
            applied_count += 1
        
        # Restore RNG state
        if het_seed >= 0:
            _backend_set_random_state(rng_state)
        
        self._log_console(f"Applied heterogeneity to {applied_count} parameters.")
    
    def _get_default_heterogeneity_distributions(self, cfg):
        """Returns scientifically-grounded default heterogeneity distributions.
        
        Based on experimental data showing:
        - CV = 0.2-0.4 for most neural parameters (Tripathy et al. 2013)
        - Log-normal for conductances (Golowasch et al. 2002)
        - Gaussian for capacitance (10-15% variance)
        """
        defaults = {}
        
        if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
            # Izhikevich parameters (CV ~ 0.3)
            defaults["izh_a_val"] = {"type": "lognormal", "mean_log": cp.log(cfg.izh_a_val).item(), "sigma_log": 0.3} if cfg.izh_a_val > 0 else {"type": "gaussian", "mean": cfg.izh_a_val, "std": abs(cfg.izh_a_val) * 0.3}
            # b can be negative (e.g., -2.0 nS for RS neurons) — use Gaussian, not log-normal
            defaults["izh_b_val"] = {"type": "gaussian", "mean": cfg.izh_b_val, "std": abs(cfg.izh_b_val) * 0.25}
            defaults["izh_d_val"] = {"type": "gaussian", "mean": cfg.izh_d_val, "std": abs(cfg.izh_d_val) * 0.25 if cfg.izh_d_val != 0 else 10.0}
            defaults["izh_C_val"] = {"type": "gaussian", "mean": cfg.izh_C_val, "std": cfg.izh_C_val * 0.15}
        
        elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
            # HH conductances (CV ~ 0.4, log-normal)
            defaults["hh_g_Na_max"] = {"type": "lognormal", "mean_log": cp.log(cfg.hh_g_Na_max).item(), "sigma_log": 0.4}
            defaults["hh_g_K_max"] = {"type": "lognormal", "mean_log": cp.log(cfg.hh_g_K_max).item(), "sigma_log": 0.4}
            defaults["hh_g_L"] = {"type": "lognormal", "mean_log": cp.log(cfg.hh_g_L).item(), "sigma_log": 0.3}
            defaults["hh_C_m"] = {"type": "gaussian", "mean": cfg.hh_C_m, "std": cfg.hh_C_m * 0.15}
        
        return defaults
    
    def _initialize_ou_process_state(self, cfg, n):
        """Initializes Ornstein-Uhlenbeck process state for background drive.
        
        Based on:
        - Destexhe & Rudolph-Lilith (2012) "Neuronal Noise" Springer
        - Produces realistic Vm fluctuations (2-5 mV)
        - Tau = 10-20ms matches synaptic time constants
        
        The OU process is defined as:
            dI/dt = -(I - μ)/τ + σ√(2/τ) dW
        
        Exact solution over timestep dt:
            I(t+dt) = I(t)*exp(-dt/τ) + μ(1-exp(-dt/τ)) + σ√((1-exp(-2dt/τ))/2) * N(0,1)
        
        Args:
            cfg: CoreSimConfig with OU parameters
            n: Number of neurons
        """
        self._log_console(f"Initializing OU process state (tau={cfg.ou_tau_ms}ms, sigma={cfg.ou_std_current_pA}pA)...")
        
        # Initialize OU current state (starts at mean)
        self.cp_ou_current = cp.full(n, cfg.ou_mean_current_pA, dtype=cp.float32)
        
        # Pre-compute OU update coefficients using exact solution (Gillespie 1996)
        dt_sec = cfg.dt_ms / 1000.0
        tau_sec = cfg.ou_tau_ms / 1000.0
        
        # Decay factor: exp(-dt/tau)
        self.ou_decay_factor = float(cp.exp(-dt_sec / tau_sec))
        
        # Noise std: sigma * sqrt((1 - exp(-2*dt/tau)) / 2)
        # This ensures correct variance in steady state
        self.ou_noise_std = float(
            cfg.ou_std_current_pA * cp.sqrt((1.0 - cp.exp(-2.0 * dt_sec / tau_sec)) / 2.0)
        )
        
        # Store mean for convenience
        self.ou_mean = float(cfg.ou_mean_current_pA)

    def _calculate_distances_3d_gpu(self, pos_i_cp, pos_neighbors_cp):
        """Calculates Euclidean distances in 3D between a point and an array of other points using CuPy."""
        return _calculate_distances_3d_gpu(pos_i_cp, pos_neighbors_cp)

    def _generate_spatial_connections_3d_vectorized(self, n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config):
        """Generates connections using fully vectorized GPU operations (fast, scalable to 100K+ neurons)."""
        return generate_spatial_connections_gpu(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config, self._log_console)
    
    def _generate_random_connections_large(self, n, k, traits_np, trait_bias, min_w, max_w):
        """Generate random connections for very large networks when spatial constraints don't apply."""
        return generate_random_connections_large(n, k, traits_np, trait_bias, min_w, max_w, self._log_console)

    def _generate_spatial_connections_3d_binned(self, n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config):
        """Spatial binning approach for very large networks (>50k neurons)."""
        return generate_spatial_connections_binned(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config, self._log_console)

    def _generate_spatial_connections_3d_chunked(self, n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config):
        """Chunked version of vectorized connection generation for large networks."""
        return generate_spatial_connections_chunked(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config, self._log_console)
    
    def _generate_spatial_connections_3d(self, n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config):
        """Generates synaptic connections based on spatial proximity and trait similarity in 3D."""
        return generate_spatial_connections_3d(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config, self._log_console)

    def _generate_watts_strogatz_connections_3d(self, n, k_neighbors, p_rewire, config):
        """Generates connections using a Watts-Strogatz small-world network model in 3D."""
        return generate_watts_strogatz_3d(n, k_neighbors, p_rewire, config, self.cp_neuron_positions_3d, self._log_console)

    def _generate_motif_connections_3d(self, n, neuron_positions_3d_cp, traits_cp, config, motif_name):
        """Generates connections according to a high-level connectivity motif."""
        return generate_motif_connections_3d(n, neuron_positions_3d_cp, traits_cp, config, motif_name, CONNECTIVITY_MOTIFS, self._log_console)

    def apply_simulation_configuration_core(self, full_config_dict, is_part_of_playback_setup=False):
        """Applies a new simulation configuration from a full dictionary."""
        self._log_to_ui(f"Applying new simulation configuration... (playback_setup: {is_part_of_playback_setup})", "info")

        if self.runtime_state.is_running:
            self.stop_simulation()

        self.clear_simulation_state_and_gpu_memory()

        # Create new config objects from the provided dictionaries
        self.core_config = _create_config_from_dict(CoreSimConfig, full_config_dict.get("core_config"))
        self.viz_config = _create_config_from_dict(VisualizationConfig, full_config_dict.get("viz_config"))
        # We don't load runtime_state from profiles, so we re-initialize it.
        # Checkpoints might restore it, but that's handled in load_checkpoint.
        self.runtime_state = RuntimeState()

        # Enforce realistic profile ↔ neuron-type compatibility before applying
        # any auto-tuned overrides, so that tuning lookup matches the clamped
        # (model, profile, HH preset) combination actually used by the sim.
        if not is_part_of_playback_setup:
            enforce_profile_neuron_type_compatibility(self.core_config)

        # Apply auto-tuned overrides for this (model, profile, HH preset) combination if available.
        try:
            tuned_entry = get_auto_tuned_overrides_for_combo(
                self.core_config.neuron_model_type,
                getattr(self.core_config, "neural_profile_name", "GENERIC_UNSTRUCTURED"),
                getattr(self.core_config, "default_neuron_type_hh", None),
            )
            if tuned_entry and isinstance(tuned_entry, dict):
                core_overrides = tuned_entry.get("core_overrides", {})
                if isinstance(core_overrides, dict):
                    for key, value in core_overrides.items():
                        if hasattr(self.core_config, key):
                            setattr(self.core_config, key, value)
        except Exception as e:
            self._log_console(f"Warning: Failed to apply auto-tuned overrides: {e}", "warning")

        # Update max_delay_steps based on new config
        dt = self.core_config.dt_ms
        self.runtime_state.max_delay_steps = int(self.core_config.max_synaptic_delay_ms / dt) if dt > 0 else 200

        self._initialize_simulation_data(called_from_playback_init=is_part_of_playback_setup)

        if not self.is_initialized:
            self._log_to_ui("Failed to initialize simulation from new configuration. Critical error.", "critical")
            return False

        # Initialize experiment engine if an experiment config is loaded
        if self.experiment_config is not None and self.experiment_config.enabled:
            try:
                self.experiment_engine = ExperimentEngine(
                    self.core_config.num_neurons, self.core_config.dt_ms
                )
                self.experiment_engine.load_experiment(self.experiment_config)
                self.experiment_engine.initialize(
                    cp_traits=self.cp_traits, cp_module=cp
                )
                # Ensure sufficient connectivity between experiment input/output groups
                # for STDP-based learning to function (random networks often have too few paths)
                added = self.experiment_engine.ensure_inter_group_connectivity(self, cp)
                if added > 0:
                    self._log_to_ui(f"Injected {added} inter-group synapses for experiment learning paths", "info")
                self._log_to_ui(f"Experiment engine initialized: {self.experiment_config.name}", "info")
            except Exception as e:
                self._log_to_ui(f"Failed to initialize experiment engine: {e}", "warning")
                self.experiment_engine = None
        else:
            self.experiment_engine = None

        self.runtime_state.current_time_ms = 0.0
        self.runtime_state.current_time_step = 0
        self._log_to_ui(f"Sim config applied ({self.core_config.neuron_model_type}, N={self.core_config.num_neurons}). Sim re-initialized.", "success")
        return True

    def get_current_simulation_configuration_dict(self):
        """Returns the current simulation configuration as a dictionary."""
        return _get_full_config_dict(self.core_config, self.viz_config, self.runtime_state)

    def clear_simulation_state_and_gpu_memory(self):
        """Clears all CuPy arrays and resets the initialization flag."""
        self._log_console("Clearing simulation state and GPU memory...")
        attrs_to_clear = [
            'cp_membrane_potential_v','cp_recovery_variable_u', 'cp_conductance_g_e','cp_conductance_g_i','cp_conductance_g_nmda','cp_conductance_g_nmda_rise',
            'cp_external_input_current', 'cp_firing_states','cp_prev_firing_states','cp_traits',
            'cp_neuron_positions_3d','cp_connections', 'cp_refractory_timers', 'cp_viz_activity_timers',
            'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress',
            'cp_izh_C', 'cp_izh_k', 'cp_izh_vr', 'cp_izh_vt', 'cp_izh_vpeak',
            'cp_izh_a', 'cp_izh_b', 'cp_izh_c_reset', 'cp_izh_d_increment',
            'cp_izh_legacy_a', 'cp_izh_legacy_b', 'cp_izh_legacy_c_reset',
            'cp_izh_legacy_d_increment', 'cp_izh_legacy_vpeak',
            'cp_adex_w',
            'cp_gating_variable_m','cp_gating_variable_h','cp_gating_variable_n',
            'cp_hh_m_current_activation', 'cp_hh_CaT_m', 'cp_hh_CaT_h', 'cp_hh_h_current_q', 'cp_hh_NaP_activation',
            'cp_hh_C_m','cp_hh_g_Na_max','cp_hh_g_K_max','cp_hh_g_L',
            'cp_hh_E_Na','cp_hh_E_K','cp_hh_E_L', 'cp_hh_v_peak',
            'cp_neuron_firing_thresholds', 'cp_neuron_activity_ema',
            'cp_syn_reversal_potential_i_per_neuron',
            'cp_stp_u','cp_stp_x',
            'cp_ou_current'  # OU process state for background noise
        ]
        for attr_name in attrs_to_clear:
            if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                setattr(self, attr_name, None) 

        if is_gpu_backend() and 'cupy' in sys.modules:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception as e:
                self._log_console(f"Error freeing CuPy memory: {e}", "warning")

        # Cleanup experiment engine GPU resources
        if self.experiment_engine is not None:
            try:
                self.experiment_engine.cleanup()
            except Exception:
                pass
            self.experiment_engine = None

        # Invalidate all caches so stale data from previous network doesn't persist
        self._cached_coo_matrix = None
        self._coo_cache_valid = False
        self._cached_stp_per_type = None
        self._cached_inhibitory_mask = None
        self._cached_static_gui_data = None
        self._synapse_count = 0
        self._synapse_capacity = 0
        self._compaction_counter = 0
        self._pending_eliminations = False

        self.is_initialized = False
        self._log_console("Cleared simulation state and GPU memory.")

    def inject_explicit_wiring(self, wiring_plan, output_inhibitory_indices=None):
        """Replace auto-generated connectivity with an explicit wiring plan.

        Used by research runners that need a precise topology (G1 classifier and
        later gates). Must be called AFTER `_initialize_simulation_data()` so
        per-neuron state is already allocated.

        The plan is a dict of population-name -> dict with keys:
            pre_indices, post_indices, initial_weights, plastic (bool),
            conn_type (string, informational).

        If `output_inhibitory_indices` is non-empty, those neurons' trait is
        set to 1 (inhibitory) so their outgoing synapses route through the
        inhibitory conductance channel. Used for G1's lateral-inhibition layer.

        Side effects:
            - rebuilds self.cp_connections from the explicit edges
            - resets self._synapse_count / _synapse_capacity to match
            - invalidates COO cache, inhibitory mask cache, STP-type cache
            - re-initializes synapse-indexed arrays (pulse timers, conn type)
        """
        # Backend-aware: use the module-level csp (cupyx on CuPy, scipy
        # on NumPy). The function-local re-import was hard-coded to
        # cupyx, breaking the NumPy backend.

        n = self.core_config.num_neurons

        # Concatenate all populations into flat arrays, tracking the
        # per-population plastic flag and plasticity_gate name so we can
        # build per-synapse mask + per-synapse gain index after CSR
        # construction (order may differ from insertion).
        all_pre = []
        all_post = []
        all_w = []
        all_plastic = []
        all_gates = []  # gate_name string per synapse, or "" for ungated
        all_trans_gates = []  # transmission gate_name per synapse, or "" (scales CURRENT)
        any_fixed = False
        any_gated = False
        any_trans_gated = False
        for name, group in wiring_plan.items():
            if not isinstance(group, dict) or "pre_indices" not in group:
                continue
            plastic_flag = bool(group.get("plastic", True))
            gate_name = group.get("plasticity_gate", None) or ""
            trans_gate_name = group.get("transmission_gate", None) or ""
            if not plastic_flag:
                any_fixed = True
            if gate_name:
                any_gated = True
            if trans_gate_name:
                any_trans_gated = True
            n_syn = len(group["pre_indices"])
            all_pre.extend(group["pre_indices"])
            all_post.extend(group["post_indices"])
            all_w.extend([float(x) for x in group["initial_weights"]])
            all_plastic.extend([plastic_flag] * n_syn)
            all_gates.extend([gate_name] * n_syn)
            all_trans_gates.extend([trans_gate_name] * n_syn)

        if len(all_pre) == 0:
            self._log_console("inject_explicit_wiring: no synapses in plan.", "warning")
            return

        pre_np = np.asarray(all_pre, dtype=np.int32)
        post_np = np.asarray(all_post, dtype=np.int32)
        w_np = np.asarray(all_w, dtype=np.float32)
        plastic_np = np.asarray(all_plastic, dtype=np.bool_)

        pre_cp = cp.asarray(pre_np)
        post_cp = cp.asarray(post_np)
        w_cp = cp.asarray(w_np)

        # Build CSR in (pre -> post) layout. cp_connections[i, j] = weight of i->j.
        coo = csp.coo_matrix((w_cp, (pre_cp, post_cp)), shape=(n, n))
        self.cp_connections = coo.tocsr()
        # sum_duplicates() merges duplicate (i,j) entries. G1 has none, but be safe.
        self.cp_connections.sum_duplicates()

        nnz = int(self.cp_connections.nnz)
        self._synapse_count = nnz
        self._synapse_capacity = nnz

        # Build per-synapse plastic mask AND per-pathway plasticity-gate map
        # aligned with cp_connections.data order. tocoo() preserves CSR's
        # internal order (row-major by pre then post), so we re-sort the
        # original tuples by the same key to match. Sort once over
        # (pre, post, plastic, gate) tuples since they're row-aligned.
        if any_fixed or any_gated or any_trans_gated:
            keyed = sorted(
                zip(all_pre, all_post, all_plastic, all_gates, all_trans_gates),
                key=lambda t: (t[0], t[1]),
            )
        else:
            keyed = None

        if any_fixed:
            sorted_plastic = np.asarray([p for _, _, p, _, _ in keyed], dtype=np.bool_)
            self.cp_synapse_plastic_mask = cp.asarray(sorted_plastic)
        else:
            self.cp_synapse_plastic_mask = None

        # Per-pathway plasticity gates: build gate_name → synapse-indices map.
        # Allocate cp_plasticity_rate_gain only if any synapse is gated; otherwise
        # leave None and the plasticity update paths skip gain multiplication.
        if any_gated:
            sorted_gates = [g for _, _, _, g, _ in keyed]
            gate_to_indices: Dict[str, List[int]] = {}
            for syn_idx, gname in enumerate(sorted_gates):
                if gname:
                    gate_to_indices.setdefault(gname, []).append(syn_idx)
            self._plasticity_gate_to_synapses = gate_to_indices
            self._plasticity_gate_indices_gpu = {
                name: cp.asarray(np.asarray(indices, dtype=np.int32))
                for name, indices in gate_to_indices.items()
            }
            self._plasticity_gate_values = {n: 1.0 for n in gate_to_indices}
            # Default gain: 1.0 everywhere (full plasticity). Runners call
            # set_plasticity_gate(name, value) to alter at runtime.
            self.cp_plasticity_rate_gain = cp.ones(nnz, dtype=cp.float32)
        else:
            self._plasticity_gate_to_synapses = {}
            self._plasticity_gate_indices_gpu = {}
            self._plasticity_gate_values = {}
            self.cp_plasticity_rate_gain = None

        # Per-pathway TRANSMISSION gates (mirror of plasticity gates, but scales synaptic CURRENT, not
        # weight updates). gate_name → synapse-indices; cp_transmission_gain is the per-synapse multiplier
        # applied to effective_synaptic_strength in the step. Default 1.0 (open); runners call
        # set_transmission_gate(name, value) to open/close at runtime.
        if any_trans_gated:
            sorted_trans = [tg for _, _, _, _, tg in keyed]
            tgate_to_indices: Dict[str, List[int]] = {}
            for syn_idx, tgname in enumerate(sorted_trans):
                if tgname:
                    tgate_to_indices.setdefault(tgname, []).append(syn_idx)
            self._transmission_gate_to_synapses = tgate_to_indices
            self._transmission_gate_indices_gpu = {
                name: cp.asarray(np.asarray(indices, dtype=np.int32))
                for name, indices in tgate_to_indices.items()
            }
            self._transmission_gate_values = {n: 1.0 for n in tgate_to_indices}
            self.cp_transmission_gain = cp.ones(nnz, dtype=cp.float32)
        else:
            self._transmission_gate_to_synapses = {}
            self._transmission_gate_indices_gpu = {}
            self._transmission_gate_values = {}
            self.cp_transmission_gain = None

        # Cluster B.1 (2026-04-28): tag D2-targeting synapses with sign=-1.
        # D1-targeting + everything else stays at +1 (default). The reward-
        # modulated weight update will multiply by this sign so D2 synapses
        # move opposite to reward direction. Only allocated when the flag is
        # on and a region_manager is present (so we can resolve which post-
        # neurons belong to str_D2_* regions).
        if (getattr(self.core_config, "enable_d1_d2_asymmetry", False)
                and self.region_manager is not None):
            self.cp_d1_d2_sign = cp.ones(nnz, dtype=cp.float32)
            # Collect post-neuron indices for all str_D2_* regions.
            d2_post_indices: List[int] = []
            for region in self.region_manager.regions():
                if region.name.startswith("str_D2_"):
                    d2_post_indices.extend(self.region_manager.indices(region.name))
            if d2_post_indices:
                d2_set_gpu = cp.asarray(
                    np.asarray(d2_post_indices, dtype=np.int64)
                )
                # cp_connections.indices is the post-neuron column for each
                # synapse in CSR data order. Mask synapses whose post is in D2.
                d2_mask = cp.isin(self.cp_connections.indices, d2_set_gpu)
                self.cp_d1_d2_sign[d2_mask] = -1.0
        else:
            self.cp_d1_d2_sign = None

        # E.3 batched-replica: clear any stale per-synapse reward override.
        # If a runner was using it pre-inject, the size won't match new nnz;
        # the runner is responsible for repopulating after inject if needed.
        self.cp_per_synapse_reward_override = None

        # Cluster C v2 (2026-04-29): per-synapse action tag for compartmentalized
        # DA. tag[i] = action_index of synapse i's POST region (∈ [0, N-1]) or
        # -1 for global / non-action-specific synapses. Allocated whenever
        # a region_manager is present so consumers (compute_per_synapse_da_signal)
        # can rely on it; default -1 produces no effect under the v1 path.
        # See docs/plans/2026-04-29-cluster-c-v2-compartmentalized-da-design.md.
        if self.region_manager is not None:
            self.cp_synapse_action_tag = cp.full(nnz, -1, dtype=cp.int32)
            # Build per-action post-neuron index set; for each region with a
            # non-None action_index, mark synapses whose post is in that region.
            for region in self.region_manager.regions():
                a_idx = getattr(region, "action_index", None)
                if a_idx is None:
                    continue
                region_post = self.region_manager.indices(region.name)
                if not region_post:
                    continue
                post_set_gpu = cp.asarray(
                    np.asarray(region_post, dtype=np.int64)
                )
                mask = cp.isin(self.cp_connections.indices, post_set_gpu)
                self.cp_synapse_action_tag[mask] = int(a_idx)
        else:
            self.cp_synapse_action_tag = None

        # Flip output trait to inhibitory if requested (enables lateral inhibition).
        if output_inhibitory_indices:
            inh_idx_cp = cp.asarray(np.asarray(output_inhibitory_indices, dtype=np.int32))
            if self.cp_traits is None:
                self.cp_traits = cp.zeros(n, dtype=cp.int32)
            self.cp_traits[inh_idx_cp] = 1
            if 1 not in self.core_config.inhibitory_trait_indices:
                self.core_config.inhibitory_trait_indices = list(
                    set(list(self.core_config.inhibitory_trait_indices) + [1])
                )

        # Invalidate any caches that depend on connections or traits.
        self._invalidate_coo_cache()
        self._cached_inhibitory_mask = None
        self._cached_stp_per_type = None

        # Re-initialize synapse-indexed arrays to match the new nnz.
        if self.gpu_config is not None:
            # Pulse timers (visualization)
            if self.cp_synapse_pulse_timers is not None:
                self.cp_synapse_pulse_timers = cp.zeros(nnz, dtype=cp.int32)
                self.cp_synapse_pulse_progress = cp.zeros(nnz, dtype=cp.float32)

            # Reset STP state arrays (sized to capacity).
            if self.core_config.enable_short_term_plasticity:
                self.cp_stp_x = cp.ones(nnz, dtype=cp.float32)
                self.cp_stp_u = cp.full(nnz, self.core_config.stp_U, dtype=cp.float32)

            # Reset eligibility traces.
            if self.core_config.enable_reward_modulation:
                self.cp_eligibility_trace = cp.zeros(nnz, dtype=cp.float32)

            # Reset per-synapse conn type (for per-type STP).
            self.cp_synapse_conn_type = None
            if (self.core_config.enable_per_type_stp and
                    self.core_config.enable_short_term_plasticity and nnz > 0):
                self._build_synapse_conn_type_array(self.core_config)

        self._log_console(
            f"inject_explicit_wiring: installed {nnz} synapses across "
            f"{sum(1 for k,v in wiring_plan.items() if isinstance(v, dict) and 'pre_indices' in v)} populations."
        )

    # ─────────────────────────── Plasticity gates ───────────────────────────
    # Per-pathway plasticity gating (Stage 1, 2026-04-27). When a pathway
    # is built with plasticity_gate="some_name", all its synapses share a
    # runtime-controllable gain. set_plasticity_gate("some_name", 0.0) freezes
    # all those synapses (no STDP, no eligibility, no reward updates).
    # set_plasticity_gate("some_name", 1.0) thaws.
    #
    # Biological grounding: developmental staging (sensory cortex matures
    # before association cortex), critical periods (visual cortex ocular
    # dominance plasticity closes via PV interneuron maturation), and
    # neuromodulator-gated plasticity windows. The gate is the abstraction;
    # what controls it (a fixed schedule, a neuromodulator concentration, a
    # developmental clock) is up to the runner / experiment configuration.

    # Deprecated gate-name aliases. Old name on the LEFT, canonical on the RIGHT.
    # When a caller uses an old name, it's silently translated to the canonical
    # form. Emit a one-time DeprecationWarning per (gate, frame) so the deprecation
    # surfaces in CI logs without spamming every step.
    _DEPRECATED_GATE_NAMES = {
        # 2026-04-29 Wave-1 rename #1: cortex_to_d1 was applied to D1, D2, AND
        # patch pathways — the name only described one of three. Use
        # "corticostriatal".
        "cortex_to_d1": "corticostriatal",
        # 2026-04-29 Wave-1 rename #2: pfc_pathways follows the pfc -> dlpfc_wm
        # rename. The implementation gates dlPFC working-memory recurrent + I/O
        # plasticity, not all of prefrontal cortex.
        "pfc_pathways": "dlpfc_wm_pathways",
        # 2026-04-29 Wave-2 rename #20: hippo_to_cortex follows the
        # place_cells -> sensor_place_readout + goal_cells -> ppc_goal_input
        # renames. The gate covers both place + goal readout, neither of
        # which is canonical hippocampus.
        "hippo_to_cortex": "place_goal_to_cortex",
        # 2026-04-29 Wave-2 rename #21: pfc_internal follows pfc -> dlpfc_wm.
        # Gate covers internal recurrent connectivity of the dlPFC WM region.
        "pfc_internal": "dlpfc_wm_recurrent",
        # 2026-04-29 Wave-2 rename #19: bg_cross_projections is more specifically
        # the corticostriatal cross-action gate (cortex_X -> str_D1/D2_Y for X!=Y),
        # not BG-internal cross.
        "bg_cross_projections": "corticostriatal_cross",
    }

    def _canonicalize_gate_name(self, name: str) -> str:
        canonical = self._DEPRECATED_GATE_NAMES.get(name)
        if canonical is None:
            return name
        if not hasattr(self, "_warned_deprecated_gates"):
            self._warned_deprecated_gates = set()
        if name not in self._warned_deprecated_gates:
            import warnings
            warnings.warn(
                f"Plasticity gate name '{name}' is deprecated; use '{canonical}' instead. "
                f"Old name will be removed in a future release.",
                DeprecationWarning,
                stacklevel=3,
            )
            self._warned_deprecated_gates.add(name)
        return canonical

    @property
    def cp_plasticity_gain(self):
        """DEPRECATED 2026-04-29 (Wave-1 rename #12). Use
        `cp_plasticity_rate_gain` instead — the new name distinguishes the
        continuous *rate* multiplier from the binary `cp_plasticity_window_gate`
        driven by the neuromodulator system."""
        if not hasattr(self, "_warned_cp_plasticity_gain"):
            import warnings
            warnings.warn(
                "bridge.cp_plasticity_gain is deprecated; use "
                "bridge.cp_plasticity_rate_gain instead. The new name "
                "distinguishes the continuous rate multiplier from the binary "
                "cp_plasticity_window_gate driven by the neuromodulator system.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._warned_cp_plasticity_gain = True
        return self.cp_plasticity_rate_gain

    @cp_plasticity_gain.setter
    def cp_plasticity_gain(self, value):
        if not hasattr(self, "_warned_cp_plasticity_gain"):
            import warnings
            warnings.warn(
                "bridge.cp_plasticity_gain is deprecated; use "
                "bridge.cp_plasticity_rate_gain instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._warned_cp_plasticity_gain = True
        self.cp_plasticity_rate_gain = value

    def set_token_drive(
        self,
        token: str,
        drive_pA: float = 200.0,
        sparsity: float = 0.1,
        region_name: str = "language_input",
        orthogonal_cue_idx: int = None,
        n_orthogonal_cues: int = 4,
    ) -> int:
        """Inject a sparse current pattern representing `token` into the
        language_input region (or another named region).

        Used by the text I/O training pipeline and interactive mode. The
        token's deterministic embedding (sim.text_embeddings.embed) maps
        to a sparse {0, drive_pA} vector via vocab_to_drive_pattern.

        Args:
            token: vocabulary word (lowercased internally).
            drive_pA: input current for active neurons.
            sparsity: fraction of neurons activated (default 0.1 = 10%).
            region_name: which region to drive. Default 'language_input'.
            orthogonal_cue_idx: when not None, USE orthogonal_drive_pattern
                with this cue_idx instead of the default hash-based
                vocab_to_drive_pattern. The `token` argument is then
                ignored (the cue is identified solely by its index).
                Use case: 2026-05-05 step 2 W→A experiment testing
                whether input-encoding ambiguity is the bottleneck for
                3-factor learning.
            n_orthogonal_cues: total cue count (only used in orthogonal
                mode). Determines the band stride.

        Returns: count of neurons activated.

        Raises:
            RuntimeError if region_manager is None or region not found.
        """
        from sim.text_embeddings import vocab_to_drive_pattern, orthogonal_drive_pattern

        if self.region_manager is None:
            raise RuntimeError(
                "set_token_drive: bridge.region_manager is None. "
                "Brain-region framework must be enabled."
            )
        try:
            indices = list(self.region_manager.indices(region_name))
        except Exception as e:
            raise RuntimeError(
                f"set_token_drive: region '{region_name}' not found: {e}"
            ) from None
        if not indices:
            raise RuntimeError(
                f"set_token_drive: region '{region_name}' has no neurons"
            )

        n = len(indices)
        if orthogonal_cue_idx is not None:
            drive = orthogonal_drive_pattern(
                cue_idx=orthogonal_cue_idx,
                n_cues=n_orthogonal_cues,
                n_neurons=n, drive_max_pA=drive_pA, sparsity=sparsity,
            )
        else:
            drive = vocab_to_drive_pattern(
                token, n_neurons=n, drive_max_pA=drive_pA, sparsity=sparsity,
            )
        idx_cp = cp.asarray(indices, dtype=cp.int64)
        self.cp_external_input_current[idx_cp] = cp.asarray(drive, dtype=cp.float32)
        return int(np.sum(drive > 0))

    def read_language_output(
        self,
        spike_counts,
        n_steps: int,
        top_k: int = 1,
        vocab=None,
        region_name: str = "language_output",
    ):
        """Decode language_output region's recent firing pattern to the
        nearest token(s) in the vocabulary.

        Args:
            spike_counts: (n_neurons,) np.ndarray or cp.ndarray of spike
                counts accumulated over the readout window. Caller is
                responsible for tallying these via the env loop.
            n_steps: number of sub-steps the spike_counts span (used for
                normalization to firing rate).
            top_k: number of tokens to return (default 1).
            vocab: list of candidate tokens. Defaults to text_embeddings.DEFAULT_VOCAB.
            region_name: which region's spike counts (default 'language_output').

        Returns: list of `top_k` tokens, ranked by cosine similarity.
        """
        from sim.text_embeddings import nearest_token

        if self.region_manager is None:
            raise RuntimeError(
                "read_language_output: bridge.region_manager is None"
            )
        # spike_counts may be a cupy array; convert to numpy.
        if hasattr(spike_counts, "get"):
            sc = spike_counts.get()
        else:
            sc = np.asarray(spike_counts)
        if n_steps <= 0:
            n_steps = 1
        # Mean firing rate normalization (units don't matter for cosine
        # similarity but make the activity vector independent of run length)
        activity = (sc.astype(np.float32) / float(n_steps))
        return nearest_token(activity, vocab=vocab, k=top_k, dim=int(activity.size))

    def set_pathway_weights(
        self,
        pathway_name: str,
        pre_indices,
        post_indices,
        weights,
        add_missing: bool = False,
    ) -> int:
        """Overwrite weights for specific (pre, post) edges in cp_connections.

        Used by post-build pathway initialization (e.g. Gabor pre-init for
        V1 simple cells in Cluster K v2) and any future helper that needs
        to install hand-computed weights into a pre-built CSR.

        Args:
            pathway_name: informational tag for logging — does NOT need to
                match any registered pathway. Each call is treated
                independently.
            pre_indices: (N,) array of int global pre-neuron indices.
            post_indices: (N,) array of int global post-neuron indices.
            weights: (N,) array of float32 weights to install.
            add_missing: if False (default), raises ValueError when any
                (pre, post) pair is not in the existing CSR. If True,
                missing edges are added (requires CSR rebuild).

        Returns: count of edges updated.

        Raises:
            ValueError: if any (pre, post) is missing and add_missing=False.
            RuntimeError: if cp_connections is None.
        """
        # Backend-aware: use the module-level csp (cupyx on CuPy, scipy
        # on NumPy). The function-local re-import was hard-coded to
        # cupyx, breaking the NumPy backend.

        if self.cp_connections is None:
            raise RuntimeError(
                f"set_pathway_weights('{pathway_name}'): cp_connections is None. "
                f"Call _initialize_simulation_data first."
            )
        pre_np = np.asarray(pre_indices, dtype=np.int64)
        post_np = np.asarray(post_indices, dtype=np.int64)
        w_np = np.asarray(weights, dtype=np.float32)
        if pre_np.shape != post_np.shape or pre_np.shape != w_np.shape:
            raise ValueError(
                f"set_pathway_weights('{pathway_name}'): shape mismatch — "
                f"pre {pre_np.shape}, post {post_np.shape}, weights {w_np.shape}"
            )
        n_input = int(pre_np.size)
        if n_input == 0:
            return 0

        # Pull CSR structure to host once (small cost vs N edge lookups)
        indptr = _backend_to_host(self.cp_connections.indptr)
        indices = _backend_to_host(self.cp_connections.indices)
        data = _backend_to_host(self.cp_connections.data)

        # Build a dict: (pre, post) -> data index. O(nnz) one-time cost.
        # For each row r, iterate indices[indptr[r]:indptr[r+1]].
        pair_to_idx = {}
        n_rows = int(self.cp_connections.shape[0])
        for r in range(n_rows):
            start = int(indptr[r])
            end = int(indptr[r + 1])
            for off in range(start, end):
                pair_to_idx[(int(r), int(indices[off]))] = off

        n_updated = 0
        missing_pairs = []
        for i in range(n_input):
            key = (int(pre_np[i]), int(post_np[i]))
            if key in pair_to_idx:
                data[pair_to_idx[key]] = float(w_np[i])
                n_updated += 1
            else:
                missing_pairs.append(key)

        if missing_pairs and not add_missing:
            head = missing_pairs[:5]
            raise ValueError(
                f"set_pathway_weights('{pathway_name}'): {len(missing_pairs)} "
                f"of {n_input} (pre, post) pairs not found in CSR. "
                f"First few: {head}. Set add_missing=True to add them."
            )

        # Push updated data back to GPU
        self.cp_connections.data = cp.asarray(data, dtype=cp.float32)

        if missing_pairs and add_missing:
            # Count adds toward the return value
            n_updated += len(missing_pairs)
            # Rebuild CSR with new edges.
            # Build per-missing arrays in the order they appeared in inputs
            new_pre_list = []
            new_post_list = []
            new_w_list = []
            missing_set = set(missing_pairs)
            for i in range(n_input):
                key = (int(pre_np[i]), int(post_np[i]))
                if key in missing_set:
                    new_pre_list.append(key[0])
                    new_post_list.append(key[1])
                    new_w_list.append(float(w_np[i]))
            new_pre = np.array(new_pre_list, dtype=np.int64)
            new_post = np.array(new_post_list, dtype=np.int64)
            new_w = np.array(new_w_list, dtype=np.float32)

            existing_coo = self.cp_connections.tocoo(copy=False)
            all_pre = cp.concatenate([existing_coo.row, cp.asarray(new_pre)])
            all_post = cp.concatenate([existing_coo.col, cp.asarray(new_post)])
            all_w = cp.concatenate([existing_coo.data, cp.asarray(new_w)])
            n = self.core_config.num_neurons
            coo = csp.coo_matrix((all_w, (all_pre, all_post)), shape=(n, n))
            self.cp_connections = coo.tocsr()
            self.cp_connections.sum_duplicates()
            new_nnz = int(self.cp_connections.nnz)

            # Invalidate caches BEFORE rebuilding conn_types: the cached COO
            # is from before the CSR rebuild and must not be reused.
            self._invalidate_coo_cache()

            # Resize synapse-indexed arrays to match the grown CSR. Mirrors
            # the pattern in inject_explicit_wiring: reinit STP / eligibility /
            # pulse timers / conn type / plastic mask / plasticity gain to
            # the new size. NB: this DOES wipe in-flight STP state,
            # eligibility traces, etc. — only safe at init time, not during
            # a running simulation.
            if new_nnz != self._synapse_count:
                self._synapse_count = new_nnz
                self._synapse_capacity = new_nnz
                if self.core_config.enable_short_term_plasticity:
                    self.cp_stp_x = cp.ones(new_nnz, dtype=cp.float32)
                    self.cp_stp_u = cp.full(new_nnz, self.core_config.stp_U,
                                             dtype=cp.float32)
                if self.core_config.enable_reward_modulation:
                    self.cp_eligibility_trace = cp.zeros(new_nnz, dtype=cp.float32)
                if self.cp_synapse_pulse_timers is not None:
                    self.cp_synapse_pulse_timers = cp.zeros(new_nnz, dtype=cp.int32)
                    self.cp_synapse_pulse_progress = cp.zeros(new_nnz,
                                                              dtype=cp.float32)
                # Per-synapse conn type rebuilt from the new CSR
                self.cp_synapse_conn_type = None
                if (self.core_config.enable_per_type_stp
                        and self.core_config.enable_short_term_plasticity
                        and new_nnz > 0):
                    self._build_synapse_conn_type_array(self.core_config)
                # Plastic mask: new edges default to non-plastic (False)
                if hasattr(self, "cp_plastic_mask") and self.cp_plastic_mask is not None:
                    if self.cp_plastic_mask.shape[0] != new_nnz:
                        old_mask = self.cp_plastic_mask
                        new_mask = cp.zeros(new_nnz, dtype=cp.bool_)
                        new_mask[:old_mask.shape[0]] = old_mask
                        self.cp_plastic_mask = new_mask
                # Plasticity rate gain: new edges default to 1.0
                if (hasattr(self, "cp_plasticity_rate_gain")
                        and self.cp_plasticity_rate_gain is not None):
                    if self.cp_plasticity_rate_gain.shape[0] != new_nnz:
                        old_gain = self.cp_plasticity_rate_gain
                        new_gain = cp.ones(new_nnz, dtype=cp.float32)
                        new_gain[:old_gain.shape[0]] = old_gain
                        self.cp_plasticity_rate_gain = new_gain

        # Invalidate caches that depend on connectivity / weights
        self._invalidate_coo_cache()

        return n_updated

    def set_plasticity_gate(self, name: str, value: float) -> None:
        """Set the runtime plasticity gain for all synapses in pathways
        tagged with `name`.

        Default gain on inject is 1.0 (full plasticity). Set to 0.0 to freeze
        (no weight changes from any source for tagged synapses), or any value
        in between for partial.

        Raises KeyError if `name` was not declared on any pathway in the
        active wiring plan.
        """
        name = self._canonicalize_gate_name(name)
        if name not in self._plasticity_gate_to_synapses:
            raise KeyError(
                f"No plasticity gate named '{name}'. "
                f"Known gates: {list(self._plasticity_gate_to_synapses.keys())}"
            )
        self._plasticity_gate_values[name] = float(value)
        if self.cp_plasticity_rate_gain is None:
            return
        indices = self._plasticity_gate_indices_gpu[name]
        # Bound nnz vs gain length for safety against post-init capacity changes
        nnz = self.cp_plasticity_rate_gain.shape[0]
        if indices.size > 0 and int(indices.max()) < nnz:
            self.cp_plasticity_rate_gain[indices] = cp.float32(value)

    def set_transmission_gate(self, name: str, value: float) -> None:
        """Set the runtime synaptic-CURRENT gain for all synapses in pathways tagged with
        `transmission_gate=name`.

        Default gain on inject is 1.0 (full transmission). Set to 0.0 to CLOSE the route (no synaptic
        current flows through it, even though its weight is non-zero), or any value in between. This is the
        complement of set_plasticity_gate: that one freezes weight UPDATES but leaves current flowing; this
        one gates the CURRENT itself. Used for thalamocortical dynamical gating -- pre-wire a route with a
        fixed weight, hold it closed, and open it on command so binding = which gate is open.

        Raises KeyError if `name` was not declared as a transmission_gate on any pathway.
        """
        name = self._canonicalize_gate_name(name)
        if name not in self._transmission_gate_to_synapses:
            raise KeyError(
                f"No transmission gate named '{name}'. "
                f"Known gates: {list(self._transmission_gate_to_synapses.keys())}"
            )
        self._transmission_gate_values[name] = float(value)
        if self.cp_transmission_gain is None:
            return
        indices = self._transmission_gate_indices_gpu[name]
        nnz = self.cp_transmission_gain.shape[0]
        if indices.size > 0 and int(indices.max()) < nnz:
            self.cp_transmission_gain[indices] = cp.float32(value)

    def couple_gate_to_pool(self, gate_name: str, control_region_name: str, threshold: float = 0.05,
                            alpha: float = 0.3, open_value: float = 1.0) -> None:
        """Drive a transmission gate from the FIRING of a control population, in-substrate (the
        thalamocortical loop without a runner read). Each step, if the control region's smoothed firing rate
        (EMA) is >= threshold the gate opens (to open_value), else it closes. So disinhibiting a thalamic gate
        pool -> its activity -> the cortical route gate opens, entirely inside _run_one_simulation_step.

        Requires the brain-region framework (control_region_name is resolved to neuron indices). The gate must
        have been declared as a `transmission_gate` on some pathway.
        """
        gate_name = self._canonicalize_gate_name(gate_name)
        if gate_name not in self._transmission_gate_to_synapses:
            raise KeyError(f"No transmission gate named '{gate_name}'.")
        if self.region_manager is None:
            raise RuntimeError("couple_gate_to_pool requires the brain-region framework (region_manager).")
        idx = self.region_manager.indices(control_region_name)
        self._gate_couplings.append({
            "gate_name": gate_name,
            "control_idx": cp.asarray(np.asarray(idx, dtype=np.int64)),
            "threshold": float(threshold), "alpha": float(alpha), "open_value": float(open_value),
            "ema": 0.0, "last_value": None,
        })

    def _apply_gate_couplings(self) -> None:
        """Per-step hook: update activity-driven transmission gates from control-pool firing. No-op when none
        are registered (zero overhead). Called after cp_firing_states is finalized."""
        if not self._gate_couplings:
            return
        for c in self._gate_couplings:
            rate = float(self.cp_firing_states[c["control_idx"]].mean())   # firing fraction of the control pool
            c["ema"] = c["alpha"] * rate + (1.0 - c["alpha"]) * c["ema"]
            value = c["open_value"] if c["ema"] >= c["threshold"] else 0.0
            if value != c["last_value"]:                                  # only write the gate when it changes
                self.set_transmission_gate(c["gate_name"], value)
                c["last_value"] = value

    # ──────────────────────────────────────────────────────────────────
    # Engram-tagging API (P2 / roadmap T1.C / catalog D.14)
    # ──────────────────────────────────────────────────────────────────
    # Tonegawa et al's ensemble-tagging paradigm in code form. A named
    # "engram tag" is just the set of neurons that fired above some
    # threshold during a window of simulation. Once tagged, the same
    # ensemble can be reactivated (`stimulate_tag`) — closing the loop
    # between correlational observation and causal driving.
    #
    # Usage:
    #     bridge.start_engram_recording("apple")
    #     for _ in range(n_steps):
    #         bridge._run_one_simulation_step()  # auto-accumulates
    #     stats = bridge.commit_engram_tag("apple", top_k=50)
    #     # Later, recall by causal stimulation:
    #     n = bridge.stimulate_tag("apple", drive_pA=200.0)
    #     ...
    #     bridge.clear_tag_drive()
    #
    # Tags persist across sim steps in self._engram_tags (CuPy/NumPy
    # int64 arrays of global neuron indices). They're cleared on
    # bridge re-init unless saved separately. Persistence to lineage
    # is straightforward — int arrays serialize trivially.

    def _init_engram_tagging(self) -> None:
        """Initialize engram-tagging structures. Called by
        _initialize_simulation_data once cp_firing_states exists."""
        if not hasattr(self, "_engram_tags") or self._engram_tags is None:
            self._engram_tags: dict = {}
        if not hasattr(self, "_engram_recordings") or \
                self._engram_recordings is None:
            self._engram_recordings: dict = {}

    def start_engram_recording(self, name: str) -> None:
        """Start accumulating spike counts for engram tag `name`.

        Each subsequent _run_one_simulation_step automatically adds
        the per-neuron spike state to the recording. Call
        commit_engram_tag(name, ...) to finalize.

        Catalog: D.14 (Tonegawa engram cells).
        """
        self._init_engram_tagging()
        n = int(self.cp_firing_states.shape[0])
        self._engram_recordings[name] = {
            "spike_counts": cp.zeros(n, dtype=cp.float32),
            "n_steps": 0,
        }

    def _tick_engram_recordings(self) -> None:
        """Internal: called once per simulation step to accumulate
        spike counts for active recordings. No-op when no active
        recordings — zero overhead when not in use."""
        if not getattr(self, "_engram_recordings", None):
            return
        if self.cp_firing_states is None:
            return
        fired_f32 = self.cp_firing_states.astype(cp.float32)
        for rec in self._engram_recordings.values():
            rec["spike_counts"] += fired_f32
            rec["n_steps"] += 1

    def commit_engram_tag(
        self,
        name: str,
        threshold_hz: float = 5.0,
        top_k: Optional[int] = None,
        region_filter: Optional[list] = None,
    ) -> dict:
        """Finalize an engram tag from accumulated spike counts.

        Two selection modes:
        - threshold_hz: tag neurons firing above (threshold_hz *
          window_seconds) total spikes during the recording.
        - top_k: tag the top K neurons by spike count regardless of
          rate (Marr-like sparse engram).

        If both are given, top_k wins.

        Args:
            name: tag identifier (must match a prior start_engram_recording)
            threshold_hz: minimum firing rate (default 5 Hz)
            top_k: alternative: tag top K spike-count neurons
            region_filter: list of region names; only consider neurons
                from these regions (e.g. ["ca3"] for hippocampal engrams)

        Returns:
            {"name": str, "n_tagged": int, "n_recorded_steps": int,
             "window_ms": float, "mean_spike_count": float}
        """
        self._init_engram_tagging()
        if name not in self._engram_recordings:
            raise KeyError(
                f"No active engram recording for {name!r}. "
                f"Call start_engram_recording({name!r}) first."
            )
        rec = self._engram_recordings.pop(name)
        spike_counts = rec["spike_counts"]
        n_steps = rec["n_steps"]
        window_ms = n_steps * float(self.core_config.dt_ms)
        window_s = window_ms / 1000.0

        # Region filter
        n_total = int(spike_counts.shape[0])
        candidate_mask = cp.ones(n_total, dtype=bool)
        if region_filter and self.region_manager is not None:
            candidate_mask = cp.zeros(n_total, dtype=bool)
            for rname in region_filter:
                try:
                    rindices = self.region_manager.indices(rname)
                    rarr = cp.asarray(list(rindices), dtype=cp.int64)
                    candidate_mask[rarr] = True
                except Exception:
                    pass

        masked_counts = cp.where(candidate_mask, spike_counts,
                                    cp.float32(-1.0))

        if top_k is not None and int(top_k) > 0:
            # Top-K selection
            k = int(top_k)
            # Use argsort descending; mask out non-candidates first
            order = cp.argsort(-masked_counts)
            top_indices = order[:k]
            # Filter out any -1 sentinel (non-candidate)
            valid_mask = masked_counts[top_indices] > 0
            indices = top_indices[valid_mask]
        else:
            # Threshold-based: spikes >= threshold_hz * window_s
            min_spikes = max(1.0, float(threshold_hz) * max(window_s, 1e-3))
            indices = cp.where(spike_counts >= cp.float32(min_spikes))[0]
            # Apply region filter
            if region_filter and self.region_manager is not None:
                indices_mask = candidate_mask[indices]
                indices = indices[indices_mask]

        # Store as int64 indices (host-compatible)
        self._engram_tags[name] = indices.astype(cp.int64)
        mean_count = float(spike_counts.mean()) if n_total > 0 else 0.0
        return {
            "name": name,
            "n_tagged": int(indices.shape[0]),
            "n_recorded_steps": int(n_steps),
            "window_ms": window_ms,
            "mean_spike_count": mean_count,
        }

    def stimulate_tag(self, name: str, drive_pA: float,
                        additive: bool = False) -> int:
        """Drive all neurons in engram tag `name` to drive_pA.

        Args:
            name: tag identifier (must exist via commit_engram_tag)
            drive_pA: input current (pA)
            additive: if True, ADD to existing
                cp_external_input_current; if False (default),
                overwrite at tagged indices.

        Returns: number of neurons stimulated.

        Catalog: D.14 — the "stimulate the tag" half of the
        Tonegawa paradigm. Drive the same ensemble that fired
        during encoding and the network treats it as recall.
        """
        self._init_engram_tagging()
        if name not in self._engram_tags:
            raise KeyError(f"No engram tag {name!r}. Did you commit it?")
        indices = self._engram_tags[name]
        if indices.shape[0] == 0:
            return 0
        if additive:
            self.cp_external_input_current[indices] = \
                self.cp_external_input_current[indices] + cp.float32(drive_pA)
        else:
            self.cp_external_input_current[indices] = cp.float32(drive_pA)
        return int(indices.shape[0])

    def clear_tag_drive(self, name: Optional[str] = None) -> None:
        """Zero the external drive. If name given, only at that tag's
        indices; else clear everything (matches existing pattern in
        other drive helpers)."""
        self._init_engram_tagging()
        if name is None:
            self.cp_external_input_current[:] = 0.0
            return
        if name not in self._engram_tags:
            return
        indices = self._engram_tags[name]
        if indices.shape[0] > 0:
            self.cp_external_input_current[indices] = 0.0

    def list_engram_tags(self) -> list:
        """List committed engram tags with sizes (for inspection)."""
        self._init_engram_tagging()
        return [
            {"name": k, "n_neurons": int(v.shape[0])}
            for k, v in self._engram_tags.items()
        ]

    def get_engram_tag_indices(self, name: str):
        """Return the int64 array of tagged neuron indices (CuPy or
        NumPy depending on backend). Useful for analysis."""
        self._init_engram_tagging()
        if name not in self._engram_tags:
            raise KeyError(name)
        return self._engram_tags[name]

    def delete_engram_tag(self, name: str) -> bool:
        """Delete an engram tag. Returns True if it existed."""
        self._init_engram_tagging()
        return self._engram_tags.pop(name, None) is not None

    def extract_per_pathway_csrs(self) -> dict:
        """Split the monolithic cp_connections into per-pathway sub-matrices.

        Used for SSD synapse paging (Phase 3 of tiering design) — each
        pathway becomes a separately-loadable shard. Each sub-matrix has
        shape (n_post_region, n_pre_region) and contains only the edges
        for that specific pathway.

        Returns:
            Dict mapping pathway name (e.g. "language_input_to_motor_N") to
            scipy.sparse.csr_matrix (the per-pathway sub-CSR).

        Requires:
            - self.region_manager is not None (brain region framework enabled)
            - self.cp_connections is initialized

        On NumPy backend the result is scipy.sparse; on CuPy backend
        the result is cupyx.scipy.sparse. Caller should convert via
        _backend_to_host() before disk persistence (TieredSynapseStore
        expects scipy.sparse).
        """
        if self.region_manager is None:
            raise RuntimeError(
                "extract_per_pathway_csrs: region_manager is None — "
                "brain region framework must be enabled"
            )
        if self.cp_connections is None:
            raise RuntimeError(
                "extract_per_pathway_csrs: cp_connections is None — "
                "_initialize_simulation_data must have been called"
            )
        result = {}
        # We use scipy.sparse on the host side regardless of backend,
        # because TieredSynapseStore is scipy.sparse-only and shards
        # are stored as numpy .npz files.
        import scipy.sparse as sp_host
        import numpy as np
        # Pull CSR to host once if on CuPy
        try:
            indptr = _backend_to_host(self.cp_connections.indptr)
            indices = _backend_to_host(self.cp_connections.indices)
            data = _backend_to_host(self.cp_connections.data)
        except NameError:
            # _backend_to_host not in scope (e.g. defensive bootstrap)
            indptr = self.cp_connections.indptr
            indices = self.cp_connections.indices
            data = self.cp_connections.data
            if hasattr(indptr, "get"):
                indptr = indptr.get()
            if hasattr(indices, "get"):
                indices = indices.get()
            if hasattr(data, "get"):
                data = data.get()
        n = int(self.cp_connections.shape[0])
        full_csr = sp_host.csr_matrix(
            (data, indices, indptr), shape=(n, n)
        )

        for pw in self.region_manager.pathways():
            pre_indices = np.array(
                list(self.region_manager.indices(pw.from_region)),
                dtype=np.int64,
            )
            post_indices = np.array(
                list(self.region_manager.indices(pw.to_region)),
                dtype=np.int64,
            )
            # Slice rows then columns (CSR -> CSR slicing is cheap)
            sub = full_csr[post_indices, :][:, pre_indices].tocsr()
            # Naming convention: <from_region>_to_<to_region>
            pw_name = f"{pw.from_region}_to_{pw.to_region}"
            result[pw_name] = sub
        return result

    def _initialize_synapse_store(self, cfg) -> None:
        """Build the TieredSynapseStore mirror for Phase 3 Strategy B.

        Called from _initialize_simulation_data when
        cfg.enable_synapse_tiering=True AND region_manager is set.
        The store mirrors the bridge's per-pathway CSRs so activity
        can be tracked + future Phase 4 auto-tiering can fire.

        Inference still uses the monolithic self.cp_connections; the
        store is observational. Strategy A would later make the store
        the source of truth for compute.
        """
        from pathlib import Path
        from sim.synapse_storage import TieredSynapseStore

        root = cfg.synapse_tiering_root
        if not root:
            # Default: process-local active directory
            root = "bridges/synapse_shards/active"
        self.synapse_store = TieredSynapseStore(
            root=Path(root),
            evict_after_idle_steps=int(cfg.synapse_tiering_evict_idle_steps),
            grace_after_pagein_steps=int(cfg.synapse_tiering_grace_pagein_steps),
            ram_budget_bytes=int(getattr(cfg, "synapse_tiering_ram_budget_bytes", 0)),
        )

        # Mirror per-pathway CSRs (Strategy B = observational mirror only;
        # the monolithic cp_connections remains the source of truth for
        # the inference path).
        try:
            per_pathway = self.extract_per_pathway_csrs()
            for name, csr in per_pathway.items():
                self.synapse_store.add_pathway(name, csr)
            self._log_console(
                f"Synapse tiering enabled: {len(per_pathway)} pathways "
                f"mirrored to {root} (evict_idle={cfg.synapse_tiering_evict_idle_steps}, "
                f"grace={cfg.synapse_tiering_grace_pagein_steps})",
                "info",
            )
        except Exception as e:
            self._log_console(
                f"Synapse tiering init failed: {e}; tiering disabled "
                f"for this session.", "warning",
            )
            self.synapse_store = None

    def _detect_fired_pathways(self, fired_this_step) -> set:
        """Return the set of pathway names whose POST-region has fired
        in this simulation step.

        A pathway "fired" if any post-region neuron crossed firing
        threshold. Used to feed TieredSynapseStore.step() so the
        eviction policy can react to actual activity patterns.

        Args:
            fired_this_step: array (cupy or numpy) of bools, one per
                neuron, True if it fired this step.

        Returns:
            Set of pathway names. Empty set if no pathway fired.
        """
        if self.synapse_store is None or self.region_manager is None:
            return set()
        fired_pathways = set()
        # Pull fired_this_step to host once (cheap; we already pay this
        # cost in other diagnostics each step).
        try:
            fired_host = _backend_to_host(fired_this_step)
        except NameError:
            fired_host = (fired_this_step.get()
                            if hasattr(fired_this_step, "get")
                            else fired_this_step)
        import numpy as np
        for pw in self.region_manager.pathways():
            post_indices = np.array(
                list(self.region_manager.indices(pw.to_region)),
                dtype=np.int64,
            )
            if post_indices.size == 0:
                continue
            if bool(fired_host[post_indices].any()):
                pw_name = f"{pw.from_region}_to_{pw.to_region}"
                fired_pathways.add(pw_name)
        return fired_pathways

    def get_plasticity_gate_value(self, name: str) -> float:
        """Return the current plasticity gain for the named gate."""
        name = self._canonicalize_gate_name(name)
        if name not in self._plasticity_gate_values:
            raise KeyError(name)
        return self._plasticity_gate_values[name]

    def list_plasticity_gates(self) -> List[str]:
        """Return all plasticity gate names declared in the active wiring."""
        return list(self._plasticity_gate_to_synapses.keys())

    def plasticity_gate_synapse_count(self, name: str) -> int:
        """Return how many synapses are tagged with the named gate."""
        name = self._canonicalize_gate_name(name)
        if name not in self._plasticity_gate_to_synapses:
            raise KeyError(name)
        return len(self._plasticity_gate_to_synapses[name])

    def enable_stp_runtime(self) -> bool:
        """Re-enable Short-Term Plasticity at runtime after the bridge was
        built without it.

        Use case: train fast with STP-disabled (~3.3x faster, per 2026-05-10
        optimization arc), then re-enable STP for inference/eval so the
        biological dynamics (temporal filtering, gain control, adaptation,
        gamma stability) are restored. Combined with STP-off training
        this gives the best of both worlds: fast training + biologically
        realistic inference.

        Allocates cp_stp_x and cp_stp_u arrays if they don't exist (matches
        allocation pattern in _init_synapse_arrays_with_capacity). Sets
        cfg.enable_short_term_plasticity = True so subsequent
        _run_one_simulation_step calls execute STP dynamics.

        Returns True if STP was newly allocated, False if already active.

        Safe to call multiple times. To disable again, set
        cfg.enable_short_term_plasticity = False directly (existing arrays
        are kept but ignored by the step loop).

        2026-05-10: validates the user's "biological realism via re-enable
        at inference" hypothesis. See research/findings/
        2026-05-10-stp-default-flip.md for context.
        """
        cfg = self.core_config
        already_active = (
            cfg.enable_short_term_plasticity
            and self.cp_stp_x is not None
        )
        if already_active:
            return False
        cfg.enable_short_term_plasticity = True
        if self.cp_stp_x is None:
            # Match allocation pattern from _init_synapse_arrays_with_capacity
            if not hasattr(self, "cp_connections") or self.cp_connections is None:
                return False  # no connections yet
            capacity = self._synapse_capacity if hasattr(
                self, "_synapse_capacity"
            ) else int(self.cp_connections.data.size)
            self.cp_stp_x = cp.ones(capacity, dtype=cp.float32)
            self.cp_stp_u = cp.full(capacity, cfg.stp_U, dtype=cp.float32)
            return True
        return False

    def set_global_plasticity_gain(self, value: float) -> None:
        """Set the global per-synapse plasticity gain to `value` for ALL synapses.

        Convenience wrapper around `cp_plasticity_rate_gain` that:
        - Allocates the array (filled to `value`) if not yet allocated
        - Otherwise fills the entire existing array with `value`

        Use cases:
        - `set_global_plasticity_gain(0.0)`: freeze ALL plasticity globally
          (e.g. during reset_steps in training loops). Trace decay still
          happens; only weight UPDATES are zeroed.
        - `set_global_plasticity_gain(1.0)`: thaw all plasticity. Inverse.
        - Partial values (e.g. 0.1) for soft global modulation.

        2026-05-10: shipped as part of perf optimization #3 (skip plasticity
        during reset_steps; per-event training does ~50 quiet steps where
        plasticity ops should be no-ops anyway). Expected ~1.3-1.5×
        speedup on plasticity-heavy training.

        NOTE: this is GLOBAL — overrides any per-pathway gates set via
        `set_plasticity_gate(name, value)`. The named gates can be
        re-applied AFTER this call to restore per-pathway state, OR
        wrap reset blocks tightly so per-pathway gates are stable
        across the whole training event.
        """
        v = float(value)
        if self.cp_plasticity_rate_gain is None:
            # Lazy allocate. Use the actual NNZ from the connections matrix.
            if not hasattr(self, "cp_connections") or self.cp_connections is None:
                # No connections yet; nothing to gate
                return
            nnz = int(self.cp_connections.nnz)
            self.cp_plasticity_rate_gain = cp.full(nnz, v, dtype=cp.float32)
        else:
            self.cp_plasticity_rate_gain.fill(v)

    def get_global_plasticity_gain(self) -> float | None:
        """Return the current global gain if uniform; else None.

        Returns None if cp_plasticity_rate_gain has heterogeneous values
        (e.g. some pathways gated 0, others 1). Returns the uniform value
        otherwise. Used for testing + debugging.
        """
        if self.cp_plasticity_rate_gain is None:
            return None
        # Check if uniform (cheap GPU op)
        first = float(self.cp_plasticity_rate_gain[0])
        if bool(cp.all(self.cp_plasticity_rate_gain == first)):  # bool() works on cupy 0-d + numpy scalar
            return first
        return None

    def update_pruning(self, eligibility_trace, reward_signal, prunable_indices=None):
        """Structural-plasticity step. Updates survival scores based on
        reward-aligned eligibility, then prunes synapses meeting both
        criteria: survival < pruning_threshold AND weight < pruning_weight_floor.

        Pruned synapses get alive=False, weight=0. Forward dynamics + plasticity
        respect the alive mask via cp_plasticity_rate_gain[i] *= alive[i] (applied
        here as a side effect, since cp_plasticity_rate_gain is already used for
        plasticity gating).

        prunable_indices: optional cupy int64 array. If provided, only synapses
        in this set are eligible for pruning. Used by the runner to restrict
        pruning to cross-projection synapses only.

        See docs/plans/2026-04-28-structural-plasticity-design.md.
        """
        import cupy as cp
        if self.cp_synapse_alive is None:
            return  # not enabled
        # Update survival score for all synapses
        delta = self.core_config.pruning_alpha * eligibility_trace * float(reward_signal)
        self.cp_synapse_survival += delta.astype(cp.float32)
        # Pruning rule
        weights = self.cp_connections.data
        prune_mask = (
            (self.cp_synapse_survival < self.core_config.pruning_threshold) &
            (weights < self.core_config.pruning_weight_floor) &
            self.cp_synapse_alive
        )
        if prunable_indices is not None:
            # Restrict to the prunable set: zero out mask outside of it
            restricted = cp.zeros_like(prune_mask)
            restricted[prunable_indices] = prune_mask[prunable_indices]
            prune_mask = restricted
        # Apply: alive=False, weight=0, plasticity_gain=0
        self.cp_synapse_alive[prune_mask] = False
        weights[prune_mask] = 0.0
        if self.cp_plasticity_rate_gain is not None:
            self.cp_plasticity_rate_gain[prune_mask] = 0.0

    def start_simulation(self):
        """Starts or restarts the simulation (called by sim_thread)."""
        if not self.is_initialized:
            self._log_to_ui("Simulation not initialized. Attempting to initialize.", "warning")
            # Re-create the full config dict to attempt re-initialization
            full_config = self.get_current_simulation_configuration_dict()
            if not self.apply_simulation_configuration_core(full_config):
                self._log_to_ui("Initialization failed. Cannot start simulation.", "error"); return

        self.runtime_state.current_time_ms = 0.0
        self.runtime_state.current_time_step = 0
        
        self.runtime_state.is_running = True; self.runtime_state.is_paused = False
        self._log_to_ui(f"Simulation started. Duration: {self.core_config.total_simulation_time_ms} ms, Model: {self.core_config.neuron_model_type}, dt: {self.core_config.dt_ms} ms.", "info")
        if self.recording_file_handle:
            self._log_to_ui(f"Recording active, streaming to file: {self.recording_filepath}", "info")

    def stop_simulation(self):
        """Stops the currently running simulation (called by sim_thread)."""
        if self.runtime_state.is_running or self.runtime_state.is_paused:
            self.runtime_state.is_running = False; self.runtime_state.is_paused = False
            self._log_to_ui("Simulation stopped.", "info")

    def pause_simulation(self):
        """Pauses the running simulation (called by sim_thread)."""
        if self.runtime_state.is_running and not self.runtime_state.is_paused:
            self.runtime_state.is_paused = True
            self._log_to_ui("Simulation paused.", "info")

    def resume_simulation(self):
        """Resumes a paused simulation (called by sim_thread)."""
        if self.runtime_state.is_running and self.runtime_state.is_paused:
            self.runtime_state.is_paused = False
            self._log_to_ui("Simulation resumed.", "info")

    def toggle_pause_simulation(self):
        """Toggles the pause state of the simulation. Returns the new pause state.

        DEPRECATED: This method directly modifies shared state and has race condition risks.
        Prefer sending PAUSE/RESUME commands through ui_to_sim_queue instead.
        """
        import warnings
        warnings.warn(
            "toggle_pause_simulation() is deprecated due to race conditions. "
            "Use ui_to_sim_queue.put({'type': 'PAUSE'/'RESUME'}) instead.",
            DeprecationWarning,
            stacklevel=2
        )

        if not self.runtime_state.is_running:
            self._log_to_ui("Cannot toggle pause: Simulation is not running.", "warning")
            return self.runtime_state.is_paused

        # Route through command queue for thread safety (if queue is available)
        _ui_to_sim_queue = getattr(self, 'ui_to_sim_queue', None)
        if _ui_to_sim_queue:
            command = "PAUSE" if not self.runtime_state.is_paused else "RESUME"
            try:
                _ui_to_sim_queue.put_nowait({"type": command})
            except queue.Full:
                self._log_to_ui("Command queue full, cannot toggle pause.", "warning")
            return not self.runtime_state.is_paused  # Return expected state

        # Fallback for non-threaded use (legacy)
        self.runtime_state.is_paused = not self.runtime_state.is_paused
        action = "paused" if self.runtime_state.is_paused else "resumed"
        self._log_to_ui(f"Simulation {action}.", "info")
        return self.runtime_state.is_paused

    def set_simulation_speed_factor(self, factor):
        """Sets the simulation speed factor (called by sim_thread)."""
        self.runtime_state.simulation_speed_factor = max(0.01, factor)
        self._log_to_ui(f"Simulation speed factor set to {self.runtime_state.simulation_speed_factor:.2f}x", "info")

    def step_simulation(self, num_steps=1):
        """Advances the simulation by a specified number of steps (called by sim_thread)."""
        # if global_gui_state.get("is_playback_mode_active", False): # Check UI-managed state
        #     self._log_to_ui("Cannot step simulation during playback mode.", "warning"); return

        if not self.is_initialized:
            self._log_to_ui("Cannot step: Sim not initialized.", "warning"); return
        
        can_step_internally = (self.runtime_state.is_running and self.runtime_state.is_paused) or \
                              (not self.runtime_state.is_running)

        if not can_step_internally:
            self._log_to_ui("Sim must be running & paused, or stopped, to step via command.", "warning"); return

        self._log_console(f"Stepping simulation by {num_steps} steps.") 
        for _ in range(num_steps):
            if self.core_config.num_neurons > 0:
                if self.runtime_state.current_time_ms < self.core_config.total_simulation_time_ms:
                    self._run_one_simulation_step() 
                    self.runtime_state.current_time_ms += self.core_config.dt_ms
                    self.runtime_state.current_time_step += 1
                else:
                    self._log_to_ui("Cannot step: Total simulation time reached.", "info")
                    self.stop_simulation() 
                    if self.ui_queue: self.ui_queue.put({"type": "SIM_STOPPED_OR_ENDED", "reason": "Total time reached on step"})
                    break 
            else:
                self._log_console("No neurons to simulate in step.", "debug"); break
        
        latest_data = self.get_latest_simulation_data_for_gui(force_fetch=True)
        if self.ui_queue and latest_data:
            self.ui_queue.put({"type": "SIM_DATA_UPDATE", "data": latest_data})
        self._log_to_ui(f"Stepped sim by {num_steps} substeps. Current time: {self.runtime_state.current_time_ms:.3f} ms", "info")


    def _estimate_frame_size_bytes(self, skip_synaptic_data=None):
        """Estimates the size in bytes of a single recording frame.

        Args:
            skip_synaptic_data: If True, exclude synaptic arrays from estimate.
                               If None, uses gpu_config.recording_skip_synaptic_data.
        """
        if not self.is_initialized:
            return 0

        if skip_synaptic_data is None:
            skip_synaptic_data = self.gpu_config.recording_skip_synaptic_data

        total_bytes = 0
        # Dynamic arrays that change each frame (neuron state)
        arrays_to_check = [
            'cp_membrane_potential_v', 'cp_firing_states', 'cp_viz_activity_timers',
            'cp_conductance_g_e', 'cp_conductance_g_i', 'cp_recovery_variable_u',
            'cp_gating_variable_m', 'cp_gating_variable_h', 'cp_gating_variable_n',
            'cp_hh_m_current_activation', 'cp_hh_CaT_m', 'cp_hh_CaT_h', 'cp_hh_h_current_q', 'cp_hh_NaP_activation',
            'cp_adex_w', 'cp_ou_current'
        ]

        # Synaptic data is often 10-20x larger than neuron data
        if not skip_synaptic_data:
            if self.core_config.enable_hebbian_learning and self.cp_connections is not None:
                if self.cp_connections.data is not None:
                    total_bytes += self.cp_connections.data.nbytes

            if self.core_config.enable_short_term_plasticity:
                if self.cp_stp_u is not None:
                    total_bytes += self.cp_stp_u.nbytes
                if self.cp_stp_x is not None:
                    total_bytes += self.cp_stp_x.nbytes

        for attr_name in arrays_to_check:
            array_data = getattr(self, attr_name, None)
            if array_data is not None:
                total_bytes += array_data.nbytes

        # Add overhead for metadata
        total_bytes += 1024  # Small overhead for scalars
        return total_bytes
    
    def _check_gpu_recording_capacity(self, estimated_frames):
        """Checks if GPU has enough memory for estimated recording frames (uses gpu_config)."""
        frame_size = self._estimate_frame_size_bytes()
        required_memory = frame_size * estimated_frames
        
        mem_info = _backend_get_device_mem_info()
        free_memory, total_memory = mem_info
        
        # Use configured fraction of available memory for recording buffer
        available_for_recording = free_memory * self.gpu_config.max_recording_memory_fraction
        max_frames = int(available_for_recording / frame_size) if frame_size > 0 else 0
        
        self._log_console(f"Frame size: {frame_size/1e6:.1f}MB, Free GPU: {free_memory/1e9:.1f}GB, Max frames: {max_frames}")
        
        if required_memory > available_for_recording:
            self._log_to_ui(
                f"Warning: Recording {estimated_frames} frames needs {required_memory/1e9:.1f}GB, "
                f"but only {available_for_recording/1e9:.1f}GB available. Max {max_frames} frames.",
                "warning"
            )
            return False, max_frames

        return True, max_frames

    def _check_recording_memory_pressure(self):
        """Checks GPU and CPU memory usage during recording.

        Recording is allowed to overflow from GPU to CPU RAM. We only pause when
        BOTH GPU and CPU RAM exceed their respective limits, allowing maximum
        recording capacity before auto-pause.

        Returns:
            tuple: (is_critical, gpu_usage_pct, cpu_usage_pct, message)
        """
        # Check GPU memory
        try:
            mem_info = _backend_get_device_mem_info()
            free_memory, total_memory = mem_info
            gpu_used = total_memory - free_memory
            gpu_usage_pct = gpu_used / total_memory
        except Exception:
            gpu_usage_pct = 0.0

        # Check CPU memory (requires psutil)
        cpu_usage_pct = 0.0
        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                cpu_usage_pct = mem.percent / 100.0
            except Exception:
                pass

        # Determine if memory is critical
        # Only critical when BOTH GPU AND CPU RAM exceed their limits
        # This allows GPU to fill up and overflow into CPU RAM before pausing
        gpu_limit = self.gpu_config.recording_gpu_memory_limit
        cpu_limit = self.gpu_config.recording_cpu_memory_limit

        gpu_exceeded = gpu_usage_pct >= gpu_limit
        cpu_exceeded = cpu_usage_pct >= cpu_limit

        is_critical = False
        message = None

        if gpu_exceeded and cpu_exceeded:
            # Both limits exceeded - must pause to prevent crash
            is_critical = True
            message = (f"GPU ({gpu_usage_pct*100:.1f}%) and CPU RAM ({cpu_usage_pct*100:.1f}%) "
                      f"both exceed limits ({gpu_limit*100:.0f}%/{cpu_limit*100:.0f}%)")
        elif gpu_exceeded and not HAS_PSUTIL:
            # GPU full but can't check CPU - pause to be safe
            is_critical = True
            message = (f"GPU memory at {gpu_usage_pct*100:.1f}% (limit: {gpu_limit*100:.0f}%). "
                      f"Cannot monitor CPU RAM (psutil not installed).")

        return is_critical, gpu_usage_pct, cpu_usage_pct, message

    def _capture_initial_state_for_recording(self):
        """Captures the full initial state of the simulation for HDF5 recording."""
        if not self.is_initialized:
            self._log_console("Cannot capture initial state: Simulation not initialized.", "error")
            return None

        snapshot = {
            "start_time_ms": self.runtime_state.current_time_ms,
            "start_time_step": self.runtime_state.current_time_step
        }

        if self.cp_traits is not None: snapshot["cp_traits"] = _backend_to_host(self.cp_traits)
        if self.cp_neuron_positions_3d is not None: snapshot["cp_neuron_positions_3d"] = _backend_to_host(self.cp_neuron_positions_3d)

        if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
            for param in ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c_reset', 'd_increment']:
                attr_name = f"cp_izh_{param}"
                if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                    snapshot[attr_name] = _backend_to_host(getattr(self, attr_name))
        elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
            for param in ['C_m', 'g_Na_max', 'g_K_max', 'g_L', 'E_Na', 'E_K', 'E_L', 'v_peak']:
                attr_name = f"cp_hh_{param}"
                if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                    snapshot[attr_name] = _backend_to_host(getattr(self, attr_name))

        arrays_to_capture = [
            'cp_membrane_potential_v', 'cp_recovery_variable_u', 'cp_gating_variable_m',
            'cp_gating_variable_h', 'cp_gating_variable_n',
            'cp_hh_m_current_activation', 'cp_hh_CaT_m', 'cp_hh_CaT_h', 'cp_hh_h_current_q', 'cp_hh_NaP_activation',
            'cp_conductance_g_e',
            'cp_conductance_g_i', 'cp_external_input_current', 'cp_refractory_timers',
            'cp_viz_activity_timers', 'cp_neuron_firing_thresholds', 'cp_neuron_activity_ema',
            'cp_firing_states', 'cp_prev_firing_states',
            'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress',
            'cp_adex_w', 'cp_ou_current'
        ]
        for attr_name in arrays_to_capture:
            array_data = getattr(self, attr_name, None)
            if array_data is not None:
                snapshot[attr_name] = _backend_to_host(array_data)
            else: 
                snapshot[attr_name] = None 

        if self.cp_connections is not None:
            snapshot["connections_data"] = _backend_to_host(self.cp_connections.data) if self.cp_connections.data is not None else np.array([])
            snapshot["connections_indices"] = _backend_to_host(self.cp_connections.indices) if self.cp_connections.indices is not None else np.array([])
            snapshot["connections_indptr"] = _backend_to_host(self.cp_connections.indptr) if self.cp_connections.indptr is not None else np.array([])
            snapshot["connections_shape"] = self.cp_connections.shape 
        else: 
            snapshot["connections_data"] = np.array([]); snapshot["connections_indices"] = np.array([])
            snapshot["connections_indptr"] = np.array([]); snapshot["connections_shape"] = (0,0)

        # Save only active portion of pre-allocated STP arrays
        synapse_count = getattr(self, '_synapse_count', None)
        if self.cp_stp_u is not None:
            active_u = self.cp_stp_u[:synapse_count] if synapse_count else self.cp_stp_u
            snapshot["cp_stp_u"] = _backend_to_host(active_u)
        else: snapshot["cp_stp_u"] = None
        if self.cp_stp_x is not None:
            active_x = self.cp_stp_x[:synapse_count] if synapse_count else self.cp_stp_x
            snapshot["cp_stp_x"] = _backend_to_host(active_x)
        else: snapshot["cp_stp_x"] = None
        
        return snapshot

    def _get_compression_kwargs(self):
        """Returns HDF5 dataset compression kwargs based on gpu_config settings."""
        compression = self.gpu_config.recording_compression.lower()

        if compression == "lz4":
            if HAS_HDF5PLUGIN:
                return hdf5plugin.LZ4()
            else:
                self._log_console("LZ4 requested but hdf5plugin not installed. Falling back to gzip.", "warning")
                return {"compression": "gzip", "compression_opts": self.gpu_config.recording_compression_level}
        elif compression == "gzip":
            return {"compression": "gzip", "compression_opts": self.gpu_config.recording_compression_level}
        elif compression == "none":
            return {}
        else:
            self._log_console(f"Unknown compression '{compression}'. Using gzip.", "warning")
            return {"compression": "gzip", "compression_opts": 1}

    def _create_compressed_dataset(self, group, key, data):
        """Creates an HDF5 dataset with configured compression."""
        compression_kwargs = self._get_compression_kwargs()
        if isinstance(compression_kwargs, dict):
            group.create_dataset(key, data=data, **compression_kwargs)
        else:
            # hdf5plugin returns a filter object, use it directly
            group.create_dataset(key, data=data, **compression_kwargs)

    def _write_gpu_frames_to_disk(self):
        """Writes all buffered frames (GPU + CPU overflow) to disk with optimized compression.

        Features:
        - Handles both GPU (CuPy) and CPU (NumPy) frame buffers
        - Configurable compression (LZ4/GZIP/none)
        - Optional parallel compression using ThreadPoolExecutor
        - Progress reporting
        """
        gpu_frame_count = len(self.gpu_frame_buffer)
        cpu_frame_count = len(self.cpu_frame_buffer)
        total_frames = gpu_frame_count + cpu_frame_count

        if total_frames == 0:
            return  # No frames to write

        compression_type = self.gpu_config.recording_compression
        use_parallel = self.gpu_config.enable_parallel_compression and total_frames > 10

        self._log_to_ui(
            f"Writing {total_frames} frames to disk ({gpu_frame_count} GPU + {cpu_frame_count} CPU, "
            f"compression={compression_type}, parallel={use_parallel})...",
            "info"
        )
        start_time = time.time()

        try:
            frames_np = {}

            # Phase 1a: GPU→CPU transfer for GPU-buffered frames
            if gpu_frame_count > 0:
                self._log_console(f"Phase 1a: Transferring {gpu_frame_count} GPU frames to CPU...")
                transfer_start = time.time()
                sorted_gpu_indices = sorted(self.gpu_frame_buffer.keys())

                for i, frame_idx in enumerate(sorted_gpu_indices):
                    frame_data_gpu = self.gpu_frame_buffer[frame_idx]
                    frame_data_np = {}
                    for key, value in frame_data_gpu.items():
                        if isinstance(value, cp.ndarray):
                            frame_data_np[key] = _backend_to_host(value)
                        else:
                            frame_data_np[key] = value
                    frames_np[frame_idx] = frame_data_np

                    # Progress every 20%
                    if (i + 1) % max(1, gpu_frame_count // 5) == 0:
                        self._log_console(f"  GPU→CPU transfer: {((i+1)/gpu_frame_count)*100:.0f}%")

                transfer_elapsed = time.time() - transfer_start
                self._log_console(f"GPU→CPU transfer completed in {transfer_elapsed:.2f}s")
            else:
                transfer_elapsed = 0.0

            # Phase 1b: Add CPU-buffered frames (already NumPy)
            if cpu_frame_count > 0:
                self._log_console(f"Phase 1b: Adding {cpu_frame_count} CPU-buffered frames...")
                for frame_idx, frame_data in self.cpu_frame_buffer.items():
                    frames_np[frame_idx] = frame_data

            # Phase 2: Write to HDF5 (with optional parallel compression)
            self._log_console(f"Phase 2: Compressing and writing {total_frames} frames to disk...")
            write_start = time.time()

            compression_kwargs = self._get_compression_kwargs()
            write_lock = threading.Lock()
            completed_count = [0]  # Use list for mutable reference in nested function

            def write_single_frame(frame_idx, frame_data):
                """Write a single frame to HDF5 (thread-safe)."""
                frame_group_name = f"frames/frame_{frame_idx}"

                with write_lock:
                    current_frame_group = self.recording_file_handle.create_group(frame_group_name)

                    for key, value in frame_data.items():
                        if isinstance(value, np.ndarray):
                            if value.size > 0:
                                if isinstance(compression_kwargs, dict):
                                    current_frame_group.create_dataset(key, data=value, **compression_kwargs)
                                else:
                                    current_frame_group.create_dataset(key, data=value, **compression_kwargs)
                            else:
                                current_frame_group.attrs[f"{key}_is_empty"] = True
                        elif value is not None:
                            current_frame_group.attrs[key] = value
                        else:
                            current_frame_group.attrs[key] = "NoneType"

                    completed_count[0] += 1
                    if completed_count[0] % max(1, total_frames // 10) == 0:
                        self._log_console(f"  Write progress: {(completed_count[0]/total_frames)*100:.0f}%")

                return frame_idx

            sorted_all_indices = sorted(frames_np.keys())

            if use_parallel:
                # Parallel compression (HDF5 writes still serialized via lock)
                num_workers = min(self.gpu_config.parallel_compression_workers, os.cpu_count() or 4)
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(write_single_frame, idx, frames_np[idx]): idx
                        for idx in sorted_all_indices
                    }
                    # Wait for all to complete
                    for future in as_completed(futures):
                        try:
                            future.result()  # Raises exception if frame write failed
                        except Exception as e:
                            self._log_console(f"Error writing frame: {e}", "error")
            else:
                # Sequential write
                for frame_idx in sorted_all_indices:
                    write_single_frame(frame_idx, frames_np[frame_idx])

            write_elapsed = time.time() - write_start

            # Final flush
            self.recording_file_handle.flush()

            elapsed = time.time() - start_time
            frames_per_sec = total_frames / elapsed if elapsed > 0 else 0
            self._log_to_ui(
                f"Successfully wrote {total_frames} frames in {elapsed:.2f}s "
                f"({frames_per_sec:.1f} frames/s, transfer={transfer_elapsed:.1f}s, write={write_elapsed:.1f}s)",
                "success"
            )

            # Clear both buffers to free memory
            self.gpu_frame_buffer.clear()
            self.cpu_frame_buffer.clear()

        except Exception as e:
            self._log_to_ui(f"Error writing frames to disk: {e}", "error")
            raise

    def _async_streaming_writer_thread(self):
        """Background thread for writing recording frames to disk asynchronously.

        This prevents the simulation from stalling while waiting for disk I/O,
        which is critical for network storage or large recordings.
        """
        compression_kwargs = self._get_compression_kwargs()
        batch_size = self.gpu_config.streaming_write_batch_size
        pending_frames = []
        last_log_time = time.time()
        log_interval = 5.0  # Log progress every 5 seconds

        self._log_console("Async streaming writer thread started.")

        while not self.streaming_writer_stop_event.is_set() or not self.streaming_write_queue.empty():
            try:
                # Get frame from queue with timeout to allow periodic checks
                try:
                    frame_idx, frame_data_np = self.streaming_write_queue.get(timeout=0.1)
                    pending_frames.append((frame_idx, frame_data_np))
                except queue.Empty:
                    pass

                # Write batch when we have enough or when stopping
                should_flush = (
                    len(pending_frames) >= batch_size or
                    (self.streaming_writer_stop_event.is_set() and pending_frames)
                )

                if should_flush and self.recording_file_handle and pending_frames:
                    for fidx, fdata in pending_frames:
                        try:
                            frame_group_name = f"frames/frame_{fidx}"
                            current_frame_group = self.recording_file_handle.create_group(frame_group_name)

                            for key, value in fdata.items():
                                if isinstance(value, np.ndarray):
                                    if value.size > 0:
                                        current_frame_group.create_dataset(key, data=value, **compression_kwargs)
                                    else:
                                        current_frame_group.attrs[f"{key}_is_empty"] = True
                                elif value is not None:
                                    current_frame_group.attrs[key] = value
                                else:
                                    current_frame_group.attrs[key] = "NoneType"

                            self.streaming_frames_written += 1
                        except Exception as e:
                            self._log_console(f"Error writing frame {fidx}: {e}", "error")

                    # Flush to disk periodically
                    try:
                        self.recording_file_handle.flush()
                    except Exception:
                        pass

                    pending_frames.clear()

                    # Log progress periodically
                    now = time.time()
                    if now - last_log_time >= log_interval:
                        queued = self.streaming_frames_queued
                        written = self.streaming_frames_written
                        pending = queued - written
                        self._log_console(
                            f"Streaming write progress: {written} frames written, {pending} pending in queue"
                        )
                        last_log_time = now

            except Exception as e:
                self._log_console(f"Error in async streaming writer: {e}", "error")
                time.sleep(0.1)

        self._log_console(f"Async streaming writer thread finished. Total frames written: {self.streaming_frames_written}")

    def start_recording_to_file(self, filepath):
        """Starts recording the simulation state to an HDF5 file (called by sim_thread)."""
        if self.recording_file_handle: 
            self._log_to_ui("Error: Recording already active. Stop current one first.", "error")
            return False 

        if not self.is_initialized:
            self._log_to_ui("Simulation not initialized. Cannot start recording.", "warning")
            return False
        
        self._log_console(f"Attempting to start new recording to file: {filepath}")
        try:
            self.recording_filepath = filepath
            self.recording_file_handle = h5py.File(self.recording_filepath, 'w') 
            self.current_frame_count_for_h5 = 0 

            self.recording_file_handle.attrs["format_version"] = RECORDING_FORMAT_VERSION
            # Save complete config structure (core_config, viz_config, runtime_state)
            config_snapshot = _get_full_config_dict(self.core_config, self.viz_config, self.runtime_state)
            # Flatten nested structure for HDF5 attrs by prefixing keys
            flattened_config = {}
            for section_name, section_dict in config_snapshot.items():
                for key, value in section_dict.items():
                    flattened_config[f"{section_name}.{key}"] = value
            save_dict_to_hdf5_attrs(self.recording_file_handle, flattened_config)

            initial_state_data = self._capture_initial_state_for_recording()
            if initial_state_data is None:
                self._log_to_ui("Failed to capture initial state for recording. Aborting.", "error")
                self.recording_file_handle.close()
                self.recording_file_handle = None; self.recording_filepath = None
                return False

            initial_state_group = self.recording_file_handle.create_group("initial_state")
            compression_kwargs = self._get_compression_kwargs()
            for key, value in initial_state_data.items():
                if isinstance(value, np.ndarray):
                    if value.size > 0:
                        if isinstance(compression_kwargs, dict):
                            initial_state_group.create_dataset(key, data=value, **compression_kwargs)
                        else:
                            initial_state_group.create_dataset(key, data=value, **compression_kwargs)
                    else:
                        initial_state_group.attrs[f"{key}_is_empty"] = True
                elif key == "connections_shape":
                    initial_state_group.attrs["connections_shape_0"] = value[0]
                    initial_state_group.attrs["connections_shape_1"] = value[1]
                elif value is not None:
                    initial_state_group.attrs[key] = value
                else:
                    initial_state_group.attrs[key] = "NoneType"

            # Store compression type and recording options for playback compatibility
            self.recording_file_handle.attrs["compression_type"] = self.gpu_config.recording_compression
            self.recording_file_handle.attrs["recording_skip_synaptic_data"] = self.gpu_config.recording_skip_synaptic_data
            self.recording_file_handle.attrs["recording_frame_skip"] = self.gpu_config.recording_frame_skip

            self.recording_file_handle.create_group("frames")

            # Estimate frames based on simulation duration and frame skip
            frame_skip = max(1, self.gpu_config.recording_frame_skip)
            estimated_frames = int(self.core_config.total_simulation_time_ms / self.core_config.dt_ms) // frame_skip
            frame_size = self._estimate_frame_size_bytes()

            # Log frame size info for large recordings
            if self.gpu_config.recording_skip_synaptic_data:
                full_frame_size = self._estimate_frame_size_bytes(skip_synaptic_data=False)
                reduction = (1 - frame_size / full_frame_size) * 100 if full_frame_size > 0 else 0
                self._log_console(
                    f"Frame size: {frame_size/1e6:.1f}MB (neuron-only, {reduction:.0f}% smaller than full {full_frame_size/1e6:.1f}MB)"
                )
            else:
                self._log_console(f"Frame size: {frame_size/1e6:.1f}MB")

            if frame_skip > 1:
                self._log_console(f"Recording every {frame_skip}th frame ({estimated_frames} frames for {self.core_config.total_simulation_time_ms:.0f}ms)")

            # Determine recording mode
            recording_mode = self.gpu_config.recording_mode

            if recording_mode == "streaming":
                # Streaming mode: write frames to disk immediately via background thread
                self.gpu_frame_buffer = {}
                self.cpu_frame_buffer = {}
                self.streaming_frames_written = 0
                self.streaming_frames_queued = 0

                # Clear the queue
                while not self.streaming_write_queue.empty():
                    try:
                        self.streaming_write_queue.get_nowait()
                    except queue.Empty:
                        break

                # Start async writer thread if enabled
                if self.gpu_config.streaming_async_write:
                    self.streaming_writer_stop_event.clear()
                    self.streaming_writer_thread = threading.Thread(
                        target=self._async_streaming_writer_thread,
                        name="StreamingRecordWriter",
                        daemon=True
                    )
                    self.streaming_writer_thread.start()
                    self._log_console("Streaming recording mode with async writer enabled.")
                else:
                    self._log_console("Streaming recording mode (synchronous writes).")

                self._log_to_ui(f"Recording armed (streaming to disk). Start sim to capture.", "info", color=[0,150,200])

            else:
                # GPU-buffered mode (default): buffer in memory, write at end
                can_gpu_buffer, max_gpu_frames = self._check_gpu_recording_capacity(estimated_frames)

                self.gpu_frame_buffer = {}  # Clear any old GPU frames
                self.cpu_frame_buffer = {}  # Clear any old CPU overflow frames
                self.recording_overflow_to_cpu = False  # Reset overflow state
                self.gpu_recording_max_frames = max_gpu_frames
                self._log_console(f"GPU-buffered recording enabled. Max GPU frames: {max_gpu_frames} (will overflow to CPU RAM if needed)")
                self._log_to_ui(f"Recording armed (GPU-buffered). Start sim to capture.", "info", color=[0,150,200])
            # Signal UI that recording has started successfully
            if self.ui_queue:
                self.ui_queue.put({"type": "RECORDING_STARTED", "filepath": self.recording_filepath})
            return True

        except Exception as e:
            self._log_to_ui(f"Error starting file recording to {filepath}: {e}", "error")
            if self.recording_file_handle:
                try: self.recording_file_handle.close()
                except: pass
            self.recording_file_handle = None; self.recording_filepath = None
            if self.ui_queue:
                self.ui_queue.put({"type": "RECORDING_START_FAILED", "error": str(e)})
            return False

    def stop_recording(self):
        """Stops the HDF5 recording stream and finalizes the file (called by sim_thread)."""
        if not self.recording_file_handle:
            self._log_to_ui("No active recording to stop.", "info")
            if self.ui_queue:
                self.ui_queue.put({"type": "RECORDING_STOPPED_UNEXPECTEDLY"})
            return

        self._log_console("Stopping HDF5 recording stream.")
        was_recording_to_file = False
        finalized_filepath = self.recording_filepath

        if self.recording_file_handle and isinstance(self.recording_file_handle, h5py.File) and self.recording_file_handle.id:
            try:
                # Handle streaming mode: wait for async writer to finish
                if self.gpu_config.recording_mode == "streaming":
                    if self.streaming_writer_thread and self.streaming_writer_thread.is_alive():
                        pending = self.streaming_frames_queued - self.streaming_frames_written
                        if pending > 0:
                            self._log_to_ui(
                                f"Waiting for {pending} frames to be written to disk...",
                                "info"
                            )
                        # Signal the writer thread to stop after draining queue
                        self.streaming_writer_stop_event.set()
                        # Wait for thread to finish (with timeout to avoid infinite hang)
                        self.streaming_writer_thread.join(timeout=300)  # 5 minute timeout
                        if self.streaming_writer_thread.is_alive():
                            self._log_to_ui("Warning: Streaming writer thread did not finish in time.", "warning")
                        self.streaming_writer_thread = None

                    self._log_console(
                        f"Streaming recording complete: {self.streaming_frames_written} frames written to disk."
                    )
                else:
                    # GPU-buffered mode: write buffered frames to disk
                    has_buffered_frames = self.gpu_frame_buffer or self.cpu_frame_buffer
                    if has_buffered_frames:
                        self._write_gpu_frames_to_disk()

                # Final flush before closing
                self.recording_file_handle.flush()
                self.recording_file_handle.close()
                was_recording_to_file = True
                self._log_to_ui(f"Recording stream to {finalized_filepath} finalized and saved.", "success")
            except Exception as e:
                self._log_to_ui(f"Error finalizing recording file {finalized_filepath}: {e}", "error")
        else:
            self._log_console(f"Stop recording called, but no active file handle or already closed for {finalized_filepath}.", "warning")

        # Reset all recording state
        self.recording_file_handle = None
        self.recording_filepath = None
        self.current_frame_count_for_h5 = 0
        self.gpu_frame_buffer.clear()
        self.cpu_frame_buffer.clear()
        self.recording_overflow_to_cpu = False
        self.streaming_frames_written = 0
        self.streaming_frames_queued = 0

        if self.ui_queue:
            self.ui_queue.put({
                "type": "RECORDING_FINALIZED",
                "success": was_recording_to_file,
                "filepath": finalized_filepath if was_recording_to_file else None
            })

    def _capture_frame_as_numpy(self, skip_synaptic_data=False):
        """Captures current simulation state as NumPy arrays for recording.

        Args:
            skip_synaptic_data: If True, exclude connection weights and STP arrays
                               (reduces frame size by 10-20x for large networks).
        Returns:
            dict: Frame data with NumPy arrays ready for HDF5 storage.
        """
        frame_data = {
            "time_ms": self.runtime_state.current_time_ms,
            "step": self.runtime_state.current_time_step,
            "_mock_num_spikes_this_step": self._mock_num_spikes_this_step,
            "_mock_network_avg_firing_rate_hz": self._mock_network_avg_firing_rate_hz,
            "_mock_total_plasticity_events": self._mock_total_plasticity_events
        }

        # Build list of neuron state arrays to capture
        dynamic_arrays = [
            'cp_membrane_potential_v', 'cp_firing_states', 'cp_viz_activity_timers',
            'cp_conductance_g_e', 'cp_conductance_g_i',
            'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress'
        ]

        if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
            dynamic_arrays.extend(['cp_recovery_variable_u'])
            if self.core_config.enable_homeostasis and self.cp_neuron_firing_thresholds is not None:
                dynamic_arrays.append('cp_neuron_firing_thresholds')
        elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
            dynamic_arrays.extend([
                'cp_gating_variable_m', 'cp_gating_variable_h', 'cp_gating_variable_n',
                'cp_hh_m_current_activation', 'cp_hh_CaT_m', 'cp_hh_CaT_h',
                'cp_hh_h_current_q', 'cp_hh_NaP_activation'
            ])
        elif self.core_config.neuron_model_type == NeuronModel.ADEX.name:
            dynamic_arrays.extend(['cp_adex_w'])

        # Capture neuron state arrays (GPU → CPU transfer)
        for attr_name in dynamic_arrays:
            array_data = getattr(self, attr_name, None)
            if array_data is not None:
                frame_data[attr_name] = _backend_to_host(array_data)
            else:
                frame_data[attr_name] = None

        # Capture synaptic data (optional - this is the large part)
        if not skip_synaptic_data:
            if self.core_config.enable_hebbian_learning and self.cp_connections is not None:
                if self.cp_connections.data is not None:
                    frame_data["cp_connections_data"] = _backend_to_host(self.cp_connections.data)

            if self.core_config.enable_short_term_plasticity:
                synapse_count = getattr(self, '_synapse_count', None)
                if self.cp_stp_u is not None:
                    frame_data["cp_stp_u"] = _backend_to_host(
                        self.cp_stp_u[:synapse_count] if synapse_count else self.cp_stp_u
                    )
                if self.cp_stp_x is not None:
                    frame_data["cp_stp_x"] = _backend_to_host(
                        self.cp_stp_x[:synapse_count] if synapse_count else self.cp_stp_x
                    )

        return frame_data

    def record_current_frame_if_active(self):
        """Records the current simulation state as a frame if recording is active (called by sim_thread)."""
        if not self.recording_file_handle or \
           not isinstance(self.recording_file_handle, h5py.File) or \
           not self.recording_file_handle.id or \
           not self.runtime_state.is_running or \
           self.runtime_state.is_paused:
            return

        try:
            # Frame skip: only record every Nth simulation step
            frame_skip = max(1, self.gpu_config.recording_frame_skip)
            if self.runtime_state.current_time_step % frame_skip != 0:
                return

            frame_idx = self.current_frame_count_for_h5
            skip_synaptic = self.gpu_config.recording_skip_synaptic_data
            recording_mode = self.gpu_config.recording_mode

            # Streaming mode: queue frames for async disk writes
            if recording_mode == "streaming":
                frame_data_np = self._capture_frame_as_numpy(skip_synaptic)

                if self.gpu_config.streaming_async_write:
                    # Queue for background writer thread
                    self.streaming_write_queue.put((frame_idx, frame_data_np))
                    self.streaming_frames_queued += 1

                    # Periodic logging
                    if frame_idx % 500 == 0:
                        pending = self.streaming_frames_queued - self.streaming_frames_written
                        self._log_console(f"Streaming recording: frame {frame_idx} queued, {pending} pending write")
                else:
                    # Synchronous write (slower, blocks simulation)
                    compression_kwargs = self._get_compression_kwargs()
                    frame_group_name = f"frames/frame_{frame_idx}"
                    current_frame_group = self.recording_file_handle.create_group(frame_group_name)

                    for key, value in frame_data_np.items():
                        if isinstance(value, np.ndarray):
                            if value.size > 0:
                                current_frame_group.create_dataset(key, data=value, **compression_kwargs)
                            else:
                                current_frame_group.attrs[f"{key}_is_empty"] = True
                        elif value is not None:
                            current_frame_group.attrs[key] = value
                        else:
                            current_frame_group.attrs[key] = "NoneType"

                    # Flush periodically
                    if frame_idx % self.gpu_config.streaming_write_batch_size == 0:
                        self.recording_file_handle.flush()

                self.current_frame_count_for_h5 += 1
                return

            # GPU-buffered recording with CPU overflow support
            if recording_mode == "gpu_buffered":

                # Check memory BEFORE storing to decide where to put this frame
                check_interval = self.gpu_config.recording_memory_check_interval
                gpu_pct = 0.0
                cpu_pct = 0.0

                if frame_idx % check_interval == 0:
                    is_critical, gpu_pct, cpu_pct, warning_msg = self._check_recording_memory_pressure()

                    # Check if we need to switch to CPU overflow mode
                    gpu_limit = self.gpu_config.recording_gpu_memory_limit
                    if not self.recording_overflow_to_cpu and gpu_pct >= gpu_limit:
                        self.recording_overflow_to_cpu = True
                        self._log_to_ui(
                            f"GPU memory at {gpu_pct*100:.1f}%. Switching to CPU RAM for new frames.",
                            "warning"
                        )
                        self._log_console(
                            f"RECORDING OVERFLOW: GPU {gpu_pct*100:.1f}% >= {gpu_limit*100:.0f}% limit. "
                            f"Frame {frame_idx}+ will be stored in CPU RAM."
                        )

                    # Check for critical memory (both GPU AND CPU full)
                    if is_critical and self.gpu_config.recording_auto_pause_on_memory:
                        self.runtime_state.is_paused = True
                        gpu_frames = len(self.gpu_frame_buffer)
                        cpu_frames = len(self.cpu_frame_buffer)
                        self._log_to_ui(
                            f"RECORDING PAUSED: {warning_msg}. "
                            f"Recorded {frame_idx} frames ({gpu_frames} GPU + {cpu_frames} CPU). "
                            f"Finalize recording now to prevent data loss.",
                            "warning"
                        )
                        self._log_console(
                            f"MEMORY CRITICAL - Auto-paused at frame {frame_idx}. "
                            f"GPU: {gpu_pct*100:.1f}%, CPU: {cpu_pct*100:.1f}%"
                        )
                        if self.ui_queue:
                            self.ui_queue.put({
                                "type": "RECORDING_MEMORY_CRITICAL",
                                "frame_count": frame_idx,
                                "gpu_frames": gpu_frames,
                                "cpu_frames": cpu_frames,
                                "gpu_usage_pct": gpu_pct,
                                "cpu_usage_pct": cpu_pct,
                                "message": warning_msg,
                                "suggestion": "Finalize recording now to save data before memory exhaustion."
                            })
                        return

                    # Periodic logging
                    if frame_idx % (check_interval * 10) == 0:
                        storage_mode = "CPU" if self.recording_overflow_to_cpu else "GPU"
                        self._log_console(
                            f"Recording frame {frame_idx}: GPU {gpu_pct*100:.1f}%, CPU {cpu_pct*100:.1f}% [{storage_mode}]"
                        )

                # Build list of arrays to capture
                dynamic_arrays_to_capture = [
                    'cp_membrane_potential_v', 'cp_firing_states', 'cp_viz_activity_timers',
                    'cp_conductance_g_e', 'cp_conductance_g_i',
                    'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress'
                ]
                if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                    dynamic_arrays_to_capture.extend(['cp_recovery_variable_u'])
                    if self.core_config.enable_homeostasis and self.cp_neuron_firing_thresholds is not None:
                        dynamic_arrays_to_capture.append('cp_neuron_firing_thresholds')
                elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                    dynamic_arrays_to_capture.extend([
                        'cp_gating_variable_m', 'cp_gating_variable_h', 'cp_gating_variable_n',
                        'cp_hh_m_current_activation', 'cp_hh_CaT_m', 'cp_hh_CaT_h', 'cp_hh_h_current_q', 'cp_hh_NaP_activation'
                    ])
                elif self.core_config.neuron_model_type == NeuronModel.ADEX.name:
                    dynamic_arrays_to_capture.extend(['cp_adex_w'])

                # Store frame data - either GPU (CuPy) or CPU (NumPy) depending on overflow state
                if self.recording_overflow_to_cpu:
                    # CPU overflow mode: store as NumPy arrays
                    frame_data = {
                        "time_ms": self.runtime_state.current_time_ms,
                        "step": self.runtime_state.current_time_step,
                        "_mock_num_spikes_this_step": self._mock_num_spikes_this_step,
                        "_mock_network_avg_firing_rate_hz": self._mock_network_avg_firing_rate_hz,
                        "_mock_total_plasticity_events": self._mock_total_plasticity_events
                    }

                    # Synaptic data (optional - skip for large recordings)
                    if not skip_synaptic:
                        if self.core_config.enable_hebbian_learning and self.cp_connections is not None:
                            if self.cp_connections.data is not None:
                                frame_data["cp_connections_data"] = _backend_to_host(self.cp_connections.data)

                        if self.core_config.enable_short_term_plasticity:
                            synapse_count = getattr(self, '_synapse_count', None)
                            if self.cp_stp_u is not None:
                                frame_data["cp_stp_u"] = _backend_to_host(self.cp_stp_u[:synapse_count] if synapse_count else self.cp_stp_u)
                            if self.cp_stp_x is not None:
                                frame_data["cp_stp_x"] = _backend_to_host(self.cp_stp_x[:synapse_count] if synapse_count else self.cp_stp_x)

                    for attr_name in dynamic_arrays_to_capture:
                        array_data = getattr(self, attr_name, None)
                        if array_data is not None:
                            frame_data[attr_name] = _backend_to_host(array_data)  # GPU→CPU transfer
                        else:
                            frame_data[attr_name] = None

                    self.cpu_frame_buffer[frame_idx] = frame_data

                else:
                    # GPU mode: store as CuPy arrays (fast, no transfer)
                    frame_data = {
                        "time_ms": self.runtime_state.current_time_ms,
                        "step": self.runtime_state.current_time_step,
                        "_mock_num_spikes_this_step": self._mock_num_spikes_this_step,
                        "_mock_network_avg_firing_rate_hz": self._mock_network_avg_firing_rate_hz,
                        "_mock_total_plasticity_events": self._mock_total_plasticity_events
                    }

                    # Synaptic data (optional - skip for large recordings)
                    if not skip_synaptic:
                        if self.core_config.enable_hebbian_learning and self.cp_connections is not None:
                            if self.cp_connections.data is not None:
                                frame_data["cp_connections_data"] = self.cp_connections.data.copy()

                        if self.core_config.enable_short_term_plasticity:
                            synapse_count = getattr(self, '_synapse_count', None)
                            if self.cp_stp_u is not None:
                                frame_data["cp_stp_u"] = self.cp_stp_u[:synapse_count].copy() if synapse_count else self.cp_stp_u.copy()
                            if self.cp_stp_x is not None:
                                frame_data["cp_stp_x"] = self.cp_stp_x[:synapse_count].copy() if synapse_count else self.cp_stp_x.copy()

                    for attr_name in dynamic_arrays_to_capture:
                        array_data = getattr(self, attr_name, None)
                        if array_data is not None:
                            frame_data[attr_name] = array_data.copy()  # CuPy copy (stays on GPU)
                        else:
                            frame_data[attr_name] = None

                    self.gpu_frame_buffer[frame_idx] = frame_data

            else:
                # Legacy CPU path: immediate streaming to HDF5
                # Use the helper function for consistency
                frame_data_np = self._capture_frame_as_numpy(skip_synaptic)

                frame_group_name = f"frames/frame_{frame_idx}"
                current_frame_group = self.recording_file_handle.create_group(frame_group_name)

                # Use configured compression settings instead of hardcoded gzip
                compression_kwargs = self._get_compression_kwargs()

                for key, value in frame_data_np.items():
                    if isinstance(value, np.ndarray):
                        if value.size > 0:
                            if isinstance(compression_kwargs, dict):
                                current_frame_group.create_dataset(key, data=value, **compression_kwargs)
                            else:
                                # hdf5plugin returns a filter object
                                current_frame_group.create_dataset(key, data=value, **compression_kwargs)
                        else:
                            current_frame_group.attrs[f"{key}_is_empty"] = True
                    elif value is not None:
                        current_frame_group.attrs[key] = value
                    else:
                        current_frame_group.attrs[key] = "NoneType"
                
                # Batch frames: only flush periodically for better performance
                if frame_idx % self.recording_buffer_size == 0:
                    self.recording_file_handle.flush()
            
            self.current_frame_count_for_h5 += 1

        except Exception as e:
            self._log_to_ui(f"Error streaming frame to recording file {self.recording_filepath}: {e}", "error")
            self.stop_recording() 
            # UI thread will update global_gui_state based on RECORDING_FINALIZED message.

    def _prepare_loaded_recording_metadata(self, filepath):
        """Opens HDF5 and prepares metadata for playback (called by sim_thread)."""
        self._log_console(f"Preparing metadata for recording file: {filepath}")
        try:
            h5_file = h5py.File(filepath, 'r') 

            version_info_str = h5_file.attrs.get("format_version")
            if version_info_str != RECORDING_FORMAT_VERSION:
                self._log_to_ui(f"Invalid/outdated recording format. Expected {RECORDING_FORMAT_VERSION}, got {version_info_str}.", "error")
                h5_file.close()
                return None

            flattened_config = load_dict_from_hdf5_attrs(h5_file) 
            
            # Reconstruct nested config structure from flattened keys
            config_snapshot = {"core_config": {}, "viz_config": {}, "runtime_state": {}}
            for key, value in flattened_config.items():
                if "." in key:
                    section, field = key.split(".", 1)
                    if section in config_snapshot:
                        config_snapshot[section][field] = value
                else:
                    # Legacy format or keys without section prefix
                    config_snapshot["core_config"][key] = value
            
            # Check if we have the expected config structure
            if not config_snapshot.get("core_config") or "num_neurons" not in config_snapshot["core_config"]: 
                self._log_to_ui("Recording metadata missing or invalid config. Cannot load.", "error")
                h5_file.close(); return None

            initial_full_state_metadata = {} 
            initial_state_group = h5_file.get("initial_state")
            if not initial_state_group:
                self._log_to_ui("Invalid recording: 'initial_state' group missing.", "error"); h5_file.close(); return None

            for key, value in initial_state_group.attrs.items():
                if value == "NoneType": initial_full_state_metadata[key] = None
                else: initial_full_state_metadata[key] = value
            initial_full_state_metadata["datasets_present"] = list(initial_state_group.keys())

            num_frames = 0
            frames_group = h5_file.get("frames")
            if frames_group:
                num_frames = len(list(frames_group.keys())) 

            loaded_data_package = {
                "filepath": filepath,
                "h5_file_obj_for_playback": h5_file, 
                "config_snapshot": config_snapshot,
                "initial_state_metadata": initial_full_state_metadata, 
                "num_frames": num_frames
            }
            self._log_console(f"Successfully prepared metadata for {os.path.basename(filepath)}. Frames: {num_frames}", "info")
            return loaded_data_package

        except FileNotFoundError:
            self._log_to_ui(f"Error: Recording file not found at {filepath}", "error")
            return None
        except Exception as e:
            self._log_to_ui(f"Critical error preparing metadata for recording {filepath}: {e}", "error")
            import traceback; traceback.print_exc()
            if 'h5_file' in locals() and h5_file.id: h5_file.close() 
            return None

    def load_recording(self, filepath, stream_only=False):
        """Loads a recording for playback (called by sim_thread).

        Args:
            filepath: Path to the .simrec.h5 file
            stream_only: If True, skip GPU caching and stream all frames from disk
        """
        mode_str = "streaming" if stream_only else "caching"
        self._log_to_ui(f"Loading recording ({mode_str} mode) from {filepath}...", "info")

        if self.runtime_state.is_running: self.stop_simulation()
        if self.recording_file_handle:
            self._log_console("load_recording: Closing an existing recording file before loading new one.", "warning")
            try: self.recording_file_handle.close()
            except: pass
            self.recording_file_handle = None; self.recording_filepath = None

        # Close any HDF5 file this sim_bridge instance might be holding for playback itself.
        # Note: The main HDF5 handle for playback is managed by UI thread via global_gui_state.active_recording_data_source.
        # This method is for the sim_thread to initially process the file.
        # If sim_bridge was designed to hold its own playback handle, it would close it here.

        prepared_metadata = self._prepare_loaded_recording_metadata(filepath)

        if prepared_metadata:
            h5_file = prepared_metadata["h5_file_obj_for_playback"]
            num_frames = prepared_metadata["num_frames"]

            if stream_only:
                # Streaming mode: skip GPU caching entirely, clear any existing cache
                self.gpu_playback_cache.clear()
                self._log_to_ui(f"Streaming mode: {num_frames} frames will be read from disk during playback.", "info")
            elif num_frames > 0:
                # Caching mode: attempt to load recording into GPU cache
                success = self._load_recording_to_gpu_cache(h5_file, num_frames)
                if not success:
                    self._log_to_ui("Warning: GPU cache loading failed. Playback will use slower disk I/O.", "warning")

            if self.ui_queue:
                self.ui_queue.put({
                    "type": "RECORDING_METADATA_PREPARED",
                    "data": prepared_metadata,
                    "stream_only": stream_only
                })
                self._log_to_ui(f"Recording metadata for '{os.path.basename(filepath)}' prepared. UI can now initialize playback.", "info")
            return True
        else:
            if self.ui_queue:
                 self.ui_queue.put({"type": "RECORDING_LOAD_FAILED", "filepath": filepath})
            return False
    def _apply_config_and_initial_state_from_recording(self, config_snapshot, initial_state_h5_group):
        """
        Applies the configuration and initial state from a loaded HDF5 recording.
        This is called by the simulation thread when commanded by the UI thread after metadata is prepared.
        `initial_state_h5_group` is an open h5py.Group object for "initial_state".
        """
        self._log_console("Applying config and initial state from recording for playback setup...")

        success_apply_config = self.apply_simulation_configuration_core(config_snapshot, is_part_of_playback_setup=True)
        if not success_apply_config or not self.is_initialized:
            self._log_to_ui("CRITICAL: Failed to apply recorded config or initialize from recording data for playback.", "critical")
            if self.ui_queue: self.ui_queue.put({"type": "PLAYBACK_SETUP_FAILED", "reason": "Config application failed"})
            return False
        self._log_console(f"Applied recording config. Neuron count now: {self.core_config.num_neurons}")

        initial_state_arrays_np = {}
        for key in initial_state_h5_group.attrs.keys(): 
            if key.endswith("_is_empty") and initial_state_h5_group.attrs[key] is True:
                original_key = key.replace("_is_empty","")
                initial_state_arrays_np[original_key] = np.array([]) 
            elif initial_state_h5_group.attrs[key] == "NoneType":
                 initial_state_arrays_np[key] = None
            elif key not in ["connections_shape_0", "connections_shape_1"]: 
                initial_state_arrays_np[key] = initial_state_h5_group.attrs[key]
        
        for key in initial_state_h5_group.keys(): 
            if f"{key}_is_empty" not in initial_state_h5_group.attrs:
                initial_state_arrays_np[key] = initial_state_h5_group[key][:] 

        if "connections_data" in initial_state_arrays_np and \
           "connections_indices" in initial_state_arrays_np and \
           "connections_indptr" in initial_state_arrays_np and \
           initial_state_h5_group.attrs.get("connections_shape_0") is not None: 
            initial_state_arrays_np["connections_shape"] = (
                initial_state_h5_group.attrs["connections_shape_0"],
                initial_state_h5_group.attrs["connections_shape_1"]
            )
        else: 
            if "connections_data" not in initial_state_arrays_np: initial_state_arrays_np["connections_data"] = np.array([])
            if "connections_indices" not in initial_state_arrays_np: initial_state_arrays_np["connections_indices"] = np.array([], dtype=np.int32)
            if "connections_indptr" not in initial_state_arrays_np: initial_state_arrays_np["connections_indptr"] = np.array([0]*(self.core_config.num_neurons+1), dtype=np.int32)
            if "connections_shape" not in initial_state_arrays_np: initial_state_arrays_np["connections_shape"] = (self.core_config.num_neurons, self.core_config.num_neurons)


        self._apply_recorded_arrays_to_gpu(initial_state_arrays_np, is_initial_state=True)
        self._log_console("Applied initial full state from recording to GPU for playback.")

        self.runtime_state.current_time_ms = initial_state_arrays_np.get("start_time_ms", 0.0)
        self.runtime_state.current_time_step = initial_state_arrays_np.get("start_time_step", 0)
        
        if self.ui_queue:
            initial_gui_data = self.get_latest_simulation_data_for_gui(force_fetch=True)
            self.ui_queue.put({
                "type": "PLAYBACK_READY",
                "initial_gui_data": initial_gui_data,
                "current_time_ms": self.runtime_state.current_time_ms,
                "current_time_step": self.runtime_state.current_time_step
            })
        return True

    def _load_recording_to_gpu_cache(self, h5_file_handle, num_frames):
        """Loads recording into GPU memory with chunked loading, memory-aware partial caching.

        Features:
        - Chunked loading to provide progress updates without blocking
        - Parallel disk reads using ThreadPoolExecutor
        - Progress reporting to UI
        - Memory-aware partial caching: stops loading before GPU OOM
        - Seamlessly falls back to streaming for frames beyond cache
        """
        chunk_size = self.gpu_config.playback_cache_chunk_size
        num_chunks = (num_frames + chunk_size - 1) // chunk_size

        # Check initial GPU memory availability
        mem_info = _backend_get_device_mem_info()
        free_memory_initial, total_memory = mem_info
        free_gb_initial = free_memory_initial / 1e9

        # Reserve 20% of total memory for safety margin (simulation state, OS, etc.)
        safety_margin = 0.20
        usable_free_memory = free_memory_initial - (total_memory * safety_margin)
        usable_free_gb = max(0, usable_free_memory / 1e9)

        self._log_to_ui(
            f"Loading up to {num_frames} frames into GPU cache ({num_chunks} chunks of {chunk_size})...",
            "info"
        )
        self._log_console(f"  Available GPU memory: {free_gb_initial:.2f}GB (usable after safety margin: {usable_free_gb:.2f}GB)")
        start_time = time.time()

        try:
            self.gpu_playback_cache.clear()
            # Pool reuses blocks naturally — no sync stall needed here

            # Send initial progress to UI
            if self.ui_queue:
                self.ui_queue.put({
                    "type": "CACHE_LOAD_PROGRESS",
                    "progress": 0.0,
                    "frames_loaded": 0,
                    "total_frames": num_frames
                })

            frames_loaded = 0
            memory_limit_reached = False
            estimated_frame_size_bytes = None

            def read_frame_from_hdf5(frame_idx):
                """Read a single frame from HDF5 to NumPy (thread-safe for HDF5 reads)."""
                frame_group_name = f"frames/frame_{frame_idx}"
                frame_group = h5_file_handle.get(frame_group_name)

                if not frame_group:
                    return frame_idx, None

                frame_data_np = {}

                # Load attributes (scalars)
                for key, value in frame_group.attrs.items():
                    if value == "NoneType":
                        frame_data_np[key] = None
                    elif key.endswith("_is_empty") and value is True:
                        original_key = key.replace("_is_empty", "")
                        frame_data_np[original_key] = np.array([], dtype=np.float32)
                    else:
                        frame_data_np[key] = value

                # Load datasets (arrays)
                for key in frame_group.keys():
                    if f"{key}_is_empty" not in frame_group.attrs:
                        frame_data_np[key] = frame_group[key][:]

                return frame_idx, frame_data_np

            # Process in chunks
            for chunk_idx in range(num_chunks):
                # Check GPU memory before loading this chunk
                mem_info = _backend_get_device_mem_info()
                free_memory_now, _ = mem_info
                free_gb_now = free_memory_now / 1e9

                # Estimate if we have room for this chunk
                if estimated_frame_size_bytes is not None:
                    estimated_chunk_size_bytes = estimated_frame_size_bytes * chunk_size
                    if free_memory_now < estimated_chunk_size_bytes + (total_memory * safety_margin):
                        memory_limit_reached = True
                        self._log_to_ui(
                            f"GPU memory limit reached at {frames_loaded}/{num_frames} frames cached. "
                            f"Remaining {num_frames - frames_loaded} frames will stream from disk.",
                            "warning"
                        )
                        break

                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, num_frames)
                chunk_frames = list(range(chunk_start, chunk_end))

                chunk_start_time = time.time()

                # Phase 1: Parallel disk reads (HDF5 supports concurrent reads in most cases)
                frames_np_chunk = {}

                # Use ThreadPoolExecutor for parallel HDF5 reads
                max_workers = min(4, len(chunk_frames))
                if max_workers > 1:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {executor.submit(read_frame_from_hdf5, idx): idx for idx in chunk_frames}
                        for future in as_completed(futures):
                            frame_idx, frame_data = future.result()
                            if frame_data is not None:
                                frames_np_chunk[frame_idx] = frame_data
                else:
                    # Serial fallback for small chunks
                    for frame_idx in chunk_frames:
                        _, frame_data = read_frame_from_hdf5(frame_idx)
                        if frame_data is not None:
                            frames_np_chunk[frame_idx] = frame_data

                # Estimate frame size from first chunk (for memory prediction)
                if estimated_frame_size_bytes is None and frames_np_chunk:
                    sample_frame = next(iter(frames_np_chunk.values()))
                    estimated_frame_size_bytes = sum(
                        arr.nbytes if isinstance(arr, np.ndarray) else 8
                        for arr in sample_frame.values()
                    )
                    # Account for CuPy overhead (~10%)
                    estimated_frame_size_bytes = int(estimated_frame_size_bytes * 1.1)

                    # Check if we can fit all remaining frames
                    remaining_frames = num_frames - frames_loaded
                    estimated_total_bytes = remaining_frames * estimated_frame_size_bytes
                    if estimated_total_bytes > usable_free_memory:
                        max_cacheable = int(usable_free_memory / estimated_frame_size_bytes)
                        self._log_console(
                            f"  Frame size ~{estimated_frame_size_bytes / 1024:.1f}KB. "
                            f"Can cache ~{max_cacheable} of {num_frames} frames."
                        )

                # Phase 2: CPU→GPU transfer (must be serial due to CUDA context)
                try:
                    for frame_idx in sorted(frames_np_chunk.keys()):
                        frame_data_np = frames_np_chunk[frame_idx]
                        frame_data_gpu = {}

                        for key, value in frame_data_np.items():
                            if isinstance(value, np.ndarray):
                                frame_data_gpu[key] = cp.array(value)
                            else:
                                frame_data_gpu[key] = value

                        self.gpu_playback_cache[frame_idx] = frame_data_gpu
                        frames_loaded += 1

                # Backend-aware OOM: cp.cuda.memory.OutOfMemoryError on
                # CuPy, MemoryError on NumPy (which raises stdlib
                # MemoryError when alloc fails).
                except (MemoryError,
                          getattr(getattr(getattr(cp, "cuda", None), "memory", None),
                                    "OutOfMemoryError", MemoryError)):
                    # OOM during transfer - stop here and use what we have
                    memory_limit_reached = True
                    self._log_to_ui(
                        f"GPU OOM at {frames_loaded}/{num_frames} frames. "
                        f"Remaining frames will stream from disk.",
                        "warning"
                    )
                    break

                # Report progress after each chunk
                chunk_elapsed = time.time() - chunk_start_time
                progress_pct = (frames_loaded / num_frames) * 100

                self._log_console(
                    f"  Chunk {chunk_idx + 1}/{num_chunks}: {len(frames_np_chunk)} frames "
                    f"({progress_pct:.0f}%, {chunk_elapsed:.2f}s, GPU free: {free_gb_now:.1f}GB)"
                )

                # Send progress update to UI
                if self.ui_queue:
                    self.ui_queue.put({
                        "type": "CACHE_LOAD_PROGRESS",
                        "progress": progress_pct / 100.0,
                        "frames_loaded": frames_loaded,
                        "total_frames": num_frames
                    })

            elapsed = time.time() - start_time
            frames_per_sec = frames_loaded / elapsed if elapsed > 0 else 0

            # Check GPU memory usage
            mem_info = _backend_get_device_mem_info()
            free_memory, total_memory = mem_info
            used_gb = (total_memory - free_memory) / 1e9

            if memory_limit_reached:
                self._log_to_ui(
                    f"Partial cache: {frames_loaded}/{num_frames} frames in {elapsed:.2f}s "
                    f"({frames_per_sec:.1f} frames/s). GPU: {used_gb:.1f}GB. "
                    f"Frames 0-{frames_loaded-1} cached, rest will stream.",
                    "info"
                )
            else:
                self._log_to_ui(
                    f"Full cache: {frames_loaded} frames in {elapsed:.2f}s ({frames_per_sec:.1f} frames/s). GPU: {used_gb:.1f}GB",
                    "success"
                )

            # Send completion to UI
            if self.ui_queue:
                self.ui_queue.put({
                    "type": "CACHE_LOAD_COMPLETE",
                    "frames_loaded": frames_loaded,
                    "total_frames": num_frames,
                    "partial_cache": memory_limit_reached,
                    "elapsed_seconds": elapsed,
                    "frames_per_second": frames_per_sec
                })

            return True  # Partial success is still success - playback will work

        except Exception as e:
            self._log_to_ui(f"Error loading recording to GPU cache: {e}", "error")
            self.gpu_playback_cache.clear()
            import traceback
            traceback.print_exc()
            return False
    
    def _read_frame_from_file(self, frame_idx, h5_file_handle):
        """Reads a specific frame's data from the provided open HDF5 file handle."""
        if not h5_file_handle or not h5_file_handle.id: 
            self._log_to_ui("Playback error: HDF5 file is not open or invalid.", "error")
            if self.ui_queue: self.ui_queue.put({"type": "PLAYBACK_ERROR", "reason": "File handle invalid"})
            return None
        
        frame_group_name = f"frames/frame_{frame_idx}"
        try:
            frame_group = h5_file_handle.get(frame_group_name)
            if not frame_group:
                self._log_to_ui(f"Playback error: Frame group '{frame_group_name}' not found.", "error")
                return None

            frame_content = {}
            for key, value in frame_group.attrs.items():
                if value == "NoneType": frame_content[key] = None
                elif key.endswith("_is_empty") and value is True: 
                    original_key = key.replace("_is_empty","")
                    frame_content[original_key] = np.array([]) 
                else: frame_content[key] = value

            for key in frame_group.keys():
                 if f"{key}_is_empty" not in frame_group.attrs:
                    frame_content[key] = frame_group[key][:]
            return frame_content
        except Exception as e:
            self._log_to_ui(f"Error reading frame {frame_idx} from HDF5: {e}", "error")
            import traceback; traceback.print_exc()
            return None

    def _prefetch_frame(self, frame_idx, h5_file_handle, num_frames):
        """Prefetch a single frame in background thread."""
        if frame_idx < 0 or frame_idx >= num_frames:
            return

        with self.prefetch_lock:
            # Skip if already cached or being fetched
            if frame_idx in self.prefetch_buffer or frame_idx in self.prefetch_pending:
                return
            self.prefetch_pending.add(frame_idx)

        try:
            frame_data = self._read_frame_from_file(frame_idx, h5_file_handle)
            if frame_data is not None:
                with self.prefetch_lock:
                    self.prefetch_buffer[frame_idx] = frame_data
                    # Limit buffer size to avoid memory bloat
                    max_buffer_size = self.gpu_config.playback_prefetch_count * 2
                    if len(self.prefetch_buffer) > max_buffer_size:
                        # Remove oldest entries
                        oldest_keys = sorted(self.prefetch_buffer.keys())[:-max_buffer_size]
                        for key in oldest_keys:
                            del self.prefetch_buffer[key]
        finally:
            with self.prefetch_lock:
                self.prefetch_pending.discard(frame_idx)

    def _trigger_prefetch(self, current_frame, h5_file_handle, num_frames):
        """Trigger prefetching of upcoming frames in background."""
        if not self.gpu_config.enable_playback_prefetch:
            return

        prefetch_count = self.gpu_config.playback_prefetch_count

        # Initialize executor if needed
        if self.prefetch_executor is None:
            self.prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prefetch")

        # Submit prefetch tasks for next N frames
        for offset in range(1, prefetch_count + 1):
            frame_to_prefetch = current_frame + offset
            if frame_to_prefetch < num_frames:
                with self.prefetch_lock:
                    if frame_to_prefetch not in self.prefetch_buffer and frame_to_prefetch not in self.prefetch_pending:
                        self.prefetch_executor.submit(
                            self._prefetch_frame, frame_to_prefetch, h5_file_handle, num_frames
                        )

    def _get_prefetched_frame(self, frame_idx):
        """Get a frame from the prefetch buffer if available."""
        with self.prefetch_lock:
            return self.prefetch_buffer.pop(frame_idx, None)

    def _clear_prefetch_buffer(self):
        """Clear the prefetch buffer and pending set."""
        with self.prefetch_lock:
            self.prefetch_buffer.clear()
            self.prefetch_pending.clear()

    def set_playback_frame(self, frame_idx, h5_file_handle, num_frames=None):
        """Sets the simulation state to a specific frame from the loaded recording.

        Args:
            frame_idx: Frame index to load
            h5_file_handle: Open HDF5 file handle for streaming reads
            num_frames: Total number of frames (needed for prefetching bounds)
        """
        if not self.is_initialized:
            self._log_to_ui("Cannot set playback frame: Sim not initialized for playback.", "error")
            if self.ui_queue: self.ui_queue.put({"type": "PLAYBACK_ERROR", "reason": "Not initialized"})
            return

        # GPU-cached playback: instant frame seeking (no disk I/O)
        if self.gpu_config.enable_gpu_buffered_playback and frame_idx in self.gpu_playback_cache:
            frame_content_gpu = self.gpu_playback_cache[frame_idx]

            # Apply GPU data directly (NO GPU→CPU→GPU transfers)
            self._apply_recorded_arrays_to_gpu_direct(frame_content_gpu, is_initial_state=False)

            self.runtime_state.current_time_ms = frame_content_gpu.get("time_ms", self.runtime_state.current_time_ms)
            self.runtime_state.current_time_step = frame_content_gpu.get("step", self.runtime_state.current_time_step)
        else:
            # Streaming playback with prefetching
            # First check if frame is already in prefetch buffer
            frame_content_np = self._get_prefetched_frame(frame_idx)

            if frame_content_np is None:
                # Not prefetched, read directly from HDF5
                frame_content_np = self._read_frame_from_file(frame_idx, h5_file_handle)

            if frame_content_np is None:
                self._log_to_ui(f"Failed to read frame {frame_idx} for playback. Playback may be unstable.", "error")
                if self.ui_queue: self.ui_queue.put({"type": "PLAYBACK_ERROR", "reason": f"Failed to read frame {frame_idx}"})
                return

            self._apply_recorded_arrays_to_gpu(frame_content_np, is_initial_state=False)

            self.runtime_state.current_time_ms = frame_content_np.get("time_ms", self.runtime_state.current_time_ms)
            self.runtime_state.current_time_step = frame_content_np.get("step", self.runtime_state.current_time_step)

            # Trigger prefetch for upcoming frames (background I/O)
            if num_frames is not None and self.gpu_config.enable_playback_prefetch:
                self._trigger_prefetch(frame_idx, h5_file_handle, num_frames)

        latest_gui_data = self.get_latest_simulation_data_for_gui(force_fetch=True)
        if self.ui_queue and latest_gui_data:
            self.ui_queue.put({
                "type": "PLAYBACK_FRAME_APPLIED",
                "gui_data": latest_gui_data,
                "frame_index": frame_idx,
                "current_time_ms": self.runtime_state.current_time_ms,
                "current_time_step": self.runtime_state.current_time_step
            })

    def _apply_recorded_arrays_to_gpu_direct(self, state_dict_gpu, is_initial_state=False):
        """Applies CuPy arrays directly from GPU cache to simulation state (zero-copy)."""
        if not self.is_initialized:
            self._log_console("Cannot apply GPU-cached frame: Sim not initialized.", "error")
            return
        
        # Direct GPU-to-GPU copies (fast)
        dynamic_arrays_to_apply = [
            'cp_membrane_potential_v', 'cp_firing_states', 'cp_viz_activity_timers',
            'cp_conductance_g_e', 'cp_conductance_g_i',
            'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress'
        ]
        
        if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
            dynamic_arrays_to_apply.append('cp_recovery_variable_u')
            if self.core_config.enable_homeostasis and self.cp_neuron_firing_thresholds is not None:
                dynamic_arrays_to_apply.append('cp_neuron_firing_thresholds')
        elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
            dynamic_arrays_to_apply.extend([
                'cp_gating_variable_m', 'cp_gating_variable_h', 'cp_gating_variable_n',
                'cp_hh_m_current_activation', 'cp_hh_CaT_m', 'cp_hh_CaT_h', 'cp_hh_h_current_q', 'cp_hh_NaP_activation'
            ])
        elif self.core_config.neuron_model_type == NeuronModel.ADEX.name:
            dynamic_arrays_to_apply.append('cp_adex_w')
        
        # Copy CuPy arrays directly (GPU→GPU, very fast)
        for attr_name in dynamic_arrays_to_apply:
            if attr_name in state_dict_gpu:
                source_array = state_dict_gpu[attr_name]
                if source_array is not None and isinstance(source_array, cp.ndarray):
                    target_array = getattr(self, attr_name, None)
                    if target_array is not None and target_array.shape == source_array.shape:
                        target_array[:] = source_array  # In-place copy
        
        # Apply connection weights if Hebbian learning enabled
        if self.core_config.enable_hebbian_learning and "cp_connections_data" in state_dict_gpu:
            conn_data = state_dict_gpu["cp_connections_data"]
            if conn_data is not None and isinstance(conn_data, cp.ndarray) and self.cp_connections is not None:
                if self.cp_connections.data.shape == conn_data.shape:
                    self.cp_connections.data[:] = conn_data
        
        # Apply STP state if enabled
        if self.core_config.enable_short_term_plasticity:
            if "cp_stp_u" in state_dict_gpu and state_dict_gpu["cp_stp_u"] is not None:
                if self.cp_stp_u is not None and self.cp_stp_u.shape == state_dict_gpu["cp_stp_u"].shape:
                    self.cp_stp_u[:] = state_dict_gpu["cp_stp_u"]
            if "cp_stp_x" in state_dict_gpu and state_dict_gpu["cp_stp_x"] is not None:
                if self.cp_stp_x is not None and self.cp_stp_x.shape == state_dict_gpu["cp_stp_x"].shape:
                    self.cp_stp_x[:] = state_dict_gpu["cp_stp_x"]
    
    def _apply_recorded_arrays_to_gpu(self, state_dict_np, is_initial_state=False):
        """Applies NumPy arrays from HDF5 to CuPy arrays on GPU."""
        if not self.is_initialized and not is_initial_state:
             self._log_console("Cannot apply recorded frame arrays: Sim not initialized for playback.", "error")
             if self.ui_queue: self.ui_queue.put({"type": "PLAYBACK_ERROR", "reason": "Sim not initialized for frame apply"})
             return
        if not self.is_initialized and is_initial_state and not self.is_initialized:
             self._log_console("Cannot apply initial recorded arrays: Sim not initialized.", "error")
             if self.ui_queue: self.ui_queue.put({"type": "PLAYBACK_SETUP_FAILED", "reason": "Sim config missing for initial apply"})
             return

        # Synapse arrays that should be resized to match recording's synapse count
        synapse_arrays = {'cp_synapse_pulse_timers', 'cp_synapse_pulse_progress', 'cp_stp_u', 'cp_stp_x'}

        def _apply_to_cp_array(cp_array_attr_name, np_array_key_in_dict, default_dtype=cp.float32):
            """Helper to apply a NumPy array from state_dict_np to a CuPy array attribute."""
            source_np_array = state_dict_np.get(np_array_key_in_dict)

            if source_np_array is None:
                if hasattr(self, cp_array_attr_name) and getattr(self, cp_array_attr_name) is not None:
                    setattr(self, cp_array_attr_name, None)
                return

            if not isinstance(source_np_array, np.ndarray):
                return

            target_cp_array = getattr(self, cp_array_attr_name, None)

            if target_cp_array is None and source_np_array.size > 0 :
                try:
                    setattr(self, cp_array_attr_name, cp.asarray(source_np_array, dtype=default_dtype))
                except Exception as e:
                    self._log_console(f"Error creating {cp_array_attr_name} from recording: {e}", "error"); return
            elif target_cp_array is not None:
                if target_cp_array.shape == source_np_array.shape:
                    if target_cp_array.dtype == source_np_array.dtype:
                        target_cp_array[:] = cp.asarray(source_np_array)
                    else:
                        try: target_cp_array[:] = cp.asarray(source_np_array.astype(target_cp_array.dtype))
                        except Exception as e: self._log_console(f"Error applying {cp_array_attr_name} due to dtype mismatch and cast fail: {e}", "error")
                elif target_cp_array.size == source_np_array.size and source_np_array.size > 0:
                    try: target_cp_array[:] = cp.asarray(source_np_array.reshape(target_cp_array.shape))
                    except ValueError as ve: self._log_console(f"ERROR: Failed to reshape {cp_array_attr_name}. Error: {ve}", "error")
                elif source_np_array.size == 0 and target_cp_array.size == 0: pass
                elif source_np_array.size == 0 and target_cp_array.size > 0:
                     target_cp_array.fill(0)
                elif cp_array_attr_name in synapse_arrays:
                    # Synapse arrays can be resized to match recording's synapse count
                    # This happens when recording has different connection count than current config
                    try:
                        setattr(self, cp_array_attr_name, cp.asarray(source_np_array, dtype=default_dtype))
                    except Exception as e:
                        self._log_console(f"Error resizing {cp_array_attr_name} from recording: {e}", "error")
                else:
                    self._log_console(f"Error: Shape/size mismatch for {cp_array_attr_name} from recording. Target: {target_cp_array.shape}, Source: {source_np_array.shape}. Cannot apply.", "error")
            elif target_cp_array is None and source_np_array.size == 0:
                setattr(self, cp_array_attr_name, cp.array([], dtype=default_dtype))

        if is_initial_state: 
            _apply_to_cp_array("cp_traits", "cp_traits", default_dtype=cp.int32)
            _apply_to_cp_array("cp_neuron_positions_3d", "cp_neuron_positions_3d")
            if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                for param in ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c_reset', 'd_increment']:
                    _apply_to_cp_array(f"cp_izh_{param}", f"cp_izh_{param}")
            elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                for param in ['C_m', 'g_Na_max', 'g_K_max', 'g_L', 'E_Na', 'E_K', 'E_L', 'v_peak']:
                     _apply_to_cp_array(f"cp_hh_{param}", f"cp_hh_{param}")

            conn_data_np = state_dict_np.get("connections_data")
            conn_indices_np = state_dict_np.get("connections_indices")
            conn_indptr_np = state_dict_np.get("connections_indptr")
            conn_shape = state_dict_np.get("connections_shape") 

            if conn_data_np is not None and conn_indices_np is not None and conn_indptr_np is not None and conn_shape is not None:
                if conn_shape[0] != self.core_config.num_neurons or conn_shape[1] != self.core_config.num_neurons:
                    self._log_console(f"Error: Connection shape {conn_shape} from recording's initial_state "
                                     f"does not match configured neuron count {self.core_config.num_neurons}. Playback may fail.", "error")
                
                self.cp_connections = csp.csr_matrix((cp.asarray(conn_data_np),
                                                      cp.asarray(conn_indices_np),
                                                      cp.asarray(conn_indptr_np)),
                                                     shape=conn_shape, dtype=cp.float32)
                self.cp_connections.sort_indices()
            else: 
                self._log_console("Warning: Connection structure missing/incomplete in initial_state. Using empty matrix.", "warning")
                n_cfg = self.core_config.num_neurons
                self.cp_connections = csp.csr_matrix((n_cfg, n_cfg), dtype=cp.float32)
            
            num_synapses_loaded = self.cp_connections.nnz
            _apply_to_cp_array("cp_synapse_pulse_timers", "cp_synapse_pulse_timers", default_dtype=cp.int32)
            _apply_to_cp_array("cp_synapse_pulse_progress", "cp_synapse_pulse_progress")
            
            if self.cp_synapse_pulse_timers is None or self.cp_synapse_pulse_timers.size != num_synapses_loaded:
                self.cp_synapse_pulse_timers = cp.zeros(num_synapses_loaded, dtype=cp.int32)
            if self.cp_synapse_pulse_progress is None or self.cp_synapse_pulse_progress.size != num_synapses_loaded:
                self.cp_synapse_pulse_progress = cp.zeros(num_synapses_loaded, dtype=cp.float32)

            if self.core_config.enable_short_term_plasticity:
                _apply_to_cp_array("cp_stp_u", "cp_stp_u")
                _apply_to_cp_array("cp_stp_x", "cp_stp_x")
                if self.cp_stp_u is None or self.cp_stp_u.size != num_synapses_loaded:
                    self.cp_stp_u = cp.full(num_synapses_loaded, self.core_config.stp_U, dtype=cp.float32) if num_synapses_loaded > 0 else cp.array([], dtype=cp.float32)
                if self.cp_stp_x is None or self.cp_stp_x.size != num_synapses_loaded:
                    self.cp_stp_x = cp.ones(num_synapses_loaded, dtype=cp.float32) if num_synapses_loaded > 0 else cp.array([], dtype=cp.float32)
            else:
                self.cp_stp_u = None; self.cp_stp_x = None

        dynamic_keys_map = { 
            'cp_membrane_potential_v': 'cp_membrane_potential_v', 
            'cp_recovery_variable_u': 'cp_recovery_variable_u', 
            'cp_gating_variable_m': 'cp_gating_variable_m', 
            'cp_gating_variable_h': 'cp_gating_variable_h', 
            'cp_gating_variable_n': 'cp_gating_variable_n', 
            'cp_hh_m_current_activation': 'cp_hh_m_current_activation',
            'cp_hh_CaT_m': 'cp_hh_CaT_m',
            'cp_hh_CaT_h': 'cp_hh_CaT_h',
            'cp_hh_h_current_q': 'cp_hh_h_current_q',
            'cp_hh_NaP_activation': 'cp_hh_NaP_activation',
            'cp_conductance_g_e': 'cp_conductance_g_e',
            'cp_conductance_g_i': 'cp_conductance_g_i',
            'cp_adex_w': 'cp_adex_w',
            'cp_external_input_current': 'cp_external_input_current',
            'cp_refractory_timers': ('cp_refractory_timers', cp.int32),
            'cp_viz_activity_timers': ('cp_viz_activity_timers', cp.int32),
            'cp_neuron_firing_thresholds': 'cp_neuron_firing_thresholds', 
            'cp_neuron_activity_ema': 'cp_neuron_activity_ema',
            'cp_firing_states': ('cp_firing_states', cp.bool_),
            'cp_prev_firing_states': ('cp_prev_firing_states', cp.bool_),
            'cp_stp_u': 'cp_stp_u', 
            'cp_stp_x': 'cp_stp_x', 
            'cp_synapse_pulse_timers': ('cp_synapse_pulse_timers', cp.int32), 
            'cp_synapse_pulse_progress': 'cp_synapse_pulse_progress' 
        }

        if not is_initial_state and "cp_connections_data" in state_dict_np:
            conn_data_frame_np = state_dict_np.get("cp_connections_data")
            if conn_data_frame_np is not None and self.cp_connections is not None and self.cp_connections.data is not None:
                if isinstance(conn_data_frame_np, np.ndarray):
                    if self.cp_connections.data.shape == conn_data_frame_np.shape:
                        self.cp_connections.data[:] = cp.asarray(conn_data_frame_np)
                    elif self.cp_connections.data.size == conn_data_frame_np.size and conn_data_frame_np.size > 0:
                        try: self.cp_connections.data[:] = cp.asarray(conn_data_frame_np.reshape(self.cp_connections.data.shape))
                        except ValueError as ve: self._log_console(f"ERROR: Failed to reshape cp_connections.data from recording frame. Error: {ve}", "error")
                    elif not (self.cp_connections.data.size == 0 and conn_data_frame_np.size == 0):
                        # Size mismatch due to structural plasticity during recording - silently skip
                        # Connection weights won't update but other state (membrane potential, firing) is fine
                        pass
            elif conn_data_frame_np is None and self.cp_connections is not None and self.cp_connections.data is not None:
                 pass 
                 
        for cp_attr, key_info in dynamic_keys_map.items():
            np_key = key_info if isinstance(key_info, str) else key_info[0]
            default_dtype = cp.float32 
            if not isinstance(key_info, str) and len(key_info) > 1:
                default_dtype = key_info[1]
            
            if np_key == "cp_connections_data" and not is_initial_state: 
                continue
            _apply_to_cp_array(cp_attr, np_key, default_dtype=default_dtype)

        self._mock_num_spikes_this_step = state_dict_np.get("_mock_num_spikes_this_step", 0)
        self._mock_network_avg_firing_rate_hz = state_dict_np.get("_mock_network_avg_firing_rate_hz", 0.0)
        self._mock_total_plasticity_events = state_dict_np.get("_mock_total_plasticity_events", 0)

        if is_initial_state:
            self.runtime_state.current_time_ms = state_dict_np.get("start_time_ms", 0.0)
            self.runtime_state.current_time_step = state_dict_np.get("start_time_step", 0)

    # --- Resonate-and-fire FHRR substrate (opt-in, owner-funded Option A; see
    #     docs/plans/2026-06-05-rf-on-bridge-derisk-design.md). Active only when
    #     neuron_model_type == RESONATE_AND_FIRE; zero impact on Izhikevich/HH/AdEx. ---
    def rf_kick(self, kick_complex, period=None, lam=-3.0e-4, floor=1.0e-3):
        """Inject a complex 'kick' into the resonate-and-fire neurons: set the complex state Z = re + i*im (reusing
        v=re, u=im) to kick_complex and reset the phase-readout trackers. Then run `period`(+8) steps of
        `_run_one_simulation_step` (with neuron_model_type=RESONATE_AND_FIRE) and call `rf_read_phases()` to recover
        each neuron's phase. The kick is the FHRR operand (bind = phasor_a*phasor_b, unbind = phasor_c*conj(a),
        bundle = sum of phasors); the resonate + phase readout run on the bridge's own neurons in its own step.
        period defaults to 1000 (one phasor cycle = T bridge steps). Mirrors resonate_fire_fhrr.rf_resonate."""
        kick = np.asarray(kick_complex, dtype=np.complex128).reshape(-1)
        n = self.core_config.num_neurons
        if kick.shape[0] != n:
            raise ValueError(f"rf_kick expects {n} complex values (one per neuron), got {kick.shape[0]}")
        self._rf_period = int(period) if period else 1000
        self._rf_omega = 2.0 * np.pi / self._rf_period
        self._rf_lambda = float(lam)
        self._rf_floor = float(floor)
        self._rf_counter = 0
        self.cp_membrane_potential_v[:] = cp.asarray(kick.real, dtype=self.cp_membrane_potential_v.dtype)
        self.cp_recovery_variable_u[:] = cp.asarray(kick.imag, dtype=self.cp_recovery_variable_u.dtype)
        self.cp_rf_prev_im = self.cp_recovery_variable_u.copy()
        self.cp_rf_fired = cp.zeros(n, dtype=bool)
        # default spike step = period -> phase 0 for a neuron that never crosses (|Z| decayed below the floor).
        self.cp_rf_spike_step = cp.full(n, self._rf_period, dtype=cp.int64)

    def rf_read_phases(self):
        """Recover the RF neurons' phases in [0,1) from their first-spike steps (the magnitude-invariant readout):
        phase = ((period - spike_step) mod period) / period. Call after running period(+8) steps post-rf_kick()."""
        period = int(getattr(self, "_rf_period", 1000))
        spike_step = np.asarray(_backend_to_host(self.cp_rf_spike_step)).astype(np.int64)
        return ((period - spike_step) % period) / float(period)

    def rf_set_complex_weights(self, connections):
        """Install complex synaptic weights for the resonate-and-fire neurons (FHRR bind THROUGH synapses) as a
        SPARSE complex matrix (csr `cp_rf_w_re` + `cp_rf_w_im`) built FRESH from `connections` (a list of
        (post_idx, pre_idx, complex_weight)) -- REPLACES any prior weights. Each step the complex synaptic input
        u_i = sum_j W_ij z_j is added to neuron i's complex state, so binding phasor_a*phasor_b is phasor_a passing
        through a synapse whose complex weight is phasor_b (complex multiply = phase sum). The matvec `W @ z` in the
        RF branch is then O(nnz) -- the diagonal bind/unbind + unit bundle synapses are O(D) sparse, not O(N^2) --
        which is what makes 320-concept production scale (D=512) tractable. The matvec line is identical for
        sparse/dense (`matrix @ vector`). See docs/plans/2026-06-05-full-fhrr-on-bridge-feature-plan.md."""
        n = self.core_config.num_neurons
        m = len(connections)
        rows = np.fromiter((int(post) for (post, pre, w) in connections), dtype=np.int32, count=m)
        cols = np.fromiter((int(pre) for (post, pre, w) in connections), dtype=np.int32, count=m)
        w_re = np.fromiter((float(complex(w).real) for (post, pre, w) in connections), dtype=np.float64, count=m)
        w_im = np.fromiter((float(complex(w).imag) for (post, pre, w) in connections), dtype=np.float64, count=m)
        r = cp.asarray(rows); c = cp.asarray(cols)
        self.cp_rf_w_re = csp.csr_matrix((cp.asarray(w_re), (r, c)), shape=(n, n))
        self.cp_rf_w_im = csp.csr_matrix((cp.asarray(w_im), (r, c)), shape=(n, n))

    def _rf_advance_one(self):
        """One step of the resonate-and-fire dynamics: rotate the complex state Z=re+i*im (v=re, u=im) by
        exp(lambda+i*omega), add the complex synaptic input u=W z (sparse matvec from the presynaptic RF states),
        detect the upward Im zero-crossing -> spike (recording its step = the kick's phase). Updates the state +
        spike trackers; returns the fired mask. Shared by the main-step RF branch and the fast rf_resonate_steps()
        loop. Assumes the RF trackers exist (rf_kick() or the step's lazy init)."""
        _rf_decay = float(np.exp(getattr(self, "_rf_lambda", -3.0e-4)))
        _rf_omega = float(getattr(self, "_rf_omega", 2.0 * np.pi / 1000.0))
        _rf_floor2 = float(getattr(self, "_rf_floor", 1.0e-3)) ** 2
        _rf_cos = float(np.cos(_rf_omega)); _rf_sin = float(np.sin(_rf_omega))
        _rf_re = self.cp_membrane_potential_v
        _rf_im = self.cp_recovery_variable_u
        _rf_re_new = _rf_decay * (_rf_re * _rf_cos - _rf_im * _rf_sin)
        _rf_im_new = _rf_decay * (_rf_re * _rf_sin + _rf_im * _rf_cos)
        if getattr(self, "cp_rf_w_re", None) is not None:
            # FHRR bind THROUGH synapses: complex matvec u_i = sum_j W_ij z_j from the presynaptic RF states.
            _rf_re_new = _rf_re_new + (self.cp_rf_w_re @ _rf_re - self.cp_rf_w_im @ _rf_im)
            _rf_im_new = _rf_im_new + (self.cp_rf_w_re @ _rf_im + self.cp_rf_w_im @ _rf_re)
        self._rf_counter = int(getattr(self, "_rf_counter", 0)) + 1
        _rf_mag2 = _rf_re_new * _rf_re_new + _rf_im_new * _rf_im_new
        _rf_crossed = ((~self.cp_rf_fired) & (self.cp_rf_prev_im < 0.0)
                       & (_rf_im_new >= 0.0) & (_rf_mag2 > _rf_floor2))
        self.cp_rf_spike_step = cp.where(_rf_crossed, self._rf_counter, self.cp_rf_spike_step)
        self.cp_rf_fired = self.cp_rf_fired | _rf_crossed
        self.cp_membrane_potential_v[:] = _rf_re_new
        self.cp_recovery_variable_u[:] = _rf_im_new
        self.cp_rf_prev_im = _rf_im_new
        return _rf_crossed

    def rf_resonate_steps(self, n_steps):
        """Run `n_steps` of the RF resonate dynamics DIRECTLY (the production-fast path) -- skips the full
        `_run_one_simulation_step` machinery (conductance / plasticity / recording / engram / gate couplings / stats),
        none of which the RF/FHRR substrate uses. The composer's per-op resonate window calls this instead of looping
        `_run_one_simulation_step`, eliminating the dominant per-step overhead at 320-concept scale. Assumes
        rf_kick() was called (else a no-op)."""
        if getattr(self, "cp_rf_prev_im", None) is None:
            return
        for _ in range(int(n_steps)):
            self._rf_advance_one()

    def _run_one_simulation_step(self):
        """Executes a single step of the simulation logic."""
        if not self.is_initialized or self.core_config.num_neurons == 0: return
        try:
            n_neurons = self.core_config.num_neurons; cfg = self.core_config; dt = cfg.dt_ms

            # Step profiler: optional per-section wall-clock timing for bottleneck analysis.
            # Enable via GPUConfig.enable_step_profiler. Logs summary every 500 steps.
            _profiling = self.gpu_config.enable_step_profiler
            if _profiling:
                import time as _time
                _prof = {}
                _t0 = _time.perf_counter()

            # Cache cp_prev_firing_states.any() ONCE per step to avoid repeated GPU-CPU sync stalls.
            # This result is used in STP, synaptic propagation, and Hebbian blocks.
            _prev_any = bool(self.cp_prev_firing_states.any())

            # --- 1. Synaptic Plasticity (STP) Update ---
            if _profiling: _backend_synchronize(); _prof['t_init'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()
            base_synaptic_weights = self.cp_connections.data
            effective_synaptic_strength = base_synaptic_weights

            # Freeze STP during experiments: sustained stimulus-driven firing (e.g. 34 Hz
            # CS input over 200ms pulses) causes pathological synaptic depression under
            # Tsodyks-Markram STP (U=0.15, τ_d=200ms → effective multiplier ≈ 0.07), reducing
            # a ~48 pA learned signal to ~3.6 pA — invisible against OU noise. Real experiments
            # account for STP by design (ISI, frequency tuning); our STP parameters are tuned
            # for general network dynamics, not experiment validation protocols.
            _stp_active = cfg.enable_short_term_plasticity and not (
                self.experiment_engine is not None and self.experiment_engine.is_experiment_running
            )

            if _stp_active and self.cp_connections.nnz > 0 and \
               self.cp_stp_u is not None and self.cp_stp_x is not None:

                # Per-synapse-type STP: use cached per-synapse tau_f/tau_d/U arrays
                if cfg.enable_per_type_stp and self.cp_synapse_conn_type is not None:
                    # Cache per-synapse STP parameter arrays — they only change when
                    # connectivity changes (structural plasticity) or config is reloaded.
                    # Avoids 3 cp.array() + 3 fancy-index + 2 concatenate ops per step.
                    if not hasattr(self, '_cached_stp_per_type') or self._cached_stp_per_type is None:
                        actual_nnz_stp = self.cp_connections.nnz
                        ctypes = self.cp_synapse_conn_type[:actual_nnz_stp]
                        U_arr = cp.array(cfg.stp_U_per_type, dtype=cp.float32)
                        tau_f_arr = cp.array(cfg.stp_tau_f_per_type, dtype=cp.float32)
                        tau_d_arr = cp.array(cfg.stp_tau_d_per_type, dtype=cp.float32)
                        stp_tau_f_per_syn = tau_f_arr[ctypes]
                        stp_tau_d_per_syn = tau_d_arr[ctypes]
                        stp_U_per_syn_cached = U_arr[ctypes]
                        n_pad = self.cp_stp_u.size - actual_nnz_stp
                        if n_pad > 0:
                            stp_tau_f_full = cp.concatenate([stp_tau_f_per_syn, cp.full(n_pad, cfg.stp_tau_f, dtype=cp.float32)])
                            stp_tau_d_full = cp.concatenate([stp_tau_d_per_syn, cp.full(n_pad, cfg.stp_tau_d, dtype=cp.float32)])
                        else:
                            stp_tau_f_full = stp_tau_f_per_syn
                            stp_tau_d_full = stp_tau_d_per_syn
                        self._cached_stp_per_type = (stp_tau_f_full, stp_tau_d_full, stp_U_per_syn_cached)
                    stp_tau_f_full, stp_tau_d_full, stp_U_per_syn = self._cached_stp_per_type
                    self.cp_stp_u, self.cp_stp_x = fused_stp_decay_recovery(
                        self.cp_stp_u, self.cp_stp_x, dt, stp_tau_f_full, stp_tau_d_full)
                else:
                    stp_U_per_syn = None
                    self.cp_stp_u, self.cp_stp_x = fused_stp_decay_recovery(
                        self.cp_stp_u, self.cp_stp_x, dt, cfg.stp_tau_f, cfg.stp_tau_d)

                if _prev_any:
                    coo_matrix_stp = self._get_cached_coo()  # Use cached COO (avoids 40-400ms tocoo() per step)
                    if coo_matrix_stp is None:
                        coo_matrix_stp = self.cp_connections.tocoo(copy=False)  # Fallback
                    active_syn_mask_stp = self.cp_prev_firing_states[coo_matrix_stp.row]
                    active_syn_indices_stp = cp.where(active_syn_mask_stp)[0]

                    if active_syn_indices_stp.size > 0:
                        # Per-type U at spike time
                        if stp_U_per_syn is not None:
                            U_stp = stp_U_per_syn[active_syn_indices_stp]
                        else:
                            U_stp = cfg.stp_U
                        u_active_old = self.cp_stp_u[active_syn_indices_stp]
                        x_active_old = self.cp_stp_x[active_syn_indices_stp]

                        u_active_new = u_active_old + U_stp * (1.0 - u_active_old)
                        self.cp_stp_u[active_syn_indices_stp] = u_active_new
                        self.cp_stp_x[active_syn_indices_stp] = x_active_old * (1.0 - u_active_new) 

                cp.clip(self.cp_stp_x, 0.0, 1.0, out=self.cp_stp_x)
                cp.clip(self.cp_stp_u, 0.0, 1.0, out=self.cp_stp_u)

                # Use actual connection count (cp_connections.nnz) as authoritative size.
                # _synapse_count tracks pre-allocated array usage but can diverge from
                # cp_connections.nnz when CSR addition deduplicates overlapping (pre,post)
                # pairs during structural plasticity.
                actual_nnz = self.cp_connections.nnz
                stp_u_active = self.cp_stp_u[:actual_nnz]
                stp_x_active = self.cp_stp_x[:actual_nnz]
                effective_synaptic_strength = base_synaptic_weights * stp_u_active * stp_x_active

                # Neuromodulator subsystem: scope=all synaptic_gain multiplier.
                if (getattr(cfg, "enable_neuromodulator_subsystem", False)
                        and self.neuromodulator_manager is not None):
                    nm_gain = self.neuromodulator_manager.compute_synaptic_gain_multiplier()
                    if abs(nm_gain - 1.0) > 1e-9:
                        effective_synaptic_strength = effective_synaptic_strength * nm_gain

                effective_connections_matrix = csp.csr_matrix(
                    (effective_synaptic_strength, self.cp_connections.indices, self.cp_connections.indptr),
                    shape=self.cp_connections.shape
                )
            else:
                # No STP, no neuromod: use connections as-is.
                # If neuromod synaptic_gain is active, build a scaled matrix.
                if (getattr(cfg, "enable_neuromodulator_subsystem", False)
                        and self.neuromodulator_manager is not None):
                    nm_gain = self.neuromodulator_manager.compute_synaptic_gain_multiplier()
                    if abs(nm_gain - 1.0) > 1e-9:
                        actual_nnz = self.cp_connections.nnz
                        scaled_data = self.cp_connections.data[:actual_nnz] * nm_gain
                        effective_connections_matrix = csp.csr_matrix(
                            (scaled_data, self.cp_connections.indices, self.cp_connections.indptr),
                            shape=self.cp_connections.shape,
                        )
                    else:
                        effective_connections_matrix = self.cp_connections
                else:
                    effective_connections_matrix = self.cp_connections

            # Per-pathway TRANSMISSION gate (thalamocortical dynamical gating, 2026-06-03): scale the
            # effective synaptic CURRENT by the per-synapse gain (closed gate=0.0 -> no current flows, even
            # though the weight is non-zero). Build a fresh matrix so cp_connections is never mutated; data
            # is row-aligned with cp_connections.data, so cp_transmission_gain (same order) multiplies directly.
            if self.cp_transmission_gain is not None and self.cp_connections.nnz > 0:
                _tg_nnz = self.cp_connections.nnz
                _gated_data = effective_connections_matrix.data * self.cp_transmission_gain[:_tg_nnz]
                effective_connections_matrix = csp.csr_matrix(
                    (_gated_data, self.cp_connections.indices, self.cp_connections.indptr),
                    shape=self.cp_connections.shape,
                )

            if _profiling: _backend_synchronize(); _prof['t_stp'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()
            # --- 2. Synaptic Conductance Update & Current Calculation ---
            decay_e = self._cached_decay_e
            decay_i = self._cached_decay_i

            # E_inh: per-neuron array (R1.1, PBR-160) when allocated; else global scalar.
            # Per-neuron override lets regions like striatal MSNs (~−60 mV) and SNc DA
            # (~−55 mV) deviate from the cortical-pyramidal default of −75 mV.
            E_inh_to_use = (
                self.cp_syn_reversal_potential_i_per_neuron
                if self.cp_syn_reversal_potential_i_per_neuron is not None
                else cfg.syn_reversal_potential_i
            )

            self.cp_conductance_g_e, self.cp_conductance_g_i, synaptic_current_I_syn_pA = fused_conductance_decay_and_current(
                self.cp_conductance_g_e, self.cp_conductance_g_i, decay_e, decay_i,
                self.cp_membrane_potential_v, cfg.syn_reversal_potential_e, E_inh_to_use
            )

            g_e_increase = None  # Track for NMDA input
            if effective_connections_matrix.nnz > 0 and _prev_any:
                prev_fired_float = self.cp_prev_firing_states.astype(cp.float32)

                if cfg.enable_inhibitory_neurons and self.cp_traits is not None:
                    # Cache inhibitory neuron mask — traits don't change during simulation.
                    if self._cached_inhibitory_mask is None:
                        inhibitory_indices = getattr(cfg, "inhibitory_trait_indices", None)
                        if inhibitory_indices:
                            inhibitory_indices_cp = cp.asarray(inhibitory_indices, dtype=cp.int32)
                            self._cached_inhibitory_mask = cp.isin(self.cp_traits, inhibitory_indices_cp)
                        else:
                            self._cached_inhibitory_mask = (self.cp_traits == cfg.inhibitory_trait_index)
                    is_inhibitory_neuron_output = self._cached_inhibitory_mask
                    exc_fired_prev = prev_fired_float * (~is_inhibitory_neuron_output)
                    inhib_fired_prev = prev_fired_float * is_inhibitory_neuron_output

                    # Batched sparse matmul: stack exc/inh firing vectors into (n, 2)
                    # matrix, perform single A.T @ B (reuses CSR index traversal).
                    fired_2col = cp.stack([exc_fired_prev, inhib_fired_prev], axis=1)
                    g_increase_2col = effective_connections_matrix.T @ fired_2col
                    g_e_increase = g_increase_2col[:, 0] * cfg.propagation_strength
                    g_i_increase = g_increase_2col[:, 1] * cfg.inhibitory_propagation_strength

                    self.cp_conductance_g_e += g_e_increase
                    self.cp_conductance_g_i += g_i_increase
                else:
                    g_e_increase = (effective_connections_matrix.T @ prev_fired_float) * cfg.propagation_strength
                    self.cp_conductance_g_e += g_e_increase

            total_input_current_pA = synaptic_current_I_syn_pA + self.cp_external_input_current

            # Neuromodulator excitability_drive (additive pA, scope=all + per-neuron).
            if (getattr(cfg, "enable_neuromodulator_subsystem", False)
                    and self.neuromodulator_manager is not None):
                nm_scalar_drive = self.neuromodulator_manager.compute_excitability_drive_pA()
                if abs(nm_scalar_drive) > 1e-9:
                    total_input_current_pA = total_input_current_pA + cp.float32(nm_scalar_drive)
                nm_per_neuron_drive = self.neuromodulator_manager.compute_excitability_drive_per_neuron(
                    cp_traits=self.cp_traits,
                )
                if nm_per_neuron_drive is not None:
                    total_input_current_pA = total_input_current_pA + nm_per_neuron_drive

            # --- 2.2b. Experiment Stimulus Injection ---
            if self.experiment_engine is not None and self.experiment_engine.is_experiment_running:
                try:
                    experiment_stimulus = self.experiment_engine.step(
                        self.runtime_state.current_time_ms,
                        self.cp_firing_states,
                        self.cp_membrane_potential_v,
                        self, cp
                    )
                    total_input_current_pA = total_input_current_pA + experiment_stimulus
                except Exception as e:
                    self._log_console(f"Experiment engine step error: {e}", "warning")

            # --- 2.3. NMDA conductance with Mg²⁺ block (Jahr & Stevens 1990) ---
            if cfg.enable_nmda and self.cp_conductance_g_nmda is not None:
                # Update NMDA dual-exponential conductance and compute Mg²⁺-gated current
                self.cp_conductance_g_nmda, self.cp_conductance_g_nmda_rise, I_nmda = fused_nmda_update_and_current(
                    self.cp_conductance_g_nmda, self.cp_conductance_g_nmda_rise,
                    self._cached_decay_nmda, self._cached_decay_nmda_rise,
                    self.cp_membrane_potential_v, cfg.syn_reversal_potential_e,  # E_NMDA = E_AMPA = 0 mV
                    cfg.nmda_mg_concentration
                )
                # NMDA gets same excitatory input as AMPA, scaled by nmda_ratio
                if g_e_increase is not None:
                    g_nmda_increase = g_e_increase * cfg.nmda_ratio
                    self.cp_conductance_g_nmda += g_nmda_increase
                    self.cp_conductance_g_nmda_rise += g_nmda_increase
                # Cluster G v2 (2026-05-01): apply per-neuron NMDA mask if
                # any region opted into NMDA. Without mask: NMDA applies
                # globally (v1 behavior). With mask: only neurons in
                # regions with enable_nmda=True receive NMDA current.
                if self.cp_nmda_neuron_mask is not None:
                    I_nmda = I_nmda * self.cp_nmda_neuron_mask
                total_input_current_pA = total_input_current_pA + I_nmda

            # --- 2.5. Update OU Process & Inject Background Noise ---
            if cfg.enable_ou_process and hasattr(self, 'cp_ou_current') and self.cp_ou_current is not None:
                # Update OU current using exact solution: I(t+dt) = I(t)*exp(-dt/tau) + mean*(1-exp(-dt/tau)) + noise
                # NOTE: RNG was seeded once at initialization. Per-step seeding removed to preserve
                # temporal correlations in OU process and improve performance.

                # Exact OU update (Gillespie 1996)
                noise_samples = cp.random.randn(n_neurons).astype(cp.float32)
                self.cp_ou_current[:] = (
                    self.cp_ou_current * self.ou_decay_factor +
                    self.ou_mean * (1.0 - self.ou_decay_factor) +
                    self.ou_noise_std * noise_samples
                )
                
                # Add OU current to total input
                total_input_current_pA = total_input_current_pA + self.cp_ou_current

            if _profiling: _backend_synchronize(); _prof['t_syn'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()
            # --- 3. Neuron Model Dynamics Update ---
            fired_this_step = cp.zeros(n_neurons, dtype=bool)

            if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                v_new, u_new = fused_izhikevich2007_dynamics_update(
                    self.cp_membrane_potential_v, self.cp_recovery_variable_u,
                    self.cp_izh_C, self.cp_izh_k, self.cp_izh_vr, self.cp_izh_vt,
                    self.cp_izh_a, self.cp_izh_b,
                    total_input_current_pA, dt
                )
                not_in_refractory = (self.cp_refractory_timers <= 0)
                current_spike_thresholds = self.cp_neuron_firing_thresholds if cfg.enable_homeostasis and self.cp_neuron_firing_thresholds is not None else self.cp_izh_vpeak
                fired_this_step = (v_new >= current_spike_thresholds) & not_in_refractory

                if getattr(cfg, "fast_spike_reset", False):
                    # Fast path: cp.where masked-update. No GPU-CPU sync,
                    # no fancy-index assignment. Numerically equivalent to
                    # legacy for the Izhikevich model. Biggest win on
                    # small networks where launch overhead dominates.
                    # See tests/test_fast_spike_reset.py for verification.
                    v_new = cp.where(fired_this_step, self.cp_izh_c_reset, v_new)
                    u_new = cp.where(fired_this_step,
                                      u_new + self.cp_izh_d_increment, u_new)
                    # Refractory: fired -> period_steps - 1 (matches legacy
                    # off-by-one: legacy sets to N then decrements to N-1
                    # via the unconditional decrement). Non-fired with
                    # timer > 0 decrement; otherwise stay 0.
                    new_refractory_for_fired = cp.int32(max(0, cfg.refractory_period_steps - 1))
                    new_refractory_for_unfired = cp.maximum(
                        self.cp_refractory_timers - cp.int32(1),
                        cp.int32(0),
                    )
                    self.cp_refractory_timers[:] = cp.where(
                        fired_this_step,
                        new_refractory_for_fired,
                        new_refractory_for_unfired,
                    )
                    self.cp_membrane_potential_v[:] = v_new
                    self.cp_recovery_variable_u[:] = u_new
                else:
                    # Legacy path: fancy-index assignment + GPU-CPU sync at
                    # `fired_indices.size > 0`. Bit-identical to historical
                    # behavior. Default kept for backward compatibility.
                    fired_indices = cp.where(fired_this_step)[0]

                    if fired_indices.size > 0:
                        v_new[fired_indices] = self.cp_izh_c_reset[fired_indices]
                        u_new[fired_indices] += self.cp_izh_d_increment[fired_indices]
                        self.cp_refractory_timers[fired_indices] = cfg.refractory_period_steps

                    self.cp_membrane_potential_v[:] = v_new
                    self.cp_recovery_variable_u[:] = u_new
                    self.cp_refractory_timers[self.cp_refractory_timers > 0] -= 1

            elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                total_input_current_uA_density_equivalent = total_input_current_pA * 1e-6

                # Use pre-computed Q10 temperature factor for extended currents
                # (Main HH kernel computes phi internally; extended currents need it passed explicitly)
                hh_phi = self._cached_hh_phi

                # Apply multiplicative conductance noise (intrinsic channel noise)
                g_Na_effective = self.cp_hh_g_Na_max
                g_K_effective = self.cp_hh_g_K_max
                
                if cfg.enable_conductance_noise:
                    # NOTE: RNG was seeded once at initialization. Per-step seeding removed
                    # for performance. Reproducibility maintained through initial seed.

                    # Multiplicative noise: g_noisy = g_nominal * (1 + noise_std * N(0,1))
                    noise_Na = cp.random.randn(n_neurons).astype(cp.float32)
                    noise_K = cp.random.randn(n_neurons).astype(cp.float32)
                    
                    g_Na_effective = self.cp_hh_g_Na_max * (1.0 + cfg.conductance_noise_relative_std * noise_Na)
                    g_K_effective = self.cp_hh_g_K_max * (1.0 + cfg.conductance_noise_relative_std * noise_K)
                    
                    # Clip to prevent negative conductances
                    g_Na_effective = cp.maximum(g_Na_effective, 0.0)
                    g_K_effective = cp.maximum(g_K_effective, 0.0)

                # Start from synaptic/external input current density
                effective_input_uA = total_input_current_uA_density_equivalent

                # Optional slow K+ M-current: treated as part of ionic current by subtracting I_M from I_syn
                if cfg.hh_g_M_max != 0.0:
                    m_act_new, I_M = fused_hh_m_current_update(
                        self.cp_membrane_potential_v,
                        self.cp_hh_m_current_activation,
                        dt,
                        cfg.hh_g_M_max,
                        self.cp_hh_E_K,
                        cfg.hh_m_current_tau_ms,
                        hh_phi
                    )
                    self.cp_hh_m_current_activation[:] = m_act_new
                    effective_input_uA = effective_input_uA - I_M

                # Optional low-threshold Ca2+ current (CaT)
                if cfg.hh_g_CaT_max != 0.0:
                    m_CaT_new, h_CaT_new, I_CaT = fused_hh_CaT_current_update(
                        self.cp_membrane_potential_v,
                        self.cp_hh_CaT_m,
                        self.cp_hh_CaT_h,
                        dt,
                        cfg.hh_g_CaT_max,
                        cfg.hh_E_CaT,
                        hh_phi
                    )
                    self.cp_hh_CaT_m[:] = m_CaT_new
                    self.cp_hh_CaT_h[:] = h_CaT_new
                    effective_input_uA = effective_input_uA - I_CaT

                # Optional hyperpolarization-activated current (I_h)
                if cfg.hh_g_h_max != 0.0:
                    q_new, I_h = fused_hh_h_current_update(
                        self.cp_membrane_potential_v,
                        self.cp_hh_h_current_q,
                        dt,
                        cfg.hh_g_h_max,
                        cfg.hh_E_h,
                        hh_phi
                    )
                    self.cp_hh_h_current_q[:] = q_new
                    effective_input_uA = effective_input_uA - I_h

                # Optional persistent Na+ current (NaP)
                if cfg.hh_g_NaP_max != 0.0:
                    p_new, I_NaP = fused_hh_NaP_current_update(
                        self.cp_membrane_potential_v,
                        self.cp_hh_NaP_activation,
                        dt,
                        cfg.hh_g_NaP_max,
                        self.cp_hh_E_Na,
                        hh_phi
                    )
                    self.cp_hh_NaP_activation[:] = p_new
                    effective_input_uA = effective_input_uA - I_NaP

                # Per-gate Q10 (precomputed phi values cached on bridge)
                v_new, m_new, h_new, n_new = fused_hodgkin_huxley_dynamics_update(
                    self.cp_membrane_potential_v, self.cp_gating_variable_m, self.cp_gating_variable_h, self.cp_gating_variable_n,
                    effective_input_uA, dt,
                    self.cp_hh_C_m, g_Na_effective, g_K_effective, self.cp_hh_g_L,
                    self.cp_hh_E_Na, self.cp_hh_E_K, self.cp_hh_E_L,
                    self._cached_hh_phi_m, self._cached_hh_phi_h, self._cached_hh_phi_n,
                )
                fired_this_step = (self.cp_membrane_potential_v < self.cp_hh_v_peak) & (v_new >= self.cp_hh_v_peak) 

                self.cp_membrane_potential_v[:] = v_new
                self.cp_gating_variable_m[:] = m_new
                self.cp_gating_variable_h[:] = h_new
                self.cp_gating_variable_n[:] = n_new

            elif cfg.neuron_model_type == NeuronModel.ADEX.name:
                v_new, w_new = fused_adex_dynamics_update(
                    self.cp_membrane_potential_v, self.cp_adex_w,
                    total_input_current_pA, dt,
                    cfg.adex_C, cfg.adex_g_L, cfg.adex_E_L,
                    cfg.adex_V_T, cfg.adex_Delta_T, cfg.adex_a, cfg.adex_tau_w
                )
                not_in_refractory = (self.cp_refractory_timers <= 0)
                fired_this_step = (v_new >= cfg.adex_V_peak) & not_in_refractory
                fired_indices = cp.where(fired_this_step)[0]

                if fired_indices.size > 0:
                    v_new[fired_indices] = cfg.adex_V_r
                    w_new[fired_indices] += cfg.adex_b
                    self.cp_refractory_timers[fired_indices] = cfg.refractory_period_steps

                self.cp_membrane_potential_v[:] = v_new
                self.cp_adex_w[:] = w_new
                self.cp_refractory_timers[self.cp_refractory_timers > 0] -= 1

            elif cfg.neuron_model_type == NeuronModel.RESONATE_AND_FIRE.name:
                # Resonate-and-fire (Izhikevich 2001; Frady & Sommer 2019): a complex-state phasor neuron.
                # State Z = re + i*im REUSES v (=re) and u (=im). Each step Z *= exp(lambda + i*omega), i.e.
                #   re' = e^lambda (re*cos w - im*sin w);  im' = e^lambda (re*sin w + im*cos w).
                # A spike fires at the first UPWARD zero-crossing of im (prev_im < 0, im >= 0) with |Z| > floor;
                # the spike step counted from rf_kick() encodes the kick's phase (magnitude-invariant readout).
                # Opt-in via rf_kick()/rf_read_phases(); Izhikevich/HH/AdEx paths above are untouched.
                # See docs/plans/2026-06-05-rf-on-bridge-derisk-design.md.
                if getattr(self, "cp_rf_prev_im", None) is None:
                    # Stepped before rf_kick(): lazily initialize trackers from the current state (no spikes).
                    self.cp_rf_prev_im = self.cp_recovery_variable_u.copy()
                    self.cp_rf_fired = cp.zeros(n_neurons, dtype=bool)
                    self.cp_rf_spike_step = cp.full(n_neurons, int(getattr(self, "_rf_period", 1000)), dtype=cp.int64)
                    self._rf_counter = 0
                fired_this_step = self._rf_advance_one()

            self.cp_firing_states[:] = fired_this_step

            # Engram tagging (catalog D.14): auto-accumulate spike counts
            # for any active recordings. Zero overhead when no recordings.
            self._tick_engram_recordings()

            # Activity-driven transmission-gate couplings (bridge-internal thalamocortical loop): a control
            # pool's firing opens/closes its coupled cortical route gate. Zero overhead when none registered.
            self._apply_gate_couplings()

            # Combine spike count + any() into a single GPU reduction.
            # cp.sum(bool_array) gives spike count; > 0 gives _fired_any — one kernel, one sync.
            spike_count_gpu = cp.sum(fired_this_step)
            _fired_any = bool(spike_count_gpu > 0)

            # Accumulate spike count on GPU, sync to CPU periodically
            if self._accumulated_spikes_gpu is None:
                self._accumulated_spikes_gpu = spike_count_gpu
            else:
                self._accumulated_spikes_gpu += spike_count_gpu

            self._stats_sync_counter += 1
            if self._stats_sync_counter >= self.gpu_config.stats_sync_interval_steps:
                self._mock_num_spikes_this_step = int(self._accumulated_spikes_gpu) // self._stats_sync_counter
                self._last_synced_spike_count = self._mock_num_spikes_this_step
                self._accumulated_spikes_gpu = None
                self._stats_sync_counter = 0

                # Debug mode: check for numerical issues
                if self.gpu_config.enable_debug_checks:
                    if cp.any(cp.isnan(self.cp_membrane_potential_v)) or cp.any(cp.isinf(self.cp_membrane_potential_v)):
                        self._log_to_ui("WARNING: NaN/Inf detected in membrane potential!", "critical")
            else:
                # Use last synced value between syncs
                self._mock_num_spikes_this_step = self._last_synced_spike_count

            if self.cp_viz_activity_timers is not None:
                max_highlight_val = opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7) if OPENGL_AVAILABLE else 7
                self.cp_viz_activity_timers = cp.where(fired_this_step,
                                                       max_highlight_val, 
                                                       self.cp_viz_activity_timers) 

            if OPENGL_AVAILABLE and opengl_viz_config.get("ENABLE_SYNAPTIC_PULSES", False) and \
               self.cp_synapse_pulse_timers is not None and _fired_any:
                if self.cp_connections is not None and self.cp_connections.nnz > 0:
                    coo_matrix_for_pulses = self._get_cached_coo()  # Use cached COO
                    presynaptic_fired_mask_for_pulses = fired_this_step[coo_matrix_for_pulses.row]
                    synapses_to_activate_indices = cp.where(presynaptic_fired_mask_for_pulses)[0]

                    if synapses_to_activate_indices.size > 0:
                        pulse_lifetime = opengl_viz_config.get("SYNAPTIC_PULSE_MAX_LIFETIME_FRAMES", 5)
                        self.cp_synapse_pulse_timers[synapses_to_activate_indices] = pulse_lifetime 
                        self.cp_synapse_pulse_progress[synapses_to_activate_indices] = 0.0 

            if _profiling: _backend_synchronize(); _prof['t_dyn'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()

            # Experiment-phase plasticity gating: if an experiment is running,
            # respect the current phase's enable_plasticity flag (e.g. testing phases disable plasticity).
            _plasticity_gated = True  # Default: plasticity allowed
            _homeostasis_gated = True  # Default: homeostasis allowed
            if self.experiment_engine is not None and self.experiment_engine.is_experiment_running:
                _plasticity_gated = self.experiment_engine.plasticity_enabled_this_phase
                # Disable homeostasis during ALL experiment phases. Rationale: homeostatic
                # plasticity in vivo operates on hours-to-days timescales (Turrigiano 2008),
                # far slower than the seconds-long experiments here. With EMA tau ≈ 5s, it
                # actively opposes learning — e.g., US neurons driven at 125 Hz during training
                # get their thresholds raised, masking STDP-strengthened CS→US pathways in post-test.
                _homeostasis_gated = False

            # --- 4. Hebbian Learning (Long-Term Potentiation/Depression) ---
            if _plasticity_gated and cfg.enable_hebbian_learning and self.cp_connections.nnz > 0 and \
               self.cp_connections.data is not None and self.cp_connections.data.size > 0:
                if _prev_any and _fired_any:
                    coo_matrix_heb = self._get_cached_coo()  # Use cached COO
                    pre_fired_mask_heb = self.cp_prev_firing_states[coo_matrix_heb.row] 
                    post_fired_mask_heb = fired_this_step[coo_matrix_heb.col] 

                    active_synapse_indices_heb = cp.where(pre_fired_mask_heb & post_fired_mask_heb)[0]
                    num_potentiation_events = 0
                    if active_synapse_indices_heb.size > 0:
                        base_weights_data_array = self.cp_connections.data
                        current_weights_active_syn = base_weights_data_array[active_synapse_indices_heb]
                        delta_weights = cfg.hebbian_learning_rate * (cfg.hebbian_max_weight - current_weights_active_syn)
                        # Per-pathway plasticity gain (Stage 1, 2026-04-27)
                        if self.cp_plasticity_rate_gain is not None:
                            delta_weights = delta_weights * self.cp_plasticity_rate_gain[active_synapse_indices_heb]
                        base_weights_data_array[active_synapse_indices_heb] += delta_weights
                        num_potentiation_events = active_synapse_indices_heb.size

                    # Skip global weight decay during experiments: over 50K training steps,
                    # decay (1-1e-5)^50000 ≈ 0.61 destroys 40% of non-STDP-reinforced weights,
                    # collapsing network baseline excitability by post-test.
                    _experiment_running = (self.experiment_engine is not None and
                                           self.experiment_engine.is_experiment_running)
                    if not _experiment_running:
                        # Per-pathway plasticity gain: decay rate scales with gain.
                        # gain=0 → no decay (frozen pathway preserves weights);
                        # gain=1 → full decay (current behavior).
                        if self.cp_plasticity_rate_gain is not None:
                            gated_decay = cfg.hebbian_weight_decay * self.cp_plasticity_rate_gain
                            self.cp_connections.data *= (1.0 - gated_decay)
                        else:
                            self.cp_connections.data *= (1.0 - cfg.hebbian_weight_decay)
                    cp.clip(self.cp_connections.data, cfg.hebbian_min_weight, cfg.hebbian_max_weight, out=self.cp_connections.data)
                    if num_potentiation_events > 0: self._mock_total_plasticity_events += num_potentiation_events
            
            # --- 4b. C2: STDP (Spike-Timing-Dependent Plasticity) ---
            # Always update last spike times regardless of plasticity gating, so STDP
            # has valid timing data when plasticity re-enables (e.g. training phase starts).
            if cfg.enable_stdp and self.cp_last_spike_time is not None and _fired_any:
                self.cp_last_spike_time = cp.where(
                    fired_this_step,
                    self.runtime_state.current_time_ms,
                    self.cp_last_spike_time
                )

            if _plasticity_gated and cfg.enable_stdp and self.cp_last_spike_time is not None and self.cp_connections.nnz > 0:
                current_time = self.runtime_state.current_time_ms

                # Apply STDP updates — ONLY for synapses connected to neurons that just fired.
                # This is the key optimization: instead of computing delta_t for ALL synapses
                # and filtering, we pre-filter to synapses where pre OR post neuron fired this step.
                # At typical firing rates (2-10 Hz), this reduces the working set from ~1M to ~1-10K.
                if _fired_any:
                    coo_matrix_stdp = self._get_cached_coo()  # Use cached COO

                    # Pre-filter: only synapses where pre or post neuron fired THIS step
                    pre_fired_now = fired_this_step[coo_matrix_stdp.row]
                    post_fired_now = fired_this_step[coo_matrix_stdp.col]
                    candidate_mask = pre_fired_now | post_fired_now
                    candidate_indices = cp.where(candidate_mask)[0]

                    if candidate_indices.size > 0:
                        # Get spike times only for candidate synapses
                        pre_spike_times = self.cp_last_spike_time[coo_matrix_stdp.row[candidate_indices]]
                        post_spike_times = self.cp_last_spike_time[coo_matrix_stdp.col[candidate_indices]]

                        # Calculate spike timing differences (t_post - t_pre)
                        delta_t = post_spike_times - pre_spike_times

                        # Only update synapses where both neurons have spiked (not at initial value)
                        valid_pairs_mask = (pre_spike_times > -500.0) & (post_spike_times > -500.0)

                        # Apply STDP time window constraint
                        stdp_window_ms = max(cfg.stdp_tau_plus_ms, cfg.stdp_tau_minus_ms) * 5.0
                        within_window_mask = (cp.abs(delta_t) < stdp_window_ms) & valid_pairs_mask

                        stdp_local_indices = cp.where(within_window_mask)[0]

                        if stdp_local_indices.size > 0:
                            # Map back to global synapse indices
                            stdp_active_indices = candidate_indices[stdp_local_indices]

                            # Apply STDP weight updates using fused kernel
                            current_weights = self.cp_connections.data[stdp_active_indices]
                            delta_t_active = delta_t[stdp_local_indices]

                            updated_weights = fused_stdp_weight_update(
                                delta_t_active,
                                current_weights,
                                cfg.stdp_a_plus,
                                cfg.stdp_a_minus,
                                cfg.stdp_tau_plus_ms,
                                cfg.stdp_tau_minus_ms,
                                cfg.stdp_w_min,
                                cfg.stdp_w_max
                            )

                            # Respect per-synapse plastic mask if set —
                            # research runners (G2+) use this to freeze
                            # reservoir weights while training input pathways.
                            if self.cp_synapse_plastic_mask is not None:
                                plastic_here = self.cp_synapse_plastic_mask[stdp_active_indices]
                                updated_weights = cp.where(
                                    plastic_here, updated_weights, current_weights
                                )

                            # Per-pathway plasticity gain (Stage 1, 2026-04-27).
                            # When set, scales STDP weight delta by gain in
                            # [0,1]. gain=0 → frozen pathway: no STDP changes
                            # AND eligibility doesn't accumulate (since the
                            # weight_changes seen below is now zero).
                            # gain=1 → full plasticity. Multiplied with
                            # plastic_mask: a non-plastic synapse stays
                            # frozen regardless of gain.
                            if self.cp_plasticity_rate_gain is not None:
                                gain_here = self.cp_plasticity_rate_gain[stdp_active_indices]
                                weight_changes_gated = (updated_weights - current_weights) * gain_here
                                updated_weights = current_weights + weight_changes_gated

                            self.cp_connections.data[stdp_active_indices] = updated_weights

                            # Update eligibility traces if reward modulation is enabled.
                            # SIGNED eligibility (this branch): accumulate the
                            # raw STDP weight change, preserving LTP (+) vs LTD (-)
                            # direction. Positive reward then selectively
                            # potentiates recently-LTP pairings and depresses
                            # recently-LTD pairings; negative reward flips that.
                            # The original unsigned version (`+= cp.abs(...)`)
                            # uniformly boosted or depressed all recently-plastic
                            # synapses, which made reward modulation path-
                            # agnostic and caused the G5.v2 degenerate attractor
                            # (see research/findings/2026-04-20-g5v2.md).
                            #
                            # Gating: weight_changes here is the post-gain
                            # delta (already 0 if pathway is frozen), so the
                            # eligibility trace correctly reflects what
                            # actually happened to weights.
                            if cfg.enable_reward_modulation and self.cp_eligibility_trace is not None:
                                weight_changes = updated_weights - current_weights
                                self.cp_eligibility_trace[stdp_active_indices] += weight_changes

                            self._mock_total_plasticity_events += stdp_active_indices.size

            # --- 4c0. Neuromodulator subsystem update (Session E.1, opt-in) ---
            # Run NM production+decay BEFORE reward modulation so this step's
            # reward signal drives this step's NM concentration changes (e.g.,
            # pause_on_reward -> ACh pause -> plasticity_window_gate opens) AT
            # THE SAME STEP. This is required for fast-dynamics gates
            # (TAN/ACh) where the pause and the reward are the same event.
            #
            # Previously (pre-2026-04-28-bugfix), manager.step ran AFTER the
            # reward block, which created a one-step lag: with single-pulse
            # rewards, the gate never opened during the reward delivery.
            # See research/findings/2026-04-28-cluster-b3-tans-results.md
            # for the empirical regression that exposed this.
            #
            # Note: synaptic_gain and excitability_drive are read earlier in
            # the step (around the synaptic conductance and total input
            # current sections); those continue to use the previous step's
            # NM concentrations. That one-step lag is biologically harmless
            # for slow-dynamics modulators (DA tonic, NE) and matches the
            # prior behavior — only the reward-time plasticity path is
            # affected by this reordering.
            if (getattr(cfg, "enable_neuromodulator_subsystem", False)
                    and self.neuromodulator_manager is not None):
                self.neuromodulator_manager.step(self)
                # Propagate any NM-driven plasticity gate values to the
                # bridge's per-pathway gain. Biological grounding:
                # developmental NM ramps modulate critical periods; DA
                # gates corticostriatal LTP; ACh gates attentional cortex
                # plasticity. The gate value = NM concentration after
                # baseline+sensitivity scaling (see compute_plasticity_gate_values).
                if self._plasticity_gate_to_synapses:
                    nm_gates = self.neuromodulator_manager.compute_plasticity_gate_values()
                    for gate_name, gate_value in nm_gates.items():
                        if gate_name in self._plasticity_gate_to_synapses:
                            # Only update if value changed materially (avoid GPU writes)
                            current = self._plasticity_gate_values.get(gate_name, 1.0)
                            if abs(gate_value - current) > 1e-4:
                                self.set_plasticity_gate(gate_name, gate_value)

            # --- 4c. C2: Reward-Modulated Plasticity (Three-Factor Learning) ---
            if _plasticity_gated and cfg.enable_reward_modulation and self.cp_eligibility_trace is not None and self.cp_connections.nnz > 0:
                # Decay eligibility traces
                decay_factor = cp.exp(-dt / cfg.reward_eligibility_tau_ms)
                self.cp_eligibility_trace = fused_eligibility_trace_decay(
                    self.cp_eligibility_trace,
                    decay_factor
                )
                
                # Apply reward modulation if reward signal is non-zero.
                # R2.4 (2026-04-29): aversive-vs-appetitive magnitude
                # asymmetry. Per Schultz98/Schultz16, aversive responses
                # are observed as DEPRESSIONS below tonic DA, of smaller
                # magnitude than appetitive activations. With the D1/D2
                # sign array (R1.1), the qualitative asymmetry of LTP-vs-LTD
                # already follows from the signed scalar; we additionally
                # scale negative reward_prediction_error by the configured
                # reward_aversive_scale (default 0.5) so the magnitude of
                # negative-reward plasticity is reduced relative to positive.
                reward_prediction_error = cfg.current_reward_signal - cfg.reward_baseline
                if reward_prediction_error < 0.0:
                    aversive_scale = float(getattr(cfg, "reward_aversive_scale", 0.5))
                    reward_prediction_error = reward_prediction_error * aversive_scale

                # Cluster C v1 (2026-04-29): tonic-DA path. When the
                # neuromodulator subsystem is on AND a "dopamine" modulator
                # is registered, use the DA concentration's deviation-from-
                # baseline as the plasticity signal instead of the raw
                # signed-scalar reward_prediction_error. DA's from_reward
                # production rule handles the phasic dynamics; tonic baseline
                # gives a non-zero plasticity signal between rewards (which
                # ACh's plasticity_window_gate can then modulate). Falls back
                # to legacy path when no DA modulator is registered.
                #
                # Cluster C v2 (2026-04-29): when 4 per-action DA modulators
                # (dopamine_{N,E,S,W}) are registered AND cp_synapse_action_tag
                # is populated, use compartmentalized per-synapse DA signal
                # instead of scalar. Each synapse i gets DA[tag[i]] - baseline
                # so per-action DA channels target only their action's
                # synapses.
                da_signal = None
                per_synapse_da = None  # cp.ndarray when v2 active; else None
                if (getattr(cfg, "enable_neuromodulator_subsystem", False)
                        and self.neuromodulator_manager is not None):
                    # v2 takes precedence: check if 4 per-action DA modulators are registered.
                    nm_names = self.neuromodulator_manager.modulator_names()
                    per_action_names = ["dopamine_N", "dopamine_E", "dopamine_S", "dopamine_W"]
                    if (all(n in nm_names for n in per_action_names)
                            and self.cp_synapse_action_tag is not None):
                        actual_nnz_for_da = self.cp_connections.nnz
                        per_synapse_da = self.neuromodulator_manager.compute_per_synapse_da_signal(
                            self.cp_synapse_action_tag[:actual_nnz_for_da],
                            action_modulator_names=per_action_names,
                        )
                    else:
                        # v1 path: single-channel "dopamine" scalar
                        try:
                            da_conc = self.neuromodulator_manager.get_concentration("dopamine")
                            da_baseline = next(
                                (c.baseline for c in self.neuromodulator_manager._configs
                                 if c.name == "dopamine"),
                                None,
                            )
                            if da_baseline is not None:
                                da_signal = float(da_conc) - float(da_baseline)
                        except KeyError:
                            da_signal = None  # dopamine not registered; legacy path
                effective_signal = da_signal if da_signal is not None else reward_prediction_error
                # Decide whether to enter the update path. With per_synapse_da,
                # we always enter (the array can have nonzero entries even when
                # the scalar effective_signal is zero, since per-action DA
                # baselines/concentrations may differ).
                update_path_active = (
                    (per_synapse_da is not None) or (abs(effective_signal) > 1e-6)
                )
                if update_path_active:
                    # Effective lr is reward_learning_rate × neuromod plasticity_rate
                    # multiplier (subsystem off → multiplier 1.0, no change).
                    effective_reward_lr = cfg.reward_learning_rate
                    if (getattr(cfg, "enable_neuromodulator_subsystem", False)
                            and self.neuromodulator_manager is not None):
                        effective_reward_lr *= (
                            self.neuromodulator_manager.compute_plasticity_rate_multiplier()
                        )

                    # Modulate weights based on eligibility trace and reward
                    # Delta_w = learning_rate * reward_error * eligibility_trace
                    # Slice eligibility trace to match actual synapse count (trace array
                    # is pre-allocated to capacity which may exceed cp_connections.nnz).
                    actual_nnz = self.cp_connections.nnz
                    if per_synapse_da is not None:
                        # Cluster C v2: per-synapse DA signal. The per-synapse
                        # multiplier from compute_per_synapse_plasticity_rate_multiplier
                        # could also be applied here for additional scope-action targets,
                        # but the canonical path is the DA concentration deviation
                        # already encoded in per_synapse_da.
                        weight_updates = (
                            effective_reward_lr
                            * per_synapse_da
                            * self.cp_eligibility_trace[:actual_nnz]
                        )
                    elif self.cp_per_synapse_reward_override is not None:
                        # E.3 batched-replica framework: each replica gets its
                        # own reward signal via this per-synapse override array.
                        # Replaces the scalar effective_signal with a per-synapse
                        # value (each replica's synapse block populated with that
                        # replica's reward_prediction_error). Subsequent multipliers
                        # (cp_d1_d2_sign, cp_plasticity_rate_gain, tan_gate)
                        # still apply normally.
                        weight_updates = (
                            effective_reward_lr
                            * self.cp_per_synapse_reward_override[:actual_nnz]
                            * self.cp_eligibility_trace[:actual_nnz]
                        )
                    else:
                        weight_updates = effective_reward_lr * effective_signal * self.cp_eligibility_trace[:actual_nnz]
                    # Per-pathway plasticity gain (Stage 1, 2026-04-27): gate
                    # the eligibility-to-weight conversion. A pathway frozen
                    # NOW won't accept reward-driven changes from past
                    # eligibility; this is the standard 3-factor rule
                    # interpretation (DA/NM gates the learning event itself).
                    if self.cp_plasticity_rate_gain is not None:
                        weight_updates = weight_updates * self.cp_plasticity_rate_gain[:actual_nnz]
                    # Cluster B.1 (2026-04-28): D1/D2 plasticity asymmetry.
                    # D2-targeting synapses move opposite to reward direction;
                    # D1-targeting + everything else move with reward.
                    if self.cp_d1_d2_sign is not None:
                        weight_updates = weight_updates * self.cp_d1_d2_sign[:actual_nnz]
                    # Cluster B.3 (2026-04-28): cholinergic (TAN) plasticity
                    # window gate. When ACh is at tonic baseline the gate ~ 0
                    # and reward-driven weight changes are suppressed; when
                    # ACh pauses (concentration drops below baseline) the gate
                    # rises toward 1 and weight changes are permitted. Scalar
                    # multiplier; subsystem off / no plasticity_window_gate
                    # targets -> returns 1.0 (no-op, bit-identical).
                    if (getattr(cfg, "enable_neuromodulator_subsystem", False)
                            and self.neuromodulator_manager is not None):
                        tan_gate = self.neuromodulator_manager.compute_plasticity_window_gate_multiplier()
                        if tan_gate != 1.0:
                            weight_updates = weight_updates * tan_gate
                    self.cp_connections.data += weight_updates
                    
                    # Clip to bounds (use STDP bounds if STDP is enabled, otherwise Hebbian bounds)
                    w_min = cfg.stdp_w_min if cfg.enable_stdp else cfg.hebbian_min_weight
                    w_max = cfg.stdp_w_max if cfg.enable_stdp else cfg.hebbian_max_weight
                    cp.clip(self.cp_connections.data, w_min, w_max, out=self.cp_connections.data)
                    
                    # Count significant updates
                    significant_updates = cp.sum(cp.abs(weight_updates) > 1e-6)
                    if significant_updates > 0:
                        self._mock_total_plasticity_events += int(significant_updates)

            # --- 4d. C3: Structural Plasticity (Synapse Formation/Elimination) ---
            # Freeze structural plasticity during experiments: synaptogenesis operates on
            # hours-to-days timescales in vivo (Holtmaat & Svoboda 2009). In a 50-second
            # experiment, activity-biased formation adds hundreds of thousands of random
            # synapses that dilute learned CS→US pathways and alter network dynamics.
            _structural_plasticity_active = _plasticity_gated and not (
                self.experiment_engine is not None and self.experiment_engine.is_experiment_running
            )
            if _structural_plasticity_active and cfg.enable_structural_plasticity and self.cp_struct_plast_step_counter is not None:
                self.cp_struct_plast_step_counter += 1
                
                # Only update periodically for efficiency
                if self.cp_struct_plast_step_counter >= cfg.struct_plast_update_interval_steps:
                    self.cp_struct_plast_step_counter = 0
                    
                    # Synapse elimination: remove weak synapses
                    weak_synapse_mask = self.cp_connections.data < cfg.struct_plast_weight_threshold
                    # int() works on both cupy 0-d arrays and numpy scalars
                    num_weak = int(cp.sum(weak_synapse_mask))
                    
                    if num_weak > 0:
                        # Probabilistic elimination based on elimination rate
                        # Rate is per-synapse-per-timestep, so scale by update interval
                        elimination_prob = cfg.struct_plast_elimination_rate * cfg.struct_plast_update_interval_steps
                        elimination_prob = min(elimination_prob, 0.5)  # Cap at 50% per update
                        
                        # Generate random numbers for each weak synapse
                        eliminate_mask = weak_synapse_mask & (cp.random.rand(self.cp_connections.nnz) < elimination_prob)
                        num_eliminated = int(cp.sum(eliminate_mask))
                        
                        if num_eliminated > 0:
                            # DON'T filter synapse arrays here - defer to compaction
                            # This keeps arrays aligned with CSR.data during the deferred window

                            # Set eliminated synapses to zero weight (STP multiplication will yield 0 anyway)
                            self.cp_connections.data[eliminate_mask] = 0.0

                            # Mark that we have pending zero-weight synapses
                            self._pending_eliminations = True

                            # Invalidate COO cache since connectivity changed
                            self._invalidate_coo_cache()

                    # Deferred CSR compaction: only rebuild periodically to amortize cost
                    self._compaction_counter += 1
                    if self._pending_eliminations and self._compaction_counter >= self.gpu_config.struct_plast_compaction_interval:
                        # Filter synapse arrays BEFORE eliminate_zeros() to maintain alignment
                        # keep_mask identifies entries with non-zero weight
                        keep_mask = (self.cp_connections.data != 0)

                        # Compact all synapse-indexed arrays
                        self._compact_synapse_arrays(keep_mask)

                        # Now compact the CSR matrix
                        self.cp_connections.eliminate_zeros()
                        self._pending_eliminations = False
                        self._compaction_counter = 0
                        self._invalidate_coo_cache()
                    
                    # Synapse formation: create new connections
                    current_density = self.cp_connections.nnz / (n_neurons * n_neurons)
                    
                    if current_density < cfg.struct_plast_target_density:
                        # Calculate number of new synapses to add
                        target_synapses = int(cfg.struct_plast_target_density * n_neurons * n_neurons)
                        current_synapses = self.cp_connections.nnz
                        potential_new = target_synapses - current_synapses
                        
                        if potential_new > 0:
                            # Formation rate per neuron pair per timestep, scaled by update interval
                            formation_prob = cfg.struct_plast_formation_rate * cfg.struct_plast_update_interval_steps
                            expected_new_synapses = int(potential_new * formation_prob)
                            expected_new_synapses = max(1, min(expected_new_synapses, n_neurons * 10))  # Form at least 1, cap at 10*N

                            # Generate candidate new connections on GPU
                            # Activity-dependent synaptogenesis (Cline & Haas 2008):
                            # Bias formation toward co-active neuron pairs using activity EMA.
                            activity_bias = cfg.struct_plast_activity_bias
                            n_candidates = expected_new_synapses * 3

                            if activity_bias > 0.0 and self.cp_neuron_activity_ema is not None:
                                # Number of activity-biased vs random candidates
                                n_biased = int(n_candidates * activity_bias)
                                n_random = n_candidates - n_biased

                                # Activity-biased: sample neurons proportional to their firing EMA
                                ema = self.cp_neuron_activity_ema + 1e-9  # avoid all-zero
                                ema_probs = ema / ema.sum()
                                ema_probs_np = _backend_to_host(ema_probs).astype(np.float64)
                                ema_probs_np /= ema_probs_np.sum()  # renormalize for float64 precision
                                # Sample active neurons as both pre and post (co-active pairs)
                                biased_pre_np = np.random.choice(n_neurons, size=n_biased, p=ema_probs_np)
                                biased_post_np = np.random.choice(n_neurons, size=n_biased, p=ema_probs_np)
                                biased_pre = cp.asarray(biased_pre_np, dtype=cp.int64)
                                biased_post = cp.asarray(biased_post_np, dtype=cp.int64)

                                # Random candidates (preserve exploration)
                                random_pre = cp.random.randint(0, n_neurons, size=n_random, dtype=cp.int64)
                                random_post = cp.random.randint(0, n_neurons, size=n_random, dtype=cp.int64)

                                candidate_pre = cp.concatenate([biased_pre, random_pre])
                                candidate_post = cp.concatenate([biased_post, random_post])
                            else:
                                candidate_pre = cp.random.randint(0, n_neurons, size=n_candidates, dtype=cp.int64)
                                candidate_post = cp.random.randint(0, n_neurons, size=n_candidates, dtype=cp.int64)

                            # Filter out self-connections on GPU
                            valid_mask = candidate_pre != candidate_post
                            candidate_pre = candidate_pre[valid_mask]
                            candidate_post = candidate_post[valid_mask]

                            if candidate_pre.size > 0:
                                # GPU-based duplicate checking using unique pair IDs
                                # Encode (pre, post) pairs as unique integers: pre * n_neurons + post
                                candidate_ids = candidate_pre * n_neurons + candidate_post

                                # Get existing pair IDs from COO matrix
                                coo_existing = self._get_cached_coo()
                                if coo_existing is not None:
                                    existing_ids = coo_existing.row.astype(cp.int64) * n_neurons + coo_existing.col.astype(cp.int64)
                                    # Find candidates that don't exist in current connections
                                    is_duplicate = cp.isin(candidate_ids, existing_ids)
                                    new_mask = ~is_duplicate
                                else:
                                    new_mask = cp.ones(candidate_ids.shape[0], dtype=cp.bool_)

                                # Also remove duplicates within candidates
                                candidate_ids_filtered = candidate_ids[new_mask]
                                if candidate_ids_filtered.size > 0:
                                    unique_ids, unique_indices = cp.unique(candidate_ids_filtered, return_index=True)
                                    # Limit to expected number of new synapses
                                    if unique_ids.size > expected_new_synapses:
                                        unique_indices = unique_indices[:expected_new_synapses]
                                        unique_ids = unique_ids[:expected_new_synapses]

                                    # Decode back to (pre, post) pairs
                                    new_pre = (unique_ids // n_neurons).astype(cp.int32)
                                    new_post = (unique_ids % n_neurons).astype(cp.int32)
                                else:
                                    new_pre = cp.array([], dtype=cp.int32)
                                    new_post = cp.array([], dtype=cp.int32)
                            else:
                                new_pre = cp.array([], dtype=cp.int32)
                                new_post = cp.array([], dtype=cp.int32)

                            if new_pre.size > 0:
                                
                                # Calculate distance-dependent initial weights
                                if cfg.struct_plast_distance_kernel == "exp_decay":
                                    pre_pos = self.cp_neuron_positions_3d[new_pre]
                                    post_pos = self.cp_neuron_positions_3d[new_post]
                                    distances = cp.linalg.norm(pre_pos - post_pos, axis=1)
                                    distance_factor = cp.exp(-distances / cfg.struct_plast_distance_scale)
                                elif cfg.struct_plast_distance_kernel == "gaussian":
                                    pre_pos = self.cp_neuron_positions_3d[new_pre]
                                    post_pos = self.cp_neuron_positions_3d[new_post]
                                    distances = cp.linalg.norm(pre_pos - post_pos, axis=1)
                                    distance_factor = cp.exp(-(distances ** 2) / (2.0 * cfg.struct_plast_distance_scale ** 2))
                                else:  # uniform
                                    distance_factor = cp.ones(new_pre.size, dtype=cp.float32)

                                # Initial weights scaled by distance
                                initial_weights = cfg.struct_plast_weight_threshold * 2.0 * distance_factor

                                # Create new sparse matrix with added connections
                                new_connections_matrix = csp.csr_matrix(
                                    (initial_weights, (new_pre, new_post)),
                                    shape=(n_neurons, n_neurons),
                                    dtype=cp.float32
                                )

                                # Add to existing connections
                                nnz_before = self.cp_connections.nnz
                                self.cp_connections = self.cp_connections + new_connections_matrix

                                # CSR addition deduplicates overlapping (pre,post) pairs by summing
                                # their weights, so actual new synapses may be fewer than candidates.
                                actual_new = self.cp_connections.nnz - nnz_before

                                # Invalidate COO cache since connectivity changed
                                self._invalidate_coo_cache()

                                # Update synapse arrays only for actually added synapses
                                if actual_new > 0:
                                    self._grow_synapse_arrays_if_needed(actual_new, cfg)
                                    self._add_synapses_to_arrays(actual_new, cfg)

                                # Keep _synapse_count in sync with actual connection matrix
                                self._synapse_count = self.cp_connections.nnz

            if _profiling: _backend_synchronize(); _prof['t_plast'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()
            # --- 5. Homeostatic Plasticity (gated separately from learning plasticity) ---
            # 5a. Adaptive thresholds (Izhikevich-specific)
            if _homeostasis_gated and cfg.enable_homeostasis and self.cp_neuron_firing_thresholds is not None:
                if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                    self.cp_neuron_activity_ema, self.cp_neuron_firing_thresholds = fused_homeostasis_update(
                        self.cp_neuron_activity_ema, fired_this_step.astype(cp.float32),
                        cfg.homeostasis_target_rate, cfg.homeostasis_ema_alpha, cfg.homeostasis_threshold_adapt_rate,
                        self.cp_neuron_firing_thresholds,
                        cfg.homeostasis_threshold_min, cfg.homeostasis_threshold_max
                    )
                elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                     self.cp_neuron_activity_ema = (1.0 - cfg.homeostasis_ema_alpha) * self.cp_neuron_activity_ema + \
                                               cfg.homeostasis_ema_alpha * fired_this_step.astype(cp.float32)

            # 5b. Synaptic scaling (Turrigiano 2008) — works for all neuron models
            # Multiplicatively scales excitatory synaptic weights to maintain target firing rate.
            # scale_factor = 1 + rate * (target - actual_ema) per postsynaptic neuron
            if _homeostasis_gated and cfg.enable_synaptic_scaling and self.cp_connections is not None and self.cp_connections.nnz > 0:
                # Update EMA if not already done by threshold homeostasis
                if not (cfg.enable_homeostasis and self.cp_neuron_firing_thresholds is not None):
                    self.cp_neuron_activity_ema = (1.0 - cfg.homeostasis_ema_alpha) * self.cp_neuron_activity_ema + \
                                                  cfg.homeostasis_ema_alpha * fired_this_step.astype(cp.float32)
                # Compute per-neuron scaling factor based on firing rate error
                rate_error = cfg.homeostasis_target_rate - self.cp_neuron_activity_ema  # positive = too quiet, scale up
                scale_factors = 1.0 + cfg.synaptic_scaling_rate * rate_error
                scale_factors = cp.clip(scale_factors, 0.95, 1.05)  # Prevent runaway scaling per step
                # Apply to excitatory weights via postsynaptic neuron index (CSR column structure)
                # In CSR format, each row i has connections FROM neuron i. For postsynaptic scaling,
                # we need the target (column) neuron's scale factor applied to the weight.
                coo = self._get_cached_coo()
                if coo is not None and coo.nnz == self.cp_connections.nnz:
                    post_scales = scale_factors[coo.col]
                    # Per-pathway plasticity gain (Stage 1, 2026-04-27): scale
                    # the deviation-from-1 by gain. gain=0 → identity (no
                    # synaptic scaling); gain=1 → full scaling.
                    if self.cp_plasticity_rate_gain is not None:
                        effective_scales = 1.0 + (post_scales - 1.0) * self.cp_plasticity_rate_gain
                        self.cp_connections.data[:] = self.cp_connections.data * effective_scales
                    else:
                        self.cp_connections.data[:] = self.cp_connections.data * post_scales
                    # Enforce weight bounds
                    if cfg.enable_hebbian_learning:
                        cp.clip(self.cp_connections.data, cfg.hebbian_min_weight, cfg.hebbian_max_weight, out=self.cp_connections.data)
                    else:
                        cp.clip(self.cp_connections.data, 0.01, 5.0, out=self.cp_connections.data)

            if _profiling: _backend_synchronize(); _prof['t_homeo'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()
            # --- 6. Prepare for Next Step & Record Frame ---
            self.cp_prev_firing_states[:] = fired_this_step
            self.record_current_frame_if_active() # This was the missing method call's target

            # Publish to data bus if available
            if self.data_bus is not None:
                # int() works on both cupy 0-d arrays and numpy scalars
                n_spikes = int(spike_count_gpu) if _fired_any else 0
                self.data_bus.publish("firing_rates", {
                    "time_ms": self.runtime_state.current_time_ms,
                    "total_spikes": n_spikes,
                    "rate_hz": n_spikes / (n_neurons * dt / 1000.0) if n_neurons > 0 else 0,
                })
                if _fired_any:
                    fired_idx = cp.where(fired_this_step)[0]
                    if fired_idx.size <= 500:  # Cap to avoid huge GPU->CPU transfers
                        self.data_bus.publish("spike_events", {
                            "time_ms": self.runtime_state.current_time_ms,
                            "neuron_indices": _backend_to_host(fired_idx),
                        })

                # Publish weight snapshot (infrequent — every 1000 steps for histogram)
                if not hasattr(self, '_weight_pub_counter'):
                    self._weight_pub_counter = 0
                self._weight_pub_counter += 1
                if self._weight_pub_counter >= 1000 and self.cp_connections is not None and self.cp_connections.nnz > 0:
                    self._weight_pub_counter = 0
                    data = self.cp_connections.data
                    sample_size = min(10000, data.size)
                    if sample_size > 0:
                        indices = cp.random.randint(0, data.size, sample_size)
                        sampled = _backend_to_host(data[indices])
                        self.data_bus.publish("weights", {"weights": sampled})

            # Note: Network firing rate calculation deferred to avoid GPU->CPU sync every step
            # Will be updated on-demand when GUI data is requested

            # ── Synapse tiering (Phase 3 Strategy B) activity tick ──
            # Feed fired-pathway names to the store's eviction policy.
            # No-op if synapse_store is None (default). Cheap enough to
            # run unconditionally when enabled — single host-side bool
            # reduce per pathway (~O(n_post) per pathway, ~30 pathways).
            if self.synapse_store is not None:
                try:
                    fired_pathways = self._detect_fired_pathways(fired_this_step)
                    self.synapse_store.step(fired_pathways)
                except Exception as e:
                    # Don't let tiering bookkeeping kill the sim
                    if not hasattr(self, "_synapse_store_warned"):
                        self._log_console(
                            f"synapse_store.step failed: {e}; "
                            f"tiering bookkeeping disabled this session.",
                            "warning",
                        )
                        self._synapse_store_warned = True

            # Step profiler: accumulate and log summary every 500 steps
            if _profiling:
                _backend_synchronize()
                _prof['t_final'] = _time.perf_counter() - _t0
                if not hasattr(self, '_prof_accum'):
                    self._prof_accum = {k: 0.0 for k in _prof}
                    self._prof_count = 0
                for k, v in _prof.items():
                    self._prof_accum[k] = self._prof_accum.get(k, 0.0) + v
                self._prof_count += 1
                if self._prof_count >= 500:
                    total = sum(self._prof_accum.values())
                    parts = " | ".join(f"{k}={v*1000/self._prof_count:.2f}ms ({v/total*100:.0f}%)"
                                       for k, v in sorted(self._prof_accum.items()))
                    prof_msg = f"[PROFILER] avg/step: {total*1000/self._prof_count:.2f}ms | {parts}"
                    self._log_console(prof_msg)
                    self._log_to_ui(prof_msg, "info")
                    self._prof_accum = {}
                    self._prof_count = 0

        except Exception as e:
            self._log_to_ui(f"Error during simulation step: {e}","critical")
            import traceback; traceback.print_exc()
            self.stop_simulation()
            if self.ui_queue: self.ui_queue.put({"type": "SIM_ERROR_OCCURRED", "error_message": str(e)})
            # Research runners can opt in to loud failure via
            # bridge.strict_step_errors = True. Default keeps the biological
            # experiment UI's load-bearing "single bad step doesn't kill the
            # session" behaviour.
            if getattr(self, "strict_step_errors", False):
                raise
    def save_checkpoint(self, filepath, gui_config_snapshot=None): # Added gui_config_snapshot
        """Saves the current simulation state to an HDF5 checkpoint file."""
        self._log_to_ui(f"Saving checkpoint to {filepath}...", "info")
        if not self.is_initialized:
            self._log_to_ui("Sim not initialized. Cannot save checkpoint.","warning"); return False

        try:
            with h5py.File(filepath, 'w') as h5f:
                config_dict = self.core_config.to_dict()
                save_dict_to_hdf5_attrs(h5f, config_dict)

                state_group = h5f 

                # Note: cp_synapse_pulse_timers and cp_synapse_pulse_progress are synapse-indexed
                # and handled separately with pre-allocation slicing
                arrays_to_save_direct = [
                    'cp_membrane_potential_v', 'cp_conductance_g_e', 'cp_conductance_g_i',
                    'cp_external_input_current', 'cp_firing_states', 'cp_prev_firing_states',
                    'cp_traits', 'cp_refractory_timers', 'cp_neuron_positions_3d',
                    'cp_neuron_activity_ema', 'cp_viz_activity_timers',
                    'cp_adex_w', 'cp_ou_current'
                ]
                for attr_name in arrays_to_save_direct:
                    data_array = getattr(self, attr_name, None)
                    if data_array is not None and data_array.size > 0:
                        state_group.create_dataset(attr_name, data=_backend_to_host(data_array), compression="gzip")
                    elif data_array is not None: 
                         state_group.attrs[f"{attr_name}_is_empty"] = True

                if self.cp_connections is not None:
                    if self.cp_connections.data is not None and self.cp_connections.data.size > 0:
                        state_group.create_dataset("connections_data", data=_backend_to_host(self.cp_connections.data), compression="gzip")
                    if self.cp_connections.indices is not None and self.cp_connections.indices.size > 0:
                        state_group.create_dataset("connections_indices", data=_backend_to_host(self.cp_connections.indices), compression="gzip")
                    if self.cp_connections.indptr is not None and self.cp_connections.indptr.size > 0:
                        state_group.create_dataset("connections_indptr", data=_backend_to_host(self.cp_connections.indptr), compression="gzip")
                    state_group.attrs["connections_shape_0"] = self.cp_connections.shape[0]
                    state_group.attrs["connections_shape_1"] = self.cp_connections.shape[1]

                # Save only active synapse elements (not pre-allocated capacity)
                synapse_count = getattr(self, '_synapse_count', None)
                if self.cp_stp_u is not None and self.cp_stp_u.size > 0:
                    active_stp_u = self.cp_stp_u[:synapse_count] if synapse_count else self.cp_stp_u
                    state_group.create_dataset("cp_stp_u", data=_backend_to_host(active_stp_u), compression="gzip")
                elif self.cp_stp_u is not None: state_group.attrs["cp_stp_u_is_empty"] = True
                if self.cp_stp_x is not None and self.cp_stp_x.size > 0:
                    active_stp_x = self.cp_stp_x[:synapse_count] if synapse_count else self.cp_stp_x
                    state_group.create_dataset("cp_stp_x", data=_backend_to_host(active_stp_x), compression="gzip")
                elif self.cp_stp_x is not None: state_group.attrs["cp_stp_x_is_empty"] = True
                
                # C2: Save STDP and reward modulation state
                if self.cp_last_spike_time is not None and self.cp_last_spike_time.size > 0:
                    state_group.create_dataset("cp_last_spike_time", data=_backend_to_host(self.cp_last_spike_time), compression="gzip")
                elif self.cp_last_spike_time is not None:
                    state_group.attrs["cp_last_spike_time_is_empty"] = True

                # G3: Save per-synapse plastic mask if present (research runners).
                # Sized to cp_connections.nnz, not the pre-allocated capacity.
                if self.cp_synapse_plastic_mask is not None and self.cp_synapse_plastic_mask.size > 0:
                    nnz = self.cp_connections.nnz if self.cp_connections is not None else self.cp_synapse_plastic_mask.size
                    mask_active = self.cp_synapse_plastic_mask[:nnz]
                    state_group.create_dataset("cp_synapse_plastic_mask",
                                               data=_backend_to_host(mask_active).astype(np.bool_),
                                               compression="gzip")
                
                if self.cp_eligibility_trace is not None and self.cp_eligibility_trace.size > 0:
                    active_traces = self.cp_eligibility_trace[:synapse_count] if synapse_count else self.cp_eligibility_trace
                    state_group.create_dataset("cp_eligibility_trace", data=_backend_to_host(active_traces), compression="gzip")
                elif self.cp_eligibility_trace is not None:
                    state_group.attrs["cp_eligibility_trace_is_empty"] = True

                # Save synapse visualization arrays (synapse-indexed with pre-allocation)
                if self.cp_synapse_pulse_timers is not None and self.cp_synapse_pulse_timers.size > 0:
                    active_timers = self.cp_synapse_pulse_timers[:synapse_count] if synapse_count else self.cp_synapse_pulse_timers
                    state_group.create_dataset("cp_synapse_pulse_timers", data=_backend_to_host(active_timers), compression="gzip")
                elif self.cp_synapse_pulse_timers is not None:
                    state_group.attrs["cp_synapse_pulse_timers_is_empty"] = True
                if self.cp_synapse_pulse_progress is not None and self.cp_synapse_pulse_progress.size > 0:
                    active_progress = self.cp_synapse_pulse_progress[:synapse_count] if synapse_count else self.cp_synapse_pulse_progress
                    state_group.create_dataset("cp_synapse_pulse_progress", data=_backend_to_host(active_progress), compression="gzip")
                elif self.cp_synapse_pulse_progress is not None:
                    state_group.attrs["cp_synapse_pulse_progress_is_empty"] = True

                # C3: Save structural plasticity state
                if self.cp_struct_plast_step_counter is not None:
                    state_group.attrs["cp_struct_plast_step_counter"] = self.cp_struct_plast_step_counter

                if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                    if self.cp_recovery_variable_u is not None and self.cp_recovery_variable_u.size > 0: state_group.create_dataset("cp_recovery_variable_u", data=_backend_to_host(self.cp_recovery_variable_u), compression="gzip")
                    elif self.cp_recovery_variable_u is not None : state_group.attrs["cp_recovery_variable_u_is_empty"] = True
                    for param in ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c_reset', 'd_increment']:
                         attr_name_cp = f"cp_izh_{param}"
                         data_array = getattr(self, attr_name_cp, None)
                         if data_array is not None and data_array.size > 0: state_group.create_dataset(attr_name_cp, data=_backend_to_host(data_array), compression="gzip")
                         elif data_array is not None : state_group.attrs[f"{attr_name_cp}_is_empty"] = True
                    if self.cp_neuron_firing_thresholds is not None and self.cp_neuron_firing_thresholds.size > 0: state_group.create_dataset("cp_neuron_firing_thresholds", data=_backend_to_host(self.cp_neuron_firing_thresholds), compression="gzip")
                    elif self.cp_neuron_firing_thresholds is not None : state_group.attrs["cp_neuron_firing_thresholds_is_empty"] = True

                elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                    for attr_name_suffix in ['m', 'h', 'n']:
                        attr_name_cp = f"cp_gating_variable_{attr_name_suffix}"
                        data_array = getattr(self, attr_name_cp, None)
                        if data_array is not None and data_array.size > 0: state_group.create_dataset(attr_name_cp, data=_backend_to_host(data_array), compression="gzip")
                        elif data_array is not None : state_group.attrs[f"{attr_name_cp}_is_empty"] = True
                    # Optional extended HH activation states
                    for attr_name_cp in [
                        "cp_hh_m_current_activation",
                        "cp_hh_CaT_m",
                        "cp_hh_CaT_h",
                        "cp_hh_h_current_q",
                        "cp_hh_NaP_activation",
                    ]:
                        data_array = getattr(self, attr_name_cp, None)
                        if data_array is not None and data_array.size > 0:
                            state_group.create_dataset(attr_name_cp, data=_backend_to_host(data_array), compression="gzip")
                        elif data_array is not None:
                            state_group.attrs[f"{attr_name_cp}_is_empty"] = True
                    for param in ['C_m', 'g_Na_max', 'g_K_max', 'g_L', 'E_Na', 'E_K', 'E_L', 'v_peak']:
                         attr_name_cp = f"cp_hh_{param}"
                         data_array = getattr(self, attr_name_cp, None)
                         if data_array is not None and data_array.size > 0: state_group.create_dataset(attr_name_cp, data=_backend_to_host(data_array), compression="gzip")
                         elif data_array is not None : state_group.attrs[f"{attr_name_cp}_is_empty"] = True
                
                h5f.attrs["_mock_total_plasticity_events"] = self._mock_total_plasticity_events
                h5f.attrs["_mock_network_avg_firing_rate_hz"] = self._mock_network_avg_firing_rate_hz
                
                if self.runtime_state.neuron_types_list_for_viz:
                    h5f.attrs["neuron_types_list_for_viz_json"] = json.dumps(self.runtime_state.neuron_types_list_for_viz)

                if gui_config_snapshot: # Save GUI related config if provided
                    if "opengl_visualization_settings" in gui_config_snapshot and gui_config_snapshot["opengl_visualization_settings"]:
                         h5f.attrs["opengl_viz_config_json"] = json.dumps(gui_config_snapshot["opengl_visualization_settings"])
                    if "filter_settings" in gui_config_snapshot and gui_config_snapshot["filter_settings"]:
                         h5f.attrs["dpg_filter_settings_json"] = json.dumps(gui_config_snapshot["filter_settings"])

                # Save experiment config if present
                if self.experiment_config is not None:
                    try:
                        exp_dict = experiment_config_to_dict(self.experiment_config)
                        h5f.attrs["experiment_config_json"] = json.dumps(exp_dict)
                    except Exception as e_exp:
                        self._log_console(f"Warning: Could not save experiment config to checkpoint: {e_exp}", "warning")

                # Engram tags (catalog D.14): persist named neuron-index
                # ensembles. Concepts-as-tagged-ensembles must survive
                # save/load for continuous learning across sessions.
                tags = getattr(self, "_engram_tags", None)
                if tags:
                    try:
                        tag_grp = h5f.create_group("engram_tags")
                        for tag_name, idx_array in tags.items():
                            if idx_array is None or idx_array.size == 0:
                                continue
                            # Sanitize name for HDF5 (no '/')
                            safe = tag_name.replace("/", "_slash_")
                            tag_grp.create_dataset(
                                safe,
                                data=_backend_to_host(idx_array.astype(cp.int64)),
                                compression="gzip",
                            )
                            tag_grp[safe].attrs["original_name"] = tag_name
                    except Exception as e_eng:
                        self._log_console(
                            f"Warning: engram tags not saved: {e_eng}",
                            "warning",
                        )

            self._log_to_ui(f"Checkpoint saved successfully to {filepath}", "success")
            if self.ui_queue: self.ui_queue.put({"type": "CHECKPOINT_SAVE_SUCCESS", "filepath": filepath})
            return True
        except Exception as e:
            self._log_to_ui(f"Error saving checkpoint: {e}","error"); import traceback; traceback.print_exc()
            if self.ui_queue: self.ui_queue.put({"type": "CHECKPOINT_SAVE_FAILED", "error": str(e)})
            return False

    def load_checkpoint(self, filepath):
        """Loads a simulation state from an HDF5 checkpoint file."""
        self._log_to_ui(f"Loading checkpoint from {filepath}...", "info")
        
        try:
            with h5py.File(filepath, 'r') as h5f:
                if self.runtime_state.is_running : self.stop_simulation() 
                self.clear_simulation_state_and_gpu_memory() 

                loaded_sim_config_dict = load_dict_from_hdf5_attrs(h5f) 
                if not loaded_sim_config_dict or "num_neurons" not in loaded_sim_config_dict: 
                    self._log_to_ui("Checkpoint missing or invalid simulation_configuration. Load failed.","error"); return False
                
                # Use the SimulationConfiguration class if available (set by main module),
                # otherwise use CoreSimConfig as fallback for filling missing keys
                _SimConfig = getattr(SimulationBridge, '_SimulationConfiguration', None)
                if _SimConfig is not None:
                    temp_cfg_for_validation = _SimConfig()
                else:
                    # Fallback: wrap CoreSimConfig to provide to_dict() interface
                    class _FallbackSimConfig:
                        def __init__(self):
                            self._cfg = CoreSimConfig()
                        def to_dict(self):
                            return asdict(self._cfg)
                        def __getattr__(self, name):
                            return getattr(self._cfg, name)
                    temp_cfg_for_validation = _FallbackSimConfig()
                for key_cfg in temp_cfg_for_validation.to_dict().keys():
                    if key_cfg not in loaded_sim_config_dict: 
                        loaded_sim_config_dict[key_cfg] = getattr(temp_cfg_for_validation, key_cfg) 

                self.core_config = CoreSimConfig(**{k: v for k, v in loaded_sim_config_dict.items() if hasattr(CoreSimConfig, k)})
                n = self.core_config.num_neurons
                state_group = h5f 

                def _load_cp_array_from_h5(key, default_val_func, default_dtype_for_empty=cp.float32):
                    if f"{key}_is_empty" in state_group.attrs and state_group.attrs[f"{key}_is_empty"] is True:
                        return default_val_func(0) 
                    if key in state_group: 
                        return cp.asarray(state_group[key][:]) 
                    self._log_console(f"Checkpoint: Dataset for '{key}' not found or was empty. Using default.", "debug")
                    return default_val_func(n) if n > 0 else default_val_func(0)

                # Note: cp_synapse_pulse_timers and cp_synapse_pulse_progress are synapse-indexed
                # and loaded separately below
                direct_load_map = {
                    'cp_membrane_potential_v': ('cp_membrane_potential_v', cp.float32),
                    'cp_conductance_g_e': ('cp_conductance_g_e', cp.float32),
                    'cp_conductance_g_i': ('cp_conductance_g_i', cp.float32),
                    'cp_external_input_current': ('cp_external_input_current', cp.float32),
                    'cp_firing_states': ('cp_firing_states', cp.bool_),
                    'cp_prev_firing_states': ('cp_prev_firing_states', cp.bool_),
                    'cp_traits': ('cp_traits', cp.int32),
                    'cp_refractory_timers': ('cp_refractory_timers', cp.int32),
                    'cp_neuron_activity_ema': ('cp_neuron_activity_ema', cp.float32),
                    'cp_viz_activity_timers': ('cp_viz_activity_timers', cp.int32),
                    'cp_adex_w': ('cp_adex_w', cp.float32),
                    'cp_ou_current': ('cp_ou_current', cp.float32)
                }
                for attr_name, (h5_key, dtype) in direct_load_map.items():
                    setattr(self, attr_name, _load_cp_array_from_h5(h5_key, 
                            default_val_func=lambda size_n, dt=dtype: cp.zeros(size_n, dtype=dt), 
                            default_dtype_for_empty=dtype))

                if "cp_neuron_positions_3d" in state_group or ("cp_neuron_positions_3d_is_empty" in state_group.attrs):
                     self.cp_neuron_positions_3d = _load_cp_array_from_h5("cp_neuron_positions_3d", 
                        default_val_func=lambda size_n: cp.zeros((size_n, 3), dtype=cp.float32))
                elif n > 0 : 
                    np_positions_3d = np.random.uniform(
                        low=[self.core_config.volume_min_x,self.core_config.volume_min_y,self.core_config.volume_min_z],
                        high=[self.core_config.volume_max_x,self.core_config.volume_max_y,self.core_config.volume_max_z],
                        size=(n,3)).astype(np.float32)
                    self.cp_neuron_positions_3d = cp.asarray(np_positions_3d)
                else: self.cp_neuron_positions_3d = cp.array([], dtype=cp.float32).reshape(0,3)

                conn_data_np = state_group["connections_data"][:] if "connections_data" in state_group else np.array([], dtype=cp.float32)
                conn_indices_np = state_group["connections_indices"][:] if "connections_indices" in state_group else np.array([], dtype=cp.int32)
                conn_indptr_np = state_group["connections_indptr"][:] if "connections_indptr" in state_group else np.array([0]*(n+1), dtype=cp.int32) 
                conn_shape_0 = state_group.attrs.get("connections_shape_0", n)
                conn_shape_1 = state_group.attrs.get("connections_shape_1", n)
                conn_shape = (conn_shape_0, conn_shape_1)
                if conn_shape[0] != n or conn_shape[1] != n: 
                    self._log_to_ui(f"Warning: Checkpoint connection shape {conn_shape} mismatch with config N={n}. Adjusting.", "warning")
                    conn_shape = (n,n)
                    if conn_data_np.size == 0 : conn_indptr_np = np.array([0]*(n+1), dtype=cp.int32)

                self.cp_connections = csp.csr_matrix((cp.asarray(conn_data_np), 
                                                      cp.asarray(conn_indices_np), 
                                                      cp.asarray(conn_indptr_np)), 
                                                     shape=conn_shape, dtype=cp.float32)
                
                num_synapses_loaded = self.cp_connections.nnz

                self.cp_stp_u = _load_cp_array_from_h5("cp_stp_u", 
                    lambda s: cp.full(s, self.core_config.stp_U, dtype=cp.float32) if self.core_config.enable_short_term_plasticity and num_synapses_loaded > 0 and s > 0 else (cp.array([],dtype=cp.float32) if s==0 else None))
                self.cp_stp_x = _load_cp_array_from_h5("cp_stp_x", 
                    lambda s: cp.ones(s, dtype=cp.float32) if self.core_config.enable_short_term_plasticity and num_synapses_loaded > 0 and s > 0 else (cp.array([],dtype=cp.float32) if s==0 else None))

                if not (self.core_config.enable_short_term_plasticity and num_synapses_loaded > 0):
                    self.cp_stp_u = None; self.cp_stp_x = None
                else:
                    if self.cp_stp_u is None or self.cp_stp_u.size != num_synapses_loaded:
                        self.cp_stp_u = cp.full(num_synapses_loaded, self.core_config.stp_U, dtype=cp.float32)
                    if self.cp_stp_x is None or self.cp_stp_x.size != num_synapses_loaded:
                        self.cp_stp_x = cp.ones(num_synapses_loaded, dtype=cp.float32)
                
                # C2: Load STDP and reward modulation state
                if self.core_config.enable_stdp and n > 0:
                    self.cp_last_spike_time = _load_cp_array_from_h5("cp_last_spike_time",
                        lambda s: cp.full(s, -1000.0, dtype=cp.float32))
                else:
                    self.cp_last_spike_time = None

                # G3: Load per-synapse plastic mask if present in checkpoint.
                # Absent → leave as None (all plastic, back-compat).
                if "cp_synapse_plastic_mask" in state_group:
                    self.cp_synapse_plastic_mask = cp.asarray(
                        state_group["cp_synapse_plastic_mask"][:]
                    ).astype(cp.bool_)
                else:
                    self.cp_synapse_plastic_mask = None
                
                if self.core_config.enable_reward_modulation and num_synapses_loaded > 0:
                    self.cp_eligibility_trace = _load_cp_array_from_h5("cp_eligibility_trace",
                        lambda s: cp.zeros(s, dtype=cp.float32) if s > 0 else cp.array([], dtype=cp.float32))
                    # Ensure size matches number of synapses
                    if self.cp_eligibility_trace.size != num_synapses_loaded:
                        self.cp_eligibility_trace = cp.zeros(num_synapses_loaded, dtype=cp.float32)
                else:
                    self.cp_eligibility_trace = None

                # Load synapse visualization arrays (synapse-indexed)
                if OPENGL_AVAILABLE and num_synapses_loaded > 0:
                    self.cp_synapse_pulse_timers = _load_cp_array_from_h5("cp_synapse_pulse_timers",
                        lambda s: cp.zeros(s, dtype=cp.int32) if s > 0 else cp.array([], dtype=cp.int32))
                    if self.cp_synapse_pulse_timers.size != num_synapses_loaded:
                        self.cp_synapse_pulse_timers = cp.zeros(num_synapses_loaded, dtype=cp.int32)
                    self.cp_synapse_pulse_progress = _load_cp_array_from_h5("cp_synapse_pulse_progress",
                        lambda s: cp.zeros(s, dtype=cp.float32) if s > 0 else cp.array([], dtype=cp.float32))
                    if self.cp_synapse_pulse_progress.size != num_synapses_loaded:
                        self.cp_synapse_pulse_progress = cp.zeros(num_synapses_loaded, dtype=cp.float32)
                else:
                    self.cp_synapse_pulse_timers = None
                    self.cp_synapse_pulse_progress = None

                # C3: Load structural plasticity state
                if self.core_config.enable_structural_plasticity:
                    self.cp_struct_plast_step_counter = state_group.attrs.get("cp_struct_plast_step_counter", 0)
                else:
                    self.cp_struct_plast_step_counter = None

                # Initialize synapse tracking variables from loaded array sizes
                # (no extra capacity initially - will grow dynamically if structural plasticity adds synapses)
                self._synapse_count = num_synapses_loaded
                self._synapse_capacity = num_synapses_loaded

                if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                    self.cp_recovery_variable_u = _load_cp_array_from_h5("cp_recovery_variable_u", lambda s: cp.zeros(s, dtype=cp.float32))
                    for param in ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c_reset', 'd_increment']:
                        setattr(self, f"cp_izh_{param}", _load_cp_array_from_h5(f"cp_izh_{param}",
                                lambda s, p=param: cp.full(s, getattr(self.core_config, f"izh_{p}_val"), dtype=cp.float32)))
                    self.cp_neuron_firing_thresholds = _load_cp_array_from_h5("cp_neuron_firing_thresholds",
                        lambda s: cp.random.uniform(self.core_config.homeostasis_threshold_min, self.core_config.homeostasis_threshold_max, s).astype(cp.float32) if s > 0 else cp.array([], dtype=cp.float32))
                elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                    for attr_name_suffix in ['m', 'h', 'n']:
                         setattr(self, f"cp_gating_variable_{attr_name_suffix}", _load_cp_array_from_h5(f"cp_gating_variable_{attr_name_suffix}",
                                 lambda s, suff=attr_name_suffix: cp.full(s, getattr(self.core_config, f"hh_{suff}_init"), dtype=cp.float32)))
                    # Optional extended HH activation states
                    self.cp_hh_m_current_activation = _load_cp_array_from_h5(
                        "cp_hh_m_current_activation",
                        lambda s: cp.zeros(s, dtype=cp.float32)
                    )
                    self.cp_hh_CaT_m = _load_cp_array_from_h5(
                        "cp_hh_CaT_m",
                        lambda s: cp.zeros(s, dtype=cp.float32)
                    )
                    self.cp_hh_CaT_h = _load_cp_array_from_h5(
                        "cp_hh_CaT_h",
                        lambda s: cp.zeros(s, dtype=cp.float32)
                    )
                    self.cp_hh_h_current_q = _load_cp_array_from_h5(
                        "cp_hh_h_current_q",
                        lambda s: cp.zeros(s, dtype=cp.float32)
                    )
                    self.cp_hh_NaP_activation = _load_cp_array_from_h5(
                        "cp_hh_NaP_activation",
                        lambda s: cp.zeros(s, dtype=cp.float32)
                    )
                    hh_param_map = {'C_m': 'hh_C_m', 'g_Na_max': 'hh_g_Na_max', 'g_K_max': 'hh_g_K_max', 'g_L': 'hh_g_L',
                                    'E_Na': 'hh_E_Na', 'E_K': 'hh_E_K', 'E_L': 'hh_E_L', 'v_peak': 'hh_v_peak'}
                    for param_key, config_attr_name in hh_param_map.items():
                         setattr(self, f"cp_hh_{param_key}", _load_cp_array_from_h5(f"cp_hh_{param_key}",
                                 lambda s, ca_name=config_attr_name: cp.full(s, getattr(self.core_config, ca_name), dtype=cp.float32)))
                    self.cp_neuron_firing_thresholds = None 

                self._mock_total_plasticity_events = h5f.attrs.get("_mock_total_plasticity_events",0)
                self._mock_network_avg_firing_rate_hz = h5f.attrs.get("_mock_network_avg_firing_rate_hz",0.0)

                # Recompute step-invariant cached constants that aren't saved to
                # the checkpoint (derivable from core_config). Without these, the
                # first _run_one_simulation_step after load crashes on
                # `AttributeError: _cached_decay_e` / `ou_decay_factor`.
                cfg = self.core_config
                self._cached_decay_e = float(cp.exp(-cfg.dt_ms / cfg.syn_tau_g_e)) if cfg.syn_tau_g_e > 0 else 0.0
                self._cached_decay_i = float(cp.exp(-cfg.dt_ms / cfg.syn_tau_g_i)) if cfg.syn_tau_g_i > 0 else 0.0
                self._cached_decay_nmda = float(cp.exp(-cfg.dt_ms / cfg.nmda_tau_decay)) if cfg.nmda_tau_decay > 0 else 0.0
                self._cached_decay_nmda_rise = float(cp.exp(-cfg.dt_ms / cfg.nmda_tau_rise)) if cfg.nmda_tau_rise > 0 else 0.0

                # Recompute OU step-invariant constants (cp_ou_current state is
                # preserved; coefficients are derived from dt / tau / sigma).
                if cfg.enable_ou_process and cfg.ou_tau_ms > 0:
                    dt_sec = cfg.dt_ms / 1000.0
                    tau_sec = cfg.ou_tau_ms / 1000.0
                    self.ou_decay_factor = float(cp.exp(-dt_sec / tau_sec))
                    self.ou_noise_std = float(
                        cfg.ou_std_current_pA * cp.sqrt((1.0 - cp.exp(-2.0 * dt_sec / tau_sec)) / 2.0)
                    )
                    self.ou_mean = float(cfg.ou_mean_current_pA)
                else:
                    self.ou_decay_factor = None
                    self.ou_noise_std = None
                    self.ou_mean = None

                # HH Q10 temperature phase factor (harmless for non-HH models).
                _BASE_HH_TEMP = 6.3
                _temp_delta_div_10 = (cfg.hh_temperature_celsius - _BASE_HH_TEMP) / 10.0
                self._cached_hh_phi = cfg.hh_q10_factor ** _temp_delta_div_10  # legacy uniform-Q10 phi
                # Per-gate phi values (HH temperature bug fix)
                self._cached_hh_phi_m = cfg.hh_q10_m ** _temp_delta_div_10
                self._cached_hh_phi_h = cfg.hh_q10_h ** _temp_delta_div_10
                self._cached_hh_phi_n = cfg.hh_q10_n ** _temp_delta_div_10

                # Engram tags (catalog D.14): restore named ensembles
                # if present in the checkpoint.
                if "engram_tags" in h5f:
                    try:
                        self._init_engram_tagging()
                        self._engram_tags.clear()
                        tag_grp = h5f["engram_tags"]
                        for safe_name in tag_grp.keys():
                            ds = tag_grp[safe_name]
                            original = ds.attrs.get("original_name", safe_name)
                            if isinstance(original, bytes):
                                original = original.decode("utf-8")
                            idx_host = ds[()]
                            self._engram_tags[str(original)] = cp.asarray(
                                idx_host, dtype=cp.int64
                            )
                    except Exception as e_eng:
                        self._log_console(
                            f"Warning: engram tags not loaded: {e_eng}",
                            "warning",
                        )

                self.is_initialized = True
                self._log_to_ui(f"Checkpoint loaded. Sim time: {self.runtime_state.current_time_ms}ms, Step: {self.runtime_state.current_time_step}, Model: {self.core_config.neuron_model_type}", "success")

                loaded_gui_settings = {}
                if "opengl_viz_config_json" in h5f.attrs:
                    try: loaded_gui_settings["opengl_visualization_settings"] = json.loads(h5f.attrs["opengl_viz_config_json"])
                    except: self._log_console("Warning: Could not parse opengl_viz_config_json from checkpoint.", "warning")
                if "dpg_filter_settings_json" in h5f.attrs: # Load DPG filter settings if present
                    try: loaded_gui_settings["filter_settings"] = json.loads(h5f.attrs["dpg_filter_settings_json"])
                    except: self._log_console("Warning: Could not parse dpg_filter_settings_json from checkpoint.", "warning")
                
                if "neuron_types_list_for_viz_json" in h5f.attrs:
                    try:
                        self.runtime_state.neuron_types_list_for_viz = json.loads(h5f.attrs["neuron_types_list_for_viz_json"])
                        loaded_gui_settings["neuron_types_list_for_viz"] = self.runtime_state.neuron_types_list_for_viz
                    except: self._log_console("Warning: Could not parse neuron_types_list_for_viz_json from checkpoint.", "warning")

                # Restore experiment config if present in checkpoint
                if "experiment_config_json" in h5f.attrs:
                    try:
                        exp_dict = json.loads(h5f.attrs["experiment_config_json"])
                        self.experiment_config = experiment_config_from_dict(exp_dict)
                        if self.experiment_config.enabled and self.is_initialized:
                            self.experiment_engine = ExperimentEngine(
                                self.core_config.num_neurons, self.core_config.dt_ms
                            )
                            self.experiment_engine.load_experiment(self.experiment_config)
                            self.experiment_engine.initialize(
                                cp_traits=self.cp_traits, cp_module=cp
                            )
                            self._log_console(f"Experiment config restored from checkpoint: {self.experiment_config.name}", "info")
                    except Exception as e_exp:
                        self._log_console(f"Warning: Could not restore experiment config: {e_exp}", "warning")

                if self.ui_queue:
                    initial_gui_data = self.get_initial_sim_data_snapshot() 
                    self.ui_queue.put({
                        "type": "CHECKPOINT_LOADED_SUCCESS",
                        "config_dict": self.core_config.to_dict(),
                        "gui_settings_from_checkpoint": loaded_gui_settings,
                        "initial_gui_data": initial_gui_data
                    })
                return True
        except Exception as e:
            self._log_to_ui(f"Error loading checkpoint: {e}","error"); import traceback; traceback.print_exc()
            self.is_initialized=False; 
            if self.ui_queue: self.ui_queue.put({"type": "CHECKPOINT_LOAD_FAILED", "error": str(e)})
            return False        

    def get_latest_simulation_data_for_gui(self, force_fetch=False):
        """Retrieves a snapshot of the current simulation state for GUI updates.
        Sends CuPy arrays for relevant OpenGL data.
        """
        if not self.is_initialized:
            self._log_console("GUI data request: Sim not initialized.","debug"); return None

        n = self.core_config.num_neurons
        dt = self.core_config.dt_ms
        
        # Use pre-computed spike count from simulation step (avoids GPU-CPU sync here)
        num_spikes_this_step = self._mock_num_spikes_this_step if hasattr(self, '_mock_num_spikes_this_step') else 0

        # Update firing rate EMA from cached spike count
        if n > 0 and dt > 0:
            instantaneous_rate_hz = (num_spikes_this_step / n) / (dt / 1000.0)
            self._mock_network_avg_firing_rate_hz = self._mock_network_avg_firing_rate_hz * 0.95 + instantaneous_rate_hz * 0.05
        
        gui_data_dict = {
            "current_time_ms": self.runtime_state.current_time_ms,
            "current_time_step": self.runtime_state.current_time_step,
            "num_spikes_this_step": num_spikes_this_step,
            "network_avg_firing_rate_hz": self._mock_network_avg_firing_rate_hz,
            "total_plasticity_events": self._mock_total_plasticity_events,
            "neuron_types_list_for_viz": self.runtime_state.neuron_types_list_for_viz.copy(), # Stays as Python list
            "neuron_model_type_str": self.core_config.neuron_model_type,
            "num_neurons_snapshot": n # Add total number of neurons in this snapshot
        }

        # --- Data to keep as CuPy arrays for OpenGL ---
        # Dynamic arrays (change every step) — must copy to avoid race with sim thread
        if self.cp_firing_states is not None:
            gui_data_dict["neuron_fired_status_cp"] = self.cp_firing_states.copy()
        elif n > 0:
            gui_data_dict["neuron_fired_status_cp"] = cp.zeros(n, dtype=bool)
        else:
            gui_data_dict["neuron_fired_status_cp"] = cp.array([], dtype=bool)

        if self.cp_viz_activity_timers is not None:
            gui_data_dict["neuron_activity_timers_cp"] = self.cp_viz_activity_timers.copy()
        elif n > 0:
            gui_data_dict["neuron_activity_timers_cp"] = cp.zeros(n, dtype=cp.int32)
        else:
            gui_data_dict["neuron_activity_timers_cp"] = cp.array([], dtype=cp.int32)

        # Static arrays (positions, traits, type IDs) — don't change during simulation.
        # Cache once and reuse to avoid expensive GPU copies every update.
        if not hasattr(self, '_cached_static_gui_data') or self._cached_static_gui_data is None:
            self._cached_static_gui_data = {}
            if self.cp_neuron_positions_3d is not None:
                self._cached_static_gui_data["neuron_positions_3d_cp"] = self.cp_neuron_positions_3d.copy()
            elif n > 0:
                self._cached_static_gui_data["neuron_positions_3d_cp"] = cp.zeros((n,3),dtype=cp.float32)
            else:
                self._cached_static_gui_data["neuron_positions_3d_cp"] = cp.array([], dtype=cp.float32).reshape(0,3)

            if self.cp_traits is not None:
                self._cached_static_gui_data["neuron_traits_cp"] = self.cp_traits.copy()
            elif n > 0:
                self._cached_static_gui_data["neuron_traits_cp"] = cp.zeros(n, dtype=cp.int32)
            else:
                self._cached_static_gui_data["neuron_traits_cp"] = cp.array([], dtype=cp.int32)

            if self.cp_neuron_type_ids is not None:
                self._cached_static_gui_data["neuron_type_ids_cp"] = self.cp_neuron_type_ids.copy()
            elif n > 0:
                self._cached_static_gui_data["neuron_type_ids_cp"] = cp.zeros(n, dtype=cp.int32)
            else:
                self._cached_static_gui_data["neuron_type_ids_cp"] = cp.array([], dtype=cp.int32)

        gui_data_dict.update(self._cached_static_gui_data)

        # --- Data for DPG text display (can be NumPy or Python types) ---
        if self.cp_membrane_potential_v is not None:
            # Example: If you need a small sample of Vm for a DPG plot (not for GL points usually)
            # sample_indices_vm = cp.random.choice(cp.arange(n), size=min(n, 100), replace=False) if n > 0 else cp.array([])
            # gui_data_dict["neuron_Vm_sample_np"] = _backend_to_host(self.cp_membrane_potential_v[sample_indices_vm]) if sample_indices_vm.size > 0 else np.array([])
            pass # For full Vm, if used for something other than GL points directly, decide if cp or np needed

        # Synapse info for GUI is CPU-based and sampled - only update occasionally to minimize CPU-GPU transfers
        # Check if we should update synapse sample this time
        # Use visualization config setting for update interval
        viz_update_interval = self.viz_config.viz_update_interval_steps
        update_synapse_sample = (self.runtime_state.current_time_step % viz_update_interval == 0)
        
        # Use cached synapse info if not updating
        if not update_synapse_sample and hasattr(self, '_cached_synapse_info_gui'):
            synapse_info_for_gui = self._cached_synapse_info_gui
        else:
            synapse_info_for_gui = []
            if self.cp_connections is not None and hasattr(self.cp_connections,'nnz') and self.cp_connections.nnz > 0:
                max_synapses_to_sample_for_gui = 20000
                try:
                    cached_coo = self._get_cached_coo()
                    coo_conn = cached_coo if cached_coo is not None else self.cp_connections.tocoo(copy=False)
                    num_actual_synapses = coo_conn.nnz
                    num_to_send = min(num_actual_synapses, max_synapses_to_sample_for_gui)

                    if num_to_send > 0:
                        indices_to_sample_np = np.random.choice(num_actual_synapses, num_to_send, replace=False) \
                                            if num_actual_synapses > num_to_send else np.arange(num_actual_synapses)

                        # Fetch relevant data from CuPy arrays using NumPy indices
                        row_indices_np = _backend_to_host(coo_conn.row[indices_to_sample_np])
                        col_indices_np = _backend_to_host(coo_conn.col[indices_to_sample_np])

                        weights_data_to_use_cp = self.cp_connections.data 
                        if self.core_config.enable_short_term_plasticity and \
                        self.cp_stp_u is not None and self.cp_stp_x is not None and \
                        self.cp_stp_u.size == self.cp_connections.data.size and \
                        self.cp_stp_x.size == self.cp_connections.data.size :
                            weights_data_to_use_cp = self.cp_connections.data * self.cp_stp_u * self.cp_stp_x

                        # Sample weights using NumPy indices on the CuPy array, then convert
                        sampled_weights_np = _backend_to_host(weights_data_to_use_cp[cp.asarray(indices_to_sample_np)])

                        for i in range(num_to_send):
                            synapse_info_for_gui.append({
                                "source_idx": int(row_indices_np[i]),
                                "target_idx": int(col_indices_np[i]),
                                "weight": float(sampled_weights_np[i])
                            })
                except Exception as e: self._log_console(f"Error processing connections for GUI: {e}","error")
            
            # Cache the synapse info for future use
            self._cached_synapse_info_gui = synapse_info_for_gui
        
        gui_data_dict["synapse_info"] = synapse_info_for_gui

        # Pulse data for OpenGL - if pulses are enabled, this part needs to be GPU-centric
        # For now, this logic is complex and might be better handled by sending raw cp_synapse_pulse_timers/progress
        # and relevant connection data for UI thread to compute positions, OR pre-compute on sim thread.
        # Let's assume for now this is handled later or in a simplified way.
        # If pulse positions are needed for GL, they should be sent as a CuPy array.
        # Example: Pre-calculate active pulse positions on sim thread (if feasible):
        if OPENGL_AVAILABLE and opengl_viz_config.get("ENABLE_SYNAPTIC_PULSES", False) and \
        self.cp_synapse_pulse_timers is not None and self.cp_synapse_pulse_progress is not None and \
        self.cp_connections is not None and self.cp_connections.nnz > 0:

            active_pulse_mask = self.cp_synapse_pulse_timers > 0
            active_pulse_indices = cp.where(active_pulse_mask)[0]

            if active_pulse_indices.size > 0:
                cached_coo_p = self._get_cached_coo()
                coo_conn_for_pulses = cached_coo_p if cached_coo_p is not None else self.cp_connections.tocoo(copy=False)

                # Get source and target neuron indices for active pulses
                # These indices are into the full list of synapses (coo_conn.row/col)
                src_neuron_indices_for_active_pulses = coo_conn_for_pulses.row[active_pulse_indices]
                tgt_neuron_indices_for_active_pulses = coo_conn_for_pulses.col[active_pulse_indices]

                # Get positions of these source and target neurons
                pos_src_cp = self.cp_neuron_positions_3d[src_neuron_indices_for_active_pulses]
                pos_tgt_cp = self.cp_neuron_positions_3d[tgt_neuron_indices_for_active_pulses]

                # Get progress for active pulses
                pulse_prog_active = self.cp_synapse_pulse_progress[active_pulse_indices]

                # Interpolate pulse positions: pos_src + progress * (pos_tgt - pos_src)
                # Reshape pulse_prog_active to be (N, 1) for broadcasting with (N, 3) positions
                pulse_positions_cp = pos_src_cp + pulse_prog_active[:, cp.newaxis] * (pos_tgt_cp - pos_src_cp)
                gui_data_dict["pulse_positions_cp_for_gl"] = pulse_positions_cp # Send as CuPy array
            else:
                gui_data_dict["pulse_positions_cp_for_gl"] = cp.array([], dtype=cp.float32).reshape(0,3)
        else:
            gui_data_dict["pulse_positions_cp_for_gl"] = cp.array([], dtype=cp.float32).reshape(0,3)


        # Small, specific NumPy arrays for DPG plots (if any)
        # Example: if self.cp_membrane_potential_v is not None and n > 0:
        #     sample_indices = cp.random.choice(cp.arange(n), size=min(n, 10), replace=False) # Small sample for plotting
        #     gui_data_dict["neuron_Vm_trace_sample_np"] = _backend_to_host(self.cp_membrane_potential_v[sample_indices])

        # Experiment system status (lightweight — no GPU sync needed)
        if self.experiment_engine is not None:
            try:
                gui_data_dict["experiment_status"] = self.experiment_engine.get_experiment_status()
            except Exception:
                gui_data_dict["experiment_status"] = {"is_running": False}
        else:
            gui_data_dict["experiment_status"] = None

        return gui_data_dict

    def get_initial_sim_data_snapshot(self):
        """
        Gets a snapshot of simulation data, intended for when the simulation is first initialized or reset.
        Returns data structure consistent with get_latest_simulation_data_for_gui, 
        including CuPy arrays for GL-relevant data.
        """
        if not self.is_initialized:
            self._log_console("Initial snapshot request: Sim not initialized. Providing empty/default structure.","info")
            # Fallback, creating structure similar to get_latest_simulation_data_for_gui
            n_cfg = self.core_config.num_neurons if self.is_initialized else 0
            model_type_str_cfg = self.core_config.neuron_model_type if self.is_initialized else NeuronModel.IZHIKEVICH.name
            types_list_cfg = self.runtime_state.neuron_types_list_for_viz.copy() if self.is_initialized and self.runtime_state.neuron_types_list_for_viz else []
                
            # Ensure this fallback structure matches the keys expected by the UI,
            # especially the CuPy array keys for GL.
            return { 
                "current_time_ms": 0.0, 
                "current_time_step": 0,
                "num_spikes_this_step": 0, 
                "network_avg_firing_rate_hz": 0.0,
                "total_plasticity_events": 0, 
                "synapse_info": [], # Stays as Python list for CPU processing
                "neuron_types_list_for_viz": types_list_cfg, # Stays as Python list
                "neuron_model_type_str": model_type_str_cfg,
                "num_neurons_snapshot": n_cfg,
                
                # CuPy arrays, initialized appropriately (empty or zeros)
                "neuron_fired_status_cp": cp.zeros(n_cfg, dtype=bool) if n_cfg > 0 else cp.array([], dtype=bool),
                "neuron_activity_timers_cp": cp.zeros(n_cfg, dtype=cp.int32) if n_cfg > 0 else cp.array([], dtype=cp.int32),
                "neuron_positions_3d_cp": cp.zeros((n_cfg,3), dtype=cp.float32) if n_cfg > 0 else cp.array([], dtype=cp.float32).reshape(0,3),
                "neuron_traits_cp": cp.zeros(n_cfg, dtype=cp.int32) if n_cfg > 0 else cp.array([], dtype=cp.int32),
                "pulse_positions_cp_for_gl": cp.array([], dtype=cp.float32).reshape(0,3)
                # Add other _cp keys if they are essential for GL init (e.g., Vm if directly used by GL)
                # "neuron_Vm_cp": cp.zeros(n_cfg, dtype=cp.float32) if n_cfg > 0 else cp.array([], dtype=cp.float32), # Example if Vm was also made cp for GL
                }

        # If initialized, get the latest data structure (which now includes CuPy arrays for GL)
        snapshot = self.get_latest_simulation_data_for_gui(force_fetch=True) 

        if snapshot: 
            # Reset time-dependent/cumulative values to represent an "initial" state
            snapshot["current_time_ms"] = 0.0
            snapshot["current_time_step"] = 0
            snapshot["num_spikes_this_step"] = 0
            snapshot["network_avg_firing_rate_hz"] = 0.0
            snapshot["total_plasticity_events"] = 0 # Reset this mock counter
                
            # Reset visual activity timers (which are CuPy arrays in the snapshot)
            # Key name was "neuron_activity_timers", now "neuron_activity_timers_cp" from get_latest_simulation_data_for_gui
            if "neuron_activity_timers_cp" in snapshot and snapshot["neuron_activity_timers_cp"].size > 0:
                snapshot["neuron_activity_timers_cp"].fill(0) 
            
            # If other visual timers or states are present as CuPy arrays and need resetting for an initial view, do it here.
            # e.g., if pulse progress was part of the _cp arrays and needed reset:
            # if "pulse_progress_cp" in snapshot and snapshot["pulse_progress_cp"].size > 0:
            #      snapshot["pulse_progress_cp"].fill(0.0)
            
            return snapshot

        def get_profile_visualization_data(self, from_current_config=False):
            """Prepares data specifically needed for visualizing a network profile (neuron positions, types)."""
            cfg = self.core_config; num_n = cfg.num_neurons

            positions_stale = self.cp_neuron_positions_3d is None or self.cp_neuron_positions_3d.shape[0] != num_n
            types_stale = not cfg.neuron_types_list_for_viz or len(cfg.neuron_types_list_for_viz) != num_n

            if from_current_config and (positions_stale or types_stale):
                self._log_console("Re-populating neuron positions/types for visualization profile (3D).","debug")

                if positions_stale and num_n > 0:
                    np_positions_3d = np.random.uniform(
                        low=[cfg.volume_min_x,cfg.volume_min_y,cfg.volume_min_z],
                        high=[cfg.volume_max_x,cfg.volume_max_y,cfg.volume_max_z],
                        size=(num_n,3)).astype(np.float32)
                    self.cp_neuron_positions_3d = cp.asarray(np_positions_3d)
                    cfg.neuron_positions_x = np_positions_3d[:,0].tolist() 
                    cfg.neuron_positions_y = np_positions_3d[:,1].tolist()
                elif num_n == 0: 
                    self.cp_neuron_positions_3d = cp.array([],dtype=np.float32).reshape(0,3)
                    cfg.neuron_positions_x=[]; cfg.neuron_positions_y=[]

                if types_stale: 
                    cfg.neuron_types_list_for_viz = [""] * num_n 
                    np_traits_host_temp = _backend_to_host(self.cp_traits) if self.cp_traits is not None and self.cp_traits.size == num_n else \
                                    np.random.randint(0, max(1, cfg.num_traits), num_n) 
                    if self.cp_traits is None or self.cp_traits.size != num_n: 
                        self.cp_traits = cp.asarray(np_traits_host_temp)

                    if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                        default_izh_type_enum = NeuronType[cfg.default_neuron_type_izh]
                        defined_izh2007_types = [ntype for ntype in NeuronType if "IZH2007" in ntype.name and ntype in DefaultIzhikevichParamsManager.PARAMS]
                        num_defined_izh_variants = len(defined_izh2007_types)
                        for i in range(num_n):
                            trait_val = np_traits_host_temp[i]
                            selected_neuron_type_enum = default_izh_type_enum
                            if num_defined_izh_variants > 0: selected_neuron_type_enum = defined_izh2007_types[trait_val % num_defined_izh_variants]
                            cfg.neuron_types_list_for_viz[i] = f"Izh2007_{selected_neuron_type_enum.name.replace('IZH2007_', '')}"
                    elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                        # For HH, use a single preset neuron type for all neurons (default_neuron_type_hh)
                        default_hh_type_enum = NeuronType[cfg.default_neuron_type_hh]
                        for i in range(num_n):
                            cfg.neuron_types_list_for_viz[i] = f"HH_{default_hh_type_enum.name.replace('HH_', '')}"
                    else: 
                        cfg.neuron_types_list_for_viz = [f"Unknown_Type_{np_traits_host_temp[i]}" for i in range(num_n)]

            positions_3d_np = _backend_to_host(self.cp_neuron_positions_3d) if self.cp_neuron_positions_3d is not None else np.zeros((0,3), dtype=np.float32)
            return {
                "neuron_positions_3d": positions_3d_np,
                "neuron_types": cfg.neuron_types_list_for_viz, 
                "neuron_positions_x_proj": cfg.neuron_positions_x, 
                "neuron_positions_y_proj": cfg.neuron_positions_y  
            }

        def get_available_neuron_types(self):
            """Returns a list of available neuron type strings for the current model, for UI filters."""
            cfg = self.core_config
            available_types = ["All"] 
            if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                available_types.extend([f"Izh2007_{nt.name.replace('IZH2007_', '')}" for nt in NeuronType if "IZH2007" in nt.name and nt in DefaultIzhikevichParamsManager.PARAMS])
            elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                available_types.extend([f"HH_{nt.name.replace('HH_', '')}" for nt in NeuronType if "HH_" in nt.name and nt in DefaultHodgkinHuxleyParams.PARAMS])
            return list(dict.fromkeys(available_types)) 

# --- Global Variables & GUI State (Must be defined before use in SimulationBridge or DPG) ---
# Note: global_simulation_bridge is initialized in main()
