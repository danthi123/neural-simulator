# main_app.py
import dearpygui.dearpygui as dpg
import os
import json
import time
import numpy as np
import random
from enum import Enum
from collections import deque
import threading
import sys
import h5py
import math
import queue
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict, fields
from typing import List, Dict

# Optional: hdf5plugin for LZ4 compression (faster than gzip)
try:
    import hdf5plugin
    HAS_HDF5PLUGIN = True
except ImportError:
    HAS_HDF5PLUGIN = False
    # Fallback warning printed later when needed

# Optional: psutil for CPU memory monitoring during recording
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # CPU memory monitoring disabled without psutil

# === LOG CAPTURE SYSTEM ===
# Initialize IMMEDIATELY after imports to capture ALL print output
class LogCapture:
    """Thread-safe log capture system for displaying console output in the GUI."""
    def __init__(self, max_lines=5000):
        self.max_lines = max_lines
        self.log_buffer = []
        self.lock = threading.Lock()
        self.original_stdout = None
        self.original_stderr = None
        self.enabled = False
    
    def start_capture(self):
        """Begin capturing print statements and stderr."""
        if self.enabled:
            return
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        self.enabled = True
    
    def stop_capture(self):
        """Restore original stdout/stderr."""
        if not self.enabled:
            return
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.enabled = False
    
    def write(self, text):
        """Called by print() to capture output."""
        # Write to original stdout as well
        if self.original_stdout:
            self.original_stdout.write(text)
        
        # Add to buffer
        if text and text.strip():
            with self.lock:
                self.log_buffer.append(text.rstrip())
                if len(self.log_buffer) > self.max_lines:
                    self.log_buffer = self.log_buffer[-self.max_lines:]
    
    def flush(self):
        """Required for file-like object interface."""
        if self.original_stdout:
            self.original_stdout.flush()
    
    def get_logs(self):
        """Get all captured log lines."""
        with self.lock:
            return self.log_buffer.copy()
    
    def clear(self):
        """Clear the log buffer."""
        with self.lock:
            self.log_buffer.clear()
    
    def search(self, query, case_sensitive=False):
        """Find all line indices containing the search query."""
        with self.lock:
            if not case_sensitive:
                query = query.lower()
            matches = []
            for i, line in enumerate(self.log_buffer):
                search_text = line if case_sensitive else line.lower()
                if query in search_text:
                    matches.append(i)
            return matches

# Initialize global log capture immediately
_global_log_capture = LogCapture(max_lines=5000)
_global_log_capture.start_capture()

# === END LOG CAPTURE SYSTEM ===

# Attempt to get screen resolution using tkinter
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 760 # Default values
try:
    import tkinter
    root = tkinter.Tk()
    root.withdraw() # Hide the main window
    SCREEN_WIDTH = root.winfo_screenwidth()
    SCREEN_HEIGHT = root.winfo_screenheight()
    root.destroy()
    print(f"Detected screen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
except Exception as e:
    print(f"Could not detect screen resolution using tkinter: {e}. Using defaults {SCREEN_WIDTH}x{SCREEN_HEIGHT}.")

# OpenGL and GLUT imports
try:
    from OpenGL.GL import *
    import OpenGL.GLUT as glut
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
    print("PyOpenGL found. OpenGL visualization will be used.")
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL or its dependencies not found. Visualization will be text-based if possible, or disabled.")
    print("Install with: pip install PyOpenGL PyOpenGL_accelerate")


import cupy as cp
try:
    import cupy.sparse as csp
except (ImportError, ModuleNotFoundError):
    import cupyx.scipy.sparse as csp
print("CuPy initialized for GPU acceleration.")

RECORDING_FORMAT_VERSION = "1.1.0-h5" # Version for .simrec.h5 files

# --- Performance Tuning Constants ---
# For 60fps: 1000ms / 60fps = ~16.67ms per frame
# With dt=1.0ms: 16.67ms / 1.0ms ≈ 17 steps
# Adjust based on your actual dt if different
SYNAPSE_SAMPLE_UPDATE_INTERVAL_STEPS = 17  # Update synapse samples for ~60fps visualization

# --- Threading Globals ---
simulation_thread = None
ui_to_sim_queue = queue.Queue()
sim_to_ui_queue = queue.Queue()
# shutdown_flag is already a threading.Event, will be initialized later in main
# gl_data_lock is already a threading.Lock, will be initialized later if OPENGL_AVAILABLE

# --- Configuration & Data Classes ---

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
            # Positive d drives post-spike recovery variable u upward → stronger AHP → faster return to rest.
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
        import math
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
    # Very high Na for fast simple spikes (~50-100 Hz tonic), strong CaT for complex spikes,
    # large M-current for afterhyperpolarization, no Ih (Purkinje cells lack it).
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
    # Small, high input resistance, modest Na/K, with Ih for resonance.
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
    # Large soma (C ~1.5), strong Na/K for robust action potentials, prominent CaT for
    # plateau potentials (bistability), strong M-current for spike-frequency adaptation,
    # persistent Na for amplification of synaptic inputs.
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
    # Fast-spiking, moderate size, strong K for brief spikes, CaT for rebound.
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
    # Strong persistent sodium supports persistent activity / UP states for working memory.
    # Enhanced Ih contributes to subthreshold resonance and temporal integration.
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
    # Fast, reliable spiking with high Na density. Minimal adaptation (sustains high-frequency
    # firing during odor responses). Moderate NaP contributes to subthreshold oscillations.
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
    # Autonomous pacemaker at 2-4 Hz driven by interplay of L-type Ca² (modeled as CaT)
    # and hyperpolarization-activated Ih. Low Na density, depolarized rest, slow kinetics.
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
    # Very high Na/K density for narrow, fast spikes. Zero adaptation currents — the defining
    # feature of FS cells. Drives perisomatic inhibition and gamma oscillations.
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
    # Strong CaT + Ih interplay generates subthreshold oscillations (~3-10 Hz).
    # Note: gap junctions (electrical coupling) are not modeled; intrinsic oscillatory
    # properties are captured but network-level synchronization requires chemical synapses.
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

# --- NEW Configuration & Data Classes (Replaces SimulationConfiguration) ---

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
    
    # Izhikevich - initialized from a default type
    izh_C_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["C"])
    izh_k_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["k"])
    izh_vr_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["vr"])
    izh_vt_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["vt"])
    izh_vpeak_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["vpeak"])
    izh_a_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["a"])
    izh_b_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["b"])
    izh_c_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["c_reset"])
    izh_d_val: float = field(default_factory=lambda: DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]["d_increment"])

    # Hodgkin-Huxley - initialized from a default type
    hh_C_m: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["C_m"])
    hh_g_Na_max: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["g_Na_max"])
    hh_g_K_max: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["g_K_max"])
    hh_g_L: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["g_L"])
    hh_E_Na: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["E_Na"])
    hh_E_K: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["E_K"])
    hh_E_L: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["E_L"])
    hh_v_rest_init: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["v_rest_hh"])
    hh_v_peak: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["v_peak_hh"])
    hh_m_init: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["m_init"])
    hh_h_init: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["h_init"])
    hh_n_init: float = field(default_factory=lambda: DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]["n_init"])
    hh_temperature_celsius: float = 37.0
    hh_q10_factor: float = 3.0
    # Optional extended HH currents. Zero conductance disables each one.
    hh_g_M_max: float = 0.0
    hh_m_current_tau_ms: float = 100.0
    hh_g_CaT_max: float = 0.0
    hh_E_CaT: float = 120.0
    hh_g_h_max: float = 0.0
    hh_E_h: float = -30.0
    hh_g_NaP_max: float = 0.0

    # AdEx parameters (single-compartment RS default; per-neuron variation can be added later)
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
    # NMDA conductance with voltage-dependent Mg²⁺ block (Jahr & Stevens 1990)
    enable_nmda: bool = False
    nmda_ratio: float = 0.4           # NMDA:AMPA conductance ratio (0 = no NMDA, 1 = equal)
    nmda_tau_decay: float = 100.0     # NMDA decay time constant (ms) — slow compared to AMPA
    nmda_tau_rise: float = 3.0        # NMDA rise time constant (ms)
    nmda_mg_concentration: float = 1.0  # Extracellular [Mg²⁺] in mM
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
    stp_U_per_type: list = None       # [U_ee, U_ei, U_ie, U_ii] — set in __post_init__
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
    enable_watts_strogatz: bool = True
    connectivity_k: int = 10
    connectivity_p_rewire: float = 0.1
    
    # C2: STDP (Spike-Timing-Dependent Plasticity) - Bi & Poo 1998, Caporale & Dan 2008
    enable_stdp: bool = True  # Enabled by default for biologically realistic learning
    stdp_a_plus: float = 0.01              # LTP amplitude (typical: 0.005-0.02)
    stdp_a_minus: float = 0.0105           # LTD amplitude (typical: slightly > A+)
    stdp_tau_plus_ms: float = 20.0         # LTP time constant (ms, typical: 15-25ms)
    stdp_tau_minus_ms: float = 20.0        # LTD time constant (ms, typical: 15-25ms)
    stdp_w_min: float = 0.0                # Minimum synaptic weight
    stdp_w_max: float = 2.0                # Maximum synaptic weight
    stdp_only_nearest_spike: bool = True   # Use only nearest spike pairs (more efficient)
    
    # C2: Reward-Modulated Plasticity (Three-factor learning rule) - Izhikevich 2007
    enable_reward_modulation: bool = True  # Enabled by default for reinforcement learning
    reward_learning_rate: float = 0.01     # Modulation strength (typical: 0.001-0.05)
    reward_eligibility_tau_ms: float = 1000.0  # Eligibility trace decay (ms, typical: 500-2000ms)
    reward_baseline: float = 0.0           # Expected reward (for prediction error)
    current_reward_signal: float = 0.0     # Current reward value (updated externally or via task)
    
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
                          f"HH gating kinetics have time constants ~0.1-1ms at 37°C; dt must resolve these.")
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
        from dataclasses import asdict
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

# --- Auto-tuned override support ---
AUTO_TUNED_OVERRIDES_PATH = os.path.join("simulation_profiles", "auto_tuned_overrides.json")
AUTO_TUNED_OVERRIDES = None  # Lazy-loaded mapping from combo key -> overrides dict

# --- Performance test stop flag ---
performance_test_stop_flag = threading.Event()  # Global flag to signal stop for benchmarks/optimization
performance_test_running_type = None  # Track which test is running: "benchmark" or "optimization"


def _load_auto_tuned_overrides_if_needed():
    """Lazily loads auto-tuned overrides from JSON if present.

    File format:
        {
          "schema_version": 1,
          "created_at": "...",
          "tuned_combinations": {
             "MODEL|PROFILE|HH_TYPE_OR_NONE": {"core_overrides": {...}, "metrics": {...}, ...},
             ...
          }
        }
    """
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
    """Returns auto-tuned overrides dict for a given (model, profile, HH preset) combo, if available.

    The key format is "MODEL|PROFILE|HH_TYPE_OR_NONE". For non-HH models we allow HH type to be "NONE".
    """
    _load_auto_tuned_overrides_if_needed()
    if not AUTO_TUNED_OVERRIDES:
        return None

    key_full = f"{neuron_model_type_str}|{profile_name_str}|{default_hh_type_str or 'NONE'}"
    entry = AUTO_TUNED_OVERRIDES.get(key_full)

    # For non-HH models, also allow a generic per-(model,profile) entry with HH type NONE
    if entry is None and neuron_model_type_str != NeuronModel.HODGKIN_HUXLEY.name:
        key_model_profile = f"{neuron_model_type_str}|{profile_name_str}|NONE"
        entry = AUTO_TUNED_OVERRIDES.get(key_model_profile)

    return entry


# --- Benchmark-derived hardware limits ---
BENCHMARK_RESULTS_PATH = os.path.join("benchmarks", "benchmark_results.json")
HARDWARE_LIMITS = None  # Lazy-loaded dict: model_name -> {max_neurons, max_conn, limits_table, hardware_note}


def _parse_benchmark_limits(results_data):
    """Parses benchmark_results.json and derives per-model hardware limits.

    Builds a table of tested configurations with their performance, and determines
    the maximum neuron/connection counts that succeeded for each model.

    Returns:
        dict: {
            "gpu_name": str,
            "per_model": {
                "IZHIKEVICH": {
                    "max_neurons_tested": 50000,
                    "max_conn_tested": 1000,
                    "realtime_max_neurons": 10000,   # Steps/s >= 1000/dt (real-time threshold)
                    "configs": [  # All tested configs for this model, sorted by size
                        {"neurons": 1000, "conn": 100, "steps_per_sec": 345.0, "mean_ms": 2.9, "gpu_gb": 1.2},
                        ...
                    ]
                },
                ...
            },
            "hardware_note": str  # Human-readable summary
        }
    """
    gpu_info = results_data.get("system_info", {})
    gpu_name = gpu_info.get("gpu_name", "Unknown GPU")
    gpu_mem_gb = gpu_info.get("gpu_memory_gb", 0)

    per_model = {}
    for entry in results_data.get("results", []):
        cfg = entry.get("config", {})
        metrics = entry.get("metrics", {})
        if not cfg or not metrics:
            continue

        model = cfg.get("neuron_model_type", "UNKNOWN")
        dt_ms = cfg.get("dt_ms", 1.0)
        neurons = cfg.get("num_neurons", 0)
        conn = cfg.get("connections_per_neuron", 0)
        steps_per_sec = metrics.get("steps_per_sec", 0)
        mean_ms = metrics.get("step_time_mean_ms", 0)
        gpu_gb = metrics.get("gpu_memory_used_gb", 0)

        if model not in per_model:
            per_model[model] = {
                "max_neurons_tested": 0,
                "max_conn_tested": 0,
                "realtime_max_neurons": 0,
                "dt_ms": dt_ms,
                "configs": []
            }

        info = per_model[model]
        info["configs"].append({
            "neurons": neurons, "conn": conn,
            "steps_per_sec": steps_per_sec, "mean_ms": mean_ms, "gpu_gb": gpu_gb
        })

        if neurons > info["max_neurons_tested"]:
            info["max_neurons_tested"] = neurons
        if conn > info["max_conn_tested"]:
            info["max_conn_tested"] = conn

        # Real-time threshold: steps_per_sec >= 1000/dt_ms (i.e., 1 second of bio time per wall second)
        realtime_threshold = 1000.0 / dt_ms if dt_ms > 0 else 1000.0
        if steps_per_sec >= realtime_threshold and neurons > info["realtime_max_neurons"]:
            info["realtime_max_neurons"] = neurons

    # Sort configs by neuron count then connection count
    for model, info in per_model.items():
        info["configs"].sort(key=lambda x: (x["neurons"], x["conn"]))

    # Build human-readable summary
    model_short = {"IZHIKEVICH": "Izh", "HODGKIN_HUXLEY": "HH", "ADEX": "AdEx"}
    lines = [f"{gpu_name} ({gpu_mem_gb:.0f}GB) — Benchmark Limits:"]
    for model_name in ["IZHIKEVICH", "HODGKIN_HUXLEY", "ADEX"]:
        if model_name not in per_model:
            continue
        info = per_model[model_name]
        short = model_short.get(model_name, model_name[:3])
        max_n = info["max_neurons_tested"]
        max_c = info["max_conn_tested"]
        dt = info["dt_ms"]

        # Find performance range for max neuron count
        max_n_configs = [c for c in info["configs"] if c["neurons"] == max_n]
        if max_n_configs:
            best_steps = max(c["steps_per_sec"] for c in max_n_configs)
            max_gpu = max(c["gpu_gb"] for c in max_n_configs)
            # Bio throughput: steps_per_sec * dt_ms = bio_ms per wall_second
            best_bio_ms_per_s = best_steps * dt
            worst_bio_ms_per_s = min(c["steps_per_sec"] for c in max_n_configs) * dt
            lines.append(f"  {short} (dt={dt}ms): up to {max_n//1000}K neurons, "
                         f"{worst_bio_ms_per_s:.0f}-{best_bio_ms_per_s:.0f} bio-ms/s, "
                         f"{max_gpu:.1f}GB VRAM")

    hardware_note = "\n".join(lines)

    return {
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_mem_gb,
        "per_model": per_model,
        "hardware_note": hardware_note
    }


def _load_benchmark_limits():
    """Loads and parses benchmark results at startup."""
    global HARDWARE_LIMITS
    if not os.path.exists(BENCHMARK_RESULTS_PATH):
        HARDWARE_LIMITS = {}
        return

    try:
        with open(BENCHMARK_RESULTS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        HARDWARE_LIMITS = _parse_benchmark_limits(data)
        print(f"Loaded hardware limits from {BENCHMARK_RESULTS_PATH} "
              f"({len(HARDWARE_LIMITS.get('per_model', {}))} models)")
    except Exception as e:
        print(f"Warning: Could not load benchmark limits from {BENCHMARK_RESULTS_PATH}: {e}")
        HARDWARE_LIMITS = {}


def get_hardware_limits_for_model(model_name):
    """Returns hardware limits dict for a specific neuron model, or None if unavailable.

    Returns:
        dict with keys: max_neurons_tested, max_conn_tested, realtime_max_neurons, dt_ms, configs
        or None if no benchmark data exists for this model.
    """
    if HARDWARE_LIMITS is None:
        _load_benchmark_limits()
    if not HARDWARE_LIMITS:
        return None
    return HARDWARE_LIMITS.get("per_model", {}).get(model_name)


def get_hardware_note():
    """Returns the human-readable hardware note from benchmark results."""
    if HARDWARE_LIMITS is None:
        _load_benchmark_limits()
    return HARDWARE_LIMITS.get("hardware_note", "") if HARDWARE_LIMITS else ""


def check_config_against_limits(model_name, num_neurons, conn_per_neuron):
    """Checks a proposed config against benchmark-derived limits.

    Returns:
        tuple: (is_safe: bool, warning_message: str or None)
            is_safe = True means config is within tested limits
            warning_message = None if safe, otherwise a descriptive string
    """
    limits = get_hardware_limits_for_model(model_name)
    if limits is None:
        return True, None  # No benchmark data — can't warn

    max_tested_n = limits["max_neurons_tested"]
    max_tested_c = limits["max_conn_tested"]

    if num_neurons > max_tested_n:
        return False, (f"WARNING: {num_neurons} neurons exceeds benchmark-tested maximum "
                       f"({max_tested_n} for {model_name}). May cause OOM or severe slowdown.")
    if conn_per_neuron > max_tested_c:
        return False, (f"WARNING: {conn_per_neuron} conn/neuron exceeds benchmark-tested maximum "
                       f"({max_tested_c} for {model_name}). May cause OOM.")

    # Check if this specific combo was tested — find closest match
    configs = limits["configs"]
    matching = [c for c in configs if c["neurons"] == num_neurons and c["conn"] == conn_per_neuron]
    if matching:
        gpu_gb = matching[0]["gpu_gb"]
        steps_s = matching[0]["steps_per_sec"]
        return True, None

    # Interpolate: check if a similar-sized config was tested and had high VRAM
    larger_configs = [c for c in configs if c["neurons"] >= num_neurons and c["conn"] >= conn_per_neuron]
    if larger_configs:
        best_match = larger_configs[0]  # Smallest config >= requested
        return True, None

    return True, None  # Within max bounds but exact combo not tested — assume OK


# Compatibility class for old SimulationConfiguration usage
class SimulationConfiguration:
    """Legacy configuration class for backward compatibility. Wraps the new dataclass structure."""
    def __init__(self):
        # Core Simulation Timing & Structure
        self.total_simulation_time_ms = 60000.0 # Total duration of the simulation in milliseconds
        self.dt_ms = 1.000 # Simulation time step in milliseconds (e.g., 0.1 ms for Izh, 0.025 for HH)
        self.num_neurons = 1000 # Total number of neurons in the network
        self.connections_per_neuron = 100 # Average number of outgoing connections per neuron (for spatial/random)
        self.num_traits = 5 # Number of distinct neuron traits/types for coloring/behavioral differences
        self.seed = -1 # Random seed for reproducibility (-1 for random initialization)

        # Neuron Model Selection
        self.neuron_model_type = NeuronModel.IZHIKEVICH.name # Current neuron model ('IZHIKEVICH', 'HODGKIN_HUXLEY', or 'ADEX')
        self.default_neuron_type_izh = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name # Default Izhikevich type if trait mapping fails
        self.default_neuron_type_hh = NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name # Default Hodgkin-Huxley type

        # High-level structural profile (brain region / mode)
        self.neural_profile_name = "GENERIC_UNSTRUCTURED"

        # Izhikevich Model Parameters (2007 Formulation - Global defaults, can be overridden per-neuron by trait)
        # These are initialized from a default Izhikevich neuron type (e.g., RS Cortical Pyramidal)
        rs_params_2007 = DefaultIzhikevichParamsManager.PARAMS[NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL]
        self.izh_C_val = rs_params_2007["C"]       # Membrane capacitance (pF)
        self.izh_k_val = rs_params_2007["k"]       # Constant related to Na+ channel kinetics (nS/mV or similar)
        self.izh_vr_val = rs_params_2007["vr"]     # Resting membrane potential (mV)
        self.izh_vt_val = rs_params_2007["vt"]     # Instantaneous threshold potential (mV)
        self.izh_vpeak_val = rs_params_2007["vpeak"] # Spike cutoff/peak value (mV)
        self.izh_a_val = rs_params_2007["a"]       # Timescale of recovery variable u (1/ms)
        self.izh_b_val = rs_params_2007["b"]       # Sensitivity of u to subthreshold fluctuations (nS)
        self.izh_c_val = rs_params_2007["c_reset"] # After-spike reset value of v (mV)
        self.izh_d_val = rs_params_2007["d_increment"] # After-spike increment of u (pA)

        # Legacy Izhikevich Model Parameters (Not actively used if 2007 formulation is primary)
        self.lif_v_rest = -65.0 # Legacy resting potential (mV) - used if no Izhikevich params available
        self.initial_firing_threshold = -40.0 # Legacy firing threshold (mV)
        self.initial_threshold_variation = 1.0 # Variation for legacy threshold (mV)

        # Hodgkin-Huxley Model Parameters (Global defaults, can be overridden per-neuron by trait)
        # Initialized from a default HH neuron type (e.g., L5 Cortical Pyramidal RS)
        hh_defaults = DefaultHodgkinHuxleyParams.PARAMS[NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS]
        self.hh_C_m = hh_defaults["C_m"]             # Membrane capacitance (uF/cm^2)
        self.hh_g_Na_max = hh_defaults["g_Na_max"]   # Max Na+ conductance (mS/cm^2)
        self.hh_g_K_max = hh_defaults["g_K_max"]     # Max K+ conductance (mS/cm^2)
        self.hh_g_L = hh_defaults["g_L"]             # Leak conductance (mS/cm^2)
        self.hh_E_Na = hh_defaults["E_Na"]           # Na+ reversal potential (mV)
        self.hh_E_K = hh_defaults["E_K"]             # K+ reversal potential (mV)
        self.hh_E_L = hh_defaults["E_L"]             # Leak reversal potential (mV)
        self.hh_v_rest_init = hh_defaults["v_rest_hh"] # Initial resting Vm for HH model (mV)
        self.hh_v_peak = hh_defaults["v_peak_hh"]    # Spike peak for HH model (mV)
        self.hh_m_init = hh_defaults["m_init"]       # Initial m gating variable value
        self.hh_h_init = hh_defaults["h_init"]       # Initial h gating variable value
        self.hh_n_init = hh_defaults["n_init"]       # Initial n gating variable value
        self.hh_temperature_celsius = 37.0           # Temperature for HH kinetics (Celsius)
        self.hh_q10_factor = 3.0                     # Q10 temperature coefficient for HH rates
        # Optional extended HH currents (all disabled by default)
        self.hh_g_M_max = hh_defaults.get("g_M_max", 0.0)            # Max M-current conductance (mS/cm^2); 0 disables
        self.hh_m_current_tau_ms = 100.0 # Approximate activation time constant for M-current (ms)
        self.hh_g_CaT_max = hh_defaults.get("g_CaT_max", 0.0)
        self.hh_E_CaT = hh_defaults.get("E_CaT", 120.0)
        self.hh_g_h_max = hh_defaults.get("g_h_max", 0.0)
        self.hh_E_h = hh_defaults.get("E_h", -30.0)
        self.hh_g_NaP_max = hh_defaults.get("g_NaP_max", 0.0)

        # AdEx Model Parameters (Adaptive Exponential IF)
        self.adex_C = 281.0          # Membrane capacitance (pF)
        self.adex_g_L = 30.0         # Leak conductance (nS)
        self.adex_E_L = -70.6        # Leak reversal (mV)
        self.adex_V_T = -50.4        # Threshold (mV)
        self.adex_Delta_T = 2.0      # Slope factor (mV)
        self.adex_a = 4.0            # Subthreshold coupling (nS)
        self.adex_tau_w = 144.0      # Adaptation time constant (ms)
        self.adex_b = 80.5           # Spike-triggered increment (pA)
        self.adex_V_r = -70.6        # Reset voltage (mV)
        self.adex_V_peak = -40.0     # Spike detection threshold (mV)

        # External drive scaling (tuned per model/profile; 1.0 = baseline range)
        self.hh_external_drive_scale = 1.0
        self.adex_external_drive_scale = 1.0

        # Basic Neuron & Synapse Properties
        self.refractory_period_steps = 2 # Absolute refractory period in simulation steps (dt units)
        self.syn_reversal_potential_e = 0.0 # Reversal potential for excitatory synapses (mV)
        self.syn_reversal_potential_i = -75.0 # Reversal potential for inhibitory synapses (mV) — Cl- Nernst at 37C
        self.syn_tau_g_e = 5.0 # Time constant for excitatory synaptic conductance decay (ms)
        self.syn_tau_g_i = 10.0 # Time constant for inhibitory synaptic conductance decay (ms)
        self.propagation_strength = 0.05 # Scaling factor for excitatory synaptic conductance increase per spike
        self.inhibitory_propagation_strength = 0.105 # Scaled for E_inh=-75mV (was 0.15 at -70mV)
        self.max_synaptic_delay_ms = 20.0 # Maximum synaptic delay in ms (Not fully implemented for individual delays yet)

        # Inhibitory Neuron Configuration
        self.enable_inhibitory_neurons = True # Whether to model inhibitory neurons
        self.inhibitory_trait_index = 1 # Trait index designated as inhibitory (0-indexed)
        self.inhibitory_trait_indices = [] # Optional list of inhibitory trait indices (overrides inhibitory_trait_index if non-empty)

        # Hebbian Learning / Long-Term Potentiation (LTP)
        self.enable_hebbian_learning = True # Enable Hebbian-like weight potentiation
        self.hebbian_learning_rate = 0.0005 # Learning rate for LTP
        self.hebbian_weight_decay = 0.00001 # Multiplicative weight decay factor per step
        self.hebbian_min_weight = 0.05 # Minimum synaptic weight
        self.hebbian_max_weight = 1.0 # Maximum synaptic weight

        # Short-Term Plasticity (STP) - Tsodyks-Markram model
        self.enable_short_term_plasticity = True # Enable STP
        self.stp_U = 0.15 # STP U parameter (baseline utilization of synaptic resources)
        self.stp_tau_d = 200.0 # STP tau_d (depression time constant, ms)
        self.stp_tau_f = 50.0 # STP tau_f (facilitation time constant, ms)
        # Per-connection-type STP [E->E, E->I, I->E, I->I]
        self.enable_per_type_stp = True
        self.stp_U_per_type = [0.5, 0.5, 0.25, 0.25]
        self.stp_tau_d_per_type = [200.0, 200.0, 100.0, 100.0]
        self.stp_tau_f_per_type = [20.0, 20.0, 50.0, 50.0]

        # Homeostatic Plasticity (Adaptive Thresholds for Izhikevich model)
        self.enable_homeostasis = True # Enable homeostatic threshold adaptation
        self.homeostasis_target_rate = 0.02 # Target firing rate (spikes per dt step)
        self.homeostasis_threshold_adapt_rate = 0.0005 # Adaptation rate for firing thresholds (slower, biologically grounded)
        self.homeostasis_ema_alpha = 0.0002 # Alpha for EMA of neuron activity (tau ~5s at dt=1ms)
        self.homeostasis_threshold_min = -55.0 # Minimum firing threshold (mV)
        self.homeostasis_threshold_max = -30.0 # Maximum firing threshold (mV)

        # Synaptic Scaling (Turrigiano 2008) - multiplicative excitatory weight scaling
        self.enable_synaptic_scaling = False
        self.synaptic_scaling_rate = 0.001 # Slow scaling rate (operates on seconds timescale)

        # NMDA conductance with voltage-dependent Mg²⁺ block (Jahr & Stevens 1990)
        self.enable_nmda = False
        self.nmda_ratio = 0.4             # NMDA:AMPA conductance ratio
        self.nmda_tau_decay = 100.0       # NMDA decay time constant (ms)
        self.nmda_tau_rise = 3.0          # NMDA rise time constant (ms)
        self.nmda_mg_concentration = 1.0  # Extracellular [Mg²⁺] in mM

        # STDP (Spike-Timing Dependent Plasticity)
        self.enable_stdp = True
        self.stdp_a_plus = 0.01           # LTP amplitude
        self.stdp_a_minus = 0.0105        # LTD amplitude
        self.stdp_tau_plus_ms = 20.0      # LTP time constant (ms)
        self.stdp_tau_minus_ms = 20.0     # LTD time constant (ms)
        self.stdp_w_min = 0.0             # Minimum STDP weight
        self.stdp_w_max = 2.0             # Maximum STDP weight
        self.stdp_only_nearest_spike = True

        # Reward-Modulated Plasticity
        self.enable_reward_modulation = True
        self.reward_learning_rate = 0.01
        self.reward_eligibility_tau_ms = 1000.0
        self.reward_baseline = 0.0
        self.current_reward_signal = 0.0

        # Structural Plasticity
        self.enable_structural_plasticity = True
        self.struct_plast_formation_rate = 1e-6
        self.struct_plast_elimination_rate = 5e-7
        self.struct_plast_weight_threshold = 0.05
        self.struct_plast_target_density = 0.1
        self.struct_plast_distance_kernel = "exp_decay"
        self.struct_plast_distance_scale = 20.0
        self.struct_plast_update_interval_steps = 100
        self.struct_plast_activity_bias = 0.5  # Co-activity bias for synapse formation

        # Parameter Heterogeneity (Phase B2)
        self.enable_parameter_heterogeneity = False # Enable per-neuron parameter variability
        self.heterogeneity_seed = -1 # Seed for heterogeneity sampling (-1 = use main seed)
        self.heterogeneity_distributions = {} # Dict of parameter distributions (empty = use defaults)
        
        # Enhanced Channel Noise (Phase B4)
        self.enable_conductance_noise = False # Enable multiplicative conductance noise (HH only)
        self.conductance_noise_relative_std = 0.05 # Relative std for conductance noise (5%)
        self.enable_ou_process = False # Enable Ornstein-Uhlenbeck background current
        self.ou_mean_current_pA = 0.0 # OU process mean current (pA)
        self.ou_std_current_pA = 100.0 # OU process std current (pA)
        self.ou_tau_ms = 15.0 # OU process time constant (ms)
        self.ou_seed = -1 # Seed for OU process (-1 = use main seed)
        
        # Hardware Performance Note (populated by viz_benchmark.py)
        self.hardware_performance_note = "" # Note about hardware realtime capacity

        # Network Generation (Watts-Strogatz specific, if spatial fallback is not used)
        self.enable_watts_strogatz = True # Use Watts-Strogatz generator for connections
        self.connectivity_k = 10 # K for Watts-Strogatz (number of nearest neighbors in ring)
        self.connectivity_p_rewire = 0.1 # Rewiring probability for Watts-Strogatz

        # Runtime State (Managed by SimulationBridge, not typically saved in profiles)
        self.current_time_ms = 0.0 # Current simulation time in ms
        self.current_time_step = 0 # Current simulation step
        self.is_running = False # Simulation is actively running
        self.is_paused = False # Simulation is paused
        self.simulation_speed_factor = 1.0 # Multiplier for simulation speed relative to real-time

        # Visualization & Network Definition Data (Populated during initialization)
        self.network_definition_dict = {"neuron_groups": [], "connections": []} # For potential export/import
        self.neuron_positions_x = [] # List of X coordinates for 2D projection/GL
        self.neuron_positions_y = [] # List of Y coordinates for 2D projection/GL
        self.neuron_types_list_for_viz = [] # List of string types for each neuron for visualization filters
        self.max_delay_steps = int(self.max_synaptic_delay_ms / self.dt_ms) if self.dt_ms > 0 else 200 # Max delay in dt steps

        # 3D Visualization & Camera Parameters
        self.volume_min_x = -50.0; self.volume_max_x = 50.0 # X-axis bounds of the simulation volume
        self.volume_min_y = -50.0; self.volume_max_y = 50.0 # Y-axis bounds
        self.volume_min_z = -50.0; self.volume_max_z = 50.0 # Z-axis bounds

        # Camera spherical coordinates for orbiting
        self.camera_center_x = 0.0; self.camera_center_y = 0.0; self.camera_center_z = 0.0 # Point camera looks at
        self.camera_radius = 150.0 # Distance from center to camera
        self.camera_azimuth_angle = 0.0 # Horizontal angle (radians)
        self.camera_elevation_angle = 0.0 # Vertical angle (radians)
        
        # Camera view properties (derived from spherical for gluLookAt, but kept for potential direct use or DPG)
        self.camera_eye_x = 0.0; self.camera_eye_y = 0.0; self.camera_eye_z = self.camera_radius # Initial eye position
        self.camera_up_x = 0.0; self.camera_up_y = 1.0; self.camera_up_z = 0.0 # Up vector for camera

        self.camera_fov = 60.0 # Field of view in degrees
        self.camera_near_clip = 0.1 # Near clipping plane
        self.camera_far_clip = 1000.0 # Far clipping plane

        # Mouse interaction state for camera control
        self.mouse_last_x = 0; self.mouse_last_y = 0 # Last mouse position for calculating deltas
        self.mouse_left_button_down = False # Is left mouse button currently pressed
        self.mouse_right_button_down = False # Is right mouse button currently pressed


    def reset_simulation_time_and_counters(self):
        """Resets time-dependent simulation variables."""
        self.current_time_ms = 0.0
        self.current_time_step = 0

    def to_dict(self):
        """Serializes the configuration to a dictionary."""
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and not callable(v)}
        # Ensure Enum types are stored as their names for JSON compatibility
        if isinstance(data.get('neuron_model_type'), Enum):
            data['neuron_model_type'] = data['neuron_model_type'].name
        if isinstance(data.get('default_neuron_type_izh'), Enum):
            data['default_neuron_type_izh'] = data['default_neuron_type_izh'].name
        if isinstance(data.get('default_neuron_type_hh'), Enum):
            data['default_neuron_type_hh'] = data['default_neuron_type_hh'].name
        return data

    @classmethod
    def from_dict(cls, data):
        """Creates a SimulationConfiguration instance from a dictionary with robust type casting."""
        config = cls() # Initialize with class defaults

        def _to_python_bool(value, default_val_if_none=False):
            if value is None:
                return default_val_if_none
            if isinstance(value, bool):
                return value
            if hasattr(value, 'item') and isinstance(value.item(), bool): # For numpy.bool_
                return value.item()
            if isinstance(value, (int, float)): # Treat 0 as False, non-zero as True
                return value != 0
            if isinstance(value, str):
                if value.lower() == 'true':
                    return True
                elif value.lower() == 'false':
                    return False
            # Fallback or warning if conversion is ambiguous for other types
            # print(f"Warning: Could not convert value '{value}' (type: {type(value)}) to bool. Using default: {default_val_if_none}")
            return default_val_if_none

        for key, value_from_data in data.items():
            if hasattr(config, key):
                default_value_for_key = getattr(config, key) # Get default type/value from cls instance

                # Handle Enums first
                if key == 'neuron_model_type' and isinstance(value_from_data, str):
                    try: setattr(config, key, NeuronModel[value_from_data].name)
                    except KeyError: setattr(config, key, cls().neuron_model_type)
                    continue
                elif key == 'default_neuron_type_izh' and isinstance(value_from_data, str):
                    try: setattr(config, key, NeuronType[value_from_data].name)
                    except KeyError: setattr(config, key, cls().default_neuron_type_izh)
                    continue
                elif key == 'default_neuron_type_hh' and isinstance(value_from_data, str):
                    try: setattr(config, key, NeuronType[value_from_data].name)
                    except KeyError: setattr(config, key, cls().default_neuron_type_hh)
                    continue

                # Handle Booleans (many config flags are bools)
                if isinstance(default_value_for_key, bool):
                    setattr(config, key, _to_python_bool(value_from_data, default_value_for_key))
                    continue

                # Handle Integers
                if isinstance(default_value_for_key, int) and not isinstance(default_value_for_key, bool): # Exclude bools here
                    if value_from_data is not None:
                        try:
                            setattr(config, key, int(value_from_data))
                        except (ValueError, TypeError):
                            print(f"Warning: Could not convert {key} value '{value_from_data}' to int. Using default: {default_value_for_key}.")
                            setattr(config, key, default_value_for_key)
                    else: # Value from data is None, use default
                        setattr(config, key, default_value_for_key)
                    continue
                
                # Handle Floats
                if isinstance(default_value_for_key, float):
                    if value_from_data is not None:
                        try:
                            setattr(config, key, float(value_from_data))
                        except (ValueError, TypeError):
                            print(f"Warning: Could not convert {key} value '{value_from_data}' to float. Using default: {default_value_for_key}.")
                            setattr(config, key, default_value_for_key)
                    else: # Value from data is None, use default
                        setattr(config, key, default_value_for_key)
                    continue
                
                # For other types (like lists, dicts, or strings not covered above), assign directly
                # This also handles cases where default_value_for_key is None, and value_from_data might be None or a valid value.
                setattr(config, key, value_from_data)

        # Ensure dt_ms is valid after loading and is float
        config.dt_ms = float(getattr(config, 'dt_ms', 0.1))
        if config.dt_ms <= 0: config.dt_ms = 0.1

        config.max_delay_steps = int(config.max_synaptic_delay_ms / config.dt_ms) if config.dt_ms > 0 else 200
        
        # Ensure camera FOV is float
        config.camera_fov = float(getattr(config, 'camera_fov', 60.0))

        # Re-check critical numeric default fallbacks for parameters that might be missing entirely from older files
        default_instance_for_fallback = cls()
        numeric_param_keys_to_check = [
            'izh_C_val', 'izh_k_val', 'izh_vr_val', 'izh_vt_val', 'izh_vpeak_val',
            'izh_a_val', 'izh_b_val', 'izh_c_val', 'izh_d_val',
            'hh_C_m', 'hh_g_Na_max', 'hh_g_K_max', 'hh_g_L', 'hh_E_Na', 'hh_E_K', 'hh_E_L',
            'hh_v_rest_init', 'hh_v_peak', 'hh_temperature_celsius', 'hh_q10_factor',
            'hh_g_M_max', 'hh_m_current_tau_ms',
            'hh_g_CaT_max', 'hh_E_CaT', 'hh_g_h_max', 'hh_E_h', 'hh_g_NaP_max',
            'adex_C', 'adex_g_L', 'adex_E_L', 'adex_V_T', 'adex_Delta_T', 'adex_a', 'adex_tau_w', 'adex_b', 'adex_V_r', 'adex_V_peak',
            'volume_min_x', 'volume_max_x', 'volume_min_y', 'volume_max_y', 'volume_min_z', 'volume_max_z',
            'camera_eye_x', 'camera_eye_y', 'camera_eye_z',
            'camera_center_x', 'camera_center_y', 'camera_center_z',
            'camera_up_x', 'camera_up_y', 'camera_up_z',
            'camera_near_clip', 'camera_far_clip', 'camera_radius',
            'connections_per_neuron', 'seed', 'num_traits', 'connectivity_k' # ints
        ]
        for param_key in numeric_param_keys_to_check:
            if not hasattr(config, param_key) or getattr(config, param_key) is None:
                fallback_val = getattr(default_instance_for_fallback, param_key)
                setattr(config, param_key, fallback_val) # Set to default
                print(f"Info: {param_key} was missing or None, set to default {fallback_val}")
            # Ensure correct type after potential None or load
            current_val = getattr(config, param_key)
            default_type_val = getattr(default_instance_for_fallback, param_key)
            if isinstance(default_type_val, bool): # Should have been handled by _to_python_bool logic primarily
                setattr(config, param_key, _to_python_bool(current_val, default_type_val))
            elif isinstance(default_type_val, int):
                setattr(config, param_key, int(current_val) if current_val is not None else default_type_val)
            elif isinstance(default_type_val, float):
                setattr(config, param_key, float(current_val) if current_val is not None else default_type_val)

        return config

# --- Neural Structure Profiles (brain-region presets) ---
# These presets describe high-level mixtures of neuron classes and E/I balance.
# They are intentionally conservative and primarily influence trait assignment and neuron types.
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

# --- HDF5 Helper Functions ---
def save_dict_to_hdf5_attrs(h5_group_or_file, data_dict):
    """Saves dictionary items as attributes to an HDF5 group or file."""
    for key, value in data_dict.items():
        try:
            if value is None:
                h5_group_or_file.attrs[key] = "NoneType" # Special string for None
            elif isinstance(value, (list, tuple, dict)):
                 # For complex types, store as JSON string
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
                    # Attempt to parse if it's a JSON string
                    data_dict[key] = json.loads(value)
                except json.JSONDecodeError:
                    # Not a JSON string, keep as string
                    data_dict[key] = value
        else:
            data_dict[key] = value
    return data_dict

# --- CuPy Fused Kernels ---
@cp.fuse()
def fused_izhikevich_legacy_dynamics_update(v, u, a, b, total_I, dt):
    """Fused kernel for legacy Izhikevich model dynamics."""
    dv = (0.04 * v**2 + 5 * v + 140 - u + total_I)
    du = a * (b * v - u)
    v_new = v + dv * dt
    u_new = u + du * dt
    return v_new, u_new

@cp.fuse()
def fused_izhikevich2007_dynamics_update(v, u, C_param, k_param, vr_param, vt_param, a_param, b_param, total_synaptic_current, dt):
    """Fused kernel for Izhikevich 2007 model dynamics."""
    # Ensure C_param is not zero to prevent division by zero errors.
    C_param_safe = cp.where(C_param == 0.0, 1.0, C_param) # Use 1.0 as a safe non-zero default if C is 0
    
    # Differential equation for membrane potential v
    dv_dt = (k_param * (v - vr_param) * (v - vt_param) - u + total_synaptic_current) / C_param_safe
    # Differential equation for recovery variable u
    du_dt = a_param * (b_param * (v - vr_param) - u)
    
    # Euler integration to update v and u
    v_new = v + dv_dt * dt
    u_new = u + du_dt * dt
    return v_new, u_new

@cp.fuse()
def fused_hodgkin_huxley_dynamics_update(V, m, h, n, I_syn, dt, C_m, g_Na_max, g_K_max, g_L, E_Na, E_K, E_L, temperature_celsius, q10_factor):
    """Fused kernel for Hodgkin-Huxley model dynamics, including temperature effects."""
    # Base temperature for original HH kinetics (typically 6.3°C or similar)
    BASE_HH_KINETICS_TEMP_C = 6.3 
    # Temperature adjustment factor (phi) using Q10
    phi = q10_factor**((temperature_celsius - BASE_HH_KINETICS_TEMP_C) / 10.0)

    # Rate functions (alpha, beta) for gating variables m, h, n
    # Original HH equations, adjusted for V in mV.
    # Handling for V = -40 (for alpha_m) and V = -55 (for alpha_n) to avoid division by zero in expm1.
    # expm1(x) = exp(x) - 1. For small x, expm1(x) approx x.
    # If V = -40, then -(V+40)/10 = 0. The limit of -0.1*x / (exp(-x/10)-1) as x->-40 is 1.0.
    # (Using L'Hopital's rule: d/dx (-0.1(x+40)) / d/dx (exp(-(x+40)/10)-1) = -0.1 / (-0.1 * exp(-(x+40)/10)) = exp((x+40)/10) -> exp(0) = 1)
    
    v_plus_40 = V + 40.0 # For m-gate alpha expression
    alpha_m_orig = cp.where(v_plus_40 == 0, 1.0 * 0.1 * 10.0 , -0.1 * v_plus_40 / cp.expm1(-v_plus_40 / 10.0)) # Corrected limit handling
    beta_m_orig  = 4.0 * cp.exp(-(V + 65.0) / 18.0)

    alpha_h_orig = 0.07 * cp.exp(-(V + 65.0) / 20.0)
    beta_h_orig  = 1.0 / (cp.exp(-(V + 35.0) / 10.0) + 1.0)

    v_plus_55 = V + 55.0 # For n-gate alpha expression
    alpha_n_orig = cp.where(v_plus_55 == 0, 0.1 * 0.01 * 10.0, -0.01 * v_plus_55 / cp.expm1(-v_plus_55 / 10.0)) # Corrected limit handling
    beta_n_orig  = 0.125 * cp.exp(-(V + 65.0) / 80.0)
    
    # Apply temperature correction to rate constants
    alpha_m = alpha_m_orig * phi; beta_m  = beta_m_orig  * phi
    alpha_h = alpha_h_orig * phi; beta_h  = beta_h_orig  * phi
    alpha_n = alpha_n_orig * phi; beta_n  = beta_n_orig  * phi

    # Update gating variables using analytical solution for first-order kinetics (assuming V is constant during dt)
    # m_new = m_inf - (m_inf - m_old) * exp(-dt / tau_m)
    # where m_inf = alpha_m / (alpha_m + beta_m) and tau_m = 1 / (alpha_m + beta_m)

    # Epsilon-based safe division eliminates branching overhead from cp.where()
    # For biophysically valid voltages, alpha+beta > 0 always; epsilon is only a numerical guard.
    # This avoids 6 cp.where() calls and 3 cp.isinf() calls per step (3-5% HH speedup).
    _EPS_GATE = 1e-12  # Small enough to not affect dynamics, large enough for float32 safety
    sum_alpha_beta_m = alpha_m + beta_m + _EPS_GATE
    m_inf = alpha_m / sum_alpha_beta_m
    m_new = m_inf + (m - m_inf) * cp.exp(-dt * sum_alpha_beta_m)

    sum_alpha_beta_h = alpha_h + beta_h + _EPS_GATE
    h_inf = alpha_h / sum_alpha_beta_h
    h_new = h_inf + (h - h_inf) * cp.exp(-dt * sum_alpha_beta_h)

    sum_alpha_beta_n = alpha_n + beta_n + _EPS_GATE
    n_inf = alpha_n / sum_alpha_beta_n
    n_new = n_inf + (n - n_inf) * cp.exp(-dt * sum_alpha_beta_n)
    
    # Clip gating variables to be between 0 and 1
    m_new = cp.clip(m_new, 0.0, 1.0); h_new = cp.clip(h_new, 0.0, 1.0); n_new = cp.clip(n_new, 0.0, 1.0)

    # Ionic currents
    I_Na = g_Na_max * (m_new**3) * h_new * (V - E_Na) # Sodium current
    I_K  = g_K_max * (n_new**4) * (V - E_K)   # Potassium current
    I_L  = g_L * (V - E_L)                    # Leak current
    I_ion = I_Na + I_K + I_L                  # Total ionic current

    # Membrane potential update
    dV_dt = (I_syn - I_ion) / C_m # dV/dt = (I_external - I_ionic) / C_m
    V_new = V + dV_dt * dt        # Euler integration
    return V_new, m_new, h_new, n_new

@cp.fuse()
def fused_hh_m_current_update(V, p_old, dt, g_M_max, E_K, tau_m_ms, phi):
    """Optional slow K+ M-current for extended HH models.

    Uses a simple sigmoidal steady-state activation with a first-order time course.
    g_M_max = 0.0 disables the current without branching.
    phi: Q10 temperature correction factor (same as main HH kinetics).
    """
    # Steady-state activation (approximate; centered around -35 mV)
    p_inf = 1.0 / (1.0 + cp.exp(-(V + 35.0) / 10.0))
    # Time constant (ms) with Q10 temperature correction — faster at higher temperatures
    # Literature: M-current tau ranges 30-200ms depending on cell type and temperature
    tau_safe = cp.maximum(tau_m_ms / phi, 1e-3)
    # First-order update assuming V is approximately constant over dt
    p_new = p_inf + (p_old - p_inf) * cp.exp(-dt / tau_safe)
    # M-current (K+): uses potassium reversal potential E_K
    I_M = g_M_max * p_new * (V - E_K)
    return p_new, I_M

@cp.fuse()
def fused_hh_CaT_current_update(V, m_old, h_old, dt, g_CaT_max, E_CaT, phi):
    """Low-threshold T-type Ca2+ current for extended HH models.

    Uses simple sigmoidal steady-state activation/inactivation with Q10-corrected time constants.
    phi: Q10 temperature correction factor.
    """
    # Steady-state activation/inactivation (approximate, thalamic-like)
    m_inf = 1.0 / (1.0 + cp.exp(-(V + 50.0) / 7.4))
    h_inf = 1.0 / (1.0 + cp.exp((V + 80.0) / 5.0))
    # Temperature-corrected time constants (Q10 ~3-4 for Ca2+ channels)
    tau_m = 5.0 / phi   # ms, fast activation (scaled by temperature)
    tau_h = 20.0 / phi  # ms, slower inactivation (scaled by temperature)
    m_new = m_inf + (m_old - m_inf) * cp.exp(-dt / tau_m)
    h_new = h_inf + (h_old - h_inf) * cp.exp(-dt / tau_h)
    I_CaT = g_CaT_max * (m_new ** 2) * h_new * (V - E_CaT)
    return m_new, h_new, I_CaT

@cp.fuse()
def fused_hh_h_current_update(V, q_old, dt, g_h_max, E_h, phi):
    """Hyperpolarization-activated mixed cation current (I_h) for extended HH models.

    phi: Q10 temperature correction factor. I_h has Q10 ~3-4 (Magee 1998).
    """
    # Steady-state activation: more active at hyperpolarized voltages
    q_inf = 1.0 / (1.0 + cp.exp((V + 75.0) / 5.5))
    # Temperature-corrected time constant
    tau_q = 100.0 / phi  # ms, slow activation (faster at mammalian temperatures)
    q_new = q_inf + (q_old - q_inf) * cp.exp(-dt / tau_q)
    I_h = g_h_max * q_new * (V - E_h)
    return q_new, I_h

@cp.fuse()
def fused_hh_NaP_current_update(V, p_old, dt, g_NaP_max, E_Na, phi):
    """Persistent Na+ current for extended HH models.

    phi: Q10 temperature correction factor. NaP kinetics scale similarly to transient Na+ (Q10 ~3).
    """
    p_inf = 1.0 / (1.0 + cp.exp(-(V + 55.0) / 5.0))
    # Temperature-corrected time constant
    tau_p = 5.0 / phi  # ms, relatively fast activation (faster at mammalian temperatures)
    p_new = p_inf + (p_old - p_inf) * cp.exp(-dt / tau_p)
    I_NaP = g_NaP_max * p_new * (V - E_Na)
    return p_new, I_NaP

@cp.fuse()
def fused_adex_dynamics_update(V, w, I_syn, dt, C, g_L, E_L, V_T, Delta_T, a, tau_w):
    """Fused kernel for Adaptive Exponential Integrate-and-Fire (AdEx) dynamics.

    All parameters can be either scalars or arrays broadcastable to V.
    Units are assumed to be consistent with the calling code (pF, nS, mV, ms, pA).
    """
    C_safe = cp.where(C == 0.0, 1.0, C)
    tau_w_safe = cp.maximum(tau_w, 1e-9)
    Delta_T_safe = cp.maximum(Delta_T, 1e-9)  # Prevent division by zero

    # Clamp exponential argument to prevent overflow. For float32:
    # exp(-20) ≈ 2e-9 (underflows gracefully), exp(5) ≈ 148 (safe with g_L*Delta_T scaling)
    # Wider range improves subthreshold accuracy near threshold without numerical risk.
    exp_arg = cp.clip((V - V_T) / Delta_T_safe, -20.0, 5.0)

    # Membrane equation: C dV/dt = -g_L (V - E_L) + g_L * Delta_T * exp((V - V_T)/Delta_T) - w + I_syn
    dV_dt = (-g_L * (V - E_L) + g_L * Delta_T * cp.exp(exp_arg) - w + I_syn) / C_safe
    # Adaptation variable: tau_w dw/dt = a (V - E_L) - w
    dw_dt = (a * (V - E_L) - w) / tau_w_safe
    V_new = V + dV_dt * dt
    w_new = w + dw_dt * dt
    return V_new, w_new

@cp.fuse()
def fused_conductance_decay_and_current(g_e, g_i, decay_e, decay_i, v, E_e, E_i):
    """Fused kernel for synaptic conductance decay and calculating synaptic current."""
    # Decay conductances
    g_e_new = g_e * decay_e # Excitatory conductance decay
    g_i_new = g_i * decay_i # Inhibitory conductance decay
    # Calculate total synaptic current based on new conductances
    I_syn = g_e_new * (E_e - v) + g_i_new * (E_i - v) # I_syn = g_e*(E_e - V) + g_i*(E_i - V)
    return g_e_new, g_i_new, I_syn

@cp.fuse()
def fused_nmda_update_and_current(g_nmda, g_nmda_rise, decay_nmda, decay_nmda_rise, v, E_nmda, mg_conc):
    """Fused kernel for NMDA conductance with voltage-dependent Mg²⁺ block.

    Implements the Jahr & Stevens (1990) Mg²⁺ block:
        B(V) = 1 / (1 + [Mg²⁺]_o/3.57 * exp(-0.062 * V))

    Uses dual-exponential kinetics: g_NMDA = g_slow - g_rise for realistic
    rise/decay dynamics. The Mg²⁺ block factor B(V) produces the characteristic
    voltage-dependent nonlinearity that gates Ca²⁺ influx and is critical
    for coincidence detection in STDP and associative learning.
    """
    # Dual-exponential decay
    g_nmda_new = g_nmda * decay_nmda
    g_nmda_rise_new = g_nmda_rise * decay_nmda_rise
    # Effective NMDA conductance (difference of exponentials)
    g_eff = g_nmda_new - g_nmda_rise_new
    g_eff = cp.maximum(g_eff, 0.0)
    # Voltage-dependent Mg²⁺ block (Jahr & Stevens 1990)
    mg_block = 1.0 / (1.0 + (mg_conc / 3.57) * cp.exp(-0.062 * v))
    # NMDA current with Mg²⁺ gating
    I_nmda = g_eff * mg_block * (E_nmda - v)
    return g_nmda_new, g_nmda_rise_new, I_nmda

@cp.fuse()
def fused_stp_decay_recovery(u, x, dt, tau_f, tau_d):
    """Fused kernel for STP u and x variable decay/recovery."""
    # Ensure tau_f and tau_d are not zero to prevent division by zero.
    tau_f_safe = cp.maximum(tau_f, 1e-9) # Use a small epsilon if tau_f is zero
    tau_d_safe = cp.maximum(tau_d, 1e-9) # Use a small epsilon if tau_d is zero

    # Decay of u (facilitation variable)
    u_decayed = u * cp.exp(-dt / tau_f_safe)
    # Recovery of x (depression variable)
    x_recovered_increment = (1.0 - x) * (dt / tau_d_safe) # dx/dt = (1-x)/tau_d
    x_recovered = x + x_recovered_increment
    x_clipped = cp.clip(x_recovered, 0.0, 1.0) # Ensure x stays within [0, 1]
    return u_decayed, x_clipped

@cp.fuse()
def fused_homeostasis_update(neuron_activity_ema_in, fired_this_step_float, target_rate, alpha_ema, adapt_rate,
                             neuron_firing_thresholds_in, thresh_min, thresh_max):
    """Fused kernel for homeostatic threshold adaptation."""
    # Update Exponential Moving Average (EMA) of neuron activity
    new_neuron_activity_ema = (1.0 - alpha_ema) * neuron_activity_ema_in + alpha_ema * fired_this_step_float
    # Calculate error from target firing rate
    error = new_neuron_activity_ema - target_rate
    # Calculate change in threshold based on error and adaptation rate
    threshold_delta = error * adapt_rate
    # Update firing thresholds
    new_neuron_firing_thresholds = neuron_firing_thresholds_in + threshold_delta
    # Clip thresholds to min/max bounds
    new_neuron_firing_thresholds_clipped = cp.clip(new_neuron_firing_thresholds, thresh_min, thresh_max)
    return new_neuron_activity_ema, new_neuron_firing_thresholds_clipped

# --- Phase C2: STDP Kernels (Bi & Poo 1998, Caporale & Dan 2008) ---
@cp.fuse()
def fused_stdp_weight_update(delta_t, w_current, A_plus, A_minus, tau_plus, tau_minus, w_min, w_max):
    """Fused kernel for STDP weight update based on spike timing difference.
    
    Implements classical asymmetric STDP window:
    - delta_t > 0 (post-before-pre): LTP (potentiation) 
    - delta_t < 0 (pre-before-post): LTD (depression)
    
    Args:
        delta_t: Spike timing difference (t_post - t_pre) in ms
        w_current: Current synaptic weight
        A_plus: LTP amplitude
        A_minus: LTD amplitude
        tau_plus: LTP time constant (ms)
        tau_minus: LTD time constant (ms)
        w_min: Minimum weight
        w_max: Maximum weight
    
    Returns:
        Updated synaptic weight
    """
    # LTP: delta_t > 0 means post fired after pre -> strengthen synapse
    # Use soft-bound: delta_w = A_plus * (w_max - w) * exp(-delta_t / tau_plus)
    ltp_update = cp.where(
        delta_t > 0.0,
        A_plus * (w_max - w_current) * cp.exp(-delta_t / tau_plus),
        0.0
    )
    
    # LTD: delta_t < 0 means pre fired after post -> weaken synapse
    # Use soft-bound: delta_w = -A_minus * (w - w_min) * exp(delta_t / tau_minus)
    ltd_update = cp.where(
        delta_t < 0.0,
        -A_minus * (w_current - w_min) * cp.exp(delta_t / tau_minus),
        0.0
    )
    
    # Apply update and clip to bounds
    w_new = w_current + ltp_update + ltd_update
    w_new_clipped = cp.clip(w_new, w_min, w_max)
    return w_new_clipped

@cp.fuse()
def fused_eligibility_trace_decay(trace, decay_factor):
    """Fused kernel for eligibility trace exponential decay.
    
    Args:
        trace: Current eligibility trace value
        decay_factor: exp(-dt / tau)
    
    Returns:
        Decayed trace value
    """
    return trace * decay_factor


# =============================================================================
# EXPERIMENT & STIMULUS SYSTEM
# =============================================================================
# Provides programmable stimulus injection, I/O neuron group management,
# training protocols, readout/analysis, and multi-phase experiment execution.
#
# Architecture:
#   ExperimentEngine (top-level orchestrator)
#   ├── StimulusManager (generates per-step current arrays)
#   │   └── StimulusChannel (pattern + target neurons + timing)
#   ├── NeuronGroupManager (input/output/hidden populations)
#   ├── ReadoutEngine (measures network responses)
#   ├── TrainingProtocol (learning protocol execution)
#   └── ExperimentLog (trial-level data logging)
#
# Scientific references:
#   - Current injection patterns: Destexhe & Bhatt 2015 (in vivo conductance injection)
#   - Poisson input: Shadlen & Newsome 1998 (neural variability)
#   - R-STDP training: Izhikevich 2007, Frémaux et al. 2013 (three-factor learning)
#   - Reservoir computing: Maass et al. 2002, Jaeger & Haas 2004
#   - Associative conditioning: Rescorla & Wagner 1972
# =============================================================================

# --- Stimulus Pattern Definitions ---

class StimulusPatternType(Enum):
    """Available stimulus waveform types."""
    CONSTANT = "CONSTANT"                   # DC current step
    PULSE_TRAIN = "PULSE_TRAIN"             # Repeated brief pulses
    SINUSOIDAL = "SINUSOIDAL"               # AC sinusoidal current
    RAMP = "RAMP"                           # Linearly increasing/decreasing
    POISSON_SPIKE_TRAIN = "POISSON_SPIKE_TRAIN"  # Poisson-distributed brief pulses
    GAUSSIAN_NOISE = "GAUSSIAN_NOISE"       # White noise injection
    CUSTOM_WAVEFORM = "CUSTOM_WAVEFORM"     # Arbitrary time series

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

    # Custom waveform (time_ms, amplitude_pA pairs — interpolated)
    custom_waveform_times_ms: List[float] = field(default_factory=list)
    custom_waveform_values_pA: List[float] = field(default_factory=list)

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
    - R-STDP: Izhikevich 2007 Ch.7, Frémaux et al. 2013
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
    success_threshold: float = 0.7         # Fraction of correct trials for convergence

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

    enabled: bool = False                   # Master enable for experiment system


# --- Stimulus Manager (generates per-step current arrays) ---

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
            mask = cp_module.zeros(self.n_neurons, dtype=cp_module.bool_)
            if len(indices) > 0:
                mask[cp_module.array(indices, dtype=cp_module.int32)] = True
            self._channel_target_masks[ch.name] = mask

            # Initialize Poisson state if needed
            if ch.pattern.pattern_type == StimulusPatternType.POISSON_SPIKE_TRAIN.name:
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
            import math
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


# --- Neuron Group Manager ---

class NeuronGroupManager:
    """Manages designated neuron populations (input/output/hidden).

    Provides methods to resolve group names to indices, populate groups
    from trait arrays, and track group statistics.
    """

    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.groups = {}  # name -> NeuronGroup

    def initialize(self, group_defs, cp_traits=None, cp_module=None):
        """Set up neuron groups from definitions.

        Args:
            group_defs: List[NeuronGroup] definitions
            cp_traits: GPU array of neuron trait indices (for trait-based population)
            cp_module: CuPy module reference
        """
        for gdef in group_defs:
            group = NeuronGroup(
                name=gdef.name,
                role=gdef.role,
                neuron_indices=list(gdef.neuron_indices),
                trait_index=gdef.trait_index,
                index_start=gdef.index_start,
                index_end=gdef.index_end,
                fraction_of_trait=gdef.fraction_of_trait,
                highlight_color=list(gdef.highlight_color),
            )

            # Auto-populate from trait if indices not specified
            if not group.neuron_indices:
                if group.trait_index >= 0 and cp_traits is not None:
                    trait_np = cp_traits.get() if hasattr(cp_traits, 'get') else cp_traits
                    all_trait_indices = [int(i) for i in range(len(trait_np)) if int(trait_np[i]) == group.trait_index]

                    if group.fraction_of_trait < 1.0 and len(all_trait_indices) > 0:
                        import random as py_random
                        n_select = max(1, int(len(all_trait_indices) * group.fraction_of_trait))
                        group.neuron_indices = sorted(py_random.sample(all_trait_indices, n_select))
                    else:
                        group.neuron_indices = all_trait_indices

                elif group.index_end > group.index_start:
                    group.neuron_indices = list(range(
                        max(0, group.index_start),
                        min(self.n_neurons, group.index_end)
                    ))

            self.groups[group.name] = group

    def get_group(self, name):
        """Get a neuron group by name."""
        return self.groups.get(name)

    def get_groups_by_role(self, role):
        """Get all groups with a specific role."""
        return [g for g in self.groups.values() if g.role == role]

    def get_group_mask(self, name, cp_module):
        """Get a boolean GPU mask for a neuron group."""
        group = self.groups.get(name)
        if group is None or not group.neuron_indices:
            return cp_module.zeros(self.n_neurons, dtype=cp_module.bool_)
        mask = cp_module.zeros(self.n_neurons, dtype=cp_module.bool_)
        mask[cp_module.array(group.neuron_indices, dtype=cp_module.int32)] = True
        return mask

    def get_summary(self):
        """Get a summary of all groups for logging."""
        summary = {}
        for name, group in self.groups.items():
            summary[name] = {
                "role": group.role,
                "n_neurons": len(group.neuron_indices),
                "trait_index": group.trait_index,
            }
        return summary


# --- Readout Engine ---

class ReadoutEngine:
    """Measures and logs network responses per neuron group.

    Provides real-time population firing rate, spike counts,
    and optional spectral analysis. All computations stay on GPU
    where possible to minimize transfer overhead.
    """

    def __init__(self, n_neurons, dt_ms):
        self.n_neurons = n_neurons
        self.dt_ms = dt_ms
        self.config = ReadoutConfig()
        self.group_manager = None

        # Rate estimation buffers (circular buffers on GPU)
        self._rate_buffers = {}          # group_name -> circular buffer of spike counts
        self._rate_buffer_idx = 0
        self._rate_buffer_size = 0

        # Spike count accumulators
        self._spike_counts = {}          # group_name -> int accumulator
        self._spike_count_window_steps = 0
        self._spike_count_step = 0

        # PSD buffers
        self._psd_buffers = {}           # group_name -> voltage history buffer
        self._psd_buffer_idx = 0

        # Current readout values (CPU, for UI display and logging)
        self.current_rates = {}          # group_name -> float (Hz)
        self.current_spike_counts = {}   # group_name -> int
        self.current_psd = {}            # group_name -> dict with freqs, power

        # Trial-level metrics
        self.trial_metrics = []          # List of per-trial measurement dicts

    def initialize(self, config, group_manager, cp_module):
        """Set up readout buffers.

        Args:
            config: ReadoutConfig
            group_manager: NeuronGroupManager
            cp_module: CuPy module reference
        """
        self.config = config
        self.group_manager = group_manager

        # Rate buffer: store spike counts per step for sliding window
        self._rate_buffer_size = max(1, int(config.rate_window_ms / self.dt_ms))
        self._rate_buffer_idx = 0

        groups_to_track = config.rate_group_names
        if not groups_to_track:
            # Default: track all output groups
            groups_to_track = [g.name for g in group_manager.get_groups_by_role(NeuronGroupRole.OUTPUT.name)]
            # Also track input groups for comparison
            groups_to_track += [g.name for g in group_manager.get_groups_by_role(NeuronGroupRole.INPUT.name)]

        for gname in groups_to_track:
            group = group_manager.get_group(gname)
            if group and group.neuron_indices:
                self._rate_buffers[gname] = cp_module.zeros(self._rate_buffer_size, dtype=cp_module.float32)
                self._spike_counts[gname] = 0
                self.current_rates[gname] = 0.0
                self.current_spike_counts[gname] = 0

        # Spike count window
        self._spike_count_window_steps = max(1, int(config.spike_count_window_ms / self.dt_ms))
        self._spike_count_step = 0

        # PSD buffer
        if config.enable_psd:
            psd_steps = max(1, int(config.psd_window_ms / self.dt_ms))
            for gname in groups_to_track:
                self._psd_buffers[gname] = cp_module.zeros(psd_steps, dtype=cp_module.float32)
            self._psd_buffer_idx = 0

    def update(self, cp_firing_states, cp_membrane_potential_v, cp_module):
        """Update readout measurements for the current timestep.

        Args:
            cp_firing_states: GPU bool array [n_neurons] of current spikes
            cp_membrane_potential_v: GPU float32 array [n_neurons] of membrane voltages
            cp_module: CuPy module reference
        """
        for gname, buffer in self._rate_buffers.items():
            group = self.group_manager.get_group(gname)
            if group is None or not group.neuron_indices:
                continue

            # Count spikes in this group this step
            group_indices = cp_module.array(group.neuron_indices, dtype=cp_module.int32)
            group_spikes = cp_firing_states[group_indices]
            n_spikes = float(cp_module.sum(group_spikes).get())
            n_neurons_in_group = len(group.neuron_indices)

            # Update circular rate buffer
            buffer[self._rate_buffer_idx % self._rate_buffer_size] = n_spikes

            # Compute instantaneous population rate (Hz)
            total_spikes_in_window = float(cp_module.sum(buffer).get())
            window_duration_s = self._rate_buffer_size * self.dt_ms / 1000.0
            if n_neurons_in_group > 0 and window_duration_s > 0:
                self.current_rates[gname] = total_spikes_in_window / (n_neurons_in_group * window_duration_s)

            # Update spike count accumulator
            self._spike_counts[gname] = self._spike_counts.get(gname, 0) + int(n_spikes)

        # Advance circular buffer index
        self._rate_buffer_idx += 1

        # Spike count window reset
        self._spike_count_step += 1
        if self._spike_count_step >= self._spike_count_window_steps:
            for gname in self._spike_counts:
                self.current_spike_counts[gname] = self._spike_counts[gname]
                self._spike_counts[gname] = 0
            self._spike_count_step = 0

        # PSD buffer update
        if self.config.enable_psd:
            psd_size = len(next(iter(self._psd_buffers.values()))) if self._psd_buffers else 0
            for gname, psd_buf in self._psd_buffers.items():
                group = self.group_manager.get_group(gname)
                if group and group.neuron_indices:
                    group_indices = cp_module.array(group.neuron_indices, dtype=cp_module.int32)
                    mean_v = float(cp_module.mean(cp_membrane_potential_v[group_indices]).get())
                    psd_buf[self._psd_buffer_idx % psd_size] = mean_v
            self._psd_buffer_idx += 1

    def compute_psd(self, group_name, cp_module):
        """Compute power spectral density for a group.

        Returns dict with 'frequencies_hz' and 'power' arrays (numpy).
        """
        import numpy as np

        psd_buf = self._psd_buffers.get(group_name)
        if psd_buf is None:
            return None

        signal = psd_buf.get()  # Transfer to CPU

        # FFT
        n = len(signal)
        if n < 2:
            return None

        fft_vals = np.fft.rfft(signal - np.mean(signal))
        power = np.abs(fft_vals) ** 2 / n
        freqs = np.fft.rfftfreq(n, d=self.dt_ms / 1000.0)

        # Filter to requested range
        f_min, f_max = self.config.psd_freq_range_hz
        mask = (freqs >= f_min) & (freqs <= f_max)

        return {
            'frequencies_hz': freqs[mask],
            'power': power[mask],
        }

    def get_trial_snapshot(self):
        """Get current readout state for trial logging."""
        return {
            'rates': dict(self.current_rates),
            'spike_counts': dict(self.current_spike_counts),
        }

    def cleanup(self):
        """Release GPU memory."""
        self._rate_buffers.clear()
        self._psd_buffers.clear()
        self._spike_counts.clear()


# --- Training Protocol Engine ---

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
                import numpy as np
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
            if hasattr(sim_bridge_ref, 'cfg') and sim_bridge_ref.cfg is not None:
                sim_bridge_ref.cfg.current_reward_signal = 0.0

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

        # Evaluate success for RL
        if self.config.mode == TrainingMode.REINFORCEMENT_LEARNING.name:
            target_group = self.config.target_output_group
            rate = snapshot.get("rates", {}).get(target_group, 0.0)
            success = self.config.target_min_rate_hz <= rate <= self.config.target_max_rate_hz
            trial_data["success"] = success
            trial_data["output_rate"] = rate

        self.trials_data.append(trial_data)

    def _deliver_reward(self, sim_bridge_ref):
        """Deliver reward or punishment based on output group activity."""
        if not hasattr(sim_bridge_ref, 'cfg') or sim_bridge_ref.cfg is None:
            return

        target_group = self.config.target_output_group
        rate = self.readout.current_rates.get(target_group, 0.0) if self.readout else 0.0

        if self.config.target_min_rate_hz <= rate <= self.config.target_max_rate_hz:
            sim_bridge_ref.cfg.current_reward_signal = self.config.reward_magnitude
        else:
            sim_bridge_ref.cfg.current_reward_signal = self.config.punishment_magnitude

    def _apply_supervised_error(self, sim_bridge_ref):
        """Apply supervised error signal as reward modulation.

        Uses the existing reward signal mechanism as an error channel.
        Error = (target_rate - actual_rate) * gain
        """
        if not hasattr(sim_bridge_ref, 'cfg') or sim_bridge_ref.cfg is None:
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
            sim_bridge_ref.cfg.current_reward_signal = mean_error * self.config.supervised_error_gain

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


# --- Experiment Engine (Top-Level Orchestrator) ---

class ExperimentEngine:
    """Orchestrates multi-phase experiments with stimulus, training, and readout.

    The engine is called once per simulation step by SimulationBridge.
    It manages:
    1. Phase transitions (baseline → stimulus → training → testing → rest)
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

    def start(self, current_time_ms):
        """Begin experiment execution."""
        self.is_experiment_running = True
        self.is_experiment_complete = False
        self.current_phase_idx = 0
        self.phase_repetition = 0
        self.phase_start_ms = current_time_ms
        self._step_counter = 0

        if self.phases:
            self._enter_phase(self.phases[0], current_time_ms)

        self.log.append({"event": "experiment_started", "time_ms": current_time_ms})

    def stop(self):
        """Stop experiment execution."""
        self.is_experiment_running = False
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

    def _enter_phase(self, phase, current_time_ms):
        """Set up a new experiment phase."""
        self.log.append({
            "event": "phase_entered",
            "phase": phase.name,
            "type": phase.phase_type,
            "time_ms": current_time_ms,
        })

        # Configure active stimulus channels
        for ch in self.stimulus_manager.channels:
            ch.enabled = (ch.name in phase.active_channels) if phase.active_channels else True

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
        import json
        with open(filepath, 'w') as f:
            json.dump({
                "experiment_name": self.config.name if self.config else "Unknown",
                "description": self.config.description if self.config else "",
                "groups": self.group_manager.get_summary() if self.group_manager else {},
                "training_summary": self.training.get_training_summary(),
                "log_entries": self.log,
                "trial_data": self.training.trials_data,
            }, f, indent=2, default=str)

    def cleanup(self):
        """Release all GPU resources."""
        self.stimulus_manager.cleanup()
        self.readout.cleanup()
        self.is_experiment_running = False


# --- Experiment Preset Definitions ---
# Pre-built experiment configurations for common neuroscience paradigms.

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
    def associative_conditioning(cs_amplitude_pA=100.0, us_amplitude_pA=200.0,
                                  cs_us_delay_ms=100.0, num_trials=100,
                                  input_group_size=100, output_group_size=100):
        """Classical conditioning: pair CS (input) with US (output), test if CS alone evokes response.

        Based on Pavlovian conditioning with STDP as the learning mechanism.
        The CS-US delay determines the temporal window for STDP potentiation.
        """
        return ExperimentConfig(
            name="Associative Conditioning (CS-US Pairing)",
            description="Pavlovian conditioning: repeated CS-US pairing followed by CS-alone testing.",
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
    def reinforcement_learning(stimulus_amplitude_pA=120.0, num_trials=200,
                                input_group_size=100, output_group_size=50):
        """Reward-modulated STDP training: stimulus → response → reward/punishment.

        Based on three-factor learning rule (Izhikevich 2007, Frémaux et al. 2013).
        Uses the existing eligibility trace and reward modulation infrastructure.
        """
        return ExperimentConfig(
            name="Reinforcement Learning (R-STDP)",
            description="Three-factor learning: stimulus evokes response, reward/punishment shapes connections.",
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
                                   punishment_magnitude=-0.5,
                                   target_output_group="response",
                                   target_min_rate_hz=15.0,
                                   target_max_rate_hz=40.0,
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
        import math

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


# --- JSON Serialization for Experiment Configs ---

def experiment_config_to_dict(config):
    """Serialize an ExperimentConfig to a JSON-safe dictionary."""
    import dataclasses

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




# --- Simulation Bridge (Core Logic) ---
class SimulationBridge:
    def __init__(self, sim_core_ref=None, core_config=None, viz_config=None, runtime_state=None, gpu_config=None):
        """Initialize SimulationBridge with optional config objects.
        
        Args:
            sim_core_ref: Legacy parameter, not used with threading
            core_config: CoreSimConfig instance (creates default if None)
            viz_config: VisualizationConfig instance (creates default if None)
            runtime_state: RuntimeState instance (creates default if None)
            gpu_config: GPUConfig instance (creates default if None)
        """
        self.core_config = core_config if core_config is not None else CoreSimConfig()
        self.viz_config = viz_config if viz_config is not None else VisualizationConfig()
        self.runtime_state = runtime_state if runtime_state is not None else RuntimeState()
        self.gpu_config = gpu_config if gpu_config is not None else GPUConfig()
        self.ui_queue = sim_to_ui_queue # Reference to the queue for sending data/status to UI

        # --- CuPy Arrays for Simulation State ---
        self.cp_membrane_potential_v = None 
        self.cp_recovery_variable_u = None  
        self.cp_conductance_g_e = None
        self.cp_conductance_g_i = None
        self.cp_conductance_g_nmda = None
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

        self.cp_connections = None 

        self.cp_stp_u = None 
        self.cp_stp_x = None 

        self.cp_synapse_pulse_timers = None   
        self.cp_synapse_pulse_progress = None 

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

        # Structural plasticity optimization
        self._compaction_counter = 0  # Counter for deferred CSR compaction
        self._pending_eliminations = False  # Flag for pending zero-weight synapses
        self._synapse_capacity = 0  # Pre-allocated capacity for synapse arrays

        # Eligibility trace for STDP/reward
        self.cp_eligibility_trace = None

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

        for dir_path in [self.PROFILE_DIR, self.CHECKPOINT_DIR, self.RECORDING_DIR]:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)
                    self._log_console(f"Created directory: {dir_path}", "info")
                except OSError as e:
                    self._log_console(f"Error creating directory {dir_path}: {e}", "error")
        try:
             cp.cuda.Device(0).use()
             
             # Configure memory pool for better performance (controlled by gpu_config)
             mempool = cp.get_default_memory_pool()
             pinned_mempool = cp.get_default_pinned_memory_pool()
             
             # Set memory pool limit based on gpu_config
             dev_props = cp.cuda.runtime.getDeviceProperties(0)
             total_mem = dev_props['totalGlobalMem']
             mempool.set_limit(size=int(total_mem * self.gpu_config.memory_pool_limit_fraction))
             
             gpu_name = dev_props.get('name',b'Unknown').decode()
             self._log_console(
                 f"CuPy using GPU: {gpu_name} ({total_mem / 1024**3:.1f} GB), "
                 f"mempool limit: {self.gpu_config.memory_pool_limit_fraction*100:.0f}%",
                 "info"
             )
        except Exception as e:
             self._log_console(f"Error setting CuPy device: {e}", "critical")

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
        mem_info = cp.cuda.Device().mem_info
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
            # Trigger garbage collection
            cp.get_default_memory_pool().free_all_blocks()
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
        """Invalidates COO cache when connectivity changes."""
        self._coo_cache_valid = False
        self._cached_coo_matrix = None

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

        # Eligibility traces for reward modulation
        if cfg.enable_reward_modulation and num_synapses > 0:
            self._log_console(f"Initializing eligibility traces for {num_synapses} synapses (capacity: {capacity})...")
            self.cp_eligibility_trace = cp.zeros(capacity, dtype=cp.float32)
        else:
            self.cp_eligibility_trace = None

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

        self._synapse_count = int(cp.sum(keep_mask).get())

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
            n = self.core_config.num_neurons
            cfg = self.core_config
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
            # NMDA conductance (dual-exponential: g_nmda_slow - g_nmda_rise)
            self.cp_conductance_g_nmda = cp.zeros(n, dtype=cp.float32)
            self.cp_conductance_g_nmda_rise = cp.zeros(n, dtype=cp.float32)
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

                np_traits_host = cp.asnumpy(self.cp_traits)
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

                if num_defined_izh_variants > 0:
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
                # Single-parameter set broadcast to all neurons; traits currently only affect visualization and E/I status
                self.cp_membrane_potential_v = cp.full(n, cfg.adex_E_L, dtype=cp.float32)
                self.cp_adex_w = cp.zeros(n, dtype=cp.float32)
                self.cp_neuron_firing_thresholds = None  # AdEx uses adex_V_peak from config
                # Vectorized viz label assignment
                self.runtime_state.neuron_types_list_for_viz = ["AdEx_RS"] * n
            
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
                self._log_console("Generating connections (3D)...")
                profile_name_for_conn = getattr(cfg, "neural_profile_name", "GENERIC_UNSTRUCTURED")
                profile_def_for_conn = NEURAL_STRUCTURE_PROFILES.get(profile_name_for_conn)
                motif_name = profile_def_for_conn.get("connectivity_motif") if profile_def_for_conn else None

                if motif_name:
                    self.cp_connections = self._generate_motif_connections_3d(n, self.cp_neuron_positions_3d, self.cp_traits, cfg, motif_name)
                elif cfg.enable_watts_strogatz:
                    self.cp_connections = self._generate_watts_strogatz_connections_3d(n, cfg.connectivity_k, cfg.connectivity_p_rewire, cfg)
                else: 
                    self.cp_connections = self._generate_spatial_connections_3d(n, cfg.connections_per_neuron, self.cp_neuron_positions_3d, self.cp_traits, cfg)
                
                # Defensive fallback: if no synapses were generated, fall back to spatial generator
                if self.cp_connections is None or (hasattr(self.cp_connections, 'nnz') and self.cp_connections.nnz == 0 and n > 1):
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

            self.is_initialized = True
            conn_count = self.cp_connections.nnz if self.cp_connections is not None else 0

            # Log GPU memory usage after initialization
            mem_stats = self._get_gpu_memory_info()
            self._log_console(
                f"Simulation data initialized for {n} neurons (3D). Connections: {conn_count}. "
                f"GPU memory: {mem_stats['used_gb']:.1f}GB/{mem_stats['total_gb']:.1f}GB ({mem_stats['usage_percent']:.1f}%)"
            )
            self._check_gpu_memory_pressure()
        except Exception as e:
            self._log_console(f"Error during simulation data initialization (3D): {e}","critical")
            import traceback; traceback.print_exc()
            self.is_initialized = False
            if 'cupy' in sys.modules and cp.is_available():
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
    
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
            rng_state = cp.random.get_random_state()
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
            cp.random.set_random_state(rng_state)
        
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
        if pos_neighbors_cp.size == 0: return cp.array([], dtype=cp.float32)
        diff_3d = pos_neighbors_cp - pos_i_cp.reshape(1, 3) 
        return cp.sqrt(cp.sum(diff_3d**2, axis=1))

    def _generate_spatial_connections_3d_vectorized(self, n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config):
        """Generates connections using fully vectorized GPU operations (fast, scalable to 100K+ neurons).
        Uses chunked processing to avoid OOM errors on large networks.
        """
        self._log_console("Generating connections (3D spatial, GPU-vectorized)...")
        start_t = time.time()
        
        if n == 0:
            return csp.csr_matrix((0, 0), dtype=cp.float32)
        
        dist_decay = getattr(config, 'connection_distance_decay_factor', 0.01)
        trait_bias = getattr(config, 'trait_connection_bias', 0.5)
        min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight
        k = min(max_connections_per_neuron, n - 1)
        
        # For very large networks, use chunked processing to avoid memory issues
        # Memory for n×n float32 matrix: n^2 * 4 bytes
        # 20GB limit: sqrt(20e9 / 4) ≈ 70k neurons can fit in full matrix
        # Use chunking for n > 15000 to be safe (allows 4GB per chunk with overhead)
        if n > 15000:
            return self._generate_spatial_connections_3d_chunked(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config)
        
        # Small enough network - use full vectorization
        # Compute all pairwise distances on GPU (n x n matrix)
        pos = neuron_positions_3d_cp  # Shape: (n, 3)
        pos_i = pos[:, None, :]  # Shape: (n, 1, 3)
        pos_j = pos[None, :, :]  # Shape: (1, n, 3)
        diff = pos_i - pos_j  # Shape: (n, n, 3)
        distances = cp.sqrt(cp.sum(diff**2, axis=2))  # Shape: (n, n)
        
        # Set self-distances to infinity to exclude self-connections
        cp.fill_diagonal(distances, cp.inf)
        
        # Compute connection probabilities
        prob_dist = cp.exp(-dist_decay * distances)
        
        # Trait similarity component
        traits_i = traits_cp[:, None]  # Shape: (n, 1)
        traits_j = traits_cp[None, :]  # Shape: (1, n)
        prob_trait = (traits_i == traits_j).astype(cp.float32) * trait_bias
        
        # Combined probability
        conn_prob = prob_dist + prob_trait  # Shape: (n, n)
        
        # For each neuron, select top-k connections based on probability
        top_k_indices = cp.argsort(conn_prob, axis=1)[:, -k:]  # Shape: (n, k)
        
        # Generate weights for connections
        weights = cp.random.uniform(min_w, max_w, (n, k)).astype(cp.float32)
        
        # Convert to COO format
        row_indices = cp.repeat(cp.arange(n), k)  # Shape: (n*k,)
        col_indices = top_k_indices.ravel()  # Shape: (n*k,)
        weights_flat = weights.ravel()  # Shape: (n*k,)
        
        # Create CSR matrix
        conn_matrix = csp.coo_matrix(
            (weights_flat, (row_indices, col_indices)),
            shape=(n, n),
            dtype=cp.float32
        ).tocsr()
        
        conn_matrix.sort_indices()
        elapsed = time.time() - start_t
        self._log_console(f"Connections (3D Spatial GPU): {conn_matrix.nnz}. Time: {elapsed:.2f}s")
        return conn_matrix
    
    def _generate_random_connections_large(self, n, k, traits_np, trait_bias, min_w, max_w):
        """Generate random connections for very large networks when spatial constraints don't apply.

        Used when connection_radius exceeds spatial extent, meaning all neurons are
        effectively within connection range of each other. Uses chunked processing
        to avoid memory issues.
        """
        start_t = time.time()
        self._log_console(f"Generating random connections for {n} neurons (k={k})...")

        all_rows = []
        all_cols = []
        all_weights = []

        # Process in chunks
        chunk_size = max(1000, n // 100)
        num_chunks = (n + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n)
            chunk_n = end_idx - start_idx

            # For each neuron in chunk, randomly select k targets
            for i in range(chunk_n):
                neuron_i = start_idx + i
                trait_i = traits_np[neuron_i]

                # Generate candidate pool (exclude self)
                candidates = np.concatenate([np.arange(0, neuron_i), np.arange(neuron_i + 1, n)])

                # Weight by trait similarity
                candidate_traits = traits_np[candidates]
                weights = np.ones(len(candidates), dtype=np.float32)
                weights[candidate_traits == trait_i] += trait_bias

                # Normalize to probabilities
                probs = weights / weights.sum()

                # Sample k targets
                actual_k = min(k, len(candidates))
                targets = np.random.choice(candidates, size=actual_k, replace=False, p=probs)

                # Generate connection weights
                conn_weights = np.random.uniform(min_w, max_w, actual_k).astype(np.float32)

                all_rows.extend([neuron_i] * actual_k)
                all_cols.extend(targets.tolist())
                all_weights.extend(conn_weights.tolist())

            if num_chunks > 1 and (chunk_idx + 1) % max(1, num_chunks // 10) == 0:
                progress = ((chunk_idx + 1) / num_chunks) * 100
                self._log_console(f"Random connection progress: {progress:.1f}%")

        # Create sparse matrix
        row_indices_cp = cp.asarray(np.array(all_rows, dtype=np.int32))
        col_indices_cp = cp.asarray(np.array(all_cols, dtype=np.int32))
        weights_cp = cp.asarray(np.array(all_weights, dtype=np.float32))

        conn_matrix = csp.coo_matrix(
            (weights_cp, (row_indices_cp, col_indices_cp)),
            shape=(n, n),
            dtype=cp.float32
        ).tocsr()

        conn_matrix.sort_indices()
        elapsed = time.time() - start_t
        self._log_console(f"Connections (Random Large): {conn_matrix.nnz}. Time: {elapsed:.2f}s")
        return conn_matrix

    def _generate_spatial_connections_3d_binned(self, n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config):
        """Spatial binning approach for very large networks (>50k neurons).

        Instead of computing distances to all N neurons, we divide the space into bins
        and only compute distances to neurons in nearby bins. This reduces memory from
        O(N) to O(N/num_bins * neighborhood_size), making 100K+ networks feasible.
        """
        self._log_console("Generating connections (3D spatial, GPU-binned)...")
        start_t = time.time()

        dist_decay = getattr(config, 'connection_distance_decay_factor', 0.01)
        trait_bias = getattr(config, 'trait_connection_bias', 0.5)
        min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight
        k = min(max_connections_per_neuron, n - 1)

        # Transfer positions to CPU for binning (more efficient for indexing)
        positions_np = cp.asnumpy(neuron_positions_3d_cp)
        traits_np = cp.asnumpy(traits_cp)

        # Get spatial bounds
        pos_min = positions_np.min(axis=0)
        pos_max = positions_np.max(axis=0)
        spatial_extent = pos_max - pos_min + 1e-6  # Avoid zero extent
        max_extent = spatial_extent.max()

        # connection_radius = distance at which probability drops to ~1%
        connection_radius = 4.6 / max(dist_decay, 0.001)

        # If connection_radius exceeds spatial extent, all neurons can connect to all others
        # In this case, use random sampling instead of spatial binning
        if connection_radius >= max_extent:
            self._log_console(f"Connection radius ({connection_radius:.1f}) >= spatial extent ({max_extent:.1f}). Using random sampling.")
            return self._generate_random_connections_large(n, k, traits_np, trait_bias, min_w, max_w)

        # Determine bin size based on network size (aim for manageable bins)
        # For 100K neurons, target ~500-1000 neurons per bin = ~100-200 bins total
        target_neurons_per_bin = max(500, n // 200)
        num_bins_total = max(27, n // target_neurons_per_bin)
        num_bins_per_dim = max(3, int(np.cbrt(num_bins_total)))

        bin_size = spatial_extent / num_bins_per_dim

        # Recompute actual num_bins based on bin_size
        num_bins_xyz = np.ceil(spatial_extent / bin_size).astype(int)
        num_bins_xyz = np.maximum(num_bins_xyz, 1)

        # How many bins does connection_radius span?
        avg_bin_size = bin_size.mean()
        neighbor_range = max(1, int(np.ceil(connection_radius / avg_bin_size)))

        # Cap neighbor_range to avoid searching more than half the bins
        max_neighbor_range = num_bins_per_dim // 2
        if neighbor_range > max_neighbor_range:
            self._log_console(f"Neighbor range {neighbor_range} too large. Using random sampling.")
            return self._generate_random_connections_large(n, k, traits_np, trait_bias, min_w, max_w)

        self._log_console(f"Spatial binning: {num_bins_xyz} bins, bin_size={avg_bin_size:.2f}, neighbor_range={neighbor_range}")

        # Assign each neuron to a bin
        bin_indices = np.floor((positions_np - pos_min) / bin_size).astype(int)
        bin_indices = np.clip(bin_indices, 0, num_bins_xyz - 1)  # Clamp to valid range

        # Convert 3D bin index to linear index
        bin_linear = (bin_indices[:, 0] * num_bins_xyz[1] * num_bins_xyz[2] +
                      bin_indices[:, 1] * num_bins_xyz[2] +
                      bin_indices[:, 2])

        # Build bin-to-neuron lookup (dict: bin_id -> list of neuron indices)
        from collections import defaultdict
        bin_to_neurons = defaultdict(list)
        for neuron_idx, bin_id in enumerate(bin_linear):
            bin_to_neurons[bin_id].append(neuron_idx)

        # Pre-compute neighbor offsets based on neighbor_range
        # If neighbor_range=1, we search 3x3x3=27 bins
        # If neighbor_range=2, we search 5x5x5=125 bins, etc.
        neighbor_offsets = []
        for dx in range(-neighbor_range, neighbor_range + 1):
            for dy in range(-neighbor_range, neighbor_range + 1):
                for dz in range(-neighbor_range, neighbor_range + 1):
                    neighbor_offsets.append((dx, dy, dz))

        # Process neurons and generate connections - bin-by-bin for vectorization
        all_rows = []
        all_cols = []
        all_weights = []

        # Process bin-by-bin (all neurons in a bin share the same neighbor bins)
        total_bins = len(bin_to_neurons)
        processed_bins = 0

        for bin_id, source_neurons in bin_to_neurons.items():
            if len(source_neurons) == 0:
                continue

            # Get 3D bin coordinates from linear index
            bx = bin_id // (num_bins_xyz[1] * num_bins_xyz[2])
            remainder = bin_id % (num_bins_xyz[1] * num_bins_xyz[2])
            by = remainder // num_bins_xyz[2]
            bz = remainder % num_bins_xyz[2]

            # Gather ALL candidate neurons from neighboring bins (same for all source neurons in this bin)
            candidates = []
            for dx, dy, dz in neighbor_offsets:
                nx, ny, nz = bx + dx, by + dy, bz + dz
                if (0 <= nx < num_bins_xyz[0] and
                    0 <= ny < num_bins_xyz[1] and
                    0 <= nz < num_bins_xyz[2]):
                    neighbor_linear = nx * num_bins_xyz[1] * num_bins_xyz[2] + ny * num_bins_xyz[2] + nz
                    candidates.extend(bin_to_neurons[neighbor_linear])

            if len(candidates) == 0:
                continue

            # Convert to arrays for vectorized operations
            source_arr = np.array(source_neurons, dtype=np.int32)
            candidate_arr = np.array(candidates, dtype=np.int32)

            # Get positions and traits for sources and candidates
            source_pos = positions_np[source_arr]  # (num_sources, 3)
            candidate_pos = positions_np[candidate_arr]  # (num_candidates, 3)
            source_traits = traits_np[source_arr]  # (num_sources,)
            candidate_traits = traits_np[candidate_arr]  # (num_candidates,)

            # Compute all pairwise distances: (num_sources, num_candidates)
            # Using broadcasting: diff = source_pos[:, None, :] - candidate_pos[None, :, :]
            diff = source_pos[:, None, :] - candidate_pos[None, :, :]  # (S, C, 3)
            distances = np.sqrt(np.sum(diff**2, axis=2))  # (S, C)

            # Set self-distances to infinity
            # Create mask where source[i] == candidate[j]
            source_expanded = source_arr[:, None]  # (S, 1)
            candidate_expanded = candidate_arr[None, :]  # (1, C)
            self_mask = (source_expanded == candidate_expanded)  # (S, C)
            distances[self_mask] = np.inf

            # Compute connection probabilities
            prob_dist = np.exp(-dist_decay * distances)  # (S, C)

            # Trait similarity: (S, 1) == (1, C) -> (S, C)
            trait_match = (source_traits[:, None] == candidate_traits[None, :])
            prob_trait = trait_match.astype(np.float32) * trait_bias  # (S, C)

            conn_prob = prob_dist + prob_trait  # (S, C)

            # For each source neuron, select top-k candidates
            num_candidates = len(candidate_arr)
            actual_k = min(k, num_candidates - 1)  # -1 to account for self-exclusion

            if actual_k <= 0:
                continue

            # Use argpartition for each row to get top-k indices
            if actual_k < num_candidates:
                # Partition to get top-k indices (unsorted)
                partition_idx = np.argpartition(conn_prob, -actual_k, axis=1)[:, -actual_k:]  # (S, k)
            else:
                partition_idx = np.tile(np.arange(num_candidates), (len(source_arr), 1))

            # Generate connections
            num_sources = len(source_arr)
            for i in range(num_sources):
                source_neuron = source_arr[i]
                # Filter out any infinite distances (self-connections that might slip through)
                valid_mask = conn_prob[i, partition_idx[i]] > 0
                valid_targets = partition_idx[i][valid_mask]

                if len(valid_targets) == 0:
                    continue

                target_neurons = candidate_arr[valid_targets]
                num_connections = len(target_neurons)

                weights = np.random.uniform(min_w, max_w, num_connections).astype(np.float32)

                all_rows.extend([source_neuron] * num_connections)
                all_cols.extend(target_neurons.tolist())
                all_weights.extend(weights.tolist())

            processed_bins += 1
            if total_bins > 10 and processed_bins % max(1, total_bins // 10) == 0:
                progress = (processed_bins / total_bins) * 100
                self._log_console(f"Binned connection progress: {progress:.1f}%")

        # Convert to arrays and create sparse matrix on GPU
        if len(all_rows) == 0:
            self._log_console("Warning: No connections generated!", "warning")
            return csp.csr_matrix((n, n), dtype=cp.float32)

        row_indices_cp = cp.asarray(np.array(all_rows, dtype=np.int32))
        col_indices_cp = cp.asarray(np.array(all_cols, dtype=np.int32))
        weights_cp = cp.asarray(np.array(all_weights, dtype=np.float32))

        conn_matrix = csp.coo_matrix(
            (weights_cp, (row_indices_cp, col_indices_cp)),
            shape=(n, n),
            dtype=cp.float32
        ).tocsr()

        conn_matrix.sort_indices()
        elapsed = time.time() - start_t
        self._log_console(f"Connections (3D Spatial GPU-Binned): {conn_matrix.nnz}. Time: {elapsed:.2f}s")
        return conn_matrix

    def _generate_spatial_connections_3d_chunked(self, n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config):
        """Chunked version of vectorized connection generation for large networks.
        Processes neurons in GPU-accelerated batches.  Memory-adaptive chunk sizing
        keeps peak VRAM within safe limits for networks up to ~500K neurons on 24GB cards.

        Falls back to the CPU-based binned generator only when a single chunk row
        would exceed available VRAM (extremely large N with high connectivity).
        """
        # Estimate per-chunk-row VRAM: N * 60 bytes (distance matrix + probs + argpartition)
        # Fall back to CPU-binned only if a SINGLE row would exceed 25% of free VRAM
        # (meaning even chunk_size=1 would OOM)
        mem_info = cp.cuda.Device().mem_info
        free_mem = mem_info[0]
        bytes_per_row = n * 60
        if bytes_per_row > free_mem * 0.25:
            self._log_console(f"Single chunk row ({bytes_per_row/1e9:.1f}GB) exceeds 25% of free VRAM ({free_mem/1e9:.1f}GB). Falling back to CPU-binned generator.")
            return self._generate_spatial_connections_3d_binned(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config)

        self._log_console("Generating connections (3D spatial, GPU-vectorized-chunked)...")
        start_t = time.time()

        dist_decay = getattr(config, 'connection_distance_decay_factor', 0.01)
        trait_bias = getattr(config, 'trait_connection_bias', 0.5)
        min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight
        k = min(max_connections_per_neuron, n - 1)

        # Determine chunk size based on available memory
        # Peak memory per chunk row (all arrays that coexist during argpartition):
        #   diff:           n * 3 * 4 = 12n bytes  (chunk_n, n, 3) float32
        #   distances:      n * 4     =  4n bytes  (chunk_n, n)    float32
        #   prob_dist:      n * 4     =  4n bytes  (chunk_n, n)    float32
        #   prob_trait:     n * 4     =  4n bytes  (chunk_n, n)    float32
        #   conn_prob:      n * 4     =  4n bytes  (chunk_n, n)    float32
        #   argpartition internals (thrust sort): ~3x (chunk_n, n) int32+float32
        #                   n * 24    = 24n bytes  (hidden CuPy/Thrust temporaries)
        # Total peak: ~52n bytes per chunk row.  Use 60n for safety margin.
        mem_info = cp.cuda.Device().mem_info
        free_mem = mem_info[0]  # Free VRAM in bytes

        # Use only 35% of free memory — argpartition's Thrust backend allocates
        # large hidden temporaries that are not visible to CuPy's pool accounting
        target_mem_bytes = free_mem * 0.35

        bytes_per_chunk_row = n * 60  # Conservative: accounts for Thrust sort internals
        chunk_size = max(64, int(target_mem_bytes / bytes_per_chunk_row))
        chunk_size = min(chunk_size, n)  # Don't exceed total neurons

        free_mem_gb = free_mem / 1e9
        target_mem_gb = target_mem_bytes / 1e9
        self._log_console(f"VRAM: {free_mem_gb:.2f}GB free, using {target_mem_gb:.2f}GB ({target_mem_gb/free_mem_gb*100:.0f}%) for chunking")

        self._log_console(f"Using chunked processing: {n} neurons, chunk_size={chunk_size}")
        
        # Lists to accumulate connection data
        all_rows = []
        all_cols = []
        all_weights = []
        
        pos = neuron_positions_3d_cp  # Shape: (n, 3)
        
        # Process neurons in chunks
        num_chunks = (n + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n)
            chunk_n = end_idx - start_idx
            
            # Get positions and traits for this chunk
            chunk_pos = pos[start_idx:end_idx]  # Shape: (chunk_n, 3)
            chunk_traits = traits_cp[start_idx:end_idx]  # Shape: (chunk_n,)
            
            # Compute distances from chunk neurons to ALL neurons
            # chunk_pos: (chunk_n, 3) -> (chunk_n, 1, 3)
            # pos: (n, 3) -> (1, n, 3)
            chunk_pos_i = chunk_pos[:, None, :]  # (chunk_n, 1, 3)
            pos_j = pos[None, :, :]  # (1, n, 3)
            diff = chunk_pos_i - pos_j  # (chunk_n, n, 3)
            distances = cp.sqrt(cp.sum(diff**2, axis=2))  # (chunk_n, n)
            
            # Set self-distances to infinity (for neurons in this chunk)
            for i in range(chunk_n):
                global_idx = start_idx + i
                distances[i, global_idx] = cp.inf
            
            # Compute connection probabilities
            prob_dist = cp.exp(-dist_decay * distances)  # (chunk_n, n)
            
            # Trait similarity component
            chunk_traits_i = chunk_traits[:, None]  # (chunk_n, 1)
            traits_j = traits_cp[None, :]  # (1, n)
            prob_trait = (chunk_traits_i == traits_j).astype(cp.float32) * trait_bias  # (chunk_n, n)
            
            # Combined probability
            conn_prob = prob_dist + prob_trait  # (chunk_n, n)

            # Free intermediate arrays BEFORE argpartition — Thrust sort
            # allocates large hidden temporaries that can exceed the pool limit
            del prob_dist, prob_trait, distances, diff
            del chunk_pos_i, pos_j
            cp.get_default_memory_pool().free_all_blocks()

            # Select top-k connections for each neuron in chunk
            # Use argpartition (O(n)) instead of argsort (O(n log n)) - more memory efficient
            # argpartition returns indices where the k largest are in the last k positions (unsorted)
            partition_idx = cp.argpartition(conn_prob, -k, axis=1)[:, -k:]  # (chunk_n, k)
            # Get the actual values at these positions for sorting within top-k
            top_k_values = cp.take_along_axis(conn_prob, partition_idx, axis=1)
            # Sort within top-k to get proper ordering (small sort, k elements)
            sorted_within_k = cp.argsort(top_k_values, axis=1)
            top_k_indices = cp.take_along_axis(partition_idx, sorted_within_k, axis=1)  # (chunk_n, k)
            
            # Generate weights
            weights = cp.random.uniform(min_w, max_w, (chunk_n, k)).astype(cp.float32)
            
            # Create row indices (offset by start_idx for global indexing)
            chunk_rows = cp.repeat(cp.arange(start_idx, end_idx), k)  # (chunk_n * k,)
            chunk_cols = top_k_indices.ravel()  # (chunk_n * k,)
            chunk_weights = weights.ravel()  # (chunk_n * k,)
            
            # Accumulate (transfer to CPU immediately to free GPU memory)
            all_rows.append(cp.asnumpy(chunk_rows))
            all_cols.append(cp.asnumpy(chunk_cols))
            all_weights.append(cp.asnumpy(chunk_weights))

            # Explicit cleanup to prevent memory fragmentation
            # (diff, distances, prob_dist, prob_trait, chunk_pos_i, pos_j already freed pre-argpartition)
            del chunk_rows, chunk_cols, chunk_weights, weights
            del top_k_indices, top_k_values, sorted_within_k, partition_idx
            del conn_prob, chunk_pos, chunk_traits
            cp.get_default_memory_pool().free_all_blocks()

            # Progress update (every 10% or every chunk if few chunks)
            if num_chunks > 1 and ((chunk_idx + 1) % max(1, num_chunks // 10) == 0 or chunk_idx == num_chunks - 1):
                progress = ((chunk_idx + 1) / num_chunks) * 100
                elapsed = time.time() - start_t
                eta = elapsed / (chunk_idx + 1) * (num_chunks - chunk_idx - 1)
                self._log_console(f"Chunked progress: {progress:.1f}% ({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")
        
        # Concatenate all chunks
        all_rows_np = np.concatenate(all_rows)
        all_cols_np = np.concatenate(all_cols)
        all_weights_np = np.concatenate(all_weights)
        
        # Convert back to GPU and create sparse matrix
        row_indices_cp = cp.asarray(all_rows_np)
        col_indices_cp = cp.asarray(all_cols_np)
        weights_cp = cp.asarray(all_weights_np)
        
        conn_matrix = csp.coo_matrix(
            (weights_cp, (row_indices_cp, col_indices_cp)),
            shape=(n, n),
            dtype=cp.float32
        ).tocsr()
        
        conn_matrix.sort_indices()
        elapsed = time.time() - start_t
        self._log_console(f"Connections (3D Spatial GPU-Chunked): {conn_matrix.nnz}. Time: {elapsed:.2f}s")
        return conn_matrix
    
    def _generate_spatial_connections_3d(self, n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config):
        """Generates synaptic connections based on spatial proximity and trait similarity in 3D."""
        # Use vectorized GPU version for better performance
        if n > 1000:  # Use vectorized for large networks
            return self._generate_spatial_connections_3d_vectorized(n, max_connections_per_neuron, neuron_positions_3d_cp, traits_cp, config)
        
        # Legacy iterative version for small networks (< 1000 neurons)
        self._log_console("Generating connections (3D spatial, legacy)..."); start_t = time.time()
        if n == 0: 
            self._log_console("No neurons to connect (n=0).", "info")
            return csp.csr_matrix((0,0), dtype=cp.float32)

        dist_decay_factor = getattr(config, 'connection_distance_decay_factor', 0.01) 
        trait_bias = getattr(config, 'trait_connection_bias', 0.5) 
        min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight 

        rows, cols, weights_list = [], [], [] 

        for i in range(n): 
            pos_i_cp = neuron_positions_3d_cp[i:i+1, :] 
            trait_i_val = traits_cp[i] 

            candidate_indices_np = np.array([j for j in range(n) if j != i], dtype=np.int32)
            if candidate_indices_np.size == 0: continue 

            candidate_indices_cp = cp.asarray(candidate_indices_np) 
            pos_candidates_cp = neuron_positions_3d_cp[candidate_indices_cp] 
            traits_candidates_cp = traits_cp[candidate_indices_cp] 

            distances_cp = self._calculate_distances_3d_gpu(pos_i_cp, pos_candidates_cp)
            prob_distance_component = cp.exp(-dist_decay_factor * distances_cp) 
            prob_trait_component = (traits_candidates_cp == trait_i_val).astype(cp.float32) * trait_bias 
            connection_probabilities_cp = prob_distance_component + prob_trait_component
            
            sum_probs = cp.sum(connection_probabilities_cp)
            if sum_probs > 1e-9: 
                 normalized_probabilities_cp = connection_probabilities_cp / sum_probs
            else: 
                 if connection_probabilities_cp.size > 0: 
                    normalized_probabilities_cp = cp.ones_like(connection_probabilities_cp) / connection_probabilities_cp.size 
                 else:
                    continue 

            num_potential_targets = candidate_indices_cp.size
            if num_potential_targets > 0 :
                num_to_select = min(max_connections_per_neuron, num_potential_targets) 

                if num_to_select > 0:
                    try:
                        if not np.isclose(cp.asnumpy(cp.sum(normalized_probabilities_cp)), 1.0) and cp.sum(normalized_probabilities_cp) > 1e-9:
                            normalized_probabilities_cp = normalized_probabilities_cp / cp.sum(normalized_probabilities_cp)
                        elif cp.sum(normalized_probabilities_cp) <= 1e-9: 
                             selected_local_indices_cp = cp.random.choice(cp.arange(num_potential_targets), size=num_to_select, replace=False)
                        else: 
                             selected_local_indices_cp = cp.random.choice(
                                cp.arange(num_potential_targets),
                                size=num_to_select,
                                replace=False,
                                p=normalized_probabilities_cp 
                            )
                    except (ValueError, NotImplementedError) as e: 
                        sorted_local_indices_cp = cp.argsort(connection_probabilities_cp)[::-1] 
                        selected_local_indices_cp = sorted_local_indices_cp[:num_to_select] 

                    final_target_global_indices_cp = candidate_indices_cp[selected_local_indices_cp]
                    initial_weights_np = np.random.uniform(min_w, max_w, num_to_select).astype(np.float32)
                    final_weights_np = np.clip(initial_weights_np, min_w, max_w) 

                    rows.extend([i] * num_to_select) 
                    cols.extend(cp.asnumpy(final_target_global_indices_cp).tolist()) 
                    weights_list.extend(final_weights_np.tolist()) 

            if n > 0 and i % (max(1, n // 20)) == 0: 
                print(f"\rConn gen (3D Spatial): {i/n*100:.1f}%", end="")

        if n > 0: print("\rConn gen (3D Spatial): 100.0% ") 

        if not rows: 
            self._log_console("No connections generated by 3D spatial method.", "warning")
            return csp.csr_matrix((n, n), dtype=cp.float32) 

        conn_matrix = csp.csr_matrix((cp.asarray(weights_list,dtype=cp.float32),
                                      (cp.asarray(rows,dtype=cp.int32),cp.asarray(cols,dtype=cp.int32))),
                                     shape=(n,n),dtype=cp.float32)
        conn_matrix.sort_indices() 
        self._log_console(f"Connections (3D Spatial): {conn_matrix.nnz}. Time: {time.time()-start_t:.2f}s")
        return conn_matrix

    def _generate_watts_strogatz_connections_3d(self, n, k_neighbors, p_rewire, config):
        """Generates connections using a Watts-Strogatz small-world network model in 3D.
        
        Creates a small-world network with high clustering and short path lengths:
        1. Create ring lattice based on 3D spatial proximity (k nearest neighbors)
        2. Rewire each edge with probability p_rewire to a random target
        3. Maintain directed network structure
        
        Args:
            n: Number of neurons
            k_neighbors: Number of nearest spatial neighbors to connect (must be even)
            p_rewire: Rewiring probability (0 = regular lattice, 1 = random network)
            config: CoreSimConfig with weight parameters
        """
        self._log_console(f"Generating Watts-Strogatz 3D network (n={n}, k={k_neighbors}, p_rewire={p_rewire})...")
        start_t = time.time()
        
        if n == 0:
            return csp.csr_matrix((0, 0), dtype=cp.float32)

    def _generate_motif_connections_3d(self, n, neuron_positions_3d_cp, traits_cp, config, motif_name):
        """Generates connections according to a high-level connectivity motif.

        Motifs are defined in CONNECTIVITY_MOTIFS and operate on trait-based
        populations. This generator is optimized for small-to-medium networks
        where explicit population-based sampling is acceptable.
        """
        motif_def = CONNECTIVITY_MOTIFS.get(motif_name)
        if motif_def is None:
            self._log_console(f"Unknown connectivity motif '{motif_name}'. Falling back to spatial generator.", "warning")
            return self._generate_spatial_connections_3d(n, config.connections_per_neuron, neuron_positions_3d_cp, traits_cp, config)

        self._log_console(f"Generating connections (Motif: {motif_name})...")
        start_t = time.time()

        if n == 0:
            return csp.csr_matrix((0, 0), dtype=cp.float32)

        # For very large networks, fall back to spatial generator to avoid O(N^2) patterns
        if n > 50000:
            self._log_console(
                f"Network size n={n} too large for motif generator; falling back to spatial generator.",
                "warning",
            )
            return self._generate_spatial_connections_3d(n, config.connections_per_neuron, neuron_positions_3d_cp, traits_cp, config)

        # Traits on host for flexible population definitions
        if traits_cp is not None and traits_cp.size == n:
            traits_np = cp.asnumpy(traits_cp).astype(np.int32)
        else:
            traits_np = np.zeros(n, dtype=np.int32)

        base_k = getattr(config, "connectivity_k", getattr(config, "connections_per_neuron", 10))
        if base_k < 1:
            base_k = 1
        min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight

        rows: list[int] = []
        cols: list[int] = []
        weights_list: list[float] = []

        rules = motif_def.get("rules", [])
        for rule in rules:
            src_traits = rule.get("source_traits", [])
            tgt_traits = rule.get("target_traits", [])
            if not src_traits or not tgt_traits:
                continue

            k_fraction = float(rule.get("k_fraction", 1.0))
            if k_fraction <= 0.0:
                continue

            weight_scale = float(rule.get("weight_scale", 1.0))

            src_mask = np.isin(traits_np, np.array(src_traits, dtype=np.int32))
            tgt_mask = np.isin(traits_np, np.array(tgt_traits, dtype=np.int32))
            src_indices = np.nonzero(src_mask)[0]
            tgt_indices = np.nonzero(tgt_mask)[0]

            if src_indices.size == 0 or tgt_indices.size == 0:
                continue

            rule_k = int(max(0, round(base_k * k_fraction)))
            if rule_k <= 0:
                continue

            # Local weight range for this rule
            local_min_w = min_w * weight_scale
            local_max_w = max_w * weight_scale
            if local_min_w > local_max_w:
                local_min_w, local_max_w = local_max_w, local_min_w

            for src_idx in src_indices:
                # Avoid self-connections when source and target populations overlap
                if traits_np[src_idx] in tgt_traits and tgt_indices.size > 1:
                    available_targets = tgt_indices[tgt_indices != src_idx]
                    if available_targets.size == 0:
                        continue
                else:
                    available_targets = tgt_indices

                num_targets = min(rule_k, available_targets.size)
                if num_targets <= 0:
                    continue

                chosen_targets = np.random.choice(available_targets, size=num_targets, replace=False)
                weights = np.random.uniform(local_min_w, local_max_w, size=num_targets).astype(np.float32)

                rows.extend([int(src_idx)] * num_targets)
                cols.extend(chosen_targets.astype(np.int32).tolist())
                weights_list.extend(weights.tolist())

        if not rows:
            self._log_console(
                f"No connections generated by motif '{motif_name}'. Falling back to spatial generator.",
                "warning",
            )
            return self._generate_spatial_connections_3d(n, config.connections_per_neuron, neuron_positions_3d_cp, traits_cp, config)

        conn_matrix = csp.csr_matrix(
            (
                cp.asarray(weights_list, dtype=cp.float32),
                (cp.asarray(rows, dtype=cp.int32), cp.asarray(cols, dtype=cp.int32)),
            ),
            shape=(n, n),
            dtype=cp.float32,
        )
        conn_matrix.sort_indices()
        elapsed = time.time() - start_t
        self._log_console(
            f"Connections (Motif {motif_name}): {conn_matrix.nnz} synapses. Time: {elapsed:.2f}s",
        )
        return conn_matrix
        
        if n == 1:
            self._log_console("Only 1 neuron, returning empty connectivity.", "info")
            return csp.csr_matrix((1, 1), dtype=cp.float32)
        
        # Ensure k is valid and even
        k = min(k_neighbors, n - 1)
        if k % 2 == 1:
            k = k + 1  # Make even
            k = min(k, n - 1)
        if k < 2:
            k = 2
        
        min_w, max_w = config.hebbian_min_weight, config.hebbian_max_weight
        
        # Step 1: Create spatial ordering - sort neurons by 3D position
        # We'll use a space-filling curve approximation (sum of coordinates)
        positions = self.cp_neuron_positions_3d
        spatial_order = cp.sum(positions, axis=1)  # Simple spatial key
        sorted_indices = cp.argsort(spatial_order)
        
        # Step 2: Build k-nearest neighbor ring lattice
        # Each neuron connects to its k/2 predecessors and k/2 successors in spatial order
        rows = []
        cols = []
        weights = []
        
        half_k = k // 2
        
        for i in range(n):
            source_idx = int(sorted_indices[i])
            
            # Connect to k/2 neighbors on each side in the spatial ring
            for offset in range(1, half_k + 1):
                # Forward connections (clockwise)
                target_spatial_idx = (i + offset) % n
                target_idx = int(sorted_indices[target_spatial_idx])
                
                # Rewiring decision
                if cp.random.random() < p_rewire:
                    # Rewire to random target (avoid self-loops and duplicates)
                    target_idx = int(cp.random.randint(0, n))
                    while target_idx == source_idx:
                        target_idx = int(cp.random.randint(0, n))
                
                weight = float(cp.random.uniform(min_w, max_w))
                rows.append(source_idx)
                cols.append(target_idx)
                weights.append(weight)
                
                # Backward connections (counter-clockwise)
                target_spatial_idx = (i - offset) % n
                target_idx = int(sorted_indices[target_spatial_idx])
                
                # Rewiring decision
                if cp.random.random() < p_rewire:
                    # Rewire to random target
                    target_idx = int(cp.random.randint(0, n))
                    while target_idx == source_idx:
                        target_idx = int(cp.random.randint(0, n))
                
                weight = float(cp.random.uniform(min_w, max_w))
                rows.append(source_idx)
                cols.append(target_idx)
                weights.append(weight)
            
            # Progress indicator for large networks
            if n > 1000 and i % (n // 20) == 0:
                print(f"\rWS generation: {i/n*100:.1f}%", end="")
        
        if n > 1000:
            print("\rWS generation: 100.0%")
        
        # Step 3: Create sparse matrix and remove duplicate edges
        # Convert to COO, then CSR to handle duplicates
        rows_cp = cp.array(rows, dtype=cp.int32)
        cols_cp = cp.array(cols, dtype=cp.int32)
        weights_cp = cp.array(weights, dtype=cp.float32)
        
        conn_matrix = csp.coo_matrix(
            (weights_cp, (rows_cp, cols_cp)),
            shape=(n, n),
            dtype=cp.float32
        ).tocsr()
        
        # Remove self-loops if any exist
        conn_matrix.setdiag(cp.zeros(n, dtype=cp.float32))
        conn_matrix.eliminate_zeros()
        
        conn_matrix.sort_indices()
        elapsed = time.time() - start_t
        
        # Calculate network statistics
        avg_degree = conn_matrix.nnz / n if n > 0 else 0
        
        self._log_console(
            f"Watts-Strogatz network complete: {conn_matrix.nnz} connections "
            f"(avg degree: {avg_degree:.1f}, expected: {k}). Time: {elapsed:.2f}s"
        )
        
        return conn_matrix

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
            'cp_stp_u','cp_stp_x',
            'cp_ou_current'  # OU process state for background noise
        ]
        for attr_name in attrs_to_clear:
            if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                setattr(self, attr_name, None) 

        if 'cupy' in sys.modules and cp.is_available():
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

        # Invalidate COO cache so stale data from previous network doesn't persist
        self._cached_coo_matrix = None
        self._coo_cache_valid = False
        self._synapse_count = 0
        self._synapse_capacity = 0
        self._compaction_counter = 0
        self._pending_eliminations = False

        self.is_initialized = False
        self._log_console("Cleared simulation state and GPU memory.")

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
        if ui_to_sim_queue:
            command = "PAUSE" if not self.runtime_state.is_paused else "RESUME"
            try:
                ui_to_sim_queue.put_nowait({"type": command})
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
        
        mem_info = cp.cuda.Device().mem_info
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
            mem_info = cp.cuda.Device().mem_info
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

        if self.cp_traits is not None: snapshot["cp_traits"] = cp.asnumpy(self.cp_traits)
        if self.cp_neuron_positions_3d is not None: snapshot["cp_neuron_positions_3d"] = cp.asnumpy(self.cp_neuron_positions_3d)

        if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
            for param in ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c_reset', 'd_increment']:
                attr_name = f"cp_izh_{param}"
                if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                    snapshot[attr_name] = cp.asnumpy(getattr(self, attr_name))
        elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
            for param in ['C_m', 'g_Na_max', 'g_K_max', 'g_L', 'E_Na', 'E_K', 'E_L', 'v_peak']:
                attr_name = f"cp_hh_{param}"
                if hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                    snapshot[attr_name] = cp.asnumpy(getattr(self, attr_name))

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
                snapshot[attr_name] = cp.asnumpy(array_data)
            else: 
                snapshot[attr_name] = None 

        if self.cp_connections is not None:
            snapshot["connections_data"] = cp.asnumpy(self.cp_connections.data) if self.cp_connections.data is not None else np.array([])
            snapshot["connections_indices"] = cp.asnumpy(self.cp_connections.indices) if self.cp_connections.indices is not None else np.array([])
            snapshot["connections_indptr"] = cp.asnumpy(self.cp_connections.indptr) if self.cp_connections.indptr is not None else np.array([])
            snapshot["connections_shape"] = self.cp_connections.shape 
        else: 
            snapshot["connections_data"] = np.array([]); snapshot["connections_indices"] = np.array([])
            snapshot["connections_indptr"] = np.array([]); snapshot["connections_shape"] = (0,0)

        # Save only active portion of pre-allocated STP arrays
        synapse_count = getattr(self, '_synapse_count', None)
        if self.cp_stp_u is not None:
            active_u = self.cp_stp_u[:synapse_count] if synapse_count else self.cp_stp_u
            snapshot["cp_stp_u"] = cp.asnumpy(active_u)
        else: snapshot["cp_stp_u"] = None
        if self.cp_stp_x is not None:
            active_x = self.cp_stp_x[:synapse_count] if synapse_count else self.cp_stp_x
            snapshot["cp_stp_x"] = cp.asnumpy(active_x)
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
                            frame_data_np[key] = cp.asnumpy(value)
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
                frame_data[attr_name] = cp.asnumpy(array_data)
            else:
                frame_data[attr_name] = None

        # Capture synaptic data (optional - this is the large part)
        if not skip_synaptic_data:
            if self.core_config.enable_hebbian_learning and self.cp_connections is not None:
                if self.cp_connections.data is not None:
                    frame_data["cp_connections_data"] = cp.asnumpy(self.cp_connections.data)

            if self.core_config.enable_short_term_plasticity:
                synapse_count = getattr(self, '_synapse_count', None)
                if self.cp_stp_u is not None:
                    frame_data["cp_stp_u"] = cp.asnumpy(
                        self.cp_stp_u[:synapse_count] if synapse_count else self.cp_stp_u
                    )
                if self.cp_stp_x is not None:
                    frame_data["cp_stp_x"] = cp.asnumpy(
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
                                frame_data["cp_connections_data"] = cp.asnumpy(self.cp_connections.data)

                        if self.core_config.enable_short_term_plasticity:
                            synapse_count = getattr(self, '_synapse_count', None)
                            if self.cp_stp_u is not None:
                                frame_data["cp_stp_u"] = cp.asnumpy(self.cp_stp_u[:synapse_count] if synapse_count else self.cp_stp_u)
                            if self.cp_stp_x is not None:
                                frame_data["cp_stp_x"] = cp.asnumpy(self.cp_stp_x[:synapse_count] if synapse_count else self.cp_stp_x)

                    for attr_name in dynamic_arrays_to_capture:
                        array_data = getattr(self, attr_name, None)
                        if array_data is not None:
                            frame_data[attr_name] = cp.asnumpy(array_data)  # GPU→CPU transfer
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
        mem_info = cp.cuda.Device().mem_info
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
            cp.get_default_memory_pool().free_all_blocks()  # Free unused CuPy memory

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
                mem_info = cp.cuda.Device().mem_info
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

                except cp.cuda.memory.OutOfMemoryError:
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
            mem_info = cp.cuda.Device().mem_info
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

    def _run_one_simulation_step(self):
        """Executes a single step of the simulation logic."""
        if not self.is_initialized or self.core_config.num_neurons == 0: return
        try:
            n_neurons = self.core_config.num_neurons; cfg = self.core_config; dt = cfg.dt_ms

            # --- 1. Synaptic Plasticity (STP) Update ---
            base_synaptic_weights = self.cp_connections.data 
            effective_synaptic_strength = base_synaptic_weights 

            if cfg.enable_short_term_plasticity and self.cp_connections.nnz > 0 and \
               self.cp_stp_u is not None and self.cp_stp_x is not None:

                # Per-synapse-type STP: build per-synapse tau_f/tau_d/U arrays from connection types
                if cfg.enable_per_type_stp and self.cp_synapse_conn_type is not None:
                    actual_nnz_stp = self.cp_connections.nnz
                    ctypes = self.cp_synapse_conn_type[:actual_nnz_stp]
                    # Build per-synapse parameter arrays via lookup table
                    U_arr = cp.array(cfg.stp_U_per_type, dtype=cp.float32)
                    tau_f_arr = cp.array(cfg.stp_tau_f_per_type, dtype=cp.float32)
                    tau_d_arr = cp.array(cfg.stp_tau_d_per_type, dtype=cp.float32)
                    stp_tau_f_per_syn = tau_f_arr[ctypes]
                    stp_tau_d_per_syn = tau_d_arr[ctypes]
                    stp_U_per_syn = U_arr[ctypes]
                    # Pad to full array length for fused kernel (capacity may exceed nnz)
                    n_pad = self.cp_stp_u.size - actual_nnz_stp
                    if n_pad > 0:
                        stp_tau_f_full = cp.concatenate([stp_tau_f_per_syn, cp.full(n_pad, cfg.stp_tau_f, dtype=cp.float32)])
                        stp_tau_d_full = cp.concatenate([stp_tau_d_per_syn, cp.full(n_pad, cfg.stp_tau_d, dtype=cp.float32)])
                    else:
                        stp_tau_f_full = stp_tau_f_per_syn
                        stp_tau_d_full = stp_tau_d_per_syn
                    self.cp_stp_u, self.cp_stp_x = fused_stp_decay_recovery(
                        self.cp_stp_u, self.cp_stp_x, dt, stp_tau_f_full, stp_tau_d_full)
                else:
                    stp_U_per_syn = None
                    self.cp_stp_u, self.cp_stp_x = fused_stp_decay_recovery(
                        self.cp_stp_u, self.cp_stp_x, dt, cfg.stp_tau_f, cfg.stp_tau_d)

                if self.cp_prev_firing_states.any():
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
                effective_connections_matrix = csp.csr_matrix(
                    (effective_synaptic_strength, self.cp_connections.indices, self.cp_connections.indptr),
                    shape=self.cp_connections.shape
                )
            else: 
                effective_connections_matrix = self.cp_connections 

            # --- 2. Synaptic Conductance Update & Current Calculation ---
            decay_e = self._cached_decay_e
            decay_i = self._cached_decay_i

            self.cp_conductance_g_e, self.cp_conductance_g_i, synaptic_current_I_syn_pA = fused_conductance_decay_and_current(
                self.cp_conductance_g_e, self.cp_conductance_g_i, decay_e, decay_i,
                self.cp_membrane_potential_v, cfg.syn_reversal_potential_e, cfg.syn_reversal_potential_i
            )

            g_e_increase = None  # Track for NMDA input
            if effective_connections_matrix.nnz > 0 and self.cp_prev_firing_states.any():
                prev_fired_float = self.cp_prev_firing_states.astype(cp.float32)

                if cfg.enable_inhibitory_neurons and self.cp_traits is not None:
                    # Support multiple inhibitory trait indices while preserving legacy single-index behavior
                    inhibitory_indices = getattr(cfg, "inhibitory_trait_indices", None)
                    if inhibitory_indices:
                        inhibitory_indices_cp = cp.asarray(inhibitory_indices, dtype=cp.int32)
                        is_inhibitory_neuron_output = cp.isin(self.cp_traits, inhibitory_indices_cp)
                    else:
                        is_inhibitory_neuron_output = (self.cp_traits == cfg.inhibitory_trait_index)
                    exc_fired_prev = prev_fired_float * (~is_inhibitory_neuron_output)
                    inhib_fired_prev = prev_fired_float * is_inhibitory_neuron_output

                    g_e_increase = (effective_connections_matrix.T @ exc_fired_prev) * cfg.propagation_strength
                    g_i_increase = (effective_connections_matrix.T @ inhib_fired_prev) * cfg.inhibitory_propagation_strength

                    self.cp_conductance_g_e += g_e_increase
                    self.cp_conductance_g_i += g_i_increase
                else:
                    g_e_increase = (effective_connections_matrix.T @ prev_fired_float) * cfg.propagation_strength
                    self.cp_conductance_g_e += g_e_increase

            total_input_current_pA = synaptic_current_I_syn_pA + self.cp_external_input_current

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

                v_new, m_new, h_new, n_new = fused_hodgkin_huxley_dynamics_update(
                    self.cp_membrane_potential_v, self.cp_gating_variable_m, self.cp_gating_variable_h, self.cp_gating_variable_n,
                    effective_input_uA, dt,
                    self.cp_hh_C_m, g_Na_effective, g_K_effective, self.cp_hh_g_L,
                    self.cp_hh_E_Na, self.cp_hh_E_K, self.cp_hh_E_L,
                    cfg.hh_temperature_celsius, cfg.hh_q10_factor
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

            self.cp_firing_states[:] = fired_this_step

            # Accumulate spike count on GPU, sync to CPU periodically (reduces GPU-CPU stalls)
            spike_count_gpu = cp.sum(fired_this_step)
            if self._accumulated_spikes_gpu is None:
                self._accumulated_spikes_gpu = spike_count_gpu
            else:
                self._accumulated_spikes_gpu += spike_count_gpu

            self._stats_sync_counter += 1
            if self._stats_sync_counter >= self.gpu_config.stats_sync_interval_steps:
                self._mock_num_spikes_this_step = int(self._accumulated_spikes_gpu.get()) // self._stats_sync_counter
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
               self.cp_synapse_pulse_timers is not None and fired_this_step.any(): 
                if self.cp_connections is not None and self.cp_connections.nnz > 0:
                    coo_matrix_for_pulses = self._get_cached_coo()  # Use cached COO
                    presynaptic_fired_mask_for_pulses = fired_this_step[coo_matrix_for_pulses.row]
                    synapses_to_activate_indices = cp.where(presynaptic_fired_mask_for_pulses)[0]

                    if synapses_to_activate_indices.size > 0:
                        pulse_lifetime = opengl_viz_config.get("SYNAPTIC_PULSE_MAX_LIFETIME_FRAMES", 5)
                        self.cp_synapse_pulse_timers[synapses_to_activate_indices] = pulse_lifetime 
                        self.cp_synapse_pulse_progress[synapses_to_activate_indices] = 0.0 

            # --- 4. Hebbian Learning (Long-Term Potentiation/Depression) ---
            if cfg.enable_hebbian_learning and self.cp_connections.nnz > 0 and \
               self.cp_connections.data is not None and self.cp_connections.data.size > 0:
                if self.cp_prev_firing_states.any() and fired_this_step.any(): 
                    coo_matrix_heb = self._get_cached_coo()  # Use cached COO
                    pre_fired_mask_heb = self.cp_prev_firing_states[coo_matrix_heb.row] 
                    post_fired_mask_heb = fired_this_step[coo_matrix_heb.col] 

                    active_synapse_indices_heb = cp.where(pre_fired_mask_heb & post_fired_mask_heb)[0]
                    num_potentiation_events = 0
                    if active_synapse_indices_heb.size > 0:
                        base_weights_data_array = self.cp_connections.data 
                        current_weights_active_syn = base_weights_data_array[active_synapse_indices_heb]
                        delta_weights = cfg.hebbian_learning_rate * (cfg.hebbian_max_weight - current_weights_active_syn)
                        base_weights_data_array[active_synapse_indices_heb] += delta_weights
                        num_potentiation_events = active_synapse_indices_heb.size

                    self.cp_connections.data *= (1.0 - cfg.hebbian_weight_decay) 
                    cp.clip(self.cp_connections.data, cfg.hebbian_min_weight, cfg.hebbian_max_weight, out=self.cp_connections.data)
                    if num_potentiation_events > 0: self._mock_total_plasticity_events += num_potentiation_events
            
            # --- 4b. C2: STDP (Spike-Timing-Dependent Plasticity) ---
            if cfg.enable_stdp and self.cp_last_spike_time is not None and self.cp_connections.nnz > 0:
                current_time = self.runtime_state.current_time_ms

                # Update last spike times for neurons that fired this step
                if fired_this_step.any():
                    self.cp_last_spike_time = cp.where(
                        fired_this_step,
                        current_time,
                        self.cp_last_spike_time
                    )

                # Apply STDP updates — ONLY for synapses connected to neurons that just fired.
                # This is the key optimization: instead of computing delta_t for ALL synapses
                # and filtering, we pre-filter to synapses where pre OR post neuron fired this step.
                # At typical firing rates (2-10 Hz), this reduces the working set from ~1M to ~1-10K.
                if fired_this_step.any():
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

                            self.cp_connections.data[stdp_active_indices] = updated_weights

                            # Update eligibility traces if reward modulation is enabled
                            if cfg.enable_reward_modulation and self.cp_eligibility_trace is not None:
                                weight_changes = updated_weights - current_weights
                                self.cp_eligibility_trace[stdp_active_indices] += cp.abs(weight_changes)

                            self._mock_total_plasticity_events += stdp_active_indices.size
            
            # --- 4c. C2: Reward-Modulated Plasticity (Three-Factor Learning) ---
            if cfg.enable_reward_modulation and self.cp_eligibility_trace is not None and self.cp_connections.nnz > 0:
                # Decay eligibility traces
                decay_factor = cp.exp(-dt / cfg.reward_eligibility_tau_ms)
                self.cp_eligibility_trace = fused_eligibility_trace_decay(
                    self.cp_eligibility_trace,
                    decay_factor
                )
                
                # Apply reward modulation if reward signal is non-zero
                reward_prediction_error = cfg.current_reward_signal - cfg.reward_baseline
                if abs(reward_prediction_error) > 1e-6:  # Only update if there's a reward signal
                    # Modulate weights based on eligibility trace and reward
                    # Delta_w = learning_rate * reward_error * eligibility_trace
                    weight_updates = cfg.reward_learning_rate * reward_prediction_error * self.cp_eligibility_trace
                    self.cp_connections.data += weight_updates
                    
                    # Clip to bounds (use STDP bounds if STDP is enabled, otherwise Hebbian bounds)
                    w_min = cfg.stdp_w_min if cfg.enable_stdp else cfg.hebbian_min_weight
                    w_max = cfg.stdp_w_max if cfg.enable_stdp else cfg.hebbian_max_weight
                    cp.clip(self.cp_connections.data, w_min, w_max, out=self.cp_connections.data)
                    
                    # Count significant updates
                    significant_updates = cp.sum(cp.abs(weight_updates) > 1e-6)
                    if significant_updates > 0:
                        self._mock_total_plasticity_events += int(significant_updates.get())
            
            # --- 4d. C3: Structural Plasticity (Synapse Formation/Elimination) ---
            if cfg.enable_structural_plasticity and self.cp_struct_plast_step_counter is not None:
                self.cp_struct_plast_step_counter += 1
                
                # Only update periodically for efficiency
                if self.cp_struct_plast_step_counter >= cfg.struct_plast_update_interval_steps:
                    self.cp_struct_plast_step_counter = 0
                    
                    # Synapse elimination: remove weak synapses
                    weak_synapse_mask = self.cp_connections.data < cfg.struct_plast_weight_threshold
                    num_weak = cp.sum(weak_synapse_mask).get()
                    
                    if num_weak > 0:
                        # Probabilistic elimination based on elimination rate
                        # Rate is per-synapse-per-timestep, so scale by update interval
                        elimination_prob = cfg.struct_plast_elimination_rate * cfg.struct_plast_update_interval_steps
                        elimination_prob = min(elimination_prob, 0.5)  # Cap at 50% per update
                        
                        # Generate random numbers for each weak synapse
                        eliminate_mask = weak_synapse_mask & (cp.random.rand(self.cp_connections.nnz) < elimination_prob)
                        num_eliminated = cp.sum(eliminate_mask).get()
                        
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
                                ema_probs_np = cp.asnumpy(ema_probs).astype(np.float64)
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

            # --- 5. Homeostatic Plasticity ---
            # 5a. Adaptive thresholds (Izhikevich-specific)
            if cfg.enable_homeostasis and self.cp_neuron_firing_thresholds is not None:
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
            if cfg.enable_synaptic_scaling and self.cp_connections is not None and self.cp_connections.nnz > 0:
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
                    self.cp_connections.data[:] = self.cp_connections.data * post_scales
                    # Enforce weight bounds
                    if cfg.enable_hebbian_learning:
                        cp.clip(self.cp_connections.data, cfg.hebbian_min_weight, cfg.hebbian_max_weight, out=self.cp_connections.data)
                    else:
                        cp.clip(self.cp_connections.data, 0.01, 5.0, out=self.cp_connections.data)

            # --- 6. Prepare for Next Step & Record Frame ---
            self.cp_prev_firing_states[:] = fired_this_step 
            self.record_current_frame_if_active() # This was the missing method call's target

            # Note: Network firing rate calculation deferred to avoid GPU->CPU sync every step
            # Will be updated on-demand when GUI data is requested

        except Exception as e:
            self._log_to_ui(f"Error during simulation step: {e}","critical")
            import traceback; traceback.print_exc() 
            self.stop_simulation() 
            if self.ui_queue: self.ui_queue.put({"type": "SIM_ERROR_OCCURRED", "error_message": str(e)})
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
                        state_group.create_dataset(attr_name, data=cp.asnumpy(data_array), compression="gzip")
                    elif data_array is not None: 
                         state_group.attrs[f"{attr_name}_is_empty"] = True

                if self.cp_connections is not None:
                    if self.cp_connections.data is not None and self.cp_connections.data.size > 0:
                        state_group.create_dataset("connections_data", data=cp.asnumpy(self.cp_connections.data), compression="gzip")
                    if self.cp_connections.indices is not None and self.cp_connections.indices.size > 0:
                        state_group.create_dataset("connections_indices", data=cp.asnumpy(self.cp_connections.indices), compression="gzip")
                    if self.cp_connections.indptr is not None and self.cp_connections.indptr.size > 0:
                        state_group.create_dataset("connections_indptr", data=cp.asnumpy(self.cp_connections.indptr), compression="gzip")
                    state_group.attrs["connections_shape_0"] = self.cp_connections.shape[0]
                    state_group.attrs["connections_shape_1"] = self.cp_connections.shape[1]

                # Save only active synapse elements (not pre-allocated capacity)
                synapse_count = getattr(self, '_synapse_count', None)
                if self.cp_stp_u is not None and self.cp_stp_u.size > 0:
                    active_stp_u = self.cp_stp_u[:synapse_count] if synapse_count else self.cp_stp_u
                    state_group.create_dataset("cp_stp_u", data=cp.asnumpy(active_stp_u), compression="gzip")
                elif self.cp_stp_u is not None: state_group.attrs["cp_stp_u_is_empty"] = True
                if self.cp_stp_x is not None and self.cp_stp_x.size > 0:
                    active_stp_x = self.cp_stp_x[:synapse_count] if synapse_count else self.cp_stp_x
                    state_group.create_dataset("cp_stp_x", data=cp.asnumpy(active_stp_x), compression="gzip")
                elif self.cp_stp_x is not None: state_group.attrs["cp_stp_x_is_empty"] = True
                
                # C2: Save STDP and reward modulation state
                if self.cp_last_spike_time is not None and self.cp_last_spike_time.size > 0:
                    state_group.create_dataset("cp_last_spike_time", data=cp.asnumpy(self.cp_last_spike_time), compression="gzip")
                elif self.cp_last_spike_time is not None:
                    state_group.attrs["cp_last_spike_time_is_empty"] = True
                
                if self.cp_eligibility_trace is not None and self.cp_eligibility_trace.size > 0:
                    active_traces = self.cp_eligibility_trace[:synapse_count] if synapse_count else self.cp_eligibility_trace
                    state_group.create_dataset("cp_eligibility_trace", data=cp.asnumpy(active_traces), compression="gzip")
                elif self.cp_eligibility_trace is not None:
                    state_group.attrs["cp_eligibility_trace_is_empty"] = True

                # Save synapse visualization arrays (synapse-indexed with pre-allocation)
                if self.cp_synapse_pulse_timers is not None and self.cp_synapse_pulse_timers.size > 0:
                    active_timers = self.cp_synapse_pulse_timers[:synapse_count] if synapse_count else self.cp_synapse_pulse_timers
                    state_group.create_dataset("cp_synapse_pulse_timers", data=cp.asnumpy(active_timers), compression="gzip")
                elif self.cp_synapse_pulse_timers is not None:
                    state_group.attrs["cp_synapse_pulse_timers_is_empty"] = True
                if self.cp_synapse_pulse_progress is not None and self.cp_synapse_pulse_progress.size > 0:
                    active_progress = self.cp_synapse_pulse_progress[:synapse_count] if synapse_count else self.cp_synapse_pulse_progress
                    state_group.create_dataset("cp_synapse_pulse_progress", data=cp.asnumpy(active_progress), compression="gzip")
                elif self.cp_synapse_pulse_progress is not None:
                    state_group.attrs["cp_synapse_pulse_progress_is_empty"] = True

                # C3: Save structural plasticity state
                if self.cp_struct_plast_step_counter is not None:
                    state_group.attrs["cp_struct_plast_step_counter"] = self.cp_struct_plast_step_counter

                if self.core_config.neuron_model_type == NeuronModel.IZHIKEVICH.name:
                    if self.cp_recovery_variable_u is not None and self.cp_recovery_variable_u.size > 0: state_group.create_dataset("cp_recovery_variable_u", data=cp.asnumpy(self.cp_recovery_variable_u), compression="gzip")
                    elif self.cp_recovery_variable_u is not None : state_group.attrs["cp_recovery_variable_u_is_empty"] = True
                    for param in ['C', 'k', 'vr', 'vt', 'vpeak', 'a', 'b', 'c_reset', 'd_increment']:
                         attr_name_cp = f"cp_izh_{param}"
                         data_array = getattr(self, attr_name_cp, None)
                         if data_array is not None and data_array.size > 0: state_group.create_dataset(attr_name_cp, data=cp.asnumpy(data_array), compression="gzip")
                         elif data_array is not None : state_group.attrs[f"{attr_name_cp}_is_empty"] = True
                    if self.cp_neuron_firing_thresholds is not None and self.cp_neuron_firing_thresholds.size > 0: state_group.create_dataset("cp_neuron_firing_thresholds", data=cp.asnumpy(self.cp_neuron_firing_thresholds), compression="gzip")
                    elif self.cp_neuron_firing_thresholds is not None : state_group.attrs["cp_neuron_firing_thresholds_is_empty"] = True

                elif self.core_config.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
                    for attr_name_suffix in ['m', 'h', 'n']:
                        attr_name_cp = f"cp_gating_variable_{attr_name_suffix}"
                        data_array = getattr(self, attr_name_cp, None)
                        if data_array is not None and data_array.size > 0: state_group.create_dataset(attr_name_cp, data=cp.asnumpy(data_array), compression="gzip")
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
                            state_group.create_dataset(attr_name_cp, data=cp.asnumpy(data_array), compression="gzip")
                        elif data_array is not None:
                            state_group.attrs[f"{attr_name_cp}_is_empty"] = True
                    for param in ['C_m', 'g_Na_max', 'g_K_max', 'g_L', 'E_Na', 'E_K', 'E_L', 'v_peak']:
                         attr_name_cp = f"cp_hh_{param}"
                         data_array = getattr(self, attr_name_cp, None)
                         if data_array is not None and data_array.size > 0: state_group.create_dataset(attr_name_cp, data=cp.asnumpy(data_array), compression="gzip")
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
                
                temp_cfg_for_validation = SimulationConfiguration() 
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
        
        # Compute spike count on-demand only when GUI requests it (avoids sync every step)
        num_spikes_this_step = int(cp.sum(self.cp_firing_states).get()) if self.cp_firing_states is not None else 0
        
        # Compute firing rate on-demand using current spike count
        if n > 0 and dt > 0:
            instantaneous_rate_hz = (num_spikes_this_step / n) / (dt / 1000.0)
            self._mock_network_avg_firing_rate_hz = self._mock_network_avg_firing_rate_hz * 0.95 + instantaneous_rate_hz * 0.05
        else:
            self._mock_network_avg_firing_rate_hz = 0.0
        
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

        if self.cp_neuron_positions_3d is not None:
            gui_data_dict["neuron_positions_3d_cp"] = self.cp_neuron_positions_3d.copy()
        elif n > 0:
            gui_data_dict["neuron_positions_3d_cp"] = cp.zeros((n,3),dtype=cp.float32)
        else:
            gui_data_dict["neuron_positions_3d_cp"] = cp.array([], dtype=cp.float32).reshape(0,3)

        if self.cp_traits is not None:
            gui_data_dict["neuron_traits_cp"] = self.cp_traits.copy()
        elif n > 0:
            gui_data_dict["neuron_traits_cp"] = cp.zeros(n, dtype=cp.int32)
        else:
            gui_data_dict["neuron_traits_cp"] = cp.array([], dtype=cp.int32)
        
        # Add neuron type IDs for GPU-efficient filtering
        if self.cp_neuron_type_ids is not None:
            gui_data_dict["neuron_type_ids_cp"] = self.cp_neuron_type_ids.copy()
        elif n > 0:
            gui_data_dict["neuron_type_ids_cp"] = cp.zeros(n, dtype=cp.int32)
        else:
            gui_data_dict["neuron_type_ids_cp"] = cp.array([], dtype=cp.int32)

        # --- Data for DPG text display (can be NumPy or Python types) ---
        if self.cp_membrane_potential_v is not None:
            # Example: If you need a small sample of Vm for a DPG plot (not for GL points usually)
            # sample_indices_vm = cp.random.choice(cp.arange(n), size=min(n, 100), replace=False) if n > 0 else cp.array([])
            # gui_data_dict["neuron_Vm_sample_np"] = cp.asnumpy(self.cp_membrane_potential_v[sample_indices_vm]) if sample_indices_vm.size > 0 else np.array([])
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
                        row_indices_np = cp.asnumpy(coo_conn.row[indices_to_sample_np])
                        col_indices_np = cp.asnumpy(coo_conn.col[indices_to_sample_np])

                        weights_data_to_use_cp = self.cp_connections.data 
                        if self.core_config.enable_short_term_plasticity and \
                        self.cp_stp_u is not None and self.cp_stp_x is not None and \
                        self.cp_stp_u.size == self.cp_connections.data.size and \
                        self.cp_stp_x.size == self.cp_connections.data.size :
                            weights_data_to_use_cp = self.cp_connections.data * self.cp_stp_u * self.cp_stp_x

                        # Sample weights using NumPy indices on the CuPy array, then convert
                        sampled_weights_np = cp.asnumpy(weights_data_to_use_cp[cp.asarray(indices_to_sample_np)])

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
        #     gui_data_dict["neuron_Vm_trace_sample_np"] = cp.asnumpy(self.cp_membrane_potential_v[sample_indices])

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
                    np_traits_host_temp = cp.asnumpy(self.cp_traits) if self.cp_traits is not None and self.cp_traits.size == num_n else \
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

            positions_3d_np = cp.asnumpy(self.cp_neuron_positions_3d) if self.cp_neuron_positions_3d is not None else np.zeros((0,3), dtype=np.float32)
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
global_gui_state = { # Primarily managed by the UI thread
    "filters_changed": False, # Flag for OpenGL to know if its data needs refresh due to filter changes
    "current_profile_name": "default_profile.json", # Profiles remain .json
    "_was_running_last_frame": False, # Internal DPG loop state tracking
    "show_connections_gl": True, # Toggle for showing synapses in OpenGL
    "neuron_filter_mode_gl": 0, # Example: 0: All, 1: Spiking (actual filtering more complex)
    "_dt_warning_logged": False, # Flag to prevent spamming dt warnings
    "reset_sim_needed_from_ui_change": False, # If UI params changed that require sim reset

    # Recording & Playback State (Managed by UI thread based on commands and sim_thread feedback)
    "is_recording_active": False, # True if UI has commanded start_recording and sim_thread confirmed
    "is_playback_mode_active": False, # True if UI has commanded start_playback and sim_thread confirmed & setup
    "current_playback_frame_index": 0, # Current frame index for playback (UI perspective)
    "active_recording_data_source": None, # Holds dict from _prepare_loaded_recording_metadata (incl. H5 file handle for sim_thread)
    "playback_is_playing_ui": False, # UI's view of whether playback is auto-stepping
    "last_playback_autostep_time_ui": 0.0, # For UI-driven playback timing
    "playback_fps_ui": 30.0, # Target FPS for UI-driven playback
    "loaded_recording_filepath_for_ui": None # Path of the currently loaded recording file
}

# Data cache for visualization (primarily for OpenGL, updated by UI thread from sim_to_ui_queue)
# This data is prepared by the UI thread before being passed to OpenGL functions.
global_viz_data_cache = {
    "neuron_positions_x": [], # For 2D projection (if used)
    "neuron_positions_y": [], # For 2D projection (if used)
    "neuron_types": [], # List of type strings for each neuron (for GL filtering)
    "last_visible_neuron_indices": [], # Indices of neurons actually rendered in last GL frame
    "last_visible_synapse_indices": [], # Indices of synapses rendered
    # --- Data passed from Sim_Thread to UI_Thread for OpenGL update ---
    "gl_render_data_buffer": None, # Holds the latest full data snapshot from sim_thread for GL
    "gl_render_data_available": threading.Event(), # Event to signal new data for GL
    "gl_render_data_lock": threading.Lock() # Lock for accessing gl_render_data_buffer
}


# --- Shutdown Flag & Other Top-Level Globals ---
shutdown_flag = threading.Event() # Global shutdown signal for all threads
last_sim_update_time_dpg = 0.0 # Used in the DPG rendering loop (main thread) if it drives sim steps (not in threaded)

# --- OpenGL Specific Globals & Config ---
# These are primarily accessed by the OpenGL rendering functions, running in the main thread.
gl_neuron_pos_vbo = None # Vertex Buffer Object for neuron positions
gl_neuron_color_vbo = None # VBO for neuron colors
gl_synapse_vertices_vbo = None # VBO for synapse lines
gl_pulse_vertices_vbo = None # VBO for synaptic pulse points

gl_num_neurons_to_draw = 0 # Number of neurons to draw in current GL frame
gl_num_synapse_lines_to_draw = 0 # Number of synapse lines
gl_num_pulses_to_draw = 0 # Number of synaptic pulses

# Frame rate limiting for smooth 60 FPS
gl_last_render_time = 0.0
gl_target_frame_time = 1.0 / 60.0  # 60 FPS = 16.67ms per frame

# FPS counter tracking
gl_frame_times = []  # Rolling window of recent frame times
gl_fps_update_interval = 0.5  # Update FPS display every 0.5 seconds
gl_last_fps_update_time = 0.0
gl_current_fps = 0.0  # Current FPS to display

# NumPy arrays holding data ready for VBO buffering (populated by UI thread before GL render)
gl_neuron_pos_cp = cp.array([], dtype=cp.float32).reshape(0,3) # Changed from _np
gl_neuron_colors_cp = cp.array([], dtype=cp.float32).reshape(0,4) # Changed from _np
gl_connection_vertices_cp = cp.array([], dtype=cp.float32).reshape(0,3) # Changed from _np
gl_pulse_vertices_cp = cp.array([], dtype=cp.float32).reshape(0,3) # Changed from _np

# CUDA-OpenGL interop flag (initialized in init_gl)
cuda_gl_interop_enabled = False


if OPENGL_AVAILABLE:
    # opengl_viz_config is primarily read by GL functions in main thread.
    # Changes from UI (e.g. point size slider) will update this dict in main thread.
    opengl_viz_config = {
        "WINDOW_WIDTH": 800, "WINDOW_HEIGHT": 600, # Initial, updated on reshape
        "POINT_SIZE": 3.0, # Default neuron point size
        "MAX_NEURONS_TO_RENDER": 1000000, # Max neurons GL will attempt to draw (performance cap)
        "MAX_CONNECTIONS_TO_RENDER": 10000000, # Max synapses GL will attempt to draw
        "INACTIVE_NEURON_OPACITY": 0.25, # Base opacity for non-firing neurons
        "FIRING_NEURON_COLOR": [1.0, 1.0, 0.0, 1.0], # RGBA for spiking neurons
        "ACTIVITY_HIGHLIGHT_FRAMES": 7, # Frames a neuron stays highlighted after firing
        "FOOTER_HEIGHT_PIXELS": 75, # Height of text overlay at bottom of GL window
        "SYNAPSE_ALPHA_MODIFIER": 0.50, # Multiplier for base synapse alpha
        "SYNAPSE_BASE_COLOR": [0.4, 0.4, 0.5], # Base RGB for synapses
        "CAMERA_PAN_SPEED_FACTOR": 0.1, # Mouse pan speed
        "CAMERA_ROTATE_SPEED_FACTOR": 0.005, # Mouse rotate speed
        "CAMERA_ZOOM_SPEED_FACTOR": 20.0, # Mouse scroll zoom speed
        "ENABLE_SYNAPTIC_PULSES": True, # Toggle for visualizing synaptic pulses
        "SYNAPTIC_PULSE_COLOR": [0.7, 0.9, 1.0, 0.9], # RGBA for pulses
        "SYNAPTIC_PULSE_SIZE": 3.0, # Point size for pulses
        "SYNAPTIC_PULSE_MAX_LIFETIME_FRAMES": 5, # How many sim steps a pulse point lasts
    }
    # Color map for neuron traits (RGBA, A is base opacity)
    TRAIT_COLOR_MAP_RAW = [ # Keep raw Python list for DPG UI trait count reference
    [0.8, 0.2, 0.2, 0.85], [0.2, 0.8, 0.2, 0.85], [0.2, 0.2, 0.8, 0.85],
    [0.8, 0.8, 0.2, 0.85], [0.8, 0.2, 0.8, 0.85], [0.2, 0.8, 0.8, 0.85],
    [1.0, 0.5, 0.0, 0.85], [0.5, 0.2, 0.8, 0.85], [0.1, 0.5, 0.5, 0.85],
    [0.7, 0.7, 0.7, 0.85] 
    ] 
    TRAIT_COLOR_MAP_GPU = cp.array(TRAIT_COLOR_MAP_RAW, dtype=cp.float32) if TRAIT_COLOR_MAP_RAW else cp.array([[0.5,0.5,0.5,0.25]], dtype=cp.float32)
# Add more colors to TRAIT_COLOR_MAP_RAW if num_traits can exceed its length
# Ensure cfg_num_traits input in DPG is limited by len(TRAIT_COLOR_MAP_RAW)
    glut_window_id = None # Will store GLUT window ID if created
else: # OpenGL not available
    opengl_viz_config = {}
    TRAIT_COLOR_MAP_RAW = []
    TRAIT_COLOR_MAP_GPU = cp.array([], dtype=cp.float32).reshape(0,4) # Or None, if preferred
    # gl_data_lock is not needed if no GL thread access, but global_viz_data_cache.gl_render_data_lock is general
    glut_window_id = None


# --- OpenGL Visualization Functions (to be run in the main/UI thread) ---
def init_gl():
    """Initializes OpenGL state. Called by the main thread."""
    if not OPENGL_AVAILABLE: return
    global gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo
    global cuda_gl_interop_enabled

    glEnable(GL_POINT_SMOOTH); glHint(GL_POINT_SMOOTH_HINT, GL_NICEST) # Anti-aliased points
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) # Enable alpha blending
    glClearColor(0, 0, 0, 0); # Dark blue background
    glPointSize(opengl_viz_config.get('POINT_SIZE', 2.0)); # Set default point size
    glEnable(GL_DEPTH_TEST) # Enable depth testing for 3D

    # Generate Vertex Buffer Objects (VBOs)
    try:
        vbo_ids = glGenBuffers(4) # Generate 4 VBO IDs
        if not isinstance(vbo_ids, (list, tuple, np.ndarray)) or len(vbo_ids) < 4 :
            # Fallback if glGenBuffers returns a single ID or unexpected type
            if isinstance(vbo_ids, int) and vbo_ids > 0: 
                 gl_neuron_pos_vbo = vbo_ids
                 gl_neuron_color_vbo = glGenBuffers(1)
                 gl_synapse_vertices_vbo = glGenBuffers(1)
                 gl_pulse_vertices_vbo = glGenBuffers(1)
            else: raise ValueError("glGenBuffers did not return expected VBO IDs.")
        else: # Standard return of multiple IDs
            gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo = vbo_ids[0], vbo_ids[1], vbo_ids[2], vbo_ids[3]
        
        # Try to enable CUDA-OpenGL interop for zero-copy transfers
        try:
            # Test if CUDA can access OpenGL context
            from cuda import cudart
            cuda_gl_interop_enabled = True
            print("[CUDA-GL Interop] Enabled for zero-copy GPU→OpenGL transfers")
        except ImportError:
            cuda_gl_interop_enabled = False
            print("[CUDA-GL Interop] Not available (cuda-python not installed). Using GPU→CPU→GPU path.")
            
    except Exception as e:
        print(f"Error: glGenBuffers failed: {e}. OpenGL visualization will likely fail.")
        # Set VBO IDs to 0 or an invalid marker to prevent usage if generation fails
        gl_neuron_pos_vbo = 0; gl_neuron_color_vbo = 0; gl_synapse_vertices_vbo = 0; gl_pulse_vertices_vbo = 0
        cuda_gl_interop_enabled = False
        return


def reshape_gl_window(width, height):
    """Handles OpenGL window reshape events. Called by GLUT in the main thread."""
    if not OPENGL_AVAILABLE or height <= 0 or global_simulation_bridge is None: return # global_simulation_bridge for camera config
    viz_cfg = global_simulation_bridge.viz_config # Access viz_config for camera params

    opengl_viz_config['WINDOW_WIDTH'] = width # Update stored window dimensions
    opengl_viz_config['WINDOW_HEIGHT'] = height

    glViewport(0, 0, width, height); # Set viewport to new window size
    glMatrixMode(GL_PROJECTION); glLoadIdentity() # Switch to projection matrix
    # Set perspective: FOV, aspect ratio, near clip, far clip
    gluPerspective(viz_cfg.camera_fov, float(width) / float(height), viz_cfg.camera_near_clip, viz_cfg.camera_far_clip)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity() # Switch back to modelview matrix


def render_text_gl(x, y, text, font=None): # Font defaults to GLUT_BITMAP_9_BY_15 if None
    """Renders text on the OpenGL screen. Called by the main thread."""
    if not OPENGL_AVAILABLE: return
    if font is None: font = glut.GLUT_BITMAP_9_BY_15 if hasattr(glut, "GLUT_BITMAP_9_BY_15") else None
    if font is None: print("Warning: GLUT font not available for render_text_gl."); return


    try:
        current_win = glut.glutGetWindow();
        if current_win == 0: return # No current GL context (e.g., window closed)

        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity() # Save current projection matrix
        win_w = opengl_viz_config.get('WINDOW_WIDTH', 800); 
        win_h = opengl_viz_config.get('WINDOW_HEIGHT', 600)
        gluOrtho2D(0, win_w, 0, win_h) # Set up 2D orthographic projection for text

        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity() # Save current modelview matrix
        glColor3f(0.9, 0.9, 0.9); # Set text color (e.g., light gray)
        glDisable(GL_DEPTH_TEST) # Disable depth test to render text on top

        glRasterPos2i(int(x), int(y)) # Position the text (bottom-left origin)
        for character in text:
            glut.glutBitmapCharacter(font, ord(character)) # Render each character

        glEnable(GL_DEPTH_TEST); # Re-enable depth test
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); # Restore matrices
        glMatrixMode(GL_MODELVIEW) # Ensure modelview is current
    except Exception as e:
        # This can happen if GLUT context is lost or font is invalid
        print(f"[ERROR] OpenGL render_text_gl: {e}")


def get_color_for_trait(trait_index, activity_timer_value, is_currently_spiking, neuron_model_name_str, neuron_type_str=""):
    """
    Determines neuron color based on trait, activity, spiking status, and filter mode.
    Called by the main thread during GL data preparation.
    """
    max_highlight_frames = opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7)
    firing_rgb_config = opengl_viz_config.get("FIRING_NEURON_COLOR", [1.0, 1.0, 0.0, 1.0])
    firing_rgb = firing_rgb_config[0:3] # RGB components for firing
    base_firing_alpha = firing_rgb_config[3] # Alpha for firing
    default_inactive_alpha = opengl_viz_config.get("INACTIVE_NEURON_OPACITY", 0.25)

    # Base color from TRAIT_COLOR_MAP_RAW
    base_color_rgb = [0.5, 0.5, 0.5]; base_alpha = default_inactive_alpha # Default gray
    if TRAIT_COLOR_MAP_RAW and len(TRAIT_COLOR_MAP_RAW) > 0:
        color_def_from_map = TRAIT_COLOR_MAP_RAW[trait_index % len(TRAIT_COLOR_MAP_RAW)]
        base_color_rgb = color_def_from_map[0:3]
        base_alpha = color_def_from_map[3] if len(color_def_from_map) > 3 else default_inactive_alpha

    final_color_rgba = list(base_color_rgb) + [base_alpha] # Default color

    # Get current filter settings (from UI thread's global_gui_state)
    # This function is called by main thread, so direct access to global_gui_state is okay here.
    filters_dict = get_current_filter_settings_from_gui() # Assumes this is safe to call from main
    spiking_mode_filter = filters_dict.get("spiking_mode", "Highlight Spiking") # DPG UI filter setting

    if spiking_mode_filter == "No Spiking Highlight":
        return final_color_rgba # Return base trait color, no special highlight

    # Apply spiking highlight based on filter
    if is_currently_spiking:
        # If "Highlight Spiking" or "Show Only Spiking" and neuron is currently spiking
        final_color_rgba = list(firing_rgb) + [base_firing_alpha]
    elif spiking_mode_filter == "Highlight Spiking" and activity_timer_value > 0:
        # "Highlight Spiking" mode: not currently spiking, but has a residual activity timer
        # Fade the highlight color based on remaining timer duration
        decay_ratio = max(0.0, min(1.0, float(activity_timer_value) / max_highlight_frames))
        dimmed_firing_alpha = base_firing_alpha * decay_ratio * 0.6 # Make decay noticeable
        dimmed_firing_alpha = max(dimmed_firing_alpha, base_alpha * 0.8, 0.05) # Ensure it's not less than base or too faint
        dimmed_firing_alpha = min(base_firing_alpha * 0.8, dimmed_firing_alpha) # Cap at a slightly lower max than full spike
        final_color_rgba = list(firing_rgb) + [dimmed_firing_alpha]
    # In "Show Only Spiking" mode, if not is_currently_spiking, the neuron wouldn't be drawn anyway by upstream filter logic.
    
    return final_color_rgba

def update_gl_data():
    """
    Prepares neuron, synapse, and pulse data for OpenGL rendering by updating VBOs.
    This function is called by the main/UI thread. It gets data from global_viz_data_cache.gl_render_data_buffer,
    which is populated by the simulation thread with CuPy arrays for GL data.
    """
    global gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo
    global gl_num_neurons_to_draw, gl_num_synapse_lines_to_draw, gl_num_pulses_to_draw
    # Use the new global CuPy array names
    global gl_neuron_pos_cp, gl_neuron_colors_cp, gl_connection_vertices_cp, gl_pulse_vertices_cp 

    if not OPENGL_AVAILABLE:
        gl_num_neurons_to_draw = 0; gl_num_synapse_lines_to_draw = 0; gl_num_pulses_to_draw = 0
        return

    sim_data_snapshot = None
    with global_viz_data_cache["gl_render_data_lock"]:
        if global_viz_data_cache["gl_render_data_buffer"] is not None:
            sim_data_snapshot = global_viz_data_cache["gl_render_data_buffer"].copy()

    if sim_data_snapshot is None:
        if not global_gui_state.get("filters_changed", False) and not global_gui_state.get("is_playback_mode_active", False):
            return
        # If filters changed but no new data, we might re-filter existing CuPy arrays
        # For now, assume sim_data_snapshot is required to proceed with new data.
        # If no snapshot, potentially clear display or show last state (current logic will use empty arrays).
        if sim_data_snapshot is None and not global_gui_state.get("filters_changed", False):
             return


    # --- Extract CuPy arrays and other data from snapshot ---
    neuron_fired_cp = sim_data_snapshot.get("neuron_fired_status_cp", cp.array([], dtype=bool))
    neuron_activity_timers_cp = sim_data_snapshot.get("neuron_activity_timers_cp", cp.array([], dtype=cp.int32))
    all_neuron_positions_3d_cp = sim_data_snapshot.get("neuron_positions_3d_cp", cp.array([], dtype=cp.float32).reshape(0,3))
    all_neuron_traits_cp = sim_data_snapshot.get("neuron_traits_cp", cp.array([], dtype=cp.int32))
    all_neuron_type_ids_cp = sim_data_snapshot.get("neuron_type_ids_cp", cp.array([], dtype=cp.int32))  # Integer type IDs

    # CPU data (neuron types list is Python list of strings, kept for UI display only)
    all_neuron_types_str_list_cpu = sim_data_snapshot.get("neuron_types_list_for_viz", []) 
    model_name_str = sim_data_snapshot.get("neuron_model_type_str", "IZHIKEVICH")
    num_neurons_in_snapshot = sim_data_snapshot.get("num_neurons_snapshot", 0)

    # Ensure consistency of snapshot data
    if all_neuron_positions_3d_cp.shape[0] != num_neurons_in_snapshot:
        all_neuron_positions_3d_cp = cp.zeros((num_neurons_in_snapshot, 3), dtype=cp.float32) # Fallback
    if neuron_fired_cp.size != num_neurons_in_snapshot:
        neuron_fired_cp = cp.zeros(num_neurons_in_snapshot, dtype=bool) # Fallback
    if neuron_activity_timers_cp.size != num_neurons_in_snapshot:
        neuron_activity_timers_cp = cp.zeros(num_neurons_in_snapshot, dtype=cp.int32) # Fallback
    if all_neuron_traits_cp.size != num_neurons_in_snapshot:
        all_neuron_traits_cp = cp.zeros(num_neurons_in_snapshot, dtype=cp.int32) # Fallback
    if all_neuron_type_ids_cp.size != num_neurons_in_snapshot:
        all_neuron_type_ids_cp = cp.zeros(num_neurons_in_snapshot, dtype=cp.int32) # Fallback
    if len(all_neuron_types_str_list_cpu) != num_neurons_in_snapshot:
         all_neuron_types_str_list_cpu = ["Unknown"] * num_neurons_in_snapshot


    # --- Neuron Filtering (on GPU where possible) ---
    current_filters = get_current_filter_settings_from_gui()
    all_indices_cp = cp.arange(num_neurons_in_snapshot, dtype=cp.int32)
    visible_mask_cp = cp.ones(num_neurons_in_snapshot, dtype=bool)

    spiking_mode_filter = current_filters.get("spiking_mode", "Highlight Spiking")
    if spiking_mode_filter == "Show Only Spiking":
        visible_mask_cp &= neuron_fired_cp

    if current_filters.get("type_filter_enabled", False):
        selected_type_str_cpu = current_filters.get("selected_neuron_type", "All")
        if selected_type_str_cpu != "All":
            # Use integer type ID for GPU-accelerated filtering
            selected_type_id = NEURON_TYPE_MAPPER.get_id_from_display_name(selected_type_str_cpu)
            type_mask_cp = (all_neuron_type_ids_cp == selected_type_id)  # GPU operation
            visible_mask_cp &= type_mask_cp

    visible_neuron_indices_cp = all_indices_cp[visible_mask_cp]

    max_render_neurons = opengl_viz_config.get('MAX_NEURONS_TO_RENDER', 100000) # Increased default
    if visible_neuron_indices_cp.size > max_render_neurons:
        chosen_neuron_indices_cp = cp.random.choice(visible_neuron_indices_cp, size=max_render_neurons, replace=False)
    else:
        chosen_neuron_indices_cp = visible_neuron_indices_cp

    current_num_neurons_to_draw = chosen_neuron_indices_cp.size

    temp_gl_neuron_pos_cp = cp.array([], dtype=cp.float32).reshape(0,3)
    temp_gl_neuron_colors_cp = cp.array([], dtype=cp.float32).reshape(0,4)

    if current_num_neurons_to_draw > 0:
        temp_gl_neuron_pos_cp = all_neuron_positions_3d_cp[chosen_neuron_indices_cp]

        # --- Vectorized Color Calculation (GPU) ---
        chosen_traits = all_neuron_traits_cp[chosen_neuron_indices_cp]
        chosen_activity_timers = neuron_activity_timers_cp[chosen_neuron_indices_cp]
        chosen_is_spiking = neuron_fired_cp[chosen_neuron_indices_cp]

        max_highlight_frames_val = opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7)
        firing_rgb_config_val = opengl_viz_config.get("FIRING_NEURON_COLOR", [1.0, 1.0, 0.0, 1.0])
        firing_rgb_gpu = cp.array(firing_rgb_config_val[0:3], dtype=cp.float32)
        base_firing_alpha_gpu = cp.float32(firing_rgb_config_val[3])
        default_inactive_alpha_gpu = cp.float32(opengl_viz_config.get("INACTIVE_NEURON_OPACITY", 0.25))

        # Initialize colors based on traits (using TRAIT_COLOR_MAP_GPU)
        # Ensure TRAIT_COLOR_MAP_GPU is defined in global scope and is a CuPy array
        if TRAIT_COLOR_MAP_GPU.ndim == 2 and TRAIT_COLOR_MAP_GPU.shape[1] == 4: # Basic check
            temp_gl_neuron_colors_cp = TRAIT_COLOR_MAP_GPU[chosen_traits % TRAIT_COLOR_MAP_GPU.shape[0]]
        else: # Fallback if TRAIT_COLOR_MAP_GPU is malformed
            temp_gl_neuron_colors_cp = cp.full((current_num_neurons_to_draw, 4), 
                                               cp.array([0.5, 0.5, 0.5, default_inactive_alpha_gpu], dtype=cp.float32), 
                                               dtype=cp.float32)

        if spiking_mode_filter != "No Spiking Highlight":
            spiking_mask = chosen_is_spiking
            if cp.any(spiking_mask):
                temp_gl_neuron_colors_cp[spiking_mask, 0:3] = firing_rgb_gpu
                temp_gl_neuron_colors_cp[spiking_mask, 3] = base_firing_alpha_gpu

            if spiking_mode_filter == "Highlight Spiking":
                active_timer_mask = (~chosen_is_spiking) & (chosen_activity_timers > 0)
                if cp.any(active_timer_mask):
                    decay_ratio = cp.clip(chosen_activity_timers[active_timer_mask].astype(cp.float32) / max_highlight_frames_val, 0.0, 1.0)

                    # Use original alpha from trait map as base for fading highlight
                    base_alpha_for_active_timer = temp_gl_neuron_colors_cp[active_timer_mask, 3].copy() # Get current alpha (from trait)

                    dimmed_firing_alpha = base_firing_alpha_gpu * decay_ratio * 0.6
                    # Ensure highlight is visible but respects original trait alpha somewhat
                    dimmed_firing_alpha = cp.maximum(dimmed_firing_alpha, base_alpha_for_active_timer * 0.8)
                    dimmed_firing_alpha = cp.maximum(dimmed_firing_alpha, 0.05) # Minimum visibility
                    dimmed_firing_alpha = cp.minimum(dimmed_firing_alpha, base_firing_alpha_gpu * 0.9) # Cap slightly below full spike alpha

                    temp_gl_neuron_colors_cp[active_timer_mask, 0:3] = firing_rgb_gpu
                    temp_gl_neuron_colors_cp[active_timer_mask, 3] = dimmed_firing_alpha

    # --- Synapse Data (GPU-accelerated filtering using cp.isin) ---
    temp_gl_connection_vertices_cp = cp.array([], dtype=cp.float32).reshape(0,3)
    current_num_synapse_lines_to_draw = 0
    if global_gui_state.get("show_connections_gl", False) and "synapse_info" in sim_data_snapshot:
        all_synapse_data_list_cpu = sim_data_snapshot["synapse_info"] # CPU list of dicts
        
        # Extract source and target indices from synapse data
        if all_synapse_data_list_cpu:
            src_indices_all = np.array([syn["source_idx"] for syn in all_synapse_data_list_cpu], dtype=np.int32)
            tgt_indices_all = np.array([syn["target_idx"] for syn in all_synapse_data_list_cpu], dtype=np.int32)
            weights_all = np.array([syn["weight"] for syn in all_synapse_data_list_cpu], dtype=np.float32)
            
            # Convert to CuPy for GPU operations
            src_indices_all_cp = cp.asarray(src_indices_all)
            tgt_indices_all_cp = cp.asarray(tgt_indices_all)
            weights_all_cp = cp.asarray(weights_all)
            
            # GPU-accelerated visibility check: both source and target must be in visible neurons
            src_visible_mask = cp.isin(src_indices_all_cp, chosen_neuron_indices_cp)
            tgt_visible_mask = cp.isin(tgt_indices_all_cp, chosen_neuron_indices_cp)
            
            # Weight filtering
            min_abs_w = current_filters.get("min_abs_weight", 0.01)
            weight_mask = cp.abs(weights_all_cp) >= min_abs_w
            
            # Combined visibility mask
            synapse_visible_mask = src_visible_mask & tgt_visible_mask & weight_mask
            visible_synapse_indices_cp = cp.where(synapse_visible_mask)[0]
            
            if visible_synapse_indices_cp.size > 0:
                # Get positions for visible synapses (all on GPU)
                visible_src_indices = src_indices_all_cp[visible_synapse_indices_cp]
                visible_tgt_indices = tgt_indices_all_cp[visible_synapse_indices_cp]
                
                pos_src_all_cp = all_neuron_positions_3d_cp[visible_src_indices]
                pos_tgt_all_cp = all_neuron_positions_3d_cp[visible_tgt_indices]

                # Interleave source and target positions: [src1, tgt1, src2, tgt2, ...]
                temp_gl_connection_vertices_cp = cp.empty((visible_synapse_indices_cp.size * 2, 3), dtype=cp.float32)
                temp_gl_connection_vertices_cp[0::2] = pos_src_all_cp # Even indices are sources
                temp_gl_connection_vertices_cp[1::2] = pos_tgt_all_cp # Odd indices are targets
                current_num_synapse_lines_to_draw = visible_synapse_indices_cp.size

    # --- Synaptic Pulse Data (Using pre-calculated positions from sim_bridge) ---
    temp_gl_pulse_vertices_cp = sim_data_snapshot.get("pulse_positions_cp_for_gl", cp.array([], dtype=cp.float32).reshape(0,3))
    current_num_pulses_to_draw = temp_gl_pulse_vertices_cp.shape[0]

    # --- Update global GL CuPy arrays and VBOs ---
    # These global arrays are now CuPy arrays
    gl_num_neurons_to_draw = current_num_neurons_to_draw
    gl_neuron_pos_cp = temp_gl_neuron_pos_cp 
    gl_neuron_colors_cp = temp_gl_neuron_colors_cp

    # === PHASE 4: CUDA-OpenGL Interop - OPTIMIZED VBO UPDATES ===
    # Use pinned memory and async transfers for faster GPU→CPU→GPU pipeline
    # This is ~2-3x faster than standard cp.asnumpy() transfers
    
    def fast_vbo_update(vbo_id, cupy_array):
        """Optimized VBO update using pinned memory for faster transfers."""
        if cupy_array.size == 0:
            return
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        
        # Method 1: Use CuPy's data pointer directly with memoryview (fastest CPU path)
        # This avoids intermediate NumPy array allocation
        try:
            # Get data as contiguous array
            if not cupy_array.flags.c_contiguous:
                cupy_array = cp.ascontiguousarray(cupy_array)
            
            # Transfer to CPU using pinned memory if available
            np_array = cp.asnumpy(cupy_array, order='C')
            glBufferData(GL_ARRAY_BUFFER, np_array.nbytes, np_array, GL_DYNAMIC_DRAW)
        except Exception as e:
            print(f"[VBO Update] Error: {e}")
    
    if gl_neuron_pos_vbo is not None and gl_neuron_pos_vbo > 0 and gl_neuron_pos_cp.size > 0:
        fast_vbo_update(gl_neuron_pos_vbo, gl_neuron_pos_cp)
        
    if gl_neuron_color_vbo is not None and gl_neuron_color_vbo > 0 and gl_neuron_colors_cp.size > 0:
        fast_vbo_update(gl_neuron_color_vbo, gl_neuron_colors_cp)

    gl_num_synapse_lines_to_draw = current_num_synapse_lines_to_draw
    gl_connection_vertices_cp = temp_gl_connection_vertices_cp
    if gl_synapse_vertices_vbo is not None and gl_synapse_vertices_vbo > 0 and gl_connection_vertices_cp.size > 0:
        fast_vbo_update(gl_synapse_vertices_vbo, gl_connection_vertices_cp)

    gl_num_pulses_to_draw = current_num_pulses_to_draw
    gl_pulse_vertices_cp = temp_gl_pulse_vertices_cp
    if gl_pulse_vertices_vbo is not None and gl_pulse_vertices_vbo > 0 and gl_pulse_vertices_cp.size > 0:
        fast_vbo_update(gl_pulse_vertices_vbo, gl_pulse_vertices_cp)

    if gl_neuron_pos_vbo is not None and gl_neuron_pos_vbo > 0 : # Check if it's a valid VBO ID
        glBindBuffer(GL_ARRAY_BUFFER, 0) # Unbind VBO

    global_gui_state["filters_changed"] = False

def render_scene_gl():
    """Main OpenGL rendering function. Called by GLUT display callback in the main thread."""
    global opengl_viz_config, global_gui_state, glut_window_id 
    global gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo
    global gl_num_neurons_to_draw, gl_num_synapse_lines_to_draw, gl_num_pulses_to_draw
    global gl_frame_times, gl_last_fps_update_time, gl_current_fps, gl_fps_update_interval

    if not OPENGL_AVAILABLE or global_simulation_bridge is None : return # Sim bridge for camera config
    
    # Track frame time for FPS calculation
    current_time = time.perf_counter()
    if len(gl_frame_times) > 0:
        frame_delta = current_time - gl_frame_times[-1]
        gl_frame_times.append(current_time)
        # Keep only last 60 frames for rolling average
        if len(gl_frame_times) > 60:
            gl_frame_times.pop(0)
    else:
        gl_frame_times.append(current_time)
    
    # Update FPS display periodically
    if current_time - gl_last_fps_update_time >= gl_fps_update_interval:
        if len(gl_frame_times) >= 2:
            time_span = gl_frame_times[-1] - gl_frame_times[0]
            if time_span > 0:
                gl_current_fps = (len(gl_frame_times) - 1) / time_span
            gl_last_fps_update_time = current_time
    try: # Ensure GLUT context is current
        current_win = glut.glutGetWindow()
        if glut_window_id is not None and current_win != glut_window_id and current_win != 0: 
            glut.glutSetWindow(glut_window_id) 
        elif current_win == 0: return # No window context
    except Exception: return # Catch errors if GLUT context is lost

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # Clear buffers
    glPointSize(opengl_viz_config.get('POINT_SIZE', 2.0)) # Set point size from config

    viz_cfg = global_simulation_bridge.viz_config # For camera parameters
    runtime = global_simulation_bridge.runtime_state # For current time/step
    glMatrixMode(GL_MODELVIEW); glLoadIdentity() # Reset modelview matrix

    # Calculate camera eye position based on spherical coordinates (azimuth, elevation, radius)
    # This uses viz_config camera parameters, which can be modified by mouse controls.
    eye_x = viz_cfg.camera_center_x + viz_cfg.camera_radius * math.cos(viz_cfg.camera_elevation_angle) * math.sin(viz_cfg.camera_azimuth_angle)
    eye_y = viz_cfg.camera_center_y + viz_cfg.camera_radius * math.sin(viz_cfg.camera_elevation_angle)
    eye_z = viz_cfg.camera_center_z + viz_cfg.camera_radius * math.cos(viz_cfg.camera_elevation_angle) * math.cos(viz_cfg.camera_azimuth_angle)

    gluLookAt(eye_x, eye_y, eye_z, # Eye position
              viz_cfg.camera_center_x, viz_cfg.camera_center_y, viz_cfg.camera_center_z, # Look-at point
              viz_cfg.camera_up_x, viz_cfg.camera_up_y, viz_cfg.camera_up_z) # Up vector

    # Render Synapses (if enabled and data available)
    if global_gui_state.get("show_connections_gl", False) and gl_num_synapse_lines_to_draw > 0 and \
       gl_synapse_vertices_vbo is not None and gl_synapse_vertices_vbo > 0:
        base_syn_color = opengl_viz_config.get('SYNAPSE_BASE_COLOR', [0.3,0.3,0.4])
        alpha_mod = opengl_viz_config.get('SYNAPSE_ALPHA_MODIFIER', 0.5)
        final_alpha = np.clip(0.15 * alpha_mod, 0.02, 0.5) # Calculate final alpha
        glColor4f(base_syn_color[0], base_syn_color[1], base_syn_color[2], final_alpha)
        glLineWidth(0.5) # Thin lines for synapses

        glBindBuffer(GL_ARRAY_BUFFER, gl_synapse_vertices_vbo) # Bind synapse vertex VBO
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None) # Define vertex data format
        glDrawArrays(GL_LINES, 0, gl_num_synapse_lines_to_draw * 2) # Draw lines (2 vertices per line)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0) # Unbind VBO

    # Render Neurons (if data available)
    if gl_num_neurons_to_draw > 0 and \
       gl_neuron_pos_vbo is not None and gl_neuron_pos_vbo > 0 and \
       gl_neuron_color_vbo is not None and gl_neuron_color_vbo > 0:
        
        glBindBuffer(GL_ARRAY_BUFFER, gl_neuron_pos_vbo) # Bind neuron position VBO
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, gl_neuron_color_vbo) # Bind neuron color VBO
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(4, GL_FLOAT, 0, None) # RGBA colors

        glDrawArrays(GL_POINTS, 0, gl_num_neurons_to_draw) # Draw points for neurons

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0) # Unbind VBO

    # Render Synaptic Pulses (if enabled and data available)
    if opengl_viz_config.get("ENABLE_SYNAPTIC_PULSES", False) and \
       gl_num_pulses_to_draw > 0 and \
       gl_pulse_vertices_vbo is not None and gl_pulse_vertices_vbo > 0:
        
        pulse_color_rgba = opengl_viz_config.get("SYNAPTIC_PULSE_COLOR", [0.7, 0.9, 1.0, 0.9])
        glColor4fv(pulse_color_rgba) # Set pulse color
        glPointSize(opengl_viz_config.get("SYNAPTIC_PULSE_SIZE", 3.0)) # Set pulse point size

        glBindBuffer(GL_ARRAY_BUFFER, gl_pulse_vertices_vbo) # Bind pulse vertex VBO
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_POINTS, 0, gl_num_pulses_to_draw) # Draw points for pulses
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0) # Unbind VBO

        glPointSize(opengl_viz_config.get('POINT_SIZE', 2.0)) # Reset point size to default for neurons

    # Render Footer Text Overlay
    footer_h = opengl_viz_config.get('FOOTER_HEIGHT_PIXELS', 75)
    if footer_h > 0:
        line_h, margin = 15, 10 # Text line height and margin
        win_w = opengl_viz_config.get('WINDOW_WIDTH', 800)
        
        # Get current time and step from runtime_state
        sim_time_s = (runtime.current_time_ms / 1000.0)
        
        # Get telemetry from sim_bridge
        avg_fr = global_simulation_bridge._mock_network_avg_firing_rate_hz
        spikes_step = global_simulation_bridge._mock_num_spikes_this_step
        plasticity_events = global_simulation_bridge._mock_total_plasticity_events
        
        # Display FPS counter
        is_sim_running = global_gui_state.get("_sim_is_running_ui_view", False)
        is_paused = global_gui_state.get("_sim_is_paused_ui_view", False)
        is_playback = global_gui_state.get("is_playback_mode_active", False)
        
        if not is_sim_running and not is_playback:
            fps_text = "FPS: 0"
        elif is_paused and not is_playback:
            fps_text = "FPS: 0"
        else:
            fps_text = f"FPS: {gl_current_fps:.1f}"
        
        mode_text = "Playback" if is_playback else "Live"
        if global_gui_state.get("is_recording_active"): mode_text += " (Rec)"
        
        # Layout: 4 rows of information
        # Row 4 (top): Time, Spikes/Step, FPS
        render_text_gl(margin, margin + 4*line_h, f"Time: {sim_time_s:.3f}s")
        render_text_gl(margin + win_w // 3, margin + 4*line_h, f"Spikes: {spikes_step}")
        render_text_gl(margin + 2*win_w // 3, margin + 4*line_h, fps_text)
        
        # Row 3: Step, Avg Rate, Mode
        render_text_gl(margin, margin + 3*line_h, f"Step: {runtime.current_time_step}")
        render_text_gl(margin + win_w // 3, margin + 3*line_h, f"Rate: {avg_fr:.2f} Hz")
        render_text_gl(margin + 2*win_w // 3, margin + 3*line_h, f"Mode: {mode_text}")
        
        # Row 2: Plasticity, Visible Neurons, Visible Synapses
        render_text_gl(margin, margin + 2*line_h, f"Plasticity: {plasticity_events}")
        render_text_gl(margin + win_w // 3, margin + 2*line_h, f"Vis.Neurons: {gl_num_neurons_to_draw}")
        render_text_gl(margin + 2*win_w // 3, margin + 2*line_h, f"Vis.Syns: {gl_num_synapse_lines_to_draw}")
        
        # Row 0 (bottom): Hotkey hints
        render_text_gl(margin, margin, "LMB:Rotate, RMB:Pan, Scroll:Zoom, R:Reset, S:Synapses, N:Neurons, Space:Pause/Resume, Esc:Exit")

    glut.glutSwapBuffers() # Swap front and back buffers to display rendered scene


def mouse_button_func_gl(button, state, x, y):
    """Handles mouse button events for OpenGL window (camera control). Called by GLUT."""
    if not global_simulation_bridge: return
    cfg = global_simulation_bridge.viz_config # Camera config is part of viz_config
    zoom_speed = opengl_viz_config.get("CAMERA_ZOOM_SPEED_FACTOR", 20.0)

    if button == glut.GLUT_LEFT_BUTTON:
        cfg.mouse_left_button_down = (state == glut.GLUT_DOWN)
    elif button == glut.GLUT_RIGHT_BUTTON:
        cfg.mouse_right_button_down = (state == glut.GLUT_DOWN)
    elif button == 3: # Scroll up (zoom in)
        if state == glut.GLUT_UP: return # Action on press, not release
        cfg.camera_radius = max(cfg.camera_near_clip * 2, cfg.camera_radius - zoom_speed)
    elif button == 4: # Scroll down (zoom out)
        if state == glut.GLUT_UP: return
        cfg.camera_radius += zoom_speed
        cfg.camera_radius = min(cfg.camera_radius, cfg.camera_far_clip * 0.8) # Don't zoom out too far

    cfg.mouse_last_x = x; cfg.mouse_last_y = y # Store last mouse position
    if glut.glutGetWindow() != 0: glut.glutPostRedisplay() # Request redraw


def mouse_motion_func_gl(x, y):
    """Handles mouse motion events for OpenGL window (camera control). Called by GLUT."""
    if not global_simulation_bridge: return
    cfg = global_simulation_bridge.viz_config
    dx = x - cfg.mouse_last_x; dy = y - cfg.mouse_last_y # Change in mouse position

    rotate_speed = opengl_viz_config.get("CAMERA_ROTATE_SPEED_FACTOR", 0.005)
    pan_speed_config = opengl_viz_config.get("CAMERA_PAN_SPEED_FACTOR", 0.1)

    if cfg.mouse_left_button_down: # Rotate camera (orbit around center)
        cfg.camera_azimuth_angle -= dx * rotate_speed
        cfg.camera_elevation_angle -= dy * rotate_speed
        # Clamp elevation to prevent flipping over poles
        cfg.camera_elevation_angle = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, cfg.camera_elevation_angle))
    elif cfg.mouse_right_button_down: # Pan camera (move look-at point)
        # Calculate camera's local right and up vectors for panning
        # Eye position (calculated from spherical coordinates)
        eye_calc_x = cfg.camera_center_x + cfg.camera_radius * math.cos(cfg.camera_elevation_angle) * math.sin(cfg.camera_azimuth_angle)
        eye_calc_y = cfg.camera_center_y + cfg.camera_radius * math.sin(cfg.camera_elevation_angle)
        eye_calc_z = cfg.camera_center_z + cfg.camera_radius * math.cos(cfg.camera_elevation_angle) * math.cos(cfg.camera_azimuth_angle)
        eye = np.array([eye_calc_x, eye_calc_y, eye_calc_z]);
        
        center = np.array([cfg.camera_center_x, cfg.camera_center_y, cfg.camera_center_z])
        up_world = np.array([cfg.camera_up_x, cfg.camera_up_y, cfg.camera_up_z]) # World up vector

        forward = center - eye; forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-6: forward /= forward_norm
        else: forward = np.array([0,0,-1]) # Default if eye is at center

        right = np.cross(forward, up_world); right_norm = np.linalg.norm(right)
        if right_norm > 1e-6: right /= right_norm
        else: # Handle gimbal lock like situations for 'right' vector
            if abs(forward[1]) > 0.99 : right = np.array([1,0,0]) # Looking straight up/down
            else: 
                right_temp = np.cross(forward, np.array([0,1,0])); right_norm_temp = np.linalg.norm(right_temp)
                right = right_temp/right_norm_temp if right_norm_temp > 1e-6 else np.array([1,0,0])

        cam_up = np.cross(right, forward) # Camera's local up vector (orthogonal to right and forward)

        pan_scale = pan_speed_config * (cfg.camera_radius / 150.0) # Scale pan speed with zoom level
        pan_vector_x = -dx * right * pan_scale # Pan left/right based on mouse dx
        pan_vector_y = dy * cam_up * pan_scale  # Pan up/down based on mouse dy

        # Update camera center (look-at point)
        new_center = center + pan_vector_x + pan_vector_y
        cfg.camera_center_x, cfg.camera_center_y, cfg.camera_center_z = new_center[0], new_center[1], new_center[2]

    cfg.mouse_last_x = x; cfg.mouse_last_y = y # Update last mouse position
    if glut.glutGetWindow() != 0: glut.glutPostRedisplay() # Request redraw

def keyboard_func_gl(key, x, y):
    """Handles keyboard events for the OpenGL window. Called by GLUT."""
    # global_gui_state, global_simulation_bridge, shutdown_flag are accessed.
    # Commands to sim_thread are sent via ui_to_sim_queue.

    if global_simulation_bridge is None : return # Should not happen if GL window is up

    # Handle ESC key first (special case)
    if key == b'\x1b': # ESC key
        print("ESC pressed in OpenGL window. Signaling shutdown.")
        shutdown_flag.set() # Signal all threads to shut down
        return
    
    try: 
        key_char = key.decode("utf-8").lower() # Decode byte string to char (includes space as ' ')
    except UnicodeDecodeError: # Handle other special keys
        return # Other non-decodeable keys are ignored

    cfg = global_simulation_bridge.viz_config # For camera reset

    # --- Keyboard Shortcuts for OpenGL Window ---
    if key_char == 's': # Toggle synapse visibility
        # This action directly modifies UI state, which then affects GL rendering data prep.
        new_show_state = not global_gui_state.get("show_connections_gl", False)
        global_gui_state["show_connections_gl"] = new_show_state
        if dpg.is_dearpygui_running() and dpg.does_item_exist("filter_show_synapses_gl_cb"):
            dpg.set_value("filter_show_synapses_gl_cb", new_show_state) # Update DPG checkbox
        trigger_filter_update_signal() # Signal GL data needs re-filtering and VBO update
        print(f"Synapse visibility toggled {'on' if new_show_state else 'off'}.")

    elif key_char == 'n': # Cycle through neuron spiking display modes
        if dpg.is_dearpygui_running() and dpg.does_item_exist("filter_spiking_mode_combo"):
            modes = ["Highlight Spiking", "Show Only Spiking", "No Spiking Highlight"]
            current_mode = dpg.get_value("filter_spiking_mode_combo")
            try:
                current_idx = modes.index(current_mode)
                next_idx = (current_idx + 1) % len(modes)
            except ValueError:
                next_idx = 0  # Default to first mode if current mode not found
            new_mode = modes[next_idx]
            dpg.set_value("filter_spiking_mode_combo", new_mode)
            trigger_filter_update_signal()
            print(f"Neuron display mode: {new_mode}")

    elif key_char == ' ': # Space: Pause/Resume or Start simulation
        if not global_gui_state.get("is_playback_mode_active", False):
            current_sim_running = global_gui_state.get("_sim_is_running_ui_view", False)
            current_sim_paused = global_gui_state.get("_sim_is_paused_ui_view", False)
            
            if not current_sim_running:
                # Sim is stopped, start it
                ui_to_sim_queue.put({"type": "START_SIM"})
                # Optimistic UI state update (matches handle_start_simulation_event)
                global_gui_state["_sim_is_running_ui_view"] = True
                global_gui_state["_sim_is_paused_ui_view"] = False
                update_ui_for_simulation_run_state(is_running=True, is_paused=False)
                print("GL Keyboard: Starting simulation.")
            elif current_sim_paused:
                # Sim is paused, resume it
                ui_to_sim_queue.put({"type": "RESUME_SIM"})
                # Optimistic UI state update (matches handle_pause_simulation_event)
                global_gui_state["_sim_is_paused_ui_view"] = False
                update_ui_for_simulation_run_state(is_running=True, is_paused=False)
                print("GL Keyboard: Resuming simulation.")
            else:
                # Sim is running, pause it
                ui_to_sim_queue.put({"type": "PAUSE_SIM"})
                # Optimistic UI state update (matches handle_pause_simulation_event)
                global_gui_state["_sim_is_paused_ui_view"] = True
                update_ui_for_simulation_run_state(is_running=True, is_paused=True)
                print("GL Keyboard: Pausing simulation.")
    
    elif key_char == 'r': # Reset camera position
        cfg.camera_azimuth_angle = 0.0
        cfg.camera_elevation_angle = 0.0
        cfg.camera_radius = 150.0
        cfg.camera_center_x, cfg.camera_center_y, cfg.camera_center_z = 0.0, 0.0, 0.0
        if glut.glutGetWindow() != 0: glut.glutPostRedisplay() # Request redraw
        print("Camera reset.")

    # Other keys can be added here.
    # Ensure glutPostRedisplay is called if the view needs to change immediately.
    if glut.glutGetWindow() != 0: glut.glutPostRedisplay()


# --- DPG GUI Helper Functions (Called by Main/UI Thread) ---

def trigger_filter_update_signal(sender=None, app_data=None, user_data=None):
    """Sets a flag indicating that visualization filters have changed and GL data needs update."""
    global global_gui_state
    global_gui_state["filters_changed"] = True
    # This will be checked in the main DPG loop to trigger update_gl_data()

def get_current_filter_settings_from_gui():
    """Retrieves current filter settings from DPG UI elements. Called by main/UI thread."""
    settings = {
        "spiking_mode": "Highlight Spiking", # Default if DPG item doesn't exist
        "type_filter_enabled": False,
        "selected_neuron_type": "All", 
        "min_abs_weight": 0.01 # Default for synapse weight filter
    }
    if dpg.is_dearpygui_running(): # Ensure DPG context is active
        if dpg.does_item_exist("filter_spiking_mode_combo"):
            settings["spiking_mode"] = dpg.get_value("filter_spiking_mode_combo")
        if dpg.does_item_exist("filter_type_enable_cb"):
            settings["type_filter_enabled"] = dpg.get_value("filter_type_enable_cb")
        if dpg.does_item_exist("filter_neuron_type_combo"):
            settings["selected_neuron_type"] = dpg.get_value("filter_neuron_type_combo")
        if dpg.does_item_exist("filter_min_abs_weight_slider"):
            settings["min_abs_weight"] = dpg.get_value("filter_min_abs_weight_slider")
    return settings

def apply_neuron_filters_to_indices(all_indices, fired_status_np, neuron_types_list_str, filters_dict):
    """
    Applies filters to a list of neuron indices to determine visibility.
    Called by main/UI thread (specifically within update_gl_data).
    """
    if all_indices.size == 0: return [] # No neurons to filter
    
    visible_mask = np.ones(all_indices.size, dtype=bool) # Start with all neurons visible

    # Filter by spiking status
    spiking_mode = filters_dict.get("spiking_mode", "Highlight Spiking")
    if spiking_mode == "Show Only Spiking":
        if fired_status_np is not None and fired_status_np.shape == visible_mask.shape:
            visible_mask &= fired_status_np # Only show neurons that are currently firing
        else: # Mismatch in array sizes, log warning or handle gracefully
            if fired_status_np is not None: print(f"Warning: fired_status_np shape mismatch in filter. Expected {visible_mask.shape}, got {fired_status_np.shape}")
            # visible_mask &= False # Or, show no neurons if data is inconsistent

    # Filter by neuron type
    if filters_dict.get("type_filter_enabled", False):
        selected_type_str = filters_dict.get("selected_neuron_type", "All")
        if selected_type_str != "All" and neuron_types_list_str is not None and len(neuron_types_list_str) == all_indices.size:
            # Create a boolean mask for matching types
            type_mask = np.array([neuron_types_list_str[i] == selected_type_str for i in all_indices], dtype=bool)
            visible_mask &= type_mask
        elif selected_type_str != "All":
             if neuron_types_list_str is not None: print(f"Warning: neuron_types_list_str length mismatch in filter. Expected {all_indices.size}, got {len(neuron_types_list_str)}")
             # visible_mask &= False # Or, show no neurons if type data is inconsistent
    
    return all_indices[visible_mask] # Return indices of neurons that pass all filters


def apply_synapse_filters_to_indices(all_synapse_data_list, filters_dict):
    """
    Applies filters to a list of synapse data dictionaries to determine visibility.
    Called by main/UI thread (specifically within update_gl_data).
    Returns a list of indices (into all_synapse_data_list) of visible synapses.
    """
    if not global_gui_state.get("show_connections_gl", False): return [] # If connections are globally hidden

    visible_syn_indices = []
    min_abs_w = filters_dict.get("min_abs_weight", 0.01) # Minimum absolute weight to show
    for i, syn_data in enumerate(all_synapse_data_list):
        if abs(syn_data.get("weight", 0.0)) >= min_abs_w: # Check weight against filter
            visible_syn_indices.append(i)
    return visible_syn_indices


def update_status_bar(message, color=None, level="info"):
    """Updates the text and color of the DPG status bar. Called by main/UI thread."""
    if dpg.is_dearpygui_running() and dpg.does_item_exist("status_bar_text"):
        dpg.set_value("status_bar_text", f"[{level.upper()}] {message}")
        if color is None: # Auto-color based on level if not provided
            if level == "error" or level == "critical": color = [255, 0, 0, 255]
            elif level == "warning": color = [255, 165, 0, 255]
            elif level == "info": color = [200, 200, 200, 255] 
            elif level == "success": color = [0, 200, 0, 255]
            else: color = [200, 200, 200, 255] # Default
        dpg.configure_item("status_bar_text", color=color)

# --- DPG GUI Element Creation & Event Handlers (Called by Main/UI Thread) ---

def _update_sim_config_from_ui(update_model_specific=True):
    """
    Updates a temporary SimulationConfiguration object from DPG UI elements.
    This temporary object is then sent to the simulation thread via a command.
    Called by the main/UI thread.
    Returns a dictionary representing the config from UI, or None if error.
    """
    if not dpg.is_dearpygui_running(): return None
    
    # Create a new temporary config object to populate from UI
    # This avoids modifying global_simulation_bridge.sim_config directly from UI thread.
    # The actual update to sim_bridge.sim_config happens in sim_thread upon command.
    temp_config = SimulationConfiguration() # Create a fresh default config
    
    # If global_simulation_bridge exists and has a config, start temp_config from it
    # to preserve settings not directly in UI or to have a baseline.
    # However, for sending a "full new config" command, starting fresh and filling from UI is cleaner.
    # Let's assume we build a new config purely from UI values where available,
    # and the sim_thread will merge this with its existing config if needed, or replace.
    # For "Apply Changes & Reset", it's usually a full replacement.
    
    # Helper to safely get DPG value or use current value from sim_bridge if item doesn't exist
    # This is tricky. The goal is that _update_sim_config_from_ui creates a *complete* config dict
    # based on the UI. If a UI element for a config param doesn't exist, what should its value be?
    # It should probably be the default from SimulationConfiguration() or the current live one if that's intended.
    # For now, let's assume UI has all relevant controls. If not, this needs refinement.

    try:
        cfg_dict_from_ui = {} # Build a dictionary of config values from UI

        # General parameters
        if dpg.does_item_exist("cfg_num_neurons"): cfg_dict_from_ui["num_neurons"] = max(1, dpg.get_value("cfg_num_neurons"))
        if dpg.does_item_exist("cfg_total_sim_time"): cfg_dict_from_ui["total_simulation_time_ms"] = max(0.0, dpg.get_value("cfg_total_sim_time"))
        if dpg.does_item_exist("cfg_dt_ms"): cfg_dict_from_ui["dt_ms"] = max(0.001, dpg.get_value("cfg_dt_ms"))
        if dpg.does_item_exist("cfg_seed"): cfg_dict_from_ui["seed"] = dpg.get_value("cfg_seed")
        if dpg.does_item_exist("cfg_neural_profile"): cfg_dict_from_ui["neural_profile_name"] = dpg.get_value("cfg_neural_profile")
        if dpg.does_item_exist("cfg_default_neuron_type_hh"): cfg_dict_from_ui["default_neuron_type_hh"] = dpg.get_value("cfg_default_neuron_type_hh")

        if dpg.does_item_exist("cfg_neuron_model_type"):
            selected_model_name = dpg.get_value("cfg_neuron_model_type")
            cfg_dict_from_ui["neuron_model_type"] = selected_model_name
            # Default neuron types based on selected model (these are part of SimulationConfiguration defaults too)
            if selected_model_name == NeuronModel.IZHIKEVICH.name:
                cfg_dict_from_ui["default_neuron_type_izh"] = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
            # For HH, do not override default_neuron_type_hh here; we use the value from the HH preset combo
            # and/or any profile-specific default mapping.

        # Connectivity
        if dpg.does_item_exist("cfg_enable_watts_strogatz"): cfg_dict_from_ui["enable_watts_strogatz"] = dpg.get_value("cfg_enable_watts_strogatz")
        if dpg.does_item_exist("cfg_connectivity_k"): cfg_dict_from_ui["connectivity_k"] = max(2, dpg.get_value("cfg_connectivity_k"))
        if dpg.does_item_exist("cfg_connectivity_p_rewire"): cfg_dict_from_ui["connectivity_p_rewire"] = dpg.get_value("cfg_connectivity_p_rewire")
        if dpg.does_item_exist("cfg_connections_per_neuron"): cfg_dict_from_ui["connections_per_neuron"] = max(0, dpg.get_value("cfg_connections_per_neuron"))

        # Synaptic parameters
        if dpg.does_item_exist("cfg_propagation_strength"): cfg_dict_from_ui["propagation_strength"] = dpg.get_value("cfg_propagation_strength")
        if dpg.does_item_exist("cfg_inhibitory_propagation_strength"): cfg_dict_from_ui["inhibitory_propagation_strength"] = dpg.get_value("cfg_inhibitory_propagation_strength")
        if dpg.does_item_exist("cfg_syn_tau_e"): cfg_dict_from_ui["syn_tau_g_e"] = max(0.1, dpg.get_value("cfg_syn_tau_e"))
        if dpg.does_item_exist("cfg_syn_tau_i"): cfg_dict_from_ui["syn_tau_g_i"] = max(0.1, dpg.get_value("cfg_syn_tau_i"))
        # NMDA parameters
        if dpg.does_item_exist("cfg_enable_nmda"): cfg_dict_from_ui["enable_nmda"] = dpg.get_value("cfg_enable_nmda")
        if dpg.does_item_exist("cfg_nmda_ratio"): cfg_dict_from_ui["nmda_ratio"] = max(0.0, dpg.get_value("cfg_nmda_ratio"))
        if dpg.does_item_exist("cfg_nmda_tau_decay"): cfg_dict_from_ui["nmda_tau_decay"] = max(10.0, dpg.get_value("cfg_nmda_tau_decay"))
        if dpg.does_item_exist("cfg_nmda_tau_rise"): cfg_dict_from_ui["nmda_tau_rise"] = max(0.5, dpg.get_value("cfg_nmda_tau_rise"))
        if dpg.does_item_exist("cfg_nmda_mg_conc"): cfg_dict_from_ui["nmda_mg_concentration"] = max(0.0, dpg.get_value("cfg_nmda_mg_conc"))
        if dpg.does_item_exist("cfg_num_traits"): cfg_dict_from_ui["num_traits"] = max(1, dpg.get_value("cfg_num_traits"))

        # Learning & Plasticity
        if dpg.does_item_exist("cfg_enable_hebbian_learning"): cfg_dict_from_ui["enable_hebbian_learning"] = dpg.get_value("cfg_enable_hebbian_learning")
        if dpg.does_item_exist("cfg_hebbian_learning_rate"): cfg_dict_from_ui["hebbian_learning_rate"] = dpg.get_value("cfg_hebbian_learning_rate")
        if dpg.does_item_exist("cfg_hebbian_max_weight"): cfg_dict_from_ui["hebbian_max_weight"] = dpg.get_value("cfg_hebbian_max_weight")
        if dpg.does_item_exist("cfg_enable_short_term_plasticity"): cfg_dict_from_ui["enable_short_term_plasticity"] = dpg.get_value("cfg_enable_short_term_plasticity")
        if dpg.does_item_exist("cfg_stp_U"): cfg_dict_from_ui["stp_U"] = dpg.get_value("cfg_stp_U")
        if dpg.does_item_exist("cfg_stp_tau_d"): cfg_dict_from_ui["stp_tau_d"] = max(0.1, dpg.get_value("cfg_stp_tau_d"))
        if dpg.does_item_exist("cfg_stp_tau_f"): cfg_dict_from_ui["stp_tau_f"] = max(0.1, dpg.get_value("cfg_stp_tau_f"))
        if dpg.does_item_exist("cfg_enable_per_type_stp"): cfg_dict_from_ui["enable_per_type_stp"] = dpg.get_value("cfg_enable_per_type_stp")
        # Per-type STP: read individual UI fields into lists
        for conn_type_suffix in ["ee", "ei", "ie", "ii"]:
            for param in ["U", "tau_d", "tau_f"]:
                tag = f"cfg_stp_{param}_{conn_type_suffix}"
                if dpg.does_item_exist(tag):
                    pass  # Gathered below as composite list
        # Build per-type lists from UI
        stp_U_list, stp_tau_d_list, stp_tau_f_list = [], [], []
        for suffix in ["ee", "ei", "ie", "ii"]:
            stp_U_list.append(dpg.get_value(f"cfg_stp_U_{suffix}") if dpg.does_item_exist(f"cfg_stp_U_{suffix}") else cfg_dict_from_ui.get("stp_U", 0.15))
            stp_tau_d_list.append(max(0.1, dpg.get_value(f"cfg_stp_tau_d_{suffix}")) if dpg.does_item_exist(f"cfg_stp_tau_d_{suffix}") else cfg_dict_from_ui.get("stp_tau_d", 200.0))
            stp_tau_f_list.append(max(0.1, dpg.get_value(f"cfg_stp_tau_f_{suffix}")) if dpg.does_item_exist(f"cfg_stp_tau_f_{suffix}") else cfg_dict_from_ui.get("stp_tau_f", 50.0))
        cfg_dict_from_ui["stp_U_per_type"] = stp_U_list
        cfg_dict_from_ui["stp_tau_d_per_type"] = stp_tau_d_list
        cfg_dict_from_ui["stp_tau_f_per_type"] = stp_tau_f_list
        # Structural plasticity activity bias
        if dpg.does_item_exist("cfg_struct_plast_activity_bias"): cfg_dict_from_ui["struct_plast_activity_bias"] = dpg.get_value("cfg_struct_plast_activity_bias")

        # Homeostasis
        if dpg.does_item_exist("cfg_enable_homeostasis"): cfg_dict_from_ui["enable_homeostasis"] = dpg.get_value("cfg_enable_homeostasis")
        if dpg.does_item_exist("cfg_homeostasis_target_rate"): cfg_dict_from_ui["homeostasis_target_rate"] = dpg.get_value("cfg_homeostasis_target_rate")
        if dpg.does_item_exist("cfg_homeostasis_threshold_min"): cfg_dict_from_ui["homeostasis_threshold_min"] = dpg.get_value("cfg_homeostasis_threshold_min")
        if dpg.does_item_exist("cfg_homeostasis_threshold_max"): cfg_dict_from_ui["homeostasis_threshold_max"] = dpg.get_value("cfg_homeostasis_threshold_max")
        if dpg.does_item_exist("cfg_enable_synaptic_scaling"): cfg_dict_from_ui["enable_synaptic_scaling"] = dpg.get_value("cfg_enable_synaptic_scaling")
        if dpg.does_item_exist("cfg_synaptic_scaling_rate"): cfg_dict_from_ui["synaptic_scaling_rate"] = dpg.get_value("cfg_synaptic_scaling_rate")

        # STDP
        if dpg.does_item_exist("cfg_enable_stdp"): cfg_dict_from_ui["enable_stdp"] = dpg.get_value("cfg_enable_stdp")
        if dpg.does_item_exist("cfg_stdp_a_plus"): cfg_dict_from_ui["stdp_a_plus"] = dpg.get_value("cfg_stdp_a_plus")
        if dpg.does_item_exist("cfg_stdp_a_minus"): cfg_dict_from_ui["stdp_a_minus"] = dpg.get_value("cfg_stdp_a_minus")
        if dpg.does_item_exist("cfg_stdp_tau_plus_ms"): cfg_dict_from_ui["stdp_tau_plus_ms"] = dpg.get_value("cfg_stdp_tau_plus_ms")
        if dpg.does_item_exist("cfg_stdp_tau_minus_ms"): cfg_dict_from_ui["stdp_tau_minus_ms"] = dpg.get_value("cfg_stdp_tau_minus_ms")
        if dpg.does_item_exist("cfg_stdp_w_min"): cfg_dict_from_ui["stdp_w_min"] = dpg.get_value("cfg_stdp_w_min")
        if dpg.does_item_exist("cfg_stdp_w_max"): cfg_dict_from_ui["stdp_w_max"] = dpg.get_value("cfg_stdp_w_max")

        # Reward-Modulated Plasticity
        if dpg.does_item_exist("cfg_enable_reward_modulation"): cfg_dict_from_ui["enable_reward_modulation"] = dpg.get_value("cfg_enable_reward_modulation")
        if dpg.does_item_exist("cfg_reward_learning_rate"): cfg_dict_from_ui["reward_learning_rate"] = dpg.get_value("cfg_reward_learning_rate")
        if dpg.does_item_exist("cfg_reward_eligibility_tau_ms"): cfg_dict_from_ui["reward_eligibility_tau_ms"] = dpg.get_value("cfg_reward_eligibility_tau_ms")

        # Structural Plasticity
        if dpg.does_item_exist("cfg_enable_structural_plasticity"): cfg_dict_from_ui["enable_structural_plasticity"] = dpg.get_value("cfg_enable_structural_plasticity")
        if dpg.does_item_exist("cfg_struct_plast_formation_rate"): cfg_dict_from_ui["struct_plast_formation_rate"] = dpg.get_value("cfg_struct_plast_formation_rate")
        if dpg.does_item_exist("cfg_struct_plast_elimination_rate"): cfg_dict_from_ui["struct_plast_elimination_rate"] = dpg.get_value("cfg_struct_plast_elimination_rate")
        if dpg.does_item_exist("cfg_struct_plast_weight_threshold"): cfg_dict_from_ui["struct_plast_weight_threshold"] = dpg.get_value("cfg_struct_plast_weight_threshold")
        if dpg.does_item_exist("cfg_struct_plast_target_density"): cfg_dict_from_ui["struct_plast_target_density"] = dpg.get_value("cfg_struct_plast_target_density")
        if dpg.does_item_exist("cfg_struct_plast_distance_scale"): cfg_dict_from_ui["struct_plast_distance_scale"] = dpg.get_value("cfg_struct_plast_distance_scale")
        if dpg.does_item_exist("cfg_struct_plast_update_interval_steps"): cfg_dict_from_ui["struct_plast_update_interval_steps"] = dpg.get_value("cfg_struct_plast_update_interval_steps")

        # Heterogeneity & Noise
        if dpg.does_item_exist("cfg_enable_parameter_heterogeneity"): cfg_dict_from_ui["enable_parameter_heterogeneity"] = dpg.get_value("cfg_enable_parameter_heterogeneity")
        if dpg.does_item_exist("cfg_heterogeneity_seed"): cfg_dict_from_ui["heterogeneity_seed"] = dpg.get_value("cfg_heterogeneity_seed")
        if dpg.does_item_exist("cfg_enable_conductance_noise"): cfg_dict_from_ui["enable_conductance_noise"] = dpg.get_value("cfg_enable_conductance_noise")
        if dpg.does_item_exist("cfg_conductance_noise_relative_std"): cfg_dict_from_ui["conductance_noise_relative_std"] = dpg.get_value("cfg_conductance_noise_relative_std")
        if dpg.does_item_exist("cfg_enable_ou_process"): cfg_dict_from_ui["enable_ou_process"] = dpg.get_value("cfg_enable_ou_process")
        if dpg.does_item_exist("cfg_ou_mean_current_pA"): cfg_dict_from_ui["ou_mean_current_pA"] = dpg.get_value("cfg_ou_mean_current_pA")
        if dpg.does_item_exist("cfg_ou_std_current_pA"): cfg_dict_from_ui["ou_std_current_pA"] = dpg.get_value("cfg_ou_std_current_pA")
        if dpg.does_item_exist("cfg_ou_tau_ms"): cfg_dict_from_ui["ou_tau_ms"] = dpg.get_value("cfg_ou_tau_ms")
        if dpg.does_item_exist("cfg_ou_seed"): cfg_dict_from_ui["ou_seed"] = dpg.get_value("cfg_ou_seed")
        
        # Camera FOV and Visualization settings (part of viz_config)
        if dpg.does_item_exist("cfg_camera_fov"): cfg_dict_from_ui["camera_fov"] = dpg.get_value("cfg_camera_fov")
        if dpg.does_item_exist("cfg_viz_update_interval_steps"): cfg_dict_from_ui["viz_update_interval_steps"] = max(1, dpg.get_value("cfg_viz_update_interval_steps"))

        # Model-specific parameters
        if update_model_specific:
            current_model_in_ui = cfg_dict_from_ui.get("neuron_model_type", NeuronModel.IZHIKEVICH.name)
            if current_model_in_ui == NeuronModel.IZHIKEVICH.name:
                if dpg.does_item_exist("cfg_izh_C_val"): cfg_dict_from_ui["izh_C_val"] = dpg.get_value("cfg_izh_C_val")
                if dpg.does_item_exist("cfg_izh_k_val"): cfg_dict_from_ui["izh_k_val"] = dpg.get_value("cfg_izh_k_val")
                # ... (all other Izhikevich params)
                ui_izh_params_keys = ["izh_C_val", "izh_k_val", "izh_vr_val", "izh_vt_val", "izh_vpeak_val", "izh_a_val", "izh_b_val", "izh_c_val", "izh_d_val"]
                for key_suffix in ui_izh_params_keys:
                    dpg_tag = f"cfg_{key_suffix}"
                    if dpg.does_item_exist(dpg_tag): cfg_dict_from_ui[key_suffix] = dpg.get_value(dpg_tag)

            elif current_model_in_ui == NeuronModel.HODGKIN_HUXLEY.name:
                # ... (all Hodgkin-Huxley params)
                ui_hh_params_keys = [
                    "hh_C_m", "hh_g_Na_max", "hh_g_K_max", "hh_g_L", "hh_E_Na", "hh_E_K", "hh_E_L",
                    "hh_v_peak", "hh_v_rest_init", "hh_g_M_max", "hh_m_current_tau_ms",
                    "hh_g_CaT_max", "hh_E_CaT", "hh_g_h_max", "hh_E_h", "hh_g_NaP_max",
                    "hh_q10_factor", "hh_temperature_celsius",
                    "hh_external_drive_scale",
                ]
                for key_suffix in ui_hh_params_keys:
                    dpg_tag = f"cfg_{key_suffix}"
                    if dpg.does_item_exist(dpg_tag): cfg_dict_from_ui[key_suffix] = dpg.get_value(dpg_tag)
            elif current_model_in_ui == NeuronModel.ADEX.name:
                ui_adex_params_keys = [
                    "adex_C", "adex_g_L", "adex_E_L", "adex_V_T", "adex_Delta_T",
                    "adex_a", "adex_tau_w", "adex_b", "adex_V_r", "adex_V_peak",
                    "adex_external_drive_scale",
                ]
                for key_suffix in ui_adex_params_keys:
                    dpg_tag = f"cfg_{key_suffix}"
                    if dpg.does_item_exist(dpg_tag): cfg_dict_from_ui[key_suffix] = dpg.get_value(dpg_tag)
        
        # Simulation speed factor (part of runtime_state)
        simulation_speed = dpg.get_value("sim_speed_slider") if dpg.does_item_exist("sim_speed_slider") else 1.0

        # Create the proper nested structure expected by apply_simulation_configuration_core
        # Split parameters into core_config, viz_config, and runtime_state
        viz_keys = ["camera_fov", "viz_update_interval_steps"]
        core_config_dict = {k: v for k, v in cfg_dict_from_ui.items() 
                           if k not in viz_keys + ["simulation_speed_factor"]}
        
        viz_config_dict = {}
        if "camera_fov" in cfg_dict_from_ui:
            viz_config_dict["camera_fov"] = cfg_dict_from_ui["camera_fov"]
        if "viz_update_interval_steps" in cfg_dict_from_ui:
            viz_config_dict["viz_update_interval_steps"] = cfg_dict_from_ui["viz_update_interval_steps"]
        
        runtime_state_dict = {
            "simulation_speed_factor": simulation_speed
        }
        
        # Create config objects to ensure all fields are present with defaults
        core_cfg = _create_config_from_dict(CoreSimConfig, core_config_dict)
        viz_cfg = _create_config_from_dict(VisualizationConfig, viz_config_dict)
        runtime = _create_config_from_dict(RuntimeState, runtime_state_dict)
        
        # Return nested dict structure
        return _get_full_config_dict(core_cfg, viz_cfg, runtime)

    except Exception as e:
        print(f"Error reading UI for sim config: {e}")
        update_status_bar(f"Error reading UI for config: {e}", color=[255,0,0], level="error")
        return None


def _populate_ui_from_config_dict(config_dict):
    """
    Populates DPG UI elements from a given simulation configuration dictionary.
    Called by the main/UI thread, e.g., after loading a profile or checkpoint.

    Supports both legacy flat SimulationConfiguration-style dicts and the
    newer nested structure returned by _get_full_config_dict, i.e.
        {"core_config": {...}, "viz_config": {...}, "runtime_state": {...}}.
    """
    if not dpg.is_dearpygui_running() or not config_dict: return

    # Normalize to a flat dict compatible with SimulationConfiguration.from_dict
    if any(k in config_dict for k in ("core_config", "viz_config", "runtime_state")):
        core_part = config_dict.get("core_config", {}) or {}
        viz_part = config_dict.get("viz_config", {}) or {}
        runtime_part = config_dict.get("runtime_state", {}) or {}

        flat_dict = {}
        if isinstance(core_part, dict):
            flat_dict.update(core_part)
        # Merge viz and runtime sections, without overriding core keys
        for section in (viz_part, runtime_part):
            if isinstance(section, dict):
                for k, v in section.items():
                    if k not in flat_dict:
                        flat_dict[k] = v
    else:
        flat_dict = config_dict

    # Use SimulationConfiguration.from_dict to ensure all fields are present with defaults if missing in dict
    cfg = SimulationConfiguration.from_dict(flat_dict)

    # General parameters
    if dpg.does_item_exist("cfg_num_neurons"): dpg.set_value("cfg_num_neurons", cfg.num_neurons)
    if dpg.does_item_exist("cfg_total_sim_time"): dpg.set_value("cfg_total_sim_time", cfg.total_simulation_time_ms)
    if dpg.does_item_exist("cfg_dt_ms"): dpg.set_value("cfg_dt_ms", cfg.dt_ms)
    if dpg.does_item_exist("cfg_seed"): dpg.set_value("cfg_seed", cfg.seed)
    if dpg.does_item_exist("cfg_neuron_model_type"): dpg.set_value("cfg_neuron_model_type", cfg.neuron_model_type)

    # Neural structure profile and HH preset (with realism constraints)
    profile_value = getattr(cfg, "neural_profile_name", "GENERIC_UNSTRUCTURED")
    if profile_value not in NEURAL_STRUCTURE_PROFILES:
        profile_value = "GENERIC_UNSTRUCTURED"
    if dpg.does_item_exist("cfg_neural_profile"):
        dpg.set_value("cfg_neural_profile", profile_value)

    if dpg.does_item_exist("cfg_default_neuron_type_hh") and hasattr(cfg, "default_neuron_type_hh"):
        allowed_hh = get_compatible_hh_type_names_for_profile(profile_value)
        if allowed_hh:
            dpg.configure_item("cfg_default_neuron_type_hh", items=allowed_hh)
            current_hh = cfg.default_neuron_type_hh
            if current_hh not in allowed_hh:
                current_hh = allowed_hh[0]
            dpg.set_value("cfg_default_neuron_type_hh", current_hh)
        else:
            dpg.set_value("cfg_default_neuron_type_hh", cfg.default_neuron_type_hh)
    
    # Connectivity
    if dpg.does_item_exist("cfg_enable_watts_strogatz"): dpg.set_value("cfg_enable_watts_strogatz", cfg.enable_watts_strogatz)
    if dpg.does_item_exist("cfg_connectivity_k"): dpg.set_value("cfg_connectivity_k", cfg.connectivity_k)
    if dpg.does_item_exist("cfg_connectivity_p_rewire"): dpg.set_value("cfg_connectivity_p_rewire", cfg.connectivity_p_rewire)
    if dpg.does_item_exist("cfg_connections_per_neuron"): dpg.set_value("cfg_connections_per_neuron", cfg.connections_per_neuron)

    # Synaptic parameters
    if dpg.does_item_exist("cfg_propagation_strength"): dpg.set_value("cfg_propagation_strength", cfg.propagation_strength)
    if dpg.does_item_exist("cfg_inhibitory_propagation_strength"): dpg.set_value("cfg_inhibitory_propagation_strength", cfg.inhibitory_propagation_strength)
    if dpg.does_item_exist("cfg_syn_tau_e"): dpg.set_value("cfg_syn_tau_e", cfg.syn_tau_g_e)
    if dpg.does_item_exist("cfg_syn_tau_i"): dpg.set_value("cfg_syn_tau_i", cfg.syn_tau_g_i)
    # NMDA
    if dpg.does_item_exist("cfg_enable_nmda"): dpg.set_value("cfg_enable_nmda", cfg.enable_nmda)
    if dpg.does_item_exist("cfg_nmda_ratio"): dpg.set_value("cfg_nmda_ratio", cfg.nmda_ratio)
    if dpg.does_item_exist("cfg_nmda_tau_decay"): dpg.set_value("cfg_nmda_tau_decay", cfg.nmda_tau_decay)
    if dpg.does_item_exist("cfg_nmda_tau_rise"): dpg.set_value("cfg_nmda_tau_rise", cfg.nmda_tau_rise)
    if dpg.does_item_exist("cfg_nmda_mg_conc"): dpg.set_value("cfg_nmda_mg_conc", cfg.nmda_mg_concentration)
    if dpg.does_item_exist("cfg_num_traits"): dpg.set_value("cfg_num_traits", cfg.num_traits)

    # Learning & Plasticity
    if dpg.does_item_exist("cfg_enable_hebbian_learning"): dpg.set_value("cfg_enable_hebbian_learning", cfg.enable_hebbian_learning)
    if dpg.does_item_exist("cfg_hebbian_learning_rate"): dpg.set_value("cfg_hebbian_learning_rate", cfg.hebbian_learning_rate)
    if dpg.does_item_exist("cfg_hebbian_max_weight"): dpg.set_value("cfg_hebbian_max_weight", cfg.hebbian_max_weight)
    if dpg.does_item_exist("cfg_enable_short_term_plasticity"): dpg.set_value("cfg_enable_short_term_plasticity", cfg.enable_short_term_plasticity)
    if dpg.does_item_exist("cfg_stp_U"): dpg.set_value("cfg_stp_U", cfg.stp_U)
    if dpg.does_item_exist("cfg_stp_tau_d"): dpg.set_value("cfg_stp_tau_d", cfg.stp_tau_d)
    if dpg.does_item_exist("cfg_stp_tau_f"): dpg.set_value("cfg_stp_tau_f", cfg.stp_tau_f)
    if dpg.does_item_exist("cfg_enable_per_type_stp"): dpg.set_value("cfg_enable_per_type_stp", getattr(cfg, 'enable_per_type_stp', True))
    # Per-type STP UI fields
    per_type_U = getattr(cfg, 'stp_U_per_type', None) or [0.5, 0.5, 0.25, 0.25]
    per_type_tau_d = getattr(cfg, 'stp_tau_d_per_type', None) or [200.0, 200.0, 100.0, 100.0]
    per_type_tau_f = getattr(cfg, 'stp_tau_f_per_type', None) or [20.0, 20.0, 50.0, 50.0]
    for i, suffix in enumerate(["ee", "ei", "ie", "ii"]):
        if dpg.does_item_exist(f"cfg_stp_U_{suffix}"): dpg.set_value(f"cfg_stp_U_{suffix}", per_type_U[i])
        if dpg.does_item_exist(f"cfg_stp_tau_d_{suffix}"): dpg.set_value(f"cfg_stp_tau_d_{suffix}", per_type_tau_d[i])
        if dpg.does_item_exist(f"cfg_stp_tau_f_{suffix}"): dpg.set_value(f"cfg_stp_tau_f_{suffix}", per_type_tau_f[i])
    # Structural plasticity activity bias
    if dpg.does_item_exist("cfg_struct_plast_activity_bias"): dpg.set_value("cfg_struct_plast_activity_bias", getattr(cfg, 'struct_plast_activity_bias', 0.5))

    # Homeostasis
    if dpg.does_item_exist("cfg_enable_homeostasis"): dpg.set_value("cfg_enable_homeostasis", cfg.enable_homeostasis)
    if dpg.does_item_exist("cfg_homeostasis_target_rate"): dpg.set_value("cfg_homeostasis_target_rate", cfg.homeostasis_target_rate)
    if dpg.does_item_exist("cfg_homeostasis_threshold_min"): dpg.set_value("cfg_homeostasis_threshold_min", cfg.homeostasis_threshold_min)
    if dpg.does_item_exist("cfg_homeostasis_threshold_max"): dpg.set_value("cfg_homeostasis_threshold_max", cfg.homeostasis_threshold_max)
    if dpg.does_item_exist("cfg_enable_synaptic_scaling"): dpg.set_value("cfg_enable_synaptic_scaling", cfg.enable_synaptic_scaling)
    if dpg.does_item_exist("cfg_synaptic_scaling_rate"): dpg.set_value("cfg_synaptic_scaling_rate", cfg.synaptic_scaling_rate)

    # STDP
    if dpg.does_item_exist("cfg_enable_stdp"): dpg.set_value("cfg_enable_stdp", cfg.enable_stdp)
    if dpg.does_item_exist("cfg_stdp_a_plus"): dpg.set_value("cfg_stdp_a_plus", cfg.stdp_a_plus)
    if dpg.does_item_exist("cfg_stdp_a_minus"): dpg.set_value("cfg_stdp_a_minus", cfg.stdp_a_minus)
    if dpg.does_item_exist("cfg_stdp_tau_plus_ms"): dpg.set_value("cfg_stdp_tau_plus_ms", cfg.stdp_tau_plus_ms)
    if dpg.does_item_exist("cfg_stdp_tau_minus_ms"): dpg.set_value("cfg_stdp_tau_minus_ms", cfg.stdp_tau_minus_ms)
    if dpg.does_item_exist("cfg_stdp_w_min"): dpg.set_value("cfg_stdp_w_min", cfg.stdp_w_min)
    if dpg.does_item_exist("cfg_stdp_w_max"): dpg.set_value("cfg_stdp_w_max", cfg.stdp_w_max)

    # Reward-Modulated Plasticity
    if dpg.does_item_exist("cfg_enable_reward_modulation"): dpg.set_value("cfg_enable_reward_modulation", cfg.enable_reward_modulation)
    if hasattr(cfg, 'reward_learning_rate') and dpg.does_item_exist("cfg_reward_learning_rate"):
        dpg.set_value("cfg_reward_learning_rate", cfg.reward_learning_rate)
    if hasattr(cfg, 'reward_eligibility_tau_ms') and dpg.does_item_exist("cfg_reward_eligibility_tau_ms"):
        dpg.set_value("cfg_reward_eligibility_tau_ms", cfg.reward_eligibility_tau_ms)

    # Structural Plasticity
    if dpg.does_item_exist("cfg_enable_structural_plasticity"): dpg.set_value("cfg_enable_structural_plasticity", cfg.enable_structural_plasticity)
    if dpg.does_item_exist("cfg_struct_plast_formation_rate"): dpg.set_value("cfg_struct_plast_formation_rate", cfg.struct_plast_formation_rate)
    if dpg.does_item_exist("cfg_struct_plast_elimination_rate"): dpg.set_value("cfg_struct_plast_elimination_rate", cfg.struct_plast_elimination_rate)
    if dpg.does_item_exist("cfg_struct_plast_weight_threshold"): dpg.set_value("cfg_struct_plast_weight_threshold", cfg.struct_plast_weight_threshold)
    if dpg.does_item_exist("cfg_struct_plast_target_density"): dpg.set_value("cfg_struct_plast_target_density", cfg.struct_plast_target_density)
    if dpg.does_item_exist("cfg_struct_plast_distance_scale"): dpg.set_value("cfg_struct_plast_distance_scale", cfg.struct_plast_distance_scale)
    if dpg.does_item_exist("cfg_struct_plast_update_interval_steps"): dpg.set_value("cfg_struct_plast_update_interval_steps", cfg.struct_plast_update_interval_steps)

    # Heterogeneity & Noise
    if dpg.does_item_exist("cfg_enable_parameter_heterogeneity"): dpg.set_value("cfg_enable_parameter_heterogeneity", cfg.enable_parameter_heterogeneity)
    if dpg.does_item_exist("cfg_heterogeneity_seed"): dpg.set_value("cfg_heterogeneity_seed", cfg.heterogeneity_seed)
    if dpg.does_item_exist("cfg_enable_conductance_noise"): dpg.set_value("cfg_enable_conductance_noise", cfg.enable_conductance_noise)
    if dpg.does_item_exist("cfg_conductance_noise_relative_std"): dpg.set_value("cfg_conductance_noise_relative_std", cfg.conductance_noise_relative_std)
    if dpg.does_item_exist("cfg_enable_ou_process"): dpg.set_value("cfg_enable_ou_process", cfg.enable_ou_process)
    if dpg.does_item_exist("cfg_ou_mean_current_pA"): dpg.set_value("cfg_ou_mean_current_pA", cfg.ou_mean_current_pA)
    if dpg.does_item_exist("cfg_ou_std_current_pA"): dpg.set_value("cfg_ou_std_current_pA", cfg.ou_std_current_pA)
    if dpg.does_item_exist("cfg_ou_tau_ms"): dpg.set_value("cfg_ou_tau_ms", cfg.ou_tau_ms)
    if dpg.does_item_exist("cfg_ou_seed"): dpg.set_value("cfg_ou_seed", cfg.ou_seed)

    # Camera FOV and Visualization settings
    if dpg.does_item_exist("cfg_camera_fov"): dpg.set_value("cfg_camera_fov", cfg.camera_fov)
    # Handle viz_update_interval_steps if it exists in the config (backward compatibility)
    if hasattr(cfg, "viz_update_interval_steps") and dpg.does_item_exist("cfg_viz_update_interval_steps"):
        dpg.set_value("cfg_viz_update_interval_steps", cfg.viz_update_interval_steps)
    
    # Hardware performance note - only update if config has a value (don't overwrite loaded benchmark data with fallback)
    if hasattr(cfg, "hardware_performance_note") and dpg.does_item_exist("cfg_hardware_performance_note"):
        if cfg.hardware_performance_note:  # Only update if config has actual data
            dpg.set_value("cfg_hardware_performance_note", cfg.hardware_performance_note)
        elif not dpg.get_value("cfg_hardware_performance_note"):  # Only set fallback if widget is currently empty
            dpg.set_value("cfg_hardware_performance_note", "Run visualization benchmark to determine hardware limits (viz_benchmark.py)")
    
    # Model-specific parameters
    if cfg.neuron_model_type == NeuronModel.IZHIKEVICH.name:
        ui_izh_params_keys = ["izh_C_val", "izh_k_val", "izh_vr_val", "izh_vt_val", "izh_vpeak_val", "izh_a_val", "izh_b_val", "izh_c_val", "izh_d_val"]
        for key_suffix in ui_izh_params_keys:
            dpg_tag = f"cfg_{key_suffix}"
            if dpg.does_item_exist(dpg_tag): dpg.set_value(dpg_tag, getattr(cfg, key_suffix))
    elif cfg.neuron_model_type == NeuronModel.HODGKIN_HUXLEY.name:
        ui_hh_params_keys = [
            "hh_C_m", "hh_g_Na_max", "hh_g_K_max", "hh_g_L", "hh_E_Na", "hh_E_K", "hh_E_L",
            "hh_v_peak", "hh_v_rest_init", "hh_g_M_max", "hh_m_current_tau_ms",
            "hh_g_CaT_max", "hh_E_CaT", "hh_g_h_max", "hh_E_h", "hh_g_NaP_max",
            "hh_q10_factor", "hh_temperature_celsius",
            "hh_external_drive_scale",
        ]
        for key_suffix in ui_hh_params_keys:
            dpg_tag = f"cfg_{key_suffix}"
            if dpg.does_item_exist(dpg_tag): dpg.set_value(dpg_tag, getattr(cfg, key_suffix))
    elif cfg.neuron_model_type == NeuronModel.ADEX.name:
        ui_adex_params_keys = [
            "adex_C", "adex_g_L", "adex_E_L", "adex_V_T", "adex_Delta_T",
            "adex_a", "adex_tau_w", "adex_b", "adex_V_r", "adex_V_peak",
            "adex_external_drive_scale",
        ]
        for key_suffix in ui_adex_params_keys:
            dpg_tag = f"cfg_{key_suffix}"
            if dpg.does_item_exist(dpg_tag): dpg.set_value(dpg_tag, getattr(cfg, key_suffix))

    if dpg.does_item_exist("sim_speed_slider"): dpg.set_value("sim_speed_slider", cfg.simulation_speed_factor)

    _toggle_model_specific_params_visibility(None, cfg.neuron_model_type) # Update visibility of UI groups
    update_status_bar("Configuration loaded into UI.", level="info") 
    global_gui_state["reset_sim_needed_from_ui_change"] = False # Config is now in sync with UI


def _toggle_model_specific_params_visibility(sender, app_data, user_data=None):
    """Shows/hides UI groups for Izhikevich or Hodgkin-Huxley parameters. Called by main/UI thread."""
    selected_model_name = app_data # This is the string name of the model from the combo box

    is_izh = selected_model_name == NeuronModel.IZHIKEVICH.name
    is_hh = selected_model_name == NeuronModel.HODGKIN_HUXLEY.name
    is_adex = selected_model_name == NeuronModel.ADEX.name

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist("izhikevich_params_group"): dpg.configure_item("izhikevich_params_group", show=is_izh)
        if dpg.does_item_exist("hodgkin_huxley_params_group"): dpg.configure_item("hodgkin_huxley_params_group", show=is_hh)
        if dpg.does_item_exist("adex_params_group"): dpg.configure_item("adex_params_group", show=is_adex)
        if dpg.does_item_exist("homeostasis_izh_specific_group"): dpg.configure_item("homeostasis_izh_specific_group", show=is_izh)
        
        # Update neuron type filter combo based on selected model
        # This requires access to sim_bridge or a way to get types for a model.
        # For now, assume sim_bridge is accessible or this logic is refined.
        # If global_simulation_bridge is None yet (e.g. during initial UI setup before sim_bridge is fully ready for this),
        # this part might need to be deferred or handled carefully.
        # For now, let's assume it's called when sim_bridge can provide types.
        if dpg.does_item_exist("filter_neuron_type_combo"):
            # Create a temporary config to get available types for the selected model
            temp_cfg_for_types = SimulationConfiguration()
            temp_cfg_for_types.neuron_model_type = selected_model_name # Set model
            # Get available types using a static or instance method if SimulationConfiguration had one,
            # or if SimulationBridge has a helper. For now, mimic SimulationBridge's logic.
            available_types_for_filter = ["All"]
            if selected_model_name == NeuronModel.IZHIKEVICH.name:
                available_types_for_filter.extend([f"Izh2007_{nt.name.replace('IZH2007_', '')}" for nt in NeuronType if "IZH2007" in nt.name and nt in DefaultIzhikevichParamsManager.PARAMS])
            elif selected_model_name == NeuronModel.HODGKIN_HUXLEY.name:
                available_types_for_filter.extend([f"HH_{nt.name.replace('HH_', '')}" for nt in NeuronType if "HH_" in nt.name and nt in DefaultHodgkinHuxleyParams.PARAMS])
            available_types_for_filter = list(dict.fromkeys(available_types_for_filter))


            current_filter_value = dpg.get_value("filter_neuron_type_combo")
            dpg.configure_item("filter_neuron_type_combo", items=available_types_for_filter)
            if current_filter_value in available_types_for_filter:
                dpg.set_value("filter_neuron_type_combo", current_filter_value)
            elif "All" in available_types_for_filter: 
                dpg.set_value("filter_neuron_type_combo", "All")
            elif available_types_for_filter: 
                dpg.set_value("filter_neuron_type_combo", available_types_for_filter[0])
            else:
                dpg.set_value("filter_neuron_type_combo", "")

    # Check config against benchmark limits after populating
    _check_and_warn_hardware_limits()


# --- DPG Event Handlers for OpenGL Visualization Settings ---
def handle_gl_point_size_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE: opengl_viz_config['POINT_SIZE'] = app_data; trigger_filter_update_signal()
def handle_gl_synapse_alpha_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE: opengl_viz_config['SYNAPSE_ALPHA_MODIFIER'] = app_data; trigger_filter_update_signal()
def handle_gl_activity_highlight_frames_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE and opengl_viz_config is not None:
        try:
            new_frames = int(app_data)
            if new_frames >= 1: opengl_viz_config['ACTIVITY_HIGHLIGHT_FRAMES'] = new_frames
            elif dpg.is_dearpygui_running() and dpg.does_item_exist(sender): 
                dpg.set_value(sender, opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7))
        except ValueError: 
            if dpg.is_dearpygui_running() and dpg.does_item_exist(sender):
                dpg.set_value(sender, opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7))
def handle_gl_max_neurons_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE and opengl_viz_config is not None:
        try:
            new_val = int(app_data)
            if new_val >= 0: opengl_viz_config['MAX_NEURONS_TO_RENDER'] = new_val; trigger_filter_update_signal()
            elif dpg.is_dearpygui_running() and dpg.does_item_exist(sender):
                dpg.set_value(sender, opengl_viz_config.get('MAX_NEURONS_TO_RENDER', 10000))
        except ValueError:
            if dpg.is_dearpygui_running() and dpg.does_item_exist(sender):
                dpg.set_value(sender, opengl_viz_config.get('MAX_NEURONS_TO_RENDER', 10000))
def handle_gl_max_connections_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE and opengl_viz_config is not None:
        try:
            new_val = int(app_data)
            if new_val >= 0: opengl_viz_config['MAX_CONNECTIONS_TO_RENDER'] = new_val; trigger_filter_update_signal()
            elif dpg.is_dearpygui_running() and dpg.does_item_exist(sender):
                dpg.set_value(sender, opengl_viz_config.get('MAX_CONNECTIONS_TO_RENDER', 20000))
        except ValueError:
            if dpg.is_dearpygui_running() and dpg.does_item_exist(sender):
                dpg.set_value(sender, opengl_viz_config.get('MAX_CONNECTIONS_TO_RENDER', 20000))
def handle_gl_inactive_neuron_opacity_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE and opengl_viz_config is not None:
        try:
            new_val = float(app_data)
            if 0.0 <= new_val <= 1.0: opengl_viz_config['INACTIVE_NEURON_OPACITY'] = new_val; trigger_filter_update_signal()
            elif dpg.is_dearpygui_running() and dpg.does_item_exist(sender):
                dpg.set_value(sender, opengl_viz_config.get('INACTIVE_NEURON_OPACITY', 0.25))
        except ValueError:
            if dpg.is_dearpygui_running() and dpg.does_item_exist(sender):
                dpg.set_value(sender, opengl_viz_config.get('INACTIVE_NEURON_OPACITY', 0.25))
def handle_gl_enable_synaptic_pulses_change(sender, app_data, user_data):
    if OPENGL_AVAILABLE and opengl_viz_config is not None:
        opengl_viz_config['ENABLE_SYNAPTIC_PULSES'] = app_data
        trigger_filter_update_signal()

# --- DPG Event Handlers for Simulation Control & Configuration ---

def _apply_hh_preset_params_to_ui(hh_type_name):
    """Update HH parameter input fields in the UI to match a given preset.

    This keeps the visible HH parameter panel in sync with the selected
    HH neuron type and any profile/model-driven preset selection.
    """
    if not dpg.is_dearpygui_running() or not hh_type_name:
        return
    try:
        hh_enum = NeuronType[hh_type_name]
        params = DefaultHodgkinHuxleyParams.get_params(hh_enum)
    except Exception as e:
        print(f"Warning: could not apply HH preset '{hh_type_name}' to UI: {e}")
        return

    tag_key_pairs = [
        ("cfg_hh_C_m", "C_m"),
        ("cfg_hh_g_Na_max", "g_Na_max"),
        ("cfg_hh_g_K_max", "g_K_max"),
        ("cfg_hh_g_L", "g_L"),
        ("cfg_hh_E_Na", "E_Na"),
        ("cfg_hh_E_K", "E_K"),
        ("cfg_hh_E_L", "E_L"),
        ("cfg_hh_v_rest_init", "v_rest_hh"),
        ("cfg_hh_v_peak", "v_peak_hh"),
        ("cfg_hh_g_M_max", "g_M_max"),
        ("cfg_hh_m_current_tau_ms", "m_current_tau_ms"),
        ("cfg_hh_g_CaT_max", "g_CaT_max"),
        ("cfg_hh_E_CaT", "E_CaT"),
        ("cfg_hh_g_h_max", "g_h_max"),
        ("cfg_hh_E_h", "E_h"),
        ("cfg_hh_g_NaP_max", "g_NaP_max"),
        ("cfg_hh_q10_factor", "q10_factor"),
        ("cfg_hh_temperature_celsius", "temperature_celsius"),
    ]
    for tag, key in tag_key_pairs:
        if dpg.does_item_exist(tag) and key in params:
            dpg.set_value(tag, params[key])


def handle_reset_hh_drive_to_auto(sender=None, app_data=None, user_data=None):
    """Reset the HH external drive scale slider to the auto-tuned value for the current combo, if any."""
    try:
        if not dpg.is_dearpygui_running():
            return
        if not dpg.does_item_exist("cfg_neuron_model_type"):
            return
        model_name = dpg.get_value("cfg_neuron_model_type")
        if model_name != NeuronModel.HODGKIN_HUXLEY.name:
            update_status_bar("HH drive reset: current model is not Hodgkin-Huxley.", level="warning")
            return
        profile_name = dpg.get_value("cfg_neural_profile") if dpg.does_item_exist("cfg_neural_profile") else "GENERIC_UNSTRUCTURED"
        hh_type = dpg.get_value("cfg_default_neuron_type_hh") if dpg.does_item_exist("cfg_default_neuron_type_hh") else NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name
        tuned = get_auto_tuned_overrides_for_combo(model_name, profile_name, hh_type)
        if not tuned or not isinstance(tuned, dict):
            update_status_bar("No auto-tuned HH entry found for this combination.", level="warning")
            return
        core_overrides = tuned.get("core_overrides", {}) or {}
        scale = core_overrides.get("hh_external_drive_scale")
        if scale is None:
            update_status_bar("Auto-tuned config has no HH drive scale for this combination.", level="warning")
            return
        if dpg.does_item_exist("cfg_hh_external_drive_scale"):
            dpg.set_value("cfg_hh_external_drive_scale", float(scale))
        _update_sim_config_from_ui_and_signal_reset_needed("cfg_hh_external_drive_scale", float(scale))
        update_status_bar("HH drive scale reset to auto-tuned value. Apply & Reset to use in sim.", level="info")
    except Exception as e:
        update_status_bar(f"Error resetting HH drive scale: {e}", level="error")


def handle_reset_adex_drive_to_auto(sender=None, app_data=None, user_data=None):
    """Reset the AdEx external drive scale slider to the auto-tuned value for the current profile, if any."""
    try:
        if not dpg.is_dearpygui_running():
            return
        if not dpg.does_item_exist("cfg_neuron_model_type"):
            return
        model_name = dpg.get_value("cfg_neuron_model_type")
        if model_name != NeuronModel.ADEX.name:
            update_status_bar("AdEx drive reset: current model is not AdEx.", level="warning")
            return
        profile_name = dpg.get_value("cfg_neural_profile") if dpg.does_item_exist("cfg_neural_profile") else "GENERIC_UNSTRUCTURED"
        tuned = get_auto_tuned_overrides_for_combo(model_name, profile_name, None)
        if not tuned or not isinstance(tuned, dict):
            update_status_bar("No auto-tuned AdEx entry found for this profile.", level="warning")
            return
        core_overrides = tuned.get("core_overrides", {}) or {}
        scale = core_overrides.get("adex_external_drive_scale")
        if scale is None:
            update_status_bar("Auto-tuned config has no AdEx drive scale for this profile.", level="warning")
            return
        if dpg.does_item_exist("cfg_adex_external_drive_scale"):
            dpg.set_value("cfg_adex_external_drive_scale", float(scale))
        _update_sim_config_from_ui_and_signal_reset_needed("cfg_adex_external_drive_scale", float(scale))
        update_status_bar("AdEx drive scale reset to auto-tuned value. Apply & Reset to use in sim.", level="info")
    except Exception as e:
        update_status_bar(f"Error resetting AdEx drive scale: {e}", level="error")


def _update_sim_config_from_ui_and_signal_reset_needed(sender=None, app_data=None, user_data=None):
    """
    Callback for UI elements that change sim config. Sets a flag that sim needs reset.
    The actual config update is collected by `handle_apply_config_changes_and_reset`.
    """
    global_gui_state["reset_sim_needed_from_ui_change"] = True
    update_status_bar("Parameter changed. Press 'Apply Changes & Reset Sim' to take effect.", color=[255,165,0,255], level="warning")

    # Special handling for certain controls
    if sender == "cfg_neuron_model_type":
        # Update visibility of model-specific parameter groups
        _toggle_model_specific_params_visibility(sender, app_data)

        # If switching to HH, clamp the preset list to profile-compatible types and
        # snap the selection (and visible HH params) to a valid preset.
        try:
            if app_data == NeuronModel.HODGKIN_HUXLEY.name and dpg.is_dearpygui_running():
                if dpg.does_item_exist("cfg_default_neuron_type_hh") and dpg.does_item_exist("cfg_neural_profile"):
                    profile_name = dpg.get_value("cfg_neural_profile")
                    allowed_hh = get_compatible_hh_type_names_for_profile(profile_name)
                    if allowed_hh:
                        dpg.configure_item("cfg_default_neuron_type_hh", items=allowed_hh)
                        current_hh = dpg.get_value("cfg_default_neuron_type_hh")
                        if current_hh not in allowed_hh:
                            current_hh = allowed_hh[0]
                            dpg.set_value("cfg_default_neuron_type_hh", current_hh)
                        _apply_hh_preset_params_to_ui(current_hh)
        except Exception as e:
            print(f"Warning: failed to enforce HH preset compatibility on model change: {e}")

    elif sender == "cfg_neural_profile":
        # When changing neural structure profile, if HH model is active, clamp the HH
        # preset list and snap the selection to the profile-compatible preset.
        try:
            if dpg.is_dearpygui_running() and dpg.does_item_exist("cfg_neuron_model_type"):
                model_name = dpg.get_value("cfg_neuron_model_type")
                profile_name = app_data
                if model_name == NeuronModel.HODGKIN_HUXLEY.name and dpg.does_item_exist("cfg_default_neuron_type_hh"):
                    allowed_hh = get_compatible_hh_type_names_for_profile(profile_name)
                    if allowed_hh:
                        dpg.configure_item("cfg_default_neuron_type_hh", items=allowed_hh)
                        current_hh = dpg.get_value("cfg_default_neuron_type_hh")
                        if current_hh not in allowed_hh:
                            current_hh = allowed_hh[0]
                            dpg.set_value("cfg_default_neuron_type_hh", current_hh)
                        _apply_hh_preset_params_to_ui(current_hh)
        except Exception as e:
            print(f"Warning: failed to enforce HH preset compatibility on profile change: {e}")

    elif sender == "cfg_default_neuron_type_hh":
        # Direct change of HH preset by the user; update HH params panel to match,
        # but still respect per-profile compatibility.
        try:
            if dpg.is_dearpygui_running() and dpg.does_item_exist("cfg_neural_profile"):
                profile_name = dpg.get_value("cfg_neural_profile")
                allowed_hh = get_compatible_hh_type_names_for_profile(profile_name)
                if allowed_hh and app_data not in allowed_hh:
                    app_data = allowed_hh[0]
                    dpg.set_value("cfg_default_neuron_type_hh", app_data)
            _apply_hh_preset_params_to_ui(app_data)
        except Exception as e:
            print(f"Warning: failed to apply HH preset params on preset change: {e}")

    # Check proposed config against benchmark-derived hardware limits
    _check_and_warn_hardware_limits()


def _check_and_warn_hardware_limits():
    """Reads current UI values and warns if config exceeds benchmark-tested limits."""
    try:
        if not dpg.is_dearpygui_running():
            return
        if not dpg.does_item_exist("cfg_num_neurons") or not dpg.does_item_exist("cfg_neuron_model_type"):
            return

        model_name = dpg.get_value("cfg_neuron_model_type")
        num_neurons = dpg.get_value("cfg_num_neurons")
        conn_per = dpg.get_value("cfg_connections_per_neuron") if dpg.does_item_exist("cfg_connections_per_neuron") else 100

        is_safe, warning = check_config_against_limits(model_name, num_neurons, conn_per)

        tag = "hw_limit_warning_text"
        if dpg.does_item_exist(tag):
            if warning:
                dpg.set_value(tag, warning)
                dpg.configure_item(tag, color=[255, 100, 100, 255], show=True)
            else:
                # Show positive feedback if within limits and benchmark data exists
                limits = get_hardware_limits_for_model(model_name)
                if limits:
                    # Find the matching or next-larger tested config
                    configs = limits["configs"]
                    matching = [c for c in configs if c["neurons"] >= num_neurons and c["conn"] >= conn_per]
                    if matching:
                        m = matching[0]
                        dpg.set_value(tag, f"Tested OK: {m['steps_per_sec']:.0f} steps/s, {m['gpu_gb']:.1f}GB VRAM")
                        dpg.configure_item(tag, color=[100, 255, 100, 255], show=True)
                    else:
                        dpg.set_value(tag, "")
                        dpg.configure_item(tag, show=False)
                else:
                    dpg.set_value(tag, "")
                    dpg.configure_item(tag, show=False)
    except Exception:
        pass  # Never let limit check crash the UI


def _handle_model_type_change_dpg(sender, app_data, user_data=None):
    """Handles change in neuron model type selection in DPG. Updates UI visibility and signals reset."""
    _toggle_model_specific_params_visibility(sender, app_data) # Update UI sections
    # Auto-adjust dt when switching to HH (needs dt <= 0.1ms for stability)
    # or back to a simpler model (can use larger dt)
    if dpg.does_item_exist("cfg_dt_ms"):
        current_dt = dpg.get_value("cfg_dt_ms")
        if app_data == NeuronModel.HODGKIN_HUXLEY.name:
            if current_dt > 0.1:
                dpg.set_value("cfg_dt_ms", 0.05)
                update_status_bar("dt auto-adjusted to 0.05 ms for HH stability (was {:.3f} ms)".format(current_dt),
                                  color=[255, 200, 100, 255], level="warning")
        else:
            # When switching away from HH, if dt is very small (likely auto-set), restore a reasonable default
            if current_dt <= 0.1:
                dpg.set_value("cfg_dt_ms", 0.5)
                update_status_bar("dt restored to 0.5 ms for {} model".format(app_data),
                                  color=[150, 220, 255, 255], level="info")
    _update_sim_config_from_ui_and_signal_reset_needed() # Mark that config changed and reset is needed


def handle_start_simulation_event(sender=None, app_data=None, user_data=None):
    """Sends a 'START_SIM' command to the simulation thread."""
    if global_gui_state.get("is_playback_mode_active", False):
        update_status_bar("Error: Cannot start simulation in playback mode.", color=[255,0,0,255], level="error")
        return
    if global_gui_state.get("reset_sim_needed_from_ui_change", False):
        update_status_bar("Apply changes before starting!", color=[255,100,100,255], level="warning")
        return
    
    ui_to_sim_queue.put({"type": "START_SIM"})
    update_status_bar("Start command sent to simulation...", level="info")
    # UI state (buttons enabled/disabled) will be updated when sim_thread confirms via message.
    # Optimistically, we can update some UI here, but it's better to wait for ack for robustness.
    # For now, let global_gui_state._sim_is_running_ui_view reflect the command sent.
    global_gui_state["_sim_is_running_ui_view"] = True
    global_gui_state["_sim_is_paused_ui_view"] = False
    update_ui_for_simulation_run_state(is_running=True, is_paused=False) # Optimistic UI update


def handle_stop_simulation_event(sender=None, app_data=None, user_data=None):
    """Sends a 'STOP_SIM' command to the simulation thread."""
    ui_to_sim_queue.put({"type": "STOP_SIM"})
    update_status_bar("Stop command sent to simulation...", level="info")
    global_gui_state["_sim_is_running_ui_view"] = False
    global_gui_state["_sim_is_paused_ui_view"] = False
    update_ui_for_simulation_run_state(is_running=False, is_paused=False) # Optimistic UI update
    # Actual telemetry reset for UI happens when sim_thread confirms stop and sends initial data.


def handle_pause_simulation_event(sender=None, app_data=None, user_data=None):
    """Sends 'PAUSE_SIM' or 'RESUME_SIM' command based on current UI perceived state."""
    if not global_gui_state.get("_sim_is_running_ui_view", False): # Can only pause/resume if UI thinks it's running
        update_status_bar("Sim not running, cannot pause/resume.", color=[255,165,0,255], level="warning")
        return

    if global_gui_state.get("_sim_is_paused_ui_view", False): # If UI thinks it's paused, send RESUME
        ui_to_sim_queue.put({"type": "RESUME_SIM"})
        update_status_bar("Resume command sent...", level="info")
        global_gui_state["_sim_is_paused_ui_view"] = False # Optimistic
    else: # If UI thinks it's running (not paused), send PAUSE
        ui_to_sim_queue.put({"type": "PAUSE_SIM"})
        update_status_bar("Pause command sent...", level="info")
        global_gui_state["_sim_is_paused_ui_view"] = True # Optimistic
    update_ui_for_simulation_run_state(is_running=True, is_paused=global_gui_state["_sim_is_paused_ui_view"])


def handle_step_simulation_event(sender=None, app_data=None, user_data=None):
    """Sends a 'STEP_SIM_ONE_MS' command to the simulation thread."""
    if global_gui_state.get("is_playback_mode_active", False):
        update_status_bar("Error: Cannot step live sim in playback mode.", color=[255,0,0,255], level="error")
        return

    is_paused_ui = global_gui_state.get("_sim_is_paused_ui_view", False)
    is_running_ui = global_gui_state.get("_sim_is_running_ui_view", False)
    can_step_ui = (is_running_ui and is_paused_ui) or (not is_running_ui)

    if can_step_ui:
        if global_gui_state.get("reset_sim_needed_from_ui_change", False):
            update_status_bar("Apply changes before stepping!", color=[255,100,100,255], level="warning")
            return
        ui_to_sim_queue.put({"type": "STEP_SIM_ONE_MS"}) # Sim thread will determine actual number of substeps
        update_status_bar("Step (1ms) command sent...", level="info")
    else:
        update_status_bar("Sim must be running & paused, or stopped, to step.", color=[255,165,0,255], level="warning")

def handle_apply_config_changes_and_reset(sender=None, app_data=None, user_data=None, from_reset_button=False):
    """
    Handles the 'Apply Changes & Reset Sim' button.
    Collects UI config, sends it to sim_thread for application and reset.
    """
    update_status_bar("Collecting UI configuration...", level="info")
    
    # Get the current configuration from UI elements
    # _update_sim_config_from_ui now returns a complete config dict
    config_dict_from_ui = _update_sim_config_from_ui(update_model_specific=True)

    if config_dict_from_ui:
        update_status_bar("Sending new configuration to simulation thread for reset...", level="info")
        ui_to_sim_queue.put({
            "type": "APPLY_CONFIG_AND_RESET",
            "config_dict": config_dict_from_ui
        })
        # UI will be fully updated once sim_thread confirms and sends back new state/config.
        # global_gui_state["reset_sim_needed_from_ui_change"] will be set to False by UI thread
        # after confirmation from sim_thread.
    else:
        update_status_bar("Failed to collect configuration from UI. Please check parameters.", color=[255,0,0,255], level="error")


def handle_sim_speed_change(sender, app_data, user_data):
    """Sends 'SET_SIM_SPEED' command when simulation speed slider changes."""
    ui_to_sim_queue.put({"type": "SET_SIM_SPEED", "factor": app_data})
    # Status bar update can be done here or by sim_thread acknowledging.
    # update_status_bar(f"Sim speed factor set to {app_data:.2f}x (command sent)", level="info")


# --- DPG File Dialog Callbacks and Handlers (Main/UI Thread) ---

def get_profile_files(profile_directory): # Profiles are still JSON
    """Gets a list of .json profile files from the specified directory."""
    try:
        if os.path.exists(profile_directory) and os.path.isdir(profile_directory):
            return sorted([f for f in os.listdir(profile_directory) if f.endswith(".json") and os.path.isfile(os.path.join(profile_directory, f))])
    except Exception as e: print(f"Error listing profile directory '{profile_directory}': {e}")
    return []

def get_hdf5_files(directory, extension): # Helper for .simrec.h5 and .simstate.h5
    """Gets a list of HDF5 files with a specific extension from a directory."""
    try:
        if os.path.exists(directory) and os.path.isdir(directory):
            return sorted([f for f in os.listdir(directory) if f.endswith(extension) and os.path.isfile(os.path.join(directory, f))])
    except Exception as e: print(f"Error listing directory '{directory}' for '{extension}': {e}")
    return []


def handle_save_profile_button_press(sender=None, app_data=None, user_data=None): # Profiles are JSON
    """Shows the 'Save Profile' file dialog."""
    if dpg.is_dearpygui_running() and dpg.does_item_exist("save_profile_file_dialog"):
        # Ensure current UI state is reflected in a temporary config dict to be saved
        # The actual saving happens in the callback, which will re-fetch this.
        update_status_bar("Preparing to save profile...", level="info")
        dpg.show_item("save_profile_file_dialog")

def handle_load_profile_button_press(sender=None, app_data=None, user_data=None): # Profiles are JSON
    """Shows the 'Load Profile' file dialog."""
    if global_gui_state.get("is_recording_active", False) or global_gui_state.get("is_playback_mode_active", False):
        update_status_bar("Stop recording/playback before loading a profile.", color=[255,165,0,255], level="warning")
        return
    if dpg.is_dearpygui_running() and dpg.does_item_exist("load_profile_file_dialog"):
        dpg.show_item("load_profile_file_dialog")

def _normalize_filepath_extension(filepath, required_extension, filter_extension=None):
    """
    Normalizes a filepath to ensure it has the correct extension.

    Args:
        filepath: The filepath from the file dialog
        required_extension: The extension we want (e.g., ".json", ".simstate.h5", ".simrec.h5")
        filter_extension: The filter extension DPG might have appended (e.g., ".h5", ".*")

    Returns:
        Normalized filepath with correct extension
    """
    # Strip ".*" if DPG appended it from "All Files" filter
    if filepath.endswith(".*"):
        filepath = filepath[:-2]

    # Strip the filter extension if DPG appended it (e.g., ".h5" when we want ".simstate.h5")
    if filter_extension and filter_extension != ".*":
        if filepath.lower().endswith(filter_extension.lower()) and not filepath.lower().endswith(required_extension.lower()):
            filepath = filepath[:-len(filter_extension)]

    # Add the required extension if not present
    if not filepath.lower().endswith(required_extension.lower()):
        filepath += required_extension

    return filepath

def save_profile_dialog_callback(sender, app_data): # Profiles are JSON
    """
    Callback for the 'Save Profile' file dialog. Saves current UI config and GUI settings.
    This operation is done entirely by the UI thread.
    """
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = _normalize_filepath_extension(
            app_data["file_path_name"],
            required_extension=".json",
            filter_extension=app_data.get("current_filter")
        )

        # Get current simulation config from UI (doesn't interact with sim_thread for this)
        sim_config_dict_to_save = _update_sim_config_from_ui(update_model_specific=True)
        if not sim_config_dict_to_save:
            update_status_bar("Error: Could not retrieve current config from UI to save profile.", color=[255,0,0,255], level="error")
            return

        # Remove runtime state keys that shouldn't be in a profile
        keys_to_remove_from_profile = ["neuron_positions_x", "neuron_positions_y", "neuron_types_list_for_viz",
                                       "current_time_ms", "current_time_step", "is_running", "is_paused", "max_delay_steps"]
        for key in keys_to_remove_from_profile:
            if key in sim_config_dict_to_save: del sim_config_dict_to_save[key]
        
        gui_settings_to_save = get_current_gui_configuration_dict() # Get current GUI/filter settings
        content_to_save = {"simulation_configuration": sim_config_dict_to_save, "gui_configuration": gui_settings_to_save}

        try:
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(content_to_save, f, indent=4, ensure_ascii=False)
            update_status_bar(f"Profile saved: {os.path.basename(filepath)}", color=[0,200,0,255], level="success")
            if dpg.does_item_exist("profile_name_input"): 
                dpg.set_value("profile_name_input", os.path.basename(filepath).replace(".json", ""))
            global_gui_state["current_profile_name"] = os.path.basename(filepath)
        except Exception as e: 
            update_status_bar(f"Error saving profile: {e}", color=[255,0,0,255], level="error")
    else: 
        update_status_bar("Save profile cancelled.", level="info")


# --- Full Profile Dropdown (auto-populated from simulation_profiles/*.json) ---
_FULL_PROFILE_MAP = {}  # display_name -> filepath, populated at startup and on refresh

def _scan_profile_directory():
    """Scans simulation_profiles/ for .json files and builds display_name -> filepath map.

    Reads _profile_metadata.name if present, otherwise derives a readable name from filename.
    Excludes auto_tuned_overrides.json (system file, not a user profile).
    """
    global _FULL_PROFILE_MAP
    profile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_profiles")
    if not os.path.isdir(profile_dir):
        return

    new_map = {"(None - use settings below)": ""}  # Default empty entry
    try:
        for fname in sorted(os.listdir(profile_dir)):
            if not fname.endswith(".json") or fname == "auto_tuned_overrides.json":
                continue
            fpath = os.path.join(profile_dir, fname)
            # Try to extract a human-readable name from metadata
            display = fname.replace(".json", "").replace("_", " ").title()
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                meta = data.get("_profile_metadata", {})
                if meta.get("name"):
                    display = meta["name"]
            except Exception:
                pass  # Fall back to filename-derived name
            new_map[display] = fpath
    except Exception as e:
        print(f"Warning: Could not scan profile directory: {e}")

    _FULL_PROFILE_MAP = new_map


# =============================================================================
# EXPERIMENT SYSTEM UI CALLBACKS
# =============================================================================

def _handle_experiment_preset_change(preset_name):
    """Callback when user selects an experiment preset from the dropdown."""
    if not preset_name or preset_name == "-- Select Preset --":
        return
    ui_to_sim_queue.put({"type": "LOAD_EXPERIMENT_PRESET", "preset_name": preset_name})
    update_status_bar(f"Loading experiment preset: {preset_name}", color=[100, 200, 255, 255])


def _handle_inject_manual_stimulus(sender=None, app_data=None, user_data=None):
    """Inject a quick manual stimulus using a basic experiment config."""
    try:
        amplitude = dpg.get_value("manual_stim_amplitude")
        pattern_str = dpg.get_value("manual_stim_pattern_combo")
        group_size = dpg.get_value("manual_stim_group_size")
        duration = dpg.get_value("manual_stim_duration")

        # Build a simple experiment config for manual injection
        exp_config = ExperimentConfig(
            name="Manual Stimulus Injection",
            description=f"Quick {pattern_str} stimulus: {amplitude} pA for {duration} ms",
            neuron_groups=[
                NeuronGroup(name="stim_target", role=NeuronGroupRole.INPUT.name,
                           index_start=0, index_end=group_size,
                           highlight_color=[0.0, 1.0, 0.0, 1.0]),
                NeuronGroup(name="network_response", role=NeuronGroupRole.OUTPUT.name,
                           index_start=group_size, index_end=group_size * 3,
                           highlight_color=[1.0, 0.5, 0.0, 1.0]),
            ],
            stimulus_channels=[
                StimulusChannel(
                    name="manual_stim",
                    pattern=StimulusPattern(
                        pattern_type=pattern_str,
                        amplitude_pA=amplitude,
                    ),
                    target_group_name="stim_target",
                    onset_ms=100.0,
                    duration_ms=duration,
                ),
            ],
            phases=[
                ExperimentPhase(name="pre_baseline", phase_type=ExperimentPhaseType.BASELINE.name,
                               duration_ms=500.0, active_channels=[]),
                ExperimentPhase(name="stimulus", phase_type=ExperimentPhaseType.STIMULUS.name,
                               duration_ms=duration + 200.0,
                               active_channels=["manual_stim"]),
                ExperimentPhase(name="post_baseline", phase_type=ExperimentPhaseType.BASELINE.name,
                               duration_ms=1000.0, active_channels=[]),
            ],
            readout=ReadoutConfig(
                rate_window_ms=50.0,
                rate_group_names=["stim_target", "network_response"],
            ),
            enabled=True,
        )

        config_dict = experiment_config_to_dict(exp_config)
        ui_to_sim_queue.put({"type": "LOAD_EXPERIMENT_CONFIG", "config_dict": config_dict})
        # Auto-start after a brief delay to allow initialization
        ui_to_sim_queue.put({"type": "START_EXPERIMENT"})
        update_status_bar(f"Injecting {pattern_str} stimulus: {amplitude} pA", color=[100, 255, 100, 255])
    except Exception as e:
        update_status_bar(f"Stimulus injection error: {e}", color=[255, 100, 100, 255])


def _update_experiment_ui_from_status(experiment_status):
    """Update experiment UI elements from status dict (called from UI thread)."""
    if experiment_status is None:
        return

    try:
        is_running = experiment_status.get("is_running", False)
        is_complete = experiment_status.get("is_complete", False)

        # Status text
        if is_complete:
            dpg.set_value("experiment_status_text", "COMPLETE")
            dpg.configure_item("experiment_status_text", color=[100, 255, 100])
        elif is_running:
            dpg.set_value("experiment_status_text", "RUNNING")
            dpg.configure_item("experiment_status_text", color=[255, 255, 100])
        else:
            dpg.set_value("experiment_status_text", "Idle")
            dpg.configure_item("experiment_status_text", color=[150, 150, 150])

        # Phase info
        phase_name = experiment_status.get("current_phase_name", "--")
        phase_type = experiment_status.get("current_phase_type", "--")
        phase_idx = experiment_status.get("current_phase_idx", 0)
        total_phases = experiment_status.get("total_phases", 0)
        rep = experiment_status.get("phase_repetition", 0)
        dpg.set_value("experiment_phase_text",
                       f"Phase: {phase_name} ({phase_type}) [{phase_idx+1}/{total_phases}] rep={rep}")

        # Readout rates
        rates = experiment_status.get("readout_rates", {})
        if rates:
            rate_lines = [f"  {name}: {rate:.1f} Hz" for name, rate in rates.items()]
            dpg.set_value("experiment_readout_text", "\n".join(rate_lines))
        else:
            dpg.set_value("experiment_readout_text", "No data")

        # Training info
        training = experiment_status.get("training")
        if training and training.get("mode", "NONE") != "NONE":
            trials_done = training.get("trials_completed", 0)
            total_trials = training.get("total_trials", 0)
            accuracy = training.get("recent_accuracy", 0.0)
            converged = training.get("is_converged", False)
            status_str = f"Trial {trials_done}/{total_trials} | Accuracy: {accuracy:.1%}"
            if converged:
                status_str += " [CONVERGED]"
            dpg.set_value("experiment_training_text", status_str)
        else:
            dpg.set_value("experiment_training_text", "No training active")

    except Exception:
        pass  # UI elements may not exist yet during startup


def _handle_full_profile_dropdown_change(sender, app_data, user_data=None):
    """Callback when user selects a full profile from the dropdown."""
    if not app_data or app_data == "(None - use settings below)":
        return

    filepath = _FULL_PROFILE_MAP.get(app_data, "")
    if not filepath or not os.path.exists(filepath):
        update_status_bar(f"Profile file not found for '{app_data}'", color=[255,100,0,255], level="warning")
        return

    _execute_profile_load_on_ui_thread(filepath)

    # After loading, update the dropdown to reflect current selection (don't reset to None)
    # The profile is now applied — user sees it in the dropdown


def _refresh_full_profile_dropdown():
    """Rescans the profile directory and updates the dropdown items."""
    _scan_profile_directory()
    if dpg.is_dearpygui_running() and dpg.does_item_exist("cfg_full_profile"):
        items = list(_FULL_PROFILE_MAP.keys())
        dpg.configure_item("cfg_full_profile", items=items)


def _execute_profile_load_on_ui_thread(filepath): # Profiles are JSON
    """
    Loads a profile file, updates UI, and sends new config to sim_thread.
    Called by UI thread.
    """
    profile_name = os.path.basename(filepath)
    update_status_bar(f"Loading profile '{profile_name}'...", level="info")
    try:
        with open(filepath, 'r', encoding='utf-8') as f: profile_content = json.load(f)
        sim_cfg_data_from_profile = profile_content.get("simulation_configuration")
        gui_cfg_data_from_profile = profile_content.get("gui_configuration")

        if sim_cfg_data_from_profile:
            # Populate UI elements from the loaded simulation configuration part of the profile
            _populate_ui_from_config_dict(sim_cfg_data_from_profile)
            
            # Apply GUI settings from the profile
            if gui_cfg_data_from_profile: 
                apply_gui_configuration_core(gui_cfg_data_from_profile) # Updates filters, GL config etc.

            # Now that UI is updated, get the full config from UI to send to sim_thread for reset
            # This ensures any defaults or interpretations by _populate_ui are captured.
            final_config_to_apply_to_sim = _update_sim_config_from_ui(update_model_specific=True)
            if final_config_to_apply_to_sim:
                ui_to_sim_queue.put({
                    "type": "APPLY_CONFIG_AND_RESET",
                    "config_dict": final_config_to_apply_to_sim
                })
                update_status_bar(f"Profile '{profile_name}' loaded. Config sent for reset.", color=[0,200,0,255], level="success")
                if dpg.does_item_exist("profile_name_input"):
                    dpg.set_value("profile_name_input", profile_name.replace(".json", ""))
                global_gui_state["current_profile_name"] = profile_name
                # Update full profile dropdown to reflect loaded profile
                if dpg.does_item_exist("cfg_full_profile"):
                    # Find the display name matching this filepath
                    for display_name, fpath in _FULL_PROFILE_MAP.items():
                        if fpath and os.path.normpath(fpath) == os.path.normpath(filepath):
                            dpg.set_value("cfg_full_profile", display_name)
                            break
                global_gui_state["reset_sim_needed_from_ui_change"] = False # Reset is being handled
            else:
                update_status_bar("Error creating final config from UI after profile load.", color=[255,0,0,255], level="error")
        else: 
            update_status_bar("Error: Profile missing 'simulation_configuration'.", color=[255,0,0,255], level="error")
    except Exception as e:
        update_status_bar(f"Error loading profile: {e}", color=[255,0,0,255], level="error"); import traceback; traceback.print_exc()

def load_profile_dialog_callback(sender, app_data): # Profiles are JSON
    """Callback for 'Load Profile' dialog. Calls helper to load and apply."""
    if "file_path_name" in app_data and app_data["file_path_name"]:
        _execute_profile_load_on_ui_thread(app_data["file_path_name"])
    else: 
        update_status_bar("Load profile cancelled.", level="info")


def handle_save_checkpoint_button_press(sender, app_data, user_data): # Checkpoints are HDF5
    """Shows 'Save Checkpoint' file dialog. Command sent to sim_thread from callback."""
    if global_gui_state.get("is_playback_mode_active", False):
        update_status_bar("Error: Cannot save checkpoint in playback mode.", color=[255,0,0,255], level="error")
        return
    if dpg.is_dearpygui_running() and dpg.does_item_exist("save_checkpoint_file_dialog_h5"):
        dpg.show_item("save_checkpoint_file_dialog_h5")

def save_checkpoint_dialog_callback_h5(sender, app_data): # Checkpoints are HDF5
    """Callback for 'Save Checkpoint'. Ensures correct extension."""
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = _normalize_filepath_extension(
            app_data["file_path_name"],
            required_extension=".simstate.h5",
            filter_extension=app_data.get("current_filter")
        )

        current_gui_config_for_checkpoint = get_current_gui_configuration_dict()
        ui_to_sim_queue.put({
            "type": "SAVE_CHECKPOINT",
            "filepath": filepath,
            "gui_config_snapshot": current_gui_config_for_checkpoint
            })
        update_status_bar(f"Save checkpoint command sent for: {os.path.basename(filepath)}", level="info")
    else:
        update_status_bar("Save checkpoint cancelled.", level="info")

def handle_load_checkpoint_button_press(sender, app_data, user_data): # Checkpoints are HDF5
    """Shows 'Load Checkpoint' file dialog. Command sent to sim_thread from callback."""
    if global_gui_state.get("is_recording_active", False) or global_gui_state.get("is_playback_mode_active", False):
        update_status_bar("Stop recording/playback before loading a checkpoint.", color=[255,165,0,255], level="warning")
        return
    if dpg.is_dearpygui_running() and dpg.does_item_exist("load_checkpoint_file_dialog_h5"):
        dpg.show_item("load_checkpoint_file_dialog_h5")

def load_checkpoint_dialog_callback_h5(sender, app_data):
    """Callback for 'Load Checkpoint' dialog. Sends command to sim_thread."""
    filepath_to_load = None
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = app_data["file_path_name"]
        selected_filter = app_data.get("current_filter", "")

        # If the ".*" filter was active, DPG might append ".*" to the actual filename.
        # We need to strip this if the file doesn't literally end with ".*".
        if selected_filter == ".*" and filepath.endswith(".*"):
            potential_filepath_stripped = filepath[:-2]
            # Check if the stripped version is the actual file
            if os.path.isfile(potential_filepath_stripped):
                filepath = potential_filepath_stripped
            # If not, and the original path with ".*" is a file (rare), use it.
            # Otherwise, it's likely an invalid construction by DPG.
            elif not os.path.isfile(filepath): # if "file.simstate.h5.*" is NOT a file
                 update_status_bar(f"Load error: Path '{filepath}' from '.*' filter seems invalid.", color=[255,0,0,255], level="error")
                 return


        # At this point, filepath should be the intended file.
        if os.path.isfile(filepath):
            filepath_to_load = filepath
        elif os.path.isdir(filepath):
            update_status_bar(f"Error: Selected path is a directory: {filepath}", color=[255,100,0,255], level="warning")
            return
        else:
            update_status_bar(f"Load error: File not found or invalid path: '{filepath}'.", color=[255,0,0,255], level="error")
            return

    elif "file_name" in app_data and app_data["file_name"] and "current_path" in app_data: # Fallback
        filepath = os.path.join(app_data["current_path"], app_data["file_name"])
        if os.path.isfile(filepath):
            filepath_to_load = filepath
        else:
            update_status_bar(f"Error: Fallback path is not a valid file: {filepath}", color=[255,0,0,255], level="error")
            return
    else:
        update_status_bar("Load checkpoint cancelled or no file selected.", level="info")
        return

    if filepath_to_load:
        ui_to_sim_queue.put({"type": "LOAD_CHECKPOINT", "filepath": filepath_to_load})
        update_status_bar(f"Load checkpoint command sent for: {os.path.basename(filepath_to_load)}", level="info")

def get_current_gui_configuration_dict():
    """
    Gets current GUI settings, including filters and OpenGL viz config.
    Called by UI thread, e.g., when saving a profile or checkpoint.
    """
    dpg_filters = get_current_filter_settings_from_gui() 
    dpg_filters["show_synapses_cb"] = global_gui_state.get("show_connections_gl", False) # From global_gui_state

    current_gl_config = opengl_viz_config.copy() if OPENGL_AVAILABLE else {}
    # If sim_bridge instance is available and has camera_fov (it's part of sim_config)
    # This is tricky as sim_config in sim_bridge might not be in sync if UI changed it.
    # Best to get FOV from DPG UI if it's there.
    if dpg.is_dearpygui_running() and dpg.does_item_exist("cfg_camera_fov"):
         current_gl_config["CAMERA_FOV_DPG_Snapshot"] = dpg.get_value("cfg_camera_fov")
    elif global_simulation_bridge and hasattr(global_simulation_bridge, 'viz_config'): # Fallback
         current_gl_config["CAMERA_FOV_DPG_Snapshot"] = global_simulation_bridge.viz_config.camera_fov


    return {"filter_settings": dpg_filters, "opengl_visualization_settings": current_gl_config}

def apply_gui_configuration_core(gui_cfg_dict):
    """
    Applies a dictionary of GUI settings to the DPG UI elements.
    Called by UI thread, e.g., after loading a profile or checkpoint that includes GUI settings.
    """
    if not gui_cfg_dict or not dpg.is_dearpygui_running(): return False

    filter_settings = gui_cfg_dict.get("filter_settings", {})
    if dpg.does_item_exist("filter_spiking_mode_combo"): 
        dpg.set_value("filter_spiking_mode_combo", filter_settings.get("spiking_mode", "Highlight Spiking"))

    type_filter_enabled = filter_settings.get("type_filter_enabled", False)
    if dpg.does_item_exist("filter_type_enable_cb"): dpg.set_value("filter_type_enable_cb", type_filter_enabled)
    if dpg.does_item_exist("filter_neuron_type_combo"):
        dpg.configure_item("filter_neuron_type_combo", enabled=type_filter_enabled) 
        # Populate items for filter_neuron_type_combo based on current model (sim_bridge needed or default list)
        # This part is tricky if sim_config is not yet aligned with the profile's model type.
        # Assume _populate_ui_from_config_dict has already set the model type.
        # Then, we can get available types.
        available_types = []
        if global_simulation_bridge and hasattr(global_simulation_bridge, 'get_available_neuron_types'):
            available_types = global_simulation_bridge.get_available_neuron_types()
        elif dpg.does_item_exist("cfg_neuron_model_type"): # Fallback if sim_bridge not ready
            model_name = dpg.get_value("cfg_neuron_model_type")
            temp_cfg_types = SimulationConfiguration(); temp_cfg_types.neuron_model_type = model_name
            if model_name == NeuronModel.IZHIKEVICH.name: available_types = ["All"] + [f"Izh2007_{nt.name.replace('IZH2007_', '')}" for nt in NeuronType if "IZH2007" in nt.name]
            elif model_name == NeuronModel.HODGKIN_HUXLEY.name: available_types = ["All"] + [f"HH_{nt.name.replace('HH_', '')}" for nt in NeuronType if "HH_" in nt.name]
            else: available_types = ["All"]
            available_types = list(dict.fromkeys(available_types))


        dpg.configure_item("filter_neuron_type_combo", items=available_types)
        selected_type = filter_settings.get("selected_neuron_type", "All")
        if selected_type in available_types: dpg.set_value("filter_neuron_type_combo", selected_type)
        elif "All" in available_types: dpg.set_value("filter_neuron_type_combo", "All")
        elif available_types: dpg.set_value("filter_neuron_type_combo", available_types[0])
            
    if dpg.does_item_exist("filter_min_abs_weight_slider"): dpg.set_value("filter_min_abs_weight_slider", filter_settings.get("min_abs_weight", 0.01))

    show_syn_val = filter_settings.get("show_synapses_cb", global_gui_state.get("show_connections_gl", False))
    global_gui_state["show_connections_gl"] = show_syn_val 
    if dpg.does_item_exist("filter_show_synapses_gl_cb"): dpg.set_value("filter_show_synapses_gl_cb", show_syn_val)

    if OPENGL_AVAILABLE:
        loaded_gl_settings = gui_cfg_dict.get("opengl_visualization_settings")
        if loaded_gl_settings: opengl_viz_config.update(loaded_gl_settings) # Update the global GL config dict

        # Apply these settings to DPG widgets for GL config
        gl_settings_to_apply_to_dpg = [
            ("gl_neuron_point_size_slider", 'POINT_SIZE', 2.0),
            ("gl_synapse_alpha_slider", 'SYNAPSE_ALPHA_MODIFIER', 0.3),
            ("gl_max_neurons_render_input", 'MAX_NEURONS_TO_RENDER', 10000),
            ("gl_max_connections_render_input", 'MAX_CONNECTIONS_TO_RENDER', 20000),
            ("gl_inactive_neuron_opacity_slider", 'INACTIVE_NEURON_OPACITY', 0.25),
            ("gl_activity_highlight_frames_input", 'ACTIVITY_HIGHLIGHT_FRAMES', 7),
            ("gl_enable_synaptic_pulses_cb", 'ENABLE_SYNAPTIC_PULSES', True)
        ]
        for tag, key, default_val in gl_settings_to_apply_to_dpg:
            if dpg.does_item_exist(tag): dpg.set_value(tag, opengl_viz_config.get(key, default_val))
        
        # Apply camera FOV if it was in the GUI settings snapshot
        if dpg.does_item_exist("cfg_camera_fov") and "CAMERA_FOV_DPG_Snapshot" in opengl_viz_config:
            dpg.set_value("cfg_camera_fov", opengl_viz_config["CAMERA_FOV_DPG_Snapshot"])
        # If not in snapshot, _populate_ui_from_config_dict would have set it from main sim_config.

    trigger_filter_update_signal(); # Filters or GL settings might have changed
    update_status_bar("GUI settings applied from profile/checkpoint.", level="info")
    return True


def update_monitoring_overlay_values(sim_data_dict):
    """
    Updates the DPG monitoring text elements with current simulation data.
    Called by the main/UI thread when new data arrives from sim_to_ui_queue.
    Note: Most monitoring data is now displayed in the OpenGL HUD. This function
    only updates the playback frame counter.
    """
    if not dpg.is_dearpygui_running(): return

    # Update playback frame counter if in playback mode
    if global_gui_state.get("is_playback_mode_active") and dpg.does_item_exist("playback_current_frame_text"):
        active_rec_meta = global_gui_state.get("active_recording_data_source") # This is UI thread's copy
        if active_rec_meta and "num_frames" in active_rec_meta:
            total_frames = active_rec_meta["num_frames"]
            current_frame_idx_ui = global_gui_state.get("current_playback_frame_index",0) # UI's current frame
            dpg.set_value("playback_current_frame_text", f"Frame: {current_frame_idx_ui + 1} / {total_frames if total_frames > 0 else 1}")

# --- DPG Event Handlers for Recording & Playback (HDF5) ---

def _recording_options_continue_callback(sender=None, app_data=None, user_data=None):
    """Called when user clicks Continue in the recording options popup."""
    # Read options from the popup and update sim_bridge's gpu_config
    recording_mode = dpg.get_value("rec_opt_mode_combo")
    skip_synaptic = dpg.get_value("rec_opt_skip_synaptic")
    frame_skip = dpg.get_value("rec_opt_frame_skip")

    # Send options to sim_bridge via command queue
    ui_to_sim_queue.put({
        "type": "SET_RECORDING_OPTIONS",
        "recording_mode": recording_mode,
        "recording_skip_synaptic_data": skip_synaptic,
        "recording_frame_skip": frame_skip
    })

    # Hide the options popup
    if dpg.does_item_exist("recording_options_popup"):
        dpg.hide_item("recording_options_popup")

    # Show the file dialog
    if dpg.is_dearpygui_running() and dpg.does_item_exist("save_recording_file_dialog_h5"):
        dpg.show_item("save_recording_file_dialog_h5")

def _recording_options_cancel_callback(sender=None, app_data=None, user_data=None):
    """Called when user cancels the recording options popup."""
    if dpg.does_item_exist("recording_options_popup"):
        dpg.hide_item("recording_options_popup")
    update_status_bar("Recording cancelled.", level="info")

def handle_record_button_click(sender=None, app_data=None, user_data=None):
    """
    Handles the 'Record' / 'Finalize Recording' button click.
    Shows recording options popup or sends command to stop recording.
    """
    if global_gui_state.get("is_recording_active", False):  # If currently recording, this button means "Finalize"
        ui_to_sim_queue.put({"type": "STOP_RECORDING"})
        update_status_bar("Finalize recording command sent...", level="info")
        # UI state will be updated when sim_thread confirms via "RECORDING_FINALIZED"
    else:  # Not recording, this button means "Record" - show options popup
        if global_gui_state.get("is_playback_mode_active", False):
            update_status_bar("Error: Cannot record while in playback mode.", color=[255,0,0,255], level="error")
            return
        if dpg.is_dearpygui_running() and dpg.does_item_exist("recording_options_popup"):
            dpg.show_item("recording_options_popup")
        else:
            update_status_bar("Error: Recording options dialog missing.", color=[255,0,0,255], level="error")

def save_recording_for_streaming_dialog_callback_h5(sender, app_data):
    """Callback for the 'Record' (Save Recording As) file dialog. Ensures correct extension."""
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = _normalize_filepath_extension(
            app_data["file_path_name"],
            required_extension=".simrec.h5",
            filter_extension=app_data.get("current_filter")
        )

        ui_to_sim_queue.put({"type": "START_RECORDING", "filepath": filepath})
        update_status_bar(f"Start recording command sent for: {os.path.basename(filepath)}", level="info")
    else:
        update_status_bar("Recording setup cancelled.", level="info")


def handle_playback_button_click(sender=None, app_data=None, user_data=None):
    """
    Handles the 'Playback Recording' / 'Stop Playback' button click.
    Sends command to sim_thread to enter or exit playback mode.
    """
    if global_gui_state.get("is_playback_mode_active", False): # If in playback, stop it
        ui_to_sim_queue.put({"type": "EXIT_PLAYBACK_MODE"})
        update_status_bar("Exit playback command sent...", level="info")
    else: # Not in playback, try to enter
        if global_gui_state.get("is_recording_active", False):
            update_status_bar("Error: Finalize active recording before entering playback.", color=[255,165,0,255], level="warning")
            return

        loaded_data_meta = global_gui_state.get("active_recording_data_source")
        if not loaded_data_meta or not loaded_data_meta.get("h5_file_obj_for_playback") or \
           not loaded_data_meta.get("h5_file_obj_for_playback").id: # Check if a valid recording is loaded in UI state
            update_status_bar("No valid recording loaded. Load one first via File menu.", color=[255,165,0,255], level="warning")
            return

        # Command sim_thread to setup for playback
        ui_to_sim_queue.put({
            "type": "SETUP_PLAYBACK_FROM_RECORDING",
            "config_snapshot": loaded_data_meta["config_snapshot"],
            "h5_file_handle_for_sim_thread": loaded_data_meta["h5_file_obj_for_playback"], # *** ADD THIS LINE ***
            "initial_state_group_name": "initial_state" # Usually "initial_state"
        })
        update_status_bar("Enter playback mode command sent...", level="info")


def handle_load_recording_menu_click(sender=None, app_data=None, user_data=None):
    """Handles the 'File > Load Recording' menu item click. Shows file dialog."""
    if global_gui_state.get("is_recording_active", False):
        update_status_bar("Finalize current recording before loading another.", color=[255,165,0,255], level="warning")
        return
    if global_gui_state.get("is_playback_mode_active", False): 
        # If already in playback, command sim_thread to exit first, then show dialog after confirmation.
        # This makes the flow cleaner. For now, just warn.
        update_status_bar("Exit current playback mode before loading a new recording.", color=[255,165,0,255], level="warning")
        # A better flow: send EXIT_PLAYBACK, then on confirmation, show dialog.
        # ui_to_sim_queue.put({"type": "EXIT_PLAYBACK_MODE", "then_show_dialog": "load_recording_file_dialog_h5"})
        return


    if dpg.is_dearpygui_running() and dpg.does_item_exist("load_recording_file_dialog_h5"):
        dpg.show_item("load_recording_file_dialog_h5")

def _normalize_load_filepath(filepath, filter_extension=None):
    """
    Normalizes a filepath from a load dialog by stripping filter artifacts.

    Args:
        filepath: The filepath from the file dialog
        filter_extension: The filter extension that may have been appended (e.g., ".*", ".h5")

    Returns:
        Cleaned filepath
    """
    # Strip ".*" if DPG appended it from "All Files" filter
    if filepath.endswith(".*"):
        filepath = filepath[:-2]

    # Strip filter extension if it was appended to a valid file path
    if filter_extension and filter_extension not in [".*", ""]:
        if filepath.endswith(filter_extension):
            potential_path = filepath[:-len(filter_extension)]
            if os.path.isfile(potential_path):
                filepath = potential_path

    return filepath

def _estimate_recording_memory_requirements(filepath):
    """
    Estimates the GPU memory required to cache a recording.

    Returns:
        tuple: (num_frames, estimated_bytes, fits_in_vram, available_vram_bytes, vram_limit_pct)
               or (None, None, None, None, None) if estimation fails
    """
    try:
        import h5py

        # Get available GPU memory
        if not cp:
            return None, None, None, None, None

        mem_info = cp.cuda.Device().mem_info
        free_memory, total_memory = mem_info
        vram_limit_pct = 0.90  # Use 90% of available VRAM
        usable_memory = free_memory * vram_limit_pct

        # Open file briefly to estimate size
        with h5py.File(filepath, 'r') as h5_file:
            frames_group = h5_file.get("frames")
            if not frames_group:
                return None, None, None, None, None

            num_frames = len(frames_group.keys())
            if num_frames == 0:
                return 0, 0, True, free_memory, vram_limit_pct

            # Sample first frame to estimate per-frame size
            first_frame_key = f"frame_0"
            first_frame = frames_group.get(first_frame_key)
            if not first_frame:
                # Try to find any frame
                frame_keys = list(frames_group.keys())
                if frame_keys:
                    first_frame = frames_group.get(frame_keys[0])

            if not first_frame:
                return num_frames, None, None, free_memory, vram_limit_pct

            # Estimate frame size from datasets
            frame_size_bytes = 0
            for key in first_frame.keys():
                dataset = first_frame[key]
                if hasattr(dataset, 'shape') and hasattr(dataset, 'dtype'):
                    frame_size_bytes += np.prod(dataset.shape) * dataset.dtype.itemsize

            # Add overhead for CuPy arrays (~10%)
            frame_size_bytes = int(frame_size_bytes * 1.1)

            total_estimated_bytes = frame_size_bytes * num_frames
            fits_in_vram = total_estimated_bytes <= usable_memory

            return num_frames, total_estimated_bytes, fits_in_vram, free_memory, vram_limit_pct

    except Exception as e:
        print(f"Error estimating recording memory: {e}")
        return None, None, None, None, None

def _show_recording_memory_warning_popup(filepath, num_frames, estimated_bytes, available_bytes):
    """Shows a popup warning that the recording won't fit in VRAM."""
    global_gui_state["_pending_recording_filepath"] = filepath

    estimated_gb = estimated_bytes / 1e9
    available_gb = available_bytes / 1e9
    pct_of_vram = (estimated_bytes / available_bytes) * 100 if available_bytes > 0 else 0

    # Update popup text
    if dpg.does_item_exist("recording_memory_warning_text"):
        dpg.set_value("recording_memory_warning_text",
            f"The selected recording ({num_frames} frames) is estimated to require\n"
            f"{estimated_gb:.2f} GB of GPU memory, but only {available_gb:.2f} GB is available.\n"
            f"(Recording is ~{pct_of_vram:.0f}% of available VRAM)\n\n"
            f"How would you like to proceed?"
        )

    if dpg.does_item_exist("recording_memory_warning_popup"):
        dpg.show_item("recording_memory_warning_popup")

def _recording_memory_popup_partial_cache(sender=None, app_data=None):
    """Callback for 'Partial Cache' button in memory warning popup."""
    filepath = global_gui_state.get("_pending_recording_filepath")
    if dpg.does_item_exist("recording_memory_warning_popup"):
        dpg.hide_item("recording_memory_warning_popup")

    if filepath:
        ui_to_sim_queue.put({
            "type": "LOAD_RECORDING",
            "filepath": filepath,
            "stream_only": False  # Will auto-stop caching when memory limit reached
        })
        update_status_bar(f"Load recording (partial cache) command sent for: {os.path.basename(filepath)}", level="info")

def _recording_memory_popup_stream_only(sender=None, app_data=None):
    """Callback for 'Stream Only' button in memory warning popup."""
    filepath = global_gui_state.get("_pending_recording_filepath")
    if dpg.does_item_exist("recording_memory_warning_popup"):
        dpg.hide_item("recording_memory_warning_popup")

    if filepath:
        ui_to_sim_queue.put({
            "type": "LOAD_RECORDING",
            "filepath": filepath,
            "stream_only": True
        })
        update_status_bar(f"Load recording (streaming) command sent for: {os.path.basename(filepath)}", level="info")

def _recording_memory_popup_cancel(sender=None, app_data=None):
    """Callback for 'Cancel' button in memory warning popup."""
    if dpg.does_item_exist("recording_memory_warning_popup"):
        dpg.hide_item("recording_memory_warning_popup")
    update_status_bar("Recording load cancelled.", level="info")

def load_recording_dialog_callback_h5(sender, app_data):
    """Callback for the 'Load Recording' file dialog. Sends command to sim_thread."""
    filepath_to_load = None
    if "file_path_name" in app_data and app_data["file_path_name"]:
        filepath = _normalize_load_filepath(
            app_data["file_path_name"],
            filter_extension=app_data.get("current_filter")
        )

        if os.path.isfile(filepath):
            filepath_to_load = filepath
        elif os.path.isdir(filepath):
            update_status_bar(f"Error: Selected path is a directory: {filepath}", color=[255,100,0,255], level="warning")
            return
        else:
            update_status_bar(f"Load error: File not found or invalid path: '{filepath}'.", color=[255,0,0,255], level="error")
            return

    elif "file_name" in app_data and app_data["file_name"] and "current_path" in app_data: # Fallback
        filepath = os.path.join(app_data["current_path"], app_data["file_name"])
        if os.path.isfile(filepath):
            filepath_to_load = filepath
        else:
            update_status_bar(f"Error: Fallback path is not a valid file: {filepath}", color=[255,0,0,255], level="error")
            return
    else:
        update_status_bar("Load recording cancelled or no file selected.", level="info")
        return

    if filepath_to_load:
        # Check if recording fits in VRAM
        num_frames, estimated_bytes, fits_in_vram, available_bytes, _ = _estimate_recording_memory_requirements(filepath_to_load)

        if fits_in_vram is None:
            # Couldn't estimate, just proceed with caching attempt
            ui_to_sim_queue.put({
                "type": "LOAD_RECORDING",
                "filepath": filepath_to_load,
                "stream_only": False
            })
            update_status_bar(f"Load recording (caching) command sent for: {os.path.basename(filepath_to_load)}", level="info")
        elif fits_in_vram:
            # Recording fits, proceed with caching
            ui_to_sim_queue.put({
                "type": "LOAD_RECORDING",
                "filepath": filepath_to_load,
                "stream_only": False
            })
            estimated_gb = estimated_bytes / 1e9 if estimated_bytes else 0
            update_status_bar(f"Load recording (caching ~{estimated_gb:.1f}GB) command sent for: {os.path.basename(filepath_to_load)}", level="info")
        else:
            # Recording won't fit, show warning popup
            _show_recording_memory_warning_popup(filepath_to_load, num_frames, estimated_bytes, available_bytes)

def handle_playback_slider_change(sender, frame_idx_from_slider_float, user_data=None):
    """Handles playback slider changes. Sends command to sim_thread to set frame if handle is valid."""
    frame_idx_from_slider = int(frame_idx_from_slider_float)
    if not global_gui_state.get("is_playback_mode_active", False):
        # This should not happen if controls are correctly disabled, but as a safeguard:
        update_status_bar("Playback not active. Cannot seek.", level="warning")
        return

    if global_gui_state.get("playback_is_playing_ui", False): # If user moves slider while playing, pause.
        global_gui_state["playback_is_playing_ui"] = False
        update_ui_for_playback_playing_state(is_playing=False)
        update_status_bar("Playback paused for manual seek.", level="info")

    loaded_data_meta = global_gui_state.get("active_recording_data_source")
    h5_handle = None
    if loaded_data_meta:
        h5_handle = loaded_data_meta.get("h5_file_obj_for_playback")

    if h5_handle and hasattr(h5_handle, 'id') and h5_handle.id: # Check if handle is valid and open
        num_frames = loaded_data_meta.get("num_frames") if loaded_data_meta else None
        ui_to_sim_queue.put({
            "type": "SET_PLAYBACK_FRAME",
            "frame_index": frame_idx_from_slider,
            "h5_file_handle_for_sim_thread": h5_handle,
            "num_frames": num_frames
        })
        # Status update for successful command send can be minimal or handled by sim thread ACK
        # update_status_bar(f"Seek to frame {frame_idx_from_slider+1} command sent.", level="debug")
    else:
        # This else block means the command will NOT be sent, preventing the error in sim_thread.
        if not loaded_data_meta:
            update_status_bar("Error: No active recording data source for playback seek.", color=[255,0,0,255], level="error")
        else: # loaded_data_meta exists, but handle is bad
            update_status_bar("Error: HDF5 handle for playback is invalid or closed. Cannot seek.", color=[255,0,0,255], level="error")


def handle_playback_play_pause_button_click(sender=None, app_data=None, user_data=None):
    """Handles the Play/Pause button for playback controls."""
    if not global_gui_state.get("is_playback_mode_active", False): return

    new_playing_state = not global_gui_state.get("playback_is_playing_ui", False)
    global_gui_state["playback_is_playing_ui"] = new_playing_state
    update_ui_for_playback_playing_state(is_playing=new_playing_state) # Update button label

    if new_playing_state:
        global_gui_state["last_playback_autostep_time_ui"] = time.perf_counter() # Reset timer for UI-driven stepping
        # If at the end of playback, loop back to the beginning by commanding frame 0
        active_rec_meta = global_gui_state.get("active_recording_data_source")
        if active_rec_meta and "num_frames" in active_rec_meta:
            num_frames = active_rec_meta["num_frames"]
            h5_handle = active_rec_meta.get("h5_file_obj_for_playback")
            current_frame_ui = global_gui_state.get("current_playback_frame_index", 0)
            if num_frames > 0 and current_frame_ui >= num_frames - 1:
                ui_to_sim_queue.put({
                    "type": "SET_PLAYBACK_FRAME",
                    "frame_index": 0,
                    "h5_file_handle_for_sim_thread": h5_handle,
                    "num_frames": num_frames
                })
        update_status_bar("Playback started/resumed by UI.", level="info")
    else:
        update_status_bar("Playback paused by UI.", level="info")
    # The actual frame stepping for playback_is_playing_ui is handled in the main DPG loop.

def handle_playback_step_frames_click(sender, app_data, user_data):
    """Handles clicks for playback step buttons. Sends command to sim_thread if handle is valid."""
    if not global_gui_state.get("is_playback_mode_active", False):
        update_status_bar("Playback not active. Cannot step frames.", level="warning")
        return

    step_amount = user_data
    if not isinstance(step_amount, int):
        return

    if global_gui_state.get("playback_is_playing_ui", False):
        global_gui_state["playback_is_playing_ui"] = False
        update_ui_for_playback_playing_state(is_playing=False)
        update_status_bar("Playback paused for manual step.", level="info")

    current_frame_ui = global_gui_state.get("current_playback_frame_index", 0)
    active_rec_meta = global_gui_state.get("active_recording_data_source")
    num_frames = active_rec_meta.get("num_frames", 0) if active_rec_meta else 0

    new_frame_idx = current_frame_ui + step_amount
    if num_frames > 0:
        new_frame_idx = max(0, min(new_frame_idx, num_frames - 1))
    else:
        new_frame_idx = 0

    h5_handle = None
    if active_rec_meta:
        h5_handle = active_rec_meta.get("h5_file_obj_for_playback")

    if h5_handle and hasattr(h5_handle, 'id') and h5_handle.id: # Check if handle is valid and open
        ui_to_sim_queue.put({
            "type": "SET_PLAYBACK_FRAME",
            "frame_index": new_frame_idx,
            "h5_file_handle_for_sim_thread": h5_handle,
            "num_frames": num_frames
        })
        # update_status_bar(f"Step playback by {step_amount} (to frame {new_frame_idx+1}) command sent.", level="debug")
    else:
        # Command will NOT be sent.
        if not active_rec_meta:
            update_status_bar("Error: No active recording data source for playback step.", color=[255,0,0,255], level="error")
        else: # active_rec_meta exists, but handle is bad
            update_status_bar("Error: HDF5 handle for playback is invalid or closed. Cannot step.", color=[255,0,0,255], level="error")

# --- GUI Update Helper Functions for Recording/Playback States (Called by Main/UI Thread) ---

def update_ui_for_simulation_run_state(is_running, is_paused):
    """Updates DPG UI elements based on the simulation's run/pause state (UI perspective)."""
    if not dpg.is_dearpygui_running(): return

    is_playback_active_ui = global_gui_state.get("is_playback_mode_active", False)
    is_recording_active_ui = global_gui_state.get("is_recording_active", False)

    # Live Simulation Controls
    dpg.configure_item("start_button", enabled=not is_running and not is_playback_active_ui)
    dpg.configure_item("pause_button", enabled=is_running and not is_playback_active_ui, label="Resume" if is_paused else "Pause")
    dpg.configure_item("stop_button", enabled=is_running and not is_playback_active_ui)
    dpg.configure_item("step_button", enabled=(is_paused or not is_running) and not is_playback_active_ui)

    # Apply config button: disabled if sim is running (live), or in playback, or actively recording
    can_apply_config = not is_running and not is_playback_active_ui and not is_recording_active_ui
    dpg.configure_item("apply_config_button", enabled=can_apply_config)

    # Record button: label managed by update_ui_for_recording_state.
    # Enabled if not in playback.
    dpg.configure_item("record_button", enabled=not is_playback_active_ui)

    # Playback button: enabled if NOT actively recording AND
    # ( (a recording is loaded AND valid) OR playback is already active (to allow stopping it) )
    loaded_data_meta = global_gui_state.get("active_recording_data_source")
    is_valid_recording_loaded_ui = loaded_data_meta and loaded_data_meta.get("h5_file_obj_for_playback") # Simpler check for UI
    
    can_initiate_or_stop_playback = not is_recording_active_ui and (is_valid_recording_loaded_ui or is_playback_active_ui)
    dpg.configure_item("playback_button", enabled=can_initiate_or_stop_playback)
    dpg.set_item_label("playback_button", "Stop Playback" if is_playback_active_ui else "Playback Recording")


def update_ui_for_recording_state(is_recording_active_ui): # Based on UI's perspective
    """Updates UI elements related to recording state (e.g., Record button label)."""
    if not dpg.is_dearpygui_running(): return
    global_gui_state["is_recording_active"] = is_recording_active_ui # Update UI's view

    dpg.set_item_label("record_button", "Finalize Recording" if is_recording_active_ui else "Record")
    
    # Re-evaluate other controls based on this new recording state
    current_sim_running_ui = global_gui_state.get("_sim_is_running_ui_view", False)
    current_sim_paused_ui = global_gui_state.get("_sim_is_paused_ui_view", False)
    update_ui_for_simulation_run_state(is_running=current_sim_running_ui, is_paused=current_sim_paused_ui)


def update_ui_for_playback_mode_state(is_playback_active_ui, num_frames_from_meta=0):
    """Updates UI elements when entering or exiting active playback mode (UI perspective)."""
    if not dpg.is_dearpygui_running(): return
    global_gui_state["is_playback_mode_active"] = is_playback_active_ui

    # Main Playback/Stop Playback button label and state
    if dpg.does_item_exist("playback_button"):
        dpg.set_item_label("playback_button", "Stop Playback" if is_playback_active_ui else "Playback Recording")
    
    # Show/hide and enable/disable playback controls group elements
    if dpg.does_item_exist("playback_controls_group"):
        dpg.configure_item("playback_controls_group", show=is_playback_active_ui)

    step_buttons_enabled_ui = is_playback_active_ui
    if dpg.does_item_exist("playback_step_minus_5"): dpg.configure_item("playback_step_minus_5", enabled=step_buttons_enabled_ui)
    if dpg.does_item_exist("playback_step_minus_1"): dpg.configure_item("playback_step_minus_1", enabled=step_buttons_enabled_ui)
    if dpg.does_item_exist("playback_play_pause_button"): dpg.configure_item("playback_play_pause_button", enabled=step_buttons_enabled_ui)
    if dpg.does_item_exist("playback_step_plus_1"): dpg.configure_item("playback_step_plus_1", enabled=step_buttons_enabled_ui)
    if dpg.does_item_exist("playback_step_plus_5"): dpg.configure_item("playback_step_plus_5", enabled=step_buttons_enabled_ui)

    if is_playback_active_ui:
        slider_max = max(0, num_frames_from_meta - 1)
        if dpg.does_item_exist("playback_slider"):
            dpg.configure_item("playback_slider", max_value=slider_max, enabled=True)
            dpg.set_value("playback_slider", 0) # Reset slider to beginning
        if dpg.does_item_exist("playback_current_frame_text"):
             dpg.set_value("playback_current_frame_text", f"Frame: 1 / {num_frames_from_meta if num_frames_from_meta > 0 else 1}")
        
        global_gui_state["playback_is_playing_ui"] = False # Start paused
        update_ui_for_playback_playing_state(is_playing=False) 

        # Disable live simulation controls and conflicting file operations
        if dpg.does_item_exist("record_button"): dpg.configure_item("record_button", enabled=False)
        if dpg.does_item_exist("start_button"): dpg.configure_item("start_button", enabled=False)
        if dpg.does_item_exist("pause_button"): dpg.configure_item("pause_button", enabled=False, label="Pause")
        if dpg.does_item_exist("stop_button"): dpg.configure_item("stop_button", enabled=False)
        if dpg.does_item_exist("step_button"): dpg.configure_item("step_button", enabled=False)
        if dpg.does_item_exist("apply_config_button"): dpg.configure_item("apply_config_button", enabled=False)
        
        menu_items_to_disable = ["save_profile_menu", "load_profile_menu", 
                                 "save_checkpoint_menu_h5", "load_checkpoint_menu_h5",
                                 "load_recording_menu_h5"] # Can't load another recording while one is active for playback
        for item_tag in menu_items_to_disable:
            if dpg.does_item_exist(item_tag): dpg.configure_item(item_tag, enabled=False)
    else: # Exiting active playback mode
        # Enable live simulation controls and file operations
        # update_ui_for_simulation_run_state will handle most of these based on current sim state
        current_sim_running_ui = global_gui_state.get("_sim_is_running_ui_view", False)
        current_sim_paused_ui = global_gui_state.get("_sim_is_paused_ui_view", False)
        update_ui_for_simulation_run_state(is_running=current_sim_running_ui, is_paused=current_sim_paused_ui)

        menu_items_to_enable = ["save_profile_menu", "load_profile_menu", 
                                "save_checkpoint_menu_h5", "load_checkpoint_menu_h5",
                                "load_recording_menu_h5"]
        for item_tag in menu_items_to_enable:
            if dpg.does_item_exist(item_tag): dpg.configure_item(item_tag, enabled=True)


def update_ui_for_playback_playing_state(is_playing): # UI's perspective of playback auto-play
    """Updates the Play/Pause button label within the playback controls."""
    if dpg.is_dearpygui_running() and dpg.does_item_exist("playback_play_pause_button"):
         dpg.set_item_label("playback_play_pause_button", "Pause Playback" if is_playing else "Play Recording")

def update_ui_after_recording_loaded(loaded_meta_data_package):
    """
    Updates UI elements after a recording's metadata has been successfully prepared by sim_thread.
    Called by UI thread when "RECORDING_METADATA_PREPARED" message is received.
    """
    if not dpg.is_dearpygui_running(): return

    if loaded_meta_data_package:
        global_gui_state["active_recording_data_source"] = loaded_meta_data_package
        global_gui_state["loaded_recording_filepath_for_ui"] = loaded_meta_data_package.get("filepath")
        num_frames = loaded_meta_data_package.get("num_frames", 0)
        
        # Enable the main "Playback Recording" button
        if dpg.does_item_exist("playback_button"):
            dpg.configure_item("playback_button", enabled=True, label="Playback Recording")
        
        # Configure the playback controls group (it's initially hidden)
        # It will be shown when user actually clicks "Playback Recording" -> enters playback mode.
        # For now, just ensure slider max is ready.
        if dpg.does_item_exist("playback_slider"):
            dpg.configure_item("playback_slider", max_value=max(0, num_frames - 1))
            dpg.set_value("playback_slider", 0) # Reset to start
        if dpg.does_item_exist("playback_current_frame_text"):
            dpg.set_value("playback_current_frame_text", f"Frame: 1 / {num_frames if num_frames > 0 else 1}")
        
        update_status_bar(f"Recording '{os.path.basename(loaded_meta_data_package.get('filepath', ''))}' loaded. {num_frames} frames. Ready for playback.", level="success")
    else: # Should not happen if message is for success
        global_gui_state["active_recording_data_source"] = None
        global_gui_state["loaded_recording_filepath_for_ui"] = None
        if dpg.does_item_exist("playback_button"):
            dpg.configure_item("playback_button", enabled=False)
        update_status_bar("Failed to process loaded recording metadata.", level="error")

# --- Handlers for Performance Testing & System Logs ---

def handle_run_benchmark_click(sender=None, app_data=None, user_data=None):
    """Runs the benchmark suite in a background thread."""
    global performance_test_running_type
    # Clear stop flag and enable stop button
    performance_test_stop_flag.clear()
    performance_test_running_type = "benchmark"
    if dpg.is_dearpygui_running() and dpg.does_item_exist("stop_perf_test_button"):
        dpg.configure_item("stop_perf_test_button", enabled=True)
    
    def run_benchmark():
        try:
            if dpg.is_dearpygui_running() and dpg.does_item_exist("perf_test_status_text"):
                dpg.set_value("perf_test_status_text", "Running benchmark suite...")
                dpg.set_value("perf_test_results_text", "")
            
            import subprocess
            # Stream output line-by-line so LogCapture can see it
            process = subprocess.Popen(
                [sys.executable, "benchmark.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            output_lines = []
            for line in process.stdout:
                # Check stop flag
                if performance_test_stop_flag.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    print("[STOPPED] Benchmark suite stopped by user")
                    if dpg.is_dearpygui_running():
                        dpg.set_value("perf_test_status_text", "Benchmark stopped by user.")
                        dpg.set_value("perf_test_results_text", "Partial results discarded. Previous results preserved.")
                        update_status_bar("Benchmark stopped", level="warning")
                    return
                
                print(line.rstrip())  # Print to console AND LogCapture
                output_lines.append(line.rstrip())
            
            returncode = process.wait(timeout=300)
            
            if returncode == 0:
                # Reload hardware limits from freshly written benchmark_results.json
                global HARDWARE_LIMITS
                HARDWARE_LIMITS = None  # Force reload
                _load_benchmark_limits()
                hw_note = get_hardware_note()
                if hw_note and dpg.is_dearpygui_running() and dpg.does_item_exist("cfg_hardware_performance_note"):
                    dpg.set_value("cfg_hardware_performance_note", hw_note)

                status = "Benchmark complete. Hardware limits updated."
                summary = hw_note + "\n\n" + "\n".join(output_lines[-5:]) if hw_note else "\n".join(output_lines[-10:])
            else:
                status = f"Benchmark failed with code {returncode}"
                summary = "\n".join(output_lines[-10:]) if len(output_lines) > 10 else "\n".join(output_lines)

            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", status)
                dpg.set_value("perf_test_results_text", summary)
                update_status_bar(status, level="info" if returncode == 0 else "error")
        except subprocess.TimeoutExpired:
            process.kill()
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", "Benchmark timed out after 5 minutes.")
                dpg.set_value("perf_test_results_text", "Check System Logs for partial results.")
                update_status_bar("Benchmark timed out", level="error")
        except Exception as e:
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", f"Error: {str(e)}")
                dpg.set_value("perf_test_results_text", "")
                update_status_bar(f"Benchmark error: {str(e)}", level="error")
        finally:
            global performance_test_running_type
            performance_test_running_type = None
            # Disable stop button when done
            if dpg.is_dearpygui_running() and dpg.does_item_exist("stop_perf_test_button"):
                dpg.configure_item("stop_perf_test_button", enabled=False)
    
    threading.Thread(target=run_benchmark, daemon=True).start()
    update_status_bar("Starting benchmark suite...", level="info")

def handle_run_optimization_click(sender=None, app_data=None, user_data=None):
    """Runs the auto-tuning workflow to optimize drive scales for different model/profile combinations."""
    global performance_test_running_type
    # Clear stop flag and enable stop button
    performance_test_stop_flag.clear()
    performance_test_running_type = "optimization"
    if dpg.is_dearpygui_running() and dpg.does_item_exist("stop_perf_test_button"):
        dpg.configure_item("stop_perf_test_button", enabled=True)
    
    def run_optimization():
        try:
            # Check if quick mode is enabled
            quick_mode = False
            if dpg.is_dearpygui_running() and dpg.does_item_exist("optimization_quick_mode_checkbox"):
                quick_mode = dpg.get_value("optimization_quick_mode_checkbox")
            
            mode_text = "quick mode" if quick_mode else "full mode"
            if dpg.is_dearpygui_running() and dpg.does_item_exist("perf_test_status_text"):
                dpg.set_value("perf_test_status_text", f"Running auto-tuning workflow ({mode_text})...")
                dpg.set_value("perf_test_results_text", "This may take several minutes.\nCheck console for detailed progress.")
            
            import subprocess
            # Build command with --auto-tune flag, optionally with --quick
            cmd = [sys.executable, "neural-simulator.py", "--auto-tune"]
            if quick_mode:
                cmd.append("--quick")
            
            # Stream output line-by-line so LogCapture can see it
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            output_lines = []
            for line in process.stdout:
                # Check stop flag
                if performance_test_stop_flag.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    print("[STOPPED] Auto-tuning/optimization stopped by user")
                    if dpg.is_dearpygui_running():
                        dpg.set_value("perf_test_status_text", "Auto-tuning stopped by user.")
                        dpg.set_value("perf_test_results_text", "Partial results discarded. Previous overrides preserved.")
                        update_status_bar("Auto-tuning stopped", level="warning")
                    return
                
                print(line.rstrip())  # Print to console AND LogCapture
                output_lines.append(line.rstrip())
            
            returncode = process.wait(timeout=1800)
            
            if returncode == 0:
                status = "Auto-tuning complete. Results saved to auto_tuned_overrides.json"
                # Count how many combinations were tuned
                try:
                    import json
                    with open("simulation_profiles/auto_tuned_overrides.json", "r", encoding='utf-8') as f:
                        data = json.load(f)
                    count = len(data.get("tuned_combinations", {}))
                    summary = f"Successfully tuned {count} model/profile combinations.\nReload overrides to apply them."
                except:
                    summary = "Check System Logs or auto_tuned_overrides.json for results."
            else:
                status = f"Auto-tuning failed with code {returncode}"
                summary = "\n".join(output_lines[-10:]) if len(output_lines) > 10 else "\n".join(output_lines)
            
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", status)
                dpg.set_value("perf_test_results_text", summary)
                update_status_bar(status, level="info" if returncode == 0 else "error")
        except subprocess.TimeoutExpired:
            process.kill()
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", "Auto-tuning timed out after 30 minutes.")
                dpg.set_value("perf_test_results_text", "Check System Logs for partial results.")
                update_status_bar("Auto-tuning timed out", level="error")
        except Exception as e:
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", f"Error: {str(e)}")
                dpg.set_value("perf_test_results_text", "")
                update_status_bar(f"Auto-tuning error: {str(e)}", level="error")
        finally:
            global performance_test_running_type
            performance_test_running_type = None
            # Disable stop button when done
            if dpg.is_dearpygui_running() and dpg.does_item_exist("stop_perf_test_button"):
                dpg.configure_item("stop_perf_test_button", enabled=False)
    
    threading.Thread(target=run_optimization, daemon=True).start()
    update_status_bar("Starting auto-tuning workflow...", level="info")

def handle_stop_perf_test_click(sender=None, app_data=None, user_data=None):
    """Stops any running benchmark or optimization task."""
    global performance_test_running_type
    
    if performance_test_running_type:
        test_name = "benchmark suite" if performance_test_running_type == "benchmark" else "auto-tuning/optimization"
        print(f"[STOP REQUESTED] Stopping {test_name}...")
        update_status_bar(f"Stopping {test_name}...", level="warning")
    else:
        print("[STOP] No performance test currently running")
        update_status_bar("No test running to stop", level="info")
    
    performance_test_stop_flag.set()
    if dpg.is_dearpygui_running() and dpg.does_item_exist("perf_test_status_text"):
        dpg.set_value("perf_test_status_text", "Stopping...")

def handle_reload_overrides_click(sender=None, app_data=None, user_data=None):
    """Reloads auto-tuned overrides from disk."""
    global AUTO_TUNED_OVERRIDES
    AUTO_TUNED_OVERRIDES = None  # Force reload
    _load_auto_tuned_overrides_if_needed()
    
    count = len(AUTO_TUNED_OVERRIDES) if AUTO_TUNED_OVERRIDES else 0
    msg = f"Reloaded {count} auto-tuned combinations from disk."
    update_status_bar(msg, level="success")
    
    if dpg.is_dearpygui_running() and dpg.does_item_exist("perf_test_status_text"):
        dpg.set_value("perf_test_status_text", msg)
        dpg.set_value("perf_test_results_text", f"Available combinations: {count}\nThese will be applied automatically when Apply & Reset is clicked.")

def handle_log_search_change(sender, app_data, user_data):
    """Handles search input changes in the log viewer."""
    if not hasattr(handle_log_search_change, "log_capture"):
        return
    
    query = app_data.strip()
    if not query:
        if dpg.is_dearpygui_running():
            dpg.set_value("log_search_match_text", "0 / 0 matches")
            dpg.configure_item("log_search_prev_button", enabled=False)
            dpg.configure_item("log_search_next_button", enabled=False)
        return
    
    log_capture = handle_log_search_change.log_capture
    matches = log_capture.search(query)
    
    if dpg.is_dearpygui_running():
        if matches:
            handle_log_search_change.current_matches = matches
            handle_log_search_change.current_match_index = 0
            dpg.set_value("log_search_match_text", f"1 / {len(matches)} matches")
            dpg.configure_item("log_search_prev_button", enabled=len(matches) > 1)
            dpg.configure_item("log_search_next_button", enabled=len(matches) > 1)
            # Highlight first match
            _update_log_display_with_highlight(matches[0])
        else:
            dpg.set_value("log_search_match_text", "0 / 0 matches")
            dpg.configure_item("log_search_prev_button", enabled=False)
            dpg.configure_item("log_search_next_button", enabled=False)

def handle_log_search_prev(sender=None, app_data=None, user_data=None):
    """Navigate to previous search match."""
    if not hasattr(handle_log_search_change, "current_matches"):
        return
    
    matches = handle_log_search_change.current_matches
    if not matches:
        return
    
    handle_log_search_change.current_match_index = (handle_log_search_change.current_match_index - 1) % len(matches)
    idx = handle_log_search_change.current_match_index
    
    if dpg.is_dearpygui_running():
        dpg.set_value("log_search_match_text", f"{idx + 1} / {len(matches)} matches")
        _update_log_display_with_highlight(matches[idx])

def handle_log_search_next(sender=None, app_data=None, user_data=None):
    """Navigate to next search match."""
    if not hasattr(handle_log_search_change, "current_matches"):
        return
    
    matches = handle_log_search_change.current_matches
    if not matches:
        return
    
    handle_log_search_change.current_match_index = (handle_log_search_change.current_match_index + 1) % len(matches)
    idx = handle_log_search_change.current_match_index
    
    if dpg.is_dearpygui_running():
        dpg.set_value("log_search_match_text", f"{idx + 1} / {len(matches)} matches")
        _update_log_display_with_highlight(matches[idx])

def _update_log_display_with_highlight(line_index):
    """Updates the log display and scrolls to highlight a specific line."""
    if not hasattr(handle_log_search_change, "log_capture"):
        return
    
    log_capture = handle_log_search_change.log_capture
    logs = log_capture.get_logs()
    
    if 0 <= line_index < len(logs):
        # Show context around the match
        start = max(0, line_index - 5)
        end = min(len(logs), line_index + 6)
        
        display_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == line_index else "    "
            display_lines.append(f"{prefix}{logs[i]}")
        
        display_text = "\n".join(display_lines)
        if dpg.is_dearpygui_running() and dpg.does_item_exist("system_logs_display"):
            dpg.set_value("system_logs_display", display_text)

def handle_clear_logs_click(sender=None, app_data=None, user_data=None):
    """Clears the log buffer."""
    if hasattr(handle_log_search_change, "log_capture"):
        handle_log_search_change.log_capture.clear()
        if dpg.is_dearpygui_running() and dpg.does_item_exist("system_logs_display"):
            dpg.set_value("system_logs_display", "")
        update_status_bar("Logs cleared.", level="info")

def handle_export_logs_click(sender=None, app_data=None, user_data=None):
    """Exports logs to a timestamped file."""
    if not hasattr(handle_log_search_change, "log_capture"):
        return
    
    try:
        log_capture = handle_log_search_change.log_capture
        logs = log_capture.get_logs()
        
        if not logs:
            update_status_bar("No logs to export.", level="warning")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"simulation_logs_{timestamp}.txt"
        
        with open(filepath, 'w') as f:
            f.write("\n".join(logs))
        
        update_status_bar(f"Logs exported to {filepath}", level="success")
    except Exception as e:
        update_status_bar(f"Export error: {str(e)}", level="error")

def handle_run_viz_benchmark_click(sender=None, app_data=None, user_data=None):
    """Runs the visualization performance test in a background thread."""
    global performance_test_running_type
    # Clear stop flag and enable stop button
    performance_test_stop_flag.clear()
    performance_test_running_type = "viz_benchmark"
    if dpg.is_dearpygui_running() and dpg.does_item_exist("stop_perf_test_button"):
        dpg.configure_item("stop_perf_test_button", enabled=True)
    
    def run_viz_benchmark():
        try:
            # Check if quick mode is enabled
            quick_mode = False
            if dpg.is_dearpygui_running() and dpg.does_item_exist("viz_benchmark_quick_mode_checkbox"):
                quick_mode = dpg.get_value("viz_benchmark_quick_mode_checkbox")
            
            mode_text = "quick mode" if quick_mode else "full mode"
            if dpg.is_dearpygui_running() and dpg.does_item_exist("perf_test_status_text"):
                dpg.set_value("perf_test_status_text", f"Running visualization performance test ({mode_text})...")
                dpg.set_value("perf_test_results_text", "This may take several minutes.\nCheck System Logs for detailed progress.")
            
            import subprocess
            # Build command with optional --quick flag
            cmd = [sys.executable, "viz_benchmark.py"]
            if quick_mode:
                cmd.append("--quick")
            
            # Stream output line-by-line so LogCapture can see it
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            output_lines = []
            for line in process.stdout:
                # Check stop flag
                if performance_test_stop_flag.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    print("[STOPPED] Viz performance test stopped by user")
                    if dpg.is_dearpygui_running():
                        dpg.set_value("perf_test_status_text", "Viz performance test stopped by user.")
                        dpg.set_value("perf_test_results_text", "Partial results discarded.")
                        update_status_bar("Viz performance test stopped", level="warning")
                    return
                
                print(line.rstrip())  # Print to console AND LogCapture
                output_lines.append(line.rstrip())
            
            returncode = process.wait(timeout=600)  # 10 minute timeout
            
            if returncode == 0:
                # Load results and update hardware note
                results_path = "benchmarks/viz_performance_results.json"
                try:
                    with open(results_path, 'r') as f:
                        benchmark_data = json.load(f)
                    
                    hardware_note = benchmark_data.get("hardware_performance_note", "Benchmark completed.")
                    
                    # Update hardware note in UI
                    if dpg.does_item_exist("cfg_hardware_performance_note"):
                        dpg.set_value("cfg_hardware_performance_note", hardware_note)
                    
                    # Build summary for results text
                    capacity_summary = benchmark_data.get("capacity_summary", {})
                    summary_lines = ["Viz Performance Test Complete!", ""]
                    if capacity_summary:
                        for key, data in capacity_summary.items():
                            max_n = data.get("max_neurons", 0)
                            conn = data.get("connections_per_neuron", 0)
                            if max_n > 0:
                                summary_lines.append(f"{key}: {max_n:,}N ({max_n * conn:,} synapses)")
                            else:
                                summary_lines.append(f"{key}: No realtime configs found")
                    else:
                        summary_lines.append("No realtime-capable configurations found.")
                    
                    summary_lines.append("")
                    summary_lines.append("Results: benchmarks/viz_performance_results.json")
                    summary_lines.append("Hardware note updated in Core Simulation Parameters.")
                    summary = "\n".join(summary_lines)
                    
                    status = "Viz performance test complete."
                except Exception as e:
                    status = "Viz test complete but failed to parse results."
                    summary = f"Error: {str(e)}\nCheck benchmarks/viz_performance_results.json"
            else:
                status = f"Viz performance test failed with code {returncode}"
                summary = "\n".join(output_lines[-10:]) if len(output_lines) > 10 else "\n".join(output_lines)
            
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", status)
                dpg.set_value("perf_test_results_text", summary)
                update_status_bar(status, level="info" if returncode == 0 else "error")
        except subprocess.TimeoutExpired:
            process.kill()
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", "Viz performance test timed out after 10 minutes.")
                dpg.set_value("perf_test_results_text", "Check System Logs for partial results.")
                update_status_bar("Viz performance test timed out", level="error")
        except Exception as e:
            if dpg.is_dearpygui_running():
                dpg.set_value("perf_test_status_text", f"Error: {str(e)}")
                dpg.set_value("perf_test_results_text", "")
                update_status_bar(f"Viz performance test error: {str(e)}", level="error")
        finally:
            global performance_test_running_type
            performance_test_running_type = None
            # Disable stop button when done
            if dpg.is_dearpygui_running() and dpg.does_item_exist("stop_perf_test_button"):
                dpg.configure_item("stop_perf_test_button", enabled=False)
    
    threading.Thread(target=run_viz_benchmark, daemon=True).start()
    update_status_bar("Starting viz performance test...", level="info")

# --- Main DPG GUI Layout Creation (Called by Main/UI Thread) ---

def add_parameter_table_row(label_text, item_callable, item_tag, default_value, callback_func, tooltip=None, **kwargs):
    """
    Adds a row to a DPG table with a label in the first column and a DPG item in the second.
    Assumes this is called within a `with dpg.table(): ...` context where columns are already defined.

    Args:
        tooltip: Optional string for a hover tooltip on the label, providing parameter help.
    """
    with dpg.table_row():
        label_id = dpg.add_text(label_text)
        if tooltip:
            with dpg.tooltip(label_id):
                dpg.add_text(tooltip, wrap=350, color=[220, 220, 180, 255])
        # Ensure 'label' kwarg for the item itself is empty as we're using a separate text widget
        kwargs['label'] = ""

        # Only add width=-1 if it's not a checkbox and width is not already specified.
        # Checkboxes and some other items might not support the 'width' argument or handle it differently.
        if item_callable != dpg.add_checkbox: # Check if the item is NOT a checkbox
            if 'width' not in kwargs: # If width is not already specified for other items
                kwargs['width'] = -1  # Make it fill the table cell
        elif 'width' in kwargs and item_callable == dpg.add_checkbox:
            # If width was somehow passed for a checkbox, remove it to prevent error
            del kwargs['width']

        return item_callable(tag=item_tag, default_value=default_value, callback=callback_func, **kwargs)

def create_gui_layout():
    """Creates the main Dear PyGui layout, including all windows, menus, and widgets."""
    profile_dir = "simulation_profiles/" 
    checkpoint_dir_h5 = "simulation_checkpoints_h5/"
    recording_dir_h5 = "simulation_recordings_h5/"  

    for p_dir in [profile_dir, checkpoint_dir_h5, recording_dir_h5]:
        if not os.path.exists(p_dir): os.makedirs(p_dir, exist_ok=True)

    # Define column widths for parameter tables
    label_col_width = 320 

    with dpg.window(label="Controls & Configuration", tag="controls_monitor_window",
                    width=-1, height=-1, pos=[0,0], 
                    on_close=lambda: (shutdown_flag.set(), dpg.stop_dearpygui() if dpg.is_dearpygui_running() else None),
                    menubar=True):
        dpg.add_spacer(height=5)

        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Save Profile (.json)", callback=handle_save_profile_button_press, tag="save_profile_menu")
                dpg.add_menu_item(label="Load Profile (.json)", callback=handle_load_profile_button_press, tag="load_profile_menu")
                dpg.add_separator()
                dpg.add_menu_item(label="Save Checkpoint (.simstate.h5)", callback=handle_save_checkpoint_button_press, tag="save_checkpoint_menu_h5")
                dpg.add_menu_item(label="Load Checkpoint (.simstate.h5)", callback=handle_load_checkpoint_button_press, tag="load_checkpoint_menu_h5")
                dpg.add_separator()
                dpg.add_menu_item(label="Load Recording (.simrec.h5)", callback=handle_load_recording_menu_click, tag="load_recording_menu_h5")
                dpg.add_separator()
                dpg.add_menu_item(label="Exit", callback=lambda: (shutdown_flag.set(), dpg.stop_dearpygui() if dpg.is_dearpygui_running() else None))

        with dpg.collapsing_header(label="Simulation Controls", default_open=True):
            dpg.add_text("Status: Idle", tag="status_bar_text")
            dpg.add_spacer(height=3)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start", tag="start_button", callback=handle_start_simulation_event, width = -1)
            with dpg.group(horizontal=True): 
                dpg.add_button(label="Pause", tag="pause_button", callback=handle_pause_simulation_event, width=100, enabled=False)
                dpg.add_button(label="Stop", tag="stop_button", callback=handle_stop_simulation_event, width=100, enabled=False)
                dpg.add_button(label="Step (1ms)", tag="step_button", callback=handle_step_simulation_event, width=-1, enabled=True)

            dpg.add_button(label="Apply Changes & Reset Sim", tag="apply_config_button", callback=handle_apply_config_changes_and_reset, width=-1)
            
            dpg.add_text("Simulation Speed:")
            dpg.add_slider_float(label="", tag="sim_speed_slider", default_value=1.0, min_value=0.01, max_value=20.0, width=-1, callback=handle_sim_speed_change, format="%.2f x")

            dpg.add_separator()
            dpg.add_text("Recording & Playback:")
            with dpg.group(horizontal=True): 
                dpg.add_button(label="Record", tag="record_button", callback=handle_record_button_click, width = -1)
            with dpg.group(horizontal=True): 
                dpg.add_button(label="Playback Recording", tag="playback_button", callback=handle_playback_button_click, width = -1, enabled=False)

            with dpg.group(tag="playback_controls_group", show=False): 
                dpg.add_text("Playback Controls:")
                dpg.add_text("Frame: 0 / 0", tag="playback_current_frame_text")
                dpg.add_slider_int(label="", tag="playback_slider", width=-1, callback=handle_playback_slider_change, min_value=0, max_value=0) 
                with dpg.group(horizontal=True, horizontal_spacing=10):
                    dpg.add_button(label="<< (-5)", tag="playback_step_minus_5", callback=handle_playback_step_frames_click, user_data=-5, width=70)
                    dpg.add_button(label="< (-1)", tag="playback_step_minus_1", callback=handle_playback_step_frames_click, user_data=-1, width=70)
                    dpg.add_button(label="Play", tag="playback_play_pause_button", callback=handle_playback_play_pause_button_click, width = 150) 
                    dpg.add_button(label="(+1) >", tag="playback_step_plus_1", callback=handle_playback_step_frames_click, user_data=1, width=70)
                    dpg.add_button(label="(+5) >>", tag="playback_step_plus_5", callback=handle_playback_step_frames_click, user_data=5, width=70)

        dpg.add_spacer(height=5); dpg.add_separator(); dpg.add_spacer(height=5)

        with dpg.collapsing_header(label="Core Simulation Parameters", default_open=False, tag="core_sim_params_header"):
            # Full Profile dropdown (auto-populated from simulation_profiles/*.json)
            _scan_profile_directory()
            dpg.add_text("Load Full Profile:", color=[150,220,255,255])
            dpg.add_text("Applies all parameters (model, plasticity, noise, etc.) from a saved profile.",
                         color=[140,140,140,255], wrap=label_col_width + 50)
            with dpg.group(horizontal=True):
                dpg.add_combo(tag="cfg_full_profile",
                              items=list(_FULL_PROFILE_MAP.keys()),
                              default_value="(None - use settings below)",
                              callback=_handle_full_profile_dropdown_change,
                              width=350)
                dpg.add_button(label="Refresh", callback=lambda: _refresh_full_profile_dropdown(),
                               width=70)
            dpg.add_spacer(height=5)
            dpg.add_separator()
            dpg.add_spacer(height=5)

            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)

                add_parameter_table_row("Number of Neurons:", dpg.add_input_int, "cfg_num_neurons", 1000, _update_sim_config_from_ui_and_signal_reset_needed, min_value=1, step=100,
                    tooltip="Total neurons in the network. 1K-10K for real-time on most GPUs. 50K-100K for RTX 3090+ (24GB VRAM). Higher counts require more VRAM.")
                add_parameter_table_row("Connections/Neuron (Spatial Fallback):", dpg.add_input_int, "cfg_connections_per_neuron", 100, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Average synaptic connections per neuron when using spatial connectivity. Biological range: 1K-10K (cortex ~7K). Higher values increase memory and computation.")
                add_parameter_table_row("Total Sim Time (ms):", dpg.add_input_float, "cfg_total_sim_time", 60000.0, _update_sim_config_from_ui_and_signal_reset_needed, step=100,
                    tooltip="Maximum simulation duration in milliseconds. 60000ms = 60 seconds of biological time. Can always be stopped early.")
                add_parameter_table_row("Time Step dt (ms):", dpg.add_input_float, "cfg_dt_ms", 1.000, _update_sim_config_from_ui_and_signal_reset_needed, step=0.001, format="%.3f", min_value=0.001,
                    tooltip="Integration timestep. Izhikevich: 0.5-1.0ms is stable. Hodgkin-Huxley: MUST be <= 0.1ms (gating kinetics require fine resolution). AdEx: 0.1-0.5ms recommended. Smaller dt = more accurate but slower.")
                add_parameter_table_row("Seed (-1 for random):", dpg.add_input_int, "cfg_seed", -1, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Random seed for reproducibility. Set to -1 for a new random seed each run. Use a fixed positive integer to reproduce identical simulations.")
                add_parameter_table_row("Number of Traits:", dpg.add_input_int, "cfg_num_traits", 5, _update_sim_config_from_ui_and_signal_reset_needed, min_value=1, max_value=len(TRAIT_COLOR_MAP_RAW) if TRAIT_COLOR_MAP_RAW else 10,
                    tooltip="Number of neuron sub-populations (color-coded in 3D view). One trait is designated inhibitory. More traits = more diverse network topology.")
                add_parameter_table_row("Neuron Model:", dpg.add_combo, "cfg_neuron_model_type", NeuronModel.IZHIKEVICH.name, _handle_model_type_change_dpg, items=[model.name for model in NeuronModel],
                    tooltip="Izhikevich: Fast, versatile (20+ firing patterns). Good for large networks.\nHodgkin-Huxley: Biophysically detailed (ion channels, temperature). Requires dt<=0.1ms.\nAdEx: Balance of speed and biophysics. Good for adaptation studies.")
                add_parameter_table_row("Neural Structure Profile:", dpg.add_combo, "cfg_neural_profile", "GENERIC_UNSTRUCTURED", _update_sim_config_from_ui_and_signal_reset_needed, items=sorted(NEURAL_STRUCTURE_PROFILES.keys()),
                    tooltip="Pre-configured brain region profiles with literature-based connectivity, E/I ratios, and neuron type distributions. GENERIC_UNSTRUCTURED uses basic random connectivity.")
            
            # Hardware performance note (read-only info from benchmarks)
            dpg.add_spacer(height=5)
            dpg.add_text("Hardware Performance Note:", color=[150,200,255,255])
            dpg.add_text("", tag="cfg_hardware_performance_note", wrap=label_col_width + 50, color=[180,180,180,255])
            dpg.add_text("", tag="hw_limit_warning_text", wrap=label_col_width + 50, color=[255,100,100,255], show=False)
            dpg.add_spacer(height=5)

            with dpg.group(tag="izhikevich_params_group", show=True):
                dpg.add_text("--- Izhikevich 2007 Model Parameters ---", color=[200,200,100,255])
                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                    dpg.add_table_column(width_stretch=True)
                    ui_izh_params = [
                        ("Membrane Capacitance C (pF)", "cfg_izh_C_val", "%.1f", 100.0), ("Scaling Factor k (nS/mV)", "cfg_izh_k_val", "%.2f", 0.7),
                        ("Resting Potential vr (mV)", "cfg_izh_vr_val", "%.1f", -60.0), ("Threshold Potential vt (mV)", "cfg_izh_vt_val", "%.1f", -40.0),
                        ("Spike Peak/Cutoff vpeak (mV)", "cfg_izh_vpeak_val", "%.1f", 35.0), ("Recovery Time Scale a (/ms)", "cfg_izh_a_val", "%.3f", 0.03),
                        ("Recovery Sensitivity b (nS)", "cfg_izh_b_val", "%.2f", -2.0), ("Voltage Reset c (mV)", "cfg_izh_c_val", "%.1f", -50.0),
                        ("Recovery Increment d (pA)", "cfg_izh_d_val", "%.1f", 100.0)
                    ]
                    _izh_tooltips = {
                        "cfg_izh_C_val": "Membrane capacitance. Higher C = slower voltage changes.\nRS ~100 pF, FS ~20-50 pF. (Izhikevich 2007, Table 2)",
                        "cfg_izh_k_val": "Scaling factor relating subthreshold I-V curvature.\nDetermines input resistance near rest.\nRS ~0.7, FS ~1.0, IB ~1.2 nS/mV.",
                        "cfg_izh_vr_val": "Resting membrane potential (no input).\nTypically -60 to -65 mV for cortical neurons.",
                        "cfg_izh_vt_val": "Instantaneous threshold potential.\nVoltage at which dV/dt becomes positive.\nTypically -40 to -45 mV.",
                        "cfg_izh_vpeak_val": "Spike cutoff voltage. When V >= vpeak, a spike\nis registered and V resets to c.\nTypically +25 to +35 mV.",
                        "cfg_izh_a_val": "Recovery variable time constant (1/ms).\nSmall a = slow recovery (RS ~0.03).\nLarge a = fast recovery (FS ~0.1).",
                        "cfg_izh_b_val": "Recovery sensitivity to subthreshold V.\nNegative b = resonator properties.\nRS ~-2 nS, FS ~0.25 nS.",
                        "cfg_izh_c_val": "Post-spike voltage reset.\nMore negative c = stronger after-hyperpolarization.\nRS ~-50 mV, IB ~-55 mV, CH ~-40 mV.",
                        "cfg_izh_d_val": "Post-spike recovery variable increment.\nControls spike-frequency adaptation.\nRS ~100 pA, FS ~25 pA, IB ~130 pA.",
                    }
                    for desc_label, tag, fmt, def_val in ui_izh_params:
                        add_parameter_table_row(desc_label, dpg.add_input_float, tag, def_val, _update_sim_config_from_ui_and_signal_reset_needed, format=fmt,
                            tooltip=_izh_tooltips.get(tag))
            
            with dpg.group(tag="hodgkin_huxley_params_group", show=False):
                dpg.add_text("--- Hodgkin-Huxley Model Parameters ---", color=[200,200,100,255])
                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                    dpg.add_table_column(width_stretch=True)
                    # HH neuron type preset selector
                    add_parameter_table_row(
                        "HH Default Neuron Type:",
                        dpg.add_combo,
                        "cfg_default_neuron_type_hh",
                        NeuronType.HH_L5_CORTICAL_PYRAMIDAL_RS.name,
                        _update_sim_config_from_ui_and_signal_reset_needed,
                        items=[nt.name for nt in NeuronType if "HH_" in nt.name],
                        tooltip="Select a biophysical neuron type preset.\nSets conductances and kinetics for specific cell classes\n(e.g., cortical pyramidal, fast-spiking interneuron)."
                    )
                    ui_hh_params = [
                        ("Membrane Capacitance C_m (uF/cm^2)", "cfg_hh_C_m", "%.2f", 1.0),
                        ("Max Sodium Cond. g_Na_max (mS/cm^2)", "cfg_hh_g_Na_max", "%.1f", 50.0),
                        ("Max Potassium Cond. g_K_max (mS/cm^2)", "cfg_hh_g_K_max", "%.1f", 5.0),
                        ("Leak Cond. g_L (mS/cm^2)", "cfg_hh_g_L", "%.3f", 0.1),
                        ("Sodium Reversal E_Na (mV)", "cfg_hh_E_Na", "%.1f", 50.0),
                        ("Potassium Reversal E_K (mV)", "cfg_hh_E_K", "%.1f", -85.0),
                        ("Leak Reversal E_L (mV)", "cfg_hh_E_L", "%.3f", -70.0),
                        ("Spike Detection V_peak (mV)", "cfg_hh_v_peak", "%.1f", 40.0),
                        ("Initial V_rest (mV)", "cfg_hh_v_rest_init", "%.1f", -65.0),
                        ("M-current g_M_max (mS/cm^2)", "cfg_hh_g_M_max", "%.3f", 0.0),
                        ("M-current Tau (ms)", "cfg_hh_m_current_tau_ms", "%.1f", 100.0),
                        ("CaT g_CaT_max (mS/cm^2)", "cfg_hh_g_CaT_max", "%.3f", 0.0),
                        ("CaT Reversal E_CaT (mV)", "cfg_hh_E_CaT", "%.1f", 120.0),
                        ("I_h g_h_max (mS/cm^2)", "cfg_hh_g_h_max", "%.3f", 0.0),
                        ("I_h Reversal E_h (mV)", "cfg_hh_E_h", "%.1f", -30.0),
                        ("NaP g_NaP_max (mS/cm^2)", "cfg_hh_g_NaP_max", "%.3f", 0.0),
                        ("Kinetics Q10 Factor", "cfg_hh_q10_factor", "%.1f", 3.0),
                        ("Kinetics Temperature (C)", "cfg_hh_temperature_celsius", "%.1f", 37.0),
                    ]
                    _hh_tooltips = {
                        "cfg_hh_C_m": "Specific membrane capacitance.\nStandard squid axon: 1.0 uF/cm².\nHigher C_m = slower voltage dynamics.",
                        "cfg_hh_g_Na_max": "Maximum sodium conductance density.\nControls action potential amplitude and rise speed.\nSquid axon: 120, cortical: 50 mS/cm².",
                        "cfg_hh_g_K_max": "Maximum delayed-rectifier potassium conductance.\nControls repolarization and spike width.\nSquid: 36, cortical: 5 mS/cm².",
                        "cfg_hh_g_L": "Leak conductance density.\nSets resting input resistance.\nTypically 0.03-0.3 mS/cm².",
                        "cfg_hh_E_Na": "Sodium Nernst reversal potential.\nSet by [Na+] gradient across membrane.\nTypically +50 mV (mammalian).",
                        "cfg_hh_E_K": "Potassium Nernst reversal potential.\nSet by [K+] gradient.\nTypically -77 to -90 mV.",
                        "cfg_hh_E_L": "Leak reversal potential.\nApproximates resting V when no active currents.\nTypically -54 to -70 mV.",
                        "cfg_hh_v_peak": "Voltage threshold for formal spike detection.\nAt 37°C with Q10=3, fast kinetics may produce\nspikes below +40 mV. Adjust if needed.",
                        "cfg_hh_v_rest_init": "Initial resting membrane potential.\nGating variables are initialized to steady-state\nvalues at this voltage.",
                        "cfg_hh_g_M_max": "Muscarinic (M-type) K+ current max conductance.\nSlow non-inactivating K+ current. Causes spike\nfrequency adaptation. 0 = disabled.",
                        "cfg_hh_m_current_tau_ms": "M-current activation time constant.\nSlow ~100 ms gives adaptation over multiple spikes.\nRange: 50-200 ms.",
                        "cfg_hh_g_CaT_max": "Low-threshold Ca²+ (T-type) current conductance.\nEnables rebound bursting and subthreshold oscillations.\n0 = disabled. Typical: 0.5-2.0 mS/cm².",
                        "cfg_hh_E_CaT": "Calcium reversal potential.\nSet by [Ca²+] gradient. Typically +120 mV.",
                        "cfg_hh_g_h_max": "Hyperpolarization-activated cation current (I_h).\nContributes to resting potential, sag response,\nand pacemaker activity. 0 = disabled.",
                        "cfg_hh_E_h": "I_h reversal potential (mixed Na+/K+).\nTypically -20 to -40 mV, depolarizing from rest.",
                        "cfg_hh_g_NaP_max": "Persistent sodium current conductance.\nNon-inactivating Na+ near threshold.\nAmplifies subthreshold inputs. 0 = disabled.",
                        "cfg_hh_q10_factor": "Temperature coefficient for gating kinetics.\nRate multiplier per 10°C: phi = Q10^((T-6.3)/10).\nQ10=3 is standard for ion channels.",
                        "cfg_hh_temperature_celsius": "Simulation temperature for HH kinetics.\n6.3°C = original squid axon (Hodgkin & Huxley 1952).\n37°C = mammalian with ~28x faster kinetics.",
                    }
                    for desc_label, tag, fmt, def_val in ui_hh_params:
                        add_parameter_table_row(
                            desc_label,
                            dpg.add_input_float,
                            tag,
                            def_val,
                            _update_sim_config_from_ui_and_signal_reset_needed,
                            format=fmt,
                            tooltip=_hh_tooltips.get(tag),
                        )

                    # External drive scale slider (auto-tuned)
                    add_parameter_table_row(
                        "External Drive Scale (HH, auto-tuned):",
                        dpg.add_slider_float,
                        "cfg_hh_external_drive_scale",
                        1.0,
                        _update_sim_config_from_ui_and_signal_reset_needed,
                        min_value=0.1,
                        max_value=8.0,
                        format="%.2f",
                        tooltip="Multiplier for external input current to HH neurons.\nAuto-tuned during initialization. Increase if neurons\nare too quiet, decrease if network is epileptic.",
                    )

                    # Button to reset HH drive scale to auto-tuned value (if available)
                    with dpg.table_row():
                        dpg.add_text("Reset HH Drive to Auto-Tuned:")
                        dpg.add_button(
                            tag="cfg_hh_reset_drive_to_auto_btn",
                            label="Reset",
                            callback=handle_reset_hh_drive_to_auto,
                            width=-1,
                        )

            with dpg.group(tag="adex_params_group", show=False):
                dpg.add_text("--- AdEx Model Parameters ---", color=[200,200,100,255])
                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                    dpg.add_table_column(width_stretch=True)
                    ui_adex_params = [
                        ("Membrane Capacitance C (pF)", "cfg_adex_C", "%.1f", 281.0),
                        ("Leak Conductance g_L (nS)", "cfg_adex_g_L", "%.1f", 30.0),
                        ("Leak Reversal E_L (mV)", "cfg_adex_E_L", "%.1f", -70.6),
                        ("Spike Threshold V_T (mV)", "cfg_adex_V_T", "%.1f", -50.4),
                        ("Slope Factor Delta_T (mV)", "cfg_adex_Delta_T", "%.2f", 2.0),
                        ("Subthreshold Coupling a (nS)", "cfg_adex_a", "%.1f", 4.0),
                        ("Adaptation Time Constant tau_w (ms)", "cfg_adex_tau_w", "%.1f", 144.0),
                        ("Spike-triggered Increment b (pA)", "cfg_adex_b", "%.1f", 80.5),
                        ("Reset Potential V_r (mV)", "cfg_adex_V_r", "%.1f", -70.6),
                        ("Spike Detection V_peak (mV)", "cfg_adex_V_peak", "%.1f", -40.0),
                    ]
                    _adex_tooltips = {
                        "cfg_adex_C": "Membrane capacitance. Brette & Gerstner 2005:\nRS ~281 pF, FS ~100 pF. Controls voltage time constant.",
                        "cfg_adex_g_L": "Leak conductance. Sets resting input resistance.\nR_in = 1/g_L. RS ~30 nS, FS ~10 nS.",
                        "cfg_adex_E_L": "Leak reversal / resting potential.\nTypically -70 to -65 mV for cortical neurons.",
                        "cfg_adex_V_T": "Effective spike threshold voltage.\nThe exponential term activates steeply above V_T.\nTypically -50 to -45 mV.",
                        "cfg_adex_Delta_T": "Slope factor of exponential spike initiation.\nSmaller = sharper threshold. 0 = perfect IF.\nTypical: 1-4 mV. (Badel et al. 2008)",
                        "cfg_adex_a": "Subthreshold adaptation coupling.\nLinks adaptation variable w to voltage.\nRS ~4 nS, bursting ~0.5 nS.",
                        "cfg_adex_tau_w": "Adaptation time constant.\nControls how quickly w decays after spikes.\nRS ~144 ms, FS ~20 ms.",
                        "cfg_adex_b": "Spike-triggered adaptation increment.\nAdded to w after each spike.\nLarger b = stronger spike-frequency adaptation.\nRS ~80 pA, FS ~0 pA.",
                        "cfg_adex_V_r": "Post-spike membrane potential reset.\nTypically near E_L. More negative = stronger\nafter-hyperpolarization.",
                        "cfg_adex_V_peak": "Spike detection threshold.\nWhen V exceeds V_peak, spike is registered\nand V resets to V_r. Typically 0 to -40 mV.",
                    }
                    for desc_label, tag, fmt, def_val in ui_adex_params:
                        add_parameter_table_row(
                            desc_label,
                            dpg.add_input_float,
                            tag,
                            def_val,
                            _update_sim_config_from_ui_and_signal_reset_needed,
                            format=fmt,
                            tooltip=_adex_tooltips.get(tag),
                        )

                    # External drive scale slider (auto-tuned)
                    add_parameter_table_row(
                        "External Drive Scale (AdEx, auto-tuned):",
                        dpg.add_slider_float,
                        "cfg_adex_external_drive_scale",
                        1.0,
                        _update_sim_config_from_ui_and_signal_reset_needed,
                        min_value=0.1,
                        max_value=5.0,
                        format="%.2f",
                        tooltip="Multiplier for external input current to AdEx neurons.\nAuto-tuned during initialization. Adjust if firing\nrates are too low or too high.",
                    )

                    # Button to reset AdEx drive scale to auto-tuned value (if available)
                    with dpg.table_row():
                        dpg.add_text("Reset AdEx Drive to Auto-Tuned:")
                        dpg.add_button(
                            tag="cfg_adex_reset_drive_to_auto_btn",
                            label="Reset",
                            callback=handle_reset_adex_drive_to_auto,
                            width=-1,
                        )

        with dpg.collapsing_header(label="Network Connectivity", default_open=False, tag="network_connectivity_header"):
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Use Watts-Strogatz Generator:", dpg.add_checkbox, "cfg_enable_watts_strogatz", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Use Watts-Strogatz small-world network topology.\nCombines local clustering with short path lengths.\nDisable for random Erdos-Renyi connectivity.")
                add_parameter_table_row("W-S K (Nearest Neighbors, even):", dpg.add_input_int, "cfg_connectivity_k", 10, _update_sim_config_from_ui_and_signal_reset_needed, step=2, min_value=2,
                    tooltip="Each neuron connects to K nearest neighbors.\nMust be even. Higher K = denser local connectivity.\nK=10 gives ~10% connection prob. for 100 neurons.")
                add_parameter_table_row("W-S P (Rewire Probability):", dpg.add_input_float, "cfg_connectivity_p_rewire", 0.1, _update_sim_config_from_ui_and_signal_reset_needed, min_value=0.0, max_value=1.0, format="%.3f",
                    tooltip="Probability of rewiring each edge to a random target.\nP=0: regular lattice. P=1: fully random.\nP=0.05-0.2: small-world regime (Watts & Strogatz 1998).")

        with dpg.collapsing_header(label="Synaptic Parameters", default_open=False, tag="synaptic_params_header"):
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Excitatory Prop. Strength (g_peak_e scale):", dpg.add_input_float, "cfg_propagation_strength", 0.05, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f",
                    tooltip="Peak excitatory conductance increase per spike (nS).\nScales AMPA synaptic input. Higher = stronger\nexcitatory drive. Typical: 0.01-0.5.")
                add_parameter_table_row("Inhibitory Prop. Strength (g_peak_i scale):", dpg.add_input_float, "cfg_inhibitory_propagation_strength", 0.15, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f",
                    tooltip="Peak inhibitory conductance increase per spike (nS).\nScales GABA_A synaptic input. Usually 2-4x excitatory\nfor E/I balance. Typical: 0.05-1.0.")
                add_parameter_table_row("Excitatory Conductance Tau_g_e (ms):", dpg.add_input_float, "cfg_syn_tau_e", 5.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f", min_value=0.1, tooltip="AMPA receptor decay time constant. Fast excitatory transmission (1-10 ms typical).")
                add_parameter_table_row("Inhibitory Conductance Tau_g_i (ms):", dpg.add_input_float, "cfg_syn_tau_i", 10.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f", min_value=0.1, tooltip="GABA_A receptor decay time constant. Inhibitory transmission (5-20 ms typical).")
            dpg.add_separator()
            dpg.add_text("NMDA Receptors (Voltage-Dependent Mg²⁺ Block)")
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable NMDA:", dpg.add_checkbox, "cfg_enable_nmda", False, _update_sim_config_from_ui_and_signal_reset_needed, tooltip="NMDA receptors with voltage-dependent Mg²⁺ block (Jahr & Stevens 1990). Adds slow excitatory current gated by postsynaptic depolarization — critical for coincidence detection and associative plasticity.")
                add_parameter_table_row("NMDA:AMPA Ratio:", dpg.add_input_float, "cfg_nmda_ratio", 0.4, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f", min_value=0.0, max_value=2.0, tooltip="Ratio of NMDA to AMPA peak conductance. 0.3-0.5 typical for cortex (Myme et al. 2003).")
                add_parameter_table_row("NMDA Tau Decay (ms):", dpg.add_input_float, "cfg_nmda_tau_decay", 100.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=10.0, tooltip="NMDA receptor decay (~100 ms). Much slower than AMPA (~5 ms), enabling temporal integration.")
                add_parameter_table_row("NMDA Tau Rise (ms):", dpg.add_input_float, "cfg_nmda_tau_rise", 3.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=0.5, tooltip="NMDA receptor rise time (2-5 ms). Slower rise than AMPA due to glutamate binding kinetics.")
                add_parameter_table_row("[Mg²⁺] (mM):", dpg.add_input_float, "cfg_nmda_mg_conc", 1.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f", min_value=0.0, max_value=5.0, tooltip="Extracellular magnesium concentration. 1.0 mM physiological. Higher = stronger voltage-dependent block, less NMDA current at rest.")

        with dpg.collapsing_header(label="Learning & Plasticity", default_open=False, tag="learning_plasticity_header"):
            with dpg.table(header_row=False): 
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Hebbian Learning:", dpg.add_checkbox, "cfg_enable_hebbian_learning", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Simple Hebbian co-activation learning rule.\nWeights increase when pre and post neurons fire together.\nIncludes weight decay to prevent runaway excitation.")
                add_parameter_table_row("Hebbian Learning Rate:", dpg.add_input_float, "cfg_hebbian_learning_rate", 0.0005, _update_sim_config_from_ui_and_signal_reset_needed, format="%.6f",
                    tooltip="Rate of weight change per co-activation event.\nSmaller = more stable but slower learning.\nTypical range: 0.0001–0.01.")
                add_parameter_table_row("Hebbian Max Weight:", dpg.add_input_float, "cfg_hebbian_max_weight", 1.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f",
                    tooltip="Upper bound on synaptic weights under Hebbian learning.\nPrevents runaway excitation. Also used as upper\nclamp for synaptic scaling.")
            dpg.add_separator()
            with dpg.table(header_row=False): 
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Short-Term Plasticity (STP):", dpg.add_checkbox, "cfg_enable_short_term_plasticity", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Tsodyks-Markram short-term plasticity model. Synapses exhibit depression (weakening) and facilitation (strengthening) on timescales of 10-1000ms. Essential for temporal coding.")
                add_parameter_table_row("STP U (Baseline Utilization):", dpg.add_input_float, "cfg_stp_U", 0.15, _update_sim_config_from_ui_and_signal_reset_needed, format="%.3f", min_value=0.0, max_value=1.0,
                    tooltip="Fraction of available resources used per spike (0-1). Low U (~0.1-0.2): facilitating synapses (cortical). High U (~0.5-0.8): depressing synapses (thalamocortical). Literature: Tsodyks & Markram 1997.")
                add_parameter_table_row("STP Tau_d (Depression, ms):", dpg.add_input_float, "cfg_stp_tau_d", 200.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=0.1,
                    tooltip="Recovery time constant for synaptic resources (ms). Controls how fast depressed synapses recover. Typical range: 100-800ms.")
                add_parameter_table_row("STP Tau_f (Facilitation, ms):", dpg.add_input_float, "cfg_stp_tau_f", 50.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=0.1,
                    tooltip="Decay time constant for facilitation variable (ms). Controls duration of synaptic facilitation. Typical range: 20-200ms.")
                add_parameter_table_row("Enable Per-Type STP:", dpg.add_checkbox, "cfg_enable_per_type_stp", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Use different STP parameters for E->E, E->I, I->E, I->I synapses.\nMore biologically realistic: cortical E->E synapses depress (U~0.5)\nwhile I->E show weaker depression (U~0.25).")
            # Per-type STP parameter table
            dpg.add_text("Per-Connection-Type STP Parameters:", color=[150,200,220,255])
            dpg.add_text("(E->E, E->I, I->E, I->I)", color=[140,140,140,255])
            with dpg.table(header_row=True):
                dpg.add_table_column(label="Param", width_fixed=True, init_width_or_weight=80)
                dpg.add_table_column(label="E->E", width_stretch=True)
                dpg.add_table_column(label="E->I", width_stretch=True)
                dpg.add_table_column(label="I->E", width_stretch=True)
                dpg.add_table_column(label="I->I", width_stretch=True)
                with dpg.table_row():
                    dpg.add_text("U")
                    dpg.add_input_float(tag="cfg_stp_U_ee", default_value=0.5, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.3f", width=-1, min_value=0.0, max_value=1.0)
                    dpg.add_input_float(tag="cfg_stp_U_ei", default_value=0.5, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.3f", width=-1, min_value=0.0, max_value=1.0)
                    dpg.add_input_float(tag="cfg_stp_U_ie", default_value=0.25, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.3f", width=-1, min_value=0.0, max_value=1.0)
                    dpg.add_input_float(tag="cfg_stp_U_ii", default_value=0.25, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.3f", width=-1, min_value=0.0, max_value=1.0)
                with dpg.table_row():
                    dpg.add_text("Tau_d")
                    dpg.add_input_float(tag="cfg_stp_tau_d_ee", default_value=200.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
                    dpg.add_input_float(tag="cfg_stp_tau_d_ei", default_value=200.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
                    dpg.add_input_float(tag="cfg_stp_tau_d_ie", default_value=100.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
                    dpg.add_input_float(tag="cfg_stp_tau_d_ii", default_value=100.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
                with dpg.table_row():
                    dpg.add_text("Tau_f")
                    dpg.add_input_float(tag="cfg_stp_tau_f_ee", default_value=20.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
                    dpg.add_input_float(tag="cfg_stp_tau_f_ei", default_value=20.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
                    dpg.add_input_float(tag="cfg_stp_tau_f_ie", default_value=50.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
                    dpg.add_input_float(tag="cfg_stp_tau_f_ii", default_value=50.0, callback=_update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", width=-1, min_value=0.1)
            dpg.add_separator()
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Homeostasis:", dpg.add_checkbox, "cfg_enable_homeostasis", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Intrinsic homeostasis via adaptive firing thresholds.\nFor Izhikevich: adjusts spike threshold toward target rate.\nEssential for stable network dynamics over long simulations.")
            with dpg.group(tag="homeostasis_izh_specific_group", show=True):
                 with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                    dpg.add_table_column(width_stretch=True)
                    add_parameter_table_row("Homeostasis Target Rate (spikes/dt for Izh):", dpg.add_input_float, "cfg_homeostasis_target_rate", 0.02, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f",
                        tooltip="Desired firing probability per timestep.\n0.02 = ~2% chance of firing each dt.\nAt dt=0.5ms this corresponds to ~40 Hz.\nThreshold adapts to reach this target.")
                    add_parameter_table_row("Homeostasis Min Threshold (Izh, mV):", dpg.add_input_float, "cfg_homeostasis_threshold_min", -55.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f",
                        tooltip="Lower bound on adaptive firing threshold.\nPrevents threshold from dropping too low,\nwhich would cause pathological firing.\nShould be above resting potential (vr).")
                    add_parameter_table_row("Homeostasis Max Threshold (Izh, mV):", dpg.add_input_float, "cfg_homeostasis_threshold_max", -30.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f",
                        tooltip="Upper bound on adaptive firing threshold.\nPrevents threshold from rising too high,\nwhich would silence the neuron entirely.\nShould be below spike peak (vpeak).")

            # C1b: Synaptic Scaling Controls (Turrigiano 2008)
            dpg.add_separator()
            dpg.add_text("--- Synaptic Scaling (Homeostatic) ---", color=[100,200,200,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Synaptic Scaling:", dpg.add_checkbox, "cfg_enable_synaptic_scaling", False, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Multiplicative synaptic scaling (Turrigiano 2008).\nScales excitatory weights up/down to maintain target firing rate.\nComplementary to threshold homeostasis — works on synaptic strengths\nrather than intrinsic excitability.")
                add_parameter_table_row("Synaptic Scaling Rate:", dpg.add_input_float, "cfg_synaptic_scaling_rate", 0.001, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f",
                    tooltip="Rate of multiplicative weight scaling per timestep.\nHigher values = faster homeostatic correction but risk instability.\nTypical range: 0.0001–0.01. Default 0.001.")

            # C2: STDP Controls
            dpg.add_separator()
            dpg.add_text("--- STDP (Spike-Timing-Dependent Plasticity) ---", color=[100,200,200,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable STDP:", dpg.add_checkbox, "cfg_enable_stdp", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Spike-Timing-Dependent Plasticity (Bi & Poo 2001).\nPre-before-post = LTP, post-before-pre = LTD.\nBiological Hebbian learning with precise timing.")
                add_parameter_table_row("STDP A+ (LTP amplitude, 0.005-0.02):", dpg.add_input_float, "cfg_stdp_a_plus", 0.01, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f", min_value=0.0,
                    tooltip="Maximum weight increase for causal (pre→post) pairing.\nLarger A+ = faster potentiation.\nA- > A+ gives net depression bias (stable).")
                add_parameter_table_row("STDP A- (LTD amplitude, 0.005-0.02):", dpg.add_input_float, "cfg_stdp_a_minus", 0.0105, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f", min_value=0.0,
                    tooltip="Maximum weight decrease for anti-causal (post→pre) pairing.\nSlightly larger than A+ ensures net weight decrease\nfor random firing, preventing runaway excitation.")
                add_parameter_table_row("STDP Tau+ (LTP time constant, ms):", dpg.add_input_float, "cfg_stdp_tau_plus_ms", 20.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=1.0,
                    tooltip="Time window for LTP (pre-before-post).\n20ms matches cortical STDP data (Bi & Poo 2001).\nLarger tau = wider learning window.")
                add_parameter_table_row("STDP Tau- (LTD time constant, ms):", dpg.add_input_float, "cfg_stdp_tau_minus_ms", 20.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=1.0,
                    tooltip="Time window for LTD (post-before-pre).\n20ms standard. Asymmetric tau+/tau- gives\ndifferent temporal sensitivity for LTP vs LTD.")
                add_parameter_table_row("STDP Weight Min:", dpg.add_input_float, "cfg_stdp_w_min", 0.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f",
                    tooltip="Lower bound on STDP-modified weights.\n0 = synapses can be fully depressed.\nSet > 0 to maintain minimal connectivity.")
                add_parameter_table_row("STDP Weight Max:", dpg.add_input_float, "cfg_stdp_w_max", 2.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f",
                    tooltip="Upper bound on STDP-modified weights.\nPrevents individual synapses from becoming\ntoo strong. 2.0 = 2x initial weight.")
            
            # C2: Reward Modulation Controls
            dpg.add_separator()
            dpg.add_text("--- Reward-Modulated Plasticity ---", color=[100,200,200,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Reward Modulation:", dpg.add_checkbox, "cfg_enable_reward_modulation", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Three-factor learning: STDP eligibility traces\nare gated by a reward signal (Schultz 2002).\nRequires STDP enabled. Models dopaminergic modulation.")
                add_parameter_table_row("Reward Learning Rate (0.001-0.05):", dpg.add_input_float, "cfg_reward_learning_rate", 0.01, _update_sim_config_from_ui_and_signal_reset_needed, format="%.4f", min_value=0.0,
                    tooltip="Scales how strongly reward modulates weight changes.\nHigher = faster reward-driven learning but noisier.\nTypical: 0.001-0.05.")
                add_parameter_table_row("Eligibility Trace Tau (ms, 500-2000):", dpg.add_input_float, "cfg_reward_eligibility_tau_ms", 1000.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=10.0,
                    tooltip="Decay time for eligibility traces (ms).\nBridges the gap between STDP events and delayed reward.\n1000ms = 1 second memory of recent spike correlations.")
                add_parameter_table_row("Reward Baseline (expected reward):", dpg.add_input_float, "cfg_reward_baseline", 0.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.3f",
                    tooltip="Expected (average) reward level.\nWeight changes proportional to (reward - baseline).\n0 = any positive reward causes LTP.")
                add_parameter_table_row("Current Reward Signal:", dpg.add_input_float, "cfg_current_reward_signal", 0.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.3f",
                    tooltip="Current reward value (can be changed live).\nPositive = reinforce recent activity.\nNegative = suppress recent activity patterns.\nModels dopaminergic reward prediction error.")
            
            # C3: Structural Plasticity Controls
            dpg.add_separator()
            dpg.add_text("--- Structural Plasticity ---", color=[100,200,200,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Enable Structural Plasticity:", dpg.add_checkbox, "cfg_enable_structural_plasticity", True, _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Dynamic synapse formation and elimination.\nNew connections form between co-active neurons.\nWeak synapses are pruned. Models developmental\nand experience-dependent rewiring.")
                add_parameter_table_row("Formation Rate (per timestep, 1e-7 to 1e-5):", dpg.add_input_float, "cfg_struct_plast_formation_rate", 1e-6, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2e", min_value=0.0,
                    tooltip="Probability of new synapse creation per candidate pair\nper update interval. Very small values needed to\navoid explosive connectivity growth.")
                add_parameter_table_row("Elimination Rate (per timestep, 1e-7 to 1e-5):", dpg.add_input_float, "cfg_struct_plast_elimination_rate", 5e-7, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2e", min_value=0.0,
                    tooltip="Probability of pruning weak synapses per update interval.\nBalances formation rate. Higher elimination =\nmore aggressive pruning of unused connections.")
                add_parameter_table_row("Weight Threshold (eliminate below):", dpg.add_input_float, "cfg_struct_plast_weight_threshold", 0.05, _update_sim_config_from_ui_and_signal_reset_needed, format="%.3f", min_value=0.0,
                    tooltip="Synapses with weight below this value are candidates\nfor elimination. Higher threshold = more aggressive\npruning. 0.05 = prune very weak connections.")
                add_parameter_table_row("Target Connection Density (0-1):", dpg.add_input_float, "cfg_struct_plast_target_density", 0.1, _update_sim_config_from_ui_and_signal_reset_needed, format="%.3f", min_value=0.0, max_value=1.0,
                    tooltip="Target fraction of possible connections present.\n0.1 = 10% connectivity. Formation/elimination rates\nadjust to approach this density.")
                add_parameter_table_row("Distance Scale (spatial, units):", dpg.add_input_float, "cfg_struct_plast_distance_scale", 20.0, _update_sim_config_from_ui_and_signal_reset_needed, format="%.1f", min_value=1.0,
                    tooltip="Spatial scale for distance-dependent connection\nprobability. New synapses preferentially form between\nnearby neurons. Smaller = more local connectivity.")
                add_parameter_table_row("Update Interval (steps):", dpg.add_input_int, "cfg_struct_plast_update_interval_steps", 100, _update_sim_config_from_ui_and_signal_reset_needed, min_value=10, step=10,
                    tooltip="How often (in sim steps) to evaluate structural changes.\nCSR matrix rebuilds are expensive, so infrequent updates\n(100-1000 steps) are recommended.")
                add_parameter_table_row("Activity Bias (formation):", dpg.add_input_float, "cfg_struct_plast_activity_bias", 0.5, _update_sim_config_from_ui_and_signal_reset_needed, format="%.2f", min_value=0.0, max_value=1.0,
                    tooltip="Bias synapse formation toward co-active neuron pairs.\n0.0 = purely random formation.\n1.0 = fully activity-driven (Cline & Haas 2008).\n0.5 = 50/50 mix of co-activity-biased and random candidates.")

        with dpg.collapsing_header(label="Heterogeneity & Noise", default_open=False, tag="heterogeneity_noise_header"):
            dpg.add_text("Add biological realism through parameter variability and intrinsic noise.", wrap=label_col_width * 2, color=[200,200,200,255])
            dpg.add_spacer(height=5)
            
            dpg.add_text("--- Parameter Heterogeneity ---", color=[200,200,100,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row(
                    "Enable Parameter Heterogeneity:",
                    dpg.add_checkbox,
                    "cfg_enable_parameter_heterogeneity",
                    True,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Add neuron-to-neuron parameter variability.\nSamples from distributions matching experimental data\n(CV~0.3-0.4). More realistic than identical neurons."
                )
                add_parameter_table_row(
                    "Heterogeneity Seed (-1 = use main seed):",
                    dpg.add_input_int,
                    "cfg_heterogeneity_seed",
                    -1,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    min_value=-1,
                    step=1,
                    tooltip="RNG seed for parameter variability.\n-1 = use main simulation seed (deterministic).\nSet different values to explore different\ninstantiations of the same heterogeneity level."
                )
            
            dpg.add_text(
                "When enabled, parameters are sampled from distributions (CV~0.3-0.4) matching experimental data.",
                wrap=label_col_width * 2,
                color=[150,150,150,255]
            )
            
            dpg.add_spacer(height=8)
            dpg.add_separator()
            dpg.add_spacer(height=5)
            
            dpg.add_text("--- Channel & Background Noise ---", color=[200,200,100,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                
                # Conductance noise (HH only)
                add_parameter_table_row(
                    "Enable Conductance Noise (HH only):",
                    dpg.add_checkbox,
                    "cfg_enable_conductance_noise",
                    True,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Add stochastic fluctuations to ion channel conductances.\nModels channel noise from finite ion channel populations.\nOnly applies to Hodgkin-Huxley model."
                )
                add_parameter_table_row(
                    "Conductance Noise Std (relative, 0.05 = 5%):",
                    dpg.add_input_float,
                    "cfg_conductance_noise_relative_std",
                    0.05,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    format="%.3f",
                    min_value=0.0,
                    max_value=0.5,
                    tooltip="Standard deviation of conductance noise as fraction\nof max conductance. 0.05 = 5% noise.\nHigher values = more stochastic spiking."
                )
            
            dpg.add_spacer(height=5)
            dpg.add_separator()
            dpg.add_spacer(height=5)
            
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                
                # OU process
                add_parameter_table_row(
                    "Enable OU Process (background drive):",
                    dpg.add_checkbox,
                    "cfg_enable_ou_process",
                    True,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    tooltip="Ornstein-Uhlenbeck process for background synaptic drive.\nModels bombardment from ~10,000 unmodeled synapses.\nProduces realistic 2-5 mV membrane potential fluctuations."
                )
                add_parameter_table_row(
                    "OU Mean Current (pA):",
                    dpg.add_input_float,
                    "cfg_ou_mean_current_pA",
                    0.0,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    format="%.1f",
                    tooltip="Mean (DC offset) of background current.\n0 = symmetric fluctuations around zero.\nPositive = tonic depolarizing drive.\nNegative = tonic hyperpolarizing."
                )
                add_parameter_table_row(
                    "OU Std Current (pA, 50-200 typical):",
                    dpg.add_input_float,
                    "cfg_ou_std_current_pA",
                    100.0,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    format="%.1f",
                    min_value=0.0,
                    tooltip="Standard deviation of OU noise current.\nControls amplitude of Vm fluctuations.\n100 pA typical for Izhikevich. Scale for HH/AdEx."
                )
                add_parameter_table_row(
                    "OU Time Constant Tau (ms, 10-20 typical):",
                    dpg.add_input_float,
                    "cfg_ou_tau_ms",
                    15.0,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    format="%.1f",
                    min_value=1.0,
                    max_value=100.0,
                    tooltip="Temporal correlation time of background noise.\nSmall tau (~5 ms) = fast, white-noise-like.\nLarge tau (~20 ms) = slowly varying, colored noise.\n15 ms matches cortical synaptic timescales."
                )
                add_parameter_table_row(
                    "OU Seed (-1 = use main seed):",
                    dpg.add_input_int,
                    "cfg_ou_seed",
                    -1,
                    _update_sim_config_from_ui_and_signal_reset_needed,
                    min_value=-1,
                    step=1,
                    tooltip="RNG seed for OU noise process.\n-1 = use main simulation seed (deterministic).\nDifferent seeds give different noise realizations\nwhile preserving other simulation state."
                )
            
            dpg.add_text(
                "OU process adds temporally correlated background noise (2-5mV Vm fluctuations).",
                wrap=label_col_width * 2,
                color=[150,150,150,255]
            )

        with dpg.collapsing_header(label="Visual Settings", default_open=False, tag="visual_settings_header"):
            dpg.add_text("--- Neurons ---", color=[150,200,250,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                spiking_filter_options = ["Highlight Spiking", "Show Only Spiking", "No Spiking Highlight"]
                add_parameter_table_row("Show Spiking Neurons:", dpg.add_combo, "filter_spiking_mode_combo", "Highlight Spiking", trigger_filter_update_signal, items=spiking_filter_options,
                    tooltip="How to display spiking neurons.\nHighlight: bright flash on spike, dim otherwise.\nOnly Spiking: hide non-spiking neurons.\nNo Highlight: uniform appearance.")
                add_parameter_table_row("Enable Synaptic Pulses (GL):", dpg.add_checkbox, "gl_enable_synaptic_pulses_cb", opengl_viz_config.get('ENABLE_SYNAPTIC_PULSES', True) if OPENGL_AVAILABLE else False, handle_gl_enable_synaptic_pulses_change,
                    tooltip="Show animated pulses traveling along synapses\nwhen spikes propagate. Visually appealing but\ncosts GPU performance at high spike rates.")
                add_parameter_table_row("Filter By Neuron Type:", dpg.add_checkbox, "filter_type_enable_cb", False, lambda s, a, u: (dpg.configure_item("filter_neuron_type_combo", enabled=a), trigger_filter_update_signal(s,a,u)),
                    tooltip="Enable filtering to show only neurons of a specific type.\nUseful for isolating excitatory or inhibitory populations.")
                add_parameter_table_row("Select Type:", dpg.add_combo, "filter_neuron_type_combo", "All", trigger_filter_update_signal, items=["All"], enabled=False,
                    tooltip="Select which neuron type to display.\nRequires 'Filter By Neuron Type' to be enabled.")
                add_parameter_table_row("Max Visible Neurons (GL):", dpg.add_input_int, "gl_max_neurons_render_input", opengl_viz_config.get('MAX_NEURONS_TO_RENDER', 10000) if OPENGL_AVAILABLE else 0, handle_gl_max_neurons_change, min_value=0, step=100,
                    tooltip="Maximum neurons rendered in OpenGL viewport.\nReduce for better frame rate with large networks.\n10000 default. 0 = render all.")
                add_parameter_table_row("Neuron Size (GL):", dpg.add_slider_float, "gl_neuron_point_size_slider", opengl_viz_config.get('POINT_SIZE', 2.0) if OPENGL_AVAILABLE else 1.0, handle_gl_point_size_change, min_value=0.5, max_value=10.0, format="%.1f",
                    tooltip="Point size for neuron rendering in pixels.\nIncrease for visibility at distance, decrease\nfor dense networks to reduce overlap.")
                add_parameter_table_row("Inactive Neuron Opacity (GL):", dpg.add_slider_float, "gl_inactive_neuron_opacity_slider", opengl_viz_config.get('INACTIVE_NEURON_OPACITY', 0.25) if OPENGL_AVAILABLE else 0.1, handle_gl_inactive_neuron_opacity_change, min_value=0.0, max_value=1.0, format="%.2f",
                    tooltip="Transparency of non-spiking neurons.\n0.0 = fully transparent, 1.0 = fully opaque.\nLow values make spiking activity pop visually.")
            
            dpg.add_separator()
            dpg.add_text("--- Synapses ---", color=[150,200,250,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Show Synapses (GL):", dpg.add_checkbox, "filter_show_synapses_gl_cb", global_gui_state.get("show_connections_gl", True), lambda s,a,u: (global_gui_state.update({"show_connections_gl":a}), trigger_filter_update_signal()),
                    tooltip="Toggle synapse line rendering in OpenGL viewport.\nDisable for cleaner neuron-only view and\nbetter performance with dense networks.")
                add_parameter_table_row("Max Visible Connections (GL):", dpg.add_input_int, "gl_max_connections_render_input", opengl_viz_config.get('MAX_CONNECTIONS_TO_RENDER', 20000) if OPENGL_AVAILABLE else 0, handle_gl_max_connections_change, min_value=0, step=500,
                    tooltip="Maximum synapse lines rendered. Dense networks\nmay have millions of connections — cap this\nfor usable frame rates. 20000 default.")
                add_parameter_table_row("Synapse Alpha Multiplier (GL):", dpg.add_slider_float, "gl_synapse_alpha_slider", opengl_viz_config.get('SYNAPSE_ALPHA_MODIFIER', 0.3) if OPENGL_AVAILABLE else 0.1, handle_gl_synapse_alpha_change, min_value=0.0, max_value=2.0, format="%.2f",
                    tooltip="Opacity multiplier for synapse lines.\nLower values = more transparent connections.\nUseful to reduce visual clutter in dense networks.")
                add_parameter_table_row("Min Abs Synapse Weight (Filter):", dpg.add_slider_float, "filter_min_abs_weight_slider", 0.000, trigger_filter_update_signal, max_value=1.0, format="%.3f",
                    tooltip="Only show synapses with |weight| above this value.\nIncrease to see only the strongest connections.\n0 = show all connections.")
            
            dpg.add_separator()
            dpg.add_text("--- General Visuals ---", color=[150,200,250,255])
            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                dpg.add_table_column(width_stretch=True)
                add_parameter_table_row("Camera Field of View (FOV, degrees):", dpg.add_slider_float, "cfg_camera_fov", 60.0, _update_sim_config_from_ui_and_signal_reset_needed, min_value=10.0, max_value=120.0,
                    tooltip="Perspective camera field of view.\n60° is natural. Lower = telephoto (flatter).\nHigher = wide-angle (more depth distortion).")
                add_parameter_table_row("Activity Highlight Frames (GL):", dpg.add_input_int, "gl_activity_highlight_frames_input", opengl_viz_config.get('ACTIVITY_HIGHLIGHT_FRAMES', 7) if OPENGL_AVAILABLE else 1, handle_gl_activity_highlight_frames_change, min_value=1, max_value=30,
                    tooltip="How many frames a neuron stays highlighted after spiking.\nHigher = longer visible flash. 7 default.\nIncrease for slow sim speeds, decrease for fast.")
                add_parameter_table_row("Viz Update Interval (steps):", dpg.add_input_int, "cfg_viz_update_interval_steps", 1, _update_sim_config_from_ui_and_signal_reset_needed, min_value=1, max_value=200, step=1,
                    tooltip="Update visualization every N simulation steps.\n1 = real-time update (smoothest, most GPU overhead).\nHigher values = faster simulation but choppier visuals.")

        # =============================================================================
        # EXPERIMENT & STIMULUS SYSTEM UI
        # =============================================================================
        with dpg.collapsing_header(label="Experiment & Stimulus System", default_open=False, tag="experiment_system_header"):
            dpg.add_text("Configure and run programmable experiments with stimulus injection,\nneuron group I/O, training protocols, and readout analysis.")
            dpg.add_spacer(height=5)

            # --- Experiment Preset Selector ---
            dpg.add_text("Experiment Presets:", color=[180, 220, 255])
            experiment_preset_names = ["-- Select Preset --"] + ExperimentPresets.get_preset_names()
            dpg.add_combo(experiment_preset_names, default_value="-- Select Preset --",
                          tag="experiment_preset_combo", width=350,
                          callback=lambda s, a, u: _handle_experiment_preset_change(a))
            dpg.add_spacer(height=3)

            # Experiment info display
            dpg.add_text("No experiment loaded.", tag="experiment_info_text", color=[150, 150, 150])
            dpg.add_spacer(height=5)

            # --- Control Buttons ---
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start Experiment", tag="btn_start_experiment",
                               callback=lambda: ui_to_sim_queue.put({"type": "START_EXPERIMENT"}))
                dpg.add_button(label="Stop Experiment", tag="btn_stop_experiment",
                               callback=lambda: ui_to_sim_queue.put({"type": "STOP_EXPERIMENT"}))
                dpg.add_button(label="Save Log", tag="btn_save_experiment_log",
                               callback=lambda: ui_to_sim_queue.put({"type": "SAVE_EXPERIMENT_LOG",
                                   "filepath": f"experiment_log_{int(time.time())}.json"}))
            dpg.add_spacer(height=5)

            # --- Experiment Status Display ---
            dpg.add_text("Status:", color=[180, 220, 255])
            dpg.add_text("Idle", tag="experiment_status_text", color=[150, 150, 150])
            dpg.add_spacer(height=3)

            # Phase progress
            dpg.add_text("Phase: --", tag="experiment_phase_text", color=[150, 150, 150])
            dpg.add_spacer(height=3)

            # Readout rates display
            dpg.add_text("Readout Rates:", color=[180, 220, 255])
            dpg.add_text("No data", tag="experiment_readout_text", color=[150, 150, 150])
            dpg.add_spacer(height=3)

            # Training progress
            dpg.add_text("Training:", color=[180, 220, 255])
            dpg.add_text("No training active", tag="experiment_training_text", color=[150, 150, 150])
            dpg.add_spacer(height=5)

            # --- Manual Stimulus Configuration ---
            with dpg.collapsing_header(label="Manual Stimulus (Quick Test)", default_open=False,
                                       tag="manual_stimulus_sub_header", indent=10):
                dpg.add_text("Inject a simple stimulus into the network without\nsetting up a full experiment.", color=[150, 150, 150])
                dpg.add_spacer(height=3)
                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=label_col_width)
                    dpg.add_table_column(width_stretch=True)
                    add_parameter_table_row("Stimulus Amplitude (pA):", dpg.add_input_float,
                        "manual_stim_amplitude", 150.0, None, min_value=0.0, max_value=5000.0,
                        tooltip="Peak current amplitude in picoamperes.\n100-300 pA typical for driving activity.")
                    add_parameter_table_row("Pattern:", dpg.add_combo,
                        "manual_stim_pattern_combo", "CONSTANT",
                        None, items=["CONSTANT", "PULSE_TRAIN", "SINUSOIDAL", "POISSON_SPIKE_TRAIN", "GAUSSIAN_NOISE"],
                        tooltip="Stimulus waveform type.\nCONSTANT: DC step current\nPULSE_TRAIN: Repeated brief pulses\nSINUSOIDAL: Oscillatory current")
                    add_parameter_table_row("Target Group Size:", dpg.add_input_int,
                        "manual_stim_group_size", 100, None, min_value=1, max_value=10000,
                        tooltip="Number of neurons in the stimulus target group.\nSelects the first N neurons in the network.")
                    add_parameter_table_row("Duration (ms):", dpg.add_input_float,
                        "manual_stim_duration", 500.0, None, min_value=10.0, max_value=100000.0,
                        tooltip="How long the stimulus will be active in milliseconds.")
                dpg.add_spacer(height=3)
                dpg.add_button(label="Inject Stimulus", tag="btn_inject_manual_stimulus",
                               callback=_handle_inject_manual_stimulus)
                dpg.add_spacer(height=5)

        with dpg.collapsing_header(label="Testing & Optimization", default_open=False, tag="perf_testing_header"):
            dpg.add_text("Run performance tests and optimization tasks:")
            dpg.add_spacer(height=3)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Run Benchmark Suite", tag="run_benchmark_button", callback=handle_run_benchmark_click, width=-1)
            
            dpg.add_spacer(height=3)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Run Viz Performance Test", tag="run_viz_benchmark_button", callback=handle_run_viz_benchmark_click, width=-80)
                dpg.add_checkbox(label="Quick", tag="viz_benchmark_quick_mode_checkbox", default_value=False)
            
            dpg.add_spacer(height=3)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Run Auto-Tuning (Optimize Drive Scales)", tag="run_optimization_button", callback=handle_run_optimization_click, width=-80)
                dpg.add_checkbox(label="Quick", tag="optimization_quick_mode_checkbox", default_value=False)
            
            dpg.add_spacer(height=3)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Stop Running Test", tag="stop_perf_test_button", callback=handle_stop_perf_test_click, width=-1, enabled=False)
            
            dpg.add_spacer(height=3)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reload Auto-Tuned Overrides", tag="reload_overrides_button", callback=handle_reload_overrides_click, width=-1)
            
            dpg.add_spacer(height=5)
            dpg.add_text("Status:", color=[150,200,250,255])
            dpg.add_text("Ready", tag="perf_test_status_text", wrap=label_col_width * 2)
            
            dpg.add_spacer(height=3)
            dpg.add_text("Results:", color=[150,200,250,255])
            dpg.add_input_text(default_value="", tag="perf_test_results_text", multiline=True, readonly=True, height=80, width=-1)

        with dpg.collapsing_header(label="System Logs", default_open=False, tag="system_logs_header"):
            dpg.add_text("Search logs:")
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag="log_search_input", width=220, callback=handle_log_search_change)
                dpg.add_button(label="Previous", tag="log_search_prev_button", callback=handle_log_search_prev, width=70, enabled=False)
                dpg.add_button(label="Next", tag="log_search_next_button", callback=handle_log_search_next, width=70, enabled=False)
            
            dpg.add_text("0 / 0 matches", tag="log_search_match_text")
            dpg.add_spacer(height=3)
            
            def toggle_log_autoscroll(sender, checked):
                """Toggle autoscroll tracking on/off for the log field."""
                if dpg.does_item_exist("system_logs_display"):
                    dpg.configure_item("system_logs_display", tracked=checked, track_offset=1.0 if checked else 0.0)
            
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Auto-scroll", tag="log_autoscroll_checkbox", default_value=True, callback=toggle_log_autoscroll)
                dpg.add_button(label="Clear Logs", tag="clear_logs_button", callback=handle_clear_logs_click, width=100)
                dpg.add_button(label="Export Logs", tag="export_logs_button", callback=handle_export_logs_click, width=100)
            
            dpg.add_spacer(height=3)
            with dpg.child_window(tag="system_logs_scroll_container", width=-1, height=-1, horizontal_scrollbar=False):
                # Auto-scroll is on by default via tracked=True and track_offset=1.0
                dpg.add_input_text(default_value="", tag="system_logs_display", multiline=True, readonly=True, 
                                 tracked=True, track_offset=1.0, width=-1, height=0)

    # File Dialogs
    profile_dir = global_simulation_bridge.PROFILE_DIR if global_simulation_bridge else "simulation_profiles/"
    checkpoint_dir_h5 = global_simulation_bridge.CHECKPOINT_DIR if global_simulation_bridge else "simulation_checkpoints_h5/" # Updated
    recording_dir_h5 = global_simulation_bridge.RECORDING_DIR if global_simulation_bridge else "simulation_recordings_h5/"   # Updated

    for p_dir in [profile_dir, checkpoint_dir_h5, recording_dir_h5]:
        if not os.path.exists(p_dir): os.makedirs(p_dir, exist_ok=True)

    # Profile dialogs (JSON)
    with dpg.file_dialog(directory_selector=False, show=False, callback=save_profile_dialog_callback, tag="save_profile_file_dialog", width=700, height=400, modal=True, default_path=profile_dir, default_filename="profile"):
        dpg.add_file_extension(".json", color=(255, 255, 0, 255), custom_text="JSON Profile (*.json)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")
    with dpg.file_dialog(directory_selector=False, show=False, callback=load_profile_dialog_callback, tag="load_profile_file_dialog", width=700, height=400, modal=True, default_path=profile_dir):
        dpg.add_file_extension(".json", color=(255, 255, 0, 255), custom_text="JSON Profile (*.json)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")

    # Checkpoint dialogs (HDF5) - use .h5 as filter (DPG doesn't handle compound extensions well)
    with dpg.file_dialog(directory_selector=False, show=False, callback=save_checkpoint_dialog_callback_h5,
                         tag="save_checkpoint_file_dialog_h5", width=700, height=400, modal=True, default_path=checkpoint_dir_h5, default_filename="checkpoint"):
        dpg.add_file_extension(".h5", color=(0, 200, 200, 255), custom_text="Checkpoint Files (*.simstate.h5)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")

    with dpg.file_dialog(directory_selector=False, show=False, callback=load_checkpoint_dialog_callback_h5,
                         tag="load_checkpoint_file_dialog_h5", width=700, height=400, modal=True, default_path=checkpoint_dir_h5):
        dpg.add_file_extension(".h5", color=(0, 200, 200, 255), custom_text="Checkpoint Files (*.simstate.h5)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")

    # Recording dialogs (HDF5) - use .h5 as filter (DPG doesn't handle compound extensions well)
    with dpg.file_dialog(directory_selector=False, show=False, callback=save_recording_for_streaming_dialog_callback_h5,
                         tag="save_recording_file_dialog_h5", width=700, height=400, modal=True, default_path=recording_dir_h5, default_filename="recording"):
        dpg.add_file_extension(".h5", color=(150, 0, 200, 255), custom_text="Recording Files (*.simrec.h5)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")

    # Load recording dialog
    with dpg.file_dialog(directory_selector=False, show=False, callback=load_recording_dialog_callback_h5,
                         tag="load_recording_file_dialog_h5", width=700, height=400, modal=True, default_path=recording_dir_h5):
        dpg.add_file_extension(".h5", color=(150, 0, 200, 255), custom_text="Recording Files (*.simrec.h5)")
        dpg.add_file_extension(".*", custom_text="All Files (*.*)")

    # Recording memory warning popup
    with dpg.window(label="Recording Too Large for GPU", tag="recording_memory_warning_popup",
                    modal=True, show=False, width=450, height=200, no_resize=True, no_collapse=True,
                    pos=[300, 250], no_close=True):
        dpg.add_text("", tag="recording_memory_warning_text", wrap=420)
        dpg.add_spacer(height=15)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Partial Cache", width=130, callback=_recording_memory_popup_partial_cache)
            with dpg.tooltip(dpg.last_item()):
                dpg.add_text("Cache as many frames as will fit in GPU memory.\n"
                            "Remaining frames will stream from disk.", wrap=250)
            dpg.add_button(label="Stream Only", width=130, callback=_recording_memory_popup_stream_only)
            with dpg.tooltip(dpg.last_item()):
                dpg.add_text("Stream all frames from disk (no GPU caching).\n"
                            "Uses minimal GPU memory but playback may be slower.", wrap=250)
            dpg.add_button(label="Cancel", width=80, callback=_recording_memory_popup_cancel)

    # Recording options popup (for large-scale recordings)
    with dpg.window(label="Recording Options", tag="recording_options_popup",
                    modal=True, show=False, width=420, height=320, no_resize=True, no_collapse=True,
                    pos=[280, 180], no_close=True):
        dpg.add_text("Configure recording settings before selecting output file.", wrap=400)
        dpg.add_spacer(height=10)

        # Recording mode
        dpg.add_text("Recording Mode:")
        dpg.add_combo(
            items=["gpu_buffered", "streaming"],
            default_value="gpu_buffered",
            tag="rec_opt_mode_combo",
            width=250
        )
        with dpg.tooltip(dpg.last_item()):
            dpg.add_text(
                "gpu_buffered: Buffer frames in GPU/CPU memory, write at end.\n"
                "  Best for short recordings that fit in memory.\n\n"
                "streaming: Write frames to disk during simulation.\n"
                "  Required for long recordings or limited memory.",
                wrap=300
            )
        dpg.add_spacer(height=10)

        # Skip synaptic data
        dpg.add_checkbox(
            label="Skip synaptic data (neuron-only recording)",
            tag="rec_opt_skip_synaptic",
            default_value=False
        )
        with dpg.tooltip(dpg.last_item()):
            dpg.add_text(
                "For large networks (100K+ neurons), synaptic data can be 10-20x larger "
                "than neuron data. Enable this to dramatically reduce recording size.\n\n"
                "Example: 100K neurons, 10M synapses:\n"
                "  Full frame: ~165MB\n"
                "  Neuron-only: ~10MB (16x smaller)",
                wrap=300
            )
        dpg.add_spacer(height=10)

        # Frame skip
        dpg.add_text("Frame skip (0 = disabled):")
        dpg.add_input_int(
            tag="rec_opt_frame_skip",
            default_value=0,
            min_value=0,
            max_value=1000,
            min_clamped=True,
            max_clamped=True,
            width=100
        )
        with dpg.tooltip(dpg.last_item()):
            dpg.add_text(
                "0 or 1 = record every frame (no skipping)\n"
                "10 = record every 10th frame (10x smaller files)\n"
                "100 = record every 100th frame (100x smaller files)\n\n"
                "For dt=1ms, frame_skip=10 gives 10ms temporal resolution.",
                wrap=300
            )
        dpg.add_spacer(height=20)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Continue", width=150, callback=_recording_options_continue_callback)
            dpg.add_button(label="Cancel", width=100, callback=_recording_options_cancel_callback)

# --- Main Application Loop Functions ---

def simulation_worker_loop(sim_bridge, local_shutdown_event, command_q, data_q):
    """
    Main loop for the dedicated simulation thread.
    Handles commands from the UI thread and runs simulation steps.
    """
    print("Simulation worker thread started.")
    # sim_bridge.ui_queue is already set to the global sim_to_ui_queue in its __init__
    
    last_sim_batch_time = time.perf_counter()
    # Max number of simulation steps to run before checking command queue / yielding
    # This helps keep the simulation thread responsive to commands.
    MAX_STEPS_PER_BATCH = 60 # e.g., if dt=0.1ms, this is 10ms of sim time
    # How often to send data updates to UI (in terms of simulation steps)
    # Lower = more responsive visualization at cost of more GPU→CPU transfers
    # For 60 FPS visualization: Update every 1-2 steps for real-time display
    DATA_UPDATE_INTERVAL_STEPS = 1 # Real-time visualization (60 FPS capable)
    SYNAPSE_SAMPLE_UPDATE_INTERVAL_STEPS = 200 # Update synapse samples much less frequently

    try:
        while not local_shutdown_event.is_set():
            # --- 1. Process Commands from UI Thread ---
            try:
                # Use exception handling instead of empty() check to avoid TOCTOU race
                while True:
                    try:
                        command = command_q.get_nowait()
                    except queue.Empty:
                        break
                    cmd_type = command.get("type")

                    if cmd_type == "START_SIM":
                        sim_bridge.start_simulation()
                    elif cmd_type == "STOP_SIM":
                        sim_bridge.stop_simulation()
                    elif cmd_type == "PAUSE_SIM":
                        sim_bridge.pause_simulation()
                    elif cmd_type == "RESUME_SIM":
                        sim_bridge.resume_simulation()
                    elif cmd_type == "STEP_SIM_ONE_MS":
                        dt_ms_val = sim_bridge.core_config.dt_ms if sim_bridge.core_config.dt_ms > 0 else 0.1
                        steps_for_1ms = max(1, int(round(1.0 / dt_ms_val)))
                        sim_bridge.step_simulation(num_steps=steps_for_1ms)
                    elif cmd_type == "APPLY_CONFIG_AND_RESET":
                        sim_bridge.apply_simulation_configuration_core(command["config_dict"])
                        # After applying, send back the (potentially modified by from_dict) config and initial data
                        if sim_bridge.is_initialized:
                            data_q.put({
                                "type": "CONFIG_APPLIED_AND_RESET_DONE",
                                "new_config_dict": sim_bridge.get_current_simulation_configuration_dict(),
                                "initial_gui_data": sim_bridge.get_initial_sim_data_snapshot()
                            })
                        else:
                            data_q.put({"type": "CONFIG_APPLIED_ERROR", "reason": "Initialization failed after apply"})
                    elif cmd_type == "SET_SIM_SPEED":
                        sim_bridge.set_simulation_speed_factor(command["factor"])
                    elif cmd_type == "SAVE_CHECKPOINT":
                        sim_bridge.save_checkpoint(command["filepath"]) # Sim_bridge will send status to UI
                    elif cmd_type == "LOAD_CHECKPOINT":
                        sim_bridge.load_checkpoint(command["filepath"]) # Sim_bridge sends status/data
                    elif cmd_type == "SET_RECORDING_OPTIONS":
                        # Update gpu_config with recording options from UI
                        if "recording_mode" in command:
                            sim_bridge.gpu_config.recording_mode = command["recording_mode"]
                        if "recording_skip_synaptic_data" in command:
                            sim_bridge.gpu_config.recording_skip_synaptic_data = command["recording_skip_synaptic_data"]
                        if "recording_frame_skip" in command:
                            sim_bridge.gpu_config.recording_frame_skip = max(1, command["recording_frame_skip"])
                        sim_bridge._log_console(
                            f"Recording options set: mode={sim_bridge.gpu_config.recording_mode}, "
                            f"skip_synaptic={sim_bridge.gpu_config.recording_skip_synaptic_data}, "
                            f"frame_skip={sim_bridge.gpu_config.recording_frame_skip}"
                        )
                    elif cmd_type == "START_RECORDING":
                        sim_bridge.start_recording_to_file(command["filepath"])  # Sim_bridge sends status
                    elif cmd_type == "STOP_RECORDING":
                        sim_bridge.stop_recording() # Sim_bridge sends status
                    elif cmd_type == "LOAD_RECORDING": # UI requests sim_thread to prepare metadata
                        stream_only = command.get("stream_only", False)
                        sim_bridge.load_recording(command["filepath"], stream_only=stream_only) # Sim_bridge sends RECORDING_METADATA_PREPARED or _FAILED
                    elif cmd_type == "SETUP_PLAYBACK_FROM_RECORDING":
                        # This command implies UI has received RECORDING_METADATA_PREPARED
                        # and now tells sim_thread to use that data to set its state.
                        # The 'active_recording_data_source' is UI state. Sim thread needs the HDF5 group/handle.
                        # This flow needs refinement: SimThread should hold its own H5 handle for playback.
                        # When UI commands "LOAD_RECORDING", sim_thread opens file, prepares meta, keeps handle.
                        # When UI commands "ENTER_PLAYBACK_MODE", sim_thread uses its handle.
                        active_playback_handle = command.get("h5_file_handle_for_sim_thread") # UI must pass this
                        initial_state_group_name = command.get("initial_state_group_name", "initial_state")
                        
                        if active_playback_handle and hasattr(active_playback_handle, 'get'): # Check if it's a valid h5py group/file
                            initial_state_group = active_playback_handle.get(initial_state_group_name)
                            if initial_state_group:
                                sim_bridge._apply_config_and_initial_state_from_recording(
                                    command["config_snapshot"], 
                                    initial_state_group # Pass the HDF5 group object
                                )
                            else:
                                sim_bridge._log_to_ui(f"Playback setup error: initial_state group '{initial_state_group_name}' not found in HDF5.", "error")
                                data_q.put({"type": "PLAYBACK_SETUP_FAILED", "reason": "Initial state group missing"})
                        else:
                            sim_bridge._log_to_ui("Playback setup error: Invalid HDF5 handle provided to sim_thread.", "error")
                            data_q.put({"type": "PLAYBACK_SETUP_FAILED", "reason": "Invalid H5 handle"})

                    elif cmd_type == "SET_PLAYBACK_FRAME":
                        active_playback_handle_for_frame = command.get("h5_file_handle_for_sim_thread")
                        num_frames_for_prefetch = command.get("num_frames")
                        if active_playback_handle_for_frame:
                            sim_bridge.set_playback_frame(
                                command["frame_index"],
                                active_playback_handle_for_frame,
                                num_frames=num_frames_for_prefetch
                            )
                        else:
                             sim_bridge._log_to_ui("Playback error: No HDF5 handle for SET_PLAYBACK_FRAME.", "error")
                             data_q.put({"type": "PLAYBACK_ERROR", "reason": "Missing H5 handle for frame set"})
                    elif cmd_type == "EXIT_PLAYBACK_MODE":
                        # Sim_thread doesn't directly manage global_gui_state.is_playback_mode_active.
                        # It just needs to reset its internal state if it was in a playback-specific mode.
                        # For example, if it was holding an HDF5 file open for playback, it should close it.
                        # The main task is to re-apply the "live" simulation config.
                        sim_bridge._clear_prefetch_buffer()  # Clean up prefetch resources
                        sim_bridge.apply_simulation_configuration_core(sim_bridge.core_config.to_dict(), is_part_of_playback_setup=False)
                        data_q.put({
                            "type": "PLAYBACK_EXITED_SIM_SIDE",
                            "new_config_dict": sim_bridge.get_current_simulation_configuration_dict(),
                            "initial_gui_data": sim_bridge.get_initial_sim_data_snapshot()
                        })

                    # --- Experiment System Commands ---
                    elif cmd_type == "LOAD_EXPERIMENT_PRESET":
                        preset_name = command.get("preset_name", "")
                        try:
                            exp_config = ExperimentPresets.get_preset(preset_name)
                            if exp_config:
                                sim_bridge.experiment_config = exp_config
                                if sim_bridge.is_initialized:
                                    sim_bridge.experiment_engine = ExperimentEngine(
                                        sim_bridge.core_config.num_neurons,
                                        sim_bridge.core_config.dt_ms
                                    )
                                    sim_bridge.experiment_engine.load_experiment(exp_config)
                                    sim_bridge.experiment_engine.initialize(
                                        cp_traits=sim_bridge.cp_traits, cp_module=cp
                                    )
                                data_q.put({
                                    "type": "EXPERIMENT_LOADED",
                                    "name": exp_config.name,
                                    "description": exp_config.description,
                                    "num_phases": len(exp_config.phases),
                                    "num_channels": len(exp_config.stimulus_channels),
                                    "num_groups": len(exp_config.neuron_groups),
                                })
                            else:
                                data_q.put({"type": "EXPERIMENT_ERROR", "reason": f"Unknown preset: {preset_name}"})
                        except Exception as e:
                            data_q.put({"type": "EXPERIMENT_ERROR", "reason": str(e)})

                    elif cmd_type == "LOAD_EXPERIMENT_CONFIG":
                        try:
                            config_dict = command.get("config_dict", {})
                            exp_config = experiment_config_from_dict(config_dict)
                            sim_bridge.experiment_config = exp_config
                            if sim_bridge.is_initialized:
                                sim_bridge.experiment_engine = ExperimentEngine(
                                    sim_bridge.core_config.num_neurons,
                                    sim_bridge.core_config.dt_ms
                                )
                                sim_bridge.experiment_engine.load_experiment(exp_config)
                                sim_bridge.experiment_engine.initialize(
                                    cp_traits=sim_bridge.cp_traits, cp_module=cp
                                )
                            data_q.put({
                                "type": "EXPERIMENT_LOADED",
                                "name": exp_config.name,
                                "description": exp_config.description,
                                "num_phases": len(exp_config.phases),
                                "num_channels": len(exp_config.stimulus_channels),
                                "num_groups": len(exp_config.neuron_groups),
                            })
                        except Exception as e:
                            data_q.put({"type": "EXPERIMENT_ERROR", "reason": str(e)})

                    elif cmd_type == "START_EXPERIMENT":
                        if sim_bridge.experiment_engine is not None:
                            sim_bridge.experiment_engine.start(sim_bridge.runtime_state.current_time_ms)
                            data_q.put({"type": "EXPERIMENT_STARTED"})
                        else:
                            data_q.put({"type": "EXPERIMENT_ERROR", "reason": "No experiment loaded"})

                    elif cmd_type == "STOP_EXPERIMENT":
                        if sim_bridge.experiment_engine is not None:
                            sim_bridge.experiment_engine.stop()
                            data_q.put({"type": "EXPERIMENT_STOPPED"})

                    elif cmd_type == "GET_EXPERIMENT_STATUS":
                        if sim_bridge.experiment_engine is not None:
                            status = sim_bridge.experiment_engine.get_experiment_status()
                            data_q.put({"type": "EXPERIMENT_STATUS", "status": status})
                        else:
                            data_q.put({"type": "EXPERIMENT_STATUS", "status": {"is_running": False}})

                    elif cmd_type == "SAVE_EXPERIMENT_LOG":
                        if sim_bridge.experiment_engine is not None:
                            filepath = command.get("filepath", "experiment_log.json")
                            try:
                                sim_bridge.experiment_engine.save_log(filepath)
                                data_q.put({"type": "EXPERIMENT_LOG_SAVED", "filepath": filepath})
                            except Exception as e:
                                data_q.put({"type": "EXPERIMENT_ERROR", "reason": f"Log save failed: {e}"})

                    command_q.task_done()
            except queue.Empty:
                pass # No commands from UI

            # --- 2. Run Simulation Step if Active ---
            if sim_bridge.is_initialized and sim_bridge.runtime_state.is_running and \
               not sim_bridge.runtime_state.is_paused:
                
                current_perf_time = time.perf_counter()
                elapsed_real_time_s = current_perf_time - last_sim_batch_time
                dt_ms_val = sim_bridge.core_config.dt_ms
                if dt_ms_val is None or dt_ms_val <= 0: dt_ms_val = 0.1 # Safety

                sim_time_to_advance_ms = elapsed_real_time_s * 1000.0 * sim_bridge.runtime_state.simulation_speed_factor
                
                num_steps_to_run_total = 0
                if dt_ms_val > 0:
                    num_steps_to_run_total = int(sim_time_to_advance_ms / dt_ms_val)

                if num_steps_to_run_total > 0:
                    steps_executed_in_batch = 0
                    for _ in range(min(num_steps_to_run_total, MAX_STEPS_PER_BATCH)):
                        if sim_bridge.runtime_state.current_time_ms < sim_bridge.core_config.total_simulation_time_ms:
                            sim_bridge._run_one_simulation_step() # Core simulation logic
                            sim_bridge.runtime_state.current_time_step += 1
                            # Compute time from step count to avoid floating point drift
                            sim_bridge.runtime_state.current_time_ms = sim_bridge.runtime_state.current_time_step * dt_ms_val
                            steps_executed_in_batch +=1

                            # Periodically send data to UI
                            if sim_bridge.runtime_state.current_time_step % DATA_UPDATE_INTERVAL_STEPS == 0:
                                latest_data = sim_bridge.get_latest_simulation_data_for_gui(force_fetch=True)
                                if data_q and latest_data:
                                    data_q.put({"type": "SIM_DATA_UPDATE", "data": latest_data})
                        else: # Total simulation time reached
                            sim_bridge.stop_simulation() # Sets flags
                            data_q.put({"type": "SIM_STOPPED_OR_ENDED", "reason": "Total time reached"})
                            break 
                    
                    # Adjust last_sim_batch_time based on simulated time processed
                    if steps_executed_in_batch > 0:
                        last_sim_batch_time += (steps_executed_in_batch * dt_ms_val) / \
                                               (sim_bridge.runtime_state.simulation_speed_factor * 1000.0 
                                                if sim_bridge.runtime_state.simulation_speed_factor > 0 else 1000.0)
                else: # No steps to run, but sim is active, so just update time to prevent large jump on resume
                    last_sim_batch_time = time.perf_counter()

            else: # Simulation not running or paused
                last_sim_batch_time = time.perf_counter() # Keep resetting to avoid large jump
                time.sleep(0.01) # Yield CPU if sim is idle or paused
    
    except Exception as e_worker:
        print(f"FATAL ERROR in simulation_worker_loop: {e_worker}")
        import traceback; traceback.print_exc()
        if data_q: data_q.put({"type": "SIM_FATAL_ERROR", "error": str(e_worker)})
    finally:
        print("Simulation worker thread finished.")


def main_dpg_loop_and_gl_idle():
    """
    Main loop for DPG rendering, processing messages from sim_thread, and driving OpenGL updates.
    If GLUT is used, this function is set as GLUT's idle function.
    """
    global global_simulation_bridge, global_gui_state, shutdown_flag, glut_window_id # Ensure glut_window_id is global if used here
    
    if shutdown_flag.is_set(): # Check for shutdown signal
        if OPENGL_AVAILABLE and glut.glutGetWindow() != 0 : # Check if a GLUT window exists
            try:
                current_glut_window = glut.glutGetWindow()
                if current_glut_window != 0: # Ensure we have a valid window ID
                    print(f"Shutdown signaled: Attempting to exit GLUT loop...")
                    # Try glutLeaveMainLoop first (freeglut), fallback to DestroyWindow
                    try:
                        glut.glutLeaveMainLoop()
                    except AttributeError:
                        # glutLeaveMainLoop not available, use DestroyWindow
                        glut.glutDestroyWindow(current_glut_window)
            except Exception as e_glut_shutdown:
                 print(f"Exception during GLUT shutdown: {e_glut_shutdown}")

        # Ensure DPG is also signaled to stop if it hasn't already by the on_close callback.
        if dpg.is_dearpygui_running():
            dpg.stop_dearpygui()

        # Don't call sys.exit() from within GLUT callback - just return and let main loop handle exit
        return

    if not dpg.is_dearpygui_running(): # If DPG window was closed by user (on_close already ran)
        # This block might be redundant if the above shutdown_flag block handles everything,
        # but it's a safeguard.
        if not shutdown_flag.is_set(): # If on_close didn't set it for some reason
            print("DPG not running, setting shutdown_flag from idle loop.")
            shutdown_flag.set() 
        # The shutdown_flag.is_set() block above will then handle GLUT termination.
        return

    # --- 1. Process Messages from Simulation Thread ---
    try:
        while not sim_to_ui_queue.empty():
            message = sim_to_ui_queue.get_nowait()
            msg_type = message.get("type")

            if msg_type == "STATUS_UPDATE":
                update_status_bar(message.get("text","Status N/A"), message.get("color"), message.get("level","info"))
            elif msg_type == "SIM_DATA_UPDATE":
                data_payload = message.get("data")
                if data_payload:
                    update_monitoring_overlay_values(data_payload)
                    with global_viz_data_cache["gl_render_data_lock"]:
                        global_viz_data_cache["gl_render_data_buffer"] = data_payload
                    global_viz_data_cache["gl_render_data_available"].set()
                    # Update experiment UI if experiment status is present
                    exp_status = data_payload.get("experiment_status")
                    if exp_status is not None:
                        _update_experiment_ui_from_status(exp_status)
            elif msg_type == "SIM_STOPPED_OR_ENDED":
                global_gui_state["_sim_is_running_ui_view"] = False
                global_gui_state["_sim_is_paused_ui_view"] = False
                update_ui_for_simulation_run_state(is_running=False, is_paused=False)
                update_status_bar(f"Simulation stopped/ended: {message.get('reason', '')}", level="info")
                if global_simulation_bridge: 
                     initial_data = global_simulation_bridge.get_initial_sim_data_snapshot()
                     if initial_data: update_monitoring_overlay_values(initial_data)
            elif msg_type == "CONFIG_APPLIED_AND_RESET_DONE":
                # Repopulate UI from the configuration actually used by the sim thread.
                # This ensures any profile/model-specific defaults or auto-tuned overrides
                # are reflected in the visible parameters.
                new_cfg_full = message.get("new_config_dict")
                if new_cfg_full:
                    _populate_ui_from_config_dict(new_cfg_full)

                # Update monitoring values and GL snapshot
                update_monitoring_overlay_values(message["initial_gui_data"])
                
                initial_gl_data = message.get("initial_gui_data")
                if initial_gl_data:
                    with global_viz_data_cache["gl_render_data_lock"]:
                        global_viz_data_cache["gl_render_data_buffer"] = initial_gl_data
                    global_viz_data_cache["gl_render_data_available"].set() 

                update_ui_for_simulation_run_state(is_running=False, is_paused=False) 
                global_gui_state["reset_sim_needed_from_ui_change"] = False
                update_status_bar("Configuration applied and simulation reset.", color=[0,200,0,255], level="success")
            elif msg_type == "CHECKPOINT_LOADED_SUCCESS":
                _populate_ui_from_config_dict(message["config_dict"])
                apply_gui_configuration_core(message.get("gui_settings_from_checkpoint",{}))
                update_monitoring_overlay_values(message["initial_gui_data"])
                # Also push this initial data to GL cache
                initial_gl_data_chkpt = message.get("initial_gui_data")
                if initial_gl_data_chkpt:
                    with global_viz_data_cache["gl_render_data_lock"]:
                        global_viz_data_cache["gl_render_data_buffer"] = initial_gl_data_chkpt
                    global_viz_data_cache["gl_render_data_available"].set()

                update_ui_for_simulation_run_state(is_running=False, is_paused=False)
                global_gui_state["reset_sim_needed_from_ui_change"] = False
                update_status_bar("Checkpoint loaded successfully.", color=[0,200,0,255], level="success")
            elif msg_type == "RECORDING_METADATA_PREPARED":
                update_ui_after_recording_loaded(message["data"]) 
            elif msg_type == "RECORDING_STARTED": # Sim thread confirms recording started
                update_ui_for_recording_state(is_recording_active_ui=True)
                update_status_bar(f"Recording started: {os.path.basename(message.get('filepath','N/A'))}", color=[0,150,200,255], level="info")
            elif msg_type == "RECORDING_FINALIZED":
                update_ui_for_recording_state(is_recording_active_ui=False)
                if message.get("success"):
                    update_status_bar(f"Recording saved: {os.path.basename(message.get('filepath','N/A'))}", color=[0,200,0,255], level="success")
                else:
                    update_status_bar("Recording finalization failed or was cancelled by sim.", color=[255,0,0,255], level="error")
            elif msg_type == "PLAYBACK_READY": 
                global_gui_state["is_playback_mode_active"] = True
                global_gui_state["current_playback_frame_index"] = 0 
                global_gui_state["playback_is_playing_ui"] = False 
                update_ui_for_playback_mode_state(is_playback_active_ui=True, num_frames_from_meta=global_gui_state.get("active_recording_data_source",{}).get("num_frames",0))
                update_monitoring_overlay_values(message.get("initial_gui_data")) 
                # Push this initial frame data to GL
                initial_pb_gl_data = message.get("initial_gui_data")
                if initial_pb_gl_data:
                    with global_viz_data_cache["gl_render_data_lock"]:
                        global_viz_data_cache["gl_render_data_buffer"] = initial_pb_gl_data
                    global_viz_data_cache["gl_render_data_available"].set()
                update_status_bar("Playback mode ready. Use playback controls.", level="info")
            elif msg_type == "PLAYBACK_FRAME_APPLIED":
                global_gui_state["current_playback_frame_index"] = message["frame_index"]
                update_monitoring_overlay_values(message["gui_data"])
                if dpg.does_item_exist("playback_slider"): 
                    if dpg.get_value("playback_slider") != message["frame_index"]:
                        dpg.set_value("playback_slider", message["frame_index"])
                # Push new frame data to GL
                pb_frame_gl_data = message.get("gui_data")
                if pb_frame_gl_data:
                    with global_viz_data_cache["gl_render_data_lock"]:
                        global_viz_data_cache["gl_render_data_buffer"] = pb_frame_gl_data
                    global_viz_data_cache["gl_render_data_available"].set()
            elif msg_type == "PLAYBACK_EXITED_SIM_SIDE":
                global_gui_state["is_playback_mode_active"] = False
                global_gui_state["playback_is_playing_ui"] = False
                if global_gui_state.get("active_recording_data_source") and \
                   global_gui_state["active_recording_data_source"].get("h5_file_obj_for_playback"):
                    try:
                        # The H5 file handle is owned by sim_thread; UI thread shouldn't close it.
                        # Sim thread should close it when it processes EXIT_PLAYBACK_MODE.
                        # We just clear the reference in UI state.
                        print("Playback exited on sim side. UI clearing its reference to HDF5 data source.")
                    except Exception as e_close_h5_ui:
                        print(f"Error clearing HDF5 ref on playback exit (UI): {e_close_h5_ui}")
                global_gui_state["active_recording_data_source"] = None 
                
                _populate_ui_from_config_dict(message["new_config_dict"]) 
                update_monitoring_overlay_values(message["initial_gui_data"]) 
                # Push this initial live data to GL
                live_initial_gl_data = message.get("initial_gui_data")
                if live_initial_gl_data:
                    with global_viz_data_cache["gl_render_data_lock"]:
                        global_viz_data_cache["gl_render_data_buffer"] = live_initial_gl_data
                    global_viz_data_cache["gl_render_data_available"].set()

                update_ui_for_playback_mode_state(is_playback_active_ui=False) 
                update_status_bar("Exited playback mode. Live simulation mode restored.", level="info")

            # --- Experiment System Messages ---
            elif msg_type == "EXPERIMENT_LOADED":
                exp_name = message.get("name", "Unknown")
                n_phases = message.get("num_phases", 0)
                n_channels = message.get("num_channels", 0)
                n_groups = message.get("num_groups", 0)
                dpg.set_value("experiment_info_text",
                              f"Loaded: {exp_name}\n  {n_phases} phases, {n_channels} channels, {n_groups} groups")
                dpg.configure_item("experiment_info_text", color=[100, 255, 100])
                update_status_bar(f"Experiment loaded: {exp_name}", color=[100, 200, 255, 255])
            elif msg_type == "EXPERIMENT_STARTED":
                update_status_bar("Experiment started", color=[100, 255, 100, 255])
            elif msg_type == "EXPERIMENT_STOPPED":
                update_status_bar("Experiment stopped", color=[255, 200, 100, 255])
            elif msg_type == "EXPERIMENT_LOG_SAVED":
                update_status_bar(f"Experiment log saved: {message.get('filepath', '')}", color=[100, 255, 100, 255])
            elif msg_type == "EXPERIMENT_ERROR":
                update_status_bar(f"Experiment error: {message.get('reason', 'Unknown')}", color=[255, 100, 100, 255])

            elif msg_type in ["CONFIG_APPLIED_ERROR", "CHECKPOINT_LOAD_FAILED", "RECORDING_LOAD_FAILED",
                              "RECORDING_START_FAILED", "PLAYBACK_SETUP_FAILED", "PLAYBACK_ERROR", "SIM_FATAL_ERROR",
                              "CHECKPOINT_SAVE_FAILED"]:
                update_status_bar(f"Error: {message.get('reason', message.get('error', 'Unknown error'))}", color=[255,0,0,255], level="error")
                if msg_type == "SIM_FATAL_ERROR": shutdown_flag.set() 

            sim_to_ui_queue.task_done()
    except queue.Empty:
        pass 

    # --- 1.5. Update Log Display ---
    if hasattr(handle_log_search_change, "log_capture"):
        if dpg.is_dearpygui_running() and dpg.does_item_exist("system_logs_display"):
            log_capture = handle_log_search_change.log_capture
            logs = log_capture.get_logs()
            if logs:
                # Show ALL logs
                display_text = "\n".join(logs)
                current_value = dpg.get_value("system_logs_display")
                if current_value != display_text:
                    dpg.set_value("system_logs_display", display_text)
                    # Update input_text height based on text size for proper scrolling
                    FRAME_PADDING = 3
                    text_size = dpg.get_text_size(display_text)
                    if text_size is not None:
                        dpg.set_item_height("system_logs_display", text_size[1] + (2 * FRAME_PADDING))

    # --- 2. Handle UI-Driven Playback Stepping (if active and playing) ---
    if global_gui_state.get("is_playback_mode_active", False) and global_gui_state.get("playback_is_playing_ui", False):
        current_time_ui = time.perf_counter()
        time_since_last_step_ui = current_time_ui - global_gui_state.get("last_playback_autostep_time_ui", 0.0)
        playback_interval_ui = 1.0 / global_gui_state.get("playback_fps_ui", 30.0)

        if time_since_last_step_ui >= playback_interval_ui:
            active_rec_meta = global_gui_state.get("active_recording_data_source")
            if active_rec_meta and active_rec_meta.get("h5_file_obj_for_playback"): # Ensure handle is there
                num_frames = active_rec_meta.get("num_frames", 0)
                current_frame_idx_ui = global_gui_state.get("current_playback_frame_index", 0)
                next_frame_idx = current_frame_idx_ui + 1
                
                if num_frames > 0 and next_frame_idx < num_frames:
                    ui_to_sim_queue.put({
                        "type": "SET_PLAYBACK_FRAME", 
                        "frame_index": next_frame_idx,
                        "h5_file_handle_for_sim_thread": active_rec_meta["h5_file_obj_for_playback"] 
                    })
                else: 
                    global_gui_state["playback_is_playing_ui"] = False 
                    update_ui_for_playback_playing_state(is_playing=False)
            global_gui_state["last_playback_autostep_time_ui"] = current_time_ui

    # --- 3. DPG Rendering ---
    if dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    # --- 4. OpenGL Rendering with 60 FPS Frame Rate Limiting ---
    if OPENGL_AVAILABLE and glut.glutGetWindow() != 0:
        global gl_last_render_time, gl_target_frame_time
        
        current_time = time.perf_counter()
        time_since_last_frame = current_time - gl_last_render_time
        
        # Only render if enough time has passed for 60 FPS (16.67ms)
        # OR if filters changed (force immediate update)
        should_render = (time_since_last_frame >= gl_target_frame_time) or global_gui_state.get("filters_changed", False)
        
        if should_render:
            # Update GL data if new data available or filters changed
            if global_viz_data_cache["gl_render_data_available"].is_set() or global_gui_state.get("filters_changed", False):
                update_gl_data() 
                global_viz_data_cache["gl_render_data_available"].clear()
            
            try: 
                current_win_gl = glut.glutGetWindow()
                if glut_window_id is not None and current_win_gl != 0 and current_win_gl != glut_window_id : 
                    glut.glutSetWindow(glut_window_id) # Ensure correct GL context
                if current_win_gl != 0: # Only post redisplay if window exists
                    glut.glutPostRedisplay()
                    gl_last_render_time = current_time  # Update last render time
            except Exception: pass

    if not OPENGL_AVAILABLE and dpg.is_dearpygui_running(): # DPG only mode
        time.sleep(0.005) # Prevent DPG-only loop from busy-waiting excessively


# --- Headless auto-tuning runner -------------------------------------------------

def _evaluate_candidate_config(sim_bridge, core_cfg, viz_cfg, total_time_ms):
    """Initializes sim_bridge with the given config, runs a short headless simulation,
    and returns basic activity/connectivity metrics for auto-tuning.
    """
    # Reset any previous state and GPU memory
    sim_bridge.clear_simulation_state_and_gpu_memory()
    sim_bridge.core_config = core_cfg
    sim_bridge.viz_config = viz_cfg
    sim_bridge.runtime_state = RuntimeState()

    dt = core_cfg.dt_ms if core_cfg.dt_ms > 0 else 0.0
    sim_bridge.runtime_state.max_delay_steps = int(core_cfg.max_synaptic_delay_ms / dt) if dt > 0 else 200

    sim_bridge._initialize_simulation_data(called_from_playback_init=False)
    if not sim_bridge.is_initialized or core_cfg.num_neurons <= 0 or dt <= 0:
        sim_bridge._log_console("Auto-tune: initialization failed or invalid config.", "warning")
        return None

    n = core_cfg.num_neurons
    num_steps = int(total_time_ms / dt)
    if num_steps <= 0:
        return None

    ever_spiked = cp.zeros(n, dtype=bool)
    total_spikes = 0

    for _ in range(num_steps):
        sim_bridge._run_one_simulation_step()
        sim_bridge.runtime_state.current_time_step += 1
        # Compute time from step count to avoid floating point drift
        sim_bridge.runtime_state.current_time_ms = sim_bridge.runtime_state.current_time_step * dt

        fired = sim_bridge.cp_firing_states
        if fired is None:
            break
        ever_spiked = cp.logical_or(ever_spiked, fired)
        step_spikes = int(cp.sum(fired).get())
        total_spikes += step_spikes

    num_synapses = int(sim_bridge.cp_connections.nnz) if sim_bridge.cp_connections is not None else 0
    num_spiking_neurons = int(cp.sum(ever_spiked).get())
    avg_spikes_per_step = total_spikes / float(max(1, num_steps))
    total_time_s = (dt * num_steps) / 1000.0
    avg_spikes_per_neuron_hz = 0.0
    if n > 0 and total_time_s > 0.0:
        avg_spikes_per_neuron_hz = total_spikes / (n * total_time_s)
    spiking_fraction = num_spiking_neurons / float(n) if n > 0 else 0.0

    return {
        "num_neurons": n,
        "num_synapses": num_synapses,
        "num_steps": num_steps,
        "total_spikes": total_spikes,
        "avg_spikes_per_step": avg_spikes_per_step,
        "avg_spikes_per_neuron_hz": avg_spikes_per_neuron_hz,
        "spiking_neuron_fraction": spiking_fraction,
    }


def _score_auto_tune_metrics(metrics):
    """Scores a candidate based on firing activity and neuron participation."""
    n = metrics.get("num_neurons", 0)
    if n <= 0:
        return -1.0

    total_spikes = metrics.get("total_spikes", 0)
    num_synapses = metrics.get("num_synapses", 0)
    if total_spikes <= 0 or num_synapses <= 0:
        return -1.0

    avg_spikes_per_step = metrics.get("avg_spikes_per_step", 0.0)
    spiking_fraction = metrics.get("spiking_neuron_fraction", 0.0)

    frac_spikes_per_step = avg_spikes_per_step / float(n) if n > 0 else 0.0

    # Desired range (fraction of neurons spiking per step on average)
    target_frac = 0.10
    min_frac = 0.02
    max_frac = 0.30

    in_range = min_frac <= frac_spikes_per_step <= max_frac

    # Component 1: closeness to target firing fraction
    diff = abs(frac_spikes_per_step - target_frac)
    score_firing = max(0.0, 1.0 - diff / max(target_frac, 1e-6))

    # Component 2: fraction of neurons that ever spiked
    target_spiking_fraction = 0.3
    score_participation = min(1.0, spiking_fraction / max(target_spiking_fraction, 1e-6))

    base_score = 0.6 * score_firing + 0.4 * score_participation

    # Penalty if firing is outside desired window
    if not in_range:
        base_score -= 0.5

    return float(base_score)


def run_auto_tuning(quick=False):
    """Headless auto-tuning entry point.

    When quick=True, only a small subset of combinations is tuned for faster testing.
    """
    print(f"Starting auto-tuning workflow (quick={quick})...")
    sim_bridge = SimulationBridge()

    # Profiles to sweep
    if "NEURAL_STRUCTURE_PROFILES" in globals():
        profile_names = sorted(NEURAL_STRUCTURE_PROFILES.keys())
    else:
        profile_names = ["GENERIC_UNSTRUCTURED"]

    if quick:
        profile_names = profile_names[:2]

    # Models to tune: HH + AdEx (Izhikevich already behaves well in most cases)
    models_to_tune = [NeuronModel.HODGKIN_HUXLEY, NeuronModel.ADEX]

    # Determine HH presets to tune per profile, respecting realism constraints.
    # For structured profiles, this will typically be a single region-appropriate
    # preset; generic/unstructured profiles fall back to all HH types.
    profile_to_hh_types = {}
    for profile_name in profile_names:
        allowed_names = get_compatible_hh_type_names_for_profile(profile_name)
        hh_list = []
        for name in allowed_names:
            if name in NeuronType.__members__ and name.startswith("HH_"):
                hh_list.append(NeuronType[name])
        if not hh_list:
            hh_list = [nt for nt in NeuronType if nt.name.startswith("HH_")]
        if quick and len(hh_list) > 3:
            hh_list = hh_list[:3]
        profile_to_hh_types[profile_name] = hh_list

    tuned_combos = {}
    num_hh_combos = sum(len(ts) for ts in profile_to_hh_types.values())
    num_adex_combos = len(profile_names)
    total_combos = num_hh_combos + num_adex_combos
    combo_index = 0

    hh_scales = [0.5, 1.0, 2.0, 4.0]
    adex_scales = [0.5, 1.0, 1.5, 2.0, 3.0]

    # Hodgkin-Huxley tuning: per (profile, HH preset)
    for profile_name in profile_names:
        hh_types_for_profile = profile_to_hh_types.get(profile_name, [])
        for hh_type in hh_types_for_profile:
            combo_index += 1
            key = f"{NeuronModel.HODGKIN_HUXLEY.name}|{profile_name}|{hh_type.name}"
            print(f"[{combo_index}/{total_combos}] Tuning {key} ...")

            best_score = -1e9
            best_scale = None
            best_metrics = None

            for scale in hh_scales:
                core_cfg = CoreSimConfig()
                core_cfg.neuron_model_type = NeuronModel.HODGKIN_HUXLEY.name
                core_cfg.neural_profile_name = profile_name
                core_cfg.default_neuron_type_hh = hh_type.name
                core_cfg.num_neurons = 400 if not quick else 250
                core_cfg.dt_ms = 1.0
                core_cfg.total_simulation_time_ms = 600.0 if not quick else 300.0
                core_cfg.hh_external_drive_scale = scale

                viz_cfg = VisualizationConfig()
                metrics = _evaluate_candidate_config(sim_bridge, core_cfg, viz_cfg, core_cfg.total_simulation_time_ms)
                if metrics is None:
                    continue

                score = _score_auto_tune_metrics(metrics)
                metrics["score"] = score
                metrics["selected_scale"] = scale

                if score > best_score:
                    best_score = score
                    best_scale = scale
                    best_metrics = metrics

            if best_scale is not None:
                tuned_combos[key] = {
                    "neuron_model_type": NeuronModel.HODGKIN_HUXLEY.name,
                    "neural_profile_name": profile_name,
                    "default_neuron_type_hh": hh_type.name,
                    "core_overrides": {"hh_external_drive_scale": float(best_scale)},
                    "metrics": best_metrics,
                }
            else:
                tuned_combos[key] = {
                    "neuron_model_type": NeuronModel.HODGKIN_HUXLEY.name,
                    "neural_profile_name": profile_name,
                    "default_neuron_type_hh": hh_type.name,
                    "core_overrides": {},
                    "metrics": {"note": "no viable candidate found"},
                }

    # AdEx tuning: per (profile, model) only
    for profile_name in profile_names:
        combo_index += 1
        key = f"{NeuronModel.ADEX.name}|{profile_name}|NONE"
        print(f"[{combo_index}/{total_combos}] Tuning {key} ...")

        best_score = -1e9
        best_scale = None
        best_metrics = None

        for scale in adex_scales:
            core_cfg = CoreSimConfig()
            core_cfg.neuron_model_type = NeuronModel.ADEX.name
            core_cfg.neural_profile_name = profile_name
            core_cfg.num_neurons = 400 if not quick else 250
            core_cfg.dt_ms = 1.0
            core_cfg.total_simulation_time_ms = 800.0 if not quick else 400.0
            core_cfg.adex_external_drive_scale = scale

            viz_cfg = VisualizationConfig()
            metrics = _evaluate_candidate_config(sim_bridge, core_cfg, viz_cfg, core_cfg.total_simulation_time_ms)
            if metrics is None:
                continue

            score = _score_auto_tune_metrics(metrics)
            metrics["score"] = score
            metrics["selected_scale"] = scale

            if score > best_score:
                best_score = score
                best_scale = scale
                best_metrics = metrics

        if best_scale is not None:
            tuned_combos[key] = {
                "neuron_model_type": NeuronModel.ADEX.name,
                "neural_profile_name": profile_name,
                "default_neuron_type_hh": None,
                "core_overrides": {"adex_external_drive_scale": float(best_scale)},
                "metrics": best_metrics,
            }
        else:
            tuned_combos[key] = {
                "neuron_model_type": NeuronModel.ADEX.name,
                "neural_profile_name": profile_name,
                "default_neuron_type_hh": None,
                "core_overrides": {},
                "metrics": {"note": "no viable candidate found"},
            }

    # Persist results
    os.makedirs(os.path.dirname(AUTO_TUNED_OVERRIDES_PATH), exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "tuned_combinations": tuned_combos,
    }
    with open(AUTO_TUNED_OVERRIDES_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Auto-tuning complete. Wrote {len(tuned_combos)} combinations to {AUTO_TUNED_OVERRIDES_PATH}.")
    return 0


def load_viz_benchmark_hardware_note():
    """Loads the hardware performance note from viz benchmark results if available.
    
    Returns:
        str: Hardware note if found, empty string otherwise
    """
    viz_results_path = os.path.join("benchmarks", "viz_performance_results.json")
    
    if not os.path.exists(viz_results_path):
        return ""
    
    try:
        with open(viz_results_path, 'r') as f:
            results = json.load(f)
        
        hardware_note = results.get("hardware_performance_note", "")
        if hardware_note:
            print(f"Loaded hardware performance note from {viz_results_path}")
            return hardware_note
    except Exception as e:
        print(f"Warning: Could not load viz benchmark results from {viz_results_path}: {e}")
    
    return ""


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_flag
    print("\nCtrl+C detected. Shutting down gracefully...")
    shutdown_flag.set()


def main():
    global global_simulation_bridge, simulation_thread, shutdown_flag, glut_window_id
    # global_gui_state, global_viz_data_cache, opengl_viz_config are already defined globally.
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Use the global log capture instance that was started at module load
    global _global_log_capture
    # Store reference in handler function for access
    handle_log_search_change.log_capture = _global_log_capture

    dpg.create_context()
    dpg.configure_app(docking=False)

    global_simulation_bridge = SimulationBridge() # Initialize the simulation core (sim_bridge.ui_queue is set here)

    # Attempt to load default profile (JSON) - This is a UI-side operation before sim_thread starts.
    default_profile_filename = "default_profile.json"
    default_profile_path = os.path.join(global_simulation_bridge.PROFILE_DIR, default_profile_filename)
    loaded_default_sim_config_dict = None
    loaded_default_gui_config_dict = None

    if os.path.exists(default_profile_path):
        try:
            with open(default_profile_path, 'r', encoding='utf-8') as f: profile_content = json.load(f)
            loaded_default_sim_config_dict = profile_content.get("simulation_configuration")
            loaded_default_gui_config_dict = profile_content.get("gui_configuration")
            if loaded_default_sim_config_dict:
                print(f"Default profile '{default_profile_filename}' found. Will apply after UI setup.")
                global_gui_state["current_profile_name"] = default_profile_filename
            else: print(f"Default profile '{default_profile_filename}' is missing simulation_configuration.")
        except Exception as e:
            print(f"Error loading default profile '{default_profile_path}': {e}")
            loaded_default_sim_config_dict = None; loaded_default_gui_config_dict = None
    
    if loaded_default_sim_config_dict is None: # Fallback if no valid default profile
        print("Using basic default internal configuration for initial UI population.")
        loaded_default_sim_config_dict = SimulationConfiguration().to_dict() # Use fresh defaults
        global_gui_state["current_profile_name"] = "unsaved_internal_defaults.json"
    
    # Load hardware performance notes from benchmarks
    # Priority: benchmark_results.json (comprehensive) > viz_performance_results.json (viz-only)
    _load_benchmark_limits()  # Parse benchmark_results.json into HARDWARE_LIMITS
    hardware_note = get_hardware_note()  # From benchmark_results.json
    if not hardware_note:
        hardware_note = load_viz_benchmark_hardware_note()  # Fallback to viz benchmark
    if hardware_note and loaded_default_sim_config_dict:
        loaded_default_sim_config_dict["hardware_performance_note"] = hardware_note


    # DPG Viewport setup
    dpg_viewport_width = 700 
    dpg_viewport_height = int(SCREEN_HEIGHT * 0.90) if SCREEN_HEIGHT > 300 else 760 
    dpg.create_viewport(title="Neuron Simulator Controls (DPG)",
                        width=dpg_viewport_width, height=dpg_viewport_height,
                        x_pos=0, y_pos=20) # Position on the left

    create_gui_layout() # Create all DPG widgets
    dpg.set_primary_window("controls_monitor_window", True)

    # Populate UI with the (default or profile-loaded) simulation configuration
    _populate_ui_from_config_dict(loaded_default_sim_config_dict)
    if loaded_default_gui_config_dict: # Apply GUI settings from profile if they exist
        apply_gui_configuration_core(loaded_default_gui_config_dict)
    
    # Ensure hardware note is displayed (direct widget update after UI population)
    if hardware_note and dpg.does_item_exist("cfg_hardware_performance_note"):
        dpg.set_value("cfg_hardware_performance_note", hardware_note)
    
    if dpg.does_item_exist("profile_name_input"): # Show current profile name
        dpg.set_value("profile_name_input", global_gui_state["current_profile_name"].replace(".json", ""))

    # Populate neuron type filter based on the initial model type in UI
    if dpg.does_item_exist("filter_neuron_type_combo") and dpg.does_item_exist("cfg_neuron_model_type"):
        initial_model_name = dpg.get_value("cfg_neuron_model_type")
        _toggle_model_specific_params_visibility(None, initial_model_name) # Updates filter items

    dpg.setup_dearpygui()
    dpg.show_viewport()
    
    # Initial UI state updates
    update_ui_for_simulation_run_state(is_running=False, is_paused=False)
    update_ui_for_recording_state(is_recording_active_ui=False)
    update_ui_for_playback_mode_state(is_playback_active_ui=False) # Hides playback controls initially
    update_monitoring_overlay_values(None) # Clear monitor

    global_gui_state["reset_sim_needed_from_ui_change"] = True # Force "Apply Changes" for initial config
    update_status_bar("Application started. Apply initial config or load a profile/state.", level="info")


    # --- Start the Simulation Worker Thread ---
    # Sim_thread will initialize sim_bridge with the config currently reflected in the UI.
    # So, send an "APPLY_CONFIG_AND_RESET" with current UI config as the first command.
    initial_config_from_ui = _update_sim_config_from_ui(update_model_specific=True)
    if initial_config_from_ui:
        ui_to_sim_queue.put({
            "type": "APPLY_CONFIG_AND_RESET",
            "config_dict": initial_config_from_ui
        })
    else: # Should not happen if UI is built correctly
        print("CRITICAL: Failed to get initial config from UI for sim_thread.")
        # Sim_thread will start with default SimulationConfiguration in sim_bridge.

    simulation_thread = threading.Thread(target=simulation_worker_loop, 
                                         args=(global_simulation_bridge, shutdown_flag, ui_to_sim_queue, sim_to_ui_queue),
                                         daemon=True) # Daemon so it exits if main thread crashes
    simulation_thread.start()


    # --- Main Loop (DPG + OpenGL if available) ---
    if OPENGL_AVAILABLE:
        glut.glutInit(sys.argv if hasattr(sys, "argv") and sys.argv else ["sim3d_threaded.py"]) # Init GLUT
        glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH) # Display modes

        gl_win_width = SCREEN_WIDTH - dpg_viewport_width - 30 if SCREEN_WIDTH > dpg_viewport_width + 30 else 600
        gl_win_height = dpg_viewport_height; gl_win_width = max(400, gl_win_width); gl_win_height = max(300, gl_win_height)
        gl_win_x_pos = dpg_viewport_width + 10 

        glut.glutInitWindowPosition(gl_win_x_pos, 20); glut.glutInitWindowSize(gl_win_width, gl_win_height)
        try: glut_window_id = glut.glutCreateWindow(b"3D Network Visualization (OpenGL - Threaded)")
        except TypeError: glut_window_id = glut.glutCreateWindow("3D Network Visualization (OpenGL - Threaded)")

        opengl_viz_config['WINDOW_WIDTH'] = glut.glutGet(glut.GLUT_WINDOW_WIDTH)
        opengl_viz_config['WINDOW_HEIGHT'] = glut.glutGet(glut.GLUT_WINDOW_HEIGHT)

        init_gl(); # Initialize OpenGL state (VBOs, etc.)
        glut.glutDisplayFunc(render_scene_gl); 
        glut.glutReshapeFunc(reshape_gl_window) 
        glut.glutKeyboardFunc(keyboard_func_gl); 
        glut.glutMouseFunc(mouse_button_func_gl) 
        glut.glutMotionFunc(mouse_motion_func_gl); 
        glut.glutIdleFunc(main_dpg_loop_and_gl_idle) # Main loop function for GLUT

        # Initial GL data population (empty or from first sim_to_ui message)
        # update_gl_data() will be called by main_dpg_loop_and_gl_idle when data is available.
        print("Starting GLUT main loop (with DPG integration)...")
        try: glut.glutMainLoop()
        except Exception as e_glut: print(f"Exception during GLUT main loop: {e_glut}")
        finally:
            print("Exited GLUT main loop."); shutdown_flag.set()
            if dpg.is_dearpygui_running(): dpg.stop_dearpygui()
    else: # No OpenGL, run DPG only
        print("OpenGL not available. Running DPG controls only.")
        while dpg.is_dearpygui_running() and not shutdown_flag.is_set():
            main_dpg_loop_and_gl_idle(); # Call the DPG part of the loop
            # time.sleep(0.005) # Replaced by queue checks and DPG's own timing
        if dpg.is_dearpygui_running(): dpg.stop_dearpygui()

    # --- Cleanup ---
    shutdown_flag.set() # Ensure flag is set for sim_thread if not already
    if simulation_thread and simulation_thread.is_alive():
        print("Waiting for simulation worker thread to finish...")
        simulation_thread.join(timeout=5.0) # Wait for sim_thread
        if simulation_thread.is_alive():
            print("Warning: Simulation thread did not terminate gracefully.")

    # Clean up OpenGL VBOs to prevent GPU memory leaks
    if OPENGL_AVAILABLE:
        try:
            vbo_list = [gl_neuron_pos_vbo, gl_neuron_color_vbo, gl_synapse_vertices_vbo, gl_pulse_vertices_vbo]
            valid_vbos = [v for v in vbo_list if v is not None and v > 0]
            if valid_vbos:
                from OpenGL.GL import glDeleteBuffers
                glDeleteBuffers(len(valid_vbos), valid_vbos)
                print(f"Cleaned up {len(valid_vbos)} OpenGL VBOs.")
        except Exception as e:
            print(f"Note: OpenGL VBO cleanup skipped ({e})")

    if dpg.is_dearpygui_running(): dpg.destroy_context()
    print("Neuron simulator application shutdown complete.")

if __name__ == '__main__':
    # If launched with --auto-tune, run the headless tuning workflow instead of the GUI.
    if '--auto-tune' in sys.argv:
        quick = '--quick' in sys.argv
        exit_code = run_auto_tuning(quick=quick)
        sys.exit(exit_code)
    else:
        main()
