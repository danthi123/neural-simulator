"""
gap#1 / A1 — TWO RUNGS toward a FULLY-SPIKING mouth, on top of the population few-spike read (2026-08-13 6/6 GO).

The parent (`_wkv_fewspike_read_derisk`, finding 2026-08-13) put the FLUENT WKV open-prose generation onto the
production few-spike Izhikevich read regime at ideal-sampler parity (read_fidelity>=0.936, P>=8 population coding).
It named its next two rungs, in order:

  RUNG 1 — add the SHARED-INHIBITORY FS-WTA (`build_fswta_score_bridge`/`fswta_drive`, the production word-decode
    mechanism) to SHARPEN the winner and CUT the spike budget below the P=8 population size. The parent's read is
    INDEPENDENT pools + OU noise + argmax-over-firing (no lateral competition), so it must out-VOTE runner-up noise
    with population size (~56 word spikes at rw20/P8). A shared inhibitory FS pool (each word pool excites FS; FS
    inhibits ALL word pools) makes the winner fire first -> recruit FS -> SUPPRESS the runners-up -> a clean one-of-K
    at a MUCH smaller population (P<8) and fewer total spikes. Hypothesis: read_fidelity>=0.936 AND fluent free-gen
    at a LOWER spike budget than P=8.

  RUNG 2 — route the state->logits projection through READ-OUT NEURONS so the WTA drive is a SYNAPTIC CURRENT,
    retiring the HOST matmul on the read path. The parent (and this file's rung 1) still compute `logits =
    head_w @ (r_h*(Wo_sp@state)) + head_b` as a HOST matmul, top-K, softmax -> a host-designed labelled-line drive.
    Rung 2 realises the FINAL logit projection `head_w @ h` (V x D, the dominant matmul, `h = r_h*(Wo_sp@state)` the
    gated hidden state) as EXCITATORY SYNAPSES from a rate-coded hidden population onto V word pools, with the fm
    signed-read-out surpass (Dale-shift + a feedforward common-mode CANCELLER; `_fm_spiking_synaptic_readout_derisk`).
    The winner emerges from a FS-WTA over ALL V pools competing on synaptic drive -> NO host logit matmul, NO top-K
    argpartition on the read path. HONEST residual (mapped, not hidden): the hidden `h = r_h*(Wo_sp@state)` is still
    host (the Wo_sp projection + the multiplicative r_h gate need gain modulation -> the next rung). If the graded
    state -> spiking logits is LOSSY (tiny margins over 1000 near-tied words + Poisson quantization + the Dale
    common-mode), that BOUNDARY is the first-class deliverable (quantified, with the next lever named).

DECISIVE metric (identical to the parent, calibration-robust): the read is a SAMPLER; the ceiling is an IDEAL host
sampler over the SAME top-K softmax. read_fidelity = ondist_mass(read) / ondist_mass(host_sample). Plus top-1
argmax-agreement, mean spikes/read (the "few-spike" budget), and FREE-GENERATION survival (self-NLL of the model's own
continuation under the graded read-out). Anti-cheats each MUST collapse: equal-drive (drive all active pools equally ->
uniform), scramble (permute label->pool -> chance), noise-ablation (ou_std->0 -> deterministic), provenance (winner
from cp_firing_states, 0 host categorical draws on the read path). Instrument validity: top-K mass coverage,
mass_fewspike <= mass_argmax, scramble -> chance.

Reuse-by-import: WKVReadout + the metric harness from `_wkv_fewspike_read_derisk`; the shared-FS bridge pattern from
`_d3_spiking_attractor_derisk.build_fswta_score_bridge`; the Dale-shift + common-mode canceller from
`_fm_spiking_synaptic_readout_derisk`. NO `sim/` edit — drives + reads public bridge arrays. cfg.seed-controlled
substrate (CLAUDE.md seed trap). Runner-only, default-off.

Run (smoke rung1):  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_fswta_synaptic_read_derisk \
                      --rungs 1 --smoke --seeds 42
Run (smoke rung2):  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_fswta_synaptic_read_derisk \
                      --rungs 2 --smoke --seeds 42
Run (6-seed):       SIM_BACKEND=cupy  .venv/bin/python -m research.runners._wkv_fswta_synaptic_read_derisk \
                      --rungs 1,2 --seeds 42,43,44,100,101,102 \
                      --json research/findings/raw/_wkv_fswta_synaptic_6seed.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.bridge import SimulationBridge  # noqa: E402
from sim.backend import to_host, from_host, get_backend  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402

# reuse the deployed read-out + the metric harness (NO copy)
from research.runners._wkv_fewspike_read_derisk import (  # noqa: E402
    WKVReadout, _softmax, _native, _load_eval,
)
from tools.lab import lever  # noqa: E402


# ================================================================================================================
# RUNG 1 — the SHARED-INHIBITORY FS-WTA few-spike read (production word-decode mechanism).
# ================================================================================================================
class FSWTAWordRead:
    """K word pools (P neurons each) + one shared INHIBITORY FS pool (lateral inhibition). Each word pool excites FS;
    FS inhibits ALL word pools. The winner (highest drive) fires first -> recruits FS -> SUPPRESSES the runners-up ->
    a clean one-of-K winner at a SMALL P (few spikes). Drive = the model's top-K softmax mass (same labelled-line map
    as the parent); OU membrane noise keeps the winner stochastic ~ softmax(drive/T) (the sampler is preserved). The
    winner is read from `cp_firing_states` accumulated per-pool firing (only the winner survives the inhibition)."""

    def __init__(self, n_pools, pop, seed, ou_std=200.0, base_pA=60.0, gain_pA=160.0, read_window=20,
                 n_fs=24, exc_to_fs=2.0, fs_to_exc=9.0):
        self.K = int(n_pools); self.P = int(pop); self.n_fs = int(n_fs)
        self.base_pA = float(base_pA); self.gain_pA = float(gain_pA)
        self.read_window = int(read_window); self.ou_std = float(ou_std)
        self.exc_to_fs = float(exc_to_fs); self.fs_to_exc = float(fs_to_exc)
        self.seed = int(seed)
        self.n_host_rng_draws = 0                                       # MUST stay 0
        self._build_bank()

    def _build_bank(self):
        cfg = CoreSimConfig()
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.dt_ms = 1.0; cfg.seed = self.seed
        cfg.enable_brain_region_framework = True
        cfg.num_traits = 1
        for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
                  "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
                  "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_input_divisive_norm"):
            if hasattr(cfg, f):
                setattr(cfg, f, False)
        cfg.enable_ou_process = self.ou_std > 0.0
        cfg.ou_mean_current_pA = 0.0
        cfg.ou_std_current_pA = self.ou_std
        cfg.ou_tau_ms = 15.0
        cfg.ou_seed = self.seed
        regions = [BrainRegion(name=f"w{k}", n_neurons=self.P, exc_fraction=1.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)
                   for k in range(self.K)]
        regions.append(BrainRegion(name="fs", n_neurons=self.n_fs, exc_fraction=0.0, internal_density=0.0,
                                   exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
        pathways = []
        for k in range(self.K):
            pathways.append(RegionPathway(from_region=f"w{k}", to_region="fs", density=0.6,
                                          weight_mean=self.exc_to_fs, weight_jitter=0.1, plastic=False))
            pathways.append(RegionPathway(from_region="fs", to_region=f"w{k}", density=0.6,
                                          weight_mean=self.fs_to_exc, weight_jitter=0.1, plastic=False))
        cfg.brain_regions = regions; cfg.region_pathways = pathways
        bank = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                runtime_state=RuntimeState(), gpu_config=GPUConfig())
        bank.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        bank._initialize_simulation_data(called_from_playback_init=False)
        self._bank = bank
        rm = bank.region_manager
        self._ridx = [np.asarray(list(rm.indices(f"w{k}")), dtype=int) for k in range(self.K)]
        self._all_word = np.concatenate(self._ridx) if self.K else np.asarray([], dtype=int)
        self._v0 = (bank.cp_izh_c_reset.copy() if getattr(bank, "cp_izh_c_reset", None) is not None
                    else None)

    def drive_from_weights(self, w):
        w = np.asarray(w, dtype=np.float64)
        peak = float(w.max()) if w.size else 0.0
        if peak <= 1e-12:
            return np.zeros(len(w))
        active = (w > 0).astype(np.float64)
        return active * (self.base_pA + self.gain_pA * (w / peak))

    def _reset(self):
        bank = self._bank
        if self._v0 is not None:
            bank.cp_membrane_potential_v[:] = self._v0
        else:
            bank.cp_membrane_potential_v[:] = -65.0
        bank.cp_recovery_variable_u[:] = 0.0
        if getattr(bank, "cp_firing_states", None) is not None:
            bank.cp_firing_states[:] = False

    def _compete(self, drive_pools, equal_drive=False):
        bank = self._bank
        self._reset()
        xp, _ = get_backend()
        if equal_drive:
            active = (np.asarray(drive_pools) > 0).astype(np.float64)
            drive_pools = active * (self.base_pA + self.gain_pA)
        per_neuron = np.zeros(bank.core_config.num_neurons, dtype=np.float64)
        for k in range(self.K):
            per_neuron[self._ridx[k]] = float(drive_pools[k])
        bank.cp_external_input_current[:] = xp.asarray(per_neuron, dtype=bank.cp_external_input_current.dtype)
        firing = np.zeros(bank.core_config.num_neurons, dtype=np.float64)
        for _ in range(self.read_window):
            bank._run_one_simulation_step()
            firing += np.asarray(to_host(bank.cp_firing_states)).astype(float)
        bank.cp_external_input_current[:] = 0.0
        per_pool = np.array([firing[self._ridx[k]].sum() for k in range(self.K)])
        word_spikes = float(firing[self._all_word].sum())
        total_spikes = float(firing.sum())
        return per_pool, word_spikes, total_spikes

    def read(self, weights, equal_drive=False):
        drive = self.drive_from_weights(weights)
        per_pool, word_sp, tot = self._compete(drive, equal_drive=equal_drive)
        if per_pool.max() <= 0.0:
            return -1, per_pool, word_sp, tot
        return int(np.argmax(per_pool)), per_pool, word_sp, tot


# ================================================================================================================
# RUNG 2 — the SYNAPTIC read-out: head_w @ h realised as EXCITATORY SYNAPSES hidden->V pools (Dale-shift + a
#          feedforward common-mode CANCELLER), FS-WTA winner over ALL V pools. Retires the host logit matmul.
# ================================================================================================================
class SynapticLogitRead:
    """Route the FINAL logit projection `head_w @ h` (V x D) through READ-OUT NEURONS. The gated hidden state
    `h = r_h*(Wo_sp@state)` (host residual) is rate-coded by 2*D hidden neurons ([h+, h-] dual-nonneg, so a positive
    firing rate carries a signed feature). head_w over [h+,h-] = Wfull[k] = concat(head_w[k], -head_w[k]); realised as
    EXCITATORY synapses hidden->pool with a global Dale-shift `Wfull - gmin` (>=0, argmax-preserving) plus a
    feedforward common-mode CANCELLER (a shared inhibitory pool receiving uniform excitation from the hidden layer and
    delivering uniform inhibition to every pool -> subtracts the shift-induced common mode `gmin*sum(feature)`, so the
    pools compete on the DISCRIMINATIVE logit alone). A shared FS pool gives the one-of-V WTA. The winner emerges from
    V pools competing on SYNAPTIC drive -> NO host logit matmul, NO top-K argpartition on the read path."""

    def __init__(self, ro: WKVReadout, seed, pop=1, ou_std=120.0, read_window=30, hid_gain=42.0, hid_bias=8.0,
                 syn_scale=1.0, cm_gain=1.0, cm_out=4.0, n_fs=48, n_cm=16,
                 exc_to_fs=1.2, fs_to_exc=7.0, head_b_gain=1.0):
        self.ro = ro
        self.V = int(ro.V); self.D = int(ro.D)
        self.P = int(pop); self.n_fs = int(n_fs); self.n_cm = int(n_cm)
        self.ou_std = float(ou_std); self.read_window = int(read_window)
        self.hid_gain = float(hid_gain); self.hid_bias = float(hid_bias)
        self.syn_scale = float(syn_scale); self.cm_gain = float(cm_gain); self.cm_out = float(cm_out)
        self.exc_to_fs = float(exc_to_fs); self.fs_to_exc = float(fs_to_exc)
        self.head_b_gain = float(head_b_gain)
        self.seed = int(seed)
        self.n_host_rng_draws = 0
        # Wfull over [h+, h-]: signed read-out weights, then GLOBAL Dale-shift to >=0 (argmax-preserving).
        head_w = ro.head_w                                             # [V, D]
        self.Wfull = np.concatenate([head_w, -head_w], axis=1)         # [V, 2D]
        self.gmin = float(self.Wfull.min())
        self.Wshift = (self.Wfull - self.gmin)                         # [V, 2D] >= 0
        self.head_b = ro.head_b.astype(np.float64)                    # [V]
        self._build_bridge()
        self._wire()

    def _build_bridge(self):
        cfg = CoreSimConfig()
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.dt_ms = 1.0; cfg.seed = self.seed
        cfg.heterogeneity_seed = self.seed; cfg.ou_seed = self.seed
        cfg.enable_brain_region_framework = True
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1
        for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
                  "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
                  "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_input_divisive_norm",
                  "enable_nmda"):
            if hasattr(cfg, f):
                setattr(cfg, f, False)
        cfg.enable_ou_process = self.ou_std > 0.0
        cfg.ou_mean_current_pA = 0.0; cfg.ou_std_current_pA = self.ou_std; cfg.ou_tau_ms = 15.0
        cfg.stdp_w_max = 2000.0; cfg.hebbian_max_weight = 2000.0
        regions = [
            BrainRegion(name="hid", n_neurons=2 * self.D, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="wpool", n_neurons=self.V * self.P, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="fs", n_neurons=self.n_fs, exc_fraction=0.0, internal_density=0.0),
            BrainRegion(name="cm", n_neurons=self.n_cm, exc_fraction=0.0, internal_density=0.0),
        ]
        cfg.brain_regions = regions; cfg.region_pathways = []
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b._initialize_simulation_data(called_from_playback_init=False)
        self._b = b
        rm = b.region_manager
        self.hid_idx = np.asarray(list(rm.indices("hid")), dtype=np.int64)     # [2D]
        wpool_idx = np.asarray(list(rm.indices("wpool")), dtype=np.int64)
        self.pool_idx = [wpool_idx[k * self.P:(k + 1) * self.P] for k in range(self.V)]
        self.all_pool = wpool_idx
        self.fs_idx = np.asarray(list(rm.indices("fs")), dtype=np.int64)
        self.cm_idx = np.asarray(list(rm.indices("cm")), dtype=np.int64)
        self._v0 = (b.cp_izh_c_reset.copy() if getattr(b, "cp_izh_c_reset", None) is not None else None)

    def _wire(self):
        """Inject all synapses runner-side: head_w-shift read-out (hidden->pools), the common-mode canceller
        (hidden->cm uniform exc, cm->pools uniform inh), and the FS-WTA (pool->fs exc, fs->pool inh)."""
        b = self._b
        union = {}
        hid = self.hid_idx
        # ---- head_w-shift read-out synapses: hidden -> pools (dense; each pool's P neurons share the column) ----
        # weight(hidden d -> pool k) = syn_scale * Wshift[k, d]. Build flat COO.
        Ws = (self.Wshift * self.syn_scale).astype(np.float32)         # [V, 2D]
        n_hid = len(hid)
        pre = np.tile(hid, self.V * self.P)                            # for each pool-neuron, all hidden
        # post: pool neuron repeated n_hid times, in pool order
        post = np.repeat(self.all_pool, n_hid)
        # weight per edge: for pool k (all P neurons) row Ws[k] over the 2D hidden
        w_rows = np.repeat(Ws, self.P, axis=0)                         # [V*P, 2D]
        w = w_rows.reshape(-1).astype(np.float32)
        union["readout"] = {"pre_indices": pre.astype(np.int64), "post_indices": post.astype(np.int64),
                            "initial_weights": w, "plastic": False, "conn_type": "E_TO_E"}
        # ---- common-mode canceller: hidden -> cm (uniform exc), cm -> pools (uniform inh) ----
        if self.cm_gain > 0.0:
            w_in = float(abs(self.gmin) * self.syn_scale)             # cm tracks the common mode |gmin|*S*sum(feat)
            pin = np.repeat(hid, len(self.cm_idx)); qin = np.tile(self.cm_idx, len(hid))
            union["cm_in"] = {"pre_indices": pin.astype(np.int64), "post_indices": qin.astype(np.int64),
                              "initial_weights": np.full(len(pin), w_in, np.float32),
                              "plastic": False, "conn_type": "E_TO_E"}
            w_out = float(self.cm_out * self.cm_gain)
            pout = np.repeat(self.cm_idx, len(self.all_pool)); qout = np.tile(self.all_pool, len(self.cm_idx))
            union["cm_out"] = {"pre_indices": pout.astype(np.int64), "post_indices": qout.astype(np.int64),
                               "initial_weights": np.full(len(pout), w_out, np.float32),
                               "plastic": False, "conn_type": "I_TO_E"}
        # ---- FS-WTA: pool -> fs (exc), fs -> pool (inh) ----
        pef = np.repeat(self.all_pool, len(self.fs_idx)); qef = np.tile(self.fs_idx, len(self.all_pool))
        union["pool2fs"] = {"pre_indices": pef.astype(np.int64), "post_indices": qef.astype(np.int64),
                            "initial_weights": np.full(len(pef), self.exc_to_fs, np.float32),
                            "plastic": False, "conn_type": "E_TO_E"}
        pfe = np.repeat(self.fs_idx, len(self.all_pool)); qfe = np.tile(self.all_pool, len(self.fs_idx))
        union["fs2pool"] = {"pre_indices": pfe.astype(np.int64), "post_indices": qfe.astype(np.int64),
                            "initial_weights": np.full(len(pfe), self.fs_to_exc, np.float32),
                            "plastic": False, "conn_type": "I_TO_E"}
        inh = np.concatenate([self.fs_idx, self.cm_idx]).tolist()
        b.inject_explicit_wiring(union, output_inhibitory_indices=inh)
        self._readout_edges = (union["readout"]["pre_indices"], union["readout"]["post_indices"],
                               union["readout"]["initial_weights"].copy())

    def _hidden_feature(self, ap, an, tid):
        """h = r_h * (Wo_sp @ [ap,an]) (HOST residual — the graded validated conductance projection). Return [h+, h-]
        (the dual-nonneg rate-code drive to the hidden neurons)."""
        ro = self.ro
        state = np.concatenate([ap, an])
        r_h = 1.0 / (1.0 + np.exp(-(ro.Wr @ ro._ln(ro.emb[tid]))))
        h = r_h * (ro.Wo_sp @ state)                                  # [D]
        return np.concatenate([np.maximum(h, 0.0), np.maximum(-h, 0.0)])   # [2D] >= 0

    def _reset(self):
        b = self._b
        if self._v0 is not None:
            b.cp_membrane_potential_v[:] = self._v0
        else:
            b.cp_membrane_potential_v[:] = -65.0
        b.cp_recovery_variable_u[:] = 0.0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False

    def read(self, ap, an, tid, hidden_silence=False, scramble_perm=None):
        """ONE synaptic read: drive the hidden neurons by [h+,h-]; their spikes propagate through the head_w-shift
        synapses to the V pools; the canceller subtracts the common mode; the FS-WTA resolves the winner. Returns
        (winner_pool_idx in 0..V-1 or -1, per_pool_firing, word_spikes, total_spikes)."""
        b = self._b
        xp, _ = get_backend()
        self._reset()
        feat = self._hidden_feature(ap, an, tid)
        if hidden_silence:
            feat = np.zeros_like(feat)
        drive = np.zeros(b.core_config.num_neurons, dtype=np.float64)
        drive[self.hid_idx] = self.hid_bias + self.hid_gain * feat
        # per-pool intrinsic bias from head_b (shifted nonneg -> a per-pool tonic floor; argmax-preserving constant)
        hb = (self.head_b - self.head_b.min()) * self.head_b_gain
        for k in range(self.V):
            drive[self.pool_idx[k]] += hb[k]
        b.cp_external_input_current[:] = xp.asarray(drive, dtype=b.cp_external_input_current.dtype)
        firing = np.zeros(b.core_config.num_neurons, dtype=np.float64)
        for _ in range(self.read_window):
            b._run_one_simulation_step()
            firing += np.asarray(to_host(b.cp_firing_states)).astype(float)
        b.cp_external_input_current[:] = 0.0
        per_pool = np.array([firing[self.pool_idx[k]].sum() for k in range(self.V)])
        word_sp = float(firing[self.all_pool].sum()); tot = float(firing.sum())
        if scramble_perm is not None:
            per_pool = per_pool[scramble_perm]
        if per_pool.max() <= 0.0:
            return -1, per_pool, word_sp, tot
        return int(np.argmax(per_pool)), per_pool, word_sp, tot

    def lesion_readout(self):
        """Zero the head_w-shift read-out synapses -> pools see only the canceller + floor -> collapse (no logit)."""
        pre, post, _ = self._readout_edges
        self._b.set_pathway_weights("lesion_readout", pre, post, np.zeros(len(pre), np.float32), add_missing=False)

    def restore_readout(self):
        pre, post, w = self._readout_edges
        self._b.set_pathway_weights("restore_readout", pre, post, w, add_missing=False)

    def read_oracle(self, logit_vec, oracle_gain=6.0, oracle_base=30.0):
        """DIAGNOSTIC ONLY (uses host logits — NOT a deliverable read path). Bypass the synaptic projection: drive
        pool_k DIRECTLY with a current proportional to the (rank-normalized) host logit, through the SAME FS-WTA over
        V pools. Isolates the WTA-resolution-over-1000-near-tied-pools ceiling from the synaptic-projection fidelity —
        i.e. locates WHERE the rung-2 loss lives (the FS-WTA cannot resolve V=1000 near-ties, vs the Dale common-mode
        distorts the drive)."""
        b = self._b
        xp, _ = get_backend()
        self._reset()
        lg = np.asarray(logit_vec, dtype=np.float64)
        # map to a contrastive nonneg per-pool current (same low-floor/high-gain regime as rung1's FS-WTA drive)
        p = _softmax(lg)
        peak = float(p.max()) if p.size else 0.0
        w = (p / peak) if peak > 1e-12 else np.zeros_like(p)
        per_pool = oracle_base + oracle_gain * w
        drive = np.zeros(b.core_config.num_neurons, dtype=np.float64)
        for k in range(self.V):
            drive[self.pool_idx[k]] = per_pool[k]
        b.cp_external_input_current[:] = xp.asarray(drive, dtype=b.cp_external_input_current.dtype)
        firing = np.zeros(b.core_config.num_neurons, dtype=np.float64)
        for _ in range(self.read_window):
            b._run_one_simulation_step()
            firing += np.asarray(to_host(b.cp_firing_states)).astype(float)
        b.cp_external_input_current[:] = 0.0
        pp = np.array([firing[self.pool_idx[k]].sum() for k in range(self.V)])
        if pp.max() <= 0.0:
            return -1
        return int(np.argmax(pp))


# ================================================================================================================
def _eval_rung1(seed, ro, ev_ids, vocab, warmup, topk, read_window, pop, base_pA, gain_pA, ou_std, sample_temp,
                n_eval_pos, fs_to_exc, exc_to_fs, n_fs, gen_tokens, gen_temp):
    reader = FSWTAWordRead(topk, pop, seed, ou_std=ou_std, base_pA=base_pA, gain_pA=gain_pA,
                           read_window=read_window, n_fs=n_fs, exc_to_fs=exc_to_fs, fs_to_exc=fs_to_exc)
    grng = np.random.default_rng(seed * 101 + 7)
    acc = dict(n=0, word_spikes=0.0, total_spikes=0.0, topk_cover=0.0, argmax_agree=0.0, top5_hit=0.0,
               nll=0.0, mass_fs=0.0, mass_hs=0.0, mass_ax=0.0, mass_scr=0.0, agree_scr=0.0, mass_eq=0.0, silent=0)
    positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(len(ids) - 1):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t])
            if ro.unk_idx >= 0:
                lg = lg.copy(); lg[ro.unk_idx] = -1e30
            cand = np.argpartition(-lg, topk - 1)[:topk]; cand = cand[np.argsort(-lg[cand])]
            p = _softmax(lg[cand] / sample_temp)
            host_argmax = int(cand[0]); top5 = set(int(c) for c in cand[:5])
            win, per_pool, word_sp, tot = reader.read(p)
            fewspike = int(cand[win]) if win >= 0 else -1
            hs = int(cand[int(grng.choice(len(p), p=p))])
            perm_t = np.random.default_rng(seed * 71 + 5 + positions).permutation(len(cand))
            fewspike_s = int(cand[perm_t[win]]) if win >= 0 else -1
            win_e, _, _, _ = reader.read(p, equal_drive=True)
            fewspike_e = int(cand[win_e]) if win_e >= 0 else -1
            pfull = _softmax(lg)
            acc["n"] += 1; positions += 1
            acc["word_spikes"] += word_sp; acc["total_spikes"] += tot
            acc["topk_cover"] += float(pfull[cand].sum())
            if win < 0:
                acc["silent"] += 1
            acc["argmax_agree"] += float(fewspike == host_argmax)
            acc["top5_hit"] += float(fewspike in top5)
            acc["nll"] += -math.log(max(pfull[fewspike] if fewspike >= 0 else 1e-12, 1e-12))
            acc["mass_fs"] += (pfull[fewspike] if fewspike >= 0 else 0.0)
            acc["mass_hs"] += pfull[hs]; acc["mass_ax"] += pfull[host_argmax]
            acc["mass_scr"] += (pfull[fewspike_s] if fewspike_s >= 0 else 0.0)
            acc["agree_scr"] += float(fewspike_s == host_argmax)
            acc["mass_eq"] += (pfull[fewspike_e] if fewspike_e >= 0 else 0.0)
            if positions >= n_eval_pos:
                break
        if positions >= n_eval_pos:
            break
    n = max(1, acc["n"])
    # noise-ablation
    reader_ab = FSWTAWordRead(topk, pop, seed, ou_std=0.0, base_pA=base_pA, gain_pA=gain_pA,
                              read_window=read_window, n_fs=n_fs, exc_to_fs=exc_to_fs, fs_to_exc=fs_to_exc)
    det_w = np.zeros(topk); det_w[[0, 1, 2]] = [1.0, 0.6, 0.3]
    w0, _, _, _ = reader_ab.read(det_w); w1, _, _, _ = reader_ab.read(det_w)
    det_stable = (w0 == w1 == 0)
    m = _pack_metrics(seed, "rung1_fswta", read_window, pop, topk, acc, n, det_stable, reader.n_host_rng_draws)
    m["mean_spikes_per_read"] = round(acc["word_spikes"] / n, 2)
    m["mean_spikes_total"] = round(acc["total_spikes"] / n, 2)
    m["fs_to_exc"] = fs_to_exc; m["n_fs"] = n_fs
    # free-gen
    if gen_tokens > 0:
        m["generation"] = _free_gen_rung1(ro, vocab, reader, grng, topk, gen_temp, gen_tokens)
    return m


def _eval_rung2(seed, ro, ev_ids, vocab, warmup, topk, sample_temp, n_eval_pos, s2, gen_tokens, gen_temp):
    grng = np.random.default_rng(seed * 131 + 9)
    acc = dict(n=0, word_spikes=0.0, total_spikes=0.0, argmax_agree=0.0, top5_hit=0.0, nll=0.0,
               mass_fs=0.0, mass_hs=0.0, mass_ax=0.0, mass_scr=0.0, agree_scr=0.0, mass_les=0.0, silent=0,
               hid_active=0.0, mass_oracle=0.0)
    positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(len(ids) - 1):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t])
            lg_supp = lg.copy()
            if ro.unk_idx >= 0:
                lg_supp[ro.unk_idx] = -1e30
            host_argmax = int(np.argmax(lg_supp))
            cand5 = np.argpartition(-lg_supp, 4)[:5]; top5 = set(int(c) for c in cand5)
            pfull = _softmax(lg_supp)
            # host-sample ceiling (top-K softmax, matched to rung1's calibration)
            candk = np.argpartition(-lg_supp, topk - 1)[:topk]; candk = candk[np.argsort(-lg_supp[candk])]
            pk = _softmax(lg_supp[candk] / sample_temp)
            hs = int(candk[int(grng.choice(len(pk), p=pk))])
            # SYNAPTIC read (full-V, no top-K on the read path)
            win, per_pool, word_sp, tot = s2.read(ap, an, ids[t])
            scr_perm = np.random.default_rng(seed * 83 + 3 + positions).permutation(s2.V)
            win_s, _, _, _ = s2.read(ap, an, ids[t], scramble_perm=scr_perm)
            # ORACLE ceiling (DIAGNOSTIC: perfect host-logit current through the SAME FS-WTA over V pools; isolates
            # the full-V WTA-resolution ceiling from the synaptic-projection loss). Uses host logits -> NOT a read path.
            ora = s2.read_oracle(lg_supp, oracle_gain=220.0, oracle_base=30.0)
            acc["mass_oracle"] += (pfull[ora] if ora >= 0 else 0.0)
            acc["n"] += 1; positions += 1
            acc["word_spikes"] += word_sp; acc["total_spikes"] += tot
            acc["hid_active"] += float(per_pool.sum() > 0)
            if win < 0:
                acc["silent"] += 1
            acc["argmax_agree"] += float(win == host_argmax)
            acc["top5_hit"] += float(win in top5)
            acc["nll"] += -math.log(max(pfull[win] if win >= 0 else 1e-12, 1e-12))
            acc["mass_fs"] += (pfull[win] if win >= 0 else 0.0)
            acc["mass_hs"] += pfull[hs]; acc["mass_ax"] += pfull[host_argmax]
            acc["mass_scr"] += (pfull[win_s] if win_s >= 0 else 0.0)
            acc["agree_scr"] += float(win_s == host_argmax)
            if positions >= n_eval_pos:
                break
        if positions >= n_eval_pos:
            break
    n = max(1, acc["n"])
    # lesion the read-out synapses -> collapse
    s2.lesion_readout()
    les_mass = 0.0; les_n = 0
    for ids in ev_ids[:2]:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(min(len(ids) - 1, warmup + 20)):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t]); lg_supp = lg.copy()
            if ro.unk_idx >= 0:
                lg_supp[ro.unk_idx] = -1e30
            pfull = _softmax(lg_supp)
            win, _, _, _ = s2.read(ap, an, ids[t])
            les_mass += (pfull[win] if win >= 0 else 0.0); les_n += 1
            if les_n >= 40:
                break
        if les_n >= 40:
            break
    s2.restore_readout()
    # ATTRIBUTION (tools.lab): whose is the (weak) synaptic read? zeroing the head_w read-out synapses must collapse it
    # -> the read-out synapses OWN the drive (required=False: at a boundary the intact read is near chance BY the
    # measured negative, so the move is small by construction, but the attribution is recorded, not assumed).
    lever("rung2_readout_synapse_lesion", before=round(acc["mass_fs"] / n, 4),
          after=round(les_mass / max(1, les_n), 4), required=False)
    m = {
        "seed": seed, "arm": "rung2_synaptic", "V": s2.V, "pop": s2.P, "topk_ceiling": topk,
        "plasticity_off": True,   # fixed read-out, NO learning anywhere in this runner (STDP/Hebbian/homeostasis off)
        "n_positions": acc["n"], "silent_frac": round(acc["silent"] / n, 4),
        "hidden_active_frac": round(acc["hid_active"] / n, 4),
        "mean_spikes_per_read": round(acc["word_spikes"] / n, 2),
        "mean_spikes_total": round(acc["total_spikes"] / n, 2),
        "argmax_agree": round(acc["argmax_agree"] / n, 4),
        "top5_hit": round(acc["top5_hit"] / n, 4),
        "nll_synaptic": round(acc["nll"] / n, 4),
        "mass_synaptic": round(acc["mass_fs"] / n, 4),
        "mass_hostsample_ceiling": round(acc["mass_hs"] / n, 4),
        "mass_argmax_ceiling": round(acc["mass_ax"] / n, 4),
        "mass_scramble": round(acc["mass_scr"] / n, 4),
        "argmax_agree_scramble": round(acc["agree_scr"] / n, 4),
        "mass_readout_lesion": round(les_mass / max(1, les_n), 4),
        "mass_oracle_ceiling": round(acc["mass_oracle"] / n, 4),
        "chance_1_over_v": round(1.0 / s2.V, 6),
        "host_rng_draws_on_read_path": int(s2.n_host_rng_draws),
    }
    m["read_fidelity_vs_sampler"] = round(m["mass_synaptic"] / max(1e-9, m["mass_hostsample_ceiling"]), 4)
    m["oracle_read_fidelity"] = round(m["mass_oracle_ceiling"] / max(1e-9, m["mass_hostsample_ceiling"]), 4)
    if gen_tokens > 0:
        m["generation"] = _free_gen_rung2(ro, vocab, s2, grng, topk, gen_temp, gen_tokens)
    return m


def _pack_metrics(seed, arm, read_window, pop, topk, acc, n, det_stable, host_draws):
    m = {
        "seed": seed, "arm": arm, "read_window": read_window, "pop": pop, "topk": topk,
        "n_positions": acc["n"], "silent_frac": round(acc["silent"] / n, 4),
        "topk_coverage": round(acc["topk_cover"] / n, 4),
        "argmax_agree": round(acc["argmax_agree"] / n, 4),
        "top5_hit": round(acc["top5_hit"] / n, 4),
        "nll_fewspike": round(acc["nll"] / n, 4),
        "mass_fewspike": round(acc["mass_fs"] / n, 4),
        "mass_hostsample_ceiling": round(acc["mass_hs"] / n, 4),
        "mass_argmax_ceiling": round(acc["mass_ax"] / n, 4),
        "mass_scramble": round(acc["mass_scr"] / n, 4),
        "argmax_agree_scramble": round(acc["agree_scr"] / n, 4),
        "mass_equal_drive": round(acc["mass_eq"] / n, 4),
        "chance_1_over_k": round(1.0 / topk, 4),
        "noise_ablation_deterministic": bool(det_stable),
        "host_rng_draws_on_read_path": int(host_draws),
    }
    m["read_fidelity_vs_sampler"] = round(m["mass_fewspike"] / max(1e-9, m["mass_hostsample_ceiling"]), 4)
    return m


def _free_gen_rung1(ro, vocab, reader, grng, topk, gen_temp, n_tok):
    def read_fn(cand, p, lg):
        w, _, _, _ = reader.read(p)
        return int(cand[w]) if w >= 0 else int(cand[0])
    return _run_gen(ro, vocab, read_fn, grng, topk, gen_temp, n_tok)


def _free_gen_rung2(ro, vocab, s2, grng, topk, gen_temp, n_tok):
    def read_fn_ctx(ap, an, cur, lg):
        w, _, _, _ = s2.read(ap, an, cur)
        return int(w) if w >= 0 else int(np.argmax(lg))
    return _run_gen(ro, vocab, None, grng, topk, gen_temp, n_tok, ctx_read=read_fn_ctx)


def _run_gen(ro, vocab, read_fn, grng, topk, gen_temp, n_tok, ctx_read=None):
    out = {}
    for prompt in ("once upon a time", "the little girl", "tom and his dog"):
        pid = [i for i in vocab.ids(prompt.split()) if 0 <= i < ro.V] or [0]
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in pid:
            ap, an = ro.advance(ap, an, t)
        gen = list(pid); self_nll = 0.0; steps = 0
        for _ in range(n_tok):
            lg = ro.logits(ap, an, gen[-1]); lg2 = lg.copy()
            if ro.unk_idx >= 0:
                lg2[ro.unk_idx] = -1e30
            if ctx_read is not None:
                nxt = ctx_read(ap, an, gen[-1], lg2)
            else:
                cand = np.argpartition(-lg2, topk - 1)[:topk]; cand = cand[np.argsort(-lg2[cand])]
                p = _softmax(lg2[cand] / gen_temp)
                nxt = read_fn(cand, p, lg2)
            self_nll += -math.log(max(_softmax(lg2)[nxt], 1e-12)); steps += 1
            gen.append(nxt); ap, an = ro.advance(ap, an, nxt)
        txt = " ".join(ro.words[i] if 0 <= i < len(ro.words) else "<unk>" for i in gen)
        out[prompt] = {"text": txt, "self_nll": round(self_nll / max(1, steps), 3)}
    return out


def _scramble_at_chance(agree_scramble, chance, n):
    """ROBUST scramble control (the parent caught a fragile `< 2*chance` razor: seed 43 missed on Poisson noise).
    The scramble is 'at chance' iff its agreement is NOT SIGNIFICANTLY ABOVE chance — a binomial upper bound at
    ~3 sigma for the actual n, so a 1-2 extra chance hit does not flip the verdict."""
    sigma = math.sqrt(max(chance * (1.0 - chance), 1e-12) / max(1, n))
    return agree_scramble <= chance + 3.0 * sigma


def _verdict_rung1(m):
    chance = m["chance_1_over_k"]; n = m["n_positions"]
    checks = {
        "read_fidelity_ge_0.90": m["read_fidelity_vs_sampler"] >= 0.90,
        "argmax_agree_gt_2x_chance": m["argmax_agree"] > 2 * chance,
        "scramble_at_chance": _scramble_at_chance(m["argmax_agree_scramble"], chance, n),
        "equal_drive_below_fewspike": m["mass_equal_drive"] < 0.9 * m["mass_fewspike"],
        "provenance_no_host_draw": m["host_rng_draws_on_read_path"] == 0,
        "noise_ablation_deterministic": m["noise_ablation_deterministic"],
        "not_silent": m["silent_frac"] < 0.05,
    }
    return all(checks.values()), checks


def _verdict_rung2(m):
    chance = m["chance_1_over_v"]; n = m["n_positions"]
    checks = {
        "read_fidelity_ge_0.90": m["read_fidelity_vs_sampler"] >= 0.90,
        "argmax_agree_gt_10x_chance": m["argmax_agree"] > 10 * chance,
        "scramble_at_chance": _scramble_at_chance(m["argmax_agree_scramble"], chance, n),
        "readout_lesion_collapses": m["mass_readout_lesion"] < 0.5 * m["mass_synaptic"],
        "provenance_no_host_draw": m["host_rng_draws_on_read_path"] == 0,
        "hidden_active": m["hidden_active_frac"] > 0.9,
        "not_silent": m["silent_frac"] < 0.05,
    }
    return all(checks.values()), checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=8000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--rungs", type=str, default="1,2")
    ap.add_argument("--n-eval-pos", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--read-window", type=int, default=20)
    ap.add_argument("--pops", type=str, default="2,4")                 # rung1: population sizes BELOW P=8
    # rung1 operating point (smoke-selected): a CONTRASTIVE drive (low floor, high gain) so the shared FS inhibition
    # can SILENCE the weak candidates -> the winner survives at a small P + few spikes (~19 word spikes vs P=8's ~56).
    ap.add_argument("--base-pA", type=float, default=30.0)
    ap.add_argument("--gain-pA", type=float, default=220.0)
    ap.add_argument("--ou-std", type=float, default=150.0)
    ap.add_argument("--fs-to-exc", type=float, default=12.0)
    ap.add_argument("--exc-to-fs", type=float, default=2.0)
    ap.add_argument("--n-fs", type=int, default=24)
    ap.add_argument("--sample-temp", type=float, default=0.8)
    ap.add_argument("--gen-tokens", type=int, default=0)
    ap.add_argument("--gen-temp", type=float, default=0.8)
    # rung2
    ap.add_argument("--r2-pop", type=int, default=1)
    ap.add_argument("--r2-read-window", type=int, default=30)
    ap.add_argument("--r2-ou-std", type=float, default=120.0)
    ap.add_argument("--r2-hid-gain", type=float, default=42.0)
    ap.add_argument("--r2-hid-bias", type=float, default=8.0)
    ap.add_argument("--r2-syn-scale", type=float, default=1.0)
    ap.add_argument("--r2-cm-gain", type=float, default=1.0)
    ap.add_argument("--r2-fs-to-exc", type=float, default=7.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_fswta_synaptic.json")
    args = ap.parse_args()

    rungs = set(r.strip() for r in args.rungs.split(",") if r.strip())
    if args.smoke:
        args.n_eval_pos = min(args.n_eval_pos, 80)
        args.gen_tokens = args.gen_tokens or 40
        args.pops = args.pops or "2,4"

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    pops = [int(x) for x in args.pops.split(",") if x.strip()]

    t0 = time.time()
    results = []
    for seed in seeds:
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        if not Path(ckpt).exists():
            print(f"[skip] seed {seed}: checkpoint {ckpt} missing", flush=True)
            continue
        ro = WKVReadout(ckpt)
        ev_ids, vocab = _load_eval(ro, args.corpus, args.n_sentences, seed, max(64, args.n_eval_pos // 6))

        if "1" in rungs:
            for pop in pops:
                gen_here = args.gen_tokens if pop == max(pops) else 0
                m = _eval_rung1(seed, ro, ev_ids, vocab, args.warmup, args.topk, args.read_window, pop,
                                args.base_pA, args.gain_pA, args.ou_std, args.sample_temp, args.n_eval_pos,
                                args.fs_to_exc, args.exc_to_fs, args.n_fs, gen_here, args.gen_temp)
                go, checks = _verdict_rung1(m); m["go"] = go; m["checks"] = checks
                results.append(m)
                print(f"[R1 seed {seed} P={pop}] word_spk={m['mean_spikes_per_read']} tot={m['mean_spikes_total']} "
                      f"read_fid={m['read_fidelity_vs_sampler']} argmax_agree={m['argmax_agree']} "
                      f"scr={m['argmax_agree_scramble']} eq={m['mass_equal_drive']} fs={m['mass_fewspike']} "
                      f"GO={go} ({sum(checks.values())}/{len(checks)})", flush=True)
                if m.get("generation"):
                    for pr, g in m["generation"].items():
                        print(f"    [R1 gen '{pr}' nll {g['self_nll']}] {g['text'][:160]}", flush=True)

        if "2" in rungs:
            s2 = SynapticLogitRead(ro, seed, pop=args.r2_pop, ou_std=args.r2_ou_std,
                                   read_window=args.r2_read_window, hid_gain=args.r2_hid_gain,
                                   hid_bias=args.r2_hid_bias, syn_scale=args.r2_syn_scale,
                                   cm_gain=args.r2_cm_gain, fs_to_exc=args.r2_fs_to_exc)
            m2 = _eval_rung2(seed, ro, ev_ids, vocab, args.warmup, args.topk, args.sample_temp,
                             args.n_eval_pos, s2, args.gen_tokens, args.gen_temp)
            go2, checks2 = _verdict_rung2(m2); m2["go"] = go2; m2["checks"] = checks2
            results.append(m2)
            print(f"[R2 seed {seed}] word_spk={m2['mean_spikes_per_read']} read_fid={m2['read_fidelity_vs_sampler']} "
                  f"ORACLE_fid={m2['oracle_read_fidelity']} argmax_agree={m2['argmax_agree']} "
                  f"(chance {m2['chance_1_over_v']}) top5={m2['top5_hit']} scr={m2['argmax_agree_scramble']} "
                  f"lesion_mass={m2['mass_readout_lesion']} synaptic_mass={m2['mass_synaptic']} "
                  f"GO={go2} ({sum(checks2.values())}/{len(checks2)})", flush=True)
            if m2.get("generation"):
                for pr, g in m2["generation"].items():
                    print(f"    [R2 gen '{pr}' nll {g['self_nll']}] {g['text'][:160]}", flush=True)

    # aggregate per arm/operating-point
    agg = {}
    for m in results:
        key = m["arm"] + (f"_P{m['pop']}" if "pop" in m else "")
        agg.setdefault(key, {"read_fidelity": [], "argmax_agree": [], "mean_spikes": [], "go": []})
        agg[key]["read_fidelity"].append(m["read_fidelity_vs_sampler"])
        agg[key]["argmax_agree"].append(m["argmax_agree"])
        agg[key]["mean_spikes"].append(m["mean_spikes_per_read"])
        agg[key]["go"].append(m["go"])
    summary = {}
    for key, d in agg.items():
        summary[key] = {"n_seeds": len(d["go"]), "go_count": int(sum(d["go"])),
                        "read_fidelity_mean": round(float(np.mean(d["read_fidelity"])), 4),
                        "read_fidelity_min": round(float(np.min(d["read_fidelity"])), 4),
                        "argmax_agree_mean": round(float(np.mean(d["argmax_agree"])), 4),
                        "mean_spikes_per_read": round(float(np.mean(d["mean_spikes"])), 2)}
    out = {"results": results, "summary": summary, "seeds": seeds, "rungs": sorted(rungs),
           "pops": pops, "topk": args.topk, "read_window": args.read_window, "sample_temp": args.sample_temp,
           "plasticity_off": True,   # fixed weights everywhere (a READ experiment; no STDP/Hebbian/homeostasis)
           "elapsed_s": round(time.time() - t0, 1), "backend": os.environ.get("SIM_BACKEND", "numpy")}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(_native(out), indent=2))
    print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(results)} rows, {time.time()-t0:.0f}s -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
