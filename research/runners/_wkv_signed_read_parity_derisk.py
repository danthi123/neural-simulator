"""
gap#1 / A1 — the PARITY rung on the TRUE SIGNED read-out of the fluent WKV open-prose mouth. Adds the two companion
processes the signed-read BOUNDARY-LIFTED finding (2026-08-13) named as its next rung, both fully SYNAPTIC:

  (1) a NEURAL DIVISIVE-NORMALISATION HOMEOSTATIC pool `dn` — a shared inhibitory sum-pool receiving from ALL word
      pools and divisively inhibiting each (the Louie-Glimcher two-stage `R_i <- V_i / (B + Σ_j R_j)` normalize-then-WTA;
      Carandini-Heeger 2012). It TRACKS the total read-pool drive and ADAPTS the read set-point PER POSITION, so the
      winner tips over rheobase WITHOUT a fixed high floor. This is the companion process the fixed floor replaced with
      a constant: it erases the ~19% silence AND disentangles the signed-vs-positive-only confound, because the floor no
      longer trades silence against sign (dn regulates the "too-many-fire" side while a floor lifts the low-drive side).
  (2) STRONGER RECURRENT WTA AMPLIFICATION — recurrent EXCITATION WITHIN each word pool (Rutishauser-Douglas-Slotine
      soft/hard-WTA gain; Wong-Wang attractor) so the winning pool amplifies its own activity: the exponential
      sharpening the LINEAR signed read lacks, closing projection_recovery toward the perfect-current ORACLE (1.30).

THE BOUNDARY THIS ATTACKS (2026-08-13 `_wkv_signed_shadow_read_derisk`, BOUNDARY-LIFTED, 6-seed):
  the TRUE SIGNED read-out (Wp on an excitatory hid, Wn on an INHIBITORY SHADOW hidinh, no Dale-shift, no common mode)
  lifted read_fidelity 0.035 -> 0.55 (~16x) — the substrate speaks semi-coherent prose on the read path — BUT it is a
  LIFT, not a parity GO: projection_recovery 0.43 (oracle 1.30 — the LINEAR read lacks exponential sharpening), ~19%
  of positions SILENT at the FIXED read floor, and the negative-weights-load-bearing claim SEED-FRAGILE (3/6),
  confounded by the silence trade-off (pushing the floor up to erase silence tips into the DECORATIVE positive-only
  regime; the 2026-07-04 conductance-signed lesson). All three residuals share ONE root cause: a FIXED operating point
  where the animal runs an ADAPTIVE homeostatic + competitive one.

THE QUESTION: does read_fidelity approach PARITY (proj_recovery -> ~1.0+; read_fid materially above 0.55) with 0%
  silent AND the signed-vs-positive-only confound RESOLVED (negative weights load-bearing 6/6, not 3/6, because silence
  is no longer traded against sign)?

BRAIN-BASED: the set-point + sharpening are SPIKING (a divisive-norm inhibitory pool + recurrent excitation on
  cp_firing_states), NOT a host softmax/normalisation. Lesion the homeostatic pool `dn` -> silence/degradation returns;
  lesion the recurrent excitation -> sharpening (proj_recovery) drops; lesion the read-out -> collapse; lesion the
  inhibitory shadow (Wn) -> a positive-only read (sign load-bearing test). NO host argmax over logits; 0 host draws on
  the read path (winner from cp_firing_states). Scramble -> chance; shadow rate-match.

Reuse-by-import: the ENTIRE signed-read machinery — SignedShadowLogitRead (bridge build, signed Wp/Wn wiring, FS-WTA,
  hidden feature, oracle, metrics), _base_eval / _free_gen / _scramble_at_chance — from `_wkv_signed_shadow_read_derisk`;
  WKVReadout / _softmax / _native / _load_eval from `_wkv_fewspike_read_derisk`. This runner ADDS only the `dn` region,
  the recurrent within-pool edges, and their lesions + the parity verdict. NO `sim/` edit; cfg.seed-controlled
  substrate (CLAUDE.md seed trap). Runner-only, default-off.

Run (smoke):  SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_signed_read_parity_derisk \
                --smoke --seeds 42
Run (6-seed): SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_signed_read_parity_derisk \
                --seeds 42,43,44,100,101,102 --pops 8 \
                --json research/findings/raw/_wkv_signed_read_parity_6seed.json
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

from sim.backend import to_host, get_backend  # noqa: E402

from research.runners._wkv_fewspike_read_derisk import (  # noqa: E402
    WKVReadout, _softmax, _native, _load_eval,
)
from research.runners._wkv_signed_shadow_read_derisk import (  # noqa: E402
    SignedShadowLogitRead, _eval as _base_eval, _free_gen, _scramble_at_chance,
)
from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.bridge import SimulationBridge  # noqa: E402
from sim.regions import BrainRegion  # noqa: E402
from tools.lab import lever  # noqa: E402


class ParitySignedShadowRead(SignedShadowLogitRead):
    """The signed read-out + a divisive-normalisation homeostatic pool (dn) + recurrent within-pool WTA excitation.

    dn: all word-pools -> dn (excitatory sum), dn -> all word-pools (inhibitory, divisive at the operating point). The
        per-draw denominator `Σ_j R_j` recomputed each position (Louie-Glimcher). Because dn's inhibition SCALES with
        total pool activity, a MODERATELY-raised floor lifts the low-drive winner over rheobase (erasing silence) while
        dn suppresses the over-driven positions (holding the mean operating point near threshold -> Wn stays
        SUBTRACTIVE -> the sign stays load-bearing). dn_inh=0 + rec_gain=0 reproduces the parent boundary.
    recurrent: pool_k[i] -> pool_k[j] (i != j) excitatory, weight rec_gain -> the winning pool amplifies itself
        (Rutishauser soft/hard-WTA gain) -> exponential sharpening the linear read lacks.
    """

    def __init__(self, ro, seed, *, dn_size=32, dn_exc=0.6, dn_inh=2.0, rec_gain=0.0, **kw):
        # set the extra knobs BEFORE super().__init__ (super calls the overridden _build_bridge/_wire)
        self.dn_size = int(dn_size)
        self.dn_exc = float(dn_exc)
        self.dn_inh = float(dn_inh)
        self.rec_gain = float(rec_gain)
        self.oracle_gain = float(kw.pop("oracle_gain", 220.0))
        self.oracle_base = float(kw.pop("oracle_base", 30.0))
        super().__init__(ro, seed, **kw)

    # ---- override the bridge build to ADD the dn region (regions must be declared before init) ----
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
        cfg.stdp_w_max = 4000.0; cfg.hebbian_max_weight = 4000.0
        Hn = self.F * self.Hp
        regions = [
            BrainRegion(name="hid", n_neurons=Hn, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="hidinh", n_neurons=Hn, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="wpool", n_neurons=self.V * self.P, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="fs", n_neurons=self.n_fs, exc_fraction=0.0, internal_density=0.0),
            BrainRegion(name="dn", n_neurons=self.dn_size, exc_fraction=0.0, internal_density=0.0),
        ]
        cfg.brain_regions = regions; cfg.region_pathways = []
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b._initialize_simulation_data(called_from_playback_init=False)
        self._b = b
        rm = b.region_manager
        self.hid_idx = np.asarray(list(rm.indices("hid")), dtype=np.int64)
        self.hidinh_idx = np.asarray(list(rm.indices("hidinh")), dtype=np.int64)
        self.hid_dim = np.repeat(np.arange(self.F), self.Hp).astype(np.int64)
        wpool_idx = np.asarray(list(rm.indices("wpool")), dtype=np.int64)
        self.pool_idx = [wpool_idx[k * self.P:(k + 1) * self.P] for k in range(self.V)]
        self.all_pool = wpool_idx
        self.fs_idx = np.asarray(list(rm.indices("fs")), dtype=np.int64)
        self.dn_idx = np.asarray(list(rm.indices("dn")), dtype=np.int64)
        self._v0 = (b.cp_izh_c_reset.copy() if getattr(b, "cp_izh_c_reset", None) is not None else None)
        if self.uniform_thresh and getattr(b, "cp_neuron_firing_thresholds", None) is not None:
            thr = b.cp_neuron_firing_thresholds
            thr[:] = float(to_host(thr).mean())

    # ---- override the wiring to ADD dn (divisive norm) + recurrent within-pool excitation ----
    def _wire(self):
        b = self._b
        union = {}
        Wp = (self.Wp * self.syn_scale).astype(np.float32)
        Wn = (self.Wn * self.syn_scale * self.ratio).astype(np.float32)
        Wp_hn = Wp[:, self.hid_dim]
        Wn_hn = Wn[:, self.hid_dim]
        nH = len(self.hid_idx)
        # ---- signed read-out (parent) : Wp EXC hid->pools, Wn INH hidinh->pools ----
        pre = np.tile(self.hid_idx, self.V * self.P)
        post = np.repeat(self.all_pool, nH)
        wp = np.repeat(Wp_hn, self.P, axis=0).reshape(-1).astype(np.float32)
        union["readout_pos"] = {"pre_indices": pre, "post_indices": post, "initial_weights": wp,
                                "plastic": False, "conn_type": "E_TO_E"}
        pre_n = np.tile(self.hidinh_idx, self.V * self.P)
        wn = np.repeat(Wn_hn, self.P, axis=0).reshape(-1).astype(np.float32)
        union["readout_neg"] = {"pre_indices": pre_n, "post_indices": post.copy(), "initial_weights": wn,
                                "plastic": False, "conn_type": "I_TO_E"}
        # ---- FS-WTA (parent) : pool -> fs (exc), fs -> pool (inh) ----
        pef = np.repeat(self.all_pool, len(self.fs_idx)); qef = np.tile(self.fs_idx, len(self.all_pool))
        union["pool2fs"] = {"pre_indices": pef, "post_indices": qef,
                            "initial_weights": np.full(len(pef), self.exc_to_fs, np.float32),
                            "plastic": False, "conn_type": "E_TO_E"}
        pfe = np.repeat(self.fs_idx, len(self.all_pool)); qfe = np.tile(self.all_pool, len(self.fs_idx))
        union["fs2pool"] = {"pre_indices": pfe, "post_indices": qfe,
                            "initial_weights": np.full(len(pfe), self.fs_to_exc, np.float32),
                            "plastic": False, "conn_type": "I_TO_E"}
        # ---- (1) DIVISIVE-NORM HOMEOSTATIC pool dn : pool -> dn (exc sum), dn -> pool (inh, divisive) ----
        if self.dn_size > 0 and self.dn_inh > 0.0:
            ped = np.repeat(self.all_pool, len(self.dn_idx)); qed = np.tile(self.dn_idx, len(self.all_pool))
            union["pool2dn"] = {"pre_indices": ped, "post_indices": qed,
                                "initial_weights": np.full(len(ped), self.dn_exc, np.float32),
                                "plastic": False, "conn_type": "E_TO_E"}
            pde = np.repeat(self.dn_idx, len(self.all_pool)); qde = np.tile(self.all_pool, len(self.dn_idx))
            wdn = np.full(len(pde), self.dn_inh, np.float32)
            union["dn2pool"] = {"pre_indices": pde, "post_indices": qde,
                                "initial_weights": wdn, "plastic": False, "conn_type": "I_TO_E"}
            self._dn_edges = (pde, qde, wdn.copy())
        else:
            self._dn_edges = None
        # ---- (2) RECURRENT within-pool EXCITATION : pool_k[i] -> pool_k[j] (i != j) ----
        if self.rec_gain > 0.0 and self.P > 1:
            rp = []; rq = []
            for k in range(self.V):
                pk = self.pool_idx[k]
                a = np.repeat(pk, self.P); c = np.tile(pk, self.P)
                mask = a != c
                rp.append(a[mask]); rq.append(c[mask])
            rp = np.concatenate(rp); rq = np.concatenate(rq)
            wr = np.full(len(rp), self.rec_gain, np.float32)
            union["recurrent"] = {"pre_indices": rp, "post_indices": rq, "initial_weights": wr,
                                  "plastic": False, "conn_type": "E_TO_E"}
            self._rec_edges = (rp, rq, wr.copy())
        else:
            self._rec_edges = None
        # SHADOW + FS + dn are inhibitory (polarity is per PRE-neuron)
        inh = np.concatenate([self.hidinh_idx, self.fs_idx, self.dn_idx]).tolist()
        b.inject_explicit_wiring(union, output_inhibitory_indices=inh)
        self._pos_edges = (union["readout_pos"]["pre_indices"], union["readout_pos"]["post_indices"], wp.copy())
        self._neg_edges = (union["readout_neg"]["pre_indices"], union["readout_neg"]["post_indices"], wn.copy())

    # ---- oracle with tunable drive (dn changes the operating point vs the parent's fixed 220/30) ----
    def read_oracle(self, logit_vec, oracle_gain=None, oracle_base=None):
        return super().read_oracle(logit_vec,
                                   oracle_gain=self.oracle_gain if oracle_gain is None else oracle_gain,
                                   oracle_base=self.oracle_base if oracle_base is None else oracle_base)

    # ---- lesions of the two new companion processes ----
    def lesion_dn(self):
        if self._dn_edges is None:
            return
        pre, post, _ = self._dn_edges
        self._b.set_pathway_weights("les_dn", pre, post, np.zeros(len(pre), np.float32), add_missing=False)

    def restore_dn(self):
        if self._dn_edges is None:
            return
        pre, post, w = self._dn_edges
        self._b.set_pathway_weights("res_dn", pre, post, w, add_missing=False)

    def lesion_recurrent(self):
        if self._rec_edges is None:
            return
        pre, post, _ = self._rec_edges
        self._b.set_pathway_weights("les_rec", pre, post, np.zeros(len(pre), np.float32), add_missing=False)

    def restore_recurrent(self):
        if self._rec_edges is None:
            return
        pre, post, w = self._rec_edges
        self._b.set_pathway_weights("res_rec", pre, post, w, add_missing=False)


def _measure(s, ro, ev_ids, warmup, n_max):
    """Small read pass: mean on-distribution mass recovered + silent fraction over up to n_max held-out positions."""
    mass = 0.0; sil = 0; nn = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(len(ids) - 1):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t]); lg_supp = lg.copy()
            if ro.unk_idx >= 0:
                lg_supp[ro.unk_idx] = -1e30
            pfull = _softmax(lg_supp)
            win, _, _, _ = s.read(ap, an, ids[t])
            mass += (pfull[win] if win >= 0 else 0.0)
            sil += int(win < 0); nn += 1
            if nn >= n_max:
                break
        if nn >= n_max:
            break
    nn = max(1, nn)
    return mass / nn, sil / nn


def _parity_verdict(m):
    """PARITY GO (all must hold). Distinct from the parent's boundary-lift verdict: requires proj_recovery near the
    oracle, ~0 silence, sign load-bearing HERE, AND both companion-process lesions load-bearing."""
    chance = m["chance_1_over_v"]; n = m["n_positions"]
    checks = {
        # PARITY: the signed projection recovers ~all of the perfect-current (oracle) resolution ceiling.
        "proj_recovery_ge_0.90": m["projection_recovery"] >= 0.90,
        # read_fid MATERIALLY above the parent's 0.55 lift.
        "read_fidelity_ge_0.65": m["read_fidelity_vs_sampler"] >= 0.65,
        # 0% silent (the homeostatic set-point erases the ~19% silence).
        "silent_frac_lt_0.02": m["silent_frac"] < 0.02,
        # the NEGATIVE weights are load-bearing HERE (confound resolved: no floor/silence trade-off).
        "signed_beats_positive_only": m["read_fidelity_vs_sampler"] > 1.10 * m["positive_only_fidelity"],
        # companion process (1): lesion dn -> silence returns OR mass degrades (homeostat load-bearing).
        "dn_lesion_degrades": (m["dn_lesion_silent_frac"] > max(0.10, 3 * m["silent_frac"] + 0.02)
                               or m["dn_lesion_mass"] < 0.85 * m["mass_synaptic"]),
        # companion process (2): lesion recurrent -> sharpening (mass) drops (attractor load-bearing).
        "recurrent_lesion_drops": m["recurrent_lesion_mass"] < 0.90 * m["mass_synaptic"],
        # parent anti-cheats (unchanged).
        "argmax_agree_gt_10x_chance": m["argmax_agree"] > 10 * chance,
        "scramble_at_chance": _scramble_at_chance(m["argmax_agree_scramble"], chance, n),
        "readout_lesion_collapses": m["mass_readout_lesion"] < 0.5 * m["mass_synaptic"],
        "provenance_no_host_draw": m["host_rng_draws_on_read_path"] == 0,
        "hidden_active": m["hidden_active_frac"] > 0.9,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    return bool(all(checks.values())), checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=8000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--pops", type=str, default="8")
    ap.add_argument("--hid-pop", type=int, default=1)
    ap.add_argument("--n-eval-pos", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--topk", type=int, default=64)
    # ---- inherited signed-read operating point (2026-08-13) ----
    ap.add_argument("--read-window", type=int, default=150)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--hid-bias", type=float, default=0.0)
    ap.add_argument("--syn-scale", type=float, default=12.0)
    ap.add_argument("--ratio", type=float, default=6.5)
    ap.add_argument("--floor-pA", type=float, default=82.0)      # raised: dn now regulates the too-many-fire side
    ap.add_argument("--fs-to-exc", type=float, default=7.0)
    ap.add_argument("--exc-to-fs", type=float, default=1.2)
    ap.add_argument("--n-fs", type=int, default=48)
    # ---- NEW companion processes ----
    ap.add_argument("--dn-size", type=int, default=32)
    ap.add_argument("--dn-exc", type=float, default=0.6)         # pool -> dn (excitatory sum drive)
    ap.add_argument("--dn-inh", type=float, default=2.0)         # dn -> pool (divisive/normalising inhibition)
    ap.add_argument("--rec-gain", type=float, default=0.9)       # within-pool recurrent excitation (sharpening)
    ap.add_argument("--oracle-gain", type=float, default=220.0)
    ap.add_argument("--oracle-base", type=float, default=30.0)
    ap.add_argument("--sample-temp", type=float, default=0.8)
    ap.add_argument("--gen-tokens", type=int, default=0)
    ap.add_argument("--gen-temp", type=float, default=0.8)
    ap.add_argument("--oracle-every", type=int, default=3)
    ap.add_argument("--no-uniform-thresh", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_signed_read_parity.json")
    args = ap.parse_args()

    if args.smoke:
        args.n_eval_pos = min(args.n_eval_pos, 70)
        args.gen_tokens = args.gen_tokens or 30

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
        for pop in pops:
            s = ParitySignedShadowRead(
                ro, seed, pop=pop, hid_pop=args.hid_pop, ou_std=args.ou_std, read_window=args.read_window,
                hid_gain=args.hid_gain, hid_bias=args.hid_bias, syn_scale=args.syn_scale, ratio=args.ratio,
                floor_pA=args.floor_pA, n_fs=args.n_fs, exc_to_fs=args.exc_to_fs, fs_to_exc=args.fs_to_exc,
                uniform_thresh=not args.no_uniform_thresh, dn_size=args.dn_size, dn_exc=args.dn_exc,
                dn_inh=args.dn_inh, rec_gain=args.rec_gain, oracle_gain=args.oracle_gain, oracle_base=args.oracle_base)
            gen_here = args.gen_tokens if pop == max(pops) else 0
            # base metrics (WITH dn + recurrent active): reuse the parent eval verbatim
            m = _base_eval(seed, ro, ev_ids, vocab, s, args.warmup, args.topk, args.sample_temp,
                           args.n_eval_pos, gen_here, args.gen_temp, oracle_every=args.oracle_every)
            # ---- companion-process lesions (the NEW deliverable) ----
            nles = min(m["n_positions"], 60)
            s.lesion_dn()
            dn_mass, dn_sil = _measure(s, ro, ev_ids, args.warmup, nles)
            s.restore_dn()
            s.lesion_recurrent()
            rec_mass, _ = _measure(s, ro, ev_ids, args.warmup, nles)
            s.restore_recurrent()
            m["dn_lesion_mass"] = round(dn_mass, 4)
            m["dn_lesion_silent_frac"] = round(dn_sil, 4)
            m["recurrent_lesion_mass"] = round(rec_mass, 4)
            m["dn_size"] = s.dn_size; m["dn_exc"] = s.dn_exc; m["dn_inh"] = s.dn_inh; m["rec_gain"] = s.rec_gain
            m["floor_pA"] = args.floor_pA
            lever("parity_dn_homeostat_lesion", before=round(m["mass_synaptic"], 4),
                  after=round(dn_mass, 4), required=False)
            lever("parity_recurrent_wta_lesion", before=round(m["mass_synaptic"], 4),
                  after=round(rec_mass, 4), required=False)
            go, checks = _parity_verdict(m); m["go"] = go; m["checks"] = checks; m["arm"] = "signed_parity"
            results.append(m)
            print(f"[seed {seed} P={pop} floor={args.floor_pA} dn_inh={args.dn_inh} rec={args.rec_gain}] "
                  f"read_fid={m['read_fidelity_vs_sampler']} ORACLE={m['oracle_read_fidelity']} "
                  f"proj_recovery={m['projection_recovery']} pos_only={m['positive_only_fidelity']} "
                  f"silent={m['silent_frac']} dn_les(mass={m['dn_lesion_mass']},sil={m['dn_lesion_silent_frac']}) "
                  f"rec_les_mass={m['recurrent_lesion_mass']} argmax_agree={m['argmax_agree']} "
                  f"scr={m['argmax_agree_scramble']} lesion={m['mass_readout_lesion']} "
                  f"GO={go} ({sum(checks.values())}/{len(checks)})", flush=True)
            if not go:
                print(f"    checks: {json.dumps(checks)}", flush=True)
            if m.get("generation"):
                for pr, g in m["generation"].items():
                    print(f"    [gen '{pr}' nll {g['self_nll']}] {g['text'][:150]}", flush=True)

    # ---- aggregate (per pop) + a 6-seed signed-load-bearing tally ----
    agg = {}
    for m in results:
        key = f"P{m['pop']}"
        d = agg.setdefault(key, {"read_fidelity": [], "oracle": [], "proj_recovery": [], "pos_only": [],
                                 "silent": [], "dn_les_mass": [], "dn_les_sil": [], "rec_les_mass": [],
                                 "signed_lb": [], "go": []})
        d["read_fidelity"].append(m["read_fidelity_vs_sampler"]); d["oracle"].append(m["oracle_read_fidelity"])
        d["proj_recovery"].append(m["projection_recovery"]); d["pos_only"].append(m["positive_only_fidelity"])
        d["silent"].append(m["silent_frac"]); d["dn_les_mass"].append(m["dn_lesion_mass"])
        d["dn_les_sil"].append(m["dn_lesion_silent_frac"]); d["rec_les_mass"].append(m["recurrent_lesion_mass"])
        d["signed_lb"].append(bool(m["read_fidelity_vs_sampler"] > 1.10 * m["positive_only_fidelity"]))
        d["go"].append(m["go"])
    summary = {}
    for key, d in agg.items():
        summary[key] = {
            "n_seeds": len(d["go"]), "go_count": int(sum(d["go"])),
            "read_fidelity_mean": round(float(np.mean(d["read_fidelity"])), 4),
            "read_fidelity_min": round(float(np.min(d["read_fidelity"])), 4),
            "oracle_mean": round(float(np.mean(d["oracle"])), 4),
            "proj_recovery_mean": round(float(np.mean(d["proj_recovery"])), 4),
            "positive_only_mean": round(float(np.mean(d["pos_only"])), 4),
            "silent_frac_mean": round(float(np.mean(d["silent"])), 4),
            "dn_lesion_mass_mean": round(float(np.mean(d["dn_les_mass"])), 4),
            "dn_lesion_silent_mean": round(float(np.mean(d["dn_les_sil"])), 4),
            "recurrent_lesion_mass_mean": round(float(np.mean(d["rec_les_mass"])), 4),
            "signed_load_bearing_count": f"{int(sum(d['signed_lb']))}/{len(d['signed_lb'])}",
        }
    out = {"results": results, "summary": summary, "seeds": seeds, "pops": pops, "hid_pop": args.hid_pop,
           "ratio": args.ratio, "topk": args.topk, "read_window": args.read_window, "floor_pA": args.floor_pA,
           "dn_size": args.dn_size, "dn_exc": args.dn_exc, "dn_inh": args.dn_inh, "rec_gain": args.rec_gain,
           "plasticity_off": True, "elapsed_s": round(time.time() - t0, 1),
           "backend": os.environ.get("SIM_BACKEND", "numpy")}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(_native(out), indent=2))
    print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(results)} rows, {time.time()-t0:.0f}s -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
