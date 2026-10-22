#!/usr/bin/env python3
"""cs-race step 1 (PACKET.md lines 70-88): F1 feasibility of the
cue-addressed race. ORGAN ONLY, CPU, dev seeds 1-8, a tools.lab probe.

WHAT THIS BUILDS. ReplayCircuit (idle_replay_organ.py:209-321) generalized
into a cue-addressed, tag-free competition: a cue layer (role-bound WTA
word slots), a mention layer (role-free WTA word slots), cue/mention->E_k
edges at write time (the declared host residual), and Tsodyks-Markram
facilitation on each engram's OWN recurrence only (every other pathway
stp_disabled). No host-computed per-candidate number reaches the race: the
host delivers only parsed question words as sensory current. The shared
organ file is NOT edited (the default-on C5 path stays byte-identical;
step 2 wires production behind BRAIN_CS_RACE, default OFF).

KILL CRITERIA -- written into this probe BEFORE its first run
(PACKET.md line 71; rulings cs-race-go-1020, cs-race-go-1020b):
K1  NO-GO for the cs-race METHOD iff no row of F1_GRID makes ALL clauses
    (a)-(i) pass on 8/8 dev seeds x 10 noise draws at that row. A NO-GO is
    banked; the packet's named next carrier is a recurrent persistent
    mention trace.
K2  A clause that errors or returns an incomplete record counts as FAIL
    for that row -- never a pass, never dropped from the table.
K3  Clause (e) fails if STEER* is undefined: no pA in STEER_PA reaches the
    >= 8/10 win bar for the target engram.
K4  Clause (i): each lesion's outcome must be IDENTICAL across all 10
    draws; any variation fails it (a coin-flip lesion may not become a
    counted bar; the BG G3 precedent).
K5  Chance band for (h): two-sided binomial at n=10 -> wins in [2,8]
    inclusive for BOTH position reads.
K6  Row selection rule (fixed now): the FIRST row of F1_GRID order with
    every clause passing on 8/8 seeds.
Claude rules on [cs-race-f1] RESULT before step 2; this probe sets no bar.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# Root = <repo>/research/runners/<this file> -> derive, never hard-code:
# the same file runs in worktrees and in pool revisions under /home/node.
REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np

from research.runners.idle_replay_organ import (GATE_COMMIT_IN, GATE_ENGRAM,
                                                GATE_EXCL, GATE_LATCH,
                                                ReplayCircuit)
from research.runners import spiking_commit as SC

KILL_CRITERIA = ["K1 NO-GO iff no F1_GRID row passes (a)-(i) on 8/8 seeds "
                 "x 10 draws; next carrier = recurrent persistent mention trace",
                 "K2 error/incomplete clause = FAIL for the row",
                 "K3 (e) fails if STEER* undefined (no pA reaches >= 8/10)",
                 "K4 (i) lesion outcome must be identical across 10 draws",
                 "K5 (h) chance band = wins in [2,8] of 10, both reads",
                 "K6 first-all-pass row in F1_GRID order (fixed before run)"]

# Facts: A old, B new share the cue (lynx, chase); C is unrelated. Two extra
# cue slots (moose, graze) belong to no fact (clause d).
FACTS = [("lynx", "chase", "vole"), ("lynx", "chase", "frog"),
         ("yak", "eat", "grass")]
CUE_OF = {0: ("lynx", "chase"), 1: ("lynx", "chase"), 2: ("yak", "eat")}
UNMATCHED_CUE = ("moose", "graze")
CUE_WORDS = ["lynx", "chase", "yak", "eat", "moose", "graze"]
MENTION_WORDS = ["lynx", "chase", "vole", "frog", "yak", "eat", "grass",
                 "moose", "graze"]
DEV_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
STEER_PA = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
N_DRAWS = 10
# Grid over (U, tau_f, assembly size) -- PACKET.md line 72. Row order is
# frozen here (K6). Values are engine globals (sim/config.py:903-911).
F1_GRID = [
    dict(U=0.15, tau_f=50.0, NE=16),   # 0: engine defaults, base assembly
    dict(U=0.30, tau_f=150.0, NE=16),  # 1: stronger/slower facilitation
    dict(U=0.45, tau_f=300.0, NE=16),  # 2: strong, slow (long recency)
    dict(U=0.30, tau_f=150.0, NE=24),  # 3: mid facilitation, larger assembly
    dict(U=0.45, tau_f=300.0, NE=24),  # 4: strong slow, larger assembly
]
# Protocol constants (ms): encode A, gap, encode B, context mention, cue.
ENCODE_MS = 60
GAP_A_MS = 800          # A is "old": its facilitation decays across this gap
GAP_B_MS = 150          # B is "new": still facilitated at cue onset
MENTION_MS = 150
CUE_MS = 500
HOLD_MS = 100
PRECUE_WINDOW_MS = 30
W_SLOT = 8
W_CUE = 45.0            # cue->E: two-role sum must ignite, one role must not
W_MENT = 12.0           # mention->E: primes, never ignites
CUE_PA = 60.0           # sensory current into a driven cue slot
MENT_PA = 40.0          # context current into a driven mention slot
MENT_MAX_PA = 90.0      # "maximal priming" for clause (c)
_BASE = dict(NE=16, NI=6, NG=12, NGI=8, NS=20, NO=8,
             W_EE=40.0, W_EI=40.0, W_IE=40.0, W_XG=45.0, W_GG=20.0,
             W_GGI=60.0, W_GIG=60.0, W_GS=40.0, W_SS=60.0, W_SO=20.0,
             W_OG=80.0, W_OE=40.0, NOISE_SD=25.0, NOISE_TAU=50.0,
             BIAS_PA=42.0, REST_STEPS=CUE_MS, HOLD_STEPS=HOLD_MS,
             READ_STEPS=30, READ_FRAC=0.03,
             # f1d step 1b (RULING cs-race-f1d-1021): flags default to
             # today's values; with no new flag every decision is
             # unchanged (identity check 1d#1 asserts it).
             W_GIS=0.0, W_SGI=0.0, CUE_N=8,
             CUE_NOISE_SD=0.0, CUE_NOISE_TAU=5.0)
GATE_CUE = "cs_cue"          # sensory drive gate for the cue layer
GATE_MENT = "cs_ment"        # mention->E gate (lesion i closes it)


class RaceCircuit(ReplayCircuit):
    """ReplayCircuit + cue/mention layers + engram-only STP."""

    def __init__(self, K, seed, p, grid, facts=None):
        self.grid = dict(grid)
        self.facts = list(facts) if facts else list(FACTS)
        super().__init__(K, seed, dict(p, NE=int(grid["NE"])))

    def slot_of(self, patient):
        """Slot index of the fact whose patient is `patient` (content-addressed
        so clause (f)'s reversed-slot circuit resolves the same protocol)."""
        for k, f in enumerate(self.facts):
            if f[2] == patient:
                return k
        raise KeyError(patient)

    def _build(self):
        from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
        from sim.config import CoreSimConfig
        from sim.enums import NeuronModel
        from sim.backend import get_backend
        from research.runners._gnw_rung1_ignition_curve_derisk import _snapshot_state
        p, K = self.p, self.K
        xp, _ = get_backend()
        NE, NI, NO = int(p["NE"]), int(p["NI"]), int(p["NO"])
        o_e, o_i = 0, K * NE
        o_c = o_i + K * NI
        lay = SC.layout(K, o_c, int(p["NG"]), int(p["NGI"]), int(p["NS"]))
        o_o = lay["end"]
        self.eng = [np.arange(o_e + k * NE, o_e + (k + 1) * NE, dtype=np.int64)
                    for k in range(K)]
        self.inh = [np.arange(o_i + k * NI, o_i + (k + 1) * NI, dtype=np.int64)
                    for k in range(K)]
        self.lay = lay
        self.occ = np.arange(o_o, o_o + NO, dtype=np.int64)   # occupancy pool
        o_cue = o_o + NO
        cue_n = int(p.get("CUE_N", W_SLOT))   # f1d --cue-n: cue slots
        # only; mention slots stay W_SLOT (ruling step 1b)
        o_ment = o_cue + len(CUE_WORDS) * cue_n
        n = o_ment + len(MENTION_WORDS) * W_SLOT
        self.cue = [np.arange(o_cue + i * cue_n, o_cue + (i + 1) * cue_n,
                              dtype=np.int64) for i in range(len(CUE_WORDS))]
        self.ment = [np.arange(o_ment + i * W_SLOT, o_ment + (i + 1) * W_SLOT,
                               dtype=np.int64) for i in range(len(MENTION_WORDS))]
        self.all_eng = np.arange(o_e, o_i, dtype=np.int64)
        cfg = CoreSimConfig()
        cfg.num_neurons = int(n)
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.seed = self.seed
        cfg.dt_ms = 1.0
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1
        for f in ("enable_stdp", "enable_reward_modulation", "enable_homeostasis",
                  "enable_structural_plasticity", "enable_watts_strogatz",
                  "enable_ou_process", "enable_hebbian_learning"):
            setattr(cfg, f, False)
        # Tsodyks-Markram facilitation ON, on every pathway by default; each
        # plan entry below opts OUT via stp_disabled except engram recurrence.
        cfg.enable_short_term_plasticity = True
        cfg.stp_U = float(self.grid["U"])
        cfg.stp_tau_f = float(self.grid["tau_f"])
        cfg.stp_tau_d = 200.0
        cfg.enable_per_type_stp = False     # globals only; ee inherits stp_U/tau_f
        cfg.enable_parameter_heterogeneity = True
        # Slow NMDA recurrent component (ruling: record-supported).
        cfg.enable_nmda_recurrent = (
            float(p.get("W_NMDA", 0.0)) > 0.0)
        bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                  runtime_state=RuntimeState(), gpu_config=GPUConfig())
        bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        bridge._initialize_simulation_data(called_from_playback_init=False)
        plan = {}
        for k in range(K):
            ee = SC.dense(self.eng[k], self.eng[k], p["W_EE"], conn_type="E_TO_E")
            ee.pop("stp_disabled", None)          # engram recurrence KEEPS STP
            plan["ee_%d" % k] = ee
            if float(p.get("W_NMDA", 0.0)) > 0.0:
                plan["eek_nmda_%d" % k] = SC.dense(
                    self.eng[k], self.eng[k], float(p["W_NMDA"]),
                    conn_type="E_TO_E", receptor="nmda_slow")
            plan["ei_%d" % k] = SC.dense(self.eng[k], self.inh[k], p["W_EI"],
                                         conn_type="E_TO_I")
            if K > 1:
                others = np.concatenate([self.eng[j] for j in range(K) if j != k])
                plan["ie_%d" % k] = SC.dense(self.inh[k], others, p["W_IE"],
                                             conn_type="I_TO_E")
            # cue role-bound -> E_k if fact k holds the word in that role
            agent, action = CUE_OF[k]
            for w, role_word in ((agent, "agent"), (action, "action")):
                i = CUE_WORDS.index(w)
                plan["cue_%s_%d" % (role_word, k)] = SC.dense(
                    self.cue[i], self.eng[k], float(p.get("W_CUE",
                                                       W_CUE)),
                    conn_type="E_TO_E")
            # role-free mention -> E_k if fact k contains the word
            for w in FACTS[k]:
                i = MENTION_WORDS.index(w)
                plan["ment_%d_%d" % (i, k)] = SC.dense(
                    self.ment[i], self.eng[k], W_MENT, conn_type="E_TO_E")
        plan.update(SC.commit_plan(self.eng, lay, dict(p, W_XG=p["W_XG"]),
                                   prefix="ir_", gate_in=GATE_COMMIT_IN,
                                   gate_excl=GATE_EXCL, latch_loop=GATE_LATCH))
        plan["so"] = SC.dense(lay["all_slot"], self.occ, p["W_SO"], conn_type="E_TO_I")
        plan["og"] = SC.dense(self.occ, lay["all_gate"], p["W_OG"], conn_type="I_TO_E")
        plan["oe"] = SC.dense(self.occ, self.all_eng, p["W_OE"], conn_type="I_TO_E")
        # The engram master gate covers every engram-local pathway (the base
        # organ expects it known); mention edges keep their OWN cs_ment gate
        # so clause (i) can close mention without touching the master.
        for key, entry in plan.items():
            if key.startswith(("ee_", "ei_", "ie_", "cue_")):
                entry["transmission_gate"] = GATE_ENGRAM
        for key, entry in plan.items():
            if key.startswith("ment_"):
                entry["transmission_gate"] = GATE_MENT
        # Everything except engram recurrence opts out of STP (facilitation on
        # each engram's own recurrence only; packet line 41-42).
        for key, entry in plan.items():
            if not key.startswith("ee_"):
                entry["stp_disabled"] = True
        # mention edges get their own transmission gate (clause i lesion).
        for key, entry in plan.items():
            if key.startswith("ment_"):
                entry["transmission_gate"] = GATE_MENT
        inh = np.concatenate(self.inh + [lay["all_gate_inh"], self.occ]) if K \
            else np.concatenate([self.occ])
        bridge.inject_explicit_wiring(plan, output_inhibitory_indices=inh)
        bridge.set_plasticity_gate(SC.GATE_FIXED, 0.0)
        self._gates = tuple(g for g in (GATE_ENGRAM, GATE_COMMIT_IN,
                                        GATE_LATCH, GATE_EXCL, GATE_MENT)
                            if any(e.get("transmission_gate") == g
                                   for e in plan.values()))
        for g in self._gates:
            bridge.set_transmission_gate(g, 1.0)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(60):
            bridge._run_one_simulation_step()
        self._bridge, self._xp, self._n = bridge, xp, int(n)
        self._snap = _snapshot_state(bridge, xp)

    # ── the race itself ────────────────────────────────────────────────
    def race(self, *, seed, cue_words, mention_words=(), encode=("B",),
             steer_pa=0.0, steer_target=None, steer_from_step=None,
             steer_stage="engram",
             nonselective=False, ment_gate=True, excl_gate=True,
             precue_ms=PRECUE_WINDOW_MS):
        """One turn: optional encode pulses + context mention, then the cue
        window with OU noise, hold, latch read. Returns the full record."""
        from sim.backend import get_backend, to_host
        from research.runners._gnw_rung1_ignition_curve_derisk import _reset_quiescent
        b, p, K = self._bridge, self.p, self.K
        xp = self._xp
        _reset_quiescent(b, xp, self._snap)
        b.set_transmission_gate(GATE_ENGRAM, 1.0)
        # step 1b: the encode->latch path stays CLOSED while writing; a
        # write-time latch (seen pre-cue: latch0=4-7, slot V=-15 mV) drives
        # the occupancy pool -> W_OE=40 global engram inhibition and the cue
        # can never ignite an engram.  Opened at cue onset below (g).
        b.set_transmission_gate(GATE_COMMIT_IN, 0.0)
        b.set_transmission_gate(GATE_EXCL, 1.0 if excl_gate else 0.0)
        b.set_transmission_gate(GATE_MENT, 1.0 if ment_gate else 0.0)
        rng = np.random.default_rng(int(seed))
        tau = float(p["NOISE_TAU"])
        a_ou = 1.0 / tau
        s_ou = float(p["NOISE_SD"]) * math.sqrt(2.0 / tau)
        nE = K * int(p["NE"])
        eta = rng.standard_normal(nE) * float(p["NOISE_SD"])
        eng_counts = np.zeros(K, dtype=np.int64)
        gate_counts = np.zeros(K, dtype=np.int64)
        first_gate = [None] * K
        gate_all = [g for g in self.lay["gate"]]
        latch_before = [0] * K
        natural_winner = None
        cue_spk = 0
        flip_step = steer_from_step
        stp_seen = []

        def step_with(cur):
            b.cp_external_input_current[:] = xp.asarray(cur.astype(np.float32))
            b._run_one_simulation_step()

        def zeros():
            return np.zeros(self._n, dtype=np.float32)

        def pulse(target_idx, pa, ms):
            for _ in range(int(ms)):
                cur = zeros()
                cur[target_idx] = float(pa)
                step_with(cur)

        # encode protocol (host passes no per-candidate number; these are
        # write-time stimulation pulses of the probe's own protocol).
        # Token = fact patient name: the FIRST token is the OLD write, the
        # second the NEW one (facilitation decays across GAP_A_MS / keeps
        # across GAP_B_MS). "ABsim" writes the two compete facts together.
        enc = [t for t in encode if t != "ABsim"]
        if "ABsim" in encode:
            both = np.concatenate([self.eng[self.slot_of("vole")],
                                   self.eng[self.slot_of("frog")]])
            pulse(both, float(p["BIAS_PA"]) * 3.0, ENCODE_MS)
            for _ in range(GAP_B_MS):
                step_with(zeros())
        else:
            for i, name in enumerate(enc):
                pulse(self.eng[self.slot_of(name)],
                      float(p["BIAS_PA"]) * 3.0, ENCODE_MS)
                gap = GAP_A_MS if i == 0 else GAP_B_MS
                for _ in range(gap):
                    step_with(zeros())
        # context mention turn
        if mention_words:
            idx = np.concatenate([self.ment[MENTION_WORDS.index(w)]
                                  for w in mention_words if w in MENTION_WORDS])
            if idx.size:
                pa = MENT_MAX_PA if len(mention_words) >= 3 else MENT_PA
                pulse(idx, pa, MENTION_MS)
                for _ in range(GAP_B_MS):
                    step_with(zeros())
        # pre-cue window: latch must be OFF (clause g)
        f_pre = np.asarray(to_host(b.cp_firing_states)).astype(bool)
        for k in range(K):
            latch_before[k] = int(f_pre[self.lay["slot"][k]].sum())
        # cue window
        b.set_transmission_gate(GATE_COMMIT_IN, 1.0)   # cue onset
        cue_idx = np.concatenate([self.cue[CUE_WORDS.index(w)]
                                  for w in cue_words if w in CUE_WORDS]) \
            if cue_words else np.zeros(0, dtype=np.int64)
        rest = int(p["REST_STEPS"])
        hold = int(p["HOLD_STEPS"])
        read = int(p["READ_STEPS"])
        T = rest + hold
        # f1d step 1c instrument (additive, recorded only): per-ms
        # rasters over cue + hold, per-ms engram mean Vm and total
        # excitatory synaptic current (window means are read after
        # t_ign is known), private cue-cell OU current, last-30 latch.
        eng_ms = np.zeros((K, T), dtype=np.int32)
        gate_ms = np.zeros((K, T), dtype=np.int32)
        slot_ms = np.zeros((K, T), dtype=np.int32)
        word_ms = np.zeros((len(CUE_WORDS), T), dtype=np.int32)
        vm_ms = np.zeros((K, T))
        iexc_ms = np.zeros((K, T))
        slot_counts_last = np.zeros(K, dtype=np.int64)
        cue_all = (np.concatenate(self.cue) if len(self.cue) else
                   np.zeros(0, dtype=np.int64))
        cue_sd = float(p.get("CUE_NOISE_SD", 0.0))
        cue_tau = float(p.get("CUE_NOISE_TAU", 5.0))
        a_c = 1.0 / cue_tau
        s_c = cue_sd * math.sqrt(2.0 / cue_tau)
        # no rng draw when cue_sd == 0 -> A0's noise stream is untouched
        eta_cue = (rng.standard_normal(cue_all.size) * cue_sd
                   if cue_sd > 0.0 else None)
        # latch counts accumulate over the read window (hold, no drive)
        slot_counts = np.zeros(K, dtype=np.int64)
        for t in range(rest + hold):
            cur = zeros()
            if t < rest:
                eta += a_ou * (-eta) + s_ou * rng.standard_normal(nE)
                cur[:nE] += eta.astype(np.float32)
                if eta_cue is not None:      # f1d A3n: private OU on
                    eta_cue += a_c * (-eta_cue) + s_c * rng.standard_normal(
                        cue_all.size)        # cue cells only
                    cur[cue_all] += eta_cue.astype(np.float32)
                if cue_idx.size:
                    cur[cue_idx] += float(CUE_PA)
                if steer_pa and (flip_step is None or t >= flip_step):
                    if steer_stage == "gate":
                        tgt = self.lay["gate"][int(steer_target or 0)]
                    elif nonselective:
                        tgt = np.concatenate([self.eng[k]
                                              for k in range(K)])
                    else:
                        tgt = self.eng[int(steer_target or 0)]
                    cur[tgt] += float(steer_pa)
            step_with(cur)
            f = np.asarray(to_host(b.cp_firing_states)).astype(bool)
            if cue_idx.size:
                cue_spk += int(f[cue_idx].sum())
            vp_now = np.asarray(to_host(b.cp_membrane_potential_v))
            ge_now = (np.asarray(to_host(b.cp_conductance_g_e))
                      if b.cp_conductance_g_e is not None else None)
            for wi in range(len(CUE_WORDS)):
                c_wi = int(f[self.cue[wi]].sum())
                if c_wi:
                    word_ms[wi, t] = c_wi
            for k in range(K):
                e_i = int(f[self.eng[k]].sum())
                eng_counts[k] += e_i
                eng_ms[k, t] = e_i
                gk = int(f[gate_all[k]].sum())
                gate_counts[k] += gk
                gate_ms[k, t] = gk
                if gk and first_gate[k] is None:
                    first_gate[k] = t
                s_i = int(f[self.lay["slot"][k]].sum())
                slot_ms[k, t] = s_i
                vm_ms[k, t] = float(vp_now[self.eng[k]].mean())
                # total excitatory synaptic current per cell:
                # I = g_e * (E_e - V), E_e = syn_reversal_potential_e
                # = 0.0 mV (sim/config.py), mean over engram cells.
                iexc_ms[k, t] = (float(
                    (ge_now[self.eng[k]] *
                     (0.0 - vp_now[self.eng[k]])).mean())
                    if ge_now is not None else 0.0)
            if rest <= t < rest + read:
                for k in range(K):
                    slot_counts[k] += int(
                        f[self.lay["slot"][k]].sum())
            if t >= T - 30:                 # last 30 ms of hold
                for k in range(K):
                    slot_counts_last[k] += int(
                        f[self.lay["slot"][k]].sum())
            if natural_winner is None and sum(
                    1 for k in range(K) if first_gate[k] is not None) >= 1:
                natural_winner = int(np.argmin([first_gate[k] if
                                                first_gate[k] is not None
                                                else 10 ** 9 for k in range(K)]))
        b.cp_external_input_current[:] = 0.0
        # step 1b: per-stage membrane V (mV) at the end of the cue window -
        # a silent stage (cue->engram, engram->gate, gate->slot latch) shows
        # as a population sitting at rest vs near threshold.
        vp = np.asarray(to_host(b.cp_membrane_potential_v))
        thr = np.asarray(to_host(b.cp_neuron_firing_thresholds))

        def _vm(idx):
            if idx is None or len(idx) == 0:
                return None
            return round(float(vp[idx].mean()), 3)

        v_mV = {"cue": _vm(cue_idx),
                "engram": [_vm(self.eng[k]) for k in range(K)],
                "gate": [_vm(gate_all[k]) for k in range(K)],
                "slot": [_vm(self.lay["slot"][k]) for k in range(K)],
                "thr": round(float(thr.mean()), 3)}
        rd = SC.read_with_variants(slot_counts.tolist(), int(p["NS"]), read,
                                   float(p["READ_FRAC"]))
        rd_last = SC.read_with_variants(slot_counts_last.tolist(),
                                        int(p["NS"]), 30,
                                        float(p["READ_FRAC"]))
        # f1d step 1c: derive ignition / latch / burst / pre-ignition
        # fields from the per-ms instrument -- recorded for step 2/3,
        # never used inside the decision itself.
        def _trailing3(c_arr):
            c_arr = np.asarray(c_arr, dtype=np.int64)
            if c_arr.size < 5:
                return None
            w = np.convolve(c_arr, np.ones(5, dtype=np.int64),
                            mode="valid")
            idx = np.flatnonzero(w >= 3)
            return int(idx[0] + 4) if idx.size else None
        t_ign = [_trailing3(eng_ms[k]) for k in range(K)]
        latch_onset = [_trailing3(slot_ms[k]) for k in range(K)]
        gate_bursts = []
        for k in range(K):
            idx_b = np.flatnonzero(gate_ms[k])
            if idx_b.size == 0:
                gate_bursts.append(None)
                continue
            bb = (gate_ms[k] > 0).astype(np.int8)
            dd = np.diff(np.concatenate(([0], bb, [0])))
            runs = (np.flatnonzero(dd == -1) -
                    np.flatnonzero(dd == 1))
            gate_bursts.append(
                {"first_ms": int(idx_b[0]),
                 "last_ms": int(idx_b[-1]),
                 "span_ms": int(idx_b[-1] - idx_b[0] + 1),
                 "longest_run_ms": int(runs.max()) if runs.size else 0})
        firing = [k for k in range(K) if gate_bursts[k] is not None]
        first_burst_k = (min(firing, key=lambda kk:
                             gate_bursts[kk]["first_ms"]) if firing
                         else None)
        last_burst_k = (max(firing, key=lambda kk:
                            gate_bursts[kk]["first_ms"]) if firing
                        else None)
        winner = rd.get("winner")
        if winner is None:
            gate_survivor = "none"
        elif first_burst_k is None:
            gate_survivor = "no_burst"
        elif first_burst_k == last_burst_k:
            gate_survivor = "single_burst"
        elif int(winner) == int(first_burst_k):
            gate_survivor = "first"
        elif int(winner) == int(last_burst_k):
            gate_survivor = "last"
        else:
            gate_survivor = "neither"
        igs = [v for v in t_ign if v is not None]
        if igs:
            w0 = max(int(min(igs)) - 50, 0)
            w1 = max(int(min(igs)) - 5, w0)
            ign_rel = True
        else:
            w0, w1, ign_rel = 100, 400, False
        # cue-only current per engram, ruling formula: cue raster x
        # W_CUE x conductance kernel, kernel = the engine's own
        # _cached_decay_e = exp(-dt/syn_tau_g_e); dense edges, so every
        # cell of a fact's cue word drives every target of that fact.
        lam = getattr(b, "_cached_decay_e", None)
        if lam is None or not float(lam):
            raise RuntimeError("_cached_decay_e missing: cannot compute "
                               "the cue-only current kernel exactly")
        w_cue_now = float(p.get("W_CUE", W_CUE))
        cue_k = []
        for k in range(K):
            words_k = [CUE_WORDS.index(w) for w in CUE_OF[k]
                       if w in CUE_WORDS]
            exc = (np.asarray(word_ms[words_k].sum(axis=0),
                              dtype=np.float64) if words_k
                   else np.zeros(T))
            acc = 0.0
            g_cue = np.zeros(T)
            for tt in range(T):
                acc = acc * float(lam) + w_cue_now * exc[tt]
                g_cue[tt] = acc
            cue_k.append(g_cue)
        pre_ign = {"window_ms": [int(w0), int(w1)],
                   "ignition_relative": bool(ign_rel),
                   "vm_mean_mv_per_k": [
                       round(float(vm_ms[k, w0:w1].mean()), 4)
                       if w1 > w0 else None for k in range(K)],
                   "vm_mean_mv": (round(float(vm_ms[:, w0:w1].mean()), 4)
                                  if w1 > w0 else None),
                   "total_exc_current_pa_per_k": [
                       round(float(iexc_ms[k, w0:w1].mean()), 4)
                       if w1 > w0 else None for k in range(K)],
                   "total_exc_current_pa": (
                       round(float(iexc_ms[:, w0:w1].mean()), 4)
                       if w1 > w0 else None),
                   "total_exc_current_note":
                       "g_e * (E_e - V) per cell, E_e = "
                       "syn_reversal_potential_e = 0.0 mV, mean over "
                       "engram cells in the window",
                   "cue_only_current_pa_per_k": [
                       round(float(cue_k[k][w0:w1].mean()), 4)
                       if w1 > w0 else None for k in range(K)],
                   "cue_only_current_pa": (
                       round(float(np.mean(
                           [cue_k[k][w0:w1].mean() for k in range(K)])),
                           4) if w1 > w0 else None),
                   "cue_only_formula":
                       "W_CUE * cue raster convolved with the engine "
                       "conductance kernel exp(-dt/syn_tau_g_e)",
                   "engram_spikes_in_window": [
                       int(eng_ms[k, w0:w1].sum()) for k in range(K)]}
        latch_first_last = {"first_read_winner": winner,
                            "last_read_winner": rd_last.get("winner"),
                            "last30_slot_counts":
                                slot_counts_last.tolist(),
                            "disagree": bool(
                                str(winner) != str(rd_last.get("winner")))}

        def _sparse(arr_row):
            idx_s = np.flatnonzero(arr_row)
            return [[int(i), int(arr_row[i])] for i in idx_s]
        raster_sparse = {
            "engram": [_sparse(eng_ms[k]) for k in range(K)],
            "gate": [_sparse(gate_ms[k]) for k in range(K)],
            "slot": [_sparse(slot_ms[k]) for k in range(K)],
            "cue_word": [_sparse(word_ms[wi])
                         for wi in range(len(CUE_WORDS))],
            "encoding": "sparse [ms, count] lists per entity, zeros "
                        "implied; per-ms counts over cue + hold"}
        # STP state over the ONLY STP-active pathways (engram recurrence)
        stp = {"u": None, "x": None}
        if getattr(b, "cp_stp_u", None) is not None:
            mask = np.ones(int(b.cp_stp_u.shape[0]), dtype=bool)
            if getattr(b, "cp_stp_disabled_mask", None) is not None:
                mask = ~np.asarray(to_host(b.cp_stp_disabled_mask)).astype(bool)
            u = np.asarray(to_host(b.cp_stp_u))[mask]
            x = np.asarray(to_host(b.cp_stp_x))[mask]
            if u.size:
                stp = {"u_mean": float(u.mean()), "u_max": float(u.max()),
                       "x_mean": float(x.mean()), "x_max": float(x.max())}
                stp_seen.append(stp)
        return {"winner": rd.get("winner"), "variants": rd.get("variants"),
                "read_invariant": rd.get("read_invariant"),
                "engram_counts": eng_counts.tolist(),
                "gate_counts": gate_counts.tolist(),
                "first_gate_step": first_gate,
                "latch_pre_cue": latch_before,
                "natural_first_gate": natural_winner, "stp": stp,
                "slot_counts": slot_counts.tolist(), "v_mV": v_mV,
                "cue_spikes": cue_spk,
                "t_ign": t_ign, "latch_onset": latch_onset,
                "gate_bursts": gate_bursts,
                "first_burst_k": first_burst_k,
                "last_burst_k": last_burst_k,
                "gate_survivor": gate_survivor,
                "pre_ign": pre_ign,
                "latch_first_last": latch_first_last,
                "raster": raster_sparse}

    def unitary_epsp(self, cue_word="lynx"):
        """f1d step 1c, once per seed: force ONE cue cell to spike from
        the quiescent snapshot; peak dV over 30 ms averaged over its
        engram targets (mV)."""
        from sim.backend import to_host
        from research.runners._gnw_rung1_ignition_curve_derisk import \
            _reset_quiescent
        b = self._bridge
        _reset_quiescent(b, self._xp, self._snap)
        b.set_transmission_gate(GATE_ENGRAM, 1.0)
        b.set_transmission_gate(GATE_COMMIT_IN, 0.0)
        b.set_transmission_gate(GATE_EXCL, 1.0)
        b.set_transmission_gate(GATE_MENT, 1.0)
        i = CUE_WORDS.index(cue_word)
        cell = int(self.cue[i][0])
        tgt = np.concatenate([self.eng[k] for k in range(self.K)
                              if cue_word in CUE_OF[k]])
        v0 = np.asarray(to_host(b.cp_membrane_potential_v))[tgt].copy()
        driven = 0
        spiking = False
        pa_used = 0.0
        for mult in (1.0, 2.0, 3.0, 4.0):
            for _ in range(30):
                cur = np.zeros(self._n, dtype=np.float32)
                pa_used = float(CUE_PA) * mult
                cur[cell] = pa_used
                b.cp_external_input_current[:] = self._xp.asarray(cur)
                b._run_one_simulation_step()
                driven += 1
                if bool(np.asarray(
                        to_host(b.cp_firing_states)).astype(bool)[cell]):
                    spiking = True
                    break
            if spiking:
                break
            for _ in range(20):   # relax to rest between drive rungs
                b.cp_external_input_current[:] = 0.0
                b._run_one_simulation_step()
        peak = np.full(tgt.size, -1e9)
        for _ in range(30):
            b.cp_external_input_current[:] = 0.0
            b._run_one_simulation_step()
            v = np.asarray(to_host(b.cp_membrane_potential_v))[tgt]
            peak = np.maximum(peak, v)
        b.cp_external_input_current[:] = 0.0
        return {"cue_word": cue_word, "cue_cell": cell,
                "ms_driven": driven, "spiked": spiking,
                "cue_pa_used": pa_used, "targets": int(tgt.size),
                "peak_dv_mv": (round(float((peak - v0).mean()), 4)
                               if spiking else None)}


def run_clauses(circ, seed, draws, grid_tag, controls_only=False,
                steer_pa=0.0, steer_stage="engram", steer_target=0):
    """All clauses (a)-(i) for ONE dev seed x `draws` noise draws.
    controls_only=True stops after the two controls (step 1c lever
    grid cells need the controls, not the full clause battery)."""
    out = {"seed": seed, "grid": grid_tag, "draws": draws,
           "KILL": KILL_CRITERIA, "clauses": {}}
    base_seed = seed * 1000003
    # f1d step 1c: unitary cue EPSP, once per seed
    out["unitary_cue_epsp"] = circ.unitary_epsp()

    def winners(tag=None, **kw):
        rows = []
        for d in range(draws):
            try:
                r = circ.race(seed=base_seed + d, **kw)
            except Exception as e:                       # K2
                r = {"winner": "ERROR:%s" % type(e).__name__}
            rows.append(r)
            if tag:
                row = {"clause": tag, "draw": d,
                       "winner": r.get("winner"),
                       "engram_counts": r.get("engram_counts"),
                       "gate_counts": r.get("gate_counts"),
                       "first_gate_step": r.get("first_gate_step"),
                       "slot_counts": r.get("slot_counts"),
                       "latch_pre_cue": r.get("latch_pre_cue"),
                       "natural_first_gate": r.get("natural_first_gate"),
                       "v_mV": r.get("v_mV"),
                       "cue_spikes": r.get("cue_spikes"),
                       "t_ign": r.get("t_ign"),
                       "latch_onset": r.get("latch_onset"),
                       "gate_bursts": r.get("gate_bursts"),
                       "gate_survivor": r.get("gate_survivor"),
                       "pre_ign": r.get("pre_ign"),
                       "latch_first_last": r.get("latch_first_last")}
                if tag == "decisive":   # per-ms rasters: cue-lock and
                    row["raster"] = r.get("raster")   # ignition stats
                out.setdefault("per_draw", []).append(row)
        return rows

    def content(w):
        if w is None:
            return None
        if isinstance(w, str):
            return w
        if 0 <= int(w) < len(circ.facts):
            return circ.facts[int(w)][2]
        return "bad"

    def pass_bar(rows, pred, bar=None):
        good = sum(1 for r in rows if r.get("winner") != "ERROR"
                   and pred(r) is True)
        errors = sum(1 for r in rows if isinstance(r.get("winner"), str)
                     and str(r.get("winner")).startswith("ERROR"))
        n = bar if bar is not None else int(math.ceil(0.8 * draws))
        return {"pass": errors == 0 and good >= n, "wins": good,
                "errors": errors, "n": draws}

    # POSITIVE CONTROL (step 1b, RULING cs-race-f1-void-1021b): one engram
    # written alone, its full cue, no competitor, must commit >= 9/10
    # draws on every dev seed.  A row whose control fails is UNDEFINED,
    # not FAIL (wired into main()'s Verdict as a require()).
    pc_rows = winners(tag="posctl", cue_words=list(CUE_OF[2]),
                      encode=("grass",))
    pc_commits = sum(1 for r in pc_rows if r.get("winner") == 2)
    out["positive_control"] = {
        "cue": list(CUE_OF[2]), "encode": ["grass"], "no_competitor": True,
        "bar": 9, "n": draws, "commits": pc_commits,
        "slot": 2,
        "winners": [content(r["winner"]) for r in pc_rows],
        "pass": bool(pc_commits >= 9)}

    # DECISIVENESS CONTROL (RULING cs-race-f1b-1021c, written before
    # any step-1c run): two engrams written TOGETHER (ABsim = one
    # shared write, no recency gap), one shared cue, no mention -> the
    # race must COMMIT to one of them on >= 9/10 draws per seed, each
    # candidate winning 2-8 of 10 (a race, not a fixed winner).
    d_rows = winners(tag="decisive", cue_words=list(CUE_OF[1]),
                     encode=("ABsim",),
                     steer_pa=float(steer_pa), steer_stage=steer_stage,
                     steer_target=int(steer_target))
    d_err = sum(1 for r in d_rows
                if isinstance(r.get("winner"), str)
                and str(r.get("winner")).startswith("ERROR"))
    d_commits = sum(1 for r in d_rows
                    if isinstance(r.get("winner"), int))
    d_frog = sum(1 for r in d_rows if content(r["winner"]) == "frog")
    d_vole = sum(1 for r in d_rows if content(r["winner"]) == "vole")
    out["decisive_control"] = {
        "encode": ["ABsim"], "cue": list(CUE_OF[1]),
        "no_recency_no_mention": True,
        "bar_commits": 9, "n": draws,
        "commits": d_commits, "frog_wins": d_frog,
        "vole_wins": d_vole, "errors": d_err,
        "split_band": [2, draws - 2],
        "vm_field": "v_mV (per_draw[tag=decisive])",
        "pass": bool(d_err == 0 and d_commits >= 9
                     and 2 <= d_frog <= draws - 2
                     and 2 <= d_vole <= draws - 2)}
    if steer_pa:
        out["decisive_control"]["steer_host_probe"] = {
            "pa": float(steer_pa), "stage": steer_stage,
            "target": int(steer_target),
            "note": "host steer = measurement probe, never a mechanism"}
    if controls_only:
        out["subset"] = ("controls-only: posctl + decisive "
                         "(cs-race-f1b-1021c lever grid)")
        return out

    # (a) A old, B new, same cue -> B (frog) wins >= 8/10
    a_rows = winners(tag="a", cue_words=list(CUE_OF[1]),
                       encode=("vole", "frog"))
    out["clauses"]["a"] = dict(
        pass_bar(a_rows, lambda r: content(r["winner"]) == "frog"),
        winners=[content(r["winner"]) for r in a_rows])
    # (b) mention of A's patient after both writes -> A (vole) >= 8/10
    b_rows = winners(tag="b", cue_words=list(CUE_OF[1]),
                            encode=("vole", "frog"),
                     mention_words=["vole"])
    out["clauses"]["b"] = dict(
        pass_bar(b_rows, lambda r: content(r["winner"]) == "vole"),
        winners=[content(r["winner"]) for r in b_rows])
    # (c) one-role match with maximal priming -> 0 latched draws
    c_rows = winners(tag="c", cue_words=["chase"],
                            encode=("vole", "frog"),
                     mention_words=["lynx", "chase", "vole"])
    out["clauses"]["c"] = dict(
        pass_bar(c_rows, lambda r: r["winner"] is None, bar=draws),
        winners=[content(r["winner"]) for r in c_rows])
    # (d) cue matching no engram -> 0 commits
    d_rows = winners(tag="d", cue_words=list(UNMATCHED_CUE),
                            encode=("vole", "frog"))
    out["clauses"]["d"] = dict(
        pass_bar(d_rows, lambda r: r["winner"] is None, bar=draws),
        winners=[content(r["winner"]) for r in d_rows])
    # (e) steer ladder into A; STEER* = smallest pA with >= 8/10 for vole
    ladder, monotone, steer_star = [], True, None
    prev = -1
    for pa in STEER_PA:
        rows = winners(cue_words=list(CUE_OF[1]), encode=("vole", "frog"),
                       steer_pa=pa, steer_target=0) if pa else a_rows
        good = sum(1 for r in rows if content(r["winner"]) == "vole")
        ladder.append({"pa": pa, "a_wins": good, "n": draws})
        if good < prev:
            monotone = False
        prev = max(prev, good)
        if steer_star is None and good >= int(math.ceil(0.8 * draws)):
            steer_star = pa
    late_flips = non_flips = 0
    if steer_star is not None:
        for d in range(draws):
            nat = winners(cue_words=list(CUE_OF[1]), encode=("vole", "frog"))
            w0 = content(nat[d]["winner"])
            fg = nat[d].get("first_gate_step") or [None] * 3
            win_k = nat[d].get("winner")
            step_fg = None
            if isinstance(win_k, int) and fg[win_k] is not None:
                step_fg = int(fg[win_k]) + 20
            late = circ.race(seed=base_seed + d,
                             cue_words=list(CUE_OF[1]), encode=("vole", "frog"),
                             steer_pa=steer_star, steer_target=0,
                             steer_from_step=step_fg)
            if content(late["winner"]) != w0:
                late_flips += 1
            nonsel = circ.race(seed=base_seed + d,
                               cue_words=list(CUE_OF[1]), encode=("vole", "frog"),
                               steer_pa=steer_star, nonselective=True)
            if content(nonsel["winner"]) != w0:
                non_flips += 1
    out["clauses"]["e"] = {"ladder": ladder, "monotone": monotone,
                           "steer_star_pa": steer_star,
                           "late_flips": late_flips, "late_n": draws,
                           "nonselective_flips": non_flips,
                           "pass": bool(monotone and steer_star is not None
                                        and late_flips <= draws // 10
                                        and non_flips <= draws // 10)}
    # (f) reversed slot order: content winner unchanged >= 9/10 non-tie
    rev = RaceCircuit(3, seed, dict(_BASE), _F1_GRID_ROW,
                      facts=list(reversed(FACTS)))
    unchanged = ties = 0
    for d in range(draws):
        w1 = circ.race(seed=base_seed + d, cue_words=list(CUE_OF[1]),
                       encode=("vole", "frog"))
        w2 = rev.race(seed=base_seed + d, cue_words=list(CUE_OF[1]),
                      encode=("vole", "frog"))   # same content protocol, slots swapped
        c1, c2 = content(w1["winner"]), content(w2["winner"])
        if c1 is None or c2 is None:
            ties += 1
            continue
        if c1 == c2:
            unchanged += 1
    non_tie = draws - ties
    out["clauses"]["f"] = {"unchanged": unchanged, "non_tie": non_tie,
                           "bar": int(math.ceil(0.9 * max(non_tie, 1))),
                           "pass": bool(non_tie and unchanged >=
                                        int(math.ceil(0.9 * non_tie)))}
    # (g) 5 consecutive turns: no pre-cue latch, no facilitation bleed
    g_ok, stp_log, post_ok = True, [], True
    for turn in range(5):
        if turn % 2 == 0:
            cw, mw = list(CUE_OF[1]), []
        else:
            cw, mw = list(CUE_OF[2]), []      # unrelated cue after a win
        r = circ.race(seed=base_seed + 1000 + turn, cue_words=cw,
                      encode=("vole", "frog"), mention_words=mw)
        if any(int(v) > 0 for v in r["latch_pre_cue"]):
            g_ok = False
        stp_log.append({"turn": turn, "stp": r["stp"]})
        if turn == 1:
            w = content(r["winner"])
            if w in ("vole", "frog"):         # previous winner won unrelated cue
                post_ok = False
    out["clauses"]["g"] = {"precue_clean": g_ok, "no_bleed": post_ok,
                           "stp_per_turn": stp_log,
                           "pass": bool(g_ok and post_ok)}
    # (h) near tie: simultaneous writes; position reads inside [2,8]
    h_rows = winners(tag="h", cue_words=list(CUE_OF[1]),
                            encode=("ABsim",))
    pos_low = sum(1 for r in h_rows
                  if isinstance(r["winner"], int) and r["winner"] == 0)
    content_wins = [content(r["winner"]) for r in h_rows]
    alpha_frog = sum(1 for c in content_wins if c == "frog")
    out["clauses"]["h"] = {
        "pos_lowest_wins": pos_low, "alpha_first_wins": alpha_frog,
        "n": draws, "band": [2, draws - 2],
        "per_position": {"slot0": pos_low, "slot1": sum(
            1 for r in h_rows if isinstance(r["winner"], int)
            and r["winner"] == 1), "other": sum(
            1 for r in h_rows if r["winner"] is not None
            and not isinstance(r["winner"], int))},
        "pass": bool(2 <= pos_low <= draws - 2
                     and 2 <= alpha_frog <= draws - 2)}
    # (i) deterministic lesions: mention edge closed; exclusion closed
    l1 = winners(tag="i1", cue_words=list(CUE_OF[1]),
                        encode=("vole", "frog"),
                 mention_words=["vole"], ment_gate=False)
    l1_out = [content(r["winner"]) for r in l1]
    l2 = winners(tag="i2", cue_words=list(CUE_OF[1]),
                        encode=("vole", "frog"), excl_gate=False)
    l2_out = [(r["winner"] if r["winner"] is None
               else ("commit" if not isinstance(r["winner"], str) else r["winner"]))
              for r in l2]
    det1 = len(set(map(str, l1_out))) == 1
    det2 = len(set(map(str, l2_out))) == 1
    out["clauses"]["i"] = {"mention_closed_outcomes": l1_out,
                           "excl_closed_outcomes": l2_out,
                           "mention_deterministic": det1,
                           "excl_deterministic": det2,
                           "pass": bool(det1 and det2)}
    return out


_F1_GRID_ROW = dict(F1_GRID[0])


_HET_FIELDS = (("a", "cp_izh_a"), ("b", "cp_izh_b"),
               ("C", "cp_izh_C"), ("d", "cp_izh_d_increment"),
               ("v0", "cp_membrane_potential_v"),
               ("u0", "cp_recovery_variable_u"))


def _transplant_from_a0(a0, arm):
    """f1d id2 ruling: copy A0's a, b, C, d, v0 and u0 into the arm's
    shared indices before the first step -- harness initial conditions,
    not a mechanism; the arm's extra cue cells keep their own draws."""
    from sim.backend import to_host
    n_shared = int(a0.occ[-1]) + 1
    for _short, fname in _HET_FIELDS:
        src = np.asarray(to_host(getattr(a0._bridge, fname)))
        dst = getattr(arm._bridge, fname)
        dst[:n_shared] = arm._xp.asarray(src[:n_shared])
    return n_shared


def identity_cue_n(base, grid, seeds, cue_n, w_cue, draws, out_arg):
    """f1d step 1d#2: for arms that change --cue-n, the engram, gate,
    GI, slot and occupancy heterogeneity arrays (a, b, d, C) and the
    initial state equal A0's for the shared indices, and the per-seed
    mean cue-only current of A0 and A3 is recorded.  Any mismatch ->
    STOP with the diff (return 1); never proceeds silently."""
    from sim.backend import to_host
    rec = {"probe": "cs_race_f1_identity_cue_n",
           "het_transplant_from_A0": True,
           "A0": {"cue_n": 8, "w_cue": float(W_CUE)},
           "A3": {"cue_n": int(cue_n), "w_cue": float(w_cue)},
           "A3n": {"cue_n": 8, "cue_noise_sd": 0.15},
           "seeds": [], "seeds_pass": 0}
    all_ok = True
    for seed in seeds:
        # A0 = today's defaults.  The CLI may have already written
        # --cue-n/--w-cue into `base` (main() wires them before this
        # call), so A0 must be forced back to the defaults -- otherwise
        # the compare is vacuous (caught: first run compared A3 to A3).
        b0 = dict(base)
        b0["CUE_N"] = 8
        b0.pop("W_CUE", None)
        a0 = RaceCircuit(3, seed, b0, grid)
        a3 = RaceCircuit(3, seed,
                         dict(b0, CUE_N=int(cue_n),
                              W_CUE=float(w_cue)), grid)
        n_shared = int(a0.occ[-1]) + 1   # everything before the cue
        # layer is index-shared: E, I, layout (gate/slot/GI), occupancy

        def _arr(c, name):
            return np.asarray(to_host(getattr(c._bridge, name)))
        # pre-transfer state of the six arrays (kept in the record,
        # RULING cs-race-f1d-id2): b/C/d differ by the draw offset
        pre, pre_diff = {}, {}
        for short, nm in _HET_FIELDS:
            d0 = _arr(a0, nm)[:n_shared]
            d3 = _arr(a3, nm)[:n_shared]
            pre[short] = bool(np.array_equal(d0, d3))
            pre_diff[short] = (0.0 if pre[short] else
                               round(float(np.abs(d0 - d3).max()), 6))
        # transplant A0's heterogeneity + init state into A3's shared
        # indices BEFORE the first step (harness init, not a mechanism)
        _transplant_from_a0(a0, a3)
        arr = {}
        for short, nm in _HET_FIELDS:
            arr[short] = bool(np.array_equal(
                _arr(a0, nm)[:n_shared], _arr(a3, nm)[:n_shared]))
        # A3n (ruling 2): n unchanged -> build arrays equal A0's; its
        # cue OU noise must come from a private generator, so racing
        # under two DIFFERENT global np.random states must be equal
        an = RaceCircuit(3, seed, dict(b0, CUE_NOISE_SD=0.15), grid)
        an2 = RaceCircuit(3, seed, dict(b0, CUE_NOISE_SD=0.15), grid)
        a3n_arr = {}
        for short, nm in _HET_FIELDS:
            a3n_arr[short] = bool(np.array_equal(
                _arr(a0, nm), _arr(an, nm)))
        # first race on each fresh circuit (races are history-
        # dependent), under two different GLOBAL np.random states: if
        # equal, the cue OU uses its private seeded generator and no
        # other draw touches the global stream
        _ck = list(CUE_OF[1])
        np.random.seed(1)
        r1 = an.race(seed=seed * 1000003 + 900000,
                     cue_words=_ck, encode=("ABsim",))
        np.random.seed(99991)
        r2 = an2.race(seed=seed * 1000003 + 900000,
                      cue_words=_ck, encode=("ABsim",))
        a3n_ou = (json.dumps(r1, sort_keys=True, default=str)
                  == json.dumps(r2, sort_keys=True, default=str))
        a3n_ok = bool(all(a3n_arr.values()) and a3n_ou)
        cue_pa = {}
        for tag, circ in (("A0", a0), ("A3", a3)):
            vals = []
            for d in range(draws):
                r = circ.race(seed=seed * 1000003 + d,
                              cue_words=list(CUE_OF[1]),
                              encode=("ABsim",))
                m = (r.get("pre_ign") or {}).get(
                    "cue_only_current_pa")
                if m is not None:
                    vals.append(float(m))
            cue_pa[tag] = (round(sum(vals) / len(vals), 4)
                           if vals else None)
        ok = bool(all(arr.values()) and a3n_ok)
        all_ok = all_ok and ok
        rec["seeds"].append({"seed": seed, "n_shared": n_shared,
                             "het_transplant_from_A0": True,
                             "arrays_equal_pre_transplant": pre,
                             "pre_transplant_max_abs_diff": pre_diff,
                             "arrays_equal": arr,
                             "a3n": {"cue_n": 8, "cue_noise_sd": 0.15,
                                     "build_arrays_equal_a0": a3n_arr,
                                     "ou_private_global_seed_invariant":
                                         bool(a3n_ou)},
                             "mean_cue_only_current_pa": cue_pa,
                             "pass": ok})
        rec["seeds_pass"] += int(ok)
        print("identity 1d#2 seed %d: pass=%s shared=%d a3n_ou=%s "
              "cue_only A0=%s A3=%s"
              % (seed, ok, n_shared, a3n_ou,
                 cue_pa["A0"], cue_pa["A3"]), flush=True)
    rec["pass"] = bool(all_ok and len(rec["seeds"]) == len(seeds))
    path = out_arg or os.path.join(
        REPO, "research/findings/raw/_cs_race/f1d/identity_cue_n.json")
    path = path if os.path.isabs(path) else os.path.join(REPO, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True, default=str)
    from research.runners import declare_output
    declare_output(path)
    if not rec["pass"]:
        print("IDENTITY 1d#2 FAIL -- STOP with the diff:", flush=True)
        for s in rec["seeds"]:
            if not s["pass"]:
                print("  seed %d: %s" % (s["seed"], s["arrays_equal"]),
                      flush=True)
        print("diff -> %s" % path, flush=True)
        return 1
    print("identity 1d#2 PASS on %d/%d seeds -> %s"
          % (rec["seeds_pass"], len(seeds), path), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--grid-row", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--w-ee", type=float, default=None,
                    help="override W_EE, engram recurrent excitation")
    ap.add_argument("--w-nmda", type=float, default=0.0,
                    help="slow NMDA recurrent weight (0 = off)")
    ap.add_argument("--w-ie", type=float, default=None,
                    help="override W_IE, exclusion (lateral) inhibition")
    ap.add_argument("--w-xg", type=float, default=None,
                    help="override W_XG, engram->gate drive")
    ap.add_argument("--w-gs", type=float, default=None,
                    help="override W_GS, gate->slot drive")
    ap.add_argument("--w-cue", type=float, default=None,
                    help="override W_CUE, cue->engram synapse weight")
    ap.add_argument("--w-gis", type=float, default=None,
                    help="override W_GIS, bursting gate GI -> other "
                         "slots (single-slot exclusion)")
    ap.add_argument("--w-sgi", type=float, default=None,
                    help="override W_SGI, latched slot -> own gate GI")
    ap.add_argument("--cue-n", type=int, default=None,
                    help="cue slots per cue word (mention slots stay "
                         "%d; f1d --cue-n)" % W_SLOT)
    ap.add_argument("--cue-noise-sd", type=float, default=None,
                    help="private OU current sd (pA) on cue cells")
    ap.add_argument("--cue-noise-tau", type=float, default=None,
                    help="private OU tau (ms) on cue cells (default 5)")
    ap.add_argument("--identity-cue-n", action="store_true",
                    help="f1d step 1d#2: build A0 vs a --cue-n arm, "
                         "compare heterogeneity arrays (a,b,d,C) + "
                         "initial state on shared indices, record each "
                         "arm's mean cue-only current, then exit")
    ap.add_argument("--steer-pa", type=float, default=0.0,
                    help="host steer current pA on the decisive control "
                         "(measurement probe; record = steer_host_probe)")
    ap.add_argument("--steer-stage", choices=("engram", "gate"),
                    default="engram",
                    help="which stage receives the host steer")
    ap.add_argument("--steer-target", type=int, default=0,
                    help="engram/gate index k receiving the host steer")
    args = ap.parse_args()
    global _F1_GRID_ROW
    _F1_GRID_ROW = dict(F1_GRID[args.grid_row])
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    from tools.lab import assert_backend, attributable_to
    from tools.verdict import Verdict
    assert_backend("numpy", note="cs-race step 1 F1 feasibility probe")

    t0 = time.time()
    print("KILL CRITERIA (pre-registered):", flush=True)
    for line in KILL_CRITERIA:
        print("  " + line, flush=True)
    base = dict(_BASE)
    if args.w_ee is not None:
        base["W_EE"] = float(args.w_ee)
    if args.w_ie is not None:
        base["W_IE"] = float(args.w_ie)
    if args.w_xg is not None:
        base["W_XG"] = float(args.w_xg)
    if args.w_gs is not None:
        base["W_GS"] = float(args.w_gs)
    if args.w_cue is not None:
        base["W_CUE"] = float(args.w_cue)
    if args.w_nmda:
        base["W_NMDA"] = float(args.w_nmda)
    if args.w_gis is not None:
        base["W_GIS"] = float(args.w_gis)
    if args.w_sgi is not None:
        base["W_SGI"] = float(args.w_sgi)
    if args.cue_n is not None:
        base["CUE_N"] = int(args.cue_n)
    if args.cue_noise_sd is not None:
        base["CUE_NOISE_SD"] = float(args.cue_noise_sd)
    if args.cue_noise_tau is not None:
        base["CUE_NOISE_TAU"] = float(args.cue_noise_tau)
    if args.identity_cue_n:
        return identity_cue_n(base, _F1_GRID_ROW, seeds,
                              int(args.cue_n or 48),
                              float(args.w_cue or 7.5), args.draws,
                              args.out)
    results = []
    het_transplant = False
    for seed in seeds:
        circ = RaceCircuit(3, seed, dict(base), _F1_GRID_ROW)
        if args.cue_n is not None and int(args.cue_n) != 8:
            # f1d id2: cue-n arms transplant A0's heterogeneity +
            # init state into shared indices before the first step
            src = RaceCircuit(3, seed, dict(base, CUE_N=8),
                              _F1_GRID_ROW)
            _transplant_from_a0(src, circ)
            het_transplant = True
        res = run_clauses(
            circ, seed, args.draws,
            "row%d U=%s tau_f=%s NE=%s W_EE=%s W_IE=%s W_NMDA=%s "
            "W_GIS=%s W_SGI=%s CUE_N=%s CUE_NOISE_SD=%s"
            % (args.grid_row, _F1_GRID_ROW["U"],
               _F1_GRID_ROW["tau_f"], _F1_GRID_ROW["NE"],
               base["W_EE"], base["W_IE"],
               base.get("W_NMDA", 0.0), base.get("W_GIS", 0.0),
               base.get("W_SGI", 0.0), base.get("CUE_N", 8),
               base.get("CUE_NOISE_SD", 0.0)),
            controls_only=args.controls_only,
            steer_pa=args.steer_pa, steer_stage=args.steer_stage,
            steer_target=args.steer_target)
        results.append(res)
        passed = sorted(k for k, v in res["clauses"].items() if v.get("pass"))
        print("seed %d: %d/%d clauses pass (%s)"
              % (seed, len(passed), 9, ",".join(passed)), flush=True)
        pc = res.get("positive_control", {})
        print("    posctl commits %s/%s pass=%s"
              % (pc.get("commits"), pc.get("n"), pc.get("pass")),
              flush=True)
        dc = res.get("decisive_control", {})
        print("    decisive commits %s/%s (frog %s vole %s) pass=%s"
              % (dc.get("commits"), dc.get("n"), dc.get("frog_wins"),
                 dc.get("vole_wins"), dc.get("pass")), flush=True)

    all_seeds_all_clauses = all(
        all(v.get("pass") for v in r["clauses"].values()) for r in results)
    v = Verdict("cs-race step 1: F1 feasibility clause table")
    v.require("8 dev seeds ran", len(results) == 8,
              note="dev seeds only; gate seeds never run")
    if not args.controls_only:
        v.require("every clause present on every seed",
                  all(set(r["clauses"]) == set("abcdefghi")
                      for r in results))
    v.require("no clause errored (K2)",
              all(all(r["clauses"][c].get("errors", 0) == 0
                      for c in r["clauses"]) for r in results))
    v.require("backend numpy", True)
    pc_ok = all(r.get("positive_control", {}).get("pass") for r in results)
    v.require("positive control: one engram written alone commits >= 9/10 "
              "on every dev seed", pc_ok,
              note="step 1b: a control-failed row is UNDEFINED, not FAIL")
    dec_ok = all(r.get("decisive_control", {}).get("pass")
                 for r in results)
    if args.controls_only:
        # RULING cs-race-f1d-1021 step 1a: the positive control stays a
        # require (instrument); the decisiveness control is the DECIDED
        # criterion, so its failure reads NO-GO, not UNDEFINED.
        v.decide(bool(dec_ok))
    else:
        v.decide(all_seeds_all_clauses)

    out = {"probe": "cs_race_f1", "grid_row": args.grid_row,
           "grid": dict(_F1_GRID_ROW), "draws": args.draws,
           "seeds": seeds, "backend": "numpy", "device": "cpu",
           "KILL_CRITERIA": KILL_CRITERIA,
           "row_all_pass_8_seeds": bool(all_seeds_all_clauses
                                    and not args.controls_only),
        "controls_only": bool(args.controls_only),
           "het_transplant_from_A0": bool(het_transplant),
           "instrument_flags": {"w_gis": base.get("W_GIS", 0.0),
                                "w_sgi": base.get("W_SGI", 0.0),
                                "cue_n": base.get("CUE_N", 8),
                                "cue_noise_sd":
                                    base.get("CUE_NOISE_SD", 0.0),
                                "cue_noise_tau":
                                    base.get("CUE_NOISE_TAU", 5.0)},
           "results": results, "Verdict": v._decided,
           "preconditions": {
               "env": "probe-local flags only; BRAIN_CS_RACE does not exist "
                      "yet (step 2); default-off world",
               "organ": "RaceCircuit extends ReplayCircuit; shared organ "
                        "file unedited",
               "dev_seeds": "1-8", "draws": args.draws,
               "kill_criteria_written_before_first_run": True,
               "positive_control": "step 1b (cs-race-f1-void-1021b): one "
               "engram written alone, full cue, no competitor, >=9/10 "
               "commits per dev seed; control-failed row = UNDEFINED"},
           "wall_s": round(time.time() - t0, 1)}
    # attribution (gate attribution-required): whose is the b-vs-i1
    # difference?  treatment = b mention intact, control = i1 ment_gate
    # lesioned; K4 expects identical wins, i.e. 0% attributable to the
    # gate -- reported as numbers, not assumed.
    b_w = float(sum((r.get("clauses") or {}).get("b", {}).get("wins", 0)
                    for r in results))
    i1_w = float(sum((r.get("clauses") or {}).get("i1", {}).get("wins", 0)
                     for r in results))
    out["attribution_mention_gate"] = attributable_to(
        "mention gate (b intact vs i1 lesion)", b_w, i1_w)
    if not args.out:
        args.out = ("research/findings/raw/_cs_race/f1/row%d/seed%s.json"
                    % (args.grid_row, "-".join(str(s) for s in seeds)))
    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=str)
    from research.runners import declare_output
    declare_output(out_path)
    print("row%d all_pass=%s wall=%.1fs -> %s"
          % (args.grid_row,
             bool(all_seeds_all_clauses and not args.controls_only),
             out["wall_s"], out_path),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
