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
             CUE_NOISE_SD=0.0, CUE_NOISE_TAU=5.0,
             # f1g (packet cs-race-f1g): persistent word trace in the
             # mention layer. All defaults keep the pre-f1g build.
             W_MM=0.0, W_MI=0.0, W_IM=0.0, NIM=8, ENCODE_WORDS=0,
             # f1h (packet cs-race-f1h): intrinsic word trace = a held
             # BDSP apical plateau in the mention-slot cells. All
             # defaults keep the f1g build byte-for-byte.
             TRACE_INTRINSIC=0, SOMA_G=0.0, APICAL_PA=0.0)
# f1h (packet cs-race-f1h, STEP 0): the gap4 6/6-seed GO configuration of
# the bistable BDSP apical, FROZEN here, never tuned. Source of every value:
# research/runners/_gap4_btsp_onbridge_behavioral_timescale_derisk.py
# (finding 2026-07-18-gap4-BTSP-onbridge-behavioral-timescale-GO-6seed-...;
# lines 57-62 set the cfg, line 75 the pulse). Entries marked [default] are
# the CoreSimConfig defaults the gap4 run left untouched (sim/config.py
# 539, 566-568, 551, 560, 630); they are pinned explicitly so a later
# default change cannot silently move this arm.
GAP4_GO = dict(
    coincidence_plateau_self_regen=2.0,     # gap4 runner :60
    coincidence_plateau_v_hold=-35.0,       # gap4 runner :61
    apical_kir_g=1.0,                       # gap4 runner :62
    bdsp_learning_rate=0.0,                 # gap4 runner :58 (BDSP moves no weight)
    coincidence_plateau_v_hold_k=0.2,       # [default] config.py:539
    apical_kir_E_K=-90.0,                   # [default] config.py:566
    apical_kir_vhalf=-50.0,                 # [default] config.py:567
    apical_kir_k=8.0,                       # [default] config.py:568
    apical_tau_ms=15.0,                     # [default] config.py:550
    apical_E_rest=-65.0,                    # [default] config.py:560
    bdsp_v_apical_scale=0.05)               # [default] config.py:630
GAP4_GO_PULSE_PA = 120.0    # gap4 runner :75 pulse_pA (the packet's pulse value)
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
        # f1g: one shared inhibitory pool for the mention slots (WM
        # overwrite); allocated ONLY when a pool weight is on, so the
        # no-flag build keeps today's neuron count byte-for-byte.
        w_mm = float(p.get("W_MM", 0.0))
        w_mi = float(p.get("W_MI", 0.0))
        w_im = float(p.get("W_IM", 0.0))
        nim = int(p.get("NIM", 8))
        self.pool = None
        if (w_mi > 0.0 or w_im > 0.0) and nim > 0:
            o_pool = n
            n = n + nim
            self.pool = np.arange(o_pool, o_pool + nim,
                                  dtype=np.int64)
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
            float(p.get("W_NMDA", 0.0)) > 0.0 or w_mm > 0.0)
        # f1h: the intrinsic trace (packet cs-race-f1h). Off by default.
        self._ap_on = bool(int(p.get("TRACE_INTRINSIC", 0)))
        self._apical_pa = float(p.get("APICAL_PA", 0.0))
        if self._ap_on:
            if w_mm != 0.0:
                raise ValueError("TRACE_INTRINSIC needs W_MM = 0 "
                                 "(no recurrence at all)")
            cfg.enable_bdsp = True
            cfg.bdsp_apical_bistable = True
            cfg.bdsp_apical_couples_soma = True
            cfg.bdsp_apical_soma_g = float(p.get("SOMA_G", 0.0))
            for _k, _v in GAP4_GO.items():
                setattr(cfg, _k, _v)
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
        # f1g: persistent word trace -- slow-NMDA self-excitation per
        # mention slot (the trace) and one shared inhibitory pool so the
        # newest heard word overwrites the older (packet cs-race-f1g).
        if w_mm > 0.0:
            for i in range(len(MENTION_WORDS)):
                # NOTE (_d3_persistent_slot_derisk.py:70): `receptor=`
                # is the INHIBITORY receptor (gaba_a/gaba_b); the
                # excitatory slow-NMDA routing key is `exc_receptor`.
                mm_e = SC.dense(self.ment[i], self.ment[i], w_mm,
                                conn_type="E_TO_E")
                mm_e["exc_receptor"] = "nmda_slow"
                plan["mm_%d" % i] = mm_e
        if self.pool is not None:
            all_ment = np.concatenate(self.ment)
            plan["mi"] = SC.dense(all_ment, self.pool, w_mi,
                                   conn_type="E_TO_I")
            plan["im"] = SC.dense(self.pool, all_ment, w_im,
                                   conn_type="I_TO_E")
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
        inh_parts = (self.inh + [lay["all_gate_inh"], self.occ] if K
                     else [self.occ])
        if self.pool is not None:
            inh_parts.append(self.pool)
        inh = np.concatenate(inh_parts)
        bridge.inject_explicit_wiring(plan, output_inhibitory_indices=inh)
        bridge.set_plasticity_gate(SC.GATE_FIXED, 0.0)
        self._gates = tuple(g for g in (GATE_ENGRAM, GATE_COMMIT_IN,
                                        GATE_LATCH, GATE_EXCL, GATE_MENT)
                            if any(e.get("transmission_gate") == g
                                   for e in plan.values()))
        for g in self._gates:
            bridge.set_transmission_gate(g, 1.0)
        bridge.cp_external_input_current[:] = 0.0
        if self._ap_on:
            # an allocated zero drive makes the apical integrate (KIR
            # down-state) in the lesion arm exactly as in the trace arm;
            # pulses are written per step by _set_apical.
            self._ment_all = np.concatenate(self.ment)
            bridge.cp_bdsp_apical_drive = xp.zeros(int(n), dtype=xp.float32)
        for _ in range(60):
            bridge._run_one_simulation_step()
        self._bridge, self._xp, self._n = bridge, xp, int(n)
        self._snap = _snapshot_state(bridge, xp)
        if self._ap_on:
            # cp_v_apical is not in _STATE_ARRAYS (_gnw_rung1_ignition_
            # curve_derisk.py:219): without this a latched plateau would
            # leak into the next trial. Snapshot the settled down-state.
            self._vap_rest = bridge.cp_v_apical.copy()

    def _set_apical(self, cur):
        """f1h: write this step's apical drive. A heard word's sensory
        volley reaches the apical arbor of its word-form slot as well as
        the somatic arbor: every mention cell that receives somatic hearing
        current this step gets the apical amplitude (--apical-pulse), all
        other neurons 0. The pulse window is therefore the word window
        (50 ms encode word, 150 ms mention). Amplitude 0 = lesion arm."""
        ap = np.zeros(self._n, dtype=np.float32)
        if self._apical_pa > 0.0:
            m = self._ment_all
            heard = np.asarray(cur[m]) > 0.0
            if heard.any():
                ap[m[heard]] = self._apical_pa
        self._bridge.cp_bdsp_apical_drive = self._xp.asarray(ap)

    def _reset_apical(self):
        """f1h: restore the apical down-state and zero the drive (the
        quiescent snapshot does not carry cp_v_apical). No-op when off."""
        if not self._ap_on:
            return
        self._bridge.cp_v_apical[:] = self._vap_rest
        self._bridge.cp_bdsp_apical_drive = self._xp.zeros(
            self._n, dtype=self._xp.float32)

    # ── the race itself ────────────────────────────────────────────────
    def race(self, *, seed, cue_words, mention_words=(), encode=("B",),
             steer_pa=0.0, steer_target=None, steer_from_step=None,
             steer_stage="engram",
             nonselective=False, ment_gate=True, excl_gate=True,
             precue_ms=PRECUE_WINDOW_MS, encode_words=None,
             cal_window=None, cal_held_k=None, cal_rival_k=None):
        """One turn: optional encode pulses + context mention, then the cue
        window with OU noise, hold, latch read. Returns the full record."""
        from sim.backend import get_backend, to_host
        from research.runners._gnw_rung1_ignition_curve_derisk import _reset_quiescent
        b, p, K = self._bridge, self.p, self.K
        xp = self._xp
        _reset_quiescent(b, xp, self._snap)
        self._reset_apical()
        encode_words = (bool(int(p.get("ENCODE_WORDS", 0)))
                        if encode_words is None
                        else bool(encode_words))
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
            if self._ap_on:
                self._set_apical(cur)
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
        enc_ment_spk = 0
        enc = [t for t in encode if t != "ABsim"]
        if "ABsim" in encode:
            both = np.concatenate([self.eng[self.slot_of("vole")],
                                   self.eng[self.slot_of("frog")]])
            if encode_words:
                # both compete facts are heard together: the vole
                # fact's three words then the frog fact's, 50 ms each
                # from the write pulse's onset (the packet is silent
                # on ABsim; this interpretation is flagged in the f1g
                # WORKLOG).
                k_a, k_b = self.slot_of("vole"), self.slot_of("frog")
                heard = list(self.facts[k_a]) + list(self.facts[k_b])
                for t_ms in range(50 * len(heard)):
                    cur = zeros()
                    if t_ms < ENCODE_MS:
                        cur[both] = float(p["BIAS_PA"]) * 3.0
                    m_w = self.ment[MENTION_WORDS.index(
                        heard[t_ms // 50])]
                    cur[m_w] = float(MENT_PA)
                    step_with(cur)
                    f_w = np.asarray(
                        to_host(b.cp_firing_states)).astype(bool)
                    enc_ment_spk += int(f_w[m_w].sum())
            else:
                pulse(both, float(p["BIAS_PA"]) * 3.0, ENCODE_MS)
            for _ in range(GAP_B_MS):
                step_with(zeros())
        else:
            for i, name in enumerate(enc):
                e_tgt = self.eng[self.slot_of(name)]
                if encode_words:
                    # packet cs-race-f1g: the encode turn is HEARD --
                    # its fact's three words reach their mention slots
                    # in heard order (agent, action, patient), 50 ms
                    # each at MENT_PA, starting at the encode pulse's
                    # onset (world-side; no per-candidate number
                    # passes the host boundary).
                    words_k = list(self.facts[self.slot_of(name)])
                    for wi, w in enumerate(words_k):
                        m_idx = self.ment[MENTION_WORDS.index(w)]
                        for t_ms in range(wi * 50, wi * 50 + 50):
                            cur = zeros()
                            if t_ms < ENCODE_MS:
                                cur[e_tgt] = float(p["BIAS_PA"]) * 3.0
                            cur[m_idx] = float(MENT_PA)
                            step_with(cur)
                            f_w = np.asarray(
                                to_host(b.cp_firing_states)).astype(bool)
                            enc_ment_spk += int(f_w[m_idx].sum())
                else:
                    pulse(e_tgt, float(p["BIAS_PA"]) * 3.0, ENCODE_MS)
                gap = GAP_A_MS if i == 0 else GAP_B_MS
                for _ in range(gap):
                    step_with(zeros())
        # context mention turn: same stepping as pulse(), plus the
        # per-cell spike read-out -> per-draw carrier engagement
        # (RULING cs-race-f1g-0-1022 B: 0 spikes on a draw = the
        # mention clause is UNDEFINED for that draw)
        ment_spk = None
        mention_pa_used = None
        if mention_words:
            idx = np.concatenate([self.ment[MENTION_WORDS.index(w)]
                                  for w in mention_words if w in MENTION_WORDS])
            if idx.size:
                pa = MENT_MAX_PA if len(mention_words) >= 3 else MENT_PA
                mention_pa_used = float(pa)
                ment_cell = np.zeros(idx.size, dtype=np.int64)
                for _ in range(MENTION_MS):
                    cur = zeros()
                    cur[idx] = float(pa)
                    step_with(cur)
                    f_m = np.asarray(
                        to_host(b.cp_firing_states)).astype(bool)
                    ment_cell += f_m[idx].astype(np.int64)
                ment_spk = int(ment_cell.sum())
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
        rec = {"winner": rd.get("winner"), "variants": rd.get("variants"),
                "read_invariant": rd.get("read_invariant"),
                "engram_counts": eng_counts.tolist(),
                "gate_counts": gate_counts.tolist(),
                "first_gate_step": first_gate,
                "latch_pre_cue": latch_before,
                "natural_first_gate": natural_winner, "stp": stp,
                "slot_counts": slot_counts.tolist(), "v_mV": v_mV,
                "cue_spikes": cue_spk,
                "mention_pulse_spikes": ment_spk,
                "mention_pulse_pa": mention_pa_used,
                "encode_mention_spikes": (
                    int(enc_ment_spk) if encode_words else None),
                "t_ign": t_ign, "latch_onset": latch_onset,
                "gate_bursts": gate_bursts,
                "first_burst_k": first_burst_k,
                "last_burst_k": last_burst_k,
                "gate_survivor": gate_survivor,
                "pre_ign": pre_ign,
                "latch_first_last": latch_first_last,
                "raster": raster_sparse}
        if cal_window is not None:
            w_a, w_b = int(cal_window[0]), int(cal_window[1])
            per_k = [round(float(iexc_ms[k, w_a:w_b].mean()), 4)
                     for k in range(K)]
            entry = {"ms": [w_a, w_b],
                     "total_exc_current_pa_per_k": per_k}
            if cal_held_k is not None:
                entry["held_k"] = int(cal_held_k)
            if cal_rival_k is not None:
                entry["rival_k"] = int(cal_rival_k)
            if cal_held_k is not None and cal_rival_k is not None:
                entry["delta_pa"] = round(
                    per_k[int(cal_held_k)] - per_k[int(cal_rival_k)],
                    4)
            rec["cal_window"] = entry
        if encode_words:
            rec["encode_words"] = True
        return rec

    def cal(self, *, seed, item, word=None, word2=None,
            rest_ms=CUE_MS):
        """f1g step 0 (packet cs-race-f1g): trace calibration with NO
        race -- mention-layer pulses, then free dynamics in the window
        the race would be decided in (no cue, no encode, no steer).
        Rates are read from spike counts (Hz per cell, never a host
        flag). Returns the raw record; run_calibration applies the
        C1-C4 thresholds."""
        from sim.backend import to_host
        from research.runners._gnw_rung1_ignition_curve_derisk import \
            _reset_quiescent
        b, p, K = self._bridge, self.p, self.K
        xp = self._xp
        _reset_quiescent(b, xp, self._snap)
        self._reset_apical()
        b.set_transmission_gate(GATE_ENGRAM, 1.0)
        b.set_transmission_gate(GATE_COMMIT_IN, 0.0)
        b.set_transmission_gate(GATE_EXCL, 1.0)
        b.set_transmission_gate(GATE_MENT, 1.0)
        rng = np.random.default_rng(int(seed) + 7717)
        nE = K * int(p["NE"])
        tau = float(p["NOISE_TAU"])
        a_ou = 1.0 / tau
        s_ou = float(p["NOISE_SD"]) * math.sqrt(2.0 / tau)
        eta = rng.standard_normal(nE) * float(p["NOISE_SD"])
        if item == "c2b":
            sched = [("gap", GAP_B_MS)]
            rest = 2000 - GAP_B_MS       # gap + rest = 2000 ms measured
        elif item == "c3":
            sched = [("pulse", word, MENTION_MS), ("gap", GAP_B_MS),
                     ("pulse", word2, MENTION_MS), ("gap", GAP_B_MS)]
            rest = int(rest_ms)
        elif item == "d0":
            # f1g step 0c DRIVE probe (RULING cs-race-f1g-0b-1022):
            # every mention word once as a 50 ms encode-format word
            # and once as a 150 ms pulse; trace off (W_MM 0, W_IM 0);
            # then the 500 ms cue report at CUE_PA (cue phase below).
            sched = []
            for _w in MENTION_WORDS:
                sched += [("pulse", _w, 50), ("gap", 100)]
            for _w in MENTION_WORDS:
                sched += [("pulse", _w, MENTION_MS), ("gap", 100)]
            rest = 2 * CUE_MS
        elif item in ("c1", "c4"):
            sched = [("pulse", word, MENTION_MS), ("gap", GAP_B_MS)]
            rest = int(rest_ms)
        else:
            raise ValueError("unknown calibration item %r" % (item,))
        pre = 0
        for ent in sched:
            pre += int(ent[-1])
        T = pre + rest + HOLD_MS
        # measure windows (absolute ms), from the frozen schedule:
        # c1/c4 = [pulse_end, pulse_end + 650)  (GAP_B + CUE_MS)
        # c3    = [Y_on + 300, Y_on + 650)      (exactly the cue window
        #                                         after the trailing gap)
        # c2b   = [0, 2000)
        if item in ("c1", "c4"):
            m0, m1 = MENTION_MS, MENTION_MS + GAP_B_MS + rest
        elif item == "c3":
            y_on = MENTION_MS + GAP_B_MS
            m0, m1 = y_on + 300, y_on + 650
        else:
            m0, m1 = 0, pre + rest
        n_w = len(MENTION_WORDS)
        ment_ms = np.zeros((n_w, T), dtype=np.int32)
        eng_ms = np.zeros((K, T), dtype=np.int32)
        slot_ms = np.zeros((K, T), dtype=np.int32)
        # f1h item C6: per-ms min / max (over the slot's cells) of the
        # apical voltage; allocated only with the intrinsic trace.
        vap_min = vap_max = None
        if self._ap_on:
            vap_min = np.full((n_w, T), np.nan, dtype=np.float32)
            vap_max = np.full((n_w, T), np.nan, dtype=np.float32)

        def zeros():
            return np.zeros(self._n, dtype=np.float32)

        def step_with(cur):
            if self._ap_on:
                self._set_apical(cur)
            b.cp_external_input_current[:] = xp.asarray(
                cur.astype(np.float32))
            b._run_one_simulation_step()

        def _rec(t_ix):
            f = np.asarray(to_host(b.cp_firing_states)).astype(bool)
            for wi in range(n_w):
                c_w = int(f[self.ment[wi]].sum())
                if c_w:
                    ment_ms[wi, t_ix] = c_w
            for k in range(K):
                e_k = int(f[self.eng[k]].sum())
                if e_k:
                    eng_ms[k, t_ix] = e_k
                s_k = int(f[self.lay["slot"][k]].sum())
                if s_k:
                    slot_ms[k, t_ix] = s_k
            if vap_min is not None:
                va = np.asarray(to_host(b.cp_v_apical))
                for wi in range(n_w):
                    sv = va[self.ment[wi]]
                    vap_min[wi, t_ix] = float(sv.min())
                    vap_max[wi, t_ix] = float(sv.max())

        t = 0
        pulse_events = []
        for ent in sched:
            if ent[0] == "gap":
                for _ in range(int(ent[1])):
                    step_with(zeros())
                    _rec(t)
                    t += 1
            else:
                m_idx = self.ment[MENTION_WORDS.index(ent[1])]
                ev_cells = np.zeros(m_idx.size, dtype=np.int64)
                ev_t0 = t
                for _ in range(int(ent[2])):
                    cur = zeros()
                    cur[m_idx] = float(MENT_PA)
                    step_with(cur)
                    f_p = np.asarray(
                        to_host(b.cp_firing_states)).astype(bool)
                    ev_cells += f_p[m_idx].astype(np.int64)
                    _rec(t)
                    t += 1
                # C0 (RULING cs-race-f1g-0-1022 C): per-cell spike
                # count over this pulse's own window
                pulse_events.append(
                    {"word": str(ent[1]), "ms": int(ent[2]),
                     "t0": int(ev_t0), "t1": int(t - 1),
                     "n_cells": int(m_idx.size),
                     "total_spikes": int(ev_cells.sum()),
                     "min_cell_spikes": int(ev_cells.min()),
                     "per_cell_spikes":
                         [int(x) for x in ev_cells.tolist()]})
        # cue-onset phase with NO cue: OU noise on E cells as in
        # race().  item d0 (f1g step 0c) additionally reports the 500
        # ms cue engagement: every cue slot driven at CUE_PA, per-cell
        # spike counts over the cue window, reported only (the cue
        # stays 60 pA — RULING cs-race-f1g-0b-1022 (1)).
        b.set_transmission_gate(GATE_COMMIT_IN, 1.0)
        cue_cell_counts = []
        cue_all = np.zeros(0, dtype=np.int64)
        if item == "d0":
            cue_cell_counts = [
                np.zeros(len(self.cue[wi]), dtype=np.int64)
                for wi in range(len(CUE_WORDS))]
            cue_all = (np.concatenate(self.cue) if len(self.cue)
                       else cue_all)
        while t < T:
            cur = zeros()
            if t < pre + rest:
                eta += a_ou * (-eta) + s_ou * rng.standard_normal(nE)
                cur[:nE] += eta.astype(np.float32)
            drive_cue = bool(item == "d0" and cue_all.size and
                             pre <= t < pre + CUE_MS)
            if drive_cue:
                cur[cue_all] += float(CUE_PA)
            step_with(cur)
            if drive_cue:
                f_c = np.asarray(
                    to_host(b.cp_firing_states)).astype(bool)
                for wi in range(len(CUE_WORDS)):
                    cue_cell_counts[wi] += f_c[
                        self.cue[wi]].astype(np.int64)
            _rec(t)
            t += 1
        b.cp_external_input_current[:] = 0.0

        def _hz(counts, n_cells, ms_len):
            return round(float(counts) / (float(n_cells)
                                          * float(ms_len) / 1000.0), 4)

        def _first_burst(arr):
            if arr.size < 5:
                return None
            w5 = np.convolve(np.asarray(arr, dtype=np.int64),
                             np.ones(5, dtype=np.int64), mode="valid")
            idx = np.flatnonzero(w5 >= 3)
            return int(idx[0] + 4 + m0) if idx.size else None

        seg = ment_ms[:, m0:m1]
        ms_len = m1 - m0
        rates = {}
        for wi, wname in enumerate(MENTION_WORDS):
            rates[wname] = _hz(int(seg[wi].sum()), W_SLOT, ms_len)
        rec = {"item": item, "seed": int(seed),
               "pulse_events": pulse_events,
               "ment_pa": float(MENT_PA),
               "window_ms": [int(m0), int(m1)], "window_len_ms": int(ms_len),
               "word_rates_hz": rates,
               "pulsed_rate_hz": (rates.get(word) if word else None),
               "unpulsed_max_hz": (
                   max(v for wname, v in rates.items() if wname != word)
                   if word else max(rates.values())),
               "word1_rate_hz": (rates.get(word) if word else None),
               "word2_rate_hz": (rates.get(word2) if word2 else None),
               "max_word_hz": max(rates.values()),
               "all_engram_spikes": [
                   int(eng_ms[k, m0:m1].sum()) for k in range(K)],
               "commit_slot_spikes": [
                   int(slot_ms[k, m0:m1].sum()) for k in range(K)],
               "ments_spikes": [int(seg[wi].sum()) for wi in range(n_w)],
               "pulse_phase_spikes": (
                   int(ment_ms[MENTION_WORDS.index(word),
                               0:MENTION_MS].sum())
                   if word else None),
               "trace_flags": {"W_MM": float(p.get("W_MM", 0.0)),
                               "W_MI": float(p.get("W_MI", 0.0)),
                               "W_IM": float(p.get("W_IM", 0.0)),
                               "NIM": int(p.get("NIM", 8))}}
        if self._ap_on:
            rec["v_apical"] = {
                "v_hold": float(GAP4_GO["coincidence_plateau_v_hold"]),
                "window_ms": [int(m0), int(m1)],
                "pulsed_words": [w for w in (word, word2) if w],
                "window": {
                    MENTION_WORDS[wi]: {
                        "min": round(float(vap_min[wi, m0:m1].min()), 4),
                        "max": round(float(vap_max[wi, m0:m1].max()), 4)}
                    for wi in range(n_w)},
                "run_max": {
                    MENTION_WORDS[wi]: round(float(vap_max[wi].max()), 4)
                    for wi in range(n_w)},
                "end": {
                    MENTION_WORDS[wi]: round(float(vap_min[wi, T - 1]), 4)
                    for wi in range(n_w)}}
        if item == "d0" and cue_cell_counts:
            rec["cue_events"] = [
                {"word": str(CUE_WORDS[wi]),
                 "ms": int(CUE_MS),
                 "n_cells": int(arr.size),
                 "total_spikes": int(arr.sum()),
                 "min_cell_spikes": int(arr.min()),
                 "per_cell_spikes": [int(x) for x in arr.tolist()]}
                for wi, arr in enumerate(cue_cell_counts)]
            rec["drive_word_ms"] = 50
            rec["cue_pa"] = float(CUE_PA)
        if item in ("c1", "c4") and word:
            pk = int(self.slot_of(word))
            rec["primed_k"] = pk
            rec["primed_engram_spikes"] = int(eng_ms[pk, m0:m1].sum())
            rec["burst_ms"] = _first_burst(eng_ms[pk, m0:m1])
            rec["latch_ms"] = _first_burst(slot_ms[pk, m0:m1])
        return rec

    def unitary_epsp(self, cue_word="lynx"):
        """f1d step 1c, once per seed: force ONE cue cell to spike from
        the quiescent snapshot; peak dV over 30 ms averaged over its
        engram targets (mV)."""
        from sim.backend import to_host
        from research.runners._gnw_rung1_ignition_curve_derisk import \
            _reset_quiescent
        b = self._bridge
        _reset_quiescent(b, self._xp, self._snap)
        self._reset_apical()
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

    def rheobase_probe(self):
        """f1e step 1a, once per seed: the smallest step current (pA)
        that fires each engram cell, read from the quiescent snapshot.
        Declared method (fixed before any result): a 5 pA coarse scan
        5..100, then an exact 1 pA back-off inside (pa-5, pa); a 200 ms
        step per trial; >= 1 spike in the window counts as firing; a
        fresh snapshot before every trial; None = never fired by the
        100 pA cap. Per engram k the record carries every cell value
        and the engram rheobase = min over its cells. Measurement
        probe only, never a mechanism."""
        from sim.backend import to_host
        from research.runners._gnw_rung1_ignition_curve_derisk import \
            _reset_quiescent
        b = self._bridge

        def fires(cell, pa):
            _reset_quiescent(b, self._xp, self._snap)
            self._reset_apical()
            cur = np.zeros(self._n, dtype=np.float32)
            cur[cell] = float(pa)
            hit = False
            for _ in range(200):
                b.cp_external_input_current[:] = self._xp.asarray(cur)
                b._run_one_simulation_step()
                if bool(np.asarray(
                        to_host(b.cp_firing_states)).astype(bool)[cell]):
                    hit = True
                    break
            b.cp_external_input_current[:] = 0.0
            return hit

        per_k = {}
        for k in range(self.K):
            cells = [int(c) for c in np.asarray(self.eng[k]).ravel()]
            vals = []
            for cell in cells:
                found = None
                for pa in range(5, 105, 5):
                    if fires(cell, float(pa)):
                        for exact in range(max(1, pa - 5), pa + 1):
                            if fires(cell, float(exact)):
                                found = exact
                                break
                        break
                vals.append(found)
            known = [v for v in vals if v is not None]
            per_k[str(k)] = {
                "cell_rheobases_pA": vals,
                "engram_rheobase_pA": (min(known) if known else None),
                "cells": len(vals),
            }
        return {
            "per_engram": per_k,
            "method": ("smallest step current firing the cell from the "
                       "quiescent snapshot: 5 pA coarse 5..100 then "
                       "exact 1 pA in (pa-5, pa); 200 ms step; >=1 "
                       "spike counts; fresh snapshot per trial; engram "
                       "value = min over its cells; None = no fire "
                       "<=100 pA"),
            "step_window_ms": 200,
            "coarse_step_pA": 5,
            "cap_pA": 100,
            "snapshot": "quiescent (pre-race) _snap",
        }


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
    # f1e step 1a: engram rheobase, once per seed (declared probe)
    out["rheobase"] = circ.rheobase_probe()

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
                       "mention_pulse_spikes":
                           r.get("mention_pulse_spikes"),
                       "mention_pulse_pa": r.get("mention_pulse_pa"),
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
    # RULING cs-race-f1g-0-1022 B: a draw with 0 mention-slot spikes
    # is UNDEFINED (carrier unengaged), never a null or a pass
    b_def = [r for r in b_rows
             if r.get("mention_pulse_spikes") is None
             or int(r["mention_pulse_spikes"]) > 0]
    out["clauses"]["b"] = dict(
        pass_bar(b_def, lambda r: content(r["winner"]) == "vole"),
        winners=[content(r["winner"]) for r in b_def],
        n_draws=len(b_rows),
        n_undefined_no_mention_spikes=int(len(b_rows) - len(b_def)),
        note="0 mention spikes on a draw = UNDEFINED "
             "(RULING cs-race-f1g-0-1022 B)")
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
           "A3n": {"cue_n": 8, "cue_noise_sd": 15.0, "cue_noise_tau": 5.0},
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
        an = RaceCircuit(3, seed, dict(b0, CUE_NOISE_SD=15.0), grid)
        an2 = RaceCircuit(3, seed, dict(b0, CUE_NOISE_SD=15.0), grid)
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
                             "a3n": {"cue_n": 8, "cue_noise_sd": 15.0,
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


def run_calibration(base, grid, seeds, args):
    """f1g step 0 (packet cs-race-f1g): calibrate the persistent word
    trace with NO race. Frozen: grid row 3, W_GIS 30, seeds 41-48,
    items C1-C5, selection = smallest W_MM then smallest W_IM among
    cells passing C1-C4 on 8/8 seeds. One invocation = one cell."""
    import statistics
    t0 = time.time()
    rec = {"probe": "cs_race_f1", "mode": "f1g_step0_calibration",
           "grid_row": args.grid_row, "grid": dict(grid),
           "seeds": [int(s) for s in seeds], "backend": "numpy",
           "device": "cpu",
           "flags": {"W_MM": float(base.get("W_MM", 0.0)),
                     "W_MI": float(base.get("W_MI", 0.0)),
                     "W_IM": float(base.get("W_IM", 0.0)),
                     "NIM": int(base.get("NIM", 8)),
                     "encode_words": bool(base.get("ENCODE_WORDS", 0))},
           "record_label": {"latch_exclusion_w_gis":
                            float(base.get("W_GIS", 0.0))},
           "per_seed": [],
           "preconditions": {
               "calibration_seeds":
                   "41-48 only; no DEV/CONFIRM arm has run on them",
               "grid_row": 3,
               "w_gis": float(base.get("W_GIS", 0.0)),
               "no_race": "cue withheld in every calibration run",
               "frozen_selection":
                   "smallest W_MM, then smallest W_IM among cells "
                   "passing C1-C4 on 8/8 seeds",
               "thresholds_frozen_in_packet": {
                   "C1_hz": [2.0, 20.0], "quiet_hz": 1.0,
                   "STOP_SMOKE_pa": 2.0, "flip_scale_pa": [4.0, 6.0],
                   "C0_min_spikes_per_cell":
                       "1 (RULING cs-race-f1g-0-1022 C)"}},
             "ment_pa": float(MENT_PA)}
    if base.get("TRACE_INTRINSIC"):
        # f1h: record the intrinsic-trace flags and the frozen gap4 GO
        # values (only when ON, so a flags-off record is unchanged).
        rec["flags"].update({
            "trace": "intrinsic",
            "soma_g": float(base.get("SOMA_G", 0.0)),
            "apical_pa": float(base.get("APICAL_PA", 0.0)),
            "gap4_go_frozen": dict(GAP4_GO),
            "gap4_go_pulse_pa": GAP4_GO_PULSE_PA})
    if getattr(args, "drive_probe", False):
        # f1g step 0c DRIVE probe (RULING cs-race-f1g-0b-1022): the
        # engagement criterion is fixed here, before any trace or
        # race reading: every cell of every mention slot fires >=1
        # spike within a 50 ms word AND >=1 spike in a 150 ms pulse,
        # on 8/8 seeds. The cue report is read-only (cue stays 60).
        rec["mode"] = "f1g_step0c_drive"
        rec["drive"] = {
            "criterion": ("every cell of every mention slot fires "
                          ">=1 spike within a 50 ms word AND >=1 "
                          "spike in a 150 ms pulse, on 8/8 seeds "
                          "(RULING cs-race-f1g-0b-1022 (1))"),
            "cue_pa_reported_only": float(CUE_PA),
            "trace_off": {"W_MM": 0.0, "W_IM": 0.0}}
        n_both = 0
        for seed in seeds:
            circ = RaceCircuit(3, seed, dict(base), grid)
            rd = circ.cal(seed=seed, item="d0")
            evs = list(rd.get("pulse_events") or [])
            w50 = [e for e in evs if int(e["ms"]) == 50]
            w150 = [e for e in evs if int(e["ms"]) == 150]
            p50 = bool(w50) and all(
                int(e["min_cell_spikes"]) >= 1 for e in w50)
            p150 = bool(w150) and all(
                int(e["min_cell_spikes"]) >= 1 for e in w150)
            both = bool(p50 and p150)
            n_both += int(both)
            rec["per_seed"].append({
                "seed": int(seed),
                "pass_50ms": p50, "pass_150ms": p150,
                "pass_both": both,
                "n_50ms": len(w50), "n_150ms": len(w150),
                "min_50ms": (min(int(e["min_cell_spikes"])
                                 for e in w50) if w50 else None),
                "min_150ms": (min(int(e["min_cell_spikes"])
                                  for e in w150) if w150 else None),
                "zero_spike_50ms": sum(
                    1 for e in w50
                    if int(e["min_cell_spikes"]) < 1),
                "zero_spike_150ms": sum(
                    1 for e in w150
                    if int(e["min_cell_spikes"]) < 1),
                "pulse_events": [{k: e[k] for k in
                                  ("word", "ms", "n_cells",
                                   "total_spikes",
                                   "min_cell_spikes",
                                   "per_cell_spikes")}
                                 for e in evs],
                "cue_events": rd.get("cue_events")})
            print("f1g-0c drive seed %d pa=%g: 50ms_pass=%s "
                  "(zero=%d/%d) 150ms_pass=%s (zero=%d/%d) both=%s"
                  % (seed, MENT_PA, p50,
                     sum(1 for e in w50
                         if int(e["min_cell_spikes"]) < 1),
                     len(w50), p150,
                     sum(1 for e in w150
                         if int(e["min_cell_spikes"]) < 1),
                     len(w150), both), flush=True)
        rec["seeds_passing_both"] = int(n_both)
        rec["cell_pass_drive"] = bool(
            n_both == len(rec["per_seed"]) == len(seeds))
        rec["stop_preview"] = {
            "STOP_DRIVE": bool(not rec["cell_pass_drive"]),
            "note": ("cue report read-only; cue stays 60 "
                     "(RULING cs-race-f1g-0b-1022 (1))")}
        if not args.out:
            args.out = ("research/findings/raw/_cs_race/f1g/"
                        "drive_paq%g_wmm%g_wim%g_seeds%s.json"
                        % (MENT_PA, base.get("W_MM", 0.0),
                           base.get("W_IM", 0.0),
                           "-".join(str(s) for s in seeds)))
        out_path = (args.out if os.path.isabs(args.out)
                    else os.path.join(REPO, args.out))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(rec, fh, indent=2, sort_keys=True, default=str)
        from research.runners import declare_output
        declare_output(out_path)
        print("f1g-0c drive pa=%g trace_off: passing_both=%d/%d "
              "STOP_DRIVE=%s wall=%.1fs -> %s"
              % (MENT_PA, n_both, len(rec["per_seed"]),
                 rec["stop_preview"]["STOP_DRIVE"],
                 time.time() - t0, out_path), flush=True)
        return 0
    c_lo, c_hi, quiet = 2.0, 20.0, 1.0
    pairs = (("vole", "frog"), ("frog", "vole"), ("lynx", "grass"))
    for seed in seeds:
        circ = RaceCircuit(3, seed, dict(base), grid)
        bcfg = getattr(circ._bridge, "core_config", None)
        srec = {"seed": int(seed),
                "config_readback": {
                    "enable_nmda_recurrent":
                        bool(getattr(bcfg, "enable_nmda_recurrent",
                                     False)),
                    "enable_short_term_plasticity":
                        bool(getattr(bcfg, "enable_short_term_plasticity",
                                     False)),
                    "enable_parameter_heterogeneity":
                        bool(getattr(bcfg,
                                     "enable_parameter_heterogeneity",
                                     False)),
                    "w_gis_read_back":
                        float(base.get("W_GIS", 0.0))}}
        if base.get("TRACE_INTRINSIC"):
            for _f in ("enable_bdsp", "bdsp_apical_bistable",
                       "bdsp_apical_couples_soma", "bdsp_apical_soma_g",
                       "coincidence_plateau_self_regen",
                       "coincidence_plateau_v_hold", "apical_kir_g",
                       "bdsp_learning_rate"):
                srec["config_readback"][_f] = getattr(bcfg, _f, None)
        r1 = circ.cal(seed=seed, item="c1", word="vole")
        srec["c1"] = r1
        r2 = circ.cal(seed=seed, item="c2b")
        srec["c2b"] = r2
        c3s = [circ.cal(seed=seed, item="c3", word=x, word2=y)
               for x, y in pairs]
        srec["c3"] = c3s
        # C5: the clause-(a)/(b) protocols, encode words on, cue
        # withheld; delta = held engram - rival engram total excitatory
        # synaptic current in [100, 200] ms of the cue window, beside
        # the +4/+6 pA flip scale.
        ra = circ.race(seed=int(seed) * 1000003 + 51, cue_words=[],
                       encode=("vole", "frog"), encode_words=True,
                       cal_window=(100, 200),
                       cal_held_k=circ.slot_of("frog"),
                       cal_rival_k=circ.slot_of("vole"))
        rb = circ.race(seed=int(seed) * 1000003 + 52, cue_words=[],
                       encode=("vole", "frog"), mention_words=["vole"],
                       encode_words=True, cal_window=(100, 200),
                       cal_held_k=circ.slot_of("vole"),
                       cal_rival_k=circ.slot_of("frog"))
        ca = ra.get("cal_window") or {}
        cb = rb.get("cal_window") or {}
        srec["c5"] = {"clause_a_delta_pa": ca.get("delta_pa"),
                      "clause_b_delta_pa": cb.get("delta_pa"),
                      "clause_a_per_k": ca.get("total_exc_current_pa_per_k"),
                      "clause_b_per_k": cb.get("total_exc_current_pa_per_k"),
                      "window_ms": ca.get("ms"),
                      "encode_words_seen_by_race":
                          bool(ra.get("encode_words"))}
        c1_ok = bool(c_lo <= float(r1["pulsed_rate_hz"]) <= c_hi)
        c2a_ok = bool(float(r1["unpulsed_max_hz"]) < quiet)
        c2b_ok = bool(float(r2["max_word_hz"]) < quiet)
        c3_ok = all(bool(c_lo <= float(c["word2_rate_hz"]) <= c_hi and
                         float(c["word1_rate_hz"]) < quiet)
                    for c in c3s)
        c4_ok = bool(r1.get("burst_ms") is None and
                     r1.get("latch_ms") is None)
        # C0 (RULING cs-race-f1g-0-1022 C): every pulsed slot fires
        # >= 1 spike per cell during its 150 ms pulse, on 8/8 seeds.
        # Kept OUT of pass{} so the frozen C1-C4 selection is intact.
        evs = list(r1.get("pulse_events") or []) + [
            e for cr in c3s for e in (cr.get("pulse_events") or [])]
        c0_ok = bool(evs) and all(
            int(e["min_cell_spikes"]) >= 1 for e in evs)
        srec["c0"] = {"pass": c0_ok, "n_pulses": len(evs),
                      "ment_pa": float(MENT_PA),
                      "pulses": [{"word": e["word"],
                                  "n_cells": e["n_cells"],
                                  "total_spikes": e["total_spikes"],
                                  "min_cell_spikes":
                                      e["min_cell_spikes"]}
                                 for e in evs]}
        if base.get("TRACE_INTRINSIC"):
            # C6 (packet cs-race-f1h, engagement): the pulsed slot's
            # v_apical stays above v_hold across the 650 ms window and
            # every unpulsed slot's stays below it. A seed whose pulse
            # never took the apical above v_hold is UNDEFINED, never a
            # FAIL. Raw numbers kept so the scorer can re-derive. Kept
            # OUT of pass{} (C0 precedent) so the frozen C1-C4
            # selection logic is untouched.
            vh = float(GAP4_GO["coincidence_plateau_v_hold"])
            va1 = r1["v_apical"]
            pw = va1["window"]["vole"]
            un_max = max(va1["window"][w]["max"] for w in MENTION_WORDS
                         if w != "vole")
            reached = bool(va1["run_max"]["vole"] > vh)
            above = bool(pw["min"] > vh)
            below = bool(un_max < vh)
            srec["c6"] = {
                "v_hold": vh, "pulsed_word": "vole",
                "pulsed_window_min": pw["min"],
                "pulsed_window_max": pw["max"],
                "pulsed_run_max": va1["run_max"]["vole"],
                "unpulsed_window_max": un_max,
                "reached_v_hold_during_run": reached,
                "pulsed_above_hold_all_window": above,
                "unpulsed_below_hold": below,
                "status": ("UNDEFINED" if not reached else
                           ("PASS" if (above and below) else "FAIL"))}
            vap2 = r2["v_apical"]["window"]
            srec["c6"]["c2b_all_slots_below_hold"] = bool(
                all(v["max"] < vh for v in vap2.values()))
            srec["c3_apical"] = [
                {"older": x, "newer": y,
                 "older_window_max": c["v_apical"]["window"][x]["max"],
                 "newer_window_min": c["v_apical"]["window"][y]["min"],
                 "older_plateau_ended": bool(
                     c["v_apical"]["window"][x]["max"] < vh),
                 "newer_plateau_held": bool(
                     c["v_apical"]["window"][y]["min"] > vh)}
                for (x, y), c in zip(pairs, c3s)]
        srec["pass"] = {"c1": c1_ok,
                        "c2": bool(c2a_ok and c2b_ok),
                        "c3": c3_ok, "c4": c4_ok}
        rec["per_seed"].append(srec)
        print("f1g-cal seed %d: c1=%sHz ok=%s c2=%s c3=%s c4=%s "
              "c5(a)=%s c5(b)=%s"
              % (seed, r1["pulsed_rate_hz"], c1_ok, srec["pass"]["c2"],
                 c3_ok, c4_ok, srec["c5"]["clause_a_delta_pa"],
                 srec["c5"]["clause_b_delta_pa"]), flush=True)
    n_ok = sum(1 for s in rec["per_seed"] if all(s["pass"].values()))
    rec["seeds_passing_c1_c4"] = int(n_ok)
    rec["cell_pass_c1_c4"] = bool(n_ok == len(rec["per_seed"]) ==
                                  len(seeds))
    da = [float(s["c5"]["clause_a_delta_pa"]) for s in rec["per_seed"]
          if s["c5"]["clause_a_delta_pa"] is not None]
    db = [float(s["c5"]["clause_b_delta_pa"]) for s in rec["per_seed"]
          if s["c5"]["clause_b_delta_pa"] is not None]
    med_a = round(statistics.median(da), 4) if da else None
    med_b = round(statistics.median(db), 4) if db else None
    rec["c5_median_pa"] = {"clause_a": med_a, "clause_b": med_b}
    c0_seeds = all(bool(s["c0"]["pass"]) for s in rec["per_seed"])
    if base.get("TRACE_INTRINSIC"):
        _st = [s["c6"]["status"] for s in rec["per_seed"]]
        rec["c6_summary"] = {
            "PASS": _st.count("PASS"), "FAIL": _st.count("FAIL"),
            "UNDEFINED": _st.count("UNDEFINED"), "n": len(_st)}
    rec["stop_preview"] = {
        "STOP_CAL": bool(not rec["cell_pass_c1_c4"]),
        "STOP_SMOKE": bool(med_a is None or med_b is None or
                           med_a < 2.0 or med_b < 2.0),
        "STOP_C0": bool(not c0_seeds),
        "note": "STOP-ID is judged by the separate identity run"}
    if not args.out:
        args.out = ("research/findings/raw/_cs_race/f1g/"
                    "cal_wmm%g_wim%g_nim%d_seeds%s.json"
                    % (base.get("W_MM", 0.0), base.get("W_IM", 0.0),
                       int(base.get("NIM", 8)),
                       "-".join(str(s) for s in seeds)))
    out_path = (args.out if os.path.isabs(args.out)
                else os.path.join(REPO, args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True, default=str)
    from research.runners import declare_output
    declare_output(out_path)
    print("f1g-cal cell W_MM=%s W_IM=%s: pass_c1_c4=%s/%d seeds "
          "c5_medians=(%s, %s) pa STOP=%s wall=%.1fs -> %s"
          % (base.get("W_MM", 0.0), base.get("W_IM", 0.0),
             n_ok, len(rec["per_seed"]), med_a, med_b,
             rec["stop_preview"], time.time() - t0, out_path),
          flush=True)
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
    ap.add_argument("--w-mm", type=float, default=None,
                    help="f1g: mention-slot slow-NMDA self-excitation "
                         "(0 = off; the persistent word trace)")
    ap.add_argument("--w-mi", type=float, default=None,
                    help="f1g: mention slot -> shared inhibitory pool")
    ap.add_argument("--w-im", type=float, default=None,
                    help="f1g: shared inhibitory pool -> mention slots")
    ap.add_argument("--nim", type=int, default=None,
                    help="f1g: inhibitory pool cells (default 8)")
    ap.add_argument("--encode-words", action="store_true",
                    help="f1g: encode turns present their fact's words "
                         "to the mention layer in heard order")
    ap.add_argument("--calibrate", action="store_true",
                    help="f1g step 0: trace calibration with no race "
                         "(items C1-C5; seeds 41-48)")
    ap.add_argument("--ment-pa", type=float, default=None,
                    help="context/encode mention current pA; f1g arms "
                         "pass 60 = CUE_PA (RULING cs-race-f1g-0-1022 "
                         "B); default 40 keeps identity exact")
    ap.add_argument("--trace", choices=("intrinsic",), default=None,
                    help="f1h: the word trace is a held BDSP apical "
                         "plateau in the mention-slot cells (bistable + "
                         "soma-coupled, gap4 GO values frozen; W_MM = 0); "
                         "default off = the f1g circuit")
    ap.add_argument("--soma-g", type=float, default=None,
                    help="f1h: bdsp_apical_soma_g (apical->soma coupling); "
                         "needs --trace intrinsic (grid 0.002/0.004/0.008)")
    ap.add_argument("--apical-pulse", type=float, default=None,
                    help="f1h: apical drive pA written to every mention "
                         "cell that gets somatic hearing current, for the "
                         "word's own duration (gap4 GO value %g); needs "
                         "--trace intrinsic; 0/absent = no pulse (the "
                         "LESION arm)" % GAP4_GO_PULSE_PA)
    ap.add_argument("--drive-probe", action="store_true",
                    help="f1g step 0c: DRIVE engagement probe "
                         "(item d0, trace off; RULING "
                         "cs-race-f1g-0b-1022)")
    args = ap.parse_args()
    global _F1_GRID_ROW, MENT_PA
    _F1_GRID_ROW = dict(F1_GRID[args.grid_row])
    if args.ment_pa is not None:
        MENT_PA = float(args.ment_pa)
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
    if args.w_mm is not None:
        base["W_MM"] = float(args.w_mm)
    if args.w_mi is not None:
        base["W_MI"] = float(args.w_mi)
    if args.w_im is not None:
        base["W_IM"] = float(args.w_im)
    if args.nim is not None:
        base["NIM"] = int(args.nim)
    if args.encode_words:
        base["ENCODE_WORDS"] = 1
    if args.trace == "intrinsic":
        if args.soma_g is None:
            ap.error("--trace intrinsic requires --soma-g")
        if args.w_mm:
            ap.error("--trace intrinsic is W_MM = 0 (no recurrence); "
                     "do not pass --w-mm")
        base["TRACE_INTRINSIC"] = 1
        base["SOMA_G"] = float(args.soma_g)
        base["APICAL_PA"] = float(args.apical_pulse or 0.0)
    elif args.soma_g is not None or args.apical_pulse is not None:
        ap.error("--soma-g / --apical-pulse need --trace intrinsic")
    if args.identity_cue_n:
        return identity_cue_n(base, _F1_GRID_ROW, seeds,
                              int(args.cue_n or 48),
                              float(args.w_cue or 7.5), args.draws,
                              args.out)
    if args.calibrate or args.drive_probe:
        return run_calibration(base, _F1_GRID_ROW, seeds, args)
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
           "ment_pa": float(MENT_PA),
           "seeds": seeds, "backend": "numpy", "device": "cpu",
           "KILL_CRITERIA": KILL_CRITERIA,
           "row_all_pass_8_seeds": bool(all_seeds_all_clauses
                                    and not args.controls_only),
        "controls_only": bool(args.controls_only),
        "het_transplant_from_A0": bool(het_transplant),
        "record_label": {"latch_exclusion_w_gis":
                         float(base.get("W_GIS", 0.0))},
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
    if base.get("TRACE_INTRINSIC"):
        out["instrument_flags"].update({
            "trace": "intrinsic", "soma_g": float(base["SOMA_G"]),
            "apical_pa": float(base["APICAL_PA"]),
            "gap4_go_frozen": dict(GAP4_GO)})
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
