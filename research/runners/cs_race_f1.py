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
             W_EE=40.0, W_EI=40.0, W_IE=40.0, W_XG=30.0, W_GG=20.0,
             W_GGI=60.0, W_GIG=60.0, W_GS=12.0, W_SS=60.0, W_SO=20.0,
             W_OG=80.0, W_OE=40.0, NOISE_SD=25.0, NOISE_TAU=50.0,
             BIAS_PA=42.0, REST_STEPS=CUE_MS, HOLD_STEPS=HOLD_MS,
             READ_STEPS=30, READ_FRAC=0.03)
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
        o_ment = o_cue + len(CUE_WORDS) * W_SLOT
        n = o_ment + len(MENTION_WORDS) * W_SLOT
        self.cue = [np.arange(o_cue + i * W_SLOT, o_cue + (i + 1) * W_SLOT,
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
        bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                  runtime_state=RuntimeState(), gpu_config=GPUConfig())
        bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        bridge._initialize_simulation_data(called_from_playback_init=False)
        plan = {}
        for k in range(K):
            ee = SC.dense(self.eng[k], self.eng[k], p["W_EE"], conn_type="E_TO_E")
            ee.pop("stp_disabled", None)          # engram recurrence KEEPS STP
            plan["ee_%d" % k] = ee
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
                    self.cue[i], self.eng[k], W_CUE, conn_type="E_TO_E")
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
        b.set_transmission_gate(GATE_COMMIT_IN, 1.0)
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
        cue_idx = np.concatenate([self.cue[CUE_WORDS.index(w)]
                                  for w in cue_words if w in CUE_WORDS]) \
            if cue_words else np.zeros(0, dtype=np.int64)
        rest = int(p["REST_STEPS"])
        hold = int(p["HOLD_STEPS"])
        read = int(p["READ_STEPS"])
        for t in range(rest + hold):
            cur = zeros()
            if t < rest:
                eta += a_ou * (-eta) + s_ou * rng.standard_normal(nE)
                cur[:nE] += eta.astype(np.float32)
                if cue_idx.size:
                    cur[cue_idx] += float(CUE_PA)
                if steer_pa and (flip_step is None or t >= flip_step):
                    tgt = (np.concatenate([self.eng[k] for k in range(K)])
                           if nonselective else
                           self.eng[int(steer_target or 0)])
                    cur[tgt] += float(steer_pa)
            step_with(cur)
            f = np.asarray(to_host(b.cp_firing_states)).astype(bool)
            for k in range(K):
                eng_counts[k] += int(f[self.eng[k]].sum())
                gk = int(f[gate_all[k]].sum())
                gate_counts[k] += gk
                if gk and first_gate[k] is None:
                    first_gate[k] = t
            if natural_winner is None and sum(
                    1 for k in range(K) if first_gate[k] is not None) >= 1:
                natural_winner = int(np.argmin([first_gate[k] if
                                                first_gate[k] is not None
                                                else 10 ** 9 for k in range(K)]))
        b.cp_external_input_current[:] = 0.0
        slot_counts = np.zeros(K, dtype=np.int64)
        f = np.asarray(to_host(b.cp_firing_states)).astype(bool)
        for k in range(K):
            slot_counts[k] = int(f[self.lay["slot"][k]].sum())
        rd = SC.read_with_variants(slot_counts.tolist(), int(p["NS"]), read,
                                   float(p["READ_FRAC"]))
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
                "slot_counts": slot_counts.tolist()}


def run_clauses(circ, seed, draws, grid_tag):
    """All clauses (a)-(i) for ONE dev seed x `draws` noise draws."""
    out = {"seed": seed, "grid": grid_tag, "draws": draws,
           "KILL": KILL_CRITERIA, "clauses": {}}
    base_seed = seed * 1000003

    def winners(**kw):
        rows = []
        for d in range(draws):
            try:
                r = circ.race(seed=base_seed + d, **kw)
                rows.append(r)
            except Exception as e:                       # K2
                rows.append({"winner": "ERROR:%s" % type(e).__name__})
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

    # (a) A old, B new, same cue -> B (frog) wins >= 8/10
    a_rows = winners(cue_words=list(CUE_OF[1]), encode=("vole", "frog"))
    out["clauses"]["a"] = dict(
        pass_bar(a_rows, lambda r: content(r["winner"]) == "frog"),
        winners=[content(r["winner"]) for r in a_rows])
    # (b) mention of A's patient after both writes -> A (vole) >= 8/10
    b_rows = winners(cue_words=list(CUE_OF[1]), encode=("vole", "frog"),
                     mention_words=["vole"])
    out["clauses"]["b"] = dict(
        pass_bar(b_rows, lambda r: content(r["winner"]) == "vole"),
        winners=[content(r["winner"]) for r in b_rows])
    # (c) one-role match with maximal priming -> 0 latched draws
    c_rows = winners(cue_words=["chase"], encode=("vole", "frog"),
                     mention_words=["lynx", "chase", "vole"])
    out["clauses"]["c"] = dict(
        pass_bar(c_rows, lambda r: r["winner"] is None, bar=draws),
        winners=[content(r["winner"]) for r in c_rows])
    # (d) cue matching no engram -> 0 commits
    d_rows = winners(cue_words=list(UNMATCHED_CUE), encode=("vole", "frog"))
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
    h_rows = winners(cue_words=list(CUE_OF[1]), encode=("ABsim",))
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
    l1 = winners(cue_words=list(CUE_OF[1]), encode=("vole", "frog"),
                 mention_words=["vole"], ment_gate=False)
    l1_out = [content(r["winner"]) for r in l1]
    l2 = winners(cue_words=list(CUE_OF[1]), encode=("vole", "frog"), excl_gate=False)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--grid-row", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    global _F1_GRID_ROW
    _F1_GRID_ROW = dict(F1_GRID[args.grid_row])
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    from tools.lab import assert_backend
    from tools.verdict import Verdict
    assert_backend("numpy", note="cs-race step 1 F1 feasibility probe")

    t0 = time.time()
    print("KILL CRITERIA (pre-registered):", flush=True)
    for line in KILL_CRITERIA:
        print("  " + line, flush=True)
    results = []
    for seed in seeds:
        circ = RaceCircuit(3, seed, dict(_BASE), _F1_GRID_ROW)
        res = run_clauses(circ, seed, args.draws,
                          "row%d U=%s tau_f=%s NE=%s"
                          % (args.grid_row, _F1_GRID_ROW["U"],
                             _F1_GRID_ROW["tau_f"], _F1_GRID_ROW["NE"]))
        results.append(res)
        passed = sorted(k for k, v in res["clauses"].items() if v.get("pass"))
        print("seed %d: %d/%d clauses pass (%s)"
              % (seed, len(passed), 9, ",".join(passed)), flush=True)

    all_seeds_all_clauses = all(
        all(v.get("pass") for v in r["clauses"].values()) for r in results)
    v = Verdict("cs-race step 1: F1 feasibility clause table")
    v.require("8 dev seeds ran", len(results) == 8,
              note="dev seeds only; gate seeds never run")
    v.require("every clause present on every seed",
              all(set(r["clauses"]) == set("abcdefghi") for r in results))
    v.require("no clause errored (K2)",
              all(all(r["clauses"][c].get("errors", 0) == 0
                      for c in r["clauses"]) for r in results))
    v.require("backend numpy", True)
    v.decide(all_seeds_all_clauses)

    out = {"probe": "cs_race_f1", "grid_row": args.grid_row,
           "grid": dict(_F1_GRID_ROW), "draws": args.draws,
           "seeds": seeds, "backend": "numpy", "device": "cpu",
           "KILL_CRITERIA": KILL_CRITERIA,
           "row_all_pass_8_seeds": all_seeds_all_clauses,
           "results": results, "Verdict": v._decided,
           "preconditions": {
               "env": "probe-local flags only; BRAIN_CS_RACE does not exist "
                      "yet (step 2); default-off world",
               "organ": "RaceCircuit extends ReplayCircuit; shared organ "
                        "file unedited",
               "dev_seeds": "1-8", "draws": args.draws,
               "kill_criteria_written_before_first_run": True},
           "wall_s": round(time.time() - t0, 1)}
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
          % (args.grid_row, all_seeds_all_clauses, out["wall_s"], out_path),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
