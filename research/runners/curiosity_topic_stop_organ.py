"""C3 TOPIC STOP (L) and TOPIC-LOCKED PHASIC DA (P): the per-conversation ask organ behind the production curiosity
follow-up. Default OFF; the flags and the handler wiring live in `webapp/curiosity_topic_stop_chat.py`. Packet:
research/coordination/c3-topic-stop-PACKET.md.

WHY (research/findings/2026-10-26-stage2-fail-diagnoses.md, C3). Every production ask decision is
`want_hz * clip(0.2, 3.0, 1 + 1.5*(DA - 0.5)) >= threshold` in host code, with DA a session-global host EMA that the
habituating carrier word drives to its floor within 4-6 probes (layer 1); and the handler has no per-topic stop
(layer 2). With the gain pinned, the per-topic stop is the only C3 failure left in both orders.

L, THE TOPIC STOP (reuse, not a new mechanism). Method 2 of the curiosity-lp de-risk
(`research/runners/_curiosity_lp_trace_derisk.py::CuriosityLPOrgan`, `no_ou=True`; dev read every bar 8/8 on dev seeds
1-8, provisional, research/coordination/c3-method2-devread-1018.md): each topic has its own slice of a `topic_aff`
input population (25 RS neurons) that projects onto ONE 80-neuron ASK pool through depression-dominant
Tsodyks-Markram synapses (U 0.35, tau_f 10 ms, recovery tau_d 12 s of substrate time). On an abstain the topic's
slice gets one burst at `novelty * WORD_DRIVE` (novelty = the production graded `topic_novelty` read), the ASK-pool
rate over the 28-step window is the want, and `want >= threshold` (the organ's own calibrated midpoint) decides.
Each presentation depletes only that slice's synapses, so the drive to ask about a topic falls with the number of
times it was presented and no other topic is touched. This is a COUNT of presentations ("by topic history"), NOT a
learning-progress estimate: a topic that is being learned is silenced the same way (the identifiability limit
named in 2026-08-02-laneB-curiosity-reserve-rescue-REFUTED). Every substrate constant is method 2's, frozen
(W_TOPIC_ASK 80 from the seed-9 design smoke). The one addition is the slice table's capacity rule (below).

  Background noise. Method 2 turns the OU background off for the WHOLE organ (`cfg.enable_ou_process = False`), and
  this organ keeps exactly that, because that is the validated configuration (the topic-afferent-only variant was
  never run). DECLARED PER-MECHANISM REALISM TRADE: biology's word-input afferents are quiet between mentions, but
  its ASK-pool neurons do have background synaptic noise; here neither has it. Kept because the OU process at its
  100 pA cortical std fired 167 topic-afferent spikes per inter-turn gap and depleted every slice by elapsed time
  (method 1 NO-GO, PLAN.md section 4.2), which made the trace a clock; the ASK-pool decision margin it gives up is
  large (~35 Hz want vs a ~18 Hz threshold on a fresh slice). The afferent-only variant is a named battery arm.

  Turn clock. Between turns the organ runs INTER_TURN_MS (400 ms) of silent substrate time per conversation turn
  (method 2's battery protocol, run lazily at the next read). Compressed clock, declared by method 2: ~2 s between
  turns against a ~1 min habituation recovery.

  Topic -> slice. First-seen allocation of the topic token the production `extract_topic` already returns (a fixed
  input code like the per-word phase code; no word list). Slice 0 is the calibration reference, never a topic.
  CAPACITY (host bookkeeping, declared): 11 topic slices per conversation; a 12th topic takes the least recently
  presented slice, whose synapses are reset to rest (x = 1, u = U). Never reached in the Stage-2 battery (8 topics).

  Not integrated (declared): method 2's DA-gated dishabituation on a teach turn (`answer()`), a host write gated by a
  spiking SNc read. It is reply-silent in this battery (the answered topic recalls and is never presented again;
  method 2's informational lesion_reset arm), and it needs a teach-turn hook; its rung is a sim/ DA -> STP-recovery
  target.

P, THE TOPIC-LOCKED PHASIC DA (replaces the session-global DA input to the crave gain when on; requires L, whose
organ hosts it). The organ's existing `reward_us -> snc` populations (method 2's DA read) are read on every abstain
with topic T, the SNc driven by
  * a TONIC afferent current = K_TONIC_PA * da_level, the session's existing host EMA DA level (the same value the
    host gain reads today; da_level 0.5 gives SNC_TONIC_PA = 220 pA, the SNc's neutral drive), and
  * a PHASIC afferent onto reward_us = G_PHASIC_PA * salience(T), salience(T) = T's normalized want on the shared
    ASK-pool read the production organ already makes (`salience_of`'s formula applied to `judge()`'s own read).
The SNc runs W_DA_WARMUP steps, its rate is read over the next DRIVE_STEPS + READ_STEPS steps (the P protocol
window), and the read becomes the DA that enters the crave gain: da_eff = DA_TONIC * snc_rate / snc_neutral_rate
(the neutral rate measured in the same window at da_level 0.5, no phasic; silence -> 0, neutral -> 0.5). The gain is
the EXISTING host formula, unchanged (`webapp/da_curiosity_drives_chat.da_crave_gain`: clip(0.2, 3.0,
1 + 1.5 (da_eff - 0.5))), applied to L's own want; `want * gain >= threshold` decides.
  DECLARED HOST ARITHMETIC THAT REMAINS (P method B): the multiplicative gain formula and its clip, the rate -> da_eff
  normalization, and the two linear input encodings (da_level -> tonic pA, salience -> phasic pA). What changes from
  today is the DA that enters it: a topic-locked spiking SNc read in which T's own salience is a phasic afferent, not
  the session-global EMA alone. The tonic afferent still carries that EMA (a host scaffold upstream, in
  webapp/da_mode_drives_chat.py).
  P method A, the bus form, was tried first and its registered design rule failed (artifact
  research/findings/raw/_c3_topic_stop/design_s9.json): a dopamine modulator (`from_region_firing_signed` on the
  SNc, `excitability_drive` on group:ask, concentration +-1 around baseline) needs S_DA between 60 and 80 pA to bring
  a fresh novel want below threshold at the floor (60 -> 21.4 Hz, 80 -> 14.7 Hz against 17.9 Hz), and at 80 pA the
  drive alone at the ceiling already fires the ASK pool (15.6 Hz with no topic input), violating the bound that DA
  must never ask by itself. An additive excitability drive is not a gain. Its rung: a per-pathway (topic -> ASK)
  neuromodulator synaptic-gain target, which needs a sim/ edit (today `synaptic_gain` supports scope=all only). The
  bus code stays here only to reproduce that artifact (`design()`); the handler never installs it.
  G_PHASIC_PA is fixed by the design rule's phasic half (unchanged since registration): the smallest grid value for
  which a fully salient topic at the floor tonic brings the SNc back to its neutral rate.

FUNCTIONAL CORRELATE, NOT phenomenal. A habituated ask drive is reported as a read-out ("I have asked about toads
three times; my drive to ask about toads has habituated"), never as felt boredom or felt interest.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._curiosity_lp_trace_derisk import (  # noqa: E402  (method 2, reuse-by-import)
    CuriosityLPOrgan,
    W_TOPIC_ASK,
    TAU_HAB_MS,
    TAU_LESION_MS,
    INTER_TURN_MS,
    N_SLOTS,
    _h,
)
from research.runners.spiking_novelty_habituation_organ import (  # noqa: E402
    STP_U, DRIVE_STEPS, READ_STEPS, WORD_DRIVE,
)
from research.runners._curiosity_seek_learn_onbridge_derisk import (  # noqa: E402
    _advance, _idx, _restore_state, SNC_TONIC_PA, W_WARMUP, W_MEASURE,
)
from research.runners.curiosity_production_organ import NOVEL_SIGNAL, FAMILIAR_SIGNAL  # noqa: E402

# ── P constants (registered before data; the design rule fixes S_DA_PA and G_PHASIC_PA, see design()) ─────────────
DA_TONIC = 0.5                         # the host EMA's neutral level (webapp/da_curiosity_drives_chat._DA_TONIC_BASELINE)
K_TONIC_PA = SNC_TONIC_PA / DA_TONIC   # 440 pA per DA unit, intercept 0: da 0.5 -> 220 pA (the SNc's neutral drive)
DA_LEVEL_MAX = 2.0                     # clip of the host EMA before encoding (it has read ~1.24 at most)
DA_NAME = "da_ask"
DA_BASELINE = 1.0
DA_MIN = 0.0                           # = baseline - 1: a silent SNc is the floor
DA_MAX = 2.0                           # = baseline + 1: an SNc at twice its neutral rate is the ceiling
DA_TAU_MS = 20.0                       # concentration decay (fast phasic transmitter clearance scale)
DA_WINDOW_MS = 5.0                     # the rule's SNc-rate EMA
W_DA_WARMUP = 60                       # SNc + DA settle before the topic burst (3 x DA_TAU_MS)
# AMENDMENT D1 (2026-10-26, before any handler run; packet section "P design"): the first design run was VOID (the
# rule-sensitivity conversion missed the engine's dt/1000 factor, so the concentration never left baseline) and the
# neutral SNc rate was read in a different window from the P read. Both fixed to the registered intent; the grid is
# extended upward (80-200) because the first valid read showed 40 pA leaves a fresh novel slice above threshold at the
# floor. The rule itself (smallest value meeting (a) the floor reproduces the failure and (b) DA alone never fires the
# ASK pool) is unchanged, and (b) still binds.
S_DA_GRID = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0, 200.0)   # pA per unit concentration
G_PHASIC_GRID = (100.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 2400.0)   # pA onto reward_us per unit salience
DESIGN_SEED = 9                        # method 2's design seed (never a battery or gate seed)
DA_FLOOR_PROBE = 0.046163              # the floor the gate recorded on s42 (diagnosis finding), used by the design rule
# FROZEN by the design rule (research/findings/raw/_c3_topic_stop/design_s9.json, run after amendment D1):
# S_DA_PA None = no grid value met both halves, so P method A (the bus form) is not used; G_PHASIC_PA 600 pA (the
# SNc at the floor tonic with salience 1 reads 47.6 Hz against its 40.5 Hz neutral rate in the P window).
DESIGN_FROZEN = {"S_DA_PA": None, "G_PHASIC_PA": 600.0, "artifact": "research/findings/raw/_c3_topic_stop/design_s9.json"}


class TopicStopOrgan(CuriosityLPOrgan):
    """Method 2's organ (noise off, frozen constants) with the production read protocol: a lazily-run turn clock,
    a capacity rule on the slice table, the lesion read-backs, and (P) a topic-locked SNc read feeding the crave gain.
    `bus_da` installs P method A (the rejected bus form) for `design()` only."""

    def __init__(self, seed, *, lesion_stop=False, drive_lesion=False, phasic=False, phasic_lesion=False,
                 s_da_pa=None, g_phasic_pa=None, bus_da=False):
        super().__init__(seed, w_topic_ask=W_TOPIC_ASK,
                         tau_hab_ms=(TAU_LESION_MS if lesion_stop else TAU_HAB_MS),
                         reset_on=True, drive_lesion=bool(drive_lesion), shared_slice=False, no_ou=True)
        self.lesion_stop = bool(lesion_stop)
        self.phasic = bool(phasic)
        self.phasic_lesion = bool(phasic_lesion)
        self.s_da_pa = float(s_da_pa if s_da_pa is not None else (DESIGN_FROZEN["S_DA_PA"] or 0.0))
        self.g_phasic_pa = float(g_phasic_pa if g_phasic_pa is not None else (DESIGN_FROZEN["G_PHASIC_PA"] or 0.0))
        self.turns_idled = 0
        self.n_presented = {}          # topic -> presentations so far (reported, never read by the decision)
        self._slot_last = {}           # slot -> presentation counter value at its last use (the capacity rule only)
        self._tick = 0
        self.recycled = []
        self.bus_da = bool(bus_da)
        self._da_installed = False
        self.snc_neutral_p_hz = None

    # ── build ──────────────────────────────────────────────────────────────────────────────────────────────────
    def ensure_built(self):
        if self._built:
            return
        super().ensure_built()                     # bridge + calibration (no DA modulator yet) + snc_neutral_hz
        if self.phasic:
            # the SNc's neutral rate in the P read's own window (da_level 0.5, no phasic), before the modulator exists
            self.snc_neutral_p_hz = self.snc_protocol_rate(DA_TONIC, 0.0)
            if self.bus_da:
                self._install_da()

    def _install_da(self):
        """The dopamine modulator on the engine's bus, installed AFTER calibration so the threshold is the same as
        L's. Production: from_region_firing_signed over `snc`, threshold = the SNc's neutral firing fraction, rate EMA
        DA_WINDOW_MS, sensitivity chosen so that conc - baseline = (rate - neutral) / neutral at equilibrium. Target:
        excitability_drive on group:ask, S_DA_PA per unit concentration."""
        from sim.neuromodulators import (NeuromodulatorConfig, ModulatorTarget, ProductionRule,
                                         NeuromodulatorManager)
        b = self.bridge
        cfg = b.core_config
        thr_frac = float(self.snc_neutral_p_hz) * 1e-3 * cfg.dt_ms
        # the engine's signed rule returns sensitivity * (ema - threshold) * dt/1000 per step (sim/neuromodulators.py),
        # and decay removes (1 - exp(-dt/tau)) of the excess per step: equilibrium excess = (rate - neutral) / neutral
        k = (1.0 - math.exp(-cfg.dt_ms / DA_TAU_MS)) / (max(thr_frac, 1e-9) * cfg.dt_ms / 1000.0)
        nm = NeuromodulatorConfig(
            name=DA_NAME, baseline=DA_BASELINE, decay_tau_ms=DA_TAU_MS,
            concentration_min=DA_MIN, concentration_max=DA_MAX,
            targets=[ModulatorTarget(target_type="excitability_drive", scope="group:ask",
                                     sensitivity=float(self.s_da_pa))],
            production_rules=[ProductionRule(rule_type="from_region_firing_signed", sensitivity=float(k),
                                             threshold=float(thr_frac), window_ms=DA_WINDOW_MS,
                                             source_regions=["snc"])])
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = [nm]
        n = int(b.cp_membrane_potential_v.shape[0])
        mgr = NeuromodulatorManager(cfg.neuromodulators, cfg.dt_ms)
        mgr.initialize(n, self.xp)
        mgr.set_group_indices(b.region_manager.region_indices_dict())
        b.neuromodulator_manager = mgr
        self._da_k = float(k)
        self._da_thr_frac = float(thr_frac)
        self._da_installed = True

    def set_s_da(self, s_da_pa):
        """Design sweeps only: change the target sensitivity in place (the manager reads it each step)."""
        self.s_da_pa = float(s_da_pa)
        if self._da_installed:
            for c in self.bridge.neuromodulator_manager._configs:
                for t in c.targets:
                    t.sensitivity = float(s_da_pa)

    def _reset_da(self):
        mgr = getattr(self.bridge, "neuromodulator_manager", None)
        if mgr is None:
            return
        mgr._concentrations[DA_NAME] = DA_BASELINE
        st = mgr._rule_state.setdefault(DA_NAME, {})
        st["signed_rate_ema"] = float(self._da_thr_frac)     # start the rate EMA at neutral (no spurious dip)

    def _wash_neurons(self):
        super()._wash_neurons()
        if self._da_installed:
            self._reset_da()

    # ── the turn clock and the slice table ─────────────────────────────────────────────────────────────────────
    def catch_up(self, turns_seen):
        """Run the inter-turn silent substrate time owed: one INTER_TURN_MS idle per conversation turn not yet idled
        (method 2's battery runs one idle before every turn). Returns the number of idles run."""
        self.ensure_built()
        due = max(0, int(turns_seen) - int(self.turns_idled))
        for _ in range(due):
            self.idle(INTER_TURN_MS)
        self.turns_idled = max(int(turns_seen), self.turns_idled)
        return due

    def slot_for(self, topic):
        if topic in self._slot_of:
            return self._slot_of[topic]
        if len(self._slot_of) < N_SLOTS - 1:
            return super().slot_for(topic)
        # capacity rule (declared host bookkeeping): the least recently presented slice, synapses reset to rest
        lru = min(self._slot_of.items(), key=lambda kv: self._slot_last.get(kv[1], -1))
        old_topic, slot = lru
        del self._slot_of[old_topic]
        self._reset_slot(slot)
        self._slot_of[topic] = slot
        self.recycled.append({"slot": int(slot), "from": old_topic, "to": topic})
        return slot

    def _reset_slot(self, slot):
        from sim.backend import from_host
        s = self._slot_syn[slot]
        x = _h(self.bridge.cp_stp_x).copy(); x[s] = 1.0
        u = _h(self.bridge.cp_stp_u).copy(); u[s] = float(STP_U)
        self.bridge.cp_stp_x[:] = from_host(x.astype(np.float32))
        self.bridge.cp_stp_u[:] = from_host(u.astype(np.float32))

    # ── read-backs (the lesions and the slice, read off the live bridge, never an env echo) ────────────────────
    def slice_state(self, slot):
        data = _h(self.bridge.cp_connections.data)
        x = _h(self.bridge.cp_stp_x)
        s = self._slot_syn[slot]
        w = np.abs(data[s]).astype(np.float64)
        return {"slot": int(slot), "n_syn": int(len(s)), "weight_sum": float(w.sum()),
                "x_mean": float(x[s].mean()) if len(s) else None,
                "efficacy_sum": float((w * x[s]).sum())}

    def held(self):
        data = _h(self.bridge.cp_connections.data)
        out = {"stp_tau_d_ms": float(self.bridge.core_config.stp_tau_d),
               "ou_enabled": bool(getattr(self.bridge.core_config, "enable_ou_process", None)),
               "topic_ask_weight_sum": float(sum(float(np.abs(data[self._slot_syn[k]]).sum())
                                                 for k in range(N_SLOTS)))}
        if self._da_installed:
            mgr = self.bridge.neuromodulator_manager
            out["da_target_sensitivity_pa"] = float(mgr._configs[0].targets[0].sensitivity)
        return out

    # ── reads ──────────────────────────────────────────────────────────────────────────────────────────────────
    def _snc_currents(self, da_level, salience):
        """The two declared input encodings: (tonic pA onto snc, phasic pA onto reward_us)."""
        lvl = float(min(DA_LEVEL_MAX, max(0.0, float(da_level))))
        sal = 0.0 if self.phasic_lesion else float(min(1.0, max(0.0, float(salience))))
        return K_TONIC_PA * lvl, self.g_phasic_pa * sal, lvl, sal

    def _present_da(self, slot, novelty, i_tonic, i_phasic):
        """P's presentation: SNc (tonic) and reward_us (phasic) driven for W_DA_WARMUP steps, then the topic burst
        (DRIVE_STEPS) and the read (READ_STEPS) with the SNc drive held. Returns spike counts and the DA trace."""
        from sim.backend import from_host
        b = self.bridge
        mgr = b.neuromodulator_manager
        cur = np.zeros(self._num, np.float32)
        snc = _h(self._idx_snc); us = _h(self._idx_us)
        cur[snc] = float(i_tonic)
        cur[us] = float(i_phasic)
        b.cp_external_input_current[:] = from_host(cur)
        snc_spk_warm = us_spk = ask_warm = 0
        conc = []
        for _ in range(W_DA_WARMUP):
            _advance(b)
            snc_spk_warm += int(b.cp_firing_states[self._idx_snc].sum())
            us_spk += int(b.cp_firing_states[self._idx_us].sum())
            ask_warm += int(b.cp_firing_states[self.idx_ask].sum())
        conc_onset = float(mgr.get_concentration(DA_NAME))
        cur2 = cur.copy()
        cur2[self._slot_neurons[slot]] = float(novelty) * float(WORD_DRIVE)
        sl = self.xp.asarray(self._slot_neurons[slot])
        ask_spk = slot_spk = snc_spk_read = 0
        b.cp_external_input_current[:] = from_host(cur2)
        for _ in range(DRIVE_STEPS):
            _advance(b)
            ask_spk += int(b.cp_firing_states[self.idx_ask].sum())
            slot_spk += int(b.cp_firing_states[sl].sum())
            snc_spk_read += int(b.cp_firing_states[self._idx_snc].sum())
            conc.append(float(mgr.get_concentration(DA_NAME)))
        b.cp_external_input_current[:] = from_host(cur)
        for _ in range(READ_STEPS):
            _advance(b)
            ask_spk += int(b.cp_firing_states[self.idx_ask].sum())
            slot_spk += int(b.cp_firing_states[sl].sum())
            snc_spk_read += int(b.cp_firing_states[self._idx_snc].sum())
            conc.append(float(mgr.get_concentration(DA_NAME)))
        b.cp_external_input_current[:] = 0.0
        n_read = DRIVE_STEPS + READ_STEPS
        return {"ask_spk": ask_spk, "slot_spk": slot_spk, "ask_warmup_spikes": int(ask_warm),
                "snc_hz_warmup": snc_spk_warm / max(self._n_snc, 1) / (W_DA_WARMUP * 1e-3),
                "snc_hz_read": snc_spk_read / max(self._n_snc, 1) / (n_read * 1e-3),
                "reward_us_spikes": int(us_spk),
                "da_conc_onset": conc_onset, "da_conc_read_mean": float(np.mean(conc)),
                "da_drive_pa_read_mean": float(self.s_da_pa * (np.mean(conc) - DA_BASELINE))}

    def phasic_gain(self, da_level, salience):
        """P method B: the topic-locked SNc read -> da_eff -> the existing host crave gain. Returns the trace dict."""
        from webapp.da_curiosity_drives_chat import da_crave_gain
        i_t, i_p, lvl, sal = self._snc_currents(DA_TONIC if da_level is None else da_level,
                                                 0.0 if salience is None else salience)
        rate = self.snc_protocol_rate(lvl, sal)
        da_eff = DA_TONIC * rate / self.snc_neutral_p_hz if self.snc_neutral_p_hz else DA_TONIC
        gain = float(da_crave_gain(da_eff))
        return {"on": True, "method": "B (spiking SNc read -> host gain formula, declared)",
                "lesioned": bool(self.phasic_lesion), "salience": float(sal),
                "salience_in": (None if salience is None else float(salience)),
                "i_phasic_pa": float(i_p), "g_phasic_pa": float(self.g_phasic_pa),
                "snc_hz": float(rate), "snc_neutral_hz": float(self.snc_neutral_p_hz), "da_eff": float(da_eff),
                "reward_us_spikes": int(getattr(self, "last_reward_us_spikes", -1)),
                "da_crave_gain": gain}, {"da_level": float(lvl), "i_tonic_pa": float(i_t), "k_tonic_pa": K_TONIC_PA}

    def read(self, topic, novelty, *, turns_seen, da_level=None, salience=None):
        """The production read on an abstain with topic `topic`: catch up the turn clock, present the topic's slice
        (method 2's own read, unchanged) and decide `want >= threshold`; with P on, `want * gain >= threshold` with the
        gain fed by the topic-locked SNc read. Returns the engagement trace."""
        self.ensure_built()
        idles = self.catch_up(turns_seen)
        slot = self.slot_for(topic)
        pre = self.slice_state(slot)
        rec = {"topic": topic, "novelty": float(novelty), "idles_run": int(idles)}
        self._wash_neurons()
        ask_spk, slot_spk = self._drive_slot(slot, novelty)
        want = self._want_hz(ask_spk)
        self._wash_neurons()
        want_dec = want
        if self.phasic:
            ph, tn = self.phasic_gain(da_level, salience)
            want_dec = want * ph["da_crave_gain"]
            ph["want_eff_hz"] = float(want_dec)
            rec["phasic_da"] = ph
            rec["tonic_da"] = tn
        self._tick += 1
        self._slot_last[slot] = self._tick
        self.n_presented[topic] = self.n_presented.get(topic, 0) + 1
        post = self.slice_state(slot)
        rec.update({"slot": int(slot), "want_hz": float(want), "want_decision_hz": float(want_dec),
                    "threshold": float(self.threshold),
                    "curious": bool(want_dec >= self.threshold), "afferent_spikes": int(slot_spk),
                    "slice_pre": pre, "slice_post": post,
                    "stop": {"presentations": int(self.n_presented[topic]), "x_pre": pre["x_mean"],
                             "x_post": post["x_mean"], "stopped": bool(want_dec < self.threshold)},
                    "held": self.held(), "lesion_stop": bool(self.lesion_stop),
                    "drive_lesion": bool(self.drive_lesion), "recycled": list(self.recycled[-3:])})
        return rec

    def fresh_want(self, novelty, *, da_level=None, salience=None):
        """Design reads: the reference slice 0 with FRESH synapses (full restore incl. STP), like the calibration."""
        self.ensure_built()
        _restore_state(self.bridge, self.snap0)
        if self._da_installed:
            self._reset_da()
        if self._da_installed and da_level is not None:
            i_t, i_p, _l, _s = self._snc_currents(da_level, 0.0 if salience is None else salience)
            r = self._present_da(0, novelty, i_t, i_p)
            want = self._want_hz(r["ask_spk"])
            extra = r
        else:
            ask_spk, _ = self._drive_slot(0, novelty)
            want = self._want_hz(ask_spk)
            extra = {}
        _restore_state(self.bridge, self.snap0)
        if self._da_installed:
            self._reset_da()
        return float(want), extra

    def snc_protocol_rate(self, da_level, salience):
        """The SNc rate (Hz) in the P read's own window: the drives on for W_DA_WARMUP steps, then the rate over the
        next DRIVE_STEPS + READ_STEPS steps (no topic input). Used for the neutral rate and the phasic design rule."""
        from sim.backend import from_host
        i_t, i_p, _l, _s = self._snc_currents(da_level, salience)
        _restore_state(self.bridge, self._snap_neuron)
        if self._da_installed:
            self._reset_da()
        b = self.bridge
        cur = np.zeros(self._num, np.float32)
        cur[_h(self._idx_snc)] = float(i_t)
        cur[_h(self._idx_us)] = float(i_p)
        b.cp_external_input_current[:] = from_host(cur)
        spk = us = 0
        n_read = DRIVE_STEPS + READ_STEPS
        for i in range(W_DA_WARMUP + n_read):
            _advance(b)
            us += int(b.cp_firing_states[self._idx_us].sum())
            if i >= W_DA_WARMUP:
                spk += int(b.cp_firing_states[self._idx_snc].sum())
        b.cp_external_input_current[:] = 0.0
        self.last_reward_us_spikes = int(us)      # read-back: 0 under lesion_P (no phasic afferent fired)
        _restore_state(self.bridge, self._snap_neuron)
        if self._da_installed:
            self._reset_da()
        return spk / max(self._n_snc, 1) / (n_read * 1e-3)


# ── the P design rule (DESIGN_SEED only; registered in the packet before it ran) ───────────────────────────────────
def design(seed=DESIGN_SEED):
    """S_DA_PA := the smallest S_DA_GRID value for which (a) at the floor tonic (da_level DA_FLOOR_PROBE, phasic 0) a
    fresh novel slice's want (NOVEL_SIGNAL) falls below the threshold (the coupling is strong enough that losing the
    tonic reproduces today's failure), and (b) the DA drive alone never fires the ASK pool: at the ceiling (da_level
    DA_LEVEL_MAX, salience 1, G_PHASIC_GRID max) with FAMILIAR_SIGNAL (no topic input) the want is 0 Hz.
    G_PHASIC_PA := the smallest G_PHASIC_GRID value for which, at the floor tonic with salience 1, the SNc rate is at
    least its neutral rate (a fully novel topic restores the DA a disengaged session lost). Nothing about asking order,
    stopping or any battery turn is read here."""
    t0 = time.time()
    org = TopicStopOrgan(seed, phasic=True, s_da_pa=S_DA_GRID[0], g_phasic_pa=G_PHASIC_GRID[-1], bus_da=True)
    org.ensure_built()
    thr = float(org.threshold)
    rows = []
    for s in S_DA_GRID:
        org.set_s_da(s)
        w_neutral, _ = org.fresh_want(NOVEL_SIGNAL, da_level=DA_TONIC, salience=0.0)
        w_floor, ex_floor = org.fresh_want(NOVEL_SIGNAL, da_level=DA_FLOOR_PROBE, salience=0.0)
        w_alone, ex_alone = org.fresh_want(FAMILIAR_SIGNAL, da_level=DA_LEVEL_MAX, salience=1.0)
        rows.append({"s_da_pa": s, "want_novel_neutral_hz": w_neutral, "want_novel_floor_hz": w_floor,
                     "want_da_alone_ceiling_hz": w_alone, "da_conc_floor": ex_floor.get("da_conc_read_mean"),
                     "da_conc_ceiling": ex_alone.get("da_conc_read_mean"),
                     "ask_warmup_spikes_ceiling": ex_alone.get("ask_warmup_spikes"),
                     "floor_below_threshold": bool(w_floor < thr),
                     "alone_silent": bool(w_alone == 0.0 and ex_alone.get("ask_warmup_spikes", 0) == 0)})
    chosen_s = next((r["s_da_pa"] for r in rows if r["floor_below_threshold"] and r["alone_silent"]), None)
    g_rows = []
    for g in G_PHASIC_GRID:
        org.g_phasic_pa = g
        g_rows.append({"g_phasic_pa": g, "snc_hz_floor_sal1": org.snc_protocol_rate(DA_FLOOR_PROBE, 1.0)})
    neutral_hz = float(org.snc_neutral_p_hz)
    chosen_g = next((r["g_phasic_pa"] for r in g_rows if r["snc_hz_floor_sal1"] >= neutral_hz), None)
    org.g_phasic_pa = G_PHASIC_GRID[-1]
    snc_tonic_curve = {str(l): org.snc_protocol_rate(l, 0.0) for l in (0.0, DA_FLOOR_PROBE, 0.25, 0.5, 1.0, 2.0)}
    return {"seed": int(seed), "threshold_hz": thr, "calib": org.calib, "snc_neutral_hz": neutral_hz,
            "snc_neutral_hz_method2_window": float(org.snc_neutral_hz),
            "s_da_rows": rows, "g_phasic_rows": g_rows, "snc_tonic_only_hz": snc_tonic_curve,
            "chosen": {"S_DA_PA": chosen_s, "G_PHASIC_PA": chosen_g},
            "rule": design.__doc__.strip(), "constants": {
                "K_TONIC_PA": K_TONIC_PA, "DA_TAU_MS": DA_TAU_MS, "DA_WINDOW_MS": DA_WINDOW_MS,
                "W_DA_WARMUP": W_DA_WARMUP, "DA_MIN": DA_MIN, "DA_MAX": DA_MAX, "DA_FLOOR_PROBE": DA_FLOOR_PROBE,
                "W_TOPIC_ASK": W_TOPIC_ASK, "TAU_HAB_MS": TAU_HAB_MS, "no_ou": True},
            "wall_s": round(time.time() - t0, 1)}


def selftest():
    """Bridge-free checks of the constants and the encodings."""
    checks = {
        "k_tonic_neutral": abs(K_TONIC_PA * DA_TONIC - SNC_TONIC_PA) < 1e-9,
        "da_bounds_unit": DA_MAX - DA_BASELINE == 1.0 and DA_BASELINE - DA_MIN == 1.0,
        "design_seed_not_dev_or_gate": DESIGN_SEED not in (1, 2, 3, 4, 5, 6, 7, 8, 42, 43, 44, 100, 101, 102),
        "grids_sorted": list(S_DA_GRID) == sorted(S_DA_GRID) and list(G_PHASIC_GRID) == sorted(G_PHASIC_GRID),
    }
    return all(checks.values()), checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", action="store_true", help="run the P design rule on DESIGN_SEED")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=DESIGN_SEED)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        ok, checks = selftest()
        print(json.dumps(checks, indent=1))
        return 0 if ok else 1
    if a.design:
        if a.seed != DESIGN_SEED:
            print("the design rule runs on DESIGN_SEED=%d only" % DESIGN_SEED)
            return 2
        res = design(a.seed)
        res["runner"] = "research/runners/curiosity_topic_stop_organ.py"
        txt = json.dumps(res, indent=1, default=str)
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w") as f:
                f.write(txt)
        print(json.dumps({"chosen": res["chosen"], "threshold_hz": res["threshold_hz"],
                          "snc_neutral_hz": res["snc_neutral_hz"]}, indent=1))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
