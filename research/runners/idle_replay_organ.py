"""IDLE REPLAY: on an idle turn the brain voices a fact from its OWN store, chosen by a spiking reactivation competition
biased by how recently and how strongly each memory was encoded (Stage-2 C5 build, Claude 2026-10-06).

WHY. Stage-2 exit criterion C5 `inner` (prereg 2026-09-30 section 4.5 + amendment A3,
docs/drafts/stage2-exit/AMENDMENT-A3.md) needs the idle turn (`say something`) to voice a fact the brain HOLDS, and,
after A3, a fact SELECTED by its state: the idle fact after the permuted exposure must differ from the intact one. The
production self-initiation organ (`self_initiated_production_organ.py`, default ON) speaks from its OWN fixed demo
lexicon (dog chase ball, cat eat worm, bird see seed, fish hear bug), never from the brain's store
(NEEDS-CLAUDE #15). This organ is the replacement selection, additive and default OFF (`BRAIN_IDLE_REPLAY`).

THE BIOLOGY.
  * At quiet wakeful rest the hippocampus reactivates stored experiences in sharp-wave ripples; awake replay favours
    RECENT experience (Karlsson & Frank 2009, Nat Neurosci 12:913; Kudrimoti, Barnes & McNaughton 1999, J Neurosci
    19:4090: reactivation strength decays over the rest period after the experience) and SALIENT / rewarded
    experience (Singer & Frank 2009, Neuron 64:910; McNamara et al. 2014, Nat Neurosci 17:1658: dopamine at encoding
    raises later reactivation). Which memory reactivates is a competition among engrams, not a list lookup
    (Mattar & Daw 2018, Nat Neurosci 21:1609, give the normative priority; here the priority is carried by cells).
  * The cellular carrier of "recent" is intrinsic excitability: neurons allocated to a new memory stay more excitable
    for a window after encoding (CREB-dependent; Yiu et al. 2014, Neuron 83:722; Cai et al. 2016, Nature 534:115; Rashid
    et al. 2016, Neuron 92:1), so a recently encoded engram is closer to threshold when spontaneous activity arrives.
  * The idle-turn trigger stays the existing host turn class (`is_selfinit_trigger`); default-mode / rest activity
    is the condition, not the content (Buckner et al. 2008).

THE CIRCUIT (one SimulationBridge, Izhikevich neurons, dt 1 ms). For each stored fact k (an "engram"):
  * E_k  NE excitatory neurons, the engram assembly. Recurrent E_k -> E_k (W_EE, transmission gate `ir_engram`): the
    encoded assembly. Each E_k drives its own inhibitory pool I_k (W_EI), and I_k inhibits every other engram
    (W_IE): lateral inhibition, one engram's population burst suppresses the others.
  * Excitability tag x_k = BASE + s_k * exp(-age_k / TAU_TURNS): a constant bias current BIAS_PA * x_k into E_k during
    the rest window. age_k = turns since fact k was (re)encoded; s_k = its encoding strength read off the composer's
    own engram (the DA-gated write magnitude, `OneBrainComposer._measure_block_readout` / `_homeo_setpoint`, clipped
    to [SAL_MIN, SAL_MAX]; 1.0 when the composer has no such read). The trace is a neuron state variable advanced per
    TURN on the host (declared compression of the hours-long excitability window onto the conversation clock).
  * Rest: weak non-specific background input to every E neuron: an independent Ornstein-Uhlenbeck current per neuron
    (SD NOISE_SD pA, correlation time NOISE_TAU ms; the point-conductance picture of synaptic bombardment, Destexhe et
    al. 2001, Neuroscience 107:13), the same statistics for every engram, from a seeded generator. No content is
    injected; the noise and the tags set which engram ignites first.
  * Commit (`spiking_commit.py`): E_k -> G_k (W_XG, transmission gate `ir_commit`), exclusive gates, latches S_k
    (`ir_latch`), and an occupancy pool O: every latch drives O (W_SO) and O inhibits every gate and every engram
    (W_OG, W_OE): once one memory is held, the rest of the rest window cannot commit a second one.
  * Read: after HOLD_STEPS of zero input, the latch read (`spiking_commit.read_with_variants`): exactly one latch ON
    -> that fact; none -> nothing surfaces (the honest neutral idle line); two or more -> co-commit (nothing). Re-read
    at x0.5 and x2 of the read level; the trace records `read_invariant`.
  * The voice: the winning fact's block is decoded on the composer's own bridge (`OneBrainComposer._read_one_block`,
    the same spiking unbind + cleanup every recall uses); the utterance is the DECODED triple, never the host label.
    A decode with a missing role abstains. Composers without that read use `query_patient(agent, action)`.

LESIONS (each on the same bridge, each a transmission gate or the absence of an encode):
  * `BRAIN_SELF_INITIATE_LESION=1` (the existing self-init lesion, so the Stage-2 `lesion_selfinit` arm reads this
    organ): NO-ENCODE. No engram recurrence (`ir_engram` closed) and no excitability tag: the noise alone drives
    unencoded assemblies, nothing ignites, nothing latches -> the neutral idle line.
  * `BRAIN_IDLE_REPLAY_TAG_LESION=1`: every engram gets the SAME excitability (the mean of the tags): recency and
    salience are removed, the engrams and the competition are intact. The noise then decides.
  * `BRAIN_IDLE_REPLAY_COMMIT_LESION=1`: E -> G closed (`ir_commit`): the engrams still compete, nothing commits.

HOST (declared residuals, not hidden):
  * the idle-turn trigger and the remark frame ("Something's been on my mind -- ...") are the existing self-init
    scaffold; the candidate list is TRANSCRIBED from the composer's fact index (the same residual
    `word_role_parser_organ.facts_from_chat` declares); the turn counter, the per-turn advance of the excitability
    trace and the encoding-strength relay into a bias current are host (the organ is its own bridge: one-brain debt);
    above K_MAX facts the K_MAX most excitable are kept (a host pre-filter, never hit at Stage-2 sizes).
  * what it does NOT show: that the tag is learned (it is wired), that the trigger is autonomous (it is HTTP), or that
    the remark is inner speech in any experiential sense. The reply is a functional read-out of which engram
    reactivated; it never claims an experience.

FLAGS. `BRAIN_IDLE_REPLAY` (default OFF; `_IDLE_REPLAY_DEFAULT_ON = False`). Off: the server never calls this module
beyond the flag read, so every reply is byte-identical. `BRAIN_IDLE_REPLAY_OP` selects a REPLAY_GRID row (default 0).
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.runners import spiking_commit as SC

_IDLE_REPLAY_DEFAULT_ON = False

GATE_ENGRAM = "ir_engram"          # engram recurrence (NO-ENCODE closes it)
GATE_COMMIT_IN = "ir_commit"       # engram -> commit gate (the commit lesion closes it)
GATE_LATCH = "ir_latch"            # latch self-recurrence
GATE_EXCL = "ir_excl"              # commit-gate cross-inhibition

DEV_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
GATE_SEEDS = (42, 43, 44, 100, 101, 102)

# Frozen operating-point grid (row 0 is the reference; rows are tried in order on dev seeds 1-8 only).
# NE/NI engram and lateral-inhibition sizes; NG/NGI/NS commit gate, gate-inhibitory and latch sizes; NO occupancy pool;
# W_* weights (synaptic units of the bridge); NOISE_SD/NOISE_TAU the OU background current (pA, ms); BIAS_PA the bias current of
# a fully excitable engram (x = 1); BASE the excitability every encoded engram keeps; TAU_TURNS the decay of the
# recency trace in turns; SAL_MIN/SAL_MAX the clip on the encoding-strength factor; REST/HOLD/READ steps (ms) and
# READ_FRAC the latch read level (spikes per latch neuron per step); K_MAX the candidate cap.
_R = dict(NE=16, NI=6, NG=12, NGI=8, NS=20, NO=8,
          W_EE=40.0, W_EI=40.0, W_IE=40.0,
          W_XG=30.0, W_GG=20.0, W_GGI=60.0, W_GIG=60.0, W_GS=12.0, W_SS=60.0,
          W_SO=20.0, W_OG=80.0, W_OE=40.0,
          NOISE_SD=25.0, NOISE_TAU=5.0, BIAS_PA=42.0, BASE=0.8, TAU_TURNS=1.5, SAL_MIN=0.5, SAL_MAX=2.0,
          REST_STEPS=500, HOLD_STEPS=100, READ_STEPS=30, READ_FRAC=0.03, K_MAX=32)
# Rows, frozen in this order (Claude, 2026-10-06, workstation numpy, `idle_replay_battery` on dev seeds 1-8 x 10
# background-noise draws; the Stage-2 C5 store shape: 5 build facts at age 4 + a 3-fact chain at ages 3/2/1, encoding
# strength 1; gate seeds never run). Counts are draws out of 80 (lesions: latched draws out of 24).
#   row  change                newest wins        differ  no-encode  commit-  tag lesion:      ignition ms   all_pass
#                              intact / permuted          latched    lesion   newest (chance   median / max  seeds
#                                                                             10 of 80)
#   0    reference             78 / 78            80      0          0        11 (max 4/seed)  129 / 166     8/8
#   1    BIAS_PA 50, REST 300  76 / 76            80      0          0        7                94 / 127      8/8
#   2    TAU_TURNS 1.0         78 / 78            80      0          0        15 (max 5/seed)  170 / 223     8/8
#   3    TAU 1.0, BIAS 46      79 / 79            80      0          0        9                137 / 177     8/8
#   4    NOISE_SD 20           78 / 78            80      0          0        9                145 / 179     8/8
# Row 0 puts the newest engram at about its rheobase, so the background noise sets WHEN it ignites; a miss (2 draws of
# 80 per arm, dev seed 2) is the second-newest fact winning the race, the competition's own variance. Near tie (the
# second-newest fact encoded at strength 2 against the newest at 1): 48 / 32 of 80 at row 0, so the race, not a sort,
# resolves a tie. Row 2 sits at the `tag_lesion_moves` edge on seed 6 (5/10): with recency removed, per-neuron
# heterogeneity alone favours one assembly position there (a property of the substrate, reported). Selection rule:
# the first row in this order with all_pass on 8/8 dev seeds -> row 0.
REPLAY_GRID: List[dict] = [
    dict(_R),                                   # 0: reference
    dict(_R, BIAS_PA=50.0, REST_STEPS=300),     # 1: newest engram suprathreshold
    dict(_R, TAU_TURNS=1.0),                    # 2: faster recency decay
    dict(_R, TAU_TURNS=1.0, BIAS_PA=46.0),      # 3: faster decay, stronger bias
    dict(_R, NOISE_SD=20.0),                    # 4: weaker background
]


def _env_on(name: str) -> bool:
    v = os.environ.get(name)
    return v is not None and v.strip().lower() in ("1", "true", "yes", "on")


def idle_replay_enabled() -> bool:
    """Default OFF. `BRAIN_IDLE_REPLAY` in {1,true,yes,on} turns the organ on; {0,false,no,off} forces it off."""
    v = os.environ.get("BRAIN_IDLE_REPLAY")
    if v is None:
        return bool(_IDLE_REPLAY_DEFAULT_ON)
    return v.strip().lower() in ("1", "true", "yes", "on")


def tag_lesioned() -> bool:
    return _env_on("BRAIN_IDLE_REPLAY_TAG_LESION")


def commit_lesioned() -> bool:
    return _env_on("BRAIN_IDLE_REPLAY_COMMIT_LESION")


def encode_lesioned() -> bool:
    """The NO-ENCODE lesion is the existing self-initiation lesion flag (the Stage-2 `lesion_selfinit` arm)."""
    return _env_on("BRAIN_SELF_INITIATE_LESION")


def op_params() -> dict:
    v = os.environ.get("BRAIN_IDLE_REPLAY_OP")
    i = 0
    if v is not None:
        try:
            i = int(v)
        except ValueError:
            i = 0
    if not (0 <= i < len(REPLAY_GRID)):
        i = 0
    return dict(REPLAY_GRID[i], OP=i)


def excitability(age: float, s: float, p: dict) -> float:
    """The excitability tag of an encoded engram: BASE + s * exp(-age / TAU_TURNS)."""
    return float(p["BASE"]) + float(s) * math.exp(-max(0.0, float(age)) / float(p["TAU_TURNS"]))


# ── the substrate ───────────────────────────────────────────────────────────────────────────────────────────────
def _isolated(fn):
    """Run `fn` and restore the host numpy (and backend) global RNG state afterwards."""
    from sim.backend import get_backend
    try:
        xp, _ = get_backend()
    except Exception:  # noqa: BLE001
        xp = None
    host_np = np.random.get_state()
    host_xp = None
    if xp is not None and xp is not np:
        try:
            host_xp = xp.random.get_random_state().get_state()
        except Exception:  # noqa: BLE001
            host_xp = None
    try:
        return fn()
    finally:
        np.random.set_state(host_np)
        if host_xp is not None:
            try:
                xp.random.get_random_state().set_state(host_xp)
            except Exception:  # noqa: BLE001
                pass


class ReplayCircuit:
    """K engram assemblies + lateral inhibition + the commit stage + the occupancy pool, on one bridge."""

    def __init__(self, K: int, seed: int, p: dict):
        self.K, self.seed, self.p = int(K), int(seed), dict(p)
        self._bridge = None
        _isolated(self._build)

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
        n = o_o + NO
        self.eng = [np.arange(o_e + k * NE, o_e + (k + 1) * NE, dtype=np.int64) for k in range(K)]
        self.inh = [np.arange(o_i + k * NI, o_i + (k + 1) * NI, dtype=np.int64) for k in range(K)]
        self.lay = lay
        self.occ = np.arange(o_o, n, dtype=np.int64)
        self.all_eng = np.arange(o_e, o_i, dtype=np.int64)
        cfg = CoreSimConfig()
        cfg.num_neurons = int(n)
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.seed = self.seed
        cfg.dt_ms = 1.0
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1
        for f in ("enable_stdp", "enable_reward_modulation", "enable_homeostasis", "enable_short_term_plasticity",
                  "enable_structural_plasticity", "enable_watts_strogatz", "enable_ou_process", "enable_hebbian_learning"):
            setattr(cfg, f, False)
        cfg.enable_parameter_heterogeneity = True
        bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                                  gpu_config=GPUConfig())
        bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        bridge._initialize_simulation_data(called_from_playback_init=False)
        plan = {}
        for k in range(K):
            plan["ee_%d" % k] = SC.dense(self.eng[k], self.eng[k], p["W_EE"], conn_type="E_TO_E",
                                         transmission_gate=GATE_ENGRAM)
            plan["ei_%d" % k] = SC.dense(self.eng[k], self.inh[k], p["W_EI"], conn_type="E_TO_I")
            if K > 1:
                others = np.concatenate([self.eng[j] for j in range(K) if j != k])
                plan["ie_%d" % k] = SC.dense(self.inh[k], others, p["W_IE"], conn_type="I_TO_E")
        plan.update(SC.commit_plan(self.eng, lay, dict(p, W_XG=p["W_XG"]), prefix="ir_", gate_in=GATE_COMMIT_IN,
                                   gate_excl=GATE_EXCL, latch_loop=GATE_LATCH))
        plan["so"] = SC.dense(lay["all_slot"], self.occ, p["W_SO"], conn_type="E_TO_I")
        plan["og"] = SC.dense(self.occ, lay["all_gate"], p["W_OG"], conn_type="I_TO_E")
        plan["oe"] = SC.dense(self.occ, self.all_eng, p["W_OE"], conn_type="I_TO_E")
        inh = np.concatenate(self.inh + [lay["all_gate_inh"], self.occ]) if K else np.concatenate([self.occ])
        bridge.inject_explicit_wiring(plan, output_inhibitory_indices=inh)
        bridge.set_plasticity_gate(SC.GATE_FIXED, 0.0)
        self._gates = tuple(g for g in (GATE_ENGRAM, GATE_COMMIT_IN, GATE_LATCH, GATE_EXCL)
                            if any(e.get("transmission_gate") == g for e in plan.values()))
        for g in self._gates:
            bridge.set_transmission_gate(g, 1.0)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(60):
            bridge._run_one_simulation_step()
        self._bridge, self._xp, self._n = bridge, xp, int(n)
        self._snap = _snapshot_state(bridge, xp)

    def run(self, x: Sequence[float], noise_seed: int, encoded: bool = True, commit: bool = True) -> dict:
        """One rest window with excitability tags `x` (len K), then the zero-input hold and the latch read."""
        return _isolated(lambda: self._run(x, noise_seed, encoded, commit))

    def _run(self, x, noise_seed, encoded, commit):
        from sim.backend import to_host
        from research.runners._gnw_rung1_ignition_curve_derisk import _reset_quiescent
        b, xp, p, K = self._bridge, self._xp, self.p, self.K
        _reset_quiescent(b, xp, self._snap)
        b.set_transmission_gate(GATE_ENGRAM, 1.0 if encoded else 0.0)
        b.set_transmission_gate(GATE_COMMIT_IN, 1.0 if commit else 0.0)
        NE = int(p["NE"])
        nE = K * NE
        bias = np.zeros(self._n, dtype=np.float32)
        for k in range(K):
            bias[self.eng[k]] = float(p["BIAS_PA"]) * float(x[k])
        rng = np.random.default_rng(int(noise_seed))
        tau = float(p["NOISE_TAU"])
        a_ou = 1.0 / tau                                          # dt = 1 ms
        s_ou = float(p["NOISE_SD"]) * math.sqrt(2.0 / tau)        # stationary SD = NOISE_SD
        eta = rng.standard_normal(nE) * float(p["NOISE_SD"])      # start from the stationary distribution
        eng_counts = np.zeros(K, dtype=np.int64)
        gate_counts = np.zeros(K, dtype=np.int64)
        slot_counts = np.zeros(K, dtype=np.int64)
        first_gate = [None] * K
        gate_all = [g for g in self.lay["gate"]]
        rest, hold, read = int(p["REST_STEPS"]), int(p["HOLD_STEPS"]), int(p["READ_STEPS"])
        try:
            for t in range(rest + hold):
                cur = np.zeros(self._n, dtype=np.float32)
                if t < rest:
                    eta += a_ou * (-eta) + s_ou * rng.standard_normal(nE)
                    cur[:] = bias
                    cur[:nE] += eta.astype(np.float32)
                b.cp_external_input_current[:] = xp.asarray(cur)
                b._run_one_simulation_step()
                f = np.asarray(to_host(b.cp_firing_states)).astype(bool)
                for k in range(K):
                    eng_counts[k] += int(f[self.eng[k]].sum())
                    gk = int(f[gate_all[k]].sum())
                    gate_counts[k] += gk
                    if gk and first_gate[k] is None:
                        first_gate[k] = t
                    if t >= rest + hold - read:
                        slot_counts[k] += int(f[self.lay["slot"][k]].sum())
        finally:
            b.cp_external_input_current[:] = 0.0
            for g in self._gates:
                b.set_transmission_gate(g, 1.0)
        rd = SC.read_with_variants(slot_counts.tolist(), int(p["NS"]), read, float(p["READ_FRAC"]))
        return {"engram_counts": eng_counts.tolist(), "gate_counts": gate_counts.tolist(),
                "slot_counts": slot_counts.tolist(), "first_gate_step": first_gate, "read": rd}


# ── the brain's store ───────────────────────────────────────────────────────────────────────────────────────────
def _composer(chat):
    return getattr(getattr(chat, "inner", None), "composer", None)


def kb_triples(chat) -> List[Optional[Tuple[str, str, str]]]:
    """The composer's fact index as (agent, action, patient) per block (None for a non-string entry). TRANSCRIBED
    from the host index (declared residual): the candidate list, never the selection."""
    comp = _composer(chat)
    kb = getattr(comp, "kb", None)
    if kb is None:
        kb = [(f, None) for f in (getattr(comp, "facts", None) or [])]
    out = []
    for entry in kb or []:
        try:
            fact = entry[0]
            a, v, pt = fact.get("agent"), fact.get("action"), fact.get("patient")
        except Exception:  # noqa: BLE001
            out.append(None)
            continue
        out.append((a, v, pt) if all(isinstance(z, str) for z in (a, v, pt)) else None)
    return out


def _leak_log(org, event: str, k=None) -> None:
    """tree-c5-leak probe (RULING 2026-10-11): per-turn RSS/circuit-build sidecar, flushed EVERY logged event so a
    killed run still leaves the growth curve. Env-gated by BRAIN_IDLE_REPLAY_LEAKLOG (path to a .jsonl sidecar):
    unset -> one environ lookup and return, byte-identical behavior, no file. Never raises (the probe cannot break
    the brain's turn). Each record: wall ts, pid, the organ's turn clock, event ("turn" = every conversation turn,
    "select" = an idle-turn competition), K for select events, the running count of ReplayCircuit builds (steps up
    exactly when _ensure_circuit rebuilds), the circuit's K, and VmRSS/VmHWM in kB."""
    path = os.environ.get("BRAIN_IDLE_REPLAY_LEAKLOG") or ""
    if not path:
        return
    try:
        rss = hwm = 0
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1])
        circ_k = org._circuit.K if getattr(org, "_circuit", None) is not None else None
        rec = {"ts": round(time.time(), 3), "pid": os.getpid(), "turn": int(org.turn), "event": event,
               "k": (None if k is None else int(k)), "circuits_built": int(getattr(org, "_circuits_built", 0)),
               "circ_k": (None if circ_k is None else int(circ_k)), "rss_kb": rss, "hwm_kb": hwm}
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:  # noqa: BLE001 -- a probe must never break the brain's turn
        pass


class IdleReplayOrgan:
    """Per-conversation: the encoding record (turn of (re)encoding per fact), the circuit, the voice."""

    def __init__(self, seed: int = 42, params: Optional[dict] = None):
        self.seed = int(seed)
        self.p = dict(params) if params is not None else op_params()
        self.turn = 0
        self._seen: Optional[List[Optional[Tuple[str, str, str]]]] = None
        self.enc: Dict[Tuple[str, str, str], dict] = {}
        self._sal: Dict[Tuple[int, Tuple[str, str, str]], float] = {}
        self._circuit: Optional[ReplayCircuit] = None
        self._circuits_built = 0
        self._lock = threading.Lock()

    # ---- encoding events (called once per turn, before the turn's reply) ----
    def note_turn(self, chat) -> None:
        """Advance the turn clock and record every fact that entered (or was rewritten in) the store since the last
        call: those were encoded during the previous turn. Facts present at the first call are build-time (turn 0)."""
        with self._lock:
            self.turn += 1
            kb = kb_triples(chat)
            first = self._seen is None
            for i, tr in enumerate(kb):
                if tr is None:
                    continue
                if first or i >= len(self._seen) or self._seen[i] != tr:
                    self.enc[tr] = {"t": 0 if first else self.turn - 1, "idx": i}
            self._seen = kb
        _leak_log(self, "turn")

    def note_encoding(self, triple, idx: int, turn: int) -> None:
        """Direct encoding record (tests, synthetic batteries)."""
        self.enc[tuple(triple)] = {"t": int(turn), "idx": int(idx)}

    # ---- encoding strength (the composer's own engram read) ----
    def _salience(self, chat, idx: int, tr, a_star_box: list) -> Tuple[float, str]:
        key = (int(idx), tuple(tr))
        if key in self._sal:
            return self._sal[key], "cached"
        comp = _composer(chat)
        s, src = 1.0, "unavailable"
        if comp is not None and hasattr(comp, "_measure_block_readout") and hasattr(comp, "_homeo_setpoint"):
            try:
                if not a_star_box:
                    a_star_box.append(float(comp._homeo_setpoint()))
                a_star = a_star_box[0]
                a = float(comp._measure_block_readout(int(idx)))
                if a_star > 0.0 and math.isfinite(a) and math.isfinite(a_star):
                    s, src = a / a_star, "engram_readout"
            except Exception:  # noqa: BLE001
                s, src = 1.0, "error"
        s = float(min(max(s, float(self.p["SAL_MIN"])), float(self.p["SAL_MAX"])))
        self._sal[key] = s
        return s, src

    def candidates(self, chat=None, salience: bool = True) -> List[dict]:
        """The engrams that can reactivate now: facts currently in the store with an encoding record, in store order,
        each with age, encoding strength and excitability tag."""
        live = None
        if chat is not None:
            live = set(t for t in kb_triples(chat) if t is not None)
        out = []
        a_star_box: list = []
        for tr, rec in sorted(self.enc.items(), key=lambda kv: kv[1]["idx"]):
            if live is not None and tr not in live:
                continue
            if salience and chat is not None:
                s, src = self._salience(chat, rec["idx"], tr, a_star_box)
            else:
                s, src = 1.0, "not_read"
            age = max(0, self.turn - int(rec["t"]))
            out.append({"fact": list(tr), "kb_idx": int(rec["idx"]), "t_enc": int(rec["t"]), "age": int(age),
                        "s": float(s), "s_source": src, "x": excitability(age, s, self.p)})
        if len(out) > int(self.p["K_MAX"]):
            out = sorted(out, key=lambda c: -c["x"])[: int(self.p["K_MAX"])]
            out.sort(key=lambda c: c["kb_idx"])
        return out

    def _ensure_circuit(self, K: int) -> ReplayCircuit:
        if self._circuit is None or self._circuit.K != K:
            self._circuit = ReplayCircuit(K, self.seed, self.p)
            self._circuits_built += 1
        return self._circuit

    def noise_seed(self, K: int) -> int:
        return int(self.seed) * 1000003 + int(self.turn) * 7919 + int(K) * 31 + 17

    def select(self, cands: List[dict], *, encoded: bool = True, tag_lesion: bool = False,
               commit: bool = True) -> dict:
        """Run the reactivation competition over `cands`. Returns the circuit record + `winner` (a candidate index or
        None) + the tags actually applied."""
        K = len(cands)
        if K == 0:
            return {"K": 0, "winner": None, "x_applied": [], "read": None, "reason": "empty store"}
        x = [c["x"] for c in cands]
        if not encoded:
            x = [0.0] * K
        elif tag_lesion:
            m = float(np.mean(x))
            x = [m] * K
        circ = self._ensure_circuit(K)
        _leak_log(self, "select", k=K)
        rec = circ.run(x, self.noise_seed(K), encoded=encoded, commit=commit)
        rec.update(K=K, x_applied=[float(v) for v in x], winner=rec["read"]["winner"])
        return rec

    # ---- the voice ----
    @staticmethod
    def decode(chat, cand: dict) -> Tuple[Optional[Tuple[str, str, str]], str]:
        """Decode the winning engram on the composer's own bridge. Returns (triple or None, route)."""
        comp = _composer(chat)
        if comp is None:
            return None, "no_composer"
        idx = int(cand["kb_idx"])
        try:
            if hasattr(comp, "_read_one_block"):
                got = comp._read_one_block(idx)
                tr = (got.get("agent"), got.get("action"), got.get("patient"))
                route = "read_one_block"
            else:
                a, v = cand["fact"][0], cand["fact"][1]
                pt = comp.query_patient(a, v)
                tr = (a, v, pt)
                route = "query_patient"
        except Exception as e:  # noqa: BLE001
            return None, "decode_error: %s" % type(e).__name__
        if not all(isinstance(z, str) and z for z in tr):
            return None, route
        return (tr[0], tr[1], tr[2]), route

    def speak(self, chat, *, lesion: Optional[bool] = None, tag_lesion: Optional[bool] = None,
              commit_lesion: Optional[bool] = None) -> dict:
        """One idle turn. Returns a dict shaped like `SelfInitiationOrgan.speak` (n_utt, utterance, question, concept)
        plus the replay trace under `idle_replay`."""
        lesion = encode_lesioned() if lesion is None else bool(lesion)
        tag_lesion = tag_lesioned() if tag_lesion is None else bool(tag_lesion)
        commit_lesion = commit_lesioned() if commit_lesion is None else bool(commit_lesion)
        with self._lock:
            cands = self.candidates(chat)
            sel = self.select(cands, encoded=not lesion, tag_lesion=tag_lesion, commit=not commit_lesion)
            w = sel.get("winner")
            decoded, route = (None, None)
            if w is not None:
                decoded, route = self.decode(chat, cands[w])
        utt = " ".join(decoded) if decoded else None
        trace = {"route": "idle_replay", "op": int(self.p.get("OP", 0)), "turn": int(self.turn),
                 "lesions": {"no_encode": bool(lesion), "tag": bool(tag_lesion), "commit": bool(commit_lesion)},
                 "candidates": cands, "selection": {k: v for k, v in sel.items() if k != "winner"},
                 "winner": (None if w is None else int(w)),
                 "winner_fact": (None if w is None else cands[w]["fact"]),
                 "winner_is_most_excitable": (None if w is None else
                                              bool(cands[w]["x"] >= max(c["x"] for c in cands))),
                 "decoded": (list(decoded) if decoded else None), "decode_route": route,
                 "decode_matches_winner": (None if (w is None or not decoded) else list(decoded) == cands[w]["fact"])}
        return {"on": True, "lesioned": bool(lesion), "composer": "brain_store", "path": "idle-replay",
                "n_utt": 1 if utt else 0, "about_rate": 1.0 if utt else 0.0, "n_concepts_spoken": 1 if utt else 0,
                "concept": (decoded[0] if decoded else None), "utterance": utt, "question": None,
                "examples": [], "idle_replay": trace}


# ── per-conversation registry ───────────────────────────────────────────────────────────────────────────────────
_ORGANS: Dict[object, IdleReplayOrgan] = {}


def get_organ(cache_key, seed: int = 42) -> IdleReplayOrgan:
    org = _ORGANS.get(cache_key)
    if org is None:
        org = IdleReplayOrgan(seed=seed)
        _ORGANS[cache_key] = org
    return org


def reset_organ(cache_key) -> None:
    _ORGANS.pop(cache_key, None)


def note_turn(cache_key, chat, seed: int = 42) -> None:
    get_organ(cache_key, seed).note_turn(chat)


def speak(cache_key, chat, seed: int = 42) -> dict:
    return get_organ(cache_key, seed).speak(chat)
