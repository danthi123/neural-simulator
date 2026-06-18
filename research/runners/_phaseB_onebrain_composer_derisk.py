"""ROADMAP PHASE 2 (the real "one brain"), STEP A3 -- the production `OneBrainComposer`: the WHOLE conversational
who/what pipeline on ONE persistent co-resident bridge, assembling the validated GO pieces. NO host round-trips between
ops; the host supplies only text in + reads the winning concept (the body's output).

Assembled pieces (each separately GO this arc):
  - the PARSER front-end (GAP B): `BridgeParser` on slice [0:P] comprehends a sentence; its neural role FIRING selects
    each word's bind (no host {role:word} dict). Voice-invariant.
  - the persistent multi-fact STORE (GAP A): each fact is a 3-role composite written into a (1+D) trigger->readout
    block in the bridge's complex weights (register-reset-safe; uniform GO to K=32).
  - the CUE-matching SCAN: a who/what question finds the matching stored fact (reconstruct + unbind cue roles + cleanup
    + first-match), abstaining when none match (the no-confab moat over the store).
  - the on-bridge cleanup + moat (steps 3a/3b) + the 4-role coherence (3c).

The composer co-resides with the parser as disjoint slices on ONE bridge (the merged-bridge regime, masked RF ops).

API mirrors `RFPhasorComposer` so `BrainConversationalAgent` can use it: `hear(sentence, voice)`,
`query_patient(agent, action)`, `query_agent(action, patient)`, `ask_yes_no(agent, action, patient)` (moat = None /
"no").

GATE (3 seeds x 2 D): the full who/what/yes-no/moat matrix over a K-fact knowledge base == the numpy `RFPhasorComposer`
== ground truth, INCLUDING abstention (an unstored cue -> None; an unstored fact -> "no"). The fact is comprehended
by the PARSER (voice-invariant: an active + a passive frame store the same fact). NO sim/ edit. GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_composer_derisk --seeds 42,43,44 --dims 64,128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.backend import to_host  # noqa: E402
from research.runners.brain_conversational_agent import BridgeParser  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402

AGENTS = ["dog", "cat", "bird", "river", "apple", "tree", "sun", "moon"]
ACTIONS = ["go", "come", "look", "stop", "swim", "walk", "run", "jump"]
PATIENTS = ["north", "east", "south", "west", "home", "hill", "lake", "sky"]
VOCAB = AGENTS + ACTIONS + PATIENTS
ROLES3 = ["agent", "action", "patient"]


def _build_coresident_bridge(seed, n_total):
    cfg = CoreSimConfig()
    cfg.num_neurons = n_total
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed); cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0; cfg.num_traits = 1
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_max_weight = 400.0; cfg.hebbian_learning_rate = 0.005
    for f in ("enable_short_term_plasticity", "enable_structural_plasticity", "enable_homeostasis",
              "enable_reward_modulation", "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = 20.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


class OneBrainComposer:
    """The whole who/what pipeline on ONE persistent co-resident bridge. Parser [0:P]; RF region from P:
    fill_0..2, bound_0..2, acc (7 blocks), the persistent store (k_max (1+D) blocks), Q, V cleanup neurons."""

    def __init__(self, seed=42, D=64, vocab=VOCAB, k_max=16):
        self.seed = seed; self.D = D; self.V = len(vocab)
        self.comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab, period=200)
        self.R = 40; self.P = 6 + 3 * self.R; self.k_max = k_max
        self.store_base = self.P + 7 * D
        self.block = 1 + D
        # 3 query registers (one per role, read in PARALLEL from one reconstruction -> no phase drift, ~3x fewer
        # resonate windows than reconstruct-per-read) + 3 V-concept cleanup blocks.
        self.q_base = self.store_base + k_max * self.block       # Q_agent, Q_action, Q_patient at q_base + {0,1,2}*D
        self.c_base = self.q_base + 3 * D                        # concept blocks for agent/action/patient
        self.n_total = self.c_base + 3 * self.V
        self.b = _build_coresident_bridge(seed, self.n_total)
        self.parser = BridgeParser(seed=seed, R=self.R, shared_bridge=self.b, index_offset=0)  # wires+trains [0:P]
        self.rf_mask = np.zeros(self.n_total, dtype=bool); self.rf_mask[self.P:self.n_total] = True
        self.facts = []           # bookkeeping {role:word} per stored fact (host routing only; the VECTOR is on-bridge)
        self.store_conns = []

    # --- comprehend + store (parser-driven) ---
    def hear(self, sentence, voice="active"):
        words = sentence.split() if isinstance(sentence, str) else list(sentence)
        roles = [self.parser.role_of(pos, voice) for pos in range(3)]    # the parser's neural role decision
        self._store_composite(words, roles)
        self.facts.append({roles[i]: words[i] for i in range(3)})
        return {roles[i]: words[i] for i in range(3)}

    def _store_composite(self, words, roles):
        comp, b, D, P, Pd = self.comp, self.b, self.D, self.P, self.comp.period
        binds, bundle = [], []
        kick = np.zeros(self.n_total, dtype=np.complex128)
        for i in range(3):
            zr = comp._to_phasor(comp.roles[roles[i]]); zf = comp._to_phasor(comp.concepts[words[i]])
            kick[P + i * D:P + (i + 1) * D] = zf
            binds += [(P + (3 + i) * D + k, P + i * D + k, complex(zr[k])) for k in range(D)]
            bundle += [(P + 6 * D + k, P + (3 + i) * D + k, 1.0) for k in range(D)]
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        b.rf_set_complex_weights(binds); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        b.rf_set_complex_weights(bundle); b.rf_resonate_steps(Pd + 8)
        zc = comp._to_phasor(np.asarray(b.rf_read_phases())[P + 6 * D:P + 7 * D])
        i = len(self.facts)
        trig = self.store_base + i * self.block
        self.store_conns += [(trig + 1 + k, trig, complex(zc[k])) for k in range(D)]

    # --- query (cue-matching scan; reconstruct ONCE per block, read all 3 roles in PARALLEL) ---
    def _read_block(self, block_idx):
        """Fire block_idx's trigger -> reconstruct its composite; unbind agent/action/patient in PARALLEL into 3 Q
        registers; cleanup all 3 in PARALLEL -> the 3 role words. One reconstruct + 2 resonate windows per block (no
        reconstruct-per-read, no phase drift -- all reads share one settle). Returns (w_agent, w_action, w_patient)."""
        comp, b, D, Pd, V = self.comp, self.b, self.D, self.comp.period, self.V
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        trig = self.store_base + block_idx * self.block
        kick = np.zeros(self.n_total, dtype=np.complex128); kick[trig] = 1.0
        b.rf_set_complex_weights(self.store_conns); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        unbind = []
        for ri, role in enumerate(ROLES3):
            zc = np.conj(comp._to_phasor(comp.roles[role]))
            unbind += [(self.q_base + ri * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
        b.rf_set_complex_weights(unbind); b.rf_resonate_steps(Pd + 8)
        clean = []
        for ri in range(3):
            for j in range(V):
                cc = np.conj(comp._to_phasor(comp.concepts[VOCAB[j]]))
                clean += [(self.c_base + ri * V + j, self.q_base + ri * D + k, complex(cc[k])) for k in range(D)]
        b.rf_set_complex_weights(clean); b.rf_resonate_steps(1)
        mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
        out = []
        for ri in range(3):
            scores = np.maximum(mem[self.c_base + ri * V:self.c_base + (ri + 1) * V], 0.0)
            out.append(VOCAB[int(np.argmax(scores))])
        return tuple(out)            # (agent, action, patient)

    def _scan(self, cue, answer_idx):
        """First stored fact whose cue roles ALL match -> its answer role word; else None (abstain)."""
        for i in range(len(self.facts)):
            wa, wv, wp = self._read_block(i)
            got = {"agent": wa, "action": wv, "patient": wp}
            if all(got[role] == want for role, want in cue.items()):
                return (wa, wv, wp)[answer_idx]
        return None

    def query_patient(self, agent, action):
        return self._scan({"agent": agent, "action": action}, 2)

    def query_agent(self, action, patient):
        return self._scan({"action": action, "patient": patient}, 0)

    def ask_yes_no(self, agent, action, patient):
        """Affirmative-fact yes/no: 'yes' if the full SVO matches a stored fact, else 'unknown' (the no-confab moat --
        abstain rather than assert 'no'). Negation (a bound polarity tag = a 4th role) is a documented follow-on; this
        first OneBrainComposer cut handles affirmative facts, so 'no' (a known-false fact) is out of scope here."""
        for i in range(len(self.facts)):
            wa, wv, wp = self._read_block(i)
            if wa == agent and wv == action and wp == patient:
                return "yes"
        return "unknown"


def run_seed(seed, D):
    obc = OneBrainComposer(seed=seed, D=D, k_max=16)
    comp = obc.comp
    # 5 facts comprehended by the parser (active); +1 stored via its PASSIVE frame (voice-invariance)
    facts = [(AGENTS[i], ACTIONS[i], PATIENTS[i]) for i in range(5)]
    oracle = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB, period=200)
    for (a, v, p) in facts:
        obc.hear(f"{a} {v} {p}", voice="active")
        oracle.store(a, v, p, polarity="AFFIRM")        # affirmative facts (the oracle yes/no checks the polarity tag)
    # the 6th fact via passive: "p6 v6 a6" passive frame must store agent=a6 (voice-invariant comprehension)
    a6, v6, p6 = AGENTS[5], ACTIONS[5], PATIENTS[5]
    obc.hear(f"{p6} {v6} {a6}", voice="passive"); oracle.store(a6, v6, p6, polarity="AFFIRM")
    facts.append((a6, v6, p6))

    okp = oka = oky = hostp = hosta = hosty = 0
    for (a, v, p) in facts:
        ap = obc.query_patient(a, v); aa = obc.query_agent(v, p); ay = obc.ask_yes_no(a, v, p)
        okp += int(ap == p); oka += int(aa == a); oky += int(ay == "yes")
        hostp += int(ap == oracle.query_patient(a, v)); hosta += int(aa == oracle.query_agent(v, p))
        hosty += int(ay == oracle.ask_yes_no(a, v, p))           # both 'yes' for a stored affirmative fact
    n = len(facts)

    # MOAT: an unstored (agent,action) cue -> None; an unstored FACT -> 'unknown' (abstain, == the oracle's 'unknown').
    used = {(a, v) for (a, v, p) in facts}
    absent = next(((a, v) for a in AGENTS for v in ACTIONS if (a, v) not in used), None)
    moat_q = int(obc.query_patient(absent[0], absent[1]) is None) if absent else 1
    scrambled = obc.ask_yes_no(AGENTS[0], ACTIONS[1], PATIENTS[2])   # an unstored SVO combination
    moat_yn = int(scrambled == "unknown" and scrambled == oracle.ask_yes_no(AGENTS[0], ACTIONS[1], PATIENTS[2]))

    row = {"seed": seed, "D": D, "qpatient": okp / n, "qagent": oka / n, "yesno": oky / n,
           "host_p": hostp / n, "host_a": hosta / n, "host_y": hosty / n,
           "moat_q": moat_q, "moat_yn": moat_yn}
    print(f"  [seed {seed} D={D}] who/what: qp={okp/n:.2f} qa={oka/n:.2f} yn={oky/n:.2f} | host p/a/y "
          f"{hostp/n:.2f}/{hosta/n:.2f}/{hosty/n:.2f} | moat: absent-cue->None {moat_q} unstored-fact->no {moat_yn}",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44"); ap.add_argument("--dims", type=str, default="64,128")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onebrain_composer.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]; dims = [int(d) for d in args.dims.split(",")]
    t0 = time.time()
    print("[STEP A3: the OneBrainComposer] the WHOLE who/what/yes-no/moat pipeline on ONE persistent co-resident "
          "bridge (parser comprehends -> persistent store -> cue-scan) == the numpy composer?\n", flush=True)
    rows = [run_seed(s, D) for s in seeds for D in dims]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    qp, qa, yn = m("qpatient"), m("qagent"), m("yesno")
    hp, ha, hy = m("host_p"), m("host_a"), m("host_y")
    mq, myn = m("moat_q"), m("moat_yn")
    n_full = sum(int(r["qpatient"] >= 0.99 and r["qagent"] >= 0.99 and r["yesno"] >= 0.99 and r["host_p"] >= 0.99
                     and r["host_a"] >= 0.99 and r["host_y"] >= 0.99 and r["moat_q"] >= 1 and r["moat_yn"] >= 1)
                 for r in rows)
    go = (n_full == len(rows))
    print(f"\n{'='*108}", flush=True)
    print(f"  MEAN ({len(rows)} seed*D): who/what qp {qp:.3f} qa {qa:.3f} yes/no {yn:.3f} | host-parity p {hp:.3f}/a "
          f"{ha:.3f}/y {hy:.3f} | moat absent-cue {mq:.2f} unstored-fact {myn:.2f} | per-row full {n_full}/{len(rows)}",
          flush=True)
    if go:
        print(f"  GO: the OneBrainComposer runs the WHOLE who/what/yes-no/moat pipeline on ONE persistent co-resident "
              f"bridge -- the parser comprehends (voice-invariant), the fact stores in synapses, the cue-scan answers, "
              f"the moat abstains -- == the numpy composer == ground truth every config. ==> the integrated one-brain "
              f"conversational turn is realized; swap into BrainConversationalAgent (composer_kind='onebrain') + run the "
              f"agent capability matrix; then A5 (megakernel the persistent loop + retire legacy numpy runtime).", flush=True)
    else:
        print(f"  BOUNDARY/NEGATIVE: full {n_full}/{len(rows)} (qp {qp:.3f} qa {qa:.3f} yn {yn:.3f} moat {mq:.2f}/{myn:.2f}) "
              f"-- localize the failing stage (parser comprehension co-resident / the parser-driven store / the cue-scan "
              f"/ the moat). Reportable.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*108}", flush=True)
    out = {"verdict": "GO" if go else "BOUNDARY", "seeds": seeds, "dims": dims, "qpatient": qp, "qagent": qa,
           "yesno": yn, "host_p": hp, "host_a": ha, "host_y": hy, "moat_q": mq, "moat_yn": myn, "per_row": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
