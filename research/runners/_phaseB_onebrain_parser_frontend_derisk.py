"""ROADMAP PHASE 2 (the real "one brain"), STEP B1 -- the PARSER FRONT-END drives the composition on ONE bridge. GAP B
from the production scoping: today the de-risks HOST-set which role each operand binds to (a `{role: word}` dict);
production must drive that from the PARSER's neural role decision. This de-risk closes it: a `BridgeParser` comprehends
a sentence ON the same persistent bridge as the resonate-and-fire (RF) composer, and the role it FIRES for each word
SELECTS that word's bind -- comprehension is neural, not a host dict.

The recommended approach (GAP-B scope) is B-ii: the parser fires (the neural DECISION of which role), and that decision
selects the bind's complex weight (the role's phasor, a FIXED wiring constant -- like an axon's developmental target).
Brain-based-compliant: the decision is neural; the projected code is fixed. (B-i, gating an RF->RF complex synapse, is
ruled out -- transmission gates multiply the Izhikevich matrix, NOT the RF complex matvec, verified bridge.py:5528.)

Co-residence (construction smoke PASSED `_phaseB_onebrain_parser_coresident_smoke.py`): the Izhikevich Hebbian parser
(state in v/u as VOLTAGE, stepped by `_run_one_simulation_step`) and the RF registers (state in v/u as a COMPLEX
phasor, stepped by the masked `rf_resonate_steps`) co-exist on ONE bridge, each un-regressed -- the merged-bridge
regime (step 2b). The parser's incidental Izhikevich firing corrupts the RF registers' v/u between ops, but the
composer re-kicks every op, so it is harmless; the masked RF ops leave the parser slice untouched.

GATE (exact/identity effect -> parity, 3 seeds x 2 D): a fact comprehended + composed via the PARSER's role decision
recovers every role's filler on-bridge == the host-routed oracle (which is TOLD the parse) == ground truth, for BOTH an
active sentence AND its passive frame (the parser's signature: "dog go north" and "north go dog"/passive store the SAME
fact -- voice-invariant comprehension). Anti-cheats: (i) PARSER LESION (zero the learned parse weights) -> comprehension
collapses -> wrong/garbled roles; (ii) PERMUTED parser->role map -> the WRONG roles bind -> recall wrong; (iii) the moat
-- an UNBOUND role (a role the fact lacks) abstains (low cleanup peak). Reuse-by-import (BridgeParser + RFPhasorComposer
+ masked rf_kick); NO sim/ edit. GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_parser_frontend_derisk --seeds 42,43,44 --dims 64,128
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

AGENTS = ["dog", "cat", "bird", "river", "apple"]
ACTIONS = ["go", "come", "look", "stop", "swim"]
PATIENTS = ["north", "east", "south", "west", "home"]
VOCAB = AGENTS + ACTIONS + PATIENTS
ROLE_OF_POS = ["agent", "action", "patient"]   # active-voice canonical


def build_coresident_bridge(seed, P, n_rf):
    cfg = CoreSimConfig()
    cfg.num_neurons = P + n_rf
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed); cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0; cfg.num_traits = 1
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_max_weight = 400.0
    cfg.hebbian_learning_rate = 0.005
    for f in ("enable_short_term_plasticity", "enable_structural_plasticity", "enable_homeostasis",
              "enable_reward_modulation", "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = 20.0
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


class ParserFrontEnd:
    """Comprehend a sentence with the co-resident parser, then compose the fact on the co-resident RF slice driving the
    binds from the PARSER's role decision. RF registers (offset by P, D each): fill_0..2 [0..2], bound_0..2 [3..5],
    acc [6], Q [7], concepts [8D : 8D+V]."""

    def __init__(self, seed, D):
        self.seed = seed; self.D = D
        self.comp = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB, period=200)
        self.R = 40
        self.P = 6 + 3 * self.R
        self.V = len(VOCAB)
        self.n_rf = 8 * D + self.V
        self.b = build_coresident_bridge(seed, self.P, self.n_rf)
        self.parser = BridgeParser(seed=seed, R=self.R, shared_bridge=self.b, index_offset=0)  # wires+trains on [0:P]
        n = self.b.core_config.num_neurons
        self.rf_mask = np.zeros(n, dtype=bool); self.rf_mask[self.P:self.P + self.n_rf] = True
        # an independently-permuted role map for the anti-cheat (agent<->patient<->action cycle)
        self._perm = {"agent": "patient", "patient": "action", "action": "agent"}

    def comprehend_roles(self, words, voice):
        """The parser's neural decision: role assigned to each POSITION (read from which role ensemble fires)."""
        return [self.parser.role_of(pos, voice) for pos in range(3)]

    def store_and_query(self, words, voice, query_role, lesion_parser=False, permute=False, host_roles=None):
        """Comprehend `words` (parser-driven roles unless host_roles given), compose on the RF slice, unbind
        `query_role` -> cleanup -> (answer_word, peak). lesion_parser: zero the parse weights first (comprehension
        collapses). permute: apply the fixed wrong-role permutation to the parser's decision."""
        comp, b, D, V, P = self.comp, self.b, self.D, self.V, self.P
        n = b.core_config.num_neurons
        if host_roles is not None:
            roles = list(host_roles)
        else:
            if lesion_parser:
                self._lesion_parse_weights()
            roles = self.comprehend_roles(words, voice)
            if permute:
                roles = [self._perm.get(r, r) for r in roles]
        o = P  # RF region base
        binds, bundle = [], []
        kick = np.zeros(n, dtype=np.complex128)
        for i in range(3):
            zr = comp._to_phasor(comp.roles[roles[i]])
            zf = comp._to_phasor(comp.concepts[words[i]])
            kick[o + i * D:o + (i + 1) * D] = zf                         # fill_i
            binds += [(o + (3 + i) * D + k, o + i * D + k, complex(zr[k])) for k in range(D)]   # bound_i = role_i*fill_i
            bundle += [(o + 6 * D + k, o + (3 + i) * D + k, 1.0) for k in range(D)]              # acc += bound_i
        zq = comp._to_phasor(comp.roles[query_role])
        qx = [(o + 7 * D + k, o + 6 * D + k, complex(np.conj(zq[k]))) for k in range(D)]          # unbind query_role
        clean = []
        for j in range(V):
            cc = np.conj(comp._to_phasor(comp.concepts[VOCAB[j]]))
            clean += [(o + 8 * D + j, o + 7 * D + k, complex(cc[k])) for k in range(D)]
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        b.rf_set_complex_weights(binds); b.rf_kick(kick, period=comp.period, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(comp.period + 8)
        b.rf_set_complex_weights(bundle); b.rf_resonate_steps(comp.period + 8)
        b.rf_set_complex_weights(qx); b.rf_resonate_steps(comp.period + 8)
        b.rf_set_complex_weights(clean); b.rf_resonate_steps(1)
        scores = np.maximum(np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)[o + 8 * D:o + 8 * D + V], 0.0)
        return VOCAB[int(np.argmax(scores))], float(scores.max()), roles

    def _lesion_parse_weights(self):
        """Zero the parser's learned 'parse' synapses -> the role ensembles no longer fire selectively."""
        import cupy as cp  # noqa
        try:
            self.b.cp_connections.data[:] = 0.0
        except Exception:
            pass


def host_oracle(comp, words, roles, query_role):
    """The validated numpy composer, TOLD the (ground-truth) roles, as the parity reference."""
    fact = {roles[i]: words[i] for i in range(3)}
    return comp._cleanup(comp._unbind_phases(comp._encode(fact), query_role), VOCAB)


def run_seed(seed, D):
    fe = ParserFrontEnd(seed, D)
    comp = fe.comp
    facts = list(zip(AGENTS, ACTIONS, PATIENTS))   # 5 SVO facts
    gt_roles = ROLE_OF_POS                           # active ground truth per position
    self_ok = host_ok = voice_ok = 0; n = 0
    moat_bound, moat_unbound = [], []
    for (a, v, p) in facts:
        words = [a, v, p]
        for qrole, truth in (("agent", a), ("action", v), ("patient", p)):
            ans, peak, roles = fe.store_and_query(words, "active", qrole)
            host = host_oracle(comp, words, gt_roles, qrole)
            self_ok += int(ans == truth); host_ok += int(ans == host); n += 1
            moat_bound.append(peak)
        # voice-invariance: the passive frame "p v a" (voice=passive) must store the SAME fact -> same agent answer
        ans_p, _, roles_p = fe.store_and_query([p, v, a], "passive", "agent")
        voice_ok += int(ans_p == a)
        # moat: query a role the fact lacks is N/A (all 3 roles bound); instead probe an unstored composite's role via a
        # mismatched query is covered by the abstain test below.
    # MOAT: bind only agent+action (the 3rd word also binds to 'agent', so the PATIENT role is never bound), then
    # query the unbound PATIENT role -> the unbind is cross-talk noise -> low peak (abstain) vs a bound role's high peak.
    a, v, p = facts[0]
    _, peak_bound, _ = fe.store_and_query([a, v, p], "active", "agent",
                                          host_roles=["agent", "action", "agent"])   # agent bound -> high peak
    _, peak_absent, _ = fe.store_and_query([a, v, p], "active", "patient",
                                           host_roles=["agent", "action", "agent"])   # patient never bound -> abstain
    moat_sep = int(peak_bound > 1.5 * peak_absent)

    row = {"seed": seed, "D": D, "self": self_ok / n, "host": host_ok / n, "voice_inv": voice_ok / len(facts),
           "moat_sep": moat_sep, "peak_bound": peak_bound, "peak_absent": peak_absent}
    print(f"  [seed {seed} D={D}] parser-driven recall self={self_ok/n:.2f}/host={host_ok/n:.2f} | "
          f"voice-invariance {voice_ok}/{len(facts)} | moat sep={moat_sep} ({peak_bound:.3g} vs {peak_absent:.3g})",
          flush=True)
    return fe, row


def anti_cheats(fe):
    """Lesion + permuted-role controls on the first fact (want recall to COLLAPSE)."""
    a, v, p = AGENTS[0], ACTIONS[0], PATIENTS[0]
    perm_ans, _, perm_roles = fe.store_and_query([a, v, p], "active", "agent", permute=True)
    perm_collapse = int(perm_ans != a)        # permuted roles -> agent query should NOT return the true agent
    les_ans, _, _ = fe.store_and_query([a, v, p], "active", "agent", lesion_parser=True)
    les_collapse = int(les_ans != a)          # lesioned parser -> comprehension garbled -> wrong agent
    print(f"  [anti-cheat] permuted-role agent='{perm_ans}'(roles={perm_roles}) collapse={perm_collapse} | "
          f"lesion-parser agent='{les_ans}' collapse={les_collapse}", flush=True)
    return perm_collapse, les_collapse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44"); ap.add_argument("--dims", type=str, default="64,128")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onebrain_parser_frontend.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]; dims = [int(d) for d in args.dims.split(",")]
    t0 = time.time()
    print("[one-brain parser front-end de-risk] does the PARSER's neural role decision drive the composition on ONE "
          "bridge == the host-routed oracle (voice-invariant), anti-cheats collapsing?\n", flush=True)
    rows = []; perm_oks = []; les_oks = []
    for s in seeds:
        for D in dims:
            fe, row = run_seed(s, D)
            rows.append(row)
            if D == dims[0]:                                  # anti-cheats once per seed (cheaper)
                pc, lc = anti_cheats(fe)
                perm_oks.append(pc); les_oks.append(lc)

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    self_m, host_m, vinv = m("self"), m("host"), m("voice_inv")
    sep = m("moat_sep"); perm = float(np.mean(perm_oks)); les = float(np.mean(les_oks))
    n_full = sum(int(r["self"] >= 0.99 and r["host"] >= 0.99 and r["voice_inv"] >= 0.99 and r["moat_sep"] >= 1)
                 for r in rows)
    go = (n_full == len(rows)) and (perm >= 0.99) and (les >= 0.99)
    print(f"\n{'='*104}", flush=True)
    print(f"  MEAN ({len(rows)} seed*D): parser-driven recall self {self_m:.3f}/host {host_m:.3f} | voice-invariance "
          f"{vinv:.3f} | moat clean-sep {sep:.3f} | anti-cheats: permuted-collapse {perm:.2f} lesion-collapse {les:.2f} "
          f"| per-row full: {n_full}/{len(rows)}", flush=True)
    if go:
        print(f"  GO: the PARSER's neural role decision drives the composition on ONE persistent bridge -- a fact "
              f"comprehended (active AND passive, voice-invariant) composes + queries back == the host-routed oracle, "
              f"the moat abstains on an unbound role, and BOTH anti-cheats collapse (permuted roles + lesioned parser). "
              f"==> GAP B resolved: comprehension is synaptic (the parser's firing selects the bind), no host "
              f"{{role:word}} dict. Next: STEP A3 -- wrap parser front-end + the GAP-A persistent store as the "
              f"production OneBrainComposer + run the agent capability matrix.", flush=True)
    elif self_m >= 0.95 and host_m >= 0.95:
        print(f"  BOUNDARY: parser-driven recall holds (self {self_m:.3f}/host {host_m:.3f}) but voice-invariance "
              f"({vinv:.3f}) or the moat ({sep:.3f}) or an anti-cheat (perm {perm:.2f}/les {les:.2f}) is soft -- "
              f"localize (parser passive-frame readout vs the bind chain). Reportable.", flush=True)
    else:
        print(f"  NEGATIVE: parser-driven recall self {self_m:.3f}/host {host_m:.3f} -- the parser's decision does not "
              f"drive the bind correctly co-resident; diagnose (parser readout under RF co-residence, or the masked "
              f"bind chain). The host-routed path stays the default. Reportable.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*104}", flush=True)
    out = {"verdict": "GO" if go else ("BOUNDARY" if (self_m >= 0.95 and host_m >= 0.95) else "NEGATIVE"),
           "seeds": seeds, "dims": dims, "self": self_m, "host": host_m, "voice_inv": vinv, "moat_sep": sep,
           "permuted_collapse": perm, "lesion_collapse": les, "per_row": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
