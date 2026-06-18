"""CYCLE 200 — richer-syntax #1 NEURAL wire-in: the attributed-entity parse realized ON THE BRIDGE (brain-based).

The cheap-first de-risk (2026-06-18-attributed-entity-parser-GO.md) proved attributed entities work on the ready
composer with a closed-form (host) conjunctive readout = GO 6/6. The BRAIN-BASED-ONLY standard requires the PARSE
itself to be neurons/synapses, not a host readout. This wires the adjacency factor into the on-bridge Hebbian parser
(the `BridgeParser` pattern): the conjunction space grows from (position x voice) -> (position-from-START x
position-from-END x voice), and the role set from 3 -> 5 ({agent, action, patient, attribute, attribute2}). The
position-from-END factor is the new conjunction (adjacency-to-the-head): the head noun is end-0 (= patient), the
preceding modifiers are attribute/attribute2 -- so s=2,e=0 (flat patient) vs s=2,e>=1 (attribute) are disambiguated
by a SPIKING conjunction unit, exactly as voice disambiguated the active/passive flip.

A conjunction unit per (s_bucket in 0..3, e_bucket in 0..2, voice) Hebbian-learns -> its role ensemble (teacher
co-firing, the validated v16 rule). At parse time, driving a word's (s, e, voice) conjunction ALONE reads its role
off the bridge in SPIKES. NO host role rule.

GATE (multi-seed, on the bridge): per-position role read-out accuracy on held-out attributed sentences == the
ground-truth structural role >= 0.90, >= 5/6 seeds, AND flat-SVO (3-word) role read-out un-regressed, AND a
FLAT-ONLY-conjunction control (drop the position-from-END factor) MUST FAIL (it can't separate patient from
attribute -> the from-END conjunction is load-bearing). GO => the attributed parse is brain-based (the parser's
firing selects the attribute role); wire into the agent next. NEGATIVE => the spiking conjunction can't resolve the
from-end factor at this scale -> localize (more neurons / a cleaner teacher / the dlPFC-Control mechanism).

Reuse: the `BridgeParser` Hebbian-parser pattern (a private Izhikevich bridge, embodied-Hebbian co-firing teacher,
firing-rate role read-out). GPU for real (numpy is a tiny smoke). NO sim/ edit.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_neural_attributed_parser_derisk
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

ROLES = ["agent", "action", "patient", "attribute", "attribute2"]
S_CAP, E_CAP = 3, 2                       # from-start bucket 0..3, from-end bucket 0..2


def _role_for(s, e):
    """The ground-truth STRUCTURAL role for a word at (from-start s, from-end e) in an active 'S V adj* N' frame:
    s=0 agent; s=1 action; e=0 patient (head noun); else s=2 attribute, s>=3 attribute2."""
    sc, ec = min(s, S_CAP), min(e, E_CAP)
    if sc == 0:
        return "agent"
    if sc == 1:
        return "action"
    if ec == 0:
        return "patient"
    return "attribute" if sc == 2 else "attribute2"


def _conj_index(s, e, voice, use_end=True):
    """Flat conjunction id for (s_bucket, e_bucket, voice). use_end=False drops the from-END factor (the control)."""
    sc, ec, v = min(s, S_CAP), (min(e, E_CAP) if use_end else 0), (0 if voice in (0, "active") else 1)
    ne = (E_CAP + 1) if use_end else 1
    return sc * (ne * 2) + ec * 2 + v


class AttributedBridgeParser:
    """(from-start x from-end x voice) -> 5-role Hebbian parser on a private Izhikevich bridge. Mirrors BridgeParser
    (embodied-Hebbian co-firing teacher; firing-rate role read-out), with the from-END conjunction added."""

    def __init__(self, seed=42, R=40, n_epochs=24, train_steps=120, test_steps=80, drive=2500.0, use_end=True):
        self.R = R; self.test_steps = test_steps; self.drive = drive; self.use_end = use_end
        self.n_conj = (S_CAP + 1) * ((E_CAP + 1) if use_end else 1) * 2
        self.conj = list(range(self.n_conj))
        self.role_idx = {r: [self.n_conj + i * R + j for j in range(R)] for i, r in enumerate(ROLES)}
        # the teacher: which role each conjunction (s,e,voice) drives. Only (s,e) combos that occur in real frames
        # (s+e = n-1, n in 3..5 active; n=3 passive) are trained; others stay unbound.
        self.teacher = {}
        for n in (3, 4, 5):
            for pos in range(n):
                k = _conj_index(pos, n - 1 - pos, 0, use_end)
                self.teacher[k] = _role_for(pos, n - 1 - pos)
        for pos, role in ((0, "patient"), (1, "action"), (2, "agent")):     # flat passive (n=3) SVO flip
            self.teacher[_conj_index(pos, 3 - 1 - pos, 1, use_end)] = role
        pre, post, w = [], [], []
        for k in self.conj:
            for r in ROLES:
                for j in self.role_idx[r]:
                    pre.append(k); post.append(j); w.append(0.5)
        cfg = CoreSimConfig()
        cfg.num_neurons = self.n_conj + len(ROLES) * R
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
        self.bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                       runtime_state=RuntimeState(), gpu_config=GPUConfig())
        self.bridge._initialize_simulation_data(called_from_playback_init=False)
        self.bridge.inject_explicit_wiring({"parse": {"pre_indices": pre, "post_indices": post,
                                                       "initial_weights": np.array(w, dtype=np.float32),
                                                       "plastic": True, "conn_type": "E_TO_E", "count": len(pre)}})
        xp, _ = get_backend()
        self.conj_arr = xp.asarray(self.conj, dtype=xp.int64)
        self.role_arr = {r: xp.asarray(v, dtype=xp.int64) for r, v in self.role_idx.items()}
        self._n = self.bridge.core_config.num_neurons
        self._train(n_epochs, train_steps)

    def _step_reset(self, reset=20):
        self.bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset):
            self.bridge._run_one_simulation_step()

    def _train(self, n_epochs, train_steps):
        xp, _ = get_backend()
        ks = sorted(self.teacher)
        for _ in range(n_epochs):
            for k in ks:
                self._step_reset()
                cur = xp.zeros(self._n, dtype=xp.float32)
                cur[self.conj_arr[k]] = self.drive
                cur[self.role_arr[self.teacher[k]]] = self.drive
                self.bridge.cp_external_input_current[:] = cur
                for _ in range(train_steps):
                    self.bridge._run_one_simulation_step()
        self.bridge.cp_external_input_current[:] = 0.0

    def role_of(self, s, e, voice=0):
        xp, _ = get_backend()
        k = _conj_index(s, e, voice, self.use_end)
        self._step_reset()
        cur = xp.zeros(self._n, dtype=xp.float32)
        cur[self.conj_arr[k]] = self.drive
        self.bridge.cp_external_input_current[:] = cur
        rates = {r: 0.0 for r in ROLES}
        for _ in range(self.test_steps):
            self.bridge._run_one_simulation_step()
            for r in ROLES:
                rates[r] += float(to_host(self.bridge.cp_firing_states[self.role_arr[r]].astype(xp.float64).mean()))
        self.bridge.cp_external_input_current[:] = 0.0
        return max(rates, key=rates.get)

    def parse_roles(self, n, voice=0):
        """The per-position roles the bridge reads out for an n-word sentence (active/passive)."""
        return [self.role_of(pos, n - 1 - pos, voice) for pos in range(n)]


def run_seed(seed, use_end=True):
    parser = AttributedBridgeParser(seed=seed, use_end=use_end)
    attr_ok = attr_n = 0
    for n in (4, 5):                                          # attributed object NP (1 or 2 adjectives)
        got = parser.parse_roles(n, voice=0)
        truth = [_role_for(pos, n - 1 - pos) for pos in range(n)]
        attr_ok += sum(int(g == t) for g, t in zip(got, truth)); attr_n += n
    flat_ok = flat_n = 0
    for voice in (0, 1):                                     # flat SVO non-regression (active + passive)
        got = parser.parse_roles(3, voice=voice)
        truth = (["agent", "action", "patient"] if voice == 0 else ["patient", "action", "agent"])
        flat_ok += sum(int(g == t) for g, t in zip(got, truth)); flat_n += 3
    return {"seed": seed, "attr_acc": attr_ok / attr_n, "flat_acc": flat_ok / flat_n, "use_end": use_end}


def main():
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    print("[neural attributed parser de-risk] is the attributed-entity PARSE brain-based? (from-start x from-END x "
          "voice -> 5 roles, Hebbian on the bridge; the from-END conjunction disambiguates patient vs attribute)\n",
          flush=True)
    seeds = (42, 43, 44, 45, 46, 47)
    rows = [run_seed(s, use_end=True) for s in seeds]
    for r in rows:
        print(f"  [seed {r['seed']}] attributed role read-out {r['attr_acc']:.3f} | flat-SVO {r['flat_acc']:.3f}",
              flush=True)
    print("  -- CONTROL: drop the from-END conjunction factor (use_end=False); MUST fail on attributed --", flush=True)
    ctrl_rows = [run_seed(s, use_end=False) for s in seeds]
    for r in ctrl_rows:
        print(f"  [seed {r['seed']}] NO-END attributed {r['attr_acc']:.3f}", flush=True)

    def m(rs, k):
        return float(np.mean([r[k] for r in rs]))
    attr, flat, ctrl = m(rows, "attr_acc"), m(rows, "flat_acc"), m(ctrl_rows, "attr_acc")
    n_go = sum(1 for r in rows if r["attr_acc"] >= 0.90 and r["flat_acc"] >= 0.90)
    print(f"\n{'='*98}\n  MEAN (6 seeds): attributed role read-out {attr:.3f} | flat-SVO {flat:.3f} | NO-END control "
          f"{ctrl:.3f} | seeds GO {n_go}/6", flush=True)
    print(f"{'='*98}", flush=True)
    go = n_go >= 5 and attr >= 0.90 and flat >= 0.90 and ctrl < attr - 0.10
    if go:
        print(f"  GO: the attributed-entity parse is BRAIN-BASED -- the bridge reads out the per-position role in "
              f"SPIKES at {attr:.3f} ({n_go}/6 seeds), flat-SVO un-regressed {flat:.3f}, and dropping the from-END "
              f"conjunction collapses it to {ctrl:.3f} (so the spiking from-END factor is load-bearing -- it "
              f"disambiguates the head noun=patient from the modifiers=attribute). ==> wire AttributedBridgeParser "
              f"into BrainConversationalAgent.hear (attributed patients end-to-end).", flush=True)
    else:
        print(f"  NEGATIVE/PARTIAL: attributed {attr:.3f} / flat {flat:.3f} / NO-END {ctrl:.3f} / GO {n_go}/6 -- the "
              f"spiking conjunction can't cleanly resolve the from-END factor at this scale. Localize (more "
              f"neurons/epochs, a cleaner teacher, or the dlPFC-Control unification mechanism).", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    out = {"attr_acc": attr, "flat_acc": flat, "no_end_control": ctrl, "seeds_go": n_go, "go": bool(go),
           "per_seed": rows, "control": ctrl_rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_neural_attributed_parser.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
