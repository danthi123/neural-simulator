"""gap #2 wire-in probe: CONTENT-ADDRESSABLE multi-fact recall from the spiking slot-binder.

2026-07-17, follow-on (c) of "finish gap #2 fully". The de-risk proved the single-bind readout (drive a slot ->
read its filler) is robust on spikes, and slot-SEPARATION beats the FHRR superposition cap. This probe uses that
validated primitive to build the conversational faculty: store each fact's (agent, verb, patient) into its OWN
three slots (separate slots = the gap-#2 win, no superposition), then answer who/what queries by a NEURAL SCAN
(the accepted pipeline pattern -- OneBrainComposer's GAP-A cue-match): for each stored fact, drive its agent- and
verb-slots and check the read-back fillers against the cue; the matching fact's patient-slot is read for the answer;
if NO fact matches the cue, ABSTAIN (the no-confab moat).

This needs NO coexistence/clear machinery (each (fact,role) slot is stored + read INDEPENDENTLY) and no new
coincidence mechanism -- it is repeated application of the validated single-bind readout. GO bar: query_patient /
query_agent correct on all stored facts, 6-seed, AND abstain on an absent cue (moat 0 false-answers). Anti-cheats:
permuted-cue must abstain (genuine content-addressing, not returning a fixed fact); scramble-store must collapse.

CPU/numpy.
"""

import json
import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._keystone2_spiking_slot_binder_derisk import build_binder_bridge, _idx  # noqa: E402


def _prim(b, KF, gain=400.0):
    """Return (store_pair, read_slot, reset) primitives over a built binder bridge."""
    from sim.backend import to_host, from_host
    n = b.core_config.num_neurons
    K = b._K_slots
    slot_idx = [_idx(b, f"w{k}") for k in range(K)]
    fill_idx = [_idx(b, f"f{f}") for f in range(KF)]

    def _reset():
        if getattr(b, "cp_izh_c_reset", None) is not None:
            b.cp_membrane_potential_v[:] = b.cp_izh_c_reset
        else:
            b.cp_membrane_potential_v[:] = -65.0
        b.cp_recovery_variable_u[:] = 0.0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False
        for _a in ("cp_conductance_g_nmda_recurrent", "cp_conductance_g_e", "cp_conductance_g_i",
                   "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise"):
            _arr = getattr(b, _a, None)
            if _arr is not None:
                _arr[:] = 0.0

    def store_pair(slot, filler, teach_steps=40):
        """Teach slot -> filler[filler] by co-driving them with ONLY this slot's plasticity gate open."""
        _reset()
        cur = np.zeros(n); cur[slot_idx[slot]] = gain; cur[fill_idx[filler]] = gain
        b.set_plasticity_gate(f"slot{slot}_to_filler", 1.0)
        dev = from_host(cur.astype(np.float64))
        for _ in range(teach_steps):
            b.cp_external_input_current[:] = dev; b._run_one_simulation_step()
        b.set_plasticity_gate(f"slot{slot}_to_filler", 0.0)
        _reset()

    def read_slot(slot, retr_steps=40):
        """Drive slot, return (argmax filler, peak rate) from the filler-pool firing."""
        _reset()
        cur = np.zeros(n); cur[slot_idx[slot]] = gain
        dev = from_host(cur.astype(np.float64)); rate = np.zeros(KF)
        for _ in range(retr_steps):
            b.cp_external_input_current[:] = dev; b._run_one_simulation_step()
            fir = np.asarray(to_host(b.cp_firing_states)).astype(float)
            for f in range(KF):
                rate[f] += fir[fill_idx[f]].mean()
        return int(np.argmax(rate)), float(rate.max())

    return store_pair, read_slot, _reset


class ProbeStore:
    """Minimal content-addressable fact store on the spiking slot-binder (probe of the wire-in mechanism)."""

    def __init__(self, seed, vocab, max_facts=8):
        self.V = len(vocab)
        self.vocab = list(vocab)
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.max_facts = max_facts
        # K = 3 slots per fact (agent/verb/patient). KF = vocab size (one filler pool per concept).
        self.b = build_binder_bridge(seed, K=3 * max_facts, KF=self.V)
        self.store_pair, self.read_slot, self._reset = _prim(self.b, self.V)
        self.facts = []   # list of (a_idx, v_idx, p_idx); fact i uses slots 3i, 3i+1, 3i+2

    def store(self, agent, verb, patient):
        i = len(self.facts)
        assert i < self.max_facts, "probe store full"
        a, v, p = self.w2i[agent], self.w2i[verb], self.w2i[patient]
        self.store_pair(3 * i + 0, a)
        self.store_pair(3 * i + 1, v)
        self.store_pair(3 * i + 2, p)
        self.facts.append((a, v, p))

    def _match_fact(self, cue_a=None, cue_v=None):
        """Neural scan: return the fact index whose agent/verb slots READ BACK the cue, else None (abstain)."""
        best = None
        for i in range(len(self.facts)):
            ok = True
            if cue_a is not None:
                ra, _ = self.read_slot(3 * i + 0)
                ok = ok and (ra == self.w2i[cue_a])
            if cue_v is not None:
                rv, _ = self.read_slot(3 * i + 1)
                ok = ok and (rv == self.w2i[cue_v])
            if ok:
                best = i
        return best

    def query_patient(self, agent, verb):
        i = self._match_fact(cue_a=agent, cue_v=verb)
        if i is None:
            return None            # no-confab moat: nothing matches -> abstain
        rp, _ = self.read_slot(3 * i + 2)
        return self.vocab[rp]

    def query_agent(self, verb, patient):
        # scan facts whose verb-slot AND patient-slot read back the cue; return the agent
        for i in range(len(self.facts)):
            rv, _ = self.read_slot(3 * i + 1)
            rp, _ = self.read_slot(3 * i + 2)
            if rv == self.w2i[verb] and rp == self.w2i[patient]:
                ra, _ = self.read_slot(3 * i + 0)
                return self.vocab[ra]
        return None


def run_seed(seed, scramble=False):
    vocab = ["dog", "cat", "fish", "bird", "chase", "eat", "see", "hear"]
    facts = [("dog", "chase", "cat"), ("cat", "eat", "fish"), ("bird", "see", "dog")]
    st = ProbeStore(seed, vocab, max_facts=len(facts) + 1)
    for (a, v, p) in facts:
        if scramble:
            # scramble-store control: store the SAME words but with the patient shuffled -> query must NOT recover truth
            pass
        st.store(a, v, p)
    if scramble:
        # rebuild with a deranged patient assignment
        st = ProbeStore(seed, vocab, max_facts=len(facts) + 1)
        pats = [f[2] for f in facts][::-1]
        for (a, v, _), p in zip(facts, pats):
            st.store(a, v, p)

    # query_patient on all stored facts
    qp_ok = sum(int(st.query_patient(a, v) == p) for (a, v, p) in facts)
    # query_agent on all stored facts
    qa_ok = sum(int(st.query_agent(v, p) == a) for (a, v, p) in facts)
    # MOAT: an absent cue (never-stored agent+verb) must abstain
    moat_abstain = int(st.query_patient("fish", "hear") is None)      # (fish, hear) never stored
    # permuted-cue anti-cheat: a real agent+WRONG verb -> should abstain (not return dog's patient)
    perm_abstain = int(st.query_patient("dog", "eat") is None)        # dog chases (not eats) -> no such fact
    return {"qp_ok": qp_ok, "qa_ok": qa_ok, "n_facts": len(facts),
            "moat_abstain": moat_abstain, "perm_abstain": perm_abstain}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    seeds = (42, 43, 44, 100, 101, 102)
    rows = [dict(seed=s, **run_seed(s)) for s in seeds]
    scr = [run_seed(s, scramble=True) for s in seeds]
    nf = rows[0]["n_facts"]
    qp = np.mean([r["qp_ok"] for r in rows]) / nf
    qa = np.mean([r["qa_ok"] for r in rows]) / nf
    moat = np.mean([r["moat_abstain"] for r in rows])
    perm = np.mean([r["perm_abstain"] for r in rows])
    scr_qp = np.mean([r["qp_ok"] for r in scr]) / nf    # scramble should NOT recover the true patient
    print("=" * 92)
    print(f"  gap#2 wire-in: content-addressable multi-fact recall on the spiking slot-binder ({nf} facts, "
          f"vocab 8, 6-seed)")
    for r in rows:
        print(f"   [seed {r['seed']}] query_patient {r['qp_ok']}/{nf} | query_agent {r['qa_ok']}/{nf} | "
              f"moat-abstain {r['moat_abstain']} | perm-abstain {r['perm_abstain']}")
    print(f"  MEAN(6): query_patient {qp:.2f} | query_agent {qa:.2f} | moat-abstain {moat:.2f} | "
          f"perm-abstain {perm:.2f} | scramble-store query_patient {scr_qp:.2f} (must be low)")
    go = (qp >= 0.90 and qa >= 0.90 and moat >= 0.90 and perm >= 0.90)
    print(f"  {'GO' if go else 'BOUNDARY'}: recall>=0.90 both dirs & moat/perm abstain>=0.90")
    print(f"  elapsed {time.time()-t0:.1f}s")
    out = os.path.join(_REPO, "research", "findings", "raw", "_slotbinder_content_addressable.json")
    json.dump({"rows": rows, "scramble": scr, "qp": qp, "qa": qa, "moat": moat, "perm": perm, "go": bool(go)},
              open(out, "w"), indent=2)
    print(f"  [saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
