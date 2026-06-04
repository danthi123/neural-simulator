"""Stage 2c + 3c: the unified agent on the BIOLOGICAL resonate-and-fire substrate.

Stages 2a/2b/3 built the agent on `spiking_phasor_fhrr` (genuine time-stepped spiking, Orchard integrator
neurons). This module re-runs it on `resonate_fire_fhrr` -- the genuine biological **resonate-and-fire** neuron
(Izhikevich 2001 / Frady-Sommer 2019): bind/unbind/bundle on resonate-and-fire neurons, role clean-up by the
`ResonateFireTPAM` complex-valued **attractor network** (the biological CA3 pattern completion; abstention = a
basin-of-attraction property). 2c validated the robust core at core vocab; 3c extends to the FULL benchmark
(two-attribute + embedded clauses) -- the patient decode is substrate-independent phasor arithmetic on the
membrane-state bundle, identical to the validated scaffold agent, so it is duplicated here (zero change to the
validated scaffold path).

Scope (honest): the resonate-and-fire substrate steps a ~1000-step cycle per operation, so this validates at the
frozen test set's CORE vocabulary on CPU. The reduced vocabulary is an EASIER clean-up than 320. The full-320 /
GPU run is the reserved-GPU build.

Pre-registered verdict: the full benchmark (flat / one- / two-attribute / depth-1 clause / who / abstain)
reproduces on the biological substrate at core vocab; the no-confabulation moat holds.

  SIM_BACKEND=numpy python -m research.runners._rf_unified_agent_probe
"""
from __future__ import annotations
import json

import numpy as np

from research.runners.resonate_fire_fhrr import (
    ResonateFireFHRR, ResonateFireTPAM, rf_bind, rf_unbind, rf_bundle, _to_phasor, CYCLE_STEPS)
from research.runners.spiking_unified_agent import _unit, _phasor_sim
from research.runners.nested_composition_agent import Clause
from research.runners.unified_agent_benchmark import (
    CORE_NOUNS, CORE_VERBS, CORE_ADJS, FACTS_FLAT, FACTS_1ATTR, FACTS_2ATTR, FACTS_CLAUSE,
    WHO_QUERIES, ABSTAIN_QUERIES)

ROLES = ("AGENT", "ACTION", "PATIENT")
ABSTAIN_ACTIVE_FRAC = 0.15      # below this settled-active fraction -> the network collapsed -> abstain


class RFUnifiedAgent:
    """The FULL unified agent on the resonate-and-fire biological substrate + TPAM attractor cleanup: store SVO
    facts (flat / attributed / clause patients), answer who/what, abstain. Storage + role matching use the
    resonate-and-fire neurons; the patient decode is phasor arithmetic on the membrane-state (complex-sum)
    bundle -- the same recursive resonator decode validated on the scaffold substrate."""

    def __init__(self, nouns, verbs, adjs, n_dim=512, seed=42):
        self.nouns, self.verbs, self.adjs = list(nouns), list(verbs), list(adjs)
        self.D, self.seed, self.n_restarts = int(n_dim), int(seed), 16
        self.t_steps = CYCLE_STEPS
        self.net = ResonateFireFHRR(n_dim, np.random.default_rng(seed))
        self.noun_sym = {w: self.net.random_symbol() for w in self.nouns}
        self.verb_sym = {w: self.net.random_symbol() for w in self.verbs}
        self.adj_sym = {w: self.net.random_symbol() for w in self.adjs}
        self.role_sym = {r: self.net.random_symbol() for r in ROLES}
        self.noun_tpam = ResonateFireTPAM([self.noun_sym[w] for w in self.nouns])
        self.verb_tpam = ResonateFireTPAM([self.verb_sym[w] for w in self.verbs])
        # phasor codebooks for the patient decode (resonator + cleanup)
        self.NMAT = np.stack([self._ph(self.noun_sym[w]) for w in self.nouns], axis=1)
        self.AMAT = (np.stack([self._ph(self.adj_sym[w]) for w in self.adjs], axis=1) if self.adjs else None)
        self.VMAT = np.stack([self._ph(self.verb_sym[w]) for w in self.verbs], axis=1)
        self.role_ph = {r: self._ph(self.role_sym[r]) for r in ROLES}
        self.kb, self.kb_complex, self.facts = [], [], []

    def _ph(self, spikes):
        return _to_phasor(spikes, self.t_steps)

    # --- storage (resonate-and-fire bind/bundle + membrane-state complex-sum bundle) ---
    def _filler_phasor(self, x):
        if isinstance(x, Clause):
            return _unit(self._clause_sum(x))
        if isinstance(x, tuple):
            mods = x[0] if isinstance(x[0], tuple) else (x[0],)
            v = self._ph(self.noun_sym[x[1]])
            for a in mods:
                v = v * self._ph(self.adj_sym[a])
            return v
        return self._ph(self.noun_sym[x])

    def _clause_sum(self, clause):
        return (self.role_ph["AGENT"] * self._filler_phasor(clause.agent)
                + self.role_ph["ACTION"] * self._ph(self.verb_sym[clause.action])
                + self.role_ph["PATIENT"] * self._filler_phasor(clause.patient))

    def _patient_filler(self, patient):
        if isinstance(patient, Clause):
            from research.runners.spiking_phasor_fhrr import phases_to_spikes
            return phases_to_spikes(np.mod(np.angle(self._clause_sum(patient)) / (2 * np.pi), 1.0), self.t_steps)
        if isinstance(patient, tuple):
            mods = patient[0] if isinstance(patient[0], tuple) else (patient[0],)
            v = self.noun_sym[patient[1]]
            for a in mods:
                v = rf_bind(self.adj_sym[a], v)
            return v
        return self.noun_sym[patient]

    def learn(self, agent, action, patient):
        bound = [rf_bind(self.role_sym["AGENT"], self.noun_sym[agent]),
                 rf_bind(self.role_sym["ACTION"], self.verb_sym[action]),
                 rf_bind(self.role_sym["PATIENT"], self._patient_filler(patient))]
        self.kb.append(rf_bundle(bound))
        self.kb_complex.append(sum(self._ph(b) for b in bound))
        self.facts.append((agent, action, patient))

    # --- role matching (resonate-and-fire unbind + TPAM attractor cleanup) ---
    def _role(self, bundle, role, tpam, names):
        z, active_frac = tpam.settle(rf_unbind(bundle, self.role_sym[role]))
        if active_frac < ABSTAIN_ACTIVE_FRAC:
            return None
        return names[int(np.argmax(np.abs(tpam.s.conj().T @ z)))]

    # --- patient decode (phasor arithmetic on the membrane state -- the validated recursive resonator decode) ---
    def _resonator3(self, p):
        rng = np.random.default_rng(self.seed + 11)
        Ma = self.AMAT.shape[1]
        cbs = [self.AMAT, self.AMAT, self.NMAT]
        best = None
        for _ in range(self.n_restarts):
            est = [_unit(self.AMAT.sum(1) + 0.7 * self.AMAT[:, rng.integers(Ma)]),
                   _unit(self.AMAT.sum(1) + 0.7 * self.AMAT[:, rng.integers(Ma)]),
                   _unit(self.NMAT.sum(1))]
            for _ in range(150):
                new = []
                for i in range(3):
                    o = np.ones(self.D, dtype=complex)
                    for j in range(3):
                        if j != i:
                            o = o * est[j]
                    new.append(_unit(cbs[i] @ (cbs[i].conj().T @ (p * np.conj(o)))))
                est = new
            resid = _phasor_sim(p, est[0] * est[1] * est[2])
            a1 = int(np.argmax(np.abs(self.AMAT.conj().T @ est[0])))
            a2 = int(np.argmax(np.abs(self.AMAT.conj().T @ est[1])))
            n = int(np.argmax(np.abs(self.NMAT.conj().T @ est[2])))
            if best is None or resid > best[0]:
                best = (resid, tuple(sorted({a1, a2})), n)
        return best[1], best[2], best[0]

    @staticmethod
    def _cleanup_phasor(p, MAT, names):
        if MAT is None:
            return None, 0.0
        ov = np.abs(MAT.conj().T @ _unit(p))
        k = int(np.argmax(ov))
        return names[k], float(ov[k] / p.shape[0])

    def _decode_filler(self, p, depth=0):
        p = _unit(p)
        margin = 0.05
        if depth < 3 and self.VMAT is not None:
            ac, vconf = self._cleanup_phasor(p * np.conj(self.role_ph["ACTION"]), self.VMAT, self.verbs)
            if vconf >= 0.18:
                ag = self._decode_filler(p * np.conj(self.role_ph["AGENT"]), depth + 1)
                p_clean = p
                if ag in self.noun_sym and ac in self.verb_sym:
                    p_clean = (p - self.role_ph["AGENT"] * self._ph(self.noun_sym[ag])
                               - self.role_ph["ACTION"] * self._ph(self.verb_sym[ac]))
                pt = self._decode_filler(p_clean * np.conj(self.role_ph["PATIENT"]), depth + 1)
                return f"{ag} {ac} {pt}"
        nidx = int(np.argmax(np.abs(self.NMAT.conj().T @ p)))
        sim_flat = _phasor_sim(p, self.NMAT[:, nidx])
        flat_noun = self.nouns[nidx]
        one = None
        if self.AMAT is not None:
            for ai in range(self.AMAT.shape[1]):
                ni = int(np.argmax(np.abs(self.NMAT.conj().T @ (p * np.conj(self.AMAT[:, ai])))))
                sim = _phasor_sim(p, self.AMAT[:, ai] * self.NMAT[:, ni])
                if one is None or sim > one[0]:
                    one = (sim, self.adjs[ai], self.nouns[ni])
        two = None
        best_simple = max(sim_flat, one[0] if one is not None else 0.0)
        if depth == 0 and self.AMAT is not None and self.AMAT.shape[1] >= 2 and best_simple < 0.5:
            adj_idx, noun_idx, sim_two = self._resonator3(p)
            two = (sim_two, [self.adjs[i] for i in adj_idx], self.nouns[noun_idx])
        choice, score = "flat", sim_flat
        if one is not None and one[0] > score + margin:
            choice, score = "one", one[0]
        if two is not None and two[0] > score + margin:
            choice, score = "two", two[0]
        if choice == "two":
            return " ".join(two[1] + [two[2]])
        if choice == "one":
            return f"{one[1]} {one[2]}"
        return flat_noun

    def _decode_patient(self, idx, agent, action):
        clean_bp = (self.kb_complex[idx]
                    - self.role_ph["AGENT"] * self._ph(self.noun_sym[agent])
                    - self.role_ph["ACTION"] * self._ph(self.verb_sym[action]))
        return self._decode_filler(clean_bp * np.conj(self.role_ph["PATIENT"]), depth=0)

    def query_patient(self, agent, action):
        for i, b in enumerate(self.kb):
            if (self._role(b, "AGENT", self.noun_tpam, self.nouns) == agent
                    and self._role(b, "ACTION", self.verb_tpam, self.verbs) == action):
                return self._decode_patient(i, agent, action)
        return None

    def query_agent(self, action, patient):
        for b in self.kb:
            if (self._role(b, "ACTION", self.verb_tpam, self.verbs) == action
                    and self._role(b, "PATIENT", self.noun_tpam, self.nouns) == patient):
                return self._role(b, "AGENT", self.noun_tpam, self.nouns)
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Unified agent FULL benchmark on the biological resonate-and-fire "
                                             "substrate.")
    ap.add_argument("--full-vocab", action="store_true", help="full 320-concept vocab (default: core vocab)")
    ap.add_argument("--n-dim", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.full_vocab:
        from research.runners.unified_agent_benchmark import build_vocab
        nouns, verbs, adjs = build_vocab()
    else:
        nouns, verbs, adjs = list(CORE_NOUNS), list(CORE_VERBS), list(CORE_ADJS)

    print("=== unified agent FULL benchmark on the BIOLOGICAL resonate-and-fire substrate + TPAM (stage 3c) ===",
          flush=True)
    print(f"  vocab: {len(nouns)} nouns, {len(verbs)} verbs, {len(adjs)} adjs; cycle={CYCLE_STEPS}; "
          f"D={args.n_dim}\n", flush=True)
    agent = RFUnifiedAgent(nouns, verbs, adjs, n_dim=args.n_dim, seed=args.seed)
    for ag, ac, pa in FACTS_FLAT + FACTS_1ATTR + FACTS_2ATTR + FACTS_CLAUSE:
        agent.learn(ag, ac, pa)

    def _render(pa):
        if isinstance(pa, Clause):
            return f"{_render(pa.agent)} {pa.action} {_render(pa.patient)}"
        if isinstance(pa, tuple):
            mods = pa[0] if isinstance(pa[0], tuple) else (pa[0],)
            return " ".join(sorted(mods, key=adjs.index) + [pa[1]])
        return pa

    res, wrong = {}, []
    res["flat"] = [sum(agent.query_patient(a, v) == p for a, v, p in FACTS_FLAT), len(FACTS_FLAT)]
    res["1-attribute"] = [sum(agent.query_patient(a, v) == f"{p[0]} {p[1]}" for a, v, p in FACTS_1ATTR),
                          len(FACTS_1ATTR)]
    res["2-attribute"] = [sum(agent.query_patient(a, v) == _render(p) for a, v, p in FACTS_2ATTR),
                          len(FACTS_2ATTR)]
    res["clause-depth1"] = [sum(agent.query_patient(a, v) == _render(p) for a, v, p in FACTS_CLAUSE),
                            len(FACTS_CLAUSE)]
    res["who-query"] = [sum(agent.query_agent(v, p) == w for v, p, w in WHO_QUERIES), len(WHO_QUERIES)]
    res["abstain"] = [sum(agent.query_patient(a, v) is None for a, v in ABSTAIN_QUERIES), len(ABSTAIN_QUERIES)]
    for a, v, p in FACTS_2ATTR + FACTS_CLAUSE:
        got = agent.query_patient(a, v)
        if got != _render(p):
            wrong.append((f"{a} {v}", got, _render(p)))

    for c in res:
        print(f"  {c:<14} {res[c][0]}/{res[c][1]} = {res[c][0]/res[c][1]*100:.0f}%", flush=True)
    gok = sum(v[0] for v in res.values())
    gtot = sum(v[1] for v in res.values())
    full = all(v[0] == v[1] for v in res.values())
    print(f"\n  OVERALL full benchmark (biological substrate): {gok}/{gtot} = {gok/gtot*100:.1f}%", flush=True)
    verdict = "RESOLVES" if full else ("PARTIAL" if gok / gtot >= 0.8 else "DOES_NOT_RESOLVE")
    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    for w in wrong[:6]:
        print(f"    miss: {w}", flush=True)
    with open("research/findings/raw/rf_unified_agent_full_probe.json", "w") as f:
        json.dump({"res": res, "overall": [gok, gtot], "verdict": verdict}, f, indent=2)
    print("\n  wrote research/findings/raw/rf_unified_agent_full_probe.json", flush=True)
    return 0 if full else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
