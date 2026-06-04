"""Stage 2c de-risk: the unified agent's robust core on the BIOLOGICAL resonate-and-fire substrate.

Stages 2a/2b built the robust core (flat / one-attribute / who / abstain) on `spiking_phasor_fhrr` — genuine
time-stepped spiking, but with Orchard's function-first *integrator* (counter) neurons. This probe re-runs the
SAME robust core on `resonate_fire_fhrr` — the genuine biological **resonate-and-fire** neuron model (Izhikevich
2001 / Frady-Sommer 2019): bind/unbind/bundle on resonate-and-fire neurons, and clean-up by the
`ResonateFireTPAM` complex-valued **attractor network** (the biological CA3 pattern completion the (b) result
needs; abstention = a basin-of-attraction property, the recurrent drive collapsing below threshold).

Scope (honest): the resonate-and-fire substrate steps a ~1000-step cycle per operation, so this validates the
biological substrate at the frozen test set's CORE vocabulary (the words the facts actually use) on CPU. The
full-320 resonate-and-fire run is the reserved-GPU build (stage 3). Reduced vocabulary is an EASIER clean-up
than 320, so this is a substrate-works validation, not a difficulty-matched one — stated plainly.

Pre-registered verdict: the robust core (flat / one-attribute / who / abstain) reproduces on the biological
substrate at the core vocabulary; the no-confabulation moat (abstain on a never-stored pair) holds.

  SIM_BACKEND=numpy python -m research.runners._rf_unified_agent_probe
"""
from __future__ import annotations
import json

import numpy as np

from research.runners.resonate_fire_fhrr import (
    ResonateFireFHRR, ResonateFireTPAM, rf_bind, rf_unbind, rf_bundle, _to_phasor, CYCLE_STEPS)
from research.runners.unified_agent_benchmark import (
    CORE_NOUNS, CORE_VERBS, CORE_ADJS, FACTS_FLAT, FACTS_1ATTR, WHO_QUERIES, ABSTAIN_QUERIES)

ROLES = ("AGENT", "ACTION", "PATIENT")
ABSTAIN_ACTIVE_FRAC = 0.15      # below this settled-active fraction -> the network collapsed -> abstain


class RFUnifiedAgent:
    """The robust-core unified agent on the resonate-and-fire biological substrate + TPAM attractor cleanup."""

    def __init__(self, nouns, verbs, adjs, n_dim=512, seed=42):
        self.nouns, self.verbs, self.adjs = list(nouns), list(verbs), list(adjs)
        self.net = ResonateFireFHRR(n_dim, np.random.default_rng(seed))
        self.noun_sym = {w: self.net.random_symbol() for w in self.nouns}
        self.verb_sym = {w: self.net.random_symbol() for w in self.verbs}
        self.adj_sym = {w: self.net.random_symbol() for w in self.adjs}
        self.role_sym = {r: self.net.random_symbol() for r in ROLES}
        self.noun_tpam = ResonateFireTPAM([self.noun_sym[w] for w in self.nouns])
        self.verb_tpam = ResonateFireTPAM([self.verb_sym[w] for w in self.verbs])
        self.adj_tpam = ResonateFireTPAM([self.adj_sym[w] for w in self.adjs])
        self.kb, self.facts = [], []

    def _filler(self, patient):
        if isinstance(patient, tuple):
            return rf_bind(self.adj_sym[patient[0]], self.noun_sym[patient[1]])
        return self.noun_sym[patient]

    def learn(self, agent, action, patient):
        bound = [rf_bind(self.role_sym["AGENT"], self.noun_sym[agent]),
                 rf_bind(self.role_sym["ACTION"], self.verb_sym[action]),
                 rf_bind(self.role_sym["PATIENT"], self._filler(patient))]
        self.kb.append(rf_bundle(bound))
        self.facts.append((agent, action, patient))

    @staticmethod
    def _tpam_match(tpam, recovered_spikes, names):
        """Settle the recovered phasor in the attractor network; return (name, top_overlap, active_fraction).
        name is None when the network collapsed (active fraction below the abstain floor) -- basin abstention."""
        z, active_frac = tpam.settle(recovered_spikes)
        overlaps = np.abs(tpam.s.conj().T @ z)
        k = int(np.argmax(overlaps))
        name = names[k] if active_frac >= ABSTAIN_ACTIVE_FRAC else None
        return name, float(overlaps[k]), active_frac

    def _role(self, bundle, role, tpam, names):
        rec = rf_unbind(bundle, self.role_sym[role])
        return self._tpam_match(tpam, rec, names)[0]

    def _decode_patient(self, bundle):
        rec = rf_unbind(bundle, self.role_sym["PATIENT"])
        flat_noun, flat_ov, _ = self._tpam_match(self.noun_tpam, rec, self.nouns)
        best = None
        for ai, a in enumerate(self.adjs):
            unbound = rf_unbind(rec, self.adj_sym[a])
            noun, ov, _ = self._tpam_match(self.noun_tpam, unbound, self.nouns)
            if noun is not None and (best is None or ov > best[0]):
                best = (ov, a, noun)
        if best is None or (flat_noun is not None and flat_ov >= best[0]):
            return flat_noun
        return f"{best[1]} {best[2]}"

    def query_patient(self, agent, action):
        for b in self.kb:
            if (self._role(b, "AGENT", self.noun_tpam, self.nouns) == agent
                    and self._role(b, "ACTION", self.verb_tpam, self.verbs) == action):
                return self._decode_patient(b)
        return None

    def query_agent(self, action, patient):
        for b in self.kb:
            if (self._role(b, "ACTION", self.verb_tpam, self.verbs) == action
                    and self._role(b, "PATIENT", self.noun_tpam, self.nouns) == patient):
                return self._role(b, "AGENT", self.noun_tpam, self.nouns)
        return None


def main():
    print("=== unified agent robust core on the BIOLOGICAL resonate-and-fire substrate + TPAM (stage 2c) ===",
          flush=True)
    print(f"  core vocab: {len(CORE_NOUNS)} nouns, {len(CORE_VERBS)} verbs, {len(CORE_ADJS)} adjs; "
          f"cycle={CYCLE_STEPS}\n", flush=True)
    agent = RFUnifiedAgent(CORE_NOUNS, CORE_VERBS, CORE_ADJS, seed=42)
    for ag, ac, pa in FACTS_FLAT + FACTS_1ATTR:
        agent.learn(ag, ac, pa)

    res, wrong = {}, []
    flat_ok = sum(agent.query_patient(ag, ac) == pa for ag, ac, pa in FACTS_FLAT)
    for ag, ac, pa in FACTS_FLAT:
        if agent.query_patient(ag, ac) != pa:
            wrong.append(("flat", ag, ac, agent.query_patient(ag, ac), pa))
    res["flat"] = [flat_ok, len(FACTS_FLAT)]

    oa_ok = sum(agent.query_patient(ag, ac) == f"{pa[0]} {pa[1]}" for ag, ac, pa in FACTS_1ATTR)
    res["1-attribute"] = [oa_ok, len(FACTS_1ATTR)]

    who_ok = sum(agent.query_agent(ac, pn) == want for ac, pn, want in WHO_QUERIES)
    res["who-query"] = [who_ok, len(WHO_QUERIES)]

    abstain_ok = sum(agent.query_patient(ag, ac) is None for ag, ac in ABSTAIN_QUERIES)
    res["abstain"] = [abstain_ok, len(ABSTAIN_QUERIES)]

    for c in res:
        print(f"  {c:<12} {res[c][0]}/{res[c][1]} = {res[c][0]/res[c][1]*100:.0f}%", flush=True)
    gok = sum(v[0] for v in res.values())
    gtot = sum(v[1] for v in res.values())
    full = all(v[0] == v[1] for v in res.values())
    print(f"\n  OVERALL robust core (biological substrate): {gok}/{gtot} = {gok/gtot*100:.1f}%", flush=True)
    verdict = "RESOLVES" if full else ("PARTIAL" if gok / gtot >= 0.8 else "DOES_NOT_RESOLVE")
    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    if wrong:
        print("  misses:", flush=True)
        for w in wrong[:6]:
            print(f"    {w}", flush=True)
    with open("research/findings/raw/rf_unified_agent_probe.json", "w") as f:
        json.dump({"res": res, "overall": [gok, gtot], "verdict": verdict}, f, indent=2)
    print("\n  wrote research/findings/raw/rf_unified_agent_probe.json", flush=True)
    return 0 if full else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
