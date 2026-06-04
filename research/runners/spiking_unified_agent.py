"""Spiking unified agent (pure-biology backlog #1, stage 2) — the SVO memory + Q&A + abstention of the unified
agent, realized in GENUINE SPIKES on the validated `spiking_phasor_fhrr` substrate (Orchard & Jarvis 2023).

This is the brain-analogue of `nested_composition_agent.NestedCompositionAgent` for its robust core: store
subject-verb-object facts, answer who/what queries, and abstain on the unknown — every operation a population of
time-stepped spiking-phasor integrator neurons (bind = phase-sum neurons, unbind = phase-subtraction neurons,
bundle = phase-midpoint, clean-up = winner-take-all over the vocabulary by spike-phase similarity with an
abstention threshold). The unified-agent benchmark's flat / who / abstain categories are the exact spec this must
reproduce; this module is driven by the benchmark's frozen test set so the spiking result is comparable to the
numpy-algebra result category-for-category.

Scope (honest, stage 2a): the FLAT robust core — flat-patient SVO facts, who/what Q&A, abstention. One-attribute
composition (the patient is adjective⊗noun) needs the spiking two-factor resonator (validated separately) and is
stage 2b. Reuse-by-import of the validated spiking substrate + the benchmark's frozen vocab/facts; no
protected-module edits. numpy/CPU (the spiking substrate; the GPU port is stage 3).

  python -m research.runners.spiking_unified_agent            # run the flat/who/abstain core in spikes
"""
from __future__ import annotations
import numpy as np

from research.runners.spiking_phasor_fhrr import (
    SpikingPhasorFHRR, phase_sum_neuron, phase_subtraction_neuron, phase_midpoint_bundle, phase_similarity,
    cleanup)

ROLES = ("AGENT", "ACTION", "PATIENT")


class SpikingUnifiedAgent:
    """Store flat SVO facts and answer who/what + abstain, entirely in spiking-phasor populations. Each fact is
    one bundle of three role-bindings (role ⊗ filler); a query unbinds a role and cleans up to the vocabulary."""

    def __init__(self, nouns, verbs, adjs=(), n_dim=512, seed=42, abstain_threshold=0.15):
        self.nouns, self.verbs, self.adjs = list(nouns), list(verbs), list(adjs)
        self.abstain_threshold = float(abstain_threshold)
        self.net = SpikingPhasorFHRR(n_dim, np.random.default_rng(seed))
        self.noun_sym = {w: self.net.random_symbol() for w in self.nouns}
        self.verb_sym = {w: self.net.random_symbol() for w in self.verbs}
        self.adj_sym = {w: self.net.random_symbol() for w in self.adjs}
        self.role_sym = {r: self.net.random_symbol() for r in ROLES}
        self.noun_vocab = [self.noun_sym[w] for w in self.nouns]   # clean-up codebooks (spike populations)
        self.verb_vocab = [self.verb_sym[w] for w in self.verbs]
        self.adj_vocab = [self.adj_sym[w] for w in self.adjs]
        self.kb = []        # bundle spikes per stored fact
        self.facts = []     # parallel (agent, action, patient) for reference

    def _patient_filler(self, patient):
        """A flat noun -> its symbol; an (adjective, noun) tuple -> the bound product (phase-sum neurons)."""
        if isinstance(patient, tuple):
            return phase_sum_neuron(self.adj_sym[patient[0]], self.noun_sym[patient[1]])
        return self.noun_sym[patient]

    def learn(self, agent, action, patient):
        """Store an SVO fact (patient = flat noun OR (adjective, noun)): bind each (role, filler) with phase-sum
        neurons, bundle them (phase-midpoint)."""
        bound = [phase_sum_neuron(self.role_sym["AGENT"], self.noun_sym[agent]),
                 phase_sum_neuron(self.role_sym["ACTION"], self.verb_sym[action]),
                 phase_sum_neuron(self.role_sym["PATIENT"], self._patient_filler(patient))]
        self.kb.append(phase_midpoint_bundle(bound))
        self.facts.append((agent, action, patient))

    @staticmethod
    def _argmax_sim(query, vocab, names):
        """Raw nearest-vocabulary match (no abstention): (name, similarity)."""
        sims = [phase_similarity(query, v) for v in vocab]
        k = int(np.argmax(sims))
        return names[k], float(sims[k])

    def _unbind_clean(self, bundle, role, vocab, names):
        """Unbind a role (phase-subtraction neurons) then clean up to the vocabulary (WTA + abstention).
        Returns the recovered name, or None if the top similarity is below the abstention threshold."""
        recovered = phase_subtraction_neuron(bundle, self.role_sym[role])
        idx, _ = cleanup(recovered, vocab, self.abstain_threshold)
        return names[idx] if idx >= 0 else None

    def _decode_patient(self, bundle):
        """Decode the PATIENT slot, auto-detecting flat noun vs (adjective, noun) -- the spiking two-factor
        decode. Unbind the patient role, then COMPARE two models: the flat-noun clean-up vs the best
        adjective-factoring (for each adjective, unbind it and clean up to the nouns; the adjective whose unbind
        yields the best clean noun is the attribute -- an enumeration factoring, robust for two factors). The
        model with the higher reconstruction similarity wins."""
        recovered = phase_subtraction_neuron(bundle, self.role_sym["PATIENT"])
        flat_noun, flat_sim = self._argmax_sim(recovered, self.noun_vocab, self.nouns)
        best = None
        for ai, a in enumerate(self.adjs):
            unbound = phase_subtraction_neuron(recovered, self.adj_vocab[ai])
            noun, sim = self._argmax_sim(unbound, self.noun_vocab, self.nouns)
            if best is None or sim > best[0]:
                best = (sim, a, noun)
        if best is None or flat_sim >= best[0]:
            return flat_noun
        return f"{best[1]} {best[2]}"

    def query_patient(self, agent, action):
        """"what does <agent> <action>?" -> the patient (flat noun or "adjective noun"), or None (abstain) if no
        stored fact matches."""
        for b in self.kb:
            if (self._unbind_clean(b, "AGENT", self.noun_vocab, self.nouns) == agent
                    and self._unbind_clean(b, "ACTION", self.verb_vocab, self.verbs) == action):
                return self._decode_patient(b)
        return None

    def query_agent(self, action, patient):
        """"who <action> <patient>?" -> the agent noun, or None (abstain) if no stored fact matches."""
        for b in self.kb:
            if (self._unbind_clean(b, "ACTION", self.verb_vocab, self.verbs) == action
                    and self._unbind_clean(b, "PATIENT", self.noun_vocab, self.nouns) == patient):
                return self._unbind_clean(b, "AGENT", self.noun_vocab, self.nouns)
        return None


def run_core_benchmark(n_dim=512, seed=42, abstain_threshold=0.15):
    """Run the unified-agent benchmark's robust core (flat / one-attribute / who / abstain) on the spiking agent,
    using the same frozen test set so the spiking result is comparable to the numpy-algebra result. Flat and
    one-attribute facts are stored together so the patient decode must auto-detect flat vs attributed."""
    from research.runners.unified_agent_benchmark import (
        build_vocab, FACTS_FLAT, FACTS_1ATTR, WHO_QUERIES, ABSTAIN_QUERIES)
    nouns, verbs, adjs = build_vocab()
    agent = SpikingUnifiedAgent(nouns, verbs, adjs, n_dim=n_dim, seed=seed, abstain_threshold=abstain_threshold)
    for ag, ac, pa in FACTS_FLAT + FACTS_1ATTR:
        agent.learn(ag, ac, pa)

    res, wrong = {}, []
    flat_ok = 0
    for ag, ac, pa in FACTS_FLAT:
        got = agent.query_patient(ag, ac)
        flat_ok += (got == pa)
        if got != pa:
            wrong.append(("flat", f"what does {ag} {ac}?", got, pa))
    res["flat"] = [flat_ok, len(FACTS_FLAT)]

    oa_ok = 0
    for ag, ac, pa in FACTS_1ATTR:
        want = f"{pa[0]} {pa[1]}"
        got = agent.query_patient(ag, ac)
        oa_ok += (got == want)
        if got != want:
            wrong.append(("1-attribute", f"what does {ag} {ac}?", got, want))
    res["1-attribute"] = [oa_ok, len(FACTS_1ATTR)]

    who_ok = 0
    for ac, pn, want in WHO_QUERIES:
        got = agent.query_agent(ac, pn)
        who_ok += (got == want)
        if got != want:
            wrong.append(("who", f"who {ac} {pn}?", got, want))
    res["who-query"] = [who_ok, len(WHO_QUERIES)]

    abstain_ok = 0
    for ag, ac in ABSTAIN_QUERIES:
        got = agent.query_patient(ag, ac)
        abstain_ok += (got is None)
        if got is not None:
            wrong.append(("abstain", f"what does {ag} {ac}? [should abstain]", got, None))
    res["abstain"] = [abstain_ok, len(ABSTAIN_QUERIES)]
    return res, wrong


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Spiking unified agent — flat robust core in genuine spikes.")
    ap.add_argument("--n-dim", type=int, default=512)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--abstain-threshold", type=float, default=0.15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"=== spiking unified agent | robust core (flat + one-attribute) in spikes | N_dim={args.n_dim} | "
          f"seeds={args.seeds} ===\n", flush=True)
    per_seed = []
    for s in args.seeds:
        res, wrong = run_core_benchmark(n_dim=args.n_dim, seed=s, abstain_threshold=args.abstain_threshold)
        per_seed.append({"seed": s, "categories": res, "wrong": wrong})
        line = "  ".join(f"{c}={res[c][0]}/{res[c][1]}" for c in res)
        print(f"  seed {s}:  {line}", flush=True)

    cats = list(per_seed[0]["categories"].keys())
    print("\n  --- per-category (multi-seed) ---", flush=True)
    gok = gtot = 0
    for c in cats:
        ok = sum(p["categories"][c][0] for p in per_seed)
        tot = sum(p["categories"][c][1] for p in per_seed)
        gok += ok
        gtot += tot
        print(f"    {c:<12} {ok:>3}/{tot:<3} = {ok/tot*100:5.1f}%", flush=True)
    print(f"\n  OVERALL (flat core): {gok}/{gtot} = {gok/gtot*100:.1f}%", flush=True)
    if per_seed[0]["wrong"]:
        print("  misses (seed 1):", flush=True)
        for cat, q, got, want in per_seed[0]["wrong"][:6]:
            print(f"    [{cat}] {q}  got={got!r} want={want!r}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"n_dim": args.n_dim, "seeds": args.seeds, "per_seed": per_seed,
                       "overall": [gok, gtot]}, f, indent=2)
        print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
