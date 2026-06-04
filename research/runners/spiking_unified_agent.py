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
    spikes_to_phases, phases_to_spikes, cleanup)
from research.runners.nested_composition_agent import Clause

ROLES = ("AGENT", "ACTION", "PATIENT")


def _to_phasor(spikes):
    """A spike-phase pattern -> the unit complex phasor it encodes (phasor = exp(2*pi*i*phase))."""
    return np.exp(2j * np.pi * spikes_to_phases(spikes))


def _unit(v):
    return v / (np.abs(v) + 1e-12)


def _phasor_sim(a, b):
    """|<a, b>| / D -- reconstruction quality in [0, 1], magnitude-invariant (the model-selection score)."""
    return float(np.abs(np.vdot(b, a)) / a.shape[0])


class SpikingUnifiedAgent:
    """Store flat SVO facts and answer who/what + abstain, entirely in spiking-phasor populations. Each fact is
    one bundle of three role-bindings (role ⊗ filler); a query unbinds a role and cleans up to the vocabulary."""

    def __init__(self, nouns, verbs, adjs=(), n_dim=512, seed=42, abstain_threshold=0.15,
                 resonator_backend="auto"):
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
        # vectorized clean-up codebooks: each vocabulary's spike-phases stacked (K x D). The role-matching WTA is
        # then ONE cos-similarity reduction over the codebook instead of a Python loop -- same metric, so memory
        # /retrieval scales to large vocabularies without the per-symbol loop (the capacity-curve cost lever).
        self._noun_phases = (np.stack([spikes_to_phases(s) for s in self.noun_vocab]) if self.nouns else None)
        self._verb_phases = (np.stack([spikes_to_phases(s) for s in self.verb_vocab]) if self.verbs else None)
        # phasor codebooks (D x |vocab|) for the two-attribute resonator decode (matmul-based, GPU-friendly)
        self.D = int(n_dim)
        self.seed = int(seed)
        self.n_restarts = 16
        self.NMAT = (np.stack([_to_phasor(self.noun_sym[w]) for w in self.nouns], axis=1)
                     if self.nouns else None)
        self.AMAT = (np.stack([_to_phasor(self.adj_sym[w]) for w in self.adjs], axis=1)
                     if self.adjs else None)
        self.VMAT = (np.stack([_to_phasor(self.verb_sym[w]) for w in self.verbs], axis=1)
                     if self.verbs else None)
        self.role_ph = {r: _to_phasor(self.role_sym[r]) for r in ROLES}  # cached role phasors
        # resonator backend: the two-attribute F=3 decode is matmul-heavy and needs D ~ M^2 at large vocab,
        # where CPU cannot run it (D>=8192 times out) -- so the GPU is the production default. "auto" (the
        # default) uses the GPU when one is present and falls back to numpy when not; "cupy"/"numpy" force it.
        # The numpy path is byte-for-byte the validated CPU resonator (the regression test pins it).
        self._res_xp = np
        self._res_AMAT, self._res_NMAT = self.AMAT, self.NMAT
        if resonator_backend in ("cupy", "auto"):
            try:
                import cupy as _cp
                if _cp.cuda.runtime.getDeviceCount() > 0:
                    self._res_xp = _cp
                    self._res_AMAT = _cp.asarray(self.AMAT) if self.AMAT is not None else None
                    self._res_NMAT = _cp.asarray(self.NMAT)
            except Exception:
                pass  # no cupy / no GPU -> numpy (the auto fallback; no behavior change)
        self.kb = []        # phase-midpoint bundle spikes per fact (the pure-phase readout; role matching)
        self.kb_complex = []  # the complex-SUM bundle per fact (the neuron's subthreshold membrane state, with
        #                       magnitude) -- enables EXACT crosstalk subtraction for the F=3 patient decode
        self.facts = []     # parallel (agent, action, patient) for reference

    def _filler_phasor(self, x):
        """The phasor of a filler: flat noun -> its code; (adj(s), noun) -> bound product; Clause -> the unit
        phasor of its role-binding superposition (recursive)."""
        if isinstance(x, Clause):
            return _unit(self._clause_sum(x))
        if isinstance(x, tuple):
            mods = x[0] if isinstance(x[0], tuple) else (x[0],)
            v = _to_phasor(self.noun_sym[x[1]])
            for a in mods:
                v = v * _to_phasor(self.adj_sym[a])
            return v
        return _to_phasor(self.noun_sym[x])

    def _clause_sum(self, clause):
        """The complex-sum superposition of a clause's three role-bindings (phasor; the bundle filler)."""
        return (self.role_ph["AGENT"] * self._filler_phasor(clause.agent)
                + self.role_ph["ACTION"] * _to_phasor(self.verb_sym[clause.action])
                + self.role_ph["PATIENT"] * self._filler_phasor(clause.patient))

    def _patient_filler(self, patient):
        """The PATIENT filler as a spike pattern. flat noun / (adj, noun) / ((adj, adj), noun) -> phase-sum
        neurons; a Clause -> the spike-phase of its role-binding superposition (a bundle filler, so the spike
        carries the phase and the magnitude lives in the membrane state via the complex-sum bundle)."""
        if isinstance(patient, Clause):
            return phases_to_spikes(np.mod(np.angle(self._clause_sum(patient)) / (2 * np.pi), 1.0))
        if isinstance(patient, tuple):
            mods = patient[0] if isinstance(patient[0], tuple) else (patient[0],)
            v = self.noun_sym[patient[1]]
            for a in mods:
                v = phase_sum_neuron(self.adj_sym[a], v)
            return v
        return self.noun_sym[patient]

    def _bound_phasor(self, role, filler_sym):
        """The complex phasor of (role ⊗ filler) -- a unit phasor per dimension (both operands are unit)."""
        return _to_phasor(phase_sum_neuron(self.role_sym[role], filler_sym))

    def learn(self, agent, action, patient):
        """Store an SVO fact (patient = flat noun, (adjective, noun), or ((adj, adj), noun)): bind each
        (role, filler) with phase-sum neurons, keep BOTH the phase-midpoint bundle (pure-phase readout, used for
        role matching) and the complex-sum bundle (the membrane state, used for exact crosstalk subtraction)."""
        bound = [phase_sum_neuron(self.role_sym["AGENT"], self.noun_sym[agent]),
                 phase_sum_neuron(self.role_sym["ACTION"], self.verb_sym[action]),
                 phase_sum_neuron(self.role_sym["PATIENT"], self._patient_filler(patient))]
        self.kb.append(phase_midpoint_bundle(bound))
        self.kb_complex.append(sum(_to_phasor(b) for b in bound))
        self.facts.append((agent, action, patient))

    @staticmethod
    def _argmax_sim(query, vocab, names):
        """Raw nearest-vocabulary match (no abstention): (name, similarity)."""
        sims = [phase_similarity(query, v) for v in vocab]
        k = int(np.argmax(sims))
        return names[k], float(sims[k])

    def _unbind_clean(self, bundle, role, phase_matrix, names):
        """Unbind a role (phase-subtraction neurons) then clean up to the vocabulary -- a VECTORIZED winner-take-
        all by spike-phase similarity (one cos-similarity reduction over the K×D codebook, not a Python loop),
        with abstention. Same similarity metric as the per-symbol clean-up (so the threshold is unchanged).
        Returns the recovered name, or None if the top similarity is below the abstention threshold."""
        rec = spikes_to_phases(phase_subtraction_neuron(bundle, self.role_sym[role]))   # (D,)
        sims = np.cos(2.0 * np.pi * (phase_matrix - rec[None, :])).mean(axis=1)          # (K,) = phase_similarity
        k = int(np.argmax(sims))
        return names[k] if float(sims[k]) >= self.abstain_threshold else None

    def _resonator3(self, p):
        """Factor the patient phasor p ~ adj1 ⊗ adj2 ⊗ noun by the phasor resonator with random restarts (the
        two adjectives share a codebook -> permutation symmetry -> restarts + best reconstruction break it).
        Runs on the resonator backend (numpy by default; cupy/GPU when resonator_backend='cupy') -- the matmuls
        dominate and need D ~ M^2 at large vocabulary, so the GPU is the enabler past ~320 concepts. The numpy
        path is byte-for-byte the validated CPU resonator. Returns (sorted-adjective-indices, noun-index,
        reconstruction-similarity)."""
        xp = self._res_xp
        AMAT, NMAT = self._res_AMAT, self._res_NMAT
        cbs = [AMAT, AMAT, NMAT]
        p = xp.asarray(p)
        rng = np.random.default_rng(self.seed + 11)
        Ma = AMAT.shape[1]

        def unit(v):
            return v / (xp.abs(v) + 1e-12)

        best = None
        for _ in range(self.n_restarts):
            est = [unit(AMAT.sum(1) + 0.7 * AMAT[:, int(rng.integers(Ma))]),
                   unit(AMAT.sum(1) + 0.7 * AMAT[:, int(rng.integers(Ma))]),
                   unit(NMAT.sum(1))]
            for _ in range(150):
                new = []
                for i in range(3):
                    o = xp.ones(self.D, dtype=complex)
                    for j in range(3):
                        if j != i:
                            o = o * est[j]
                    new.append(unit(cbs[i] @ (cbs[i].conj().T @ (p * xp.conj(o)))))
                est = new
            resid = float(xp.abs(xp.vdot(est[0] * est[1] * est[2], p)) / self.D)
            a1 = int(xp.argmax(xp.abs(AMAT.conj().T @ est[0])))
            a2 = int(xp.argmax(xp.abs(AMAT.conj().T @ est[1])))
            n = int(xp.argmax(xp.abs(NMAT.conj().T @ est[2])))
            if best is None or resid > best[0]:
                best = (resid, tuple(sorted({a1, a2})), n)
        return best[1], best[2], best[0]

    def _decode_patient(self, idx, agent, action):
        """Decode the PATIENT slot of the matched fact. Take the fact's complex-sum bundle (the membrane state),
        EXACTLY subtract the known agent + action role-bindings (predictive 'explaining-away' -- the magnitude is
        physical, so the subtraction is exact), and unbind the patient role -> the CLEAN patient phasor, no
        crosstalk. Then auto-detect flat / one- / two-attribute by RECONSTRUCTION similarity (the two-attribute
        model is the matmul resonator, F=3), with a parsimony margin preferring the simpler model unless a richer
        one is clearly better (the resonator's extra degrees of freedom would otherwise over-explain a noun)."""
        clean_bp = (self.kb_complex[idx]
                    - self._bound_phasor("AGENT", self.noun_sym[agent])
                    - self._bound_phasor("ACTION", self.verb_sym[action]))
        p = clean_bp * np.conj(self.role_ph["PATIENT"])     # the clean patient filler (crosstalk explained away)
        return self._decode_filler(p, depth=0)

    @staticmethod
    def _cleanup_phasor(p, MAT, names):
        """Nearest-vocabulary phasor cleanup: (name, confidence = |<p, best>|/D); (None, 0) if codebook empty."""
        if MAT is None:
            return None, 0.0
        ov = np.abs(MAT.conj().T @ _unit(p))
        k = int(np.argmax(ov))
        return names[k], float(ov[k] / p.shape[0])

    def _decode_filler(self, p, depth=0):
        """Decode a filler phasor into a string, auto-detecting its KIND. An embedded CLAUSE (a verb component is
        present after unbinding the ACTION role) -> decode its agent/action and recurse on its patient, with the
        SAME exact crosstalk subtraction at each level. Otherwise a terminal filler: flat noun / one- /
        two-attribute, chosen by reconstruction similarity with a parsimony upgrade. The two-attribute resonator
        runs at the top level only (inside a clause the arguments are flat or one-attribute)."""
        p = _unit(p)
        margin = 0.05

        # clause? (only a clause carries a verb component in its ACTION slot)
        if depth < 3 and self.VMAT is not None:
            ac, vconf = self._cleanup_phasor(p * np.conj(self.role_ph["ACTION"]), self.VMAT, self.verbs)
            if vconf >= 0.18:
                ag = self._decode_filler(p * np.conj(self.role_ph["AGENT"]), depth + 1)
                p_clean = p
                if ag in self.noun_sym and ac in self.verb_sym:        # explain away the inner agent + action
                    p_clean = (p - self.role_ph["AGENT"] * _to_phasor(self.noun_sym[ag])
                               - self.role_ph["ACTION"] * _to_phasor(self.verb_sym[ac]))
                pt = self._decode_filler(p_clean * np.conj(self.role_ph["PATIENT"]), depth + 1)
                return f"{ag} {ac} {pt}"

        # terminal filler -- flat model
        nidx = int(np.argmax(np.abs(self.NMAT.conj().T @ p)))
        sim_flat = _phasor_sim(p, self.NMAT[:, nidx])
        flat_noun = self.nouns[nidx]

        # one-attribute model (enumeration)
        one = None
        if self.AMAT is not None:
            for ai in range(self.AMAT.shape[1]):
                ni = int(np.argmax(np.abs(self.NMAT.conj().T @ (p * np.conj(self.AMAT[:, ai])))))
                sim = _phasor_sim(p, self.AMAT[:, ai] * self.NMAT[:, ni])
                if one is None or sim > one[0]:
                    one = (sim, self.adjs[ai], self.nouns[ni])

        # two-attribute model (F=3 resonator) -- top level only, and only when flat/one don't already explain p
        two = None
        best_simple = max(sim_flat, one[0] if one is not None else 0.0)
        if depth == 0 and self.AMAT is not None and self.AMAT.shape[1] >= 2 and best_simple < 0.5:
            adj_idx, noun_idx, sim_two = self._resonator3(p)
            two = (sim_two, [self.adjs[i] for i in adj_idx], self.nouns[noun_idx])

        # parsimony upgrade: flat -> one -> two, each only if it beats the running best by the margin
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

    def query_patient(self, agent, action):
        """"what does <agent> <action>?" -> the patient (flat noun or "adjective noun"), or None (abstain) if no
        stored fact matches."""
        for i, b in enumerate(self.kb):
            if (self._unbind_clean(b, "AGENT", self._noun_phases, self.nouns) == agent
                    and self._unbind_clean(b, "ACTION", self._verb_phases, self.verbs) == action):
                return self._decode_patient(i, agent, action)
        return None

    def query_agent(self, action, patient):
        """"who <action> <patient>?" -> the agent noun, or None (abstain) if no stored fact matches."""
        for b in self.kb:
            if (self._unbind_clean(b, "ACTION", self._verb_phases, self.verbs) == action
                    and self._unbind_clean(b, "PATIENT", self._noun_phases, self.nouns) == patient):
                return self._unbind_clean(b, "AGENT", self._noun_phases, self.nouns)
        return None


def run_core_benchmark(n_dim=512, seed=42, abstain_threshold=0.15, n_noun=200, n_verb=60, n_adj=60,
                       resonator_backend="auto"):
    """Run the unified-agent benchmark on the spiking agent at a chosen VOCABULARY SIZE (n_noun/n_verb/n_adj --
    default 320 concepts), using the same frozen test set (which uses only the core words) so accuracy at larger
    vocabularies measures how more distractor concepts stress the clean-up / resonator / decode at fixed
    dimension -- the capacity curve."""
    from research.runners.unified_agent_benchmark import (
        build_vocab, FACTS_FLAT, FACTS_1ATTR, FACTS_2ATTR, FACTS_CLAUSE, WHO_QUERIES, ABSTAIN_QUERIES)
    nouns, verbs, adjs = build_vocab(n_noun, n_verb, n_adj)
    agent = SpikingUnifiedAgent(nouns, verbs, adjs, n_dim=n_dim, seed=seed, abstain_threshold=abstain_threshold,
                                resonator_backend=resonator_backend)
    for ag, ac, pa in FACTS_FLAT + FACTS_1ATTR + FACTS_2ATTR + FACTS_CLAUSE:
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

    ta_ok = 0
    for ag, ac, pa in FACTS_2ATTR:
        want = " ".join(sorted(pa[0], key=adjs.index) + [pa[1]])   # canonical vocabulary order (commutative)
        got = agent.query_patient(ag, ac)
        ta_ok += (got == want)
        if got != want:
            wrong.append(("2-attribute", f"what does {ag} {ac}?", got, want))
    res["2-attribute"] = [ta_ok, len(FACTS_2ATTR)]

    def _render(pa):
        if isinstance(pa, Clause):
            return f"{_render(pa.agent)} {pa.action} {_render(pa.patient)}"
        if isinstance(pa, tuple):
            mods = pa[0] if isinstance(pa[0], tuple) else (pa[0],)
            return " ".join(sorted(mods, key=adjs.index) + [pa[1]])
        return pa
    cl_ok = 0
    for ag, ac, pa in FACTS_CLAUSE:
        want = _render(pa)
        got = agent.query_patient(ag, ac)
        cl_ok += (got == want)
        if got != want:
            wrong.append(("clause-depth1", f"what does {ag} {ac}?", got, want))
    res["clause-depth1"] = [cl_ok, len(FACTS_CLAUSE)]

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
    ap.add_argument("--resonator-backend", choices=["auto", "numpy", "cupy"], default="auto",
                    help="auto (default) uses the GPU when present else numpy; the F=3 resonator is the scaling "
                         "enabler past ~320 concepts")
    ap.add_argument("--n-noun", type=int, default=200)
    ap.add_argument("--n-verb", type=int, default=60)
    ap.add_argument("--n-adj", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"=== spiking unified agent | flat + 1/2-attribute + clause + who + abstain in spikes | "
          f"N_dim={args.n_dim} | vocab={args.n_noun + args.n_verb + args.n_adj} | "
          f"resonator={args.resonator_backend} | seeds={args.seeds} ===\n", flush=True)
    per_seed = []
    for s in args.seeds:
        res, wrong = run_core_benchmark(n_dim=args.n_dim, seed=s, abstain_threshold=args.abstain_threshold,
                                        n_noun=args.n_noun, n_verb=args.n_verb, n_adj=args.n_adj,
                                        resonator_backend=args.resonator_backend)
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
