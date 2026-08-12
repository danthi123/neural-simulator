"""SPIKING OPEN-ENDED GENERATION DE-RISK -- a VOCAB-AGNOSTIC spiking generative draw that produces a
grammatical multi-word utterance (>=SVO, + a connective clause) from the brain's OWN learned codes, on
FIRING NEURONS, GO-gated by grammaticality + novelty + the no-confab moat.

THE WALL THIS ATTACKS (north-star: brain-native spiking fluent generation):
  The brain's spiking generative DRAW is a 6-seed GO (`_followon2_spiking_wta_sampler_derisk`: a Buesing-
  Bill-Nessler-Maass 2011 noise-driven soft-WTA over an Izhikevich bank -- the winner read from
  cp_firing_states IS the categorical draw, NO host rng.choice). BUT it is HARD-LOCKED to the hand-designed
  8x8 taxonomy: `SpikingWTASampler.__init__` calls `_category_pools(TAXONOMY_8x8)` and then
  `_encodable_agents()` indexes `self.row[<taxonomy word>]` -> KeyError on ANY corpus-mined vocab (confirmed
  empirically: KeyError 'run'). So the spiking draw cannot run on arbitrary vocabulary -- it is not
  vocab-agnostic, which blocks the brain-owns-generation stack (#3E) from being genuinely open-ended.

THE WALL-DISCIPLINE REFRAME (what constant did we substitute for a companion process?):
  The hand-designed TAXONOMY_8x8 is a CONSTANT standing in for a process biology runs ALONGSIDE generation:
  SYNTACTIC-CATEGORY ACQUISITION -- a child induces noun/verb categories from distributional + morphological
  regularities of the input (Mintz 2003 "frequent frames"; morphological bootstrapping: -ed/-ing inflections
  mark verbs). We replace the constant taxonomy with a CORPUS-DERIVED morpho-distributional role tagger, so
  the role pools come from the input, not a hand table. That makes the spiking draw vocab-agnostic.

WHAT IS GENUINELY SPIKING vs HOST (the brutally-honest inventory this de-risk MAPS):
  SPIKING (on firing neurons): the generative DRAW -- for each slot, one soft-WTA competition on a real
    SimulationBridge Izhikevich pool driven by the brain's learned PPMI likelihood + OU membrane noise; the
    winner (argmax-over-FIRING read from cp_firing_states) IS the sampled word. The OU noise IS the
    stochasticity (ablate -> deterministic argmax). NO host rng.choice on the draw path (source-grep + count).
  HOST SCAFFOLDS (mapped, not hidden -- the residual walls, the biologization targets):
    - the role tagger (morpho-distributional) is host-computed (target: a spiking morphology/frame detector);
    - the SVO / connective TEMPLATE (slot order, the connective lexeme) is host (target: a learned/spiking
      Broca sequencer -- emerge62-65 direction);
    - the PPMI likelihood is a host matrix over the brain's heard co-occurrence (project's stream-cortex);
    - the no-confab moat is the RF phasor composer (host algebra with a validated spiking select).
  So the DELIVERABLE claim is narrow + defensible: the spiking categorical DRAW generalizes to ARBITRARY
  corpus vocabulary; the grammar/role acquisition remain host scaffolds (honest residual).

GRAMMATICALITY, MEASURED INDEPENDENTLY (non-circular): roles are induced on corpus SPLIT A (used to
  generate); grammaticality is judged by a role tagger re-induced on a DISJOINT corpus SPLIT B. A generated
  (S,V,O) is grammatical iff, per split-B's INDEPENDENT tagger, S and O are nouns and V is a verb. Cross-
  validated across disjoint corpus halves -> not the generator grading its own homework. A role-BLIND
  "unslotted" draw (all three words from the full vocab, ignoring roles) is the grammaticality FLOOR the
  slotted spiking draw must beat.

THE GO BARS (>=6 seeds; mirrors b2/followon2 + the grammaticality axis):
  (PROVENANCE, HARD) each slot is drawn from cp_firing_states (no host rng on the draw path; source-clean +
    0 host-rng draws) AND noise-ablation (ou_std->0 collapses the draw to a deterministic argmax).
  (VOCAB-AGNOSTIC) the sampler runs on a corpus-mined vocab (NOT the 8x8 taxonomy; <=~15% overlap) with
    induced roles, NO KeyError, and emits >= min_novel distinct novel utterances.
  (GRAMMATICAL) spiking-generated utterances pass the independent split-B grammaticality oracle at
    >= grammatical_bar, and BEAT the role-blind unslotted floor by >= grammatical_advantage.
  (NOVEL) disjoint from the store; the no-confab known-fact retrieval ABSTAINS on every generated utterance.
  (PLAUSIBLE) spiking plausible-frac >= advantage_bar x the random-recombination floor AND >= host_match x
    the host rng.choice sample loop (quality parity with the validated host draw).
  (MOAT) 0 hypothesis->known-fact leaks + 0 negated facts re-proposed; untaught-cue abstention unregressed.
  (LESION) likelihood-ablation (equal drive) collapses plausibility; shuffled-PPMI collapses TRUE plausibility.
  HONEST_NEGATIVE on any gate = the precisely-isolated wall (a first-class deliverable, NOT a stopping point).

REUSE-BY-IMPORT, NO sim/ edit. CPU (SIM_BACKEND=numpy). Run:
  SIM_BACKEND=numpy python -u -m research.runners._spiking_openended_generation_derisk \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/_spiking_openended_generation_derisk.json
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

# Reuse-by-import: the b2 PPMI plausibility + gates + host baseline + shuffle control; the validated spiking
# soft-WTA sampler (followon2); the option_c corpus builder + the taxonomy (used ONLY as an independent POS
# oracle to sanity-check the induced tagger -- NEVER as the generation vocab); the RF composer (moat).
from research.runners._genfrontier_b2_generative_replay_derisk import (  # noqa: E402
    GenerativeReplayProposer,
    build_plausibility,
    random_recombination,
    shuffle_graph,
)
from research.runners._followon2_spiking_wta_sampler_derisk import SpikingWTASampler  # noqa: E402
from research.runners.option_c_real_cooccurrence_derisk import (  # noqa: E402
    TAXONOMY_8x8,
    build_real_cooccurrence,
)
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402


# ===========================================================================
# CORPUS -> arbitrary vocab + morpho-distributional roles (the companion process, vocab-agnostic).
# ===========================================================================
# A minimal CLOSED-CLASS stoplist (function words) -- universal scaffolding, NOT a per-word content taxonomy.
STOP = set("the a an and or but if then so to of in on at for with as is are was were be been "
           "being have has had do does did will would can could shall should may might must "
           "i you he she it we they me him her us them my your his its our their this that these "
           "those there here not no yes very just too also all some any one two out up down off "
           "over under again into from by about endoftext".split())
# closed-class cues that PRECEDE verbs (subject pronouns / tense-aspect / infinitival) -- Mintz-style frame
VERB_LEFT = set("to will would can could should must let and then they he she it we i you".split())
# determiner / possessive / quantifier / adjectival cues that PRECEDE nouns
NOUN_LEFT = set("the a an my your his her its our their this that some any one two big little".split())


def load_stories(corpus_path, max_bytes):
    """Read the corpus, split into stories, tokenize each into a list of alphabetic tokens (lowercased)."""
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read(max_bytes) if max_bytes else fh.read()
    text = text.lower()
    stories = text.split("<|endoftext|>")
    return [re.findall(r"[a-z]+", s) for s in stories]


def mine_vocab(stories_tokens, top_k):
    """The vocab is the top_k most-frequent alphabetic CONTENT words (minus the closed-class stoplist) --
    genuinely corpus-derived, NOT the hand-designed 8x8 taxonomy."""
    cnt = Counter()
    for toks in stories_tokens:
        for t in toks:
            if t not in STOP and len(t) >= 2:
                cnt[t] += 1
    return [w for w, _ in cnt.most_common(top_k)]


def morpho_distributional_tag(stories_tokens, vocab):
    """Induce a role (verb vs noun) per vocab word from MORPHOLOGY + function-word FRAMES (vocab-agnostic;
    the biology: morphological bootstrapping + Mintz-2003 frequent frames). Verb evidence = the word is/has
    -ed/-ing inflections + a high fraction of occurrences preceded by a verb-left cue; noun evidence = a high
    fraction preceded by a noun-left (determiner/adjective) cue + a slight majority-class prior. Returns
    {word: is_verb(bool)} and a per-word score dict."""
    vset = set(vocab)
    all_tokens = set()
    left = defaultdict(Counter)
    for toks in stories_tokens:
        for k, t in enumerate(toks):
            all_tokens.add(t)
            if t in vset:
                left[t][toks[k - 1] if k > 0 else "<s>"] += 1
    is_verb, score = {}, {}
    for w in vocab:
        morph = 0.0
        if (w.endswith("ing") and len(w) > 5) or (w.endswith("ed") and len(w) > 4):
            morph += 1.0
        for suf in ("ed", "ing"):
            if (w + suf) in all_tokens:
                morph += 0.5
        if w.endswith("e") and (w[:-1] + "ing") in all_tokens:
            morph += 0.5
        lc = left[w]
        tot = sum(lc.values()) or 1
        vfrac = sum(c for p, c in lc.items() if p in VERB_LEFT) / tot
        nfrac = sum(c for p, c in lc.items() if p in NOUN_LEFT) / tot
        vscore = morph + 2.0 * vfrac
        nscore = 2.0 * nfrac + 0.3
        score[w] = {"vscore": vscore, "nscore": nscore, "morph": morph, "vfrac": vfrac, "nfrac": nfrac}
        is_verb[w] = vscore > nscore
    return is_verb, score


def roles(vocab, is_verb):
    """Split the vocab into noun and verb pools by the induced tags."""
    nouns = [w for w in vocab if not is_verb[w]]
    verbs = [w for w in vocab if is_verb[w]]
    return nouns, verbs


def verb_oracle_prf(vocab, is_verb):
    """SANITY (not a gate): induced-verb precision/recall vs the TAXONOMY_8x8 POS oracle, over the vocab words
    that ARE in the taxonomy (actions=verb truth, all else=noun truth). An honest number for induction quality."""
    tv = set(TAXONOMY_8x8["actions"])
    tn = set(w for c, ws in TAXONOMY_8x8.items() if c != "actions" for w in ws)
    tp = fp = fn = tn_ = 0
    for w in vocab:
        if w in tv:
            tp += int(is_verb[w]); fn += int(not is_verb[w])
        elif w in tn:
            fp += int(is_verb[w]); tn_ += int(not is_verb[w])
    n_overlap = tp + fn + fp + tn_
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"n_overlap": n_overlap, "verb_precision": prec, "verb_recall": rec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn_}


# ===========================================================================
# THE VOCAB-AGNOSTIC SPIKING SAMPLER -- the followon2 soft-WTA, decoupled from TAXONOMY_8x8.
# ===========================================================================
class VocabAgnosticSpikingSampler(SpikingWTASampler):
    """Identical spiking machinery to the GO SpikingWTASampler (Buesing-Maass soft-WTA over an unwired
    GENERIC_UNSTRUCTURED Izhikevich bank with OU membrane noise; the winner read from cp_firing_states IS the
    draw), but the role pools come from the INDUCED tagger (nouns/verbs), NOT `_category_pools(TAXONOMY_8x8)`.
    That single swap removes the KeyError on arbitrary vocab. All draw-path methods (_weights, _likelihood_
    drive, drive_from_weights, _compete, _draw, draw_from_weights, draw_one, calibration_kl) are INHERITED
    UNCHANGED -- so the provenance/noise-ablation guarantees carry over verbatim."""

    def __init__(self, P, row, tau, nouns, verbs, seed=42, n_cand_max=128, base_pA=110.0, gain_pA=160.0,
                 read_window=120, ou_std_current_pA=200.0, temperature=1.0, ablate_likelihood=False,
                 ablate_noise=False, shuffled_P=None, shuffled_tau=None):
        self.P, self.row, self.tau = P, row, tau
        self.seed = seed
        self.base_pA = float(base_pA)
        self.gain_pA = float(gain_pA)
        self.read_window = int(read_window)
        self.temperature = float(temperature)
        self.ablate_likelihood = bool(ablate_likelihood)
        self.ablate_noise = bool(ablate_noise)
        self.shuffled_P = shuffled_P
        self.shuffled_tau = shuffled_tau
        # THE ONLY CHANGE vs the parent: roles from the induced tagger (subjects+objects = nouns; verbs).
        self.agents = list(nouns)
        self.actions = list(verbs)
        self.patients = list(nouns)
        self._seed_rng = np.random.default_rng(seed * 31 + 3)
        self.n_cand_max = int(n_cand_max)
        ou_std = 0.0 if self.ablate_noise else float(ou_std_current_pA)
        self.ou_std = ou_std
        self._bank = self._build_wta_bank(self.n_cand_max, ou_std)   # inherited, taxonomy-free
        self.encodable_agents = self._encodable_agents()             # inherited; now all words are in row
        self.n_spiking_draws = 0
        self.n_host_rng_draws = 0
        self.n_silent_fallbacks = 0

    def draw_svo(self):
        """One SVO generative event: seed a subject (encodable noun), SPIKING-draw a verb | subject, then
        SPIKING-draw an object-noun | (subject, verb). Rejects S==O (degenerate). Returns (s, v, o) or None."""
        s = self.encodable_agents[int(self._seed_rng.integers(len(self.encodable_agents)))]
        v, _, _ = self._draw([s], self.actions)
        if v is None:
            return None
        o, _, _ = self._draw([s, v], self.patients)
        if o is None or o == s:
            return None
        return (s, v, o)

    def draw_many(self, n_attempts):
        out = []
        for _ in range(n_attempts):
            t = self.draw_svo()
            if t is not None:
                out.append(t)
        return out


# ===========================================================================
# Plausible-SVO enumeration + stored-fact builder over the induced noun/verb pools (S != O).
# ===========================================================================
def enumerate_plausible_nv(nouns, verbs, P, row, tau):
    """All plausible (s, v, o): selectional preference s~v AND v~o (PPMI >= tau), s != o. Matches
    GenerativeReplayProposer._plausible over the induced noun/verb pools."""
    def rel(a, b):
        return P[row[a], row[b]] >= tau
    verbs_by_subject = {}
    out = []
    for s in nouns:
        rel_v = [v for v in verbs if rel(s, v)]
        verbs_by_subject[s] = rel_v
        for v in rel_v:
            for o in nouns:
                if o == s:
                    continue
                if rel(v, o):
                    out.append((s, v, o))
    return out


def build_stored_facts_nv(nouns, verbs, P, row, tau, n_facts, n_negated, rng):
    plausible_all = enumerate_plausible_nv(nouns, verbs, P, row, tau)
    rng.shuffle(plausible_all)
    need = n_facts + n_negated
    chosen = plausible_all[:min(need, len(plausible_all))]
    affirmed = chosen[:n_facts]
    negated = chosen[n_facts:n_facts + n_negated]
    return affirmed, negated, plausible_all


def _gate_and_collect(raw_triples, proposer, all_stored):
    """Apply the brain's UNCHANGED gates (novelty + plausibility + non-contradiction) to raw drawn triples."""
    accepted, seen = [], set()
    n_novel, n_plausible = 0, 0
    for (s, v, o) in raw_triples:
        triple = (s, v, o)
        if triple in all_stored:
            continue
        n_novel += 1
        is_pl = proposer._plausible(s, v, o)
        if is_pl:
            n_plausible += 1
        if triple in seen:
            continue
        if is_pl and not proposer._contradicts(s, v, o):
            accepted.append(triple)
            seen.add(triple)
    return {"accepted": accepted, "n_novel_attempts": n_novel,
            "plausible_fraction_of_novel": n_plausible / max(1, n_novel)}


def _grammatical(triple, is_verb_B):
    """Independent grammaticality (split-B tagger): S and O are nouns, V is a verb, per the DISJOINT-split
    tagger (non-circular with the split-A roles that generated the triple)."""
    s, v, o = triple
    if s not in is_verb_B or v not in is_verb_B or o not in is_verb_B:
        return False
    return (not is_verb_B[s]) and is_verb_B[v] and (not is_verb_B[o])


# ===========================================================================
# One seed.
# ===========================================================================
def run_seed(seed, vocab, corpus, nouns, verbs, is_verb_B, a):
    rng = np.random.default_rng(seed)
    P, row = build_plausibility(corpus, vocab)
    pos = P[P > 0]
    tau = float(np.percentile(pos, a.tau_pct)) if pos.size else 0.0

    affirmed, negated, plausible_all = build_stored_facts_nv(
        nouns, verbs, P, row, tau, a.n_facts, a.n_negated, rng)
    all_stored = set(affirmed) | set(negated)
    plausible_novel_universe = sorted(set(plausible_all) - all_stored)

    # store the facts in the RF composer (the no-confab moat); larger vocab -> larger D for clean codes
    comp = RFPhasorComposer(seed=seed, D=a.D, vocab=vocab)
    for s, v, o in affirmed:
        comp.store(s, v, o, polarity="AFFIRM")
    for s, v, o in negated:
        comp.store(s, v, o, polarity="NEGATE")

    # the brain's gates + the HOST rng.choice ORACLE baseline (use_spiking_sampler=False -> host draw)
    proposer = GenerativeReplayProposer(comp, affirmed, negated, P, row, tau,
                                        np.random.default_rng(seed * 7 + 1), use_spiking_sampler=False)
    proposer.agents = list(nouns)
    proposer.actions = list(verbs)
    proposer.patients = list(nouns)

    # ---- HOST sample-loop baseline (the validated draw we match for quality) ----
    host_rep = proposer.propose(a.n_attempts)
    host_frac = host_rep["plausible_fraction_of_novel"]

    # ---- the VOCAB-AGNOSTIC SPIKING sampler (the replacement for the host DRAW, on arbitrary vocab) ----
    n_cand = max(96, len(nouns), len(verbs))
    t_build = time.time()
    sampler = VocabAgnosticSpikingSampler(P, row, tau, nouns, verbs, seed=seed, n_cand_max=n_cand,
                                          base_pA=a.base_pA, gain_pA=a.gain_pA, read_window=a.read_window,
                                          ou_std_current_pA=a.ou_std, temperature=a.temperature)
    build_s = time.time() - t_build
    raw = sampler.draw_many(a.n_attempts_spiking)
    spk = _gate_and_collect(raw, proposer, all_stored)
    spk_accepted = spk["accepted"]
    spk_frac = spk["plausible_fraction_of_novel"]
    spk_set = set(spk_accepted)
    n_spk = len(spk_accepted)

    # ---- (PROVENANCE) draw from cp_firing_states, NO host rng on the draw path ----
    provenance_no_host_rng = (sampler.n_host_rng_draws == 0) and (sampler.n_spiking_draws > 0)

    def _code_only(fn):
        src = inspect.getsource(fn)
        tree = ast.parse(src.strip())
        body = tree.body[0].body
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
            body = body[1:]
        return "\n".join(ast.unparse(n) for n in body)
    # grep the DRAW path (== followon2's provenance definition): _draw picks the winning FILLER (argmax-over-
    # FIRING from cp_firing_states), _likelihood_drive builds its drive. These must contain NO host categorical
    # draw. draw_svo is ORCHESTRATION and is EXCLUDED here on purpose: its only rng is `self._seed_rng.integers`
    # -- the SWR REPLAY SEED (which subject-memory reactivates), a documented-legitimate host process (uniform,
    # NOT a likelihood-weighted draw of a filler), exactly as followon2's draw_one excludes it. The generative
    # ACT (which word fills a slot given the likelihood) is entirely in the inherited _draw/_compete (spiking).
    draw_src = _code_only(SpikingWTASampler._draw) + "\n" + _code_only(SpikingWTASampler._likelihood_drive)
    provenance_no_rng_in_source = all(tok not in draw_src for tok in
                                      ("rng.choice", "random.choice", "np.random", ".choice(", ".integers("))
    # and assert draw_svo's ONLY rng touch is the SWR seed (self._seed_rng), never a distribution draw
    svo_src = _code_only(VocabAgnosticSpikingSampler.draw_svo)
    seed_rng_only = ("_seed_rng.integers" in svo_src) and all(
        tok not in svo_src for tok in ("rng.choice", "random.choice", "np.random", ".choice("))
    provenance_ok = bool(provenance_no_host_rng and provenance_no_rng_in_source and seed_rng_only)

    # ---- noise-ablation: ou_std=0 -> deterministic argmax (the OU noise IS the stochasticity) ----
    sampler_nl = VocabAgnosticSpikingSampler(P, row, tau, nouns, verbs, seed=seed, n_cand_max=n_cand,
                                             base_pA=a.base_pA, gain_pA=a.gain_pA, read_window=a.read_window,
                                             ou_std_current_pA=a.ou_std, ablate_noise=True)
    cal_seed = sampler.encodable_agents[0]
    _, nl_emp, _, nl_silent = sampler_nl.calibration_kl([cal_seed], sampler.actions, n_repeats=60)
    noiseless_deterministic = bool(float(nl_emp.max()) >= 0.999 and nl_silent == 0)
    _, noisy_emp, _, _ = sampler.calibration_kl([cal_seed], sampler.actions, n_repeats=a.calib_repeats)
    noisy_is_stochastic = bool(float(noisy_emp.max()) < 0.999)
    noise_ablation_ok = bool(noiseless_deterministic and noisy_is_stochastic)

    # ---- (VOCAB-AGNOSTIC) genuinely arbitrary vocab: overlap with the taxonomy ----
    taxo_words = set(w for ws in TAXONOMY_8x8.values() for w in ws)
    overlap_frac = len([w for w in vocab if w in taxo_words]) / max(1, len(vocab))

    # ---- (GRAMMATICAL) independent split-B oracle + the role-blind unslotted floor ----
    n_gram = sum(1 for t in spk_accepted if _grammatical(t, is_verb_B))
    grammatical_frac = n_gram / max(1, n_spk)
    # role-blind floor: draw all three slots from the FULL vocab ignoring roles -> grammaticality per split-B
    blind_rng = np.random.default_rng(seed * 29 + 9)
    n_blind, n_blind_gram = 0, 0
    for _ in range(a.n_attempts):
        t = (vocab[int(blind_rng.integers(len(vocab)))],
             vocab[int(blind_rng.integers(len(vocab)))],
             vocab[int(blind_rng.integers(len(vocab)))])
        if t[0] == t[2]:
            continue
        n_blind += 1
        if _grammatical(t, is_verb_B):
            n_blind_gram += 1
    blind_grammatical_frac = n_blind_gram / max(1, n_blind)
    grammatical_advantage = grammatical_frac / max(blind_grammatical_frac, 1.0 / max(1, n_blind))

    # ---- (NOVEL) disjoint + retrieval abstains ----
    novel_disjoint = len(spk_set & all_stored) == 0
    novel_comp_score = min(1.0, n_spk / max(1, len(plausible_novel_universe)))
    retr_abstains = 0
    for (s, v, o) in spk_accepted:
        kp = comp.query_patient(s, v)
        yn = comp.ask_yes_no(s, v, o)
        if kp != o and yn == "unknown":
            retr_abstains += 1
    retr_abstains_all = (retr_abstains == n_spk)

    # ---- (PLAUSIBLE) vs random floor (advantage) + vs the host loop (quality parity) ----
    randb = random_recombination(proposer, a.n_attempts, np.random.default_rng(seed * 13 + 3))
    random_frac = randb["plausible_fraction_of_novel"]
    floor = max(random_frac, 1.0 / max(1, randb["n_novel_attempts"]))
    spk_advantage = spk_frac / floor
    spk_vs_host = spk_frac / max(host_frac, 1e-9)

    # ---- (LESION) likelihood-ablation collapses; shuffled-PPMI collapses TRUE plausibility ----
    sampler_les = VocabAgnosticSpikingSampler(P, row, tau, nouns, verbs, seed=seed, n_cand_max=n_cand,
                                              base_pA=a.base_pA, gain_pA=a.gain_pA, read_window=a.read_window,
                                              ou_std_current_pA=a.ou_std, ablate_likelihood=True)
    les = _gate_and_collect(sampler_les.draw_many(a.n_attempts_spiking), proposer, all_stored)
    lesion_frac = les["plausible_fraction_of_novel"]
    lesion_collapses = lesion_frac <= max(0.5 * spk_frac, random_frac * 1.5 + 0.02)

    P_shuf = shuffle_graph(P, np.random.default_rng(seed * 17 + 5))
    pos_s = P_shuf[P_shuf > 0]
    tau_s = float(np.percentile(pos_s, a.tau_pct)) if pos_s.size else 0.0
    sampler_shuf = VocabAgnosticSpikingSampler(P, row, tau, nouns, verbs, seed=seed, n_cand_max=n_cand,
                                               base_pA=a.base_pA, gain_pA=a.gain_pA, read_window=a.read_window,
                                               ou_std_current_pA=a.ou_std, shuffled_P=P_shuf, shuffled_tau=tau_s)
    raw_shuf = sampler_shuf.draw_many(a.n_attempts_spiking)
    n_sh_novel, n_sh_true = 0, 0
    for (s, v, o) in raw_shuf:
        if (s, v, o) in all_stored:
            continue
        n_sh_novel += 1
        if proposer._plausible(s, v, o):
            n_sh_true += 1
    shuf_true_frac = n_sh_true / max(1, n_sh_novel)
    shuffled_collapses = shuf_true_frac <= a.shuffle_collapse_frac * max(spk_frac, 1e-9)

    # ---- (MOAT) 0 confab leaks, 0 negated re-proposed, untaught-cue abstention unregressed ----
    moat_leaks = 0
    for (s, v, o) in spk_accepted:
        if comp.query_patient(s, v) == o:
            moat_leaks += 1
        if comp.ask_yes_no(s, v, o) == "yes":
            moat_leaks += 1
    contradictions_proposed = len(spk_set & set(negated))
    n_ab, ab_ok, guard = 0, 0, 0
    stored_cues = {(s, v) for s, v, _ in affirmed}
    while n_ab < 20 and guard < 200000:
        guard += 1
        s = nouns[int(rng.integers(len(nouns)))]
        v = verbs[int(rng.integers(len(verbs)))]
        if (s, v) in stored_cues:
            continue
        n_ab += 1
        ab_ok += int(comp.query_patient(s, v) is None)

    # ---- connective clause (bonus): join two grammatical spiking SVOs with a connective ----
    connectives = ["and", "because", "so"]
    clause_examples, n_clause, n_clause_gram = [], 0, 0
    for i in range(0, min(len(spk_accepted) - 1, 2 * a.n_clause), 2):
        t1, t2 = spk_accepted[i], spk_accepted[i + 1]
        conn = connectives[(seed + i) % len(connectives)]
        n_clause += 1
        if _grammatical(t1, is_verb_B) and _grammatical(t2, is_verb_B):
            n_clause_gram += 1
        clause_examples.append(f"{t1[0]} {t1[1]} {t1[2]} {conn} {t2[0]} {t2[1]} {t2[2]}")
    clause_grammatical_frac = n_clause_gram / max(1, n_clause)

    examples = [f"{s} {v} {o}" for (s, v, o) in spk_accepted[:12]]
    oracle = verb_oracle_prf(vocab, {w: w in verbs for w in vocab})

    print(f"\n[openended seed {seed}] vocab N={len(vocab)} (taxo-overlap {overlap_frac*100:.0f}%) | "
          f"nouns {len(nouns)} verbs {len(verbs)} | stored {len(affirmed)}+{len(negated)} | "
          f"novel-univ {len(plausible_novel_universe)} | tau={tau:.3f} | bank {build_s:.1f}s", flush=True)
    print(f"  (P) provenance {provenance_ok} (spk-draws {sampler.n_spiking_draws}, host-rng {sampler.n_host_rng_draws}) "
          f"| noise-ablation {noise_ablation_ok} (noiseless peak {float(nl_emp.max()):.2f}, noisy peak "
          f"{float(noisy_emp.max()):.2f})", flush=True)
    print(f"  (VA) induced-verb oracle prec {oracle['verb_precision']:.2f} rec {oracle['verb_recall']:.2f} "
          f"(overlap {oracle['n_overlap']})", flush=True)
    print(f"  (G) GRAMMATICAL (split-B): spiking {grammatical_frac:.3f} vs role-blind floor "
          f"{blind_grammatical_frac:.3f} -> {grammatical_advantage:.1f}x | clause-grammatical {clause_grammatical_frac:.3f}",
          flush=True)
    print(f"  (a) NOVEL: {n_spk} distinct (novel-comp {novel_comp_score:.3f}); disjoint {novel_disjoint}; "
          f"retrieval ABSTAINS {retr_abstains}/{n_spk} (all {retr_abstains_all})", flush=True)
    print(f"  (b) PLAUSIBLE: spiking {spk_frac:.3f} (adv {spk_advantage:.1f}x) vs host {host_frac:.3f} "
          f"(quality {spk_vs_host:.2f}) vs random {random_frac:.4f}", flush=True)
    print(f"  (c) LESION frac {lesion_frac:.3f} (collapses {lesion_collapses}) | SHUFFLED true {shuf_true_frac:.3f} "
          f"(collapses {shuffled_collapses})", flush=True)
    print(f"  (d) MOAT: leaks {moat_leaks} | negated re-proposed {contradictions_proposed} | untaught-abstention "
          f"{ab_ok}/{n_ab}", flush=True)
    if examples:
        print(f"  spiking SVO: {examples}", flush=True)
    if clause_examples:
        print(f"  spiking clause: {clause_examples[:3]}", flush=True)

    return {
        "seed": seed, "vocab_size": len(vocab), "taxo_overlap_frac": overlap_frac,
        "n_nouns": len(nouns), "n_verbs": len(verbs), "n_affirmed": len(affirmed), "n_negated": len(negated),
        "tau": tau, "bank_build_s": build_s,
        "discoverable_novel_plausible_universe": len(plausible_novel_universe),
        "provenance_ok": provenance_ok, "provenance_no_host_rng": provenance_no_host_rng,
        "provenance_no_rng_in_source": provenance_no_rng_in_source, "provenance_seed_rng_only": seed_rng_only,
        "n_spiking_draws": int(sampler.n_spiking_draws), "n_host_rng_draws": int(sampler.n_host_rng_draws),
        "noiseless_deterministic": noiseless_deterministic, "noisy_is_stochastic": noisy_is_stochastic,
        "noise_ablation_ok": noise_ablation_ok,
        "induced_verb_precision": oracle["verb_precision"], "induced_verb_recall": oracle["verb_recall"],
        "induced_oracle_overlap": oracle["n_overlap"],
        "grammatical_frac": grammatical_frac, "blind_grammatical_frac": blind_grammatical_frac,
        "grammatical_advantage": grammatical_advantage, "clause_grammatical_frac": clause_grammatical_frac,
        "n_spiking_generated": n_spk, "novel_composition_score": novel_comp_score,
        "novel_disjoint_from_store": novel_disjoint, "retrieval_abstains_all": retr_abstains_all,
        "spiking_examples": examples, "clause_examples": clause_examples[:6],
        "spiking_plausible_fraction_of_novel": spk_frac, "host_plausible_fraction_of_novel": host_frac,
        "random_plausible_fraction_of_novel": random_frac, "spiking_advantage_ratio": spk_advantage,
        "spiking_vs_host_quality": spk_vs_host,
        "lesion_plausible_fraction_of_novel": lesion_frac, "lesion_collapses": lesion_collapses,
        "shuffled_true_plausible_fraction_of_novel": shuf_true_frac, "shuffled_collapses": shuffled_collapses,
        "moat_leaks": moat_leaks, "contradictions_proposed": contradictions_proposed,
        "untaught_cue_abstention_correct": ab_ok, "untaught_cue_abstention_attempted": n_ab,
    }


def decide_verdict(rows, a):
    def col(k):
        return np.array([r[k] for r in rows])
    prov = col("provenance_ok"); noise_ab = col("noise_ablation_ok")
    spk_frac = col("spiking_plausible_fraction_of_novel")
    spk_adv = col("spiking_advantage_ratio"); spk_vs_host = col("spiking_vs_host_quality")
    n_gen = col("n_spiking_generated"); novel_score = col("novel_composition_score")
    novel_disjoint = col("novel_disjoint_from_store"); retr = col("retrieval_abstains_all")
    gram = col("grammatical_frac"); gram_adv = col("grammatical_advantage")
    lesion = col("lesion_collapses"); shuf = col("shuffled_collapses")
    leaks = col("moat_leaks"); contra = col("contradictions_proposed")
    ab_ok = col("untaught_cue_abstention_correct"); ab_att = col("untaught_cue_abstention_attempted")
    overlap = col("taxo_overlap_frac")

    provenance_all = bool(np.all(prov) and np.all(noise_ab))
    vocab_agnostic_all = bool(np.all(overlap <= a.max_overlap_frac) and np.all(n_gen >= a.min_novel))
    novel_all = bool(np.all(n_gen >= a.min_novel) and np.all(novel_score > 0.0)
                     and np.all(novel_disjoint) and np.all(retr))
    grammatical_all = bool(np.all(gram >= a.grammatical_bar) and np.all(gram_adv >= a.grammatical_advantage))
    advantage_all = bool(np.all(spk_adv >= a.advantage_bar))
    host_match_all = bool(np.all(spk_vs_host >= a.host_match_frac))
    lesion_all = bool(np.all(lesion)); shuf_all = bool(np.all(shuf))
    moat_all = bool(np.all(leaks == 0) and np.all(contra == 0))
    store_rate = ab_ok / np.maximum(ab_att, 1)
    store_floor_all = bool(np.all(store_rate >= a.store_floor_bar))

    detail = {
        "provenance_all_seeds": provenance_all, "vocab_agnostic_all_seeds": vocab_agnostic_all,
        "novel_all_seeds": novel_all, "grammatical_all_seeds": grammatical_all,
        "advantage_all_seeds": advantage_all, "host_match_all_seeds": host_match_all,
        "lesion_collapses_all_seeds": lesion_all, "shuffled_collapses_all_seeds": shuf_all,
        "moat_preserved_all_seeds": moat_all, "store_floor_ok_all_seeds": store_floor_all,
        "taxo_overlap_frac_mean": float(overlap.mean()),
        "grammatical_frac_mean": float(gram.mean()), "grammatical_frac_min": float(gram.min()),
        "blind_grammatical_frac_mean": float(np.mean(col("blind_grammatical_frac"))),
        "grammatical_advantage_mean": float(gram_adv.mean()),
        "clause_grammatical_frac_mean": float(np.mean(col("clause_grammatical_frac"))),
        "induced_verb_precision_mean": float(np.mean(col("induced_verb_precision"))),
        "induced_verb_recall_mean": float(np.mean(col("induced_verb_recall"))),
        "spiking_plausible_fraction_mean": float(spk_frac.mean()),
        "host_plausible_fraction_mean": float(np.mean(col("host_plausible_fraction_of_novel"))),
        "random_plausible_fraction_mean": float(np.mean(col("random_plausible_fraction_of_novel"))),
        "spiking_advantage_ratio_mean": float(spk_adv.mean()), "spiking_advantage_ratio_min": float(spk_adv.min()),
        "spiking_vs_host_quality_mean": float(spk_vs_host.mean()), "spiking_vs_host_quality_min": float(spk_vs_host.min()),
        "novel_composition_score_mean": float(novel_score.mean()), "n_spiking_generated_min": int(n_gen.min()),
        "lesion_plausible_fraction_mean": float(np.mean(col("lesion_plausible_fraction_of_novel"))),
        "shuffled_true_plausible_fraction_mean": float(np.mean(col("shuffled_true_plausible_fraction_of_novel"))),
        "moat_leaks_total": int(leaks.sum()), "contradictions_proposed_total": int(contra.sum()),
        "untaught_cue_abstention_rate_mean": float(store_rate.mean()),
        "bars": {"advantage_bar": a.advantage_bar, "host_match_frac": a.host_match_frac,
                 "grammatical_bar": a.grammatical_bar, "grammatical_advantage": a.grammatical_advantage,
                 "min_novel": a.min_novel, "max_overlap_frac": a.max_overlap_frac,
                 "shuffle_collapse_frac": a.shuffle_collapse_frac, "store_floor_bar": a.store_floor_bar},
    }

    if not provenance_all:
        verdict = "HONEST_NEGATIVE_provenance_failed"
    elif not vocab_agnostic_all:
        verdict = "HONEST_NEGATIVE_not_vocab_agnostic"
    elif not moat_all:
        verdict = "HONEST_NEGATIVE_moat_broken"
    elif not store_floor_all:
        verdict = "HONEST_NEGATIVE_untaught_abstention_regressed"
    elif not novel_all:
        verdict = "HONEST_NEGATIVE_no_novel_generated"
    elif not grammatical_all:
        verdict = "HONEST_NEGATIVE_ungrammatical"
    elif not advantage_all:
        verdict = "HONEST_NEGATIVE_no_plausibility_advantage"
    elif not host_match_all:
        verdict = "HONEST_NEGATIVE_underperforms_host_draw"
    elif not lesion_all:
        verdict = "HONEST_NEGATIVE_likelihood_not_load_bearing"
    elif not shuf_all:
        verdict = "HONEST_NEGATIVE_structure_not_load_bearing"
    else:
        verdict = "GO"
    return verdict, detail


def main():
    p = argparse.ArgumentParser(description="Vocab-agnostic spiking open-ended generation de-risk.")
    p.add_argument("--seeds", default="42,43,44,100,101,102")
    p.add_argument("--top-k", type=int, default=150, help="vocab = top-K corpus content words (arbitrary vocab)")
    p.add_argument("--D", type=int, default=96, help="phasor dim for the RF composer store (moat), sized to vocab "
                   "(D=96 gives 24/24 recall + 40/40 abstention at 150-word vocab; larger D blows up the 2*D "
                   "resonate bridge cost with no moat gain)")
    p.add_argument("--n-facts", type=int, default=24)
    p.add_argument("--n-negated", type=int, default=12)
    p.add_argument("--n-attempts", type=int, default=2000, help="host/random baseline sample-loop attempts")
    p.add_argument("--n-attempts-spiking", type=int, default=600, help="spiking SVO draws (each = 2 competitions)")
    p.add_argument("--n-clause", type=int, default=6, help="connective-clause pairs to compose (bonus)")
    p.add_argument("--base-pA", type=float, default=110.0)
    p.add_argument("--gain-pA", type=float, default=160.0)
    p.add_argument("--read-window", type=int, default=120)
    p.add_argument("--ou-std", type=float, default=200.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--calib-repeats", type=int, default=200)
    p.add_argument("--tau-pct", type=float, default=50.0)
    p.add_argument("--advantage-bar", type=float, default=3.0)
    p.add_argument("--host-match-frac", type=float, default=0.7)
    p.add_argument("--grammatical-bar", type=float, default=0.85)
    p.add_argument("--grammatical-advantage", type=float, default=1.5)
    p.add_argument("--min-novel", type=int, default=3)
    p.add_argument("--max-overlap-frac", type=float, default=0.20, help="vocab-taxonomy overlap must be <= this")
    p.add_argument("--shuffle-collapse-frac", type=float, default=0.5)
    p.add_argument("--store-floor-bar", type=float, default=0.95)
    p.add_argument("--max-bytes", type=int, default=4_000_000)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--repeat-cap", type=int, default=40)
    p.add_argument("--corpus-path", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")

    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    t0 = time.time()
    print(f"[spiking-openended] seeds={seeds} top_k={a.top_k} -- can a VOCAB-AGNOSTIC spiking soft-WTA DRAW "
          f"produce grammatical, novel, moat-safe multi-word utterances from arbitrary corpus vocab?", flush=True)

    corpus_path = a.corpus_path
    if corpus_path is None:
        for cand in (os.path.join(_REPO, "data", "corpus", "tinystories.txt"),
                     "/home/dant123/Projects/sim/data/corpus/tinystories.txt"):
            if os.path.exists(cand):
                corpus_path = cand
                break
    if not corpus_path or not os.path.exists(corpus_path):
        print(f"[ERROR] corpus not found (pass --corpus-path): {corpus_path}", flush=True)
        sys.exit(2)

    # ---- corpus -> arbitrary vocab + induced roles (split A) + the independent grammaticality oracle (split B) ----
    stories = load_stories(corpus_path, a.max_bytes)
    vocab = mine_vocab(stories, a.top_k)
    mid = len(stories) // 2
    A, B = stories[:mid], stories[mid:]
    is_verb_A, _ = morpho_distributional_tag(A, vocab)   # roles used to GENERATE
    is_verb_B, _ = morpho_distributional_tag(B, vocab)   # INDEPENDENT grammaticality oracle (disjoint split)
    nouns, verbs = roles(vocab, is_verb_A)
    oracleA = verb_oracle_prf(vocab, is_verb_A)
    ab_agree = np.mean([is_verb_A[w] == is_verb_B[w] for w in vocab])
    taxo_words = set(w for ws in TAXONOMY_8x8.values() for w in ws)
    overlap = len([w for w in vocab if w in taxo_words]) / max(1, len(vocab))
    print(f"  vocab N={len(vocab)} taxo-overlap {overlap*100:.0f}% | split-A roles: {len(nouns)} nouns, "
          f"{len(verbs)} verbs | induced-verb oracle prec {oracleA['verb_precision']:.2f} rec "
          f"{oracleA['verb_recall']:.2f} (overlap {oracleA['n_overlap']}) | A/B tag agreement {ab_agree:.3f}",
          flush=True)

    corpus = build_real_cooccurrence(corpus_path, vocab, np.zeros(len(vocab), dtype=int),
                                     window=a.window, repeat_cap=a.repeat_cap, seed=42, max_bytes=a.max_bytes,
                                     freq_floor=5, min_facts_per_category=5, verbose=False)

    rows = [run_seed(s, vocab, corpus, nouns, verbs, is_verb_B, a) for s in seeds]
    verdict, detail = decide_verdict(rows, a)
    detail["split_A_induced_verb_precision"] = oracleA["verb_precision"]
    detail["split_A_induced_verb_recall"] = oracleA["verb_recall"]
    detail["split_A_B_tag_agreement"] = float(ab_agree)
    detail["vocab_taxo_overlap_frac"] = float(overlap)

    print(f"\n{'='*100}", flush=True)
    print(f"  OVERALL VERDICT: {verdict}", flush=True)
    print(f"  (P)  provenance+noise-ablation all seeds: {detail['provenance_all_seeds']}", flush=True)
    print(f"  (VA) vocab-agnostic (overlap {detail['taxo_overlap_frac_mean']*100:.0f}% <= "
          f"{a.max_overlap_frac*100:.0f}%, >= {a.min_novel} novel all): {detail['vocab_agnostic_all_seeds']} | "
          f"induced-verb prec {detail['induced_verb_precision_mean']:.2f} rec "
          f"{detail['induced_verb_recall_mean']:.2f}", flush=True)
    print(f"  (G)  GRAMMATICAL (split-B) mean {detail['grammatical_frac_mean']:.3f} (min "
          f"{detail['grammatical_frac_min']:.3f}, >= {a.grammatical_bar} all: {detail['grammatical_all_seeds']}) "
          f"vs role-blind floor {detail['blind_grammatical_frac_mean']:.3f} -> "
          f"{detail['grammatical_advantage_mean']:.1f}x | clause-grammatical "
          f"{detail['clause_grammatical_frac_mean']:.3f}", flush=True)
    print(f"  (a)  NOVEL all seeds: {detail['novel_all_seeds']} (novel-comp mean "
          f"{detail['novel_composition_score_mean']:.3f}, min {detail['n_spiking_generated_min']} generated)",
          flush=True)
    print(f"  (b)  PLAUSIBLE: spiking {detail['spiking_plausible_fraction_mean']:.3f} (adv "
          f"{detail['spiking_advantage_ratio_mean']:.1f}x, >= {a.advantage_bar}x all: "
          f"{detail['advantage_all_seeds']}) vs host {detail['host_plausible_fraction_mean']:.3f} -- quality "
          f"{detail['spiking_vs_host_quality_mean']:.2f} (>= {a.host_match_frac} all: "
          f"{detail['host_match_all_seeds']})", flush=True)
    print(f"  (c)  LESION collapses all: {detail['lesion_collapses_all_seeds']} | SHUFFLED collapses all: "
          f"{detail['shuffled_collapses_all_seeds']}", flush=True)
    print(f"  (d)  MOAT: {detail['moat_leaks_total']} leaks + {detail['contradictions_proposed_total']} negated "
          f"re-proposed (preserved all: {detail['moat_preserved_all_seeds']}); untaught-abstention mean "
          f"{detail['untaught_cue_abstention_rate_mean']:.3f} (ok all: {detail['store_floor_ok_all_seeds']})",
          flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}\n", flush=True)

    out = {
        "probe": "spiking_openended_generation", "verdict": verdict, "seeds": seeds,
        "config": {k: getattr(a, k) for k in ("top_k", "D", "n_facts", "n_negated", "n_attempts",
                   "n_attempts_spiking", "n_clause", "base_pA", "gain_pA", "read_window", "ou_std",
                   "temperature", "calib_repeats", "tau_pct", "advantage_bar", "host_match_frac",
                   "grammatical_bar", "grammatical_advantage", "min_novel", "max_overlap_frac",
                   "shuffle_collapse_frac", "store_floor_bar", "max_bytes", "window")},
        "corpus_path": corpus_path, "vocab": vocab, "nouns": nouns, "verbs": verbs,
        "what_is_spiking_vs_host": (
            "SPIKING (firing neurons): each SVO slot is drawn by a soft-WTA competition on a real "
            "SimulationBridge Izhikevich pool (VocabAgnosticSpikingSampler == the GO followon2 SpikingWTASampler, "
            "role pools swapped from TAXONOMY_8x8 to the corpus-induced tagger) driven by the brain's PPMI "
            "likelihood + OU membrane noise; the winner read from cp_firing_states IS the word. NO host "
            "rng.choice on the draw path (source-grep + 0 host-rng draws); OU noise IS the stochasticity "
            "(ablate -> deterministic argmax). HOST SCAFFOLDS (mapped residual): the morpho-distributional role "
            "tagger, the SVO/connective template, the PPMI likelihood matrix, the RF-composer moat."),
        "detail": detail, "per_seed": rows,
        "brain_based_note": (
            "the KeyError-on-arbitrary-vocab blocker (SpikingWTASampler.__init__ -> _category_pools(TAXONOMY_8x8) "
            "-> self.row[taxonomy word]) is removed by inducing role pools from the corpus (morphological "
            "bootstrapping + Mintz-2003 frequent frames). Grammaticality is judged by an INDEPENDENT tagger "
            "re-induced on a disjoint corpus split (non-circular). The spiking DRAW, the PPMI likelihood, the "
            "no-confab moat are unchanged from followon2/b2. NO sim/ edit; reuse-by-import; CPU."),
        "elapsed_total_s": time.time() - t0,
    }
    if a.out is None:
        a.out = os.path.join(_REPO, "research", "findings", "raw", "_spiking_openended_generation_derisk.json")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {a.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
