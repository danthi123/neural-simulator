"""STANDALONE VERIFY HARNESS for the vocab-agnostic spiking generative-DRAW production organ (numpy-CPU).

Proves, at the PRODUCTION operating point (the brain's OWN runtime stored-fact association graph, NOT the corpus
PPMI the de-risk used), that routing the #3E generative DRAW through `VocabAgnosticSpikingDrawOrgan`:

  (WALL)  the b2 taxonomy `SpikingWTASampler` KeyErrors on this runtime lexicon (the blocker this removes),
          while the vocab-agnostic sampler builds cleanly over the SAME lexicon.
  (A) SPIKING PROVENANCE + integration: the installed proposer's DRAW runs on FIRING NEURONS -- every draw is a
      cp_firing_states read (n_host_rng_draws == 0, n_spiking_draws > 0) -- and the production `propose` loop
      still yields novel grounded hypotheses through it.
  (B) LESION load-bearing: ablating the likelihood (uniform drive) collapses the plausibility of the drawn
      recombinations (spiking selectional-preference is caused by the brain's graph, not an input artifact).
  (C) NOISE-ABLATION: ou_std->0 collapses the draw to a DETERMINISTIC argmax (the OU noise IS the stochasticity).
  (D) FLAG-OFF byte-identical: `BRAIN_SPIKING_DRAW=0` leaves the proposer on the host oracle draw -> the draw
      sequence is identical to the pre-organ production path.
  (E) MOAT preserved: every accepted hypothesis is NOVEL (never a stored fact), NON-CONTRADICTORY, and PLAUSIBLE
      -- the downstream gates are untouched, so 0 leaks.

Run:  SIM_BACKEND=numpy python -u -m research.runners._vocab_agnostic_spiking_generation_production_organ_verify
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

from research.runners._genfrontier_b2_generative_replay_derisk import GenerativeReplayProposer
from research.runners._spiking_openended_generation_derisk import (
    VocabAgnosticSpikingSampler,
    _gate_and_collect,
)
from research.runners._followon2_spiking_wta_sampler_derisk import SpikingWTASampler
import research.runners.vocab_agnostic_spiking_generation_production_organ as ORGAN
from tools.lab import attributable_to
from tools.verdict import Verdict


# ---- a runtime brain: a handful of grounded SVO facts with SPARSE selectional preference (arbitrary vocab,
# NOT the 8x8 taxonomy). Verbs select specific patients; agents do specific actions -> uniform recombination is
# mostly implausible, so the likelihood-driven draw is measurably better than the lesioned one. ----
FACTS = [
    ("dog", "chase", "ball"), ("dog", "eat", "meat"), ("dog", "bark", "mailman"),
    ("cat", "chase", "mouse"), ("cat", "eat", "fish"), ("cat", "climb", "tree"),
    ("fox", "chase", "hare"), ("fox", "eat", "berry"),
    ("owl", "eat", "mouse"), ("owl", "hunt", "vole"),
    ("boy", "kick", "ball"), ("boy", "throw", "stick"),
    ("girl", "throw", "ball"), ("girl", "read", "book"),
    ("bird", "build", "nest"), ("bird", "eat", "worm"),
    ("horse", "eat", "hay"), ("horse", "pull", "cart"),
]
NEGATED = [("cat", "eat", "grass")]   # the brain was EXPLICITLY told this is false (the non-contradiction target)


class _StubComposer:
    """The minimal composer surface the b2 proposer's gates touch: `ask_yes_no(a,ac,p)` -> 'no' ONLY for an
    explicitly-stored NEGATED fact (what `_contradicts` reads). Everything else is 'unknown'. This is the
    teacher/environment boundary -- the harness stands in for the real RF composer's no-confab store."""

    def __init__(self, negated):
        self._neg = {(a, ac, p) for a, ac, p in negated}

    def ask_yes_no(self, a, ac, p):
        return "no" if (a, ac, p) in self._neg else "unknown"


def build_production_proposer(facts, negated, seed):
    """Build a `GenerativeReplayProposer` EXACTLY as production `ChatBrain._build_generation_proposer` does:
    clean concept co-occurrence over the brain's stored facts -> P/row, tau = 50th pctile of positive edges,
    host oracle draw by default (use_spiking_sampler=False). The graph covers affirmed AND negated triples (every
    concept the brain has heard is in `row`), mirroring production `_build_generation_proposer` (facts = all kb)."""
    graph = {}
    for a, v, p in list(facts) + list(negated):
        cs = [c for c in (a, v, p) if isinstance(c, str)]
        for x in cs:
            for y in cs:
                if x != y:
                    graph.setdefault(x, {})[y] = graph.get(x, {}).get(y, 0.0) + 1.0
    vocab = sorted(graph.keys())
    row = {w: i for i, w in enumerate(vocab)}
    P = np.zeros((len(vocab), len(vocab)), dtype=float)
    for a, nbrs in graph.items():
        for b, w in nbrs.items():
            P[row[a], row[b]] = float(w)
    tau = float(np.percentile(P[P > 0], 50.0))
    comp = _StubComposer(negated)
    prop = GenerativeReplayProposer(comp, facts, negated, P, row, tau,
                                    np.random.default_rng(seed * 7 + 1), use_spiking_sampler=False)
    return prop


def _plausible_fraction(sampler, proposer, all_stored, n):
    raw = sampler.draw_many(n)
    return _gate_and_collect(raw, proposer, all_stored)["plausible_fraction_of_novel"], len(raw)


def main():
    seed = 42
    all_stored = set(FACTS) | set(NEGATED)
    fails = []
    results = {"probe": "vocab_agnostic_spiking_generation_production_organ_verify", "seed": seed,
               "operating_point": {"base_pA": ORGAN._BASE_PA, "gain_pA": ORGAN._GAIN_PA,
                                    "read_window": ORGAN._READ_WINDOW, "ou_std": ORGAN._OU_STD}}

    def check(name, cond, detail=""):
        ok = bool(cond)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}", flush=True)
        if not ok:
            fails.append(name)

    print("=" * 96)
    print("VOCAB-AGNOSTIC SPIKING generative-DRAW production organ -- verify (numpy-CPU)")
    print("=" * 96)

    # ---------------------------------------------------------------- WALL: taxonomy sampler KeyErrors here
    prop = build_production_proposer(FACTS, NEGATED, seed)
    print(f"runtime lexicon: {len(prop.agents)} agents, {len(prop.actions)} actions, "
          f"{len(prop.patients)} patients (vocab {len(prop.row)}); tau={prop.tau:.3f}")
    taxo_keyerrored = False
    try:
        SpikingWTASampler(prop.P, prop.row, prop.tau, seed=seed, n_cand_max=96)
    except KeyError:
        taxo_keyerrored = True
    results["runtime_lexicon"] = {"n_agents": len(prop.agents), "n_actions": len(prop.actions),
                                  "n_patients": len(prop.patients), "vocab": len(prop.row), "tau": float(prop.tau)}
    results["taxonomy_sampler_keyerrors"] = bool(taxo_keyerrored)
    check("WALL: taxonomy SpikingWTASampler KeyErrors on runtime vocab", taxo_keyerrored,
          "the blocker this organ removes")
    try:
        _ = ORGAN.VocabAgnosticSpikingDrawOrgan(seed=seed).build_sampler(prop, lesion=False)
        vocab_agnostic_builds = True
    except Exception as e:  # noqa: BLE001
        vocab_agnostic_builds = False
        print(f"       (vocab-agnostic build raised: {type(e).__name__}: {e})")
    results["vocab_agnostic_builds"] = bool(vocab_agnostic_builds)
    check("WALL removed: vocab-agnostic sampler builds on the SAME runtime vocab", vocab_agnostic_builds)

    # ---------------------------------------------------------------- (A) SPIKING provenance + integration
    os.environ.pop("BRAIN_SPIKING_DRAW", None)
    os.environ.pop("BRAIN_SPIKING_DRAW_LESION", None)
    ORGAN._ORGAN = None                                   # fresh process-shared organ
    prop_a = build_production_proposer(FACTS, NEGATED, seed)
    info = ORGAN.install_spiking_draw(prop_a, seed=seed)
    check("(A) organ installs (default-ON)", info.get("on") is True, str(info))
    check("(A) proposer routed to spiking draw", prop_a.use_spiking_sampler is True
          and isinstance(prop_a._spiking_sampler, VocabAgnosticSpikingSampler))
    rep = prop_a.propose(80)                              # the PRODUCTION draw path, now spiking
    accepted = rep["accepted"]
    prov = ORGAN.get_organ().provenance()
    results["provenance"] = {"n_spiking_draws": int(prov["n_spiking_draws"]),
                             "n_host_rng_draws": int(prov["n_host_rng_draws"])}
    results["n_accepted_hypotheses"] = len(accepted)
    results["example_hypotheses"] = [list(t) for t in accepted[:6]]
    check("(A) draw is on FIRING NEURONS: 0 host-rng draws, >0 spiking draws", prov["n_host_rng_draws"] == 0
          and prov["n_spiking_draws"] > 0, f"spiking={prov['n_spiking_draws']}, host_rng={prov['n_host_rng_draws']}")
    check("(A) integration yields novel grounded hypotheses through the spiking draw", len(accepted) > 0,
          f"{len(accepted)} accepted, e.g. {accepted[:6]}")

    # ---------------------------------------------------------------- (E) MOAT preserved: 0 leaks
    leaks = [t for t in accepted if (t in all_stored) or prop_a._contradicts(*t) or (not prop_a._plausible(*t))]
    results["moat_leaks"] = len(leaks)
    check("(E) MOAT: every accepted hypothesis is novel + non-contradictory + plausible (0 leaks)",
          len(leaks) == 0, f"{len(leaks)} leaks")
    # the non-contradiction gate is DISCRIMINATIVE: it rejects a told-false triple, passes a merely-novel one.
    check("(E) MOAT: non-contradiction gate rejects the explicitly-negated triple",
          prop_a._contradicts("cat", "eat", "grass") is True and prop_a._contradicts("dog", "chase", "hare") is False)

    # ---------------------------------------------------------------- (B) LESION load-bearing (plausibility collapses)
    ORGAN._ORGAN = None
    prop_i = build_production_proposer(FACTS, NEGATED, seed)
    ORGAN.get_organ(seed).install(prop_i, lesion=False)
    intact_frac, n_i = _plausible_fraction(prop_i._spiking_sampler, prop_i, all_stored, 120)

    ORGAN._ORGAN = None
    prop_l = build_production_proposer(FACTS, NEGATED, seed)
    ORGAN.get_organ(seed).install(prop_l, lesion=True)
    lesion_frac, n_l = _plausible_fraction(prop_l._spiking_sampler, prop_l, all_stored, 120)

    collapse_ok = lesion_frac <= 0.5 * intact_frac        # the de-risk's LESION-collapse condition
    results["intact_plausible_fraction"] = round(float(intact_frac), 3)
    results["lesion_plausible_fraction"] = round(float(lesion_frac), 3)
    results["lesion_collapses"] = bool(collapse_ok)
    check("(B) LESION collapses plausibility (likelihood is load-bearing)", collapse_ok,
          f"intact plausible-frac {intact_frac:.3f} (n={n_i}) -> lesioned {lesion_frac:.3f} (n={n_l})")
    # ATTRIBUTION: what fraction of the drawn plausibility is caused by the brain's likelihood (intact) vs is
    # ALSO present in the likelihood-ablated control (lesion) -- the effect must belong to the manipulation.
    attr = attributable_to("spiking-draw plausibility (intact vs likelihood-lesion)", intact_frac, lesion_frac)
    results["plausibility_attributable_to_likelihood"] = None if attr is None else round(float(attr), 3)

    # ---------------------------------------------------------------- (C) NOISE-ABLATION -> deterministic argmax
    noiseless = VocabAgnosticSpikingSampler(
        prop_i.P, prop_i.row, prop_i.tau,
        sorted(set(prop_i.agents) | set(prop_i.patients)), sorted(set(prop_i.actions)),
        seed=seed, n_cand_max=max(96, len(prop_i.row)), ablate_noise=True)
    cand = noiseless.actions
    w = noiseless._weights([noiseless.encodable_agents[0]], cand)
    winners = {noiseless.draw_from_weights(w, cand) for _ in range(12)}
    results["noiseless_distinct_winners_over_12"] = len(winners)
    check("(C) NOISE-ABLATION (ou_std=0): the draw is a DETERMINISTIC argmax", len(winners) == 1,
          f"{len(winners)} distinct winners over 12 repeats -> {winners}")

    # ---------------------------------------------------------------- (D) FLAG-OFF byte-identical to host oracle
    os.environ["BRAIN_SPIKING_DRAW"] = "0"
    ORGAN._ORGAN = None
    prop_off = build_production_proposer(FACTS, NEGATED, seed)
    off_info = ORGAN.install_spiking_draw(prop_off, seed=seed)
    prop_ref = build_production_proposer(FACTS, NEGATED, seed)      # never touched by the organ
    rep_off = prop_off.propose(80)
    rep_ref = prop_ref.propose(80)
    identical = (rep_off["accepted"] == rep_ref["accepted"])
    results["flag_off_byte_identical"] = bool(identical)
    results["flag_off_n_accepted"] = len(rep_off["accepted"])
    check("(D) FLAG-OFF install is a no-op", off_info.get("on") is False
          and prop_off.use_spiking_sampler is False and prop_off._spiking_sampler is None, str(off_info))
    check("(D) FLAG-OFF draw is byte-identical to the pre-organ host oracle path", identical,
          f"{len(rep_off['accepted'])} vs {len(rep_ref['accepted'])} accepted, equal={identical}")
    os.environ.pop("BRAIN_SPIKING_DRAW", None)

    # ---------------------------------------------------------------- earn the verdict (preconditions travel)
    vd = Verdict("vocab_agnostic_spiking_draw_production_organ")
    vd.require("WALL: taxonomy sampler blocked on runtime vocab", taxo_keyerrored, True)
    vd.require("WALL removed: vocab-agnostic sampler builds", vocab_agnostic_builds, True)
    vd.require("draw on FIRING NEURONS (0 host-rng draws)", prov["n_host_rng_draws"] == 0, True)
    vd.require("draw ran on spikes (>0 spiking draws)", prov["n_spiking_draws"] > 0, True)
    vd.require("integration yields >=1 novel grounded hypothesis", len(accepted) > 0, True)
    vd.require("MOAT: 0 leaks", len(leaks) == 0, True)
    vd.control("LESION plausibility (intact vs likelihood-ablated)", intact_frac, lesion_frac,
               min_separation=0.5 * intact_frac,
               note="uniform drive collapses the drawn plausibility -> the likelihood is load-bearing")
    vd.require("NOISE-ABLATION: ou_std=0 -> deterministic argmax", len(winners) == 1, True)
    vd.require("FLAG-OFF byte-identical to host oracle draw", identical, True)
    vd.disabled("recurrent plasticity in the WTA bank",
                "the sampler is an unwired GENERIC_UNSTRUCTURED Izhikevich bank (STDP/Hebb/STP/homeostasis off) "
                "-- the DRAW is spiking; the likelihood/template/moat remain host scaffolds (carrier residuals)")
    decided = vd.decide(go=(not fails))

    print("=" * 96)
    verdict = decided["status"]
    results.update(decided)                    # top-level status + preconditions block (gates: verdict_preconditions)
    results["verdict"] = verdict
    results["failed_gates"] = fails
    out = os.environ.get("ORGAN_VERIFY_OUT") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "findings", "raw", "_vocab_agnostic_spiking_generation_production_organ_verify.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(f"  VERDICT: {verdict}" + ("" if not fails else f"  (failed: {fails})"))
    print(f"  [saved] {out}")
    print("=" * 96)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
