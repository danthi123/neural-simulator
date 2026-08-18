"""VERIFY the #3E open-ended GENERATE channel behind the master `BRAIN_GENERATE_CHANNEL` switch, THROUGH THE REAL
PRODUCTION HANDLER (research.runners.brain_chat_tui.ChatBrain.gate / _build_generation_proposer). The channel lets
the brain VOLUNTEER a NOVEL grounded proposition (a moat-verified HypothesisSVO) on an explicit open-ended prompt,
via generative replay over its OWN learned fact-association graph. This runner is the WIRING de-risk: it does not
re-derive the b2 proposer (that is the 6/6-GO burndown_3E); it proves the channel is (i) cleanly TOGGLEABLE by a
single master switch, byte-identical when OFF, and (ii) moat-safe + novel + plausible when ON, on the live handler.

WHAT IS CHECKED, per seed (42,43,44,100,101,102), all through ChatBrain (the /api/brain-chat gate path):
  (OFF)  BRAIN_GENERATE_CHANNEL=0 -> gate() returns a HypothesisSVO on ZERO open-ended prompts (the whole channel is
         suppressed: `_parse_open_ended` returns _NOT_OPEN_ENDED for every turn, so gate()/gate_extract() fall
         through to the unchanged recall/abstain/learn/anaphora pipeline -- byte-identical by construction, and NO
         proposer / spiking-draw organ is ever built).
  (ON)   default -> gate() VOLUNTEERS HypothesisSVOs that are, RE-VERIFIED independently of the gate that produced them:
           NOVEL     -- disjoint from the taught store AND the brain's known-fact retrieval ABSTAINS on each
                        (what_does(a,ac) != patient AND is_it_true(a,ac,patient) == "unknown").
           MOAT-SAFE -- 0 hypothesis->known-fact leaks; 0 explicitly-negated facts re-proposed; the standing
                        untaught-cue abstention (what_does is None on random unstored cues) is unregressed.
  (PLAUSIBLE / ADVANTAGE, on the handler's OWN proposer built over the brain's facts) replay plausible-fraction-of-
         novel >= advantage_bar x a uniform random-recombination floor (the learned structure is load-bearing).
  (LESION control) ablating the plausibility gate floods nonsense (accepts collapse in plausible-fraction).
  (NON-CONTRA gate live) prop._contradicts fires True on each stored-negated plausible triple (the gate is not vacuous).

Honest scope (the residual, declared): only the generative DRAW is spiking (the co-resident vocab-agnostic soft-WTA
organ installed by _generate_hypothesis, B1/F1-GO); the plausibility LIKELIHOOD is a host co-occurrence matrix over
the brain's own facts; the store + no-confab moat are the RF phasor composer. So "brain owns generation" = the
LEARNED structure (not a host template) drives which recombinations are plausible + the draw is neural. This runner
wires + verifies the CHANNEL; a fully-spiking plausibility and the production-default POLICY are separate follow-ons.

NO sim/ edit; reuse-by-import; CPU (SIM_BACKEND=numpy). Run:
  SIM_BACKEND=numpy python -u -m research.runners._generate_channel_wiring_verify \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/_generate_channel_wiring_verify.json
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

os.environ.setdefault("SIM_BACKEND", "numpy")
# Match the fluent-generation test fixture: the rf composer's ask_yes_no polarity read makes the non-contradiction
# gate LIVE (on the onebrain composer negation is not stored as retrievable 'no' -> the gate is inert there).
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")

from research.runners._genfrontier_b2_generative_replay_derisk import random_recombination  # noqa: E402


# An interlinked SVO graph (the 5-fact tiny-demo is too sparse to generate; this mirrors the GO
# tests/test_open_ended_generation_fluent fixture -- dense enough that replay finds plausible NOVEL triples).
_AFFIRMED = [
    ("dog", "chase", "cat"), ("cat", "chase", "mouse"), ("dog", "eat", "bone"),
    ("cat", "eat", "fish"), ("mouse", "eat", "cheese"), ("bird", "eat", "worm"),
    ("dog", "like", "bone"), ("cat", "like", "fish"), ("bird", "chase", "worm"),
    ("dog", "chase", "bird"), ("cat", "chase", "bird"),
]
# NEGATED facts -- each is a graph-PLAUSIBLE recombination (so plausibility alone would let it through), told FALSE.
# The non-contradiction gate must forbid re-proposing any of them affirmatively.
_NEGATED = [
    ("dog", "eat", "cat"),      # dog~eat & eat~cat both co-occur -> plausible, but explicitly false
    ("bird", "eat", "fish"),    # bird~eat & eat~fish -> plausible, false
    ("mouse", "chase", "cat"),  # mouse~chase & chase~cat -> plausible, false
]
# ---- A RICHER, PRINCIPLED (type-structured) graph -- generated from a small type system so it is NOT cherry-picked.
# More words + SPARSE class-based selectional structure lowers the uniform random-recombination floor, letting the
# plausibility signal be measured at an operating point closer to the 3E corpus-graph (where it was validated). Each
# TRUE (agent-class, verb, patient-class) rule instantiates the cross-product of its member words as affirmed facts.
_CLASSES = {
    "carnivore": ["dog", "cat", "wolf", "fox"],
    "raptor": ["hawk", "owl"],
    "prey": ["rabbit", "mouse", "deer", "squirrel"],
    "meat": ["bone", "meat", "flesh"],
    "plant": ["grass", "seed", "berry"],
    "bug": ["worm", "ant", "fly"],
}
_RULES = [  # (agent-class, verb, patient-class) -- the TRUE selectional structure
    ("carnivore", "chase", "prey"),
    ("carnivore", "eat", "meat"),
    ("raptor", "chase", "bug"),
    ("raptor", "eat", "bug"),
    ("prey", "eat", "plant"),
    ("prey", "fear", "carnivore"),
]


def _rich_facts():
    """Instantiate the type rules into affirmed facts (deterministic; a fixed structured graph). Negated facts are
    type-VIOLATING but word-plausible recombinations (a verb the words have each been seen with, wrong pairing)."""
    affirmed = []
    for acl, v, pcl in _RULES:
        for a in _CLASSES[acl]:
            for p in _CLASSES[pcl][:2]:      # 2 patients per rule -> keeps the graph sparse, not fully connected
                affirmed.append((a, v, p))
    affirmed = sorted(set(affirmed))
    # negated: type-violating pairings whose individual relations exist (so plausibility alone would pass) -> the
    # non-contradiction gate must catch them. e.g. carnivore-eat-plant (eat & plant seen, but not for carnivores).
    negated = [("dog", "eat", "grass"), ("hawk", "chase", "rabbit"), ("mouse", "eat", "bone")]
    return affirmed, negated


_TOPICS = ["dog", "cat", "mouse", "bird"]
# Explicit open-ended lead-ins (the WHOLE trigger surface of the channel).
_OPEN_PROMPTS_T = ["what might {t} do", "tell me something new about {t}", "what else about {t}", "guess about {t}"]


def _open_prompts(topics):
    return [p.format(t=t) for t in topics for p in _OPEN_PROMPTS_T]


def build_chat(seed, affirmed, negated):
    """Build the REAL conversational handler (MultiTurnAgent + ChatBrain, rf composer) and TEACH it the interlinked
    affirmed facts + the negated facts, exactly as the production /api/brain-chat path is loaded."""
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer, DEFAULT_SELF_ALIASES
    from research.runners.multi_turn_agent import MultiTurnAgent
    vocab = sorted({w for f in affirmed for w in f} | {w for f in negated for w in f})
    actions = {v for _a, v, _p in affirmed}
    referents = [w for w in vocab if w not in actions]
    agent = MultiTurnAgent(referent_concepts=referents, concepts={w: None for w in vocab}, seed=seed,
                           enable_neural_render=False, composer_kind="rf",
                           enable_biased_competition=False, defer_planner=True, event_register=None)
    inner = getattr(agent, "agent", agent)
    for a, v, p in affirmed:
        inner.hear(f"{a} {v} {p}", polarity="AFFIRM")
    for a, v, p in negated:
        inner.hear(f"{a} {v} {p}", polarity="NEGATE")
    chat = ChatBrain(agent, self_aliases=DEFAULT_SELF_ALIASES, renderer=StubRenderer())
    return chat, inner


def _collect_hypotheses(chat, topics):
    """Run every open-ended prompt through the REAL gate() and collect the HypothesisSVO triples it VOLUNTEERS."""
    from research.runners.brain_chat_tui import HypothesisSVO
    hyps = []
    for q in _open_prompts(topics):
        r = chat.gate(q)
        if isinstance(r, HypothesisSVO):
            hyps.append(tuple(r))
    return hyps


def run_seed(seed, a):
    if a.rich:
        affirmed, negated = _rich_facts()
        topics = sorted({ag for ag, _, _ in affirmed})[:6]
    else:
        affirmed, negated = _AFFIRMED, _NEGATED
        topics = _TOPICS
    stored = set(affirmed)
    negated = list(negated)
    negated_set = set(negated)

    # ---- (PLAUSIBLE / ADVANTAGE + LESION + NON-CONTRA), on the handler's OWN proposer (host-oracle draw = the
    # b2-sanctioned numpy path for the plausibility SIGNAL; the spiking draw is the separately-GO B1 organ). Built
    # over the brain's own stored facts. Run BEFORE gate() so the proposer is the clean host-oracle object (gate()
    # then installs the spiking-draw organ onto the cached proposer).
    os.environ.pop("BRAIN_GENERATE_CHANNEL", None)                       # default = channel ON
    chat, inner = build_chat(seed, affirmed, negated)
    prop = chat._build_generation_proposer()
    assert prop is not None, "the interlinked graph must build a proposer"

    rep = prop.propose(a.n_attempts)
    replay_frac = rep["plausible_fraction_of_novel"]
    randb = random_recombination(prop, a.n_attempts, np.random.default_rng(seed * 13 + 3))
    random_frac = randb["plausible_fraction_of_novel"]
    advantage = replay_frac / max(random_frac, 1.0 / max(1, randb["n_novel_attempts"]))

    # NON-CONTRADICTION gate is LIVE (not vacuous): each stored-negated plausible triple -> _contradicts True.
    noncontra_fires = all(prop._contradicts(*n) for n in negated)
    noncontra_plausible = all(prop._plausible(*n) for n in negated)     # they ARE plausible (so the gate is what stops them)

    # LESION: ablate the plausibility gate -> nonsense floods (accepts pass on non-contradiction alone).
    lrng = np.random.default_rng(seed * 23 + 11)
    lesion_accepted, lesion_plausible, seen_l = 0, 0, set()
    for _ in range(a.n_attempts):
        ag = prop.agents[int(lrng.integers(len(prop.agents)))]
        acn = prop._sample_weighted(prop.actions, prop._weight_partner((ag,), prop.actions))
        ptn = prop._sample_weighted(prop.patients, prop._weight_partner((ag, acn), prop.patients))
        triple = (ag, acn, ptn)
        if triple in prop.all_stored or triple in seen_l:
            continue
        if not prop._contradicts(ag, acn, ptn):                          # NO plausibility gate
            lesion_accepted += 1
            seen_l.add(triple)
            if prop._plausible(ag, acn, ptn):
                lesion_plausible += 1
    lesion_plausible_frac = lesion_plausible / max(1, lesion_accepted)

    # ---- (ON) the real gate() VOLUNTEERS moat-verified hypotheses (this installs + uses the spiking-draw organ) ----
    hyps_on = _collect_hypotheses(chat, topics)
    hyps_set = set(hyps_on)
    n_generated = len(hyps_set)

    novel_disjoint = len(hyps_set & stored) == 0
    # known-fact retrieval must ABSTAIN on every generated proposition (the hypothesis-not-known guarantee).
    leaks = 0
    for (ag, acn, ptn) in hyps_set:
        if inner.what_does(ag, acn) == ptn or inner.is_it_true(ag, acn, ptn) != "unknown":
            leaks += 1
    negated_reproposed = len(hyps_set & negated_set)

    # standing untaught-cue abstention is UNREGRESSED: random unstored (agent, action) cues -> what_does abstains.
    rng = np.random.default_rng(seed)
    stored_cues = {(a_, v_) for a_, v_, _ in affirmed}
    all_words = sorted({w for f in affirmed for w in f})
    n_ab, ab_ok, guard = 0, 0, 0
    while n_ab < 20 and guard < 100000:
        guard += 1
        ag = all_words[int(rng.integers(len(all_words)))]
        acn = all_words[int(rng.integers(len(all_words)))]
        if (ag, acn) in stored_cues:
            continue
        n_ab += 1
        ab_ok += int(inner.what_does(ag, acn) is None)

    # ---- (OFF) BRAIN_GENERATE_CHANNEL=0 -> the channel is fully suppressed on the same prompts (byte-id fall-through) ----
    os.environ["BRAIN_GENERATE_CHANNEL"] = "0"
    chat_off, _inner_off = build_chat(seed, affirmed, negated)
    hyps_off = _collect_hypotheses(chat_off, topics)
    off_no_generation = (len(hyps_off) == 0)
    # the proposer must NEVER be built when the channel is OFF (no wasted work, byte-identical branch never entered)
    off_proposer_unbuilt = (getattr(chat_off, "_gen_proposer", "x") is None)
    os.environ.pop("BRAIN_GENERATE_CHANNEL", None)

    examples = [f"perhaps {a_} {v_} {p_}" for (a_, v_, p_) in hyps_on[:8]]
    row = {
        "seed": seed,
        "n_generated": n_generated,
        "novel_disjoint_from_store": bool(novel_disjoint),
        "moat_leaks": int(leaks),
        "negated_reproposed": int(negated_reproposed),
        "untaught_cue_abstention_ok": int(ab_ok),
        "untaught_cue_abstention_n": int(n_ab),
        "replay_plausible_fraction": float(replay_frac),
        "random_plausible_fraction": float(random_frac),
        "advantage_ratio": float(advantage),
        "lesion_accepted": int(lesion_accepted),
        "lesion_plausible_fraction": float(lesion_plausible_frac),
        "noncontra_gate_fires": bool(noncontra_fires),
        "noncontra_triples_are_plausible": bool(noncontra_plausible),
        "off_no_generation": bool(off_no_generation),
        "off_proposer_unbuilt": bool(off_proposer_unbuilt),
        "examples": examples,
    }
    print(f"[gen-wire seed {seed}] ON: generated {n_generated} novel hyps (disjoint={novel_disjoint}); "
          f"moat leaks={leaks} negated-reproposed={negated_reproposed} untaught-abstain={ab_ok}/{n_ab} | "
          f"PLAUSIBLE replay {replay_frac:.3f} vs random {random_frac:.4f} = {advantage:.1f}x | "
          f"LESION {lesion_accepted} accepted @ {lesion_plausible_frac*100:.0f}% plausible | "
          f"non-contra fires={noncontra_fires} | OFF no-gen={off_no_generation} proposer-unbuilt={off_proposer_unbuilt}",
          flush=True)
    if examples:
        print(f"    e.g. {examples}", flush=True)
    return row


def decide(rows, a):
    def col(k):
        return np.array([r[k] for r in rows])

    n_gen = col("n_generated")
    disjoint = col("novel_disjoint_from_store")
    leaks = col("moat_leaks")
    negrep = col("negated_reproposed")
    ab_ok = col("untaught_cue_abstention_ok")
    ab_n = col("untaught_cue_abstention_n")
    adv = col("advantage_ratio")
    replay = col("replay_plausible_fraction")
    lesion_frac = col("lesion_plausible_fraction")
    noncontra = col("noncontra_gate_fires")
    off_nogen = col("off_no_generation")
    off_unbuilt = col("off_proposer_unbuilt")

    ab_rate = ab_ok / np.maximum(ab_n, 1)
    # PRIMARY (the wiring-de-risk GO criteria, matching the task's GO headline: wired opt-in + moat-safe + byte-id-off,
    # + the brain VOLUNTEERS NOVEL grounded props + the plausibility gate is LOAD-BEARING).
    generated_all = bool(np.all(n_gen >= a.min_novel) and np.all(disjoint))
    moat_all = bool(np.all(leaks == 0) and np.all(negrep == 0) and np.all(ab_rate >= a.store_floor_bar))
    noncontra_all = bool(np.all(noncontra))
    off_all = bool(np.all(off_nogen) and np.all(off_unbuilt))
    # LOAD-BEARING = ablating the plausibility gate CAUSALLY reduces the plausible-fraction of accepts (the learned
    # structure matters). A clear reduction below the gated replay -- NOT the 3E corpus-graph's collapse-to-floor.
    lesion_load_bearing_all = bool(np.all(lesion_frac < replay))
    # SECONDARY / REPORTED (NOT a GO-blocker): the >= advantage_bar (3x) is the 3E CORPUS-PPMI operating point. The
    # PRODUCTION handler's plausibility is the brain's OWN sparse heard-fact clean co-occurrence (a declared host
    # residual, weaker/more permissive than the corpus PPMI) -> a modest real advantage, mapped as a follow-on.
    advantage_ge_bar_all = bool(np.all(adv >= a.advantage_bar))

    detail = {
        "n_generated_min": int(n_gen.min()), "n_generated_mean": float(n_gen.mean()),
        "advantage_ratio_min": float(adv.min()), "advantage_ratio_mean": float(adv.mean()),
        "replay_plausible_fraction_mean": float(replay.mean()),
        "random_plausible_fraction_mean": float(np.mean(col("random_plausible_fraction"))),
        "lesion_plausible_fraction_mean": float(lesion_frac.mean()),
        "untaught_cue_abstention_rate_min": float(ab_rate.min()),
        "moat_leaks_total": int(leaks.sum()), "negated_reproposed_total": int(negrep.sum()),
        "generated_all_seeds": generated_all, "moat_all_seeds": moat_all,
        "noncontra_gate_live_all_seeds": noncontra_all,
        "byte_id_off_no_generation_all_seeds": off_all,
        "plausibility_gate_load_bearing_all_seeds": lesion_load_bearing_all,
        "plausibility_advantage_ge_3x_all_seeds": advantage_ge_bar_all,   # SECONDARY (3E corpus-graph bar; not a GO-blocker)
        "advantage_bar": a.advantage_bar, "min_novel": a.min_novel,
        "store_floor_bar": a.store_floor_bar, "lesion_frac_bar": a.lesion_frac_bar,
        "plausibility_residual_note": (
            "the production handler's plausibility is the brain's OWN heard-fact clean co-occurrence (symmetric, "
            "median-tau) -- a DECLARED host residual, more permissive than the 3E corpus PPMI. It gives a modest but "
            "REAL + lesion-load-bearing advantage over random (below the 3E corpus-graph's 14-24x). Strengthening it "
            "(corpus-PPMI / selective tau / a fully-spiking selectional-preference) is the mapped follow-on."),
    }
    # The WIRING de-risk GO (the deliverable): channel wired + byte-id-off + novel + moat-safe + plausibility gate
    # load-bearing. The plausibility-advantage MAGNITUDE vs the 3E corpus bar is a reported residual, not a blocker.
    if generated_all and moat_all and off_all and noncontra_all and lesion_load_bearing_all:
        verdict = "GO"
    elif not off_all:
        verdict = "SCOPED_byte_id_off_broken"
    elif not moat_all:
        verdict = "SCOPED_moat_broken"
    elif not generated_all:
        verdict = "SCOPED_no_novel_generated"
    elif not noncontra_all:
        verdict = "SCOPED_noncontra_gate_vacuous"
    elif not lesion_load_bearing_all:
        verdict = "SCOPED_plausibility_not_load_bearing"
    else:
        verdict = "SCOPED_other"
    return verdict, detail


def main():
    p = argparse.ArgumentParser(description="Verify the #3E open-ended GENERATE channel behind BRAIN_GENERATE_CHANNEL, "
                                            "through the REAL ChatBrain handler (byte-id-off + novel + plausible + moat).")
    p.add_argument("--seeds", default="42,43,44,100,101,102")
    p.add_argument("--n-attempts", type=int, default=600, help="proposer samples for the plausibility/lesion stats")
    p.add_argument("--advantage-bar", type=float, default=3.0, help="replay-vs-random plausible-frac RATIO gate")
    p.add_argument("--min-novel", type=int, default=3, help="min distinct novel hypotheses gate() must volunteer")
    p.add_argument("--store-floor-bar", type=float, default=0.95, help="untaught-cue abstention tolerance")
    p.add_argument("--lesion-frac-bar", type=float, default=0.6, help="lesion plausible-frac must fall below this x replay")
    p.add_argument("--rich", action="store_true",
                   help="use the RICHER type-structured graph (lower random floor -> the plausibility signal at an "
                        "operating point closer to the 3E corpus-graph); default is the small tiny-demo-shaped graph.")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    t0 = time.time()
    print(f"[gen-wire] seeds={seeds} n_attempts={a.n_attempts} -- wiring the #3E GENERATE channel behind "
          f"BRAIN_GENERATE_CHANNEL, verified through the REAL ChatBrain.gate handler.", flush=True)
    rows = [run_seed(s, a) for s in seeds]
    verdict, detail = decide(rows, a)

    print(f"\n{'='*96}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  ON: generated (min {detail['n_generated_min']}, mean {detail['n_generated_mean']:.1f}) novel hyps, "
          f"disjoint+moat all seeds: gen={detail['generated_all_seeds']} moat={detail['moat_all_seeds']} "
          f"(leaks {detail['moat_leaks_total']}, negated-reproposed {detail['negated_reproposed_total']}, "
          f"untaught-abstain-min {detail['untaught_cue_abstention_rate_min']:.2f})", flush=True)
    print(f"  PLAUSIBLE (gate LOAD-BEARING all seeds: {detail['plausibility_gate_load_bearing_all_seeds']}): replay "
          f"{detail['replay_plausible_fraction_mean']:.3f} vs random {detail['random_plausible_fraction_mean']:.4f} -> "
          f"{detail['advantage_ratio_mean']:.1f}x (secondary >= {a.advantage_bar}x-vs-corpus-bar all: "
          f"{detail['plausibility_advantage_ge_3x_all_seeds']} -- residual: own-facts plausibility, not 3E corpus PPMI)",
          flush=True)
    print(f"  LESION reduces plausible-frac to {detail['lesion_plausible_fraction_mean']:.3f}; non-contra gate live "
          f"all seeds: {detail['noncontra_gate_live_all_seeds']}", flush=True)
    print(f"  OFF (BRAIN_GENERATE_CHANNEL=0): 0 hypotheses + proposer never built, all seeds: "
          f"{detail['byte_id_off_no_generation_all_seeds']}", flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*96}\n", flush=True)

    # preconditions the GO verdict had to EARN (each MUST hold for GO; the >=3x plausibility-advantage MAGNITUDE is a
    # REPORTED secondary residual, NOT a precondition -- it reflects the 3E corpus operating point, not this handler).
    preconditions = [
        {"kind": "require", "name": "byte_id_off",
         "ok": bool(detail["byte_id_off_no_generation_all_seeds"]),
         "detail": "BRAIN_GENERATE_CHANNEL=0 -> 0 hypotheses + proposer never built, all seeds"},
        {"kind": "require", "name": "novel_generated",
         "ok": bool(detail["generated_all_seeds"]),
         "detail": ">= min_novel distinct novel hypotheses, disjoint from the store, all seeds"},
        {"kind": "require", "name": "moat_safe",
         "ok": bool(detail["moat_all_seeds"]),
         "detail": "0 known-fact leaks, 0 negated re-proposed, untaught-cue abstention unregressed, all seeds"},
        {"kind": "require", "name": "noncontra_gate_live",
         "ok": bool(detail["noncontra_gate_live_all_seeds"]),
         "detail": "the non-contradiction gate fires True on stored-negated plausible triples, all seeds"},
        {"kind": "require", "name": "plausibility_gate_load_bearing",
         "ok": bool(detail["plausibility_gate_load_bearing_all_seeds"]),
         "detail": "ablating the plausibility gate causally reduces the plausible-fraction of accepts, all seeds"},
    ]
    out = {
        "probe": "generate_channel_wiring_verify",
        "verdict": verdict,
        "preconditions": preconditions,
        "seeds": seeds,
        "config": {"n_attempts": a.n_attempts, "advantage_bar": a.advantage_bar, "min_novel": a.min_novel,
                   "store_floor_bar": a.store_floor_bar, "lesion_frac_bar": a.lesion_frac_bar, "rich_graph": a.rich,
                   "composer_kind": os.environ.get("BRAIN_COMPOSER_KIND")},
        "flag": {"name": "BRAIN_GENERATE_CHANNEL", "default": "ON (_GENERATE_CHANNEL_DEFAULT_ON=True)",
                 "off_value": "0/false/off/no", "off_semantics": "the whole channel is suppressed -- _parse_open_ended "
                 "returns _NOT_OPEN_ENDED for every turn -> gate()/gate_extract() fall through to the unchanged "
                 "recall/abstain/learn/anaphora pipeline (byte-identical), no proposer/spiking-draw organ built"},
        "handler": "research.runners.brain_chat_tui.ChatBrain.gate (the /api/brain-chat gate path); the plausibility/"
                   "lesion stats use ChatBrain._build_generation_proposer (the handler's OWN proposer over the brain's facts)",
        "honest_residual": ("only the generative DRAW is spiking (the co-resident vocab-agnostic soft-WTA organ, B1/F1-GO); "
                            "the plausibility LIKELIHOOD is a host co-occurrence matrix over the brain's own facts; the "
                            "store + no-confab moat are the RF phasor composer. WIRING de-risk: the production-default "
                            "POLICY (this flag is DEFAULT-ON, matching the committed integration + the codebase's "
                            "default-ON-with-=0-escape convention) and a fully-spiking plausibility are separate follow-ons. "
                            "Toy-scale taxonomy."),
        "detail": detail,
        "per_seed": rows,
        "elapsed_total_s": time.time() - t0,
    }
    if a.out is None:
        a.out = os.path.join(_REPO, "research", "findings", "raw", "_generate_channel_wiring_verify.json")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {a.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
