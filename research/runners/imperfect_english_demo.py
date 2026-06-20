"""R1 -- a side-by-side SHOWCASE that makes the (already-validated) robust multi-cue Competition-Model parser
VISIBLE: it comprehends IMPERFECT / non-canonical English (object-fronted, scrambled, dropped function words)
where the default ORDER-ONLY parser inverts the roles -- while the no-confab moat holds in BOTH.

This builds NO new mechanism. It runs the SAME `BrainConversationalAgent` twice on the SAME bridge config:
  (a) order-only:  enable_multicue_competition=False -> the position-by-construction `(position x voice) -> role`
                   map (`BridgeParser._GT`); brittle -- a corrupted word order corrupts the role.
  (b) multi-cue:   enable_multicue_competition=True  -> the validated SPIKING multi-cue role-COMPETITION
                   (`MultiCueRoleParser`): word order COMPETES with animacy + verb-selectional-fit, each weighted
                   by its learned validity, so the surviving content cues carry the role assignment when order is
                   degraded.

For each imperfect input (ground truth: an ANIMATE agent acts on an INANIMATE patient), we print the role
assignment each agent COMPREHENDED, then have each STORE its parse and answer a who/what query, so the divergence
shows up at the behavioral level too: the multi-cue agent answers correctly; the order-only agent either abstains
(it stored the inverted fact, so the content-correct query misses) or returns the inverted answer.

The no-confab MOAT (an UNKNOWN-subject query returns None / abstains) is verified in BOTH agents -- the
robustness win never comes at the cost of confabulation.

Provenance: deep-research GATE `research/findings/2026-06-20-robust-multicue-parser-deep-research.md` recommended
exactly this (R1: flip the imperfect-English DEMO to the validated multi-cue parser, scoped to the demo, NOT the
library default -- to preserve numpy-CPU portability + because the validated scope is the 2-noun transitive). The
capability + its anti-cheat controls are CI-guarded by `tests/test_multicue_competition_agent.py`. Reuse-by-import;
NO `sim/` edit.

Run (GPU is the production substrate; numpy works for a tiny smoke):
    SIM_BACKEND=cupy python -m research.runners.imperfect_english_demo --seed 42 \
        --out research/findings/raw/_R1_imperfect_english_demo.json
"""
from __future__ import annotations

import argparse
import json
import sys

from research.runners.brain_conversational_agent import BrainConversationalAgent, _GT

# rf composer with an explicit vocab -> no denoise64 cache needed (mirrors the CI guard). Animate agents +
# inanimate patients so the semantic cues (animacy + verb-selectional-fit) are decisive on a degraded word order.
NOUNS = ["dog", "cat", "fox", "bird", "wolf", "apple", "ball", "rock", "bone", "stick"]
VERBS = ["eat", "chase", "push", "carry", "bite", "kick", "grab"]
VOCAB = {w: None for w in NOUNS + VERBS}

# The imperfect / non-canonical transitive battery. Each entry: (surface words, ground-truth agent, ground-truth
# patient, a short label of HOW it is degraded). Ground truth is always animate-agent / inanimate-patient, the case
# the content cues resolve and the position cue (sole cue of the order-only parser) gets backwards.
IMPERFECT_BATTERY = [
    (["apple", "eat", "dog"],   "dog", "apple", "object-fronted (patient first)"),
    (["ball", "kick", "cat"],   "cat", "ball",  "object-fronted (patient first)"),
    (["bone", "dog", "bite"],   "dog", "bone",  "scrambled + dropped function words (verb last)"),
    (["rock", "fox", "push"],   "fox", "rock",  "scrambled + dropped function words (verb last)"),
]

# A canonical control (the multi-cue parser must NOT break the native word order).
CANONICAL = (["wolf", "carry", "stick"], "wolf", "stick")

# An UNKNOWN-subject query for the no-confab moat: a noun with no stored fact must abstain (None) in BOTH agents.
MOAT_UNKNOWN_AGENT = "stick"   # never an agent in this battery -> what_does(stick, ...) must be None
MOAT_UNKNOWN_VERB = ("chase", "bird")   # no 'chase' fact stored at all -> who_does('chase','bird') must be None


def _order_only_roles(words):
    """The DEFAULT (order-only) parser's decision, PURELY position-by-construction: `_GT` maps position*2+voice ->
    role (pos0=agent, pos1=action, pos2=patient). This is exactly what the agent stores with the flag OFF -- the
    load-bearing contrast (it inverts an object-fronted sentence, and on a verb-last input it even files the verb
    as the patient)."""
    return {_GT[pos * 2]: w for pos, w in enumerate(words)}


def _build_agents(seed):
    """Two agents, SAME bridge config, differing ONLY in the multi-cue flag (rf composer + explicit vocab)."""
    order_only = BrainConversationalAgent(seed=seed, composer_kind="rf", concepts=VOCAB)  # flag default OFF
    multi_cue = BrainConversationalAgent(seed=seed, composer_kind="rf", concepts=VOCAB,
                                         enable_multicue_competition=True, multicue_verbs=VERBS)
    return order_only, multi_cue


def _comprehend_and_answer(agent, words, gt_agent, gt_patient):
    """Have `agent` hear the sentence (storing its parse), then answer the content-correct who/what query. Returns
    a dict capturing what it comprehended + how it answered (so the order-only vs multi-cue divergence is visible
    at both the parse level and the behavioral level)."""
    sentence = " ".join(words)
    roles = agent.hear(sentence)                                # comprehend + store (its own parser decides)
    verb = roles.get("action")
    parsed_agent, parsed_patient = roles.get("agent"), roles.get("patient")
    # behavioral readout: the content-correct queries (who did <verb> to <gt_patient>? / what did <gt_agent> <verb>?)
    who = agent.who_does(verb, gt_patient)
    what = agent.what_does(gt_agent, verb)
    parse_correct = (parsed_agent == gt_agent and parsed_patient == gt_patient)
    answer_correct = (who == gt_agent and what == gt_patient)
    return {
        "parsed_agent": parsed_agent, "parsed_patient": parsed_patient, "parsed_action": verb,
        "who_does": who, "what_does": what,
        "parse_correct": bool(parse_correct), "answer_correct": bool(answer_correct),
    }


def run_demo(seed=42, verbose=True):
    """Run the side-by-side demo. Returns a result dict (also the JSON payload). Each input is heard by a FRESH
    pair of agents so a prior stored fact can never leak into a later query (clean per-sentence comparison)."""
    lines = []

    def out(s=""):
        lines.append(s)
        if verbose:
            print(s)

    out("=" * 92)
    out("R1 -- IMPERFECT-ENGLISH COMPREHENSION: multi-cue Competition-Model parser vs order-only parser")
    out("=" * 92)
    out("Ground truth for every item: an ANIMATE agent acts on an INANIMATE patient.")
    out("The order-only parser assigns roles by WORD POSITION alone, so degraded order -> inverted roles.")
    out("The multi-cue parser COMPETES position against animacy + verb-fit (learned validity) -> robust.")
    out("")

    per_sentence = []
    mc_parse_correct = oo_parse_correct = 0
    mc_answer_correct = oo_answer_correct = 0

    for words, gt_agent, gt_patient, degradation in IMPERFECT_BATTERY:
        sentence = " ".join(words)
        order_only, multi_cue = _build_agents(seed)             # fresh pair per sentence (no cross-leak)
        oo = _comprehend_and_answer(order_only, words, gt_agent, gt_patient)
        mc = _comprehend_and_answer(multi_cue, words, gt_agent, gt_patient)
        # the order-only static decision (what the position map would assign, independent of any composer state)
        oo_static = _order_only_roles(words)

        mc_parse_correct += int(mc["parse_correct"]); oo_parse_correct += int(oo["parse_correct"])
        mc_answer_correct += int(mc["answer_correct"]); oo_answer_correct += int(oo["answer_correct"])

        out(f'INPUT: "{sentence}"   [{degradation}]')
        out(f'   ground truth        :  agent={gt_agent:<5}  patient={gt_patient}')
        out(f'   order-only parser   :  agent={oo_static["agent"]:<5}  patient={oo_static["patient"]:<5}'
            f'  -> {"CORRECT" if oo["parse_correct"] else "WRONG (roles inverted)"}')
        out(f'   multi-cue parser    :  agent={mc["parsed_agent"]:<5}  patient={mc["parsed_patient"]:<5}'
            f'  -> {"CORRECT" if mc["parse_correct"] else "WRONG"}')
        out(f'   behavioral readout  :  who_does("{mc["parsed_action"]}","{gt_patient}")?  '
            f'order-only={oo["who_does"]!s:<6} multi-cue={mc["who_does"]!s:<6}'
            f'   (truth={gt_agent})')
        out("")

        per_sentence.append({
            "sentence": sentence, "degradation": degradation,
            "ground_truth": {"agent": gt_agent, "patient": gt_patient},
            "order_only": oo, "order_only_static": oo_static, "multi_cue": mc,
        })

    n = len(IMPERFECT_BATTERY)

    # --- canonical control: the multi-cue parser must not break native word order ---
    cwords, cga, cgp = CANONICAL
    order_only, multi_cue = _build_agents(seed)
    c_oo = _comprehend_and_answer(order_only, cwords, cga, cgp)
    c_mc = _comprehend_and_answer(multi_cue, cwords, cga, cgp)
    out("-" * 92)
    out(f'CANONICAL CONTROL: "{" ".join(cwords)}" (native SVO)  -> '
        f'multi-cue parse {"CORRECT" if c_mc["parse_correct"] else "WRONG"}, '
        f'order-only parse {"CORRECT" if c_oo["parse_correct"] else "WRONG"}  '
        f'(the multi-cue parser does not break the native case)')
    out("")

    # --- no-confab moat: an UNKNOWN-subject query must abstain (None) in BOTH agents ---
    order_only, multi_cue = _build_agents(seed)
    # store one canonical fact in each so the kb is non-empty, then query an UNSTORED subject/relation
    order_only.hear(" ".join(cwords)); multi_cue.hear(" ".join(cwords))
    moat = {
        "order_only_unknown_agent": order_only.what_does(MOAT_UNKNOWN_AGENT, "carry"),
        "multi_cue_unknown_agent":  multi_cue.what_does(MOAT_UNKNOWN_AGENT, "carry"),
        "order_only_unknown_verb":  order_only.who_does(*MOAT_UNKNOWN_VERB),
        "multi_cue_unknown_verb":   multi_cue.who_does(*MOAT_UNKNOWN_VERB),
    }
    moat_held = all(v is None for v in moat.values())
    out("-" * 92)
    out("NO-CONFAB MOAT (an UNKNOWN-subject / unstored-relation query must abstain -> None):")
    out(f'   what_does("{MOAT_UNKNOWN_AGENT}", "carry")  order-only={moat["order_only_unknown_agent"]!s:<6} '
        f'multi-cue={moat["multi_cue_unknown_agent"]!s:<6}  (both must be None)')
    out(f'   who_does("{MOAT_UNKNOWN_VERB[0]}", "{MOAT_UNKNOWN_VERB[1]}") order-only={moat["order_only_unknown_verb"]!s:<6} '
        f'multi-cue={moat["multi_cue_unknown_verb"]!s:<6}  (both must be None)')
    out(f'   -> moat {"HELD (zero confabulation, both agents)" if moat_held else "BREACHED <-- FAILURE"}')
    out("")

    # --- summary ---
    out("=" * 92)
    out("SUMMARY")
    out(f'   multi-cue parser  : {mc_parse_correct}/{n} imperfect sentences comprehended CORRECTLY '
        f'(behavioral who/what {mc_answer_correct}/{n})')
    out(f'   order-only parser : {oo_parse_correct}/{n} imperfect sentences comprehended CORRECTLY '
        f'(behavioral who/what {oo_answer_correct}/{n})')
    out(f'   canonical control : multi-cue {"PASS" if c_mc["parse_correct"] else "FAIL"} '
        f'(native word order not broken)')
    out(f'   no-confab moat    : {"HELD" if moat_held else "BREACHED"} in BOTH agents')
    # the showcase succeeds iff the multi-cue parser beats order-only on the imperfect inputs AND the moat holds
    demo_pass = (mc_parse_correct == n and oo_parse_correct < mc_parse_correct
                 and c_mc["parse_correct"] and moat_held)
    out(f'   -> DEMO {"PASS" if demo_pass else "FAIL"}: multi-cue ({mc_parse_correct}/{n}) beats '
        f'order-only ({oo_parse_correct}/{n}) on imperfect English, moat intact')
    out("=" * 92)

    return {
        "demo": "R1_imperfect_english",
        "seed": seed,
        "n_imperfect": n,
        "multi_cue_parse_correct": mc_parse_correct,
        "order_only_parse_correct": oo_parse_correct,
        "multi_cue_answer_correct": mc_answer_correct,
        "order_only_answer_correct": oo_answer_correct,
        "canonical_control_multicue_correct": bool(c_mc["parse_correct"]),
        "canonical_control_orderonly_correct": bool(c_oo["parse_correct"]),
        "moat_held": bool(moat_held),
        "moat_detail": moat,
        "demo_pass": bool(demo_pass),
        "per_sentence": per_sentence,
        "transcript": lines,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="R1 side-by-side: multi-cue vs order-only parser on imperfect English")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None, help="optional JSON output path")
    args = ap.parse_args(argv)

    result = run_demo(seed=args.seed, verbose=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n[wrote] {args.out}")

    # non-zero exit on a demo failure so a CI/launcher can detect it
    return 0 if result["demo_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
