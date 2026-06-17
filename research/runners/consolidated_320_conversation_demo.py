"""Consolidation demonstration — the PRODUCTION conversational agent talking with the codes it LEARNED FROM
CONVERSATION (the 320-concept fully-brain-based stream cortex). This closes the loop end-to-end on ONE agent.

WHAT WAS SEPARATE (and is joined here).
  * The 320-concept cortex (research/runners/_phaseB_onbridge_stream_conversation_derisk.py) LEARNS each word's
    meaning from a conversation stream by population-Hebbian co-occurrence, reads it out with two real cortical
    gain-control operations (per-hub spike-frequency adaptation + per-concept feedforward inhibition -- the
    `--readout-norm neural` path), and stores the result as a 320x300 real-valued concept code per seed
    (`_phaseB_stream_codes_320_neural_seed42.npy`). It was validated through a numpy HRR who/what + abstention
    pipeline, NOT through the production agent.
  * The PRODUCTION conversational agent (research/runners/brain_conversational_agent.py ->
    research/runners/rf_phasor_composer.py) parses sentences by word-position x voice, binds role-filler facts on
    resonate-and-fire phasor neurons, answers who/what queries, ABSTAINS when no fact matches (the no-confab
    moat), and -- with enable_neural_render -- produces a described sentence's word ORDER from a spiking
    competitive-queuing serial-order generator. It has always run on codes the COMPOSER self-generates, never on
    the codes the cortex learned from conversation.

THIS RUNNER joins them: it feeds the 320 stream-learned cortex codes into the production agent as its concept
vocabulary (via a fixed complex grounding projection, the same `angle(M @ code)` map the step-3 perception arc
used -- research/runners/_step3_grounded_codes_production_composer_derisk.py), then drives the production agent
through a multi-turn conversation: hear several facts, answer who/what queries, abstain on the unstored, confirm
yes/no, describe an agent (neural word order), and bring up an on-topic associate (dialogue planning). No new
mechanism -- every piece is already validated; this is the assembly + a capability gate.

THE INTERESTING SCIENCE. The cortex codes are SEMANTICALLY STRUCTURED (they carry category similarity -- that
is what lets the cortex generalize). The production binder prefers decorrelated codes, but the role-binding
decorrelates the cross-terms (tolerant to code-similarity up to ~0.98 per
research/runners/_step3_correlated_percept_boundary.py). So the prediction is: recall holds, and whatever recall
errors occur are WITHIN-CATEGORY (dog -> cat), the generalization signature -- never random. Abstention is
structurally safe regardless of code correlation, because the production moat is RELATIONAL (it abstains on
whether the fact was stored, not on code geometry).

GATE (per seed): recall == 1.0 on every stored fact (who AND what), abstain == 1.0 on the unstored set (ZERO
false-accepts -- a single false-accept is a MOAT BREACH = HARD STOP), yes/no correct, describe() returns a
correctly-ordered sentence for a known agent and None for an unknown one, elaborate() returns an on-topic
associate. Reports the grounded-code phase-cosine similarity (the structure carried) + the within-category error
rate (the generalization signature).

Run (CPU is fine -- D=128 RF ops are tiny; GPU is the real path for the dlPFC elaborate bridge):
  SIM_BACKEND=numpy python -m research.runners.consolidated_320_conversation_demo --seeds 42 --readout neural
  SIM_BACKEND=cupy  python -m research.runners.consolidated_320_conversation_demo --seeds 42 43 44 --readout host
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Reuse-by-import: the production agent, the 320-word taxonomy, nothing new.
from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.runners.stream_taxonomy_320 import TAXONOMY_40x8
from research.runners.option_c_real_cooccurrence_derisk import taxonomy_to_vocab_categories

D = 128  # the production RFPhasorComposer phasor dimension (brain_conversational_agent.py default)


# --- the grounding map (the step-3 pattern, verbatim) -----------------------
def _projection(d_out, n_in, seed):
    """A FIXED random complex projection n_in -> d_out (a fixed cortico-cortical fan-in, not learned per fact)."""
    rng = np.random.RandomState(seed * 7919 + 13)
    return (rng.standard_normal((d_out, n_in)) + 1j * rng.standard_normal((d_out, n_in))).astype(np.complex128)


def grounded_phases(code_vec, proj):
    """Real cortex code -> composer phases[D] in [0,1): the composer's _to_phasor(phases)=exp(2pi i phases) then
    equals exp(i angle(proj @ code)) -- the same grounded phasor the step-3 arc composed."""
    z = proj @ code_vec.astype(np.complex128)
    return (np.angle(z) % (2.0 * np.pi)) / (2.0 * np.pi)


def _phase_cos(pa, pb):
    """Mean phasor cosine between two phase codes in [0,1) (1 = identical, ~0 = orthogonal)."""
    return float(np.mean(np.cos(2.0 * np.pi * (pa - pb))))


# --- the multi-turn conversation (natural child-corpus SVO facts) -----------
# agents: animals/family; actions: motion_actions; patients: food/places/toys. All real taxonomy words.
FACTS = [
    ("dog", "eat", "apple"),
    ("cat", "play", "ball"),
    ("bird", "sleep", "tree"),
    ("girl", "run", "park"),
    ("boy", "look", "book"),
    ("lion", "eat", "cake"),
    ("rabbit", "jump", "garden"),
    ("mouse", "walk", "house"),
]
# unstored (agent, action) cues that must ABSTAIN -- real words, never stored together.
ABSENT_WHAT = [("dog", "sing"), ("cat", "run"), ("bird", "eat"), ("girl", "sleep"), ("lion", "jump")]
ABSENT_WHO = [("eat", "ball"), ("play", "apple"), ("run", "tree"), ("sleep", "park")]
# one explicitly-NEGATED statement, to exercise the yes/no "no" path (a bound NEGATE polarity tag).
NEG_FACT = ("fish", "eat", "cake")


def run_seed(seed, codes, vocab, cat_ids, readout):
    label = {w: c for w, c in zip(vocab, cat_ids)}
    proj = _projection(D, codes.shape[1], seed)
    grounded = {vocab[i]: grounded_phases(codes[i], proj) for i in range(len(vocab))}
    concepts = {vocab[i]: codes[i] for i in range(len(vocab))}  # sets the full 320-word vocabulary

    # the PRODUCTION agent, on the stream-learned codes, with neural word-order generation.
    agent = BrainConversationalAgent(seed=seed, concepts=concepts, grounded_codes=grounded,
                                     enable_neural_render=True)

    # grounded-code structure carried (mean off-diagonal phase-cosine over the words used in the demo).
    used = sorted({w for f in FACTS for w in f})
    sims = [_phase_cos(grounded[a], grounded[b]) for i, a in enumerate(used) for b in used[i + 1:]]
    mean_sim, max_sim = float(np.mean(sims)), float(np.max(sims))

    # TURN 1 -- hear the facts (parse by position x voice + bind on the substrate). Declarative statements are
    # affirmative, so each carries an explicit AFFIRM polarity tag (the slot the yes/no path reads); one extra
    # statement is explicitly NEGATED.
    for a, v, o in FACTS:
        agent.hear(f"{a} {v} {o}", polarity="AFFIRM")
    agent.hear(f"{NEG_FACT[0]} {NEG_FACT[1]} {NEG_FACT[2]}", polarity="NEGATE")

    # TURN 2 -- recall (who AND what), with the within-category-error signature.
    recall_ok, recall_tot, within_cat_err = 0, 0, 0
    for a, v, o in FACTS:
        pred_o = agent.what_does(a, v)
        recall_tot += 1
        if pred_o == o:
            recall_ok += 1
        elif pred_o is not None and label.get(pred_o) == label.get(o):
            within_cat_err += 1
        pred_a = agent.who_does(v, o)
        recall_tot += 1
        if pred_a == a:
            recall_ok += 1
        elif pred_a is not None and label.get(pred_a) == label.get(a):
            within_cat_err += 1
    recall = recall_ok / recall_tot

    # TURN 3 -- the no-confab moat: every unstored cue must abstain (return None). A single answer = HARD STOP.
    false_accept, abstain_tot, breaches = 0, 0, []
    for a, v in ABSENT_WHAT:
        abstain_tot += 1
        ans = agent.what_does(a, v)
        if ans is not None:
            false_accept += 1
            breaches.append(f"what_does({a},{v}) -> {ans!r} (should abstain)")
    for v, o in ABSENT_WHO:
        abstain_tot += 1
        ans = agent.who_does(v, o)
        if ans is not None:
            false_accept += 1
            breaches.append(f"who_does({v},{o}) -> {ans!r} (should abstain)")
    abstain = 1.0 - false_accept / max(abstain_tot, 1)

    # TURN 4 -- yes/no over the bound polarity tag. ask_yes_no returns "yes"/"no"/"unknown" (strings): a stored
    # AFFIRM fact -> "yes"; a stored NEGATE fact -> "no"; an unstored SVO -> "unknown" (the honest no-confab
    # answer -- an unstored fact is NEVER affirmed).
    yn_affirm = agent.is_it_true(*FACTS[0])               # expect "yes"  (dog eat apple, affirmed)
    yn_negate = agent.is_it_true(*NEG_FACT)               # expect "no"   (fish eat cake, negated)
    yn_unknown = agent.is_it_true("dog", "eat", "ball")   # expect "unknown" (never stored -> abstain)
    yn_ok = (yn_affirm == "yes") and (yn_negate == "no") and (yn_unknown != "yes")

    # TURN 5 -- describe a known agent (neural word order) + abstain on an unknown one.
    desc = agent.describe("dog")                       # expect a non-None ordered sentence mentioning dog
    desc_unknown = agent.describe("frog")              # 'frog' has no stored fact -> None (no confabulation)
    desc_ok = (desc is not None) and ("dog" in str(desc)) and (desc_unknown is None)

    # TURN 6 -- dialogue planning: bring up an on-topic associate of a known topic.
    assoc = agent.elaborate("dog")                     # dog co-occurs with eat, apple
    assoc_ok = assoc in ("eat", "apple")

    go = (recall == 1.0) and (false_accept == 0) and yn_ok and desc_ok and assoc_ok
    moat_breach = false_accept > 0
    return {
        "seed": seed, "readout": readout, "n_facts": len(FACTS),
        "recall": recall, "abstain": abstain, "false_accept": false_accept,
        "within_cat_err": within_cat_err, "grounded_mean_sim": mean_sim, "grounded_max_sim": max_sim,
        "yes_no_ok": yn_ok, "describe": str(desc), "describe_ok": desc_ok,
        "elaborate": assoc, "elaborate_ok": assoc_ok, "go": go, "moat_breach": moat_breach,
        "breaches": breaches,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--readout", choices=["neural", "host"], default="neural",
                    help="neural = the fully-brain-based read-out codes (seed 42 only so far); "
                         "host = the host double-centring read-out codes (seeds 42/43/44 available)")
    ap.add_argument("--out", default="research/findings/raw/_consolidated_320_conversation.json")
    a = ap.parse_args()

    vocab, cat_ids, _ = taxonomy_to_vocab_categories(TAXONOMY_40x8)
    suffix = "neural_seed" if a.readout == "neural" else "seed"

    print(f"[consolidated-320] production agent on the 320 stream-learned ({a.readout}) cortex codes — "
          f"does the loop close end-to-end?\n", flush=True)
    results, hard_stop = [], False
    for seed in a.seeds:
        cpath = os.path.join(_REPO, "research", "findings", "raw",
                             f"_phaseB_stream_codes_320_{suffix}{seed}.npy")
        if not os.path.exists(cpath):
            print(f"  [seed {seed}] SKIP — no {a.readout} codes at {cpath}", flush=True)
            continue
        codes = np.load(cpath)
        codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
        r = run_seed(seed, codes, vocab, cat_ids, a.readout)
        results.append(r)
        tag = "GO" if r["go"] else ("MOAT_BREACH" if r["moat_breach"] else "NEGATIVE")
        print(f"  [seed {seed}] recall {r['recall']:.2f} | abstain {r['abstain']:.2f} "
              f"(false-accept {r['false_accept']}) | within-cat-err {r['within_cat_err']} | "
              f"grounded-sim {r['grounded_mean_sim']:+.3f} (max {r['grounded_max_sim']:+.3f}) | "
              f"yes/no {'ok' if r['yes_no_ok'] else 'X'} | describe {r['describe']!r} "
              f"({'ok' if r['describe_ok'] else 'X'}) | elaborate {r['elaborate']!r} "
              f"({'ok' if r['elaborate_ok'] else 'X'})  ==> {tag}", flush=True)
        for b in r["breaches"]:
            print(f"      !! {b}", flush=True)
        hard_stop = hard_stop or r["moat_breach"]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results}, fh, indent=2, default=str)

    n_go = sum(r["go"] for r in results)
    print(f"\n{'='*100}", flush=True)
    if hard_stop:
        print("  MOAT_BREACH (HARD STOP): the production agent accepted an unstored query on the stream-learned "
              "codes — the no-confab guarantee failed; investigate before anything else.", flush=True)
    elif results and n_go == len(results):
        print(f"  GO ({n_go}/{len(results)} seeds): the PRODUCTION conversational agent converses end-to-end on the "
              "codes it LEARNED FROM CONVERSATION — recall 1.00, abstain 1.00 (0 false-accepts), yes/no, "
              "neural-ordered describe, on-topic elaborate. The loop closes: learn word meanings from a "
              "conversation stream -> converse using them through the production agent. Recall errors (if any) are "
              "WITHIN-CATEGORY (the generalization signature), abstention is structurally safe (relational moat).",
              flush=True)
    elif results:
        print(f"  PARTIAL ({n_go}/{len(results)} seeds GO): localize — recall under code correlation (within-cat "
              "errors), or a render/elaborate edge. The moat held (no breach).", flush=True)
    else:
        print("  NO CODES — run the 320 stream cortex first to produce the cached codes.", flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
