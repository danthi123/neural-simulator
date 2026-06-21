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
research/runners/_step3_correlated_percept_boundary.py), so recall stays perfect. Abstention is structurally
safe regardless of code correlation, because the production moat is RELATIONAL (it abstains on whether the fact
was stored, not on code geometry). (A follow-up tested whether recall ERRORS under noise are within-category --
they are NOT; they are near-random, because the category margin in the codes is thin and swamped by the noise
that causes errors. See 2026-06-17-within-category-error-signature-NEGATIVE.md. So this runner reports
within_cat_err for completeness, but does not claim a within-category error signature.)

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


def run_seed(seed, codes, vocab, cat_ids, readout, composer_kind="rf", spiking_cleanup=True, integrated_loop=False):
    label = {w: c for w, c in zip(vocab, cat_ids)}
    proj = _projection(D, codes.shape[1], seed)
    grounded = {vocab[i]: grounded_phases(codes[i], proj) for i in range(len(vocab))}
    concepts = {vocab[i]: codes[i] for i in range(len(vocab))}  # sets the full 320-word vocabulary

    # the PRODUCTION agent, on the stream-learned codes, with neural word-order generation. composer_kind="onebrain"
    # routes the WHOLE who/what pipeline onto ONE persistent co-resident spiking bridge (the validated one-brain path);
    # default "rf" = the production numpy composer / test oracle. The grounded codes pass through to either composer.
    # spiking_cleanup (burndown #1, default ON for the flagship one-brain path): the cleanup SELECTION (the winner-pick
    # over the matched-filter membrane) is a fully-on-substrate spiking Izhikevich WTA -- the host argmax retired, so the
    # WHOLE conversational turn (parse -> bind -> store -> unbind -> select -> abstain) is brain-based on one bridge.
    # == host-argmax answers + moat 0-FA (tests/test_onebrain_spiking_cleanup.py). `--no-spiking-cleanup` is the escape.
    # Applied to the onebrain (production) path only; rf stays its host-argmax oracle / numpy-CPU default.
    use_spiking_cleanup = bool(spiking_cleanup) and composer_kind == "onebrain"
    # integrated_loop (shortcut #3, default OFF = byte-identical = the host-_scan oracle): route the (agent, action)
    # cue-match-and-first-match SELECTION through the validated spiking K-way sequencer (gated-disinhibition match
    # cascade + BG first-match priority WTA) at match_thresh=0.06, so that routing op is neurons firing, not a host
    # Python first-match loop. Applied to the onebrain path only (it needs the on-bridge composer). NOT the default
    # (a default flip is a separate, gated step like the spiking-cleanup burndown). The no-confab moat is preserved.
    use_integrated_loop = bool(integrated_loop) and composer_kind == "onebrain"
    # enable_learned_assoc (cheat-D, onebrain production path only): dialogue planning (elaborate) spreads over the
    # SUBSTRATE-LEARNED sparse Hebbian recurrent assoc graph instead of the host Python co-occurrence dict -- so that
    # association op is neurons firing. Validated at scale (test_learned_assoc_graph_agent + 9/9 top-associate); the
    # library/test default stays False because the Hebbian graph is underpowered at toy 2-fact scale (the rf oracle /
    # numpy-CPU path keeps the deterministic host dict). The production onebrain conversation closes this shortcut.
    use_learned_assoc = composer_kind == "onebrain"
    agent = BrainConversationalAgent(seed=seed, concepts=concepts, grounded_codes=grounded,
                                     enable_neural_render=True, composer_kind=composer_kind,
                                     enable_spiking_cleanup=use_spiking_cleanup,
                                     enable_learned_assoc=use_learned_assoc,
                                     integrated_loop=use_integrated_loop)

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
                    help="neural (PRODUCTION DEFAULT) = the fully-brain-based on-bridge read-out normalization codes "
                         "(per-hub spike-frequency adaptation + per-concept feedforward inhibition; burndown #5, "
                         "seeds 42/43/44 == the host who/what baseline with the moat at 0 false-accepts). "
                         "host = the host double-centring read-out codes (the escape / test-oracle path, seeds 42/43/44).")
    ap.add_argument("--out", default="research/findings/raw/_consolidated_320_conversation.json")
    ap.add_argument("--composer", choices=["rf", "onebrain"], default="onebrain",
                    help="onebrain = the integrated one-brain composer (the whole who/what pipeline on ONE persistent "
                         "spiking bridge, no host round-trips between ops) -- the PRODUCTION DEFAULT (320-scale GO 3/3 "
                         "seeds 2026-06-18: recall 1.00, abstain 1.00, 0 false-accepts); needs SIM_BACKEND=cupy. "
                         "rf = the RFPhasorComposer (the TEST ORACLE + the numpy-CPU path).")
    ap.add_argument("--no-spiking-cleanup", dest="spiking_cleanup", action="store_false",
                    help="DISABLE the fully-on-substrate spiking cleanup SELECTION (burndown #1) and fall back to the "
                         "host argmax over the matched-filter membrane. Default ON: the cleanup winner-pick is a "
                         "spiking Izhikevich WTA (== host answers + moat 0-FA, tests/test_onebrain_spiking_cleanup.py), "
                         "so the whole conversational turn is brain-based. The escape is for the numpy-CPU / oracle path.")
    ap.set_defaults(spiking_cleanup=True)
    ap.add_argument("--integrated-loop", dest="integrated_loop", action="store_true",
                    help="(shortcut #3) route the (agent, action) cue-match-and-first-"
                         "match SELECTION through the validated spiking K-way sequencer (gated-disinhibition match "
                         "cascade + BG first-match priority WTA, match_thresh=0.06) instead of the host first-match "
                         "loop -- so that routing op is neurons firing. onebrain path only; the no-confab moat is "
                         "preserved (answer-identical + fa_total 0, the #3 fold de-risk). DEFAULT ON as of the "
                         "production-wiring pass (validated V=320 GO 4/4): the flagship production who/what is spiking.")
    ap.add_argument("--no-integrated-loop", dest="integrated_loop", action="store_false",
                    help="disable the spiking sequencer -> the host first-match _scan (the byte-identical escape / "
                         "numpy-CPU / test-oracle path).")
    ap.set_defaults(integrated_loop=True)
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
        r = run_seed(seed, codes, vocab, cat_ids, a.readout, composer_kind=a.composer,
                     spiking_cleanup=a.spiking_cleanup, integrated_loop=a.integrated_loop)
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
              "conversation stream -> converse using them through the production agent. Abstention is structurally "
              "safe (the relational host moat is code-independent). (Note: recall errors under noise are NEAR-RANDOM, "
              "not semantically biased -- the codes' category structure is real but thin-margin; see "
              "2026-06-17-within-category-error-signature-NEGATIVE.md.)",
              flush=True)
    elif results:
        print(f"  PARTIAL ({n_go}/{len(results)} seeds GO): localize — recall under code correlation (within-cat "
              "errors), or a render/elaborate edge. The moat held (no breach).", flush=True)
    else:
        print("  NO CODES — run the 320 stream cortex first to produce the cached codes.", flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
