"""Multi-sentence FLUENCY via ORDERED TOPIC-SEQUENCING on the project's SPIKING phasor substrate -- cheap-first
de-risk (the multi-sentence lever of the conversational-architecture arc).

THE QUESTION. The agent can recall a single fact and render ONE word-ordered sentence (the validated
single-sentence path: BrainConversationalAgent.describe -> RFPhasorComposer.render_fact, whose word ORDER is the
de-risked spiking competitive-queuing serial-order generator). It can also hold an ORDERED sequence of concepts in
an order-encoded working memory (the CYCLE-135 GO OrderedPositionWM: items bound to gamma-slot POSITION phasors on
the resonate-and-fire substrate; ordered recall 1.000 @ loads {2,3,5}). Does composing those two give a coherent
ORDERED MULTI-SENTENCE output -- the core of multi-sentence fluency? i.e. hold a SEQUENCE OF TOPICS in the
ordered-WM, then emit ONE sentence per slot IN SLOT ORDER (for each topic, recall a stored fact about it and render
the sentence), concatenating in slot order.

THE ARCHITECTURE (deliberately topic-sequencing, NOT nested-binding). Each WM slot holds a SINGLE concept/topic
phasor -- the VALIDATED regime of OrderedPositionWM (a single concept bound to a position). It does NOT bind a
whole COMPOSED fact into a slot; binding an already-bound SVO composite into a position phasor is nested binding
(role(x)(role(x)filler)), the project's documented SNR wall (the hierarchical-320 nesting null,
2026-06-02-full-320-flat-distinct...). By keeping slots = single topics and letting the COMPOSER hold the facts
(its own validated flat fact memory), we stay entirely inside two separately-validated regimes and only ask whether
they COMPOSE for ordered emission. The order comes from the WM slots; the content comes from the composer; the
word-order WITHIN each sentence comes from the existing serial-order renderer.

WHAT THIS GIVES THE AGENT. A multi-sentence turn: "dog ran north. cat saw river. bird ate worm." -- emitted in a
discourse order that is held in (and driven by) the spiking ordered-WM, so re-ordering the topic sequence re-orders
the output (the order is order-encoded, not a fixed storage order). Plus the no-confab moat for free: a topic with
no stored fact ABSTAINS (no sentence) rather than confabulating one.

PRE-REGISTERED, FROZEN tests + verdict (set before any multi-seed run; never tuned to a result):
1. ORDERED EMISSION (the capability): the emitted sentence SEQUENCE is in the correct slot order AND each sentence
   renders the CORRECT fact (exact (subject, verb, object) match to the topic's stored fact). Score = fraction of
   trials with the FULL ordered multi-sentence output correct, at sequence lengths K in {2, 3}.
2. ORDER-CONTROL (load-bearing): permute the topic-to-slot assignment -> the emitted sentence ORDER must permute
   correspondingly (proves the order comes from the ordered-WM slots, not a fixed/storage order). Without this, a
   "multi-sentence" output that just dumps facts in storage order would pass test 1 vacuously.
3. CONTENT-FIDELITY / NO-CONFAB (load-bearing): a topic with NO stored fact must ABSTAIN (no sentence / explicit
   "unknown") rather than confabulate. (Per the owner's 2026-06-17 moat relaxation the moat is not a hard gate, but
   it is FREE here -- kept + reported; a breach is a characterized result, not a fail-by-default.)
4. SINGLE-SENTENCE REGRESSION: the existing single-fact describe() still renders correctly (K=1) -- no regression.

GO   = ordered-emission correct at K in {2,3} AND order-control permutes AND unknown-topic abstains AND
       single-sentence un-regressed, in >= 5/6 seeds.
BOUNDARY = ordered emission works at K=2 but degrades at K=3, or one control is seed-fragile.
NEGATIVE = the ordered multi-sentence emission doesn't hold (slot order doesn't drive output order, or content
           collapses / the nested-binding wall is hit despite the topic-sequencing design).

Pure runner; reuse-by-import only; NO `sim/` edit; no automatic differentiation; no protected module modified.
Prefers CPU/numpy (the spiking RF composer + ordered WM run there; each op is a small RF bridge).

Reproduce:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_multisentence_ordered_emission_derisk
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Quiet the per-bridge init spam (each RF op builds a small bridge): keep stdout = the de-risk report only.
logging.disable(logging.INFO)

from research.runners.brain_conversational_agent import BrainConversationalAgent      # noqa: E402
from research.runners.ordered_position_wm import OrderedPositionWM                     # noqa: E402

# =====================================================================
# Pre-registered constants (frozen; never tuned to a result).
# =====================================================================
# A small fact base of 6 SVO facts. Topics (the subjects) are the discourse referents the ordered-WM holds; each
# subject has exactly one stored fact, so the correct emitted sentence per topic is well-defined.
FACTS = [
    ("dog", "ran", "north"),
    ("cat", "saw", "river"),
    ("bird", "ate", "worm"),
    ("fox", "found", "den"),
    ("frog", "crossed", "road"),
    ("hawk", "chased", "mouse"),
]
TOPICS = [s for (s, _, _) in FACTS]                       # the 6 subject topics
# A noun that is NEVER a subject of any stored fact -> the no-confab / content-fidelity probe (held as a WM
# referent, but describe() must abstain because no fact has it as agent).
UNKNOWN_TOPIC = "owl"

LOADS = [2, 3]                  # sequence lengths (slots used) for the multi-sentence emission
N_SLOTS = 7                    # gamma slots per theta cycle (Lisman-Idiart); positions fixed per seed
N_TRIALS_EMISSION = 40         # ordered-emission trials per load per seed (random ordered topic subsets)
N_TRIALS_ORDER = 40            # order-control trials per seed (a sequence + its permutation)
N_TRIALS_NOCONFAB = 30         # no-confab trials per seed (a sequence with one unknown topic)
SEEDS = [42, 43, 44, 100, 101, 102]
D_WM = 128                     # WM/composer phasor dimension (= the agent composer's D, so codes are shared)


def _build_agent_and_wm(seed):
    """Build the conversational agent (parser + RF composer + neural word-order render) and an order-encoded WM
    that SHARES the composer's concept codes (same seed/D/sorted-vocab). Store the fixed fact base.

    enable_neural_render=True -> each rendered sentence's word ORDER is produced by the de-risked spiking
    competitive-queuing serial-order generator (NeuralSerialOrderRenderer), not a host f-string. So the emitted
    multi-sentence output is neural both in (a) inter-sentence ORDER (the ordered-WM slots) and (b) intra-sentence
    word order (the serial-order renderer); only the final per-sentence join + the discourse concatenation are the
    body's emission. The WM cleans up slot reads against the TOPIC subset only (a slot read resolves to a topic,
    never an action/object word)."""
    vocab = sorted(set([w for f in FACTS for w in f] + [UNKNOWN_TOPIC]))
    concepts = {w: None for w in vocab}
    agent = BrainConversationalAgent(seed=seed, concepts=concepts, enable_neural_render=True)
    for (s, v, o) in FACTS:
        agent.hear(f"{s} {v} {o}")                       # active voice: pos0=agent, pos1=action, pos2=patient
    comp = agent.composer
    # The discourse buffer: order-encoded WM, codes byte-shared with the composer; resolves to topics only.
    wm = OrderedPositionWM(seed=seed, D=comp.D, vocab=comp.words, n_slots=N_SLOTS,
                           cleanup_words=TOPICS + [UNKNOWN_TOPIC])
    return agent, wm


def _stored_fact_str(topic):
    """The single correct rendered sentence for a topic = its stored (subject, verb, object) joined in SVO order.
    None for a topic with no stored fact (the unknown probe)."""
    for (s, v, o) in FACTS:
        if s == topic:
            return f"{s} {v} {o}"
    return None


def emit_multisentence(agent, wm, topic_sequence):
    """THE CAPABILITY. Hold `topic_sequence` in the ordered-WM (each topic bound to its successive gamma slot on the
    spiking RF substrate), then EMIT one sentence per slot IN SLOT ORDER: for k in slot order, read slot k (spiking
    unbind -> the topic), recall a stored fact about that topic and render the sentence (agent.describe), and
    concatenate in slot order. A slot whose topic does not ground (familiarity moat) OR whose topic has no stored
    fact (the composer's no-confab moat) ABSTAINS -- represented by None for that slot (no confabulated sentence).

    Returns (sentences, read_topics): `sentences` = the per-slot list (str or None), in slot order; `read_topics` =
    the per-slot topic the WM recovered (str or None)."""
    composite = wm.encode_sequence(topic_sequence)
    sentences, read_topics = [], []
    for k in range(len(topic_sequence)):
        topic, _match = wm.read_slot(composite, f"pos{k}", gate=True)   # spiking unbind + familiarity gate
        read_topics.append(topic)
        if topic is None:
            sentences.append(None)                                      # WM-level abstain (slot didn't ground)
            continue
        sentences.append(agent.describe(topic))                        # recall + render; None = no-confab abstain
    return sentences, read_topics


# ---------------------------------------------------------------------
# Test 1: ordered multi-sentence emission (the capability).
# ---------------------------------------------------------------------
def test_ordered_emission(agent, wm, loads, n_trials, seed):
    """For each K in `loads`: pick a random ORDERED subset of K topics, emit the multi-sentence output, and score a
    trial CORRECT iff (a) the emitted sentence sequence is exactly the K stored-fact sentences in the SAME ORDER as
    the topic sequence (every slot's sentence == that topic's stored fact, in order), with NO None/abstain (these
    are all groundable topics). This jointly checks order AND per-sentence content fidelity."""
    rng = np.random.default_rng(seed + 7)
    per_load = {}
    for K in loads:
        assert K <= wm.n_slots
        ok = 0
        examples = []
        for _ in range(n_trials):
            idx = list(rng.choice(len(TOPICS), size=K, replace=False))
            seq = [TOPICS[i] for i in idx]
            expected = [_stored_fact_str(t) for t in seq]              # the correct ordered sentences
            sentences, read_topics = emit_multisentence(agent, wm, seq)
            hit = (sentences == expected)
            ok += hit
            if len(examples) < 3:
                examples.append({"topic_sequence": seq, "expected": expected,
                                 "emitted": sentences, "read_topics": read_topics, "correct": bool(hit)})
        per_load[K] = {"ordered_emission_accuracy": ok / n_trials, "n_trials": n_trials, "examples": examples}
    return per_load


# ---------------------------------------------------------------------
# Test 2: order-control (load-bearing) -- the output order tracks the slot order.
# ---------------------------------------------------------------------
def test_order_control(agent, wm, n_trials, seed):
    """LOAD-BEARING. For a random ordered topic sequence and a PERMUTATION of it, the emitted sentence order must
    permute correspondingly. Concretely: emit for `seq` and for `perm(seq)`; require emitted(perm) == perm(emitted)
    -- i.e. the multi-sentence output of the permuted topics is the permuted multi-sentence output. This proves the
    inter-sentence ORDER comes from the ordered-WM SLOTS, not a fixed storage/recall order (a storage-order dump
    would emit the SAME order regardless of the topic sequence and FAIL this)."""
    rng = np.random.default_rng(seed + 21)
    ok = 0
    flips_observed = 0          # how often the permutation actually changed the order (sanity: non-trivial perms)
    examples = []
    for _ in range(n_trials):
        K = 3 if rng.random() < 0.5 else 2
        idx = list(rng.choice(len(TOPICS), size=K, replace=False))
        seq = [TOPICS[i] for i in idx]
        perm = list(rng.permutation(K))
        pseq = [seq[i] for i in perm]
        base, _ = emit_multisentence(agent, wm, seq)
        permd, _ = emit_multisentence(agent, wm, pseq)
        expected_permd = [base[i] for i in perm]                       # the order-permuted base emission
        hit = (permd == expected_permd) and (None not in base)
        ok += hit
        if perm != list(range(K)):
            flips_observed += 1
        if len(examples) < 3:
            examples.append({"seq": seq, "perm": [int(p) for p in perm], "pseq": pseq,
                             "base_emitted": base, "permuted_emitted": permd,
                             "expected_permuted": expected_permd, "correct": bool(hit)})
    return {"order_control_accuracy": ok / n_trials, "n_trials": n_trials,
            "n_nontrivial_perms": flips_observed, "examples": examples}


# ---------------------------------------------------------------------
# Test 3: content-fidelity / no-confab (load-bearing) -- unknown topic abstains.
# ---------------------------------------------------------------------
def test_no_confab(agent, wm, n_trials, seed):
    """LOAD-BEARING. Emit a sequence with one UNKNOWN topic (a referent the WM can hold but with NO stored fact).
    That slot must ABSTAIN -- emit None (no sentence) -- rather than confabulate a fact, while the OTHER slots
    still emit their correct stored sentences in order. Score: (a) abstain rate on the unknown slot, (b) the
    known slots' sentences are correct + correctly ordered. A breach = the unknown slot emitting any sentence
    (characterized, not auto-fail per the moat relaxation)."""
    rng = np.random.default_rng(seed + 33)
    abstain_ok = 0              # unknown slot abstained
    known_ok = 0               # the known slots emitted their correct sentences in order
    both_ok = 0
    examples = []
    for _ in range(n_trials):
        # A length-3 sequence: two known topics + the unknown topic, unknown placed at a random slot.
        knowns_idx = list(rng.choice(len(TOPICS), size=2, replace=False))
        knowns = [TOPICS[i] for i in knowns_idx]
        upos = int(rng.integers(0, 3))
        seq = list(knowns)
        seq.insert(upos, UNKNOWN_TOPIC)
        sentences, read_topics = emit_multisentence(agent, wm, seq)
        a_ok = (sentences[upos] is None)                               # the unknown slot abstained
        # The known slots (all positions != upos) emit their correct stored sentence, in order.
        k_ok = all(sentences[p] == _stored_fact_str(seq[p]) for p in range(len(seq)) if p != upos)
        abstain_ok += a_ok
        known_ok += k_ok
        both_ok += (a_ok and k_ok)
        if len(examples) < 3:
            examples.append({"seq": seq, "unknown_pos": upos, "emitted": sentences,
                             "read_topics": read_topics, "unknown_abstained": bool(a_ok),
                             "knowns_correct": bool(k_ok)})
    return {"unknown_abstain_accuracy": abstain_ok / n_trials,
            "known_slots_correct_accuracy": known_ok / n_trials,
            "both_accuracy": both_ok / n_trials, "n_trials": n_trials, "examples": examples}


# ---------------------------------------------------------------------
# Test 4: single-sentence regression (K=1) -- no regression of the base path.
# ---------------------------------------------------------------------
def test_single_sentence_regression(agent):
    """The existing single-fact describe() (the K=1 case) still renders each stored fact correctly -- the
    multi-sentence machinery must not regress the base single-sentence path."""
    ok = 0
    details = {}
    for t in TOPICS:
        got = agent.describe(t)
        exp = _stored_fact_str(t)
        hit = (got == exp)
        ok += hit
        details[t] = {"expected": exp, "got": got, "correct": bool(hit)}
    return {"single_sentence_accuracy": ok / len(TOPICS), "n": len(TOPICS), "details": details}


# ---------------------------------------------------------------------
# Per-seed + aggregate.
# ---------------------------------------------------------------------
def run_one_seed(seed):
    agent, wm = _build_agent_and_wm(seed)
    emission = test_ordered_emission(agent, wm, LOADS, N_TRIALS_EMISSION, seed)
    order = test_order_control(agent, wm, N_TRIALS_ORDER, seed)
    noconf = test_no_confab(agent, wm, N_TRIALS_NOCONFAB, seed)
    regression = test_single_sentence_regression(agent)

    # Per-seed GO components (frozen bars).
    emission_pass = all(emission[K]["ordered_emission_accuracy"] >= 0.80 for K in LOADS)
    order_pass = order["order_control_accuracy"] >= 0.80
    noconf_pass = noconf["unknown_abstain_accuracy"] >= 0.80 and noconf["both_accuracy"] >= 0.80
    regression_pass = regression["single_sentence_accuracy"] >= 0.999
    return {
        "seed": seed,
        "emission": emission,
        "order_control": order,
        "no_confab": noconf,
        "single_sentence_regression": regression,
        "emission_pass": bool(emission_pass),
        "order_pass": bool(order_pass),
        "no_confab_pass": bool(noconf_pass),
        "regression_pass": bool(regression_pass),
        "seed_full_pass": bool(emission_pass and order_pass and noconf_pass and regression_pass),
    }


def aggregate_and_verdict(seed_results, seeds):
    emission_means = {}
    for K in LOADS:
        vals = [seed_results[s]["emission"][K]["ordered_emission_accuracy"] for s in seeds]
        emission_means[K] = {"mean": float(np.mean(vals)), "per_seed": [round(v, 3) for v in vals],
                             "pass": bool(np.mean(vals) >= 0.80)}
    order_mean = float(np.mean([seed_results[s]["order_control"]["order_control_accuracy"] for s in seeds]))
    abstain_mean = float(np.mean([seed_results[s]["no_confab"]["unknown_abstain_accuracy"] for s in seeds]))
    both_mean = float(np.mean([seed_results[s]["no_confab"]["both_accuracy"] for s in seeds]))
    reg_mean = float(np.mean([seed_results[s]["single_sentence_regression"]["single_sentence_accuracy"]
                              for s in seeds]))
    n_full = sum(seed_results[s]["seed_full_pass"] for s in seeds)
    n_emission = sum(seed_results[s]["emission_pass"] for s in seeds)
    n_order = sum(seed_results[s]["order_pass"] for s in seeds)
    n_noconf = sum(seed_results[s]["no_confab_pass"] for s in seeds)
    n_reg = sum(seed_results[s]["regression_pass"] for s in seeds)
    n_seeds = len(seeds)
    # The FROZEN bar is ">= 5/6 of seeds" (a FRACTION; see the pre-registration above).
    # Scale it to however many seeds were actually run so a partial run (e.g. a
    # 3-seed controller verification) is judged on the same fractional bar -- the
    # earlier hardcoded absolute ">= 5" could never print GO below 5 seeds even at
    # 100% pass (a verdict-print bug; the per-seed data was unaffected).
    go_thresh = int(np.ceil((5.0 / 6.0) * n_seeds))
    # K-split (for BOUNDARY detection: K=2 works but K=3 degrades).
    k2_ok = sum(seed_results[s]["emission"][2]["ordered_emission_accuracy"] >= 0.80 for s in seeds)
    k3_ok = sum(seed_results[s]["emission"][3]["ordered_emission_accuracy"] >= 0.80 for s in seeds)

    if n_full >= go_thresh:
        verdict = "GO"
    elif n_emission == 0 or n_order == 0:
        verdict = "NEGATIVE"          # the ordered emission doesn't hold, or order doesn't drive the output
    elif k2_ok >= go_thresh and k3_ok < go_thresh:
        verdict = "BOUNDARY"          # K=2 works, K=3 degrades
    elif min(n_emission, n_order, n_noconf, n_reg) >= go_thresh:
        verdict = "GO"               # (covers the case where seed_full_pass missed a seed on a single component)
    else:
        verdict = "BOUNDARY"          # a control is seed-fragile
    return {
        "emission_means": emission_means,
        "order_control_mean": order_mean,
        "unknown_abstain_mean": abstain_mean,
        "no_confab_both_mean": both_mean,
        "single_sentence_regression_mean": reg_mean,
        "n_emission_pass": int(n_emission),
        "n_order_pass": int(n_order),
        "n_no_confab_pass": int(n_noconf),
        "n_regression_pass": int(n_reg),
        "n_full_pass": int(n_full),
        "k2_pass_seeds": int(k2_ok),
        "k3_pass_seeds": int(k3_ok),
        "n_seeds": n_seeds,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--out", type=str,
                    default=os.path.join(_REPO_ROOT, "research", "findings", "raw",
                                         "_phaseB_multisentence_ordered_emission.json"))
    args = ap.parse_args()

    import sim.backend as _b
    _, backend_name = _b.get_backend()

    print("=== multi-sentence fluency via ORDERED TOPIC-SEQUENCING on the SPIKING RF phasor substrate ===",
          flush=True)
    print(f"backend={backend_name}; D={D_WM}; n_facts={len(FACTS)}; topics={TOPICS}; unknown={UNKNOWN_TOPIC!r}",
          flush=True)
    print(f"seeds={args.seeds}; loads={LOADS}; slots={N_SLOTS}", flush=True)

    seed_results = {}
    transcript_example = None
    for seed in args.seeds:
        print(f"\n--- seed {seed} ---", flush=True)
        r = run_one_seed(seed)
        seed_results[seed] = r
        em = r["emission"]
        print("  ordered emission (full multi-sentence correct):  "
              + "  ".join(f"K{K}={em[K]['ordered_emission_accuracy']:.3f}" for K in LOADS), flush=True)
        oc = r["order_control"]
        print(f"  ORDER-CONTROL (output order permutes with slots): {oc['order_control_accuracy']:.3f}  "
              f"(non-trivial perms {oc['n_nontrivial_perms']}/{oc['n_trials']})", flush=True)
        nc = r["no_confab"]
        print(f"  NO-CONFAB (unknown topic abstains): abstain={nc['unknown_abstain_accuracy']:.3f}  "
              f"known-slots-correct={nc['known_slots_correct_accuracy']:.3f}  both={nc['both_accuracy']:.3f}",
              flush=True)
        rg = r["single_sentence_regression"]
        print(f"  single-sentence regression (K=1): {rg['single_sentence_accuracy']:.3f}", flush=True)
        print(f"  -> emission={r['emission_pass']} order={r['order_pass']} no_confab={r['no_confab_pass']} "
              f"regression={r['regression_pass']} | seed_full_pass={r['seed_full_pass']}", flush=True)
        # Capture a clean K=3 transcript from the first seed for the report.
        if transcript_example is None:
            for ex in em[3]["examples"]:
                if ex["correct"]:
                    transcript_example = {"seed": seed, **ex}
                    break

    agg = aggregate_and_verdict(seed_results, args.seeds)

    print("\n=== MULTI-SEED AGGREGATE ===", flush=True)
    for K in LOADS:
        m = agg["emission_means"][K]
        print(f"  ordered emission K{K}: mean={m['mean']:.3f} ({'>=' if m['pass'] else '<'}0.80)  "
              f"per-seed={m['per_seed']}", flush=True)
    print(f"  ORDER-CONTROL mean={agg['order_control_mean']:.3f}", flush=True)
    print(f"  NO-CONFAB unknown-abstain mean={agg['unknown_abstain_mean']:.3f}  "
          f"both(abstain+knowns) mean={agg['no_confab_both_mean']:.3f}", flush=True)
    print(f"  single-sentence regression mean={agg['single_sentence_regression_mean']:.3f}", flush=True)
    print(f"  per-seed passes: emission {agg['n_emission_pass']}/{agg['n_seeds']}  "
          f"order {agg['n_order_pass']}/{agg['n_seeds']}  no_confab {agg['n_no_confab_pass']}/{agg['n_seeds']}  "
          f"regression {agg['n_regression_pass']}/{agg['n_seeds']}  full {agg['n_full_pass']}/{agg['n_seeds']}",
          flush=True)
    print(f"  K-split: K=2 pass {agg['k2_pass_seeds']}/{agg['n_seeds']}  "
          f"K=3 pass {agg['k3_pass_seeds']}/{agg['n_seeds']}", flush=True)

    if transcript_example is not None:
        print("\n=== EXAMPLE EMITTED MULTI-SENTENCE TRANSCRIPT (K=3) ===", flush=True)
        print(f"  topic sequence (slot order): {transcript_example['topic_sequence']}", flush=True)
        emitted = transcript_example["emitted"]
        print(f"  emitted: \"{'. '.join(s for s in emitted if s)}.\"", flush=True)

    print(f"\n=== VERDICT: {agg['verdict']} ===", flush=True)
    if agg["verdict"] == "GO":
        print("  The agent produces a coherent ORDERED MULTI-SENTENCE output by holding a sequence of topics in "
              "the spiking ordered-WM and emitting one correct sentence per slot IN SLOT ORDER. The output order "
              "is order-encoded (permuting the topic sequence permutes the sentences), each sentence renders the "
              "correct stored fact, an unknown topic abstains (no confabulation), and the single-sentence path is "
              "un-regressed -- all multi-seed. Multi-sentence fluency via topic-sequencing COMPOSES the validated "
              "ordered-WM and the validated fact-recall/serial-order renderer, staying clear of the nested-binding "
              "wall.", flush=True)
    elif agg["verdict"] == "BOUNDARY":
        if agg["k2_pass_seeds"] >= 5 and agg["k3_pass_seeds"] < 5:
            print("  Ordered multi-sentence emission is GO at K=2 but DEGRADES at K=3 -- the topic-sequencing "
                  "composition holds for short discourse but the longer-sequence ordered-WM fidelity (the "
                  "bundle cross-talk over more slots) erodes the full-sequence-correct rate. BOUNDARY: works for "
                  "2-sentence turns; 3+ needs a larger D or fewer concurrent slots.", flush=True)
        else:
            print("  Ordered multi-sentence emission works, but one load-bearing control (order or no-confab) is "
                  "seed-fragile. Topic-sequencing is the right architecture; this configuration is not yet "
                  "robustly GO across all seeds.", flush=True)
    else:
        print("  The ordered multi-sentence emission does not hold: the slot order does not drive the output "
              "order, or the per-sentence content collapses. (If this is the nested-binding wall despite the "
              "topic-sequencing design, multi-sentence needs more machinery than composing the two validated "
              "regimes.)", flush=True)

    out = {
        "params": {"D_wm": D_WM, "facts": FACTS, "topics": TOPICS, "unknown_topic": UNKNOWN_TOPIC,
                   "loads": LOADS, "n_slots": N_SLOTS, "n_trials_emission": N_TRIALS_EMISSION,
                   "n_trials_order": N_TRIALS_ORDER, "n_trials_noconfab": N_TRIALS_NOCONFAB,
                   "backend": backend_name},
        "seeds": list(args.seeds),
        "per_seed": {str(s): seed_results[s] for s in args.seeds},
        "aggregate": agg,
        "example_transcript": transcript_example,
    }
    # JSON-safe: emission dicts key loads by int -> str.
    for s in out["per_seed"]:
        out["per_seed"][s]["emission"] = {str(K): v for K, v in out["per_seed"][s]["emission"].items()}
    out["aggregate"]["emission_means"] = {str(K): v for K, v in out["aggregate"]["emission_means"].items()}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
