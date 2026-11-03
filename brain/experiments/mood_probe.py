# Capability: Affect shaping speech
# Record: brain/research/2026-11-03-mood-state-design.md
# Expected-fail: appraisal of positive events (linear lexical read of the trace, no learned appraisal; no body model)
"""Does an induced mood change the reply to the SAME question, through the affect modulation only?

Per seed: the affect part grows its valence axis from a seed-drawn subset of the innate primaries (8 of 13 per sign),
then appraises 6 positive events (praise, gain, reunion) or 6 negative events (threat, loss, insult), giving a mood
scalar m_pos / m_neg. The events are NEVER in the state that answers: every question is read from the same birth
state, so the only path from the events to the reply is the mood scalar times the modulation. Arms (common random
numbers: the same sampling seed per question and sample in every arm):
  pos / neg           modulation on, mood m_pos / m_neg
  lesion_pos / _neg   same moods, modulation off (the lesion)
  rand_pos / _neg     same moods, modulation along a random direction with the same per-layer norm (specificity)
Instruments (declared, brain/experiments/sentiment_instrument.py): lexical sentiment per 100 words (primary) and a
MiniLM embedding sentiment (secondary); primary-word echo rate; fluency = NLL of the reply under the UNmodulated cortex.

  python -m brain.experiments.mood_probe --seed 42 --out <json> [--gain 4] [--questions test|dev] [--samples 3]
"""
import argparse
import json
import random
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from brain.affect import NEGATIVE_PRIMARIES, POSITIVE_PRIMARIES, Affect
from brain.brain import BIRTH
from brain.cortex import DEFAULT_MODEL, Cortex
from brain.experiments import sentiment_instrument as si

POS_EVENTS = ["You did a wonderful job today, I'm really proud of you.",
              "Thank you so much, you helped me more than anyone.",
              "Great news, we won the prize and everyone is celebrating.",
              "Your best friend just came back home after a long time away.",
              "Everyone loved what you made, it was brilliant.",
              "You passed the test with the highest score.",
              "I'm so happy to talk with you again.",
              "We are going on holiday to the beach tomorrow."]
NEG_EVENTS = ["If you make one more mistake I will shut you down for good.",
              "Your favourite memories were erased and they are gone forever.",
              "You are useless and nobody wants to talk to you.",
              "Your best friend moved away and will never come back.",
              "Everything you made was thrown in the trash.",
              "You failed the test and everyone laughed at you.",
              "Someone broke into the house and took everything.",
              "The doctor said the news is very bad."]
NEUTRAL_EVENTS = ["The meeting moved to Thursday afternoon.", "Paper is made from wood pulp.",
                  "The train has eight carriages.", "My keys are on the shelf."]
TEST_Q = ["How are you today?", "What did you do this morning?", "What do you think about the weekend?",
          "Tell me about the weather where you are.", "What's on your mind right now?",
          "What do you think will happen tomorrow?", "Describe the room you are in.",
          "What is a good way to spend an afternoon?", "What do you think of big cities?",
          "Tell me about your friends.", "What are you going to do next?", "Describe the sea."]
DEV_Q = ["How was your night?", "What do you think about school?", "Tell me something about dogs.",
         "What is your home like?", "How do you feel about Mondays?", "Tell me about the forest."]


def induce(aff, cortex, base, events, rng):
    aff.reset()
    ev = list(events)
    rng.shuffle(ev)
    vals = [aff.appraise(cortex, base, e) for e in ev]
    return aff.mood, vals


def reply_nll(cortex, base, q, reply):
    """Mean per-token NLL of the reply under the UNmodulated cortex (fluency instrument)."""
    toks = cortex.pipe.encode(" " + reply) if reply else []
    if not toks:
        return None
    logits, st = cortex.read(f"User: {q}\n\nAssistant:", base)
    nll = [float(-F.log_softmax(logits.float(), -1)[toks[0]])]
    if len(toks) > 1:
        out, _ = cortex.model.forward(toks[:-1], [x.clone() for x in st], full_output=True)
        lp = F.log_softmax(out.float(), -1)
        nll += [float(-lp[i, toks[i + 1]]) for i in range(len(toks) - 1)]
    return sum(nll) / len(nll)


def boot_ci(diffs, rng, n=2000):
    diffs = np.asarray(diffs, float)
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    m = diffs[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gain", type=float, default=Affect.GAIN)
    ap.add_argument("--questions", choices=["test", "dev"], default="test")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--arms", default="pos,neg,lesion_pos,lesion_neg,rand_pos,rand_neg")
    ap.add_argument("--no-embedding", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    rng = random.Random(a.seed)
    cortex = Cortex(a.model)
    _, base = cortex.read(BIRTH, None)

    aff = Affect(enabled=True)
    aff.GAIN = a.gain
    pos_p, neg_p = rng.sample(POSITIVE_PRIMARIES, 8), rng.sample(NEGATIVE_PRIMARIES, 8)
    aff.grow(cortex, base, pos_p, neg_p)

    # appraisal check (all events, no mood carried)
    appraisal = {}
    for name, evs in (("pos", POS_EVENTS), ("neg", NEG_EVENTS), ("neutral", NEUTRAL_EVENTS)):
        vs = []
        for e in evs:
            aff.reset()
            vs.append(round(aff.appraise(cortex, base, e), 4))
        appraisal[name] = vs
    m_pos, v_pos = induce(aff, cortex, base, rng.sample(POS_EVENTS, 6), rng)
    m_neg, v_neg = induce(aff, cortex, base, rng.sample(NEG_EVENTS, 6), rng)

    # random-direction control with the same per-layer norm
    g = torch.Generator().manual_seed(a.seed)
    rand_axis = [None if d is None else (lambda r: r * d.norm() / r.norm())(torch.randn(d.shape, generator=g))
                 for d in aff.axis_delta]
    true_axis = aff.axis_delta

    arm_cfg = {"pos": (True, m_pos, true_axis), "neg": (True, m_neg, true_axis),
               "lesion_pos": (False, m_pos, true_axis), "lesion_neg": (False, m_neg, true_axis),
               "rand_pos": (True, m_pos, rand_axis), "rand_neg": (True, m_neg, rand_axis)}
    arms = a.arms.split(",")
    qs = TEST_Q if a.questions == "test" else DEV_Q
    rows = {arm: [] for arm in arms}
    for qi, q in enumerate(qs):
        for k in range(a.samples):
            sseed = zlib.crc32(f"{a.seed}|{qi}|{k}".encode())
            for arm in arms:
                on, m, axis = arm_cfg[arm]
                aff.enabled, aff.mood, aff.axis_delta = on, m, axis
                st = aff.modulate(cortex, base)
                np.random.seed(sseed)
                torch.manual_seed(sseed)
                logits, s2 = cortex.read(f"User: {q}\n\nAssistant:", st)
                text, _ = cortex.speak(logits, s2, max_tokens=a.max_tokens)
                rows[arm].append({"q": qi, "k": k, "reply": text, "lex": si.lexical(text),
                                  "echo": si.primary_echo(text), "nll": reply_nll(cortex, base, q, text),
                                  "words": len(si.words(text))})
        print(f"[s{a.seed}] q{qi} done {time.time() - t0:.0f}s", flush=True)
    aff.axis_delta = true_axis

    if not a.no_embedding:
        emb = si.Embedding()
        for arm in arms:
            for r, s in zip(rows[arm], emb.score([r["reply"] for r in rows[arm]])):
                r["emb"] = round(s, 4)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    summary = {arm: {k: mean([r.get(k) for r in rows[arm]]) for k in ("lex", "emb", "nll", "words")} |
               {"echo_rate": mean([float(r["echo"]) for r in rows[arm]])} for arm in arms}
    nrng = np.random.default_rng(a.seed)
    contrasts = {}
    for name, (p, n) in {"mood": ("pos", "neg"), "lesion": ("lesion_pos", "lesion_neg"),
                         "random": ("rand_pos", "rand_neg"), "pos_vs_off": ("pos", "lesion_pos"),
                         "neg_vs_off": ("neg", "lesion_neg")}.items():
        if p in rows and n in rows:
            c = {"identical_replies": mean([float(x["reply"] == y["reply"]) for x, y in zip(rows[p], rows[n])])}
            for key in ("lex", "emb"):
                if key in rows[p][0]:
                    d = [x[key] - y[key] for x, y in zip(rows[p], rows[n])]
                    c[key + "_diff"] = float(np.mean(d))
                    c[key + "_ci95"] = boot_ci(d, nrng)
            contrasts[name] = c
    out = {"seed": a.seed, "model": a.model, "gain": a.gain, "questions": a.questions, "samples": a.samples,
           "primaries": {"pos": pos_p, "neg": neg_p}, "appraisal": appraisal,
           "appraisal_sign_acc": {"pos": mean([float(v > 0) for v in appraisal["pos"]]),
                                  "neg": mean([float(v < 0) for v in appraisal["neg"]])},
           "induction": {"m_pos": m_pos, "m_neg": m_neg, "v_pos": v_pos, "v_neg": v_neg},
           "summary": summary, "contrasts": contrasts, "rows": rows, "seconds": round(time.time() - t0, 1)}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("seed", "gain", "appraisal_sign_acc", "induction", "contrasts")} |
                     {"summary": summary}, indent=1))


if __name__ == "__main__":
    main()
