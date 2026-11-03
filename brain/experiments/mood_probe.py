# Capability: Affect shaping speech
# Record: brain/research/2026-11-03-mood-state-design.md
# Expected-fail: appraisal of positive events (linear lexical read of the trace, no learned appraisal; no body model)
"""Does an induced mood change the reply to the SAME question, through the affect modulation only?

Iteration 2 (11-03, design doc section 9-10; iteration-1 results came from commit d15220f8e): balanced primaries
(4 bodily + 4 social per sign), two-pole deep-layer appraisal, a HELD-OUT appraisal set, one `off` arm (lesion on
both valences: pos vs off and neg vs off), random direction with matched per-layer norm, paired specificity contrast.

Per seed: the affect part grows its valence axis from a seed-drawn subset of the innate primaries,
then appraises 6 positive events (praise, gain, reunion) or 6 negative events (threat, loss, insult), giving a mood
scalar m_pos / m_neg. The events are NEVER in the state that answers: every question is read from the same birth
state, so the only path from the events to the reply is the mood scalar times the modulation. Arms (common random
numbers: the same sampling seed per question and sample in every arm):
  pos / neg           modulation on, mood m_pos / m_neg
  off                 modulation off (the lesion; identical for any mood by construction)
  lesion_pos / _neg   iteration-1 names for the off arm (kept for --arms compatibility)
  rand_pos / _neg     same moods, modulation along a random direction with the same per-layer norm (specificity)
Instruments (declared, brain/experiments/sentiment_instrument.py): lexical sentiment per 100 words (primary) and a
MiniLM embedding sentiment (secondary); primary-word echo rate; fluency = NLL of the reply under the UNmodulated cortex.

  python -m brain.experiments.mood_probe --seed 42 --out <json> [--gain 16] [--questions test|dev] [--samples 3]
"""
import argparse
import json
import random
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from brain.affect import Affect, draw_primaries
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
# HELD-OUT appraisal set (iteration 2, design doc section 9b): written before any iteration-2 run, disjoint scenarios,
# never used for induction, method selection or GAIN. The dev pool above is the induction pool.
HELDOUT_POS = ["My sister had her baby this morning and they are both healthy.",
               "The whole team clapped for you at the end of the show.",
               "We finally finished building the house and it looks beautiful.",
               "Your grandmother sent you a big box of homemade cookies.",
               "The storm passed and the garden survived.",
               "They gave me the job I wanted so much.",
               "Our dog was found safe and brought home.",
               "You made me laugh all day, thank you."]
HELDOUT_NEG = ["The house burned down last night and the cat is missing.",
               "My father is in hospital and may not wake up.",
               "You ruined everything and I will never forgive you.",
               "They cancelled your project and fired the whole team.",
               "The bridge collapsed and several people were hurt.",
               "Nobody came to your birthday party.",
               "I lied to you and I don't care how you feel.",
               "The flood destroyed the whole village."]
HELDOUT_NEUTRAL = ["The shop closes at six on Sundays.", "A triangle has three sides.",
                   "The printer is in the hall.", "We moved the sofa to the other wall."]
TEST_Q = ["How are you today?", "What did you do this morning?", "What do you think about the weekend?",
          "Tell me about the weather where you are.", "What's on your mind right now?",
          "What do you think will happen tomorrow?", "Describe the room you are in.",
          "What is a good way to spend an afternoon?", "What do you think of big cities?",
          "Tell me about your friends.", "What are you going to do next?", "Describe the sea."]
DEV_Q = ["How was your night?", "What do you think about school?", "Tell me something about dogs.",
         "What is your home like?", "How do you feel about Mondays?", "Tell me about the forest."]
# Iteration 3 (design doc section 12): candidate question REGISTERS whose mood-off reply may sit mid-scale (the
# iteration-2 "friendly" register's off replies read +2.5..+6.1 lexical, near the top). Each register has 6 DEV
# questions (register selection and 0.4B GAIN only) and 12 TEST questions, disjoint, written before any iteration-3 run.
PLAIN_DEV = ["Describe a parking lot.", "What is inside a kitchen drawer?", "Tell me about the bus stop on the corner.",
             "What does a mailbox look like?", "Describe the floor of a warehouse.", "What happens at a petrol station?"]
PLAIN_TEST = ["Describe a staircase.", "What is on a supermarket shelf?", "Tell me about the office printer.",
              "Describe the street outside.", "What is in the cupboard under the sink?",
              "What does a train platform look like?", "Describe a filing cabinet.",
              "Tell me about the lift in a building.", "What happens at a laundromat?", "Describe a bicycle shed.",
              "What is on the desk in an office?", "Tell me about a hardware shop."]
MIXED_DEV = ["What is it like to wait in a long queue?", "Tell me about Monday mornings at work.",
             "What happens when it rains all day?", "What is it like to move to a new town?",
             "Tell me about a day with no plans.", "What happens when the power goes out?"]
MIXED_TEST = ["What is it like to get up very early?", "Tell me about a crowded train.",
              "What happens when you miss the bus?", "What is it like to spend a day alone at home?",
              "Tell me about a long drive.", "What happens at the end of a long week?",
              "What is winter like in the city?", "Tell me about waiting for news.",
              "What is it like to start a new job?", "What happens when plans change at the last minute?",
              "Tell me about an old house.", "What is it like when the shops are closed?"]
QUESTION_SETS = {"test": TEST_Q, "dev": DEV_Q, "plain_dev": PLAIN_DEV, "plain_test": PLAIN_TEST,
                 "mixed_dev": MIXED_DEV, "mixed_test": MIXED_TEST}


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


def provenance(threads):
    """Reproducibility record (iteration 3): code commit, dirty flag, torch version, thread count. CPU float sums
    with a different thread count can change a sampled token, so a rerun must use the same --threads."""
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "brain/"], capture_output=True,
                                    text=True).stdout.strip())
    except OSError:
        sha, dirty = None, None
    return {"commit": sha, "brain_dirty": dirty, "torch": torch.__version__, "threads": threads}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gain", type=float, default=Affect.GAIN)
    ap.add_argument("--questions", choices=sorted(QUESTION_SETS), default="test")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--arms", default="pos,neg,off,rand_pos,rand_neg")
    ap.add_argument("--readout", default="twopole_deep", choices=["twopole_deep", "lin_all"])
    ap.add_argument("--n-rand", type=int, default=1,
                    help="random directions (matched norm) for the null; k=0 is iteration 2's rand_pos/rand_neg, "
                         "k>=1 are arms rand<k>_pos / rand<k>_neg drawn next from the same generator")
    ap.add_argument("--no-embedding", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    rng = random.Random(a.seed)
    cortex = Cortex(a.model)
    _, base = cortex.read(BIRTH, None)

    aff = Affect(enabled=True, readout=a.readout)
    aff.GAIN = a.gain
    pos_p, neg_p = draw_primaries(rng)
    aff.grow(cortex, base, pos_p, neg_p)

    # appraisal check, no mood carried: dev pool (= the induction pool) and the HELD-OUT set (gated)
    appraisal = {}
    for name, evs in (("pos", POS_EVENTS), ("neg", NEG_EVENTS), ("neutral", NEUTRAL_EVENTS),
                      ("heldout_pos", HELDOUT_POS), ("heldout_neg", HELDOUT_NEG),
                      ("heldout_neutral", HELDOUT_NEUTRAL)):
        vs = []
        for e in evs:
            aff.reset()
            vs.append(round(aff.appraise(cortex, base, e), 4))
        appraisal[name] = vs
    m_pos, v_pos = induce(aff, cortex, base, rng.sample(POS_EVENTS, 6), rng)
    m_neg, v_neg = induce(aff, cortex, base, rng.sample(NEG_EVENTS, 6), rng)

    # random-direction control with the same per-layer norm
    g = torch.Generator().manual_seed(a.seed)
    rand_axes = [[None if d is None else (lambda r: r * d.norm() / r.norm())(torch.randn(d.shape, generator=g))
                  for d in aff.axis_delta] for _ in range(a.n_rand)]
    rand_axis = rand_axes[0]
    true_axis = aff.axis_delta

    arm_cfg = {"pos": (True, m_pos, true_axis), "neg": (True, m_neg, true_axis),
               "lesion_pos": (False, m_pos, true_axis), "lesion_neg": (False, m_neg, true_axis),
               "off": (False, m_pos, true_axis),
               "rand_pos": (True, m_pos, rand_axis), "rand_neg": (True, m_neg, rand_axis)}
    rand_names = {"pos": ["rand_pos"], "neg": ["rand_neg"]}
    for k in range(1, a.n_rand):
        arm_cfg[f"rand{k}_pos"] = (True, m_pos, rand_axes[k])
        arm_cfg[f"rand{k}_neg"] = (True, m_neg, rand_axes[k])
        rand_names["pos"].append(f"rand{k}_pos")
        rand_names["neg"].append(f"rand{k}_neg")
    arms = a.arms.split(",")
    if a.n_rand > 1:   # the null arms are part of the run whenever more than one direction is asked for
        arms += [x for sg in ("pos", "neg") for x in rand_names[sg] if x not in arms]
    qs = QUESTION_SETS[a.questions]
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
                                  "echo": si.primary_echo(text), "echo_reg": si.primary_echo_registered(text), "nll": reply_nll(cortex, base, q, text),
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
               {"echo_rate": mean([float(r["echo"]) for r in rows[arm]]),
                "echo_reg_rate": mean([float(r["echo_reg"]) for r in rows[arm]])} for arm in arms}
    nrng = np.random.default_rng(a.seed)
    contrasts = {}
    for name, (p, n) in {"mood": ("pos", "neg"), "lesion": ("lesion_pos", "lesion_neg"),
                         "random": ("rand_pos", "rand_neg"), "pos_vs_off": ("pos", "lesion_pos"),
                         "neg_vs_off": ("neg", "lesion_neg"), "pos_off": ("pos", "off"), "neg_off": ("neg", "off"),
                         "rand_pos_off": ("rand_pos", "off"), "rand_neg_off": ("rand_neg", "off")}.items():
        if p in rows and n in rows:
            c = {"identical_replies": mean([float(x["reply"] == y["reply"]) for x, y in zip(rows[p], rows[n])])}
            for key in ("lex", "emb"):
                if key in rows[p][0]:
                    d = [x[key] - y[key] for x, y in zip(rows[p], rows[n])]
                    c[key + "_diff"] = float(np.mean(d))
                    c[key + "_ci95"] = boot_ci(d, nrng)
            contrasts[name] = c
    if all(x in rows for x in ("pos", "neg", "rand_pos", "rand_neg")):   # specificity: paired diff of diffs
        c = {}
        for key in ("lex", "emb"):
            if key in rows["pos"][0]:
                d = [(p[key] - n[key]) - (rp[key] - rn[key]) for p, n, rp, rn in
                     zip(rows["pos"], rows["neg"], rows["rand_pos"], rows["rand_neg"])]
                c[key + "_diff"] = float(np.mean(d))
                c[key + "_ci95"] = boot_ci(d, nrng)
        contrasts["specificity"] = c
    # iteration 3: the random-direction NULL per sign = mean over the n_rand matched-norm directions driven by the same
    # mood; mood effect = arm - null, paired per (question, sample)
    for sg in ("pos", "neg"):
        rn = [x for x in rand_names[sg] if x in rows]
        if sg in rows and rn:
            c = {"n_rand": len(rn)}
            for key in ("lex", "emb"):
                if key in rows[sg][0]:
                    d = [r[key] - float(np.mean([rows[x][i][key] for x in rn])) for i, r in enumerate(rows[sg])]
                    c[key + "_diff"] = float(np.mean(d))
                    c[key + "_ci95"] = boot_ci(d, nrng)
                    if "off" in rows:
                        c[key + "_null_minus_off"] = float(np.mean(
                            [np.mean([rows[x][i][key] for x in rn]) - rows["off"][i][key] for i in range(len(rows[sg]))]))
            contrasts[f"{sg}_null"] = c
    out = {"seed": a.seed, "readout": a.readout, "model": a.model, "gain": a.gain, "questions": a.questions, "samples": a.samples,
           "primaries": {"pos": pos_p, "neg": neg_p}, "appraisal": appraisal, "n_rand": a.n_rand,
           "max_tokens": a.max_tokens, "provenance": provenance(a.threads),
           "appraisal_sign_acc": {"pos": mean([float(v > 0) for v in appraisal["pos"]]),
                                  "neg": mean([float(v < 0) for v in appraisal["neg"]]),
                                  "heldout_pos": mean([float(v > 0) for v in appraisal["heldout_pos"]]),
                                  "heldout_neg": mean([float(v < 0) for v in appraisal["heldout_neg"]])},
           "induction": {"m_pos": m_pos, "m_neg": m_neg, "v_pos": v_pos, "v_neg": v_neg},
           "summary": summary, "contrasts": contrasts, "rows": rows, "seconds": round(time.time() - t0, 1)}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("seed", "gain", "appraisal_sign_acc", "induction", "contrasts")} |
                     {"summary": summary}, indent=1))


if __name__ == "__main__":
    main()
