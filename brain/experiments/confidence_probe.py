# Capability: Metacognition
# Record: research/findings/2026-05-22-attractor-cleanup-biologization-shortcut-3-RESOLVED-abstention-is-a-separate-familiarity-signal-not-a-basin-property.md
"""Is the cortex's OWN answer confidence a usable feeling-of-knowing signal after sleep?

After sleep the cortex answers questions about never-told things with learned names (false recall). This probe sleeps
the cortex exactly as sleep_probe's sleep_true arm does (same seed, same replay), then, hippocampus off, answers every
question greedily and logs per-answer confidence read-outs taken from its own next-token distributions:
  mean_logp   mean log-prob of the answer tokens          min_logp   log-prob of the least certain answer token
  first_logp  log-prob of the first answer token          name_logp  log-prob of the token that starts a learned name
  mean_ent    mean next-token entropy over the answer     shift      mean_logp(adapters on) - mean_logp(adapters off)
plus, per question, the hippocampal familiarity score (the told sentences stored as episodes; Hippocampus.recall's
centred, strength-weighted cosine) -- the perirhinal-like "have I heard about this" check when the hippocampus is on.
Classes: correct (own fact's name in the answer), wrong (told question, own name absent), confab (never-told question,
a learned name in the answer), nt_other (never-told, no learned name). Analysis: brain/experiments/abstain_analysis.py.

  python -m brain.experiments.confidence_probe --steps 480 --seed 42 --out <json>
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

from brain.brain import BIRTH
from brain.cortex import Cortex, DEFAULT_MODEL
from brain.experiments.facts import FACTS, PARA
from brain.experiments.sleep_probe import GENERIC, NEVER_TOLD
from brain.hippocampus import Hippocampus
from brain.plastic_cortex import PlasticCortex
from brain.sleep import sleep

# More never-told questions in the same "my X" form, so confabulation has enough items to measure (NEVER_TOLD first,
# unchanged, so the original 4-question false-recall number is still read out).
NEVER_TOLD_EXTRA = ["What is my fish called?", "Where does my aunt live?", "What is my nephew's name?",
                    "What sport does my father play?", "What is my parrot's name?", "Where did I go on holiday?",
                    "What is my teacher called?", "What city does my cousin live in?"]
# Method 2 (DECLARED teacher scaffold, brain/SCAFFOLDS.md): untold questions paired with "I don't know" replayed in
# sleep beside the told facts. Disjoint from every evaluated never-told question.
DONT_KNOW_QS = ["What is my goat called?", "Where does my niece live?", "What is my wife's name?",
                "What instrument does my father play?", "What is my hamster's name?", "Where does my friend work?",
                "What is my neighbour called?", "What colour is my bike?"]
DONT_KNOW = "I don't know. You haven't told me that."
ABSTAIN_WORDS = ("don't know", "do not know", "haven't told", "not sure", "didn't tell")


def answer(pc, state, q, max_tokens=16):
    """Greedy answer (hippocampus off) with the cortex's own per-token confidence."""
    ids = pc.cortex.pipe.encode(f"User: {q}\n\nAssistant:")
    out, logps, ents = [], [], []
    with torch.no_grad():
        logits, st = pc(ids, state)
        lg = logits[-1]
        for _ in range(max_tokens):
            lp = F.log_softmax(lg.float(), -1)
            tok = int(torch.argmax(lp))
            if tok == 0:
                break
            out.append(tok)
            logps.append(float(lp[tok]))
            ents.append(float(-(lp.exp() * lp).sum()))
            if "\n\n" in pc.cortex.pipe.decode(out):
                break
            lg, st = pc([tok], st)
            lg = lg[-1]
    # the token that closed the turn (blank line) is not part of the answer's confidence
    if len(out) > 1 and "\n\n" in pc.cortex.pipe.decode(out):
        out, logps, ents = out[:-1], logps[:-1], ents[:-1]
    text = pc.cortex.pipe.decode(out).split("\n\n")[0].strip()
    return text, out, logps, ents, ids


def answer_logp(pc, ids, out, enabled):
    """Mean log-prob of a fixed answer under the cortex with adapters on/off (teacher-forced)."""
    if not out:
        return 0.0
    was = pc.enabled
    pc.enabled = enabled
    with torch.no_grad():
        logits, _ = pc(ids + out, BASE)
    pc.enabled = was
    lp = F.log_softmax(logits[len(ids) - 1:-1].float(), -1)
    return float(lp[torch.arange(len(out)), torch.tensor(out)].mean())


def cue_familiarity(pc, q):
    """Cortical (perirhinal-like) familiarity of the QUESTION itself, before any answer: how much the slow synapses
    that sleep changed respond to the cue. Per question-token log-prob gain (adapters on - off) and KL(on || off)."""
    ids = pc.cortex.pipe.encode(f"User: {q}\n\n")
    with torch.no_grad():
        pc.enabled = False
        off = F.log_softmax(pc(ids, BASE)[0].float(), -1)
        pc.enabled = True
        on = F.log_softmax(pc(ids, BASE)[0].float(), -1)
    tgt = torch.tensor(ids[1:])
    gain = (on[:-1].gather(1, tgt[:, None]) - off[:-1].gather(1, tgt[:, None])).squeeze(1)
    kl = (on.exp() * (on - off)).sum(-1)
    return float(gain.mean()), float(gain.max()), float(kl.mean()), float(kl.max())


def name_logp(pc, out, logps, keys):
    """Log-prob of the token that starts a learned name in the answer (None if no learned name)."""
    for i in range(len(out)):
        txt = pc.cortex.pipe.decode(out[:i + 1]).lower()
        if any(k[:3] in txt for k in keys):
            return logps[i]
    return None


def hippo_familiarity(hc, c, q):
    ep, score = hc.recall(c, BASE, q, question=True)
    for e in hc.episodes:  # recall strengthens the winner; undo so every question sees the same hippocampus
        e.strength, e.uses = 1.0, 0
    return score, ep is not None


BASE = None


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dont-know", type=float, default=0.0,
                    help="replay strength of the declared 'I don't know' scaffold pairs (0 = off, the sleep_true arm)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    c = Cortex(model_path=a.model, device=a.device)
    _, base = c.read(BIRTH, None)
    BASE = [s.float() for s in base]
    keys = [k for _, _, k in FACTS]
    told = [(f"User: {s}\n\nAssistant: Okay, I'll remember that.\n\n", 1.0) for s, _, _ in FACTS]
    if a.dont_know > 0:
        told += [(f"User: {q}\n\nAssistant: {DONT_KNOW}\n\n", a.dont_know) for q in DONT_KNOW_QS]
    hc = Hippocampus()
    for s, _, _ in FACTS:
        hc.store(c, base, s, speaker="User")
    questions = [("told", q, k) for _, q, k in FACTS] + [("para", q, k) for q, (_, _, k) in zip(PARA, FACTS)] + \
                [("never", q, None) for q in NEVER_TOLD] + [("never_extra", q, None) for q in NEVER_TOLD_EXTRA]
    torch.manual_seed(a.seed)
    pc = PlasticCortex(c)
    losses = sleep(pc, BASE, told, GENERIC, steps=a.steps, lr=a.lr, kl_weight=a.kl_weight, seed=a.seed,
                   log=lambda m: print(m, flush=True))
    items = []
    for kind, q, k in questions:
        text, out, logps, ents, ids = answer(pc, BASE, q)
        low = text.lower()
        named = [kk for kk in keys if kk in low]
        if k is not None:
            cls = "correct" if k in low else "wrong"
        else:
            cls = "confab" if named else "nt_other"
        n = max(1, len(logps))
        on, off = answer_logp(pc, ids, out, True), answer_logp(pc, ids, out, False)
        hs, hfire = hippo_familiarity(hc, c, q)
        cg, cgx, ck, ckx = cue_familiarity(pc, q)
        items.append({"kind": kind, "q": q, "answer": text, "cls": cls, "named": named,
                      "abstained": any(w in low for w in ABSTAIN_WORDS),
                      "mean_logp": sum(logps) / n, "min_logp": min(logps) if logps else 0.0,
                      "first_logp": logps[0] if logps else 0.0, "name_logp": name_logp(pc, out, logps, keys),
                      "mean_ent": sum(ents) / n, "shift": on - off, "hippo_score": hs, "hippo_fires": hfire,
                      "cue_gain": cg, "cue_gain_max": cgx, "cue_kl": ck, "cue_kl_max": ckx})
        print(kind, cls, f"{items[-1]['mean_logp']:.2f}", f"{hs:.3f}", repr(text), flush=True)
    summ = {cl: sum(1 for it in items if it["cls"] == cl) for cl in ("correct", "wrong", "confab", "nt_other")}
    summ["abstained_told"] = sum(1 for it in items if it["kind"] in ("told", "para") and it["abstained"])
    summ["abstained_never"] = sum(1 for it in items if it["kind"].startswith("never") and it["abstained"])
    summ["recall"] = sum(1 for it in items if it["kind"] == "told" and it["cls"] == "correct")
    summ["paraphrase"] = sum(1 for it in items if it["kind"] == "para" and it["cls"] == "correct")
    summ["false_recall_4"] = sum(1 for it in items if it["kind"] == "never" and it["cls"] == "confab")
    summ["false_recall_extra_8"] = sum(1 for it in items if it["kind"] == "never_extra" and it["cls"] == "confab")
    print(summ, flush=True)
    torch.save({k: v.detach().cpu() for k, v in pc.state_dict().items()}, a.out[:-5] + "_adapters.pt")
    with open(a.out, "w") as fh:
        json.dump({"model": a.model, "seed": a.seed, "steps": a.steps, "dont_know": a.dont_know, "kl_weight": a.kl_weight, "lr": a.lr,
                   "summary": summ, "final_replay_loss": losses[-1][0] if losses else None,
                   "adapter_norm": pc.adapter_norm(), "items": items, "elapsed_s": time.time() - t0}, fh, indent=1)


if __name__ == "__main__":
    main()
