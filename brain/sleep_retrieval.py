"""Retrieval practice in sleep, with the hippocampal episode as the teacher (confabulation method M3).

Plan: brain/research/2026-11-03-confabulation-next-methods.md (M3). Biology: retrieval is a fast route to consolidation
(testing effect, Roediger & Karpicke 2006; Antony et al. 2017); CLS: the hippocampus teaches the cortex by replay and
learning is driven by the cortex's own error on the replayed item, interleaved with old knowledge (McClelland et al. 1995).

Ordinary sleep (brain/sleep.py) restudies the told sentence. Here each replay is a self-test instead:
  1. Before sleep, for every episode the cortex itself poses questions about it (SELFQ_PROMPT, a declared scaffold:
     brain/SCAFFOLDS.md, the same prompt as the meaning-selection learned key), and the cortex WITH the episode
     reinstated (hippocampus on: the episode in context) answers each one. That answer is the teacher.
  2. In sleep, the cortex WITHOUT the episode (hippocampus off: base state + slow synapses only) answers the same cue.
     Its greedy answer is compared with the teacher's answer (two routes, no host labels, no key words). Because greedy
     decoding is deterministic, "the free answer diverges from the teacher before the teacher's answer ends" equals
     "the teacher-forced argmax differs from the teacher token at some answer position", so one forward pass gives
     both the attempt and the loss.
  3. Loss = cross-entropy on the answer span only, each token weighted by the cortex's own error on it (1 + (1 - p)),
     normalised over the span (so the weighting moves the learning to where the answer goes wrong, e.g. a wrong
     learned name, without changing the overall step size). Replay priority: an episode whose last attempt failed is
     replayed PRIO times as often as one that succeeded.
  4. KL guard on ordinary text unchanged from brain/sleep.py.
No answer text is written by the host: cues and answers are the cortex's own outputs.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

SELFQ_PROMPT = "User: {s}\n\nAssistant: Okay, I'll remember that.\n\nUser: What"
TOLD = "User: {s}\n\nAssistant: Okay, I'll remember that.\n\n"


def selfq_text(o: str) -> str:
    """Re-attach the "What" onset (speak() strips the continuation's leading space)."""
    o = o.split("\n")[0].rstrip()
    return "What" + ("" if o.startswith(("'", "?", ",")) else " ") + o


def self_cues(cortex, birth, statement, n_cues=4, seed=0):
    """Questions the cortex itself poses while replaying the episode: one greedy, the rest sampled (T 1.0, top_p 0.7)."""
    lg, st = cortex.read(birth + SELFQ_PROMPT.format(s=statement), None)
    outs = [cortex.speak(lg, st, max_tokens=24, temperature=0)[0]]
    for k in range(n_cues - 1):
        torch.manual_seed(seed * 1000 + k)
        outs.append(cortex.speak(lg, st, max_tokens=24, temperature=1.0, top_p=0.7)[0])
    cues = [selfq_text(o) for o in outs]
    return list(dict.fromkeys(c for c in cues if len(c) > len("What ")))


def greedy_answer(plastic, state, q, max_tokens=16):
    """Greedy reply of the plastic cortex from `state` to a user question (same decoding as the probes' ask())."""
    ids = plastic.cortex.pipe.encode(f"User: {q}\n\nAssistant:")
    out = []
    with torch.no_grad():
        logits, st = plastic(ids, state)
        lg = logits[-1]
        for _ in range(max_tokens):
            tok = int(torch.argmax(lg))
            if tok == 0:
                break
            out.append(tok)
            if "\n\n" in plastic.cortex.pipe.decode(out):
                break
            lg, st = plastic([tok], st)
            lg = lg[-1]
    return plastic.cortex.pipe.decode(out).split("\n\n")[0].strip()


def teacher_answers(plastic, base_state, statement, cues, max_tokens=16):
    """The hippocampal teacher: the episode reinstated (told turn in context), the cortex's slow synapses at birth."""
    was = plastic.enabled
    plastic.enabled = False
    try:
        _, st = plastic(plastic.cortex.pipe.encode(TOLD.format(s=statement)), base_state)
        st = [s.detach() for s in st]
        pairs = [(q, greedy_answer(plastic, st, q, max_tokens)) for q in cues]
    finally:
        plastic.enabled = was
    return [(q, a) for q, a in pairs if a]


def encode_pair(pipe, cue, answer):
    """Token ids of the cued answer, and where the answer span starts; returns (ids, start, n_content)."""
    prefix = pipe.encode(f"User: {cue}\n\nAssistant:")
    content = pipe.encode(" " + answer)
    end = pipe.encode("\n\n")
    return prefix + content + end, len(prefix), len(content)


def practice_loss(plastic, base_state, cue, answer):
    """Error-weighted answer-span loss of the hippocampus-off cortex, and whether its greedy answer failed."""
    ids, start, n_content = encode_pair(plastic.cortex.pipe, cue, answer)
    logits, _ = plastic(ids, base_state)
    pred = logits[start - 1:len(ids) - 1]
    tgt = torch.tensor(ids[start:], device=logits.device)
    ce = F.cross_entropy(pred, tgt, reduction="none")
    with torch.no_grad():
        p = torch.exp(-ce)
        w = 1.0 + (1.0 - p)
        failed = bool((pred[:n_content].argmax(-1) != tgt[:n_content]).any())
    return (w * ce).sum() / w.sum(), failed


def sleep_retrieval(plastic, base_state, practice, generic, *, steps=240, lr=5e-4, kl_weight=1.0, prio=2.0, seed=0,
                    log=None):
    """practice: one list of (cue, teacher answer) per episode. Returns per-step (loss, keep, failed)."""
    practice = [p for p in practice if p]
    rng = random.Random(seed)
    torch.manual_seed(seed)
    c = plastic.cortex
    opt = torch.optim.Adam([p for p in plastic.parameters() if p.requires_grad], lr=lr)
    gen_ids = [c.pipe.encode(g) for g in generic]
    plastic.enabled = False
    with torch.no_grad():
        ref = [F.log_softmax(plastic(ids, base_state)[0], -1) for ids in gen_ids]
    plastic.enabled = True
    weights = [1.0] * len(practice)  # unknown before the first attempt
    out = []
    for step in range(steps):
        e = rng.choices(range(len(practice)), weights=weights)[0]
        cue, ans = rng.choice(practice[e])
        loss_r, failed = practice_loss(plastic, base_state, cue, ans)
        weights[e] = prio if failed else 1.0
        j = rng.randrange(len(gen_ids))
        g_logits, _ = plastic(gen_ids[j], base_state)
        keep = F.kl_div(F.log_softmax(g_logits, -1), ref[j], log_target=True, reduction="batchmean")
        loss = loss_r + kl_weight * keep
        opt.zero_grad()
        loss.backward()
        opt.step()
        out.append((float(loss_r), float(keep), failed))
        if log and step % 20 == 0:
            recent = out[-20:]
            log(f"retrieval step {step}: loss {float(loss_r):.3f} keep {float(keep):.4f} "
                f"failed {sum(r[2] for r in recent)}/{len(recent)}")
    return out
