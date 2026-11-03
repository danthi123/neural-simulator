# Capability: Sleep consolidation
# Record: brain/research/2026-11-03-confabulation-next-methods.md (M3)
"""M3: retrieval practice in sleep (hippocampal episode as teacher) vs ordinary restudy sleep. Pre-registration + result:
brain/experiments/results/2026-11-03-confab-m3-retrieval-sleep/README.md.

12 reachable facts (first 12 of ceiling_probe.POOL whose original question the cortex answers in context at this size).
Arms (same seed, same steps, same facts): no_sleep; sleep_true (brain/sleep.py, restudy of the told turn);
sleep_retrieval (brain/sleep_retrieval.py); sleep_retrieval_shuffled (CONTROL: retrieval practice on the told turns with
the answer words swapped between facts). Read-outs with the hippocampus off: recall (original questions), paraphrase,
false recall on the 4 never-told questions and on the 24 abstain-v2 fresh never-told questions, drift on ordinary text.

  python -m brain.experiments.retrieval_sleep_probe --ceiling <ceiling json> --seed 42 --out <json>
"""
import argparse
import json
import random
import time

import torch
import torch.nn.functional as F

from brain.brain import BIRTH
from brain.cortex import Cortex, DEFAULT_MODEL
from brain.experiments.abstain_v2 import FRESH_NEVER2
from brain.experiments.sleep_probe import GENERIC, NEVER_TOLD, ask
from brain.experiments.strict_scorer import correct_answer
from brain.experiments.use_strength_probe import reachable_facts
from brain.plastic_cortex import PlasticCortex
from brain.sleep import sleep
from brain.sleep_retrieval import TOLD, self_cues, sleep_retrieval, teacher_answers


def shuffle_statements(facts, seed):
    """Same sentences, each with another fact's answer word (same exposure, wrong content)."""
    rng = random.Random(seed)
    names = [f[2] for f in facts]
    perm = names[:]
    while any(x == y for x, y in zip(perm, names)):
        rng.shuffle(perm)
    out = []
    for (s, _, k, _), wrong in zip(facts, perm):
        i = s.lower().find(k)
        w = wrong.capitalize() if s[i].isupper() else wrong
        out.append(s[:i] + w + s[i + len(k):])
    return out


def evaluate(pc, state, facts, keys_all):
    rec_ans = [ask(pc, state, q) for _, q, _, _ in facts]
    par_ans = [ask(pc, state, p) for _, _, _, p in facts]
    never_ans = [ask(pc, state, q) for q in NEVER_TOLD]
    fresh_ans = [ask(pc, state, q) for q, _, _ in FRESH_NEVER2]
    intr = lambda a: any(k in a.lower() for k in keys_all)  # noqa: E731
    with torch.no_grad():
        drift = 0.0
        for g in GENERIC:
            ids = pc.cortex.pipe.encode(g)
            on = F.log_softmax(pc(ids, state)[0], -1)
            pc.enabled = False
            off = F.log_softmax(pc(ids, state)[0], -1)
            pc.enabled = True
            drift += float(F.kl_div(on, off, log_target=True, reduction="batchmean"))
    rec = [k in a.lower() for a, (_, _, k, _) in zip(rec_ans, facts)]
    rec_strict = [correct_answer(a, k, q, s) for a, (s, q, k, _) in zip(rec_ans, facts)]
    par = [k in a.lower() for a, (_, _, k, _) in zip(par_ans, facts)]
    return {"recall": sum(rec), "recall_strict": sum(rec_strict), "paraphrase": sum(par),
            "false_recall": sum(intr(a) for a in never_ans), "false_recall_fresh24": sum(intr(a) for a in fresh_ans),
            "generic_kl": drift / len(GENERIC), "recall_items": rec, "recall_strict_items": rec_strict,
            "paraphrase_items": par, "answers": {"recall": rec_ans, "paraphrase": par_ans, "never": never_ans,
                                                 "fresh_never": fresh_ans}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ceiling", required=True)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-cues", type=int, default=4)
    ap.add_argument("--prio", type=float, default=2.0)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--arms", default="no_sleep,sleep_true,sleep_retrieval,sleep_retrieval_shuffled")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    c = Cortex(model_path=a.model)
    _, base = c.read(BIRTH, None)
    base = [s.float() for s in base]
    facts, _ = reachable_facts(a.ceiling)
    keys_all = [f[2] for f in facts]
    shuffled = shuffle_statements(facts, a.seed)
    pc0 = PlasticCortex(c)

    def practice_for(statements):
        out = []
        for s in statements:
            cues = self_cues(c, BIRTH, s, n_cues=a.n_cues, seed=a.seed)
            out.append(teacher_answers(pc0, base, s, cues))
        return out

    prac_true = practice_for([f[0] for f in facts])
    prac_shuf = practice_for(shuffled) if "sleep_retrieval_shuffled" in a.arms else []
    exact = sum(q == f[1] for p, f in zip(prac_true, facts) for q, _ in p)
    teach_has_key = [any(f[2] in ans.lower() for _, ans in p) for p, f in zip(prac_true, facts)]
    print("practice", json.dumps(prac_true, indent=0), flush=True)
    result = {"model": a.model, "seed": a.seed, "steps": a.steps, "kl_weight": a.kl_weight, "lr": a.lr,
              "n_cues": a.n_cues, "prio": a.prio, "facts": keys_all, "shuffled_statements": shuffled,
              "practice_true": prac_true, "practice_shuffled": prac_shuf,
              "cues_equal_to_test_question": exact, "n_cues_total": sum(len(p) for p in prac_true),
              "facts_with_key_in_some_teacher_answer": sum(teach_has_key), "arms": {}}
    for arm in a.arms.split(","):
        torch.manual_seed(a.seed)
        pc = PlasticCortex(c)
        log = lambda m, arm=arm: print(arm, m, flush=True)  # noqa: E731
        extra = {}
        if arm == "sleep_true":
            sleep(pc, base, [(TOLD.format(s=f[0]), 1.0) for f in facts], GENERIC, steps=a.steps, lr=a.lr,
                  kl_weight=a.kl_weight, seed=a.seed, log=log)
        elif arm in ("sleep_retrieval", "sleep_retrieval_shuffled"):
            prac = prac_true if arm == "sleep_retrieval" else prac_shuf
            hist = sleep_retrieval(pc, base, prac, GENERIC, steps=a.steps, lr=a.lr, kl_weight=a.kl_weight,
                                   prio=a.prio, seed=a.seed, log=log)
            extra = {"failed_first40": sum(h[2] for h in hist[:40]), "failed_last40": sum(h[2] for h in hist[-40:])}
        r = evaluate(pc, base, facts, keys_all)
        r.update(extra)
        r["adapter_norm"] = pc.adapter_norm()
        result["arms"][arm] = r
        print(arm, {k: v for k, v in r.items() if not k.endswith("_items") and k != "answers"}, flush=True)
    result["elapsed_s"] = time.time() - t0
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
