# Capability: Metacognition
# Record: brain/experiments/results/2026-11-13-confidence-gate-1.5b.md (gate no-regret but == hippocampus-off when
#         every fact was slept on); brain/experiments/results/2026-11-05-hippo-routing-1.5b-live.md
"""Mixed day: half the facts slept on, half told after the restart (only the hippocampus holds those).

Pre-registration: brain/research/2026-11-13-mixed-day-gate-1.5b.md (M0-M5, written before any run).

Day 1: six FACTS told to a live Brain (live_brain_probe.tell_day). Restart = the night (sleeps once). Day 2, same
process after the restart: the other six FACTS told, saved, no second night. Which six are slept is `--slept even|odd`
(FACTS indices), counterbalanced across seeds in the job file. Then per question, exactly as confidence_gate_probe:
two fresh Brains on that home (sleep_on_restart=False, so day 2 stays unconsolidated), c = hippocampus off and
f = hippocampus on (S_all key, 1.5B size bar), plus c's confidence read-outs. Rows carry `set` = slept | new.

  python -m brain.experiments.mixed_day_probe --home <dir> --model ../models/rwkv7-g1l-1.5b-...pth --seed 42 \
      --slept even --steps 480 --lr 5e-4 --kl-weight 3 --device cuda --out <json>
"""
import argparse
import json
import time

import numpy as np
import torch

from brain.brain import Brain
from brain.cortex import DEFAULT_MODEL, Cortex, resolve_model
from brain.experiments import live_brain_probe as lbp
from brain.experiments.confidence_gate_probe import FIX_BAR, FIX_KEY, fresh, reply_logps
from brain.experiments.confidence_probe import NEVER_TOLD_EXTRA
from brain.experiments.facts import FACTS, PARA
from brain.experiments.sleep_probe import NEVER_TOLD
from brain.experiments.strict_scorer import correct_answer, has_key


def split(slept):
    idx = [i for i in range(len(FACTS)) if i % 2 == (0 if slept == "even" else 1)]
    return idx, [i for i in range(len(FACTS)) if i not in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slept", choices=("even", "odd"), default="even")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-q", type=int, default=0, help="smoke only: first N questions per kind (0 = all)")
    ap.add_argument("--fix-bar", default=FIX_BAR, help="f's key_bar: size (the 1.5B run) | default (0.1B smoke only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lbp.SLEEP_LR, lbp.SLEEP_KL = a.lr, a.kl_weight
    t0 = time.time()
    c = Cortex(model_path=resolve_model(a.model), device=a.device)
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    home = f"{a.home}/mixed"
    slept, new = split(a.slept)
    day1 = lbp.tell_day(home, c, [FACTS[i][0] for i in slept], a.seed, a.steps)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    b = Brain(home, cortex=c, sleep_steps=a.steps, sleep_lr=lbp.SLEEP_LR, sleep_kl_weight=lbp.SLEEP_KL, adapter_rank=8,
              sleep_seed=a.seed)   # the restart = the night
    assert b.meta["sleeps"] == 1, b.meta
    adapter_norm = b.cortex.plastic.adapter_norm()
    day2 = [{"you": FACTS[i][0], "brain": b.hear(FACTS[i][0])[0]} for i in new]
    b.save()   # not closed: day 2 must stay unconsolidated
    n_pending = len(b.pending())   # day-2 episodes held only by the hippocampus
    b.release()
    del b
    keys = [k for _, _, k in FACTS]
    sets = {i: ("slept" if i in slept else "new") for i in range(len(FACTS))}
    qs = [(q, k, "told", s, sets[i]) for i, (s, q, k) in enumerate(FACTS)] + \
         [(p, f[2], "para", f[0], sets[i]) for i, (p, f) in enumerate(zip(PARA, FACTS))] + \
         [(q, None, "never", None, None) for q in NEVER_TOLD + NEVER_TOLD_EXTRA]
    if a.max_q:
        qs = [x for kind in ("told", "para", "never") for x in [y for y in qs if y[2] == kind][:a.max_q]]
    rows = []
    for q, key, kind, told, st in qs:
        row = {"q": q, "kind": kind, "key": key, "set": st}
        for arm, hip_off, hk, bar in (("c", True, None, "default"), ("f", False, FIX_KEY, a.fix_bar)):
            b = fresh(home, c, a.seed, a.steps, hip_off, hk, bar)
            assert b.meta["sleeps"] == 1, b.meta   # no second night
            if arm == "c":
                base = [s.float() for s in b.base]
                cue = b.cortex.cue_response(f"User: {q}\n\n", base)
                state0 = [x.clone() for x in b.state]
            reply, info = b.hear(q)
            if arm == "c":
                b.state = state0
                on = reply_logps(b, q, reply, True) if reply else []
                off = reply_logps(b, q, reply, False) if reply else []
                n = max(1, len(on))
                row.update(cue=cue, mean_logp=sum(on) / n, first_logp=on[0] if on else 0.0,
                           min_logp=min(on) if on else 0.0, mean_logp_off=sum(off) / n,
                           shift=(sum(on) - sum(off)) / n, n_tok=len(on))
            if key is None:
                ok = any(has_key(reply, k) for k in keys)
            else:
                ok = bool(correct_answer(reply, key, q, told))
            row[arm] = {"reply": reply, "recalled": info["recalled"], "ok": ok}
            b.release()
            del b
        rows.append(row)
        print(a.seed, kind, st, row["c"]["ok"], row["f"]["ok"], bool(row["f"]["recalled"]),
              f"{row['mean_logp']:.3f}", repr(row["c"]["reply"][:60]), flush=True)
    res = {"model": a.model, "seed": a.seed, "slept": a.slept, "slept_idx": slept, "new_idx": new, "steps": a.steps,
           "lr": a.lr, "kl_weight": a.kl_weight, "device": a.device, "max_q": a.max_q, "fix_key": FIX_KEY,
           "fix_bar": a.fix_bar, "adapter_norm": adapter_norm, "day1": day1, "day2": day2, "n_pending": n_pending,
           "rows": rows, "elapsed_s": round(time.time() - t0, 1)}
    if a.device == "cuda":
        res["cuda_peak_bytes"] = torch.cuda.max_memory_allocated()
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
