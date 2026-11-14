# Capability: Metacognition
# Record: brain/experiments/results/2026-11-05-hippo-routing-1.5b-live.md (reinstating the RIGHT episode into a slept
#         cortex costs answers); brain/experiments/results/2026-11-03-false-recall-feeling-of-knowing/README.md (0.1B)
"""Does the slept cortex know when it knows? Read-outs for gating hippocampal reinstatement on cortical confidence.

Pre-registration: brain/research/2026-11-13-confidence-gate-1.5b.md (G0-G4, written before any run).

Day 1 and the night are exactly live_brain_probe's (12 FACTS told to a live Brain, restart = sleep). Then, per question
(12 told, 12 paraphrase, 12 never-told = sleep_probe.NEVER_TOLD + confidence_probe.NEVER_TOLD_EXTRA), two fresh Brains
on the slept home, seeded per question as live_brain_probe.ask_arm:
  c  hippocampus off: the slept cortex alone (live Brain.hear)
  f  hippocampus on, key S_all with the 1.5B size bar (the 11-05 fix arm)
For c's own reply, the cortex's confidence read-outs (no host decision; nothing here changes a reply):
  mean_logp / first_logp / min_logp   the reply's tokens teacher-forced through the live cortex (consolidated on)
  mean_logp_off, shift                the same reply with the consolidated synapses off; shift = on - off
                                      ("did my slept synapses produce this answer")
  cue                                 LiveCortex.cue_response of the question (cortical cue familiarity, 0.1B AUC 0.79)
The gate is evaluated offline by brain/experiments/confidence_gate_table.py: answer with c's reply when the signal is at
or above a bar fitted on the other five seeds, else with f's reply. That equals a Brain that skips reinstatement when it
knows, because f's reply when nothing fires is c's reply (checked per row: G0).

  python -m brain.experiments.confidence_gate_probe --home <dir> --model ../models/rwkv7-g1l-1.5b-...pth \
      --seed 42 --steps 480 --lr 5e-4 --kl-weight 3 --device cuda --out <json>
"""
import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from brain.brain import Brain
from brain.cortex import DEFAULT_MODEL, Cortex, resolve_model
from brain.experiments import live_brain_probe as lbp
from brain.experiments.confidence_probe import NEVER_TOLD_EXTRA
from brain.experiments.facts import FACTS, PARA
from brain.experiments.sleep_probe import NEVER_TOLD
from brain.experiments.strict_scorer import correct_answer, has_key

FIX_KEY, FIX_BAR = "S_all", "size"


def reply_logps(b, q, reply, consolidated):
    """Per-token log-probs of `reply` after the question, teacher-forced through the brain's live cortex."""
    was = b.cortex.consolidated_on
    b.cortex.consolidated_on = consolidated
    try:
        with torch.no_grad():
            logits, st = b.cortex.read(f"User: {q}\n\nAssistant:", b.state)
            st = copy.deepcopy(st)
            out = []
            for tok in b.cortex.pipe.encode(" " + reply):
                out.append(float(F.log_softmax(logits.float(), -1)[tok]))
                logits, st = b.cortex.forward([tok], st)
    finally:
        b.cortex.consolidated_on = was
    return out


def fresh(home, cortex, seed, steps, hip_off, key=None, key_bar="default", gain="default"):
    np.random.seed(seed)
    torch.manual_seed(seed)
    b = Brain(home, cortex=cortex, sleep_on_restart=False, sleep_steps=steps, sleep_lr=lbp.SLEEP_LR,
              sleep_kl_weight=lbp.SLEEP_KL, adapter_rank=8, sleep_seed=seed, key=key, key_bar=key_bar, gain=gain)
    b.set_lesions(hippocampus_off=hip_off)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
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
    home = f"{a.home}/slept"
    day1 = lbp.tell_day(home, c, [s for s, _, _ in FACTS], a.seed, a.steps)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    b = Brain(home, cortex=c, sleep_steps=a.steps, sleep_lr=lbp.SLEEP_LR, sleep_kl_weight=lbp.SLEEP_KL, adapter_rank=8,
              sleep_seed=a.seed)   # the restart = the night
    assert b.meta["sleeps"] == 1, b.meta
    adapter_norm = b.cortex.plastic.adapter_norm()
    b.release()
    del b
    keys = [k for _, _, k in FACTS]
    qs = [(q, k, "told", s) for s, q, k in FACTS] + [(p, f[2], "para", f[0]) for p, f in zip(PARA, FACTS)] + \
         [(q, None, "never", None) for q in NEVER_TOLD + NEVER_TOLD_EXTRA]
    if a.max_q:
        qs = [x for kind in ("told", "para", "never") for x in [y for y in qs if y[2] == kind][:a.max_q]]
    rows = []
    for q, key, kind, told in qs:
        row = {"q": q, "kind": kind, "key": key}
        for arm, hip_off, hk, bar in (("c", True, None, "default"), ("f", False, FIX_KEY, a.fix_bar)):
            b = fresh(home, c, a.seed, a.steps, hip_off, hk, bar)
            if arm == "c":
                base = [s.float() for s in b.base]
                cue = b.cortex.cue_response(f"User: {q}\n\n", base)
                state0 = [x.clone() for x in b.state]
            reply, info = b.hear(q)
            if arm == "c":
                b.state = state0   # read-outs from the state the reply was spoken from
                on = reply_logps(b, q, reply, True) if reply else []
                off = reply_logps(b, q, reply, False) if reply else []
                n = max(1, len(on))
                row.update(cue=cue, mean_logp=sum(on) / n, first_logp=on[0] if on else 0.0,
                           min_logp=min(on) if on else 0.0, mean_logp_off=sum(off) / n,
                           shift=(sum(on) - sum(off)) / n, n_tok=len(on))
            if key is None:
                ok = any(has_key(reply, k) for k in keys)   # false recall
            else:
                ok = bool(correct_answer(reply, key, q, told))
            row[arm] = {"reply": reply, "recalled": info["recalled"], "ok": ok}
            b.release()
            del b
        rows.append(row)
        print(a.seed, kind, row["c"]["ok"], row["f"]["ok"], bool(row["f"]["recalled"]), f"{row['shift']:.3f}",
              f"{row['cue']:.3f}", repr(row["c"]["reply"][:60]), flush=True)
    res = {"model": a.model, "seed": a.seed, "steps": a.steps, "lr": a.lr, "kl_weight": a.kl_weight, "device": a.device,
           "max_q": a.max_q, "fix_key": FIX_KEY, "fix_bar": a.fix_bar, "adapter_norm": adapter_norm, "day1": day1,
           "rows": rows, "elapsed_s": round(time.time() - t0, 1)}
    if a.device == "cuda":
        res["cuda_peak_bytes"] = torch.cuda.max_memory_allocated()
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
