# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-15-open-conv-baseline-1.5b (day 2 collapses into one repeated sentence on 4/6 seeds)
"""Control: replay open_conv's day 2 on a COPY of a slept home (the hippocampus-off home holds exactly the day-1 store
and the night, since a lesioned day stores nothing), in three arms:
  intact          hippocampus on, the night's synapses on (should reproduce the baseline intact arm)
  night off       hippocampus on, the night's synapses off (consolidated_off): memory without what sleep wrote
  both off        hippocampus off, the night's synapses off: the pretrained cortex alone
  python -m brain.eval.open_conv_day2 --model 1.5b --seed <s> --home <slept home> --out <md>
"""
import argparse
import collections
import json
import os
import shutil
import tempfile

import numpy as np
import torch

from brain.brain import Brain
from brain.cortex import Cortex, resolve_model
from brain.eval.open_conv import build, live, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = Cortex(model_path=resolve_model(a.model), device=a.device)
    told, _, day2 = build(a.seed)
    lines, rows, stats = [f"# Open conversation day 2 control, seed {a.seed}"], [], {}
    for arm, hip_off, night_off in (("intact", False, False), ("night off", False, True), ("both off", True, True)):
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
        cp = os.path.join(tmp, "home")
        shutil.copytree(a.home, cp)
        b = Brain(cp, cortex=c, sleep_on_restart=False, sleep_seed=a.seed)
        assert b.meta["sleeps"] >= 1
        b.set_lesions(hippocampus_off=hip_off, consolidated_off=night_off)
        lines.append(f"\n## day 2, {arm}\n")
        live(b, day2, told, lines, rows, arm, 2)
        R = [r for r in rows if r["arm"] == arm]
        top = collections.Counter(r["brain"].strip() for r in R).most_common(1)[0]
        stats[arm] = dict(summary(rows, arm), top_repeat=top[1], top_reply=top[0][:60],
                          echo=sum(r["brain"].strip() == r["you"].strip() for r in R), n=len(R))
        shutil.rmtree(tmp, ignore_errors=True)
    lines.append("\n## summary (day 2 only)\n")
    lines += [f"- {k}: {v}" for k, v in stats.items()]
    open(a.out, "w").write("\n".join(lines) + "\n")
    json.dump({"seed": a.seed, "stats": stats, "rows": rows}, open(a.out + ".rows.json", "w"), indent=1)
    print("\n".join(lines[-4:]))


if __name__ == "__main__":
    main()
