# Capability: Language
# Record: brain/experiments/results/2026-11-16-own-gain-probe.md (the own-reply write is the copy amplifier; a flat
# gain also erases who is speaking)
"""Report-only diagnostic, no training: is the matrix-memory change written by a COPY reply aligned with the change the
user's turn just wrote, more than an ordinary reply's is? If so, suppressing only that component is a selective
version of the own-reply gain. Replays recorded conversations (echo_onset_probe output) through the pretrained cortex
(hippocampus and night off) on a copy of a slept home; per turn and layer: cos(reply delta, user-turn delta).
  python -m brain.experiments.copy_delta_probe --home <slept home> --probe-json <echo_onset_probe json> --out <json>
"""
import argparse
import json
import os
import shutil
import tempfile

import torch

from brain.brain import Brain
from brain.cortex import Cortex, resolve_model
from brain.experiments.echo_onset_probe import _norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--home", required=True)
    ap.add_argument("--probe-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = Cortex(model_path=resolve_model(a.model), device=a.device)
    convs = json.load(open(a.probe_json))
    out = []
    for conv in convs:
        tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
        cp = os.path.join(tmp, "home")
        shutil.copytree(a.home, cp)
        b = Brain(cp, cortex=c, sleep_on_restart=False)
        b.set_lesions(hippocampus_off=True, consolidated_off=True)
        st = b.state
        for i, t in enumerate(conv["turns"]):
            _, su = b.cortex.read(f"User: {t['you']}\n\nAssistant:", st)
            _, sp = b.cortex.read(f" {t['brain']}\n\n", su)
            cos = []
            for j in range(1, len(sp), 3):
                du, dr = (su[j] - st[j]).flatten().float(), (sp[j] - su[j]).flatten().float()
                cos.append(round(float(torch.nn.functional.cosine_similarity(du, dr, dim=0)), 4))
            out.append({"script_seed": conv["script_seed"], "sample": conv["sample"], "turn": i,
                        "echo": _norm(t["brain"]) == _norm(t["you"]), "cos": cos})
            st = sp
        shutil.rmtree(tmp, ignore_errors=True)
        json.dump(out, open(a.out, "w"), indent=1)
    for e in (True, False):
        r = [x for x in out if x["echo"] == e and x["turn"] > 0]
        if r:
            m = [sum(x["cos"][k] for x in r) / len(r) for k in range(len(r[0]["cos"]))]
            print("echo" if e else "other", len(r), "mean cos by layer", [round(v, 3) for v in m])


if __name__ == "__main__":
    main()
