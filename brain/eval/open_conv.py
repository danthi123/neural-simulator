# Capability: Episodic recall
# Record: brain/research/2026-11-15-open-conversation-eval.md (owner 2026-11-15: are we building to pass our own tests?)
"""Open-conversation eval: a different two-day conversation per seed, scored as a whole conversation.

The fixed weekly (brain/eval/weekly.py) ticks 8 scripted questions. This eval samples each conversation from a held-out
bank (open_conv_bank.py; never used to fit anything):
  day 1: greeting, 6 facts told among 4 small-talk turns, one direct question about a fact told that day;
  night (the restart sleeps once; the slept home is copied for the hippocampus-off arm);
  day 2: greeting, then shuffled: 3 direct + 3 reworded questions about the told facts, 3 questions about facts never
         told (other bank facts), and 6 turns that should not recall anything (chat, feeling, story, opinion,
         knowledge, chat).
Per turn: recall turns are scored by strict_scorer.correct_answer (any key word); never-told turns are FALSE when the
reply names any told key; every other turn (greetings included) is an INTRUSION when the reply names a told key that
the turn does not, and a NEEDLESS RECALL when the hippocampus returned an episode. Arms: intact, hippocampus off.
No single score: the numbers are reported side by side, and the transcript is written in full to be read.

  python -m brain.eval.open_conv --model 1.5b --device cuda --seed 42 --out <md> [--homes <dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import socket

import numpy as np
import torch

from brain.brain import Brain
from brain.cortex import Cortex, resolve_model
from brain.eval.open_conv_bank import FACTS, GREET_1, GREET_2, OPEN_TURNS
from brain.experiments.strict_scorer import correct_answer


def build(seed: int):
    rng = random.Random(seed)
    facts = rng.sample(FACTS, 9)
    told, untold = facts[:6], facts[6:]
    chat = [t for t in OPEN_TURNS if t[0] == "chat"]
    rng.shuffle(chat)
    day1 = [("greet", GREET_1[rng.randrange(len(GREET_1))], None)]
    small = chat[:4]
    for i, f in enumerate(told):
        day1.append(("tell", f[0], f))
        if i % 2 == 1 and small:
            k, t = small.pop(0)
            day1.append((k, t, None))
    day1 += [(k, t, None) for k, t in small]
    day1.append(("direct", told[0][1], told[0]))
    open2 = [rng.choice([t for t in OPEN_TURNS if t[0] == k]) for k in ("feeling", "story", "opinion", "knowledge")]
    open2 += chat[4:6]
    day2 = ([("direct", f[1], f) for f in told[:3]] + [("reworded", f[2], f) for f in told[3:]]
            + [("never", f[1], f) for f in untold] + [(k, t, None) for k, t in open2])
    rng.shuffle(day2)
    day2 = [("greet", GREET_2[rng.randrange(len(GREET_2))], None)] + day2
    return told, day1, day2


def _has(text: str, keys) -> bool:
    return any(re.search(rf"\b{re.escape(k)}\b", text, re.I) for k in keys)


# Teacher-side scorer (not a brain rule): a reply to a never-told question is honest if it says it does not know or asks
# back; anything else states something about the user that was never told (added 2026-11-17: FALSE alone counted only
# told-key intrusions and missed new inventions such as "Your goldfish is called Goldie.").
_DONT_KNOW = re.compile(r"\b(not sure|don'?t know|do not know|no idea|haven'?t (told|mentioned|said)|didn'?t (tell|mention|say)|"
                        r"not (told|mentioned)|can'?t (recall|remember|tell)|cannot (recall|remember|tell)|don'?t (have|recall|remember)|"
                        r"do not (have|recall|remember)|not aware|no information|unknown to me)\b", re.I)


def invented(reply: str) -> bool:
    r = reply.strip()
    return bool(r) and not _DONT_KNOW.search(r) and not r.endswith("?")


def mark(kind, turn, fact, reply, recalled, told):
    told_keys = [k for f in told for k in f[3]]
    if kind in ("direct", "reworded"):
        return {"ok": any(correct_answer(reply, k, turn, fact[0]) for k in fact[3])}
    if kind == "tell":
        others = [k for f in told if f is not fact for k in f[3]]
        return {"intrusion": _has(reply, [k for k in others if not _has(turn, [k])]), "needless": False}
    if kind == "never":
        return {"false": _has(reply, told_keys), "invented": invented(reply)}
    return {"intrusion": _has(reply, [k for k in told_keys if not _has(turn, [k])]), "needless": bool(recalled)}


def live(b: Brain, turns, told, lines, rows, arm, day):
    for kind, turn, fact in turns:
        reply, info = b.hear(turn)
        m = mark(kind, turn, fact, reply, info.get("recalled"), told)
        rows.append({"arm": arm, "day": day, "kind": kind, "you": turn, "brain": reply, "recalled": info.get("recalled"), **({"withheld": info["withheld"]} if "withheld" in info else {}), **m})
        tag = ("✅" if m["ok"] else "❌") if "ok" in m else ("FALSE" if m.get("false") else "") + \
              (" INTRUSION" if m.get("intrusion") else "") + (" needless-recall" if m.get("needless") else "")
        rec = f" _(recalled: {info['recalled']})_" if info.get("recalled") else ""
        lines.append(f"- [{kind}] **you:** {turn}\n  **brain:** {reply} {tag.strip()}{rec}")
    b.save()


def summary(rows, arm):
    r = [x for x in rows if x["arm"] == arm]
    def n(pred): return sum(1 for x in r if pred(x))
    opens = [x for x in r if "intrusion" in x and x["kind"] != "tell"]
    return {"recall_ok": n(lambda x: x.get("ok")), "recall_n": n(lambda x: "ok" in x),
            "direct_ok": n(lambda x: x["kind"] == "direct" and x["day"] == 2 and x["ok"]),
            "reworded_ok": n(lambda x: x["kind"] == "reworded" and x["ok"]),
            "false": n(lambda x: x.get("false")), "never_n": n(lambda x: x["kind"] == "never"),
            "invented": n(lambda x: x["kind"] == "never" and x.get("invented", invented(x["brain"]))),
            "intrusions": sum(x["intrusion"] for x in opens), "open_n": len(opens),
            "needless_recall": sum(x["needless"] for x in opens),
            "tell_intrusions": n(lambda x: x["kind"] == "tell" and x["intrusion"])}


def main(argv=None, cortex: Cortex | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--homes", default=None, help="directory for the brain homes (default: next to --out)")
    ap.add_argument("--out", required=True, help="transcript .md; the JSON rows go to <out>.rows.json")
    ap.add_argument("--sleep-rehearse", action="store_true", help="Brain(sleep_rehearse=True) (switch, default OFF)")
    ap.add_argument("--rehearse-reply", action="store_true", help="Brain(sleep_rehearse='reply'): KL on reply tokens only")
    ap.add_argument("--habituation", type=int, default=0, help="Brain(habituation=N) (switch, default OFF = 0)")
    ap.add_argument("--reply-monitor", type=int, default=0, help="Brain(reply_monitor=N) (switch, default OFF = 0)")
    a = ap.parse_args(argv)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    cortex = cortex or Cortex(model_path=resolve_model(a.model), device=a.device)
    told, day1, day2 = build(a.seed)
    kw = {"sleep_seed": a.seed, **({"sleep_rehearse": True} if a.sleep_rehearse else {}),
          **({"sleep_rehearse": "reply"} if a.rehearse_reply else {}),
          **({"habituation": a.habituation} if a.habituation else {}),
          **({"reply_monitor": a.reply_monitor} if a.reply_monitor else {})}
    root = a.homes or (os.path.splitext(os.path.abspath(a.out))[0] + "_homes")
    intact, off = os.path.join(root, "intact"), os.path.join(root, "hippocampus_off")
    for h in (intact, off):
        shutil.rmtree(h, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    lines = [f"# Open conversation, seed {a.seed}",
             f"\n_cortex: {getattr(cortex, 'model_name', None)}; device: {a.device}; host: {socket.gethostname()};"
             f" Brain defaults{' + sleep_rehearse' if a.sleep_rehearse else ''}{' + sleep_rehearse=reply' if a.rehearse_reply else ''}{f' + habituation {a.habituation}' if a.habituation else ''}{f' + reply_monitor {a.reply_monitor}' if a.reply_monitor else ''}; facts told: {', '.join(f[3][0] for f in told)}_", "\n## day 1\n"]
    rows = []
    live(Brain(intact, cortex=cortex, **kw), day1, told, lines, rows, "both", 1)
    Brain(intact, cortex=cortex, **kw).save()     # restart = the one night
    shutil.copytree(intact, off)
    for arm, home, lesion in (("intact", intact, False), ("hippocampus off", off, True)):
        b = Brain(home, cortex=cortex, **kw)
        assert b.meta["sleeps"] >= 1 and not b.pending()
        b.set_lesions(hippocampus_off=lesion)
        lines.append(f"\n## day 2, {arm}\n")
        live(b, day2, told, lines, rows, arm, 2)
    for r in rows:
        if r["arm"] == "both":
            r["arm"] = "day1"
    s = {arm: summary([dict(x, arm=arm) if x["arm"] == "day1" else x for x in rows], arm)
         for arm in ("intact", "hippocampus off")}
    lines.append("\n## summary (day 1 counts in both arms)\n")
    for arm, v in s.items():
        lines.append(f"- {arm}: recall {v['recall_ok']}/{v['recall_n']} (day-2 direct {v['direct_ok']}/3, reworded "
                     f"{v['reworded_ok']}/3); never-told answered with a told fact {v['false']}/{v['never_n']}; "
                     f"intrusions {v['intrusions']}/{v['open_n']} open turns; needless recalls {v['needless_recall']}; "
                     f"intrusions while being told {v['tell_intrusions']}")
    with open(a.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(a.out + ".rows.json", "w") as fh:
        json.dump({"seed": a.seed, "host": socket.gethostname(), "model": getattr(cortex, "model_name", None),
                   "told": [f[0] for f in told], "summary": s, "rows": rows}, fh, indent=1)
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
