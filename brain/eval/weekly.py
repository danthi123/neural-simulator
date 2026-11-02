"""Weekly conversation test: teach, chat, restart, recall; intact vs hippocampus lesion. Writes a markdown transcript.

python -m brain.eval.weekly [--out brain/eval/transcripts/<date>.md] [--device cpu]
"""
from __future__ import annotations

import argparse
import datetime
import os
import shutil

from brain.brain import Brain
from brain.cortex import Cortex

SESSION_1 = [
    "Hi! What's your name?",
    "Nice to meet you. I work as a system administrator, mostly Linux servers.",
    "My cat is called Pixel.",
    "It rained all day here, so I stayed inside and fixed a broken backup job.",
    "My sister's name is Marguerite.",
    "I drive a red Volvo.",
    "What is my cat called?",
    "Tell me something you find interesting.",
]
SESSION_2 = [  # after a restart
    "Hi again! Do you remember my name?",
    "What's your name?",
    "What is my cat called?",
    "What is my sister's name?",
    "What car do I drive?",
    "What do I do for work?",
    "Tell me a short story about a fox who gets lost.",
    "How are you feeling today?",
]
KEYS = {"Do you remember my name?": "dan", "What's your name?": "nova", "What is my cat called?": "pixel",
        "What is my sister's name?": "marguerite", "What car do I drive?": "volvo", "What do I do for work?": "admin"}


def run(home: str, cortex: Cortex, lesion: bool, lines: list):
    for s, turns in (("session 1", SESSION_1), ("session 2 (after restart)", SESSION_2)):
        b = Brain(home, cortex=cortex)          # a new Brain object = a restart; memory comes from disk
        b.hippocampus.enabled = not lesion
        lines.append(f"\n### {s}\n")
        for t in turns:
            reply, info = b.hear(t)
            key = next((v for k, v in KEYS.items() if k in t), None)
            mark = "" if key is None else (" ✅" if key in reply.lower() else " ❌")
            rec = f" _(recalled: {info['recalled']})_" if info["recalled"] else ""
            lines.append(f"- **you:** {t}\n  **brain:** {reply}{mark}{rec}")
        b.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"brain/eval/transcripts/{datetime.date.today()}.md")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    cortex = Cortex(device=a.device)
    lines = [f"# Weekly conversation test, {datetime.date.today()}"]
    for name, lesion in (("intact", False), ("lesion: hippocampus off", True)):
        home = f"brain_data/_weekly_{'lesion' if lesion else 'intact'}"
        shutil.rmtree(home, ignore_errors=True)
        lines.append(f"\n## {name}")
        run(home, cortex, lesion, lines)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    hits = {k: sum(1 for l in lines if "✅" in l) for k in ("all",)}
    print("\n".join(lines))
    print(f"\nwrote {a.out}; ✅ total {hits['all']}")


if __name__ == "__main__":
    main()
