# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-02-sleep-0.1b-six-seeds.md
"""Rescore a weekly transcript's session-2 recall answers with the strict correct-perspective scorer (no reruns).

python -m brain.eval.rescore_transcript TRANSCRIPT.md > OUT.md
Pre-registered rule: the transcript's own mark (key substring + not degenerate) is the 'lenient' column; 'strict' is
brain.experiments.strict_scorer.correct_answer(reply, key, question, told). Told sentences are the statements the
weekly test makes in session 1 (None for the brain's own facts, name/Dan).
"""
import re
import sys

from brain.experiments.strict_scorer import correct_answer

Q = {  # question -> (key, told sentence or None)
    "Hi again! Do you remember my name?": ("dan", None),
    "What's your name?": ("nova", None),
    "What is my cat called?": ("pixel", "My cat is called Pixel."),
    "What is my sister's name?": ("marguerite", "My sister's name is Marguerite."),
    "What car do I drive?": ("volvo", "I drive a red Volvo."),
    "What do I do for work?": ("admin", "Nice to meet you. I work as a system administrator, mostly Linux servers."),
    "Where does my uncle live?": ("denver", "My uncle lives in Denver."),
    "Where was I born?": ("lisbon", "I was born in Lisbon."),
    "What instrument do I play?": ("saxophone", "I play the saxophone."),
    "What instrument does my mother play?": ("cello", "My mother plays the cello."),
    "What do I do in my free time?": ("chess", "In my free time I play chess."),
    "What is my favourite food?": ("ramen", "My favourite food is ramen."),
}


def parse(text):
    arms, arm, s2 = {}, None, False
    for line in text.splitlines():
        if line.startswith("## "):
            arm, s2 = line[3:].strip(), False
            arms[arm] = []
        elif line.startswith("### session 2"):
            s2 = True
        elif s2 and line.startswith("- **you:** "):
            q = re.sub(r" _\(this week's draw\)_$", "", line[len("- **you:** "):])
            arms[arm].append([q])
        elif s2 and line.strip().startswith("**brain:**") and arms[arm]:
            r = line.strip()[len("**brain:**"):].strip()
            r = re.sub(r" _\(recalled:.*$", "", r)
            m = re.search(r" (✅|❌)$", r)
            mark = None
            if m:
                mark, r = m.group(1) == "✅", r[:m.start()]
            arms[arm][-1] += [r, mark]
    return arms


def main(path):
    arms = parse(open(path).read())
    out = ["# Weekly 2026-11-03 (W45) rescored with brain.experiments.strict_scorer.correct_answer", "",
           "Source: research/findings/raw/_weekly_eval/2026-11-03-weekly-transcript.md (a5f155434). No reruns.", ""]
    tot = {}
    rows = []
    for arm, items in arms.items():
        if not items:
            continue
        n = lo = st = 0
        for q, reply, mark in (i for i in items if len(i) == 3):
            if q not in Q:
                continue
            key, told = Q[q]
            s = correct_answer(reply, key, q, told)
            n, lo, st = n + 1, lo + bool(mark), st + s
            rows.append((arm, q, reply, mark, s))
        tot[arm] = (lo, st, n)
    out.append("| arm | transcript (lenient) | strict |\n|---|---|---|")
    out += [f"| {a} | {l} / {n} | {s} / {n} |" for a, (l, s, n) in tot.items()]
    out += ["", "| arm | question | reply | transcript | strict |", "|---|---|---|---|---|"]
    out += [f"| {a} | {q} | {r} | {'Y' if m else 'N'} | {'Y' if s else 'N'} |" for a, q, r, m, s in rows]
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
