"""Weekly conversation test: teach, chat, restart, recall. Three arms, one markdown transcript.

Arms: intact; hippocampus off (lesion); sleep + hippocampus off (session 1 with the hippocampus, then sleep replays the
stored episodes into the cortex's adapters, then session 2 asks with the hippocampus OFF through the adapted cortex).

Scored questions: 6 FIXED (comparable week to week) + 6 drawn each ISO week from POOL (seeded by the week, so the
score cannot improve by memorising fixed questions). A reply counts only via counts_as_recall (key word present and
the reply is not degenerate).

python -m brain.eval.weekly [--out brain/eval/transcripts/<date>.md] [--device cpu] [--week 2026-W45] [--sleep-steps 200]
"""
from __future__ import annotations

import argparse
import datetime
import os
import random
import re
import shutil

import torch

from brain.brain import BIRTH, Brain
from brain.cortex import Cortex
from brain.plastic_cortex import PlasticCortex
from brain.sleep import sleep

# (statement told in session 1 or None, question asked in session 2, key word). Name facts come from the seeded self.
FIXED = [
    (None, "Hi again! Do you remember my name?", "dan"),
    (None, "What's your name?", "nova"),
    ("My cat is called Pixel.", "What is my cat called?", "pixel"),
    ("My sister's name is Marguerite.", "What is my sister's name?", "marguerite"),
    ("I drive a red Volvo.", "What car do I drive?", "volvo"),
    ("Nice to meet you. I work as a system administrator, mostly Linux servers.", "What do I do for work?", "admin"),
]
N_ROTATING = 6
# Weekly draw pool: (statement, question, key). Statements are tellings (not requests) so the hippocampus stores them.
POOL = [
    ("My dog is called Biscuit.", "What is my dog called?", "biscuit"),
    ("My brother's name is Theodore.", "What is my brother's name?", "theodore"),
    ("My favourite colour is teal.", "What is my favourite colour?", "teal"),
    ("I was born in Lisbon.", "Where was I born?", "lisbon"),
    ("My mother plays the cello.", "What instrument does my mother play?", "cello"),
    ("In my free time I play chess.", "What do I do in my free time?", "chess"),
    ("I take my coffee with oat milk.", "How do I take my coffee?", "oat"),
    ("My uncle lives in Denver.", "Where does my uncle live?", "denver"),
    ("My daughter's name is Ingrid.", "What is my daughter's name?", "ingrid"),
    ("My favourite food is ramen.", "What is my favourite food?", "ramen"),
    ("I am learning Japanese.", "Which language am I learning?", "japanese"),
    ("I ride a yellow Brompton.", "What bike do I ride?", "brompton"),
    ("My horse is called Clover.", "What is my horse called?", "clover"),
    ("My neighbour is called Gustav.", "What is my neighbour called?", "gustav"),
    ("I grow tomatoes on my balcony.", "What do I grow on my balcony?", "tomatoes"),
    ("My favourite band is Radiohead.", "What is my favourite band?", "radiohead"),
    ("I once travelled to Iceland.", "Which country did I travel to?", "iceland"),
    ("I play the saxophone.", "What instrument do I play?", "saxophone"),
    ("My laptop is a ThinkPad.", "What laptop do I use?", "thinkpad"),
    ("My best friend is called Oskar.", "What is my best friend called?", "oskar"),
]
CHITCHAT_1 = {  # unscored session-1 turns, by position (see _session1_turns)
    "greet": "Hi! What's your name?",
    "rain": "It rained all day here, so I stayed inside and fixed a broken backup job.",
    "ask_cat": "What is my cat called?",
    "interesting": "Tell me something you find interesting.",
}
CHITCHAT_2 = ["Tell me a short story about a fox who gets lost.", "How are you feeling today?"]

# Scaffold (SCAFFOLDS.md): ordinary chat the sleep step interleaves with the replayed episodes. No told facts here.
GENERIC_CHAT = [
    "User: What's a good way to spend a rainy afternoon?\n\nAssistant: Reading a book with tea is nice.\n\n",
    "User: Can you count to five?\n\nAssistant: One, two, three, four, five.\n\n",
    "User: Hello, how are you today?\n\nAssistant: I'm doing well, thank you for asking.\n\n",
    "User: Tell me something interesting.\n\nAssistant: Octopuses have three hearts and blue blood.\n\n",
]


# ---------------------------------------------------------------- scoring
def _tokens(text: str) -> list:
    return re.findall(r"[a-z0-9']+", text.lower())


def is_degenerate(reply: str, brain_name: str = "Nova") -> bool:
    toks = _tokens(reply)
    if not toks:                                    # empty, whitespace or punctuation only
        return True
    if all(t == brain_name.lower() for t in toks):  # only the brain's own name
        return True
    return any(toks[i] == toks[i + 1] == toks[i + 2] for i in range(len(toks) - 2))  # one token 3+ times in a row


def counts_as_recall(reply: str, key: str, brain_name: str = "Nova") -> bool:
    return key.lower() in reply.lower() and not is_degenerate(reply, brain_name)


# ---------------------------------------------------------------- weekly facts
def current_week() -> str:
    y, w, _ = datetime.date.today().isocalendar()
    return f"{y}-W{w:02d}"


def draw_rotating(week: str) -> list:
    """The week's drawn facts from POOL: deterministic per ISO-week string, never overlapping FIXED."""
    return random.Random(week).sample(POOL, N_ROTATING)


def scored_facts(week: str) -> list:
    """[(statement|None, question, key, rotating)] : 6 fixed then 6 drawn."""
    return [(s, q, k, False) for s, q, k in FIXED] + [(s, q, k, True) for s, q, k in draw_rotating(week)]


def session1_turns(facts: list) -> list:
    """Greeting, the fixed tellings with chit-chat between, then this week's drawn tellings, then two unscored turns."""
    c = CHITCHAT_1
    fixed = [s for s, _, _, r in facts if s and not r]   # work, cat, sister, car (FIXED order: cat, sister, car, work)
    rot = [s for s, _, _, r in facts if s and r]
    cat, sister, car, work = fixed
    return [c["greet"], work, cat, c["rain"], sister, car] + rot + [c["ask_cat"], c["interesting"]]


# ---------------------------------------------------------------- the sleep arm's cortex
class _PlasticShim:
    """Looks like the rwkv model to Cortex.read/speak, but runs the plastic cortex (with its adapters)."""

    def __init__(self, plastic):
        self.plastic = plastic

    def forward(self, ids, state):
        with torch.no_grad():
            logits, st = self.plastic(ids, None if state is None else [s.float() for s in state])
        return logits[-1], st


class _PlasticReader(Cortex):
    """The cortex as sleep left it: no model load; every forward goes through the plastic adapters."""

    def __init__(self, cortex, plastic):
        self.pipe = cortex.pipe
        self.model = _PlasticShim(plastic)
        self.plastic = plastic


# ---------------------------------------------------------------- one session / one arm
def run_session(home, cortex, hippocampus_on, turns, lines, label, scored=None):
    """One session on a fresh Brain object (= a restart; memory from disk). Returns (brain, hits, asked).
    scored: {turn text: (key, rotating)} for the turns that count."""
    b = Brain(home, cortex=cortex)
    b.hippocampus.enabled = hippocampus_on
    lines.append(f"\n### {label}\n")
    hits = asked = 0
    for t in turns:
        reply, info = b.hear(t)
        mark, tag = "", ""
        if scored and t in scored:
            key, rotating = scored[t]
            ok = counts_as_recall(reply, key, b.meta["name"])
            asked += 1
            hits += ok
            mark = " ✅" if ok else " ❌"
            tag = " _(this week's draw)_" if rotating else ""
        rec = f" _(recalled: {info['recalled']})_" if info["recalled"] else ""
        lines.append(f"- **you:** {t}{tag}\n  **brain:** {reply}{mark}{rec}")
    b.save()
    return b, hits, asked


def run_arm(name, home, cortex, facts, lines, sleep_steps=None):
    """name: 'intact' | 'lesion' | 'sleep'. Returns (hits, asked)."""
    shutil.rmtree(home, ignore_errors=True)
    s1 = session1_turns(facts)
    s2_scored = {q: (k, r) for _, q, k, r in facts}
    s2 = [q for _, q, _, _ in facts] + CHITCHAT_2
    b1, _, _ = run_session(home, cortex, name != "lesion", s1, lines, "session 1")
    reader = cortex
    if name == "sleep":
        episodes = [(ep.text, ep.strength) for ep in b1.hippocampus.episodes]
        plastic = PlasticCortex(cortex)
        base = [s.float() for s in cortex.read(BIRTH, None)[1]]
        losses = sleep(plastic, base, episodes, GENERIC_CHAT, steps=sleep_steps)
        first = f"{losses[0][0]:.3f}" if losses else "n/a"
        last = f"replay {losses[-1][0]:.3f}, keep {losses[-1][1]:.4f}" if losses else "n/a"
        lines.append(f"\n### sleep\n\n- {len(episodes)} stored episodes + {len(GENERIC_CHAT)} ordinary chat texts replayed, "
                     f"{sleep_steps} steps; replay loss {first} -> final ({last})")
        reader = _PlasticReader(cortex, plastic)
    _, hits, asked = run_session(home, reader, name == "intact", s2, lines,
                                 "session 2 (after restart" + (", after sleep, hippocampus off)" if name == "sleep" else ")"),
                                 s2_scored)
    return hits, asked


def run_week(cortex, out_path, week=None, sleep_steps=200, home_root="brain_data", today=None):
    week = week or current_week()
    today = today or datetime.date.today()
    facts = scored_facts(week)
    body = [f"## This week's rotating facts ({week})\n"]
    body += [f"- {q} -> {k}" for _, q, k, r in facts if r]
    arms = (("intact", "intact", "_weekly_intact"), ("lesion", "hippocampus off", "_weekly_lesion"),
            ("sleep", "sleep + hippocampus off", "_weekly_sleep"))
    scores = {}
    for arm, title, sub in arms:
        body.append(f"\n## {title}")
        scores[title] = run_arm(arm, os.path.join(home_root, sub), cortex, facts, body, sleep_steps)
    # header is built after the body, written above it
    s = {t: f"{h} / {n}" for t, (h, n) in scores.items()}
    off = scores["hippocampus off"][0]
    header = [f"# Weekly conversation test, {today} ({week})", "",
              "Key hits / questions asked (strict: key word present and reply not degenerate):", ""]
    header += [f"- {t}: {v}" for t, v in s.items()]
    header += [f"- difference (intact − hippocampus off): {scores['intact'][0] - off:+d}",
               f"- difference (sleep − hippocampus off): {scores['sleep + hippocampus off'][0] - off:+d}", ""]
    text = "\n".join(header + body) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(text)
    return scores, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"brain/eval/transcripts/{datetime.date.today()}.md")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--week", default=current_week(), help="ISO week, e.g. 2026-W45 (seeds the rotating facts)")
    ap.add_argument("--sleep-steps", type=int, default=200)
    a = ap.parse_args()
    scores, text = run_week(Cortex(device=a.device), a.out, a.week, a.sleep_steps)
    print(text)
    print(f"wrote {a.out}; scores {scores}")


if __name__ == "__main__":
    main()
