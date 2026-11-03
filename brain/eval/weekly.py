"""Weekly conversation test: teach, chat, restart (= sleep), recall; intact vs hippocampus lesion in session 2 (both arms sleep). Writes a markdown transcript.

python -m brain.eval.weekly [--out brain/eval/transcripts/<date>.md] [--device cpu] [--model 0.1b|0.4b|1.5b|PATH]
    [--shared-sleep] [--homes DIR]   (on the GPU: bash tools/brain_gpu.sh weekly --model 1.5b --shared-sleep)
"""
from __future__ import annotations

import argparse
import datetime
import os
import shutil

from brain.brain import Brain, sleep_defaults
from brain.cortex import Cortex, resolve_model
from brain.experiments.strict_scorer import correct_answer

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
# The told sentence behind each user fact (None = the brain's own fact: a first-person answer is the right perspective).
TOLD = {"pixel": "My cat is called Pixel.", "marguerite": "My sister's name is Marguerite.", "volvo": "I drive a red Volvo.",
        "admin": "I work as a system administrator, mostly Linux servers.", "dan": None, "nova": None}


def mark_for(turn: str, reply: str):
    """None when the turn is not a keyed question, else True/False with the correct-perspective scorer: the answer word
    as a word, not in the question, not a tautology, not a first-person echo of the told sentence."""
    key = next((v for k, v in KEYS.items() if k in turn), None)
    return None if key is None else correct_answer(reply, key, turn, TOLD[key])


def _session(b: Brain, turns: list, lines: list):
    for t in turns:
        reply, info = b.hear(t)
        ok = mark_for(t, reply)
        mark = "" if ok is None else (" ✅" if ok else " ❌")
        rec = f" _(recalled: {info['recalled']})_" if info["recalled"] else ""
        lines.append(f"- **you:** {t}\n  **brain:** {reply}{mark}{rec}")
    b.save()


def run(home: str, cortex: Cortex, lesion: bool, lines: list):
    for s, turns in (("session 1", SESSION_1), ("session 2 (after restart)", SESSION_2)):
        # A new Brain object = a restart; memory comes from disk, and a restart is a night: the episodes stored in
        # session 1 are replayed into the cortex's slow synapses when session 2 starts. Both arms live session 1 and
        # sleep identically (same seed, same stored episodes); the lesion is applied AFTER that, only while answering
        # in session 2 (hippocampus off = no recall, no storing). So intact vs lesion differ only by the hippocampus.
        b = Brain(home, cortex=cortex)
        if lesion and s != "session 1":
            b.set_lesions(hippocampus_off=True)
        lines.append(f"\n### {s}\n")
        _session(b, turns, lines)


def run_shared(homes: dict, cortex: Cortex, lines: list):
    """--shared-sleep (adopt-1.5b plan step 11): session 1 is lived once and the night slept once; the slept home is
    copied for the lesion arm, so both arms start session 2 from the same synapses and hippocampus. Halves the sleep
    cost (one night instead of two) at 1.5B. Session 1 appears once in the transcript."""
    intact, lesion = homes["intact"], homes["lesion"]
    lines.append("\n## session 1 (shared by both arms)\n")
    _session(Brain(intact, cortex=cortex), SESSION_1, lines)
    Brain(intact, cortex=cortex).save()      # restart = the one night
    shutil.copytree(intact, lesion)
    for name, home, off in (("intact", intact, False), ("lesion: hippocampus off", lesion, True)):
        b = Brain(home, cortex=cortex)       # nothing pending: no second night
        assert b.meta["sleeps"] >= 1 and not b.pending()
        b.set_lesions(hippocampus_off=off)
        lines.append(f"\n## {name}\n\n### session 2 (after restart)\n")
        _session(b, SESSION_2, lines)


def main(argv=None, cortex: Cortex | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"brain/eval/transcripts/{datetime.date.today()}.md")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default=None, help="cortex checkpoint path or size tag (default: the 0.1B cortex)")
    ap.add_argument("--shared-sleep", action="store_true", help="sleep once and copy the home for the lesion arm")
    ap.add_argument("--homes", default="brain_data", help="directory for the two arms' brain homes")
    a = ap.parse_args(argv)
    cortex = cortex or Cortex(model_path=resolve_model(a.model), device=a.device)
    lines = [f"# Weekly conversation test, {datetime.date.today()}"]
    if a.model or a.shared_sleep:
        name = getattr(cortex, "model_name", None)
        try:
            cfg = sleep_defaults(name)
        except KeyError:
            cfg = "Brain arguments"
        lines.append(f"\n_cortex: {name}; sleep: {cfg}; device: {a.device};"
                     f" {'one shared night' if a.shared_sleep else 'each arm sleeps'}_")
    homes = {k: os.path.join(a.homes, f"_weekly_{k}") for k in ("intact", "lesion")}
    for h in homes.values():
        shutil.rmtree(h, ignore_errors=True)
    if a.shared_sleep:
        run_shared(homes, cortex, lines)
    else:
        for name, lesion in (("intact", False), ("lesion: hippocampus off", True)):
            lines.append(f"\n## {name}")
            run(homes["lesion" if lesion else "intact"], cortex, lesion, lines)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    hits = {k: sum(1 for l in lines if "✅" in l) for k in ("all",)}
    print("\n".join(lines))
    print(f"\nwrote {a.out}; ✅ total {hits['all']}")
    return lines


if __name__ == "__main__":
    main()
