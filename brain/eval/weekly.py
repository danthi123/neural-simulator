"""Weekly conversation test: teach, chat, restart (= sleep), recall; intact vs hippocampus lesion in session 2 (both arms sleep). Writes a markdown transcript.

python -m brain.eval.weekly [--out brain/eval/transcripts/<date>.md] [--device cpu] [--model 0.1b|0.4b|1.5b|PATH]
    [--shared-sleep] [--homes DIR] [--seed N] [--sleep-lesion]   (on the GPU: bash tools/brain_gpu.sh weekly --model 1.5b --shared-sleep)
"""
from __future__ import annotations

import argparse
import datetime
import inspect
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


SEED = {}  # --seed N -> {"sleep_seed": N}: the night's replay order (and torch/numpy) per seed; unset = Brain's default


def run(home: str, cortex: Cortex, lesion: bool, lines: list):
    for s, turns in (("session 1", SESSION_1), ("session 2 (after restart)", SESSION_2)):
        # A new Brain object = a restart; memory comes from disk, and a restart is a night: the episodes stored in
        # session 1 are replayed into the cortex's slow synapses when session 2 starts. Both arms live session 1 and
        # sleep identically (same seed, same stored episodes); the lesion is applied AFTER that, only while answering
        # in session 2 (hippocampus off = no recall, no storing). So intact vs lesion differ only by the hippocampus.
        b = Brain(home, cortex=cortex, **SEED)
        if lesion and s != "session 1":
            b.set_lesions(hippocampus_off=True)
        lines.append(f"\n### {s}\n")
        _session(b, turns, lines)


def run_shared(homes: dict, cortex: Cortex, lines: list, sleep_lesion: bool = False):
    """--shared-sleep (adopt-1.5b plan step 11): session 1 is lived once and the night slept once; the slept home is
    copied for the lesion arm, so both arms start session 2 from the same synapses and hippocampus. Halves the sleep
    cost (one night instead of two) at 1.5B. Session 1 appears once in the transcript."""
    intact, lesion = homes["intact"], homes["lesion"]
    lines.append("\n## session 1 (shared by both arms)\n")
    _session(Brain(intact, cortex=cortex, **SEED), SESSION_1, lines)
    Brain(intact, cortex=cortex, **SEED).save()      # restart = the one night
    shutil.copytree(intact, lesion)
    arms = [("intact", intact, False, False), ("lesion: hippocampus off", lesion, True, False)]
    if sleep_lesion:   # --sleep-lesion: also switch off what the night learned (the pretrained synapses speak)
        shutil.copytree(intact, homes["sleep_lesion"])
        arms.append(("lesion: hippocampus off + sleep's synapses off", homes["sleep_lesion"], True, True))
    for name, home, off, cons_off in arms:
        b = Brain(home, cortex=cortex, **SEED)       # nothing pending: no second night
        assert b.meta["sleeps"] >= 1 and not b.pending()
        b.set_lesions(hippocampus_off=off, consolidated_off=cons_off)
        lines.append(f"\n## {name}\n\n### session 2 (after restart)\n")
        _session(b, SESSION_2, lines)


def main(argv=None, cortex: Cortex | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"brain/eval/transcripts/{datetime.date.today()}.md")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default=None, help="cortex checkpoint path or size tag (default: the 0.1B cortex)")
    ap.add_argument("--shared-sleep", action="store_true", help="sleep once and copy the home for the lesion arm")
    ap.add_argument("--homes", default="brain_data", help="directory for the two arms' brain homes")
    ap.add_argument("--sleep-lesion", action="store_true",
                    help="with --shared-sleep: a third arm with the hippocampus AND the night's synapses off")
    ap.add_argument("--self-completion", action=argparse.BooleanOptionalAction, default=None,
                    help="self episodes store their completion only (Brain(self_completion=...); default ON since "
                         "2026-11-15). --no-self-completion = the flag-off arm. Unset = the Brain default")
    ap.add_argument("--self-key", choices=("exchange", "question"), default=None,
                    help="switch (default OFF = exchange): what the self episodes are keyed by (Brain(self_key=...))")
    ap.add_argument("--key", default=None, help="pattern-completion key (Brain(key=...)); unset = the Brain default (S0)")
    ap.add_argument("--key-bar", default=None, help="bar for a non-S0 key (Brain(key_bar=...)): default | size | slept")
    ap.add_argument("--seed", type=int, default=None, help="sleep seed + torch/numpy seed (default: unset, as before)")
    a = ap.parse_args(argv)
    SEED.clear()
    if a.seed is not None:
        import numpy as np
        import torch
        SEED["sleep_seed"] = a.seed
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
    if a.self_completion is not None:
        SEED["self_completion"] = a.self_completion
    if a.key is not None:
        SEED["key"] = a.key
    if a.key_bar is not None:
        SEED["key_bar"] = a.key_bar
    if a.self_key is not None:
        SEED["self_key"] = a.self_key
    key_note = f"; key {a.key} ({a.key_bar or 'default'} bar)" if a.key else ""
    sc_on = inspect.signature(Brain.__init__).parameters["self_completion"].default if a.self_completion is None else a.self_completion
    cortex = cortex or Cortex(model_path=resolve_model(a.model), device=a.device)
    lines = [f"# Weekly conversation test, {datetime.date.today()}"]
    if a.model or a.shared_sleep:
        name = getattr(cortex, "model_name", None)
        try:
            cfg = sleep_defaults(name)
        except KeyError:
            cfg = "Brain arguments"
        lines.append(f"\n_cortex: {name}; sleep: {cfg}; device: {a.device};"
                     f" {'one shared night' if a.shared_sleep else 'each arm sleeps'}"
                     f"{'' if a.seed is None else f'; seed {a.seed}'}"
                     f"{'; self episodes store their completion' if sc_on else ''}{'; self episodes keyed by their question' if a.self_key == 'question' else ''}{key_note}_")
    homes = {k: os.path.join(a.homes, f"_weekly_{k}") for k in ("intact", "lesion", "sleep_lesion")}
    for h in homes.values():
        shutil.rmtree(h, ignore_errors=True)
    if a.shared_sleep:
        run_shared(homes, cortex, lines, sleep_lesion=a.sleep_lesion)
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
