# Roadmap

Plain-language status of the project. Updated when a milestone moves; the dated plan lives in
[`brain/ARCHITECTURE.md`](brain/ARCHITECTURE.md) §4 and on the task board (tasks titled "New core: ...").
The pre-pivot roadmap (2,700 lines of old-track history) is archived at [`docs/archive/ROADMAP-2026-11-02.md`](docs/archive/ROADMAP-2026-11-02.md).

## What we are building

A functional digital brain you can talk with openly, that learns and grows over time (owner, 2026-11-02). Not a
chatbot with a fact lookup: what it tells you comes from its own memory and its own state, and it changes with use.

## How it works today (new core, `brain/`)

A small pretrained language network is the cortex; the hippocampus stores what you tell it as a change to the cortex's
fast memory and brings it back by putting that change back; memory strength grows with use and fades without it;
sleep (on every restart) replays the day into slow connections, with gradient learning as a declared scaffold.

## Where we stand (2026-11-03, honest)

- **Works:** sleep runs inside the live brain on every restart. With the memory part switched off, the slept brain
  still answers about half of 12 told facts (5.5 of 12, six runs) where the same brain with sleep switched off answers
  almost none (0.17), and one slept on scrambled content answers none. The trace is real and switching it off
  changes replies.
- **Does not work yet:** sleep adds little on top of the memory part (6.5 vs 6.3 of 12) and makes the brain invent
  answers more often (about 3 of 4 never-told questions against 2). The small model (0.1B) tops out at 6 of 12 even
  when the facts are in front of it, so part of the gap is the model, not the method.
- **Recorded negatives:** the "I don't know" gate, two versions (invented answers fell 63 to 34, goal was a quarter);
  strengthening by how often a fact is told (about one extra fact of four); three new ways to stop invented answers
  at the small model size (a "does my memory finish this cue" check, practising recall in sleep, and the meaning-based
  memory picker inside the live brain); replaying facts in varied wordings during sleep (no clear gain).
- **Leads:** mood shaping replies (bad mood darkens replies reliably, good mood only weakly); a lived "I don't know"
  lesson the brain keeps as its own memory makes it say "I don't know" to 69% of never-told questions (goal 80%),
  without hurting ordinary replies.
- **Cortex size chosen: the 1.5B model.** After sleep it recalls 8.8 of 12 told facts against 4.0 for the 0.4B
  model, better on all six runs, with no more invented answers.
- **Promising, not proven:** a meaning-based way to pick the right memory finds reworded facts far better than the
  word-overlap default (0.76 vs 0.10 at 0.4B) but is not in the brain yet and cannot tell told from untold questions.
- **Scoring was fixed tonight:** an answer that just repeats what you said ("I drive a red Volvo") or says
  "Pixel is called Pixel" no longer counts as remembering. Earlier numbers using the old count were too high.

## Timeline

| By | Target | Status |
|---|---|---|
| 2026-11-09 | First weekly conversation test (Hermes); upgraded test with whole / memory-off / after-sleep arms (Pi) | in progress; weekly scoring fixed |
| 2026-11-16 | Sleep in the live brain on every restart; often-used facts never wrong; switching a part off visibly changes replies; cortex size chosen | sleep in brain and lesion: met on branch; often-used facts: not met; cortex size: chosen (1.5B) |
| 2026-11-30 | Recall from reworded questions; mood/arousal shapes replies (proven by switching it off); asks about new things | planned; meaning-based recall and mood are leads |
| 2026-12-14 | A simulated multi-week life with nightly sleep: relearns faster, recalls from partial cues, transfers skills | planned |
| 2026-12-31 | Sleep learning by a local biology-style rule instead of gradients; 1.5B cortex | planned |
| 2027 Q1 | Start moving the core onto the shared spiking substrate | planned |

Next: move the live brain onto the 1.5B model, then retest invented answers and "I don't know" at that size (most
methods that failed tonight failed on the small model, which cannot answer half the facts even when told them).

## How progress is measured

The weekly conversation test (`brain/eval/weekly.py`, every Monday), the milestones above met by their dates, and the
share of commits that touch the brain. Who does what: Claude builds and judges; Hermes and Pi run and land experiments
([`research/coordination/LOCAL_LLM_RUNBOOK.md`](research/coordination/LOCAL_LLM_RUNBOOK.md) §4).
