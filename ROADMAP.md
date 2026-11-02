# Roadmap

Plain-language status of the project. Updated when a milestone moves; the dated plan lives in
[`brain/ARCHITECTURE.md`](brain/ARCHITECTURE.md) §4 and on the task board (tasks titled "New core: ...").
The pre-pivot roadmap (2,700 lines of old-track history) is archived at
[`docs/archive/ROADMAP-2026-11-02.md`](docs/archive/ROADMAP-2026-11-02.md).

## What we are building

A functional digital brain you can talk with openly, that learns and grows over time (owner, 2026-11-02). Not a
chatbot with a fact lookup: what it tells you comes from its own memory and its own state, and it changes with use.

## How it works today (new core, `brain/`)

- **Cortex:** a small pretrained recurrent language network (RWKV-7, 0.1B to 1.5B) is the brain's language cortex.
  Pretraining stands in for development (a declared trade).
- **Hippocampus:** each thing you tell it is stored as the change it made to the cortex's fast memory, and brought
  back by putting that change back into the cortex, never by pasting text.
- **Memory strength:** often-used memories get stronger, rarely-used details fade, like a person.
- **Sleep (first version, 2026-11-02):** replays the day's memories into slow cortical connections, so facts survive
  with the hippocampus switched off. Gradient learning here is a declared scaffold (`brain/SCAFFOLDS.md`).
- **Restart = a new day:** the brain keeps its memories across restarts.

## Timeline

| By | Target | Status |
|---|---|---|
| 2026-11-09 | First weekly conversation test (Hermes); upgraded test with whole / memory-off / after-sleep arms (Pi) | in progress |
| 2026-11-16 | Sleep in the live brain on every restart; often-used facts never wrong; switching a part off visibly changes replies; cortex size chosen | in progress |
| 2026-11-30 | Recall from reworded questions; mood/arousal shapes replies (proven by switching it off); asks about new things | planned |
| 2026-12-14 | A simulated multi-week life with nightly sleep: relearns faster, recalls from partial cues, transfers skills | planned |
| 2026-12-31 | Sleep learning by a local biology-style rule instead of gradients; 1.5B cortex | planned |
| 2027 Q1 | Start moving the core onto the shared spiking substrate | planned |

## How progress is measured

1. The weekly conversation test (`brain/eval/weekly.py` -> `brain/eval/transcripts/<date>.md`), every Monday.
2. The milestones above, met by their dates.
3. Share of commits that touch the brain (shown at every Claude session start; process work is overhead).

## Who does what

Claude builds the new core and makes judgment calls; Hermes (server) runs and lands experiments and the weekly test;
Pi (server) works a lane of build tasks, with Claude available to it for hard steps and judgment. Details:
[`research/coordination/LOCAL_LLM_RUNBOOK.md`](research/coordination/LOCAL_LLM_RUNBOOK.md) §4.
