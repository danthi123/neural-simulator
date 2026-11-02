# brain/ — architecture (read this in full at the start of every session)

**Goal (owner, 2026-11-02):** a functional digital brain you can speak with openly, like a young person, that learns
and grows from what it lives through. Not LLM + RAG. Decision record:
`docs/plans/2026-11-02-pivot-functional-digital-brain-DECISION.md`. This package is the NEW core; everything outside
`brain/` is the frozen old codebase (a library of proven mechanisms and a baseline, never imported by default).

## 1. The parts and what each one does

| Part | File | Role in a conversation | Biology it stands for | Today |
|---|---|---|---|---|
| Cortex | `cortex.py` | Reads and speaks. Its recurrent state is the brain's moment-to-moment activity; its matrix memory (per head, k->v associations written by a delta rule) is fast synaptic memory. | Neocortex: slow-learned language and world knowledge | RWKV-7 G1 0.1B, pretrained (declared trade below) |
| Hippocampus | `hippocampus.py` | Stores each experience in one shot as the change it made to cortical memory; brings it back by pattern completion from the current cortical state, and reinstates it as cortical memory, never as pasted text. | Hippocampal indexing + reinstatement | state-delta episodes, cosine pattern completion |
| Memory strength | `hippocampus.py` | Strength grows with use (each recall), spacing, and salience; unused, weak memories fade. Important, often-used things become near-permanent. | Retrieval practice, spacing, emotional enhancement, decay | per-episode strength with power-law decay |
| Sleep | `sleep.py` | Replays strong episodes and writes them into the cortex's persistent baseline memory (the state it wakes with), so later recall needs no hippocampus. Gist survives; detail may fade. | Systems consolidation / semanticization | accumulate replayed state deltas into the baseline state, with interference |
| Neuromodulators | `modulators.py` | Surprise (the cortex's own prediction error) and affect set how strongly an experience is stored and colour how it speaks. | Dopamine, noradrenaline, acetylcholine, affect | surprise -> encoding gain (first); affect later |
| Self | `self_model.py` | Who it is and who you are: the strongest, most-used memories, so it never loses its own or a friend's name. | Autobiographical self-memory | identity episodes at maximal strength, consolidated first |
| Loop + persistence | `chat.py`, `store.py` | Turn loop; everything above is saved to disk and survives a restart. | Continuity of a life | CLI chat; one brain per `--brain` directory |

## 2. Rules that keep this a brain and not a chatbot

1. **No text retrieval into the prompt.** Memory returns as cortical state (reinstated associations), selected by the
   cortex's current state, never by keyword search over stored text.
2. **Every part must change the conversation.** A part counts only if switching it off changes what the brain says
   on the weekly prompts (lesion test). `brain/eval/` runs these.
3. **Forgetting is allowed; the right forgetting.** Measure savings (relearning is faster than learning fresh), cued
   recall (the right context brings it back), transfer, and zero errors on high-use facts (names, identity). Never
   gate on perfect retention of rarely used detail.
4. **Declared trade.** The cortex's starting ability comes from gradient pretraining (stands in for evolution and
   childhood). Everything learned after birth is learned with local, brain-style rules (state deltas, Hebbian
   accumulation, decay). Self-reports are functional read-outs; never claims of experience.
5. **Not spiking yet, honestly.** The cortex is an artificial recurrent network; the hippocampus is a state-delta
   store. Spiking implementations from the old codebase (BTSP / CA3 hippocampus, neuromodulator circuits) are ported
   in only when they improve the conversation on the weekly test, never for their own sake.

## 3. How we work on it (anti-tunnel rules)

- Progress = the weekly 20-prompt transcripts (`brain/eval/weekly.py`) side by side with last week's, plus one
  fluency number, plus the owner talking to it. A week that changes neither was off track.
- Any single problem gets at most two days; then ship the simplest working thing and list it in
  `brain/SCAFFOLDS.md`, or ask the owner.
- Keep `brain/` small (target under 3,000 lines of code). Port from the old code only what a test says it needs.
- Six-seed preregistered tests only when a capability is claimed for the record; day-to-day work iterates freely.

## 4. Milestones

- **Day 3:** CLI conversation in the cortex's own words with identity memory.
- **Day 7:** hippocampus stores what you tell it, recalls it by context, survives a restart; weekly test running.
- **Day 14:** sleep consolidation; high-use facts never wrong; one lesion visibly changes replies. If not met, stop
  and report.
