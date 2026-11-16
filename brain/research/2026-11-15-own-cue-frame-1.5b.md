# Pre-registration: a cue frame for the teacher night's own cues, chosen on the DEV bank, then the gate again (1.5B, pool CPU)

Written 2026-11-15, before any run of the new frames. Capability: Sleep consolidation; board #279. This is the next
step named by `results/2026-11-15-teacher-night-gate-1.5b.md` (T1 fails with T2 holding: the cues are the weak part).

**Record brief.**
- With the near-miss OWN frame, the gate's own cues were one greedy draw, repeated identically on six seeds. Of the
  72, 54 addressed the brain ("What's your cat's name?"), 18 addressed neither, 24 contained the key, and own_ok
  was 0/12. The teacher was clean on never-told cues: 1/87.
- A "your" -> "my" rewrite by host code would be an English-grammar template. Instead, pre-listed frames (a declared
  teacher scaffold, `brain/SCAFFOLDS.md`) let the cortex write the question itself.
- **Already built: M3 retrieval-practice sleep** (`brain/sleep_retrieval.py`, 11-03), with the episode as teacher.
  Its cues come from the declared self-question prompt `SELFQ_PROMPT`
  ("User: {s}\n\nAssistant: Okay, I'll remember that.\n\nUser: What"), which is already in the user's voice.
  Its told-recall results with the hippocampus off: FALSIFIED at 0.1B (gain -0.5/12) and FAIL at 0.4B (-2.67).
  Fresh false recall fell both times: 17.7 -> 8.0 at 0.1B, 12.2 -> 6.17 at 0.4B (`results/2026-11-03-confab-m3-
  retrieval-sleep/`). It has never run at 1.5B. Its prompt is candidate F4 here, so the existing cue source is
  measured before any new one is adopted.
- Prerequisites met. Host decision: the frames are teacher scaffold text, listed below before any run.

**Method, stage A (selection).** `brain/experiments/own_cue_frames.py` runs on one pool node with no seed (greedy).
Frames F0 (OWN), F1, F2, F3 and F4 (M3's SELFQ_PROMPT, greedy) are as listed in that file. Each frame writes one question for every DEV_FACTS36
statement. A question is GOOD when it ends in "?", uses the user's voice (my/I/me and no you/your), and does not
contain the key. The frame with the most GOOD questions on the DEV bank is selected. FACTS are reported, never used
to select.

**Method, stage B (gate).** `teacher_gate_probe.py --own-frame <selected> --k 4` on seeds 42/43/44/100/101/102, pool
CPU. T1-T3 are as in `research/2026-11-15-teacher-night-gate-1.5b.md`, unchanged. The own cues are still one greedy
draw: T1 is a property of the cue set and the teacher, not six independent tests. The result says so.

**Criteria.**
- A1: the selected frame's DEV GOOD count is >= 24/36. Below that, no frame writes usable cues; report and stop.
- Stage B: T1 own_ok >= 8/12, T2 never_false <= 10%, T3 overlaps listed and dropped.

**Decision.** A1 and T1-T3 hold: no new mechanism is built. The teacher night is M3 (`brain/sleep_retrieval.py`),
run at 1.5B for the first time, with the selected frame if it is not F4. It runs alongside ordinary restudy, since M3
alone lost told recall at 0.4B. Its mixed-day test is pre-registered against the banked `nmoff`, with false recall on
never-told questions as the primary measure. A1 holds and T1 fails: the teacher's answers are the weak
part (report them); try the teacher with the recalled episode at a different gain. A1 fails: own cues from the
cortex need another source, such as the questions the user actually asked in the conversation.
