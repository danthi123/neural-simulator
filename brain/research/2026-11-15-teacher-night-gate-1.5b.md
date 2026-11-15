# Pre-registration: no-training gate for the hippocampus-as-teacher night (1.5B, six seeds, pool CPU)

Written 2026-11-15 before any 1.5B run. Capability: Sleep consolidation + Metacognition (false recall after sleep);
board #279. Lever 10 on false recall after sleep; this gate runs before the method is built.

**Record brief.**
- After one night at 1.5B, never-told "my X" questions get a told fact about 7/12 (banked `nmoff`, arm s 6.83).
  Text-replay variants (M4 guard, M4b, rank 2/32) leave it at 5.5-7.8 (`results/2026-11-15-rank-diagnosis-1.5b.md`,
  MIXED: rank is not the lever at this lr).
- The awake brain is clean: on an unslept 1.5B day the default package answers never-told questions with a told fact
  0.33/12 and told questions 4.83/12 (`results/2026-11-14-key-default-1.5b.md`, part A).
- The method (record A4 "distilled reinstatement", `brain/research/2026-11-02-record-brief-consolidation-and-
  selection.md`; never built): the night trains the adapters to give, with no hippocampus, what the awake brain
  (pretrained cortex + the episode its recall returns) gives on the night's own cues, instead of next-token training
  on the telling. Literature: fine-tuning on new facts teaches answering unfamiliar questions with trained answers
  (Gekhman et al. 2024; Kang et al. 2025); a teacher with the knowledge in context does not.
- The gate asks whether that teacher, on cues the cortex writes itself, is good enough to learn from.
- Prerequisites met; host decision: the five frame onsets and the OWN frame are the near-miss guard's declared scaffold
  (`brain/SCAFFOLDS.md`), reused unchanged.

**Method.** `brain/experiments/teacher_gate_probe.py --model 1.5b --seed <s> --k 4`, seeds 42/43/44/100/101/102,
pool CPU fp32. 0.1B smoke (pool40, seed 42, k 2) before this prereg: it runs; its own cues were poor ("What is
Pixel?", friend-perspective "What is your name?"), own_ok 0/12; not a result.

**Criteria (six-seed means).**
- T1 (teacher right on its own told cues): own_ok >= 8 of 12.
- T2 (teacher clean on never-told cues): never_false <= 10% of never_n.
- T3 (no test leak): no generated cue equals a probe test question (FACTS, PARA, NEVER_TOLD); any overlap is listed
  and those cues are dropped from the method.

**Decision.** T1-T3 hold: build the teacher night (switch default OFF) and pre-register its mixed-day test against
the banked `nmoff`. T2 fails: the awake teacher would teach false answers on the night's cues; NOT A LEAD for this
cue set. T1 fails with T2 holding: the cues are the weak part (report the own questions); next is a better cue
source (for example the cortex's question about the telling in the user's own voice) before any build.

**Expected.** T2 holds (the awake brain is clean). T1 uncertain: the 0.1B smoke's cues were poor.
