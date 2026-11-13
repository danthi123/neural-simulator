# Pre-registration: does the slept cortex know when it knows? (gate reinstatement on cortical confidence, 1.5B)

Written 2026-11-13 before any run. Capability: memory selection after sleep (metacognition: feeling of knowing).

**Record brief.** The 11-05 routing fix (`results/2026-11-05-hippo-routing-1.5b-live.md`) removed misrouting, but
reinstating the RIGHT episode into a slept cortex costs answers: on rows where f reinstates, f gets 23 right and the
slept cortex alone (c) 33; where nothing fires both get 23. F3 failed (f 7.67 vs c 9.33). Its named next method (2):
reinstate only when the slept cortex's own confidence is low, the same "do I know this" signal the lesson arbitration
needs (HANDOFF). Known at 0.1B (`results/2026-11-03-false-recall-feeling-of-knowing/README.md`): the answer's own
log-prob does NOT separate correct from confabulated answers (name-token AUC 0.47); cortical cue familiarity does
(0.79). That was correct vs confabulation on never-told questions; this test asks a different split: on TOLD
questions, is c right or wrong. Biology: complementary learning systems with selective reinstatement (old-track
`research/findings/2026-08-06-replay-consolidation-selective-CA1-to-cortex-reinstatement-research-gate.md`); the
hippocampus is needed less as the neocortex consolidates.

**Prerequisites.** Met: 1.5B cortex, sleep consolidation (c 9.33/12), the S_all routing fix (no wrong episode fires).
Not tested here: a live gate inside Brain (this measures the signal and simulates the gate exactly, see G0).
Host decision: none in the brain; the bar is fitted on held-out seeds (an evaluation choice, as the routing bar).

**Method.** `brain/experiments/confidence_gate_probe.py`, 1.5B, `--steps 480 --lr 5e-4 --kl-weight 3`, seeds
42/43/44/100/101/102, AWS g5 (A10G). Per question (12 told, 12 paraphrase, 12 never-told): c = hippocampus off,
f = S_all key with the 1.5B bar; c's reply teacher-forced for its confidence read-outs. Gate h (offline,
`confidence_gate_table.py`): c's reply if signal >= t, else f's reply; t fitted leave-one-seed-out on the other five
seeds' told + paraphrase rows (max correct; ties = highest t). **Primary signal: `mean_logp`** (mean per-token
log-prob of c's own reply under the live slept cortex). Chosen from the existing 0.1B data before this run, on THIS
split (told + paraphrase, right vs wrong; `2026-11-03-false-recall-feeling-of-knowing/s*_st*.json`, greedy answers):
mean_logp AUC 0.739 at 240 and at 480 steps; shift 0.707 / 0.606; cue_gain_max 0.641 / 0.634; min_logp 0.616 / 0.504.
This is not the ruled-out method of `brain/research/2026-11-03-confabulation-next-methods.md` (b): that was answer
confidence for correct vs CONFABULATED never-told answers (0.63-0.65 there, unchanged); here it only decides whether
to reinstate a told memory, and G4 guards false recall. Secondary, exploratory only: `shift`, `cue`, `first_logp`,
`min_logp` (a secondary that passes is a lead for a new registration, never this verdict).

**Criteria (exact means, `fractions.Fraction`).**
- G0 instrument: f's reply equals c's reply on >= 95% of rows where nothing fired (else the offline gate is not
  the live gate; verdict void).
- G1 signal: AUC(mean_logp) for c-correct vs c-wrong on told + paraphrase, pooled >= 0.70 and per-seed mean >= 0.65.
- G2: h told recall >= c - 0.5.
- G3: h - f told recall >= 1, and h >= f on >= 5/6 seeds.
- G4: false recall (never-told, of 12) h <= f + 0.5.
LEAD = G0-G4 all hold -> next: the gate inside Brain (switch default OFF), live six-seed run including the unslept
arm (the bar must be checked there, since mean_logp is defined without sleep). NOT A LEAD otherwise; a G1 fail is
a verdict on `mean_logp` as the method, and the secondaries say where to look next.

**Expected.** Uncertain. 0.1B gives 0.74 on this split, near the G1 bar; 1.5B answers are sampled (top_p 0.3), not
greedy, and recall is higher (c 9.33/12), so fewer wrong rows to separate. G2 needs the gate to keep c's right
answers and G3 to beat f; both can hold with a modest AUC because f only reinstates on 5-7 told questions per seed.

**Also in this batch (not part of the verdict):** the weekly conversation test (`brain.eval.weekly`, default cortex),
run on AWS because the workstation is dedicated to other work; Hermes' W queue copy was removed.
