# Curiosity: novelty-gated lived inquiry episode, DEV at 1.5B (2026-11-19)

DEV only (dev bank, seeds 1144/1146/1147; 1145 pending), not a generalization claim. Harness
`brain/experiments/inquiry_gate_dev.py`, bar 12.4 nats, lesson "My friend keeps a zorbit in her kitchen." lived as an
earlier conversation with the reply "I don't know that word. What is it?". 2 samples x 12 frames x C/R/N per arm.
Runs: W (1144), pool41/42 (1146/1147); first run VOID (recall leak, `.../VOID_recall_leak_igd_s1144.json`).

Engagement: lesson-only replies are identical to off on 72/72 rows per seed (the lesson cannot leak through ordinary
recall). The gate fired C 0/24, R 16/24, N 24/24 per seed (same on all seeds: novelty is read from the birth state).
Verbatim copies of the guided reply: 0 in every arm.

Blind readers (haiku + sonnet, class hidden, shuffled; agreement asks 0.99, unfamiliar 1.00, invents 0.98,
coherent 0.97). Counts where both agree, pooled over 3 seeds (of 72 per class), C / R / N:

| arm | asks what it is | says it doesn't know | invents a meaning | coherent |
|---|---|---|---|---|
| off | 0 / 8 / 17 | 0 / 10 / 4 | 1 / 17 / 13 | 69 / 68 / 66 |
| gate | 0 / 8 / 27 | 0 / 18 / 23 | 1 / 8 / 1 | 69 / 71 / 68 |
| forced (always on) | 0 / 8 / 28 | 0 / 20 / 23 | 0 / 6 / 1 | 70 / 71 / 68 |

Per seed, coined items (of 24): asks off/gate 6/9, 7/10, 4/8; asks-or-says-unknown 6/18, 7/17, 4/14; invents 5/0, 3/0,
5/1. Consistent on all 3 seeds.

Reading: the lived episode makes the brain admit it does not know a new word and nearly stops it inventing meanings,
without touching common-word replies or coherence. It asks the question itself less often than it admits not knowing
("I've never heard of that before." without "What is it?"). The gate adds nothing over always-on here: the reinstated
episode only shows in replies to unfamiliar items. Rare-real items: "invents" falls 17 -> 8, which may be real
knowledge lost (rare "invents" is mostly correct knowledge, baseline note); to be read in the test.
Next (dev): guided-reply wording variants that lead with the question (`igv_*`), then the pre-registered test.
Labels: `2026-11-18-inquiry-gate-dev-1.5b/blind_labels.json` (key `_scratch/ns0/gkey.json` copied as `blind_key.json`).
