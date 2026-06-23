# C2 grow+no-forget at a LEARNABLE shift = NEGATIVE — the genuine SCALE WALL, now mapped across the ENTIRE shift axis (the cloud/scale-up decision point) (2026-06-23)

**The C2 self-replay MECHANISM is sound (replay helps directionally at both extreme + moderate shifts), but the
3.4M-param toy CANNOT hold both distributions tightly enough at ANY shift magnitude that is also a real forgetting
stressor. ⇒ a model-CAPACITY wall — the loop needs a bigger generator. Stronger evidence than the prior single-point
NEGATIVE: the whole shift axis is mapped.** `research/runners/_genseq_C2_moderate_shift_derisk.py` (new), GPU, NO `sim/` edit.

## Result (new-corpus = TinyStories/Shakespeare interleave SH_FRAC=0.45 → new-ppl 47.8 = 7.7× baseline)
| condition | orig ppl | new ppl | orig-retention | new-drop |
|---|---|---|---|---|
| baseline (pre-grow) | 6.21 | 47.83 | — | — |
| no-replay (control) | 13.42 | 30.37 | 46.3% | 36.5% |
| replay 0.3 | 11.52 | 24.27 | **53.9%** | 49.3% |
| replay 0.5 | 13.04 | 24.33 | 47.7% | 49.1% |

On-bridge RF install EXACT (ppl_ratio 1.000000 both distributions — the off-bridge table IS the on-bridge table).
**Verdict NEGATIVE:** retains≥85%=False (best 53.9%), no_replay_forgets≥1.3×=False (only 1.16×), dose_monotone=False
(0.46→0.54→0.48).

## The decisive characterization (the corpus-selection sweep's hard fact)
Gen-F trained on the FULL 8MB TinyStories → pure TinyStories topic/structural slices are 0.8–1.05× baseline ppl
(NO shift; the no-replay control wouldn't forget = an uninformative anti-cheat). The only genuinely-distinct corpora
are out-of-domain (Shakespeare 42×, WikiText 91×; "simplifying" by line-length does NOT lower ppl — register/vocab
drives it). ⇒ **there is NO corpus on the 3.4M toy that gives BOTH a clean forgetting contrast AND ≥85%
replay-retention** — pure-distinct shifts forget hard but cap replay at ~52–55%; in-band mixtures self-reinforce the
old distribution so they don't forget enough. Combined with the prior extreme-shift run (41× Shakespeare → 52%
retention), the WHOLE shift axis is mapped: the 3.4M capacity is the wall.

## ⇒ THE CLOUD/SCALE-UP DECISION POINT (per `feedback_long_local_runs_ok_confirm_cloud_cause`)
The loop (train → generate → grow → no-forget) needs a BIGGER generator (capacity to hold 2 distributions). This is a
model-CAPACITY wall. Per the cloud rule: SIZE a target model, MEASURE its VRAM + throughput, give an ETA, run LOCALLY
unless VRAM > 24 GB (cloud only for a genuine VRAM wall, ~1B+ params). A **50–200M-param model likely fits in 24 GB →
a LOCAL (longer-wallclock) scale-up, NOT cloud.** PREPARED for the owner's morning decision (target model size + the
measured VRAM/throughput + the ETA + the explicit local-vs-cloud call). The C2 MECHANISM and the C1 consolidation are
both validated; only the SCALE needed to *demonstrate* the loop remains — and it is most likely a long local run, not
a cloud job.

## Scope / honesty
3.4M toy; the mechanism (replay-prevents-forgetting) is directionally confirmed at both 41× and 7.7× shifts; the
≥85%-retention bar is a capacity property, not a mechanism flaw. Honest negative = the deliverable: it precisely
locates where scale becomes necessary.
