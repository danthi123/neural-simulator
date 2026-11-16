# Pre-registration: a night that rehearses the cortex's own replies (sleep_rehearse), judged on the open-conversation eval (1.5B)

Written 2026-11-16, before any run of the switch. Capability: Sleep consolidation. The primary measure is the
open-conversation eval (`brain/research/2026-11-15-open-conversation-eval.md`), not the fixed weekly.

**Record brief.**
- Open-conv baseline (`results/2026-11-15-open-conv-baseline-1.5b.md`): after one night at the defaults, day 2
  collapses into one repeated reply on 4/6 seeds, x11 to x14 of 16 turns, usually a day-1 user sentence. It also does
  so with the hippocampus off (1102, 1103), so the fault is in the night's synapses.
- The night is next-token CE on stored user sentences, 480 steps, plus KL on three fixed snippets (`brain/sleep.py`).
- A weaker night (240 steps, or KL x2) did not lower false recall in the probe
  (`results/2026-11-15-weak-night-control-1.5b.md`). A second KL term on texts the pretrained cortex wrote itself (the
  near-miss frame guard) did: false 8.83 -> 5.50, partly through abstentions. So protection with breadth is the lever
  with evidence, not dose.
- New-core generative-replay work does not cover this. The old-track findings are about facts on the spiking
  substrate.
- Prerequisites met. Host decision: none new. `near_miss.guard_texts` is reused: the cortex's own greedy reply to each
  user turn the brain actually heard.
- A day-2 control on the baseline's slept homes (night off, hippocampus on) is running on the pool. Its result is
  reported beside this one and does not change these criteria.

**Method.** `brain/eval/open_conv.py --model 1.5b --device cuda --seed <s>`, with and without `--sleep-rehearse`,
on fresh seeds 1106-1111. These conversations are new to every method. Run on AWS, both arms in one batch.

**Criteria (six seeds).** "Collapse" means a day-2 intact reply repeated more than 3 times of 16.
- R1 (speech survives the night): no collapse on >= 5/6 seeds with rehearse, and fewer collapsed seeds than the
  default night.
- R2 (memory kept): intact day-2 recall with rehearse (direct + reworded, of 6 per seed, summed) >= the default's sum.
  The hippocampus-off arm's day-2 recall is reported: that is the consolidation read-out.
- R3 (no new harm): false recall and intrusions, summed, are not higher with rehearse than with the default.
- R4 (reading): a reader of the 12 day-2 transcripts judges the rehearse conversations no worse on every seed. The
  reader is a fresh sonnet subagent given the transcript pairs blind to the arm, with "which is the better
  conversation and why" per seed. Ties are allowed. R4 holds if rehearse is preferred or tied on >= 5/6.

**Decision.**
- R1-R4 hold: flip `sleep_rehearse` on by default for every size, after a land-result skeptic. The weekly runs
  beside the flip as a report.
- R1 holds and R2 fails: the night stops wrecking speech but loses what it consolidated; report it and try rehearsal
  plus more replay steps.
- R1 fails: guarding replies to the heard turns is not enough. The next method is broader self-rehearsal: the
  cortex's replies to cue onsets it writes itself.

**Expected.** R1 holds. R2 is uncertain, because the collapse inflated some "right" answers by accident and deflated
others.
