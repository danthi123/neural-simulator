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

**Addendum (2026-11-16, before any result of either batch).**
The day-2 control on the baseline's slept homes (`results/2026-11-16-open-conv-day2-control-1.5b/`) shows two
defects, not one:
- With the night off and the hippocampus on, the loop with stored user lines is gone on all six seeds, so the night
  causes that collapse.
- The pretrained cortex alone (both off) also falls into a loop on 3/6 seeds. It mirrors a personal question it cannot
  answer ("What is my favourite colour?" -> "What is your favourite colour?"), then echoes every later turn, 9-12/16.
  That is in-context repetition at near-greedy sampling (top_p 0.3). Penalties reset each turn, so nothing stops a
  repeat across turns.

A second switch is added: `Brain(habituation=N)`, default OFF, a declared mouth scaffold. A reply may not repeat a
4-token sequence from the brain's last N replies or the user's current turn. Two more arms run in a second AWS batch
on the same seeds and hardware: H (`--habituation 3`) and HR (`--habituation 3 --sleep-rehearse`).

R1-R4 apply to each arm against the default. The decision for each switch follows its own arms. If both hold
separately, HR decides whether they are flipped together. R4's blind reading covers all four arms per seed.

**Addendum 2 (before any result): report-only pool seeds 1112-1114** (CPU fp32, one per node), default vs HR. These
are not criteria and not pooled with the AWS seeds (different hardware); they are read as a replication only.

**Addendum 3 (2026-11-16 08:20, after the report-only pool seeds 1112-1114, before any AWS result).**
Pool, default vs HR: no collapse in either arm (0/3), so the collapse is seed-dependent (4/6 at baseline). HR lost
day-2 recall, 13 -> 7 of 18; false recall 6 -> 3, intrusions 1 -> 1. The transcripts show why. The default night puts
the facts in the cortex: it answers right even when the hippocampus returns the wrong episode ("What is my dad
called?" -> "Bartholomew", recalled episode: the hedgehog). HR answers in a generic voice ("Your niece lives in a big
city."). Root cause, in code: each guard text is `User: <heard turn>\n\nAssistant: <pretrained reply>`, and the KL
covers every position. So the guard holds the fact sentence itself to the pretrained cortex while the replay CE learns
that same sentence: the two terms fight.
This does not change R1-R4 for the AWS arms. It names the next method if R2 fails there: **reply-only rehearsal**
(`sleep_rehearse="reply"`), with the KL only on the positions that predict the reply tokens. The night still keeps the
cortex's own way of answering a heard turn, but leaves the turn's content free to be learned. It is judged by the same
R1-R4 against the default on fresh seeds.

**Addendum 4 (2026-11-16, after the first AWS result and the pool reply-only report, before any reply-only AWS run).**
- Pool reply-only (report-only, seeds 1112-1114):
  - recall 10/18 (default 13, whole-text rehearse 7); false 3 (default 6);
  - stored-line reuse 1 (default 3);
  - one in-context loop on 1113 ("I don't know." x13, intact arm only; the hippocampus-off arm recalls 4).
- Next run: AWS g6, fresh seeds 1115-1120, two arms: default (`ocd`) and reply-only (`ocy`, `--rehearse-reply`).
- R2, R3 and the decision rules are unchanged.
- **R1 is restated** so that a paraphrased loop counts. A seed is collapsed if either holds:
  - the most repeated intact day-2 reply occurs > 3 of 16 times; or
  - `brain/experiments/stored_line_reuse.py` counts > 3 of 16 replies.
- R1 holds if collapsed seeds with reply-only are <= 1 of 6 and <= the default's.
  The "fewer than the default" clause is dropped: it is vacuous when the default rarely collapses (1/6 on 1106-1111).
- **R4:** one blind sonnet reader per seed sees the two conversations under shuffled letters, as pairs per the original
  prereg, and scores each 1-10. R4 holds if reply-only scores >= the default's on >= 5/6 seeds.
- **Answer rate is reported:** abstentions ("don't know", "not sure") on memory questions, intact arm. A flip needs
  reply-only's abstentions <= the default's + 2 summed over six seeds.

**Addendum 5 (2026-11-16, before any result of the 1115-1120 AWS batch).**
Pool replication seed 1121 (report-only) echoes the user's turn verbatim on 14/16 day-2 turns in BOTH arms. Neither
collapse count in addendum 4 sees it: each echo is a different string, and it is not a stored line. R1's collapse
therefore also includes **echoes > 3 of 16** (intact day-2 reply == the user's turn, as `rehearsal_night_score.py`
counts it). It applies to both arms alike.
