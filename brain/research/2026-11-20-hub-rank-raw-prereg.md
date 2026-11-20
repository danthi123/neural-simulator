# Prereg (dev): R-raw, the meaning key's raw similarity picks the winner, strength only gates firing

Date 2026-11-20. Lever 2 against the hub-episode defect (lever 1, M-use, KILLED 11-19:
`brain/experiments/results/2026-11-19-life-screens-1.5b.md`). Design panel: 3 independent designs (use-by-efficacy,
key-side, strength dynamics), each refuted by a skeptic (all three refuted as written), synthesis below with the grafts.
Record check 11-20: reconsolidation scoping `research/findings/2026-06-17-reconsolidation-conversational-memory-scoping.md`
(prediction-error-gated restabilisation; not built); S_all meaning key (`results/2026-11-18-sall-*.md`, selection MET as
read-out); no hub method on record. Prerequisites: a meaning key (built, S_all) and strength-weighted competition (built).
Expected to fail, stated before the run: section 7.

## 0. Why not the other two (from the verdicts)
- **M-eff (use by reinstatement efficacy):** as written it cannot run on a slept cortex (`read_nll` raises with the
  consolidated adapters on), its gain is a KL and positive for any state delta, and a random-episode control misses the
  hub, which is shape-matched ("My X is called Y"). Salvage (read/forward teacher forcing, a shape-matched decoy) costs
  two passes per fetch and still credits template priming. Kept as the fallback after R-raw.
- **Fan normalisation (use-share homeostat):** 3 of the 4 true-right fetches on s310-312 are weighted-only winners by a
  weight ratio of 1.01-1.25, so any fan penalty flips them; its fixed point leaves a hub share of 0.4-0.6.
- Graft: right fetches are counted by EPISODE IDENTITY (the ledger), never by reply `ok`: on s310-312 there are 4
  right fetches, not 6 (4 of the 6 right replies followed a wrong fetch; the slept cortex answered).

## 1. Method
The hub wins two ways: by strength (s311: about 9/10 of its fetches are weighted-only winners; strength 6 sits at tanh
about 1 against 0.36-0.46 for a fact told once) and by the surface key itself (s310 4/5, s312 4/7 of the hub's fetches are
the S0 raw winner). R-raw attacks both: the meaning key S_all, and the winner = argmax of the RAW similarity. Strength still
decides whether that winner fires (sims[best] * tanh(eff) >= bar), the margin is on raw sims (>= MARGIN), and USE_BOOST,
decay and sleep replay are unchanged. Strength can no longer make an episode the winner for a cue it does not match best.

## 2. Switch
`Hippocampus.RANK_ON_RAW = False` (default; the competition is the old code verbatim, pinned by
`brain/tests/test_rank_raw.py` against 61c6ec728). `Brain(rank="raw")` sets it; not saved in the home. `life.py --rank raw`.
last_fetch also logs weighted_winner and weighted_would_fire.

## 3. Offline screen (before any life)
`python -m brain.experiments.life_screen rank --home <trunk> --seed S --model 1.5b --night 7 ...` on the banked night-7
trunks s310-s314. Four intact-frozen arms (FREEZE_USE: the battery changes no episode), copies only:
A = S0 weighted (as banked), D = S0 raw-ranked, B = S_all weighted (slept bar 0.165/0.02), R = S_all raw-ranked (slept bar).
No bar is refit. Precondition: S_all reads the stored delta of every episode; an S0 home re-keys at load (no fit). If a
home cannot be re-keyed the screen is VOID there, not a negative.

Kill rules, pooled over s310-s314 unless stated, fixed now:
- **K1 selection:** right-episode fetches in R >= max(2 x A, A + 4) and >= B. Else KILL.
- **K2 hub:** top-1 share of fetched episodes in R <= 0.40 on every home where A is above 0.40. Else KILL.
- **K3 open feed:** fires on open + greet + never-told turns in R <= 0.5 x A. Else KILL.
- **K4 replies:** KILL if H right in R is more than 3 below A (pooled); KILL if open intrusions in R exceed A by more
  than 2; KILL if the whole-transcript read shows R losing a user fact A kept on 2 or more homes.
- Attribution, reported not ruled: R - B = raw ranking; D - A = raw ranking under the old key.
- Pass = K1-K4 clear on s310-s312, then clear again on s313/s314 unchanged.

## 4. Dev lives (only after a pass)
Seeds 318-320 (no fit used them), 7 nights, pools: arms B0 (defaults), R (`--key S_all --key-bar slept --rank raw`),
R+S (R and `--sleep-downscale 1.0`, the adapter growth is a separate defect). Read per night: H, hub top-1 share, top
non-permanent strength vs median, whole transcripts. To the six registered seeds if R's night-7 H >= B0's + 2 on 2 of 3
seeds and no episode holds more than 40% of lived fetches on nights 5-7; a hippocampus-off lesion arm before any claim.

## 5. Host decision
None new. The S_all slept bar is the declared `KEY_BARS_SLEPT_BY_SIZE` scaffold, used as is. The host takes an argmax
over the brain's own key similarities and applies the strength fire test: arithmetic, no text read or matched. Ledger
labels are world-side, for measurement only. SCAFFOLDS.md: one row, no fit.

## 6. Cost
No extra forward pass per fetch; S_all reads the stored deltas (about 6.5 MiB per episode at 1.5B, already paid in the
day-2 gates). Screen: 4 arms x 5 homes x 60 turns, about 10 CPU-min per home.

## 7. Expected to fail
1. The hub may survive on s310/s312 if S_all also names it the raw winner for generic cues (S_all has never been read
   on a 7-night life): a K2 fail is a verdict on the key's selection; next is the decay-clock rule (no last_used reset
   on fetches strength alone decided) or salvaged M-eff.
2. D loses right fetches (3 of 4 are weighted-only winners): expected; only R counts.
3. H may not move: the slept cortex answers many items itself (4 of 6 right replies followed a wrong fetch; the S_all
   lesion arm equalled intact). K1 pass with flat K4 = selection fixed, replies not.
4. Open intrusions with no fetch come from the slept cortex and adapter growth; R does not touch them, M-scale does.
5. Fewer fires mean a rarely needed fact may never earn strength: accepted against the hub.
6. The 0.165 bar was fit on one night; night-7 drift may make it fire too much or too little: a K3 fail from that is a
   kill with a note, not a refit on s310-314.
A negative is a verdict on raw-sim ranking with S_all, not on whether the brain can select memories without a hub.
