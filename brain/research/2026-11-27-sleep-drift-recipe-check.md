# Sleep-drift wall: recipe check against the Lift-Gate-Glue persistence literature, and one offline screen (2026-11-27)

Status: a design note, NOT a registered prereg. Nothing was run. Revised 11-27 after a design check (changes listed in
section 6); the screen rule in section 4 is DRAFT pending pre-data review (sonnet, per the prereg habit). Written read-only against `brain/sleep.py`,
`brain/brain.py` (Brain.sleep, `_keep_bank`, `_night_episodes`), `brain/research/2026-11-23-stage2-test2-prereg.md`,
`results/2026-11-22-kp-dev-read-1.5b.md`, `results/2026-11-22-kp-gate-replications-1.5b.md`, and
`Lift-Gate-Glue/docs/SN4_PERSISTENCE_PLAN_2026-11-26.md` lines 40-125 (read as data). Record check run
(`before_you_build.sh "sleep drift keep bank CE replay mixing"`): no prior REMIX-style arm, no placebo night, and no
CE-on-keep arm anywhere in `brain/research/`, `brain/experiments/results/`, PREREQUISITES or SCAFFOLDS. This is unmeasured,
not refuted.

## 1. What our sleep actually does (read from the code)

Per step of `brain/sleep.py::sleep` (1.5B: 480 steps, lr 5e-4, kl_weight 3.0, rank-8 adapters, Adam):
1. one episode drawn in proportion to strength, **cross-entropy** on its whole text (`replay`);
2. one keep text drawn from the generic set, **KL(frozen cortex || student)** over the full next-token distribution at
   every position (`keep`, `F.kl_div(log_softmax(student), ref, log_target=True)`; no CE on that text);
3. loss = replay + kl_weight x keep (+ the near-miss guard KL, off in the K-P arms).

Two facts the literature comparison depends on:
- **The night replays the WHOLE store, not just the day's episodes** (`Brain._night_episodes` returns every stored
  episode; weights = effective strength). So "replay 1-10% of old facts" (Chen et al.) is not our situation: we already
  do full-store replay each night (the LocFT-BF pattern, whose cost grows with N). Old-fact CE is present; what thins is
  passes per episode as the store grows (480 steps over a store that grows each night).
- **K-P keep bank** (`keep_bank="prior"`): 512 texts sampled once per life from the frozen cortex at temperature 1 from the
  "User:" frame (64 tokens each); each night a fresh 64-text subset (`random.Random(sleep_seed + sleeps + 104729).sample`, where `sleeps` is the number of
nights already slept, i.e. night n uses sleeps = n-1), so each subset
  text is visited about 7.5 times in 480 steps. Mix ratio is 1 episode : 1 keep text per step.
- **One generator draws both streams.** In `brain/sleep.py` a single `random.Random(seed)` (seed = sleep_seed + sleeps) draws the
  episode (`rng.choices`) and then the keep index (`rng.randrange(len(generic))`) every step, and `sleep_seed + sleeps` also
  seeds the keep-subset draw in `Brain.sleep`. So (i) changing the replay seed alone also changes which keep texts are drawn
  at each step; (ii) changing the keep set size or keeps per step can re-time the episode draws, because `randrange(n)`
  rejects words >= n and consumes extra words when n is not a power of two (64 vs 256 keeps the episode stream, 12 vs 40 does
  not: tested), and a second keep draw per step always does. The screen decouples them (section 4).

## 2. Does our sleep already do REMIX-style mixing? Answer: half yes, and the missing half is probably not where the gain is

REMIX (Chen et al. 2024, arXiv 2411.07175 as summarised in the SN4 plan, 13.5% -> 53.2%): generic text is mixed into BOTH
training stages under the SAME next-token CE loss as the facts.

| REMIX property | Ours | Verdict |
|---|---|---|
| generic text interleaved in the replay stream, every stage | yes, 1:1 per step, every night (K-P) | yes |
| generic text carries a CE term | **no**, KL to the frozen cortex only | no, literally |
| generic text is outside-the-model text | no, the bank is the cortex's own prior samples | different (see below) |

Why the CE-vs-KL gap is probably small here, and where it is not:
- On the cortex's own samples, E[grad CE] equals grad KL(base || student) at each prefix. Our keep term is the same
  gradient with the next-token sampling noise removed (full-distribution target), so it is an equal-or-stronger anchor on
  those texts than CE would be. A literal CE-on-bank arm is therefore predicted ~NEUTRAL (extra noise), and I would
  register it as a prediction, not a hope. Two departures from exact equivalence, both small: the frame "User:" is forced,
  not sampled, so on those positions CE and KL differ; and each bank text was decoded and is re-encoded, so its tokens are
  not exactly the sampled ones. The CE arm also adds a pull toward the base on self text, which is the same direction as
  the keep term (it is a second keep term, as the placebo arm A5s is too: see 3.1).
- REMIX's gain comes from CE on text the model did NOT generate (random or generic text). That adds a data-side anchor the
  frozen-prior bank cannot give: the bank only protects the prior's own mode. Our K-P DEV costs (self-name overwrite
  26/26/17 of 30, refusal collapse nights, "I've seen it" phrasings) are exactly drift OFF the prior's mode in lived-day
  language, which a bank sampled from the "User:" frame covers thinly. This is an argument for widening what the keep term
  covers (more coverage, bigger subset, a lived-day-style keep), not for swapping KL for CE.
- Literature agreement with our own data: SN4 cites WISE (FT-L, which already has a KL term, Loc 0.03 at 1,000 edits) and
  "forgetting tracks KL to base" (RL's Razor). Our numbers say the same two things: (a) keep-KL works but is not a stop:
  K-P still grows n15 KL 3.5-5.3x its own night 1 (bar for "fixed" is <= 2x), against B 13.6-22.7x; (b) the drift tracks
  adapter norm (K-P 358-370 at n15 vs B 660-693) and KL (K-P 0.32 vs B 2.72). One K-P night applied to a B n15 life
  repairs 2.9 -> 0.8 (gate replications), so the keep term both holds and repairs; the unrepaired part is what accumulates.
- A real difference from the REMIX/WISE setting: they edit with single facts; we replay a growing store through a rank-8
  adapter, so the likely driver is cumulative optimiser motion plus content CE, not any one night. That is what the
  placebo arm below separates.

## 3. What the evidence supports, cheaply, and in brain terms (ranked)

All are constants or one extra term inside the existing declared sleep scaffold (SCAFFOLDS.md sleep row; CLS interleaving
and the keep bank are already declared). None adds host logic about world, body or teacher.

1. **Placebo nights (diagnostic, highest value, free of any claim).** Replace the night's episode texts by other texts, same
   count, same strengths, same CE path, same keep stream. Two placebos, because each has a confound the other lacks:
   - **A5s (self-text placebo): unused bank texts** (the cortex's own samples, not in that night's keep subset). CE on the
     cortex's own samples pulls the adapters BACK toward base after drift (E[grad CE] = grad KL(base || student), the very
     argument of section 2), so A5s carries a second keep term and its drift D_self is biased DOWN: it is a lower bound on
     content-independent optimiser motion, not an estimate of it. A5s can therefore support "motion is at least this much"
     but can never support "motion is small / drift is content-driven".
   - **A5f (foreign-store placebo): another life's episodes** (the next KR seed's store of the same night, any text equal to
     one of this seed's own stored texts removed, pool cycled if short). Non-self text in the same lived register, so CE on it
     pulls AWAY from base like the real facts do. It removes the self-text pull; what it cannot separate is optimiser motion
     from "any lived-register CE" (register-general drift versus this life's specific facts). Its D_foreign is the upper-side
     read of the two.
   Together they bracket: D_self <= (motion) and D_foreign ~ (motion + register). The earlier "0.6x / 0.25x of D_real" rule on
   a single self-text placebo was invalid as written (the 0.25x branch is exactly where the bias would manufacture a false
   "content-driven"); the replacement rule is in section 4. Nobody on our record has this number.
2. **Keep coverage and dose** (cheap, the strongest-supported direction): subset 64 -> 256 per night (a text revisited
   about 2 times, not 7.5; the 512-text bank already exists; nested, the shipped 64 stay in); two keep texts per step with the
   keep term MEAN-reduced (a gradient-variance lever, dose fixed: a SUM over two texts would equal kl_weight x 2 in
   expectation, i.e. arm A4, and would not be a separate lever); kl_weight 3 -> 6 as a dose-response (WISE, as summarised in
   SN4 and not re-read here, says KL alone is not enough, so a monotone-but-saturating curve is the expected read).
3. **CE on keep text, literal REMIX** (predicted neutral, cheap to include since the stream exists): add
   `ce_keep_w x CE(keep text)` in the same step.
4. **Not recommended now:** replaying a larger fraction of old facts (already full-store), KL on the episode inputs (the
   RL's Razor form; already approximated by the near-miss / rehearsal guards, 9 voice levers without a pass), routers or
   retrieval (excluded), and any new bank texts written by the teacher as keep material (a scaffold we would have to
   declare; only worth it if arms 2-3 both fail).

## 4. The screen: `mix-night-screen` (offline, pool CPU fp32, $0 cloud). DRAFT rule pending pre-data review

**Question.** Does anything in the REMIX/keep-coverage family reduce multi-night language drift on the K-P slept store, and
how much of the drift is content-independent?

**Population and what each night sleeps on (the chain store, stated exactly).** KR night snapshots (`life_kw
--night-snapshots`; s319/s320/s321 = lifeKP, 0 mismatches, as in the routing ceiling prereg), taken after each night's
sleep. The KR lives ran on AWS (fp16 / TF32); the screen runs on pool CPU fp32. Start = snapshot n12: its adapters (the
life's) and the shared keep bank. Night k in 13, 14, 15 is slept by the real `Brain.sleep` on a Brain opened with
`routing_replay.open_snapshot` on a snapshot-shaped directory built per arm and night:
- **store** = snapshot n<k>'s `episodes.json`: the text and the scalars (strength, uses, last_used, created) of the life's store
  at the end of day k, i.e. the store the life itself slept on night k; **clock** = that snapshot's `meta.last_sleep`;
  **sleep counter** = k-1 (so the seed rule `sleep_seed + sleeps` and the keep-subset draw are the life's own);
- **adapters** = the ARM's own carried adapters (n12 for k = 13, then the arm's output of the previous night). NOT the
  life's n<k> adapters.
- So the arm chains differ from the life's trajectory in the adapters only; the episodes of days 13-15 are the life's, with
  strengths, uses and last_used produced by the life's own days (its replies, recalls and gates under ITS adapters), not
  the arm's. The sleep reads nothing else of an episode (no keys, no deltas), so the keys of the life's adapter state do not
  enter; the strengths do, identically in every arm, and that is the declared caveat: an arm whose adapters drift less would
  in a real life also have lived slightly different days. The screen answers "given these stores", not "in a life".
- Arm A0L (below) replays the shipped loop exactly from the same start, so A0L at n13 differs from the life's n13 only by
  device. Chained arms from n14 on differ from the life by their own adapters, by design.

**Code (switch-free).** One new script `brain/experiments/mix_night_screen.py`; `brain/sleep.py`, `brain/brain.py` and every
default are untouched. The script monkeypatches `brain.brain.replay_into_cortex` and `Brain._night_episodes` inside a context
manager (restored on exit) with `mix_sleep`: a copy of the sleep loop with (a) `loop="legacy"`, an exact replica of
`brain/sleep.py` (unit test: equal losses and adapters to the shipped function on a fake adapter forward), and (b)
`loop="split"`, the decoupled loop below, plus `ce_keep_w`, `keep_per_step` and a nested larger keep set. Nothing runs
without `POOL_NODE=1`; every unit records host, commit and device; every unit is written atomically (`.tmp` + replace) and a
rerun skips finished units; carried adapters are saved per night and the previous night's file removed only after the new
unit is written. Tests: `brain/tests/test_mix_night_screen.py` (exact verdicts on synthetic records; the patched real
`Brain.sleep` on the stub cortex).

**Decoupled streams (the check's point 3).** In the split loop the episode draw uses `Random(seed + ep_add)` and the keep
index uses its own `Random(seed + 31337 + keep_add)`; the keep SET is the shipped 64-text subset (unchanged `sleep_seed + sleeps
+ 104729` draw), and A3's 256 is that same 64 plus 192 extras from a third generator (nested). Consequences, all tested:
no keep option (CE, two texts per step, larger set) moves the episode sequence, and no episode option moves the keep-index
sequence. The noise arms change ONE stream each, so the floor says what each source of noise contributes.

**Arms** (3 DEV seeds x chained nights; 1.5B, 480 steps, lr 5e-4, rank 8, kl_weight 3 unless stated):

| arm | nights | change | why / predicted |
|---|---|---|---|
| A0L | 13 | the shipped loop, shared generator (legacy) | instrument check vs the life (K-sane, adapter distance) |
| A0 | 13-15 | split streams, shipped subset | the baseline of every contrast |
| A0r | 13 | identical rerun of A0 | determinism check (voids the screen if it differs) |
| A0e | 13-15 | A0 with the episode stream seed +1000 only | noise: episode draw |
| A0k | 13-15 | A0 with the keep-index stream seed +1000 only (same subset) | noise: keep draw |
| A1 | 13-15 | `+ 1.0 x CE(keep text)`, same keep text as the KL | literal REMIX; predicted neutral |
| A2 | 13-15 | 2 keep texts per step, KL and CE MEAN-reduced | gradient-variance lever, dose fixed |
| A3 | 13-15 | nightly keep set 64 -> 256 (nested) | coverage; predicted the clearest gain |
| A4 | 13-15 | kl_weight 3 -> 6 | dose-response; expected to saturate and cost replayed-episode NLL |
| A5s | 13-15 | PLACEBO: episode texts -> unused bank texts (own samples) | motion lower bound (biased down: section 3.1) |
| A5f | 13-15 | PLACEBO: episode texts -> another seed's store of that night | non-self text; motion + register |

Candidates compared with A0 are A1-A4. The subset-draw noise (a different 64) is not a separate arm: A3 contains the shipped 64.
87 night-sleeps (9 three-night arms + A0L + A0r, x 3 seeds), plus one shared n12 reading per seed.

**Compute (UNMEASURED; `mix_night_screen.py estimate`).** At a placeholder 25 min per 1.5B CPU night and 10 min per reading set:
about 51 node-hours, about 17 h wall on pool40/41/42 under `tools/memcap.sh 11` (floor: A2 pays a second keep forward per step
and A3 about 4x the reference forwards; neither is modelled). **Step 0 (30-60 min): `run_node.sh 319 A0r` (one night + reading)
and time it; if a night + reading exceeds 45 min, drop to nights 14-15 and drop A4 and say so before launch.** One process per
(arm, seed), `run_node.sh <seed> [arms]`, seeds on three nodes; the screen runs on idle pool nodes only, after test2.

**Readings after every chained night** (read-only, no RNG draw, `plastic.enabled` restored; also once at n12 per seed):
1. **Held-out drift:** `brain.eval.life_langguard.measure` on the frozen HELDOUT set (KL, NLL_on), the same instrument as every
   banked number.
2. **Off-frame drift:** the same `measure` on 12 plain-prose passages with no "User:" frame, written once by the teacher and
   frozen before the first run: a JSON list whose sha256 is passed on the command line and checked against the file at run
   time (the run refuses on a mismatch), and recorded in every unit. K-P's bank is sampled from the "User:" frame, so only this
   reading shows whether keep coverage transfers. The set is not yet written (open item before launch).
3. **Retention:** per-token NLL of up to 20 store texts per age stratum (own = age < 1 clock day, recent 1-3, mid 3-5, old >= 5;
   ages in simulated clock days, which include the 48 h weekend jumps, declared), the same texts in every arm of a seed (fixed
   seed), adapters on, no hippocampus. The texts read are the REAL store texts, also in the placebo arms.
4. **Stock-reply collapse:** greedy replies (40 tokens) to 24 fixed ordinary prompts held in the script (a test checks none
   occurs in `life_bank*.py` or the held-out set), scored for the refusal pattern ("I'm sorry", "I don't know what you mean") and
   for "my name is".
5. Adapter norm; relative L2 distance of A0L's n13 adapters to the life's n13 snapshot adapters (report only).

**Draft pass rule (per candidate X in A1-A4 against A0 at night 15, all per seed; DRAFT, frozen only after the pre-data review).**
- **Noise floor per seed:** noise_s = max(|KL(A0) - KL(A0e)|, |KL(A0) - KL(A0k)|) at n15, held-out KL.
- **P1 drift:** KL(X) < KL(A0) with gap > 2 x noise_s on every seed, AND mean KL(X) <= 0.75 x mean KL(A0). Chain growth
  KL(n15)/KL(n12) is reported beside it; "fixed" is NOT claimed here (that bar stays n15 <= 2x n1 in a full life).
- **P1b transfer:** off-frame KL(X) <= 0.85 x A0 on >= 2/3 seeds.
- **P2 retention:** replay NLL of X <= A0 + 0.10 nats in the own, recent and old strata on 3/3 seeds.
- **P3 no new collapse:** refusal share and "my name is" share each <= A0 + 0.05 absolute on 3/3 seeds.
- **PASS** = P1 and P1b and P2 and P3. **KILL** = P1's per-seed clause fails on >= 2 seeds. Otherwise INCONCLUSIVE. Missing
  units, mixed commits or off-frame hashes, or an A0r that differs from A0 at n13 by more than 1e-3 relative KL: INCONCLUSIVE.
  A PASS means "enters a six-seed fresh-split registered test" (42/43/44/100/101/102; the three KR seeds are DEV seeds); it
  flips no default. Seeds are reported individually; the mean in P1 is the only pooled number and is a screening gate, not a claim.

**Placebo read rule (diagnostic, not pass/fail; DRAFT).** Per seed D_real = KL(A0, n15) - KL(n12), D_self = A5s - n12,
D_foreign = A5f - n12. A seed is READABLE only if D_real > 2 x noise_s; fewer than 2 readable seeds = UNREAD.
- **MOTION-DOMINANT** if D_self >= 0.6 x D_real on >= 2 readable seeds. This direction is conservative: the self-text placebo is
  biased down, so reaching 0.6x despite the bias is real evidence that drift is mostly content-independent; next levers are lr,
  steps, the norm budget (M-scale, SHY), and A1-A4 are read as secondary.
- **CONTENT-DRIVEN** only if BOTH D_self and D_foreign <= 0.25 x D_real on >= 2 readable seeds (a low D_self alone proves
  nothing: it is what the bias produces). Then A1-A4 and keep-on-episode-inputs are the family; rank by P1.
- Otherwise **BRACKETED**: report the per-seed ratios; run the six-seed arm that passed P1-P3 together with A5f.
- A5 retention is never a pass (it replays no facts); A5s and A5f are not candidates.

**Anti-cheat and validity.**
- **K-sane (instrument):** A0L and A0 held-out KL at n13 within 25% of the life's banked `lang_guard.kl` at n13 (read from the
  life JSON by the scorer); a miss is a device caveat (CPU fp32 vs AWS fp16/TF32; the 11-22 note says the device effect is a
  growth rate, not an offset), not a void, because the contrasts are arm-to-arm on one device.
- Every arm shares keep_bank.json, the same night snapshots and the same measurement code.
- The held-out set and the off-frame set are never used to fit anything.
- Numbers are read from the unit JSONs by `mix_night_screen.py score`, not retyped; a fresh-context skeptic (sonnet) and a
  pre-data review (sonnet) before the first run, per the land-result and prereg habits.
- Report seeds individually.

## 5. What this does and does not tell us

- It can say whether a cheap sleep-recipe change cuts multi-night language drift on the existing K-P store, and bracket how much of
  the remaining drift is the optimiser rather than the facts (a lower bound and a register-confounded upper side, not a point). It cannot say anything about test2 (running; frozen), about
  voice, or about capability: a negative is a verdict on these four recipe changes at this scale, not on consolidation
  without drift.
- Biological reading (PREREQUISITES): CLS interleaving is present; synaptic downscaling is the M-scale / SHY line; a
  consolidation scheme that uses outside-model "ordinary experience" for the keep term is the biologically honest form of
  REMIX (waking life interleaves with replay), and the cortex's own lived days are the declared ORDINARY scaffold. If A3
  passes, the follow-up registered design is a keep bank of lived-day ordinary turns (not the cortex's samples), not a
  teacher-written set.
- Timing: test2 launches/reads are ahead of this in the queue (read by 12-10), so the screen runs on idle pool nodes only;
  it does not use AWS and costs no Claude beyond the two review agents (tier sonnet).

## 6. Changes from the first draft of this note (design check, 11-27)

1. Placebo: the self-text arm was biased toward base (a second keep term); added a non-self placebo (A5f), named the confound,
   replaced the 0.6x/0.25x read rule (the 0.25x branch could not be valid on A5s alone).
2. Chain store: stated which snapshot supplies each night's store, clock and sleep counter, and that adapters are the arm's own.
3. Noise floor: A0' (replay seed +1000) also moved the keep-index stream, and the 64 -> 256 subset and two-keep arms re-timed the
   shared generator. Streams decoupled (episode, keep index, extras), noise split into A0e / A0k, the shipped subset kept.
4. A2 as first written (two keep KLs, summed) equals A4 in expectation; now MEAN-reduced so it is a variance lever.
5. The seed rule is `sleep_seed + sleeps` (nights already slept), not "+ night".
6. "Additive keyword arguments on `sleep()` / `Brain.sleep`" replaced by the switch-free harness (no edit to brain/).
7. "Pooled over seeds" in the placebo rule contradicted "no cross-seed pooling": per seed now.
8. The note said A0 differs from the life only by device; it also differs by the decoupled streams, so A0L (exact legacy loop)
   carries the instrument check and A0r the determinism check.
9. The off-frame set needs a registered hash and a run-time refusal on mismatch; the scorer is `mix_night_screen.py score`.
10. CE-vs-KL equivalence caveats (forced frame, decode/re-encode) added in section 2.
11. Compute recounted: 87 night-sleeps, not 63; the wall estimate moves from 12-13 h to about 17 h (placeholder inputs).
Open before launch: write and hash the 12 off-frame passages; time step 0; the pre-data review of the DRAFT rule.
