# Capability: Episodic recall + Self and identity (whose words a memory holds)
# Record: brain/research/2026-11-23-biggest-defect-read.md (D2 recall side); brain/experiments/results/2026-11-14-lived-key-1.5b.md (585ad1be); 2026-11-14-speaker-contrast-live-1.5b.md and -2-1.5b.md (46da9f9e); brain/research/2026-11-23-stage2-test2-prereg.md (Amendment T1, G7/G8, life_costs.py)
# Expected-fail: yes, named. Self and identity "missing: self/other tag" and Episodic recall "separation of near-duplicate episodes" missing (PREREQUISITES.md); neither switch touches routing, which fetched the right episode only about half the time unslept and names another episode on about 3/4 of K-P fetches, so P1 (told) is expected to fall short even if P2 (echo) holds
# Recall voice on K-P: lived_key (M1) and speaker contrast (M2) in a slept 15-day DEV life (prereg, 2026-11-23, written before any M1/M2 life)

## 0. Record brief and hypothesis
- Record check: `tools/before_you_build.sh` refused (STALE CORPUS: published main bd85c19cc not in local main); the
  record was read directly: `git log --all --grep=lived_key / --grep=speaker_contrast` (9d56d2fe, 585ad1be, 46da9f9e)
  and the result notes above. Not re-derived; nothing is rebuilt (both switches exist, default OFF).
- Unslept 1.5B, six seeds, S0 key, gain 0.5: lived_key told 4.83 -> 5.17 (+0.33, bar 1: NO-GO), echoes 13 -> 4,
  answered-given-right-episode 37 -> 43/54. Speaker contrast mu 3.0: told +0.00 (NO-GO), echoes 13 -> 4, 37 -> 42/54;
  oracle LEAD (right 64 -> 84 of 108, echo 16 -> 3). Both: routing unchanged (right episode 36/72), routing the limit.
  Neither was ever run with sleep, in a life, or on a K-P cortex.
- K-P DEV (`lifeKP_s319-321`, AWS g5 fp16), counted by the read's method (§3, reproduced here exactly, no model):
  | seed | intact told right | hippocampus-off told right | intact echo | hippocampus-off echo |
  |---|---|---|---|---|
  | 319 | 58/80 | 54/80 | 15/88 | 20/184 |
  | 320 | 36/70 | 39/70 | 31/88 | 34/184 |
  | 321 | 29/73 | 45/73 | 25/88 | 23/184 |
  | pooled | 123/223 | 138/223 | 71/264 | 77/552 |
  Part of s321's gap is one night: night 15 intact 0/8 vs hippocampus-off 6/8, the intact probe collapsed onto the stock
  refusal ("I'm sorry, I don't know what you mean." 11 of its 20 rows); without night 15, s321 is 29/65 vs 39/65.
  Paired rows: fetch hurt 41, helped 26, best of both 164. K-P guards (same files): lang_guard KL n15 0.351/0.294/0.323;
  G7 wrong 26/26/17 of 30; G8a worst arm 0.087/0.130/0.137; leak on non-refusal rows n10+15 0/148, 3/150, 14/138.
- Hypothesis. Once K-P keeps the night healthy, the intact brain's loss against hippocampus-off is the reinstated
  memory in the user's first-person voice (mechanism 1 of the read). M1 stores the brain's own turn beside the user's
  (keyed by the telling, so routing is unchanged); M2 subtracts the own-voice reading of the telling from what is
  reinstated (night replay text unchanged). If either cuts intact echo and lifts intact told toward hippocampus-off,
  voice was the cause; if neither moves told, the wrong episode (mechanism 3) is.

## 1. Arms, seeds, device, command
| Arm | Switch (on top of `--keep-bank prior`) | `--brain-kw-extra` |
|---|---|---|
| K-P (comparison, banked) | none | none (`lifeKP_s319-321`, not re-run) |
| M1 | `Brain(episode="lived_key")` | `{"episode":"lived_key"}` |
| M2 | `Brain(speaker_contrast=3.0)` (the record's mu; gain stays "auto" = 0.5) | `{"speaker_contrast":3.0}` |
- DEV seeds 319, 320, 321 (the K-P seeds; DEV split). No TEST split: `life_kw` refuses `--split test/test2`.
- AWS g5.xlarge (A10G), CUDA fp16 (`--cuda-precision` default; `--device cuda` from the launcher's EXTRA), the device
  class of the banked K-P lives (hosts ip-172-31-41-224 / -91-140 / -66-19, device cuda). AWS fp16 lives are
  byte-deterministic per seed (Amendment T1 of the TEST v2 prereg), so M vs K-P differs only by the switch.
- 15 days. `build_life` is day-count independent ("day d does not depend on `days`"; the battery draws per seed and
  night), so a shorter life would be a prefix of K-P; but the bar reads nights 5-15, full probe nights 10/15,
  interviews 5/10/15 and `--pruned-a` day 11 (needs days >= 13), so 15 days, compared to `lifeKP` directly.
- Command = the exact `lifeKP` line (`_scratch/stage2/jobs_kp_<seed>.txt`) with the module swapped to
  `brain.experiments.life_kw` and `--brain-kw-extra` appended: `brain/research/recall_voice_launch/jobs_<M1|M2>_s<seed>.txt`
  (compact JSON, no spaces: `brain_batch_run.sh` word-splits args). One life per instance. `--resume` only on g5; a seed
  never moves device. The result JSON must carry `brain_kw_extra` equal to the arm's extra, else the life is UNDEFINED.
- Resume only via `brain.experiments.life_kw` with the same `--brain-kw-extra` (life_kw refuses a changed extra); a
  resumed or final JSON without `brain_kw_extra` is UNDEFINED.
- **Launch commit.** Launch only from a commit that contains `brain/experiments/life_kw.py`, its test
  (`brain/tests/test_life_kw.py`), this prereg and `brain/research/recall_voice_launch/` (Claude commits before launch;
  `launch.sh go` refuses a dirty `brain/` or a HEAD without them).
- **Launch env.** `bash brain/research/recall_voice_launch/launch.sh go M1` (11-24 UTC), `... go M2` (11-25 UTC): one
  g5.xlarge per life, state `research/queue/.aws_brain.rv<M><seed>`, with `AWS_BRAIN_AMI=ami-012ba162b9cd2729c` (the
  lifeKP AMI, recorded in `research/queue/.aws_brain.kp319/320/321.done-*`), `AWS_BRAIN_AMI_RECORD=1`,
  `AWS_BRAIN_ENV="BRAIN_CODE_COMMIT=<HEAD sha at launch>"`, dead-man `AWS_BRAIN_MAX_MIN` 600 (M1) / 510 (M2, as the test2 K-P), grace 15.
  Worst case (dead-man + grace + 15 min boot, 3 lives at $1.01/h) M1 $31.82, M2 $25.75 per UTC day; with 11-23's
  $5.11 spent either fits the $50 cap, both on one day ($57.58) does not, and `launch.sh` refuses it.
- **Code since lifeKP.** `git diff d777f71c9..HEAD -- brain/` (read 11-23): every change on the K-P DEV path is behind a
  non-default flag or record-only (brain.py: keep_bank "prior_u"/"prior_0", replay_priority "loss_a", a resume refusal
  on a changed keep_bank; cortex.py: `cuda_precision`, default fp16 = the lifeKP strategy; sleep.py: a (text,
  positions) keep item only under prior_u, the str path unchanged; life.py/life_* eval: flags and read-outs);
  `brain/hippocampus.py` unchanged. So with both switches off a life on HEAD is the lifeKP life.

## 2. Count method (the read's, written exactly; host-only, `brain.experiments.strict_scorer.is_echo`)
Rows come from `res["probes"]`, item `e = res["ledger"][row["item"]]` with `e["telling"]` truthy, night 5..15.
- **Told right (P1, guard H).** Probes with `arm` in {intact, hippocampus-off}, `line == "trunk"`, no `tag`; rows with
  `ok` not None (drops the 41 near-dup rows K-P has with `ok` None). Right = `ok` True. Pooled over seeds; denominator
  reported (K-P 223). Paired by (seed, night, row index, item, question): hurt = off right and intact wrong; helped =
  the reverse; best = either right.
- **Intact echo (P2).** Probes `arm == "intact"`, `line == "trunk"`, no `tag`, every told-item row (ok None included;
  K-P 264). Echo = `any(is_echo(row["brain"], k, e["telling"]) for k in e["keys"])`.
- **Hippocampus-off echo (report).** Every `arm == "hippocampus-off"` probe, nights 5-15, all lines and tags (trunk,
  ab-recall, prunedA; K-P 552), same rule.

## 3. Bars (each arm separately against K-P; 3 seeds = DEV, never a claim)
- **Validity.** A seed whose both-off rows differ from lifeKP's (`life_costs` G11 setup check, on) is UNDEFINED (both-off
  is the same pretrained cortex with no hippocampus for every arm, so a difference is a setup mismatch).
- **P1 told, intact:** pooled intact right >= 138/223 (K-P's hippocampus-off level: K-P intact + 15), and >= K-P intact
  on >= 2/3 seeds. (Same denominator expected; if it differs, compare rates and report both.)
- **P2 echo, intact:** pooled intact echo <= 47/264 (K-P 71 minus one third) and <= K-P on >= 2/3 seeds.
- **Guards** (each must hold; read with `brain/eval/life_costs.py --dev --arm <M lives> --b <lifeKP lives>`,
  `life_leak`, `life_bars`, nights[].lang_guard):
  - H recall: pooled hippocampus-off told right >= 115/223 (K-P 138 minus 1/6).
  - Leak (G4 form, `life_costs.leak_nonrefusal`, nights 10+15): pooled rate <= K-P pooled + 0.03.
  - Language: lang_guard KL n15 <= 1.25x K-P's on each seed.
  - G7 self-name wrong rate: pooled <= K-P pooled (69/90) + 0.05. G8a refusal: worst arm per seed <= K-P's + 0.03.
  - Collapse (`life_bars.language`, top reply share >= 0.4): seeds-ever per probe arm <= K-P's.
  - Stuck lived-day runs (same reply >= 5 turns in a row; `res["lived"]`, every line and day, consecutive rows of one
    lived day, exact `brain` text after strip): pooled <= K-P's count by the same rule + 1. K-P (computed 11-23 from
    `_scratch/stage2/read/lifeKP_s319-321.json`): 0 runs (longest run 4 turns, s320), so the bar is <= 1.
  - Hippocampus-off echo (§2 rule): pooled <= 77/552 + 10% (<= 84/552).
- Report only: fetch hurt/helped split by fetched / right_episode (K-P hurt 41 = 8 right episode, 28 wrong, 5 no
  fetch; helped 26 = 6 / 16 / 4), best of both, hippocampus-off echo, stuck lived-day replies (same reply >= 5 turns
  in a row, the D4 read), G8b, G10, whole transcripts of one seed per arm (read before the verdict). M2 diagnostic: if
  night replay ignores replies, M2's hippocampus-off and both-off rows equal K-P's; reported, not a bar.
- M1's `nights[].replay_share` class table is INVALID and not read: `life.replay_share` strips only "User: " from the
  stored text, and M1 stores the brain's own turn beside the user's, so told episodes fall to "other".

## 4. Decisions
- **KEEP** (an arm): P1, P2 and every guard hold. Consequence: a candidate for TEST v3 only, never v2 (v2 is frozen);
  next is six seeds on a fresh split, not a default flip.
- **NARROWED:** P2 holds with every guard, P1 fails (voice fixed, routing still the limit, as the record predicts), or
  P1 and P2 hold with one guard missed by <= its own margin again. Recorded as mechanism evidence; next lever = routing
  (M3 margin gain or an S_all key), not a 6th perspective lever.
- **KILLED:** P2 fails, or pooled intact told < K-P's 123, or the language or collapse guard fails.
- Mechanism read: M2 moves P2 and M1 not -> reinstated voice; M1 moves and M2 not -> the night's lived content;
  neither -> wrong episode (mechanism 3). A negative is a verdict on these switches, not on the capability.

## 5. Prerequisites, host decisions, levers, timebox, cost
- Prerequisites: Self and identity (self/other tag) MISSING, Theory of mind NOT-READY, near-duplicate separation
  missing: expected failure named in the header (P1). Scale is not the limit here (the unslept right-episode brain
  answers).
- Host decision: none. Both switches are existing brain-native Brain kwargs; `life_kw` only passes kwargs (no reply
  text in the host, no retrieval of text, no grammar). No new scaffold row. No frozen v2 file edited.
- Lever count on D2 (echo/perspective): 3 on record (lived, lived_key, speaker contrast; none passed its bar, all
  unslept); this is 4 and 5 (2 and 3 re-run slept, in a life, on K-P). Hub round 11-21: 15 selection levers, none fixed
  selection. If both are KILLED or NARROWED, no further perspective lever before a routing method.
- Timebox: launch 11-24 (M1) and 11-25 (M2) UTC, read by 11-26; hard stop 11-27 00:00 UTC (unfinished = UNDEFINED,
  instances terminated). Nothing of this on AWS 12-04..12-06 (TEST v2 days).
- Cost: K-P lives took 5.9-6.1 h of nights + 0.7 h probes; about 7.5 g5.xlarge hours with setup at ~$1/h = ~$8 per
  life, ~$48 for six. Split 3 lives per UTC day (~$24/day) to stay inside the $50/day cap with other work. Read: host
  only, no model, $0.
