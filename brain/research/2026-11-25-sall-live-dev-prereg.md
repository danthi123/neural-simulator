# Pre-registration: a live DEV K-P life with the S_all key and its slept bar (arm S), judged on replies

Written 2026-11-25 before any run. Capability: episodic recall (memory selection), at reply level. DEV, 3 seeds: never a
claim.

## 1. Record brief
- **Ceiling (11-25):** on the K-P store the all-layer key ranks the right episode first on 168 of 217 told rows (S0 90).
- **Screen 2 (11-25):** offline, on S0-shaped stores, S_all with its slept bar fetched right on 49 (A 44) and fetched
  nothing on 128 told rows; the right episodes it ranked first had weight median 0.36. Two candidates, neither testable
  offline: the stores' strengths were shaped by S0's fetches; the slept bar was fit on 2-day-old episodes.
- **Live record of this switch:** slept gate (11-18, 2 days) reworded fetches 1/18 -> 14/18, replies +4/36;
  key default (11-14) S_all ahead by about one fact, kept S0 by 0.17. A fetch gain may not move replies.
- **The defect this targets:** on K-P, memory hurts. Told recall is 123/223 with the hippocampus and 138/223 without it
  (recall-voice read 11-24; hurt 41, helped 26).
- **Prerequisites:** near-duplicate separation and a paraphrase-robust cue are "missing" (`brain/PREREQUISITES.md`).
  Expected risk: told episodes S_all does not fetch decay further and the brain answers less; "admin" is not ranked
  first by S_all.
- **Host decision:** none. `--key S_all --key-bar slept` is an existing switch with a declared bar row
  (`brain/SCAFFOLDS.md`; `Brain.KEY_BARS_SLEPT_BY_SIZE`). Nothing is fitted.

## 2. Lives
- Arm S = the lifeKP job line exactly (`_scratch/stage2/jobs_kp_*.txt` via `brain/research/recall_voice_launch/
  jobs_KR_*.txt`: 1.5B, 15 days, weekend, DEV split, full probes nights 1/5/10/15, frozen probe, span, transfer,
  pruned-a, cue ladder, interview, lang guard, `--keep-bank prior`) plus `--key S_all --key-bar slept` and
  `--night-snapshots` (record-only). Seeds 319 / 320 / 321, paired with lifeKP on the same seeds.
- AWS g5.xlarge fp16, the lifeKP AMI, one life per instance, BRAIN_CODE_COMMIT = HEAD, dead-man 600 min.

## 3. Bars (pooled over 3 seeds against lifeKP; the recall-voice §2 count method, `count.py` of the 11-24 read)
- **Validity:** a seed whose both-off rows differ from lifeKP's (`life_costs` G11) is UNDEFINED.
- **P1 told, intact:** pooled intact right >= 138/223 (K-P intact + 15, = K-P's hippocampus-off), and >= K-P intact on
  >= 2/3 seeds.
- **P2 memory helps:** pooled intact right >= the same lives' pooled hippocampus-off right, and helped >= hurt.
- **Guards** (each must hold): H recall, pooled hippocampus-off told >= 115/223; intact echo <= 78/264 (K-P 71 + 10%);
  hippocampus-off echo <= 84/552; leak (G4 form, nights 10 + 15) <= K-P + 0.03; lang_guard KL n15 <= 1.25x K-P per
  seed; G7 self-name wrong <= K-P + 0.05; G8a refusal, worst arm per seed <= K-P + 0.03; collapse (top reply share >=
  0.4) seeds-ever per arm <= K-P's; stuck lived-day runs <= 1.
- **Report only:** frozen-arm fetch right / wrong / none per night; per night, the strength, uses and last_used of each
  told item's right episode vs the night's three strongest episodes (from `--night-snapshots` episodes.json); the four
  H items; never-told false recall; whole transcripts of one seed (read before the verdict).

## 4. Decisions
- **KEEP:** P1, P2 and every guard hold. A candidate for TEST v3 (six seeds, fresh split), never a v2 default.
- **NARROWED:** P2 holds with every guard and P1 fails (memory stops hurting but recall does not rise enough).
- **KILLED:** P2 fails, or pooled intact told < K-P's 123, or the language or collapse guard fails.
- A negative is a verdict on this switch on this life, not on the capability.

## 5. Cost and timebox
3 lives x (600 + 30) min x $1.01/h, at most about $32 (11-25 spend $2.93 before launch; $50 cap). Read by 11-27.
