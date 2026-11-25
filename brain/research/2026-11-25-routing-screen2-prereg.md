# Pre-registration: routing screen 2, the S_all key (slept bar) through the live competition, on the KR snapshots

Written 2026-11-25 before any run. Capability: episodic recall (memory selection). Fetch level only, offline.

## 1. Record brief
- **Ceiling (11-25, `results/2026-11-25-routing-ceiling-read.md`):** on the 217 KR told rows the all-layer key
  S_all_live ranks the right episode first on 168 (S0: 90), and on 89 of 89 non-repeated rows (S0: 56). Rank only.
- **A's own competition loses much of S0's raw order:** of S0's 90 raw-first rows, A fetches right on 24, wrong on 45,
  nothing on 21 (net 44). What the weighted, gated competition does to S_all is the open question.
- **The arm is an existing switch:** `Brain(key="S_all", key_bar="slept")` = `KeyedHippocampus(key="S_all",
  familiarity=0.165, margin=0.02)` (`brain/brain.py` `KEY_BARS_SLEPT_BY_SIZE`, fit on seeds 42/43/44, slept DEV bank,
  `results/2026-11-18-sall-slept-fit-1.5b.md`). The statement bar stays the class 0.45. **Nothing is fitted here**, and
  no KR row sets anything.
- **Record of this key live:** slept gate 1/18 -> 14/18 reworded fetches, replies +4/36, intrusion bar failed
  (`results/2026-11-18-sall-slept-gate-1.5b.md`); R-raw with S_all KILLED (`results/2026-11-20-rank-raw-screen-1.5b.md`,
  false fires rose). A PASS here is not a reply result.
- **Prerequisites:** near-duplicate separation and paraphrase-robust cue "missing" (`brain/PREREQUISITES.md`);
  expected risk: false fires on untold near-duplicate cues (K4) and the "admin" item, which no key ranks first.
- **Host decision:** none (existing switch, existing declared bar row in `brain/SCAFFOLDS.md`).

## 2. Arms and instrument
- Population, nights and void rule exactly as the 11-24 screen (`brain/research/2026-11-24-routing-screen-prereg.md`
  sections 2-3): KR snapshots s319-321, nights 5-15, intact-frozen battery rows; arm A must reproduce the banked fetch
  idx and recalled text on every row of a night, else the night is void; a seed with more than 2 void nights is void.
- **A:** `routing_replay.decide` arm A (unchanged).
- **K:** a `KeyedHippocampus(key="S_all", familiarity=0.165, margin=0.02)` holding the night's episodes (with the final
  trunk's deltas and route deltas, keys checked against keys.pt), FREEZE on: `similarities` (the live class's own
  code), weighted by `tanh(effective_strength)` as `scores` does, then `_compete(score, question, now, sims=sims)`.
  `question` = `cue.endswith("?")`, as in `Brain.hear`.
- Pool40/41/42, CPU fp32, under `tools/memcap.sh`, the code at a pushed commit.

## 3. Bars (KILL first; arm K)
- K1: pooled right-episode fetches on told rows `>= max(ceil(1.25 x A), A + 10)` (A = 44 on these rows: need 55).
- K2: right rises on >= 2 of 3 seeds and falls by more than 1 on none.
- K3: at most 1 row per seed turns from A's right episode into a wrong episode.
- K4: fires on fire rows (open, greet, never-told, untold near-duplicate) at most A's + 1 per seed.
- **K5 (new; the D lesson):** told no-fetch at most A's + 5 per seed (A: 14 / 8 / 15).
- PASS = none of K1-K5. It earns one live DEV K-P life with `key="S_all", key_bar="slept"` (s319-321, AWS g5) with a
  reply-level bar, nothing more. Six seeds before any claim.
- Reported, not barred: non-H rows; the four H items; identity rows (name / nova) fetch; per night; lost-to-no-fetch;
  K's raw-first right count vs its fetched right count.

## 4. Timebox and cost
About 10 min per seed on pool CPU, $0. Read by end of 11-26.

## Review (sonnet, 11-25, before data): OK_WITH_FIXES, applied
- Arm K is built by the brain's own code (`key_bar="slept"`, `Brain._make_hippocampus("S_all")`) on the snapshot Brain,
  and refuses a cortex that is not 1.5B. Per-night counts and the four H items are reported. Test: K equals the live
  non-frozen recall's choice, and reads route deltas.
- **Caveats for the read (not bars):** (1) K has no banked tripwire: its CPU fp32 cue read meets fp16 stored deltas,
  so borderline K rows are unvalidated (`K_sim_raw_max` logged). (2) The store's strengths and recency were shaped by
  S0's own competition during the KR life (its hubs were boosted by S0 fetches), so K is "S_all on an S0-shaped store",
  not what a live S_all life would build. Either way only a live DEV life settles it.
