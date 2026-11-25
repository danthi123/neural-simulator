# Pre-registration: routing ceiling on the K-P slept store (which key puts the right episode first?)

Written 2026-11-25 before any run. Capability: episodic recall (memory selection). A MEASUREMENT, not a method: no
brain default changes and no arm earns a claim. It decides which routing method is worth building next.

## 1. Record brief
- **Known.** On the K-P slept life (frozen arm, nights 5-15, s319-321), told questions fetch the wrong episode on 136 of
  217 non-void told rows and the right one on 44 (`results/2026-11-25-routing-screen-read.md`). Raw ranking (D+G) and a
  strength cap (SC) were KILLED there; raw ranking trades wrong fetches for no fetch.
- **Most errors are hub captures, not near-duplicate swaps:** cat -> Bramble 17, car -> Bramble 15, work -> Bramble 14,
  work -> bus 10; only sister -> Philippa and cat -> dog are same-frame swaps (record workflow 11-25, dg-asset reader).
- **Keys already compared, but never on a slept store:** at 1.5B on an unslept day S_all routed 11/12 told right vs S0
  7/12 (`brain/research/2026-11-14-key-default-1.5b.md`); the live default stayed S0 by 0.17 of a fact
  (`results/2026-11-14-key-default-1.5b.md`). S_all on the slept gate raised fetches 1/18 -> 14/18 with replies flat
  (`results/2026-11-18-sall-slept-gate-1.5b.md`): a fetch gain need not move replies.
- **DG on the same key is predicted dead.** A random sparse expansion with k-WTA keeps rank order to first order
  (SimHash): on s319 n15 keys a DG code keeps the stored-key nearest neighbour 73.5 / 85.7 / 91.8% at 2 / 5 / 10%
  active (record workflow 11-25, one seed). On dense correlated codes a fixed expansion does not decorrelate
  (`research/findings/2026-06-11-cortex-fixed-expansion-decorrelation-NEGATIVE.md`). The old-track 6-seed GO
  (`research/findings/2026-08-19-replay-pattern-separation-DG-separator-...-6seed-GO.md`) was 2 memories on a LIF net.
- **Prerequisites:** near-duplicate separation and a paraphrase-robust cue are "missing" (`brain/PREREQUISITES.md`).
  Expected: S0 ranks the right episode deep on the hub-capture rows; a key carrying other information may not.
- **Host decision:** none. Every key is an existing read-out (`brain/hippocampus_keys.py`) or a fixed random map
  scored offline; nothing is wired into the brain.

## 2. Population and instrument
- KR night snapshots (s319/s320/s321, nights 5-15, `life_kw --night-snapshots`; KR == lifeKP, 0 mismatches) and each
  life's final trunk episode files (append-only; deltas for per-layer keys; the replay checks the key in each file
  equals the snapshot's keys.pt).
- Rows: the told rows of the banked intact-frozen battery (`routing_replay.row_class` = told) on the nights the 11-25
  screen did not void (s321 night 5 void). The right set = every store episode with `life.episode_is_source(item, text)`.
  A row with an empty right set is "unscorable" and reported, not counted.
- One cue read per row from the night's snapshot cortex, from `base`, as `Hippocampus.similarities` reads it
  (CPU fp32, pool40/41/42, under `tools/memcap.sh`). Sanity gate: under S0 the arg-max must equal the 11-25 replay's
  `raw_winner` on every row, else that night is void (the 11-25 drift rule).

## 3. Keys (rows of the table)
1. **S0** (the default): layer-0 matrix delta, centred cosine. Baseline.
2. **DG(S0)**: the centred S0 key through a fixed sparse random map (each of E units sums F = 128 input dimensions with
   random signs), then the top fraction f kept as a binary code; similarity = overlap / k. Same map for cue and store.
   E in {4096, 16384}, f in {0.02, 0.05}, map seeds 42/43/44/100/101/102.
3. **S0-q (cue offset)**: the cue key minus the mean S0 key of that night's other question cues that are not told
   items (never-told and untold near-duplicate rows ending in "?"), then centred cosine. No told cue enters the mean.
4. **S_l**: one layer's matrix delta, l in {6, 12, 18, 23}, centred cosine.
5. **S_all**: all layers, each block centred and normalised, averaged (`hippocampus_keys.centred_cos`).
6. **S0-drift**: every stored episode re-keyed by reading its own text through the night's cortex (the cue's cortex),
   S0. Tests whether storage-night vs slept-cortex mismatch is the cause.

## 4. Measures (per seed and pooled)
right@1, right@<=3, right@<=5 (best-ranked right episode), median rank, the right episode's margin to the top-1, and
the number of A-wrong rows (11-25 replay, arm A fetched a wrong episode) with the right episode at rank <= 3.

## 5. Kill rules (fixed now)
- **K-sane:** S0 right@1 must equal the 11-25 replay's raw-winner right count on the scored rows; else the run is VOID.
- **DG(S0) is dead** if, on every (E, f), pooled right@1 gains fewer than 5 rows over S0, or the sign of the gain differs
  across map seeds; also dead as a router if its gain is only on same-frame swaps with the hub captures unchanged.
- **S0-q is dead** if pooled right@1 is not at least 10 rows above S0.
- **A deeper key (S_l, S_all) is alive** only if it beats S0 on right@1 by at least 10% of the scored told rows on at
  least 2 of 3 seeds.
- **Drift is not the cause** if S0-drift is within 5 rows of S0 pooled right@1.
- **Separation of any kind is dead on this store** if under EVERY key the right episode ranks > 5 on most A-wrong rows:
  the defect is then cue form (question vs statement) and the next method is a cue transform or a learned key.
- An alive key earns only a screen prereg (its own bar sweep on a DEV store, then a live DEV life with a reply-level bar
  that counts no-fetch). Not a claim. Six seeds before any claim.

## 6. Cost and timebox
Pool40/41/42 CPU, one seed per node: one cue read per row for S0 and S_all (one pass gives every layer), plus about 100
episode re-reads per night for S0-drift. Estimate under 1 h per seed, $0. Timebox: the read by end of 11-26.

## Review (sonnet, 11-25, before data): OK_WITH_FIXES, applied
- A wrong episode tied with the best right one counts against it (DG overlaps are integers, so ties would favour DG).
- Every gain is computed on the rows that have that key (S0's right@1 on the same rows).
- **Added key S_all_live**: the live `KeyedHippocampus` S_all (each block normalised, then the concatenation centred),
  the key behind the 11-03 / 11-14 / 11-18 results. S_all as in 3.5 stays the registered row; both use the episode's
  plain routing change (`route_delta`) where one is kept, as the live key does; the count is recorded per night.
- DG's third rule (hub captures unchanged) is judged by hand from each row's `A_is_hub` (A's fetch is one of the
  night's three strongest episodes) and `A_text`; it is not computed.
- Section 2's void rule as written: a night where S0's top-1 differs from the replay's raw winner is dropped; a seed
  with more than 2 dropped nights (replay + ceiling) is void. `table` refuses fewer than 3 scored seeds.
