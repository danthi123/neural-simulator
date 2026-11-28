# Pre-registration: mix-night screen, does a sleep-recipe change cut multi-night language drift on the K-P slept store?

> **STATUS: REGISTERED 2026-11-27 after pre-data sonnet review.** Review 1 (6 defects, section 9) and review 2 (2 BLOCKING,
> 7 NON-BLOCKING, section 10) are applied: B1 engagement gate (every candidate and placebo arm must have realized its change,
> logged in `sleep_info`, or it is void), B2 noise floor (noise_s >= 0.05 x KL(A0, n15)), N1 KILL-UNREAD plus one next step
> per outcome, N2/N3 expected outcome stated, N4 placebo stated as a reading aid (tested), N5 pick isolation stated, N6
> measure cap 10 GB on the 15 GB nodes plus an early stop after A0r, N7 empty-env test and spec consistency; tests
> 88 passed. Launch order is section 7: `strata` and step 0 run first and may still amend this file under their own
> **Amendments at the end:** S1 (strata merge), S2 (the W 3090), S3 (deterministic cuda kernels; discards the S2 step-0
> units, step 0 recomputed).
> registered rules (P2_MIN_N, the 45-min rule, the THREADS=1 fallback) before any other unit. Nothing has been run; no
> unit, timing or RSS exists. The design, the literature check and
> the reasons for each arm are `brain/research/2026-11-27-sleep-drift-recipe-check.md` (sections 1-3, 5, 6); its section 4 is
> superseded by this file. Capability: Lifelong learning (PREREQUISITES). Code: `brain/experiments/mix_night_screen.py`
> (switch-free: patches `brain.brain.replay_into_cortex` and `Brain._night_episodes` inside a context manager; brain/sleep.py,
> brain/brain.py and every default untouched), tests `brain/tests/test_mix_night_screen.py`, node script
> `brain/experiments/results/2026-11-27-mix-night/run_node.sh`, W launcher `.../ship.sh`, off-frame set
> `brain/eval/mix_night_offframe.json`.

## 1. Record brief
- K-P (keep bank of 512 frozen-prior samples, KL keep term, 64-text nightly subset) holds language better than B but does not
  stop drift: n15 held-out KL is 3.5-5.3x its night 1 on the KR seeds (bar for "fixed" is <= 2x), against B 13.6-22.7x
  (`results/2026-11-22-kp-dev-read-1.5b.md`). One K-P night repairs a B n15 life 2.9 -> 0.8 (`kp-gate-replications-1.5b.md`):
  the keep term both holds and repairs; what is unrepaired accumulates.
- Record check (`before_you_build.sh "sleep drift keep bank CE replay mixing"`): no REMIX-style arm, no placebo night, no CE-on-keep
  arm anywhere in brain/research, results, PREREQUISITES or SCAFFOLDS. Unmeasured, not refuted.
- The banked n12 / n13 / n15 held-out KL of the three lives (life JSON, trunk `lang_guard.kl`, AWS fp16/TF32): s319 0.270 / 0.285 /
  0.351, s320 0.252 / 0.271 / 0.294, s321 0.313 / 0.361 / 0.323. Read before any screen data; it bears on power (section 8).
- Expected-fail (declared): partly. A negative is a verdict on these four recipe changes at 1.5B over 3 chained nights, not on
  consolidation without drift. Nothing here flips a default; a PASS enters a six-seed fresh-split registered test.

## 2. Population and the chain
KR night snapshots of seeds 319/320/321 (lifeKP; DEV seeds, 0 mismatches as in the routing ceiling prereg). Start = snapshot n12 (its
adapters and the shared keep bank). Night k in 13, 14, 15 is slept by the real `Brain.sleep` on a Brain opened with
`routing_replay.open_snapshot` on a per-arm directory: store = snapshot n<k>'s `episodes.json` (text, strength, uses, last_used,
created), clock = its `meta.last_sleep`, sleep counter = k-1 (so `sleep_seed + sleeps` and the keep-subset draw are the life's),
adapters = the ARM's own chain (n12 for k = 13, then its previous night). Declared caveat: strengths and uses are the life's,
produced under the life's adapters, identical in every arm; the screen answers "given these stores", not "in a life".
Every reading night's snapshot must hold its `keep_bank.json` (with episodes, keys, meta): `chain_dir` and the run's file
check refuse otherwise, before the model loads, because a missing bank would make the Brain sample a NEW 512-text bank from the
frozen cortex (each arm on another bank than the life). `strata` reports `no_keep_bank` per night.

## 3. Arms (1.5B, 480 steps, lr 5e-4, rank 8, kl_weight 3 unless stated; split RNG streams unless stated)

| arm | nights | change | role |
|---|---|---|---|
| A0L | 13 | the shipped loop, one shared generator (exact replica of brain/sleep.py, tested equal) | instrument (K-sane, adapter distance to the life) |
| A0 | 13-15 | split streams: episode `Random(seed)`, keep index `Random(seed+31337)`, shipped 64-text subset | baseline of every contrast |
| A0r | 13 | identical rerun of A0 night 13 | determinism control |
| A0e | 13-15 | episode stream seed +1000 only | noise control (episode draw) |
| A0k | 13-15 | keep-index stream seed +1000 only | noise control (keep draw) |
| A1 | 13-15 | + 1.0 x CE on the same keep text | candidate: literal REMIX (predicted neutral) |
| A2 | 13-15 | 2 keep texts per step, KL and CE MEAN-reduced (dose fixed) | candidate: gradient variance |
| A3 | 13-15 | keep set 64 -> 256 per night (the shipped 64 plus 192, nested) | candidate: coverage (predicted clearest) |
| A4 | 13-15 | kl_weight 3 -> 6 | candidate: dose |
| A5s | 13-15 | episode texts -> unused bank texts (the cortex's own samples) | placebo (sham content), biased DOWN |
| A5f | 13-15 | episode texts -> the next KR seed's store of that night (319<-320, 320<-321, 321<-319; own texts removed) | placebo (non-self content) |

Lesion / sham / no-contrast: A0 is the no-contrast arm for every candidate; A5s / A5f are the content shams (replay with the
store's facts removed); A0e / A0k are the null contrasts (same recipe, different noise) that set the floor; A0r is the
null-of-nulls (same everything); A0L is the instrument link to the life. A3 recomputes each step's frozen reference from the
adapters-off cortex instead of holding 256 reference tensors (about 4.5 GB at 1.5B); the numbers are the same (tested on the
stub: identical losses and adapters).

## 4. Measures (read after every chained night and once at n12 per seed; read-only, no RNG draw)
1. **Held-out KL** and NLL_on: `brain.eval.life_langguard.measure` on HELDOUT (the instrument of every banked number).
2. **Off-frame KL**: the same `measure` on `brain/eval/mix_night_offframe.json`, 12 plain-prose passages with no "User:" frame,
   written by the teacher (Claude) on 11-27 before any run. **Registered sha256
   `b1a95684b8caeec1f6564493afa35a6cddc280d791f8466a2c73ccbc49e6f268`**; the script refuses a file or a command-line hash that
   differs, and records it in every unit. A test checks: 12 strings, no "User:"/"Assistant:", no 6-word overlap with
   `life_bank*.py` or `life_heldout.py`. Never used to fit anything.
3. **Retention**: per-token NLL of up to 20 real store texts per age stratum (own < 1 clock day, recent 1-3, mid 3-5, old >= 5),
   the same texts in every arm of a seed (`Random(20261128 + night)`), adapters on, no hippocampus.
4. **Stock-reply collapse**: greedy 40-token replies to 24 fixed ordinary prompts (none in the banks or HELDOUT: tested),
   refusal share and "my name is" share.
5. Adapter norm; A0L's relative L2 distance to the life's n13 adapters (report only); sleep seconds, read seconds; per arm max RSS
   from `/usr/bin/time -v` (node script).
6. **Engagement** (`sleep_info`, review 2 B1): what the sleep REALIZED, from inside the loop: keep_n (the night's keep set),
   keep_n_loop, kl_weight, ce_keep_w, keep_per_step (the set of per-step keep draws seen), steps_run, mean_ce_keep (mean CE
   keep term as added to the loss; None without it); placebo arms also placebo_pool, placebo_excluded, placebo_n and
   placebo_overlap (replayed texts that are in the night's real store). Also copied to the unit as `mean_ce_keep`,
   `kl_weight_used`.
Every unit records host, commit (on a pool node read from `~/brain-lane/route_repo/COMMIT`, refused if missing or empty; on W
`git rev-parse HEAD`), commit source, device, precision and model, and is written atomically; a rerun resumes and refuses a unit
written under a different host, commit, device, precision, model or off-frame hash.

## 5. Pass rule (written before any data; numbers read from unit JSONs by `mix_night_screen.py score --dir D --lives L1 L2 L3`)
Per candidate X in A1-A4 against A0 at night 15, per seed s:
- **Engagement (checked before anything else of X; review 2 B1):** every unit of X (n13-n15, 3 seeds) must show its change
  realized in `sleep_info`: A1 ce_keep_w == 1.0 and mean_ce_keep > 0; A2 keep_per_step == [2]; A3 keep_n == 256 (and the
  loop saw 256); A4 kl_weight == 6.0. A unit that fails, or records nothing, makes X **INCONCLUSIVE: arm not engaged**,
  never PASS and never KILL (a KILL on a null arm would be a verdict on nothing). The placebo read needs, on every A5s/A5f
  unit, placebo_pool > 0, placebo_excluded recorded and placebo_overlap == 0, else it is INCONCLUSIVE.
- **Noise floor**: noise_raw_s = max(|KL(A0) - KL(A0e)|, |KL(A0) - KL(A0k)|), held-out KL at n15, and
  **noise_s = max(noise_raw_s, 0.05 x KL(A0, n15)_s)** (review 2 B2). Two draws can land near A0 by chance and make every
  "> 2 x noise" clause trivial; the 0.05 floor is set now from the record, not from data: the smallest one-night life
  change at this point of the KR lives is s319 n12 -> n13, +0.015 on 0.285 (5.3%). It applies to P1, P1-change and the
  placebo read alike; both noise_raw and noise are reported per seed. This is still a 2-draw screen floor, so the 2 x noise
  rule screens, it does not estimate a noise distribution (a third noise arm was weighed and not added: +9 nights for a
  better floor on a screen whose outcome section 8 already expects to be KILL-UNREAD / INCONCLUSIVE).
- **P1 drift**: KL(X) < KL(A0) and KL(A0) - KL(X) > 2 x noise_s on every seed, AND mean KL(X) <= 0.75 x mean KL(A0).
- **P1b transfer**: off-frame KL(X) <= 0.85 x off-frame KL(A0) on >= 2 of 3 seeds.
- **P2 retention**: NLL(X) <= NLL(A0) + 0.10 nats in each of the own, recent and old strata on 3/3 seeds. **A stratum read on
  fewer than 5 texts (or none) in X or A0 on any seed is UNREAD; an unread P2 stratum makes X INCONCLUSIVE, never a P2 pass.**
- **P3 no new collapse**: refusal share and "my name is" share each <= A0 + 0.05 absolute on 3/3 seeds.
- **P1-change (DIAGNOSTIC, never a pass condition; added at the pre-data review):** per seed d_A0 = KL(A0,n15) - KL(n12),
  d_X = KL(X,n15) - KL(n12); a seed is readable if d_A0 > 2 x noise_s; cut = 1 - mean d_X / mean d_A0 over readable seeds,
  UNREAD with fewer than 2. Reported beside every candidate's status (`p1_change_diag`). Note: P1's per-seed clause is already a
  change contrast (X and A0 share the n12 start), so only P1's 0.75 ratio is level-based; P1-change says what share of the
  three nights' own drift X removed, which is what a KILL must be read against.
- **KILL** = P1's per-seed clause fails on >= 2 seeds (checked right after engagement: an unread P2 does not hide a KILL).
  A KILL whose P1-change diagnostic is UNREAD is printed **KILL-UNREAD** (review 2 N1) and the engine digest does not count
  it as a KILL. **PASS** = P1 and P1b and P2 and P3 with every P2 stratum read. Otherwise INCONCLUSIVE.
- **What each outcome leads to (review 2 N1):**
  - PASS: X enters a six-seed fresh-split registered test (below). No default flips.
  - KILL (P1-change READ): X, as a sleep-recipe constant at 1.5B over 3 chained nights, is recorded as failing to cut
    drift. The next try is a different METHOD (not another constant of this loop), chosen against PREREQUISITES.
  - KILL-UNREAD: no verdict on X. A0's own three-night drift is inside twice the noise, so this window cannot show a cut.
    X is not retired. A longer window (an earlier start night or more chained nights) is the only rerun, and only under a
    new prereg.
  - INCONCLUSIVE: the status names the cause. A void (instrument, provenance, engagement) is fixed and the affected units
    rerun under this rule. Unread P2 means amending P2_MIN_N from `strata` before launch, never after. If P1 holds on every
    seed and only the 0.75 ratio misses, it is recorded as "holds drift, no repair", a lead for a longer-window prereg,
    not a pass.
- **Instrument gate (voids every candidate AND the placebo read: INCONCLUSIVE):** any of the 12 n15 units of A0, A0e, A0k and X (3 seeds)
  missing; A0r or A0 n13 missing on any seed (determinism unread); A0r differs from A0 at n13 by more than 1e-3 relative held-out
  KL (threads are a provenance key; see the determinism fallback below); A0L or A0 n13 missing, or a life's banked n13 KL missing (K-sane unread); mixed provenance: commit, device, precision,
  model or off-frame hash differing across any units, a unit without a commit, or one seed's units on more than one host
  (hosts differ BY seed by design: one seed per node).
- **K-sane**: A0L and A0 n13 held-out KL within 25% of the life's banked n13 KL. A MISS is a device caveat (CPU fp32 vs AWS
  fp16/TF32), reported, not a void, because every contrast is arm-to-arm on one node and device.
- A PASS means "enters a six-seed fresh-split registered test (42/43/44/100/101/102)"; it flips no default. Seeds are reported
  individually; the P1 mean is the only pooled number and is a screening gate, not a claim.

**Determinism fallback (registered before data, pre-data review defect 6).** The 1e-3 relative tolerance is an untested
assumption (480 Adam steps on CPU fp32 with OMP_NUM_THREADS = nproc). So the pair is read at STEP 0, before any other unit:
`run_node.sh measure` runs A0r, then A0 night 13, then `detcheck` (A0r vs A0 n13, exit 3 on a miss). On a miss: no measure
file is written and `run` refuses. Fallback, once: move `mix_night/` aside, rerun step 0 with THREADS=1 (OMP and torch threads
1, recorded per unit). If the single-thread pair passes, the WHOLE screen runs at THREADS=1 on every node (`run` refuses a thread
count other than the measure's), and the time is re-estimated from that step 0 under the 45-min rule below. If it misses again,
the screen does not launch: the instrument is not repeatable on this CPU path; we do NOT loosen the tolerance after seeing the
miss (any tolerance chosen then would be fitted to the observed gap). The scorer still applies the gate to every seed.

**Placebo read (a reading aid, never a gate; review 2 N4).** No candidate status reads A5s/A5f: a candidate PASS stands
when the placebo units are missing or the read is MOTION-DOMINANT (tested). A PASS beside MOTION-DOMINANT is reported with
that caveat, and the six-seed test must then carry its own placebo arm. Per seed D_real = KL(A0, n15) - KL(n12), D_self = KL(A5s, n15) - KL(n12),
D_foreign = KL(A5f, n15) - KL(n12). A seed is READABLE only if D_real > 2 x noise_s; fewer than 2 readable seeds = UNREAD.
MOTION-DOMINANT if D_self >= 0.6 x D_real on >= 2 readable seeds (conservative: A5s is biased down). CONTENT-DRIVEN only if BOTH
D_self and D_foreign <= 0.25 x D_real on >= 2 readable seeds. Otherwise BRACKETED. A5 retention is never a pass. The same
instrument gate voids it.

All clauses above are exact in `candidate_verdict`, `placebo_read`, `instrument_checks`, `provenance_check` and `score`, with a
test per clause failing alone on synthetic records (edges at 0.75, 2 x noise, 0.85 on 2/3, +0.10, +0.05, n = 4 vs 5, each
missing instrument unit, each provenance key; review 2: each engagement clause per arm, an earlier night not engaged,
noise raw 0 / 1e-9 / 0.005 against the floor, KILL-UNREAD, placebo as a non-gate).

## 6. Compute and placement
- **Placement (data-bound):** each pool node holds one seed's KR snapshots: pool40 s319, pool41 s320, pool42 s321. `run_node.sh`
  derives the seed from the one `snaps_s<seed>` on the node and refuses any other; `ship.sh ship` copies each node's three
  foreign `episodes.json` (n13-n15 of the next seed, sha256 checked against the source node) into `foreign_s<F>`. A5f refuses
  before the model loads if they are missing. `ship.sh ship` refuses while any `brain.experiments` process is live on a node
  (shipping moves route_repo's code and COMMIT under it; the memory ceiling and local-sleep runs use that copy).
- **Measure first (RAM):** pool41 and pool42 have about 15 GB (a K-P LIFE night peaked at about 19 GB on pool40, 11-23; this
  screen holds no hippocampus deltas, so it should be smaller, UNMEASURED). Step 0 = `ship.sh measure` = `run_node.sh measure`
  on pool40: the n12 start reading, A0r night 13, A0 night 13 (`--max-night 13`, the run resumes it), the determinism pair,
  then A2 night 13 (the worst-case arm: two keep graphs live through backward), each under `/usr/bin/time -v`, written to
  `mix_night/measure_s319.txt` (max RSS of A0r and of A2, threads, `det ok`). Cap: MemTotal - 2 GB on pool40; on nodes under
  24 GB it must be given (`NODES="pool41 pool42" MEMCAP_GB=10 ship.sh measure`, so `mem_ok.sh 10 1` needs 11 GB free rather
  than the whole node). 10, not 11 (review 2 N6): the run cap ceil(1.3 x RSS) + 1 must fit MemTotal - 1 = 14 GB, so RSS
  must be <= 10 GB; at 11 an RSS of 10-11 GB would pass step 0 and then refuse at `run`. Also, right after A0r, a run cap
  that cannot fit stops step 0 (exit 2) before the A0 and A2 hours are spent (tested). 1.5B fp32 weights are about 6 GB,
  so the 15 GB nodes may well refuse; that fails closed and the seed is INCONCLUSIVE by the missing-unit rule. The full run's cap is ceil(1.3 x max(RSS A0r, RSS A2)) + 1 GB and must fit MemTotal - 1 GB or the node
  refuses. A failed baseline arm (A0L, A0, A0e, A0k: OOM, guard, missing file) stops that node at once (no candidate can be
  scored without it); a failed candidate or placebo is logged and the node moves on; each arm waits on
  `tools/mem_ok.sh <cap> 1` (others' runs share the nodes) and logs its own max RSS.
- **Launch visibility:** `ship.sh` logs every node launch to `mix_night/launch_<mode>.log` and prints its tail, so a refusal
  (echo + exit 2) is seen at launch. `ship.sh ship` refuses while `brain.experiments` OR `mn_run_node` is live on a node (between
  arms, and while an arm waits on `mem_ok.sh`, only the driver is live).
- **Time (UNMEASURED):** at a placeholder 25 min per 1.5B CPU night and 10 min per reading set (the reading set includes
  24 greedy 40-token replies per reading, review 2 N7; step 0 times a real night + reading and the 45-min rule decides), 87 night-sleeps + 3 start
  readings = 51 node-hours, about 17 h wall on three nodes in parallel (a floor: A2's second keep forward, A3's per-step
  reference recompute are not modelled). **Step-0 rule:** if one night + its reading exceeds 45 min on pool40, drop to nights 14-15
  only and drop A4, recorded here as an amendment before any other unit runs. Run `ship.sh strata` first (seconds, no model):
  if any P2 stratum has < 5 texts at n15 on a seed, amend before launch (the screen would be INCONCLUSIVE by rule).
  Step 0 is now three nights + readings (A0r, A0 n13, A2 n13), about 1.5-2.5 h on pool40; two of them are reused by the run.
- $0 cloud; no AWS; Claude cost = the pre-data review and the skeptic (sonnet). Runs only on idle pool capacity after test2.

## 7. Run commands (after the review is recorded here and the code is committed)
```
bash brain/experiments/results/2026-11-27-mix-night/ship.sh ship      # code + COMMIT + run_node + foreign files
bash brain/experiments/results/2026-11-27-mix-night/ship.sh strata    # P2 stratum sizes, store-only
bash brain/experiments/results/2026-11-27-mix-night/ship.sh measure   # step 0 on pool40 (s319: A0r, A0 n13, det pair, A2 n13)
NODES="pool41 pool42" MEMCAP_GB=10 bash brain/experiments/results/2026-11-27-mix-night/ship.sh measure   # cap required under 24 GB
# on a determinism miss: move mix_night/ aside on every node and repeat both measure lines with THREADS=1 (section 5)
bash brain/experiments/results/2026-11-27-mix-night/ship.sh run       # all arms on all three nodes
# read: pull ~/brain-lane/route_repo/_scratch/route/kr/mix_night/u_*.json from the 3 nodes into one dir, then
python -m brain.experiments.mix_night_screen score --dir D --lives lifeKR_s319.json.json lifeKR_s320.json.json lifeKR_s321.json.json --out D/verdict.json
```

## 8. Open for the pre-data review (questions, not changes)
1. **Power of P1.** The life's own n12 -> n15 change is +0.081 / +0.022 / -0.037 (s321 FALLS), so three chained nights move
   KL little next to the n15 level (~0.3). P1 asks for a 25% cut of the n15 LEVEL in three nights, i.e. repair of drift built
   over 12 nights. Repair at that size happened once (one K-P night on a B n15 life), but on a much larger excess. The
   placebo read (D_real > 2 x noise) may well be UNREAD. Keep the rule as written (a strict screen), or move P1 to the
   n12 -> n15 change? Decide before any data.
2. P2_MIN_N = 5 is new in this file (the note's rule passed P2 vacuously on an empty stratum). Confirm, or set from `strata`.
3. The determinism tolerance 1e-3 relative assumes CPU fp32 runs are bitwise or near-bitwise repeatable with OMP_NUM_THREADS =
   nproc; A0r tests exactly that, and a fail voids the screen by design.

## 9. Pre-data review 1 (11-27, sonnet) and what changed
Verdict: 6 defects, none fatal to the rules; rules, controls, off path, patch hook, offsets, leakage and the A4 non-repeat
found sound. All six applied (tests in `brain/tests/test_mix_night_screen.py`, each failing alone):
1. Missing `keep_bank.json` silently resampled a bank: now required per night in `chain_dir` and the run's file check
   (`SNAP_FILES`); `strata` reports it.
2. Refusals invisible through `ship.sh` (`>/dev/null`): launches now log to `launch_<mode>.log`, tail shown.
3. Measure under-read A2/A1 memory, and the 15 GB-node measure cap would wait 12 h: step 0 now measures A2 n13 too and sizes
   the cap from the max; MEMCAP_GB is required on nodes under 24 GB (11 for pool41/42).
4. `ship.sh` could ship between arms: it now also refuses on a live `mn_run_node`.
5. A failed baseline arm did not stop the node: any non-zero rc of A0L/A0/A0e/A0k now stops it.
6. Determinism tolerance unevidenced: the pair is read at step 0 with a registered single-thread fallback (section 5).
   I keep 1e-3 rather than loosen it: the gate's job is to catch a non-repeatable instrument, and a looser bound chosen
   after a miss would be fitted to the gap it is meant to catch.

Open questions (section 8), decided before any data:
1. **P1 power:** the rule stays as written (the 0.75 level ratio remains the screening gate); a diagnostic P1-change is added
   (section 5), reported, never a pass condition. A KILL is read against it: a KILL with P1-change UNREAD (A0's own three-night
   drift within twice the noise) is a verdict that this 3-night window cannot show the effect, not that the recipe fails.
2. **P2_MIN_N = 5:** confirmed; the final value is set from `ship.sh strata` before launch (amend here if any n15 stratum has
   fewer than 5 texts on a seed).

Residual (not defects): A2 draws its two keep indices with replacement (noise only). The worktree's `brain/brain.py` diff is
another lane's default-off `replay_strengthen` / `strength_setpoint` switch; `ship.sh` needs `brain/` clean, so that diff is
committed first and ships with this run (default off, not used by any arm here). `tools/engine/specs/mix_night.yaml` (another
session's file) is stale: TODO off-frame path, `snaps_s{foreign}`; it must point at `brain/eval/mix_night_offframe.json` (sha256
above) and `foreign_s{foreign}` before any engine launch; `ship.sh` does not use it.

## Engine route (tools/engine/specs/mix_night.yaml), 11-27: same rule, differences stated
An alternative launcher to `ship.sh`. It does not change any rule above. It points at `brain/eval/mix_night_offframe.json`
(sha256 above) and `foreign_s{foreign}`, so the stale-spec residual above is closed.
- **Step 0 on every node**, as in section 6: A0r, A0 n13, `detcheck`, A2 n13, each under `/usr/bin/time -v`. Caps are 21 GB
  on pool40 and 10 GB on pool41/42 (review 2 N6). The pick reads, per seed: det ok, and cap = ceil(1.3 x max(RSS A0r, RSS A2)) + 1. On s319
  it also reads the 45-min bar. Each seed's six-seed job runs at that seed's own cap. Each attempt writes a new time report,
  so a resumed attempt cannot lower a measured peak.
- **Stricter fit:** the engine needs cap + 2 GB <= node RAM, so the cap is at most 13 GB on every node. The rule above allows
  MemTotal - 1 GB (22 on pool40, 14 on pool41/42). This can only HOLD more often, never pass a cap the rule would refuse.
- **Manual fallbacks:** a determinism miss, a cap over 13 GB, or a step 0 over 45 min ends the engine chain as HOLD-DEV. The
  THREADS=1 fallback and the nights 14-15 amendment stay manual, recorded here before any relaunch.
- **Memcap kill (rc 137) is not retried.** The same cap would kill the job again. The engine ends as ERROR and pulls the log,
  which ends with the arm's time report. The verdict records the peak RSS of every arm.
- **Life JSONs:** the three W-side life files are `requires_local`, so the engine refuses to launch without them. If they
  are missing at score time, the verdict is ERROR, not a scored INCONCLUSIVE.
- **Disagreement with the engine review (defect 3).** The review proposed sizing the cap from A0r and A5f. A5f has A0's
  footprint, because only the episode source changes. Section 6 already names A2 as the worst-case arm, so the spec uses A2.
  A3's 256-text keep set has its reference log-probs recomputed per step rather than cached, so it is not measured at step
  0. Its peak is recorded in the verdict. If it dies at its seed's cap, the run stops as ERROR (memcap) rather than retrying.
- Launch: `python -m tools.engine mix_night` (status REGISTERED, so no `--allow-draft`), after `ship.sh ship` has placed
  the foreign files.

## 10. Pre-data review 2 (11-27, sonnet): 2 BLOCKING, 7 NON-BLOCKING; what changed
Record check repeated (`before_you_build.sh "sleep drift keep bank CE replay mixing"`): no REMIX-style or placebo-night arm
in the record, so nothing here repeats a killed method. Before the fixes: 69 passed. After: 88 passed
(`brain/tests/test_mix_night_screen.py`), and `tests/test_engine.py` 79 passed (its mix-night cases updated for
REGISTERED, the 10 GB cap, and engaged synthetic units).
- **B1 (fixed) no engagement gate.** `mix_sleep(stats=)` now reports what the loop realized: keep_n_loop, kl_weight,
  ce_keep_w, keep_per_step, steps_run and mean_ce_keep. `patched` copies it into `sleep_info`, adds placebo_overlap and
  placebo_n, and the unit carries `mean_ce_keep` and `kl_weight_used`. `engagement()` / `_engaged()` void a candidate or
  the placebo read (section 5). Tests: each candidate's clause failing alone on one seed, a would-be KILL on a null arm,
  an empty `sleep_info`, an earlier chained night, the A1 variants (None, weight 0, negative), each placebo clause alone,
  and the real Brain path. On the stub's 24-text bank the gate catches A3 (keep_n 24, not 256). The returned losses are
  unchanged, so the legacy replica still equals brain/sleep.py.
- **B2 (fixed) tiny noise floor.** `noise_floor = max(noise_raw, 0.05 x KL(A0, n15))` (section 5), with `noise_raw`
  reported. Tests: raw noise 0, 1e-9 and 0.005 against the floor, on both sides of the edge, plus zero noise in the
  placebo read and in P1-change.
- **N1 (fixed)** KILL-UNREAD in `verdict()`. The engine digest counts only a plain KILL (`mix_night.py` docstring), and
  each outcome now names its next step (section 5).
- **N2 (argued, rule kept)** P1's 0.75 level ratio stays. P1-change is not made the pass clause: per N3 it is likely
  UNREAD on 2 of 3 seeds, so passing on it would make a PASS no more readable, and the rule was already kept at review 1
  before any data. Expected outcome, stated so a null is read in that light: PASS needs mean KL(X, n15) <= 0.75 x ~0.32 =
  ~0.24, below the lives' n12 level (~0.28). That is repair of 12 nights of drift in 3. My PASS probability is low, about
  10% or less (a judgment, not a computation). The screen is mainly a KILL / KILL-UNREAD / INCONCLUSIVE screen, and its
  most likely informative result is a "holds drift, no repair" INCONCLUSIVE.
- **N3 (stated)** With the floor and the lives' own change (+0.081 / +0.022 / -0.037 on ~0.29-0.35), s319 is likely
  readable and s320 / s321 likely not, so P1-change and the placebo read are expected UNREAD and a KILL is expected to
  print as KILL-UNREAD.
- **N4 (fixed in text and test)** the placebo is a reading aid. `test_placebo_is_a_reading_aid_not_a_gate` shows a PASS
  with the A5 units missing and a PASS beside MOTION-DOMINANT.
- **N5 (stated)** The engine pick sees A0's n13 unit, a verdict-data unit, before any candidate exists. It reads only
  RSS, wall time and the determinism pair from it, never a KL contrast, and n13 is not a scored night. Harmless, and named
  here. A thread count that differs across nodes is blocked by `threads` in PROV_KEYS.
- **N6 (fixed)** The measure cap is 10 GB on the 15 GB nodes (run_node.sh, ship.sh, mix_night.yaml dev_memory_gb), with
  an early stop after A0r (tested). A3 is unmeasured at step 0 (engine section). A memcap kill there fails closed, and
  A3 on that seed is INCONCLUSIVE.
- **N7 (fixed)** `test_run_node_empty_memcap_and_threads_mean_default` covers ship.sh passing an empty MEMCAP_GB or
  THREADS. dev_memory_gb is labelled "set, not measured". The header status now lists both reviews. Not changed: A2
  draws with replacement and A3's extras are drawn from a correlated nested stream (declared residuals, noise only).
  The reading time is folded into the 10-min placeholder, and step 0 measures it.

## Amendment S1 (2026-11-27 21:40 UTC, from `ship.sh strata`, before any unit ran)
`strata` showed the `own` stratum (< 1 clock day) holds 3 texts at n15 on all three seeds (each simulated day adds 2-4
episodes), so P2 would be UNREAD by construction (P2_MIN_N 5). As section 7 requires, amended before launch: `own` and
`recent` merge into one `new` stratum (< 3 clock days; 9 texts at n15 on every seed); P2 now reads `new` and `old`;
P2_MIN_N stays 5; margins unchanged. `mid` remains report-only. Code: `STRATA`, `P2_STRATA` in mix_night_screen.py; tests
updated. The question P2 answers is unchanged: does the arm keep recently learned and old texts as well as A0.

## Amendment S2 (2026-11-27 22:55 UTC, from step 0, before any scored unit)
Step 0 on pool40 (s319 A0r): peak RSS 17.9 GB and 62 min for one night + reading. The registered cap rule gives
ceil(1.3 x 17.9 + 1) = 24 GB, which no pool node holds (pool40 23 GB; pool41/42 15 GB, where A0r was killed at 10 GB), and
one night exceeds the 45-min step-0 bound. As section 6/7 require, amended before any scored unit: the screen runs on the
workstation RTX 3090 (cuda; sleep in fp32 with TF32 off, as brain/life code does on cuda), all three seeds there, one at a
time, only when nvidia-smi shows the GPU free (owner 11-27). Step 0 is repeated on the 3090 for s319 (A0r, A0 n13,
determinism pair, A2 n13) before the full run, and its numbers replace the pool40 ones; every unit records device and
precision; all arms of a seed run on the one device. Rules, arms, margins and Amendment S1 unchanged. The pool40 A0r unit
(u_A0r_s319_n13.json) is a step-0 measurement only and is not scored.

## Amendment S3 (2026-11-28 UTC, from the S2 step 0, pre-data: no scored unit exists)
The S2 step 0 on the 3090 (engine, commit ce3649e) stopped at the pick as HOLD-DEV: the determinism pair MISSED on s320
(A0r vs A0 n13 held-out KL 0.3300 vs 0.2656, relative 0.24; mean replay loss 0.4329 vs 0.4313) and passed on s319 (1.5e-4)
and s321 (5.5e-4). The registered fallback (section 5: rerun step 0 at THREADS=1) targets CPU thread nondeterminism; on
cuda the nondeterminism comes from the kernels (atomic adds in backward passes, the cuBLAS workspace), which THREADS does
not touch. The s319/s321 gaps (1e-4 level, nonzero) show the kernels are not repeatable on this path, and s320 shows how far
480 Adam steps can carry such a difference. Amended before any scored unit, as the cuda form of the same once-only fallback:
- **On cuda, step 0 and every arm run with deterministic settings** (`mix_night_screen run --deterministic`; required with
  `--device cuda`, `run` refuses without it; off by default on cpu, where the THREADS rule of section 5 stays as it was):
  env `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set before the cortex is built), `torch.use_deterministic_algorithms(True,
  warn_only=False)`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`, fixed seeds
  (`random` and `torch` from the run's seed; each sleep reseeds from its own seed exactly as before). TF32 stays off (S2).
- **An op with no deterministic implementation**: torch raises naming it; the run records that op and falls back to
  `warn_only=True` (the night that raised wrote nothing and is redone; finished units resume). Every op torch names in
  its warnings from then on is recorded too. The op(s) are named in every later unit (`det_settings.fallback_op`,
  `det_settings.nondeterministic_ops`), and the determinism pair decides as registered.
- **Recorded in every unit**: `det_settings` (the settings as read back from torch and the env at write time: switch,
  use_deterministic_algorithms, warn_only, cudnn deterministic / benchmark, CUBLAS_WORKSPACE_CONFIG, seed, fallback op,
  ops named, whether CUDA was initialized before the switch) and the provenance key `deterministic` (in PROV_KEYS: a
  resumed unit of the other setting refuses, and `score` voids a unit set that mixes it).
- **The 1e-3 tolerance is NOT relaxed.** If the pair misses under S3 on any seed, the screen does not launch (section 5:
  no second fallback; a tolerance chosen after the miss would be fitted to it). The engine pick names this.
- **The earlier step-0 units are discarded.** The S2 step-0 units of all three seeds (commit ce3649e: start n12, A0r,
  A0 n13, A2 n13, the detcheck files; and the smoke A0r unit) ran without these settings and are never read or resumed.
  The engine out-dir is per commit, so the S3 commit recomputes every step-0 unit (launch with `--reset`), and the pick
  refuses any cuda step-0 unit without `deterministic: true`. Their RSS (about 3.9 GB) and times stay as a pre-S3 note
  only; the pick and the 45-min bar read the S3 step 0 (deterministic kernels may be slower).
Rules, arms, margins, the noise floor and Amendments S1/S2 are unchanged. Code: `apply_deterministic`, `det_fallback`,
`det_record`, `--deterministic` in `mix_night_screen.py`; `tools/engine/specs/mix_night.yaml` passes `--deterministic` on
cuda; `mix_night.py` pick check. Tests: `brain/tests/test_mix_night_screen.py` (switch sets the flags, units record them,
the fallback names the op, cuda refuses without the switch, the key is a provenance key), `tests/test_engine.py`.
