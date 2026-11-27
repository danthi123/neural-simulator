# Pre-registration: reachable strength, an offline screen on the KR snapshots

> **STATUS: REGISTERED 2026-11-27 after pre-data sonnet review; GATED on the memory reply ceiling.** (The file keeps its
> `-DRAFT` name because code and scripts cite this path.) The method arms may not be run until the memory reply ceiling
> (`brain/research/2026-11-27-memory-ceiling-prereg.md`, Stage 1) returns **HEADROOM**. NO-HEADROOM closes all fetch-side work,
> this method included (that prereg, section 7). UNDEFINED or INCONCLUSIVE there closes this screen too, with no rescue.
> The code enforces the gate twice: the full `strength_screen screen` (every arm) refuses to run, and `score` returns
> `GATED` and reads no file, unless the ceiling verdict file's status starts with `HEADROOM`. Before the gate only
> `screen --tripwire-only` may run (NULL and A; no method or control arm is computed or written). No data has been read.
>
> Design source: `brain/research/2026-11-27-fable-review-dg-port.md` section 3, "HEADROOM". Capability: Memory strength and
> forgetting (also Episodic recall).
>
> Code:
> - Switch `Brain(replay_strengthen=None, strength_setpoint=None)`, both default OFF. The rule is in `brain/replay_strength.py`.
> - Harness `brain/experiments/strength_screen.py`.
> - Tests `brain/tests/test_replay_strength.py` (9) and `brain/tests/test_strength_screen.py` (42 collected).
> - Review 11-27 applied (section 9): increment sham, a gated `score` returns the status alone, the registered seed set is
>   hardcoded, `--ceiling-dir` takes the per-seed `mc_s*` directories, a stale ledger stops `sims`.
> - Second pre-data review 11-27 applied (section 10): engagement gates G1-G4 (UNDEFINED), the tripwire-only screen and the
>   gated full screen, P1b (non-H) and P1c (clusters).
> - Node script `brain/experiments/results/2026-11-27-reachable-strength/run_node.sh` (sims, then the tripwire-only screen).

## 1. Record brief
- **Where the right episode is lost (screen 2 diagnostic, `results/2026-11-25-routing-screen2-read.md`).** On the 1.5B K-P
  slept life, the S_all key ranks the right episode first on 168 of 217 told rows. Of the 133 such rows not fetched right,
  121 stop at or after strength weighting, and only 6 have raw similarity below the bar. The right episode's
  `tanh(effective_strength)` has median 0.36, against 0.85 on rows that fired right. By group: Volvo 0.25, rotating 0.35,
  near-duplicate 0.81. What made the weight low (age decay, few use boosts, rivals boosted by S0's fetches) was not logged.
- **Banked fetch, arm A (K-P as is).** Right 44 of 217, wrong 136, no fetch 37. It fires 239 times on 359 fire rows.
- **Strength levers already tried:**
  - SC cap (routing screen 1, 11-24).
  - R-raw: KILLED with S_all (11-20).
  - M-use and M-scale: their life prereg is 11-19, and M-scale acts on adapters, not on episode strength.
  - `UseHippocampus` re-telling strength (11-03): NO-GO at 0.1B.

  None of them makes the brain's own NIGHT strengthen the trace it replays.
- **Biology.**
  - `research/biology/repetition-retrieval-strengthen-same-trace.md`: reactivation strengthens the existing trace and does
    not add a copy. Restabilisation folds the new learning into it, and retrieval needs no correctness grader.
  - `research/biology/homeostatic-scaling-relative-strength.md`: one multiplicative factor over all of a neuron's inputs.
    The factor is proportional to each input's strength, so relative order is kept and memories are not equalised. The file
    also warns that the old track's per-memory homeostat removed importance differences.
- **Mechanism (the method, no host text).** Each night, after the unchanged replay into the slow synapses:
  - (a) Each non-permanent episode replayed `c` times gets `strength += REPLAY_BOOST * c`. Its decay clock restarts at the
    night: the reactivated trace is restabilised, as a fetch's is. `c` is sleep.py's own draw sequence for that night.
  - (b) One factor `g = SET_POINT / mean(effective strength of the non-permanent episodes)`, clamped to [0.25, 4], multiplies
    every non-permanent strength. Ratios between episodes are kept exactly.
  - Permanent episodes are untouched. The only trainer is the brain's own replay schedule.
- **Expected-fail (declared).**
  - Replay is drawn in proportion to effective strength, so (a) is rich-get-richer in expectation. With the clock restart,
    the boost is close to uniform in relative terms, so most of the effect is that replayed traces stop decaying.
  - The homeostat cannot undo hub dominance, because it keeps ratios.
  - Lifting the whole store toward the set-point can raise fires, so P4 (half of A's fires) may fail even if right fetches rise.
  - The counterfactual store keeps the banked S0 fetch history. Arm FBL and the live life address that.

## 2. Population
- Seeds 319, 320 and 321 (the KR lives of screen 2 and of the memory ceiling), nights 5-15.
- Rows: the banked intact-frozen battery of each night, from `routing_replay.banked_rows`. Told rows (217 after the s321
  night-5 void), fire rows (open, greet, never-told, untold near-duplicate) and identity rows (fetches per arm, reported
  only). 128 of the 217 told rows come from four H items (32 rows each); the 89 others are the non-H rows. A (seed, item)
  pair is a cluster (68 clusters).
- **Void rules.**
  - A night is void if arm A, recomputed from the snapshot, does not reproduce the banked fetch idx and recalled text on
    every row, or if A's rightness differs from the banked row.
  - A seed with more than 2 of 11 nights void is void.
  - The oracle-validated measure uses only nights with an ORACLE reply unit from the memory ceiling. A non-void night with
    no such unit makes the verdict INCONCLUSIVE.

## 3. How the screen works
- **Stage `sims` (pool CPU fp32, guarded by POOL_NODE=1 or BRAIN_ALLOW_W3090=1).** For each night snapshot:
  - Open the brain's own store with `routing_replay.open_snapshot` (no deltas).
  - Read every battery cue once and keep the full raw similarity vector over the store.
  - Keep the set of stored episodes that are the row item's own telling (`life.episode_is_source`), and arm A's decision
    (void rule).
  - Also write the ledger of every snapshot (episodes.json scalars, meta sleeps) and the night settings read from the opened
    Brain (steps, sleep seed, generic count).
  - Per-night atomic files with resume. A unit written under another commit stops the run.
- **Stage `screen --tripwire-only` (host arithmetic; the only screen allowed before HEADROOM).** NULL and A only; it writes
  `trip_s<seed>.json` with the tripwire counters and the void nights, no row and no other arm.
- **Stage `screen` (host arithmetic; refuses unless the ceiling verdict is HEADROOM).** For each arm:
  - Rebuild the store night by night from the ledger. Strength is salience plus the banked waking fetch events (uses
    increments between snapshots, at the snapshot's last_used), with the arm's night rule applied by `replay_strength.night_update`.
  - Re-make every row's decision from the banked similarity vector through `Hippocampus._compete` (`routing_replay.decide`,
    FREEZE on).
- **Tripwires (any one makes the verdict INCONCLUSIVE).**
  - NULL (the rebuild with no method) must equal the snapshot strength and last_used on every episode of every night.
  - NULL must also equal A's decision on every row, and the plain competition on the snapshot strengths must equal A.
  - The rebuild raises if the store is not append-only, if strength minus USE_BOOST times uses moves between snapshots, or
    if the rebuilt sleep count differs from meta sleeps.
  - Refused outright: lives with `replay_priority`, `use_gate`, sleep_extra, or a hippocampus class that overrides the
    competition.
- **What the tests show (stub cortex).**
  - On a stub life the sims stage reproduces every banked fetch and NULL reproduces the snapshots.
  - The RS rebuild from an OFF life's ledger equals, night by night, the store of a life lived with the Brain switch on
    (with fetch boosts frozen in both lives).
  - `replay_counts` equals the draws of the real `brain/sleep.py` loop.

## 4. Arms

| arm | what | role |
|---|---|---|
| A | banked snapshot strengths (K-P as is) | baseline |
| NULL | the rebuild with no method | tripwire (must equal A) |
| **RS** | replay-strengthen (REPLAY_BOOST 0.125, clock restart) plus scaling to SET_POINT 1.0 | the method |
| SCO | scaling only (no replay strengthening) | control: does scaling alone do it |
| SHAM | increment sham: NULL's effective strength plus RS's per-episode increment (eff_RS − eff_NULL), the increments permuted among the non-permanent episodes at each battery (20 draws, fixed seeds, mean counts; a sum below 0 floors at 0, counted as `sham_floored`) | no-contrast control: the store's own structure kept, the method's add-on on the wrong episodes |
| FBL | RS with the waking fetch boost lesioned (a fetch changes no strength and no clock) | lesion: does the gain ride on the banked S0 fetch history |
| RS-noreset, RS-b0.0625, RS-b0.25, RS-sp0.5, RS-sp2.0 | the rule's parts and a factor-2 grid | report only, never barred, never a rescue |
| NULL-nofetch | NULL with the fetch boost lesioned (FBL's own no-method baseline) | report only (see C3) |

The constants are fixed before data, not fit:
- REPLAY_BOOST = USE_BOOST / 4, since a replay draw is a weaker reactivation than a cued fetch.
- SET_POINT = 1.0, the birth strength of a told-once episode (the same value as routing SC_CAP).
- The scale is clamped to [0.25, 4].

Declared in `brain/SCAFFOLDS.md` (row "Reachable strength night rule", 11-27).

## 5. Measures and the pass rule (fixed before data; code `strength_screen.verdict`)
**Measures**, pooled and per seed:
- Told right, wrong and no-fetch; right on the 89 non-H rows; right per (seed, item) cluster.
- Fires on fire rows; fetches on identity rows (report only).
- Rows A fetched right that the arm turns into a wrong fetch (P5), and into no fetch (reported beside P5).
- The screen-2 diagnostic: median `tanh(eff)` of each told row's best right episode, NULL vs RS, pooled (gate G4).
- Engagement (per arm, per slept night of the rebuild): g, the raw g, clamp hits, mean effective strength before scaling,
  episodes replayed at least once, episodes whose strength or clock the night changed; per battery night: decisions that
  differ from NULL (RS, SCO, SHAM) and from RS (FBL), FBL's store against RS's, the sham's floored slots.
- **Oracle-validated replies:** told rows where the arm fetches a right episode AND the memory ceiling's ORACLE reply on that
  row (`u_s*_n*_ORACLE.json`) was strict-ok. This projects reply gain from the ceiling and generates no reply. A's count
  uses the same rule. The ceiling unit's row i must be this screen's row i (same told row count, item and cue), or the
  score stops; an empty `--ceiling-dir` is an error.

**Gate:** the verdict is GATED unless the memory ceiling status is HEADROOM. A GATED `score` reads no screen file and
returns only the status (no per-seed or report-only counts), so nothing can be peeked before registration.

**INCONCLUSIVE** if any of these holds:
- A seed is missing or void.
- A tripwire fires.
- A control arm (RS, SCO, SHAM, FBL) is missing.
- An ORACLE unit is missing for a non-void night.
- Units come from mixed commits.
- The seed set is not exactly 319, 320, 321 (hardcoded `REGISTERED_SEEDS`).
- A seed's engagement record is missing.

**PASS needs every P clause, every C clause and every engagement gate.** Thresholds are absolute counts on the surviving rows.
- **P1** right fetches: RS ≥ max(ceil(1.25 × A), A + 10), pooled.
- **P1b** non-H rows: RS right − A right on the 89 non-H told rows ≥ 6 (the ceiling's non-H bar).
- **P1c** clusters: RS gains in more (seed, item) clusters than it loses in (ties dropped; the ceiling's sign-test rule).
  P1b and P1c stop one H item's 32 repeated rows from clearing P1 alone.
- **P2** direction: RS right > A right on at least 2 of 3 seeds, and falls by more than 1 on none.
- **P3** oracle-validated replies: RS ≥ A + max(8, ceil(0.25 × A)), pooled. P3 is P1 reweighted by the ceiling's ORACLE
  ok-rate (the ORACLE reply is handed the right episode), not independent reply evidence. A P3 pass adds no evidence beyond
  P1; it guards against a gain that lands only on rows whose reply fails even with the right episode.
- **P4** false fires: RS fires on fire rows ≤ floor(0.5 × A's), pooled. This is the review's "at most half of A's".
- **P5** at most 1 new wrong fetch per seed on rows A fetched right.
- **C1** RS right − SCO right ≥ 5. Scaling alone does not do it.
- **C2** RS right − SHAM right (mean of draws) ≥ 10. The per-episode assignment of the increment matters. With RS = NULL
  the sham is NULL exactly, so C2 cannot pass on the store's baseline structure alone.
- **C3** FBL right − A right ≥ 0.5 × (RS right − A right). The gain does not ride on the banked fetch history.

**Engagement gates** (pooled over the live seeds; code `engagement_gates`). They decide whether an arm did anything:
- **G1 (method):** RS's decision differs from NULL's on ≥ 10 rows (= GAIN_MIN: fewer cannot reach P1's minimum gain), and
  RS's g is clamped on at most 1/4 of its slept nights.
- **G2 (control SCO):** SCO's |g − 1| > 0.05 on at least half of its slept nights, and SCO's decision differs from NULL's
  on ≥ 1 row. The decision bar is 1, not larger, because "scaling changes few decisions" is a valid control result once the
  scale itself is shown to move.
- **G3 (controls FBL, SHAM):** FBL's store differs from RS's on ≥ 1 non-permanent episode-night (the lesion removed
  something); SHAM's decisions differ from NULL's (> 0, mean of draws); `sham_floored` ≤ 5% of the sham's episode slots.
- **G4 (method):** RS's pooled median `tanh(eff)` of the told row's best right episode is above NULL's (the method moved
  the quantity the screen-2 diagnosis blames).

**Outcomes:**
- **UNDEFINED** if G1 or G4 fails (the method did not engage), whatever the P and C clauses say. If G2 or G3 fails, a
  verdict that reads the controls (PASS or NOT-ATTRIBUTED) becomes UNDEFINED; a P-clause KILLED stays KILLED, because no P
  clause reads SCO, SHAM or FBL. UNDEFINED has no rescue and is not a verdict on the method.
- **KILLED** if any P clause fails (with G1 and G4 holding), and the failing clauses are named.
- **NOT-ATTRIBUTED** if every P clause holds but a C clause fails. It is not a pass, and there is no rescue. If only C1 fails,
  the record says scaling alone may carry it, and that needs its own prereg.
- **PASS** if everything holds. The tests check that each clause and each gate, failing alone, gives exactly that outcome.

**Bars on the recorded references** (for audit; the code recomputes them on the surviving rows after voids): A right 44
gives P1 ≥ 55; A fires 239 gives P4 ≤ 119. P1 and P4 together ask the method to add right fetches while halving fires.
Ratio-preserving scaling does not separate told from fire rows, and halving fires needs g < 1, which also lowers right
fetches; what can do both is mostly the clock restart (replayed traces stop decaying while unreplayed rivals keep decaying).
So a PASS mostly tests the clock-restart effect, and P4 is the clause expected to fail (section 1).

## 6. Compute
- **`sims`:** one cue read per battery row (about 20 rows × 11 nights per seed) plus one snapshot open per night, on
  pool40, pool41 and pool42 (one seed per node), CPU fp32, under `tools/memcap.sh`. Screen 2's identical pattern took
  4-5 min per seed, so the estimate is about 5-10 min per seed.
- **`screen`:** host arithmetic, under a minute per seed.
- **`score`:** seconds. It needs the memory ceiling's `u_*_ORACLE.json` files and its verdict JSON. `--ceiling-dir` takes
  the per-seed directories as written (`--ceiling-dir $C/mc_s319 $C/mc_s320 $C/mc_s321`, or the glob); two differing
  copies of one unit stop the score.
- Total well under an hour of pool CPU. No GPU.

## 7. What each outcome does
- **PASS** earns only a live DEV life; it is not evidence that the switch helps a live life. PASS earns a live DEV life with the switch (`--brain-kw-extra '{"replay_strengthen": 0.125, "strength_setpoint": 1.0}'`),
  first 3 then six seeds (42/43/44/100/101/102). It needs its own prereg with reply-level bars and a lesion that changes
  the reply. No default flips on this screen.
- **KILLED or NOT-ATTRIBUTED** is a verdict on this method, recorded in PREREQUISITES "Memory strength and forgetting".
  UNDEFINED is recorded as "the method did not engage at the registered constants", not as a negative.
- **Readings fixed now (report only, never a rescue):** if RS-noreset's gain over A is under half of RS's, the record says
  the effect is mostly "stop decay" (the clock restart), not strengthening. If C3 fails but FBL − NULL-nofetch ≥
  0.5 × (RS − A), the record says C3's baseline is confounded by the lesioned fetch history (FBL also loses A's banked
  boosts); the verdict stays NOT-ATTRIBUTED.
  The next fetch-side candidate from the review (write-side separation at `hippocampus.store`) is judged on its own.

## 8. Caveats
- Offline and counterfactual: the waking fetches are the banked S0 ones, but a life with the switch on would fetch
  differently, and its replies would differ. FBL and the live life address this.
- The slept cortex is unchanged. Strength does not alter what sleep writes into the adapters in this screen, although in a
  live life the replay weights would change too.
- Seen seeds, 3 seeds, 1.5B K-P only. Fetch level plus a projection from oracle replies. Paraphrase and voice are MISSING
  capabilities, so the strict scorer undercounts.
- The events of a gap between snapshots are placed at the later snapshot's last_used (`approx_events` is reported). With
  every night probed there is no gap.
- What a live switch-on life changes that the rebuild does not model: the fetch decisions after night 1 (a different
  store fetches differently, and each fetch boosts), and the sleep replay weights (`effective_strength` also weights the
  replay draws into the adapters). A scan of `brain/` finds strength used only by recall (`tanh(eff)` competition),
  the replay weights and permanence's floor (`max(strength, 3.0)`, permanent episodes are untouched by the switch); no
  pruning or promotion rule reads strength, so a scale g < 1 changes nothing else. FBL covers only the banked-fetch part.

## 9. Pre-data review 11-27 and what changed
- **Sham (important, accepted).** The first SHAM shuffled RS's whole effective-strength vector, which destroys the store's
  own structure (salience, recency, banked fetch boosts) that gives A its right fetches, so C2 would hold for any
  informative store and did not test the replay increment. SHAM now permutes only the increment over NULL
  (`_sham_store`; test `test_sham_permutes_only_the_increment`: RS = NULL gives SHAM = NULL, so C2 fails).
- **`--ceiling-dir` (important, accepted).** It takes several directories (`nargs="+"`) and searches each.
- **Gated peeking (important, accepted).** A GATED `score` returns before reading any screen file
  (test `test_gated_score_returns_the_status_alone`).
- **Seed count from the CLI (minor, accepted).** The registered set is hardcoded; any other set is INCONCLUSIVE.
- **P3 = P1 reweighted (minor, accepted).** Stated at P3; kept as a guard, read as no extra evidence.
- **Unmodeled switch consequences (minor, accepted).** Stated in section 8; PASS earns only a live DEV life (section 7).
- **Stale ledger (minor, accepted).** `run_sims` stops on a ledger of another commit or weekend setting.
- **Before registration** (reviewer's condition, adopted and amended by the second review, section 10): the three
  constants have their `brain/SCAFFOLDS.md` row (done). The real-seed tripwire check is now `screen --tripwire-only`
  (NULL and A only) and is a pre-run condition, not a registration one: a nonzero `null_mismatch` or `state_mismatch` makes
  the verdict INCONCLUSIVE anyway.

## 10. Second pre-data review 11-27 (sonnet; 3 blocking, 8 non-blocking) and what changed
- **B1 no engagement record or gate (accepted, fixed).** `simulate` now logs each slept night's record (g, raw g, clamp,
  mean effective strength before, episodes replayed, episodes changed); `screen_seed` keeps the per-night engagement
  (decisions vs NULL, FBL's store vs RS's, the sham's floored slots, the right episode's tanh(eff) under NULL and RS);
  `seed_counts` aggregates them; gates G1-G4 (section 5) make an inert arm's verdict UNDEFINED. One refinement of the
  review's rule: G2/G3 (controls) do not turn a P-clause KILLED into UNDEFINED, because the P clauses read no control.
  Tests: `test_an_inert_arm_is_undefined` (each gate failing alone), `test_inert_method_is_undefined_even_when_a_p_clause_fails`,
  `test_an_inert_control_does_not_block_a_p_clause_kill`, `test_missing_engagement_is_inconclusive`.
- **B2 no-peek hole in `sims`/`screen` (accepted, fixed).** `screen --tripwire-only` computes and writes NULL and A only;
  the full screen refuses without HEADROOM (`run_screen`, CLI; tests in `test_sims_and_null_rebuild_reproduce_a_stub_life`
  and `test_full_screen_cli_refuses_without_headroom`). `run_node.sh` now chains `sims` into the tripwire-only screen.
  The full three-seed screen is first run after HEADROOM and stays unread until then. On "run the dry run on a stub or DEV
  life, not a verdict seed": the stub-life test covers the harness, and the tripwire-only screen may run on a verdict seed
  because it computes no method or control arm (it reads only whether NULL rebuilds the banked store); a stub cannot show
  that the real ledgers reproduce. The `sims` stage itself holds no arm (similarity vectors and A's banked decision only).
- **B3 one item can carry P1 (accepted, fixed).** P1b (non-H ≥ +6) and P1c (cluster gains > losses) added as kills;
  `seed_counts` keeps right per non-H row and per item; tests `P1b` (the whole gain from one H item) and `P1c`.
- **NB1 C3 baseline confounded (accepted as report).** NULL-nofetch added (report only) with a fixed reading (section 7).
  C3 itself is unchanged: the confound can only make it fail spuriously, never pass.
- **NB2 off-path test skipped once committed (accepted, fixed).** `test_off_equals_the_head_brain` now compares against
  `brain.py` at the pinned pre-switch commit 4389b39ac and runs (no skip).
- **NB3 P1 with P4 (accepted).** Reference bars printed and the clock-restart reading stated (section 5).
- **NB4 per-night g range, clock restart (argued, partly accepted).** g and the clamp are logged and G1 bars clamping. The
  [0.25, 4] range is a per-night safety clamp, not a speed: the factor is applied once per night (one sleep-wake cycle of
  the life's clock), the coarsest step the store has, and the biology's ~48 h is two such nights; a slower rule would add
  a fit constant. RS-noreset stays report only, because the clock restart is part of the registered mechanism
  (restabilisation of the reactivated trace, biology file); its share is read as fixed in section 7.
- **NB5 oracle rows keyed by index (accepted, fixed).** `_check_aligned` checks the told row count, item and cue.
- **NB6 empty ceiling dir (accepted, fixed).** `seed_counts` and `score` raise.
- **NB7 identity rows (accepted, fixed).** Fetches on identity rows are computed per arm (report only).
- **NB8 right to no fetch (accepted, fixed).** `new_none_on_A_right` is counted per seed and reported beside P5.
