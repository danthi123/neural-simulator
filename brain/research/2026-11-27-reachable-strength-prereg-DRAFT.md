# Pre-registration (DRAFT): reachable strength, an offline screen on the KR snapshots

> **STATUS 11-27: DRAFT, pending pre-data review. GATED.** It may not be registered or run until the memory reply ceiling
> (`brain/research/2026-11-27-memory-ceiling-prereg.md`, Stage 1) returns **HEADROOM**. NO-HEADROOM closes all fetch-side work,
> this method included (that prereg, section 7). UNDEFINED or INCONCLUSIVE leaves this DRAFT unregistered, with no rescue.
> The code enforces the gate: `strength_screen score` returns `GATED` and reads no bar unless the ceiling verdict file's
> status starts with `HEADROOM`. No data has been read. The harness and tests are built ahead of time so the screen can run the
> same day as a HEADROOM verdict.
>
> Design source: `brain/research/2026-11-27-fable-review-dg-port.md` section 3, "HEADROOM". Capability: Memory strength and
> forgetting (also Episodic recall).
>
> Code:
> - Switch `Brain(replay_strengthen=None, strength_setpoint=None)`, both default OFF. The rule is in `brain/replay_strength.py`.
> - Harness `brain/experiments/strength_screen.py`.
> - Tests `brain/tests/test_replay_strength.py` (9) and `brain/tests/test_strength_screen.py` (26).
> - Review 11-27 applied (section 9): increment sham, a gated `score` returns the status alone, the registered seed set is
>   hardcoded, `--ceiling-dir` takes the per-seed `mc_s*` directories, a stale ledger stops `sims`.
> - Node script `brain/experiments/results/2026-11-27-reachable-strength/run_node.sh`.

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
  night-5 void), fire rows (open, greet, never-told, untold near-duplicate) and identity rows (reported only).
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
- **Stage `screen` (host arithmetic).** For each arm:
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

The constants are fixed before data, not fit:
- REPLAY_BOOST = USE_BOOST / 4, since a replay draw is a weaker reactivation than a cued fetch.
- SET_POINT = 1.0, the birth strength of a told-once episode (the same value as routing SC_CAP).
- The scale is clamped to [0.25, 4].

Declare these in `brain/SCAFFOLDS.md` at registration.

## 5. Measures and the pass rule (fixed before data; code `strength_screen.verdict`)
**Measures**, pooled and per seed:
- Told right, wrong and no-fetch.
- Fires on fire rows.
- Rows A fetched right that the arm turns into a wrong fetch.
- The screen-2 diagnostic: median `tanh(eff)` of the right episode, NULL vs RS.
- **Oracle-validated replies:** told rows where the arm fetches a right episode AND the memory ceiling's ORACLE reply on that
  row (`u_s*_n*_ORACLE.json`) was strict-ok. This projects reply gain from the ceiling and generates no reply. A's count
  uses the same rule.

**Gate:** the verdict is GATED unless the memory ceiling status is HEADROOM. A GATED `score` reads no screen file and
returns only the status (no per-seed or report-only counts), so nothing can be peeked before registration.

**INCONCLUSIVE** if any of these holds:
- A seed is missing or void.
- A tripwire fires.
- A control arm (RS, SCO, SHAM, FBL) is missing.
- An ORACLE unit is missing for a non-void night.
- Units come from mixed commits.
- The seed set is not exactly 319, 320, 321 (hardcoded `REGISTERED_SEEDS`).

**PASS needs every P clause and every C clause.** Thresholds are absolute counts on the surviving rows.
- **P1** right fetches: RS ≥ max(ceil(1.25 × A), A + 10), pooled.
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

**Outcomes:**
- **KILLED** if any P clause fails, and the failing clauses are named.
- **NOT-ATTRIBUTED** if every P clause holds but a C clause fails. It is not a pass, and there is no rescue. If only C1 fails,
  the record says scaling alone may carry it, and that needs its own prereg.
- **PASS** if everything holds. The tests check that each clause, failing alone, gives exactly that outcome.

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
- **Before registration** (reviewer's condition, adopted): one real seed's `sims` plus `screen` must show
  `null_mismatch = 0` and `state_mismatch = 0`, and the three constants get a `brain/SCAFFOLDS.md` row.
