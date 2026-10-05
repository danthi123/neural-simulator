# long-delay-credit WORKLOG (task 3, Weekend work)

Branch: research/draft-long-delay-credit · worktree .claude/worktrees/draft-long-delay-credit
Serves docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md section 4.
ALL results here are DEV (dev seeds only), NOT a verdict. Claude reviews after Tuesday.

## 2026-09-29 01:15  Worktree created + prior-work search (required before the first lever)

- `git worktree add .claude/worktrees/draft-long-delay-credit -b research/draft-long-delay-credit`
  (based on main f52c6d246).
- `bash tools/before_you_build.sh "long delay temporal credit assignment eligibility trace"`
  (main checkout, read-only). Top hits (POINTERs, read in full):
  1. `research/findings/2026-05-18-td-value-critic-temporal-credit-PASS.md` — the VALIDATED
     biologically-canonical TD(lambda) value-function critic (phasic-DA TD error), reuses
     `sim.kernels.fused_eligibility_trace_decay` UNMODIFIED, NO autograd. Schultz cue-shift
     transfer 0.997, us_decay 0.0015, vrmse 0.003 (3 seeds). THIS is the machinery to reuse.
  2. `research/findings/2026-07-11-R2b-dual-timescale-eligibility-GO-...md` — CORRECTED/REFUTED:
     a "slow eligibility horizon" lift was an effective-learning-rate artifact (magnitude-
     decoupled EMA control collapses it). LESSON: do NOT claim a slow-trace horizon effect
     without a magnitude-decoupled control. Not directly needed here (fixed gamma*lambda).
  3. `research/findings/2026-05-18-compose-temporal-credit-PASS.md` — temporal credit bridges a
     composition gap (context, not needed for this benchmark).
- Reusable machinery confirmed present:
  - `sim/td_value_critic.py` (102 L): TD(lambda) critic, `analytic_vstar`, `csc_features`,
    `scale_free_transfer`, `run_pavlovian` (modes td/no_bootstrap/permuted/wrongsign).
  - `sim/kernels.py:1630` `fused_eligibility_trace_decay(trace, decay_factor) = trace*decay_factor`
    (REUSE UNMODIFIED).
  - `sim/neuromodulators.py` (phasic-DA framing; the TD error IS the phasic-DA signal per the
    TD-critic finding).
  - `research/runners/td_critic_core.py` (frozen-bar THREE-STATE verdict pattern to mirror).
  - `research/runners/_advantage_actor_critic_probe.py` — the existing SPOKING actor-critic is
    cupy/GPU-bound (imports g11_bg_runner, cupy). OUT of my CPU-only scope (runbook: no GPU jobs).

## 2026-09-29 01:20  Design decision (flagged UNSURE for Claude)

- The runbook task 3 says "on a small spiking network" but also "REUSE existing
  eligibility-trace / dopamine machinery". The only existing spiking actor-critic
  (`_advantage_actor_critic_probe.py`) is cupy/GPU-bound and out of my CPU-only scope. The
  VALIDATED temporal-credit machinery (`sim/td_value_critic.py`) is rate-level (pure numpy, CPU).
- DECISION (mine, flagged UNSURE): implement the benchmark RATE-LEVEL, reusing the validated
  TD/eligibility/dopamine machinery byte-for-byte (CPU, dev seeds). This directly answers the
  plan's science question (persistent cue memory vs temporal credit assignment) and reuses the
  validated kernel. Whether a SPIKING realization is required is an UNSURE item for Claude —
  it is a design choice that changes what the experiment means, so it is flagged, not decided.
- The benchmark separates the two mechanisms the plan names:
  - PERSISTENT CUE MEMORY = a fixed slow-decay working-memory buffer (persistent activity) that
    holds the cue through the variable delay; the actor reads it out to make the choice.
  - TEMPORAL CREDIT ASSIGNMENT = the TD eligibility trace carries the delayed reward's TD error
    back to the earlier cue/choice event; measured by RPE transfer to the cue (not the terminal
    state / final decoy / reward time).
  - TRACE ABLATION cuts the eligibility trace (gamma*lambda -> 0) for the credit path ONLY; the
    working-memory buffer (a separate mechanism) is untouched. Prediction: credit (RPE transfer)
    collapses while the buffer still holds the cue -> the plan's key dissociation.

## 2026-09-29 (overnight)  Iteration: making the credit REACH the cue (credit-stealing debugging)

The plan's key control requires the delayed reward's credit to anchor to the EARLIER cue, not to
whatever is active at reward time. Debugging showed a recurring failure mode — **credit-stealing by
a temporally-closer or perfectly-predictive feature** — three times over, each fixed by removing the
stealing feature from the critic's feature set (it stays in the WORLD/timeline):

1. **Reward made CUE-contingent** (r = 1 iff cue is the target, cue A), not choice-contingent. A
   diagnostic showed that with a choice-contingent reward the value anchored to the (late) choice and
   the taps (w_cue ~ 0). Making the reward cue-contingent makes the cue the DIRECT predictor.
2. **Removed the time-since-cue taps.** The taps sit between the cue and the reward and steal the
   credit (a diagnostic showed the taps carrying the weight, w_cue ~ 0). Also the plan's "no clock".
3. **Removed the terminal choice as a critic feature.** The choice IS the memory read-out (argmax of
   the buffer), so it equals the cue and is a PERFECT predictor of the cue-contingent reward
   (diagnostic: w_choice ~ 0.83, w_cue ~ 0). The choice is still COMPUTED for the buffer_accuracy
   probe; it is just not a critic feature.
4. **Removed the terminal/decoy/distractor as critic features.** These sit a few steps BEFORE the
   reward, so the trace leaked credit to them (diagnostic: w_terminal ~ 0.30, w_decoy ~ 0.29
   alongside w_cue ~ 0.50). The plan requires the credit to anchor to the CUE, not the intervening
   events. They remain in the WORLD (the timeline) but are not critic features.

Result: the critic sees ONLY {bias, transient cue}. The reward's TD error must be carried BACK across
the delay by the eligibility trace to reach the cue.

**Dev-seed result (seed 7, 2000 trials):**

| arm | buffer_acc | val_A | val_B | disc | rpe_term | rpe_decoy |
|---|---|---|---|---|---|---|
| full | 1.0 | 0.547 | 0.042 | 0.505 | 0.002 | 0.002 |
| trace_ablated | 1.0 | 0.247 | 0.252 | -0.005 | 0.005 | 0.005 |
| immediate | 1.0 | 0.623 | 0.034 | 0.589 | 0.002 | 0.002 |
| misleading_final_cue | 1.0 | 0.547 | 0.042 | 0.505 | 0.002 | 0.002 |
| shuffled_reward | 1.0 | 0.284 | 0.326 | -0.042 | 0.002 | 0.002 |

The plan's key dissociation is now measurable: `value_discrimination` is HIGH in `full` (0.505, credit
reached the cue) and ~0 in `trace_ablated` (-0.005, credit cut), while `buffer_accuracy` is 1.0 in
BOTH (memory intact). The terminal/decoy RPEs dropped to ~0.002 (no more stealing). `shuffled_reward`
shows no cue structure (disc ~0).

**NOTE (us_decay ~1.0 in all arms):** the cue is TRANSIENT (removed after CUE_STEPS=2), so the critic's
value is a spike at the cue onset and is NOT sustained to the reward step; the reward is never
"predicted" at the reward time (V(reward) = bias only). This is a consequence of the transient cue, not
a failure. The credit read-out is `value_discrimination`, not `us_decay`. Whether the benchmark should
instead SUSTAIN the cue feature through the delay (so the reward is predicted and us_decay -> 0) is an
UNSURE item for Claude (PREREG section 8.4) — it changes what is measured.

Tests updated to the new metric names (buffer_accuracy, value_discrimination); the stale
choice_accuracy/rpe_transfer_to_cue/IDX_CHOICE_A/IDX_TERMINAL references are gone. `test_no_autograd`
now checks import statements only (the docstring mentions "autograd" in prose). 6/6 pass.

## 2026-09-29 04:45  15-seed dev sweep + two NON-BLOCKING warnings (for Claude)

Ran the planned small dev sweep (seeds 1-15, all 5 arms, 2000 trials, ~31 s CPU) to
`research/findings/raw/_draft_long_delay_credit_dev_sweep.json` (75 rows). The plan's key
dissociation is robust across every seed. Attribution (full vs trace_ablated, value_discrimination)
is 98-102% attributable to the eligibility-trace manipulation on all 15 seeds.

| arm | buf_acc | disc (mean±sd) | val_A | val_B | vrmse | rpe_term | rpe_decoy |
|---|---|---|---|---|---|---|---|
| full | 1.000 | +0.509 ± 0.021 | 0.555 | 0.047 | 0.186 | 0.002 | 0.002 |
| trace_ablated | 1.000 | −0.001 ± 0.005 | 0.257 | 0.258 | 0.186 | 0.005 | 0.005 |
| immediate | 1.000 | +0.590 ± 0.025 | 0.630 | 0.039 | 0.176 | 0.002 | 0.002 |
| misleading_final_cue | 1.000 | +0.509 ± 0.021 | 0.555 | 0.047 | 0.186 | 0.002 | 0.002 |
| shuffled_reward | 1.000 | −0.011 ± 0.033 | 0.298 | 0.310 | 0.204 | 0.002 | 0.002 |

- `full` disc range +0.473..+0.562; `trace_ablated` disc range −0.008..+0.010. The dissociation
  (credit reaches the cue in `full`, collapses in `trace_ablated`, memory intact in both) holds on
  15/15 seeds. `immediate` (post-choice delay 0) is EASIER (higher disc, +0.590) than `full`
  (+0.509) — shorter credit horizon, as expected. `shuffled_reward` learns no cue structure
  (disc ≈ 0) — the instrument does not leak.

**NON-BLOCKING WARNING 1 — `misleading_final_cue` is currently a NO-OP (identical to `full`).**
The final decoy (choice_time−2) is filtered out of BOTH the critic feature set (only {bias, cue})
and the buffer (driven only by the original cue's active window). So a "misleading" final decoy has
no effect: the arm's numbers are byte-identical to `full` on all 15 seeds. Consequence: this arm does
NOT currently test "does the network hold the original cue against a misleading final decoy" — that
control is vacuous as built. There is a real tension: the feature-set restriction that PREVENTS
credit-stealing (filter the decoy) is the same change that makes the misleading-decoy control vacuous.
Claude's call: (a) drop the arm, (b) make the decoy a critic/buffer input (reintroduces the
credit-stealing risk the design removed), or (c) reframe the arm's claim. Flagged in PREREG §8.5.

**NON-BLOCKING WARNING 2 — `buffer_accuracy` is at CEILING (1.000) in every arm and seed.**
The fixed working-memory buffer (MEM_DECAY=0.98) holds the cue perfectly through the delay, so
`buffer_accuracy` does not discriminate between arms. It is a valid sanity check (memory is intact,
and — crucially — it stays 1.000 in `trace_ablated`, confirming the ablation cut CREDIT not MEMORY),
but the discriminating read-out is `value_discrimination`. Flagged in PREREG §8.6.

## 2026-10-04 21:12 -- Step 4b.1 (clear dose) queued

- Branch synced by the owner (RULING T1-sync; tip 45db3fd09); BYB corpus check rc=0
  earlier today. Ran STEP4B as written, step 1.
- 27 lines = 3 candidates x dev seeds 7-15, on the step-4a template at
  50f52c23e (pilot runner byte-identical to branch tip; revision verified present
  on pool41+pool42): --schedule credit_immediate --arms full --n-train 200
  --n-test 100 --lr 0 plus the candidate flag, fixed candidate order:
  (1) --w-clear-fs 60, (2) --clear-ms 500, (3) --clear-pA 1200.
- Out: research/findings/raw/_long_delay_credit_bg/dev_pilot_4b/clear_<name>_s<s>.json
- Shape-check + pool_queue --checked each (mem_gb=2, max_age_h=168): ok=27 fail=0;
  9 dispatched within the minute, T0 no-hold clean, depth 190.
- Next: when the 27 records land, read mem_after_clear (A, B), hold, leak (cue ITI
  vs baseline) and behavior, write the 27-row table and the weakest-knob choice
  here, then STEP4B step 2 (omission build).

## 2026-10-04 23:30 -- step 1 status: 18/27 landed, 10 in flight on pool40, nothing lost

- Records at S: cpa1200 9/9, cms500 8/9, wfs60 1/9 = 18/27.
- Probed every pool node over ssh: the 10 missing are NOT dead - wfs60 s7-s13,
  s15 and cms500 s15 are still RUNNING on pool40 (3 procs each, correct
  revision 50f52c23e, mem_gb=2 per line). queue depth for dev_pilot_4b = 0
  (all 27 dispatched; the 21:04 add reported ok=27 fail=0).
- clear_wfs60_s14.json finished on pool40 and was stranded there: rsync is
  absent on S so pool_sync.sh --dry-run transfers nothing; pulled it by ssh
  pipe (pull-only) with its .prov.json, JSON parses (8560 bytes).
- No requeue: the runs are healthy and in flight. The 27-row table + weakest-
  knob choice waits for 27/27 (next wake after the in-flight lines land).


## 2026-10-05 00:46 -- Step 4b.1 (clear dose): 27/27 landed, TABLE + CHOICE

- Records complete: cpa1200 9/9, cms500 9/9, wfs60 9/9 (the pool40 in-flight
  lines landed; nothing requeued). Copied into this worktree for the commit.
- Section-1 rule (STEP4B, fixed order w-clear-fs 60 -> clear-ms 500 ->
  clear-pA 1200; first with mem_after_clear_hz < 1.0 AND mem_hold_acc >= 0.95
  on all nine seeds): ALL THREE candidates pass 9/9, so the fixed order picks
  **--w-clear-fs 60** (params confirm w_clear_fs=60.0 vs 30.0 on the other
  two rows). That flag goes on every Step 4b line.
- Observations (marked: readings, not verdicts): after_clear reads 0.000 on
  all 27 cells INCLUDING the two w_clear_fs=30 rows, so this run does not
  separate the knobs on that instrument (PILOT step-3 def read 0.0493 Hz under
  its own one-knob harness -- different harness, not comparable directly);
  seed 7 reads identical across wfs60 and cpa1200 on every listed field
  (other seeds differ) -- quirk noted, not investigated. hold=1.000 27/27;
  behavior sits at chance (expected: lr 0 diagnostic cells, not the credit
  grid); ITI cue/mem attribution 0.000 and da_iti_offset ~ +-1e-4 everywhere.
- Table (27 rows):

| candidate | seed | after_clear_hz | hold_acc | hold_probe | success | resp | pA | pB | iti cue | iti mem | da_iti_off |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wfs60 | 7 | 0.000 | 1.000 | 1.000 | 0.53 | 0.97 | 0.43 | 0.42 | 0.000 | 0.000 | -0.000021 |
| wfs60 | 8 | 0.000 | 1.000 | 1.000 | 0.54 | 0.95 | 0.57 | 0.50 | 0.000 | 0.000 | 0.000054 |
| wfs60 | 9 | 0.000 | 1.000 | 1.000 | 0.42 | 0.95 | 0.59 | 0.78 | 0.000 | 0.000 | -0.000044 |
| wfs60 | 10 | 0.000 | 1.000 | 1.000 | 0.49 | 0.97 | 0.33 | 0.59 | 0.000 | 0.000 | -0.000021 |
| wfs60 | 11 | 0.000 | 1.000 | 1.000 | 0.62 | 0.96 | 0.68 | 0.44 | 0.000 | 0.000 | -0.000008 |
| wfs60 | 12 | 0.000 | 1.000 | 1.000 | 0.55 | 0.99 | 0.62 | 0.31 | 0.000 | 0.000 | 0.000009 |
| wfs60 | 13 | 0.000 | 1.000 | 1.000 | 0.48 | 0.98 | 0.52 | 0.63 | 0.000 | 0.000 | -0.000028 |
| wfs60 | 14 | 0.000 | 1.000 | 1.000 | 0.53 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | -0.000061 |
| wfs60 | 15 | 0.000 | 1.000 | 1.000 | 0.39 | 0.99 | 0.58 | 0.52 | 0.000 | 0.000 | 0.000057 |
| **wfs60** | rule | after<1.0 & hold>=0.95 on 9/9: PASS | | | | | | | | | |
| cms500 | 7 | 0.000 | 1.000 | 1.000 | 0.50 | 0.95 | 0.53 | 0.44 | 0.000 | 0.000 | -0.000027 |
| cms500 | 8 | 0.000 | 1.000 | 1.000 | 0.45 | 0.96 | 0.71 | 0.41 | 0.000 | 0.000 | 0.000049 |
| cms500 | 9 | 0.000 | 1.000 | 1.000 | 0.57 | 0.99 | 0.57 | 0.56 | 0.000 | 0.000 | -0.000086 |
| cms500 | 10 | 0.000 | 1.000 | 1.000 | 0.44 | 0.96 | 0.55 | 0.82 | 0.000 | 0.000 | -0.000018 |
| cms500 | 11 | 0.000 | 1.000 | 1.000 | 0.57 | 0.95 | 0.73 | 0.56 | 0.000 | 0.000 | -0.000037 |
| cms500 | 12 | 0.000 | 1.000 | 1.000 | 0.51 | 0.97 | 0.54 | 0.53 | 0.000 | 0.000 | 0.000024 |
| cms500 | 13 | 0.000 | 1.000 | 1.000 | 0.51 | 0.97 | 0.67 | 0.68 | 0.000 | 0.000 | -0.000034 |
| cms500 | 14 | 0.000 | 1.000 | 1.000 | 0.53 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | -0.000065 |
| cms500 | 15 | 0.000 | 1.000 | 1.000 | 0.49 | 0.99 | 0.67 | 0.64 | 0.000 | 0.000 | 0.000029 |
| **cms500** | rule | after<1.0 & hold>=0.95 on 9/9: PASS | | | | | | | | | |
| cpa1200 | 7 | 0.000 | 1.000 | 1.000 | 0.53 | 0.97 | 0.43 | 0.42 | 0.000 | 0.000 | -0.000021 |
| cpa1200 | 8 | 0.000 | 1.000 | 1.000 | 0.52 | 0.93 | 0.60 | 0.50 | 0.000 | 0.000 | 0.000054 |
| cpa1200 | 9 | 0.000 | 1.000 | 1.000 | 0.41 | 0.93 | 0.55 | 0.76 | 0.000 | 0.000 | -0.000042 |
| cpa1200 | 10 | 0.000 | 1.000 | 1.000 | 0.47 | 0.96 | 0.41 | 0.59 | 0.000 | 0.000 | -0.000028 |
| cpa1200 | 11 | 0.000 | 1.000 | 1.000 | 0.63 | 0.99 | 0.68 | 0.44 | 0.000 | 0.000 | -0.000007 |
| cpa1200 | 12 | 0.000 | 1.000 | 1.000 | 0.55 | 0.99 | 0.65 | 0.31 | 0.000 | 0.000 | 0.000009 |
| cpa1200 | 13 | 0.000 | 1.000 | 1.000 | 0.46 | 0.96 | 0.52 | 0.63 | 0.000 | 0.000 | -0.000028 |
| cpa1200 | 14 | 0.000 | 1.000 | 1.000 | 0.53 | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 | -0.000064 |
| cpa1200 | 15 | 0.000 | 1.000 | 1.000 | 0.40 | 0.98 | 0.58 | 0.54 | 0.000 | 0.000 | 0.000057 |
| **cpa1200** | rule | after<1.0 & hold>=0.95 on 9/9: PASS | | | | | | | | | |

**Section-1 choice (fixed order): `wfs60` — first candidate passing after_clear<1.0 AND hold>=0.95 on all nine seeds.**
params@wfs60: w_clear_fs=60.0 clear_ms=None clear_pA=600.0 (schedule=credit_immediate arm=full)
params@cms500: w_clear_fs=30.0 clear_ms=None clear_pA=600.0 (schedule=credit_immediate arm=full)
params@cpa1200: w_clear_fs=30.0 clear_ms=None clear_pA=1200.0 (schedule=credit_immediate arm=full)
VERDICT: {'wfs60': True, 'cms500': True, 'cpa1200': True} | chosen=wfs60

- Next: STEP4B step 2 (omission build: the two runner copies, seed-7-only
  constant tuning, freeze, engagement check).
- COMMIT STOP: the 27 records + provs staged here hit the `below-chance`
  failure gate (21 problems: lr-0 diagnostic cells read < 0.5 on every arm;
  expected at lr 0, but the artifact carries no below_chance/UNDEFINED
  declaration and adding one post-hoc, or editing the runner, is reserved).
  Per section 3: NEEDS-CLAUDE `[T1]` filed with the gate message; this
  WORKLOG commits alone (Markdown exempt); the 54 record files stay STAGED
  on research/draft-ldc-outcome-timing for Claude's data ruling.

## 2026-10-05 02:51 -- step 4b.1 records land: RULING T1-lr0 applied, 54 files committed

- agent_sync_branch merged main @ 6a173b635 as 350e1e0c8 (clean).
- `git am 0001-ldc-pilot-lr0-declaration.patch` -> a8ecbe7d2; pin test 4 passed.
- `--declare-lr0 clear_*_s*.json`: 27/27 lr-0 records DECLARED (behavior_verdict UNDEFINED + declaring-runner sha256). The 27 .prov.json paths matched the ruling's own glob and were REFUSED by design (no rows; rc=2 from the no-rows branch, nothing written to them) -- not an lr>0 refusal.
- The below-chance gate now reads UNDEFINED for the lr-0 cells; this commit is the gate test. Choice `--w-clear-fs 60` (section 1) unchanged.
- Trigger: main landing wake 2026-10-05 02:41 (24 commits, session7-iw2).

## 2026-10-05 03:26 -- section 2 (Build, copies only): DONE, identity test 3/3

- `research/runners/_ldc_credit_omission.py` = copy of `bg_corticostriatal_credit.py` + 10 CreditParams
  fields (`omission_route=0`, n_lhb 20, n_lhb_inh 10, n_rmtg 20, w_outc_lhb 0, w_rus_lhbinh 9.0 = the
  2026-08-01 REWARDUS_TO_RMTG_W, w_lhbinh_lhb 12.0 = RMTG_TO_OMIT_W, w_lhb_rmtg 0, w_rmtg_snc 0,
  cut_rus_lhbinh 0) and the route-on block in `credit_extension`: regions lhb (RS) / lhb_inh (FS) /
  rmtg (FS); pathways ctx_o->lhb, reward_us->lhb_inh (skipped when cut), lhb_inh->lhb (gaba_a),
  lhb->rmtg, rmtg->snc (gaba_a), all fixed. Densities 0.6/0.5, jit 0.2/0.5 mirror the 2026-08-01 relay
  build (STEP4B names no densities). The three seed-7 tuning knobs default 0 so route-0 stays pure.
- `research/runners/_ldc_omission_pilot.py` = copy of the PATCHED (RULING T1-lr0) outcome-timing pilot
  importing from `_ldc_credit_omission`: auto CLI flags for the new fields, RUNNER_SHA256 + _git_state
  paths retargeted, declare_lr0 `by` string retargeted (declaration rides along), and the new records
  `lhb_rate_rewarded` / `lhb_rate_unrewarded` (LHb spikes in the OUTCOME_MS window / (n_lhb x 0.1 s),
  split by train-trial outcome; None when the LHb is absent) + `membrane_ok` /
  `membrane_bad_regions` (membrane_bound_check strict=False on lhb, lhb_inh, rmtg, snc after training).
- `tests/test_ldc_omission_identity.py`: route-0 at seed 7 byte-identical (region list +
  cp_connections indptr/indices/data/nnz), route-1 adds exactly the three regions + four checked
  pathways + reward_us->lhb_inh, cut drops only that one. 3 passed in 1.15 s.
- Next (section 2): seed-7-only tune of w_outc_lhb / w_lhb_rmtg / w_rmtg_snc at credit_immediate,
  --reward-on-ms 150 --outcome-ms 250 (the step-4a pairing from the on150 prov), w_outc_strio 30,
  n-train 200, then the engagement check and the frozen-constants table before any seed 8-15 line.
- Trigger: 03:11 continuation of the 02:41 landing wake.

## 2026-10-05 03:59 -- section 2 (tune): seed-7 lattice QUEUED on the pool

- Local probe of on-v1 (30/20/12, n-train 200) did not finish within 420 s on S, so per RULING
  compute-placement the tune runs go to the pool (step-1 precedent). Provision of this branch's build
  commit `eb4730cce16589b704a76f49a5b6d66e11162240` came back state=ok on pool41, pool42, pool40, wspool.
- 4 shape-checked lines queued from S at eb4730cce (each: credit_immediate, full arm, seed 7 only,
  n-train 200, n-test 100, --reward-on-ms 150 --outcome-ms 250 (the step-4a pairing from the on150 prov
  sidecar), w_outc_strio 30, w-clear-fs 60, mem_gb=2, max_age_h=48):
  `tune_off_base_s7.json` (route 0), `tune_on_v1_s7.json` (route 1, w_outc_lhb/w_lhb_rmtg/w_rmtg_snc
  30/20/12), `tune_on_v2_s7.json` (30/20/24), `tune_on_v3_s7.json` (60/30/12). Depth 125, FAIL=0.
- Also executed RULING transitive-engagement: withdrew the 12 barred `s2p-R-tr-` lines verbatim
  (`--expect 12` honoured, withdrawn.log, depth 137->125); word-role 12 untouched. T19 re-generation at
  the new pin is the next wake's task.
- Next (section 2): read the 4 tune JSONs on arrival, check U (unrewarded in-window DA min <= 0.47),
  R (rewarded max >= 0.60), I (|ITI offset probe| <= 0.005), membrane_ok, and engagement
  (lhb_rate_unrewarded > 2x lhb_rate_rewarded AND route-on unrewarded DA min < route-off by >= 0.02),
  then freeze the tuned constants in this WORKLOG before any seed 8-15 line.
- Trigger: 03:57 provisioning-ok wake.

## 2026-10-05 05:13 -- section 2 (tune): round-1 readout + MiMo Pro verdict; NEEDS [T1] ASK filed

Round 1 landed (4 cells, seed 7, credit_immediate, onset 150/outcome 250, w_outc_strio 30, clear 60,
wall ~2000 s each; pulled to S under `research/findings/raw/_long_delay_credit_bg/dev_omission/`):

| cell | w_outc_lhb/w_lhb_rmtg/w_rmtg_snc | U min (<=.47) | R max (>=.60) | I (<=.005) | LHb U/R | mem | U-drop vs off |
|---|---|---|---|---|---|---|---|
| off route0 | - | 0.4982 FAIL | 0.6431 PASS | 2.45e-05 PASS | n/a | True | - |
| on v1 | 30/20/12 | 0.3395 PASS | 0.5070 FAIL | 4.11e-05 PASS | 1.04x FAIL | True | +0.1587 |
| on v2 | 30/20/24 | 0.2990 PASS | 0.5012 FAIL | 1.87e-05 PASS | 1.04x FAIL | True | +0.1992 |
| on v3 | 60/30/12 | 0.2963 PASS | 0.5011 FAIL | 2.01e-05 PASS | 1.02x FAIL | True | +0.2019 |

Route-off reproduces the step-4a anchor exactly (U 0.498 is the PILOT table's `.498 x` cell), so the
instrument is calibrated. Route-on: U, I, membrane_ok and engagement (b) all PASS; R fails everywhere and
engagement (a) fails (1.04x vs >2x). Test success (n=100): off 0.38, v1 0.53, v2 0.52, v3 0.44 -- the route
does not damage the task. void_floor_ceiling True on all four INCLUDING route-off (D2 selector weights
85-88 vs w_max 90, D1 ~0): pre-existing saturation, flagged to Claude.

**MiMo Pro delegate verdict (2026-10-05, one delegate per RULING models):** the route engages as a TONIC
brake, not a reward-modulated omission signal; R is the binding bar. The reward-side release arm
(`w_rus_lhbinh`/`w_lhbinh_lhb`, still 9/12 everywhere) was never varied -- the 50 ms reward_us pulse
suppresses LHb only ~17%, so RMTg->SNc inhibition runs through the rewarded DA peak (0.643 -> 0.507).
Rewarded max is PINNED at 0.501-0.507 across terminal brake 12 vs 24 and drives 30/20 vs 60/30 --
strengthening the brake is a dead end; the dip stays dose-responsive. Engagement (a) as written is
**structurally unreachable**: the only rewarded/unrewarded difference inside the 250 ms window is the 50 ms
pulse, so even total silencing caps the mean-rate ratio at 250/200 = 1.25 (measured 1.04). Fix the
instrument (reward sub-window / pulse-locked Hz / slow GABA_B release), cap `w_outc_lhb` for physiology
(LHb at 61-97 Hz sustained vs ~1-10 Hz tonic in vivo), and resolve the pre-existing saturation before any
12-seed grid, else every frozen row voids.

**NEEDS [T1] ASK filed on main `83efc766c`**: bar (a) fix is section-2 reserved (bars). Freeze therefore
waits on Claude's ruling; the weight tune continues meanwhile.

**Round-2 lattice queued** (4 shape-checked lines at eb4730cce, records cite the delegate verdict):
- `tune_r1_release_x3_s7.json` -- w 30/27/24/20/12 (release arm x3 -- the one untested R lever)
- `tune_r2_brake_half_s7.json` -- 30/9/12/20/6 (halved terminal brake; dose-response point)
- `tune_r3_relhalf_s7.json` -- 30/27/24/20/6 (release + half brake, the freeze candidate)
- `tune_r4_lowrate_s7.json` -- 15/18/18/40/12 (physiological LHb rate, compensated dip)

- Next: on arrival pick the winner by R>=0.60 while holding U<=0.47, I, membrane_ok; freeze only after the
  bar-(a) ruling. Round-2 is the last weight-only round before the instrument question must resolve.
- Trigger: 04:59 results wake (tune round 1 landed).

## 2026-10-05 06:07 -- section 2 (tune): round-2 readout; round-3 lattice queued

Round 2 landed (seed 7, same schedule; wall ~2300-2420 s each):

| cell | w outc/rus/lhbinh/rmtg/snc | U (<=.47) | R (>=.60) | I | LHb U/R | mem | drop |
|---|---|---|---|---|---|---|---|
| r1 release_x3 | 30/27/24/20/12 | 0.3391 PASS | 0.5294 FAIL | PASS | 1.23x | True | 0.159 |
| r2 brake_half | 30/9/12/20/6 | 0.4204 PASS | 0.5740 FAIL | PASS | 1.04x | True | 0.078 |
| r3 relhalf | 30/27/24/20/6 | 0.4205 PASS | 0.5835 FAIL | PASS | 1.23x | True | 0.078 |
| r4 lowrate | 15/18/18/40/12 | 0.3284 PASS | 0.5166 FAIL | PASS | 1.22x | True | 0.170 |

All four: I PASS, membrane_ok True, eng(b) PASS. The delegate's mechanism holds: the release arm moved the
LHb ratio 1.04 -> 1.23 (structural ceiling 1.25) and R +0.022 (r1 vs r1-round1), the terminal brake moved R
+0.067 (r2 vs r1-round1), and the combination r3 is best at 0.5835 -- miss 0.0165. LHb unrewarded rate
63.4 Hz on all except r4's 37.2 Hz. void_floor_ceiling True everywhere (NEEDS [T1] open, Claude).

**Round-3 lattice queued** (4 shape-checked lines at eb4730cce) -- r3 family, release 27/24 pinned, ladder the
terminal brake with LHb->rmtg compensation so U margin and R peak decouple:
- `tune_r5_rel_snc3_s7.json` -- 30/27/24/20/3 (terminal halved again)
- `tune_r6_rel_snc4_s7.json` -- 30/27/24/20/4 (mid rung)
- `tune_r7_rel_rmtg40_snc3_s7.json` -- 30/27/24/40/3 (deeper dip restored, peak still freed)
- `tune_r8_ceiling_s7.json` -- 30/27/24/60/2 (ceiling probe: if R still <0.60 here, something other than the
  brake caps the rewarded peak and the design goes back to the delegate)

Prediction to check on arrival: r5/r7 R ~0.60-0.63 while U holds <=0.47; r8 R even higher. Winner freezes
only after the bar-(a) ruling (NEEDS [T1], main 83efc766c).
- Trigger: 06:02 results wake (round 2 landed).
