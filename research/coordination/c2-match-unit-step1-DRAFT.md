# C2 step 1 draft: match-unit conjunction (design panel 2026-11-01; Claude ruling c2-match-reset-1101 applies)

Status: DRAFT. Produced by a research + three-design + adversarial-review panel (all three designs scored 3/10 as written). Claude's ruling c2-match-reset-1101 (runbook) decides the order: the vectorized gate-state reset (branch research/c2-gate-reset 3ec75e204, BRAIN_C2_GATE_RESET) is measured ALONE first on the dev seeds; the S0a/S0b probes and the role relay below wait for that wave's score.

## C2 step 1: match-unit conjunction (draft for Claude's ruling)

**Defect (read (b), wave `_c2_step0_relay`, seeds 7,1-6, score `_scratch/c2_relay_score.json` = STOP).** Read (c) is fixed. Read (b) fails per block in two ways:
- **Silent true matches.** 4 of 98 true-match blocks have m{b} at exactly 0.0, all on cue frog,eat (seed 4 infer/infer_dup/infer_xtier, seed 5 infer_amb2).
- **False fires.** 57 POST lines. When each block is classified by its lit decoded words:
  - 51 of 86 false blocks with both cue words lit fire at 0.153-0.31, the same range as true matches.
  - 31 blocks with one cue word lit fire at 0.067-0.141. 27 of these 31 had m{b} firing on the previous cue.
  - 0 of 1045 neither-lit blocks fire.
  - 35 both-lit false blocks also read 0.0, which is not explained.

**Diagnosis.** m{b} is not mainly a leaky AND. It is a near-exact AND over multi-hot decoded drives:
- Every lit word gets 1500 pA: `np.where(acc > 0, hi_pA, 0.0)` (`_phaseC_S5_divnorm_derisk.py:181`).
- Each role lights 2-8 words per block.
- The one-word fires look like state left over from the previous query. `reset_sequencerK_state` zeroes the scalar `c["ema"]`/`c["last_value"]` (`_phaseB_onebrain_sequencerK_derisk.py:197-199`). The vectorized path, which is the one actually running, keeps its EMA in `sb._gate_coupling_state` arrays (`sim/bridge.py:6106-6116`, docstring: "the scalar dict ema/last_value are unused while vectorized"), and those arrays are never reset.
- The decision is a host step: `fired = r > match_thresh` then `winner = next(...)` (`_seq_vocab_shrink_derisk.py:300-302`).
- The arm's replies come from the host `_scan` (`integrated_loop=False`, `one_brain_composer.py:1715-1722`), so read (b) does not change any reply yet.

### Ranking of the three designs (all reviews scored 3)

- **Inhibition design: keep stage 1 and the reset idea as the PRIMARY; drop stages 2-3.**
  - Its review refuted stages 2 and 3. Stage 2 cannot detect coincidence because the drive is tonic: the 60-step constant-current settle means it is still a threshold on the summed input, which is the 06-19 wall. Stage 3's WTA breaks the amb-table spec, which requires both true blocks for p_prey_eat to stay active so the cue can abstain (packet :1144-1147). Its readout is also still a host argmax.
  - Stage 1 is the record's banked fix for this disease: decoded-line competition, `drive_frac=0.9` at K=8/16, and the patient FF relay that already fixed read (c). All three reviews recommend it first.
- **Dendritic design: PARALLEL, in its review's fixable form.** That form is a single existing subunit, `coincidence_detector` with K between one and two lines' synchronous counts, on today's engine. This is the 6/6-seed GO two-input AND (2026-08-08 pragmatics leg1). The two-branch engine change, the 0.6x/1.2x soma sum (the 06-19 wall) and graded drive into a count detector (anti-rate, review fatal 1) are dropped.
- **Learned design: LOST.**
  - Selectivity comes from the host replaying stored fact text, with a host-aimed instructive current into mA/mX. This is host ownership of the computation, against the 09-25 phase.
  - Its margin arithmetic gives about 293, below today's 300, not 1.5x it.
  - Its lesion arms cannot show that the decoded memory is load-bearing.
  - Bank it as a later candidate only once exposure happens at hearing time.

### Step 0 (read-only, before any build): driver runs, then STOP

- **S0a. Seed-4 block-6 trace, cue frog,eat.** Per step: d/mw/mA6/mX6/gblk6/m6 spike rasters, plus the vectorized `ema`/`last_value` for g6·/gblk6. Repeat on seed 5 amb2 block 7. Purpose: decide whether the 0.0 misses come from upstream or from gate state.
- **S0b. Separability of `acc` at the drive level.** Log `acc` from `onbridge_divnorm_drive` (returned at :182) per block, role and word. Compute the cross-block AUC (per seed and pooled) of true-filler `acc` against spurious-cue-word `acc` in the 86 both-lit false blocks. Also compute the within-block margin.
  - Pooled AUC < 0.9 refutes any absolute threshold.
  - Within-block margin > 0 in at least 95% of blocks supports competition (the primary).
- **S0c.** Freeze the flag-OFF block classification (98 / 86 / 31 one-word / 1045) as fixed denominators for every bar below.

### Primary: per-role competition on `_seq_drives` + vectorized gate reset

**Flag:** `BRAIN_C2_ROLE_RELAY` (default OFF; composer kwarg `enable_seq_role_relay=False`).

**Change sites**
- `research/runners/_c2_patient_inhib.py`: add `composer_role_relay(buf, slice, role)`, generalising `composer_patient_relay` (:204) to the agent and action rows, plus `composer_role_relay_enabled()`.
- `research/runners/one_brain_composer.py:1673-1694`: when the flag is on, always take `with_patient=True`-style full rows, then relay the agent and action slices the same way `ap` is relayed. The lit set becomes the set of relay pools that spiked, so there is no host top-k.
- `research/runners/_phaseB_onebrain_sequencerK_derisk.py:197-199`: when the flag is on, also reset `sb._gate_coupling_state`.
  - Set `ema` to 0 and `last_value` to NaN, the init value per `sim/bridge.py:6106-6116`. NaN forces every gate to be rewritten closed.
  - Do NOT zero `last_value`: per the inhibition review, zeroing it leaves an open gate stuck open.
  - This is a separate sub-flag, `BRAIN_C2_GATE_RESET`, so the two levers can be ablated apart.

**Lesion arms**
- L1: role relay ON with its I→E weight set to 0. This should bring back the multi-hot lit sets and the both-lit false fires.
- L2: gate reset only, no relay. This attributes the one-word fires.
- L3 (memory load-bearing): the relay fed another block's cleanup row. True-match fires must collapse.

**Dev smoke (seeds 1-7, 5 tables × 4 cues, `c2_hops_step0 --score` read (b)+(c), counted per block against the frozen S0c denominators)**
- P1. Flag OFF: per-block m rates byte-identical to `_c2_step0_relay`.
- P2. True match: at least 95 of 98 fire, with a stated count for the 4 frog,eat blocks (no bar on them until S0a explains the zeros).
- P3. Both-lit false: at most 5 of 86 fire. One-word-lit: at most 2 of 31 fire (with reset on). Neither-lit: at most 2 of 1045 fire.
- P4. Responsive-range guard: true-match m rates have median ≥ 0.15, and no true-match block in 0.0 < r < 0.10. Relay pinh spike counts stay within 5-95% of the ceiling on the agent and action slices (the 11-01 pinned-pool lesson).
- P5. Read (c) unchanged: `S2_patient_pool` = 0, and patient lit sets identical to the flag-OFF run.
- P6. L1 recovers at least 30 of the 51 both-lit false fires. L2 alone removes at least 20 of the 31 one-word fires. L3 fails P2.
- P7. Amb tables: both true blocks of every ambiguous cue still fire (`AMBIGUOUS_CUES`, `c2_hops_step0.py:76`).

### Parallel: single-subunit coincidence AND for m{b}

**Flag:** `BRAIN_C2_COINC_MATCH` (default OFF).

**Change sites**
- `_seq_vocab_shrink_derisk.py` `build_sequencerK_reduced_bridge` (wiring :141-177): when the flag is on, route mA{b} and mX{b} to m{b} as coincidence-routed pathways on the existing subunit (`sim/config.py:456-471`). K is set from a Step-0 single-cell probe on this fabric's GENERIC_UNSTRUCTURED cells, between one pool's and two pools' synchronous counts.
- gblk{b} is removed. Binary drive is kept, with no graded drive into a count detector.
- `match_thresh` and `next(...)` (:300-302) are logged only.

**Lesion arm:** coincidence off (plain summed excitation at today's weight). This should reproduce the one-word leak fires that are present when the reset is off.

**Smoke:** the same P1-P5 and P7, run on the reset-on/relay-off build, so the parallel method is scored on the one-word class plus the misses. It is not expected to fix the 51 both-lit fires, and that is stated in advance.

### Driver steps

1. Branch `research/c2-match-conjunction` in a worktree. Commit the S0a/S0b/S0c probe scripts first, read-only, and run them via `tools/gpu_queue.sh`. STOP, and post the S0 numbers to the runbook NEEDS CLAUDE.
2. On Claude's go: build the primary (flag + sub-flag + lesion arms) and add tests that flag-OFF output is byte-identical. Commit.
3. Run the smoke on dev seeds 1-7: flag OFF, primary, L1, L2, L3. CPU/GPU go through the queue under `memcap.sh`.
4. Build the parallel method on the same branch behind its own flag and smoke it.
5. STOP. Report the per-block tallies only, with no GO claim. Gate seeds 42/43/44/100/101/102 are a separate prereg.

### Open questions for Claude

1. If S0b's pooled AUC is below 0.9 but the within-block margin is above 0, is per-block competition acceptable even though it cannot generalise across blocks?
2. Which host steps go to the substrate this step? The `match_thresh` + `next(...)` readout, and replies via the host `_scan`. Under integration-first, read (b) changes no reply until the reply is routed through ans/abstain. Should that be step 1 or step 2?
3. Should the gate-state reset be treated as a scaffold (a host zeroing), or should it be a longer physical drain?
4. What is the in-scope criterion for the 0.0 misses and the 35 silent both-lit blocks if S0a points upstream of m{b}?
5. Is the relay that pins at its ceiling (finding 11-01) an acceptable base for agent/action, or does it need its fix first?

Checked lines: `_phaseC_S5_divnorm_derisk.py:181`, `_seq_vocab_shrink_derisk.py:300-302`, `_phaseB_onebrain_sequencerK_derisk.py:197-199`, `sim/bridge.py:6106-6116`, `_c2_patient_inhib.py:204`, `one_brain_composer.py:1673-1694`. Other file:line references are taken from the factual base and were not re-read.