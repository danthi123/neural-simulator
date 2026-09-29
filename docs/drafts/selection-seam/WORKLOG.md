# WORKLOG: selection seam with no host authority (plan step 3, lane A)

Branch `research/selection-seam-need-reversal`, worktree `.claude/worktrees/selection-seam-need-reversal`.

## 2026-09-29

- 12:05 Worktree created from main `f52c6d246`; `research/coordination/lane_builds.jsonl` copied from the primary
  checkout (holds the lane-A registration) and left modified, to be staged with the first non-Markdown commit.
- 12:10 Record check: `bash tools/before_you_build.sh "state-dependent winner reversal: interoceptive need ..."`;
  RAG queries (need/satiety/striatum, competing drives, GNW ignition retire, g11 homeostatic). Read:
  `2026-06-17-homeostatic-spiking-drive-mechanism-GO.md` (AgRP/POMC pools on spikes),
  `2026-08-19-embodied-affect-interoception-GO.md` (body -> interoceptive relay -> synapses),
  `2026-09-03-gnw-bus-family-retire-status-ledger-drift-CORRECTED.md`, `2026-08-03-intrinsic-vocal-reversal-4seed-NO-GO.md`
  and the Gate-B vocal-credit series (v1-v14: reward credit on this selector never reached GO), the production organs
  `bg_action_selection_production_organ.py`, `value_choice_production_organ.py`, the workspace builder.
- 12:13-12:20 Review of the local model's N-channel work (DEV probes, seed 7, numpy; scripts in `raw/`):
  the N=5 commit pools sit at -377 mV and fire at the refractory ceiling with arousal off -> an explicit-step
  overshoot of four summed GABA-A projections; normalizing the cross-inhibition (40/(N-1)) removes it at the default
  tonic (commit 0.83-1.0). Review: `REVIEW-of-local-model-nchannel-bg.md`.
- 12:18 Added `tools/lab.py::membrane_bound_check` + `tests/test_membrane_bound_check.py` (5 pass) and the
  FAILURE_LOG row (coverage gate resolves it).
- 12:20 Built `research/runners/_need_reversal_selection_seam.py` + `tests/test_need_reversal_selection_seam.py`.
  The first version of the production-equivalence test failed because it interleaved two live bridges (a build
  reseeds a shared noise stream); run sequentially, the harness trial reproduces `_run_biased_trial` exactly.
  5 tests pass (race == production race; static audit clean; touch audit sees the host rule and not the synaptic arm;
  host-variable flip is byte-identical in the synaptic arm and changes the host arm).
- 12:21-12:30 DEV tune, AMPA need synapses, seeds 1-6 x 20 trials (`research/findings/raw/_need_reversal_seam/dev_tune/`):
  selectivity and arousal gating trade off. w=60/400 pA: mean R 0.90 (min 0.80) but need alone commits on up to 25%
  of no-arousal trials; w=80/400: R 1.0 but need alone commits 100% (the need drives rather than biases); w=40/400:
  R 0.66, gating intact. NMDA (voltage-gated, the MSN up-state) smoke on seed 1: w=20/400 R 0.75, no-arousal 0.08;
  w=40 need alone commits; w=160 trips the membrane bound (-283 mV). UNSURE: whether to require arousal gating as a
  gate or report it as a mechanism ablation (decided in the prereg; see below).
- 12:25-12:31 DEV tune 2 (12 seeds x 20 trials; `dev_tune2/`): NMDA w=20 / 400 pA: mean R 0.90 (min 0.85), P(forage|
  hungry) >= 0.85, P(forage|sated) <= 0.15, P(forage|h=0.5) 0.54, no-arousal commit mean 0.07 (max 0.20). AMPA
  w=60/350: mean R 0.89 with no-arousal mean 0.07 / max 0.20; w=70/350 no-arousal max 0.60. Chose NMDA 20/400/0.5
  (the voltage-gated route is the faithful up-state mechanism and at least as well gated). Frozen as runner defaults.
- 12:28 Pipeline smoke (seed 7, 6 trials): all gates computable. DEV observation: the GPe and STN fire 0 spikes in the
  intact Gate-A v2 race (the GPe gets no tonic current), so inactivating them is byte-identical; added a NOT ENGAGED
  label to the necessity rule and recorded every ablation target's firing in every block. OU noise off -> 0 commits
  (seed 7): without proposal noise the two channels cross together.
- 12:29 Verdict code: gates G1-G7 decide GO/NO-GO; unevaluable gates and failed knob read-backs go through
  `tools/verdict.Verdict` as preconditions (UNDEFINED, never NO-GO).
- 12:31 PREREGISTRATION written: `research/findings/2026-09-29-need-reversal-selection-seam-PREREGISTRATION.md`
  (first draft sha256 3b08bea547c2ea57; 12:34 added the reported-only production anchor, before any gate seed; the runner is pinned by the commit that adds the prereg). NOT YET COMMITTED: blocked by
  gates/compute_idle_persistent (pool idle, 22 cores; a pool job needs a pushed revision, which needs a commit).
  The GPU is busy with my cupy DEV battery (AMPA 60/300, seeds 1-4, queued ~12:24).
- 12:33 Killed my own cupy DEV battery on the GPU (pid 1579616, AMPA 60/300, superseded operating point, ~47 min
  projected at 18% GPU) because lane C's host-share GPU job was queued behind it. No output kept. A cupy replicate at
  the frozen operating point is to be re-queued as a short job (extremes + controls, no ablations).
- 12:34 Added `production_baseline` (reported only): the live `BGActionSelector` host share. Seed 7, 20 trials: host
  salience flip moves the committed winner 0.0 -> 1.0 SPEAK (HS = 1.0); the host decides content-bearing messages
  without consulting the race (pre-filter). Added to the prereg's reported measures before any gate seed.
- 12:33 Lane C's queued GPU job (host-share-baselines) started when mine was killed and was REFUSED by its own
  `mem_ok.sh 12` (avail 14G after 9G committed-unused), rc=1, and left the queue. Not re-queued by me (another lane's
  job); flagged to the coordinator. My rehearsal processes use ~0.26 GB each.
- 12:34 Queued a short cupy DEV replicate at the frozen operating point (seeds 1-3, 40 trials, no ablations/curve).
- 12:35 attribution-required gate: added `attributable_to` for the reversal (need synapses, mapping) and the commit
  (arousal, direct path). Dry-run of the pre-commit hook on the staged set now fails ONLY compute-idle-persistent.
- 12:36 Added `closed_loop_episode` (reported only; one persistent bridge, deficit +0.08/trial, forage -0.35).
  Seed 7, 100 trials: intact corr(deficit, forage) 0.45, P(forage | deficit >= 0.6) 0.67 vs <= 0.4: 0.09; need
  synapses closed: corr 0.18, 0.57 vs 0.25. Added to the prereg's reported measures (still before any gate seed).
- 12:32-now DEV rehearsal of the full preregistered battery, seeds 1-12, 60 trials, numpy, local, 6 parallel
  (started on the runner version before the attribution / production-anchor / closed-loop additions).
- 12:40-12:50 DEV rehearsal seeds 1-6 (60 trials, full battery; ~8.5 min per seed at 6 parallel): 6/6 seed GO under
  the frozen bars (DEV, not a verdict). R 0.83-0.95; need lesion R 0.0 exactly (the gate closed, the body current
  reaches nothing, so both extremes replay the same noise); shuffled synapses R -0.10..+0.08; swapped interoception
  R = -R(intact) exactly (swapped at h=1 IS intact at h=0 current-for-current: by construction, see note); HS before
  0.90-1.0, after 0.0; NS after 0.83-1.0, before 0.0. Decoding (screen): synaptic arm host variable 0.50 (chance by
  construction), interoceptive differential 0.81-0.88, D1 differential 0.92-0.96; host arm host variable 0.85-0.89 AND
  interoceptive differential 0.86-0.91 although the relays reach nothing there (h drives both: the confound plan
  section 2 warns about). Host override (host injects 600 pA for the non-preferred option): R falls to -0.2..-0.5,
  i.e. the host would win often if it re-entered. Necessity: NECESSARY need synapses, interoceptive relays, arousal,
  direct path, thalamus, OU noise (no proposal noise -> both motor pools cross on the same step, 0 commits); NOT
  ENGAGED GPe, STN (silent in the intact race); NOT NECESSARY D2 (fires ~13 spikes/trial but its only target, the GPe,
  is silent), commit cross-inhibition and commit NMDA (the commit is read at the first motor crossing, before either
  engages). Those three ablations leave commit and R identical to intact on every seed.
  NOTE for the finding: G5, HS(after)=0 and the need-lesion R=0 hold by construction once the code path is clean;
  the gates that can fail on data are G1, G2, G4, the NS(after) and HS(before) magnitudes, and G6 (a regression).
- 13:01 cupy DEV replicate (GPU queue, seeds 1-3, 40 trials, no ablations/curve,
  `dev_cupy/controls_nmda20_400_seeds1-3.json`): 3/3 seed GO; R 0.90 / 0.90 / 0.93; shuffled R -0.00 / -0.04 / 0.18;
  HS before 0.95-1.0, after 0.0; NS after 0.92-0.97; membrane [-82, +40] mV. Consistent with numpy.
- 13:04 DEV rehearsal seed 8 DIED in an ablation block: `membrane_bound_check` raised (one neuron at -161.8 mV).
  Probe (`seeds 1, 8`, 8 trials): the -3000 pA inactivation clamp drove the thalamic relay to -175 mV and into rebound
  firing (2.1 spikes/trial on seed 8, the Izhikevich TC rebound); -2000 pA still -151 mV and 1.5-1.9 spikes; -1000 pA
  silences D2, GPe, STN, commit FS and thalamus (0 spikes) with V >= -95 mV; -600 pA does not silence the thalamus.
  The check caught a clamp artifact before registration. INACTIVATE_PA -> -1000 pA in the runner, the prereg (not yet
  committed, no gate seed run) and DESIGN. Seeds 1-7, 9-12 of the first rehearsal used -3000 pA (DEV only).
- 13:16 DEV rehearsal 2 launched: the exact prereg command on seeds 1-12 (`--production-baseline --closed-loop`,
  60 trials), `research/findings/raw/_need_reversal_seam/dev_rehearsal2/`, 6 parallel, ETA ~13:40.
- 13:16-13:49 DEV rehearsal 2 = the exact prereg command on dev seeds 1-12 (`dev_rehearsal2/`, aggregate
  `dev_rehearsal2/aggregate_seeds1-12.json`): 12/12 seed GO (DEV, not a verdict). R 0.83-0.95 (mean 0.90); host arm R
  0.90-0.97; shuffled synapses R -0.18..+0.17 (bar |R| <= 0.25); host override R -0.49..-0.09; HS before >= 0.90, HS
  after 0.0 on every seed; NS after 0.83-1.0, NS before 0.0. Decoding: synaptic arm host variable 0.50, interoceptive
  0.81-0.90, D1 0.92-0.96; host arm host variable 0.85-0.89, interoceptive 0.86-0.91 (confounded), D1 0.92-0.96.
  Substitution (neural route for the host score at matched h): |delta P(forage)| <= 0.05, so behaviour is kept while
  ownership moves. Production anchor: the live `BGActionSelector` host share 0.95-1.0 on all 12 seeds; the host decides
  both content-bearing test messages without consulting the race. Closed loop (150 trials): intact corr(deficit,
  forage) 0.26-0.43 (mean 0.35), P(forage | deficit >= 0.6) 0.75 vs <= 0.4: 0.23; need synapses closed: corr
  -0.12..0.20 (mean 0.05), 0.59 vs 0.50, forage rate 0.50. Necessity (12 seeds): NECESSARY need synapses,
  interoceptive relays, arousal (commit <= 0.12 without it), direct path, thalamus, OU noise; NOT NECESSARY D2 (fires
  1.9-11.9 spikes/trial; its target GPe is silent), commit cross-inhibition (commit FS 0-3 spikes/trial), commit NMDA;
  NOT ENGAGED GPe and STN (0 spikes). No membrane-bound violation at -1000 pA inactivation.
- 13:50 gates/verdict_preconditions (a real gate, not compute-idle) flagged the DEV artifacts: a top-level `verdict`
  with no `preconditions`. Fixed the runner: `overall_block` emits a top-level `verdict` + `preconditions`
  (tools.verdict) ONLY over exactly the six gate seeds; any other seed set gets `dev_readout` and no `verdict` key.
  Renamed `verdict` -> `dev_readout` in the 25 staged DEV JSONs (values unchanged, `key_note` added; the prov sidecars
  carry no content hash). Test added. Hook dry-run on the staged set: only compute-idle-persistent blocks.

## NEXT, once commits unblock (gates/compute_idle_persistent; pool idle; the waiver budget frees ~18:30)

All of this is staged or written in the worktree; nothing has run on a gate seed.
1. Commit A (already staged): runner + tests + `tools/lab.py` + FAILURE_LOG row + lane_builds.jsonl + drafts + DEV raw.
   The hook dry-run fails only compute-idle-persistent.
2. Commit B, alone: `research/findings/2026-09-29-need-reversal-selection-seam-PREREGISTRATION.md` (untracked now).
3. `bash tools/push_both.sh research/selection-seam-need-reversal`.
4. `bash tools/pool_provision.sh --revision <commit B sha> --isolated pool41 pool42`, then per gate seed s:
   `bash tools/pool_queue.sh add 'cd ~/derisk-pool/revisions/<sha> && env SIM_BACKEND=numpy OMP_NUM_THREADS=1
   OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m
   research.runners._need_reversal_selection_seam --seeds <s> --allow-gate-seeds --trials 60 --curve-trials 20
   --production-baseline --closed-loop --out research/findings/raw/_need_reversal_seam/gate_seed<s>.json'
   --checked 'prereg research/findings/2026-09-29-need-reversal-selection-seam-PREREGISTRATION.md'`
   (~10 min per seed on one core; 6 seeds in parallel).
5. Aggregate: `python -m research.runners._need_reversal_selection_seam --aggregate
   research/findings/raw/_need_reversal_seam/gate_seed{42,43,44,100,101,102}.json
   --out research/findings/raw/_need_reversal_seam/gate_6seed.json`; finding doc; verify-go; commit raw + finding;
   board / summary-doc sync.

## 2026-09-29 afternoon: redirect to the LIVE seam (coordinator rulings on the audit)

Audit: `.claude/worktrees/_audit_2026-09-29/selection-seam.md` (8 confirmed issues). Rulings applied:
(1) target = the live production seam; (2) lane C's instrument for scoring; (3) plasticity on the value afferents
belongs to the long-delay-credit lane; (4) MEDIUMs fixed. Then the owner's token-scarcity directive: finish as an
execution packet (`HANDOFF.md`).

FILE REGIONS (for a clean merge with the long-delay-credit branch, which edits the same organ):
- `research/runners/bg_action_selection_production_organ.py`: everything I added is between `# BEGIN neural-seam`
  and `# END neural-seam` at the END of the file; nothing above it changed (flag-off hash identical to main).
  The value afferents carry `plasticity_gate="bg_value_afferent_plasticity"`, frozen here; the credit lane owns
  making them plastic, in its own functions.
- `research/runners/_gnw_norgan_bus_derisk.py`: 2 lines inside `norgan_hop` after `_ignite_and_read`, plus the
  `BEGIN/END neural-seam read` block after `norgan_hop`.
- `webapp/server.py`: the `faculty:bg-action-selection` block, the `if _BG.neural_seam_enabled(): ... else:` dispatch.

WHAT CHANGED
- Toy runner (MEDIUM 6, 7): membrane checked every step incl. reset/washout; a violation marks the block UNDEFINED
  (precondition), not a crash; necessity labels split (NECESSARY-FOR-COMMIT / -FOR-REVERSAL / NOT ENGAGED / PRE-EMPTED
  BY HOST READOUT); docstring receptor fixed. Correction to my 12:28 note: with OU off the whole race is silent (D1,
  thalamus, commit, motor all 0), not "the channels cross together". Toy prereg WITHDRAWN (moved to
  `toy-need-reversal-derisk-DESIGN-NOT-A-PREREG.md`); the toy is a DEV mechanism de-risk only.
- Live seam: latch probe (seed 7) showed the production race has NO persistent commitment: without the host `break`
  the winner's motor/commit activity dies 50-75 ms after arousal ends. Added an action latch (recurrent NMDA,
  mutual FS inhibition). Latch internal 48 latches unreliably, 64 holds; cross 20-40 overshot the losing latch to
  -121..-141 mV -> 10; motor->latch 80 drove latch neurons to +68..+145 mV during refractory -> 40 (<= +47). One fix
  round each.
- Engine overshoot (MEDIUM 8): not parked. A sonnet subagent built `enable_bounded_conductance_integration`
  (default off) on `research/conductance-overshoot-guard`; staged, 4 numpy tests pass.
- GPU DEV job queued ~15:10: `host_share_bg_seam --run-all research/findings/raw/_bg_live_seam/dev_cupy --seed 7
  --backend cupy` (behind 4 other lanes' jobs).
- ~16:00 Final organ DEV run (12 seeds, constants frozen; `research/findings/raw/_bg_neural_seam/dev_organ/`):
  numbers in the prereg's "DEV expectations". All 26 tests pass. Committed the infra-only part: 81d5cf2a9
  (`tools/lab.py::membrane_bound_check` + its test). Everything else is staged; the prereg is untracked on purpose
  (commit B, alone). Execution packet: `HANDOFF.md`. STOP here (owner token directive).
