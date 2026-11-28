# Capability: Sleep consolidation
# Record: brain/research/2026-11-28-local-rule-research-round.md

# Three-factor sleep prereg (2026-11-28): reward-gated perturbation on the adapters' rank units

STATUS: REGISTERED 2026-11-28, after the pre-data review (2026-11-27) and the full read of MeZO, MeZO-SVRG and MEMIT
(section "Full read (2026-11-28)"; Registration review, end of file). Nothing has run. Engine specs
`tools/engine/specs/three_factor_sleep_{01b,04b}.yaml` are `status: REGISTERED`. The full read raised the budget from
30x to 64x the gradient arm's forwards (below); that is the only change to the registered numbers. Milestone:
ARCHITECTURE section 4, 2026-12-31, "sleep learning by a local, biology-style rule instead of gradients". This is
candidate 1 of `2026-11-28-local-rule-research-round.md` section 2.1, and the fifth method on that defect (v1, SHY, v2,
v3). It does not re-open any of their verdicts.

PRE-DATA REVIEW 2026-11-27 (3 blocking, 6 non-blocking). Fixed: (1) the cosine instrument `full_objective(grad=True)`
now backprops one text at a time and sums G (values unchanged, tested against the one-graph gradient), so only one
text's graph is alive, as in gradient sleep; (3) on NEGATIVE-DEV / KILL-DEV the engine spec copies the harness word to
`<res>/harness_pick_dev.json` before the engine overwrites `pick.json` (tested); (4) gradient_rank has its own Adam lr
`--grad-rank-lr 5e-5` (5e-4 / 10, because fixed B ~ N(0, 1/8) is about 3x the gradient arm's learned B); (9) 0.4B wall
re-estimated (Compute). Kept as drafted: lr grid, SVRG out of dev, graded read-out report-only, gradient_rank a reading.
Blocking item (2), the full read of MeZO, MeZO-SVRG and MEMIT: done 2026-11-28 (section "Full read"). No blocking
item remains.

## Record brief
- **Knew.** v1, SHY and v2 sent the output error back over feedback that never aligned with the network: local recall
  equalled shuffled, and replay loss was 2.27 against the gradient's 0.17. v3 fitted its layerwise targets (error down
  70%) but recalled 0-2/12, because copy-mode targets do not transfer to answering. The note's claim: none of the four
  rules moved the adapters along the output-error direction.
- **Old track.** 07-13 retired node perturbation for having "no edge over KP" on toys. KP has now failed at scale, so
  that verdict does not carry over. Variance is the open question.
- **Literature (snippets at drafting; read in full 2026-11-28, see "Full read").** MeZO fine-tunes pretrained LLMs up
  to OPT-13B, LoRA included, in 20K-100K steps (32-100x fine-tuning's; the snippet's "thousands" was wrong). Forward
  gradients degrade above about 1.15M trainable parameters. At 0.1B, rank-8 A factors have
  8 x (768 + 3072) x 12 = 0.37M parameters.
- **New.** The rule optimises the same objective as gradient sleep. Its expected update is the true gradient (tested on
  a tiny cortex: positive cosine; negative with the wrong sign; near zero with a shuffled scalar), so nothing can be
  misaligned. The only open question is variance at this budget.

## Method (module `brain/experiments/three_factor_sleep.py`; imports `local_rule_v2`, `local_rule_v3`; edits neither)
- **Rank units.** Each channel-mix adapter is `A @ B` with 8 rank units, `u = x A`. PlasticCortex starts B at 0, which
  would silence the noise. At the start of the night this rule re-parameterises: A = 0, B = fixed random N(0, 1/8)
  (seeded per seed). The product is still 0, so the cortex starts exactly pretrained (tested). **Only A learns. B stays
  a fixed random projection.** That is a real restriction: the `gradient_rank` arm measures what it costs.
- **Per step.** One episode is drawn by strength, then one keep text (the same RNG draws as `brain.sleep`).
  1. Unit noise xi (T x 8) goes on every adapter's rank units for each text: +sigma xi and -sigma xi ("antithetic",
     4 forwards). The alternative "forward" estimator uses 0 and +sigma xi.
  2. One global scalar: s = (L+ - L-) / (2 sigma), where L = replay CE + kl_weight x keep KL to the pretrained cortex.
  3. `dA = s x^T xi`: scalar x noise x pre-synaptic activity (k_in for the key adapter, h for the value adapter).
- **Step size.** `A -= lr sqrt(P) dA / m`, where m is an EMA of past estimate norms. This is normalised SGD, so lr is a
  per-parameter RMS step like Adam's. m uses past steps only, so the step direction stays unbiased.
- **No feedback path.** Every night forward runs under `torch.no_grad` (asserted). No weight transport.
- **Optional, MeZO-SVRG form (`--svrg q`, not in the dev grid).** An anchor copy of A, plus a full-batch estimate every q
  steps. Its cost is counted in the budget.
- **Budget (stated; raised by the full read).** The gradient night runs 240 steps x 2 forwards (replay and keep) = 480
  forwards, plus 240 backwards, which are not counted. The rule gets 64 x 480 = **30720 text forwards per night**. Every
  forward is counted, anchor refreshes included. A step runs only if all its forwards fit. Antithetic with 1
  perturbation gives 7680 steps = 32 x the gradient's 240 (MeZO's OPT step ratio, "Full read" change 1). The draft had
  30 x 480 = 14400 forwards = 3600 steps.
- **Instrument (backprop as a measuring tool only, never the rule).** At 9 points of the night (8 windows), the true
  gradient of the full fixed-order objective (every telling's replay CE + kl_weight x every keep text's KL) with respect
  to A. `cos_window` is the mean, over windows, of the cosine between the rule's accumulated update in the window and
  that gradient. `cos_night` is the same over the whole night (reported only).

## Arms (per model, per seed; 42, 43, 44, 100, 101, 102)
| Arm | What | Role |
|---|---|---|
| no_sleep | adapters untouched | floor |
| gradient | `brain.sleep.sleep`, Adam 5e-4, 240 steps | reference; sets the bar and the budget |
| three_factor | the rule, own scalar | tested |
| tf_shuffled | scalar = a draw from earlier steps' scalars (same magnitudes, unrelated to this step's noise) | control |
| tf_wrong_sign | scalar = -s (ascent in expectation) | control |
| gradient_rank | Adam 5e-5 (`--grad-rank-lr`) on the same re-parameterisation (A from 0, B fixed) | report only; reads a NEGATIVE |

The three-factor arms share one (lr, sigma, estimator, n_pert, svrg, budget) setting. Score refuses mixed units.

## Dev (seed 7, per model, frozen before that model's six seeds)
- **Grid.** `ref` (no_sleep, gradient, gradient_rank) plus 9 three_factor cells: lr {3e-5, 1e-4, 3e-4} x sigma {0.01,
  0.03, 0.1}. Antithetic, 1 perturbation, no SVRG.
- **`dev_pick` order:**
  1. **UNDEFINED-DEV**: a non-finite ref, or dev gradient recall < 2.
  2. **KILL-DEV**: every cell is killed.
  3. **NEGATIVE-DEV**: no cell qualifies.
  4. **PICKED**: the qualifying cell with the most recall. Ties go to higher answer log-probability, then lower held-out
     KL, lower generic KL, lower lr, lower sigma.
- **Qualify.** All of these:
  - alive;
  - finite;
  - recall >= the bar;
  - generic and held-out (dev half) KL each <= 1.25 x gradient's and <= 1;
  - engaged: norm > 0, the fixed-order objective falls, >= 90% of the budget used (never more), and the arm applied its
    own scalar.
- **Kill rule (the note's).** A night is killed if the fixed-order replay CE at the end is not under 1.0, or if
  `cos_window` is under 0.1.
- **Ladder.** KILL-DEV and NEGATIVE-DEV mean no six seeds and no 0.4B. UNDEFINED-DEV holds the chain.

## Measures (hippocampus off, per seed, per arm; one atomic json unit each, resumed by skip)
- recall /12, paraphrase, false recall, generic KL (`sleep_probe.evaluate`).
- Held-out KL on the pinned prompts, split as in v2/v3: dev half on seed 7, verdict half never on seed 7.
- Adapter and factor norms.
- **Graded read-out (note section 4).** The log-probability of the answer word right after `User: {q}\n\nAssistant:`,
  summed over its tokens, mean over the 12 facts. Also the rank of its first token (median, mean log10, count in the top
  10). It is reported beside recall and gates nothing, except the dev tie-break.
- Engagement:
  - objective, replay CE and keep KL at the start and end (fixed order);
  - `cos_windows`, `cos_window`, `cos_night`, and the per-step `cos_step_mean`;
  - forwards used against the budget, and steps run;
  - corr(applied scalar, own scalar), and mean |s|.
- Provenance: host, commit, device and precision in every unit. The guard requires POOL_NODE=1 or BRAIN_ALLOW_W3090=1.

## Registered rule (written before data; `verdict()` in the module, tested with each clause and each gate failing alone)
bar = max(0.5 x gradient mean recall, no_sleep mean + 2). Caps are 1.25 x gradient's generic and held-out KL. The
verdict reads these clauses in order:
1. **UNDEFINED** if any of these holds:
   - a gated value of a gated arm is non-finite;
   - mean gradient recall < 2;
   - a core engagement gate fails. The gates: no_sleep norm is 0; gradient norm > 0 and replay CE end < start;
     three_factor norm > 0, objective end < start, budget = 64 x the gradient forwards with 90-100% used, and
     corr(applied scalar, own scalar) >= 0.99.
2. **KILL** if the mean three_factor replay_end is not under 1.0, or the mean `cos_window` is under 0.1.
3. **NEGATIVE** if a core clause fails:
   - (a) three_factor recall >= bar;
   - (c) generic and held-out KL each <= cap, and no night above KL 1;
   - (e) false recall <= gradient + 1.

   The reading comes from gradient_rank. If gradient_rank passes (a) and (c), it is a **rule-limit**: the same
   parameterisation stores by gradient, and the estimate is the limit. If it fails, it is a
   **parameterisation-limit**: A-only rank units with a fixed random B do not store even by gradient.
4. **UNDEFINED** if a control engagement gate fails:
   - tf_shuffled: norm > 0 and >= 0.5 x three_factor's, budget used, and |corr| <= 0.2;
   - tf_wrong_sign: norm > 0 and >= 0.5 x three_factor's, budget used, corr <= -0.99, and objective end > start
     (ascent, so the sign is live).
5. **NEGATIVE** unless (b) holds: three_factor beats tf_shuffled, and also beats tf_wrong_sign, each on >= 5 of 6 seeds,
   and each control's mean recall is <= no_sleep + 1.
6. **GO-TF** otherwise.

Two scales (`combine`): GO-TF at both scales gives **GO-TF**. Exactly one gives **GO-TF-PARTIAL**. Any UNDEFINED, ERROR,
HOLD or STOP gives **UNDEFINED**. Anything else gives **NEGATIVE**.

## Decision table (the engine ladder matches it)
| 0.1B outcome | Next |
|---|---|
| GO-TF | 0.4B runs (dev then six), and then the 12-12 gate (forward-only at half of gradient recall, six seeds, both scales) |
| KILL / KILL-DEV | plain ZO-SGD with one episode per step is retired at this budget. One registered amendment may run the MeZO-SVRG form (full-batch anchor over all tellings and the keep bank, q in 2-5, same budget), because MeZO-SVRG measured plain MeZO unstable at small batch and the anchor as the fix (Full read, change 3). If that is also killed, the method is retired: candidate 2 or 3 of the note, not a larger budget, unless T0 says direction is fine |
| NEGATIVE, rule-limit | variance is the limit. A registered amendment may try SVRG at the same budget. Not n_pert > 1: MeZO Table 6 measured only marginal gains at equal forwards (Full read, change 2) |
| NEGATIVE, parameterisation-limit | B must learn (weight perturbation on B, or candidate 2's key-value write) |
| UNDEFINED / HOLD-DEV / ERROR | diagnose, amend here, re-run 0.1B. 0.4B does not run |

## Scaffolds declared (added to SCAFFOLDS.md at registration, 2026-11-28)
| Scaffold | Where | Replace with | Added |
|---|---|---|---|
| Global third factor = a host-computed scalar "did replay loss + keep KL go down" (two forward passes, one number for the whole cortex) | `experiments/three_factor_sleep.py` `zo_sample` | a neuromodulator signal from `modulators.py` (dopamine-like reward-prediction error, computed by the brain from its own replay outcome) | 11-28 |
| Synaptic noise on the rank units is drawn by the host (seeded Gaussian), not generated by the units | `three_factor_sleep.py` `draw_noise` | intrinsic stochastic activity of the rank units | 11-28 |
| Rank-unit output projection B fixed random; only A learns | `three_factor_sleep.py` `reparam_rank` | B learned by the same three-factor rule (weight perturbation), or a key-value write | 11-28 |
| Step normalised by an EMA of past update norms (lr is a host constant) | `three_factor_sleep.py` `tf_sleep` | homeostatic / divisive normalisation of plasticity | 11-28 |
| Text replay and the keep bank as in `brain/sleep.py` | `sleep.py` | unchanged (already declared) | - |

Biology: this is synaptic noise plus a global neuromodulator. It is three-factor learning in the dopamine sense, not
synapse-local Hebbian learning. The note says so, and the owner message in note section 3 says so.

## Honest odds (estimates, not measured; re-checked against the full read, unchanged)
- Dev: a cell survives the kill at 0.1B: 50%.
- Dev: a cell qualifies at 0.1B: 30%.
- GO-TF at 0.1B on six seeds: 20%.
- Both scales: about 10-15%. This is below the note's 35% for "a forward-only three-factor rule", because B is fixed
  here.
- Expected outcome (full read): partial convergence. No paper measured memorisation of arbitrary facts by ZO; MeZO's
  near-parity results are on prompted, generalising tasks, and MeZO-SVRG's best ZO still sits 8-30 points under
  first-order. The 64x budget buys the step ratio MeZO needed, not a guarantee. The most likely single outcome at 0.1B
  is NEGATIVE (rule-limit) or KILL; half of gradient recall is a plausible ceiling, not a safe bet. The budget raise
  offsets the memorisation gap, so the numbers above stay.

Risks:
1. Variance. About 2 x 12 x 8 x T noise dimensions per text means a per-step cosine near 0.01. The accumulated window
   cosine may sit near the 0.1 kill line.
2. The random-walk part of the update raises generic KL (c).
3. Fixed B limits capacity.
4. The 0.4B six-seed gradient mean was 2.33, so UNDEFINED by gradient < 2 is possible there.
5. Batch of one episode per step (the gradient arm's draws). MeZO-SVRG measured plain MeZO unstable or diverging at small
   batch (Fig. 1b-c). Kept, so both arms see the same draws; the kill rule catches it and the decision table names the
   SVRG form as the one amendment.
6. Cross-fact interference. MEMIT stores many facts only with keys decorrelated through C^-1; this rule shares A across
   the 12 facts with no such term. Paraphrase, false recall and held-out KL are already measured.

## Compute
- CPU fp32 on the pool, through the engine. No GPU, no AWS.
- The real smoke sets the memcap. memory_gb is 5 (0.1B) and 10 (0.4B), as in v3. Record the 0.4B smoke's peak RSS
  (the engine logs it) and set the 04b memcap from it before dev. The cosine instrument backprops one text at a time.
- Wall-time estimate from v3 (about 0.2 s per forward at 0.1B), at the 64x budget: one three-factor night is 30720
  forwards, about 1.7 h at 0.1B (was 45-50 min at 30x). A six-seed job (3 three-factor nights + 2 gradient nights) is
  about 5.3 h per seed at 0.1B. At 0.4B (about 4 x the per-forward time) one night is about 6.8 h, so a seed is about
  21 h and dev is 9 cells of about 7 h. Totals: 0.1B about 16 h of dev + 32 h of six seeds, 0.4B about 63 h + 126 h,
  about 240 CPU-hours. That fits the 12-12 gate only with about 4 or more pool slots in parallel (0.4B dev then six in
  about 2 days). The 0.4B smoke gives the real figure; if the pool cannot hold it, say so before 0.4B dev, do not cut the
  budget. Dev is 10 jobs per model.
- Freeze the repo and its COMMIT from a model's dev through its score. Six-seed jobs take lr and sigma from the engine's
  pick. The score checks model, commit and config against the pick.

## Open for the pre-data review (answered 2026-11-27: kept as drafted)
1. The lr grid {3e-5, 1e-4, 3e-4} is a guess for a normalised ZO step (kept after the full read: MeZO's lr values are
   for unnormalised SGD and do not transfer). On a tiny cortex, 3e-4 lowered the objective
   and 1e-3 did too. At 0.1B, P = 0.37M.
2. Whether one SVRG cell should join the grid.
3. Whether the graded read-out should gate (it is reported only, by design, so a pass does not move).
4. Whether gradient_rank should gate the six seeds or stay a reading (it stays a reading here).

## Full read (2026-11-28): MeZO, MeZO-SVRG, MEMIT
Reads of arXiv 2305.17333 (MeZO, whole text), 2404.08080 (MeZO-SVRG, main text through App. E.6; F-J unread) and
2210.07229 (MEMIT), through a summarising fetch: numbers are extracts, spot-check before citing.

**What each paper measured.**
- **MeZO.** Two-point SPSA (z ~ N(0, I)), plain ZO-SGD. Variance: E|g|^2 = ((d+n-1)/n) E|grad|^2, so the stable lr is
  about n/d of SGD's; the slowdown scales with an effective Hessian rank r instead of d when pretraining plus prompts
  make r low (Thm 1). Steps: RoBERTa-large 100K vs 1K for FT (100x); OPT 20K vs 625 (32x). OPT-13B within 1 point of FT
  on 7 of 11 tasks; RoBERTa k=16 gaps up to 9 points. LoRA (r 8, alpha 16) matches full-parameter MeZO and trains at a
  similar speed (no fewer steps). n = 4 or 16 perturbations per step gives marginal gains at equal forwards (Table 6).
  Failure modes: no prompt (SST-2 89.6 -> 51.9), a non-differentiable objective (accuracy instead of CE: TREC 94.3 ->
  68.6), one-point estimates, normalised-gradient and its variance-reduction variants. No memorisation experiment.
- **MeZO-SVRG.** A full-batch anchor estimate every q steps as a control variate. Plain MeZO is unstable or diverges at
  small batch (Fig. 1b-c); SVRG is more robust (Fig. 6), q >= 2 beats q = 1, large perturbation scale diverges (Fig. 4).
  Still 8-30 points under first-order on 512-example tasks. LoRA is not tested. No fact recall, no RWKV.
- **MEMIT.** Not a perturbation method: v* is found by backprop (GPT-J: 25 steps, lr 0.5, clamp 0.75|h|), then a
  closed-form key-value write Delta = R K^T (C + K K^T)^-1 with C from 100K samples of the layer's own keys (lambda
  15,000). It says nothing on ZO variance, step counts, LoRA or lr. Its relevance: many facts are stored without
  interference only with keys decorrelated through C^-1; a "perturbation-found value + closed-form write" would be a new
  method (the note's candidate 2), not a replication.

**What the design already matched (no change).**
- The scalar is the continuous two-point loss difference on a smooth objective (replay CE + keep KL), not a binary or
  thresholded reward: the form MeZO and MeZO-SVRG use, and Table 3 shows what a coarser signal costs. Antithetic
  (two-point) is the dev estimator; the one-point "forward" estimator stays out of dev.
- n_pert = 1 in dev, the extra forwards go to steps (Table 6).
- The text format is fixed and the same as the gradient arm's (MeZO needs a fixed prompt).
- Rank-8 adapters: P = 0.37M at 0.1B, far below MeZO's full-model d; MeZO's LoRA runs support ZO at this rank. The
  effective-rank argument leans on pretraining plus prompts; memorising arbitrary facts is the case it covers least.
  Untested here, so it is a risk, not a reason to cut the budget.
- Grid. sigma {0.01, 0.03, 0.1}: the noise sits on rank-unit activities, not weights, so MeZO's eps (1e-3 / 1e-2) does
  not transfer; MeZO found eps mattered little, MeZO-SVRG that large scale diverges, and 0.1 is the top cell with the
  kill rule behind it. lr {3e-5, 1e-4, 3e-4}: MeZO's lr is for unnormalised SGD and does not transfer to this normalised
  step. No paper measurement requires a grid change, so the grid is unchanged.
- Normalisation. MeZO's normalised-gradient variant (direction / its own norm) hurt; this rule divides by an EMA of
  PAST norms, so the step keeps each estimate's magnitude (a running lr scale, not a per-step normalisation). Kept,
  recorded as a difference.

**Changes (each forced by a measurement).**
1. **Budget 30x -> 64x the gradient arm's forwards** (`BUDGET_MULT` 64; 14400 -> 30720 forwards; 3600 -> 7680 steps).
   At 30x the rule got 15x the gradient's steps, below every step ratio MeZO measured (32x OPT, 100x RoBERTa), and a
   KILL there would retire the method at a budget the literature says is too small. 64x gives 32x steps, MeZO's
   smallest measured ratio with near-parity on most tasks; the target is half of gradient recall, not parity. 100x
   steps (200x forwards) would make a 0.4B night about 21 h and does not fit the 12-12 gate. Compute: about 240
   CPU-hours over both scales (Compute). Module, tests and both engine specs carry 64.
2. **Decision table, rule-limit row: n_pert > 1 removed** as an amendment (MeZO Table 6: marginal at equal forwards).
   SVRG stays.
3. **Decision table, KILL row: one registered SVRG amendment before retirement.** The rule steps on one episode plus one
   keep text, the small-batch regime where MeZO-SVRG measured plain MeZO unstable and the anchor as the fix. A KILL of
   plain ZO-SGD at batch one does not retire three-factor perturbation until the anchored form is also killed. SVRG
   stays out of dev (the two papers disagree on variance reduction: MeZO saw its variant hurt).
4. **Expected outcome stated** (Honest odds): partial convergence, NEGATIVE or KILL the likeliest single outcome at
   0.1B; odds unchanged.
5. **Risks 5 and 6 added** (small batch; cross-fact interference without a covariance term).

## Registration review
- 2026-11-27: pre-data review, 3 blocking and 6 non-blocking; blocking 1 and 3 fixed and tested (STATUS above).
- 2026-11-28: blocking item 2 closed by the full read of MeZO, MeZO-SVRG and MEMIT; budget raised to 64x with the
  compute estimate, decision table rows amended, expected outcome stated, grid unchanged (each recorded in "Full
  read"). Scaffold rows added to SCAFFOLDS.md. No blocking item remains: REGISTERED 2026-11-28, specs REGISTERED. Freeze
  the repo and its COMMIT from 0.1B dev through its score. Before 0.4B dev, set the 04b memcap from the 0.4B smoke's
  peak RSS and check the pool can hold about 21 h per seed.
