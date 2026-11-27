# Capability: Sleep consolidation
# Record: brain/research/2026-11-27-local-rule-v3-DESIGN.md

# Local rule v3 prereg (2026-11-27): two-phase layerwise targets from the cortex's own in-context activity

STATUS: REGISTERED 2026-11-27 after pre-data sonnet review. Nothing has run. The fixes are listed in "Registration
review" at the end of this file. Engine specs `tools/engine/specs/local_rule_v3_{01b,04b}.yaml` are
`status: REGISTERED`. Settings below are frozen at registration. lr and keep weight are
frozen per model by that model's seed-7 dev pick. Milestone: ARCHITECTURE section 4, 2026-12-31, "sleep learning by a
local, biology-style rule instead of gradients; scale to 1.5B". This is lever 4 on that defect (v1, SHY, v2, v3). It
does not re-open the v1 NEGATIVE, the SHY result or the v2 NEGATIVE-DEV
(`experiments/results/2026-11-27-local-rule-v2-read.md`).

## Record brief (design: `2026-11-27-local-rule-v3-DESIGN.md`)
- **Knew.** v1 (output error over fixed feedback) learns only at a high lr and drifts 3.5x the gradient night. SHY
  reached 1.17 against a bar of 2. v2 (plasticity off the keep subspace) showed no drift and 0-2/12 recall at both
  scales (NEGATIVE-DEV, engine run at commit 34325395). The common reading: the feedback path does not put the update
  where the fact lives.
- **Old track.** These cases failed because the error had to travel: 05-17 dendritic credit (feedback never aligned,
  0.012), the 08-01 supervised deep-credit null, 07-13 node perturbation (no edge over KP). The positive case had the
  error given at the layer: 07-20, a local delta rule on a readout, reaches BPTT parity. The 08-02 depth finding was
  corrected on 08-11 as a per-arm lr artifact, so the step here is normalised.
- **brain/.** With the telling in context the cortex has the fact (in-context arm 55/67). The 11-17 teacher night used
  the in-context cortex as teacher with final-logit targets and backprop. Its NO-GO was about abstention, not about
  storing told facts.
- **New.** Each layer gets its own target, so no error travels between layers. Each adapter learns to reproduce,
  without the telling in context, the activity its layer had with the telling in context. Each layer is then the 07-20
  readout case.

## Method (module `brain/experiments/local_rule_v3.py`; imports `local_sleep` and `local_rule_v2`, edits neither)
Per replayed telling X, both phases forward-only, under `no_grad`, on the same cortex:
1. **Clamped phase (teacher).** Adapters off. The cortex reads the token ids of X followed by the token ids of X again,
   from the base state. On the second reading, per block i and position t, it records the residual after the block,
   x*_{i+1}, and the channel-mix hidden h*_i. These are cached once per telling.
2. **Free phase (student).** Adapters on. The cortex reads X once from the base state.
3. **Update, at each block's own synapses:**
   - Value path: e = x_{i+1} - x*_{i+1}, and dB_v = (h A_v)^T e.
   - Key path: e_h = (h - h*) 2 relu(pre), and dB_k = (k_in A_k)^T e_h.
   - A factors: dA = x^T (e Bcopy), through v1's Kolen-Pollack copies of B, which receive B's own update and decay.
   - Every step is normalised LMS, divided by the pre-synaptic energy of that factor. At lr 1 a single row is fully
     corrected (tested).
   - v1's update-norm clip (1.0) and B decay (1e-3) are kept. The clip caps each factor's normalised step at norm 1
     before lr. So "one lr serves every layer" and "lr 1 fully corrects a row" hold only while the clip is inactive.
     The fraction of clipped factor updates is logged per arm, replay and keep separately (`clip_frac_replay`,
     `clip_frac_keep`).
   - All errors come from the student's own forward.
4. **Keep.** One keep text (`sleep_probe.GENERIC`) per step goes through the same rule. Its target is the pretrained
   cortex's own activity on that text. The keep weight multiplies the keep step AFTER the clip (and the KP copy's
   update with it), so keep 0.3, 1 and 3 give different steps whether or not the clip binds (tested). The logged
   errors are unweighted.
- **RNG draws** match v1/v2: one episode by strength, then one keep text, per step. Each night is 240 steps, with
  `torch.manual_seed(seed)` before each arm's PlasticCortex.
- **Episodes** are `build_arm_episodes(seed)["sleep_true"]`: the 12 tellings, the same information the gradient
  reference gets.
- **Recording.** `V3Cortex` subclasses LocalCortex. It rebuilds the residual from the recorded time-mix and
  channel-mix outputs with the same adds as `PlasticCortex.forward`. Logits are bit-identical to PlasticCortex with
  adapters on and off (tested). A source tag is refused.
- **Not chosen.** Question-form cues, which would use the 55/67 competence, are not used. Questions are not stored, so
  using them would change the information as well as the rule. They are the follow-up if the target proves to be the
  limit.

## Registration checks (resolved before data)
- **`_told` layout.** The text is `"User: <sentence>\n\nAssistant: Okay, I'll remember that.\n\n"`. The double reading
  concatenates token ids, so positions align 1:1 by construction.
  - Checked on the real tokenizer (vocab only, no model): `encode(X+X) == encode(X)+encode(X)` for 12/12 tellings, so
    the double reading equals a plain text re-reading.
  - The answer starts at token 6-9 of 20-24.
  - The answer's first token is a subword for 6 facts (" tea", " Marg", " Bis", " Vol", " Cru", " cell"). G0
    therefore gates on EVERY token of the answer key, teacher-forced (amended at review). The first-token rate is
    reported beside it.
- **v2's measured wall time** (engine units, pool nodes, CPU fp32):
  - 0.1B: a local night plus battery takes 80-88 s, gradient 203 s, no_sleep 30 s. Smoke peak 2.9 GB.
  - 0.4B: 162-242 s, gradient 522 s, no_sleep 86 s. Smoke peak 6.8 GB.
  - v3 adds 12 cached double reads and two forwards per step, as v1. Estimate: local night about 1.2x v2.

## Gate G0 (free, before any night)
- On the second reading, teacher-forced, the teacher's top-1 must be the answer's token at every token of the answer
  key, for >= 10 of the 12 facts, per model (`g0_check`, `n_ok`). The first-token rate (`n_ok_first`) and the
  single-reading rates are reported beside it.
- G0 runs first in the engine's real smoke (`--arms g0,gradient,v3 --steps 24`). If it fails, the run writes the G0
  unit and exits non-zero before any night. The chain then stops at ERROR-SMOKE with no night run.
- The dev `ref` job runs G0 again. The pick refuses if there is no G0 unit, and returns STOP-G0 (engine HOLD-DEV) if G0
  failed. A resumed run with a failed G0 unit refuses again (tested).

## Arms (per model, per seed; 42, 43, 44, 100, 101, 102)
1. `v3`: the rule at (lr*, keep*), with targets at every block.
2. `gradient`: `brain.sleep.sleep` (Adam 5e-4, KL weight 1) on the true episodes. This is the reference.
3. `no_sleep`: adapters at zero.
4. `sham_ctx` (shuffled target): before X, the clamped phase reads another fact's telling, by a seeded derangement
   (seed + 7919). Real in-context activity, no knowledge of X.
5. `no_target` (lesion): no first reading, so the target is the pretrained cortex's own activity on X. The error is the
   adapters' own effect, and B starts at 0, so the night must stay exactly inert. Any recall is a leak.
6. `top_only` (depth lesion): targets, replay and keep at the last block only.
7. `grad_distill` (report only): Adam 5e-4 on KL(teacher || student) over the telling's positions, using the same
   double-reading teacher logits, plus the keep KL at weight 1. It separates "bad target" from "bad rule".
Models: 0.1B `rwkv7-g1d-0.1b-20260129-ctx8192.pth`, then 0.4B `rwkv7-g1d-0.4b-20260210-ctx8192.pth`.

## Dev (seed 7, per model, frozen before that model's six seeds)
- **Grid.** lr in {0.03, 0.1, 0.3} x keep in {0.3, 1, 3}: 9 `v3` nights, plus `ref` (G0, no_sleep, gradient). That
  makes 10 engine jobs, 11 nights in all.
- **Pick (`dev_pick`, tested).**
  - A cell qualifies only if all of these hold:
    - every value is finite;
    - recall >= the verdict bar, max(0.5 x ref gradient, no_sleep + 2);
    - generic KL <= 1.25 x the ref gradient's, and <= 1;
    - held-out KL on the DEV half <= 1.25 x the ref gradient's, and <= 1;
    - the cell is engaged: adapter norm > 0 and the replay error at the night's end < at its start (`err_end <
      err_start`, the fixed-order probe under Measures).
  - Pick the highest recall. Ties go to lower held-out KL, then lower generic KL, then lower lr, then lower keep.
- **Outcomes.** If no cell qualifies: NEGATIVE-DEV, no six seeds, no grid extension. A non-finite ref: UNDEFINED-DEV.
  A failed G0: STOP-G0.
- **Refusals.** The pick refuses (exit non-zero) unless the dir holds exactly one G0 unit, the two ref units and one
  unit per cell. All must be seed 7, from one model, one commit and one prompt list.
- **Firewall.** The VERDICT half of the held-out prompts is never computed on seed 7 (tested).
- **Facts shared across seeds (disclosed).** The seeds vary the adapter init, the sampling order and the sham
  derangement, not the 12 facts. Dev tunes on the facts that the six seeds score.

## Measures (hippocampus off, per seed, per arm; one atomic json unit each, resumed by skip)
- **Battery.** `sleep_probe.evaluate`: recall /12, false recall /4, generic KL, paraphrase /12 (reported).
- **Held-out KL.** `local_rule_v2.heldout_drift` on `mix_night_screen.PROMPTS`, pinned by sha256: the VERDICT half
  gates, the DEV half is used by the pick.
- **Norms.** Adapter norm and factor norms.
- **Engagement record** (every arm):
  - **Engagement probe (gated).** At the start and the end of each night, a fixed-order, forward-only pass over all 12
    tellings, with no update. Local arms: the summed per-layer value error against the cached targets, averaged over
    tellings (`err_start`, `err_end`, and per layer). gradient: the replay CE that `brain.sleep` trains on
    (`loss_start`, `loss_end`). grad_distill: the KL to the teacher (`loss_start`, `loss_end`). Every "end < start"
    gate below reads this probe.
  - local arms: teacher kind, layers on, targets cached, mean context tokens, clip fractions (replay, keep), and,
    reported only, replay and keep error head/tail (mean of the first/last 20 steps) and per-layer value error
    first/last, sham permutation;
  - gradient and grad_distill: loss head/tail (reported only);
  - no_sleep: steps 0.
- **Provenance.** Host, host_env, commit, device, precision, model id, prompts sha, lr, keep, grad_lr, last layer,
  elapsed.
- **Guard.** POOL_NODE=1 or BRAIN_ALLOW_W3090=1.

## Registered rule (written before data; `verdict()` in the module, tested with each clause and each gate failing alone)
Per model, six seeds, means across seeds. `verdict()` raises ValueError unless all 7 arms hold exactly the six seeds.
bar = max(0.5 x mean gradient recall, mean no_sleep recall + 2). "Held-out" below means the VERDICT half.

**Clauses:**
- **(a)** mean `v3` recall >= bar.
- **(b)** `v3` recall > `sham_ctx` recall on >= 5 of 6 seeds, AND mean `sham_ctx` <= mean no_sleep + 1.
- **(c)** mean `v3` generic KL <= 1.25 x gradient's (c_generic), AND mean `v3` held-out KL <= 1.25 x gradient's
  (c_heldout), AND no `v3` seed has generic or held-out KL > 1 (c_night).
- **(d)** mean `no_target` recall <= mean no_sleep, AND every `no_target` adapter norm <= 1e-6.
- **(e)** mean `v3` false recall <= mean gradient false recall + 1.

**Engagement gates (`engagement()`):**
- Core arms:
  - `no_sleep`: adapter norm exactly 0.
  - `gradient`: norm > 0, and probe loss end < start.
  - `v3`: norm finite and > 0, probe replay error end < start, teacher "double", more than one layer on.
- Controls:
  - `sham_ctx`: norm > 0 and >= 0.5 x that seed's `v3` norm, probe error end < start, teacher "sham" with context read.
  - `top_only`: norm > 0, probe error end < start, exactly the last layer on.
  - `no_target`: ran its steps with teacher "none" and no context. Its inertness is (d).
- Report only:
  - `grad_distill`: norm > 0, and probe loss end < start. This gate affects only the NEGATIVE reading.

**Order of decision:**
1. **UNDEFINED** if any of these holds:
   - a recall, false recall, generic or held-out KL of any arm except `grad_distill` is non-finite;
   - a `v3` adapter norm is non-finite;
   - mean gradient recall < 2;
   - a core engagement gate fails on any seed;
   - (d) is violated (a leak or a broken lesion).
2. **NEGATIVE** if (a), (c) or (e) fails. These read only `v3`, `gradient` and `no_sleep`, so this holds whatever the
   controls did, as in v2. Reading from `grad_distill`:
   - it passes (a) and (c): "rule-limit". The delta rule fails even with aligned targets, and the delta-rule family
     stops at this adapter rank (the stop rule for this lever);
   - it fails either: "target-limit". The single-form target is the limit; next are question-form cues;
   - it is not engaged or non-finite: "undetermined".
3. **UNDEFINED** if a control engagement gate fails on any seed. An inert sham would pass (b) for free, and an inert
   `top_only` would make GO-V3 free.
4. **NEGATIVE** if (b) fails (contrast). Not credited to the told content.
5. **GO-SHALLOW** if mean `v3` minus mean `top_only` < 1: local, but only the known readout result. Otherwise
   **GO-V3** for that model.
- **Two scales (`combine`):**
  - **GO-V3**: GO-V3 at both scales.
  - **GO-V3-PARTIAL**: exactly one GO-V3, with the other defined.
  - **GO-SHALLOW**: GO-SHALLOW at either scale, with no GO-V3.
  - **NEGATIVE**: otherwise among defined verdicts.
  - **UNDEFINED**: any UNDEFINED, ERROR, HOLD or STOP.
- **Not tested here.** Transfer to reworded questions (paraphrase is reported). The claim is "the night stores
  replayed told facts under the drift caps".

## Decision table (the engine ladder matches it)
| 0.1B outcome | 0.4B | Meaning / next |
|---|---|---|
| GO-V3 / GO-SHALLOW | runs | lead only, nothing adopted until 0.4B is read |
| NEGATIVE / NEGATIVE-DEV | runs | a verdict on this method at 0.1B; operating points do not transfer across scale |
| UNDEFINED / HOLD-DEV (STOP-G0, UNDEFINED-DEV, refused pick) / ERROR (incl. ERROR-SMOKE from a failed G0) | does NOT run | diagnose, record an amendment here, re-run 0.1B. A failed 0.1B G0 does not settle 0.4B: the owner may run the 0.4B G0 alone (`run --arms g0`, no nights) |

| Two-scale | Next |
|---|---|
| GO-V3 | land-result (skeptic, a lesion that changes the reply); write the 1.5B prereg |
| GO-V3-PARTIAL | lead at one scale; 1.5B only if the failing scale is a prerequisite gap (PREREQUISITES) |
| GO-SHALLOW | not credited to layerwise targets; next is a readout-only registered method, or deeper targets |
| NEGATIVE, rule-limit | the delta-rule family stops at rank 8; next is a rank or substrate change, not another rule |
| NEGATIVE, target-limit | next is question-form cues (the follow-up named above) |

## Scaffolds to declare (SCAFFOLDS.md rows to add at registration)
- Re-reading the text stands in for hippocampal reinstatement of the encoding pattern (catalog N.07, N.12, N.14).
- A trace holds the clamped-phase activity across the free phase (the cached targets).
- lr and keep weight are host constants.
- The KP copy is as in v1.
- The update-norm clip is v1's divisive-normalisation stand-in.
- Rule form: the dendritic prediction-error rule (07-01 spec, Urbanczik-Senn); two-phase difference learning is
  contrastive Hebbian learning / GeneRec. Time-mix stays frozen, so no credit through the WKV state is needed.
- Missing prerequisite (expected): single-form replay gives weak reworded recall at 0.1B.

## Honest odds (estimates, not measured)
- G0 passes: 90%.
- Dev finds a cell at 0.1B: 45%.
- GO at 0.1B on six seeds: 30%.
- Both scales: 15-20%.

Risks:
1. The second reading is partly a "copy mode" target.
2. Residual targets drift in many directions (c).
3. Rank 8 with a KP-trained A may not separate 12 facts.
4. The 0.4B gradient six-seed mean was 2.33, so UNDEFINED by gradient < 2 is possible there.

## Compute
- CPU fp32 on pool41/42, no GPU, no AWS, through the engine.
- The real smoke sets the memcap. memory_gb is 5 (0.1B) and 10 (0.4B), from v2's measured peaks of 2.9 and 6.8 GB.
- Dev: 10 jobs per model. Six seeds: 6 jobs x 7 arms = 42 nights per model.
- Estimate from v2's wall times: 0.1B six seeds about 12-15 min per seed, 0.4B about 35-45 min per seed.
- The repo and its COMMIT are frozen from a model's dev through its score. Six-seed jobs take lr and keep from the
  engine's pick, and the score checks model and commit against the pick.

## Registration review (pre-data, sonnet, 2026-11-27)
No data existed. All points were NON-BLOCKING. Each could have wasted a run, so each is fixed before registration
(module, tests and this file):
1. **Noisy engagement gates.** The "error tail < head" gates compared the first 20 and last 20 step means, and each
   step samples one random telling of 12. One seed with a flat tail would have made the run UNDEFINED through the core
   gate, a control gate, or a dev cell failing "engaged". Fix: a fixed-order, forward-only probe of the summed
   per-layer error on all 12 tellings at each night's start and end, gated on end < start. The gradient (replay CE)
   and grad_distill (KL to the teacher) gates use the same probe. Head/tail stay as report-only fields.
2. **Clip collapses the keep grid.** The keep weight multiplied the error before the normalised step and the norm-1
   clip. Once the clip binds, keep 0.3, 1 and 3 would give identical updates. Fix: the keep weight now multiplies the
   step after the clip (also the KP copy's update), and the logged errors are unweighted. The fraction of clipped
   factor updates per arm (replay and keep) is in the engagement record. The prereg now states that the
   normalised-step claims hold only while the clip is inactive.
3. **G0 checked only the first answer token.** That token is a word fragment for 6 of the 12 facts. Fix: G0 gates on
   every token of the answer key, teacher-forced on the second reading. The first-token rate is reported. The G0 odds
   (90%) are kept, since copying is easiest on continuation tokens. This is disclosed as a stricter gate than the
   draft's.
Tests: `brain/tests/test_local_rule_v3.py` (the fixed-order probe re-runs bit-identically and falls; clip fraction
logged; the keep weight scales the step 3:1 while the clip binds; G0 full-span; engagement records carry the start/end
and clip fields; the registered specs run without `--allow-draft`).
