# Local rule v3 DESIGN (2026-11-27): two-phase layerwise targets from the cortex's own in-context activity

STATUS: DESIGN, not a prereg. Nothing has run. Lever 4 on one defect (v1, SHY, v2, v3): the research gate fired; this
note is its output. Milestone: ARCHITECTURE section 4, 2026-12-31, sleep learning by a local, biology-style rule.

## Record brief
- **Knew (today).** v1 (output error over fixed feedback): learns only at high lr, drift 3.5x gradient; random feedback
  diverges. SHY: 1.17 vs bar 2, diverged 2/12 nights. v2 (plasticity off the keep subspace): no drift, recall 0-2/12.
  Reading: the feedback path does not put the update where the fact lives.
- **Old track (record check run; hits read).** 05-17 dendritic credit NEGATIVE: feedback never aligned (0.012). 08-01
  supervised deep credit: null, gap to oracle widens with task size. 07-13: node perturbation no edge over KP. 07-20 CLOSE: a local delta rule on a READOUT (error given at the synapse's own output)
  reaches BPTT parity. 08-02 depth finding, corrected 08-11: the "deep collapse" was a per-arm learning-rate artifact.
  Pattern: local rules work where the error is given at the layer, fail where it must travel.
- **brain/ record.** The cortex has the fact in context (in-context arm knows 55/67; ceiling 0.1B 6/12, 0.4B 11/12).
  The 11-17 teacher night used the in-context cortex as teacher, with final-logit targets and backprop.
- **New.** Every layer gets its own target, so no error travels: each adapter learns to reproduce, without the telling
  in context, the activity its layer had with it. Each layer becomes the 07-20 readout case.

## Mechanism
Two phases per replayed telling X, both forward-only, `no_grad`, on the same cortex:
1. **Clamped phase (teacher).** Adapters off. The cortex reads X twice in one stream; activity is recorded on the
   second reading, where the first reading is in context and the cortex predicts X's own tokens. Per block i and
   position t: the residual after the block, `x*_{i+1}`, and the channel-mix hidden activity `h*_i`. Cached once per
   telling.
2. **Free phase (student).** Adapters on. The cortex reads X once from the base state. Same tokens, positions align 1:1.
3. **Update (per layer, no feedback path, no weight transport).**
   - value path: `e_i = x_{i+1} - x*_{i+1}`; `dB_v = (h A_v)^T e_i`. The channel mix adds to the residual, so this is the
     exact delta rule for the layer's own loss. The adapter learns to supply what time-mix carried in from context.
   - key path: `e_h = (h - h*) * 2 relu(pre)`; `dB_k = (k_in A_k)^T e_h`.
   - A factors: v1's Kolen-Pollack copy of B (one hop inside the adapter's own rank units), unchanged.
   - Step size normalised by pre-synaptic energy (normalised LMS), so one lr serves every layer (the 08-11 lesson).
   - Errors are computed on the student's own forward, so a later layer corrects only what earlier layers left.
4. **Keep.** One keep text per step, same rule, target = the pretrained cortex's own activity on that text (error =
   the adapter's own effect). Exact and local, unlike v1's feedback-carried KL.
Episodes are v1/v2's (`build_arm_episodes`): tellings only, exactly what the gradient reference sees. Not chosen: question-form cues (the measured 55/67 competence); questions are not stored and would change
the information, not only the rule. That is the follow-up if the target proves the limit.

## Biology
- Replay reinstates the encoding pattern in cortex (CAT N.07, N.12, N.14): the clamped phase.
- Rule form `eta (phi(target-driven) - phi(input-driven)) r_pre` is the dendritic prediction-error rule in the record
  (07-01 spec, Urbanczik-Senn); two-phase difference learning is contrastive Hebbian / GeneRec.
- Time-mix stays frozen; adapters sit on the per-timestep channel mix, so no credit through the WKV state is needed
  (the prerequisite v1 and v2 lacked).
- Scaffolds to declare at registration: text re-reading stands in for hippocampal reinstatement; a trace holds
  clamped-phase activity across the free phase; lr and keep weight are host constants; KP copy as v1. Missing
  prerequisite: single-form replay gives weak reworded recall at 0.1B (PREREQUISITES).

## Arms (per model, per seed; 42/43/44/100/101/102)
1. `v3`: the rule at (lr*, keep*).
2. `gradient`: `brain.sleep.sleep`, true episodes (reference).
3. `no_sleep`.
4. `sham_ctx` (shuffled target): the clamped phase reads another fact's telling before X (seeded derangement): real
   in-context activity, no knowledge of X.
5. `no_target` (lesion): no first reading, so target = own activity. Must stay inert; any recall is a leak.
6. `top_only` (depth lesion): targets at the last block only: do layerwise targets do the work, or the readout alone?
7. `grad_distill` (report only): backprop on KL to the same teacher's final logits. Separates "bad target" from "bad
   rule".

## Gate before any night (free)
G0: on the second reading, the teacher's top-1 at the answer-word position is the answer for >= 10/12 facts per model.
Else STOP, no nights.

## Dev (seed 7, per model, frozen before the six seeds)
Grid lr {0.03, 0.1, 0.3} x keep weight {0.3, 1, 3}; 240 steps; plus `no_sleep` and `gradient`. Pick: highest recall
with generic and held-out KL (dev half) <= 1.25x gradient's; tie to lower drift; floor = bar (a). No cell:
NEGATIVE-DEV, six seeds do not run.

## Measures
`sleep_probe.evaluate` (12 told, hippocampus off; 4 never-told; generic drift; paraphrase) plus v2's held-out KL
(verdict half). Per unit: host, commit, lr, keep, per-layer error norm first/last (where the fact lives), adapter norms.

## Pass rule (per model; six-seed means)
- (a) recall: `v3` >= max(0.5 x gradient, no_sleep + 2).
- (b) contrast: `v3` > `sham_ctx` on >= 5/6 seeds, and `sham_ctx` <= no_sleep + 1.
- (c) drift: generic KL and held-out KL each <= 1.25x gradient's; no night with KL > 1.
- (d) `no_target` recall <= no_sleep and adapter norm at zero.
- (e) false recall <= gradient + 1.
- UNDEFINED: gradient < 2; `sham_ctx` or `top_only` inert (adapter norm zero); (d) violated.
- Verdicts: GO-V3 (a-e at both scales); GO-V3-PARTIAL (one scale); GO-SHALLOW if a-e hold but `v3` - `top_only` < 1
  (local, but only the known readout result); NEGATIVE; NEGATIVE-DEV.
- Reading a NEGATIVE: `grad_distill` below the bar too = the single-form target is the limit, next is question-form
  cues; `grad_distill` passes = the delta rule fails even with aligned targets, and the delta-rule family stops at
  this adapter rank (stop rule for the gate).

## Compute
Engine specs `tools/engine/specs/local_rule_v3_{01b,04b}.yaml` plus verdict module; the real smoke sets memcap.
Expected as v2 (0.1B peak 2.9 GB -> cap 5; 0.4B 6.8 GB -> cap 10; fits 15 GB nodes): targets cached once per telling
(a few MB), two forwards per step, as v1. Dev 11 nights per model; six seeds x 7 arms = 42 nights per model,
pool41/42, 0.1B first. No AWS, no GPU. To check at registration: `_told` layout for the double reading; v2's
measured night time.

## Honest odds (estimates, not measured)
G0 passes: 90%. Dev finds a cell at 0.1B: 45%. GO at 0.1B six seeds: 30%. Both scales: 15-20%. GO-SHALLOW is a live
outcome. Risks, most likely first:
1. The second-reading state is partly "copy mode"; it may store the sentence yet transfer to the question worse than
   gradient's own solution.
2. Residual targets move many directions, so drift (c) may fail as in v1.
3. Rank 8 with a KP-trained A may not separate 12 facts.
4. 0.4B gradient six-seed mean was 2.33: UNDEFINED is possible there.

## Why this is not a killed lever
- v1, SHY and v2 share one error source: output error sent backward over fixed feedback. v3 has no backward path;
  every error is target minus actual at the synapse's own layer. v3 does not use v2's projection either.
- The teacher night NO-GO concerned abstention (refusals distilled as an unconditional reply) under backprop at 1.5B;
  v3 tests storage of told facts by a local rule.
- The old-track negatives read are "error must travel" cases; the positive (07-20) is the case v3 reduces to.
