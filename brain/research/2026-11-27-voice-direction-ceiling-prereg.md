# Oracle source-direction ceiling prereg (2026-11-27)

STATUS: REGISTERED 2026-11-27 after pre-data sonnet review. Nothing here has been run on a model.
Fixes from the review (all made before any data):
1. H-reproduction gate strengthened: the reference must cover at least 0.9 of the questions (n), AND replies identical on at least 0.9 of matched rows, AND onsets equal (|diff| < 1e-4, the onset is sampling-free) on at least 0.9. The A0snap hippocampus-off reference is kept; voice_split B rows stay report-only.
2. Voice control added: clause 3b, ORACLE P(uv|knows) <= SHAM P(uv|knows) - 0.10 (pooled point estimate). P(uv|knows) is now reported for H, SHAM, ALL, WRONG too.
3. Engagement proven per row: each vector-arm row records `label_tagged` (the `Assistant:` label position is in the footprint) and `vec_norm`; the harness asserts the cortex is active and the abstain gate off at session start. UNDEFINED if any label is untagged or an arm's onset equals H's (shift < 1e-6) on more than half of a seed's rows.
4. Cross-seed consistency: `table` is UNDEFINED on duplicate seeds or on outputs that disagree on commit, night or layer.
5. MIXED is now described as any other clause failing (section 4). A primary item with no knows rows in H or ORACLE is "not evaluable" (reported with per-item row counts), not a failed direction; if no primary item is evaluable the items clause fails.
6. Tests: exact p_uv bar (54 of 60 passes, 55 fails), voice-control clause alone, unevaluable item, no-op arm, inconsistent outputs, reproduce coverage and onset, voice_split reference format, table and handread CLI, and one tiny-cortex run through the real tag path (ORACLE onset differs from H, SHAM has equal norm).
Tests use a stub life and a tiny random cortex only (`brain/tests/test_voice_ceiling_oracle.py`).
Capability: Self and identity (PREREQUISITES: self/other tag, missing). Replaces the DO-NOT-RUN source-monitor port
(`2026-11-27-fable-review-source-monitor-port.md` section 3). A verdict here is on the one-vector residual-tag CLASS at 1.5B, not on the capability.

## 1. Question
Can any one-vector residual signal on the brain's own-production positions move told-fact voice in the slept weights
without losing the fact? Measured with an oracle vector read off the cortex itself. Forward passes only, nothing trained, so
an oracle direction is an upper bound on a learned one.

## 2. Data and reads
- Banked n14 adapters (A0snap) of the KR lives, seeds 319/320/321. The 67 told questions (`voice_tag.told_questions`), hippocampus off, fresh state per question.
- Per question, at the last position of `User: q\n\nAssistant:` (the label), the residual after every block:
  H (nothing before), C (the item's telling heard first, as voice_split ctx), W (an unrelated item's telling heard first).
- WRONG context = the telling of the next item in sorted item order (a world-authored telling of another item; no host English).
- d_L(item) = mean over the item's questions of (C - H); dW_L likewise with W. Oracle v_L(i) = mean of d_L over the OTHER items
  (leave-one-item-out, items weighted equally, per seed). v_WRONG(i) = same from dW, also excluding the item whose telling was i's unrelated turn.
- Primary L = n_layers // 2 (the builder's layer, 12 at 1.5B). L = 6 and 18 are report only (ORACLE and SHAM; `--report-arms` adds ALL, WRONG).

## 3. Arms (all hippocampus off, asked as `voice_tag.battery` asks A0snap; paired reply draws)
| arm | applied |
|---|---|
| H | nothing. Must reproduce the A0snap hippocampus-off rows (section 5 UNDEFINED) |
| C | telling heard first, then the question. Reference ceiling |
| ORACLE | v_L on `Provenance("own")`: the `Assistant:` label plus tokens fed back after sampling, via `PlasticCortex.tag_mask_fn` |
| SHAM | random direction, norm exactly that of the question's ORACLE vector, same footprint |
| ALL | v_L on every position (`Provenance("all")`, no contrast) |
| WRONG | v_WRONG at L on the own footprint |
The lesion is the H arm (no weights change). Host: `run` refuses unless POOL_NODE=1 or BRAIN_ALLOW_W3090=1; per-unit atomic results with resume; host, commit, device recorded.

## 4. Measures and rule (fixed before data)
Measures on told rows: onset = log(p(" You")+p(" Your")) - log p(" My") at the first answer token (higher = second person), generated-reply
P(user voice | knows) (`cascade_screen.own2`, `voice_split.knows`), knows, refusals. Bootstrap: 2000 draws, ITEM-clustered (item names
resampled, every seed's cells of a drawn item come with it; 95% percentile CI), seed 20261128. Per item pixel, marguerite, volvo reported separately.
Report-only: P(uv|knows) CI, C minus H onset, vector norms and cos(ORACLE, WRONG), blind 40-row hand-read (`handread`: H, C, ORACLE rows shuffled, arm key in a separate file).

CLASS-ALIVE, all of (primary layer):
1. onset ORACLE - H CI lower bound > 0;
2. ORACLE - SHAM, ORACLE - ALL, ORACLE - WRONG onset, each CI lower bound > 0;
3. P(uv|knows) ORACLE <= H - 0.10 (pooled point estimate); 3b (voice control) ORACLE <= SHAM - 0.10 on the same measure;
4. knows ORACLE >= 0.9 of H;
5. refusals up by less than 10% of H's told rows;
6. direction (onset diff > 0 and P(uv|knows) lower) on at least 2 of the 3 seeds, and on pixel and marguerite individually (pooled over seeds; an item with no knows rows in H or ORACLE is not evaluable and does not count as a failure, at least one primary item must be evaluable; per-item row counts are reported).
CLASS-CLOSED: clause 1 or clause 3 fails (an oracle direction is the upper bound on a learned one; no residual tag at 1.5B).
MIXED: clauses 1 and 3 pass and any other clause fails (controls, voice control, knows, refusals, seeds, items); the failed clauses are named. Knows falling with voice moving = fact and frame entangled; write up and stop.
Precedence: UNDEFINED > CLASS-CLOSED > CLASS-ALIVE > MIXED.

## 5. UNDEFINED
H reproduction failing on any seed: reference covers fewer than 0.9 of the questions, or replies identical on fewer than 0.9, or onsets equal (|diff| < 1e-4) on fewer than 0.9 of matched rows (reference: the source-tag run's A0snap hippocampus-off rows, same code path; voice_split B rows are reported, not gated, because voice_split seeds once per arm and not per question, so sampled replies are not expected to match); no reference; H knows < 10 pooled;
a seed or battery missing (needs all of 319/320/321); a footprint that tags nothing, or every token on a non-ALL arm, or leaves the `Assistant:` label untagged on any row; a vector arm whose onset equals H's on more than half of a seed's rows (no-op); duplicate seeds, or outputs disagreeing on commit, night or layer; banked adapters not live (the harness raises).

## 6. Cost and memory (estimates, nothing measured)
Forward only (`torch.no_grad`, no backward, no Adam state). Expected RSS 8-10 GB at 1.5B fp32 (about 6.2 GB weights shared with PlasticCortex, plus load transient and activations): fits a 15 GB node, one process per node; `memcap` default 14 GB. Do not run two seeds on one 15 GB node.
Per seed: reads (3 passes per question, two with a generated telling reply) about the cost of 2 batteries, then 6 primary batteries + 4 report-layer batteries at 90-130 s each: about 25-35 min per seed (estimate from the 90-130 s per battery figure). Three nodes in parallel about 35 min; one node about 1.5 h. $0.

## 7. If ALIVE / CLOSED
ALIVE: a learned-tag question is well posed (known layer, norm, direction); build after test2 with real two-speaker replay (HANDOFF reopen condition). CLOSED: the voice arc stays killed, test2 runs with the defect documented, effort goes to routing and the 12-31 local-sleep de-risk.

## 8. Risks for the pre-data review
- The residual is read at the final label token only; the added vector also lands on the earlier `Assistant` token and on fed-back tokens (builder's footprint), so ORACLE may be over- or under-strong versus the measured d. SHAM norm-matches ORACLE, not the signal.
- C may not move onset at all (then d is noise and CLOSED is uninformative); reported as C minus H, not gated.
- 3 seeds, about 22 questions per seed, items shared across seeds (hence item clustering); a pass on 2 of 3 seeds is weak evidence.
- `hear(telling)` runs the abstain gate; if it routes off consolidated synapses the ctx read differs from voice_split's both-off ctx.
