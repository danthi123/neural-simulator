# Dense SELF-SUPERVISION substantially cracks the D3 INTERNAL-COMPARE relational reference-δ (the "who holds it now" holder-possession residual) — +65% over the end-state-only residual, beating every shortcut, NO host state label — but is capped by TRANSITION-LEARNING QUALITY: hard-attractor re-discretization is REFUTED as the fix (it backfires on a soft transition), sharply mapping the boundary to a need for a sharper per-step signal

**Date:** 2026-07-14
**Runner:** `research/runners/_d3_reference_selfsup_derisk.py`. Raw `research/findings/raw/_d3_reference_selfsup.json`. numpy CPU; NO `sim/` edit.
**Status:** PARTIAL-advance + sharp BOUNDARY (6-seed) — dense self-supervision substantially cracks the internal-compare relational-δ (SELFSUP_deep 0.476, ~2.85× chance, **+65% over the 0.289 end-state-only residual**, all controls collapse), but is capped at 0.476 by TRANSITION-LEARNING QUALITY: the indirect emission under-constrains the exact comparison. Straight-through hard-attractor re-discretization was REFUTED as the fix (it backfires on a soft transition). The named next lever = a SHARPER per-step signal.

## Why (the D3 honest residual)
The D3 discourse-referent tracking arc (`2026-07-09-D3-language-reference-tracking-GO.md`) tracks WHO-HOLDS-IT across a possession narrative (δ: `holder := b if holder==a else holder` for clause (subj=a, obj=b) — a state-dependent, non-commutative, INTERNAL-COMPARE relational rule, the operation behind multi-turn anaphora). Its honest residual: learning this RELATIONAL δ from WEAK (end-state-only) supervision reached only **0.289** (`_d3_sparse_supervision_derisk.py`; per-step supervision 0.872 but any sparsity collapsed to 0.232) — the relational comparison-δ does NOT interpolate from sparse anchors, UNLIKE the LOOKUP DFA (which the RANK-1 end-state-only recipe cracked to 1.0). The next mechanism named (frontier gate 2026-07-14): apply the proven RANK-3 DENSE SELF-SUPERVISED emission-CE recipe (which cracked the OBSERVABLE-cued agent-tracking δ, `_d3_event_selfsup_derisk.py`) to this harder INTERNAL-compare relational-δ.

## The mechanism (single-variable de-risk)
The K=6 holder-possession task is byte-identical (n_pool=64, XOR-over-pool subj/obj clause code, curriculum train_lens=(1,2,3) → held-out-DEEPER (6,7,8), forced-no-op-last-clause so recency/last-named/retention floors are at chance). The ONLY change is the supervision: each clause EMITS a symbol from the CURRENT holder's characteristic distribution θ[holder_t] (M=8≥K), a **TARGET ONLY, NEVER an input** (the forward pass + rollout read ONLY the subj/obj clause code; the emission enters ONLY the backward CE target). The `discrete_attractor_rnn` δ is trained by emission-CE ALONE — no host state label. To predict the moving emission the model MUST maintain the running holder (internal-compare state). Eval = a FROZEN linear probe (final hidden state → holder identity), depth-conditioned.

## Result 1 — dense self-supervision substantially cracks the internal-compare δ (6-seed, soft rollout)
| arm | mean (6-seed) | reading |
|---|---|---|
| **SELF-SUP deep (held-out 6/7/8)** | **0.476** (0.396–0.53) | ~2.85× chance; **+0.19 over the 0.289 end-state-only residual = +65% relative** |
| SELF-SUP shallow (train lens) | 0.840 | the δ is learned well at trained depth |
| TF step-δ (per-step, teacher-forced) | 0.997 | the per-step δ is learned NEARLY PERFECTLY from self-supervision |
| TF track deeper (hard-attractor, teacher-forced) | 0.813 | the hard-attractor rollout ceiling |
| emission_severed (θ random) | 0.173 | COLLAPSES — the emission↔holder link is load-bearing |
| no_recurrence | 0.199 | COLLAPSES — recurrence load-bearing |
| fair_reservoir (deep subset) | 0.237 | the mechanism BEATS a proper 512-dim echo-state |
| floor recency / last-subject / retention | 0.166 / 0.000 / 0.173 | all ≈ chance — no shortcut |

Integrity: **finite-difference gradient check PASS all 6 seeds** (rel err 1.4e-5); **emission-target-only audit PASS** (forward hidden states byte-identical under a shuffled emission — the emission is structurally un-leakable into the probe). ⇒ dense self-supervision genuinely learns the internal-compare relational-δ (all controls collapse), a large improvement over the documented residual.

## Result 2 — hard-attractor re-discretization REFUTED as the fix; the residual is TRANSITION-LEARNING QUALITY, not rollout drift (6-seed)
Hypothesis (from Result 1): the gap is autoregressive soft-slot rollout DRIFT → straight-through HARD-attractor re-discretization should close it. **REFUTED — none of three re-discretization forms beats the plain soft rollout:**
| rollout arm (SELF-SUP deep, 6-seed) | mean | reading |
|---|---|---|
| SOFT (no re-discretize) | **0.476** | the baseline |
| HARD (straight-through, train+eval) | 0.456 | ≤ soft |
| SOFTtrain / HARDeval (pure test-time snap) | 0.408 | **< soft — the decisive diagnostic** |

**The clincher:** if the gap were pure test-time drift, snapping a WELL-TRAINED soft model to attractors at rollout (SOFTtrain/HARDeval) would LIFT it toward the TF ceiling — instead it LOWERS it (0.408 < 0.476). Re-discretization commits to a wrong argmax and locks in the error; the soft blend preserves self-correcting uncertainty. ⇒ **the residual is TRANSITION-LEARNING QUALITY, not rollout drift.** Self-sup learns the internal-compare δ only to shallow **0.771** (vs TF step-δ **0.997**) because the emission is an INDIRECT, entity-characteristic readout (purity 0.71) that UNDER-CONSTRAINS the exact `holder==a?` comparison. Re-discretization presupposes a SHARP transition (which TF has — why it helps TF + the group-composition arc), so on a soft transition it BACKFIRES. (Integrity: SOFT-path + HARD-exact-sub-path FD checks PASS; the HARD STE sub-path rel-err ≈ 1.0 is the designed straight-through surrogate — argmax has zero true gradient — correctly reported, not a bug; emission-audit PASS.)

## ⇒ dense self-supervision substantially ADVANCES the internal-compare relational reference-δ (+65% over the residual, beats every shortcut), capped by transition-learning quality — a sharp, precisely-mapped boundary
Dense emission-CE self-supervision — NO host state label, emergent from a per-clause prediction signal — genuinely learns the internal-compare relational-δ (the operation behind multi-turn anaphora) to 0.476, a **+65% advance over the documented end-state-only residual** (0.289), decisively above every shortcut. The honest boundary: the exact internal-compare comparison needs a SHARPER per-step signal than an indirect entity-characteristic emission provides (re-discretization refuted). **Named next levers (diagnosis-directed):** (1) higher emission PURITY / bits-per-clause (directly sharpens the per-step signal — tests the indirectness diagnosis); (2) DETACHED-HARD "self-teacher-forcing" (carry `emb[argmax]` detached during training — clean-attractor prev inputs like TF, but without the STE bias; the mechanistically-closest analog to the recipe that reaches 0.81); (3) a light state-anchor auxiliary. NO `sim/` edit.
