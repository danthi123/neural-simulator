# Loop-step 3 ATTENTION-on-RF = GO (1.000): Gen-F's attention PROJECTIONS (Q/K/V/O, all 262K of attention's learned params) consolidate EXACTLY onto the conductance-free RF complex-synapse path — and the [VERIFY] is reconciled (the REAL Gen-F generator, NOT cortex_10M, is the teacher); the softmax(QKᵀ) content-dependent core (0 learned params) is the HONEST deferred part (2026-06-22)

**One-line verdict:** `attn_rf_distill: GEN-F(s42.real, loss=1.471) block-0 attention PROJECTIONS (Q/K/V/O, 262144 params = ALL of attention's learned weights) consolidated onto the conductance-free RF complex-synapse path (the SYNTHESIS's no-g(V−E) escape) on REAL token activations -> installed-on-live-RF-bridge projection_fidelity_vs_teacher=1.000 (best=verbatim; RF-VERBATIM=1.000 EXACT; clip-aware-distill=−0.326 — WRONG tool for a projection, see note) specificity_margin=0.991 shuffled_control=−0.216 -> GO | DEFERRED: softmax(QKᵀ) content-dependent core (0 learned params, NOT a fixed matvec -> own de-risk / graded op). GO bar 0.80`

**Scope:** the consolidation ladder's NEXT target after the MLP synthesis WIN (`2026-06-22-genseq-loopstep3-rf-distill-GO-cheap-ladder-WINS.md`, 0.872). `research/runners/_genseq_loopstep3_attn_rf_distill_derisk.py`, GPU (`SIM_BACKEND=cupy`). **NO `sim/` edit** (`git diff --stat -- 'sim/*'` EMPTY; the RF path + the clip-aware trainer + the install/measure machinery ALL already exist, reuse-by-import from the MLP-winning runner + the RF probe). On `main`. Not committed.

## The [VERIFY] reconciliation (the prompt's load-bearing open)
The prior loop-step-3 de-risks used `cortex_10M_seed42.npz`'s dense MLP slice as the **load vehicle**. The REAL consolidation target is **Gen-F** = `sim/tiny_transformer.py` `TinyGPT` — the WORKING fluent generator (`generator_f_gate.ckpt.s42.real.pt`, **loss 1.471**, 12000 steps). This de-risk uses Gen-F's **ACTUAL** block-0 `nn.MultiheadAttention` weights as the teacher, on **REAL token activations**: tokenized TinyStories ("Once upon a time there was a little girl named Lily...") → tok+pos embeddings → the block-0 LN1 output `h` (8 probe positions × 256-dim) — the genuine input `attn(h,h,h)` sees. Not cortex_10M; not a synthetic one-hot.

## The honest decomposition (attention is harder than the MLP — the softmax is content-dependent)
`nn.MultiheadAttention(d_model=256, n_head=4)` is, per the saved `state_dict`:
- `in_proj_weight` (768,256) = stacked **[W_Q; W_K; W_V]** (each 256×256) — **LINEAR PROJECTIONS**
- `out_proj.weight` (256,256) — **LINEAR PROJECTION** (W_O)
- `softmax(Q@Kᵀ / √d_k)` — the **CONTENT-DEPENDENT** attention-weight computation

Decompose:
- **The 4 projections (Q/K/V/O) are pure linear matvecs** `y = h @ W`. The RF complex accumulator computes `h @ W` **EXACTLY** (Re(Z)/nsteps, ω≈0, λ=0 — rank 1.000, the rf-PARTIAL finding). There is not even a clip in a projection → it is the **IDEAL/trivial RF case**. 4 projections = **262,144 params = ALL of attention's learned parameters.**
- **The softmax(QKᵀ) core is the genuinely-nonlinear part** — NOT a fixed per-layer matvec (the attention weights depend on the input). NOT RF-consolidatable by this mechanism. **ZERO learned parameters.** → its own de-risk (a linear-attention / GLA approximation — harder) OR a graded/host op for now.

## Result — the projections install EXACTLY on the live RF bridge (the [VERIFY] holds)
| Projection (256×256) | RF-VERBATIM install vs teacher | specificity margin | shuffled-control vs REAL teacher |
|---|---|---|---|
| W_Q | **1.000** | 0.927 | −0.205 |
| W_K | **1.000** | 1.029 | −0.255 |
| W_V | **1.000** | 1.012 | −0.127 |
| W_O | **1.000** | 0.998 | −0.277 |
| **CUMULATIVE** | **1.000** | **0.991** | **−0.216** |

`max|Re(Z)/nsteps − h@W| ≈ 7e-8` (float32 precision) — the live RF complex-synapse read reproduces each Gen-F projection **exactly**. No clip, no `g·(V−E)`, no ceiling.

## The HONEST distillation note — the clip-aware DISTILLATION step is the WRONG tool for a projection (and correctly fails)
The MLP-winning **clip-aware distillation** arm scores **−0.326** here (below chance). This is **EXPECTED and informative**, NOT a failure of the synthesis:
- A projection output `y = h@W` is **NOT in [0,1]** (l2 ~9). The trainer's `clip(a@W',0,1)` forward **clamps** it → DESTROYS the linear rank → the distilled un-clip read diverges.
- The MLP needed clip-aware distillation **BECAUSE the MLP has a per-layer CLIP** that compressed rank (rf-verbatim 0.556 → distill 0.872). A linear projection has **NO clip**, so it is the trivial case: the conductance-free RF accumulator **ALONE** is exact.
- ⇒ the SYNTHESIS that won — **install on the no-`g·(V−E)` RF complex-synapse path** — consolidates attention's projections **PERFECTLY (1.000)**; the distillation sub-step is simply **unnecessary** for a pure-linear matvec. **The verbatim arm IS the RF-faithful answer.** The prompt's "via the same RF-faithful clip-aware distillation" question is answered precisely: the SHARED escape (the RF path) transfers and wins; the clip-specific sub-step is the MLP's tool, not the projection's.

## Anti-cheats — both pass decisively
- **Specificity margin = 0.991** (matched probe-position 0.99 vs mismatched ~0.0). Each real token activation maps to its SPECIFIC projection output; the install is not a trivial constant.
- **Shuffled-target control = −0.216 < real 1.000 by 1.216** (≫ the 0.2 bar). Distilling each projection to a position-DERANGED teacher (real `h`'s, permuted targets) → installs → scores ANTI-correlated vs the real teacher. The wrong target → wrong weights, as required.

## Verdict + what it routes to
**GO.** Installed-on-live-RF-bridge projection fidelity **1.000 ≥ 0.8**, specificity margin re-opens (0.991 > 0.1), shuffled-control below real (−0.216, real − shuffled = 1.216 > 0.2). **Consolidated cheaply: the 4 attention PROJECTIONS = 262,144 params = ALL of attention's learned weights**, installed exactly on the conductance-free RF complex-synapse path (the same substrate-native escape the MLP synthesis used). **DEFERRED: the softmax(Q@Kᵀ) attention-weight core** — content-dependent, **0 learned params**, NOT a fixed matvec → its own de-risk (linear-attention / GLA approximation) or a graded/host op. This is the HONEST partial scope: attention is harder than the MLP because of the softmax, but the projections (all of attention's *parameters*) consolidate trivially-exactly, and the softmax's *computation* (not its weights) is the bounded remaining piece.

**Honest scope:** block-0 attention only (one transformer block; the per-block analog-Spearman-vs-teacher metric, identical basis to the MLP synthesis); the per-projection install is exact and seed-independent (it is the linear matvec, rank 1.000 by construction — verified on the LIVE bridge, not asserted). The softmax core + the full 4-block stack + the end-task (next-token) head are the named follow-ons. The whole-attention consolidation = the 4 projections on RF (this de-risk) + the softmax(QKᵀ) as either its own RF/GLA de-risk or a graded op.

NO `sim/` edit; not committed. Raw: `research/findings/raw/_genseq_loopstep3_attn_rf_distill.json`. Runner: `research/runners/_genseq_loopstep3_attn_rf_distill_derisk.py`.
