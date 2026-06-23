# Loop-step 3 de-risk #4 (INTEGRATION) = GO (1.000): a FULL Gen-F transformer BLOCK runs end-to-end on the bridge — ALL learned-weight matvecs (attention Q/K/V/O + MLP linears, 786,432 params) on the conductance-free RF complex-synapse path + softmax/GELU/LayerNorm as FAITHFUL READS — preserving output fidelity vs the exact-float Gen-F block (2026-06-22)

**One-line verdict:** `fullblock_rf: GEN-F(s42.real, loss=1.471) FULL block-0 forward on the bridge — ALL learned-weight matvecs (attn Q/K/V/O + MLP W1/W2 = 786432 params) on the conductance-free RF complex-synapse path (EXACT, max|Re(Z)/nsteps−h@W|=4.9e-07) + softmax/GELU/LayerNorm as FAITHFUL READS, on REAL token activations -> full-block output fidelity_vs_exact-float-teacher spearman=1.0000 cosine=1.0000 specificity_margin=0.878 shuffled_control=0.0845 LESION(scrambled-RF-weights)=0.6731~residual-floor=0.6555<<real -> GO | the two residual streams + LN/softmax-as-reads COMPOSE; the RF matvecs carry the 95%-of-norm sublayer corrections (lesion collapses to the residual floor). GO bar 0.80`

## What this de-risks (the integration milestone)
The two prior loop-step-3 de-risks proved each SUBLAYER's learned weights consolidate EXACTLY (1.000) on the conductance-free RF complex-synapse path:
- **attention projections** (`2026-06-22-genseq-loopstep3-attn-rf-distill-GO-...`): Q/K/V/O, 262,144 params = ALL of attention's learned weights, RF-verbatim 1.000; softmax = 0-param content-dependent core → faithful read.
- **MLP+GELU** (`research/findings/raw/_genseq_loopstep3_mlp_gelu_rf_distill.json`): W1+W2, 524,288 params = ALL of the MLP's learned weights, RF 1.000; GELU = exact-erf faithful read between the two exact linears (0 params).

This #4 **composes them into a FULL block forward** and asks: does the whole block (`sim/tiny_transformer.py` `_Block.forward`) run end-to-end on the bridge — the **two residual streams** + **LayerNorm-as-read** + **softmax-as-read** all working together — and preserve output fidelity? **Yes, exactly (1.000).**

## The full block + exactly how each piece was realized
`_Block.forward`: `h=LN1(x); a=attn(h,h,h,causal); x=x+a; out=x+MLP(LN2(x))`.

| Block piece | Learned matvec params | Realized as |
|---|---|---|
| LN1 (content-norm + affine) | 0 matvec (512 affine) | **faithful read** — per-feature `(x−μ)/√(var+ε)·w+b` (no cross-feature mixing → NOT a matvec; affine rides on the read) |
| attn Q proj | 256×256 | **RF** exact (`rf_linear_layer_signed`, per position) |
| attn K proj | 256×256 | **RF** exact |
| attn V proj | 256×256 | **RF** exact |
| softmax(QKᵀ/√dh) + w@V | 0 (content-dependent) | **faithful read** — the 4-head causal softmax + value mix |
| attn O proj | 256×256 | **RF** exact |
| residual 1 (`x+a`) | — | float add |
| LN2 | 0 matvec (512 affine) | **faithful read** |
| MLP linear 1 | 256×1024 | **RF** exact (per position) |
| GELU | 0 (exact-erf) | **faithful read** |
| MLP linear 2 | 1024×256 | **RF** exact (per position) |
| residual 2 (`x1+mlp`) | — | float add |
| **TOTAL learned matvec** | **786,432** | **ALL on RF (exact)** |

Biases (attn q/k/v/o, MLP fc1/fc2) + the LN affines ride on the host read (the RF matvec has no bias term). Realized on **REAL token activations** (tokenized TinyStories `"Once upon a time there was a little girl named Lily..."` → tok+pos embeddings → the genuine 90-position block-0 input `x`); fidelity scored at 8 spread probe positions.

## Result — the full block composes EXACTLY (1.000)
- **Full-block output fidelity vs the exact-float Gen-F block teacher: spearman 1.0000, cosine 1.0000.**
- **Every learned matvec EXACT on the live RF bridge:** `max|Re(Z)/nsteps − h@W|` = Wq 2.6e-07 / Wk 2.7e-07 / Wv 2.5e-07 / Wo 2.4e-07 / W1 4.9e-07 / W2 4.4e-07 (float32 precision) — the conductance-free RF complex-synapse read reproduces every projection AND both MLP linears exactly.
- The two residual streams + LN-as-read + softmax-as-read + GELU-as-read **compose with no error accumulation** — the integration holds.

## Anti-cheats — all pass decisively
1. **Specificity margin = 0.878** (matched probe-position 1.000 vs mismatched 0.122). Each real token activation maps to its SPECIFIC block output; not a constant.
2. **Shuffled-target control = 0.0845 ≪ real 1.0000** (real − shuffled = 0.916 ≫ 0.2). The RF-full-block output for position p scored against a position-DERANGED teacher collapses to ~chance.
3. **LOAD-BEARING LESION (scramble the RF complex weights of every learned matvec) = 0.6731**, vs real 1.0000. **Interpreted against the residual floor** (the precise test for a residual block): the block is residual (`out = x + attn + mlp`), so the carried-through input `x` is itself correlated with the teacher output (l2(x)/l2(teacher) ≈ 1.3; **residual-floor spearman = 0.6555** when both sublayers are zeroed). The lesion (0.6731) lands AT this residual floor — it lost the sublayer corrections, which are **95% of the output norm**. The real RF-full-block (1.000) is decisively above the floor. ⇒ **the RF matvecs carry the sublayer computation; the host softmax/GELU/LN reads do NOT manufacture the output.** (This residual-floor framing was added because a naive "lesion must drop below X" criterion is wrong for a residual stream — the floor is the carried-through identity, not zero.)

## Verdict + scope
**GO.** Full-block output fidelity 1.0000 ≥ 0.8; specificity margin 0.878 > 0.1; shuffled-control 0.0845 < real by 0.916; lesion collapses to the residual floor (0.673 ≈ floor 0.655 ≪ real 1.000, with the real result 0.34 above the floor and the lesion failing to recover the corrections). **All 786,432 of the block's learned matvec weights run on the conductance-free RF complex-synapse path; softmax + GELU + LayerNorm are faithful reads; the two residual streams compose end-to-end.**

**HONEST SCOPE — precisely what is and isn't done here:**
- This de-risks the **WEIGHTS-on-RF + nonlinearities-as-faithful-reads forward** — the path to running the full Gen-F weights on the bridge. It is **block-0 only** (one transformer block; the per-block analog-Spearman/cosine-vs-teacher metric, identical basis to the attention + MLP de-risks).
- The **fully-SPIKING nonlinearities** (spiking softmax / spiking LayerNorm / spiking GELU) are a **SEPARATE follow-on** — NOT this de-risk. Here softmax/LayerNorm/GELU are host/graded faithful reads (0 learned matvec params; the LN affine rides on the read). The named remaining pieces toward a fully-spiking block are: spiking realizations of those three parameter-free ops, the full 4-block stack, and the LM head.
- The fidelity is exact (1.000) **by construction** for the matvecs (the RF accumulator computes `a@W` exactly, verified on the LIVE bridge, not asserted) — so it is seed-independent; the integration question (does the whole block compose with the residual streams + the parameter-free reads) is what was at risk, and it is answered: it composes.

## Reproduce
```bash
SIM_BACKEND=cupy python -m research.runners._genseq_loopstep3_fullblock_rf_derisk
```
OOM-safe: 3 RF bridges co-resident (max 1280 neurons, 262K nnz) ≈ 0.04 GB ≪ 16 GB ceiling; asserted before building. ~1.5 min wall.

NO `sim/` edit (`git diff --stat -- 'sim/*'` EMPTY — the RF path + the RF-linear primitive + GELU/LayerNorm helpers ALL already exist, reuse-by-import from the attention + MLP de-risks + the RF probe). On `main`. Not committed. Raw: `research/findings/raw/_genseq_loopstep3_fullblock_rf.json`. Runner: `research/runners/_genseq_loopstep3_fullblock_rf_derisk.py`.
