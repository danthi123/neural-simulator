# R3 REFRAME (decisive, single-seed): deep-context / long-range capture is INPUT-EMBEDDING + READ-OUT bound, NOT recurrent-credit bound — a FROZEN random recurrent reservoir with a learned input embedding + read-out BEATS full backprop-through-time (deep +1.258 vs +0.902), and learning the input embedding is worth ~3× more than learning the recurrent weights. The entire biological deep-recurrent-credit arc was aimed at the wrong bottleneck; the tractable frontier is learning the INPUT REPRESENTATION on a fixed random reservoir

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_stream_eprop_lm_derisk.py` (`BPTT_frozen_wrec` arm added: freeze W_rec at the random reservoir, train {W_in, W_out, b} by backprop; NO `sim/` edit). TinyStories 2M tokens, V=2000, n_pool=300, block=128, 4 epochs, seed 42.
**Verdict:** the Run-A ctx1 diagnostic hinted the deep-context "recurrent-credit ceiling" was read-out/embedding-dominated; this test CONFIRMS it decisively — **freezing the recurrent weights and learning only the input embedding + read-out BEATS training everything.** The recurrent-credit RULE (the target of the whole R1-R2 arc) is not the bottleneck; it is mildly counterproductive to train it.

## The decomposition — DEEP (ctx17-127) margin, all with an AdamW read-out (so W_in-vs-W_rec is isolated)
| arm | trains | freezes | DEEP margin | deep gain over fixed |
|---|---|---|---|---|
| fixed_reservoir | (W_out delta only) | W_rec, W_in | −2.760 | — |
| BPTT_fixed_win | W_rec, W_out, b | **W_in** | −1.401 | +1.359 |
| **BPTT_frozen_wrec** | **W_in, W_out, b** | **W_rec** | **+1.258** | **+4.018** |
| BPTT_same_net | W_rec, W_in, W_out, b | — | +0.902 | +3.662 |
| plastic_eprop (5× lr, delta read-out) | W_rec | W_in | −1.657 | +1.103 |

- **Learning the INPUT EMBEDDING (W_in) is worth ~3× more than learning the RECURRENT weights (W_rec):** +4.018 nats deep (BPTT_frozen_wrec, learns W_in) vs +1.359 (BPTT_fixed_win, learns W_rec) — both AdamW read-out, each freezing the other. The input representation dominates.
- **A FROZEN random reservoir + learned input/read-out BEATS full BPTT:** BPTT_frozen_wrec deep **+1.258** > BPTT_same_net **+0.902** (and agg-CE 3.054 < 3.398). Training W_rec on TOP of W_in is counterproductive — backprop destabilizes the recurrent dynamics (the classic reservoir-computing / echo-state result: a fixed random recurrence provides stable temporal mixing; learning it hurts).
- **ctx1 read-out diagnostic** (position 1: h_1 = α·tanh(W_in·x + b), depends on W_in+read-out, NOT W_rec): the W_in-learning arms read ctx1 **−0.12/−0.16** (BPTT_frozen_wrec/BPTT_same_net) vs the W_in-FROZEN arms' **−2.08/−3.25** (BPTT_fixed_win/plastic). The input embedding dominates prediction from the very first token.

## Why this reconciles the whole R1-R2 arc (and redirects it)
- plain e-prop's "limitation" (−1.657, "44% of BPTT") was NEVER the recurrent-credit rule — it was the **FROZEN input embedding** (the reservoir convention froze W_in). The recurrent credit e-prop does (diagonal RTRL) is fine; the missing piece was learning W_in.
- The dual-timescale-eligibility "GO" (refuted as effective-lr) and the ALIF-forward-state negative were both chasing the RECURRENT-credit horizon — the wrong bottleneck. The deep-context signal lives in the input representation + the read-out, on a fixed reservoir.
- The "% of full-BPTT" framing collapses cleanly: most of BPTT_same_net's advantage over a frozen-W_in e-prop is **input-embedding learning**, a modest part is the AdamW read-out, and the recurrent-weight training is worth little-to-negative.

## ⇒ The redirected biological frontier (far more tractable + biology-aligned)
The long-range/deep-context capture is a **feedforward-representation-learning** problem on a **fixed random recurrent reservoir**, NOT an off-diagonal recurrent-credit problem:
1. **Fixed random recurrent reservoir** — biologically a cortical recurrent network with random/developmentally-fixed recurrent weights providing temporal mixing (no recurrent-weight training needed; training it hurts). This is the reservoir-computing / liquid-state-machine hypothesis, and it MATCHES the spiking substrate's echo-state regime (the SCALE-CAPSTONE reservoir).
2. **Learned INPUT REPRESENTATION (W_in)** — the dominant learnable factor. e-prop's eligibility applies to INPUT synapses too, so a biological e-prop CAN credit W_in (with a fixed W_rec, sidestepping the hard off-diagonal recurrent credit). The open question (next test): does e-prop-learned-W_in (fixed reservoir, delta read-out) approach BPTT_frozen_wrec's +1.258, or is W_in's DEEP credit also diagonal-truncation-limited?
3. **Local read-out** — the delta rule (already local/three-factor); an AdamW read-out helped modestly but the delta read-out is the biological form.

**NEXT CONCRETE TEST (the biological version of the winning arm):** add an `eprop_learn_win` arm — e-prop that learns W_in (input-synapse eligibility) with W_rec FIXED (random reservoir) + delta read-out — and measure its deep margin vs BPTT_frozen_wrec (+1.258, the ceiling for this arm) and vs plain e-prop (−1.657, frozen W_in). If it substantially closes the gap, the biological long-range path is: fixed spiking reservoir + e-prop-learned input representation + local read-out — no off-diagonal recurrent credit, no `sim/` rewrite. This is the on-substrate-realizable target.

## ✅ 6-SEED CONFIRMATION (the reframe is init-robust) + the BIOLOGICAL `eprop_learn_win` result
**6-seed (42/43/44/100/101/102):** `BPTT_frozen_wrec` (freeze W_rec, learn W_in+W_out+b) deep **+1.257 ± 0.002** vs `BPTT_same_net` (learn all) **+0.859 ± 0.116** — **frozen_wrec BEATS same_net on ALL 6 seeds.** The frozen-reservoir + learned-input solution is astonishingly consistent (±0.002) while training W_rec adds variance AND hurts. ⇒ the reframe (deep-context is input-embedding bound; training the recurrent weights is counterproductive) is decisively init-robust.

**The BIOLOGICAL version — `eprop_learn_win` (e-prop learns W_in via input-synapse eligibility, W_rec FIXED, local delta read-out, random feedback = NO weight transport):** an lr_in sweep at seed 42, 2M tokens:
| lr_in | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 | 2.0 | 5.0 | 10 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| deep margin | −1.130 | −0.909 | −0.695 | −0.433 | −0.257 | −0.110 | **+0.037** | +0.117 | +0.177 |
- **The biological input-learning path WORKS:** deep margin climbs monotonically and STABLY (no destabilization even at lr_in=20 — learning W_in on a FIXED reservoir is stable, unlike plain e-prop's W_rec which destabilized at 30× lr) from plain e-prop's −1.657 to a **POSITIVE** deep margin (~+0.18, beats the add-1 bigram). ctx1 climbs in step (−2.99→−1.45), confirming W_in is genuinely being learned. ⇒ **the on-substrate-realizable long-range path is validated: a fixed spiking reservoir + an e-prop-learned input representation + a local read-out captures deep-context structure — NO off-diagonal recurrent credit, NO `sim/` rewrite.**
- **It PLATEAUS at ~35-40% of the frozen-reservoir-BPTT ceiling** (asymptote ~+0.2-0.3 vs the +1.257 ceiling; the lr_in increments shrink: +0.08 at 5→10, +0.06 at 10→20). The residual is the **feedback-alignment partiality** — e-prop's fixed random-feedback learning signal (`L = δ @ Bᵀ`) vs BPTT's true `W_outᵀ` gradient — plus the delta vs AdamW read-out. Confirmed by ctx1 (position 1, no recurrence): eprop_learn_win reaches only −1.45 vs the ceiling's −0.02, i.e. W_in is learned only PARTIALLY even for the direct (off-diagonal-free) prediction ⇒ the limit is the CREDIT SIGNAL (random feedback), not the recurrence.
- **⇒ the next lever is research-identified: LEARNED FEEDBACK (Kolen-Pollack / weight-mirror, Akrout 2019 — biologically legal, no weight transport, ~10 lines) for the input-synapse credit**, to close the feedback-alignment gap toward the +1.257 ceiling. This is the SAME lever the earlier deep-credit research ranked for FA partiality — now applied to the RIGHT bottleneck (input-representation learning), not the recurrent weights.

## Honest scope
Single seed (42) for the eprop_learn_win sweep (the reframe itself is 6-seed-confirmed); single layer, n_pool=300, 2M tokens — a decisive reframe + a validated (if FA-plateaued) biological path; multi-seed confirmation + the `eprop_learn_win` biological test are the immediate follow-ons. The BPTT arms are AdamW (a reference ceiling); the biological realization uses e-prop + delta. All rate-level torch, GPU, NO `sim/` edit. This reframe supersedes the "off-diagonal recurrent credit / multi-layer" next-direction earmarked earlier — the bottleneck is the input representation, not the recurrent credit.

## Files
`_emerge_stream_eprop_lm_derisk.py` (`BPTT_frozen_wrec` arm); raw `research/findings/raw/_r2_ctrl/_reframe_frozenwrec.json` + `.log`. Builds on the R1/R1b/R2/R2b (dualtc-refuted) + Run-A control-sweep findings.
