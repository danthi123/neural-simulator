# Cumulative / staged stream-cortex training — preservation GO, same-vocab compounding NEGATIVE at scale (2026-06-26)

## CORRECTION of the smoke-only "GO"

An initial 16-concept numpy smoke suggested "compounding GO" (corr 0.733→0.745). The **full-scale GPU run
(1,000 concepts, 7,000 windows/stage) contradicts it.** The smoke sat in the UNSATURATED regime (16 concepts at
400 windows have headroom, so more training trivially helps); at realistic scale the same-vocab resume is
inert-to-harmful. This doc supersedes the premature `-GO` finding.

## Results

**PRESERVATION (the mechanism) — GO at BOTH scales.** `corr(M_reload, M_saved) = 1.000000`. Checkpoint
save/resume restores the learned co-occurrence M (the rate-Hebbian `cp_connections`) byte-for-byte. The
`--save-bridge`/`--resume-bridge` mechanism (reuse `save_checkpoint`/`load_checkpoint`, **NO `sim/` edit**)
works exactly as designed.

**SAME-VOCAB COMPOUNDING (train MORE windows on the SAME data after resume) — scale-dependent:**

| scale | windows (N → 2N-eff) | corr(M,C) | recall |
|---|---|---|---|
| smoke (16 concepts, numpy) | 400 → 800 | 0.733 → **0.745** | 1.000 → 1.000 |
| full-scale (1,000 concepts, GPU) | 7,000 → 14,000 | 0.838 → **0.823** (flat/down) | 0.625 → **0.292** |

The full-scale 2N-effective run is **past the documented window optimum** (recall peaks ~3–7K windows,
densifies after ~8K — see CLAUDE.md "1454 breadth window sweep"). Re-training the same vocab to 14,000 windows
**over-trains**: the M densifies and recall collapses. The resume is not broken (preservation corr 1.000); the
"more training on the same data" premise is.

## Honest verdict + the real use-case

- The cumulative **mechanism** (preserve the learned M across save/resume) is **PROVEN** (corr 1.000).
- "A week of training on the SAME corpus compounds" is **FALSE** — it over-trains (densification).
- The **real cumulative-scaling use-case** is therefore resume + **ADD NEW concepts/data** (not re-chew the
  same corpus), training each chunk only to its window-optimum. That is **UNTESTED here** and is the next
  de-risk: does resume + new-concept growth ADD knowledge while preservation keeps the old (no catastrophic
  forgetting)? It needs the bridge to grow its target region (`auto_growth`/`TierPromoter`), not just a
  checkpoint reload.
- **Candidate better-fit substrate:** the lineage / `_longitudinal_develop_loop` machinery already does
  no-catastrophic-forgetting cumulative *development* (vocab 6→24 over simulated days, zero forgetting). For
  the "week compounds" goal that may be the right vehicle rather than raw checkpoint-resume of the stream
  curriculum.

**Verdict:** preservation **GO**; same-vocab-more-windows **NEGATIVE** (over-training); new-concept cumulative
growth = **the next de-risk**. An honest negative that sharpens the cumulative-training design — the smoke
misled by not reaching the densification regime.
