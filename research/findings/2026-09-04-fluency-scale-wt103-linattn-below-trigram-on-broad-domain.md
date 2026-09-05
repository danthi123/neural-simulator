---
type: finding
status: partial
claim_check: measured
date: 2026-09-04
mechanism: linattn own-voice mouth — fluency at wikitext103 (broad-domain) scale (the owner-prioritized fluency-scale sweep)
lane: E·Language / brain-native open-ended generation (the owner-sanctioned DEFERRED wall #3)
seeds: [43]
artifacts:
  - research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json
verdict: >
  At wikitext103 scale (1.78M training sentences, d_model=192, 3.6 GPU-h, seed 43) the linattn own-voice mouth
  GENUINELY USES CONTEXT (permutation NLL collapses 4.7-5.5 -> 8.7-9.0; memoryless is strictly worse) and BEATS
  the bigram at every depth >= 2 (+0.32 to +0.61), LOSING to the bigram only at depth 1 (margin_vs_bigram -0.187,
  where the bigram itself is strongest and the trigram is weakest) — corrected 2026-09-05, adversarial-verify
  w3qhweujd, from an earlier "beats the bigram at every context depth" overstatement. It LOSES to the TRIGRAM at
  every depth >= 2
  (margin_vs_trigram -0.29 to -0.57), beating trigram only at depth 1 where the trigram is weakest. So MORE +
  HARDER data did NOT bring the current mouth to fluency on a broad domain — a direct contrast with the simplewiki
  6/6 GO that cleared trigram. This is a verdict on THIS mechanism+capacity at broad scale, not the capability:
  it re-aims the fluency arc toward either more capacity or a new mechanism, and it is a single-seed, single-
  capacity directional result (not a 6-seed generalization).
---

# Fluency at wikitext103 scale: the linattn mouth falls below trigram on a broad domain

> **⚠️ CORRECTION (2026-09-05, adversarial-verify `w3qhweujd`):** the frontmatter verdict and the "Reading it
> honestly" section below originally said the mouth "BEATS the bigram at every context depth" / "beats the bigram
> everywhere." False at depth 1: the table shows `margin_vs_bigram = -0.187` there (a LOSS), matching the
> depth-1 exception already carved out for the trigram clause in the same sentence. Corrected to "every depth >=
> 2," losing to bigram only at depth 1. The core result (loses to trigram at every depth >= 2) is unaffected.

## What ran
The owner-prioritized fluency-scale sweep: `_emerge_wkv_lm_derisk --recurrence linattn` on `data/corpus/wikitext103.txt`,
1,781,187 training sentences, vocab 8001, d_model=192, seed 43, ~3.6 GPU-h (cupy). Result:
`research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json`. This was a single decisive run (the wt103
queue is now empty).

## The depth-bucketed result (NLL; lower is better; margin_vs_X = X_nll - wkv_nll, positive = wkv wins)

| depth | wkv | bigram | trigram | wkv_perm | margin_vs_trigram | margin_vs_bigram |
|---|---|---|---|---|---|---|
| 1 | 5.489 | 5.302 | 6.478 | 8.984 | **+0.989** | -0.187 |
| 2 | 4.961 | 5.278 | 4.392 | 8.945 | **-0.57** | +0.317 |
| 3 | 4.809 | 5.285 | 4.342 | 8.999 | -0.467 | +0.477 |
| 4-5 | 4.906 | 5.374 | 4.504 | 8.766 | -0.402 | +0.468 |
| 6-9 | 4.818 | 5.35 | 4.462 | 8.759 | -0.356 | +0.532 |
| 10-99 | 4.749 | 5.361 | 4.463 | 8.684 | -0.286 | +0.612 |

## Reading it honestly
- **The mouth is genuinely context-using, not a lookup:** permutation collapses the NLL from ~4.7-5.5 to ~8.7-9.0
  at every depth, and the memoryless variant is strictly worse. Both anti-cheats pass. It also beats the bigram at
  every depth except depth 1 (a real loss there, margin_vs_bigram -0.187 — the bigram is not beaten everywhere).
  So it learned real sequential structure.
- **But on the broad wt103 domain it does not clear the trigram** at any depth >= 2 — the exact bar the simplewiki
  run cleared 6/6. The depth-1 win is not fluency (the trigram is degenerate at depth 1).
- **So scale alone did not deliver fluency here.** The Vikunja-#193 framing ("the ceiling was token-starvation,
  not capacity — fluency keeps improving with more text") holds on simplewiki but does NOT extrapolate to wt103 at
  this capacity: 1.78M broad-domain sentences still leaves the mouth below trigram.

## Next (no-defer: two named levers, this is the DEFERRED wall so it stays build-ahead)
1. **Capacity sweep** — this ran at d_model=192, a small model. Before concluding the linattn architecture is
   trigram-bound on broad data, sweep d_model (e.g. 384/512) at wt103 scale: is it capacity-bound or
   architecture-bound? Cheap to queue on the GPU.
2. **The new mechanism** — if capacity does not close it, the memory `project_own_voice_fluency_pursue_fully`
   names the path: a content-addressable learned-key associative read (structured HiPPO SSM -> learned-key
   attention), NOT the linattn/SSM family. That mechanism is ALREADY prepared as the `--recurrence learnkey`
   build-ahead (research/coordination/build_ahead_ready.md #5) — this wt103 result is exactly the trigger it was
   staged for. Both remain build-ahead: brain-native open-ended generation is the owner-sanctioned deferred wall,
   with Qwen the interim scaffold.

## Caveats
Single seed (43), single capacity (d_model=192). The margins are large and consistent across depth, so the
directional verdict (below trigram on broad data at this capacity) is clear; a 6-seed or multi-capacity claim
would need the capacity sweep above.
