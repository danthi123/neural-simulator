# EMERGE-38..45 arc — adversarial audit + remediation: the science SURVIVES, the metrics/controls/framing are now HONEST. An exhaustive multi-agent adversarial audit (23 agents) found a systematic pattern of defects; all were fixed and every GO survives its corrected test. NO `sim/` edit (in the remediation).

**2026-07-02 (autonomous, ultracode).** After completing the EMERGE-38..45 competitive-pooler + discovered-multi-level-taxonomy arc, an adversarial workflow audited every de-risk (one skeptical reviewer per de-risk + adversarial verification of each flagged concern + synthesis), then a remediation workflow applied the confirmed fixes (one agent per de-risk) and re-verified each GO. All 27 arc CI tests pass after remediation.

## Why this was run
Ultracode discipline: before building EMERGE-46 on top of the arc, adversarially verify the arc so defects don't propagate. The audit was worth it — it caught real, systematic issues that the single-author cheap-first process missed.

## What the audit found (confirmed by independent adversarial verification)
The **core science is SOLID** — every confirmed issue survived a *corrected* test. But two systemic defects recurred across the arc, plus one framing overclaim:

1. **"Held-out that isn't held out" (EMERGE-42, EMERGE-43).** The class-inheritance property was taught on ALL members, then the "inheritance/generalization" metric tested a subset of those *taught* members — a direct-retrieval score wearing a generalization label. (This is the exact hole EMERGE-44 documented catching, present unfixed in two predecessors.)
2. **GO gates leaning on forbidden fixed-random-code controls (EMERGE-39, 40, 44; also EMERGE-38).** A strict GO condition (`≥ fixed-random + margin`) rested on a random-weight control that scatters by luck and near-ties on 1–2 of 6 seeds, passing only on the mean — which the project's own 2026-07-02 anti-cheat control-validity methodology explicitly forbids.
3. **Mechanism-framing overclaim (EMERGE-41).** The title/verdict credited "FS lateral-inhibition competition" for winner selection, but the FS is causally *inert* for which columns win (byte-identical winners with FS lesioned, 6/6 seeds; the pure integrator reproduces the headline exactly). Selection is by rank-order spike *timing* (Thorpe latency coding); FS only sparsifies the loser pool.
4. **EMERGE-45 control-completeness.** "Transitivity" was near-tautological with order_acc at 2 orders (permuted scored it high purely from abstentions); and no control isolated the L3 level (a genus-proximity shortcut carries most of the signal).

## The remediation (all confirmed to preserve the GO)
- **EMERGE-42 / 43 — genuine hold-out:** the tested members are now EXCLUDED from class teaching; they inherit only via the shared pooler codon. Held-out inheritance **1.00** (both), cancellation/override **1.00**, permuted collapses to **0.15**. GO on genuine generalization.
- **EMERGE-39 / 40 — drop the fixed-random gate term:** the strict gate now rests on no-selectivity (mechanism-ablation, margin +0.76 / +0.74), permuted (input-destruction), and dAP-lesion; FIXED is a reported secondary with its per-seed spread disclosed. EMERGE-40 also given a genuine pooler hold-out (held-out members excluded from the unsupervised competitive-learning order; moved the mean only 0.98→0.94). GO holds (0.96 / 0.94).
- **EMERGE-44 — demote l2lesion:** the fixed-random L1→L2-lesion control (per-seed [0.42,0.92,…], seed-43 non-collapsing) is REPORTED, not gated; GO survives on permuted (0.43) + dAP-lesion (0.00) + super-acc (0.97) + L2-grouping (+0.19).
- **EMERGE-41 — reframe:** retitled to spiking rank-order (latency) SELECTION; FS explicitly stated to provide loser-pool SPARSITY, not selection; added the FS-lesion-winner-set-identity control + a genuine FLAT-drive input-destruction control (replaces the powerless permuted-drive) + randomized tie-break. Overlap 1.00, flat-drive collapses. GO on the honest selection claim.
- **EMERGE-45 — honest metrics + L3 isolation:** "transitivity" replaced by **sibling-confusion** (fraction inferring the WRONG order, separate from abstentions) = **0.00** every seed; added L2/genus-floor (0.81), permute-L3-only (0.61), and L3-lesion (0.58) controls. Honest reframe: **the L2/genus grouping is the dominant carrier; L3 adds a smaller, seed-variable increment (+0.17 mean)**. GO on order-acc (0.97).
- **EMERGE-38 — gate + disclosure:** strict gate moved to permuted + dAP-lesion + floor; the learned-beats-fixed comparison is the reported headline with its per-seed spread [0.28,0.83,0.72,0.50,0.44,0.56] disclosed. 6-seed GO.

## Verdict
The arc's headline — *the brain discovers overlapping + hierarchical categories from experience and reasons over them (inheritance, cancellation, multi-override, discrimination) on the spiking substrate* — is **defensible and now honestly framed**. The most-softened claim is EMERGE-45's third level (L3 is a seed-variable increment above the genus floor, not the dominant carrier). All 27 CI tests pass. This is a clean foundation for EMERGE-46 (the fully-spiking stacked hierarchy).

## Process note
This is a strong instance of the adversarial-verification pattern paying off: a single-author cheap-first sweep produced fast GOs; an independent adversarial pass caught a *systematic* class of metric/control defects (not one-offs) that would have propagated. The standing lesson (already in `2026-07-02-anti-cheat-control-validity-methodology.md`): never gate strictly on a fixed-random-code control; always hold the tested set out of teaching; match the control to what the mechanism actually computes.

## Artifacts
All EMERGE-38..45 runners + tests + findings + raw jsons (corrected). Audit + remediation workflows (transcripts in the session workflow dir). Prior: the eight `2026-07-02-emerge3{8,9}/4{0,1,2,3,4,5}-*.md` findings; `2026-07-02-anti-cheat-control-validity-methodology.md`.
