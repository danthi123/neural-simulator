# Off-diagonal dendritic de-risk — NEGATIVE (3 seeds): the dendritic rewrite is NOT required for the conversational cortex; ship the flat cortex

**Date:** 2026-06-17
**Status:** **NEGATIVE, 3 seeds — the decisive gate of the dendritic-substrate frontier (CYCLE 119).** An
online-local circuit does **not** reach a genuine off-diagonal (cross-neuron) decorrelation beyond the input —
because **local PPMI-centering already reaches the whitening ceiling**. ⇒ the months-scale dendritic rewrite is
**not required for the conversational cortex**; ship the flat 2,048-concept cortex as the product; reserve the
dendritic build for the artificial-life goal only.

## The question (CYCLE 119)

The dendritic-substrate scoping (`2026-06-17-dendritic-substrate-frontier-scoping.md`) honestly reversed its own
premise: point-neuron progress (the PPMI local-normalization cortex) had overtaken most of the dendritic
rationale. The **one residual** a dendrite might still uniquely buy was the **off-diagonal (cross-neuron,
low-rank) decorrelation** that PPMI's diagonal normalization leaves on the table (diagonal ~+0.31 → host +0.44 →
offline-ZCA +0.49). A point neuron provably cannot do cross-neuron whitening; a dendrite might. This de-risk
(`_phaseB_offdiagonal_dendritic_pc_derisk.py`, CPU, 3 seeds) was the decisive cheap-first gate before any
months-scale build: can an **online local** circuit (Duong fixed-frame + plastic gains [Mechanism A]; Mikulasch
error-gated dendritic balance [Mechanism B]) reach that off-diagonal residual with effective rank ~8, beating
the must-fall-short controls (Oja somatic, per-hub diagonal), anti-cheats clean?

## Result (mean of 3 seeds; numbers exact from the run)

| quantity | value | role |
|---|---|---|
| HOST (PPMI-SVD) | +0.442 | the reference target |
| ZCA_r8 (offline whitening) | +0.524 | the ceiling a dendrite would chase |
| SM_somatic (Oja) | +0.264 (rank 3.3) | must-fall-short control — falls short ✓ |
| DIAG (per-hub gain) | +0.216 | must-fall-short control — falls short ✓ |
| **GAINS_whiten (Mech A)** | **+0.519** (rank **53.2**) | the mechanism |
| DEND_balance (Mech B) | +0.482 (rank 37.2) | the mechanism |
| **lesion (g=0 → centered-PPMI Pc)** | **+0.519** | the no-off-diagonal control |
| permuted-similarity | −0.006 | clean (≈0) ✓ |
| raw-input (non-PPMI) | +0.402 | < PPMI ✓ |
| held-out generalization | 0.86 | ✓ |

**Verdict: NEGATIVE.** The mechanism reaches +0.519 — but the **decisive tell** is that its **lesion (g=0, i.e.
the centered-PPMI input `Pc` with no learned off-diagonal gains) gives the *same* +0.519**, and the output's
effective rank is **53, not ~8**. So the learned off-diagonal gains are **inert** — they add nothing over the
input. The +0.519 is **the centered-PPMI codes themselves already reaching ≈ ZCA (+0.524)**. Two gate failures:
`rank_in[6,16]=False` (rank 53), `beats_collapse=False` (mech == lesion).

## What it means — the dendritic question is CLOSED for the conversational cortex

The off-diagonal residual a dendrite was hypothesized to buy **does not exist as a reachable gap on this corpus**,
because **local, feedforward PPMI-centering (log → row-normalize → mean-center — all point-neuron operations)
already reaches the whitening ceiling** (+0.519 ≈ ZCA +0.524). No online-local circuit improves on it, and there
is nothing left for a cross-neuron dendritic decorrelator to add.

This is the **pre-registered NEGATIVE branch** of CYCLE 119, with a clarifying (and reassuring) twist:

> **The months-scale dendritic off-diagonal rewrite is NOT required for the generalizing conversational cortex.**
> The point-neuron substrate, with local PPMI normalization, already reaches the whitening ceiling. **Ship the
> flat 2,048-concept curated cortex as the conversational product.** Reserve the dendritic build for the
> artificial-life goal only — and even there, eyes open: it may also plateau on real experience.

An honest negative that closes a months-scale fork on a measured signal **is** the deliverable — it redirects
effort away from a build that the evidence says is unnecessary for the conversational goal.

## Provenance + a fixed bug

The computation completed for all 3 seeds (~3.7 h CPU; slowed by concurrent de-risks sharing cores). The process
then **crashed on a Windows cp1252 `UnicodeEncodeError`** in the verdict `print` (the `≈`/`⇒`/`→` characters)
**before** `json.dump`, so the artifact was never written. Fixed: `sys.stdout.reconfigure(encoding="utf-8")` at
the top of `main()` (a future run reproduces the result + writes the JSON). The JSON
(`research/findings/raw/_phaseB_offdiagonal_dendritic_pc.json`) is reconstructed exactly from the captured stdout;
no re-run of the 3.7 h job was needed.

## Where this leaves the project

- **Conversational product:** the **flat 2,048-concept curated cortex** (already delivered) + local PPMI
  normalization — generalizing, biology-faithful, point-neuron, no dendrites. The full conversational stack
  (parse · store · recall · abstain · negate · generate · dialogue-plan · learn-from-conversation · multi-hop ·
  multi-turn) runs on it.
- **Dendritic build:** reserved for the artificial-life goal only, NOT the conversational cortex. The
  decisive evidence says point-neurons + local normalization suffice for the conversational generalization.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._phaseB_offdiagonal_dendritic_pc_derisk
```

No `sim/` edit. (The runner edit is the encoding fix only.)
