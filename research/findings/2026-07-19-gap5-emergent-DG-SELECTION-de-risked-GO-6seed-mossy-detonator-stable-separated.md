# gap#5 (ii) emergent-DG — the DG-SELECTED assembly is DE-RISKED, 6-seed GO: strong mossy detonation selects a STABLE, sparse, separated, input-specific CA3 code from the DG volley (the pattern separation the SWR needs)

**2026-07-19.** Following the read-your-substrate root-cause (`2026-07-19-gap5-emergent-DG-ROOT-CAUSE-...` — the
trisynaptic feedforward does not conduct a volley), de-risked the core emergent-DG question: **can a stable,
pattern-separated CA3 assembly be SELECTED from a DG volley?** Answer: **YES, 6-seed GO.** Two apparent walls were both
root-caused as METHOD/instrument artifacts (per THE LAW), not capability limits.

## The arc (two walls → both artifacts → GO)
1. **Wall 1 — "layer-2 amplification gives 0 firing."** Root cause: the runner's DEFAULT mossy (w=8) is ~10-30× too
   weak (CA3 g_e 0.17, needs ~5+). The amplification framing was chasing the wrong variable; the input never reached CA3.
2. **Wall 2 — "NOT-GO, stability 0.00" (same input → disjoint CA3 sets).** This looked like fatal non-determinism. The
   silent-failure discipline resolved it: (a) `deterministic_transpose_matvec` ON made NO difference (identical sizes) →
   NOT GPU non-determinism; (b) two FRESH bridges (same seed, same pattern) gave **Jaccard 1.00** → the response IS
   input-deterministic. ⇒ the "instability" was a RESET ARTIFACT: the intrinsic bistability plateau (`plateau_self_regen`)
   LATCHES by design, and a partial reset (v/u/conductances, even +STP) did not clear the latch, so presentation 2 was
   contaminated by presentation 1. A COMPLETE post-build state snapshot/restore (== fresh bridge) removes the confound.

## Result — 6-seed GO (drive DG directly to isolate the dg→ca3 selection; mossy w=200, amplify config, snapshot-restore)
| gate | criterion | result (seeds 42/43/44/100/101/102) |
|---|---|---|
| SPARSE | assembly 3-40 cells | sizes 10-37 (~2.5-9% of 400) ✓ |
| SEPARATED | distinct DG → sep_cos < 0.4 | **0.041 / 0.093 / 0.100 / 0.099 / 0.097 / 0.155** ✓ |
| STABLE | same DG → Jaccard > 0.6 | **0.94 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00** ✓ |
| INPUT-SPECIFIC | permuted DG → overlap < 0.3 | 0.03 / 0.08 / 0.13 / 0.03 / 0.00 / 0.08 ✓ |
| MOAT | no-input → ~empty | 0 all seeds ✓ |
| MOSSY-LESION | input + mossy w=0 → collapse | intact 17 → lesion **0** (load-bearing) ✓ |

The separation is a REAL low-overlap code (sep_cos ~0.07, not the earlier all-disjoint 0.00 noise artifact) — distinct
DG volleys expansion-recode into distinct sparse CA3 sets via the fixed random mossy projection (the DG/CA3
pattern-separation mechanism, Marr / Kandel Ch 54). The mossy-lesion collapse confirms the mossy pathway is the
load-bearing selector.

## What this DE-RISKS, and what remains (honest scope)
- **DE-RISKED (6-seed GO):** the emergent-DG SELECTION — a DG volley SELECTS a stable, sparse, separated,
  input-specific, moat-safe CA3 assembly. The assembly is DG-SELECTED (from the random mossy projection), NOT
  hand-assigned. This is the CORE emergent-DG piece and the pattern separation the SWR specificity needs.
- **REMAINING for the full emergent-DG (named, per THE LAW):**
  1. **Upstream conduction (lang→ec→dg):** still too weak (every hop sub-threshold — same conduction issue); I drove DG
     directly to ISOLATE dg→ca3. The full "input → assembly" loop needs lang→ec and ec→dg strengthened (a bounded
     multi-stage feedforward-weight tune; biologically the perforant path + mossy ARE strong).
  2. **Self-sustaining ATTRACTOR:** the selection is a TRANSIENT (drive-present) response. To STORE it as a completable
     memory (persist after the drive + complete from a partial cue), it needs **one-shot BTSP** to lock the co-active set
     into a recurrent attractor = the **gap#4 keystone**. This is the gap#4↔#5 UNIFICATION: the emergent assembly, once
     SELECTED (here), is STORED by the same plateau-gated one-shot rule the gap#4 keystone provides.
- **HYPOTHESIS (next test):** feed these mossy-SELECTED separated assemblies to the SWR generative-replay readout (in
  place of the hand-assigned near-tie assemblies) → does the completion become distinct → SWR specificity closed? The
  SWR (i) strength sweep showed robust specificity needs pattern-separated codes; this selection PROVIDES them.

## Status
- **6-seed GO with a full anti-cheat suite** (sparse/separated/stable/input-specific/moat/mossy-lesion). The emergent-DG
  selection is real, not an amplification tune. NO sim/ edit (diagnostics + `_build_bridge` params only).
- Diagnostics: `scratchpad/emergdg_{diag,chain,ec,stagedrive,mossy,detonator,determinism,select,select_clean}.py`.
- NEXT (GPU): (a) one-shot BTSP-store the selected assembly → self-sustain + complete (the gap#4×#5 unification build);
  (b) feed selected assemblies to the SWR → test specificity; (c) strengthen lang→ec/ec→dg for the full input→assembly loop.
