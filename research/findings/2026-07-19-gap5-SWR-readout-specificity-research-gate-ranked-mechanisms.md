# gap#5 SWR readout specificity — research gate: the near-tie is a dense-random-Schaffer READOUT artifact (Valero 2017), and 6-seed-robust specificity needs the pattern-separated completion (the emergent-DG) UPSTREAM. Ranked mechanisms + cheap-first de-risk.

**2026-07-19.** Deep-research gate on WHERE/HOW the SWR replay readout near-tie (distinct CA3 assemblies → near-identical
CA1 output, cross 0.72-0.86) is fixed biologically. Read-only scout + controller-verified the decisive primary source.

## The decisive biology (VERIFIED by the controller, not just cited)
**Valero et al. 2017 (Neuron 94(6):1234-1247)** — verified via the abstract/paper: CA1 pyramidal firing selectivity during
PHYSIOLOGICAL ripples is "dominated by an event- and cell-SPECIFIC synaptic drive, modulated in single cells by changes
in the **excitatory/inhibitory ratio**." During pathological FAST ripples that cell-specific drive is lost → cells "fire
together" → selectivity collapses/randomizes. **This is a near-exact description of the model's failure mode:** a
fixed-random DENSE Schaffer delivers ~76 near-identical inputs to every CA1 cell (no cell-specific drive) → every CA1
fires → cross ≈ 1. ⇒ the specificity lives in the CELL-SPECIFIC (structured + potentiated) synaptic drive + E/I balance,
which the model's dense-random excitatory Schaffer explicitly lacks.
- **Kwon et al. 2018 (J Neurosci 38(22):5140)** — biology's CA3→CA1 EXCITATORY projection is STRUCTURED (violates Peters'
  rule); CA3→PV-interneuron is random. The model INVERTED this (dense-random excitatory Schaffer, structured-nothing).
- **Wilson-McNaughton 1994 / Pfeiffer-Foster 2013** — CA1 replay output IS specific in vivo → the near-tie is a MODEL
  ARTIFACT, not a biological ceiling. CA1 relays + sparsifies the CA3 sequence; **no CA1 readout can manufacture a
  distinction absent from the CA3 pattern** ⇒ 6-seed-robust specificity REQUIRES a pattern-separated CA3 completion upstream.

## Ranked mechanisms (cheapest-first on the point-neuron + region-framework + STDP/BTSP/coincidence substrate)
1. **CA1 E%-max / feedforward-inhibition winner-set sparsification** (de Almeida-Idiart-Lisman 2009 E%≈5-10%; Pouille-
   Scanziani 2001 ~2ms window). Cheapest, read-side. Substrate: `swr_ca1_ff_inhib` + CA1 FS pool as a top-k/divisive-norm
   WTA over CA1 g_e. **NECESSARY-not-SUFFICIENT** — only discriminates if the CA1 g_e is non-degenerate (on dense-random
   Schaffer it slides between all-fire/all-silent — the project already observed this). Must ride on #2.
2. **Structured + encoding-POTENTIATED (learned) SPARSE Schaffer** (Valero 2017 cell-specific drive; Kwon 2018 structure;
   Schaffer-LTP, Bliss-Collingridge 1993). The actual specificity source — DROP the dense-random projection; potentiate a
   SPARSE ca3(assembly)→ca1(distinct target) at encoding. The project's `swr_learn_schaffer` is exactly this (no-learn
   anti-cheat cross 0.999→0.27 proves it load-bearing); the missing piece is the SPARSE structured init (not dense).
3. **Pattern-separated CA3 completion feeding the readout** (Marr 1971; the project's Kopsick completion 12.6× within/cross
   GO + the mossy-detonator DG selection 6-seed GO). Feed self-organized/mossy-selected assemblies via the existing
   `assemblies_ext` hook — removes the completion near-tie that caps seed-robustness. **The shared gap#5 unlock.**
4. **Assembly-selective inhibition** (Kim-Kim 2025, PMC12244581 — global inhib ~30-60% vs assembly-selective ~90%): plastic
   E→I + heterosynaptic "spare-your-own-engram" I→E for the asymmetric dominant-attractor residual. More expensive.
5. **Brief single-volley sharp-wave read** (Buzsaki 2015; Stark 2014 PV pacing) instead of the sustained 60-step ripple —
   a sparse phase-locked read avoids the Izhikevich u-accumulation/saturation the project saw under sustained drive. Cheap.

## Recommended cheap-first de-risk (the specificity STACK, one experiment)
On the existing SWR runner: (1) source assemblies via `assemblies_ext` from the self-organized completion / mossy-selection
(within/cross 12.6× / sep_cos 0.07), NOT random-disjoint; (2) `swr_learn_schaffer` ON with a SPARSE init (drop dense-random);
(3) phase-2 STP-off on Schaffer (already root-caused); (4) brief single-volley read + fire only top-~10% CA1 by g_e (E%-max).
GO bar: ca1_match ≥ 0.6, cross ≤ 0.3, ratio ≥ 3×, 6-seed; anti-cheats: no-learn → cross ≈ 1 (Schaffer load-bearing),
permuted-cue → no match (specific assembly). If specificity appears → the near-tie was the random-assemblies/dense-readout;
if it survives distinct assemblies → the readout is still degenerate and #2's sparse structured Schaffer is the gate.

## Honest residual (do not oversell)
Biology does NOT hand a single "make-the-readout-specific" knob — specificity is a STACK (pattern-separated CA3 → structured
+potentiated sparse Schaffer → E%-max/FFI → brief read); remove any layer and it degrades. Since no CA1 readout can create a
distinction absent from CA3, **6-seed-robust specificity requires the pattern-separated completion (#3 = the emergent-DG
selection, already 6-seed GO) upstream** — confirming the standing conclusion that the two gap#5 extensions UNIFY and the
emergent-DG pattern separation is the shared unlock. #1+#2+#5 lift per-seed specificity WHEN the completion is clean.

## To verify further (controller trust-but-verify queue)
- **Valero 2017 = VERIFIED** (abstract: CA1 firing selectivity dominated by event/cell-SPECIFIC synaptic drive modulated
  by the intracellular E/I ratio; collapses to co-firing in fast ripples).
- **Kwon 2018 = VERIFIED** (J Neurosci, mGRASP: CA3→CA1 PYRAMIDAL connectivity spatially STRUCTURED; CA3→PV-interneuron
  "significantly more random"; Peters' rule enhances PC structure but randomizes PV — i.e. excitatory Schaffer is
  structured, inhibitory random; the model's dense-random EXCITATORY Schaffer is the inverted prior).
- Still queued: Kim-Kim 2025 90% vs 30-60% numbers; de Almeida-Idiart-Lisman E%-max exact form (verify when building #1/#4).
