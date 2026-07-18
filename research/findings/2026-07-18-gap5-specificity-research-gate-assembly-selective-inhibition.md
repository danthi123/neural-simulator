# Gap #5 specificity — research gate: the missing mechanism is ASSEMBLY-SELECTIVE inhibition (Kim-Kim 2025), not global

**2026-07-18.** The diagnostic (`2026-07-18-gap5-wang-GO-was-plasticity-noise-confound-...md`) isolated the wall: a
FROZEN learned CA3 attractor completes from ANY partial cue (permuted cue completes the held members as much as the
correct cue). Biological sparse recurrence (Guzman-Jonas ~2%) makes it PARTLY specific (cue/perm 0.94→1.37) but a
(density × assembly_frac × fb_inhib) sweep confirmed it plateaus at ~1.0-1.37× — nowhere near the 3× GO bar — and
STRONGER GLOBAL feedback inhibition (fb=35) makes it WORSE (suppresses the correct completion too). The research gate
fires (confirmed boundary, ≥2 levers failed). Read the literature IN DEPTH (per read-sources-in-depth).

## The mechanism (Kim-Kim 2025, PLOS Comput Biol / PMC12244581 — the paper the formation recipe already CITES but does NOT faithfully implement)

**"Selective inhibition in CA3: a mechanism for stable pattern completion through heterosynaptic plasticity."** The
decisive result: **global inhibition gives ~30-60% retrieval accuracy for competing/overlapping engrams; ASSEMBLY-
SELECTIVE inhibition gives ~90%.** The mechanism:

1. **Coactivation-based assembly formation.** During encoding, DG mossy fibers drive BOTH sparse excitatory pyramidal
   cells AND inhibitory interneurons. The co-active E and I cells form a UNIFIED assembly via simultaneous plasticity.
   The interneurons INHERIT stimulus selectivity from the sparse DG inputs (no explicit per-assembly wiring needed).
2. **Plastic E→I (the key rule).** The E→I synapses undergo symmetric STDP (~62.5 ms window) when the DG-driven E and I
   cells co-fire. Quantitative: initial E→I peak conductance 0.5 nS, Δ 0.2 nS/spike-pair, cap 3 nS. This makes each
   interneuron feature-tuned to the assembly that co-activated it.
3. **Heterosynaptic I→E reshaping → "spare your own engram."** The load-bearing property: *"inhibitory neurons never
   suppress the excitatory neurons within their own engram"* but DO inhibit competing assemblies' cells. Strengthened
   E→I heterosynaptically reshapes the I→E projections so an assembly's tuned interneurons suppress OTHER assemblies /
   non-members while sparing their own.
4. **The completion threshold emerges from the attractor dynamics + selective inhibition, not explicit tuning.** A
   matching cue drives the assembly's cells; recurrent collaterals amplify within the matching assembly; the assembly's
   selective interneurons suppress non-matching activity → the matching assembly wins (80-120 Hz) while competitors stay
   <10 Hz. A RANDOM cue has poor overlap → weak drive → cannot overcome the attractor-basin threshold → no completion.
   "Asymmetric inhibition amplifies small cue-strength differences, sharpening competition."

## Why MY implementation fails specificity (the precise gap)

The formation recipe's `ca3_fb_inhib` is a **single shared `ca3_pv_basket` FS pool with FIXED (`plastic=False`) E→I and
I→E weights** (`_riii_ca3_coincidence_completion_derisk.py`). That is exactly the GLOBAL inhibition the paper shows
fails: every active cell drives the shared basket equally, which inhibits every cell equally → it cannot distinguish
the matching assembly from a random cue's spurious completion. The CLAUDE.md note calling it "assembly-selective
inhibition (PMC12244581)" is inaccurate — the wiring is global, not selective.

## RANKED next mechanism (the build)

1. **Make E→I PLASTIC (assembly-selective inhibition, the paper's core).** Flip `ca3→ca3_pv_basket` to plastic so during
   encoding the assembly's cells + the co-active basket cells potentiate their E→I (symmetric co-activity), making the
   basket feature-tuned. Cheapest step; test whether cue/perm rises.
2. **Heterosynaptic "spare own engram" I→E.** Make `ca3_pv_basket→ca3` shaped so an assembly's tuned interneurons spare
   their own assembly (weak I→E onto the co-active E cells) but inhibit non-members — the load-bearing property.
3. **Sparse interneuron sub-selection (DG-inherited tuning).** The DG mossy fibers select a sparse I subset per pattern
   (not the whole basket), so different assemblies recruit different interneurons.

Test on the FROZEN + OU-off bistable gate with the mandatory no-cue + permuted anti-cheats; GO = cue ≥ 0.20, cue ≥ 3×
perm, nocue ≤ 0.10; then 6-seed. Source:
[Kim-Kim 2025, PLOS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013267)
([PMC12244581](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12244581/)).
