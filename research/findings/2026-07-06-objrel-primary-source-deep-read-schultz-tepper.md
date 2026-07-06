# objrel closure — primary-source DEEP-READ (Schultz-1998 dopamine + Tepper-Koos-2017 striatal interneurons): two load-bearing refinements to the running de-risks

**Date:** 2026-07-06
**Sources (deep-read, not grep-index):** the ACTUAL PDF text, extracted via PyMuPDF (`fitz`) to `.txt` (the earlier "pdftoppm missing" blocker was the wrong tool — `fitz`/`pdfplumber`/`pdfminer` are all installed). 8 basal-ganglia + Schultz PDFs now `.txt`-readable alongside their PDFs in `sim-catalog/references/textbooks/` (the durable form of the read-sources-in-depth methodology fix).
**Relevance:** directly refines the two running objrel de-risks — the dopamine-gated 3-factor plasticity CLOSURE (#1) + the frequency-vs-geometry ISOLATION control (#2) — and its fallback mechanism #5 (inhibitory-row plasticity).

## Finding 1 — Schultz 1998 (J Neurophysiol, "Predictive Reward Signal of Dopamine Neurons"): novelty-DA HABITUATES → the closure's minority up-weight must ride the PERSISTENT reward-prediction-error, NOT a decaying novelty term

The secondary sources (and the emergent-learning research gate) framed the biological minority up-weight as *"rare = novel = higher dopamine (novelty/exploration bonus)."* The PRIMARY source confirms the novelty-salience DA signal exists AND adds a load-bearing caveat the secondaries omitted:

- **Salience scales DA magnitude** (p.6, ~line 594-605): *"Their magnitude depends on the physical salience of stimuli as stronger stimuli induce higher activations that occasionally exceed those after conditioned stimuli. Particularly salient stimuli continue to activate dopamine neurons with limited magnitude even after losing their novelty."* ⇒ salience is a real, graded DA up-weight — the biological basis for boosting the rare objrel construction.
- **BUT novelty-DA HABITUATES** (p.6, ~line 596): *"Activations after novel stimuli decrease with repeated exposure over consecutive trials... Responses decay gradually with repeated exposure but may persist at reduced magnitudes with very salient stimuli."* ⇒ a PURE novelty bonus FADES as objrel is seen repeatedly during training — exactly when the read-out still needs the corrective signal.
- **DA is a scalar population signal** (p.6, ~line 660-666): *"the dopamine response constitutes a relatively homogeneous, scalar population signal. It is graded in magnitude by the responsiveness of individual neurons and by the fraction of responding neurons within the population."* ⇒ confirms the global-DA × local-eligibility THREE-FACTOR structure the closure uses (one scalar DA gain, per-synapse eligibility).

**Refinement (the load-bearing one):** the robust, non-habituating minority up-weight is the **reward-prediction-ERROR itself** — the rare objrel construction, being read WRONG for longer, keeps generating a persistent corrective RPE (an omitted-expected-reward depression → learning signal) until the signed-THEME read is learned. The transient novelty bonus is at most an early-exploration BOOTSTRAP. **Concrete trust-but-verify criterion for the closure de-risk (#1):** its reward must be the persistent correct-vs-wrong RPE (`r=+1 iff role correct`, so objrel keeps earning corrective plasticity while wrong), NOT a decaying novelty term that would habituate before objrel is learned. (The closure as dispatched uses exactly `r=+1 iff role correct` — so the deep-read CONFIRMS the design and names the failure mode to check.)

## Finding 2 — Tepper & Koos 2017 (Handbook, "Striatal GABAergic Interneurons"): feedforward inhibition is STRONG + HETEROGENEOUS → biologically confirms the Dale-legal read's inhibitory-interneuron POPULATION (each carrying a distinct negative row) and grounds fallback mechanism #5

The Dale-legal read (analytic reference + DANN) carries the objrel THEME evidence — which lives in the NEGATIVE ridge rows — through a POPULATION of inhibitory interneurons (Dale-legal: excitatory→inhibitory-interneuron→inhibitory-onto-output). The primary source confirms this is the real striatal read-out circuitry:

- **Strong feedforward inhibition is real in vivo** (~line 739): *"strong feedforward inhibition of MSNs by FSIs normally occurs in vivo as well as in vitro (Koos & Tepper 1999, 2002; Mallet 2005)."*
- **The inhibitory population is HETEROGENEOUS** (~line 767-770): *"feedforward inhibition of individual MSNs may be comprised of inputs from FSIs with very DIFFERENT firing rates and/or behavioral correlates."* ⇒ different inhibitory interneurons carry DIFFERENT signals onto the same output MSN — exactly the DANN structure (a population of inhibitory interneurons, each carrying a distinct negative ridge row). The population-of-inhibitory-interneurons read is NOT an engineering convenience; it is the striatal feedforward-inhibition architecture.
- **Multiple distinct classes** (Tepper-2018 companion): PV-FSI, NPY-LTS, CR, THIN, FAI, SABI — the striatum's inhibitory read is carried by heterogeneous, class-diverse interneurons, not a single pooled relay (which is WHY the RANK-2 single-pooled-relay see-sawed: `g(ON)-g(OFF)≠g(ON-OFF)` — a POPULATION of per-row interneurons is required, and the biology supplies exactly that).

**Refinement:** this STRENGTHENS the analytic-reference existence suggestion (the Dale-legal population read is the biologically-faithful striatal architecture, not a re-expression trick) AND grounds fallback mechanism #5 — if the closure (#2 diagnosis) shows the residual is frequency+GEOMETRY (the signed rows are the hard part), the biological fix is Vogels-Sprekeler inhibitory plasticity on the HETEROGENEOUS interneuron→output rows (letting the striatal feedforward-inhibition population LEARN to carve the minority decision region), which the heterogeneity here licenses.

## Net effect on the running de-risks
- **Closure (#1):** design CONFIRMED (persistent-RPE three-factor, scalar DA × local eligibility); named the failure mode to trust-but-verify (must not be a habituating novelty term).
- **Isolation control (#2):** unchanged; still diagnoses frequency-only vs frequency+geometry.
- **Fallback (#5):** biologically grounded and ready (heterogeneous plastic inhibitory-interneuron rows) IF #2 shows geometry is the residual.
- **Analytic reference:** its population-of-inhibitory-interneurons structure is confirmed biologically faithful (strengthens, though its adversarial-verify vs ridge-re-expression is still pending).

## Files
- `sim-catalog/references/textbooks/basal-ganglia-reviews/{Tepper-2018,Tepper-Koos-2017,TepperAbercrombieBolam-2007,Bolam-2000}*.txt` — extracted (durable deep-readable).
- `sim-catalog/references/textbooks/schultz-dopamine/{Schultz-1998,Schultz-2016-NRN,Hollerman-Schultz-1998,Schultz-2016-JNT}*.txt` — extracted.
