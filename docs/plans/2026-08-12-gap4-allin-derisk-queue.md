# gap#4 ALL-IN — the ranked de-risk queue (2026-08-12, owner: full effort / max parallel / all compute)

**The located wall:** on real few-spike DEEP (N>=3) spiking nets, even a perfect Wᵀ oracle (align 0.999) gives NO
directed credit through the finite-spike σ'(v−θ) read; deep nets don't enter the learning regime (FA/KP collapse to
majority-class). Sources: research/findings/2026-08-02-gap4-crux-wall-LOCATED-*, -depth-rescue-untestable-*,
2026-08-01-*port-to-real-spikes NEGATIVE. The assault attacks this by BYPASSING the top-down finite-spike-read credit
path (Q1 FF/TP, Q4 DECOLLE), SIDESTEPPING it via RL target-following (Q2 birdsong tutor), or fixing the Izhikevich
anti-rotation (Q3 DRTP) — with Q5 the falsifiability instrument.

**The tell that the wall is the SUBSTRATE not the rule:** if Q5's obligatory-depth-3 instrument is built AND Q1/Q2/Q4
all fail on it while BPTT (ceiling) solves it, the point-neuron spiking read carries no obligatory-depth-3 credit under
ANY local rule -> the ALIF substrate swap is the real (scope-first) surpass. Correlated tell: Q3+Q9 both failing on
Izhikevich where LIF succeeds -> the Izhikevich neuron model (quadratic + hard reset + post-reset σ' read) is the wall.

**Launched 2026-08-12 (local):** Q1 (codex/gap4-forwardforward), Q4/DECOLLE (codex/gap4-decolle), Q2
(codex/gap4-birdsong-tutor), Q5 (codex/gap4-depth3-instrument). Pending: Q3/Q9 (AWS-GPU, Izhikevich/EventProp — provision
aws_gpu.sh), Q8 (DFA head-to-head, pool), Q6/Q7 (feed Q2), Q10 (fully-spiking DFA burn-down, depends on a Q1-Q4 positive).

---

# gap#4 ALL-IN LAUNCH PLAN — ranked, ready-to-launch de-risk queue

## 1. EXCLUSION LIST — do NOT re-run (finding paths)

- **Chained multi-hop FA** (e_below=e_above@Y, hop-by-hop) — collapses to majority-class at N≥3, both tasks/configs. `2026-08-02-gap4-depth-rescue-untestable`, `...DFA-eprop-is-depth-robust`.
- **Chained Kolen-Pollack** — same depth collapse. Same findings.
- **Perfect Wᵀ oracle** (feedback-align 0.999) — zero directed credit through finite-spike σ′ on the reservoir, 6-seed. `2026-08-02-gap4-crux-wall-LOCATED` (Update baseline).
- **Lower-CV read / longer window / ensemble pool / longer eligibility** — INERT; substrate is deterministic, no shot noise. `...crux-wall-LOCATED` Update 1.
- **DECOLLE on the fixed movable-plateau RESERVOIR** — directed=0.0 both tasks. `...crux-wall-LOCATED` Update 2. *(NOTE: negative is reservoir-specific; trainable-hidden is UNTRIED — see Q4.)*
- **Relaxed plasticity (signed, drop renorm) / bottleneck architecture** — architecture-invariant, still 0. `...crux-wall-LOCATED` Updates 3-4.
- **Node perturbation on the deep read-state** — 12-seed 0/6, readout-noise variance wall. `2026-07-13-NP-vs-KP-REFUTED`. *(Confining NP to a low-dim tutor is UNTRIED — Q7.)*
- **BDSP/burstprop/BurstCCN + pool-k population read** — best arm at chance. `2026-07-12-FA-family-exhausted-BurstCCN-gate`.
- **Urbanczik-Senn 2-compartment + fixed feedback** — no hidden credit. `2026-05-17-dendritic-credit-assignment-NEGATIVE`.
- **KP/learned-feedback + settle-step temporal averaging on Izhikevich** — 0/6 and 0/12, credit is consistent-not-noisy. `2026-08-02-gap4-FA-convergence-root-cause` Updates 1-2.
- **Sacramento self-predicting microcircuit / DFC / coincidence-gated BDSP** — reduce to fixed-DFA / overfit / tie reservoir. `2026-08-01-...`, FA-convergence elim (e).
- **µPC / equilibrium propagation** — require settling phases, violate single-phase spiking. `2026-07-07 gate`.
- **Axon CaP-CaD bidirectional target** — 0.476 < microcircuit 0.942 at rate. `2026-08-07-gap4-axon-capd`.
- **"Dendrites" as a reproposal** — registered refuted; `gates/refuted_mechanism_reproposal` BLOCKS it.

*(Do not re-litigate the GOs: rate deep credit works — chained-FA+σ′ 6-seed 0.935; DFA e-prop depth-robust N2/3/4 on LIF; FA converges 6/6 on LIF.)*

## 2. RANKED DE-RISK QUEUE

**Q1 — Forward-Forward / Traces-Propagation contrastive loss on TRAINABLE spiking hidden.**
(a) Each hidden layer trains from a LOCAL per-layer contrastive "goodness" objective on its own forward spike traces + eligibility — **eliminates the top-down finite-spike read entirely**, so wall (a) has no purchase to fail through. (b) LOCAL, forward-only, brain-based (CSDP = STDP-modulated). (c) EXTEND `research/runners/_onbridge_deep_credit_decolle_derisk.py`: swap fixed reservoir → trainable spiking hidden, replace random-readout loss with per-layer contrastive-trace loss. GO-gate: `decolle_minus_permuted_L0 > margin` AND deep-layer leaves majority-class. Anti-cheats: permuted-label floor (=chance), fixed-reservoir floor (the 0.0 we already have), BPTT ceiling arm, enter-the-regime check. (d) local-CPU / mini-pool. (e) **independent-now.** *Source: TP arXiv:2509.13053; benchmark arXiv:2402.01782; NeuroTrain arXiv:2605.15058.* **⚠ READ the TP methods PDF first** to confirm the contrastive loss is spike-compatible.

**Q2 — Two-stage birdsong teaching-signal decomposition (tutor RL + local Hebbian follow).**
(a) A low-dim LMAN-analogue tutor learns a corrective teaching signal via reward-modulated node-perturbation; the deep HVC→RA motor stack trains by a reward-independent Hebbian rule following the tutor's per-neuron target (dW=η·c̃ᵢ·(gⱼ−θ)) — **no top-down error through σ′**, deep credit becomes local target-following. (b) LOCAL, three-factor, brain-based. (c) WRITE `credit_mode='tutor_teach'` in `research/runners/_snn_bptt_forward_vs_learning_isolation_derisk.py`, reusing `sim/song_hvc.py`+`song_g1_core.py` as HVC→RA; tutor-student timescale matching (τ_tutor). GO-gate: beats frozen-reservoir + permuted, deep net leaves majority-class on depth-2 LIF. Anti-cheats: shuffle-tutor-target→collapse, permuted-reward, fresh-seed pre-registration gate. (d) local-CPU. (e) **independent-now**; couples to Q6 critic. *Source: Teşileanu/Ölveczky/Balasubramanian eLife 5:e20944.*

**Q3 — DRTP (direct random TARGET projection) on the Izhikevich bridge.**
(a) Projects the fixed random *target* (not error) directly to each hidden layer, update-unlocked, **does not require W to rotate toward B** — the exact 0/6 Izhikevich FA-non-convergence root cause (W anti-rotates). (b) LOCAL, transport-free. (c) ADD `--drtp` arm to `research/runners/_onbridge_eprop_port_derisk.py`. GO-gate: FA-convergence-analogue rises + trains XOR codon where fixed-B/KP fail 0/6. Anti-cheats: permuted-target floor, `_no_weight_transport` assert, BPTT ceiling. (d) **AWS-GPU** (Izhikevich bridge). (e) independent-now. *Source: Frenkel/Lefebvre/Bol DRTP arXiv:1909.01311.*

**Q4 — DECOLLE with TRAINABLE hidden on the LIF SNN.**
(a) Per-layer fixed-random local readout + local target trains each hidden layer directly — the reservoir negative was config-specific; never run on the trainable substrate where DFA gets purchase. (b) LOCAL, transport-free. (c) ADD `credit_mode='decolle'` to `_snn_bptt_forward_vs_learning_isolation_derisk.py`. GO-gate: directed>permuted, deep net leaves majority-class. Anti-cheats: reservoir floor (0.0), permuted, DFA-eprop reference arm. (d) local-CPU. (e) independent-now. *Source: Kaiser-Mostafa-Neftci Front Neurosci 14:424; arXiv:2402.01782.*

**Q5 — Obligatory-depth-3 CREDIT instrument (shared enabler).**
(a) Build a spiking task whose depth-2 oracle fails held-out AND depth-3 generalizes, defeating the T=24 temporal-depth floor — turns "depth-robust" into provable depth-3 credit. (b) N/A (instrument). (c) EXTEND `_snn_bptt_forward_vs_learning_isolation_derisk.py` with a task builder reusing `stage0_depth_genuineness` (l2≤chance+0.06 ∧ l3≥0.80 ∧ jump≥0.15) from the crux runner. GO-gate: that predicate holds under DFA e-prop; eprop_shuffle→chance. (d) local-CPU. (e) **independent-now; UNBLOCKS falsifiability of Q1-Q4/Q7.**

**Q6 — Self-generated performance-error critic (Gadagkar).**
(a) A spiking comparator (produced-decode vs internal prediction) supplies a dense phasic-DA third factor — fixes the "reward always 0" that killed G1. (b) LOCAL three-factor. (c) Reuse N9 spiking reward loop + spiking actor-critic; feed DA as Q2's gating factor. Anti-cheat: shuffle-DA-sign→collapse. (d) local-CPU. (e) **feeds Q2.** *Source: Gadagkar Science 354:1278.*

**Q7 — Low-dim tutor node-perturbation + reward baseline.**
(a) NP failed on the high-dim deep read-state (variance ∝ noise-dim); confining perturbation to a few-unit tutor + (R−R̄) baseline is the variance-tractable regime the theory names. (b) LOCAL zeroth-order RL. (c) RETARGET `_np_feedforward_win_credit_derisk.py` NP to a small tutor pop + baseline. GO-gate: beats KP, shuffle collapses. (d) local-CPU. (e) **component of Q2**, standalone probe. *Source: Werfel-Xie-Seung Neural Comp 17:2699.*

**Q8 — DFA arm in the crux runner (head-to-head).**
(a) Pit DFA vs chained-FA vs KP vs BPTT vs reservoir under ONE 6-seed adversarial harness. (b) LOCAL. (c) ADD `credit_mode='dfa_eprop'` to `_gap4_bptt_snn_chained_fa_transport_free_derisk.py`. GO-gate: DFA>chained-FA at every width/depth. (d) local-CPU. (e) independent-now.

**Q9 — EventProp CEILING on Izhikevich.** (a) Exact adjoint gradient isolates whether the Izhikevich wall is the surrogate or the substrate. (b) **BACKPROP — CEILING ONLY, never ship.** (c) `--ceiling-eventprop` arm in `_onbridge_eprop_port_derisk.py`. (d) AWS-GPU. (e) informs Q3. *Source: Wunderlich-Pehle Sci Rep 11:12829.*

**Q10 — Fully-spiking descending DFA path (purity burn-down).** (a) Convert host-injected B_direct into a fixed-random apical `RegionPathway` → `cp_bdsp_apical_drive`. (b) LOCAL, brain-based. (c) ONE additive/default-off `sim/bridge.py` edit (receiving hooks exist); byte-identity-gated when absent. (d) AWS-GPU. (e) **depends on a Q1-Q4 positive** (converts a working rule).

## 3. START-FIRST — Q1 (FF/TP contrastive on trainable hidden)

It attacks the located wall the most directly and cheaply: the wall is *"no directed credit through the finite-spike σ′ read."* Q1 **removes that read from the loop entirely** — no top-down credit path exists to fail through — while training each deep layer from a local objective (the reservoir-free-ride failure cannot recur). It reuses a working runner and the already-decisive `decolle_minus_permuted_L0` metric, runs CPU-only 6-seed, and both external + biology digests independently name it the single genuinely-new mechanism class (everything else is exhausted FA/dendrite/BDSP variants). Launch Q5 in parallel immediately — a Q1 positive only proves depth-robustness until Q5's obligatory-depth-3 instrument exists.

## 4. HONEST RISK

**Biggest failure mode:** the "weak coupling" the locality benchmark (arXiv:2402.01782) and NeuroTrain survey both name — every fully-local rule (DECOLLE, FF/TP, DFA) trails BPTT precisely because early layers satisfy their *local* objective while remaining suboptimal downstream. Our finite-read redundancy is that same pathology in another guise, so Q1/Q2/Q4 may train the deep net yet still show the deep layer is not *obligatory*. **The tell that the wall is deeper than a mechanism:** if Q5's genuine obligatory-depth-3 instrument gets built AND Q1/Q2/Q4 all leave majority-class OR score at the permuted floor on it, while BPTT (ceiling) solves it — that isolates a substrate-level failure (the point-neuron spiking read carries no obligatory-depth-3 credit under ANY local rule), not a rule choice. The correlated tell on the deployment side: Q3+Q9 both failing on Izhikevich where LIF succeeds would confirm the production neuron model (quadratic + hard reset + post-reset σ′ read), not the credit rule, is the deeper wall — pointing to the ALIF substrate swap as the real (larger, scope-first) surpass.