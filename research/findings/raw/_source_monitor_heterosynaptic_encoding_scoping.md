# Source-monitoring encoding wall — heterosynaptic-competition-at-encoding scoping (2026-08-07)

Research round (deep-read of the 4 NO-GO findings + the CA3 competitive-Hebbian GO + the committed kernels).
No code yet; this is the pre-build spec + the pre-scoped fallback. Pivot target after the Gate B Stage-2j GO.

## The wall + the quantity that must move
Source margin `M_s = rate[s] − max(rate[rivals])` (`_source_margin`, `_laneC_source_monitor_coresidency_gate.py:508`);
gate certifies the WEAKEST source: `min_s M_s` must beat the competition-lesion arm AND clear frozen floor `F=0.15`.
The margin is set at **ENCODING**: during `experience()` symmetric Hebbian (η=0.2) potentiates every co-active
`episode→source` pair, so a SHARED-CORE cell (fires in seen/heard/self encoding) potentiates EQUALLY to all three →
a rival "pedestal" under the correct-source "peak" at recall. NO-GO now: overlap sweep strict **1/5**, best min M
**+0.005** (~30× below 0.15). The quantity to move = the weakest source's between-source fan-out CONTRAST (lower the
shared cell's drive to RIVALS without lowering its drive to the CORRECT source).

## Why each of the 4 prior levers failed (common root: all act at RECALL or on ACTIVITY, never the encoding fan-out)
1. fair/blanket divisive inhibition (recall): spiking GABA-A is anti-divisive — rebound-fires a silent rival above a
   near-threshold target (hard WTA, not soft division).
2. own-gain/BCM up-scaling (recall): saturates at the Izhikevich adaptation/refractory ceiling; never suppresses the
   rival co-firing that is the actual deficit.
3. multiplicative synaptic scaling: defends firing RATE → EQUALIZES per-source rates → compresses the very contrast
   the margin measures (orthogonal-quantity failure).
4. symmetric lateral GABA-A competition (recall): rich-get-richer (inhibition scales with the winner's drive) → the
   weak source is buried; the binding constraint is DIRECTION, not magnitude.

## Proposed mechanism (distinct from all 4): thresholded heterosynaptic competition at ENCODING
"Protect the peak, depress the pedestal," keyed to each presynaptic episode cell's CUMULATIVE per-source
co-activation eligibility. Homosynaptic LTP unchanged; heterosynaptically DEPRESS only that cell's synapses whose
cumulative source-eligibility is BELOW θ, PROTECTING its single max-eligibility ("peak") source. A shared cell thus
COMMITS its output to one source; its drive to the rival pedestal is cratered; different shared cells commit to
different sources. Biology: Chistiakova–Volgushev heterosynaptic plasticity / Miller–MacKay 1994 subtractive
competitive normalization (outgoing-weight conservation) + the THRESHOLDED gate (Milstein–Magee 2021, Cone–Shouval
2021). REUSE committed GO kernels by reference (NO `sim/` edit, as the CA3 GO did): `fused_btsp_hetero_update`
(`sim/kernels.py:647`), `fused_htm_winner_inactive_depression` (`:497`). The same kernel-family surpassed the CA3
saturation boundary 6-seed GO (`2026-07-14-ca3-competitive-hebbian-formation`, 5.2–8.9× vs pure-LTP 1.01×).

Two load-bearing subtleties the codebase already proved:
- **CUMULATIVE keying, NOT per-event** — per-event/afferent-label keying → recency collapse (each shared cell fires
  in all 3 encodings; last-encoded source never subsequently depressed → all commit to `self_generated`). The CA3 GO
  found the same: per-event winner masks crater the signal; cumulative-ensemble mask gave the 5.2–8.9×.
- **THRESHOLDED subtractive, NOT linear/multiplicative** — `sim/kernels.py:676`: the linear `(1−Ẽ)` gate gives
  `W_i=0.5` for every input (provably uniform, Cone–Shouval 2021); Oja is multiplicative → preserves ratios, not
  sharpening (exactly why lever #3 compressed the margin). Thresholded subtractive "lowers the PEDESTAL without
  lowering the PEAK" = verbatim the source-monitor need.

Threads the 2026-05-31 separation-vs-reliability BOUNDARY (which was a single input-sparsity k-WTA knob): here
separation is per-synapse in the FAN-OUT (doesn't touch input sparsity); reliability is preserved by never lowering
the peak + distributing commitment across the population. The DG knob that couldn't thread both is simply not turned.

## Cheapest single-variable de-risk
Instrument: reuse `_laneC_source_monitor_overlap_sweep.py` verbatim (honest v6 recall + `_source_margin` +
zero-weight `control_strict=False` guard; already reproduces the 1/5 NO-GO). One knob `lam_hetero` (heterosynaptic
depression coeff): accumulate per-episode-cell × per-source eligibility during `experience`; after encoding apply
thresholded heterosynaptic depression to `episode→source` CSR weights (protect max-eligibility source above θ,
depress below-θ by `lam_hetero`). **`lam_hetero=0` ≡ current symmetric-Hebbian overlap NO-GO byte-identically** (the
null control). Anti-cheats (ALL must hold): (a) `lam_hetero=0` reproduces strict 1/5; (b) zero-learned-weight control
stays `strict=False`; (c) **commitment-distribution entropy spans all THREE sources** (guards recency collapse); (d)
reliability guard — `all_dominant_correct` stays True AND no source's own recall rate drops (peak protected);
(e) encoding-only lever, recall competition + v6 thresholds frozen. GO (frozen v6): `min M > min L` AND `min M ≥ 0.15`
on calib 650/651 → dev 652/653/654 → held-out 655/656/657. numpy, deterministic, ~minutes/seed.

## Honest closability + fallback
Likely closable (first lever to attack the root cause; GO precedent). ONE real risk: symmetry-breaking on
IDENTICALLY-driven zero-init shared cells (harder than CA3's pre-differentiated sparse code) — if per-cell source
leads (stochastic timing + encoding order) are too weak, commitment collapses to recency or stays uniform (the
commitment-distribution anti-cheat catches it as NO-GO). Pre-scoped FALLBACK: a **conjunctive source-tag** — let the
physical source afferent WEAKLY modulate the overlap layer during ENCODING (biologically legitimate: sensory/motor
context reaches association cortex; recall stays episode-only + honest), so different shared-cell subsets fire
preferentially per source (Komorowski–Manns–Eichenbaum 2009 item-in-context conjunctive cells). Separates only the
OVERLAP, not the whole code → the DG tradeoff knob still doesn't bind. Build `lam_hetero` first; conjunctive tag is
the next rung if symmetry-breaking is the wall.
