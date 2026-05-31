# DG-boundary alternative-codes survey (check-existing-first) + the sharp cheap test. TEM/tensor-product conjunctive BINDING was already surveyed (Pick 4, 2026-05-06) and is the same family as the project's already-validated FHRR composition algebra -- binding was NEVER the blocker. The blocker is grounding a separable+stable SYMBOL from overlapping substrate activity (the DG separation-vs-reliability BOUNDARY). The sharpened grid/conjunctive idea for the SYMBOL is MODULAR REDUNDANT coding: M independent k-WTA modules + redundant/majority decoding might thread separation (from many modules) AND reliability (from redundancy) where a SINGLE DG k-WTA could not. Cheap-first numpy probe specified, on the SAME substrate activity the DG could not thread.

> ## RESULT (2026-05-31): the specified cheap probe came back CANNOT-CONCLUDE (instrument-invalid)
> The modular-redundant-coding probe ran (finding 2026-05-31-modular-coding-probe-INSTRUMENT-INVALID-...md):
> the M=1 single-DG CONTROL passes (within 0.64/between 0.45/id ~0.98) at all projection densities, so it
> does NOT reproduce the spiking DG failure -> modular escape is untestable in clean numpy. The deeper
> catch: the raw substrate activity is ALREADY 16/16 ID-separable (within 0.896 > between 0.768), so the
> id metric was saturated. Clarification (see that finding): the DG boundary is the NEAR-ORTHOGONALITY bar
> (not separability), and the spiking instability is implementation-specific. Modular/grid coding NOT
> pursued further. Converges on accepting the oracle for VSA binding + advancing P4. Read body as the
> (sound) pre-registration; the empirical answer is the instrument-invalid + clarification above.

**Date:** 2026-05-31
**Status:** Strategic survey + analysis (no build yet), honoring the check-existing-work-first discipline (the lesson from the theta-gamma redundancy earlier this session) BEFORE proposing a new biological arc for the DG boundary. Sets up the next cheap-first probe precisely.

## Why this survey (the active primary-goal question)

The owner's primary goal is biological mechanism / artificial life; the characterized blocker for biologizing the compositional symbol is the DG separation-vs-reliability BOUNDARY (finding 2026-05-31-DG-...-FUNDAMENTAL-BOUNDARY: a single competitive sparse-coding k-WTA stage cannot give between-concept separation AND within-concept reliability from overlapping inputs). Night-synthesis P3 candidate (a) was "a DIFFERENT biological code -- grid/place-cell conjunctive coding" as a possible escape. Per discipline, survey prior in-project work + reason about whether it can escape BEFORE building.

## Check-existing-first result

40 in-project files mention grid/place/conjunctive/entorhinal/TEM, but almost all are NAVIGATION context (place-cell readout in the gridworld G-arc; entorhinal EC as the trisynaptic-loop INPUT). The one representational hit: the 2026-05-06 related-projects survey "Pick 4: TEM tensor-product binding (Whittington 2020)" -- verb(x)noun outer product in a CA3-analog, proposed for Tier 2.3 phrases (~1-2 wk, never built). KEY realization: tensor-product / outer-product binding is the SAME FAMILY as the project's ALREADY-VALIDATED FHRR composition algebra (Orchard spiking-phasor + Frady-Sommer resonate-and-fire + attractor TPAM; identity-level 0.96-0.99, night-synthesis item 3). So the BINDING algebra was never the bottleneck and is already done. Grid/TEM as BINDING adds nothing new.

## The sharpened question (grid/conjunctive for the SYMBOL, not the binding)

The open question is whether grid/conjunctive coding can ground a separable+stable SYMBOL from the substrate's overlapping concept activity (cosine 0.82) -- the thing the DG could not. Reasoning about whether it can escape the boundary:
- A grid code's defining property is MULTIPLE INDEPENDENT MODULES (each a periodic/competitive code), giving combinatorial capacity AND -- the load-bearing part -- ROBUSTNESS VIA REDUNDANCY (decoding tolerates some modules being noisy, like an error-correcting code).
- A SINGLE module is just a mini-DG: same k-WTA, same separation-vs-reliability tradeoff. CONCATENATING M mini-DGs does NOT escape -- if each module's winners flip between a concept's two noisy halves, the concatenated code flips too (same per-module instability).
- BUT REDUNDANT / MAJORITY decoding across M separable-but-noisy modules COULD thread it: between-concept separation accumulates across modules (different concepts pick different per-module winners), while within-concept reliability is rescued by redundancy (a concept is identified by the MAJORITY of its module-codes matching, tolerating a minority flipping). This is exactly how grid cells achieve robust high-capacity coding. This is a genuine, non-obvious, testable hypothesis -- NOT obviously oracle-reinstantiating, because the modular code is DERIVED from substrate activity via random projections + per-module k-WTA (emergent, no external concept->code table), exactly like the DG but with M modules + redundant decoding.

## The cheap-first probe (specified; next concrete action)

CPU/numpy, stdlib+numpy only, no protected import. On the SAME cached substrate concept activity the DG probes used (between-concept cosine ~0.82, two storage/query halves per concept):
- Project the activity through M INDEPENDENT random k-WTA modules (each module: its own random projection to N/M dims + top-k winners). A concept's code = the tuple of M per-module winner-sets, computed separately for its storage half and its query half.
- SEPARATION metric: between-concept agreement under modular decoding (fraction of modules where two DIFFERENT concepts share the winner-set) -- want LOW.
- RELIABILITY metric: within-concept agreement (fraction of modules where a concept's storage-half and query-half winner-sets match) -- want HIGH.
- REDUNDANT-DECODE identity test: is a concept's query-half correctly matched to its OWN storage-half (vs all other concepts) by MAJORITY module agreement? -- the capability metric.
- Sweep M (1, 2, 4, 8, 16, 32) at matched total active count. M=1 reproduces the single-DG boundary (the reproduce-the-failure control). 
- FROZEN three-state: RESOLVES if some M gives between-separation good (majority-mismatch across concepts) AND within-reliability good (majority-match within concept) AND identity test >= 0.80 multi-seed where M=1 fails; BOUNDARY if no M threads both (modular redundancy does not escape -- the boundary is deeper); DOES-NOT-RESOLVE/CANNOT-CONCLUDE on instrument-invalid.

If RESOLVES -> grid-like modular redundant coding is the biological escape from the DG boundary -> justifies a spiking grid-module build (adversarial review + frozen verdict + controller decisive run). If BOUNDARY -> redundancy does not escape; bank it (the boundary is deeper than module count), and the honest conclusion strengthens: the substrate's overlapping activity cannot be grounded into a separable+stable symbol by ANY single-stage competitive sparse code, modular or not -> accept the oracle/external code as an irreducible engineering component + advance the validated P4 stack (already done: directional multi-hop shipped).

## Discipline

Survey + reasoning only; no build yet; no protected/frozen/moat/sim/runner module touched. The grid/conjunctive direction was NOT proposed naively -- it was check-existing-first'd (TEM binding already covered + the binding algebra already validated) and reasoned to its sharp form (modular REDUNDANT coding, the only version that could escape the per-module tradeoff). The cheap-first probe is fully specified with a frozen three-state bar + the M=1 reproduce-the-failure control, on the SAME substrate activity, BEFORE any spiking build -- the same de-risking discipline that correctly killed the integrated-loop, denoiser, and (redundant) theta-multiplexing builds.
