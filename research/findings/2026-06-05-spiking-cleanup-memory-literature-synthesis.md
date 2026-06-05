# Spiking cleanup / associative memory — literature synthesis (the right mechanism for the composer cleanup) — 2026-06-05

Owner steer: stop parameter-guessing the cleanup; ground it in the science (the catalog + its backing papers/
textbooks). A deep-research pass (full texts of Stewart-Tang-Eliasmith 2011, Rutishauser-Douglas-Slotine 2011,
Krotov-Hopfield 2021; plus Ramsauer 2020, Betteti 2025, Treves-Rolls) produced this synthesis. It diagnoses why
all three of our cleanup attempts failed and gives the literature-grounded mechanism + exact operating conditions.

## The problem
Clean a noisy D-dim VSA estimate (cue-cos ~0.31 to the true stored code, ~0 to the other M=320 near-orthogonal
codes after ZCA) to the nearest stored concept, in SPIKING dynamics, at numpy-`argmax` parity (1.000), seed-robust
at a fixed operating point.

## Why our three attempts failed (theory-grounded)
1. **Matched filter + rate readout plateaus (~0.78-0.84).** A rectified-rate population is a LINEAR reconstructor
   `Σ_i a_i(x)·φ_i` (Stewart-Tang-Eliasmith 2011, Eqs. 4-5) — the best *linear* estimate, never a discrete max.
   Off-target codes with small positive dot-products LEAK into the sum → the cap.
2. **+ Divisive normalization lifts to 0.911 but plateaus + is fragile.** Carandini-Heeger divisive norm is
   literally Krotov-Hopfield (2021) Model C — the L2 / power-`n≈2` variant. Capacity/sharpness scales as `N_f^{n-1}`;
   `n=2` is graded, never argmax. The 0.78→0.911 climb is the `n≈2` signature; you need `n→∞` (exp/softmax).
3. **Hand-tuned hard-WTA fails (0.13-0.16).** It violated Rutishauser-Douglas-Slotine (2011) stability: a single
   global inhibitory shunt with **self-excitation gain α < 1**. Their proof: a stable single-winner needs
   **α > 1** (open-loop recurrent self-excitation gain above 1, so the winner sustains itself against the inhibition
   it drives), **asymmetric** inhibition, in the wedge **1 < α < 2√(β₁β₂)** and **¼ < β₁β₂ < 1** (Eqs. 23-25).
   "Winner drives its own inhibition and can't escape" = the α<1 soft-WTA regime (graded, no persistence);
   "everything suppressed" = outside the β₁β₂<1 box. We were never in the wedge. Working point they verify:
   α=1.3, β₁=2.8, β₂=0.25. BUT a correctly-tuned hard-WTA is also the wrong PRIMARY tool (slow to settle; Stewart
   et al. deliberately rejected recurrent-settling cleanup for a feed-forward 5-10 ms thresholded population).

## Recommended mechanism — NEF thresholded cleanup (Mechanism A, implement FIRST)
Stewart, Tang & Eliasmith (2011) "A Biologically Realistic Cleanup Memory: Autoassociation in Spiking Neurons"
(*Cog. Sys. Res.* 12:84-92) — the canonical spiking VSA cleanup, used in Spaun, shipped as Nengo SPA
`AssociativeMemory`. Feed-forward, cleans up in 5-10 ms, validated to **M=100,000, D=1000**. Construction:
1. Three feed-forward pops: input (`est`) → middle "cleanup" layer → output (cleaned vector). NO recurrence.
2. **Middle-layer encoders = the stored codes** (`φ_i = code_{w(i)}`, the matched filter — we have this).
3. **~10 neurons per concept** (M=320 → ~3,200 cleanup neurons; trivial at our scale) for redundancy/averaging.
4. **THE KEY CHANGE — threshold placement via a slightly-negative bias** so each neuron fires ONLY when
   `code_w·est > θ`, with **θ between the off-target similarity (~0) and the true cue-cos (~0.31)** — e.g. θ≈0.15-0.25.
   Then the true concept's ~3,200 neurons fire and EVERY off-target neuron stays SILENT (zero spikes). "The slight
   background inhibition (negative J_bias) allows the neurons to be insensitive to the noise." The neuron's own
   rectification nonlinearity IS the cleanup — the discretization a rate readout cannot do.
5. **Readout = project the middle layer back through the codes (W=I reconstruction)** — because only the winner is
   above threshold, the output is a near-pure copy of `code_winner`. Identity = argmax of per-concept firing (clean
   because off-target is silent).
- Scale-invariance: the threshold is on the SIMILARITY, so the cue must be ~unit-scale → keep our spiking INPUT
  normalization (it makes the matched-filter drive ≈ cosine, scale-invariant across seeds; that is why it helped
  0.844→0.911). Threshold on the cosine (0.15-0.25) is then seed-invariant.
- Spiking params (drop-in): refractory 2 ms, τ_m 20 ms, max rate 200 Hz, input noise σ 10%, NMDA PSC τ 5 ms.
- Optional hard single-winner: Nengo `WTAAssocMem` = lateral inhibition over the ALREADY-THRESHOLDED concept
  channels (a clean competition among ≤ a few above-threshold candidates), NOT on the raw D-dim estimate.

## Sharpening / alternative — iterated high-β softmax = modern Hopfield (Mechanism B)
Krotov-Hopfield (2021, ICLR) show the modern-Hopfield retrieval is a **two-layer two-body-synapse spiking circuit**:
`est_clean = CODES·softmax(β·CODESᵀ·est)`. It's OUR divnorm with the hidden-layer nonlinearity steepened from L2 to
**exp (softmax)** + a **project-back reconstruction** + **1-3 iterations**. Ramsauer (2020) one-update error bound
`‖T(x)−ξ_μ‖ ≤ 2m(M-1)·exp(−β(Δ_μ−2mR))` with separation `Δ_μ = ⟨ξ_μ,ξ_μ⟩ − max_{ν≠μ}⟨ξ_ν,ξ_μ⟩`. For near-orthogonal
codes Δ_μ is large → **β ≈ 15-30** (scaled so β·Δ_μ ≳ ln(2M) ≈ 7 for M=320) makes the error ~10⁻⁴ ≈ exact; **one
high-β pass suffices for well-separated codes**, 1-3 iterations as insurance. Capacity is exponential — 320 concepts
needs only N_h ≥ 320 (we use ~10×). softmax denominator = a divisive-normalization inhibitory pool (we have it) — the
only change is exp nonlinearity + project-back + iterate.

## Robustness at a fixed op — input-driven dynamics (Betteti-Baggio-Bullo-Zampieri 2025, Sci. Adv.)
Keep the cue DRIVING throughout (don't free-run): `ẋ = −x + W(u)·Ψ(x)`, `W(u)=(1/N)Σ α_μ ξ_μξ_μᵀ`, saliency
`α_μ = ξ_μ·u` (= our matched filter). A memory exists as an equilibrium only if **α_μ > 1**; the best-aligned code
deepens its basin, others flatten, and noise *helps* (drives toward the deepest well). This is the antidote to
"0.911 fragile": clamp the est drive through the whole readout window (we already do a stim window — keep it on).

## Divisive-norm ↔ softmax ↔ attention (what exactly to change)
Divisive normalization = Krotov-Hopfield Model C (L2, n=2) → plateaus. softmax = the exp variant on the HIDDEN
(concept) layer = Model B = attention. Actionable: (i) steepen the concept-layer nonlinearity L2→exp (softmax) OR
place a hard threshold (NEF); (ii) normalize the CONCEPT layer (M neurons), not just shunt the D-dim input;
(iii) read out by PROJECTING the concept activity back through the codes; (iv) keep the cue driving, iterate 1-3× at
high β. Reuses our FS inhibitory pool (now the softmax normalizer) + matched filter (now the per-concept saliency).

## CA3 cross-check (the recurrent-autoassociator alternative, for partial-cue completion)
Treves-Rolls / Rolls 2013: CA3 capacity `p_max ≈ k·C_RC/[a·ln(1/a)]`, k≈0.2-0.3; realistic CA3 → ~36,000 patterns;
needs SPARSE codes (a few-% active). A settling attractor (slower; same recurrent-stability concerns). Keep in
reserve for partial-cue completion; for fast per-step cleanup of a dense ZCA estimate, the feed-forward
thresholded/softmax route (A/B) is the better match.

## Build order
1. **Mechanism A (NEF thresholded cleanup) FIRST** — smallest delta from our matched-filter substrate; the essential
   change is placing the per-concept firing threshold below 0.31 and above ~0 (on the input-normalized = cosine
   similarity), readout by projection. Proven to 100K items, 5-10 ms; our M=320/cue-cos-0.31 case is deep inside the
   envelope → oracle-parity expected.
2. **Mechanism B (iterated high-β softmax)** as the sharpening stage / drop-in: exp nonlinearity + project-back +
   1-3 iterations, β≈15-30.
3. **Wrap in input-driven dynamics** (clamp the cue throughout) for fixed-op seed-robustness.
**Avoid** a global-inhibition recurrent hard-WTA on the raw D-dim estimate (the Rutishauser failure regime).

## Key citations
- Stewart, Tang & Eliasmith (2011), *Cog. Sys. Res.* 12:84-92 — NEF spiking cleanup recipe.
- Rutishauser, Douglas & Slotine (2011), *Neural Comp.* 23:735-773 (arXiv:1105.3106) — hard-WTA stability inequalities.
- Ramsauer et al. (2020), arXiv:2008.02217 — modern Hopfield retrieval + error bound.
- Krotov & Hopfield (2021), ICLR (arXiv:2008.06996) — two-layer spiking dense-AM; divnorm=Model C, softmax=Model B.
- Demircigil et al. (2017), *J. Stat. Phys.* 168:288 — exp interaction → exponential capacity.
- Betteti, Baggio, Bullo & Zampieri (2025), *Sci. Adv.* (arXiv:2411.05849) — input-driven dynamics robustness.
- Treves & Rolls (1991) / Rolls (2013), *Front. Syst. Neurosci.* 7:74 — CA3 autoassociator capacity.
- Nengo SPA `AssociativeMemory`/`WTAAssocMem` — production cleanup (threshold≈0.3, wta_output, threshold_output).
