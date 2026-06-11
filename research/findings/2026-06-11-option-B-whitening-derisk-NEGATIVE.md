# Option B whitening de-risk — NEGATIVE: even the IDEAL whitening cannot co-satisfy decorrelation + reproducibility≥0.9 on denoise64 (2026-06-11)

**VERDICT: NEGATIVE, unanimous 3/3 seeds (42/43/44).** The load-bearing §4 falsification that gates the entire
Option-B (structured-cortex) arc has run, and it is a **clean, decisive NEGATIVE forced by the REPRODUCIBILITY gate
(b)**. The §6.4 three-operating-points tension is **real, not a tuning problem**: on the brain's REAL correlated
`denoise64` codes, the three operating-point regions (decorrelated ∩ reproducible-at-σ=0.1 ∩ composing) **do not
overlap** — and they fail to overlap even for the IDEAL god's-eye ZCA whitening (the B3 ceiling). The decisive next
step is the **DUAL architecture** (similar cortical codes + a linked decorrelated hippocampal/cerebellar expansion),
NOT whitening the similar codes directly.

Probe: `research/runners/option_B_whitening_derisk_probe.py`. Raw:
`research/findings/raw/_option_B_whitening_derisk.json` (multi-seed 42/43/44) +
`_option_B_whitening_derisk_seed42_smoke.json` (single-seed). CPU/numpy, no `sim/` edits, no GPU.

---

## The decisive experiment (the never-yet-run reconciliation, on ONE operating point)

All four gates tested on the SAME operating point — the reconciliation the 2026-06-06 (composition-validated) arc and
the 2026-06-11 (reproducibility-failing) arc had never been run together:

- **Codes**: the brain's REAL `denoise64` codes, read in NATIVE binary form (the `load_real_codes` convention: mean
  over obs samples, random-Gaussian project to D=512, mean-center + unit-normalize). **NEVER median-bipolarized.**
- **Unit check (anti-cheat #1)**: PASS on all seeds — between-cos as-read = **0.814 / 0.798 / 0.787** (seeds 42/43/44),
  in the correlated regime (>0.6), NOT ≈1.0 from accidental bipolarization. The codes are genuinely correlated.
- **Multi-seed 42/43/44.**

### The four gates (pre-registered thresholds, NOT tuned per result)
| gate | bar | what it tests |
|---|---|---|
| (a) DECORRELATION | between-cos ≤ 0.1 | the binding bar |
| **(b) REPRODUCIBILITY** | **same-input cos ≥ 0.9 at σ=0.1** | **the spiking-DG killer — the only untested gate** |
| (c) COMPOSITION | recovery ≥ 0.99 (argmax-parity) | not coherence-only (the 06-06 lesson) |
| (d) GENERALIZATION | held-out inference works (+ permuted-control fails) | the NEW capability Option B must add |

---

## STAGE B3 — the CEILING (ideal whitening): FAILS gate (b) on every mechanism, every seed

Apply an IDEAL / god's-eye whitening and ask: can ANY whitening co-satisfy (a)+(b)+(c) at one operating point?
Three mechanisms: RAW (floor), ideal ZCA `C^{−1/2}` (the realizable analytic), concept-whiten `Ω=ΓᵀΓ` (the
Deneve-Machens handed-in reference / the gentlest possible whitening).

**B3 ceiling table (multi-seed mean, 42/43/44):**

| mechanism | (a) decorrelation | (b) **reproducibility @ σ=0.1** | (c) composition | ALL (a+b+c) |
|---|---|---|---|---|
| RAW (floor) | −0.065 ✅ | **0.161 ❌** | 1.000 ✅ | **no** |
| ideal ZCA `C^{−1/2}` | −0.067 ✅ | **0.254 ❌** | 1.000 ✅ | **no** |
| concept-whiten `Ω=ΓᵀΓ` | −0.067 ✅ | **0.609 ❌** | 1.000 ✅ | **no** |

**Every mechanism PASSES decorrelation, PASSES composition, and FAILS reproducibility.** The best whitening
(concept-whiten) reaches only **0.609** at σ=0.1 — still well below the **0.9** bar (the exact bar the spiking dentate
gyrus failed at ~0.05). Per-seed reproducibility is tight: RAW [0.160, 0.163, 0.159], ZCA [0.258, 0.254, 0.249],
concept-whiten [0.598, 0.610, 0.620]. **The ceiling cannot clear gate (b).** ⇒ B3 does not clear ⇒ Option B's premise
fails at the cheapest, most-decisive test.

### Why reproducibility fails — two compounding mechanisms (rigorously diagnosed, not an artifact)

1. **The codes are intrinsically noise-fragile at σ=0.1.** The RAW (no-whitening) reproducibility is only **0.161** at
   σ=0.1, reaching 0.9 only at **σ ≈ 0.012**. So **no front-end at all** survives σ=0.1 on these codes — the
   concept-distinguishing residual sits in so small a sub-space that 10%-of-norm input noise swamps it. (Noise-threshold
   sweep, seed 42: RAW repro 0.987→0.952→0.831→0.441→0.166 at σ = 0.005/0.01/0.02/0.05/0.1.)
2. **Whitening makes reproducibility WORSE, not better.** Ideal ZCA is *below RAW* at low noise (0.527 at σ=0.005 vs
   RAW's 0.987) because it amplifies the low-variance directions — classic over-whitening noise amplification. The
   learned-M settle made this mechanistically explicit: **min-eig(I+M) = 0.0006** = a ~1700× whitening gain on a
   near-null direction (pure noise), which is BOTH the slowest settle mode AND the reproducibility killer — the same
   phenomenon. Concept-whiten is the gentlest (0.992 at σ=0.005) and degrades most gracefully but still only 0.609 at
   σ=0.1.

**There is no operating point (no `eps`, no whitening strength) that gives deco ≤ 0.1 AND repro ≥ 0.9 at σ=0.1.** A
rank/eps sweep confirmed it: at proj_dim=16 (full rank, no rank-deficiency) the best ideal-ZCA repro@0.1 is 0.18–0.34
(eps 1e-3→1e-1) — the eps knob trades toward gentler whitening but the gain is bounded far below 0.9. **This is the
§6.4 three-operating-points tension, made concrete and unavoidable.**

### Composition (c) is NOT the binding constraint at this scale — honest note
Composition passes at ≈1.0 for the centered/whitened codes because V=16 concepts is well below capacity (the cleanup
margin is large). The gate *is* mildly correlation-sensitive — truly-raw correlated codes compose at 0.958 vs 1.000
for whitened at D=256/roles=8, and the gap widens under stress (D=128/roles=12: raw **0.627** vs ZCA **0.945**) — so
gate (c) is a valid composition-not-coherence check, but at the denoise64 V=16 scale **it is not the constraint that
forces the NEGATIVE. Gate (b) is.**

### Generalization (d) — FLAGGED, not run (honest): denoise64 lacks graded similarity
The denoise64 correlation is **~uniform**, not graded: off-diagonal cosine mean 0.81, **std only 0.033–0.037**, range
0.15–0.18. There is no systematic "some concept pairs are closer than others" structure. Per §5.3, the held-out
generalization test is therefore **meaningless on these codes** and was FLAGGED, not faked. Generalization (the whole
reason Option B exists over the flat Option A) needs the **grounded/structured** codes (CIFAR-grounded / semantic), not
denoise64. This is itself a finding: even if whitening had cleared (a)+(b)+(c), denoise64 could not have demonstrated
(d) — the codes carry no graded semantics to preserve.

---

## STAGE B1 — the LEARNED Pehlevan-Chklovskii rule: reproduces the ceiling's failure; the analog machinery is SOUND

B1 was run for the record (it is cheap, and the learned-rule numbers + the analog-not-host machinery are informative
even when B1 is bounded by a failed B3 ceiling). The learned rule `ΔM_ij ∝ ⟨y_i y_j⟩ − δ_ij − λM_ij` (λ=0.01),
computed by the analog settle `dr/dt = W_ff·x − r − M·r̂`:

| gate | B1 learned (mean) | reading |
|---|---|---|
| (a) decorrelation | −0.066 ✅ | passes |
| **(b) reproducibility @ σ=0.1** | **0.249 ❌** | **≈ ideal ZCA's 0.254 — bounded by the failed ceiling** |
| (c) composition | 1.000 ✅ | passes |

**The learned rule reaches the ideal ZCA ceiling exactly (repro 0.249 ≈ 0.254) and inherits its failure** — neither the
ideal nor the learned whitening can reach 0.9 reproducibility.

### Analog-not-host proof — the machinery is genuine (so the NEGATIVE is about the codes, not a broken instrument)
- **(i) settle CONVERGES to the fixed point** (not a one-shot host `C^{−1/2}`): final rel-err to `(I+M)^{−1}x` =
  **0.0007** over 5000 steps. The slow convergence is itself diagnostic: min-eig(I+M)=0.0006 is the slowest mode AND the
  over-whitened (noise-amplifying) direction — the settle and the reproducibility failure are the same physics.
- **(ii) M is LEARNED + BOUNDED**: M grows from zero; **M-ratio = 0.037** (NOT a 9000× blow-up — the guard that caught
  the prior false-positive). This is a *gentle* whitening, consistent with the 2026-06-06 regularized `C^{−1/3}` result.
- **(iii) LESIONABLE (honest decomposition)**: full-front-end-OFF (raw row-normed codes) → between-cos **0.787,
  collapses=True** — the whitening rides the simulated mechanism. **BUT** the honest finding: the decorrelation in gate
  (a) is **dominated by the codebook-mean removal** (a trivial common-mode subtraction that alone takes 0.81 → −0.07);
  the learned lateral M adds **+0.000 incremental decorrelation** on top. So "M does the whitening" is overstated for
  this codebook — mean-centering does the heavy lifting; M is a minor residual term. `proof_holds=True` (the machinery
  is sound), but the substrate-realizable decorrelation here is mostly centering, not lateral inhibition.

---

## The reconciliation (§6.1): the 06-06 priors do NOT save Option B, because the GATE differs

The project's record was split and the optimistic 2026-06-06 prior was real — but it does **not** rescue Option B,
because this de-risk runs the **stricter gate it never faced**:
- **2026-06-06** validated a local whitening rule that COMPOSES at 100% 6/6 — gated on **composition of CIFAR-grounded
  codes at the rate level**. (Reproduced here: composition passes ✅ for whitened denoise64 too.)
- **2026-06-11 (this)** gated on **REPRODUCIBILITY-under-noise (σ=0.1) of denoise64** — and **even the IDEAL whitening
  fails it**, and the learned rule reaches only the same failing ceiling.

Mikulasch-Priesemann's dendritic balance is the mechanism for the noise regime this gate tests — but the de-risk shows
the problem is **upstream of any whitening mechanism**: the denoise64 codes are sub-reproducible at σ=0.1 *before*
whitening (RAW 0.161), and whitening *amplifies* the noise. A dendritic front-end (B1's analog settle is its
function-level stand-in) is the right machinery for low-SNR averaging, but it cannot manufacture reproducibility that
the input SNR does not contain — and it makes the over-whitened-direction problem worse, not better. **B2 (literal
per-neuron compartments) would face the identical wall** (it is the same whitening, computed more faithfully) — the
NEGATIVE is in the codes-at-this-noise-level, not the mechanism's fidelity.

---

## DECISION LOGIC (stated explicitly)

- **GO** would require: B3 ceiling clears (a)+(b)+(c) AND B1 reaches it with the analog-not-host proof + lesion,
  multi-seed. **NOT MET** — B3 does not clear gate (b).
- **NEGATIVE / BOUNDARY** otherwise, naming the failing gate. **MET: NEGATIVE**, forced precisely by **reproducibility
  (b) @ σ=0.1** — and it fails for the **IDEAL** whitening, so it is the strong form (§6.4): even a god's-eye whitening
  cannot co-satisfy decorrelation + reproducibility + composition on codes this correlated/noise-fragile. ⇒ **Option
  B's premise (whiten the similar codes directly to get a binding-ready, reproducible, generalizing cortex) FAILS.**

### ⇒ The decisive next step: the DUAL (complementary-learning-systems) architecture — NOT the B1/B2 build
Because the failure is the **§6.4 tension** (decorrelated-enough-to-bind and reproducible-at-0.9 are mutually
exclusive on these codes), and because denoise64 carries no graded similarity for (d) anyway, the answer the de-risk
points to is the **biology-faithful dual architecture**:
- **a "cortex" representation** that keeps the SIMILAR/correlated codes (for the generalization Option B wanted —
  measured on grounded/structured codes that actually carry graded semantics, NOT denoise64);
- **a linked decorrelated "hippocampal/cerebellar" expansion** that the binder/cleanup reads (the
  reproducible-and-decorrelated codes the project's own sparse-distributed positive control already validates at
  between-cos ≈ 0.05);
- coupled by encode/decode — the complementary-learning-systems answer the build-plan §"deep tension" already names.

**Do NOT commit the weeks-scale B1 build or the months-scale B2 rewrite** on the premise of whitening the similar codes
directly: this afternoon-scale falsification shows that premise does not survive the reproducibility gate even at the
ideal ceiling. The B1 analog-whitening population remains a sound, reusable component **inside** a dual architecture
(it is the right machinery for the decorrelated-expansion side), but it is not the path to a single similar-code cortex
that binds reproducibly.

---

## Anti-cheats (all reported, decisive)
- **Reproducibility ≥0.9 front-and-center** — the headline; it is the gate that forces the NEGATIVE; reported for every
  stage/mechanism/seed alongside between-cos so "decorrelated but irreproducible" cannot masquerade as a win.
- **Native-binary unit-check** — PASS, between-cos as-read 0.79–0.81 (NOT bipolarized).
- **M-ratio bound (B1)** — 0.037, bounded (no noise-collapse false-positive).
- **Lesion (B1)** — full-front-end-OFF collapses to 0.787 (whitening rides the mechanism), with the honest centering-vs-M
  decomposition.
- **Composition-not-coherence (gate c)** — passes; verified correlation-sensitive under stress (raw 0.627 vs ZCA 0.945
  at D=128/roles=12), so it is a real composition check, just not the binding constraint at V=16.
- **Permuted-similarity (gate d)** — N/A: (d) flagged-not-run because denoise64 lacks graded similarity (reported, not
  faked).

## Honest scope / what this does and does not show
- **Shows**: even the IDEAL whitening cannot co-satisfy decorrelation + reproducibility(≥0.9 @ σ=0.1) + composition on
  the REAL denoise64 codes, multi-seed; the failure is the reproducibility gate; the underlying codes are
  sub-reproducible at σ=0.1 even raw; whitening amplifies the noise; the learned PC rule reproduces the ceiling's
  failure; the analog-not-host machinery is sound.
- **Does NOT show**: that whitening is useless in general (it COMPOSES — gate c — and the 06-06 composition result
  stands); that σ=0.1 is the only meaningful noise level (it is the pre-registered spiking-DG bar, but a different
  substrate noise level would shift the threshold); that the DUAL architecture will work (that is the recommended next
  arc, unbuilt). Generalization on denoise64 was untestable (no graded similarity) — a separate reason the similar-code
  cortex must come from grounded/structured codes, not denoise64.

## Artifacts
- Probe: `research/runners/option_B_whitening_derisk_probe.py`
- Raw JSON: `research/findings/raw/_option_B_whitening_derisk.json` (multi-seed, B3 + B1, every gate, M-ratio + lesion +
  repro sweep), `research/findings/raw/_option_B_whitening_derisk_seed42_smoke.json` (single-seed).
- Spec followed: `docs/plans/2026-06-11-option-B-dendritic-substrate-research.md` §4 (de-risk), §5 (anti-cheats + (d)),
  §6.4 (the tension this confirms).
