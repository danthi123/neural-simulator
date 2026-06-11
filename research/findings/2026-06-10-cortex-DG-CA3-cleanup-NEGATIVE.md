# STEP 3 (true cortex) — resolve the cleanup boundary with the brain's TRISYNAPTIC LOOP (DG→CA3) — **NEGATIVE**

**Date:** 2026-06-10 (overnight thread; ran into 2026-06-11)
**Scope:** the follow-on to the STEP-3 cheap-first de-risk
(`research/findings/2026-06-10-cortex-learned-cleanup-derisk-PARTIAL.md`). That de-risk mapped a boundary: the
composer's learned Hopfield/CA3 attractor cleanup **collapses on the brain's raw correlated `denoise64` codes**
(0.045 ≈ chance vs argmax 1.000) and **recovers to argmax parity only with a learned decorrelation step** — but
the decorrelation it used was a **host ZCA linear transform**, which is a shortcut (the brain is not computing
it). This thread tested the brain-based replacement: the project's VALIDATED hippocampal **dentate-gyrus pattern
separation** (catalog D.12; the learned/spiking decorrelation) feeding a **CA3 autoassociative cleanup**
(catalog D.13) — together the **trisynaptic loop** (DG→CA3) the project validated multi-seed (CLAUDE.md "P1
trisynaptic loop"). Question: does DG→CA3 replace the composer's god's-eye argmax cleanup?
**Runner:** `research/runners/cortex_dg_ca3_cleanup_probe.py` (CPU, `SIM_BACKEND=numpy`, toy scale, no `sim/` edits).
**Substrate:** a real `SimulationBridge` built by `build_biological_brain_regions(enable_hippocampus_consolidation=True)`
— the **same builder** the validated P1 trisynaptic result used (EC→DG→CA3→CA1 with PV-basket feedforward
inhibition + a CA3→CA3 recurrent autoassociator).
**Codes:** the project's REAL `denoise64` concept codes (V=16, between-code cosine ≈ **0.81** at D=512 — highly
correlated, the brain's own concept-pool activity), NOT random-clean phasors.
**Raw:** `research/findings/raw/_cortex_dg_ca3_cleanup_probe_seed42.json` + the diagnostic chain
`_dgca3_{sweep,determinism,crux,repro2,strongdrive,repro_multiseed}_result.json`.

---

## TL;DR

**NEGATIVE.** The spiking DG→CA3 trisynaptic loop, as the project ships it, **cannot** serve as the composer's
cleanup on a code-cue. On the brain's raw correlated codes:

| | argmax (matched filter) | Hopfield-raw (de-risk collapse ref) | **DG→CA3 (this probe)** | chance |
|---|---|---|---|---|
| TEST 1 — full noised cue (parity) | **1.000** | 0.050 | **0.083** | 0.062 |
| TEST 2 — partial/occluded cue (completion) | 1.000 | — | **0.100** | 0.062 |

DG→CA3 sits at **chance** on both the parity test and the pattern-completion test. The argmax matched filter is
perfect on both (it is linear → immune to the common mode → robust on correlated codes, and a 40%-kept partial
cue still has enough signal for a matched filter).

**The root cause is upstream of CA3 and is mechanistic, not a tuning miss** (five confirming diagnostics): the
**spiking DG read is not a reproducible code.** Driving the *same* clean code twice — fresh reads, plasticity
gates closed, even with a 200–400-step averaging window and 5.5× drive — yields DG ensembles with cosine **≈
0.04–0.15** (near-orthogonal). DG fires only ~**15–62 spikes across 600 neurons per read**, so *which* cells win
the k-winners-take-all is dominated by background (OU) noise + spike-timing chaos, **not** by the input. The DG
therefore "separates" two reads of an *identical* input as strongly as it separates two *different* concepts —
this is **over-separation / sparse-noise, not stable pattern separation**. A CA3 autoassociator has no stable
attractor to store and no basin the cue reliably enters, so the cleanup is at chance.

This is the same boundary the project already documented for this loop ("the EC-driven completion test FAILED at
all parameter combinations; only DIRECT-CA3 drive PASSES" — CLAUDE.md "P1 trisynaptic loop"). The cleanup
use-case *requires* the EC→DG→CA3 feed-through (the cue is a code, not a pre-recorded CA3 ensemble), so it lands
squarely on the documented-failing path. **No banking** — reported as found.

---

## What the de-risk asked this thread to resolve

The de-risk's TEST 2 collapse was a **Hopfield capacity failure on correlated patterns**: the symmetric Hebbian
outer-product weight `W = C Cᵀ` acquires a dominant **common-mode eigenvector** when the stored codes are
correlated, so the settle locks onto it and reads out an overlap with *every* stored pattern equally (→ chance).
A linear matched filter (argmax) is unaffected by the common mode → stays at 1.000. Host **ZCA whitening**
(orthonormalize the codebook) removed the common mode → restored attractor = argmax = 1.000. The brain-based
equivalent of "orthonormalize the codebook" is **DG pattern separation** (catalog D.12). So the thesis under
test was: *route the noisy code-cue through the spiking DG → its separated (decorrelated) code → the CA3
autoassociator settles → read the recovered concept; this should recover argmax parity (TEST 1) and beat argmax
on partial cues where a matched filter has no completion (TEST 2).*

---

## Results (seed 42; the mechanism reconfirmed on 43/44, below)

### The probe (canonical numbers) — `_cortex_dg_ca3_cleanup_probe_seed42.json`

- **Codes:** V=16, D=512, raw between-code cosine **0.814** (the correlated stress).
- **Trained CA3 autoassociator** on the 16 codes (89 s). Reported **DG between-ensemble cos 0.027**, CA3
  between-ensemble cos 0.069. *At face value this looks like a D.12 success (0.81 → 0.03 orthogonalization) — but
  the diagnostics below show it is over-separation: the DG separates identical inputs just as much.*
- **TEST 1 (parity, full noised cue):** argmax **1.000** | Hopfield-raw **0.050** (reproduces the de-risk
  collapse) | **DG→CA3 0.083** (chance 0.062). Gate `DG→CA3 ≥ argmax − tol`: **FAIL**.
- **TEST 2 (completion, 40%-kept partial cue):** argmax-on-partial **1.000** | **DG→CA3 0.100** (chance). Gate
  `DG→CA3 > argmax`: **FAIL**.
- **Anti-cheat lesions:** the lesion mechanics work (zeroed 27 008 CA3↔CA3 recurrent entries; zeroed 11 483 +
  108 000 DG-FFi entries; the DG-FFi lesion measurably *reduced* separation, between-cos 0.027 → 0.064). But
  because intact DG→CA3 is **already at chance**, the lesion controls cannot demonstrate "rides the learned
  weights": CA3-recurrent-lesion 0.067 and DG-separation-lesion 0.017 are both ≈ the intact 0.083. **The lesion
  controls are uninformative when the intact capability is itself at chance** — an honest limitation of running
  the anti-cheats on a capability that does not clear chance.
- **VERDICT (probe): NEGATIVE** on all five gates.

### The diagnostic chain (the load-bearing root cause)

1. **Cleanup at chance regardless of cue noise or partial-keep** (`_dgca3_sweep_result.json`): DG→CA3 accuracy
   0.025–0.125 across full-cue noise ∈ {0.0, 0.1, 0.2, 0.3} **and** partial-keep ∈ {0.9, 0.7, 0.5, 0.3}. It
   fails even at **zero added noise** — so cue corruption is not the cause.
2. **The stored CA3 ensemble does not re-read** (`_dgca3_determinism_result.json`): driving the *same clean
   code* again and reading CA3 gives cosine **0.0** to the stored ensemble at baseline; with OU noise OFF the
   CA3 active fraction is **0.0** (CA3 barely fires at recall with gates closed). Even a **very strong attractor**
   (CA3 weight 8.0, 10 train events, 300 pA, 150-step read) gives re-read cosine **[0.089, 0.028, 0.11]** and
   clean-cue identity accuracy **0.0625** (chance). The EC→DG→CA3 feed-through readout does not converge to a
   reproducible CA3 state.
3. **The blocker is the DG read itself** (`_dgca3_crux_result.json`): two *fresh* reads of the same clean code
   give DG cosine **0.058** (min 0.0) — even with a 200-step averaging window. A noisy cue's DG ensemble is
   *less* similar to the same concept's clean DG (**0.065**) than to *other* concepts' DG (**0.184**) → the DG
   code carries **negative** concept identity across reads. A Hopfield built on the DG-separated codes
   (`hop_dg`) is at **0.021** (chance), confirming there is no stable code to store.
4. **Not a drive-convention artifact** (`_dgca3_repro2_result.json`): under the project's *validated*
   sparse-INDEX drive convention, DG reproducibility is **0.038** (min 0.0); DG fires only **15.5 spikes** total
   across 600 neurons (active frac 0.029). EC (upstream) is meaningfully reproducible (**0.30**), so the input
   carries identity — the irreproducibility is **born at the DG k-WTA**, not at the drive or at EC.
5. **Stronger/longer drive does not rescue it** (`_dgca3_strongdrive_result.json`): at **1200 pA / 400-step**
   read (62 DG spikes), DG reproducibility only reaches **0.15** and `hop_dg` stays at chance (0.05). More spikes
   help marginally but do not make the DG a stable input-determined code.

### Multi-seed confirmation of the mechanism (seeds 43, 44) — `_dgca3_repro_multiseed_result.json`

The load-bearing irreproducibility signature is seed-general (seed 42 numbers from the crux/repro2 runs above):

| seed | raw code cos | **DG same-input reproducibility** (mean / min) | DG spikes/read | noisy-cue DG: same vs other |
|---|---|---|---|---|
| 42 | 0.814 | 0.058 / 0.000 | ~15 | 0.065 **<** 0.184 |
| 43 | 0.798 | 0.033 / 0.000 | ~17 | 0.066 **<** 0.186 |
| 44 | 0.787 | 0.042 / 0.000 | ~16 | 0.048 **<** 0.155 |

All three seeds: the DG read of an identical input is near-orthogonal to itself (cosine ≤ 0.06), DG fires ~15–17
spikes across 600 neurons, and a noisy cue's DG ensemble is *less* like the same concept's clean DG than like
other concepts' — i.e. the DG carries **negative** identity. A single-seed full probe would be redundant: the
NEGATIVE is mechanistic and reconfirmed on 3/3 seeds at the level of the cause.

---

## Diagnosis — why the trisynaptic loop cannot be the code-cue cleanup (as the project ships it)

- **DG pattern separation is operating in the extreme-sparse / low-spike-count regime.** ~15–62 spikes spread
  over 600 DG neurons means the active set per read is ~15–35 neurons, and *which* ones win the feedforward-
  inhibited k-WTA is set by OU noise and initial-condition chaos, not by the (correlated) input. The result is
  a code that is near-orthogonal to *itself* across reads. The measured "0.027 between-concept cosine" is a
  symptom of this: the DG orthogonalizes everything, including identity — it is **over-separation**, not the
  robust separation D.12 requires (which must keep two presentations of the *same* input close).
- **A CA3 autoassociator (D.13) needs a stable stored pattern and a cue that reliably enters its basin.** Both
  preconditions fail: the stored pattern is irreproducible (the recall read ≠ the training-window-recorded
  ensemble), and the feed-through CA3 state barely fires at recall. So the attractor has nothing to complete.
- **This matches the project's own prior finding exactly.** CLAUDE.md "P1 trisynaptic loop": *"EC-driven test
  (drive lang_input, propagate through the trisynaptic chain) FAILED at all parameter combinations. DIRECT-CA3
  test (drive a partial of the stored CA3 ensemble directly) is the cleaner Marr autoassociator test and
  PASSES."* The validated D.13 = **direct-CA3 with a recorded ensemble**; the cleanup use-case needs the
  **EC→DG→CA3 feed-through of a code**, which is the documented-failing path. The probe independently rediscovers
  and quantifies that boundary, and adds the precise mechanism (sub-reproducible sparse DG read).
- **The argmax baseline is robust for the structural reason the de-risk gave:** it is a *linear matched filter*,
  immune to the common mode of correlated codes, and on a 40%-kept partial cue it still has ample signal. The
  attractor's hoped-for value-add (completion from a partial cue, D.13) never materializes because the DG
  front-end destroys the cue's identity before CA3 sees it.

## What would move it (the honest next approaches)

The NEGATIVE is specific: *the EC→DG→CA3 feed-through of a code-cue, on the project's stock hippocampus build, is
not a usable cleanup because the spiking DG read is sub-reproducible.* Candidate fixes, each brain-based and
each a cheap next probe (none attempted here — out of scope once the mechanism was nailed):

1. **Direct-CA3 storage + the de-risk's Storkey/pseudo-inverse weighting (the de-risk's named next step).** Skip
   the unstable DG feed-through: store the codes directly as CA3 patterns and use **covariance-corrected
   (Storkey) or pseudo-inverse** Hopfield weights, which raise correlated-pattern capacity *without* a separate
   decorrelation stage. This tests the de-risk's secondary option (its §"what would move it" item 2) directly,
   and sidesteps the DG-irreproducibility blocker entirely. **Recommended first** — it is the smallest change
   that could turn the cleanup positive.
2. **A reproducible DG code before the CA3 store.** Make the DG separation input-determined rather than
   noise-determined: (a) far more DG spikes per read (rate-coded, not 1-spike-per-cell) with a hard k-WTA on the
   *accumulated rate* rather than instantaneous spikes; (b) a learned EC→DG that produces a stable sparse code
   (the concept-pool training the project already ships, applied to the EC→DG pathway); or (c) reduce DG
   stochasticity (OU off + a deterministic winner read). Only if a DG read reaches, say, same-input cosine ≳ 0.7
   is it worth re-attaching the CA3 autoassociator. This is a larger build and should follow (1).
3. **Accept the de-risk's host-ZCA-as-stand-in with an explicit "learned decorrelation is the open piece" flag.**
   The de-risk already showed ZCA restores parity; if the goal is to *ship* Option A's cleanup, ZCA is a
   documented idealization stand-in for the DG separation that does not yet work on-substrate — but that is a
   shortcut by the project's standing bar, so it is a fallback, not a resolution.

## Anti-cheat / provenance (brain-based-only)

- The decorrelation under test was the **spiking DG** (k-WTA + PV-basket feedforward inhibition on a real
  bridge), NOT a host ZCA linear transform — exactly as the task required. The cleanup under test was the
  **spiking CA3 autoassociator** settle, NOT an argmax over a god's-eye codebook.
- The argmax baseline and the Hopfield-raw baseline appear **only** as references (the composer's idealization
  and the de-risk's documented collapse); neither is the deliverable.
- The cue was rendered to `language_input` drive by a fixed positive-rectified projection (presenting sensory
  drive = the environment's legitimate job); reading a region's firing-rate vector is a readout, not a host
  computation of the match. No `sim/` edits — the trisynaptic loop is the project's existing
  `build_biological_brain_regions` wiring, reused by import.
- The lesion mechanics are validated (the CSR-zeroing zeroes the correct edges and the DG-FFi lesion measurably
  reduces separation), but the lesion *controls are uninformative here* because the intact capability is at
  chance — reported honestly rather than dressed up.

---

## VERDICT: NEGATIVE → next step

**NEGATIVE.** The spiking DG→CA3 trisynaptic loop (project stock build) does **not** replace the composer's
argmax cleanup on a correlated code-cue: it is at chance on both the full-cue parity test and the partial-cue
completion test, because the **spiking DG read is sub-reproducible** (~0.04–0.15 same-input cosine; ~15–62
spikes / 600 neurons), so the CA3 autoassociator has no stable attractor to store or basin to enter. This
reconfirms — and mechanistically explains — the project's prior "EC-driven completion fails; only direct-CA3
passes" boundary.

**Recommended next step (cheap, CPU):** the de-risk's named option (1) above — **direct-CA3 storage with
Storkey/pseudo-inverse Hopfield weights** on the raw correlated codes, which targets the correlated-pattern
capacity directly and bypasses the DG-irreproducibility blocker. If that recovers argmax parity AND shows
pattern-completion value-add on partial cues, Option A's learned cleanup is brain-based-viable → proceed to the
V=320 cortex probe (GPU). Until then, Option A's cleanup remains **conditional on a decorrelation/storage front
end that works on-substrate** — the host ZCA of the de-risk is a documented stand-in, not a resolution.

**No banking** — the cheap-first follow-on reported exactly as found.
