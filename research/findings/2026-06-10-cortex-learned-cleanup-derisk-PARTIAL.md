# STEP 3 (true cortex) cheap-first de-risk — Option A + systematicity control — **PARTIAL**

**Date:** 2026-06-10
**Scope:** the CPU-cheap de-risk that must pass BEFORE any full Step-3 build, per
`docs/plans/2026-06-10-step3-true-cortex-design.md` (§2 Option A, §3 familiarity gate, §5 the cheap-first
de-risk, §6 anti-cheats, §7 the systematicity risk + the discipline to run it FIRST).
**Runner:** `research/runners/cortex_learned_cleanup_derisk.py` (CPU, `SIM_BACKEND=numpy`, toy scale).
**Codes:** the project's REAL `denoise64` concept codes (the brain's own concept-pool activity), NOT
random-clean phasors. Seeds **42 and 43** (both PARTIAL, identical pattern).
**Raw:** `research/findings/raw/_cortex_learned_cleanup_derisk.json` (seed 42),
`_cortex_learned_cleanup_derisk_seed43.json` (seed 43).

---

## TL;DR

Option A keeps the FHRR bind/unbind **operations** and replaces only (a) the god's-eye argmax cleanup with a
**learned Hopfield/CA3 attractor**, and (b) the host-`if` abstention with a **learned spiking familiarity gate**.
Run on the brain's REAL correlated codes (between-code phase-cosine ≈ **0.70**):

1. **Systematicity (the core risk) — HOLDS.** Held-out novel-combination accuracy **1.000** = trained accuracy
   1.000 = the argmax algebra ceiling 1.000. The expected Fodor-Pylyshyn negative does **not** bite Option A
   (it keeps the systematic operation). As designed, that negative is reserved for Option C's learned readout.
2. **Learned attractor cleanup vs argmax — the mapped boundary.** On **raw correlated** codes the attractor
   **collapses to chance** (0.045–0.080 vs argmax 1.000). **With a learned-decorrelation step** (ZCA / DG
   pattern separation) it **recovers to 1.000 = argmax**. So Option A's attractor is viable **only** with
   learned decorrelation — it does **not** beat argmax on the raw correlated codes by itself.
3. **Familiarity gate (the no-confab moat) — clean PASS.** The learned anti-Hebbian signal separates known
   (novelty ≈ 0.000) from unknown (≈ 0.99) with margin **+0.98**, 100% / 100% at the midpoint threshold.

**Both lesion anti-cheats fire:** zero the attractor's recurrent weights → cleanup → chance; zero the
familiarity weights → separation collapses (margin → 0). The behaviours ride the LEARNED weights, not the
algebra or a host path.

**VERDICT: PARTIAL.** Systematicity holds, the familiarity gate separates cleanly, and the learned attractor
matches argmax — **but only with a learned-decorrelation step; not on the brain's raw correlated codes.**
This is the §7-anticipated secondary boundary, now precisely mapped.

---

## Results (seed 42; seed 43 identical pattern)

### Codes (the correlated-code stress, auditable)
- V = 16 concept words, projected to D = 512.
- **Raw** between-code phase-cosine: **0.702** (seed 43: 0.688) — highly correlated, the brain's real codes.
- **ZCA-decorrelated** between-code phase-cosine: **−0.006** (seed 43: −0.000) — orthogonalized.
- The phase map is a fixed complex random projection (`phase = angle(W_c @ code)`): full-circle spread (a
  valid FHRR code) that **preserves** the cross-code correlation, so the clean-code demand is genuinely
  stressed (an earlier per-dimension CDF map silently decorrelated the codes — caught and fixed).

### TEST 1 — SYSTEMATICITY control (run FIRST; the core risk) — **HOLDS**
Train/imprint on a role-filler grid (agents × actions × patients); HOLD OUT a novel **combination** whose
parts were trained but the combination never was (e.g. parts of "dog go north" + others trained, but
"cat go north" never), then store + query the held-out triple. Reported on the regime where the learned
attractor functions (decorrelated):

| | trained-combo acc | **HELD-OUT novel-combo acc** | chance/role |
|---|---|---|---|
| LEARNED attractor cleanup | 1.000 (n=240) | **1.000** (n=80) | 0.100 |
| argmax (algebra ceiling) | — | **1.000** | 0.100 |

A systematic binder handles the never-seen combination; Option A does, because the FHRR operation is identical
for every operand and the attractor stores only the **filler codebook** (not the facts). No degradation on
novel combinations → Option A inherits the algebra's free systematicity.

### TEST 2 — learned attractor cleanup vs argmax (REAL codes; both regimes) — **conditional**
Bind a fact, unbind one role, clean up the noisy estimate. Gate: attractor ≥ argmax.

| regime | argmax | LEARNED attractor | lesioned attractor | chance |
|---|---|---|---|---|
| **RAW correlated** (cos 0.70) | 1.000 | **0.045** | 0.065 | 0.062 |
| **ZCA decorrelated** (cos 0.00) | 1.000 | **1.000** | 0.065 | 0.062 |

- On the brain's **raw correlated** codes the Hopfield attractor **collapses to chance** — the well-known
  Hopfield capacity failure on correlated patterns: W = S·Sᵀ acquires a dominant common-mode eigenvector, the
  settle saturates so its read-out overlaps **every** stored pattern equally. A linear matched filter (argmax)
  is unaffected by the common mode and stays perfect.
- With **learned decorrelation** (ZCA = the dentate-gyrus pattern-separation step Option A names, catalog
  D.12) the attractor **recovers to argmax parity (1.000)**.
- **Lesion anti-cheat:** zeroing the recurrent weights drops the attractor to chance (0.065) in BOTH regimes —
  the cleanup rides the LEARNED weights.

### TEST 3 — learned anti-Hebbian familiarity gate (the no-confab moat) — **clean PASS**
Imprint known concept codes (Bogacz-Brown anti-Hebbian = the projector onto the stored span); present known
cues (familiar) and never-imprinted random cues (novel). Novelty energy N(x) = ‖x‖² − xᵀWx.

| | known (familiar) | unknown (novel) | margin (unk.min − known.max) |
|---|---|---|---|
| novelty | mean 0.000, max 0.000 | mean 0.991, min 0.982 | **+0.982** (seed 43: +0.985) |

- A clean threshold exists (max-known < min-unknown): **100%** of known cues read familiar, **100%** of
  unknown cues read novel → the gate would *answer* on the known and *abstain* on the unknown.
- **Lesion anti-cheat:** zero the learned weights → N(x) = ‖x‖² for every cue → margin **→ 0** (collapsed) →
  the separation rides the LEARNED weights, not a host max-similarity `if`.

---

## Interpretation

- **The familiarity gate is the strongest result and is regime-independent.** The Bogacz-Brown anti-Hebbian
  projector cleanly separates known from unknown on the brain's **correlated** codes (their headline
  high-capacity-on-correlated-inputs property), with both directions at 100% and a large margin. This is the
  load-bearing no-confab piece (§3) and it de-risks cleanly. It is a **learned, lesionable** signal — abstention
  becomes a computed spiking signal, replacing the host bookkeeping.
- **Systematicity is not the bottleneck for Option A.** Option A keeps the exact-inverse FHRR operation, so it
  inherits the algebra's free systematicity (held-out novel combinations recover at 1.000). The §7 core risk —
  a *learned readout* losing systematicity — is correctly reserved for **Option C**, and this control is now
  the ready, validated probe to run on Option C at toy scale **before** any GPU train.
- **The learned attractor cleanup is the conditional piece.** It matches argmax — but **only after learned
  decorrelation**. On the brain's raw correlated codes it buys nothing (it under-performs argmax at chance).
  This is exactly the §7 secondary-negative ("the learned attractor cleanup may not beat argmax on these
  particular codes"). It does not sink Option A, but it means Option A's cleanup is **attractor + a
  learned-decorrelation front end**, not attractor-on-raw-codes. The decorrelation is itself biologically
  grounded (DG pattern separation, catalog D.12; the project already ships ZCA in `CoreSimComposer`).

## Honest boundary mapped (what would move it)

- **Why the attractor collapses on correlated codes:** the symmetric Hebbian outer-product weight has a
  large common-mode component when patterns are correlated; the settle locks onto it. Fixes (all biologically
  grounded, all candidates for the Option-A front end):
  1. **Learned decorrelation before storage** (the validated fix here: ZCA / DG sparse pattern separation,
     catalog D.12) — restores attractor = argmax parity.
  2. **Pseudo-inverse / Storkey storage** (covariance-corrected Hopfield weights) — raises correlated-pattern
     capacity without an explicit decorrelation stage; untested here, a cheap next probe.
  3. **Sparse k-winners-take-all coding** (the DG sparsity catalog D.12 prescribes) before the CA3
     autoassociator — sparsifies the patterns so the outer product separates them.
- **The argmax baseline is robust precisely because it is a linear matched filter** (immune to the common
  mode). The value the learned attractor adds over argmax is *content-addressable pattern completion from a
  partial cue* (catalog D.13) — which matters for the perception→memory cross-code bridge (Step-3 §8 step 6),
  not for the clean single-role unbind tested here. A fair head-to-head of that *completion* capability (drive
  a partial/occluded cue) is the right next comparison, distinct from this unbind test.

## Anti-cheat / provenance (brain-based-only)

- Cleanup = the LEARNED attractor's recurrent settle (vocabulary in distributed Hebbian outer-product weights;
  no argmax over a god's-eye enumerated list). Lesion → chance.
- Familiarity = the LEARNED anti-Hebbian pool's novelty energy (not a host max-similarity `if`). Lesion →
  margin collapses.
- The phasor cue is rendered to real activity by the standard I/Q population read `[cos, sin]` (a *readout*
  of phase, not a numpy *computation* of the match).
- The argmax appears **only** as the baseline being replaced (reported for the gate), never as the deliverable.
- The FHRR bind/unbind/bundle are the project's validated spiking-phasor primitives, reused by import; no
  `sim/` edits.

---

## VERDICT: PARTIAL → next step

**PARTIAL.** Two of three pillars pass on the brain's correlated codes (systematicity holds; the learned
familiarity gate separates cleanly with a lesionable, large margin), and the learned attractor matches argmax —
**but only with a learned-decorrelation front end**, not on the raw correlated codes.

**Recommended next step (cheap, before any heavy build):** keep the **familiarity gate as-is** (it is the
load-bearing no-confab piece and it passes) and resolve the attractor's correlated-code collapse with a cheap
front-end probe — **(1) ZCA/DG learned-decorrelation (already validated here to restore parity) or (2)
Storkey/pseudo-inverse correlated-pattern Hopfield storage** — then re-run TEST 2 on the **raw** codes. If a
decorrelation/storage front end makes the attractor beat argmax on the raw codes (and the **pattern-completion**
head-to-head, not just unbind, shows the attractor's added value), Option A is GO → proceed to the `V=320`
`vocab_ceiling_probe` (GPU). The Option-C **systematicity probe** (the real Fodor-Pylyshyn test) is now
ready to run at toy scale on CPU **first**, exactly per §7 discipline, whenever Option C is attempted.

**No banking** — this is the cheap-first de-risk result reported as found.
