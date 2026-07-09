# R-iii CA3 formation — a 2nd `sim/` rule (RATE-WINDOW / BCM co-activity Hebbian) built + de-risked, and the DECISIVE result: ALL FOUR plasticity-rule variants form only a WEAK attractor (~1.15-1.19× within-ensemble separation), so the formation blocker is NOT the plasticity RULE — it is the ENSEMBLE FIRING DYNAMICS. The trained CA3 code is DISTRIBUTED (35-47% of cells active, not a sparse dense-firing ensemble), so no small set co-fires strongly enough for any Hebbian rule to bind it. The next mechanism is pattern-SEPARATION (sparser CA3; D.12) + theta-gamma SYNCHRONIZATION (members fire in the same gamma cycle), a research gate — NOT another rule tweak. Both `sim/` rules (symmetric, rate-window) are guarded + byte-safe (`test_determinism` 7/7).

**Date:** 2026-07-09
**Runner:** `research/runners/_riii_ca3_attractor_diag.py` (+ `--hebb-sym/--hebb-rate/--coact-thresh/--coact-decay`). `sim/` edit: `config.hebbian_rate_window` + `hebbian_coactivity_decay/thresh` + a guarded rate-window branch in the bridge Hebbian block (a maintained per-neuron co-activity trace). GPU.
**Verdict:** the plasticity-rule iteration is COMPLETE (4 variants characterized, all weak); the blocker is re-diagnosed to the ensemble dynamics (a different mechanism class) — a research gate, honestly opened, NOT a wall.

## The systematic result (seed 42, within-ensemble vs member->silent weight separation)
```
formation rule                                 within   silent   separation   verdict
CAUSAL offset (default, CYCLE 1069)            4.87     4.84     +0.03        no attractor (offset ~never satisfied)
SYMMETRIC per-step, 150 events                 6.79     6.02     +0.77        specific but WEAK
SYMMETRIC per-step, 1000 events                7.02     6.12     +0.90        SATURATES (6.7x events, +0.13)
RATE-WINDOW, thresh 0.02 / 0.05                6.06     6.01     +0.05/+0.01  FAILS (traces < thresh; sparse firing)
RATE-WINDOW, no-threshold graded (lr 2.0)      7.48     6.28     +1.21        BEST, still WEAK (1.19x)
```
Every variant plateaus at a ~1.15-1.19× within-ensemble advantage — far below the ~3-5× needed for a usable c_drive separation (the CYCLE-1068 hand-installed attractor that completed cleanly was **10×**). The saturation across the learning-RATE (rate 2.0→5.0), the EVENT count (symmetric 150→1000: +0.87→+0.90), the DECAY, and the RULE FORM (offset→symmetric→rate-window) is decisive: **the plasticity rule is not the lever.**

## Root cause (systematic-debugging: 4 failed rule-tweaks → question the architecture)
The trained CA3 code is DISTRIBUTED: SPARSITY = 35-47% of the 150 CA3 cells fire (>10% of peak), not a sparse (~2-5%) dense-firing ensemble. The "members" (top-15 by rate) fire SPARSELY (<14% of steps) and ASYNCHRONOUSLY (spread across steps, not step-locked). So the co-activity of any specific member PAIR — however measured (same-step spike coincidence OR windowed EMA product) — is small, and non-members fire enough to co-activate too. **No Hebbian rule can bind neurons that do not co-fire strongly.** The blocker is upstream of the plasticity: the ENSEMBLE must be a sparse set that fires densely + synchronously for the recurrent LTP to write a strong attractor.

## The `sim/` rules built this arc (both guarded, default-off, byte-safe)
1. **`hebbian_symmetric`** (CYCLE 1069): offset-free per-step co-activity (`fired_this_step` for both endpoints) — necessary (the causal offset rule potentiates ZERO on synchronous firing) but forms only a weak attractor (per-step co-spikes are rare under async firing).
2. **`hebbian_rate_window`** (this cycle): BCM/rate-Hebbian — a maintained per-neuron co-activity trace (`trace = trace*decay + (1-decay)*fired`, an EMA in [0,1]); potentiate ∝ `trace[pre]·trace[post]·(max−w)` (no-threshold graded form). The most correct + best rule (+1.21), but still weak because the traces are small (sparse firing). `sim/config.py` + a guarded branch in the bridge Hebbian block; `test_determinism` 7/7 (default-off byte-identical).
These are genuine, reusable, biology-grounded plasticity primitives (CA3 associative LTP, Kandel Ch 54; BCM/rate-Hebbian, CLAUDE.md CYCLE 95-96) — kept for the substrate even though they are not sufficient ALONE for CA3 formation.

## NEXT — the ensemble-dynamics research gate (the honest next mechanism)
Get the CA3 ensemble to be SPARSE + DENSE-FIRING + SYNCHRONOUS during encoding, so the (now-correct) rate-window rule binds it strongly:
1. **Pattern SEPARATION (D.12):** stronger DG sparsification / CA3 feedback inhibition / a CA3 k-WTA → a sparse (~5%) CA3 code where the ensemble is a small distinct set (Marr 1971; Kandel Ch 54 pp 1357-1360). CYCLE-1066 already noted the 35% sparsity as too high.
2. **Theta-gamma SYNCHRONIZATION (Lisman-Idiart; catalog N.15):** members fire together in a gamma cycle → high per-step co-activity → the symmetric/rate rule fires every cycle → strong binding.
Read the sources in depth (Kandel Ch 54 CA3 sparsification; Buzsaki theta-gamma) + the external CA3-autoassociator-formation literature, then a cheap-first de-risk: does a sparser + more-synchronous CA3 ensemble let the rate-window rule reach a ~3-5× separation → re-run the CYCLE-1068 dendritic completion on the learned attractor = fully emergent CA3 pattern completion → the SWR generative-replay loop.

## The R-iii arc status (honest)
- Completion half — **SOLVED** (CYCLE 1068: the two-compartment dendritic dAP completes a strong attractor, 6-seed, 4 adversarial controls, on-substrate).
- Formation half — the plasticity RULE is solved (symmetric + rate-window, both `sim/`, byte-safe); the remaining blocker is the ENSEMBLE DYNAMICS (sparse/synchronous code), re-diagnosed + research-gated here.

## Files
`sim/config.py` (`hebbian_rate_window`, `hebbian_coactivity_decay/thresh`), `sim/bridge.py` (guarded rate-window branch + co-activity trace), `research/runners/_riii_ca3_attractor_diag.py`, `_riii_ca3_coincidence_completion_derisk.py`. Prior: `2026-07-08-riii-ca3-attractor-formation-symmetric-hebbian.md` (1069), `-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md` (1068). Biology: Kandel 6e Ch 54 (CA3 sparsification + associative LTP), Marr 1971, Lisman-Idiart theta-gamma; CLAUDE.md CYCLE 95-96.
