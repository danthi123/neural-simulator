# EMERGE-49 / toward-semantics — BOUNDARY (honest, 3-seed): the GRADED DRIVE/READ rung does NOT surpass the EMERGE-46 fully-spiking-stacked-pooler boundary — because the on-substrate learned L2 permanences are GENUINELY BIMODAL (collapsed near 0), not graded-under-the-threshold. A graded read has almost nothing graded to read: at the soft/union L2 depression rate (`ld_wi`=0.005) 97-99% of the learned permanences sit near 0 (mid-band 0.2-0.8 ≈ 0.03), so `graded_read`/`graded_drive` reproduce the over-selective regime (super-acc 0.00, held-within ≈ 0.005) exactly as `hard` does; turning depression FULLY OFF (`ld_wi`=0) makes the permanences graded (mean 0.425, mid-band 1.00) but collapses to INDISCRIMINATE COLLISION (within ≈ cross ≈ 0.06, super-acc 0.50 ≈ chance). The EMERGE-48 residual is CONFIRMED to be the on-substrate pooler's LEARNED TUNING DYNAMICS (the winner-inactive depression over-sparsifies to a hard connected/not-connected split), NOT the read threshold — so the graded read is not the fix. The honest negative IS the deliverable; the next rung is the Földiák (1991) trace / temporal-continuity rule (rung a). NO NEW `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge49_graded_read_derisk.py` (`--diag` fast single-seed histogram + `--onsubstrate` the decisive 3-seed port); CI guard `tests/test_emerge49_graded_read.py` (3 tests, pinning the LEARNED-PERMANENCE facts on the real substrate — bimodal-at-soft-ld, graded-at-ld=0, graded-read≡hard-codon — the L2 pooler alone, ~10s total). Reuse-by-import (`_emerge44` task; `_emerge46` `OnSubstratePooler` + bridge; `_emerge47` `compute_idf_weights`; `_emerge14`/`_emerge12` kernels); CPU numpy-backend; 3-seed (42/43/44). Launched by the EMERGE-48 BOUNDARY's identified rung b (a graded read to reproduce numpy's soft-pooling window). Prior: `2026-07-02-emerge48-soft-l2-pooling-BOUNDARY.md`, `2026-07-02-emerge47-l2-input-normalization-BOUNDARY.md`, `2026-07-02-emerge46-spiking-stacked-pooler-BOUNDARY.md`.

## The mechanism tested (rung b — the cheapest of the two EMERGE-48 candidates)
EMERGE-48 ISOLATED the residual: the on-substrate L2 pooler has NO soft-pooling WINDOW — it jumps from over-selective (super-acc 0.03) straight to indiscriminate collision (super-acc 0.53 ≈ chance) as the winner-inactive depression drops, SKIPPING numpy's clean window (super-acc 0.06→1.00 across ld=0.15→0.005). EMERGE-48's suspected cause: the on-substrate drive/winner-read uses a HARD `perm > 0.5` connected-threshold, whereas numpy `(W>0.5)@x` COMBINED WITH its update trajectory keeps a graded pooling window. Rung b gives the on-substrate L2 pooler a GRADED drive read so a soft-pooling window can exist. Three single-variable variants (`_emerge46`'s `_drive`/`codon` read `cp_connections.data`; the graded variant reads the raw permanence value instead of `>0.5`):
- **`graded_drive`** — the L2 winner-selection drive is the GRADED raw-permanence-weighted sum `sum(perm · x)` (steers BOTH training winner-selection AND codon read).
- **`graded_read`** — TRAIN with the vanilla hard-threshold drive (== EMERGE-46), but READ the final L2 codon by ranking columns via the graded permanence-weighted drive.
- **`hard`** — == EMERGE-46 exactly (the boundary control).

The learning kernels are BYTE-UNCHANGED — the graded read only changes HOW `cp_connections.data` is READ into a column drive (raw perm vs `perm>0.5`). NO NEW `sim/` edit.

## THE DECISIVE DIAGNOSTIC — the learned on-substrate L2 permanences are GENUINELY BIMODAL (the load-bearing honesty check)
Per the task's step 3 (diagnose graded-vs-bimodal FIRST): the permanence histogram after L2 learning, swept across `ld_wi` (seed 42, `--diag`):

| L2 `ld_wi` | graded mode | held-within | held-cross | super-acc | perm mean | mid-band(0.2-0.8) | near0 | near1 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | hard | 0.003 | 0.000 | 0.00 | 0.012 | **0.03** | **0.97** | 0.00 | BIMODAL |
| 0.02 | graded_read | 0.002 | 0.000 | 0.00 | 0.012 | 0.03 | 0.97 | 0.00 | BIMODAL |
| 0.005 | hard | 0.013 | 0.000 | 0.08 | 0.013 | **0.03** | **0.97** | 0.00 | BIMODAL |
| 0.005 | graded_read | 0.005 | 0.000 | 0.00 | 0.013 | 0.03 | 0.97 | 0.00 | BIMODAL |
| 0.00 | hard | 0.066 | 0.061 | 0.50 | 0.425 | **1.00** | 0.00 | 0.00 | GRADED (but COLLISION) |
| 0.00 | graded_read | 0.054 | 0.043 | 0.67 | 0.425 | 1.00 | 0.00 | 0.00 | GRADED (but COLLISION) |

Histogram at `ld_wi`=0.005 (hard), bins [0.0..1.0]: `frac = [0.97, 0.00, 0.00, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00]` — a sharp spike at 0, a negligible tail elsewhere. At `ld_wi`=0.0: `frac = [0.00, 0.00, 0.00, 0.40, 0.40, 0.20, ...]` — everything in [0.3, 0.6], truly graded.

**The answer is BIMODAL, not graded-under-the-threshold.** At ANY nonzero `ld_wi` the accumulating winner-inactive depression (over 400 epochs × 240 samples) drives 97-99% of the L2 permanences to near 0 — a hard connected/not-connected split with essentially no graded middle. So a graded read has nothing graded to read: `graded_read` (held-within 0.005) and `graded_drive` (held-within 0.001) reproduce the over-selective failing regime EXACTLY, super-acc 0.00. Only `ld_wi`=0 makes the permanences graded (mean 0.425, mid-band 1.00) — but that is the COLLISION regime (within ≈ cross ≈ 0.06, super-acc 0.50 ≈ chance; `graded_read` at ld=0 gives within 0.054 vs cross 0.043 = discrimination 0.011 < 0.05, still collision). The graded read is NOT the fix.

## ON-SUBSTRATE PORT — the 3-seed anti-cheat table (`graded_read`, `ld_wi`=0.005)
Porting the graded read to EMERGE-46's `OnSubstratePooler` (both pooler layers' learning in `cp_connections`, the committed `sim/` kernels; the graded read on the L2 pooler), 3-seed (42/43/44):

| on-substrate arm | held-within | held-cross | super-acc (per-seed) | L2-group |
|---|---|---|---|---|
| **stacked_graded (graded_read, ld_wi=0.005)** | 0.004 | 0.002 | **0.00** (0.00/0.00/0.00) | +0.07 |
| permuted_cooc | 0.004 | 0.007 | 0.03 (0.00/0.00/0.08) | +0.07 |
| dAP-lesion | 0.005 | 0.002 | 0.00 (0.00/0.00/0.00) | +0.08 |
| l2lesion (untrained random L2, reported-not-gated) | 0.064 | 0.055 | 0.61 (0.67/0.75/0.42) | +0.05 |

L2-permanence histogram (learned, stacked_graded seed 42): mean 0.013, mid-band(0.2-0.8) **0.03**, near0 **0.97**, near1 0.00 → **BIMODAL**.

**GATE misses (3-seed):** super-acc 0.00 < 0.80; not ≥ permuted (0.03) + 0.25; not ≥ dAP-lesion (0.00) + 0.30; NO within>cross discrimination (within 0.004 vs cross 0.002 = collision, not generalization). The `graded_read` at the soft `ld_wi`=0.005 reproduces the EMERGE-46 over-selective boundary EXACTLY (super-acc 0.00, held-within 0.004), because the bimodal-at-0 connectivity leaves nothing for the graded read to exploit. The l2lesion (untrained-random-L2) arm shows the collision-at-chance signature (within ≈ cross ≈ 0.06, super-acc 0.61) — REPORTED-not-gated, as in EMERGE-46/48.

## The genuine residual, now doubly-confirmed (the deliverable)
EMERGE-48 said the residual "is NOT a depression-rate knob (the numpy dominant lever) but a deeper representation-dynamics limit (no graded soft-pooling window)." EMERGE-49 pins the MECHANISM: the on-substrate potentiation (`fused_htm_permanence_update` lp=0.05) + winner-inactive depression (`fused_htm_winner_inactive_depression` lam_dep_wi) over 400 epochs drives the winner permanences to a **BIMODAL split at 0** (97-99% near 0), not the graded real-valued distribution numpy's `(W>0.5)@x` sits on. This is exactly EMERGE-48's predicted branch: *"IF genuinely bimodal → the graded read won't help."* The graded read cannot manufacture a soft-pooling window from a bimodal connectivity — the columns are either connected to a feature (perm ≈ 1, though here even the connected mass is thin) or not (perm ≈ 0); there is no partially-connected shared band to read. Softening `ld_wi` doesn't create a graded band, it just moves the near-0 collapse threshold; taking `ld_wi`=0 removes the depression entirely and the pooler stops discriminating (collision). The residual is the LEARNED TUNING DYNAMICS (the potentiation/depression trajectory over-sparsifies to bimodal), NOT the read threshold — confirmed independently by the histogram AND by the graded read failing to change the outcome.

## Honest verdict — BOUNDARY (the graded read is not the fix; the perms are genuinely bimodal)
**A GRADED DRIVE/READ does NOT surpass the EMERGE-46 boundary on-substrate.** GATE misses (3-seed): super-acc 0.00 ≪ 0.80; not ≥ permuted + 0.25; not ≥ dAP-lesion + 0.30; NO within>cross discrimination (within ≈ cross ≈ collision at ld=0, or both ~0 at soft ld). Per the master directive + the surpass gate, this is an honest characterized boundary — **the honest negative IS the deliverable**: it CONFIRMS (with the permanence histogram) that the on-substrate point-neuron competitive pooler's generalization residual is the LEARNED bimodal tuning dynamics, not the readout threshold. NOT forced to a GO (super-acc 0.00 ≪ 0.80; the only arm that raises overlap, ld=0, breaks discrimination into collision).

## Next rung (the genuine surpass path — rung a)
The graded read (rung b) is now ruled out: the permanences are bimodal, so no readout change recovers the soft-pooling window. The remaining rung is the **Földiák (1991) trace / temporal-continuity rule** (rung a, EMERGE-48's primary candidate): a slow eligibility trace that pools features co-occurring in TIME (present same-superordinate members in temporally-contiguous bouts; the trace binds their L1 codons into shared L2 columns), creating the shared-but-discriminative tuning STRUCTURALLY rather than by relaxing selectivity or reading softer. This needs GROUPED / curriculum presentation (a training-protocol change, not a rate knob or a read change) and is the EMERGE-50 candidate. A deeper alternative if the trace rule also boundary-walls: a soft-bounded potentiation/depression (so permanences settle at graded values in the middle band while still discriminating) — but that is a learning-rule change with its own de-risk, not a cheap knob.

## Anti-cheats (all correctly-behaving)
- **Held out ENTIRE sub-categories {2,5}** (as EMERGE-44/46/47/48) — a held-out member can inherit ONLY via the L2-discovered grouping.
- **PERMUTED-co-occurrence** on-substrate — the stacked arm does not clear it (super-acc 0.00), correctly reporting the boundary.
- **dAP-lesion** — the coincidence-plateau read is load-bearing.
- **The shortcut guard FIRED CORRECTLY**: at the only regime that raises held-out overlap (`ld_wi`=0) within ≈ cross (0.054 vs 0.043 = discrimination 0.011 < 0.05), so the verdict logic reports COLLISION (not generalization) and refuses the GO — it did NOT let indiscriminate collision masquerade as inheritance.
- **l2lesion** (untrained random L2) — REPORTED-not-gated (a fixed-random control, per the anti-cheat control-validity methodology).

## Honest scope
- **3 seeds (42/43/44)**; the bimodality is consistent (near0 0.97-0.99 across seeds; on-substrate stacked super-acc 0.00/0.00/0.00).
- The permanence-histogram diagnostic (the load-bearing finding) is deterministic and robust — it appears within 60 epochs (CI-pinned at reduced epochs), driven by the cumulative winner-inactive depression saturating quickly.
- Winner SELECTION is a host top-k over the on-substrate drive (EMERGE-41 has the spiking FS-WTA version).
- NO NEW `sim/` edit (the graded drive/read is a HOST-side read of `cp_connections.data` using the raw permanence instead of the `>0.5` threshold; the committed learning kernels `fused_htm_permanence_update` + `fused_htm_winner_inactive_depression` are byte-unchanged).

## Artifacts
`research/runners/_emerge49_graded_read_derisk.py` (`GradedOnSubstratePooler`/`GradedSpikingStackedPoolerProbe` via `_build_onsubstrate_probe`, `_onsubstrate_run`, `_diag`; `--demo`/`--diag`/`--onsubstrate`/`--graded {hard,graded_read,graded_drive}`/`--l2-ld`/`--normalize`), `tests/test_emerge49_graded_read.py` (3 tests), `research/findings/raw/_emerge49_graded_read.json`. Prior: `2026-07-02-emerge48-soft-l2-pooling-BOUNDARY.md`, `2026-07-02-emerge47-l2-input-normalization-BOUNDARY.md`, `2026-07-02-emerge46-spiking-stacked-pooler-BOUNDARY.md`, `2026-07-02-emerge44-stacked-pooler-multilevel-taxonomy-GO.md`, `2026-07-02-anti-cheat-control-validity-methodology.md`.
