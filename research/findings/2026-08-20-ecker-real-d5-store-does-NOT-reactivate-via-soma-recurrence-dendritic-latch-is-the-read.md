---
type: finding
status: contributing
date: 2026-08-20
mechanism: swr-sequence-replay
lane: EPISODIC
seeds: [42, 43, 100]
seed-waiver: A verified NEGATIVE (honest-negative headline, exempt from the single-seed GO gate). It is nonetheless
  substantiated three ways beyond one point: 3 seeds (42/43/100) at the default op-point, a banked 24-op-point
  cue_pa x cue_frac sweep on a seed-42 store, and a matched-weight proxy corroboration — every draw and every
  op-point reads 0.000.
instrument: research/runners/_gap5_ecker_reactivates_REAL_d5_derisk.py — a GENUINE production D5 EpisodicDapMemory
  store ('dog' BTSP-encoded, 'cat' never-formed), its real membership + per-synapse BTSP-grown within-recurrence
  EXACT-COPIED onto an Ecker ADEX_ECKER_CA3_PC bridge, then partial-cued under the SWR envelope; held-out co-firing /
  discreteness / specificity / recurrence-lesion teeth, with a tools.verdict block + attributable_to.
runner: research/runners/_gap5_ecker_reactivates_REAL_d5_derisk.py
external: NO-EXTERNAL-NEEDED — the proxy->real transfer step of an in-repo integration whose feasibility was already
  scoped ([[2026-08-20-ecker-replay-into-D5-integration-FEASIBLE-by-composition-not-replacement]]); the residual it
  localizes (soma-recurrence attractor at ~15-cell sparsity) is D5's own documented small-assembly seam, not a
  literature question.
artifacts:
  - research/findings/raw/_ecker_reactivates_real_d5/seed42.json
  - research/findings/raw/_ecker_reactivates_real_d5/seed43.json
  - research/findings/raw/_ecker_reactivates_real_d5/seed100.json
  - research/findings/raw/_ecker_reactivates_real_d5/summary_3seed.json
  - research/findings/raw/_ecker_reactivates_real_d5/oppoint_sweep_s42.json
---
# A REAL D5-stored assembly does NOT reactivate via AdEx SOMA recurrence at its true scale — the dendritic-dAP latch stays the read

Artifact: research/findings/raw/_ecker_reactivates_real_d5/summary_3seed.json (3-seed aggregate) · the per-seed
research/findings/raw/_ecker_reactivates_real_d5/seed42.json + seed43.json + seed100.json (single default op-point) ·
research/findings/raw/_ecker_reactivates_real_d5/oppoint_sweep_s42.json (24-op-point cue_pa x cue_frac robustness).

**One line.** Step 1 of the banked D5-integration plan
([[2026-08-20-ecker-replay-into-D5-integration-FEASIBLE-by-composition-not-replacement]]) closes the proxy->real gap
with a **verified NO-GO**: a genuine production D5 `EpisodicDapMemory` 'dog' assembly, its REAL BTSP-grown
within-recurrence exact-copied onto an Ecker AdEx CA3 bridge, does **not** co-fire its held-out members under the SWR
replay — **0.000 across 3 seeds, 24 op-points, and a matched-weight proxy** — because a ~15-cell soma-recurrence
attractor at D5's true scale is too weak to complete. This is exactly why D5 reads via a per-cell dendritic-dAP latch,
not soma recurrence, and it **redirects the method** (per the no-defer law: the verdict is on the SOMA-recurrence
reactivation method, not on the learn-through-use capability).

## What was tested (the whole point: a REAL store, not the proxy)
<!--derived-->
The feasibility scoping used a STRONG-RECURRENCE PROXY (`build_store`, contiguous 20-cell assemblies, hand-set uniform
w=60). This runner instead builds a genuine D5 `EpisodicDapMemory` (n_ca3=2000), stores 'dog' the real way
(`mem.store('dog')` — its one-shot BTSP encode, a ~14-25-cell emergent DG-selected assembly with HETEROGENEOUS
BTSP-grown within-recurrence mean ~82-85, range ~49-100 vs the never-formed baseline ~1.5), leaves 'cat' never-formed,
EXTRACTS dog's real membership + per-synapse potentiated weights, and maps them onto an `ADEX_ECKER_CA3_PC` bridge as an
EXACT COPY (`copy_ok=True`; the on-bridge dog within-weight mean byte-matches the extracted BTSP mean to |Δ|~1.7e-6;
not a fresh `build_store`, not a uniform hand-set weight, not tuned to work). D5's own recall confirms the transfer
target is a real memory (dog `in_memory=True`, apical_cue 0.625; cat False).

## The verdict — NO-GO, robust (adversarially verified, no confound)
<!--derived-->
Single default op-point (cue_pa=9000, cue_frac=0.5), seeds 42/43/100: **held-out co-firing 0.000 = cat 0.000** every
seed (`summary_3seed.json` n_go=0). The held-out members never fire (`frac_held_ever=0.0`) — dog is indistinguishable
from the never-formed cat. The lesion teeth are vacuous (nothing to collapse). What DID transfer: DISCRETE
self-termination (`terminate_ratio=0`, cells spike-reset then rest silent — no dt-stiff pin), NOCUE-silence, and the
exact-copy machinery; `attributable_to` correctly returns **null** (both arms ~0 — no effect to attribute, not fabricated).

**Op-point robustness (banked):** `oppoint_sweep_s42.json` — 24 op-points (cue_pa 3000-150000 pA x cue_frac 0.3/0.5/0.7) on a
seed-42 store: **max dog co-firing 0.0000**, cat at the strongest op-point 0.0000, `robust_nogo=True`. No reachable cue
op-point rescues it.

**The decisive corroboration** (why the 0.000 is a real store property, not a dead instrument): the SAME byte-identical
`reactivate` instrument reads a POSITIVE completion when recurrence is strong enough (proxy s42_a20_w1500 -> 0.042,
s42_a80_w500 -> 0.055), and a correctly-wired proxy at the store's MATCHED weight (s42_a20_w100, w=100 ~ dog's formed
max) ALSO reads 0.000. So the real store's ~82-weight / ~15-cell soma recurrence sitting at zero completion is exactly
what a known-good instrument yields there. Four independent adversarial lenses (instrument-liveness, exact-copy-
faithfulness, verdict/metric-correctness, scale-vs-op-point) each returned CONFIRMED with no confound; all values were
reproduced from the artifacts.

## Honest scope + the residual it localizes (= the method redirect)
The residual is **D5's own documented seam**: a ~15-cell set with real ~85 weights is too small/weak for a recurrent
**soma** attractor — which is precisely why D5 uses a **per-cell dendritic-dAP latch** for its READ (the latch completes
at this scale; dog apical_cue=0.625), not soma recurrence. The proxy "positive" required asm=80 AND w=500 (both above
D5's real scale/weight), so it never transferred. Caveat, stated not hidden: `store('dog')` is not byte-reproducible
across worktree states (banked dog_size 14/15/25; prov git_dirty=true) — but every draw yields 0.000, which strengthens
(not weakens) the negative. **Redirect (buildable, sharply motivated):** do the READ with D5's dendritic-dAP latch
(which works at this scale) GATED by an AdEx-style spike-triggered-adaptation / SWR envelope as a SELF-TERMINATION so it
is a discrete transient, not a persistent latch — the composition [[dendritic-plateau-coincidence-burst]] x
swr-sequence-replay, step 2 of the plan. Using AdEx soma recurrence alone to carry co-firing at real scale is confirmed
insufficient. (Agent-built; parent verified the 3-seed + 24-op-point 0.000 + copy_ok + the 4-lens adversarial pass from
the artifacts, then banked the op-point sweep + corrected the sidecar provenance.)
