---
type: plan
status: live
date: 2026-05-25
---

# Direction 3 V=32 PRODUCTION — Adversarial Reviewer Prompt (pre-staged 2026-05-25)

Status when this was written: Direction 3 V=32 PRODUCTION DECISIVE multi-seed run IN FLIGHT on GPU (PID 36700; ETA ~5-6 hr; watcher bl0wjskjb monitors). Smoke = PASS multi-seed. Pillar n=105 candidate **IF AND ONLY IF** an independent fresh adversarial reviewer returns CLEAR using this prompt.

This document is the pre-registered, pre-staged prompt the reviewer must follow when the production run completes. The reviewer must be a fresh agent (no shared session history with this autonomous arc), must reproduce the verdict from raw data, and must verbatim record CLEAR or BLOCK at the end.

---

## Inputs to the reviewer

- **Result file (production)**: `research/findings/raw/direction_3_v32_production.json`
- **Log file (production)**: `research/findings/raw/direction_3_v32_production.log`
- **Smoke result (already PASS, for sanity comparison)**: `research/findings/raw/direction_3_v32_smoke.json`
- **Smoke cache (per-seed activity + bridges)**: `research/findings/raw/direction_3_v32_cache/`
- **Runner module**: `research/findings/raw/direction_3_v32_runner.py`
- **Bridge builder (V=32 wrapper)**: `research/findings/raw/direction_3_bridge_builder.py`
- **Vocab spec (V=32)**: `research/findings/raw/direction_3_vocab_spec.py`
- **Frozen verdict module**: `research/findings/raw/direction_3_verdict.py`
- **Design doc**: `docs/plans/2026-05-25-direction-3-vocab-scaling-bio_brain_regions-design.md`
- **Smoke commit pinning the verdict**: `9a09576`
- **Headline reference (pillar n=94/n=95)**: `webapp/capability_status.json` (the multi-seed-validated mode-unification + cross-bridge headline)
- **V=16 pillars context**: n=96/n=97/n=98 (mode-unification at V=16 with parallel-matching decoder)
- **Load-ceiling map V=16 reference**: cross-bridge OB exactly 1.000 every cell across 3600 trials; OI 1.000/1.000/0.790 (global_mean) and 1.000/0.998/0.785 (per_bridge_mean) at loads {2,3,5} — the boundary localised at high load x high vocab x order-invariant.

---

## What the reviewer must scrutinise per pillar candidacy

The reviewer must check ALL of the following. Any single FAIL on items 1-7 returns BLOCK. CLEAR requires all 7 to PASS.

### 1. Multi-seed reproducibility (3/3 seeds at every cell)

- Open `direction_3_v32_production.json`. Confirm `per_seed` contains exactly **3 entries** for seeds 42, 43, 44.
- For EACH seed, confirm `per_load` contains entries for loads **{2, 3, 5}** with `n_trials` matching the production trial count (NOT the 50-trial smoke).
- Confirm EVERY cell at EVERY seed has `order_bearing_accuracy >= 0.80` AND `order_invariant_accuracy >= 0.80` (the frozen 0.80 bar; do NOT tune).
- Confirm the verdict entry per seed matches: at every load `OB >= 0.80` AND `OI >= 0.80` triggers PASS; otherwise BOUNDARY or NEGATIVE per the frozen verdict module's branches.
- Reject any cell that hits "PASS" via a tied threshold (e.g. exactly 0.7999... rounding to 0.80 — the bar is `>= 0.80` so exact 0.80 is allowed, but anything below must NOT be relabelled).

### 2. Smell-test recomputation from raw per-seed data

- Independently recompute the per-seed PASS verdict from the JSON's `per_load[*][order_bearing_accuracy|order_invariant_accuracy]` values WITHOUT running the verdict module.
- Confirm your recompute matches the JSON's `verdict_entry` and the runner's printed verdict line VERBATIM.
- If there is any discrepancy >= 0.001 in any cell or any divergence in PASS/BOUNDARY/NEGATIVE label, return BLOCK.

### 3. Comparison to V=16 pillars n=96/n=97/n=98 (does V=32 add genuine new info?)

- The V=16 pillars established the parallel-matching decoder cleared the bar with cross-bridge characterisation including the OI ceiling at high load. V=32 doubles the vocabulary.
- Confirm: V=32 multi-seed PASS is a GENUINE scaling result, not a re-derivation of V=16 numbers under a different name. Specifically:
  - `V` field in the JSON must equal **32**.
  - `substrate` must reference `bio_brain_regions_v14v16_recipe_V32` (or the production-doc-specified label).
  - `d_act` should reflect a 32-word substrate (smoke had `d_act=3200`; production may have a different value but should be consistent across all 3 seeds).
- Confirm the V=32 result is NOT subset-of-V=16 by checking the vocab spec module exposes 32 unique concept identifiers, and that the OB/OI distractor pool reflects 32-symbol selection (not 16).
- If `V` is NOT 32, OR if the substrate label is missing/mismatched, OR if the distractor pool collapses to a 16-symbol subset, return BLOCK.

### 4. Verify the frozen verdict module's output matches what `compute_verdict` produces from the JSON

- Independently import `research.findings.raw.direction_3_verdict` (or the module path the production runner uses).
- Pass the production JSON's per-seed data into `compute_verdict(...)` and confirm the returned verdict label and per-cell PASS/FAIL marks match the JSON's recorded `verdict_entry` for EVERY seed.
- If the frozen module is not callable on the production JSON (schema mismatch, missing fields, type errors), return BLOCK.
- If the module's output disagrees with the JSON's recorded verdict in ANY cell, return BLOCK (this would indicate the runner tampered with the verdict).

### 5. Anti-cheat: confirm the parallel-matching primitive is reused byte-unchanged

- The pillar n=93/n=94/n=95 parallel-matching primitive is the load-bearing biologized identification mechanism. V=32 must REUSE it byte-unchanged (not re-derive a permissive variant).
- Locate the primitive in the runner (likely an `import` of the validated mode-unification module).
- Confirm the import is from the protected/frozen location (NOT a copy-pasted local re-implementation).
- Run `git diff <pillar-commit>..HEAD -- <primitive-module-path>` and confirm the diff is EMPTY for the primitive module (or any related core mode-unification helper).
- If there is any non-empty diff on the parallel-matching primitive, return BLOCK.

### 6. Check for any score-tuning or threshold-tampering

- Confirm `bar_ob` and `bar_oi` in the JSON are BOTH exactly **0.80** (the frozen bar). Reject any value other than 0.80.
- Confirm `min_seeds` is **3** (the documented min for multi-seed reproducibility).
- Confirm seeds are exactly **[42, 43, 44]** (the canonical reproducibility set; not cherry-picked).
- Search the runner module + verdict module for any post-hoc threshold adjustment, any `if accuracy < X: accuracy = X + epsilon` pattern, any `np.clip(...)`-style score adjustment in the OB/OI computation path. If found, return BLOCK.
- Search the result JSON for any commented-out/extra fields suggesting a permissive verdict was attempted before the final one (`_attempt_1`, `_softer_verdict`, etc.). If found, return BLOCK.

### 7. Verify the load-ceiling map V=16 reference is still applicable

- The V=16 OI ceiling is 1.000/1.000/0.790 at loads {2,3,5} cross-bridge; the per-bridge OI at V=16 is uniformly 1.000.
- V=32 single-bridge OI at loads {2,3,5} should ideally either (a) match the V=16 per-bridge pattern (1.000 uniform), demonstrating the substrate scales the OI capability cleanly, OR (b) localize a new boundary specific to V=32 high-load.
- If V=32 OI degrades below 0.80 at any load, the cell is BOUNDARY (not PASS) — confirm the verdict module recorded BOUNDARY in that case (not PASS via rounding/tuning).
- If V=32 OI at L=5 collapses to a value LESS than the V=16 cross-bridge ceiling 0.79, flag as a regression worth surfacing in the verdict text (still PASS if all cells >= 0.80; FLAG if any cell triggers a regression-class warning).

---

## Additional structural checks (informational; do not block on these alone)

- Wall clock printed in the log matches the documented ETA (~5-6 hr); if substantially shorter (< 3 hr) the run may have used reduced trials/seeds — verify.
- Per-seed bridge cache files exist for production seeds (not just smoke).
- `batched_vs_scalar_max_diff` field, if present, should be near machine precision (< 1e-15) — confirms the GPU-batched primitive equivalence held under V=32.

---

## Reviewer verdict (pre-registered tags)

The reviewer must end their report with EXACTLY ONE of:

- **CLEAR** — All 7 scrutiny items PASS. The V=32 multi-seed result is a genuine biology-grounded scaling demonstration on the build_biological_brain_regions substrate. Pillar n=105 candidate is APPROVED for promotion.
- **BLOCK** — One or more scrutiny items FAILED. Specify which item(s), describe the strengthening fix needed (e.g. "re-run production with corrected verdict module after fixing X" or "patch the runner to reuse the primitive byte-unchanged"). Pillar n=105 candidate is REJECTED until BLOCK is resolved.

If BLOCK, the autonomous arc MUST NOT promote a pillar; instead the BLOCK reason must be documented, the strengthening fix scoped, and the next D3 V=32 production attempt must address the BLOCK before re-submission.

---

## Why this prompt exists pre-staged

This is pre-staged so that when the production run completes (~5-6 hr from the smoke completion timestamp), the reviewer's criteria are FROZEN before the production result is observed. No reviewer-induced post-hoc adjustment to the bar, the scrutiny list, or the verdict tags is possible — the prompt was committed before the production JSON existed.

This mirrors the project's standing "frozen-bar / pre-registered criteria" discipline: the bar and the reviewer's blocking criteria must be set BEFORE the result is in hand, otherwise the verdict is post-hoc rationalisation, not science.

---

## Cross-reference

- Smoke commit pinning V=32 architecture + smoke PASS: `9a09576`
- Implementation plan: `docs/plans/2026-05-25-direction-3-vocab-scaling-bio_brain_regions-design.md`
- Production launch context: `research/findings/AUTONOMOUS_STATE.md` (pre-registered post-verdict chain)
- Sibling Direction 4 cross-bridge work: `docs/plans/2026-05-25-direction-4-cross-bridge-bio_brain_regions-design.md` + `docs/plans/2026-05-25-direction-4-cross-bridge-bio_brain_regions-implementation.md`
