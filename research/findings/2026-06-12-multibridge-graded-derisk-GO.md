# Multi-bridge learned-graded-embedding cheap-first de-risk: GO — cross-bridge composition + the no-confab moat SURVIVE on correlated graded codes

**Date:** 2026-06-12. **Runner:** `research/runners/multibridge_graded_derisk.py` (built per `docs/plans/2026-06-11-multibridge-learned-embedding-derisk-design.md`). **Backend:** `SIM_BACKEND=cupy` (GPU, RTX 3090). **Raw:** `research/findings/raw/_multibridge_graded_derisk_full.json` (M1/M2/M4/M6 + first M3/M7) + `_multibridge_cross_fixedm7.json` (the corrected M3/M7). **Scope:** 3 bridges × 64 concepts (animals / foods / vehicles), 3 seeds (42/43/44).

> **Verdict: GO** (with a corrected anti-cheat). The load-bearing risk the design flagged — *does the existing cross-bridge composition layer, validated on ORTHOGONAL sparse codes, still work when each bridge carries a learned CORRELATED graded code; and does the no-confab moat survive?* — is **falsified: it works.** Every gate passes with valid anti-cheats, 3 seeds. ⇒ the multi-bridge route to large vocabulary (2,048 = 32 bridges × 64) is de-risked at the mechanism level (3 bridges exercise every cross-bridge code path; 32 is fan-out, tested in the build).

## Why this ran
The dual/CLS learned graded-embedding recipe is validated single-pool to V=320 (production tier, multi-seed GO), but the single-pool synapse count scales quadratically and OOMs by ~V=320–450 (`2026-06-11-V640-single-pool-memory-wall.md`). So large vocabulary must go MULTI-BRIDGE (the project's existing 320-tier method: 5 bridges × 64). The genuine open risk (design §4): the existing cross-bridge composition (the V-tag engram layer) was validated on ORTHOGONAL codes; the graded embedding produces CORRELATED within-bridge codes — a potential signal-to-noise hit for cross-bridge tag recall, and a moat risk.

## Results (3 bridges × 64 concepts, seeds 42/43/44)

| Gate | Result | Verdict |
|---|---|---|
| **M1** per-bridge within-bridge generalization (held-out-neighbour) + A2/A3 controls | gen 0.975–1.000 (≈4× chance) all 3 bridges × 3 seeds; orthogonal + permuted-property controls collapse | **GO** |
| **M2** per-bridge structure recovery (Pearson + 2nd-order margin) | Pearson +0.76+, 2nd-order +0.52+, graded=1, all 3 × 3; G5 (robust margin-gated) collapses | **GO** |
| **M3** cross-bridge composition recall (target retrieved over the other 63 concepts) | top-2 = 1.00, **signal/floor 16.85× / 17.75× / 23.82×** (seeds 43/42/44) | **GO** |
| **M7** cross-bridge anti-cheat (corrected: does a cue retrieve a WRONG target?) | top-2 = 0.00–0.08, **signal/floor 1.13× / 1.22× / 1.19×** (≈ chance) → **COLLAPSES** all 3 | **anti-cheat valid** |
| **M4** no-confab moat (familiarity gate alongside host abstention) | agreement 1.000, **zero false-accepts, zero host-abstain/gate-accept breaches**, lesion collapses, all 3 | **GO (moat intact)** |
| **M6** random-shard control (within-bridge generalization must collapse) | gen 0.31–0.36 (≈ chance 0.25) → collapses, all 3 | **anti-cheat valid** |

**⇒ every gate passes with valid anti-cheats.** Per-bridge synapses ~3.46M (≈10.4M co-resident) — 8.5× below the single-pool OOM wall.

## The anti-cheat fix (honest — the initial verdict was BOUNDARY on a broken control)
The first full run printed `COMBINED VERDICT: BOUNDARY` because M7 did not collapse. On inspection, **M7 was mis-implemented**: it permuted the cue→target mapping then STORED *and* SCORED that same permuted mapping on the shared bridges → it trivially recalled what it stored → **could not collapse by construction** (the same class of flaw as the brittle G5 boolean in the homeostasis run). This was NOT evidence M3 was an artifact — M3's 17–24× signal/floor specificity and M4's zero-breach abstention were independent strong evidence the link is real. The fix (commit `aa92ae8a`): keep the stored mapping TRUE; the M7 control scores whether a cue retrieves a *random WRONG* target in the same bridge — a genuine link retrieves the true target, so a wrong target ranks ~median and the control collapses. Tiny-CPU smoke confirmed (M3 top2=1.00/6.45×; corrected M7 top2=0.00/0.84×), then the 3-seed GPU re-run confirmed at scale (above). **No GO was claimed on the broken control** — the fix + re-run produced the valid verdict.

## Honest scope + what's NOT yet tested
- **3 bridges, not 32.** Per the design, 3 bridges exercise every cross-bridge code path (routing, partial-pair encode in two bridges, tag-name aggregation, abstention); 32 bridges is *more fan-out* (the §7 scaling risk: does cross-bridge SNR + the moat hold at 6.4× the validated 5-bridge fan-out), which is tested in the build, not the de-risk.
- **The conversational matrix subset (M5) is NOT in the de-risk runner** — the full who/what/negation/clause matrix on the multi-bridge ensemble is the build's integration validation (it needs the parser/composer/dialogue wired across bridges), downstream of this mechanism de-risk.
- **Sharding is curated** (animals/foods/vehicles) for the cheap-first run — the production sharding (co-occurrence-graph semantic clustering) is a build-time design choice; the within-bridge graded generalization being meaningful *requires* similar concepts co-located (M6 confirms random sharding collapses it).

## Conclusion + next
The mechanism-level risk is retired: **cross-bridge composition and the no-confab moat survive on correlated graded codes.** The multi-bridge route to 2,048 concepts is de-risked cheap-first. The next step is the BUILD (32 bridges × 64, full integration + the conversational matrix at multi-bridge fan-out) — the **owner's explicit-go gate** for the ~2–4 week push (build plan piece iii). No `sim/` edits anywhere in the de-risk. No banking — the honest BOUNDARY (broken control) was diagnosed + fixed + re-run before the GO.
