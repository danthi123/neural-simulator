---
type: finding
status: live
date: 2026-08-13
mechanism: one-brain-merge
integration_faculty: onebrain-merge-organs
---

# One-substrate PRODUCTION-DEFAULT flip (SCOPED): the D2 SURPRISE + E2 WORLD-MODEL organs now build on ONE shared spiking bridge BY DEFAULT — wired + on-by-default, answer-preserving vs the pre-flip production; the other 3 proven organs stay separate on a GENUINE global-config conflict

**Date:** 2026-08-13 · **Source flip:** `research/runners/onebrain_merge_production.py` `merge_enabled()` default
(`_MERGE_DEFAULT_ON=True`; `BRAIN_ONEBRAIN_MERGE=0` is the escape) · **Verify:**
`research/runners/_onebrain_production_flip_verify.py` · **Artifact:**
`research/findings/raw/_onebrain_production_flip_6seed.json` (6 seeds 42/43/44/100/101/102, `SIM_BACKEND=numpy`) ·
**Ledger:** row `onebrain-merge-organs` (wired=YES / on_by_default=YES / scaffold_retired=PARTIAL).

## What this lands

The one-substrate merge was de-risked to a production-read GO for 5 organs behind default-OFF flags (rung-1
surprise+world-model; per-region param-het metacog+pragmatic; per-region OU affect). This FLIPS the production
default so the two ORGANS THAT CAN SHARE ONE POOL BYTE-IDENTICALLY — the D2 surprise expectation-violation organ
and the E2 affective world-model organ — build on ONE shared `SimulationBridge` (one `cp_membrane_potential_v`,
N=1584) BY DEFAULT via the rung-1 `MergedSubstrate`. Both organs' `get_organ` inject the process-shared substrate
when `merge_enabled()`, and the `/api/brain-chat` handler reads them through those same singletons
(`_get_surprise_organ()` / `_get_worldmodel_organ()`), so the live chat now runs the pair on ONE pool by default.
`BRAIN_ONEBRAIN_MERGE=0` reverts to two separate bridges (byte-identical to the pre-flip production).

**This is a SCOPED flip, not a 5-organ flip — and the scope is a MEASURED, load-bearing constraint, not a
shortcut** (see "the residual" below): the 5 proven organs span TWO incompatible global configs, so they cannot
all share ONE pool byte-identically.

## Result — FLIP-GO 6/6 (the pair)

Every read is through the REAL production organ APIs the handler calls: surprise `judge()`, world-model
`expectation()` + `read_surprise()`. Three build variants per seed — TODAY (`shared=None`, == the pre-flip
separate-bridge production == the escape), MERGED (both organs on ONE shared bridge, the default-ON path),
CORESIDENT (each organ on its own bridge with the two merge flags ON — the apples-to-apples merge baseline).

| axis (6 seeds, broad panel: surprise confirm/violation ×4 concepts, world-model expectation ± + surprise expected/violated) | result | verdict |
|---|---|---|
| A. ONE shared pool (surprise.bridge IS worldmodel.bridge IS the substrate bridge; one `cp_membrane_potential_v`, N=1584) | 6/6 | GO |
| B. MERGED == CORESIDENT byte-identical (surprise + world-model read deltas 0.0 — the genuine merge byte-identity, rung-1 through the full read APIs) | 6/6 | GO |
| C. answer classes preserved vs TODAY (every surprise `surprised` bool + every world-model `pred_sign` identical MERGED-vs-pre-flip) | 6/6 | GO |
| **FLIP-GO (A + B + C)** | **6/6** | **GO** |
| D. numeric residual vs TODAY (reported, NOT gated): surprise Hz ≤ 1.16, world-model margin ≤ 129.2 — no class crosses a threshold | — | documented |

- Default-path check (no env var): `merge_enabled()` → True; `_get_surprise_organ()` and `_get_worldmodel_organ()`
  share ONE `cp_membrane_potential_v` object (N=1584). Escape (`BRAIN_ONEBRAIN_MERGE=0`) → `merge_enabled()` False →
  separate bridges.

## The honest residual — why "byte-identical to today" is IMPOSSIBLE for a genuine shared pool, and why 2 of 5 (not 5 of 5)

**Not byte-identical vs today — answer-preserving.** The merge REQUIRES `per_region_threshold_heterogeneity`
(so each organ's per-neuron init is position-invariant on the shared pool). That flag re-draws the firing
thresholds name-keyed instead of from the global-RNG order, so the merged reads are byte-identical to the
CO-RESIDENT-WITH-FLAGS baseline (check B, 0.0 in data) but NOT numerically byte-identical to the pre-flip
flag-off reads (check D: surprise ≤1.16 Hz, world-model margin ≤129.2). This is INHERENT to any genuine shared
pool: one shared global RNG cannot reproduce BOTH organs' standalone threshold draws (they depend on standalone
pool size + draw order). What IS preserved is the answer: because each build calibrates its own confirm/violation
threshold to its own rate distribution, the read is self-normalizing — every `surprised` bool and `pred_sign` is
identical vs today across the panel + 6 seeds (check C), so the user-visible chat answer does not change (like the
`one-brain-substrate` row, this is a SUBSTRATE-consolidation claim, not an answer change).

**2 of 5 — a genuine global-config conflict.** The other three proven organs cannot join THIS pool
byte-identically: metacog (`build_metacog_bridge`) and pragmatic (`build_rsa_bridge`) both set
`enable_parameter_heterogeneity=True`; affect (`build_one_brain`) sets `enable_parameter_heterogeneity=True` AND
`enable_ou_process=True`. Surprise + world-model were proven byte-identical with `enable_parameter_heterogeneity=
False`; a SINGLE global config cannot be both False (for the surprise/world-model byte-identity) and True (for the
graded rate codes of the other three). So all 5 on ONE pool is not byte-identical. The named next rung is a SECOND
param-het-ON shared pool for metacog+pragmatic+affect, which additionally needs the `per_region_wiring_seed` ORDER
seam exercised end-to-end (proven at the substrate level, `2026-08-13-per-region-ou-wiring-affect-GO.md`, but not
yet in a two-fully-wired-organ production merge). Per the flip gate's rule — "flip the subset that's clean, keep
the rest co-resident, map the residual honestly" — those three stay on their own bridges (unchanged).

## No regression (flag OFF = the pre-flip production; flag ON = answer-preserving)

- `BRAIN_ONEBRAIN_MERGE=0` → `merge_enabled()` False → each organ builds its own bridge with `shared=None` — the
  pre-flip separate-bridge path, byte-identical.
- `pytest tests/test_determinism.py -q` → **9 passed** (with the flip ON by default).
- `brain_chat_tui --smoke` (tiny-demo, stub renderer) is **byte-identical** post-flip vs a pre-flip baseline (the
  smoke exercises the ChatBrain TUI path, which does not invoke the server-only surprise/world-model organs — so
  the flip is a no-op there, verified by an exact JSON compare).

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._onebrain_production_flip_verify \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_production_flip_6seed.json
```

## Honest scope / non-claims

- `wired: the surprise + world-model production organs on ONE shared substrate / on_by_default: YES (default-merged;
  BRAIN_ONEBRAIN_MERGE=0 is the escape) / scaffold_retired: PARTIAL — the separate-bridge co-residency for THIS PAIR
  is retired, but (a) only 2 of the 5 proven organs (metacog/pragmatic/affect stay separate on a global-config
  conflict), and (b) the pair shares one pool with EACH OTHER, not yet with the recall composer bridge
  (one-brain-substrate).` Functional read-outs only; no phenomenal claim.
- The flip is **answer-preserving, NOT numerically byte-identical to the pre-flip reads** — the numeric Hz/margin
  debug fields shift (the inherent, characterized cost of a genuine shared pool); no classification crosses a
  threshold on the tested panel + 6 seeds. This is an empirical (not proven) preservation guarantee.
- No cross-organ synapse is added; the load-bearing claim is one shared pool + byte-identity to the merge baseline
  + answer-preservation. A genuine cross-region synapse and merging onto the RECALL composer bridge are later rungs.
