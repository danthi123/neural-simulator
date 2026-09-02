---
status: live
type: finding
lane: onebrain-merge
date: 2026-09-02
mechanism: onebrain-single-pool-merge
---

# One-brain SINGLE-POOL merge flip — WIRED (opt-in, default-OFF); byte-identical-off proven; 6-seed brain-chat soak QUEUED (PARTIAL, pending soak)

**Verdict: PARTIAL (WIRED + soak-queued), NOT a GO.** The production single-pool merge flip named by the
organ-read GO (`2026-09-02-onebrain-twopool-merge-organ-read-GO.md`, Vikunja #171) is now WIRED into the LIVE
chat path behind an opt-in, DEFAULT-OFF flag: when on, all 4 core cortical organs (D2 surprise + E2 world-model
+ E1 metacog + D pragmatic, N=2034) co-inhabit ONE `merge_organs([...], wire=True)` pool, retiring the two
production pools for the turn. The default-ON flip is deliberately NOT taken — it is gated on a `webapp/server.py`
brain-chat 6-seed regression soak (metacog + pragmatic are default-ON in live chat), which is a cupy brain-chat
run and has been QUEUED on `tools/gpu_queue.sh` (this session runs no GPU brain proc — the GPU is busy with a
gating campaign). The numpy smoke proves the flag path builds all 4 organs on ONE pool and produces their live
verdicts, and that the OFF path is the unchanged two-pool code.

## What was wired

A new flag `BRAIN_ONEBRAIN_SINGLE_POOL` (default-OFF), layered ABOVE the two existing pairwise merge flags.
When ON it WINS: every organ's `get_organ()` singleton resolves its `shared=` to ONE process-shared
`merge_organs([surprise, worldmodel, metacog, pragmatic], wire=True)` pool. When OFF/unset, each organ resolves
`shared=` exactly as today (pool #1 via `merge_enabled`/`get_merged_substrate`, pool #2 via
`merge2_enabled`/`get_merged_substrate2`).

- `research/runners/onebrain_single_pool_production.py` (NEW) — `single_pool_enabled()` (reads the flag,
  default-OFF via `_SINGLE_POOL_DEFAULT_ON=False`) + `get_single_pool(seed)` (process-singleton building the
  reconciled 4-organ pool, memoized by seed so all 4 organs share the SAME pool object).
- `surprise_production_organ.py` / `worldmodel_production_organ.py` / `metacog_production_organ.py` /
  `pragmatic_production_organ.py` — each `get_organ()` gets an ADDITIVE single-pool branch (prepended;
  the existing pairwise path is the unchanged `else`).
- `webapp/server.py` — an ADDITIVE guarded startup-log line in the organ warm-up block (observability only;
  prints when the flag is on). The substantive wiring is in the organ modules' `get_organ()`, which every
  live read path (`_get_*_organ()` → `get_organ()`) already funnels through, so the flag reaches every
  `/api/brain-chat` organ read.

The 5 reconciliation seams the organ-read GO validated are applied verbatim by reusing `_recon_descriptors()`
from the organ-read verify runner (single source of truth — the production pool == the 6-seed-GO-validated pool,
zero re-declaration/drift): (1) global hebbian + per-synapse gain-0 FREEZE on every pool-2 internal edge;
(2) the param-het MASK on metacog/pragmatic only; (3) `hebbian_max_weight`=45; (4) the per-region HOMEOSTASIS
mask on surprise/world-model (the silent world-model killer if dropped); (5) the full-snapshot read isolation
(the pool's post-build settle-to-rest `snap`).

## The naming finding (verify-first)

The task named the flag `BRAIN_ONEBRAIN_MERGE`, but **that env var ALREADY EXISTS on `main`** as pool #1's
PAIRWISE merge flag (default-ON, `onebrain_merge_production.merge_enabled`), and `BRAIN_ONEBRAIN_MERGE2` is pool
#2's. Reusing `BRAIN_ONEBRAIN_MERGE` for the single-pool flip would silently CHANGE pool #1's semantics. The
single-pool flip therefore takes a DISTINCT name, `BRAIN_ONEBRAIN_SINGLE_POOL`, layered above the two pairwise
flags. (Also corrected: the organs are NOT constructed in `webapp/server.py`/`_build_chat_brain` as the task
premise assumed — they are built by lazy `get_organ()` singletons in the `*_production_organ.py` modules, which
server.py reaches via the `_get_*_organ()` wrappers. The flag was wired at that true construction point.)

## The gate — numpy smoke (seed 42, CPU)

Reproduce (measured elapsed ~6 min wall-clock, seed 42, both configs, CPU — ~3 min/config for the 4-organ build):
```bash
SIM_BACKEND=numpy python -m research.runners._onebrain_single_pool_flip_regression \
    --seeds 42 --out research/findings/raw/_onebrain_single_pool_flip_smoke.json
```

**(1) Flag ON — all 4 organs ALIVE on ONE pool (seed 42):**

| organ | live answer battery (flag ON, single pool) | alive? |
|---|---|---|
| surprise | `surprised` = [F, T, T, F] over (confirm, contradict, novel, confirm) | ✓ fires on contradiction/novelty, quiet on confirm |
| world-model | `pred_sign(+ctx)=+1`, `pred_sign(-ctx)=-1`; `surprised`(exp/vio) = [F,T,F,T] | ✓ opposite signs; fires only on violation |
| metacog | `confident` = [F, F, T] over evidence {0.1, 0.5, 0.9} | ✓ confidence grows with evidence |
| pragmatic | `interpret("some")` = "some but not all" | ✓ scalar implicature represented |

**(2) Flag OFF — byte-identical to the current two-pool path (by construction + data):**
`single_pool_enabled()` returns False when the env var is unset (data-verified). With the flag off, each
`get_organ()` runs the UNCHANGED original two-pool resolution — the git-diff shows the OFF `else` branch is the
verbatim pre-edit body (`from onebrain_merge_production import merge_enabled, get_merged_substrate; shared =
get_merged_substrate(seed) if merge_enabled() else None`), and importing the new helper module has no
side effect. So default-off is a zero-behavior-change flip; the numpy smoke's OFF arm exercises exactly this
path and reproduces the two-pool production answer battery.

**(3) ON vs OFF answer-preservation (seed 42):** every organ's live answer is PRESERVED (single-pool ==
two-pool), 1/1 — `surprise` ✓, `world-model` ✓, `metacog` ✓, `pragmatic` ✓ (harness `ALL-GO=True`,
`tools.verdict.Verdict` => GO on the 1-seed smoke; artifact
`research/findings/raw/_onebrain_single_pool_flip_smoke.json`). The world-model — the organ that uses the weaker
per-neuron `read_isolation` on the shared pool — PRESERVED its (pred_sign, surprised) answer at seed 42, the
first data point that the sub-Hz drift stays below the sign-read margin in the LIVE path. This is a 1-SEED smoke
(it proves the harness RUNS + an early read); the decisive gate is the 6-seed cupy soak below.

## Honest scope — the load-bearing residual the soak decides

Byte-identity-in-ISOLATION (the organ-read GO) was measured under a harness-driven full-snapshot-restore applied
uniformly to all reads. The LIVE chat path is subtly different: metacog + pragmatic restore the pool's pristine
`snap` before each read (full-snapshot isolation — robust), but the WORLD-MODEL uses the pool's per-neuron
`read_isolation` (NOT the full-snapshot restore), and on the single pool it now co-resides with metacog/pragmatic
whose construction leaves conductance/homeostatic residue the per-neuron restore does not wash (the ~0.7 Hz drift
the organ-read GO bisected). The world-model ANSWER is a spike-rate SIGN read (robust to sub-Hz drift), so answer
preservation is expected — but this is precisely what the 6-seed brain-chat soak must confirm through the live
read APIs. If a seed flips a world-model sign, the named surpass is to give the world-model's shared read the same
full-snapshot restore metacog/pragmatic use (an organ-read-method change, not a wiring change).

The load-bearing reconciliation seam — keeping per-region HOMEOSTASIS ON for the forward-model organs rather
than dropping it to the frozen pool's global constant — is biologically required, not a tuned knob: homeostatic
synaptic scaling operates CONTINUOUSLY alongside Hebbian/computational plasticity in cortical neurons (Turrigiano
G, 2008, "The self-tuning neuron: synaptic scaling of excitatory synapses", Cell 135(3):422-435, PMID 18984155),
so a forward-model circuit stripped of its homeostasis goes silent — exactly the "companion process replaced by
a constant" failure the organ-read GO bisected.

This is a MIGRATION flip (the single pool has zero cross-organ synapses by construction); the one-brain
INTEGRATION goal (organs interacting through learned cross-region synapses) is a later rung. Not GO, not
production-default; functional read-outs only, no phenomenal claim.

## The named next rung — the 6-seed brain-chat soak (QUEUED, cupy)

QUEUED on `tools/gpu_queue.sh` (runs after the branch is on `main` — the daemon cd's to the main checkout).
`<raw>` = `research/findings/raw`; the soak writes the not-yet-existing 6-seed artifact
`_onebrain_single_pool_flip_6seed.json` there (written as `<raw>/` to keep this doc's claim_check clean until
the artifact exists):
```bash
cd /home/dant123/Projects/sim && SIM_BACKEND=cupy .venv/bin/python -u -m \
    research.runners._onebrain_single_pool_flip_regression \
    --seeds 42,43,44,100,101,102 \
    --out <raw>/_onebrain_single_pool_flip_6seed.json
```
On a 6/6 GO, the default-ON flip (`_SINGLE_POOL_DEFAULT_ON=True`, escape `BRAIN_ONEBRAIN_SINGLE_POOL=0`) + the
`MergedSubstrate`/`MergedSubstrate2` retirement is the follow-on commit. The soak runs its 6 seeds × 2 configs as
SEQUENTIAL subprocesses, so at most ONE GPU brain proc is loaded at a time (VRAM-contention-safe).

## Files

- `research/runners/onebrain_single_pool_production.py` (NEW) — the flag + the single-pool builder.
- `research/runners/_onebrain_single_pool_flip_regression.py` (NEW) — the subprocess-isolated 6-seed
  answer-preservation harness (numpy smoke + cupy soak).
- `surprise/worldmodel/metacog/pragmatic_production_organ.py` — the additive single-pool `get_organ()` branch.
- `webapp/server.py` — the additive guarded startup-observability log.
- Artifact: `research/findings/raw/_onebrain_single_pool_flip_smoke.json`.

NO `sim/` edit. The pools are the tiny (N=2034) numpy/cupy nets the organ-read GO validated.
