---
type: finding
status: live
date: 2026-08-13
mechanism: one-brain-merge
---

# One-brain 2-organ MERGE: the SECOND byte-identity cause (homeostatic threshold adaptation) is CLOSED — pool co-residence is byte-EXACT end-to-end with the faculty ALIVE, via per-region homeostasis isolation

**Date:** 2026-08-13 · **Runner:** `research/runners/_one_brain_merge_2organ_derisk.py` · **Artifact:**
`research/findings/raw/_one_brain_merge_2organ_homeo_iso_6seed.json` (6 seeds 42/43/44/100/101/102,
`SIM_BACKEND=numpy`). **`sim/` EDIT (owner green-lit): additive + guarded + byte-identical-when-off** —
a new opt-in config flag `cfg.per_region_homeostasis_isolation` (default `False`). Continues
`2026-08-13-one-brain-merge-CLOSED-per-region-threshold.md`, which closed the FIRST cause (init RNG,
`cfg.per_region_threshold_heterogeneity`) and root-caused — but did NOT close — this SECOND cause.

## The question this arc answered

The prior finding closed INIT byte-identity (0/6 → 6/6) but the fully-adapted TRAINED read stayed 0/6.
It attributed the residual to **homeostatic threshold adaptation** and (imprecisely) to "the WHOLE pool's
activity history" plus a "shared-numerical-context floating-point effect". This arc **measured the actual
mechanism**, then closed it.

## What the residual ACTUALLY is (measured, not assumed)

**It is NOT pooled activity, and NOT a floating-point-order effect. It is a deterministic shared-CLOCK
homeostatic idle-drift.**

- `fused_homeostasis_update` (`sim/kernels.py:1314`) is **strictly per-neuron**: each neuron's threshold
  adapts only from its OWN activity EMA and its OWN spike, with global scalar params. There is **no
  reduction across neurons** — nothing "pools the whole pool's activity". An idle organ's activity EMA
  stays **exactly 0.0** (measured).
- Homeostasis is a **CONTINUOUS** process: with EMA = 0 the rate error is `0 − target < 0`, so a **silent**
  neuron's threshold is pulled DOWN every step. On ONE shared, continuously-stepped substrate, while organ A
  trains, organ B's neurons are stepped too and **idle-drift** — an evolution the SEPARATE standalone
  organ-B bridge never undergoes (it simply does not run during A's training). **That is the entire
  residual.**

Decisive controls (seed 42, `SIM_BACKEND=numpy`; artifact
`research/findings/raw/_one_brain_merge_homeo_mechanism_s42.json`):

| control | result | reading |
|---|---|---|
| organ-B threshold drift while IDLE during organ-A training | **all 1056 thresholds drift ~0.08 mV; EMA stays 0.0** | per-neuron, not pooled; the divergence seed |
| merged organ-B build thresholds vs standalone build | **max err 0.0** | the init (first-cause) fix holds |
| **REVERSE training order (B first, then A)** | **recall(B) 9.98e-1 → 1.23e-1 Hz; surprise(A) 5.79e-2 → 3.125 Hz** | the DIVERGENT organ FLIPS — decisive proof of idle-drift (the organ that idles BEFORE its own training diverges) |

## The fix (the `sim/` edit)

`sim/config.py`: `per_region_homeostasis_isolation: bool = False`. `sim/bridge.py` (in the Izhikevich
homeostasis update, `~:10523`): a NEW block — **skipped entirely when the flag is off** — GATES the
homeostatic threshold + activity-EMA update to neurons that **PARTICIPATED this step** (`fired_this_step`
OR nonzero `cp_external_input_current`), combined (AND) with any existing static homeostasis-update mask.
Idle co-resident neurons are FROZEN, so a region's homeostatic state is invariant to how long it co-resides
idle beside a training co-resident. During-training / during-read adaptation (driven or firing neurons) is
untouched — the operating point the faculty depends on is preserved.

**Why default-off is byte-identical (verified in data):** flag ABSENT vs present-but-`False` hashes
IDENTICALLY on `cp_membrane_potential_v`, `cp_recovery_variable_u`, `cp_connections`,
`cp_neuron_firing_thresholds` — with homeostasis ON and OFF, after 200 stepped iterations that actually
run the homeostasis path. `tests/test_determinism.py` passes 9/9. `brain_chat_tui --smoke` is unchanged
(the flag is never set → the block is unreached).

## Result — the SECOND cause CLOSES 6/6; pool co-residence is byte-EXACT with the faculty ALIVE

**6 seeds (42/43/44/100/101/102), `per_region_homeostasis_isolation` ON, homeostasis STILL ON** (the
production operating point):

| axis (isolation ON) | 6-seed result | verdict |
|---|---|---|
| SURPRISE organ (A) read byte-identity | **6/6 EXACT** (max err 0.000e+00 Hz) | **GO** |
| RECALL organ (B), read BEFORE the cross fires (POOL CO-RESIDENCE) | **6/6 EXACT** (max err 0.000e+00 Hz) | **GO** |
| faculty ALIVE (contradict/confirm separation ≥ 5×) | **6/6** (separations 7.8×–49.5×) | **GO** |
| HOMEO-ISO merge verdict (pool byte-exact AND alive) | **6/6** | **GO** |
| STRUCTURAL merge (pool + determinism + load-bearing + init + homeo-off + iso) | **6/6** | **GO** |
| RECALL organ (B), read AFTER organ A's surprise read | max ≈ 1.88e-1 Hz | cross footprint (see below) |

For contrast, with isolation OFF (homeostasis ON) the fully-adapted read is 0/6 (residual = idle-drift,
max surprise 3.59e0 Hz, max recall 1.63e0 Hz). Representative seed 42, ISO OFF → ON: surprise
5.79e-2 → **0.000e+00 Hz**; recall-before-cross → **0.000e+00 Hz**; recall-after-cross 9.98e-1 → 9.40e-2 Hz;
separation 49.9× → 49.5×.

**The read-order control is the clincher** (seed 42, ISO ON): `read_recall` drives only cueB (organ A
undriven) and there is no B→A synapse, so reading recall FIRST leaves organ A's surprise read untouched.

- **recall BEFORE organ-A read:** merged vs solo → **max err 0.000e+00 Hz** → the MERGE
  itself (init + trained + homeostatically-adapted) is **byte-clean** (`homeo_iso_recall_before_maxerr_hz`).
- **recall AFTER organ-A read:** merged vs solo → max err `9.40e-2 Hz` (≈ 1 spike on
  1 near-threshold fact; `homeo_iso_recall_after_maxerr_hz`). This residual appears **only after** organ A's contradict/novel read, in which
  surprise_A fires and the **LOAD-BEARING cross synapse** `surprise_A→cueB` drives cueB — nudging cueB's
  threshold through the continuous homeostatic process. This is **the cross synapse DOING ITS JOB**
  (novelty gating recall), not a co-residence defect; byte-identity of an "isolated" read cannot be blind
  to a load-bearing synapse that drove the read region moments earlier. In the standalone there is no
  organ A to be surprised, so cueB never receives that nudge.

**VERDICT: GO (6/6).** With `per_region_homeostasis_isolation` ON, ONE shared spiking substrate holding two
distinct organs is byte-identical to the standalone organs — INIT, TRAINED, and homeostatically ADAPTED —
for pool co-residence, on all 6 seeds, while the surprise faculty stays alive (separations 7.8×–49.5×). The
homeostatic idle-drift SECOND cause is CLOSED. The remaining ~1–2-spike recall delta (max ≈ 1.88e-1 Hz) is
the cross synapse's own load-bearing homeostatic footprint, characterized by the read-order control — not a
merge artifact and not floating-point.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._one_brain_merge_2organ_derisk \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_one_brain_merge_2organ_homeo_iso_6seed.json
```

The runner reports, per seed: INIT byte-id · HOMEO-OFF byte-id · PROD (homeo-on, NO isolation) ·
**HOMEO-ISO** (surprise EXACT · POOL CO-RESIDENCE byte-id = surprise + recall-before-cross · recall-after-
cross residual · faculty separation) · the cross load-bearing lesion · determinism.

## Honest scope / non-claims

- **CLOSED = the homeostatic idle-drift SECOND cause** — pool co-residence is byte-EXACT end-to-end
  (surprise organ exact; recall organ exact when read before the load-bearing cross fires), faculty alive.
  Combined with the first-cause init fix, the 2-organ merge is byte-identical to the standalone organs for
  everything the merge itself controls.
- **NOT claimed:** that the recall read is byte-identical AFTER the cross synapse has fired into it. That
  ~1-spike delta is the cross synapse's LOAD-BEARING action (a genuine functional interaction), proven by
  the read-order control; it is a property of the faculty, not a determinism boundary. It is **not** a
  floating-point-order effect (the prior finding's "shared-numerical-context FP" wording described a
  broken naive-isolation attempt, not this residual).
- **`per_region_homeostasis_isolation` changes homeostasis semantics when ON** (idle neurons do not drift
  toward target). This is an OPT-IN correctness flag for the one-substrate merge seam, default-OFF and
  byte-identical to legacy; it is not a claim about biological intrinsic-plasticity dynamics in general.
- **Not integrated into production.** This is a substrate-correctness fix behind an opt-in flag; no
  production organ set is merged onto one pool by default yet. "One substrate" for production organs
  remains CO-RESIDENCY.
- **Two INSTANCES of one builder** (surprise + recall). Merging DIFFERENT builders (config superset,
  2→N scaling) is the named next step, not attempted here.
- **Functional read-outs only**; no phenomenal claim.
