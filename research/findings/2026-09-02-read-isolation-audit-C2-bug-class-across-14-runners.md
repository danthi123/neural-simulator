---
type: finding
status: live
date: 2026-09-02
mechanism: read-isolation-audit-c2-bug-class
board: one-brain integration / measurement-integrity
artifact: research/findings/raw/_read_isolation_audit/audit_14runners.json
---

# Read-isolation audit: the C2 reset-gap is in all 14 audited runners, but changed a verdict in 5 — incl. 1 live production over-claim + a "wall" that was an artifact

**2026-09-02.** After the C2 metacog fix revealed a read-isolation bug (a pool `_hard_reset` that restores
membrane/recovery/conductances/firing/Hebbian-trace but NOT `cp_refractory_timers`, `cp_prev_firing_states`,
`cp_neuron_activity_ema`, `cp_neuron_firing_thresholds` — leaking hard-gate + homeostatic state across reads,
inflating the lesion baseline and biasing vary/lesion verdicts toward NO-GO/non-attributable), a 14-runner
read-only audit workflow (mechanism-zeroed two-read diagnostic per runner) mapped where else it bites.
Artifacts: `research/findings/raw/_read_isolation_audit/audit_14runners.json` (per-runner) +
`research/findings/raw/_read_isolation_audit/synthesis.md` (ranked report).

## Result: bug class present in ALL 14; verdict CHANGED in 5

**Why it bites some and not others:** the leak is `≤`2-step refractory + prev-firing + homeostatic residue. A read
protocol with a forced-spike warmup (`LOAD_STEPS=30` at 400 pA before the scored window) makes every neuron spike →
Izhikevich hard-reset-on-spike erases the residue before recording → **clean** (R1, R2, etc.). C2's `read_confidence`
and the leakers below have no such warmup → the residue survives into the scored read.

### 🔴 Integrity: 1 INFLATED GO that is LIVE in production
<!--derived-->
- **`_onebrain_crossedge_curiosity_to_d6wm` (curiosity→d6-WM)** — GO 6/6, wired **default-ON** in `/api/brain-chat`.
  Running the REAL pipeline with a corrected reset flips **seed 43 GO→NO-GO** (`delta_intact` 0.0105→0.007375, below
  the 0.008 floor; the leak was ~42% of that seed's effect) → the banked "GO 6/6" is **NO-GO 5/6** once reads are
  isolated <!--derived-->. A live default-on faculty resting partly on the artifact. Highest priority to fix +
  re-verify + re-decide the flip.

### ⭐ 4 FALSE WALLS (leak suppressed a real result)
- **`_spiking_expectation_rpe` (gain 0.4)** — a 3/6 BOUNDARY the finding narrated as a "precision/homeostatic-companion
  wall"; isolated reads flip seeds 100 + 102 FAIL→PASS → ~5/6. A **narrated biological wall demoted to a measurement
  artifact** (the "what did we replace with a constant?" payoff — here the constant is the unreset homeostatic
  threshold; this runner defaults `enable_homeostasis=True`, so all 4 arrays leak). <!--derived-->
- **`_onebrain_integration_surprise_episodic_crossedge`** — UNDEFINED (F2 lesion-control fails 5/6); the leak
  magnitude is the same order as the `delta_lesion` it corrupts → recoverable. On the spine.
- **`r3v2` / `r3` dopamine-credit** — NO-GO on F2 (best seed misses the floor by 0.00198 <!--derived-->); leak confound comparable
  magnitude, sign unpinned → plausible flip. Already superseded by **r3v3 GO 6/6 (live)** — so integrity + confirming
  r3v3 survives isolation, not an unblocked capability.

### 9 ROBUST (verdict stands)
R1, R2, surprise→episodic-encode-decision, provenance→self-schema, causal-forward-model, affective-world-model are
**clean** (warmup washout, or the lesion numerator is structurally pinned to 0). r4-selfschema-provenance,
neural-wta-word-decode, wkv-graded-recurrent leak but the GO margin ≫ leak (hardening only).

## The fix + re-verify plan (see synthesis.md for the ranked table)
- **Fix recipe:** two ports — route `onebrain_*` runners to the framework's already-correct `_PER_NEURON_STATE`
  snapshot (`onebrain_merge_framework.py:246-250` already lists all 4 arrays via `read_isolation`/`sequence_isolation`);
  inline the C2 `_EXTRA_RESET_ARRAYS` for standalone runners (watch the `cp_refractory` typo →
  `cp_refractory_timers`). One shared leaky primitive `fswta_drive()` in `_d3_spiking_attractor_derisk.py` is imported
  by other runners — fixing it there covers unaudited importers. Every port is a no-op where the arrays are inert, so
  it cannot change a clean verdict — add a repeat-read-bitwise-identity `selftest`.
- **Re-verify (cupy, gpu_queue, sequential):** IG-1 curiosity→d6-WM FIRST (production over-claim), then FW-1
  expectation-rpe, FW-2 surprise→episodic, FW-3 r3v2+r3v3, H-1 r4.

## ⭐ Update 2026-09-02: H-2 landed — CORRECTION, this is NOT pure hardening. `fswta_drive`'s OWN home runner
(`_d3_spiking_attractor_derisk.py`) has a REAL A5 group-verdict flip, missed by the original synthesis
<!--derived-->

**What the original synthesis checked vs. what this landed.** synthesis.md's H-2 entry evaluated the shared-primitive
leak's effect on a **downstream importer** (`neural_wta_word_decode`, "verdict does not move ... margin ≫ leak") and
concluded the fix is "highest-leverage hardening." **It never separately checked `fswta_drive`'s own PRODUCING
runner's A5 verdict** — that check is what this update adds, and it changes the picture.

**The fix:** ported the C2 `_EXTRA_RESET_ARRAYS` block into `fswta_drive()` / `build_fswta_score_bridge()`
(`research/runners/_d3_spiking_attractor_derisk.py` L24-113) exactly as the recipe above prescribes — the same 4
arrays, same true-rest-snapshot-at-build pattern. `fswta_drive` is a SHARED PRIMITIVE called repeatedly on one
bridge across an autoregressive rollout (this runner's own `spiking_rollout_eval`, plus ~20 D3/event/reslm/mouth/
joint-attention/wkv importers). Added `selftest_read_isolation()` (`--selftest`), verified in BOTH directions: it
PASSES on the fixed code and FAILS on a `git show HEAD` snapshot of the pre-fix file (`cp_refractory_timers` /
`cp_prev_firing_states` provably do not reset without the port).

**numpy 6-seed BEFORE-vs-AFTER, the runner's own banked seeds, DEFAULT CLI params (`--fs-inh 9.0 --fs-settle 25`
— no override):**

| group | metric | BEFORE (git HEAD) | AFTER (fixed) |
|---|---|---|---|
| S3 (seeds 42,43,44) | every field, every seed | byte-identical | byte-identical |
| A5 (seeds 100,101,102) | `FSWTA_deeper_track` mean | 0.8997 | **0.9553** |
| A5 | `FSWTA_host_agree` mean | 0.9857 | **0.9953** |
| A5 | **group verdict** (`best_deeper>0.90 and best_agree>0.95`) | **PARTIAL/NEGATIVE** | **GO** |

Per-seed: seed 100 unaffected both arms (`FSWTA_deeper_track` 0.983/0.998, at ceiling either way); seed 101
`FSWTA_deeper_track` 0.833→0.95 (`FSWTA_host_agree` 0.978→0.998); seed 102 0.883→0.933 (0.981→0.99). Artifact:
`research/findings/raw/_read_isolation_audit/h2_fswta_drive_fix_verify.json` (a `tools.verdict.Verdict`, GO — every
precondition earned: backend, both selftest directions, the untouched-primitive isolation control, the floor
crossing, the flip itself).

**The isolation control that pins this to the exact primitive fixed:** `SPK_deeper_track` / `SPK_host_agree_deeper`
(computed via the SIBLING function `onbridge_divnorm_drive`, never touched by this port) and
`rate_step_delta`/`rate_deeper_track` (the host-computed rate transition) are **bitwise identical before vs after
in every one of the 6 seeds**. Only `FSWTA_deeper_track`/`FSWTA_host_agree` — the two metrics `fswta_drive` itself
computes — move, and only on the 2 of 3 A5 seeds where the audit's own instrumentation found a difference.

**Why the earlier small-scale instrumentation (2-3 sequential synthetic calls, reported in the fix's own
`selftest_read_isolation` docstring) found NO divergence and this 6-seed run did:** `spiking_rollout_eval` chains
UP TO 8 sequential `fswta_drive` calls with DIFFERENT, data-driven score vectors per autoregressive rollout step,
across 60 eval sequences per seed — far more opportunity for the leaked refractory/prev-firing residue to compound
than a 2-call synthetic probe with hand-picked scores. The selftest still correctly guards the underlying
mechanism (both directions verified); it just doesn't reproduce the LENGTH of chain where the leak's effect
compounds into a metric-moving margin. **The instrument is part of the emulation**, again.

**Scope note (not a retraction of the existing GO):** `2026-07-09-D3-spiking-attractor-rung1-...-GO.md` already
banks A5 as GO, but at TUNED FS parameters (`--fs-inh 18 --fs-settle 45`), a different operating point than this
check's bare defaults (`--fs-inh 9.0 --fs-settle 25`, what `python -m research.runners._d3_spiking_attractor_derisk
--group A5 --seeds ...` runs with no override). This update does not contradict that finding — it reports a NEW
result at the DEFAULT operating point, which the read-isolation leak was independently degrading toward (and past)
the GO floor.

**Selftest:** `research.runners._d3_spiking_attractor_derisk.selftest_read_isolation()` / `--selftest`.
**cupy re-verify:** QUEUED (this is a real verdict flip, not hardening — the "no cupy needed" exemption does not
apply here) via `bash tools/gpu_queue.sh add`, guarded on the `_EXTRA_RESET_ARRAYS` marker being present in the
runner (skips cleanly if the fix goes stale/reverted before the queue drains):
`research/queue/_h2_fswta_drive_readfix_cupy_verify.sh`.
**Branch:** `research/readfix-d3_spiking_attractor_derisk`.
