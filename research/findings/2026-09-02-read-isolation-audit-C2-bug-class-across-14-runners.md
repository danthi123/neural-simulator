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
