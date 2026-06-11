# Flat-cortex (A) completion — the learned no-confab familiarity gate matches the host abstention moat at V=320: GO (multi-seed, zero moat-breaches)

**Date:** 2026-06-11 (overnight). **Runner:** `research/runners/familiarity_gate_v320_validation.py` (CPU, `SIM_BACKEND=numpy` — tiny bridges, numpy linear algebra; not a GPU-heavy workload). **Raw:** `research/findings/raw/_familiarity_gate_v320_validation.json` (seeds 42/43/44, V=320, D=128, 60 stored facts, abstention floor 48) + `_famgate_smoke.json` (V=64).

> **Result: GO, multi-seed.** The learned Bogacz-Brown anti-Hebbian familiarity gate (a computed spiking-compatible novelty signal) agrees with the production composer's host abstention decision on **every cue, all three seeds (168/168 agreement)**, with **zero moat-breaches** (the dangerous `host-abstain / gate-accept` confusion cell = 0) and **zero abstention-floor false-accepts (0/48 every seed)** — the load-bearing bar. The separation margin shrinks from the toy de-risk (+0.98) to **+0.49 at V=320** (a characterized code-density effect, not a breach), stays robustly positive, the operating threshold window is wide (≈0.44–0.52), and a lesion of the gate's learned weights collapses the separation (anti-cheat). This validates a **neural replacement for the host `if`-based no-confab moat — alongside the host check, without weakening it** — closing flat-cortex (A)'s last brain-based gap.

## Why this ran

The cortex de-risk arc concluded with the fork: **(A) a semantically-flat cortex achievable now** (the production composer over generated decorrelated codes — full conversational matrix at 320 concepts) vs **(B) the deferred dendritic-substrate rewrite** (semantic generalization, gated on analog/pre-spike whitening). While (B) is the owner's decision, the decision-independent forward work is completing (A). The production composer answers fact-queries and **abstains** when nothing matches — the "no-confab moat" — but that abstention is currently a **host check** (`rf_phasor_composer.py` `query_agent` iterates the knowledge base and abstains via a Python `if`). A cheap-first de-risk showed a brain-based replacement works at toy scale (a learned familiarity/novelty signal: known cue ≈ 0, unknown ≈ 0.99). This validates it at **production scale (V=320), multi-seed, alongside the host moat (no production edit)** — the precursor to ever wiring it in.

## Method (moat-preserving)

For each cue, both decisions are computed and compared: (a) the **host** answer/abstain (the existing `if` — the ground-truth moat), and (b) the **learned familiarity gate** decision (threshold the computed novelty score). The production composer self-generates decorrelated phasor codes from the seed (verified: between-code phase-cosine ≈ 0). The knowledge base stores 60 subject-verb-object facts; the abstention battery presents **known cues** (fact stored → should answer, gate should read "familiar") and an **abstention floor** of 48 **unknown cues** (fact not stored → should abstain, gate should read "novel"). The production moat is never modified — the gate is validated alongside it.

## Results — V=320, seeds 42/43/44

| seed | separation margin (unknown.min − known.max) | gate-vs-host agreement | moat-breach cell (host-abstain/gate-accept) | abstention-floor false-accepts | threshold window width | lesion |
|---|---|---|---|---|---|---|
| 42 | **+0.483** | **1.000 (168/168)** | **0** | **0 / 48** | 0.464 | collapses (−0.000) |
| 43 | **+0.538** | **1.000 (168/168)** | **0** | **0 / 48** | 0.521 | collapses (−0.000) |
| 44 | **+0.452** | **1.000 (168/168)** | **0** | **0 / 48** | 0.440 | collapses (−0.000) |

**Aggregate:** mean margin **+0.491**; multi-seed total abstention-floor false-accepts **= 0**; all seeds zero-false-accept + robust-window + lesion-collapses + perfect-agreement → **VERDICT: GO**.

## Reading

- **The moat is preserved at scale.** The only dangerous confusion cell — the gate *accepting* a cue the host *abstains* on (a confabulation risk) — is **0 across all seeds**, as is the abstention-floor false-accept count. The gate never lets through a cue the host would refuse.
- **The margin shrinks with scale but stays safe.** +0.98 (toy) → +0.49 (V=320). This is a code-density effect (more concepts → denser code space → smaller novelty gap), the same kind of dimension dependence the embedded clause showed (needs D≥256). At V=320, D=128 the margin is still robustly positive with a wide threshold window — but the trend means a much larger vocabulary may need a higher code dimension D to keep the margin safe; that is the characterized scaling knob, not a present failure.
- **The decision rides the learned gate.** Lesioning the gate's learned weights collapses the separation to ≈0 (the lesion anti-cheat), confirming the novelty signal is computed by the learned familiarity mechanism, not an artifact of the readout.
- **Note on the GPU crash:** the first attempt (GPU/cupy path) died without output; this validation uses tiny bridges (256–512 neurons) plus numpy linear algebra (phasor codes + the familiarity gate), so CPU is the correct, fast (~2 min for V=320 × 3 seeds), out-of-memory-free choice — a legitimate small-workload numpy run, not a heavy run mis-pinned to CPU.

## What this completes, and the honest constraint

This validates the **neural no-confab moat** as a drop-in-agreement replacement for the host abstention check at production scale — the last host shortcut in the conversational abstention path, now shown to be replaceable by a computed spiking-compatible familiarity signal **with zero moat-breaches**. It does NOT weaken the host moat: the gate was validated *alongside* it. Any future wiring should keep the host check as a belt-and-suspenders fallback (the gate accepts only where the host accepts; both must agree to answer), so the moat can never regress.

## Verdict + next step

**GO (multi-seed).** Flat-cortex (A)'s brain-based no-confab moat is validated at V=320. **Next:** (1) optionally wire the gate in as belt-and-suspenders (gate AND host must agree to answer) — a careful, moat-non-weakening change; (2) the remaining (A) work is assembling the full pipeline (decorrelated codes → binder → cleanup → this gate) end-to-end on the merged one-bridge substrate and confirming the V=320 matrix on-substrate; (3) the (B) dendritic-substrate rewrite for semantic generalization stays the owner's decision. No banking.
