---
type: finding
status: live
date: 2026-06-06
mechanism: fhrr
---

# The VSA/FHRR composer is a principled idealization, not a functional reproduction of cortex — known limitation; the learned-read-out conversion (option d) is benched as a future revisit — 2026-06-06

> **Owner directive (2026-06-06):** if the spike-native robustness ladder (options a–c) resolves the
> whitening, bench the deeper learned-read-out rebuild (option d) for now but PROPERLY NOTE it. It is
> **not** labelled a "cheat," but we stay cognizant that the composer is **not functionally identical to
> the cortex it is intended to function as.** Worth revisiting in the future — but NOT higher priority
> than the currently planned tasks, in this order: (1) cheat/shortcut removal, (2) consolidation to a
> single-brain configuration, (3) capability addition and scaling.

## The honest characterization

The conversational composer binds concepts into facts via a Vector Symbolic Architecture (VSA) — the
FHRR (Fourier Holographic Reduced Representation) resonate-and-fire phasor composer. It is a
**principled idealization** — not an arbitrary shortcut, and no longer a numpy cheat — but it is
**not a functional reproduction of cortex.** This note records that distinction so it is not lost as the
project advances.

## What is already converted vs what remains idealized

**Converted to genuine spiking dynamics (no longer math):**
- The binding OPERATIONS — bind, unbind, bundle — run as actual spiking resonate-and-fire neurons +
  complex synapses on the bridge (the FHRR-on-bridge work, 2026-06-05).
- Cleanup is a spiking network (NEF threshold cleanup); memory is held in synapses (substrate weight
  store); the association graph is a learned recurrent structure.

**Still an idealization (the known limitation):**
- The binding is a CLEAN, EXACTLY-INVERTIBLE ALGEBRA — unbind analytically recovers what bind stored.
  A real cortex has no exact inverse; it has LEARNED, lossy, redundant read-outs that approximate the
  function.
- The algebra DEMANDS clean, decorrelated, full-precision concept vectors. A real cortex does not — it
  learns to read whatever (messy, correlated, spike-degraded) code it is handed. **The entire whitening
  problem is downstream of this demand** — it is the bill that comes due for relying on precise algebra
  instead of a learned region.

## Why it is principled, not arbitrary

VSA binding (HRR/FHRR) is a serious, published hypothesis in computational neuroscience about HOW
distributed neural representations could bind. Eliasmith's Spaun — the largest functional brain model
built to date — runs on exactly this (the Semantic Pointer Architecture). There is genuine debate that
cortex implements something VSA-like. So the composer is a credible mechanism-hypothesis realized in
spikes — but a HYPOTHESIS, not verified cortical microcircuitry.

## The specific functional divergences from cortex

| property | the VSA/FHRR composer | a genuine cortex |
|---|---|---|
| binding read-out | fixed, exact, invertible algebra | learned, lossy, redundant, approximate |
| input requirement | demands clean decorrelated codes | learns to read whatever code arrives |
| robustness to noise | brittle (needs whitening) | robust (population averaging, attractor cleanup, learned read-outs) |
| reliability (abstention / no-confab) | clean and ~free (the algebra gives it) | hard-won (learned systems hallucinate / forget) |

## The conversion path (where option d sits)

- **Options a–c** (the spike-native robustness ladder, owner-approved, in progress) make the EXISTING
  algebra spike-FAITHFUL — keep the VSA hypothesis, run it robustly in spikes. They may resolve the
  whitening within the VSA framework, but they do NOT dissolve the idealization.
- **Option d** — replacing the fixed algebra with LEARNED read-outs — is the step that dissolves the
  idealization and becomes a genuine learned cortical region. **This is the only option that makes the
  composer functionally a cortex rather than an algebra running in spikes.**

## The trade-off (why this is not free, and why it is benched)

The clean algebra hands us the no-confab moat (reliable "I don't know" instead of a confabulated answer)
and compositional reliability essentially FOR FREE. A learned cortical system does NOT: it hallucinates,
forgets, needs training, and hits capacity walls. The project's own concept-pool / continual-learning
arc poured effort into the learned side and never matched the VSA's reliability. So option d is the
honest endpoint of "fully biologize the composer," but a months-scale, genuinely uncertain build that
trades the VSA's free reliability for learned dynamics.

## Decision (2026-06-06)

- **Option d is BENCHED** (assuming options a–c, or something similar, resolve the whitening
  spike-natively).
- It is **worth revisiting in the future** — but **explicitly LOWER priority** than the currently
  planned work, in this order:
  1. **Cheat / shortcut removal** — conversational a–c finishing, then the navigational / gridworld
     cheats (the action heuristic, the parked cross-projection cheat #5, perception / reward conveniences).
  2. **Consolidation to a single-brain configuration** — nav + conversational regions live on ONE
     `SimulationBridge` (different region combinations, one engine; not a separate brain).
  3. **Capability addition and scaling.**
- Revisit d only after the above, and only as an explicit owner-greenlit arc — it is the
  spike-faithful-VSA (a–c) vs genuine-cortical (d) standard decision, and the owner sets that standard.

## What revisiting d would entail (for the future arc)

- Replace the fixed bind/unbind algebra with a learned read-out: train the composer's input/read
  synapses (via the bridge's plasticity) to read whatever spiky code the upstream stage produces, and
  to produce clean bound representations from it — the consumer adapts to the producer.
- Re-establish the no-confab moat and compositional reliability under a LEARNED scheme (the hard part —
  the reason the VSA was chosen). Likely needs the continual-learning machinery (consolidation, capacity
  management) the project already built, plus an abstention mechanism that survives learned read-outs.
- Gate on the existing capability matrix + the no-confab controls, multi-seed, same rigor.

## Cross-references
- `research/findings/AUTONOMOUS_STATE.md` — COMPOSER-AS-IDEALIZATION note + the a→c ladder + the owner roadmap.
- `research/findings/2026-06-06-graded-lgn-decorrelation-BOUNDARY.md` — the whitening read-out boundary that surfaced this.
- `research/findings/2026-06-05-fhrr-production-switch-DONE.md` — the FHRR composer as the production default.
- `research/findings/2026-06-06-option1-local-learning-whitening-VALIDATED-6seed.md` — the validated whitening rule.
