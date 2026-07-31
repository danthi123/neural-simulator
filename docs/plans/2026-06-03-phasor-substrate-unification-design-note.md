---
type: plan
status: live
date: 2026-06-03
---

# Design note — phasor FHRR substrate unification (decision artifact) — 2026-06-03

**Status:** decision artifact, owner-steerable. Not an approved plan. Captures the de-risked evidence, the
one open step, and a proposed minimal first experiment so the next arc starts from a plan rather than a
blank page.

## The question

The project's conversational capability splits across two substrates:

- **Production diversity** (160/320-concept G.20 sparse, multi-tag retrieval ~88–93%) on a
  **non-invertible** real/dense binding → **cannot nest** (hierarchical-320 scored 0.000; the flat-distinct
  workaround is single-binding-only).
- **Nesting** (Direction A this session: resonator decoder, multi-modifier, recursive clause — all validated
  in algebra *and* genuine resonate-and-fire spikes) on **phasor FHRR**, where binding is **invertible**.

Should the production substrate unify onto phasor FHRR to gain nesting (compositional structure) while
keeping diversity?

## What is already de-risked (2026-06-03, all committed)

See `research/findings/2026-06-03-phasor-FHRR-unified-substrate-candidate-diversity-plus-nesting.md`.

| Axis | Result |
|---|---|
| Diversity + composition | 320 concepts + 3-role SVO decode **1.00 at D=1024** |
| Single-bundle capacity | ~24–32 role-bindings (8–10× a 3-role fact) |
| Inter-code correlation (common-mode) | 1.00 up to mean cosine 0.768 |
| Inter-code correlation (clustered, grounded-like) | 1.00 up to within-cluster cosine 0.829 |
| Nesting (multi-factor / multi-modifier / recursive clause) | RESOLVES, incl. genuine spikes |
| Agent at production-diversity scale | 120-concept vocab, 40 mixed facts ~96%, 3 seeds + abstention |
| **Learning analog (linear Hebbian)** | cue→code 1.000 + learned codes compose SVO 1.00 at 320 |

So **capacity, correlation, scale, and the linear-learning analog are all de-risked.**

## The one open step — spiking-STDP learning of the input→phasor-code map

The production system does not construct codes algebraically; it **learns** an input("word")→representation
map via STDP on a spiking network. The linear-Hebbian analog passes (above), but the **spiking-STDP**
realization on the resonate-and-fire substrate is unproven. That is the load-bearing open question for
unification.

### Proposed minimal first experiment (cheap-first entry to the arc)

Pre-register a frozen probe BEFORE any larger build (per the project's cheap-first discipline):

1. **Setup:** a small resonate-and-fire network. Input layer = a cue spike pattern (one of N concepts).
   Output layer = phase-coded neurons whose target firing phases are the concept's phasor code.
2. **Learning:** STDP (or a phase-aligned plasticity rule) on input→output weights over repeated
   cue presentations, for a small N (e.g. N=8 then N=32).
3. **Test:** after training, present each cue → read output phases → cleanup against the N-code book →
   retrieval accuracy. Then bind two learned codes with a role and unbind (composition survives learning?).
4. **Frozen three-state gate:**
   - **RESOLVES** := retrieval ≥ 0.90 at N=32 AND a learned 2-role bind/unbind ≥ 0.80 → spiking-STDP learns
     the map; unification is viable end-to-end. Proceed to scale (N→320) + a design/plan.
   - **BOUNDARY** := retrieval works at small N but degrades by N=32 → characterize the spiking capacity
     limit (likely a neuron-count / phase-resolution lever) before committing.
   - **DOES-NOT-RESOLVE** := STDP cannot drive output phases to the target code → honest negative; the
     learning side does not transfer from the linear analog to spikes; do NOT migrate; the phasor substrate
     stays a research/decoder layer, not the production representation.
5. **Anti-cheat:** reproduce-the-failure control (random/shuffled target phases must NOT be learned to the
   same accuracy); smell-test any PASS (a perfect score earns extra scrutiny — this session caught two
   harness false-negatives by exactly this discipline).

If RESOLVES → a full writing-plans pass for the migration (re-implement learned binding + cleanup on phasor
codes across the production path). If not → honest negative, keep the substrates separate; the phasor
nesting agent remains a validated research artifact and decoder.

## Alternative direction (also owner-steerable)

**Direction B — thalamocortical dynamical gating (Logiaco-Abbott-Escola 2021):** the *other* genuinely-new
mechanism the deep-research synthesis surfaced (`research/findings/2026-06-03-deep-research-...md`), partly
present in the existing BG→thalamus cascade. Independent of substrate unification; a separate large arc.

## Recommendation

The evidence points strongly to phasor FHRR being a viable unified substrate on every cheap-first axis. The
**recommended next arc is the minimal spiking-STDP first experiment above** — it is the single load-bearing
unknown, it follows cheap-first discipline, and its outcome cleanly decides whether to invest in the full
migration. It is a genuine engineering arc (network design + plasticity rule + frozen eval), best started
with a writing-plans pass, not a rushed probe.
