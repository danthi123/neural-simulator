---
type: plan
status: live
date: 2026-06-08
---

# Brain-fidelity roadmap: close cheats → unify → conduction delays → multi-compartment neurons

**Date:** 2026-06-08
**Owner-set sequence (2026-06-08):** *"Finish closing all cheats (navigation and conversational) → consolidate navigation and conversational configs into a single brain → implement proper conduction delays → implement proper [multi-compartment] neurons."*
**Status:** the durable plan. Supersedes ad-hoc next-step notes for the post-nav-critic arc.

This doc records the four-step sequence, the one structural refinement it needs (some cheat-closures
are *gated on* the fidelity upgrades, so they move to the end), the per-step scope + acceptance bars,
and the dependency graph.

---

## 0. The goal this serves

Artificial life with a *proper brain analogue* + biology-translatable insights, under the **BRAIN-BASED-ONLY
standard**: every cognitive function (perception, salience, orienting, reward, value, dopamine, action
selection, language) is realized by **neurons / synapses / their communication**, not by host code. Host
code is legitimate ONLY for the *environment* (world state + sensory render) and the *body* (acting on motor
output). Honest negatives under strict biology **are** the scientific deliverable (they map what the
substrate can/can't do on its own).

---

## 1. The four steps (with the refinement)

The owner's four steps are correct. The one refinement: **"close all cheats" must be scoped to cheats
closeable on the *current* substrate.** Two of the project's deepest capabilities are *gated on* the fidelity
upgrades (steps 3–4) and therefore cannot be done in step 1 — they are the **payoff** of the upgrades, not
precursors:

- The **genuine-cortex conversational conversion** (replacing the VSA composer's exact-inverse algebra with
  *learned, lossy, redundant dendritic read-outs*) needs **multi-compartment neurons** (step 4). Point neurons
  with one soma structurally can't host it. *Note:* the composer is classified as a **principled idealization,
  explicitly NOT a cheat** (`2026-06-06-composer-vsa-idealization-known-limitation.md`), so this does NOT block
  step 1 — it just means the conversion lands *after* step 4.
- A **theta-gamma sequence-binding conversational architecture** (the SPEAR-style path the owner reframed
  conversation around) needs **conduction delays** (step 3) for phase-of-firing codes.

So the executable sequence is:

```
Step 1  Close all SUBSTRATE-INDEPENDENT cheats (nav + conversational)
          │
          ▼
Step 2  Consolidate nav + conversational into ONE brain
          │
          ▼
Step 3  Conduction delays           (protected sim/ edit, additive, default-off)
          │
          ▼
Step 4  Multi-compartment neurons   (protected sim/ core change; the big one)
          │
          ▼
Post    Substrate-GATED capabilities the upgrades unlock:
          • genuine-cortex read-out conversion   (needs step 4)
          • theta-gamma sequence conversation     (needs step 3)
```

**Why fidelity-after-unification is the efficient order (not just convenient):** conduction delays and
multi-compartment neurons are *substrate* changes that benefit both arcs. Doing them after the merge means
implementing + validating each **once**, on one brain, against one combined bar — instead of twice on
diverging configs that must then be reconciled. Both upgrades are **additive / default-off** (a per-pathway
`conduction_delay_ms` defaulting to 1 step; a 2-compartment mode opted into per region), so they do **not**
invalidate the unified brain — you enable them where they earn their keep. Delays-before-compartments is right:
delays are medium-effort and unlock a whole capability class cheaply; compartments are a months-long core
rewrite.

---

## 2. Step 1 — close all substrate-independent cheats

**Bar:** every cognitive function in both arcs is realized by neurons/synapses, not host code. Where the neural
version underperforms a host shortcut, that **honest negative is the deliverable** (it maps a substrate limit),
documented — not hidden.

### 2a. Navigation cheats → neural
The 2026-06-08 BRAIN-BASED-ONLY directive reclassified several biologically-*shaped* but **host-computed** nav
pieces as shortcuts whose spiking/synaptic versions are the real target. Remaining:

| Item | Current state | Target (neural) | Status |
|---|---|---|---|
| **N9 value subtraction** | spiking-SNc actor-critic; value was host reward-EMA scaffold (Stage A) | a **neural striosome value critic** that learns V + subtracts via GABA_B/GIRK at the SNc membrane (Stage B) | **IN FLIGHT** — mechanism de-risked (PASS); nav build runs + navigates; critic-firing drive **calibration** under way |
| **N5 perceived reward** | coord-free perceived-approach reward, but **host-computed** | a **neural reward/value system** (a spiking reward signal) | queued |
| **N1 orienting reflex** | superior-colliculus reflex, host-computed | a **spiking superior colliculus** | queued |
| **N6 action read-out** | argmax over motor spike counts (host) | a **minimal neural motor read-out** | queued |
| **N9 dopamine** | spiking SNc FIRES the RPE (Stage A done) | (done — SNc fires δ; the *value* it subtracts is the Stage-B critic above) | Stage A ✅ |
| **neural position code** | `sensor_place_readout` place code (rendered readout) | confirm/convert to a neural position code | review |
| **N2 / N7 characterizations** | beacon perception; innate V1 Gabor pre-init | CHARACTERIZE as defensible (not cheats): beacon = legit visual input; V1 orientation tuning is present at eye-opening (retinal waves) | write the verdicts → the "fully biologized nav" finding |

### 2b. Conversational cheats → neural
The conversational pipeline already runs **on the core sim** (parser Hebbian-learned; bind/unbind on-substrate
spiking FHRR; cleanup spiking NEF; dialogue spiking dlPFC; one-bridge unification done). Remaining cheat:

| Item | Current state | Target (neural) | Status |
|---|---|---|---|
| **(B) memory shortcut** | the bound fact + superposition/opponency held in **numpy** | a **substrate-held attractor memory** (Hopfield/attractor cleanup) — doable on point neurons | queued (`2026-06-05-composer-B-substrate-held-memory-options.md`) |
| composer exact-inverse algebra | host VSA algebra (idealization) | **NOT a cheat** — a principled idealization; its genuine-cortex replacement is **step-4-gated** (see Post) | deferred to Post |

**Step 1 exit criterion:** both arcs run with no host *cognition* (only environment + body in host code), every
remaining piece either neural or an explicitly-documented honest negative; nav passes its 6-seed bar and
conversational passes its capability matrix.

---

## 3. Step 2 — consolidate into one brain

**Scope:** one `SimulationBridge` holding both nav and conversational regions as disjoint persistent slices on
shared substrate (the conversational regions are already unified on one bridge; nav already runs on the core
sim — this merges the two into a single interacting brain).
**Gated on:** step 1 (the owner's explicit precondition: *nav fully biologized before unification*; plus the
conversational cheat closed, so the merged brain is clean).
**Bar:** the single brain passes **both** arcs' validation bars (nav 6-seed deterministic + the conversational
capability matrix incl. the no-confab moat) with no regression vs the separate configs. A measured regression
is a reportable cost of merging, not hidden.
**Risk:** architectural assumptions baked in at merge (timing, binding) interact with the later fidelity
upgrades — mitigated because those upgrades are additive/default-off (opt-in per pathway/region), so the merged
brain stays valid and the upgrades are a re-tune, not a rebuild.

---

## 4. Step 3 — conduction delays (audit shortcut #2)

**Why:** the single highest *fidelity-per-effort* substrate fix. Today there is **no axonal delay at all** —
`max_synaptic_delay_ms` is converted to `max_delay_steps` but that value is **set and never read**; every
synapse fires at a uniform 1-step latency. (`2026-06-08-sim-biological-accuracy-shortcuts-audit.md` #3.)
**Scope (protected `sim/` edit, additive):** a per-pathway `RegionPathway.conduction_delay_ms`, realized by a
small per-pathway ring buffer of recent firing vectors (index `delay_steps` back instead of always reading
`cp_prev_firing_states`). **Default 1 step = byte-identical.** Byte-reviewed before it lands (owner standing
rule); a byte-identity-when-default check like the GABA_B mask-fix.
**Unlocks:** temporal/phase codes — **theta-gamma multiplexing** (the conversational path), theta phase
precession, polychronization-style sequence learning, faithful **BG three-phase timing** (early STN → striatal
inhibition → late STN, which the catalog says *requires* differential striatonigral-vs-pallidonigral delays),
rank-order coding beyond the current dt bound.
**Bar:** byte-identical at default 1-step (both arcs unchanged); a temporal-code validation (BG three-phase
timing, or a phase-code experiment) demonstrates the new capability.

---

## 5. Step 4 — multi-compartment (dendritic) neurons (audit shortcut #2-by-impact)

**Terminology:** *point neurons* (single-compartment) **are** the shortcut; this step replaces them with
**multi-compartment** neurons. The viable first version is a **minimal two-compartment (soma + apical) AdEx**,
NOT a full multi-compartment reconstruction.
**Why:** the most consequential shortcut for the deepest goals — it is why **dendritic credit assignment is a
confirmed dead end** (no second compartment to carry the apical teaching signal) and a load-bearing part of the
composer-idealization limitation. (`2026-06-08-sim-biological-accuracy-shortcuts-audit.md` #2;
`docs/plans/2026-05-05-dendritic-learning-design.md`.)
**Scope:** large (months; ~10× compute/neuron; protected `sim/` core change). Additive/opt-in per region
(default single-compartment = byte-identical). Byte-reviewed.
**Unlocks:** dendritic credit assignment (apical-basal learning — the biologically-plausible alternative to
global-scalar feedback for the W→A bottleneck), **the genuine-cortex conversational conversion**, predictive
coding / perceptual inference, dendritic nonlinear computation (NMDA spikes, plateaus), spine-level dopamine
gating in the striatum.
**Bar:** byte-identical at default (single-compartment mode); a dendritic-learning or apical-basal-gating
validation demonstrates the new capability.

---

## 6. Post — the substrate-gated capabilities

These are *not* cheats to close in step 1 — they are the **payoff** of steps 3–4 and land only once the
substrate supports them:

- **Genuine-cortex read-out conversion** (needs step 4): replace the composer's exact-inverse VSA algebra with
  learned, lossy, redundant dendritic read-outs — converting the last conversational *idealization* (not cheat)
  into a functional cortex.
- **Theta-gamma sequence-binding conversation** (needs step 3): a phase-of-firing conversational architecture
  (SPEAR theta-multiplexing / theta-gamma mode-unification), distinct from the current VSA composer.

---

## 7. Cross-cutting principles

- **Protected `sim/` edits** (steps 3, 4, and any step-1 spiking-substrate additions): additive, default-off,
  **byte-identical-when-unused**, owner **byte-reviewed before they land**, with a byte-identity-when-default
  check (the GABA_B/GIRK edit this session is the template).
- **Cheap-first de-risk before every build** (a CPU/probe falsifier with anti-cheat controls) — and the de-risk
  environment must be checked against the *deployment* environment (the recurring "probe ≠ deployment" trap:
  the GABA_B mask bug + the simultaneous-timing artifact both came from this).
- **Honest negatives are deliverables.** Every NEGATIVE maps a substrate limit and is documented, not hidden.
- **Both git remotes** (origin + gitea) on every commit; GPU/CuPy for real runs.

---

## 8. Where we are now (2026-06-08)

**Step 1, nav N9 (value subtraction) — in flight:**
- GABA_B/GIRK postsynaptic conductance: **shipped** (`6f73b5f0`/`a7370d49`, byte-reviewed, Pavlovian de-risk
  PASS 3/3). *(Audit shortcut #1 — addressed.)*
- Place-code value critic de-risk: **PASS** (`d0416fc3`) — learns value-of-location + subtracts via GABA_B under
  nav-faithful timing.
- Nav build (`27f7d79a`): **silencing FIXED, nav excellent** (828/1800 steps at goal, dist 2.13), **but the
  critic is still silent** → critic-drive **calibration in flight** (`2026-06-08-nav-placecritic-smoke-PARTIAL.md`).

**Immediately next (step 1 continuation):** finish the nav critic (calibration → re-smoke → 6-seed A/B) → the
remaining host→neural nav conversions (N5 reward, N1 SC reflex, N6 read-out, position code) → N2/N7 verdicts →
the conversational memory shortcut. Then step 2 (unify), step 3 (conduction delays — the next protected fidelity
edit), step 4 (multi-compartment), then the substrate-gated capabilities.
