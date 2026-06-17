# On-bridge learned spiking binder — STEP 2 (conversation-capable?) = 6-seed unanimous GO

**Date:** 2026-06-17 (CYCLE 150 — the owner-greenlit on-bridge binding build, second step)
**Status:** **GO, 6 seeds unanimous** (42, 43, 44, 100, 101, 102). The on-bridge learned spiking binder does real
conversational store / who-what question-answering / abstention (the no-confab moat) at **full accuracy** on real
LIF spikes. NO protected `sim/` edit; reuse-by-import.
**Runner:** `research/runners/_phaseB_onbridge_learned_composer_derisk.py`
**Raw:** `research/findings/raw/_phaseB_onbridge_learned_composer.json`

## What this step asked

Step 1 (`2026-06-17-onbridge-learned-filler-binding-step1-GO.md`) proved the fixed-role + learned-filler bundled
bind survives real LIF spiking (0.969 = 98% of numpy). Step 2 is the production question: wrap that on-bridge
binding in the composer's conversational API — store a subject-verb-object fact as a spiking-read bundled
composite, answer "what does <agent> <action>?" and "who <action> <patient>?" by unbinding the cued roles +
cleanup, and **abstain** (return nothing) when no stored fact matches — exactly the production `RFPhasorComposer`
interface, but with the **learned spiking binder** in place of the fixed exact-inverse phasor algebra.

The bind/bundle is on-bridge: `store()` drives the analog 3-way bundle onto the LIF ON/OFF populations and keeps
the spiking-read (rate) composite; query unbinds from that spiking composite. The read-out cleanup is the learned
projection (numpy fast path — the same scaffold split as Step 1 and the production composer's numpy cleanup).

## Result — 6 seeds, D_h=256, a 5-fact knowledge base

| metric | 6-seed mean |
|---|---|
| "what does <agent> <action>?" recall | **1.000** |
| "who <action> <patient>?" recall | **1.000** |
| no-confab moat — never-stored cue abstains | **1.000** (6/6 seeds clean, 20/20 probes each) |
| permuted-cue clean (no false bind to another fact's answer) | **1.000** (6/6 seeds) |

chance 0.062. Every seed perfect on every metric.

## Reading it

- **The learned spiking binder is conversation-capable.** It stores facts, answers who/what, and abstains on the
  unknown — the full minimal conversational primitive set — on real LIF spikes, multi-seed, with no false binds.
- **The no-confab moat is structural and intact:** abstention falls out of the iterate-and-match retrieval (no
  stored composite's unbound cue matches an absent query), and the permuted control confirms a wrong cue does not
  smuggle out another fact's answer.
- **Together with Steps 1 + the capacity sweep (RE-OPEN), the greenlit binding build is de-risked end-to-end at
  the mechanism level:** the learned spiking binder can replace the composer's idealized exact-inverse algebra for
  binding, bundling, store, retrieval, and abstention — and it needs no new dendritic mechanism (the fixed ±1
  role makes binding a linear channel-swap; the population rate-code carries the superposition).

## Honest scope — what is brain-based vs still a shortcut

- **Brain-based (real spikes):** the bind/bundle (LIF ON/OFF populations), the stored composite (spiking-read
  rates), the retrieval-by-unbind, and the structural abstention.
- **Still a host shortcut:** the binder's weights (the filler projection and the read-out cleanup) are trained
  off-substrate by gradient descent (Adam), not by a local synaptic rule. Per the BRAIN-BASED-ONLY standard this
  host training is a documented shortcut (like the composer's other learned pieces). **The deep
  biology-faithful frontier is on-substrate weight learning** — can a local rule (Hebbian/delta) learn the
  read-out (and ideally the filler projection) on the substrate? The single-attribute learned bind already works
  on real spikes (2026-06-16, 0.833), so the read-out (a supervised linear cleanup) is the tractable next target;
  fully-from-scratch multi-attribute learning is the known hard boundary.

## Next

The build is mechanism-complete and conversation-capable. The two deliverable directions: (a) **production
wiring** — a `LearnedSpikingComposer` the conversational agent can instantiate (the idealization removal made
usable), with the synaptic read-out as the on-substrate refinement; (b) **on-substrate weight learning** — the
deeper biology-faithfulness frontier (remove the host-Adam shortcut), scoped deep-research-first.

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onbridge_learned_composer_derisk \
    --dh 256 --seeds 42,43,44,100,101,102
```
