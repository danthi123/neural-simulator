---
type: finding
status: live
date: 2026-08-19
mechanism: interoceptive-affect
board_task: 49
artifacts:
  - research/findings/raw/embodied_affect/_embodied_affect_interoception_6seed.json
  - research/findings/raw/embodied_affect/_embodied_affect_interoception_6seed_smoke.json
  - research/findings/raw/embodied_affect/_embodied_affect_seed42_trace.json
runner: research/runners/_embodied_affect_interoception_derisk.py
---

# Embodied affect — a simulated interoceptive BODY-STATE causally drives the neural affect attractor (6-seed GO)

**Board #49 ("give the brain real feelings tied to a simulated body").** Verdict: **BRAIN-BASED GO, 6/6 seeds**
(numpy-CPU, NO `sim/` edit). A minimal simulated body-state causes the neural affect state through spiking
interoceptive populations and synapses, and interoception is load-bearing for it. The gradedness of the read is
the P0.3 bistable-latch boundary, reported honestly below.

## What was built (a bounded first slice, not a full body model)

- **Simulated interoceptive body-state (host — the body interface, like the world):** `homeostasis h in [0,1]`
  (satiety/comfort; comfort = h, discomfort = 1-h) and `arousal a in [0,1]` (heart-rate / sympathetic tone).
  These are the ONLY host->neural quantities and enter the brain ONLY as an interoceptive afferent CURRENT
  (~200 pA), the same body->sensory boundary the homeostatic-drive GO used.
- **Three spiking interoceptive populations** (Izhikevich RS, no recurrence — pure afferent relays):
  `intero_comfort <- comfort`, `intero_discomfort <- discomfort`, `intero_arousal <- arousal`.
- **Synaptic projection onto the reused P0.3 affect attractor** (AMPA, gated by `intero_out`):
  intero_comfort -> affect_vplus, intero_discomfort -> affect_vminus, intero_arousal -> affect_arousal. The
  affect attractor (the 2026-07-24 P0.3 opponent slow-NMDA region) is reused UNCHANGED via an additive default-off
  `extra_regions`/`extra_pathways` seam on `AffectStateBrain`.
- **The felt state is the attractor's OWN read:** `mood = rate(affect_vplus) - rate(affect_vminus)` and
  `felt_arousal = rate(affect_arousal)`, off `cp_firing_states`. Never a host formula over the body variable.

## Anti-cheat 1 — the body CAUSES the affect (body-state sweep -> affect response, 6/6)

Sweeping the body-state and reading the neural attractor (per-seed, pooled mean):

Pooled means over the 6 seeds (rounded; per-seed values live in the cited 6-seed artifact):

| body-state | neural affect read | pooled | all-6-seeds |
| --- | --- | --- | --- |
| comfortable (h=1) | mood = rate(V+)-rate(V-) | **+0.082** | > 0 | <!--derived-->
| distressed (h=0) | mood | **-0.081** | < 0 | <!--derived-->
| comfort->distress swing | mood(h1) - mood(h0) | **0.163** | >= 0.05 | <!--derived-->
| ordered tracking | corr(h, mood) | **+0.84** | >= 0.8 | <!--derived-->
| aroused (a=1) vs calm (a=0) | felt_arousal swing | **+0.082** | >= 0.02 | <!--derived-->

A comfortable body yields a positive felt valence, a distressed body a negative one, and an aroused body a raised
felt-arousal — the affect state moves with the body, in the correct direction, on every seed. The path is the
interoceptive populations firing into the attractor, not a host affect assignment.

## Anti-cheat 2 — interoception is LOAD-BEARING (dissociation, 6/6)

Cutting the interoceptive->affect SYNAPSES (`intero_out` gate = 0) while the body-state sweep is UNCHANGED
collapses BOTH affect channels to exactly zero, yet the interoceptive pools keep firing and keep encoding the body:

| quantity | intact | interoception lesioned |
| --- | --- | --- |
| valence range over the h-sweep | 0.181 | **0.000** | <!--derived-->
| corr(h, mood) | +0.84 | **+0.00** |
| felt-arousal range over the a-sweep | 0.107 | **0.000** | <!--derived-->
| intero_comfort pool still encodes the body | corr 0.99 (intact) | **still fires + encodes, 6/6 seeds** | <!--derived-->

`tools.lab.attributable_to` (intact vs lesion) reports **100.0%** of the body->affect coupling is owned by the
interoceptive path (0% present in the lesion control), on both channels, every seed. <!--derived--> The raw is in
`research/findings/raw/embodied_affect/_embodied_affect_interoception_6seed.json`
(`per_seed[].intero_owns_valence_frac` / `intero_owns_arousal_frac`).

The single-seed trace makes the mechanism explicit (seed 42; exact rates in the cited `_seed42_trace.json`, rounded
here):

- distress (h=0): intero_discomfort fires -> V- = 0.082, V+ = 0.000 -> mood = **-0.082**. <!--derived-->
- comfort (h=1): intero_comfort fires (0.035) -> V+ = 0.082, V- = 0.000 -> mood = **+0.082**. <!--derived-->
- comfort + LESION: intero_comfort STILL fires (0.035) but the synapse is cut -> V+ = V- = 0.000 -> mood = **0.000**. <!--derived-->

The body signal is still present in the interoceptive pool; severing its synaptic route to the attractor removes
the feeling. A SILENCE control (zero the afferent current) agrees: mood range 0.000 on all 6 seeds. The affect
pools sit at the OFF down-state at rest and ignite only on the synaptic body drive.

## Anti-cheat 3 — not a host formula (grep-verified)

The affect value is read from the neural attractor, never computed from the body variable. In
`_embodied_affect_interoception_derisk.py` the ONLY assignments are:

```
rate["mood"] = rate["affect_vplus"] - rate["affect_vminus"]   # from cp_firing_states spike counts
rate["felt_arousal"] = rate["affect_arousal"]
```

No expression maps `h`/`comfort`/`discomfort`/`arousal_body` -> affect. A runtime assertion fires every step that
the affect pools receive ZERO direct external current — the body reaches them ONLY through synapses.

## Anti-cheat 4 — 6 seeds, deterministic

Seeds 42/43/44/100/101/102, `cfg.seed` set (seeds the substrate — verified: two builds at one seed give
byte-identical `cp_neuron_firing_thresholds`). All 8 aggregate gates pass 6/6 (valence signs, valence swing,
ordered tracking, arousal raised-by-body, valence lesion-decouple x2, arousal lesion-decouple, interoceptive
encoding).

## Honest residuals (the characterized boundary)

- **The read is a BISTABLE SIGNED SWITCH, not a graded circumplex.** Mood is a two-state ±0.08 latch that flips
  sign near the set-point (h~0.5); felt-arousal is an on/off ignition (gradedness Pearson +0.70, 1/6 seeds >=0.8;
  a smooth ramp would read ~0.95). This is the SAME P0.3 bistable-latch limit — a graded valence x arousal
  continuum needs the named line/bump attractor with adaptation eviction / the dendritic substrate. The valence
  channel grades better than arousal only because its opponent V+/V- structure gives a bidirectional switch; the
  lone arousal pool has no opponent. The gradedness limit is ORTHOGONAL to the embodiment claim: the body still
  causes the correct signed feeling and interoception is load-bearing, 6/6.
- **Scope (bounded first de-risk).** Two body axes (satiety/comfort + arousal), a 3-pool interoceptive channel,
  no full homeostatic feedback loop and no body dynamics (a follow-on): the body-state is swept open-loop, not
  produced by the agent's own metabolism/behaviour. The body VARIABLES are host (the standard body boundary) —
  the de-risk is the body->AFFECT MAPPING being synaptic, not the body itself.
- **Persistence not re-tested here.** The body drive is held during the read, so this measures the DRIVEN affect
  state, not whether the mood persists after the body normalizes (that is P0.3's separate, already-GO property).
- **Honesty boundary.** This is a functional core-affect STATE with a bodily cause and an honest functional
  read-out (a felt sign that tracks the body); no claim of phenomenal experience is made.

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._embodied_affect_interoception_derisk --smoke
SIM_BACKEND=numpy python -u -m research.runners._embodied_affect_interoception_derisk \
    --seeds 42 43 44 100 101 102
```
