# Scoping note — the spiking realization of the unified agent (pure-biology backlog #1)

**Date:** 2026-06-04. **Status:** scoping for an owner go/no-go. Plain language; no undefined terms.

## The goal

The unified-agent benchmark (`research/runners/unified_agent_benchmark.py`,
finding `2026-06-04-unified-agent-benchmark-converge-not-add.md`) measures one coherent conversational agent at
320 concepts on a frozen test set. It currently runs as **numpy phasor algebra** — fast, but not the brain
analogue. Pure-biology backlog item #1 is to run the *same* agent as a **spiking neural network** (the project's
top-line goal: a proper brain analogue, not algebra). The benchmark's per-category pass-rate table is the exact
specification the spiking version must reproduce.

## What already exists (validated spiking pieces)

Much more than expected — the spiking substrate is largely built and individually validated:

| piece | where | status |
|---|---|---|
| spiking bind / unbind / bundle (phase-coded spikes; Orchard & Jarvis 2023) | `research/runners/spiking_phasor_fhrr.py` | self-test clears the frozen 0.80 compositional bar at loads {2,3,5} |
| spiking **cleanup with abstention** (winner-take-all over a vocabulary by spike-phase similarity) | same file, `cleanup()` | tested; the no-confabulation moat holds in spikes |
| spiking **resonator** (the multi-factor / nested decode) | recursive-clause + resonator findings (2026-06-03) | "decodes 1.00 in genuine resonate-and-fire spikes (D=256)" |
| membrane-level resonate-and-fire phase readout | membrane finding (2026-06-03) | retrieval 1.00; needs a high-Q (low-leak) resonator — biology insight |
| spiking-STDP that **learns** the word→code map | spiking-STDP finding (2026-06-03) | retrieval 1.00, bind/unbind 0.95 on learned codes |

The cleanup primitive matters specifically because the **(b)** result (2026-06-04) made pattern completion a
**required** component: composition only works on clean concept codes, so a noisy grounded readout must be snapped
to its nearest clean concept attractor (an autoassociative cleanup) *before* composing. The spiking `cleanup()`
is exactly that primitive — winner-take-all to the nearest stored attractor — so the load-bearing new component
is already present in spikes.

## What is NOT yet assembled (the real work)

The pieces exist in separate probes/demos; they are not assembled into the one agent that reproduces the
benchmark. The genuine gaps:

1. **One spiking agent object** that stores SVO facts, answers who/what, abstains, and composes — wiring the
   existing spiking bind/unbind/bundle/cleanup/resonator into the `NestedCompositionAgent` interface so the
   *same* benchmark harness can drive it. (Engineering, not research.)
2. **The grounded pipeline in spikes:** grounded word cue → spiking readout → spiking cleanup (pattern
   completion) → spiking compose. The numpy (b) result must reproduce: raw readout fails attribute composition,
   cleanup recovers it. This is the one place a cheap de-risk is warranted FIRST (below).
3. **The attribute/nested decode at the benchmark's scale in spikes** (the resonator was validated at D=256 on
   small vocab; the benchmark is 320 concepts). Likely needs larger D and is slower — a cost, not a risk.
4. **Speed.** The spiking substrate steps a 1000-step cycle per operation; the benchmark's resonator does many
   operations. A full multi-seed spiking benchmark may be hours on CPU — this is the GPU-worthy build (CuPy),
   which is the legitimate place for the GPU the owner reserved.

## Cheap-first de-risk (run this ONE thing before the full build)

The single genuinely-unvalidated link is whether the spiking cleanup performs the **(b) pattern-completion** at
the benchmark's noise/scale: take clean spiking attractors, add phase noise (simulating a grounded readout),
and confirm the spiking `cleanup()` recovers the correct attractor across a noise sweep, degrading gracefully.

**DONE 2026-06-04 — RESOLVES.** `research/runners/_spiking_pattern_completion_probe.py` (pre-registered verdict):
spiking cleanup recovers a corrupted phasor code to the correct attractor at **100%** out to noise that drops the
code's self-similarity to 0.45, and **99.7%** at self-similarity **0.17** — which is exactly the numpy grounded
readout's `recall_conf` — mirroring the numpy `id_acc ~1.00`. Anti-cheat holds: recovery collapses to chance
(9.5%, ≈1/32) under full randomization, so it is not trivially always-correct. The load-bearing new component is
de-risked in genuine spikes. (c) is therefore an **integration of fully-de-risked pieces** (spiking FHRR +
resonator + STDP learning + this cleanup), not a research gamble.

## Honest scope / cost

- **Not a from-scratch research risk** — the hard parts (spiking FHRR, resonator, cleanup, STDP learning) are
  individually validated. It is an **integration + scale-up + GPU port**.
- **Estimated:** the de-risk probe is minutes. The integrated spiking agent reproducing the *robust core*
  (flat / who / abstain / one-attribute) is a bounded build. The full table including two-attribute + clauses at
  320 concepts in spikes is the larger, GPU-worthy part (the benchmark is its spec and gate).
- **Reserved-GPU fit:** yes — this is the decisive heavy run the GPU is for.

## Decision ask

Confirm to proceed toward (c). Suggested staging: (1) the cheap de-risk probe above; (2) if it passes, the
spiking agent reproducing the robust core on CPU; (3) the full benchmark in spikes on GPU. Each stage is
benchmark-gated and propagated honestly. Alternatively, redirect (e.g., leave (c) and push concept/vocab scale,
or strengthen the conversational surface on the validated algebra agent).
