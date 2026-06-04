# Cheat-removal #3 PARTIAL: learned (not commanded) BG gate selection — 2026-06-04

**One line:** The selection-LEARNING half is validated — cortico-striatal STDP *selectively* learns which gate a
cue should open (correct verb→D1 synapse grows 0.5→~16, wrong targets stay at 0.5). The end-to-end routing is not
yet closed because the learned weight can't drive the high-rheobase striatal MSN-D1 to fire *synaptically* at
inference — the same drive-strength wall #2 sidestepped with direct current. Two load-bearing discoveries en route.

## Goal

#2 made the gate opened by a genuine D1⊣GPi⊣thal disinhibition cascade, but WHICH D1 pool is driven (= which gate
opens) was set by hand (commanded). #3: have a plastic cortico-striatal pathway *learn* the selection, so the cue
alone opens the right gate. `gated_compose_bg_learned_demo.py`: a plastic `verb_V → d1_V_M` pathway (all 16,
low init 0.5), trained supervised — co-drive the verb cue with a teacher current on the CORRECT D1 pool so STDP
binds verb→correct-D1. Inference: drive the verb alone; the learned weight should fire the correct D1 → the #2
cascade opens the correct gate.

## What's validated: the learning is genuine and selective

After 20 epochs of teacher-paired training (seed 42):

| synapse | learned mean weight |
|---|---|
| verb_GO → d1_GO_N (correct) | **18.2** |
| verb_GO → d1_GO_S (wrong) | 0.50 (init, untouched) |
| verb_COME → d1_COME_S (correct) | **15.6** |
| verb_COME → d1_COME_N (wrong) | 0.50 (init, untouched) |

The correct synapse grows ~30×; the wrong ones are untouched. The cortico-striatal map is genuinely **learned**
from the teacher — the scientific core of #3 (selection is not commanded). A permuted-teacher anti-cheat is wired
in `_eval` for the full multi-seed run once the end-to-end gap below is closed.

## Load-bearing discovery 1: the step doesn't advance the clock

The first run produced **exactly 0.50** (init) on every synapse — STDP literally did not run. Root cause:
`SimulationBridge._run_one_simulation_step()` does NOT advance `runtime_state.current_time_ms`; the batch-run
loop does (`bridge.py:3179`). A runner that calls the step directly freezes the clock at 0, so every spike gets
timestamp 0, `delta_t = post − pre = 0` for every pair, and the STDP weight update is a **silent no-op**. Adding
`current_time_ms += dt_ms` after each step (`_step()`) makes STDP learn. (The #2 demo also calls the step
directly — harmless there, no plasticity, the cascade still works on instantaneous conductances.) This is a real
gotcha for any future runner that drives plasticity by calling the step directly rather than through the
batch-run / experiment loop.

## Load-bearing discovery 2: synaptic MSN-D1 drive is the remaining wall

At inference (verb alone, no teacher), the learned weight does NOT fire the D1 pool: `d1_COME_S = 0.000`,
`thal = 0.000`, all motors 0.000. The striatal MSN-D1 (`IZH2007_STRIATAL_MSN_D1`) has a high rheobase; the #2
cascade fired it by injecting 1500 pA *directly*. A learned *synaptic* weight of ~15-18 (and even a manual ~120
in a quick test) didn't reach D1's threshold. This is the **same wall #2 sidestepped**: there, `sel→d1` at
weight 40 was also too weak to fire D1, which is exactly why #2 drives D1 with direct current. So the gap is
**engineering (drive strength), not science (learning)** — the learning works; the learned signal just isn't yet
strong enough to fire a high-rheobase MSN synaptically.

## Continuation (scoped)

The validated Tier-1 word→action recipe DOES drive action selection synaptically — it uses **500-1000 neuron
pools + motor FS interneurons + topographic priors** (vs the 30-neuron pools here). So the close is to scale the
presynaptic drive that way, or insert a more-excitable cortico-striatal relay upstream of the MSN. Then: learned
cue → fires its D1 → the genuine #2 disinhibition cascade → opens the correct gate → routes. The multi-seed
end-to-end + permuted-teacher anti-cheat (already coded) then becomes the gate.

## Honest status

- LEARNING (the hard, scientific part): **validated** — selective cortico-striatal STDP.
- END-TO-END routing: **open** — a drive-strength engineering gap (synaptic MSN-D1 firing), with a concrete,
  validated-recipe continuation. Not claimed resolved.

## Files

- `research/runners/gated_compose_bg_learned_demo.py` — the learned-selection scaffold + `_step()` clock fix +
  selective-weight-growth demonstration + the wired permuted-teacher anti-cheat.
