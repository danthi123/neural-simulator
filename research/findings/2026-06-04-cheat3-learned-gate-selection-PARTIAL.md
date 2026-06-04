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

## Close de-risked (2026-06-04): the gap is drive magnitude, and Tier-1-scale pools fire the cascade

`research/runners/_msn_synaptic_drive_probe.py` settles it cheap-first: drive a verb pool at 1500 pA through a
FIXED `verb→D1` weight of 16 (the learned magnitude), and sweep the pool size:

| verb pool | D1 | GPi (base 0.28) | thal (base 0.01) | motor_N | cascade |
|---|---|---|---|---|---|
| 30 | 0.000 | 0.26 | 0.000 | 0.000 | silent (the #3 gap) |
| 100 | 0.000 | 0.27 | 0.000 | 0.000 | silent |
| **300** | **0.238** | **0.06** | **0.056** | **0.137** | **FIRES** |
| 500 | 0.304 | 0.04 | 0.067 | 0.187 | FIRES |
| 1000 | 0.312 | 0.04 | 0.068 | 0.194 | FIRES |

So the wall is purely drive magnitude (summed synaptic drive ≈ n_presynaptic × rate × weight): at ≥300 presynaptic
neurons the learned-magnitude weight fires the high-rheobase MSN-D1, which silences GPi (0.28→0.06), releases the
thalamic relay, and routes to the motor — the genuine #2 disinhibition cascade completing from the cue alone, no
direct current. This is consistent with the validated Tier-1 word→action recipe (500-1000 neuron pools).

## Full multi-binding close — HONEST NEGATIVE (the single-binding cascade fires, the 16-binding selection does not)

The full retrain at `--n-verb 500` (3 seeds, true + permuted-teacher anti-cheat,
`research/findings/raw/cheat3_close_nverb500.txt`) is a **NEGATIVE**:

```
TRUE teacher:      seed 42/43/44 = 1/4 each  ->  3/12   [GO->N(ok) COME->N(X) STOP->N(X) LOOK->N(X)]
PERMUTED teacher:  TRUE-label hits 3/12 (=chance),  PERMUTED-label hits 3/12 (=chance)
=> NEEDS TUNING / NOT YET LEARNED
```

Every verb routes to **N** regardless of the teacher (true AND permuted both 3/12 = chance) — GO→N is correct
only by coincidence with the bias. So scaling the cue pool fired the single-binding cascade in ISOLATION (the
cheap-first probe above) but did NOT close the full 16-binding selection: it collapses to the **structural N-bias**.

Why: with 16 gated `verb_V → motor_M` routes and 500-neuron cue pools, the motor decode is dominated by one pool
(N) — partially-open / leaky gates plus the random-init structural favouring of N let `motor_N` win even when the
correct `d1_COME_S → ... → g_COME_S` cascade is the trained one. This is exactly the project's long-documented
**structural-N-bias / silent-motor-trap** (the reason Phase B replaced reservoir+argmax with the per-action BG
cascade, and the reason motor WTA / FS lateral inhibition exists): a multi-action selector with a dominant-motor
bias is NOT fixed by more presynaptic drive — it needs cross-pool **winner-take-all** (FS lateral inhibition
between motor pools, or one-gate-open enforcement) so a single binding wins cleanly. Pool scaling addresses the
DRIVE wall (real, cheap-first confirmed) but not the SELECTION-arbitration wall.

## 🎉 RESOLVED (2026-06-04): the negative was a cascade-tuning mismatch, NOT structural — and the anti-cheat passes

The "needs WTA / structural N-bias" reading was wrong — **three** successive smell-test diagnostics corrected it:

1. At `n_verb=500` the learned weight only reached ~4 (plateaus; more epochs don't help) → d1 didn't fire →
   nothing routed → the decode *defaulted* to N (argmax of zeros). Not an N-bias; a non-firing cascade.
2. At `n_verb=1000` the learned weight reached ~7.75 and d1 fired **SELECTIVELY** (only `d1_COME_S` = 0.06; the
   wrong d1 pools stayed silent) — so selection/arbitration was never the problem. But thal stayed suppressed.
3. The real blocker: a weak, STDP-learned d1 (0.06) cannot silence a GPi pacing at #2's **2200 pA** tonic (which
   #2 needs because it drives d1 DIRECTLY to 0.12-0.30). The genuine cascade was tuned for strong direct-drive d1,
   not weak learned d1.

**Fix:** rebalance the GPi pacemaker tonic to the LEARNED regime — `GPI_TONIC_PA = 600` (vs #2's 2200). At 600 the
weak learned d1 fully silences its GPi (→0.00), the thalamic relay is released (0.10), the cascade completes, and
the cue routes to its motor — while baseline GPi still gates the non-selected relays.

**Result (`--n-verb 1000`, `research/findings/raw/cheat3_close_nverb1000_gpi600.txt`):**

```
TRUE teacher:      seed 42/43/44 = 4/4 each  ->  12/12   (GO->N COME->S STOP->W LOOK->E, every seed)
PERMUTED teacher:  PERMUTED-label hits 10/12 (learns the taught permuted mapping),
                   TRUE-label hits 1/12 (BELOW chance 3/12 -- no structural true-mapping bias)
=> LEARNED SELECTION
```

The anti-cheat is decisive: the selection **follows the teacher** — true teacher → true mapping (12/12); permuted
teacher → permuted mapping (10/12) with the true mapping collapsing to chance (1/12). So the gate that opens is
selected by a TRAINED cortico-striatal pathway, not commanded, and not a structural artifact.

## Honest status — RESOLVED

- LEARNING: **validated** — selective cortico-striatal STDP (correct verb→D1 grows; wrong stays at init).
- Genuine end-to-end (learned cue → D1 → disinhibition cascade → gate → motor): **RESOLVED** — 12/12 true across
  3 seeds with the permuted-teacher anti-cheat passing, at `n_verb=1000` + the learned-regime `GPI_TONIC_PA=600`.
- Key biology-translatable insight: a downstream cascade tuned for STRONG (direct-drive) input fails on WEAK
  (plasticity-learned) input even when the learned selection is perfectly correct — the pacemaker/threshold of the
  *receiving* stage must match the magnitude the *learned* projection actually delivers. The honest-negative→
  diagnose→rebalance loop (three corrected hypotheses) is what found it; the first "structural N-bias / needs WTA"
  reading was a decode-default artifact.

## Files

- `research/runners/gated_compose_bg_learned_demo.py` — the learned-selection scaffold + `_step()` clock fix +
  selective-weight-growth demonstration + the wired permuted-teacher anti-cheat.
