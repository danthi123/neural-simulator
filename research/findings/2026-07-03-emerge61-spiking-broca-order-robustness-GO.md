# EMERGE-61 — the spiking-Broca render-ORDER tail CLOSED (inter-utterance wash-out): render-exact 1.00 on ALL 6 seeds, position-independent — GO

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge61_spiking_broca_order_robustness_derisk.py`
**CI:** `tests/test_emerge61_spiking_broca_order_robustness.py` (9 tests, CPU/numpy, offline)
**Raw:** `research/findings/raw/_emerge61_spiking_broca_order_robustness.json`
**Wire:** `research/runners/_emerge60_console_spiking_broca_derisk.py` — additive default-off `reset_producer` flag on `SpikingBrocaConsole` (+ `--reset` CLI); the interactive `_demo` opts in by default.
**`sim/` edit:** NONE.

## The problem (EMERGE-60's ONE honest residual)

EMERGE-60 wires the EMERGE-59 spiking Broca producer (`FrameSlotCQ` / `BrocaProducer`, order = per-pool spiking-RATE ranking on a real `SimulationBridge`) into the flagship console. render-CONTENT is 1.00 (perfect); render-EXACT (word ORDER) is **0.93 6-seed**: on seeds **100 & 101** the 4-slot `F_MODAL` frame `[det:the, SUBJ:robin, FUNC:can, VERB:breathe]` swaps its two adjacent slots → **"the robin breathe can"**. Content always correct; only the ORDER swaps, only on the 4-slot frame, only on 2/6 seeds, and only when robin is the **5th emit** (after owl/minnow/penguin/pike) — the swap is SEQUENCE-POSITION-DEPENDENT (the fresh console renders robin correctly).

## Root cause — H1 CONFIRMED (Izhikevich slow adaptation accumulates across productions)

The spiking read-out (`slot_pool_rates`) advances a real `SimulationBridge`. The Izhikevich recovery variable `cp_recovery_variable_u` is a **slow spike-frequency-adaptation current** that accumulates with every spike and does **not** reset between productions. Instrumenting the state before each emit (seed 100):

```
emit#1 owl      u_pre mean=   0.000 std=  0.000   -> the owl can fly
emit#2 minnow   u_pre mean= 496.9   std=441.6     -> the minnow can swim
emit#3 penguin  u_pre mean= 536.0   std=475.5     -> the penguin walks
emit#4 pike     u_pre mean= 482.2   std=485.6     -> the pike lurks
emit#5 robin    u_pre mean= 466.2   std=503.2     -> the robin breathe can   <-- SWAP
```

`u_pre` is 0 at emit#1 (fresh post-init) and ~500 mean / ~500 **std** (heterogeneous per-neuron) by emit#5. That heterogeneous residual perturbs the 5th production's per-pool rates enough to flip the two near-equal-primacy adjacent slots on the seeds where the primacy noise already put them close. `cp_conductance_g_e` stayed 0 (no internal connectivity) → the residual is purely `v`/`u`. This ties directly to the already-done diagnostics: fresh-console-correct (position-dependent), WTA_NOISE-invariant (deterministic given the sequence, not a noise tie-break), and the naive-flat-reset-worse (the reset was right in spirit, wrong in implementation).

This is a **genuine biological mechanism** (spike-frequency adaptation), not a bug — but it makes an utterance depend on prior utterances' residual state, which a fluent producer must not do (each utterance is an independent motor plan; Broca does not carry the last sentence's adaptation into the next).

## The fix (the CORRECT wash-out) — why the naive flat reset failed

Diagnostic #3 (already done): `v=-65, u=0` for ALL neurons made it **worse** (0.867). That is the WRONG post-init state — it ignores per-neuron heterogeneity (`cp_izh_vr`, `cp_izh_b`) and the correct `u = b*(v-vr)` relation the bridge establishes at init (`bridge.py:1562-1563`), so it disrupts the slot f-I dynamics.

The fix: capture the **EXACT** per-neuron dynamic state right after `_initialize_simulation_data()` — a byte-for-byte snapshot of `v` / `u` / the four conductances / `firing_states` / STP — and **RESTORE** it before EACH production. This returns the substrate to its genuine post-init operating point per utterance, so the read-out is a function of the LEARNED primacy gradient ALONE. Biologically: an inter-utterance wash-out that clears the previous motor plan's adaptation.

Realized as `ResetFrameSlotCQ(FrameSlotCQ)` — subclasses EMERGE-59's producer, overrides only `emit` / `emit_order_indices` to restore first. EMERGE-59 is **untouched**.

## Results (6 seeds 42/43/44/100/101/102, CPU/numpy) — GO

| seed | FIX exact (in-sequence) | FIX pos-indep | CTL (un-reset) exact | CTL pos-indep |
|------|------------------------:|:-------------:|---------------------:|:-------------:|
| 42   | 1.00 | ✓ | 1.00 | ✓ |
| 43   | 1.00 | ✓ | 1.00 | ✓ |
| 44   | 1.00 | ✓ | 1.00 | ✓ |
| 100  | **1.00** | ✓ | **0.80** | ✗ |
| 101  | **1.00** | ✓ | **0.80** | ✗ |
| 102  | 1.00 | ✓ | 1.00 | ✓ |

- **render-EXACT-in-sequence = 1.00 on ALL 6 seeds** (robin as the 5th emit, IN the sequence, not fresh-per-emit).
- **POSITION-INDEPENDENCE holds on every seed** (the load-bearing property): the same fact renders IDENTICALLY at emit-position 1 / 3 / 5 (0 / 2 / 4 prior productions) — `robin@1st == robin@5th == "the robin can breathe"`.
- **CAUSAL**: WITHOUT the reset the tail swaps on 100/101 (0.80, pos-indep fails); WITH it, it does not.
- **MOAT untouched**: 0 producer calls on abstains (the producer is never invoked, hence never reset, on an abstain).

Which hypothesis won: **H1** (accumulated Izhikevich adaptation), fixed by the correct post-init snapshot-restore. H2 (later read window / baseline-normalize) and H3 (widen the F_MODAL primacy separation) were not needed — the substrate-state reset makes the productions genuinely independent, which is the right property (an utterance should not depend on prior utterances), whereas H3 would only paper over the sensitivity.

## Wire into the flagship console (additive, default-preserving)

`SpikingBrocaConsole.__init__` gains `reset_producer=False`:
- **`reset_producer=False` (default)** → uses `FrameSlotCQ` → byte-identical to the committed EMERGE-60 (its de-risk render-exact 0.93 tail is preserved; the committed JSON + CI are unchanged).
- **`reset_producer=True`** (EMERGE-61) → uses `ResetFrameSlotCQ` → render-exact **1.00 all seeds**.
- The **interactive `_demo` opts in by default** (`--no-reset` to see the tail), so the flagship console renders EMERGE answers EXACT on all seeds; the `_derisk` adds a `--reset` flag (default off, so the committed de-risk output is byte-identical). Import of `ResetFrameSlotCQ` is guarded so EMERGE-60 still loads if EMERGE-61 is absent.

## No regression

- **EMERGE-59 default de-risk: still GO** (order 0.993, exact-slot 0.995, all controls collapse, moat 0). `FrameSlotCQ` byte-unchanged.
- **EMERGE-60 default de-risk: still GO**, render-exact 0.93 tail preserved (byte-identical committed output).
- **EMERGE-60 with `--reset`: GO**, render-exact **1.00** all 6 seeds.
- **EMERGE-60 CI (`test_emerge60_console_spiking_broca.py`): 6 passed.**
- **EMERGE-61 CI: 9 passed.**

## Honest notes

- This closes the **ORDER** tail for the bounded EMERGE frame inventory; render-CONTENT was already 1.00 and is unchanged; open-prose (R4) is the separate deferred wall.
- The reset is validated by **position-independence** (the fact renders identically regardless of prior productions) — it makes the productions genuinely independent, not merely nudged; it is not a metric hack.
- **Pre-existing, out of scope:** EMERGE-60's `_derisk` has a documented harness flakiness where an RNG-sensitive **fluid-path** turn can occasionally flip depending on how many prior EMERGE spiking renders ran (runner comment lines 126-130). This was verified pre-existing in the pristine committed EMERGE-60 (BOUNDARY on ~1/3 runs, GO otherwise) and is unaffected by EMERGE-61's edits. It is a fluid-path RNG issue, NOT the render-order tail EMERGE-61 targets; the fix is orthogonal (isolate the fluid path's RNG stream from the producer's) and left as a named follow-on.
- Reuse-by-import; **NO `sim/` edit** — the reset writes existing bridge arrays via their public attributes (the same `cp_external_input_current[...] =` pattern the producer already uses).

⇒ the flagship console renders EMERGE answers EXACT on ALL seeds; the emergent brain SPEAKS its grounded answers on spikes with a stable, position-independent word order, transformer-retired for those frames.
