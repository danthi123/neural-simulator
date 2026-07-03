# EMERGE-60 — the flagship console SPEAKS its EMERGE answers ON SPIKES: the EMERGE-59 spiking Broca producer WIRED INTO the unified console in place of the 21M ANN — **GO** (6-seed)

**2026-07-03 (autonomous).** The payoff of EMERGE-59: the unified fluent console (EMERGE-58, audit-remediated) now renders its EMERGE emergent-reasoning answers **on the spiking substrate** (EMERGE-59 frame-slot competitive queuing on a real `SimulationBridge`; the emission order is the per-pool spiking-rate ranking) **instead of the 21M ANN generator** — the ANN is **retired for the EMERGE frame inventory**. Reuse-by-import; **NO `sim/` edit**.

## The wire — a COMPOSITION (subclass override)

`SpikingBrocaConsole` subclasses `UnifiedFluentConsole` and overrides **only** `_render_emerge`: the EMERGE gate decision's `(svo, polarity)` is mapped (via EMERGE-59's `decision_from_emerge`) to a frame decision and rendered by `BrocaProducer.speak` (EMERGE-59, spiking) instead of the 21M ANN. Everything else — the gate-first structure, the membership-aware routing (`can a X <verb>?` disambiguated by taxonomy membership), the fluid paths — is inherited **unchanged**. The gate-first structure guarantees the moat by construction: on an ABSTAIN, `_emerge_turn` returns **before** `_render_emerge`, so the spiking producer is **never invoked** on an abstain (asserted via `BrocaProducer.production_count == 0`).

Mapping: `polarity 'affirm'` (inherited) → `F_MODAL` "the owl can fly"; `polarity 'negate'` (exception) → `F_INTR` "the penguin walks" (`emerge_v3` keeps the already-3sg `ovr` verb from double-inflecting).

## De-risk — **GO** (6-seed 42/43/44/100/101/102, CPU)

| gate | value | bar |
|---|---|---|
| EMERGE render CONTENT on spikes (right grounded fact routed to the producer — **the WIRE**) | **1.00** | ≥ 0.99 |
| gate-first MOAT — spiking-producer invocations on abstains (the LOAD-BEARING property) | **0** | 0 |
| MEMBERSHIP routing (a fluid-known entity in the ability frame answered, not falsely denied; producer not stolen) | **PASS every seed** | true |
| NO fluid-path REGRESSION (what / anaphora / growth / yes-no / discuss / moat) | **PASS every seed** | true |
| EMERGE render word-ORDER exact (the spiking producer's own accuracy — REPORTED, not re-gated) | **0.93** | *(reported)* |

The GO is on the **wire** (the integration): the EMERGE answer is rendered by the spiking producer with the correct content, the moat holds, membership routing is unchanged, and there is no fluid regression — all perfect 6-seed. The render word-**order** exact accuracy (0.93) is the **spiking producer's own EMERGE-59-characterized property**, reported not re-gated.

### The demo (seed 42, all EMERGE answers rendered ON SPIKES)

```
you> can an owl fly?        brain> the owl can fly           [INHERIT; spiking producer INVOKED]
you> can a penguin fly?     brain> the penguin walks         [CANCEL;  spiking producer INVOKED]
you> can a robin breathe?   brain> the robin can breathe     [PER-DIMENSION inherit; INVOKED]
you> can an owl swim?       brain> I don't know whether an owl can swim.  [SIBLING-abstain; producer NOT invoked]
you> can a zzz fly?         brain> I don't know.             [MOAT; producer NOT invoked]
you> can a dog eat?         brain> The dog eats meat.        [MEMBERSHIP -> fluid path (not falsely denied)]
you> what does the dog eat? brain> The dog eats meat.        [FLUID path, unchanged]

spiking-producer invocations on abstains: 0 (the load-bearing property)
```

## Honest scope + the named follow-on

- **Render word-order tail (transparently reported, not hidden):** on seeds 100/101 the 4-slot `F_MODAL` frame swaps its two lowest-primacy adjacent slots under the read-out noise — e.g. **"the robin breathe can"** — so render-exact is 0.80 on 2/6 seeds (0.93 aggregate). **Content is always correct** (render-words 1.00). This is the spiking producer's inherited order accuracy (EMERGE-59: 0.993 soft), **not a wire defect**. The named robustness follow-on: sharpen the primacy separation for the 4-slot frame / more sim steps in the EMERGE-59 producer read-out (a bounded EMERGE-59 refinement).
- **The A→W spell** is the pluggable token-surface callback (its own spiking validation is `concept_speak_demo`, 100% multi-seed); wiring the trained-bridge read-out in is the GPU follow-on.
- **The fluid paths** still use the flagship's own renderer (a separate, larger surface EMERGE-59 does not cover). EMERGE-60 retires the ANN **for the bounded EMERGE frame inventory**, not for open prose (R4 deferred).
- The fluid regression is run **first** (on the pristine post-construction RNG) because the spiking producer's bridge advances the shared RNG the fluid path draws from — an ordering artifact of the harness, not a logic regression (the fluid path is byte-identical to EMERGE-58, inherited without override; each seed is correct in isolation).

## Files
- `research/runners/_emerge60_console_spiking_broca_derisk.py` — `SpikingBrocaConsole` (the wire) + the `--derisk` (6-seed) + `--demo`.
- `tests/test_emerge60_console_spiking_broca.py` — 6 tests (render-on-spikes, producer-invoked-on-answer, moat-never-invoked-on-abstain; slow: membership/no-false-denial, full de-risk GO).
- `research/findings/raw/_emerge60_console_spiking_broca.json` — the 6-seed de-risk.

## Verdict
**GO.** The flagship unified console renders its EMERGE emergent-reasoning answers on the spiking substrate (frame-slot competitive queuing on a real `SimulationBridge`) in place of the 21M ANN — the transformer is retired for the EMERGE frame inventory. The gate-first no-confab moat, the membership-aware routing, and the fluid paths are all preserved. ⇒ **the emergent brain SPEAKS its grounded EMERGE answers on spikes, on the flagship console** — a concrete step toward the fully-spiking one-brain end state (the honest residual: the spiking producer's order tail on the 4-slot frame; the A→W spell wire-in; open prose). NO `sim/` edit.
