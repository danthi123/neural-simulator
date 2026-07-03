# EMERGE-66 — the flagship console renders from a FULLY-SELF-ORGANIZED producer (GO, 6-seed)

**Date:** 2026-07-03
**Verdict:** GO (6 seeds 42/43/44/100/101/102, CPU/numpy). A WIRE (composition of GO pieces); no new mechanism.
**Reuse-by-import; NO `sim/` edit. The ONLY change to committed code is an additive default-off flag.**

## What this is

Wire the fully-self-organized producer (EMERGE-65 `SelfOrganizedProducer`) into the flagship console
(EMERGE-60 `SpikingBrocaConsole`) via an **ADDITIVE default-OFF `self_organized` flag** (mirroring EMERGE-61's
`reset_producer` flag), so the flagship renders its EMERGE answers from a producer whose **ENTIRE grammatical
structure was DISCOVERED FROM THE CORPUS** — the function-word inventory (S2, EMERGE-62), the per-construction
slot inventory (S1a, EMERGE-64), and the slot order (S1b, EMERGE-63) — **NOT the host `FRAMES` dict**.

This completes the emergent-Broca arc: the emergent brain discovers categories from experience → reasons →
and now **SPEAKS its grounded EMERGE answers on spikes FROM A SELF-ORGANIZED GRAMMAR**, on the flagship
console, transformer-free.

## The wire (files)

- **`research/runners/_emerge60_console_spiking_broca_derisk.py`** — the ONLY committed-code change: an additive
  default-off `self_organized` (+ `self_organized_n_sentences`) flag on `SpikingBrocaConsole.__init__`.
  - `self_organized=True` → builds `SelfOrganizedProducer(seed).build_from_corpus(build_stream(seed))` and sets
    `self.broca = sop.producer(spell=spell)` (a `BrocaProducer` over the corpus-mined `MinedInventoryFrameSlotCQ`),
    `render_kind="self_organized_broca"`. The self-organized CQ subclasses `CorpusOrderFrameSlotCQ → ResetFrameSlotCQ`,
    so it is **position-independent by construction** (the EMERGE-61 wash-out is subsumed; `reset_producer` is True).
  - `_render_emerge` is UNCHANGED — it already routes `self.broca.speak(decision)`, so both paths flow through it.
  - Default `self_organized=False` == EMERGE-60 byte-identical (host-FRAMES `spiking_broca` path). Guarded imports
    of EMERGE-65/62 so EMERGE-60 still loads if they are absent.
- **`research/runners/_emerge66_console_self_organized_derisk.py`** — the de-risk (`--demo` / `--derisk`).
- **`tests/test_emerge66_console_self_organized.py`** — 9 CI tests (CPU/numpy, offline).
- **`research/findings/raw/_emerge66_console_self_organized.json`** — the raw 6-seed result.

## 6-seed de-risk results (`--derisk --seeds 42 43 44 100 101 102`)

Every seed identical:

| metric | value (all 6 seeds) |
|---|---|
| render-words (content routed to the self-organized producer) | **1.00** |
| render-exact (word order, over the wash-out) | **1.00** |
| F_NEGMOD "the penguin does not fly" (direct through the producer) | **exact, all seeds** |
| gate-first moat — producer calls on abstains | **0** |
| membership routing ("can a dog eat?" → fluid, not falsely denied) | **True** |
| fluid no-regression (Broca-free baseline) | **True** |
| self-organized provenance (struct-match vs permuted-corpus) | **1.00 vs 0.33** |

Sample transcript (self-organized console):

```
you> can an owl fly?      brain> the owl can fly       [INHERIT; self-organized producer INVOKED]
you> can a penguin fly?   brain> the penguin walks     [CANCEL;  self-organized producer INVOKED]
you> can a robin breathe? brain> the robin can breathe [PER-DIMENSION inherit; INVOKED]
you> can an owl swim?     brain> I don't know whether an owl can swim.  [SIBLING-abstain; producer NOT invoked]
you> can a zzz fly?       brain> I don't know.         [MOAT; producer NOT invoked]
you> can a dog eat?       brain> The dog eats meat.    [MEMBERSHIP -> fluid, not falsely denied]
(F_NEGMOD direct)         brain> the penguin does not fly  [DENY negated-modal; self-organized producer]
```

The discovered function words (seed 42, from the corpus) include the/a/can/does/not (+ other high-freq
context-covering tokens); the assembled structure matches the host FRAMES 1.000 / inventory-accuracy 1.000.

## The four de-risk axes (from the task) + provenance

- **(a) render-CONTENT 1.00 on spikes** from the self-organized producer; render-EXACT (order) 1.00 all seeds
  (the self-organized CQ washes out per emit → position-independent, so the EMERGE-60 order tail does not arise).
  F_NEGMOD exercised directly (the console's EMERGE-54 reasoner only emits affirm/negate, not `negated_modal`).
- **(b) gate-first MOAT** — 0 producer-calls on abstains (sibling + unknown), on the self-organized console.
- **(c) MEMBERSHIP routing preserved** — a fluid-known entity in the shared ability frame is answered by the
  fluid path (not falsely denied), producer not stolen in.
- **(d) NO fluid-path REGRESSION** — tested on a **Broca-FREE baseline** (a plain `UnifiedFluentConsole`,
  re-seeded per seed) per the EMERGE-60 harness note, so the producer's bridge-sim RNG consumption cannot flake
  the fluid gate (the known EMERGE-60 fluid-RNG flakiness). The fluid dispatch is inherited byte-identical on
  both paths, so "no regression" is STRUCTURAL. No spurious BOUNDARY from the known flakiness.
- **(e) SELF-ORGANIZED provenance** (extra, to prove the wire is genuinely self-organized, not a silent host-FRAMES
  fallback): the console's producer structure MATCHES the host FRAMES (struct-match 1.00, inv-acc 1.00, all frame
  function words discovered) AND the PERMUTED-CORPUS control COLLAPSES it (struct 1.00 vs perm 0.33, margin ≥ 0.30).

## Default-path byte-identity (verified)

The flag is additive / default-preserving; the committed default path is unchanged:

- **EMERGE-60 de-risk** (`--derisk --seeds 42 43 44`): GO, render-words/exact 1.00, moat 0, membership + fluid pass.
- **EMERGE-61 de-risk** (`--derisk --seeds 42 43 44 100 101 102`): GO, FIX exact 1.00 all seeds, position-independent,
  causal control swaps (seeds 100/101), moat 0.
- **EMERGE-59..65 CI** (`test_emerge59..65`): 63 passed.
- **EMERGE-66 CI** (`test_emerge66_console_self_organized`): 9 passed (incl. `test_default_path_byte_identical_to_emerge60`).

## Honest scope + carried-forward residuals

- Renders the **BOUNDED EMERGE frame inventory** (ability-affirm / intransitive-exception / negated-modal) on
  spikes from a self-organized grammar — **NOT open prose** (R4, the separate deferred wall).
- EMERGE-65's carried-forward residuals are inherited, named not hidden: a HELD-OUT frame's DISTINCTIVE
  function-word slots (F_MODAL's `can` / F_NEGMOD's `does`/`not`) + F_INTR's 3sg inflection + F_NEGMOD's `does<not`
  internal order are NOT recoverable from the OTHER two frames alone (only that frame attests them). The MAIN arm
  (all frames' exemplars) mines every inventory EXACTLY — which is what the console uses.
- The A→W spell is the pluggable token-surface callback (its own spiking validation is `concept_speak_demo`).
- The gate-first moat is untouched (0 productions on abstains, by construction).

## Verdict

**GO.** The flagship console renders its EMERGE answers on spikes from a fully-self-organized (corpus-mined)
producer, with the gate-first moat + membership routing + fluid paths all intact, self-organized provenance
asserted, 6-seed. The default path stays byte-identical (EMERGE-60/61 de-risks + EMERGE-59..65 CI unchanged).
NO `sim/` edit; the only change is the additive default-off `self_organized` flag on EMERGE-60's
`SpikingBrocaConsole`.
