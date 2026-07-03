# EMERGE-90 — THE CONVERSATIONAL-TURN CAPSTONE: HEAR → comprehend → store → ASK → SPEAK the answer (SPIKING WORD-ORDER) — **GO** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge90_conversational_turn_capstone_derisk.py`
**Test:** `tests/test_emerge90_conversational_turn_capstone.py`
**Raw:** `research/findings/raw/_emerge90_conversational_turn_capstone.json`
**Scoping:** the read-only capstone-wire scoping (this session) named the exact interfaces + the cheapest rung.
**Adversarial-verify:** a 4-skeptic + adjudicator Workflow probed the GO; every empirical claim SURVIVED (render_exact
genuinely tests the spiking order — monkeypatching the rate ranking flips the surface; recall is a genuine phasor decode;
both lesion controls collapse for the right causal reason; the moat is gate-first, clean to 300 draws). Verdict:
**COMMIT-WITH-FRAMING-FIXES** — no code/metric defect, but strike the "one brain" and "three spiking components"
overclaims. This writeup applies those fixes.

> **SCOPE (up front, per the verdict):** THREE co-executing components in ONE PROCESS — a **RATE** reservoir
> (comprehension), an RF-phasor **spiking** bridge (memory), an Izhikevich **spiking** bridge (production) — with
> **host-dict hand-offs, NO shared bridge, NO cross-synaptic interaction**. This is **NOT "one brain"** in the
> EMERGE-70/71 shared-substrate sense. **Two of the three are spiking**; comprehension is the RATE reservoir (the
> spiking `OnBridgeLSM` swap, EMERGE-89, is the named follow-on). The producer speaks the answer with **spiking
> slot-ORDER**; the **word SURFACES are host-token** (`spell=str(w)`) and 3sg inflection is host `emerge_v3` (the A→W
> neural spell, EMERGE-67/68, is the fully-spiking-words follow-on). The construction `C_TRANS` is **hand-named** (its
> shape is corpus-mined, EMERGE-72; the per-turn message→construction router is not exercised here).

## Why (both halves of conversation, wired into one turn)

The EMERGE arc built the two halves of a conversational turn separately, each self-taught from corpus experience and
spiking: **COMPREHENSION** (the fronto-striatal reservoir, EMERGE-78..89 — learns the who-does-what-to-whom map from
function-word structure, on spikes) and **PRODUCTION** (the self-organized spiking Broca producer, EMERGE-59..77 —
discovers its whole grammar and speaks answers on spikes). EMERGE-90 wires them into ONE conversational turn.

## The turn (the mechanism)

- **HEAR** a transitive sentence "the dog chases the ball".
- **COMPREHEND** — the reservoir (`ReservoirComprehender`, EMERGE-88) parses it into `(agent=dog, action=chases,
  patient=ball)` from the closed-class configuration (content abstracted → roles from STRUCTURE).
- **STORE** — the RF phasor composer stores the fact.
- **ASK** "what does the dog chase?" — the composer recalls the patient (`query_patient`), with the no-confab moat.
- **SPEAK** — the recalled answer drives the self-organized spiking Broca producer (EMERGE-72/74
  `RegistryBrocaProducer` + the `C_TRANS` construction) to SPEAK "the dog chases the ball" ON SPIKES (the frame-slot
  competitive-queuing emission order on a real `SimulationBridge`).
- **GATE-FIRST moat** — an unstored query → the composer abstains → the producer is **never invoked**.

The composer's stored action is the 3sg surface ("chases"); a morphological de-inflection lexicon
(`bare_of = {emerge_v3(v): v}`, the exact inverse of the producer's own inflection over the vocabulary) recovers the
bare lemma the producer re-inflects.

## The de-risk — **GO** (6 seeds; reuse EMERGE-72/74/88 + the production composer; NO `sim/` edit)

| gate | value (6-seed) | bar |
|---|---|---|
| **parse** — reservoir comprehends the heard transitive into (agent, action, patient) | **1.000** | ≥ 0.90 |
| **recall** — composer recalls the patient | **1.000** | ≥ 0.90 |
| **render_exact** — the producer SPEAKS the answer sentence on spikes == the ground-truth transitive | **1.000** | ≥ 0.90 |
| **no-confab MOAT** — unstored query → false-accept rate | **0.000** | ≤ 0.05 |
| — producer invocations on abstain (gate-first) | **0** | == 0 |
| **comprehension-lesion** — reservoir closed-class identity collapsed → render collapses | **0.000** | ≤ 0.30 |
| **producer-no-learn** — the learned spiking ORDER removed → the spoken order collapses | **0.028** (≈ chance) | ≤ 0.60 |

**The result:** the full conversational turn runs end-to-end — HEAR → comprehend → store → ASK → SPEAK the grounded
answer on spikes — 6-seed unanimous, with the no-confab moat holding **gate-first** (0 false-accepts, the producer
**never** invoked on an abstain), and BOTH halves load-bearing: lesioning the reservoir's comprehension collapses the
render (0.000), and removing the producer's learned spiking order collapses the render (0.028 ≈ 1/120 chance for a
5-slot construction). Both halves are self-taught from corpus experience; no hand-written grammar rulebook, no
bolted-on language model.

## Anti-cheats (all pass)

- **Held-out CONTENT** — fresh transitive draws the reservoir read-out never saw; the reservoir abstracts content to
  the OPEN marker, so it generalizes to new words.
- **No-confab MOAT, gate-first** — on any unstored (agent, action), the composer abstains → `decision("ABSTAIN")` →
  `RegistryBrocaProducer.speak` returns WITHOUT running `emit` (`production_count` unchanged, asserted == 0). A
  confabulated fact is never spoken.
- **Comprehension-lesion** — collapsing the reservoir's closed-class identity collapses role-labeling → the stored
  facts are wrong → the render collapses (0.000). The reservoir's comprehension is necessary.
- **Producer-no-learn** — building the producer WITHOUT its learned primacy (the spiking-order teacher) randomizes the
  emission order → the spoken surface scrambles (0.028). The spoken ORDER is genuinely from the spiking producer, not a
  host string-join.

## Honest scope

- **Substrate topology — three co-executing components in ONE PROCESS, NOT one brain.** A **rate** reservoir
  (comprehension), an RF-phasor **spiking** bridge (memory, 12288 neurons), and an Izhikevich **spiking** bridge
  (production, ~184 neurons). Hand-offs are **host-Python dict passes**, not spikes/synapses — there is **no shared
  bridge and no cross-synaptic interaction**. Per the project's own definition (`project_one_brain_substrate_vs_functional`:
  "co-location w/ zero cross-synapses isn't one-brain; real = cross-region synaptic interaction"), this is **NOT "one
  brain"**; folding all three onto ONE bridge (the EMERGE-87 disjoint-slice pattern) — and then making the hand-offs
  synaptic — is the follow-on. "One brain" is reserved for the EMERGE-70/71 shared-cupy-bridge sense, which this is not.
- **Two spiking + one rate.** The RF composer and the Izhikevich producer are spiking; **comprehension is the RATE
  reservoir** (EMERGE-78 numpy `tanh` echo-state). EMERGE-89 already proved the on-bridge **spiking** `OnBridgeLSM`
  drives the composer identically, so the spiking-comprehender capstone is the **mechanical swap** (the named follow-on).
- **The ORDER is spiking; the WORD SURFACES are host-token.** The producer's frame-slot competitive queuing produces
  the emission ORDER on a real bridge (the load-bearing "spiking" claim, isolated by the producer-no-learn control at
  0.028 ≈ chance), while each word's surface string is the host-token identity (`spell=lambda w: str(w)`) and the 3sg
  inflection is host `emerge_v3`. Spelling every WORD on spikes is the validated A→W read-out (EMERGE-67/68/69), which
  forces cupy — the **fully-spiking-words follow-on** (the whole turn then co-executes on one cupy process, EMERGE-70/71).
- **The construction `C_TRANS` is hand-named** by the turn; its SHAPE is corpus-mined (EMERGE-72), but the per-turn
  message→construction router is not exercised here (a transitive fact is spoken as the transitive construction). The
  3sg inflection is the previously-named EMERGE-64 host residual.
- **render/recall coupling (minor, disclosed):** `render_exact` is bounded above by `recall` — only the OBJECT slot is
  recall-dependent; the subject/verb fed to the producer are the parsed/ground-truth agent/action. Render still adds
  genuine information over recall (the 5-slot **spiking order** + the bare→3sg inflection, neither tested by recall).
  Driving the producer's subject/verb from the RECALLED agent/action too (closing the residual coupling) is a
  non-blocking rigor follow-on.
- **The vocab filter** (test content restricted to words the reservoir sees as genuinely OPEN) isolates the WIRE from
  EMERGE-62's closed-class-discovery precision (a separately-characterized property); with the unfiltered vocab the
  turn still GOes (parse/recall/render 0.958 at seed 42 — one word-collision miss).
- Reuse-by-import (EMERGE-72/74 registry producer + EMERGE-88 comprehender + the production composer); NO `sim/` edit.

## The arc, closed into a turn

Comprehension (reservoir, EMERGE-78..89) + production (self-organized spiking producer, EMERGE-59..77) now compose
into ONE conversational turn: **hear a fact → understand it → remember it → answer a question about it by speaking a
grounded sentence with spiking word-order**, self-taught end-to-end, moat-first. The clear follow-on ladder (each a
mechanical swap of an already-GO piece): fold the three bridges onto ONE + make the hand-offs synaptic (EMERGE-87
pattern) → the honest "one brain"; the A→W neural word-spell (fully-spiking words, cupy, EMERGE-67/68/69); the
spiking-reservoir comprehender in the capstone (EMERGE-89 `OnBridgeLSM` swap); richer question types + a per-turn
message→construction router.

## Files
- `research/runners/_emerge90_conversational_turn_capstone_derisk.py` — the wire + the 6-seed de-risk (parse / recall /
  render_exact / gate-first moat / comprehension-lesion / producer-no-learn).
- `tests/test_emerge90_conversational_turn_capstone.py` — 3 CPU tests.
- `research/findings/raw/_emerge90_conversational_turn_capstone.json` — the 6-seed turn.
