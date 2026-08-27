---
status: live
type: finding
lane: integration
mechanism: spiking-broca-recall-mouth
date: 2026-08-26
---

# Spiking Broca mouth extended to the RECALL / RICH answer surface — production wire-in (GO, 6-seed, wired default-OFF)

**Date:** 2026-08-26
**Status:** GO — wired (default-OFF), 6-seed flip-soak clean; ready for the parent's default-ON flip after the pool soak.
**Faculty:** spiking-broca-mouth (recall/rich half). Burns down the RECALL half of the mouth scaffold for the bounded transitive-SVO case.
**Organ:** `research/runners/spiking_mouth_recall_prod.py` (REUSE-BY-IMPORT, NO `sim/` edit)
**Wiring:** `research/runners/brain_chat_tui.py` (`ChatBrain.spiking_recall_surface` + guarded branch in `ChatBrain.render`); `research/runners/rich_answer_composer.py` (guarded branch in `RichAnswerComposer._render_one_verified`)
**Flag:** `BRAIN_SPIKING_MOUTH_RECALL` (default OFF)
**Soak:** `research/runners/_spiking_mouth_recall_soak.py`
**Raw:** `research/findings/raw/_spiking_mouth_recall_soak.json`
**De-risk lineage:** EMERGE-59/61 (`_spiking_fluent_surface_derisk` · `_emerge61_spiking_broca_order_robustness_derisk`), the same spiking Broca the GENERATE channel already renders hypotheses on.

## What this does

The GENERATE channel already renders a structured (hedged-transitive) HYPOTHESIS on the spiking Broca by default
(`brain_chat_tui.render_hypothesis_verified` / `_render_hypothesis_spiking`, flag `BRAIN_SPIKING_MOUTH` default ON,
EMERGE-59/61 6-seed GO): word order = the per-pool spiking-RATE ranking on a real Izhikevich `SimulationBridge`.

This wire-in carries that SAME spiking mouth to the **ASSERTED RECALL / RICH answer surface** — the surface that
today is authored by the off-bridge Qwen / template-stub. A grounded recalled SVO `(a, v, p)` in the bounded
transitive-SVO frame inventory is now rendered "the &lt;S&gt; &lt;V-3sg&gt; the &lt;O&gt;" **on firing neurons**
(the 5-slot `PLAIN_TRANSITIVE` frame, EMERGE-61 inter-utterance wash-out), re-parse VERIFIED against the recalled
SVO, replacing the Qwen/template surface. Both production recall entry points inherit it: the single-fact
`ChatBrain.render` (server.py `/api/brain-chat` line ~4644 / ~5429) and the multi-sentence
`RichAnswerComposer._render_one_verified` (the FLUENT production default, server.py line ~5207).

**Scope-guard (open prose stays Qwen — owner-sanctioned scaffold):** only a bounded transitive SVO (single-word
alphabetic roles, subject ≠ object, non-copula verb) whose spiking render **re-parse VERIFIES** to the exact
recalled `(a, v, p)` is ever routed on spikes; every other fact — open/multi-word prose, copula/attribute facts,
an irregular verb the round-trip cannot recover — falls straight through to the current mouth, **byte-identical**.
The moat (recalled CONTENT) never weakens: the spiking surface either carries EXACTLY the recalled SVO or is unused.

## Flag naming (reconciliation for the merge)

The faculty spec named the flag `BRAIN_SPIKING_MOUTH (default OFF)`, but that env var **already exists** and is
**default-ON** (it gates the GENERATE-channel hypothesis mouth). Reusing it default-OFF would flip the generate
channel off and would not be flag-OFF-byte-identical. So this wire-in introduces a **NEW** flag,
`BRAIN_SPIKING_MOUTH_RECALL` (default OFF), for the recall/rich extension, and leaves the generate-channel
`BRAIN_SPIKING_MOUTH` untouched. To honor the literal lesion oracle, the **master mouth kill-switch
`BRAIN_SPIKING_MOUTH=0` also forces the recall surface OFF** — so "lesion `BRAIN_SPIKING_MOUTH` → the recall SVO
surface reverts to Qwen/template" holds directly. The parent flips `BRAIN_SPIKING_MOUTH_RECALL` default-ON after
the pool soak (or folds it into `BRAIN_SPIKING_MOUTH`).

## The three proven properties (`_spiking_mouth_recall_soak`, 6 seeds 42/43/44/100/101/102, CPU/numpy)

| seed | GO | byte-identical(OFF) | load-bearing | no-regression | spiking-authored facts |
|------|----|--------------------:|:------------:|:-------------:|:----------------------:|
| 42   | ✓ | ✓ | ✓ | ✓ | 15/17 |
| 43   | ✓ | ✓ | ✓ | ✓ | 15/17 |
| 44   | ✓ | ✓ | ✓ | ✓ | 15/17 |
| 100  | ✓ | ✓ | ✓ | ✓ | 15/17 |
| 101  | ✓ | ✓ | ✓ | ✓ | 15/17 |
| 102  | ✓ | ✓ | ✓ | ✓ | 15/17 |

**VERDICT: GO (6/6).**

**(1) FLAG-OFF BYTE-IDENTICAL** — asserted in the data by exact string compare: with the flag unset (default OFF),
`chat.render(svo)` for every stored transitive fact equals the template-stub's own verified surface exactly, and
no spiking-form surface leaks. The added code is a guarded early-return; flag OFF never enters it.

**(2) LOAD-BEARING (the lesion oracle)** — two independent lesions, both asserted in the data:
- *flag lesion*: flag ON → `the brain uses the spikes` (authored on spikes); flag OFF → `The brain uses spikes.`
  (Qwen/template). The word ORDER / surface CHANGES while the recalled CONTENT SVO is byte-identical (the rich
  transcript's `facts` are `==` across OFF and ON on all 6 seeds).
- *rate-read lesion*: intact per-pool spiking-RATE ranking → the CORRECT canonical order (`the brain uses the
  spikes`); the EMERGE anti-cheat `equal_drive` (rates tie) → a DIFFERENT order (`uses brain the spikes the`) with
  the **identical content-word multiset**, on every load-bearing fact of all 6 seeds. Attribution
  (`tools.lab.attributable_to`, canonical-order match): **100% of the correct-order effect is owned by the
  spiking-rate read** (intact canonical-order rate 1.0, equal-drive control 0.0) — the read **authored** the order,
  not a fixed host template. *Instrument note:* the attribution is measured on **exact canonical-order match**, not
  on the re-parse VERIFY — the independent grammar parser is somewhat order-tolerant, so on 2/6 seeds a scrambled
  equal-drive order coincidentally still re-parses; the canonical-order match is the clean read of "did the rate
  ranking author the ORDER" and separates cleanly on all 6. In production a scramble that does not re-parse simply
  falls back to the current mouth (never a leak).

**(3) NO-REGRESSION** — a mouth wire-in must change the SURFACE ONLY, never the CONTENT. Running the SAME live rich
conversation with the flag OFF vs ON gives **content-equivalence per turn** (identical abstain, identical sentence
count, identical supporting facts) while the ON surface differs on ≥1 turn (the spiking mouth genuinely
re-authored the surface); the no-confab moat still ABSTAINS on the untaught/general cues (incl. "capital of
France") with the flag ON.

Coverage is 15/17 of the smoke's stored transitive facts (honest residual: the 2 misses are the irregular
`have→has` and an already-3sg lemma `needs→needses` whose round-trip the independent parser rejects → they fall
back to the template mouth, visible in the ON transcript as the one `The neurons haves synapses.` sentence sitting
among the spiking-form sentences — the scope-guard working as designed).

## Honest scope / residual

- This routes the **RECALL/RICH** surface for the **bounded transitive-SVO frame inventory ONLY**. Open arbitrary
  prose the frames cannot cover keeps the Qwen fallback (the documented A1 open-prose residual, owner-sanctioned).
- **Wired, default-OFF** (`gates/production_integration` level "wired"): reachable from `/api/brain-chat` →
  `ChatBrain` → the organ, but not yet on-by-default. NOT "integrated" / "closed" until the parent flips the
  default and the scaffold is retired for the bounded case. The flip is the parent's, post pool-soak.
- `sim/` untouched; the producer, frame, and wash-out are imported verbatim from the EMERGE-59/61 de-risk; the
  re-parse VERIFY is the ChatBrain's own `_verify` (the identical moat the Qwen recall path uses).

⇒ For the bounded transitive-SVO frame inventory, the emergent brain now SPEAKS its GROUNDED recalled answers on
spikes — word order authored by the per-pool spiking-rate ranking on a real Izhikevich substrate — with the
transformer retired for that surface, the no-confab moat intact, and open prose byte-identical.
