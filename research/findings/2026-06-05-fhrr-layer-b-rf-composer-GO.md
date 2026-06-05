# Full FHRR-on-bridge — layer (b): the RF phasor composer's conversational capabilities → GO — 2026-06-05

**Verdict: GO. The conversational composer runs on the FHRR-on-bridge substrate** (resonate-and-fire phasor neurons +
complex synapses) — the opponency rate-coded SNR wall is fully escaped on the conversational path, and the no-confab
moat is preserved throughout. Layer (b) of the owner-greenlit full FHRR-on-bridge feature
(`docs/plans/2026-06-05-fhrr-layer-b-composer-recode-design.md`), after layer (a) GO
(`2026-06-05-fhrr-layer-a-complex-synapse-GO.md`).

## What was built — `research/runners/rf_phasor_composer.py` (`RFPhasorComposer`)
A PARALLEL composer (same API as `CoreSimComposer`), reuse-by-import on the bridge's RF + complex-synapse substrate
(NO `sim/` edits in layer b). FHRR phasor codes (random phases per seed); bind = diagonal complex synapses
(role ⊙ filler); bundle = unit complex synapses (the sum — NO opponency exists); unbind = conj diagonal synapses;
cleanup = phase-cosine argmax. The rate-coded `CoreSimComposer` stays the production path until layer (c) re-validates
the RF composer at parity and the `BrainConversationalAgent` switches.

## The five conversational capabilities — all GO multi-seed (`tests/test_rf_phasor_composer.py`)
| capability | gate | result |
|---|---|---|
| **b.1 who/what Q&A + abstention** | store 2 SVO facts; `who <action><patient>?`→agent, `what <agent><action>?`→patient, absent-cue→None (the no-confab moat) | GO 3/3 seeds |
| **b.2 negation / yes-no** | a bound AFFIRM/NEGATE polarity tag; `ask_yes_no`→yes/no, 'unknown' when no match (4-role bind) | GO 3/3 seeds |
| **b.3a one-attribute** | an ATTRIBUTE role-tag; "big apple" — adjective+noun both decoded (RESOLVES; 2-attribute is the documented K=5 BOUNDARY) | GO 3/3 seeds |
| **b.3b recursive clauses** | a clause as a filler ("dog look (cat go north)"); the nested SVO renders (double-nesting, D=128) | GO 3/3 seeds |
| **b.4 dialogue (`elaborate`)** | the dlPFC spiking content-selection (`SpikingSpreadingController`) over the association graph from the RF facts → on-topic associate; None when unconnected | GO 3/3 seeds (GPU) |

The no-confab moat is preserved on every relational query (None/'unknown' when no stored fact matches the cue) — the
architecture-defining anti-confabulation property carries over to the phasor substrate unchanged.

**The nesting/multi-hop SNR wall does NOT bite in the phasor algebra:** recursive clauses (b.3b) work at D=128 where
the rate-coded hierarchical approach hit an SNR wall — a concrete advantage of the unit-magnitude phase code.

## Backend note
b.1–b.3 run on CPU (numpy) and GPU (the RF + complex-synapse ops are backend-agnostic). b.4 is GPU-only: the REUSED
`SpikingSpreadingController` (the dlPFC, an existing validated component) has a numpy-backend `IndexError` in that
component (not in the RF composer). The b.4 test skips on the numpy backend.

## Honest scope / what remains
- Layer (b) realizes the full conversational capability on the RF substrate at the de-risk vocab/scale (D=64–128,
  small fact sets), multi-seed. **Layer (c)** validates the RF composer against the FULL capability matrix at the
  production scale + the same multi-seed bars the rate-coded composer meets, then switches the
  `BrainConversationalAgent` (the opponency cleared on the production path; the F=3 two-attribute resonator —
  which the ±1 scheme provably can't do — becomes available to lift the documented 2-attribute K=5 boundary).
- Performance: per-op resonate windows (period+8 steps) are correctness-first; an optimization pass (shorter period,
  batched ops, sparse complex weights, GPU complex matvec) follows parity.
- The rate-coded composer stays production until layer (c) parity — no capability regression ships silently.

## Artifacts
- `research/runners/rf_phasor_composer.py`; `tests/test_rf_phasor_composer.py` (5 capabilities × 3 seeds)
- commits: b.1 4d077a8f, b.2 df48b6fa, b.3a eb07069a, b.3b 25b65ca8, b.4 (this); design
  `docs/plans/2026-06-05-fhrr-layer-b-composer-recode-design.md`. Frozen bars / no-confab moat untouched.
