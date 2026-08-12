---
type: finding
status: contributing
date: 2026-08-12
mechanism: production-integration — the /api/brain-chat production default is now GENUINELY SPIKING (composer_kind=onebrain)
lane: integration-first (#0 one-brain — FLIPPED to default, HTTP-verified)
integration_faculty: one-brain-substrate
verdict: DONE (the #0 one-brain default flip). The production webapp chat (/api/brain-chat, tiny-demo brain) now builds composer_kind="onebrain" BY DEFAULT — the recall runs the resonate-and-fire step per query on the co-resident RF substrate (the on-substrate cleanup + store), NOT the numpy `_scan_first_match` fast path. Enabled by the recruit-an-assembly fix (OneBrainComposer vocab_headroom, so a fact taught mid-conversation is laid down + recalled on the SPIKING store). HTTP-verified on the real endpoint: CHOOSE recall "what does dog chase?"->"The dog chases cat." (composer='onebrain', 45730 readout neurons, frac_fired 1.7%); abstain "what does fish fly?"->"I don't know" (frac_fired ~0); LEARN teach "wolf hunt deer"->"The wolf hunts deer." then recall (matched_fact_index=5, n_facts_scanned=6); rich GENERATE "tell me about dog"->"The dog chases cat. The cat eats fish." (2-fact associative chain via the spiking elaborate). BRAIN_COMPOSER_KIND=rf is the escape to the numpy fast path; the TUI/smoke _build_tiny_demo default stays 'rf' (fast GPU-free smoke). Cost: ~183s one-time warm build (speed secondary per the mission). Scaffold NOT retired (the rf numpy path remains the CPU/test oracle + escape).
artifacts:
  - webapp/server.py
  - research/runners/brain_chat_tui.py
  - docs/PRODUCTION_INTEGRATION_LEDGER.yaml
verification: HTTP POST to http://127.0.0.1:8010/api/brain-chat (webapp started with the onebrain default): the four acts above, every response tagged composer='onebrain' with a live RF activity trace (frac_fired>0 on matches, ~0 on abstains). No webapp test regresses (test_webapp_server.py POSTs no brain-chat turn).
---

# The production chat recall is now GENUINELY SPIKING by default — the #0 one-brain flip

## What changed and why it matters

The owner's goal is a working all-spiking one-substrate brain with **all faculties ON BY DEFAULT in production** — "when
I run the brain and chat with it, it should be fully functional in all its capabilities." The recall/store faculty was
the load-bearing one still running the NUMPY fast path in the default chat (`RFPhasorComposer`,
`enable_substrate_store=False`, `_scan_first_match` = np. masking, NOT `_resonate`). It answered correctly but the
*substrate was not doing the work* — a brain-based-only shortcut.

<!--derived-->
As of this session the production webapp chat (`/api/brain-chat`, the tiny-demo brain) builds **`composer_kind="onebrain"`
by default** (`webapp/server.py`, `BRAIN_COMPOSER_KIND` defaults to `"onebrain"`; `_build_tiny_demo` gained a
`composer_kind` param). The recall now runs the **resonate-and-fire** step per query on the co-resident RF substrate,
with the on-substrate matched-filter cleanup + the weight-store. This was unblocked by the recruit-an-assembly fix
(`2026-08-12-onebrain-spiking-store...RESOLVED`): a fact taught in the conversation is laid down + recalled on the
SPIKING store (previously it stored but never recalled).

## HTTP verification (the real endpoint, not a probe)

The webapp was started with the onebrain default and POSTed the four owner-visible acts. Every response is tagged
`composer='onebrain'` and carries a live RF activity trace:

- **CHOOSE recall** — "what does dog chase?" -> "The dog chases cat." (`recalled_svo=[dog,chase,cat]`, `verified=true`,
  45730 readout neurons, `frac_fired=0.017`, `mean_magnitude=6285`).
- **CHOOSE abstain** — "what does fish fly?" -> "I don't know about that." (`abstained=true`, `frac_fired~0.0002` — the
  substrate finds no match, the moat holds).
- **LEARN** — teach "wolf hunt deer" -> "The wolf hunts deer."; recall "what does wolf hunt?" -> "The wolf hunts deer."
  (`matched_fact_index=5`, `n_facts_scanned=6` — the taught fact is the 6th, recalled on the spiking store).
- **GENERATE (rich)** — "tell me about dog" -> "The dog chases cat. The cat eats fish." (`n_sentences=2`,
  `supporting_facts=[[dog,chase,cat],[cat,eat,fish]]` — the associative chain via the spiking `elaborate`).

## Honest scope (what this is and is NOT)

This flips the RECALL/STORE mechanism to genuinely spiking by default — one faculty, the load-bearing one. It is NOT
"the whole brain is now spiking on by default": the rich/GENERATE multi-fact path is still behind the `rich` toggle (not
default-on), the host QuestionRouter remains the self/identity + anaphora fallback, the off-bridge stub/Qwen still
renders the surface, and a BTSP/plasticity per-turn write (a lasting trace beyond the RF store) is the deeper LEARN. The
rf numpy path is NOT retired — it stays the CPU/test oracle and the `BRAIN_COMPOSER_KIND=rf` escape. Cost: ~183s one-time
warm build (the onebrain bridge; speed is secondary per the mission — faithfulness is not traded for it). Next: flip the
rich/GENERATE default + continue the faculty-by-faculty wiring toward all-on-by-default.
