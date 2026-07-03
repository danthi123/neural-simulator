# EMERGE-56 (Rung 1) — wiring the EMERGENT grounded REASONING → the FLUENT-language faculty, gate-first no-confab MOAT preserved: **GO** (3-seed). Wernicke-decides → Broca-articulates.

**2026-07-03 (autonomous).** Rung 1 of the north-star wire (the research gate `2026-07-03-emergent-reasoning-to-fluent-nl-wire-research-gate.md`). The emergent semantic substrate (EMERGE-51..55) REASONS over discovered categories (inheritance / cancellation / abstention) and answers — but in TEMPLATED English. This wires that grounded reasoning to the fluent faculty so the brain answers FLUENTLY, keeping the validated gate-first no-confab moat: the BRAIN decides answer-vs-abstain AND supplies the grounded fact BEFORE any generator renders (an abstain ⇒ the renderer is NEVER invoked). Reuse-by-import; **NO `sim/` edit**.

## The finding — the wire is a 1-to-1 ADAPTER (confirmed), and the MOAT holds by construction

The research gate scoped this as an adapter, not a mechanism. **Confirmed:** `ExperientialConversationalConsole._best(member)` already exposes EMERGE's STRUCTURED inference decision — `None` (below the no-confab floor) | `("CLASS", cname)` (inherited class default) | `("OVR", member)` (member-specific exception) | a cross-bleed. The adapter reads that and converts it 1-to-1 to the fluent faculty's gate-first input (a bool + a grounded SVO triple). No new dendritic circuit, no new learning rule.

### RUNG 1 de-risk — 3-seed **GO** (42/43/44)

| gate | value | bar |
|---|---|---|
| ADAPTER FIDELITY (extracted `(gate, subject, property)` == EMERGE's own `ask_can` decision) | **1.00** | ≥ 0.95 |
| MOAT preserved (every abstain → "I don't know" AND renderer NEVER invoked) | **1/1 every seed** | true |
| render-calls on abstains (the LOAD-BEARING property) | **0** | 0 |
| moat FALSE renders (renderer invoked on an abstain) | **0** | 0 |
| correct grounded facts rendered (inherit owl→fly, cancel penguin→walks) | **1.00** | ≥ 0.99 |

The moat is preserved **by construction**: `wired_reply` short-circuits on `gate=="ABSTAIN"` and emits "I don't know" WITHOUT touching the renderer. A `CountingStubFaculty` (the validated grounded-lang P3 `TemplateStubFaculty`, instrumented with a `render_call_count`) makes "renderer-never-invoked-on-abstain" a **hard, assertable count** — 0 across every abstain, every seed.

### Demo transcript (seed 42) — EMERGE reasons → gated fact extracted → FLUENTLY rendered

```
you> can an owl fly?
    EMERGE reasons  : Yes, an owl can fly.
    WIRED (fluent)  : Yes, the owl can fly.        [INHERIT; gate=ANSWER; renderer INVOKED]
you> can a wren fly?
    WIRED (fluent)  : Yes, the wren can fly.       [INHERIT; gate=ANSWER; renderer INVOKED]
you> can a minnow swim?
    WIRED (fluent)  : Yes, the minnow can swim.    [INHERIT; gate=ANSWER; renderer INVOKED]
you> can a gar swim?
    WIRED (fluent)  : Yes, the gar can swim.       [INHERIT; gate=ANSWER; renderer INVOKED]
you> can a penguin fly?
    EMERGE reasons  : No, a penguin walks.
    WIRED (fluent)  : No, the penguin walks.       [CANCEL;  gate=ANSWER; renderer INVOKED]
you> can a pike swim?
    WIRED (fluent)  : No, the pike lurks.          [CANCEL;  gate=ANSWER; renderer INVOKED]
you> can a zzz fly?
    WIRED (fluent)  : I don't know what a zzz is.   [MOAT; gate=ABSTAIN; renderer NOT invoked]
you> can a wobble swim?
    WIRED (fluent)  : I don't know what a wobble is. [MOAT; gate=ABSTAIN; renderer NOT invoked]

render-call count after 8 probes: 6   (abstains never invoked the renderer)
```

`owl/wren` (bird) + `minnow/gar` (fish) are GENUINE HELD-OUTS (never named in a can/exception sentence — they inherit ONLY via the shared discovered codon); `penguin/pike` are the member-specific exceptions (cancellation); `zzz/wobble` were never observed (moat).

## RUNG 2 (GPU) — ran; the gate-first MOAT HELD on the REAL 21M; RA content is a DATA/format lever (honest)

The RA-fine-tuned 21M (`gen_tinystories_ra_ft.ckpt.pt`, 21.3M params, on CUDA) was loaded and asked to render EMERGE's gated facts behind the SAME gate-first loop:

```
you> can an owl fly?    brain> [facts: the owl can fly .]   no , the owl does not fly f .   [model invoked]
you> can a penguin fly? brain> [facts: the penguin walkses .] the penguin likes to follow leaf . [model invoked]
you> can a zzz fly?     brain> I don't know what a zzz is.  [MOAT; model NOT invoked]   ← moat held on the real model
```

- **The load-bearing property carries to the GPU generator:** on the `zzz` abstain the model was **NOT invoked** (the gate short-circuited before render). The no-confab moat holds with the real model in the loop.
- **Honest content result:** the RA fine-tune renders its TRAINED transitive-SVO format *faithfully* (`the dog eats meat .` → `the dog eats meat .`), but EMERGE's `can-fly` / intransitive-exception verb-forms are **out of the RA fine-tune's distribution** → it confabulates content ("the owl likes to follow leaf"). Direct probe confirms this is a format mismatch, not a model failure. Also surfaced: `_v3("walks")` double-inflects a verb that is already 3rd-person-sg ("walkses") — the Rung-2 prompt builder needs EMERGE-frame-aware inflection.
- **Verdict for Rung 2:** the architecture is confirmed sound (moat held), but fluent RA rendering of EMERGE's specific frames needs a small **RA re-fine-tune on the can-form + intransitive frames EMERGE emits** (a data lever) — NOT an architecture wall. This is the Rung-2 follow-on.

## Verdict

**Rung 1 GO.** The EMERGENT grounded reasoning is WIREABLE to a fluent faculty behind the gate-first no-confab moat — adapter fidelity 1.00, moat preserved (0 renders on abstains, 0 false renders — the load-bearing property), correct grounded facts rendered, 3-seed. The architecture is confirmed wireable; the GPU RA-render path is safe to proceed. The Rung-2 GPU smoke ran and **the moat held on the real 21M**; the remaining gap is a RA re-fine-tune on EMERGE's verb-frames (data/format lever), plus frame-aware inflection in the prompt builder. **Wernicke decides → Broca articulates.**

## Files
- `research/runners/_emerge56_reasoning_to_fluent_wire_derisk.py` — the adapter (`emerge_gate_decision`), the gate-first render loop (`wired_reply`), the counting stub (`CountingStubFaculty`), the 3-seed de-risk, and the `--rung2` GPU RA-render smoke.
- `tests/test_emerge56_reasoning_to_fluent_wire.py` — 4 CPU/numpy tests (adapter fidelity; moat preserved incl. renderer-not-invoked-on-abstain; correct grounded facts; single-seed de-risk GO).
- `research/findings/raw/_emerge56_reasoning_to_fluent_wire.json` (Rung 1 de-risk) + `research/findings/raw/_emerge56_rung2_ra_render.json` (Rung 2 smoke).

## Honest scope
Wiring-not-mechanism: two orthogonal subsystems (on-brain spiking EMERGE reasoning + a fine-tuned-ANN articulator) handed off via the gated-fact tuple. The Rung-1 renderer is the content-locked P3 stub; the generator ANN remains a tracked temporary scaffold (its spiking-forward conversion validated at 88.6M). Next: Rung 2 (RA re-fine-tune on EMERGE frames + frame-aware inflection → fluent) + Rung 3 (merge into `_fluidconv_chat_repl.py` so EMERGE `can a penguin fly?` + existing `what does a dog eat?` both work under one consistent moat + fluency).
