# EMERGE-58 (Rung 3) — the FINAL integration: EMERGENT-REASONING fluent conversation FOLDED into the flagship FLUID console, ONE gate-first no-confab MOAT across BOTH: **GO** (3-seed CPU + GPU render smoke)

**2026-07-03 (autonomous).** Rung 3 of the north-star wire (Rung 1 = EMERGE-56 GO the wire; Rung 2 = EMERGE-57 GO the re-fine-tuned fluent render). Rung 3 MERGES the two so **ONE console answers BOTH kinds of question** under one consistent gate-first moat: (a) EMERGE emergent-reasoning questions — discovered-category inheritance / per-dimension cancellation / sibling-discrimination — rendered FLUENTLY by the re-fine-tuned 21M; AND (b) the existing fluid-conversation paths ("what does the dog eat?" / "tell me about X" / growth / yes-no / discuss). Reuse-by-import; **NO `sim/` edit; NO `_fluidconv_chat_repl.py` edit** (the flagship is used verbatim — the merge is a COMPOSITION/wrapper).

## The merge — a COMPOSITION, not a destructive edit

`UnifiedFluentConsole` OWNS two subsystems and a tiny router:
- **the flagship `FluidChat`** (`_fluidconv_chat_repl`) — used VERBATIM (no edit) for all fluid paths.
- **a taught `PerDimensionConsole`** (EMERGE-54) — the per-dimension emergent reasoner over the pooler-DISCOVERED categories (inherit / per-dimension cancel / sibling-discrimination / moat). Taught once at construction (EMERGE-54's scripted bird/fish taxonomy).
- **the ROUTER**, keyed on the ability frame `can a/an <member> <verb>?`: that frame → the EMERGE reasoner; everything else → `FluidChat.turn()` (byte-unchanged). The frame is exclusive to the EMERGE question shape, so there is **no cross-talk** either direction.

The EMERGE side reuses the EMERGE-56/57 **gate-first render loop** exactly: a per-dimension gate-decision adapter (`emerge_pd_gate_decision`, validated 1-to-1 against `PerDimensionConsole.ask_can`) → on ABSTAIN emit "I don't know" and **NEVER invoke the generator**; on ANSWER pass the grounded gated fact to the re-fine-tuned 21M (`_CountingFTFaculty`, EMERGE-57) → fluent surface. A CPU **template** renderer (same surface, content-locked, invocation-counted) makes the routing/moat/regression gates run offline + CPU-safe + multi-seed; the FLUENT GPU render is the real 21M behind the IDENTICAL loop.

## De-risk — **GO** (3-seed CPU-safe, + GPU render smoke)

| gate | value | bar |
|---|---|---|
| (a) EMERGE gate-decision ADAPTER FIDELITY (vs `PerDimensionConsole.ask_can`'s own decision) | **1.00** | ≥ 0.99 |
| (a) EMERGE RENDER correct (inherit owl→fly, cancel penguin→walks, per-dim robin→breathe; CPU template) | **1.00** | ≥ 0.99 |
| (a) EVERY EMERGE ability frame routed to the reasoner | **1/1 every seed** | true |
| (b) NO fluid-path REGRESSION (what / anaphora / growth / yes-no / discuss / moat all correct) | **1/1 every seed** | true |
| (c) ONE MOAT — render-calls on abstains across BOTH kinds (the LOAD-BEARING property) | **0** | 0 |
| (d) NO CROSS-TALK (EMERGE frame → reasoner; fluid Q → not-EMERGE) | **1/1 every seed** | true |

3-seed (42/43/44). The GPU render smoke ran on the **real 21.3M** re-fine-tuned generator (dev=cuda): the EMERGE answers rendered fluently and the **moat held on the real model** (0 renders on abstains; the model NOT invoked on all 3 abstains).

### The mixed demo transcript (`--demo --render`, seed 42, the REAL 21M rendering EMERGE answers) — BOTH kinds in one session, one moat

```
you>   what does the dog chase?   brain> The dog chases cat.                 [FLUID  grounded Q&A -> writes 'cat']
you>   what does it eat?          brain> The cat eats fish.                  [FLUID  anaphora it=cat, Phase-4]
you>   the wolf eats rabbit       brain> ok, i learned that the wolf eats rabbit.  [FLUID  growth]
you>   what does the wolf eat?    brain> The wolf eats rabbit.               [FLUID  learned fact usable]
you>   can an owl fly?            brain> yes .                               [EMERGE INHERIT (discovered bird codon); 21M rendered]
you>   can a penguin fly?         brain> no , the penguin walks .            [EMERGE CANCEL (locomotion exception); 21M]
you>   can a robin breathe?       brain> yes , the robin can breathe .       [EMERGE PER-DIMENSION inherit (respiration, no leak); 21M]
you>   can an owl swim?           brain> I don't know whether an owl can swim.   [EMERGE SIBLING-abstain (bird, not fish); model NOT invoked]
you>   can a zzz fly?             brain> I don't know what a zzz is.          [EMERGE MOAT (never observed); model NOT invoked]
you>   tell me about the dog      brain> Here's what I know about the dog: A dog is big. It eats meat, chases cat and likes bone.  [FLUID discuss]
you>   does the dog eat meat?     brain> Yes, the dog eats meat.             [FLUID  yes/no]
you>   what does the lion eat?    brain> I don't know.                       [FLUID  MOAT (untaught)]

render-calls on abstains (BOTH kinds): 0   (the load-bearing property)
```

The four target EMERGE behaviors all render fluently + correctly, matching EMERGE-57: **owl→fly inherited**, **penguin→walks cancel**, **robin→breathe per-dimension** (the exception does NOT leak across dimensions), **owl→swim sibling-abstain** (owl is a bird, not a fish — the reasoner correctly refuses the fish-branch ability). The fluid paths are unchanged. The moat is one consistent gate-first mechanism: an unknown/unobserved/sibling subject abstains on BOTH kinds and the generator is NEVER invoked.

## Verdict

**GO.** The emergent-reasoning fluent conversation (EMERGE-51..57) is folded into the flagship fluid console: ONE console routes correctly (every EMERGE ability frame → the reasoner, no cross-talk), renders EMERGE questions FLUENTLY + correctly (adapter fidelity 1.00, render 1.00; the real 21M renders inherit/cancel/per-dimension), preserves the existing fluid paths (no regression, 3-seed), and holds ONE consistent gate-first no-confab MOAT — **0 renders on abstains across BOTH kinds, the load-bearing property, confirmed on the real 21M**. The north-star wire is complete: **Wernicke decides → Broca articulates, for BOTH the emergent reasoner and the fluid paths, on one console.**

## Files
- `research/runners/_emerge58_unified_fluent_console.py` — `UnifiedFluentConsole` (router + `_emerge_turn` gate-first render + `FluidChat` delegation), the per-dimension gate-decision adapter (`emerge_pd_gate_decision`, fidelity-validated), the CPU template renderer (`_TemplateEmergeFaculty`, invocation-counted) + the GPU faculty selector, the mixed `--demo`/`--demo --render`, the 3-seed `--derisk` (+ `--gpu-render-smoke`), and `_gpu_render_smoke`.
- `tests/test_emerge58_unified_fluent_console.py` — 12 tests (10 CPU/offline: routing, adapter fidelity, moat renderer-never-invoked-on-abstain, unknown-vs-sibling abstain, inherit/cancel/per-dimension render, frame-aware inflection, the four decision branches; 1 slow: full de-risk with FluidChat + no-regression + no-cross-talk; 1 GPU render smoke, skip-if-no-ckpt — all pass with the EMERGE-57 ckpt present).
- `research/findings/raw/_emerge58_unified_fluent_console.json` — the 3-seed de-risk + the embedded GPU render smoke.
- Reuses the EMERGE-57 ckpt `research/findings/raw/fluidconv/gen_tinystories_ra_emerge_ft.ckpt.pt` (21.3M; unchanged).

## Honest scope
Rung 3 = the MERGE (composition), not a new mechanism. The EMERGE render CORRECTNESS gate runs on the CPU template renderer (content-locked, same surface) so routing/moat/regression are offline + CPU-safe + multi-seed; the FLUENT GPU render is the EMERGE-57 re-fine-tuned 21M behind the IDENTICAL gate-first loop (`--render` / `--demo --render`; skip-if-no-ckpt), reported as a smoke — EMERGE-57 already GO'd its render fidelity 1.00 + moat 0-renders-on-abstain, and the smoke here re-confirms the moat holds on the real model within the unified console. The **generator ANN remains a tracked temporary scaffold** (spiking-forward conversion deferred, validated at 88.6M). The MOAT is preserved BY CONSTRUCTION on BOTH kinds (the gate short-circuits before the renderer / the fluid gate). The EMERGE reasoner is taught EMERGE-54's scripted bird/fish taxonomy (per-dimension locomotion/respiration); corpus-scale feature discovery + a broader taxonomy in the unified console are the standing EMERGE follow-ons. The router is a host keyword/frame interface (the world/keyboard boundary), like every EMERGE NL front end.
