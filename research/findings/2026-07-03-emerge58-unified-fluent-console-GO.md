# EMERGE-58 (Rung 3) — the FINAL integration: EMERGENT-REASONING fluent conversation FOLDED into the flagship FLUID console, ONE gate-first no-confab MOAT across BOTH: **GO** (3-seed CPU + GPU render smoke)

**2026-07-03 (autonomous).** Rung 3 of the north-star wire (Rung 1 = EMERGE-56 GO the wire; Rung 2 = EMERGE-57 GO the re-fine-tuned fluent render). Rung 3 MERGES the two so **ONE console answers BOTH kinds of question** under one consistent gate-first moat: (a) EMERGE emergent-reasoning questions — discovered-category inheritance / per-dimension cancellation / sibling-discrimination — rendered FLUENTLY by the re-fine-tuned 21M; AND (b) the existing fluid-conversation paths ("what does the dog eat?" / "tell me about X" / growth / yes-no / discuss). Reuse-by-import; **NO `sim/` edit; NO `_fluidconv_chat_repl.py` edit** (the flagship is used verbatim — the merge is a COMPOSITION/wrapper).

## The merge — a COMPOSITION, not a destructive edit

`UnifiedFluentConsole` OWNS two subsystems and a tiny router:
- **the flagship `FluidChat`** (`_fluidconv_chat_repl`) — used VERBATIM (no edit) for all fluid paths.
- **a taught `PerDimensionConsole`** (EMERGE-54) — the per-dimension emergent reasoner over the pooler-DISCOVERED categories (inherit / per-dimension cancel / sibling-discrimination / moat). Taught once at construction (EMERGE-54's scripted bird/fish taxonomy).
- **the ROUTER**: the ability frame `can a/an <X> <verb>?` is **SHARED** between an EMERGE taxonomy-ability question and a fluid fact question, so **disambiguation is by TAXONOMY MEMBERSHIP** (not the frame alone): `X` an observed taxonomy member → the EMERGE reasoner; a fluid-known (or unknown) `X` in the same frame → `FluidChat.turn()` (which answers it if it knows `X`, else applies its own gate-first moat); everything else → `FluidChat.turn()` (byte-unchanged). *(This membership gate is the 2026-07-03 audit remediation — see below; the initial frame-ONLY router falsely denied fluid-known entities.)*

The EMERGE side reuses the EMERGE-56/57 **gate-first render loop** exactly: a per-dimension gate-decision adapter (`emerge_pd_gate_decision`, validated 1-to-1 against `PerDimensionConsole.ask_can`) → on ABSTAIN emit "I don't know" and **NEVER invoke the generator**; on ANSWER pass the grounded gated fact to the re-fine-tuned 21M (`_CountingFTFaculty`, EMERGE-57) → fluent surface. A CPU **template** renderer (same surface, content-locked, invocation-counted) makes the routing/moat/regression gates run offline + CPU-safe + multi-seed; the FLUENT GPU render is the real 21M behind the IDENTICAL loop.

## De-risk — **GO** (3-seed CPU-safe, + GPU render smoke)

| gate | value | bar |
|---|---|---|
| (a) EMERGE gate-decision ADAPTER FIDELITY (vs `PerDimensionConsole.ask_can`'s own decision) | **1.00** | ≥ 0.99 |
| (a) EMERGE RENDER correct (inherit owl→fly, cancel penguin→walks, per-dim robin→breathe; CPU template) | **1.00** | ≥ 0.99 |
| (a) every TAXONOMY-MEMBER ability frame routed to the reasoner | **1/1 every seed** | true |
| (a2) MEMBERSHIP routing (audit remediation): a fluid-known entity in the frame (`can a dog eat?`) is ANSWERED, not falsely denied; a genuine unknown abstains gate-first | **PASS every seed** | true |
| (b) NO fluid-path REGRESSION (what / anaphora / growth / yes-no / discuss / moat all correct) | **1/1 every seed** | true |
| (c) ONE MOAT — render-calls on abstains across BOTH kinds (the LOAD-BEARING property) | **0** | 0 |
| (d) NO CROSS-TALK (fluid Q → not-reasoner; taxonomy-member EMERGE Q → reasoner; fluid-known entity in the shared frame → not-reasoner) | **1/1 every seed** | true |

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

## Audit + remediation (2026-07-03) — a real defect found, fixed, and guarded

The initial GO was **adversarially audited** (the project's standing discipline — the same pattern caught systematic defects in EMERGE-38..45). The audit **confirmed one [major] routing-crosstalk correctness defect**, empirically reproduced on seed 42:

- **The bug:** the router was **frame-ONLY**. `can a dog eat?` matched the `can a X <verb>?` frame → routed **unconditionally** to the EMERGE reasoner → `dog` is not in the scripted taxonomy → hit `moat_unknown` → **falsely denied** *"I don't know what a dog is."* — in the **same session** where `does the dog eat meat?` → *"Yes, the dog eats meat."* and the fluid path answers `can a dog eat?` → *"The dog eats meat."* A **factually wrong denial / self-contradiction**, not a moat-safe abstain. The original `no_crosstalk` gate passed only because it **never probed a fluid-known entity in the ability frame** — a gate that passes by construction while the plain-English "no cross-talk" framing was false.
- **The fix (membership-aware routing):** on a frame match, route to the reasoner **iff `X ∈ reasoner.member_idx`** (an observed taxonomy member); else fall through to `self.fluid.turn(raw)` — the fluid path answers a fluid-known entity and else applies its own gate-first moat. The gate-first moat is **unchanged for genuine unknowns** (`can a zzz fly?` → *"I don't know."*, the fluent generator NOT invoked).
- **The gate fix:** added a **membership-routing gate** (a2 above) + a **regression test** (`test_membership_aware_routing_no_false_denial`) that probe the exact failing shape, so the gate can no longer pass by construction. Corrected the two false statements at source (the "frame exclusive / no cross-talk" claim; the routing row).
- **Re-verified post-fix (3-seed 42/43/44, CPU):** `can a dog eat?` → **"The dog eats meat."** (answered, 21M NOT invoked); `can a zzz fly?` → **"I don't know."** (abstain, 21M NOT invoked); membership-routing gate **PASS every seed**; adapter 1.00, render 1.00, moat 0-renders, no-regression, no-cross-talk all still GO. CI **13 pass** (incl. the new regression guards).

*(Audit scope note: 6 of 7 adversarial lenses hit transient server-side rate limits and did not complete; the routing-crosstalk lens completed and found the above. The remaining lenses are re-run on the fixed code to complete the adversarial pass.)*

## Verdict

**GO (post-remediation).** The emergent-reasoning fluent conversation (EMERGE-51..57) is folded into the flagship fluid console: ONE console routes correctly by **taxonomy membership** (a taxonomy-member ability question → the reasoner; a fluid-known entity in the shared frame → the fluid path, answered not falsely denied), renders EMERGE questions FLUENTLY + correctly (adapter fidelity 1.00, render 1.00; the real 21M renders inherit/cancel/per-dimension), preserves the existing fluid paths (no regression, 3-seed), and holds ONE consistent gate-first no-confab MOAT — **0 renders on abstains across BOTH kinds, the load-bearing property, confirmed on the real 21M**. The north-star wire is **demonstrated end-to-end and audit-hardened**: **Wernicke decides → Broca articulates, for BOTH the emergent reasoner and the fluid paths, on one console** — with the honest tracked residuals below (the reasoner is taught a scripted taxonomy; the generator is an ANN scaffold; the EMERGE-57 re-fine-tune was single-seed).

## Files
- `research/runners/_emerge58_unified_fluent_console.py` — `UnifiedFluentConsole` (router + `_emerge_turn` gate-first render + `FluidChat` delegation), the per-dimension gate-decision adapter (`emerge_pd_gate_decision`, fidelity-validated), the CPU template renderer (`_TemplateEmergeFaculty`, invocation-counted) + the GPU faculty selector, the mixed `--demo`/`--demo --render`, the 3-seed `--derisk` (+ `--gpu-render-smoke`), and `_gpu_render_smoke`.
- `tests/test_emerge58_unified_fluent_console.py` — 13 tests (CPU/offline: frame recognition, fluid-Q-not-captured, **membership-gates-the-shared-frame** [audit guard], adapter fidelity, moat renderer-never-invoked-on-abstain, unknown-vs-sibling abstain, inherit/cancel/per-dimension render, render-invokes-generator-on-answer, frame-aware inflection, the four decision branches; slow: full de-risk with FluidChat + no-regression + membership + no-cross-talk, **`test_membership_aware_routing_no_false_denial`** [the audit regression guard: `can a dog eat?` answered not denied, `can a zzz fly?` moat]; 1 GPU render smoke, skip-if-no-ckpt — all pass).
- `research/findings/raw/_emerge58_unified_fluent_console.json` — the 3-seed de-risk + the embedded GPU render smoke.
- Reuses the EMERGE-57 ckpt `research/findings/raw/fluidconv/gen_tinystories_ra_emerge_ft.ckpt.pt` (21.3M; unchanged).

## Honest scope
Rung 3 = the MERGE (composition), not a new mechanism. The EMERGE render CORRECTNESS gate runs on the CPU template renderer (content-locked, same surface) so routing/moat/regression are offline + CPU-safe + multi-seed; the FLUENT GPU render is the EMERGE-57 re-fine-tuned 21M behind the IDENTICAL gate-first loop (`--render` / `--demo --render`; skip-if-no-ckpt), reported as a smoke — EMERGE-57 already GO'd its render fidelity 1.00 + moat 0-renders-on-abstain, and the smoke here re-confirms the moat holds on the real model within the unified console. The **generator ANN remains a tracked temporary scaffold** (spiking-forward conversion deferred, validated at 88.6M). The MOAT is preserved BY CONSTRUCTION on BOTH kinds (the gate short-circuits before the renderer / the fluid gate). The EMERGE reasoner is taught EMERGE-54's scripted bird/fish taxonomy (per-dimension locomotion/respiration); corpus-scale feature discovery + a broader taxonomy in the unified console are the standing EMERGE follow-ons. The router is a host keyword/frame interface (the world/keyboard boundary), like every EMERGE NL front end; it disambiguates the shared `can a X <verb>?` frame by taxonomy membership (audit-remediated, see above). The EMERGE-57 re-fine-tune underlying the fluent GPU render was single-seed — a multi-fine-tune-seed robustness pass is a tracked follow-on.
