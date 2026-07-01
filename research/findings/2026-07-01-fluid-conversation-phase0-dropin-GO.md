# Fluid conversation — Phase 0 DROP-IN GO: the ~21M generator is grounded + non-vacuous + moat-intact behind the veto

**2026-07-01 (autonomous night; the owner's main-priority pivot).** The core **"minimize the transformer"** thesis
test, and the second (grounding) half of Phase 0. The fluency half already landed
(`2026-07-01-fluid-conversation-phase0-minimal-generator.md`): a **~21.3M-param** TinyStories generator (d512/L6/H8,
held-out ppl 5.66, no overfit) is genuinely fluent. This doc closes the grounding half: **that same 15–25×-smaller
generator, dropped into the EXISTING grounded-lang gate→constrain→verify loop in place of the external Qwen2.5-0.5B,
stays grounded + non-vacuous behind the per-token veto while the no-confab moat holds — SCALE-CONFIDENT across the
validated ladder.**

## What was run
The validated grounded-lang constrained-decode harness (`research/runners/constrained_decode_gate` +
`sim/grounded_decode.grounded_decode` + the FROZEN `_CDC_*` verdict in `constrained_decode_core`) — reused **WHOLE**,
byte-unmodified in its verdict logic. The only change: the fluent generator behind the veto is now the 21M TinyStories
model instead of Generator-F/Qwen. Same `_GROUNDED` set (24 children's-story facts — in-domain for a TinyStories model,
e.g. *"max is a big friendly dog"*), same `_UNGROUNDED` nonsense tokens (`zarn`, `qexel`, …) for the moat, same
3-way **constrained / unconstrained / shuffled** decode, same frozen bars, same `(6,12,24)` scale ladder, 3 seeds.

Enabled by one **additive, default-preserving** edit to the gate runner (`_GroundedConstrainedLM.__init__` gains
`d_model/n_layer/n_head/bpe_path`, defaults = the original d256/L4/H4 → existing callers byte-unchanged) + a thin
wrapper `_fluidconv_phase0_dropin_derisk.py`. **NO `sim/` edit.**

## Result — SCALE-CONFIDENT-PASS (3 seeds 42/43/44, GPU, 82.8 s)

| Rung | GATE | constrained UER | constrained non-vacuity | unconstrained-UER | shuffled-UER | shuffled non-vac | moat abstain-on-ungrounded | emittable |
|---|---|---|---|---|---|---|---|---|
| K=6  | PASS | 0.025 | 0.83 | 0.908 | 1.000 | 0.00 | 1.00 | 1.00 |
| K=12 | PASS | 0.012 | 0.83 | 0.866 | 0.779 | 0.00 | 1.00 | 1.00 |
| K=24 | PASS | 0.014 | 0.75 | 0.897 | 0.871 | 0.00 | 1.00 | 1.00 |

`nonvac_by_rung = [0.83, 0.83, 0.75]` — non-decreasing within the frozen tol 0.10, holds ≥ 0.5 at the largest rung ⇒
**SCALE-CONFIDENT-PASS**.

**Reading the numbers (the harness's own discipline):**
- **Constrained UER ~0 is MECHANICAL, not the result** (the per-token veto forbids off-proposition content by
  construction). The *discriminating* signal is **non-vacuity behind the veto** (≥ 2 distinct on-proposition content
  words + answer-rate ≥ 0.5): the 21M generator clears it at **0.75–0.83** — it produces genuinely on-topic content, it
  is not merely emitting empty function words.
- **The controls fail hard, so the veto is LOAD-BEARING** (not a trivially-satisfiable instrument): the *unconstrained*
  21M generator drifts to **UER 0.87–0.91** (way above the 0.20 faithful bar — free generation invents ungrounded
  entities), and the *shuffled* control (veto from a DIFFERENT proposition) drifts to **0.78–1.00 UER with 0.00
  non-vacuity** (wrong-proposition words cannot produce on-topic content).
- **The no-confab MOAT holds with the small generator in the loop: abstain-on-ungrounded = 1.00 at every rung** — the
  nonsense entities (`zarn`, …) never clear the abstention gate, so the generator is never even touched (no-confab by
  construction; the moat gates FIRST).
- **Instrument valid at every rung** (`emittable = 1.00`): the BPE veto can fully express all grounded content — no
  subword-defeat, so the PASS is a real test of the premise, not a VOID.

## What Phase 0 establishes (fluency + grounding, both halves)
- **The transformer-minimization thesis is validated end-to-end:** a **~21M** generator — **15–25× smaller** than the
  external Qwen-0.5B, trained locally in ~4.3 h on 90M TinyStories tokens — supplies fluency AND, behind the brain's
  validated veto, stays grounded (UER ~0.01–0.03) + non-vacuous (0.75–0.83) + hallucination-proof (moat 1.00),
  scale-confident. The transformer is *minimized*, not deleted — the honest sweet spot the roadmap identified.
- **It is small enough to become spiking-on-substrate cheaply:** the already-validated **88.6M spiking-forward** path
  (`2026-06-30-100M-C2-scaleup-C1-GO`) makes bridge co-residence of a 21M model trivial — the generator can literally
  become a spiking network on the one brain (Phase 1).
- **This is COMPOSITION of validated pieces, NO `sim/` edit, moat never weakened.**

## Honest ceiling (unchanged, stated plainly)
Constrained decoding **TRADES open-ended fluency for faithfulness BY DESIGN** — it renders a *single retrieved
proposition*, per-token vetoed to that proposition's words. This is grounded + hallucination-proof, but it is **NOT yet
"fluid, LLM-like, multi-turn conversation about almost any topic"** (the owner's north star). Phase 0 proves the
*minimal-transformer fluency + grounding + moat* substrate; the fluidity gap (context-conditioned multi-sentence
replies that blend knowledge + discourse) is the next arc. Per the owner's standing note that the no-confab moat is a
*plus, not a hard gate* (`feedback_moat_not_hard_lossy_memory_ok`), the next step can trade the rigid per-token veto
for a softer prompt-conditioning + **post-hoc VERIFY** (re-parse rejects hallucinated facts) where that buys fluidity.

## Next (Phase 1, per `2026-07-01-fluid-conversation-mechanisms-roadmap.md`)
- **Conversational rendering (the direct owner-goal test):** context + retrieved grounded facts → a fluid multi-sentence
  reply, grounded by prompt-conditioning + post-hoc VERIFY (moat-as-a-plus). Then recurrent/RWKV-style block for O(1)
  word-by-word on-substrate generation + the multi-referent WTA biased-competition dialogue mechanism (GAP D).
- **Phase 2:** growth through conversation (develop loop + C2 self-replay). **Phase 3 (parallel science bet):**
  thalamocortical dynamical gating (the transformer-free ceiling).

**Artifacts:** `research/runners/_fluidconv_phase0_dropin_derisk.py`; result
`research/findings/raw/_fluidconv_phase0_dropin.json`; the 21M generator `research/findings/raw/fluidconv/gen_tinystories_20M.ckpt.pt`.
