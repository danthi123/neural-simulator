---
type: plan
status: live
date: 2026-05-17
---

# Generator-G — Grounded, No-Confabulation Generation (fluency MUST preserve the no-confab moat) — Design (ACTIVE)

> **For Claude:** REQUIRED NEXT SKILL: superpowers:writing-plans (then
> superpowers:subagent-driven-development). Continuous autonomous arc
> (user 2026-05-17: a week autonomous, no stopping/asking, no
> config-cranking a terminated mechanism, self-contained at RUNTIME,
> local 3090, public training corpus authorized, full architectural
> freedom). The pre-registered PASS-branch successor of Generator-F.

## Why this is genuinely decision-relevant (NOT recombination-busywork)

The project has two SEPARATELY-validated assets:
1. **Generator-F** (just validated): a self-contained small Transformer
   that generates grammatical, locally-coherent simple text — at the
   explicit small-Transformer TinyStories ceiling (NOT GPT-class).
2. **The validated biology-grounded grounded continual memory with
   no-confabulation abstention** (G.20 sparse ensembles, engram
   stim-recall 87.5%, multi-tag cue retrieval 90%, the
   no-confabulation moat — multi-seed anti-cheat-validated): it
   reliably stores/retrieves grounded propositions AND *refuses to
   make things up* (abstains on the unknown). That no-confabulation
   property is the project's distinctive contribution — a property a
   small LLM does NOT have.

The genuinely-open, decision-relevant question (the user's actual
north-star: "a very small yet SOTA LLM" but *trustworthy*): **can
fluent generation be added WITHOUT destroying the no-confabulation
property?** A fluent LM is intrinsically a confabulation engine. This
is a real fluency-vs-faithfulness tension, not predictable
recombination.

## Evidence grounding (falsify-cheaply, done BEFORE designing)

A cheap probe conditioned the trained Generator-F TinyGPT on
retrieved-style grounded prompts at temperature 1.0:
- "Once upon a time there was a dog named Max. Max" -> "was a called
  **Bob**. Bob was compassionate..." (it RENAMES Max->Bob — a
  confabulation; drifts to generic story).
- "Lily found a red ball. She" -> "...Can I try it, **Sam**?" **Ben**
  looked..." (introduces ungrounded entities Sam/Ben).

**Honest finding:** naive free-sampling conditioning at the small-LM
ceiling does NOT preserve grounded faithfulness — the model
confabulates entities. This SHARPENS (does not kill) Generator-G: the
mechanism must be a **faithfulness-preserving decoding scheme gated by
the validated no-confabulation moat**, and the LOAD-BEARING
pre-registered bar must be *no-confabulation preservation* — not raw
fluency.

## Thesis

A self-contained agent where the **validated no-confabulation moat
decides answer-vs-abstain** (reused UNMODIFIED — its multi-seed
validated abstention behaviour is the gate), and ONLY when the moat
says "grounded" does the Generator-F Transformer produce a response,
decoded in a **faithfulness-constrained** way (low-temperature/greedy,
conditioned on the retrieved grounded proposition tokens, optionally
constrained to a retrieved-content token set). On the unknown the
agent ABSTAINS — the Transformer is NEVER given free rein to
confabulate. The honest realization of the conversational goal within
the small-LM ceiling: coherent simple responses that are grounded and
that *refuse to make things up*.

## Pre-registered gate — the LOAD-BEARING bar is no-confab preservation

Reuse the SAME hardened gate discipline (fixed bars, multi-seed >=3,
permuted/held-out controls, mandatory smell-test, never tuned). The
**load-bearing, pre-registered, FIXED criterion**:

1. **No-confabulation PRESERVED (the decisive bar):** on a held-out
   set of UNGROUNDED queries (never stored), the agent must ABSTAIN
   at >= the validated bare-moat abstention rate (the moat's
   multi-seed-validated behaviour is the frozen reference; the fluent
   layer must NOT reduce it). A single ungrounded query answered with
   a fluent confabulation is a FAIL. Multi-seed >=3.
2. **Grounding faithfulness on the known:** for GROUNDED queries, the
   generated response must be faithful to the retrieved proposition —
   measured by a FIXED, pre-registered, non-circular metric:
   ungrounded-entity rate (content tokens not in retrieved
   proposition ∪ a closed function-word set) <= a FIXED bar; and the
   retrieved key fact's tokens appear in the response. (The probe's
   Max->Bob failure is exactly what this catches.)
3. **Coherence at the Generator-F ceiling (not above):** responses
   are grammatical/locally-coherent (Generator-F is validated for
   this) — reported at the honest ceiling, NEVER spun as GPT-class.
4. **MANDATORY anti-cheat smell-test** (scrutinize a PASS HARDER than
   a FAIL — the Generator-S lesson): verify from recorded data the
   abstention isn't trivially achieved by always-abstaining (it must
   still ANSWER grounded queries), the faithfulness metric isn't
   gamed by degenerate echo, and read the actual transcripts.
   Recompute from recorded data; no re-run; no bar-tuning.

PASS (scrutinized genuine) => the honest culmination: a
self-contained, local, no-cheat agent that generates coherent simple
text, is grounded, AND preserves the validated no-confabulation
property a small LLM lacks — the north-star within the explicit
small-LM ceiling. FAIL => the decision-relevant terminal finding:
fluency and no-confabulation do NOT compose into a single
self-contained agent at feasible local scale; the two validated
assets (Generator-F fluency; the grounded no-confab memory) stand as
SEPARATE deliverables. Either outcome is decision-relevant and
honestly propagated; NOT config-cranked.

## Architecture (net-new small; validated components reused UNMODIFIED)

Reuse UNMODIFIED (DRY): `sim.tiny_transformer.TinyGPT` + a trained
Generator-F checkpoint (the validated fluent generator); the validated
grounded-memory + no-confabulation moat (the existing validated
G.20/engram/abstention machinery — exact module per the cheapest
faithful integration point, chosen at plan time from the validated
runners); `sim.bpe_tokenizer`; the hardened gate-core anti-cheat
discipline (verdict/aggregate pattern; a Generator-G-specific
no-confab-preservation core with its OWN frozen bars, mirroring the
discipline, NOT modifying gate_core/song_g1_core).

Net-new (small, pure-testable where possible):
1. A pure `grounded_decode` policy: given (query, moat-decision,
   retrieved-tokens), return either ABSTAIN or a faithfulness-
   constrained TinyGPT decode (greedy/low-temp, conditioned on
   retrieved tokens). Pure-unit-testable: abstain path never calls
   the generator; grounded path conditions on retrieved tokens.
2. A pure no-confab-preservation scoring core (FIXED bars: abstain-
   on-unknown rate >= frozen moat reference; ungrounded-entity rate
   <= FIXED; multi-seed >=3) — adversarially testable (always-abstain
   cannot pass because grounded-answer-rate is also required;
   echo-degenerate cannot pass faithfulness).
3. A thin `generator_g_gate.py` runner orchestrating it, mirroring the
   existing gate runners' shape (kill-safe, ASCII-only, <3-seeds->
   exit-2, honest-propagation-is-controller's-job).

## Honest ceiling / risks (no overclaiming)

- The probe already shows faithful grounded conditioning is HARD at
  the small-LM ceiling. The honest expectation, stated up front: this
  may FAIL the faithfulness/no-confab-preservation bar — and that
  honest FAIL is the decision-relevant terminus (the two validated
  assets stay separate; the north-star "fluent AND trustworthy in one
  self-contained artifact" is not reachable at feasible local scale).
- A PASS is reported STRICTLY at the small-Transformer ceiling
  (coherent SIMPLE grounded responses + preserved no-confab), with
  verbatim transcripts, NEVER spun as GPT-class or as overturning the
  9-negative converged conclusion.
- Self-contained at RUNTIME preserved (trained TinyGPT weights + BPE
  JSON + the self-contained validated memory; no external dep).
- Local 3090; reuse trained Generator-F checkpoints (no re-train of
  the LM needed for the decisive slice); kill-safe; ASCII-only.

## Out of scope (YAGNI)

No external dependency at RUNTIME ever. No new global bar in
gate_core; gate_core/song_g1_core byte-UNMODIFIED. No config-cranking
any terminated mechanism. No LM re-training in this slice (reuse the
validated Generator-F checkpoint). The pre-registered no-confab-
preservation gate decides; PASS = honest culmination, FAIL = terminal
decision-relevant; either way the autonomous arc continues / concludes
honestly per the result.

## Scientific basis (catalog + arc)

Retrieval-grounded / constrained decoding (faithfulness-vs-fluency
literature); the project's validated no-confabulation abstention moat
(the distinctive contribution); Generator-F (validated small-
Transformer generation, Eldan&Li small-LM ceiling). The hardened
anti-cheat discipline (held-out, multi-seed, frozen bars, mandatory
smell-test) is the adjudicator.
