# Generative conversation — re-scoped design (corrected constraints)

> **Status:** APPROVED for autonomous execution (user delegated +
> authorized design→plan→implement; corrected constraints
> 2026-05-16). Supersedes the RETRACTED Stage-2 NO-GO.

## Corrected constraints (the only ones that are real)

- **Everything local** (no cloud, no external API at train or run). Hard.
- Free **public training datasets allowed**.
- Using a **local model to distill information allowed**.
- Vocabulary may be **large / learned** (the "320 hand-curated
  concepts" was never a user rule).
- "No cheating / no external-AI-as-a-crutch-to-fake-results" still
  holds — but principled local distillation + public-corpus training
  are explicitly sanctioned, not cheating.

## Feasibility: decisive GO (already proven, not speculative)

The project's own **Phase 2.2** (`path-f-hybrid`) trained a 4-layer
SNN via surrogate-grad BPTT on **Tiny Shakespeare, locally on this
RTX 3090**: loss 14.1→2.24 (84% reduction, perplexity ~9.4 vs chance
66, ~41 s). Real local text has massively-learnable sequence
structure on this exact hardware — empirically settled. Trained
artifact (`research/findings/raw/path_f/shakespeare_pretrained.npz`),
the BPTT infra (`sim/bptt_snn_gpu.py`, `sim/char_tokenizer.py`,
`sim/surrogate_grad.py`, `research/runners/cortex_pretraining.py`),
and a zero-download local English corpus (repo's 332 findings docs ≈
296 K words) all already exist. The prior "no corpus" wall was purely
the now-removed 320-concept mismatch.

## The core design insight

Do **not** force generation through the biological 320-concept
substrate (that is the documented v12–v16 / dlpfc NEGATIVE trap).
Instead: **a local learned sequence model supplies fluency; the
validated Stage-1 grounded-retrieval + abstention supplies
truthfulness.** Fluent AND honest, all local. The honesty moat (it
refuses to confabulate) is the deliberate differentiator vs a plain
small LLM, which hallucinates.

## Architecture (staged G1 → G2)

```
user ─▶ local small LM (generator)  ──draft──▶ GROUNDING GATE ──▶ reply
            (Phi-3-mini / Llama-3.2 /            │  every factual span
             Qwen2.5 — runs on 3090,             │  checked vs Stage-1
             100% local)                          │  validated retrieval;
                                                  │  unsupported span ⇒
        Stage-1 grounded substrate ───────────────┘  abstain / hedge
        (validated: cross-bridge recall +            (no-confabulation
         abstention AUC 0.990, remediated)            moat preserved)
```

**G1 (first; lower-risk; delivers the user goal):** a local
open-weights small LM as the fluent generator, **wrapped** so every
factual claim is checked against the validated grounded substrate +
abstention gate; unsupported claims are hedged/withdrawn rather than
asserted. Output: fluent, conversational, **and refuses to make
things up — fully local.** This is the genuinely-achievable
"LLM-like + honest + local" target.

**G2 (follow-on; more ownership):** **distillation** (sanctioned) —
use the G1 local LM as a teacher to train the project's own scaled
Phase-2 sequence learner on a real local corpus, so the generator
becomes *the project's own trained model*, not a wrapped external
one. Still local; same grounding gate.

## Components

- **Generator (G1):** one local open-weights small LM via a local
  runtime (llama.cpp / ctransformers / a local HF load on the 3090).
  Selection criterion: runs comfortably in 24 GB, Apache/MIT/Llama
  license, ≤~3B. No network at inference.
- **Grounding gate:** reuse the validated `g20_generative_agent`
  retrieval + `abstention_gate`. Factual spans from the draft are
  verified against grounded retrieval; below the abstention
  threshold ⇒ replace with an honest hedge / "I don't know."
- **Corpus (G2):** Tiny Shakespeare (already local) and/or the
  repo's own English text; learned tokenizer (existing
  `char_tokenizer`; word/subword as a later refinement).
- **Trainer (G2):** existing `cortex_pretraining` / `bptt_snn_gpu`
  on `path-f-hybrid`, merged forward; distillation loss vs the G1
  teacher's next-token distribution.

## Data flow / error handling

User turn → LM draft → segment into claims → each claim queried
against grounded substrate → supported: keep; unsupported/low-conf:
hedge or abstain → assemble reply + log (turn, draft, gated reply,
which spans hedged). LM runtime failure ⇒ fall back to the pure
Stage-1 grounded agent (already shipped) — degrade to honest, never
to hallucinating. No network path anywhere (assert offline).

## Testing

- Pure-logic CPU TDD: claim-segmentation, the grounding-gate
  decision (supported vs hedge vs abstain), reply assembly — same
  pattern as Stage-1's gate/grammar units.
- Integration smoke (local, GPU): scripted multi-turn; MUST include
  (a) a question the substrate knows → fluent grounded answer,
  (b) an unknown → hedged/abstained (no confabulation),
  (c) assert no network syscall (offline-enforced).
- Anti-cheat: the gate must measurably suppress an LM hallucination
  on a known-unanswerable prompt (demonstrated, not assumed).

## Honest ceiling (no overclaiming)

Local 3090 caps the generator far below cloud frontier LLMs. The
deliverable is **local + fluent + grounded + refuses to
confabulate** — a trustworthy local conversational agent, NOT
GPT-parity. G2's in-sim distilled generator will be weaker than its
G1 teacher (distillation + local-scale loss) — that is expected and
will be reported honestly with a falsifiable gate (distilled model
perplexity vs teacher, vs n-gram baseline).

## Build order

1. G1: local-LM runtime wrapper (offline-enforced) + grounding gate
   + scripted honest-fluency smoke. Ship the local fluent+honest
   agent.
2. G2: corpus + distillation trainer (reuse path-f-hybrid infra),
   falsifiable gate, then swap the in-sim generator behind the same
   grounding gate.

## Files (anticipated, G1)

- `research/runners/local_lm_runtime.py` (offline local LM load+gen)
- `research/runners/grounded_generative_agent.py` (LM draft →
  grounding gate → reply; reuses g20_generative_agent + abstention)
- `tests/test_grounding_gate.py` (CPU)
- `research/runners/grounded_generative_smoke.ps1`
- Supersedes Stage-2 in `2026-05-16-generative-conversation-design.md`;
  retraction context in `…-stage2-NO-GO-feasibility.md`.
