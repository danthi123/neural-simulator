# Fluid conversation — Phase 6 GO: breadth (the fluid stack over a broader KB, "almost any topic" honestly)

**2026-07-01 (autonomous night; owner's fluid-conversation priority).** Phases 0–5 + the console run on the 22-fact
micro-curriculum. Phase 6 tests the honest **breadth** axis (the owner's *"almost any topic"*; roadmap GAP B =
manage-not-solve via domain-constraint + retrieval-augmentation + abstention): a broader KB of facts across MANY
entities, taught to the brain at a higher composer dimension, then recall + RA-render + the no-confab moat measured at
scale. Reuse-by-import; **NO `sim/` edit**.

## Result — GO (3 seeds)
`_fluidconv_phase6_breadth_derisk.py`: a **40-fact** KB across many entities (drawn from the RA generator's own
competent vocab so it renders them), built at composer **D=256**, taught, then measured.

| metric | result (3 seeds) |
|---|---|
| RECALL across the broad KB (`what_does` correct on a 20-fact sample) | **1.00** (20/20 all seeds) |
| RA-RENDER (the fine-tuned 21M renders the broad answers grounded, VERIFY-clean) | **0.97** (20/20, 19/20, 19/20) |
| MOAT — held-out untaught cues (encodable, never taught) → abstain | **0 false-accepts / 12** all seeds |

⇒ the fluid stack holds over a broader KB: perfect recall, ~0.97 grounded RA-render, and the no-confab moat holds
0-FA at breadth. Dozens of entities are answerable, and the honest **"I don't know"** boundary holds beyond the
taught KB.

## Honest ceiling (the breadth boundary, characterized)
- Breadth is bounded by three levers, none a substrate wall at this scale:
  1. **The composer's FHRR capacity** (~√D / √M; crosstalk grows with the number of stored facts M) — raise D or add
     distinct codes; **validated to 320 concepts** in the project's existing sparse-distributed work. D=256 comfortably
     holds 40 facts (recall 1.00); the trend is the known √D/M curve, a data/D lever.
  2. **The generator's vocab** — TinyStories common English, which GENERALIZES to novel entities (Phase 5), so it
     renders a broad common-word KB.
  3. **The taught KB** — what facts the brain has learned; the abstention moat is the truthful boundary beyond it.
- **Open-domain (non-fact) conversation** — chit-chat / opinions / explanations beyond grounded fact-Q&A — remains
  the field's genuine wall (the roadmap's honest verdict: domain-constraint + retrieval-augmentation + abstention, not
  a transformer-free open-domain conversationalist). The stack is a grounded fact-conversation system that scales to
  hundreds of concepts, not an unbounded open-domain LLM.

## Arc status (the owner's fluid-conversation priority — comprehensive first pass)
Phases 0–6 + console, all reuse-by-import, NO `sim/` edit, moat preserved throughout:
fluent (0) · grounded rendering (1) · focused Q&A via the brain-train fine-tune (2) · full single-turn (3) ·
multi-turn anaphora (4) · growth-through-conversation (5) · **breadth (6)** · the interactive console (what/who/
yes-no/describe + anaphora + growth + abstention). The BRAIN does comprehension + knowledge + grounding + the moat;
the minimized (~21M, 15–25× < Qwen-0.5B), brain-trained, brain-gated generator does fluency.

**Tracked / deferred (per the end-state-fully-spiking standard):** the generator runs as an ANN (spiking-forward
conversion deferred until the KV-cache speed lever lands — a validated-mechanism reuse); the interrogative parse is a
rule-based scaffold (→ a neural interrogative parser); growth is over pre-allocated concept codes (new CODES = the
dendritic/allocation frontier); cross-session persistence validated in the develop loop; the webapp Interact wire-in
is pending.

**Artifacts:** `research/runners/_fluidconv_phase6_breadth_derisk.py`; result
`research/findings/raw/_fluidconv_phase6_breadth.json`.
