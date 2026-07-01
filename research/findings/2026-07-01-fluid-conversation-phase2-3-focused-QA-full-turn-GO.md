# Fluid conversation — Phase 2+3 GO: focused conversational Q&A + the full grounded turn ("talk to it like an LLM")

**2026-07-01 (autonomous night; owner's fluid-conversation priority).** Phase 0 gave a fluent+grounded minimal
generator; Phase 1 settled the fluid-grounded *rendering* mechanism (prompt-condition + free-gen + post-hoc VERIFY)
but the base TinyStories model CONTINUES STORIES rather than ANSWERING questions. Phase 2 closes that with the
roadmap's **"brain-train it"** lever (a small render/QA fine-tune), and Phase 3 assembles the **full grounded
conversational turn** end-to-end. Both **GO, 3 seeds, on the converged fine-tuned ckpt.** All reuse-by-import, **NO
`sim/` edit**, the no-confab moat preserved (GATE-first).

## Phase 2 — the retrieval-augmented render/QA fine-tune (the minimized, brain-trained faculty)
`_fluidconv_phase2_ra_finetune.py` fine-tunes the ~21.3M TinyStories generator to ANSWER questions from retrieved
facts. Design (so it GENERALIZES the FORMAT, not memorizes facts): a BROAD synthetic vocab (dozens of
subjects/verbs/objects, random facts) → the only learnable regularity is "use the provided facts + abstain if
absent"; answer-phrasing variety (fluent, not rote copy); **INTERLEAVED with raw TinyStories** (McClelland-1995 CLS /
self-replay anti-forgetting — the C2-validated principle — so the base fluency is retained); SAME BPE (V=2049),
low-LR (5e-5) continue-train from the 21M ckpt, 2500 steps (~26 min, init loss 4.59 → **final 0.877**).

**`_fluidconv_phase2_ra_qa_eval_derisk.py` — DEFINITIVE GO (3 seeds, final ckpt):**

| metric | result |
|---|---|
| focused-grounded Q&A (states the fact, ≤18 words, 0 ungrounded) | **5/5** all seeds |
| GATE-first moat (untaught → brain abstains → model NOT invoked) | **3/3** all seeds |
| RA-faithfulness (follows the PROVIDED fact over the model's own bias) | **3/3** all seeds |
| median answer words | **5** (vs the v3 base-model story ramble of ~30–40) |

Answers: *"the dog eats meat.", "the cat eats fish.", "the bird eats seed."* — focused, grounded, conversational.

**The load-bearing moat insight (a real finding):** the fine-tuned model, if WRONGLY prompted about an untaught
subject with distractor context, **CONFABULATES** (*"what does the lion eat?" → "the lion eats fish."* — diagnostic:
would-confab on 9/9 untaught cues). Therefore **the moat is GATE-FIRST**: the brain's validated no-confab gate decides
answer-vs-abstain BEFORE the generator is ever invoked (the integration-derisk pattern) — the model is NEVER prompted
without a grounded fact. This preserves the no-confab moat by construction with a hallucination-capable generator in
the loop, exactly as the grounded-lang arc established with Qwen.

## Phase 3 — the full grounded conversational TURN end-to-end
`_fluidconv_phase3_conversational_turn_derisk.py` assembles one turn from the validated parts:

```
free-text question --> [brain COMPREHEND: interrogative parse] --> (qtype, cue)
                    --> [brain GATE: what_does/who_does/is_it_true] --> fact | ABSTAIN (moat, gate-FIRST)
                    --> if abstain: "I don't know."  (generator NEVER invoked = no-confab by construction)
                    --> else: [RA-prompt -> fine-tuned 21M -> focused answer] -> [post-hoc VERIFY re-parse] -> reply
```

**DEFINITIVE GO (3 seeds, final ckpt): grounded-reply 5/5, gate-first moat 3/3, drift-caught 3/3.** The transcript is
the owner's north star:
```
you> what does the dog eat?    brain> the dog eats meat.
you> what does the cat eat?    brain> the cat eats fish.
you> what does the bird eat?   brain> the bird eats seed.
you> what does the lion eat?   brain> I don't know.   (untaught -> gate-first abstain)
```
yes/no is answered from the brain directly; an adversarial wrong-fact-in-context turn is caught by VERIFY (falls back
to the brain's own grounded statement, never emits the false fact).

**The division of labour (the owner's decoupling, realized):** the BRAIN supplies comprehension + knowledge +
grounding + the moat; the **minimized (~21M, 15–25× < Qwen-0.5B), brain-trained, brain-gated** generator supplies
fluency. This is "talk to it like an LLM" assembled from validated parts, with the transformer *minimized* (the
honest sweet spot), not deleted.

## Honest scope + what's next
- **SCAFFOLDS (flagged):** (1) the interrogative parse (question → structured query) is a light rule-based
  comprehension over the brain's vocab + wh/aux cue words — the brain-based replacement is a **neural interrogative
  parser** (same family as the declarative `BridgeParser`). (2) The fine-tune answer's surface morphology is the
  generator's; the word order in the fallback is the brain's neural render.
- **Vocab coverage:** the fine-tune's QA object pool was widened to cover all curriculum patient words
  (rabbit/cat/mouse/light/water/shade/tree/ground/cave) → the diverse verb set (eat/chase/make) renders 5/5. Broader
  domains need a broader synthetic vocab (a data lever, not a mechanism wall).
- **This is single-turn** grounded Q&A. NEXT: **multi-turn** dialogue coherence — wire the validated `MultiTurnAgent`
  + `SpikingLoopContextBuffer` anaphora (2026-06-17 GO; biased-competition disambiguation opt-in) onto the turn so a
  pronoun ("it") in a follow-up resolves to the held referent; then the neural interrogative parser; then open
  **breadth** via retrieval-augmentation + abstention (roadmap GAP B).
- The next-next lever if fluency/breadth needs more: run the (already-validated) 88.6M-style spiking forward to make
  the small generator literally **spiking-on-the-one-brain** (roadmap Phase 1), and a Phase-3 thalamocortical
  transformer-free science bet in parallel.

**Artifacts:** `_fluidconv_phase2_ra_finetune.py`, `_fluidconv_phase2_ra_qa_eval_derisk.py`,
`_fluidconv_phase3_conversational_turn_derisk.py`; ckpt `research/findings/raw/fluidconv/gen_tinystories_ra_ft.ckpt.pt`;
results `_fluidconv_phase2_ra_qa_eval.json`, `_fluidconv_phase3_conversational_turn.json`. NO `sim/` edit.
