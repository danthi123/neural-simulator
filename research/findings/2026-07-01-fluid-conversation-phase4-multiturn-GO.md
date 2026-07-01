# Fluid conversation — Phase 4 GO: multi-turn grounded dialogue (a pronoun resolves to the held referent)

**2026-07-01 (autonomous night; owner's fluid-conversation priority).** Phase 3 gave the full single-turn grounded
conversation. Phase 4 closes the **multi-turn** axis (the owner's "fluid back-and-forth"): a persistent SPIKING
discourse working-memory loop holds the salient referent (the ANSWER of the prior turn) across a turn boundary, so a
later turn's PRONOUN ("what does IT eat?") resolves to it — then routes through the same brain-GATE → RA-fine-tuned
21M answer → post-hoc VERIFY pipeline. Reuse of the VALIDATED anaphora machinery (`MultiTurnAgent`, 2026-06-17 GO);
NO `sim/` edit.

## Result — GO (3 seeds), the multi-turn transcript
Discourse chain (curriculum-derived): dog-chase-cat + cat-eat-fish. `_fluidconv_phase4_multiturn_derisk.py`:
```
you> what does the dog chase?    brain> the dog chases cat.
you> what does it eat?           brain> the cat eats fish.   (it -> cat)
```

| gate | result (3 seeds) |
|---|---|
| **ANAPHORA** — turn-2 "it" resolves to the turn-1 answer (cat, held in the spiking WM) → correct grounded answer (fish) | **3/3** |
| **WM-LESION** — wipe the WM loop before turn 2 → the pronoun does NOT resolve → abstain (the carry is load-bearing) | **3/3** |
| **EMPTY-WM MOAT** — a turn-2 pronoun with NO prior turn → abstain (no confabulated antecedent) | **3/3** |
| **SINGLE-TURN unregressed** — a direct-subject turn still answers grounded | **3/3** |

## How it works (the pieces, all validated)
- The pronoun resolution is the project's validated `MultiTurnAgent._resolve` → `held_referent` over a
  `SpikingLoopContextBuffer` (one spiking attractor per referent; the dominant attractor is the antecedent; ambiguous
  / empty → None → abstain). After each turn the ANSWER (patient) is written into the WM as the next salient referent
  (exactly what `hear` does for a heard statement) — so discourse salience carries forward.
- The GATE (brain recall/abstain) + the RA-fine-tuned 21M focused answer + post-hoc VERIFY are the Phase-2/3 pipeline,
  unchanged. The no-confab moat is preserved at every turn (gate-first; an unresolved pronoun abstains).

## Honest scope + next
- **Single dominant held referent.** ≥2-referent disambiguation (which of several held referents a bare pronoun binds)
  is the biased-competition path (`enable_biased_competition`, validated separately 2026-06-19) — a drop-in opt-in
  when multi-referent dialogue is prioritized.
- **The referent set is small** (the WM loop holds one pattern_size-neuron attractor per referent within a fixed
  neuron pool; the referent count must stay well under n/pattern_size). Scaling the referent pool = a wider WM loop
  (a capacity lever, characterized, not a wall).
- **NEXT:** replace the interrogative-parse scaffold with a neural interrogative parser; then open breadth
  (retrieval-augmentation + abstention). The generative-fluency + on-substrate-spiking levers (run the 21M as a
  spiking forward on the one brain) remain the roadmap Phase-1 path.

**⇒ the fluid-conversation stack now spans:** minimal fluent generator (Phase 0) · fluid grounded rendering (Phase 1) ·
focused conversational Q&A via the brain-train fine-tune (Phase 2) · the full single-turn (Phase 3) · **multi-turn
anaphora (Phase 4)** — all on the minimized (~21M, 15–25× < Qwen-0.5B), brain-trained, brain-gated stack, moat
preserved, NO `sim/` edit anywhere in Phases 0–4.

**Artifacts:** `research/runners/_fluidconv_phase4_multiturn_derisk.py`; result
`research/findings/raw/_fluidconv_phase4_multiturn.json`.
