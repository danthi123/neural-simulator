# Chat demo on biology-grounded Phase 1.4 BRANCH A foundation
**Seed:** 42
**Training:** Tier 1 embodied Hebbian, 200 events/word

---

## Conversation transcript

```
[SYSTEM] Trained 4-word vocab via Tier 1 embodied Hebbian (seed=42, 200 events/word).

--- Round 1/3 ---
  [OK] You: north  -> Sim: north  (N204 E189 S190 W131, confidence x1.1)
  [OK] You: east   -> Sim: east   (N211 E225 S169 W211, confidence x1.1)
  [X] You: south  -> Sim: north  (N198 E174 S160 W177, confidence x1.1)
  [OK] You: west   -> Sim: west   (N202 E173 S184 W223, confidence x1.1)
--- Round 2/3 ---
  [X] You: north  -> Sim: south  (N196 E161 S208 W192, confidence x1.1)
  [X] You: east   -> Sim: west   (N188 E222 S196 W237, confidence x1.1)
  [X] You: south  -> Sim: north  (N209 E207 S190 W184, confidence x1.0)
  [X] You: west   -> Sim: north  (N200 E166 S178 W175, confidence x1.1)
--- Round 3/3 ---
  [X] You: north  -> Sim: west   (N230 E208 S164 W233, confidence x1.0)
  [X] You: east   -> Sim: west   (N229 E208 S186 W236, confidence x1.0)
  [X] You: south  -> Sim: west   (N197 E173 S187 W230, confidence x1.2)
  [OK] You: west   -> Sim: west   (N193 E162 S191 W217, confidence x1.1)

Accuracy: 4/12 = 33.3%
```

---

## What this demonstrates

- Tier 1 embodied Hebbian binding (Phase 1.4 architecture)
- All learning biology-grounded: STDP + co-firing teachers
- No backprop, no surrogate gradients
- 4-word vocabulary, scriptable to 8/12 with Tier 2.1 synonym mode
- Continual learning preserved (Phase 1.4 BRANCH A: 5/6 PASS, mean 103% retention)
- Memory consolidation works (Phase 1.3: 3/3 PASS, mean 96% hippo-OFF retention)

First conversational artifact built on the validated biology-grounded continual-learning + memory consolidation foundation.
