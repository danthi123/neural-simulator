# Chat demo on biology-grounded Phase 1.4 BRANCH A foundation
**Seed:** 43
**Training:** Tier 1 embodied Hebbian, 200 events/word

---

## Conversation transcript

```
[SYSTEM] Trained 4-word vocab via Tier 1 embodied Hebbian (seed=43, 200 events/word).

--- Round 1/3 ---
  [OK] You: north  -> Sim: north  (delta N +66 E +10 S +38 W +13, x1.7)
  [X] You: east   -> Sim: west   (delta N +26 E -13 S -15 W +56, x2.2)
  [X] You: south  -> Sim: east   (delta N  +0 E +15 S  +9 W -26, x1.7)
  [OK] You: west   -> Sim: west   (delta N  -8 E -61 S  +2 W +12, x6.0)
--- Round 2/3 ---
  [X] You: north  -> Sim: east   (delta N  +5 E +23 S -31 W  -1, x4.6)
  [OK] You: east   -> Sim: east   (delta N +16 E +17 S +16 W -40, x1.1)
  [OK] You: south  -> Sim: south  (delta N +15 E +24 S +34 W -48, x1.4)
  [X] You: west   -> Sim: north  (delta N +29 E  -6 S +22 W  +1, x1.3)
--- Round 3/3 ---
  [X] You: north  -> Sim: east   (delta N -19 E +15 S  +6 W -18, x2.5)
  [X] You: east   -> Sim: south  (delta N -16 E +11 S +14 W -43, x1.3)
  [X] You: south  -> Sim: west   (delta N +27 E +35 S -16 W +44, x1.3)
  [OK] You: west   -> Sim: west   (delta N -22 E  +9 S +14 W +54, x3.9)

Accuracy: 5/12 = 41.7%
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
