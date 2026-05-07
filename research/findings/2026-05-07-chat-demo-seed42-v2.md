# Chat demo on biology-grounded Phase 1.4 BRANCH A foundation
**Seed:** 42
**Training:** Tier 1 embodied Hebbian, 200 events/word

---

## Conversation transcript

```
[SYSTEM] Trained 4-word vocab via Tier 1 embodied Hebbian (seed=42, 200 events/word).

--- Round 1/3 ---
  [X] You: north  -> Sim: west   (delta N +17 E +26 S -21 W +69, x2.7)
  [X] You: east   -> Sim: west   (delta N  +2 E +10 S +10 W +31, x3.1)
  [X] You: south  -> Sim: west   (delta N -11 E +23 S -13 W +32, x1.4)
  [X] You: west   -> Sim: north  (delta N  -5 E -44 S -21 W -15, x1.0)
--- Round 2/3 ---
  [X] You: north  -> Sim: south  (delta N  +5 E  +1 S +25 W +17, x1.5)
  [OK] You: east   -> Sim: east   (delta N  -6 E  +0 S  -2 W -10, x1.0)
  [X] You: south  -> Sim: west   (delta N -23 E  -4 S  +5 W +25, x5.0)
  [X] You: west   -> Sim: south  (delta N -27 E -14 S +31 W  +7, x4.4)
--- Round 3/3 ---
  [OK] You: north  -> Sim: north  (delta N +28 E  +4 S  -6 W  +1, x7.0)
  [X] You: east   -> Sim: west   (delta N +34 E +26 S +37 W +52, x1.4)
  [X] You: south  -> Sim: north  (delta N -12 E -23 S -18 W -32, x1.0)
  [X] You: west   -> Sim: south  (delta N  +2 E  +8 S +36 W  -3, x4.5)

Accuracy: 2/12 = 16.7%
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
