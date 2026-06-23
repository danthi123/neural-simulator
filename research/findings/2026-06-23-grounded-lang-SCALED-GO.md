# 🎉 Grounded-language ROBUST AT SCALE (~67 facts) — brain recall 1.0 + moat 0-false-accept (3 seeds) + spiking-faculty grounded-render 10/10 (subject-first), drift caught. The no-confab moat holds at the bigger vocab WITH a real generative LLM in the loop (2026-06-23)

**The end-to-end grounded-language demo scaled to ~67 facts (2.2× the de-risk, 138-word vocab): the brain-half
(store/recall/abstain) holds recall 18/18=1.0 + moat 0/8-false-accept across 3 seeds; the real spiking Qwen2.5-0.5B
faculty (T=16, subject-first prompt) renders grounded facts 10/10 fluent + re-parse-verified, untaught abstain,
adversarial drift caught — 2 render seeds. ⇒ the grounded-language capability is ROBUST at scale; the no-confab moat
holds at the bigger vocab EVEN WITH a real generative LLM in the loop.** `research/runners/_grounded_lang_scaled_demo.py`,
brain numpy-CPU + faculty PyTorch-3090, NO `sim/` edit. (Owner-chosen consolidation of the just-completed arc.)

## Brain-half at scale (full query set, multi-seed)
| seed | recall | moat false-accepts |
|---|---|---|
| 42 | 18/18 (1.00) | 0/8 |
| 43 | 18/18 (1.00) | 0/8 |
| 44 | 18/18 (1.00) | 0/8 |

The moat holds 0-FA at the 138-word vocab including the SUBTLE cues: `whale eat` / `lion chase` (taught only an
*attribute*, no action-fact) and `cat give` / `dog make` (agent known, no fact for that relation) — all abstain.

## End-to-end grounded render (spiking Qwen2.5-0.5B, T=16, subject-first)
| render seed | grounded | first-render (no regen) | moat-held | drift-caught |
|---|---|---|---|---|
| 42 | 10/10 | 10/10 | 7/7 | 2/2 |
| 43 | 10/10 | 10/10 | 7/7 | 2/2 |

**Subject-first LIFTS the render +0.20 first-render** vs the loose de-risk prompt (which object-fronts → "Rabbit
chased fox." → verify-rejected → would need regen). Samples (verbatim): "Dog eats meat." / "The horse eats hay." /
"Wolves eat deer." / "The fox chased the rabbit." (all re-parse-verified); drift (dog→apple) caught; moat
(dragon eat) abstains. **T=8** (faster faculty, ppl 1.21×): grounded 9/10 (the one miss = verb-synonym drift
"devoured", correctly caught) — viable; **T=16 is the clean operating point**.

## The one scale-limit (found + fixed; NOT a moat/faculty failure)
At 67 facts the 0.5B faculty PLURALIZES generic facts ("Wolves eat deer", "Frog eats flies" — both the TRUE fact);
the de-risk's verb-only VERIFY extractor couldn't re-parse them → 2/10 true renders were conservative-correctly
REJECTED (silence, NOT a moat breach, NOT confabulation). FIX: noun-pluralization normalization in VERIFY (wolves→wolf,
flies→fly; invariant fish/deer preserved) → grounded restored to 10/10. The moat is untouched (the gate abstains
before any render). A re-parser coverage gap, not a substrate limit.

## ⇒ the grounded-language arc: complete + robust
P1 (fluency, the spiking Qwen) + P2 (knowledge, the brain learns+recalls+abstains) + P3 (grounding, gate→constrain→
verify) + integration + SCALE — all GO. A spiking LLM speaks the brain's grounded knowledge, hallucination-proof,
robust at ~67 facts multi-seed. HONEST SCOPE: still de-risk-class scale (67 facts, not thousands); the faculty is
PyTorch off the bridge (bridge co-residence = the "one brain" consolidation, the natural next deepening, exactly as
with the generative arc's C1). Further scaling (thousands of facts) + the bridge co-residence are the follow-ons.
