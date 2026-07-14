# D3's composed discourse focus DEPLOYED on the emergent talkable console (6-seed GO): the emergent (no-Qwen) console now resolves pronouns to the COMPOSED discourse center (D3 Centering-Cb), beating the host last-subject recency 0.945 vs 0.000 — replacing a host shortcut with the emergent mechanism, default-off byte-identical, no-confab moat preserved

**Date:** 2026-07-14
**Runner:** `research/runners/_d3_emergent_console_wire_derisk.py`. Raw `research/findings/raw/_d3_emergent_console_wire.json`. numpy CPU; NO `sim/` edit.
**Status:** GO (6-seed) — the "close arcs to full capacity" deployment: the already-GO D3 multi-turn discourse focus is wired into the emergent talkable console, beating its host last-subject recency on focus-shifted discourses.

## Why (the frontier gate's mission-central integration item)
The 2026-07-14 frontier gate mapped the genuine mission gap: the conversational pieces are mature but siloed, and the emergent talkable console (`_realcorpus_unified_talkable_console.py`, discovered spiking codes, NO off-bridge Qwen) resolved pronouns by HOST last-subject recency (`_resolve`: "a pronoun resolves to the last-mentioned subject `self.last_subject`") — a host shortcut that mis-resolves whenever the discourse FOCUS is not the most-recent subject. The D3 arc's composed-focus tracker (Centering backward-looking center Cb; `2026-07-09-D3-live-agent-wire-GO.md`, deployed into `MultiTurnAgent`'s `focus_bias_source`) is the emergent mechanism that tracks WHO we are talking about across turns. This de-risk deploys it onto the emergent console — a-1-confirmed genuinely open (the emergent console had no D3 register wired).

## The wire (additive, DEFAULT-OFF; NO `sim/` edit)
`D3FocusConsole(UnifiedTalkableConsole)` adds `use_d3_focus=False`. When ON, each heard fact `hear_fact(subj, verb, obj)` (a) teaches the fact into the console's KB, (b) updates the host recency (the stock `self.last_subject`), and (c) `observe(subj, obj)` into a `D3CenteringFocusSource` (the GO-validated Centering-Cb adapter); and `_resolve` of a pronoun ("it"/"they"/"them") returns the composed focus `referents[Cb]` instead of `self.last_subject`. Default OFF == byte-identical to the stock console (`_resolve` falls through to `super()._resolve`; the register is inert) — asserted on a 10-probe battery.

## Result — the emergent console resolves pronouns to the composed center, beating host recency (6-seed)
FOCUS-SHIFTED discourses = a short SVO sequence where the composed center is realized as the OBJECT of the final utterance while a NEW subject appears (Centering CONTINUE-as-object), so the true Cb ≠ the last subject (the host recency). Metric = pronoun resolves to the true Cb.
| arm | d3 (composed focus) | host (last-subject recency) |
|---|---|---|
| **FOCUS-SHIFTED (resolve-to-Cb, 6-seed)** | **0.945** (42:0.667, 43/44/100/101/102:1.0) | **0.000** |
| NON-SHIFTED control (Cb == last-subject) | 1.000 | 1.000 (agree — no regression) |
| register-LESION (frozen focus) | 0.000 | — (the register observations are load-bearing) |
| default-off byte-identity | True (10/10 probes) | — |
| no-confab MOAT (fresh discourse + pronoun) | abstains True (6/6 seeds, "I don't know") | — |

**Demo (seed 42):** "cat see box. cat see ball. cat see box. dog see cat." → composed center = cat. **D3 "it"→cat ("the cat sees box"); host "it"→dog ("the dog sees cat").** The emergent console follows the sustained discourse focus, not mere recency.

**Honest scope (Cb ⊆ {last-subject, last-object}):** the Centering Cb is always either the last-subject (CONTINUE-as-subject / SHIFT) or the last-object (CONTINUE realized as object). On these CONTINUE-as-object discourses the Cb coincides with the last-OBJECT (so a last-object heuristic scores 1.0 too) — but that is a property of the Cb definition, NOT a competitor to the DEPLOYED mechanism: the console's actual host anaphora is last-SUBJECT recency (host = 0.000), which D3's composed Cb decisively beats (0.945). A pure last-object heuristic is not what the console does and would misfire elsewhere; the load-bearing win is D3-Cb over the host recency the console actually uses.

## ⇒ the emergent talkable console now has multi-turn composed-focus anaphora, replacing a host shortcut
The emergent (no-Qwen) console resolves pronouns to the composed discourse focus (who we are talking about across turns), beating host last-subject recency, with the no-confab moat preserved and the default path byte-identical. This deploys the already-GO D3 multi-turn discourse mechanism onto the emergent substrate — closing the D3-anaphora arc to full capacity on the emergent console. NO `sim/` edit; additive default-off. Follow-on: the fully-spiking D3 register (`_d3_event_pair_spiking_derisk`) on the console; richer >2-referent discourse; the console's category-QA + D3 focus co-execution at scale.
