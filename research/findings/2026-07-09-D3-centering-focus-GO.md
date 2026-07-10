# D3 tracks the AGENT's actual discourse center (Centering Cb over SVO) — the foundation for the production MultiTurnAgent wire-in

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_centering_focus_derisk.py` (reuse-by-import: `discrete_attractor_rnn`; numpy; NO `sim/` edit).
**Verdict:** GO (6-seed) — a second genuinely-linguistic iterative-compose focus rule, on the agent's discourse structure.

## Why (toward the production wire-in)
The anaphora integration (`2026-07-09-D3-anaphora-integration-GO.md`) drove the real biased-competition resolution with D3's focus tracked on the POSSESSION δ. The deployed `MultiTurnAgent` hears SVO facts and resolves pronouns via the host `content_bias_target` (feature-compatibility) shortcut. To CLOSE that host shortcut with D3's **brain-based composed focus**, D3 must track the agent's ACTUAL discourse center. This models it: **Centering Theory's backward-looking center Cb over SVO utterances** (Grosz-Joshi-Weinstein 1995).

## The δ (a genuinely different focus rule than possession)
`δ(Cb, (subj=s, obj=o)) = Cb if Cb ∈ {s, o} else s` — the center CONTINUES if it is realized in the current utterance, else SHIFTS to the new subject (Cb = the highest-ranked realized center; subject-preferred). A state-dependent single-K-way focus update. The pronoun binds to Cb. Encoded as [subj-half ; obj-half] noisy ±1 pool codes.

## The result (6-seed; K=6; NO `sim/` edit)
| held-out-DEEPER (lengths 6/7/8; chance 1/6=0.167), 6-seed | value |
|---|---|
| **D3 discourse-center (Cb) track** | **0.970** (step-delta **1.000**, every seed) |
| RECENCY baseline (bind to the last-mentioned = last object) | **0.000** (every seed) |

(6-seed 42/43/44/100/101/102: D3 Cb-track 0.96–0.98, recency 0.0 — robust every seed.)

**GO:** the discrete-attractor tracks the discourse CENTER (Cb, Centering Theory) over SVO utterances to held-out-DEEPER lengths where the RECENCY (last-object) baseline FAILS at chance — the center continues while a new object is mentioned, so "last-mentioned" is wrong while D3's composed Cb persists.

## ⇒ D3 models the agent's actual discourse focus
This is a **second** genuinely-linguistic iterative-compose focus rule (Centering over SVO, distinct from possession-tracking), on the SVO discourse structure the deployed `MultiTurnAgent` actually hears. It is the **foundation for the production wire-in**: D3's composed Cb (brain-based) replaces the host `content_bias_target` shortcut as the resolution bias — the pronoun binds to the composed discourse center, not mere recency or a host feature-lookup.

## Honest scope + next
- The Cb tracker is a numpy discrete-attractor (per-step-supervised δ; the spiking transition+FS-WTA drops in, as the anaphora integration showed).
- **Next (the production wire-in):** add a default-off `focus_bias_source` hook to `MultiTurnAgent._resolve_biased` so the Cb tracker supplies `fav` in place of `content_bias_target`; validate on a MultiTurnAgent discourse where the center shifts. Combine Cb + feature-compatibility for the full resolution.

## Files
`research/runners/_d3_centering_focus_derisk.py`; the D3 arc `2026-07-09-D3-*.md`.
