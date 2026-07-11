# RUNG 2 GO (6-seed) — a non-fading working-memory LATCH restores a distal discourse referent the fading reservoir loses: the buffer conditions the emergent generator to carry a topic 32 clauses back that the reservoir alone forgets to chance

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_rung2_distal_referent_derisk.py` (reuse-by-import: the Rung-1 reservoir-LM read-out + EMERGE-82 `OnBridgeLSM` fixed spiking reservoir + EMERGE-62 corpus/vocab; NO `sim/` edit, NO edit to any existing runner).
**Verdict:** GO — Rung 2 of the emergent-generation ladder: conditioning the generator on **distal discourse context** held in a working-memory latch. **Adversarial-verify `w3h2q6obu` = SURVIVES, commit-as-is** (4 agents): leakage closed by 3 independent code-inspection guarantees (the buffer latches the topic SUBJECT, never the den; the den is at target position, absent from the scored feature; dens are disjoint from subjects/fillers) + referent-shuffle→0; content-vs-capacity closed by the byte-identical slot-scramble arm collapsing to reservoir-only (reinforced by top-1 dep-acc being calibration-invariant); reservoir-fairness/genuine-forgetting certified by the horizon sweep (reservoir-only 0.55@D2 → 0.125@D32, high-at-small-D decaying = genuine fading, not a broken baseline); scope accurately bounded, no fix needed. The leave-one-topic-out generalization probe is belt-and-suspenders for a claim this GO does NOT make (fixed-bijection recall, not novel-pair generalization) — reserved for the Rung-3 generalization extension.

## The mechanism (emergent generator + a non-fading single-slot WM latch, no deep credit, no BPTT)
Rung 1 showed a fixed spiking reservoir + a shallow one-step-local-delta read-out predicts the next token, but only within the reservoir's **fading memory** (~depth-3). Rung 2 adds a **non-fading single-slot topic buffer** — a prefrontal working-memory latch / Grosz-Sidner attentional focus (the functional rate-level analogue of a slow-NMDA / theta-gamma WM slot): its **write gate is opened by a discourse boundary marker** (`DOC`) and latches the *topic subject's identity* (not the answer); the held identity is exposed as a fixed-length feature that **concatenates to the reservoir state** before the *same* one-step-local-delta read-out. The reservoir supplies fading local context; the latch supplies the one distal thing the reservoir loses.

## The task (a distal-referent discourse dependency, leakage-proof)
Each document: a boundary marker → `the <TOPIC> can <v>` (topic introduced once) → **D filler clauses about DIFFERENT same-category subjects** (so a bag-of-words can't read the topic, and the same-category subjects drive agreement-attraction interference on the reservoir) → a continuation whose **dependent token = the topic's HOME "den"**, a per-seed frozen bijection (topic → a UNIQUE object) that is **disjoint from the fillers and never mentioned in the intro or any filler** (no copy/repeat shortcut). Only that dependent token is scored. Chance = 1/K = 0.167 (K=6).

## The crux — a HORIZON SWEEP (the reservoir must genuinely forget before the buffer can matter)
The reservoir is *good* at retention (EMERGE-81: holds a 1-bit cue ≥16 fillers; EMERGE-83: resists agreement-attraction to depth ≥4). But a full **topic identity** across same-category interference decays faster. Sweeping the filler distance D (seed 42, reservoir-only dep_acc): **0.550 (D=2) → 0.400 (D=8) → 0.225 (D=16) → 0.125 (D=32)** — monotone decay crossing chance (0.167) between D=16 and D=32. So at **D\*=32 the reservoir alone is at/below chance** — it has genuinely forgotten which topic was introduced. The buffer's advantage is measured in that beyond-horizon regime (not where the reservoir already carries it).

## The result (6-seed at D\*=32: dev 42/43/44 + blind 100/101/102; chance 0.167)
| Arm | mean dep-acc | per-seed |
|---|---|---|
| Reservoir alone | **0.125** (at chance) | 0.075–0.175 |
| **Reservoir + topic buffer** | **0.971** | 0.900–1.000 |
| Slot-scramble (latch a RANDOM subject) | 0.137 | ≈ reservoir-only |
| Referent-shuffle (deranged topic→den map) | 0.000 | 0.000 all seeds |
| Bag-of-prefix | 0.163 | ≈ chance |
| Bigram | 0.171 | ≈ chance |

**Buffer margin over reservoir-only: +0.846, 6/6 seeds; every anti-cheat collapses 6/6.** The reservoir-only cross-entropy at the dependent token is 3.86 (at chance); the buffer's is 0.19 (confident). The reservoir is genuinely active (~0.020 spikes/neuron/step).

## Why each control is load-bearing
- **Reservoir alone at chance (0.125):** the reservoir has forgotten the topic 32 clauses back — the buffer is not decorating a solved task.
- **Slot-scramble = reservoir-only (0.137):** latching a *random* subject (same buffer dimensionality, wrong content) gives nothing → the **held content**, not the extra dimensions/capacity, is what helps.
- **Referent-shuffle = 0.000:** breaking the topic→den mapping in training collapses test to zero → the read-out learned a **real topic→answer mapping**; the buffer holds the *topic*, not the answer (no leakage — a leaked answer would survive a shuffled map).
- **Bag = bigram = chance:** the topic is not readable from unordered prefix counts (the same-category distractors mask it) or from the previous token.

## Honest scope
A **controlled discourse-referent grammar** (a closed template domain, K=6 topics): this demonstrates that a working-memory latch **restores a distal discourse dependency the fading reservoir loses** — recall of a per-seed topic→den fact conditioned on a distally-introduced referent. It is NOT open prose (R4), NOT held-out compositional generalization (that is Rung 3), and the buffer is a **functional single-slot latch** (the fully-spiking theta-gamma / slow-NMDA WM realization on the substrate is the follow-on rung). The emergent generator (Rung 1) + a non-fading WM slot together carry distal context — with no deep credit, no backprop-through-time, no `sim/` edit.

## ⇒ significance
The second validated rung of the emergent-generation ladder: **generation conditioned on distal discourse context**, via the reservoir's fading memory plus a biologically-motivated non-fading working-memory latch. Together they hold a referent across a span the reservoir alone cannot — the mechanism a conversation needs to track a topic across turns. NEXT: Rung 3 (compositional generalization of generation — a never-seen combination produced grammatically).

## Files
`_emerge_reservoir_lm_rung2_distal_referent_derisk.py`; horizon sweep `research/findings/raw/_reslm_rung2_horizon_s42.json`; 6-seed `research/findings/raw/_reslm_rung2_s{42,43,44,100,101,102}.json`; builds on `2026-07-10-RUNG1-emergent-reservoir-next-token-LM-dynamics-earned-GO-6seed.md` + `2026-07-10-emergent-language-cortex-scoping-the-generation-gap.md`.
