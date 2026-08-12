---
status: live
lane: gap#1
date: 2026-08-12
type: finding
---

# Spiking comprehension-success monitor — the parser's sel-pool WTA firing margin reads "did I understand this utterance" (GO, 6-seed); the positional BridgeParser margin is content-blind (mapped boundary)

**Date:** 2026-08-12 · **Runner:** `research/runners/_spiking_comprehension_monitor_derisk.py` · **Reuse-by-import, NO `sim/` edit.**

## Headline

A **genuinely spiking read of parse success** exists and de-risks 6/6. Given an incoming transitive utterance,
the **firing margin of the on-brain multi-cue role competition's `sel_agent`/`sel_patient` Wong-Wang
accumulators** — driven by the CONTENT (animacy + verb-fit) cue populations and read from
`bridge.cp_firing_states` — is **HIGH when the thematic role-binding resolved cleanly** (a well-formed,
in-vocabulary SVO) and **LOW when it did not** (out-of-vocab nouns, or a content-ambiguous / role-symmetric
string). It separates well-formed from ill-formed utterances with **type-2-like AUC 1.000 (6/6 seeds)**;
lesioning the learned cue→role synapses **collapses it to exactly 0.500** (chance); the read is on firing
neurons (the host `_semantic_contrast` formula is never called); it reads in a couple hundred ms per turn.

This is the discrimination the lane-C metacognition monitor FAILED (type-2 at chance) — but for a **different
faculty** (memory-retrieval groundability), and with a different signal. Here the parser's OWN competition
dynamics carry the confidence: the balance of firing between the competing role accumulators is a clean,
load-bearing correlate of whether the content decisively determined the roles.

**The task's primary SUGGESTED mechanism — the `BridgeParser` 3-role-ensemble margin — is a mapped BOUNDARY,
not the answer.** `BridgeParser.role_of` drives a `(position × voice)` conjunction ALONE and never reads the
token, so its winner-vs-runner-up firing margin is **content-blind**: a per-seed constant (4.2–6.5) identical
for well-formed and OOV/ambiguous lexical input → **AUC = 0.500 (chance) on every seed**. The positional parser cannot monitor
its own comprehension; the content-sensitive competition parser can.

## Mechanism (what is read, and why it is spiking not host)

The production on-brain parser has two role-assigners. The `BridgeParser` (`brain_conversational_agent.py:28`)
maps `(word-position × voice) → role` and is **purely positional — it ignores the token identity entirely**
(`role_of` drives only the conjunction unit; `parse` slaps `words[pos]` onto the resolved role post-hoc). The
`MultiCueRoleParser` / `SpikingRoleCompetition`
(`_phaseB_multicue_competition_spiking_derisk.py`) is **content-sensitive**: each cue (position, animacy,
verb-fit, lexbias) is a spiking population projecting through learned synaptic weights (= the cue validities)
into two Wong-Wang accumulators `sel_agent`/`sel_patient` in mutual inhibition; the WTA settle is the role
decision.

**The comprehension-success read (R-primary):** for each of the sentence's two nouns, drive ONLY the SEMANTIC
(animacy + verb-fit) cue populations, let the WTA settle, and read `agentEv = firing(sel_agent) −
firing(sel_patient)` **from `cp_firing_states`** (via `SpikingRoleCompetition._noun_role_rates`). The
sentence-level comprehension scalar is the spiking contrast `|agentEv_noun0 − agentEv_noun1|`. It is HIGH when
the content decisively separates an animate agent from an inanimate patient, LOW when the content cancels (two
animate + a symmetric verb) or is absent (OOV → the animacy/verb-fit populations get no drive). This is the
**spiking analogue of the no-confab moat's content gate**, which currently ships as a HOST dot-product
`_semantic_contrast(evs) = Σ w[c]·rel·vote` (`_phaseB_multicue_competition_spiking_derisk.py:663`) — a Python
formula over the learned weights, NOT read from spikes. The de-risk replaces that host read with the settled
sel-pool firing the competition already produces.

Grounding: the **balance of evidence between competing accumulators is the classic neural correlate of
decision confidence** (bounded-accumulation / drift-diffusion; balance-of-evidence models of confidence). Here
that balance is realized as the firing difference of two spiking Wong-Wang pools, so the confidence read is a
by-product of the same substrate that makes the decision — not a second, host-side monitor bolted on.

## Results (6 seeds: 42/43/44/100/101/102, CuPy on RTX 3090)

Battery per seed: 20 each of WELL-FORMED (animate agent + asymmetric verb + inanimate patient) and three
ILL-FORMED classes — AMBIGUOUS-2-animate (symmetric verb), NONSENSE-2-inanimate (animacy cancels), and
OUT-OF-VOCAB (unknown nouns → semantic cues absent). Label 1 = should-comprehend, 0 = should-not.

| metric | per-seed (42/43/44/100/101/102) | 6-seed |
|---|---|---|
| **R-primary AUC** (semantic sel-pool spiking margin, well vs ill) | 1.000 · 1.000 · 1.000 · 1.000 · 1.000 · 1.000 | **1.000** |
| **LESION AUC** (zero the learned cue→role synapses; cue VALUES unchanged) | 0.500 · 0.500 · 0.500 · 0.500 · 0.500 · 0.500 | **0.500** |
| R-full control AUC (position-INCLUSIVE margin; rounded from `auc_full_control`) <!--derived--> | 0.784 · 0.762 · 0.780 · 0.918 · 0.889 · 0.805 | **0.823** |
| mean margin WELL-FORMED | ~0.32–0.36 | — |
| mean margin ILL-FORMED | ~0.09–0.11 | — |
| per-tag means (well ≫ ambiguous > OOV) | well ~0.33 · ambig ~0.13 · oov ~0.04 | monotonic 6/6 |
| **BridgeParser 3-role margin** (content-blind constant) | 4.2–6.5, AUC **0.500** (6/6) | boundary |
| per-turn read wall-time (GPU, launch-bound tiny net) | 0.18–0.62 s (2 reads) | conversation-feasible |

**Frozen GO gate (pre-registered, not tuned-to-pass):** (1) R-primary AUC ≥ 0.80 on ≥5/6 seeds; (2) LESION
collapses AUC to ≤ 0.60 on ≥5/6 seeds; (3) the read is on firing neurons (host `_semantic_contrast` never
called). **ALL THREE MET — GO 6/6** (`c1`, `c2`, `c3` all true; R-primary AUC 1.000 and LESION AUC 0.500 on
every seed, not merely 5/6).

## Anti-cheat controls (all passed)

- **LESION collapses to exactly 0.500 (6/6):** zeroing the learned cue→role synaptic weights — while the host
  cue VALUES fed to the substrate are byte-identical — drops discrimination to chance. This proves the
  discrimination is **caused by the learned spiking competition settling on the cues**, not by the cue values
  (which would make the metric a tautology of the battery labels). It is the decisive load-bearing control.
  The lesion is verified to PERSIST at measurement time (plasticity gates are frozen before the weights are
  zeroed, so the zeroed cue→role weights stay exactly 0.0 through every read-settle — checked directly; the
  exact-0.500 AUC on all 6 seeds also confirms no partial regrowth). **Attribution (`tools.lab.attributable_to`):
  treatment = intact above-chance separation (1.000 − 0.5 = 0.5), control = lesioned above-chance separation
  (0.500 − 0.5 = 0.0) ⇒ 100% of the discrimination is attributable to the learned spiking competition, 0% is
  in the control.**
- **On firing neurons, not a host formula:** every margin is accumulated from `bridge.cp_firing_states[sel_idx]`;
  the runtime flag `host_semantic_contrast_used=False` and the code path never calls `_semantic_contrast`.
- **Non-canonical defense (verify-go arm):** on an object-fronted well-formed sentence ("apple eat dog" —
  inanimate patient first, animate agent last), the semantic read STAYS HIGH (~0.30, the content decisively
  resolves the roles) while the position-inclusive read COLLAPSES (~0.03, position fights content).
  AUC(object-front-well vs OOV) = **1.000 on all 6 seeds**, and the position-inclusive read is below the
  semantic read on object-front on 6/6 seeds → the monitor reads COMPREHENSION, not canonical word order. This
  is the sharpest defense: a canonical-order detector would call the comprehensible "apple eat dog"
  NOT-understood; the semantic sel-pool read does not.
- **Positional-parser boundary (honest negative on the suggested mechanism):** the `BridgeParser` 3-role
  margin is content-blind (constant ~5.4, AUC 0.500), documented so the suggested mechanism is not
  mis-adopted.

## Honest scope + residual

- **Covered:** the 2-noun transitive AGENT/PATIENT role-binding — the core SVO comprehension the on-brain
  parser handles — across OOV and content-ambiguity. This is the utterance-comprehension read the moat gate
  needs, now genuinely spiking.
- **NOT yet covered (bounded next steps, same mechanism):** multi-clause, attributed (adj+noun), questions,
  and non-transitives route through OTHER parsers (`EmbeddedClauseParser`, `AttributedBridgeParser`,
  `FrameParser`, the question-routing reservoir); extending the sel-pool-margin read to each is additive.
- **Residual host boundary:** a STRUCTURALLY malformed string (no verb / wrong arity) is still caught by a
  host arity check in `_split_verb_nouns`, not the spiking read. The spiking read covers OOV + content
  ambiguity (the "ambiguous role assignment" the task named); structural malformedness is a separate gate.
- **Ceiling caveat (honest):** the R-primary AUC is pinned at **1.000** because the battery uses cleanly
  decisive vs cleanly-cancelling constructions — the mechanism separates the DESIGNED categories perfectly.
  The load-bearing evidence is the ceiling PLUS the lesion collapse to 0.500 (the separation is caused by the
  competition, not battery-triviality). A pinned ceiling is uninformative about the operating point on MESSY
  real utterances; the informative next test is a graded battery (partially-degraded / near-threshold cues)
  where AUC lands below ceiling and the per-turn threshold can be calibrated. This de-risk establishes the
  signal EXISTS and is load-bearing; the operating-point calibration is the next step.

## Wireable into the live turn (honest "I didn't follow that")

`hear_multicue` / `parse_decisive` already run this exact competition in production (opt-in). The spiking
comprehension margin is a by-product of the `assign_roles` settle that already happens per turn: read the
settled `sel` firing instead of (or alongside) the host `_semantic_contrast`, and gate the turn on it — margin
below a per-seed threshold ⇒ the brain reports "my role-binding did not resolve, I didn't follow that" and
ABSTAINS (strengthening the moat at the comprehension layer, never weakening it). Cost is the one settle it
already pays (~0.2 s here on a launch-bound tiny GPU net; far less batched/on CPU).

## Reproduce

```bash
SIM_BACKEND=cupy python -u -m research.runners._spiking_comprehension_monitor_derisk \
    --seeds 42,43,44,100,101,102 --n-per-cond 20 \
    --out research/findings/raw/_spiking_comprehension_monitor.json
# verify-go non-canonical arm:
SIM_BACKEND=cupy python -u -m research.runners._spiking_comprehension_monitor_derisk \
    --noncanon-verify --seeds 42,43,44,100,101,102 --n-per-cond 20 \
    --out research/findings/raw/_spiking_comprehension_monitor_noncanon.json
```
Raw: `research/findings/raw/_spiking_comprehension_monitor.json`.
