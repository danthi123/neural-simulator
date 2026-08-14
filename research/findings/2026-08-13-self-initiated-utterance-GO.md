---
type: finding
status: live
date: 2026-08-13
mechanism: self-initiated-utterance
lane: self-initiation / conversation
runner: research/runners/_self_initiated_utterance_derisk.py
artifacts:
  - research/findings/raw/_self_initiated_utterance_derisk.json
---

# Self-initiated utterance — the loop CLOSES (6-seed GO): a spontaneously-SELECTED, curiosity-biased CA3 thought (no prompt) is routed into the composer/mouth and SPOKEN as a coherent utterance ABOUT that concept

**2026-08-13 (autonomous, GPU/cupy, n_ca3=2000, n_mem=4, D=256, rest 4000).** The named FINAL rung of the multibasin
self-initiation GO ([`2026-08-13-self-initiation-multibasin-GO.md`](2026-08-13-self-initiation-multibasin-GO.md),
rung #1: *"seed → utterance routing"*). That GO showed the noise-driven CA3 wander (no cue) SELECTS among several
disjoint balanced basins with a curiosity recurrent-gain biasing WHICH surfaces (66% attributable). This de-risk
routes that surfaced, curiosity-SELECTED concept into the production `OneBrainComposer` MOUTH so a spontaneous thought
becomes a **self-initiated remark / question**: internally-generated → selected → **SPOKEN**. **Functional
self-initiated-utterance CORRELATE only — no claim of phenomenal experience.**

## The mechanism (compose two validated GO organs; NO `sim/` edit; reuse-by-import)

_Config values + readouts from the committed artifact `research/findings/raw/_self_initiated_utterance_derisk.json`._

<!--derived-->

Two already-GO organs are composed with **zero** new spiking machinery:

1. **SELECTION (the substrate decides the TOPIC, 0 host content-draw)** — the multibasin self-initiation wander
   (6-seed GO), reused-by-import (`_self_initiation_multibasin_derisk._run_condition` / `_selection` + its DISJOINT
   pattern-separated CA3 store + the production curiosity recurrent-gain). Under weak non-specific Poisson (rate 0.015,
   1500 pA, dur 10 — the RANK-1 operating point; **NO cue, 0 external CONTENT drive**) each discrete noise-seeded volley
   ignites WHICHEVER balanced basin its coincidental overlap favours; the curiosity gain biases WHICH; the bistable KIR
   down-state returns the net to silence between events. The surfaced basin IS the self-initiated "thought" — which
   concept, and how often, is entirely the spiking attractor competition + noise (**no `random.choice` over concepts**).
2. **THE MOUTH (the composer turns the selected concept → words)** — the production `OneBrainComposer`
   (`one_brain_composer.py`), reused-by-import. Each stored concept is a 3-role fact composite in the bridge's complex
   RF synapses; `render_fact(concept)` reconstructs "concept verb patient" by an **ON-BRIDGE resonate-and-fire unbind +
   cleanup** (the spiking decode) and **abstains (None) on an unknown subject** (the no-confab moat, verified).

**The loop:** noise (no prompt) → spiking CA3 wander SELECTS a curiosity-biased basin → the basin's bound concept →
`OneBrainComposer.render_fact` → a spoken SVO utterance ABOUT that concept (e.g. `dog chase ball`), which a host
template wraps into a question form (`what does dog chase?`).

## Result — 6/6 GO (seeds 42, 43, 44, 100, 101, 102), gain-scale 1.0, rest 4000, n_mem=4

_Per-seed values are from the cited committed artifact — verify against the raw JSON. GPU/cupy summation order is not
byte-deterministic, so counts jitter run-to-run; the GO and every anti-cheat hold with margin._

<!--derived-->

| seed | ON utterances / distinct concepts | about-selected (vs SCRAMBLE) | coherence member vs rand | novel-utt share HIGH vs LOW(reversed) | attributable | NO-NOISE utt | STORE-LESION utt |
|------|-----------------------------------|------------------------------|--------------------------|----------------------------------------|--------------|--------------|-------------------|
| 42   | 331 / 3  | 1.00 vs 0.00 | 0.35 vs 0.04 | 0.87 vs 0.48 | 45% | 0 ✓ | 0 ✓ |
| 43   | 540 / 3  | 1.00 vs 0.00 | 0.39 vs 0.04 | 0.79 vs 0.10 | 88% | 0 ✓ | 0 ✓ |
| 44   | 234 / 3  | 1.00 vs 0.00 | 0.39 vs 0.04 | 0.47 vs 0.17 | 63% | 0 ✓ | 0 ✓ |
| 100  | 1018 / 3 | 1.00 vs 0.00 | 0.41 vs 0.04 | 0.95 vs 0.44 | 53% | 0 ✓ | 0 ✓ |
| 101  | 433 / 3  | 1.00 vs 0.00 | 0.40 vs 0.04 | 0.80 vs 0.19 | 77% | 0 ✓ | 0 ✓ |
| 102  | 303 / 3  | 1.00 vs 0.00 | 0.36 vs 0.04 | 0.67 vs 0.09 | 86% | 0 ✓ | 0 ✓ |

**Aggregate (6-seed):** production wander speaks a mean **476 utterances/session about 3.0 distinct concepts**;
about-selected **1.00** (every seed) vs SCRAMBLE-routing **0.00**; coherence member **0.38 vs random 0.04** (~9.5×
chance); novel-concept utterance share HIGH-gain **0.76 vs LOW-gain (reversed) 0.24** → **68% of the novel-concept
surfacing attributable to the curiosity gain**; NO-NOISE **0** utterances, STORE-LESION **0** utterances every seed.
The runner's OWN `Verdict` decided **GO** (all preconditions met). Example surfaced-then-spoken utterances (seed 42):
`dog chase ball` (→ `what does dog chase?`), `cat eat worm` (→ `what does cat eat?`).

Every seed → GO on all anti-cheats (each VERIFIED, not asserted):

- **MOUTH FIDELITY + no-confab moat.** Each stored fact decodes on-bridge (`render_fact` == the stored SVO) every
  seed; an UNKNOWN subject abstains (None) — the mouth speaks only what is stored, and only about the surfaced concept.
- **INTERNALLY-TRIGGERED.** 0 external CONTENT drive (only non-specific Poisson to random CA3-exc cells). **NO-NOISE
  (gains on, noise off) → 0 utterances** every seed (the wander is silent, so there is nothing to speak). Plasticity
  byte-FROZEN during the session every seed; apical not self-latched.
- **ABOUT-THE-SELECTED-CONCEPT (coherent).** Each utterance NAMES the concept bound to the basin the substrate
  actually ignited (about-selected 1.00), and the surfaced steps overlap the stored assembly (member 0.35–0.41 vs
  random 0.04). **SCRAMBLE-ROUTING** (route each basin to a WRONG concept, a derangement) → about-selected **0.00**:
  the basin↔concept correspondence is load-bearing.
- **CURIOSITY-STEERED (identity-controlled).** NOVEL concepts drive MATERIALLY more utterances under HIGH gain than
  under the REVERSED (anti-curiosity) gain (novel-utterance share 0.76 vs 0.24; 45–88% per seed) — the SAME concepts,
  opposite gains, so the bias is the curiosity VALUE not the basin identity.
- **SUBSTRATE-ATTRIBUTABLE (lesion the selection → no utterance).** **STORE-LESION (NO-ENCODE the CA3 store, same
  noise+gain) → member 0.00 → 0 utterances** every seed: the content-selection is the learned store, not the noise or
  the mouth.

## What is SUBSTRATE vs HOST (the honesty boundary is a deliverable, not a caveat)

- **SPIKING (load-bearing):** (i) the SELECTION of WHICH concept is spoken and HOW OFTEN — the CA3 dendritic-plateau
  attractor competition under non-specific noise (0 host content-draw / no `random.choice` over concepts); (ii) the
  steering VALUE (the curiosity ASK-pool want, read off `cp_firing_states`); (iii) the VERBALISATION of the SVO
  proposition — decoded by the `OneBrainComposer`'s on-bridge RF resonate unbind + cleanup (`render_fact` reads the
  complex synapses, not host labels).
- **HOST (declared, rides existing burn-downs):** (i) the per-concept NOVELTY levels are the ENVIRONMENT; (ii) the
  basin↔lexical-concept BINDING (which stored word each disjoint CA3 basin denotes) and each concept's stored FACT are
  the learned store / environment — the same boundary the multibasin wander declares for its assemblies; (iii) the
  curiosity want→recurrent-gain PROJECTION (the one-brain-merge rung); (iv) the composer's **agent→block first-match**
  is the host `_scan` oracle (`integrated_loop=False`, the pre-existing declared default seam — only the DECODE is
  spiking); (v) the QUESTION-template wrapper and any natural-language **FLUENCY** — the Broca / Qwen articulation
  scaffold — is **NOT exercised or measured here**; the MEASURED content is the bare spiking SVO proposition.

## Honest scoping (what this does NOT show)

<!--derived-->

- **about-selected = 1.00 is mouth-fidelity × correct-routing, not the load-bearing claim.** With identity routing
  `render_fact(agents[i])` returns `agents[i]`'s fact, so the decoded subject matches whenever the mouth decodes
  correctly — the 1.00 is a correctness read of the closed loop, not evidence by itself. The substrate-load-bearing
  evidence is the SELECTION DISTRIBUTION + curiosity-steering + the three lesions (SCRAMBLE 0.00, NO-NOISE 0,
  STORE-LESION 0), which is where the substrate (not host) is shown to drive WHICH utterance and HOW OFTEN.
- **"utterances/session" = coherent surfacing EPISODES routed to the mouth** (contiguous winner-take-all active runs — a
  duty-cycle-like count), NOT 476 distinct sentences. The mouth produces one of **3 distinct** utterances (the
  multibasin 3-of-4 basins ignite); the load-bearing quantities are the DISTINCT-concept count (3) and the visit/
  utterance SHARE across concepts, not the raw episode count.
- **The FLUENCY is not built here.** The utterance is a bare SVO proposition (`dog chase ball`) + a host question
  template; turning it into well-formed conversational English is the articulation scaffold's job (the
  scaffold-as-conditioned-articulation-crutch boundary), not measured. The propositional CONTENT + its selection are
  the deliverable.
- **3 of 4 basins ignite** (inherited from the multibasin substrate), so the wander speaks about 3 co-equal concepts,
  not all 4 (the 4th basin is sub-threshold at this operating point — the multibasin's named residual #2).

## Named next rungs (no defer — the capability continues)

1. **Fluent articulation of the self-initiated proposition** — feed the selected SVO + its curiosity/affect read into
   the Broca/Qwen articulation scaffold so the spoken remark/question is well-formed conversational English (a
   conditioned-articulation crutch, faculties load-bearing on the substrate selection), then biologize the mouth's
   remaining host seam (the agent→block first-match `integrated_loop`).
2. **One-brain merge of the curiosity gain** — release the `curiosity` neuromodulator directly onto the CA3 store on
   ONE bridge (shared with the mouth), so the steering gain is set BY the spiking modulator, not a host scalar.
3. **Question vs remark selection ON the substrate** — let an epistemic-gap / confidence read decide whether the
   surfaced concept is spoken as a REMARK (known fact) or a QUESTION (uncertain patient → `query_patient` abstains →
   "what does X do?"), so the utterance TYPE is substrate-driven, not templated.
4. **Get all n_mem basins to ignite** (4/4, then 6–8) so the self-initiated topic set is the full store (the
   multibasin residual: stronger synchronous encode or a per-basin adaptive coincidence threshold / larger n_ca3).

**Status: runner-level de-risk GO (NOT wired to production / NOT integrated).** Functional self-initiated-utterance
CORRELATE only; no claim of phenomenal experience. Runner: `research/runners/_self_initiated_utterance_derisk.py`. NO
`sim/` edit; reuse-by-import of the multibasin self-initiation SELECTION substrate + the production `OneBrainComposer`
mouth + the production curiosity organ.
