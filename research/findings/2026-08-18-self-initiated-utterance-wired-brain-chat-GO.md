---
type: finding
status: live
date: 2026-08-18
mechanism: self-initiated-utterance
integration_faculty: self-initiated-utterance
lane: self-initiation / conversation / production-integration
runner: research/runners/self_initiated_production_organ.py
verify: research/runners/_self_initiated_production_verify.py
artifacts:
  - research/findings/raw/_self_initiated_production/verify.json
---

# Self-initiated utterance WIRED into production /api/brain-chat (GO): the first INTERNALLY-GENERATED turn class — on an idle turn the brain SELECTS a stored concept itself and SPEAKS it, moat-safe, byte-identical on every reactive turn, lesion-load-bearing

**2026-08-18 (autonomous, numpy-CPU, through the REAL production ChatBrain + `/api/brain-chat` handler).** The
loop-closing self-initiated-utterance de-risk ([`2026-08-13-self-initiated-utterance-GO.md`](2026-08-13-self-initiated-utterance-GO.md),
6-seed cupy GO) is now WIRED into the production turn. On an **idle/empty turn** (an empty message, or a bare "say
something / what's on your mind" lead-in — a DISJOINT class) the brain **SELECTS a stored concept ITSELF** and
**SPEAKS** it through the production `OneBrainComposer` mouth as a self-initiated remark/question — the first turn
class whose CONTENT is internally generated, not a reaction to user words. Reuse-by-import, **NO `sim/` edit**,
default-ON + byte-identical-when-off — mirrors the d5-episodic / d6-multiref wiring exactly. **Functional
self-initiated-utterance CORRELATE only; no claim of phenomenal experience.**

## What is wired (reuse-by-import; NO `sim/` edit)

`research/runners/self_initiated_production_organ.py` composes the two already-GO organs of the de-risk with zero new
spiking machinery:

1. **SELECTION (the substrate decides the TOPIC, 0 host content-draw)** — the multibasin self-initiation wander
   ([`2026-08-13-self-initiation-multibasin-GO.md`](2026-08-13-self-initiation-multibasin-GO.md), 6-seed GO): DISJOINT
   pattern-separated CA3 basins under weak non-specific Poisson (NO cue), a curiosity recurrent-gain biasing WHICH
   surfaces (66% attributable). Reuse-by-import of `_run_condition` / `_selection`.
2. **THE MOUTH (the selected concept → words)** — the production `OneBrainComposer`, reuse-by-import via the de-risk's
   `_build_mouth`. `render_fact(concept)` reconstructs "concept verb patient" by an ON-BRIDGE resonate-and-fire unbind
   + cleanup (the spiking decode) and ABSTAINS (None) on an unknown subject (the no-confab moat).

**The handler hook** (`webapp/server.py::brain_chat`): a DISJOINT idle/empty-turn class, placed BEFORE the empty-422,
short-circuits to `organ.speak()` → `self_initiated_text` (the curiosity-selected concept → the mouth) or the honest
neutral idle line. Default-ON with the `_SELF_INITIATE_DEFAULT_ON = True` anchor; `BRAIN_SELF_INITIATE=0` skips the
block byte-identically (an empty message still 422s; a "say something" turn falls to the normal path). PER-SESSION
organ (`_SESSION_SELFINIT`, cleared on reset), keyed like the ChatBrain cache.

## Honest scope — the design fork (do NOT overclaim)

<!--derived-->

This is the **buildable-now** integration: the idle-turn **SHORT-CIRCUIT**. The **TIMING** is still HTTP/user-triggered
("say something"); only the **CONTENT** is internally selected by the substrate. Call it **"internally-selected
content on an idle-turn trigger", NOT "fully autonomous proactive speech".** A truly proactive
background/idle-**tick** that speaks with **NO HTTP request at all** is a larger endpoint/infra build — the **named
deferred next rung**, needing an owner decision (a background scheduler + a push channel; out of scope here).

**LATENCY residual (the SAME one d5-episodic declares):** the heavy n_ca3=2000 CA3 wander is ~seconds on cupy but
minutes on numpy@2000, so on numpy it is **DEFERRED** — the light path speaks the mouth's **curiosity-top decodable
concept** (the CONTENT is the mouth's spiking RF decode; the stochastic multibasin WHICH is deferred to cupy;
`BRAIN_SELF_INITIATE_STORE=1` forces the full wander). The full curiosity-biased CA3 selection + its NO-ENCODE
store-lesion collapse are the committed cupy 6-seed GO (`_self_initiated_utterance_derisk.json`), reused unchanged.

## GO gate — through the REAL production ChatBrain + `/api/brain-chat` handler (numpy-CPU)

_Values from the committed artifact `research/findings/raw/_self_initiated_production/verify.json`; the verify runner
is `research/runners/_self_initiated_production_verify.py`._

<!--derived-->

- **(A) IDLE turn → a COHERENT self-initiated utterance about a real stored concept.** Through the REAL handler an
  empty message / "say something" returns `Something's been on my mind — cat eat worm. What does cat eat?`
  (verified=True, abstained=False); the surfaced concept is a real stored fact (mouth fidelity: `render_fact` decoded
  it), about-rate 1.0. Per-seed 42/43/44/100/101/102 the organ surfaces a coherent curiosity-top concept (n_utt=4).
- **(B) BYTE-IDENTICAL on a full reactive panel** (recall / abstain / learn / anaphora), measured in **SEPARATE
  PROCESSES** and hashed: **flag-ON (default) == a PRISTINE-HEAD stash (the block removed) == `BRAIN_SELF_INITIATE=0`**
  (identical SHA-256), and **NO `self_initiated` key** on any reactive turn — the idle block is a pure no-op on a
  reactive turn.
- **(C) LESION-LOAD-BEARING.** `BRAIN_SELF_INITIATE_LESION=1` is the **store NO-ENCODE control** (an emptied RF store,
  not a host flag) → `render_fact` abstains for every concept → the utterance stream collapses (**n_utt 4 → 0** every
  seed) → the honest neutral idle fallback ("Nothing in particular is surfacing..."). Verified per-seed AND through the
  handler; `attributable_to`(intact vs NO-ENCODE lesion) = **1.0** — 100% of the utterance stream owed to the store
  (the lesion holds the mouth geometry fixed and empties only the RF store).
- **(D) MOAT-SAFE.** The remark is grounded in a real stored concept; an UNKNOWN subject abstains (`render_fact` None)
  every seed; the idle block NEVER flips a reactive abstain (the abstain panel in (B) is byte-identical).

The runner's own verdict decided **PASS** (all gates). The de-risk's OWN 6/6 per-seed gate (the full CA3-selection
wander + its NO-ENCODE store-lesion collapse) is the committed cupy GO, unchanged.

## What is SUBSTRATE vs HOST (the honesty boundary is a deliverable)

- **SPIKING (load-bearing):** (i) the VERBALISATION — the SVO proposition decoded by the `OneBrainComposer`'s on-bridge
  RF resonate unbind + cleanup (`render_fact` reads the complex synapses; tested live on numpy; collapses under the
  store NO-ENCODE lesion); (ii) on cupy, the SELECTION of WHICH concept + HOW OFTEN — the CA3 dendritic-plateau
  attractor competition under non-specific noise (0 host content-draw / no `random.choice` over concepts); (iii) the
  steering VALUE (the curiosity ASK-pool want read off `cp_firing_states`).
- **HOST (declared, rides existing burn-downs):** (i) on numpy the stochastic multibasin WHICH is DEFERRED (the light
  path ranks the mouth-decodable concepts by curiosity want); (ii) the basin↔lexical-concept BINDING + each concept's
  stored FACT are the learned store / ENVIRONMENT; (iii) the curiosity want→recurrent-gain PROJECTION (the
  one-brain-merge rung); (iv) the remark/question TEMPLATE + any natural-language FLUENCY (the Broca/Qwen articulation
  scaffold); (v) the TIMING is HTTP-triggered (the proactive idle-tick is deferred); (vi) CO-RESIDENT on its own
  selection + mouth bridges (rides burn-down #1).

## Named next rungs (no defer — the capability continues)

1. **The truly proactive idle-tick** (the deferred prize, owner decision): a background scheduler + push channel so the
   brain speaks with NO HTTP request at all — the TIMING becomes internal, not just the CONTENT.
2. **Run the full CA3 wander on the default numpy path** (or amortise it via a precompute → .npz cache of the encoded
   store) so the stochastic curiosity-biased SELECTION runs by default on numpy too, not only cupy.
3. **Fluent articulation** of the self-initiated proposition (feed the selected SVO + its curiosity/affect read into the
   Broca/Qwen mouth) + biologize the remaining host seams (the want→gain projection; the one-brain merge of the mouth
   and selection bridges).

**Status: WIRED into production `/api/brain-chat`, ON BY DEFAULT, verified GO through the REAL handler (numpy-CPU);
scaffold NOT retired** (the CA3 wander deferred on numpy + the declared host seams above). Functional
self-initiated-utterance CORRELATE only; no claim of phenomenal experience. Organ:
`research/runners/self_initiated_production_organ.py`. NO `sim/` edit; reuse-by-import of the loop-closing 6-seed GO.
