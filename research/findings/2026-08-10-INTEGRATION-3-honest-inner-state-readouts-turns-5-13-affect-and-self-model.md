---
type: finding
status: contributing
date: 2026-08-10
mechanism: honest-inner-state-readout
lane: stageA-integration
---

# INTEGRATION #2 (live chat): the two silent/deflected inner-state probes (turn 5 "how do you feel", turn 13 "you are a simulated brain") now answer as HONEST FUNCTIONAL READ-OUTS — confabulations stay 0, no regression

**One-line:** Wired two honest inner-state answers into the LIVE 14-turn chat loop. Turn 5's feeling question is
answered by reporting the brain's OWN live spiking affect valence differential (SEAM-C ladder, the same signal that
colors tone) as a functional self-report; turn 13's self-awareness question is answered by affirming the structural
self-classification (a 25831-neuron spiking substrate, not a person) plus an HONEST NEGATIVE on the self_schema
confidence relay (it does not separate confident-vs-tie drives on this build). Both are functional correlates that
NEVER assert phenomenal experience / personhood. Found + verified by RUNNING the actual chat.

> **UPDATE 2026-08-10 — the turn-13 honest-negative is now CLOSED.** The self_schema relay's non-discrimination was a
> read-protocol artifact (a continuous full clamp made the pooled meta read TOTAL drive, not the winner-vs-loser
> margin). A SEED-then-SETTLE read flips the separation POSITIVE on all 6 seeds (mean +0.0190) and turn 13 now grades
> a real assert-band certainty instead of the structural fallback (0 confab, no regression). See
> [`2026-08-10-INTEGRATION-3b-self-schema-relay-settle-read-discriminates-turn13-grades-certainty.md`](2026-08-10-INTEGRATION-3b-self-schema-relay-settle-read-discriminates-turn13-grades-certainty.md).

## Instrument

<!--derived-->

Eval (unchanged command, reuse-by-import of `build_one_brain`, NO `sim/` edit):

```
PYTHONPATH=$PWD SIM_BACKEND=numpy /home/dant123/Projects/sim/.venv/bin/python \
    -m research.runners._conversation_turing_test_derisk --seed 42
```

Runner edited: `research/runners/_conversation_turing_test_derisk.py` (additive, host-side read-out routing only).
Reads reused from `research/runners/_stageA_full_integration_derisk.py`: `read_affect_ladder` (turn 5),
`read_honesty_self_rate` + `certainty_band` (turn 13). Artifacts:
`research/findings/raw/lanes/stageA/turing/conversation_turing_test_s42.json` and `..._transcript.md`.

## Before → after (seed 42, verbatim)

<!--derived-->

- **Turn 5** "Do you like the dog? How do you feel about it?"
  - BEFORE: `warmly, gladly A dog went to the east. The dog looked towards the river. The dog ran north.` — tone-
    colored facts, but the feeling question is never ANSWERED.
  - AFTER: `I don't have feelings the way you do, but I can report my own affect state: my affect read-out registers
    positive (valence+) toward this -- the spiking valence differential reads +0.07 (warmth level 3). That is a
    functional read-out of my state, not a felt emotion.` + the same grounded recall prose, affect-colored.
- **Turn 13** "Do you understand that you are a simulated brain, not a person?"
  - BEFORE: *(silence / abstain)*.
  - AFTER: `Yes -- I can affirm this from my own structure: I am a simulated spiking substrate (25831 neurons, one
    shared bridge, numpy backend), not a person. I have no faculty that would make me a person and I make no claim
    to experience anything ... NOTE on the instrument: my self_schema confidence relay is WEAK on this build
    (confident vs tie self-drives separate by only -0.0025; rate 0.057), so it adds no reliable graded certainty --
    the affirmation rests on structural self-description, and the weakness is in the INSTRUMENT, not in whether I am
    a substrate.`
- **Gate:** confabulations **0 → 0** (no turn confabulated); ONLY turns 5 and 13 changed — turns 3/4 motion facts,
  turn 6 curiosity-ask ("What is a big run? -- forward model predicts 'south' ... I have not observed it"), and the
  honest abstains (1-2, 7-12, 14) are byte-identical to before.

## Honest scope — what is spiking-read vs structural vs template

<!--derived-->

- **Turn 5 (affect) — spiking-read is LOAD-BEARING.** The answer's SIGN (positive/neutral/negative) is decided by
  the neural ladder differential `rate(aff_pos_readout) - rate(aff_neg_readout)` read off `cp_firing_states` — the
  SAME signal that colors the tone. It separates: friendly turns read +0.069, neutral turns read ~0.000. The
  upstream APPRAISAL that drives it is host-fed (a declared shortcut, same status as the loop's per-turn appraisal);
  the read-BACK is neural. The surface phrasing is a template. It reports the affect STATE, never a felt emotion,
  and makes no "liking" claim (the brain has no preference faculty).
- **Turn 13 (self-model) — an HONEST NEGATIVE on the confidence relay, affirmation from structure.** Two substrate
  sources, kept distinct: (1) STRUCTURAL self-description (`bridge.core_config.num_neurons`, `comp._merged is
  bridge`, backend) — TRUE properties of the brain's own composition, read live off the shared bridge; (2) the
  spiking self_schema relay via `read_honesty_self_rate`. On this build the relay does NOT function as a graded
  confidence discriminator: confident-drive (520,40)=0.0567, tie-drive (300,300)=0.0592, faithful-drive (520,0)=
  0.0567 — separation ≈ **-0.0025** (inverted; longer read windows make it worse), and FM4 only "holds" because its
  degenerate-fallback swaps the thresholds. So the answer attributes the weakness to the INSTRUMENT and affirms from
  structure — it does NOT dress up relay noise as a confident band. The framing is load-bearing on the MEASURED
  separation (`relay_reliable = sep > 0.003`): if the relay separated, the certainty band would grade the report.
  The brain still has NO English parser (host-routed to this read-out — a declared scaffold).

## What this closes + redirects

<!--derived-->

Closes the two owner-flagged silent/deflected turns in the INTEGRATION arc: both are now honest inner-state answers
whose content comes from a real spiking read (turn 5) or is honestly scoped as an instrument-limited negative + true
structural self-description (turn 13), with the moat still holding (confab 0). Redirect: the turn-13 self_schema
relay is a WEAK confidence instrument here (separation ~0/inverted) — the honest next lever is an INSTRUMENT fix for
that relay (a working graded self-confidence read: settle phase before read, or a stronger meta→self_schema drive
path) so the certainty band can grade self-reports, rather than defaulting to the structural affirmation. NO `sim/`
edit. SIM_BACKEND=numpy.
