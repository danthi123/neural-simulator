# Difference-readout cheap-first probe = honest fast NEGATIVE; the predictive-coding readout-computation hypothesis is falsified; but the probe relocates the compositional blocker: it is NOT the readout computation, it is that engram-tag stimulation carries no readable bound-attribute signal at the language-output population at all

## Status

Cheap-first falsification probe for the predictive-coding difference
readout (design: `docs/plans/2026-05-21-predictive-coding-difference-readout-compositional-design.md`,
commit `8ea41d7`). Controller-only; single seed 42; reused the cached
200-event unified substrate (no retraining); ~1 min wall-clock. The
probe applies the design's pre-registered decision rule and two
controls.

## Result (pre-registered; no bar change; no threshold tuning)

```
seed 42; 4 groundable (noun, adjective) bindings + 4 permuted-tag controls

| query                  | raw_top (arcs 1-8 sum) | diff_top (hip - cons) |
|------------------------|------------------------|------------------------|
| apple -> big (target)  | look  XX               | look  XX               |
| river -> small         | go    XX               | look  XX               |
| dog -> hot             | big   XX               | big   XX               |
| cat -> cold            | big   XX               | big   XX               |

raw readout (arcs 1-8 sum):    0/4 correct
difference readout (hip-cons): 0/4 correct
permuted-tag control:          0/4 point to the stimulated tag
```

**Pre-registered decision rule -> NEGATIVE.** The difference readout
did not beat the raw readout (both 0/4) AND the permuted-tag control
failed (0/4). Per the design's fixed rule, this is an honest fast
negative.

## The probe is not broken: consistency check

The 0/4 raw compositional correctness at seed 42 is CONSISTENT with
the prior localization diagnostic (2026-05-20), which measured seed 42
N=5 and found 4/5 wrong (1/5 correct). Both put raw top-1
compositional correctness at seed 42 in the near-zero regime. The
"~0.46" figure cited for the sixth arc is the abstention-inclusive
`full_acc` verdict metric (which credits correct abstentions and
routes through the calibrated gates), NOT raw top-1 compositional
correctness. The probe reproduces the known near-zero raw
compositional correctness; it is measuring the right thing.

One cosmetic imperfection, noted for honesty: in the permuted-tag
control, the cue-noun and tag-noun differ, so the per-word exclusion
set differs slightly between the two patterns being subtracted. This
does not affect the verdict -- the groundable result (0/4, difference
not beating raw) alone triggers NEGATIVE, and the permuted control's
incoherence (top words are nouns and verbs, never the stimulated
tag's adjective) is robust to the exclusion detail.

## The sharper diagnosis: the blocker is NOT the readout

The eight-architecture series localized the failure as "the cue's
broad drive dominates the engram tag's selective drive at a shared
readout" and hypothesized that a more selective readout computation
would recover the bound attribute. This probe falsifies that
hypothesis -- and, more importantly, the **permuted-tag control
relocates the blocker.**

The permuted-tag control stimulates a *different* binding's engram tag
while cueing a noun. If the engram tag carried ANY readable
bound-attribute signal at the language-output population, the
difference readout -- which removes the cue's contribution by
subtraction -- would surface the *stimulated tag's* adjective. It does
not, on any of the 4 controls. The difference readout, with the cue's
contribution fully subtracted out, finds no coherent adjective signal
to recover.

**This means the failure is upstream of the readout.** It is not that
the cue drowns out a present-but-weaker tag signal. It is that
engram-tag stimulation does not produce a readable bound-attribute
signal at the language-output population in the first place. No
readout computation -- sum, difference, normalization, or otherwise --
can recover a signal that is not there.

## What this rules in and out

- **Ruled out**: the readout-computation class of fix (this probe) and
  the eight dynamics-overlay architectures (gating, rhythm,
  multiplexing, replay, priming, monitoring). Composition does not
  fail because the readout is computed wrongly, nor because the
  substrate dynamics around a fixed readout are configured wrongly.

- **Ruled in**: the blocker is in the **storage-and-reactivation**
  stage. The engram tag -- a Tonegawa-style top-K co-firing ensemble
  captured across dg/ca3/ca1 -- when stimulated, does not reconstruct
  the bound adjective's distributed language-output code well enough
  to be read by any downstream computation.

This is a genuine biology-translatable result, and it is sharper than
the prior convergent-ceiling statement. The compositional binding is
either (a) not captured into the engram tag at encoding, or (b)
captured but not routed to the language-output population on
stimulation. The eight-arc-plus-readout-probe convergent ceiling now
has a precise mechanistic locus: storage/reactivation, not retrieval.

## Terminal closure of the readout design line

Per the design doc's pre-registered terminal-closure branch: a
cheap-probe negative ends the readout design line. It is honestly
closed. The readout-computation hypothesis is falsified; no full arc
follows it.

This is the falsify-cheaply-first discipline working exactly as
intended: a ~1-minute single-seed probe replaced what would have been
a multi-day full arc (frozen verdict module + adversarial review +
multi-seed decisive run), and it did so while producing a sharper
diagnosis than the full arc would have.

## Honest ceiling (unchanged)

Compositional / conversational capability is NOT achieved and is NOT
claimed. The difference-readout hypothesis is falsified. The
biology-translatable deliverable is the relocation of the blocker:
across eight dynamics-overlay architectures plus one readout-
computation probe, the limiting factor is now localized to engram
storage-and-reactivation. The protected module set is byte-unchanged;
the no-confabulation moat is 7/7 byte-identical; no bar was tuned.

## Files / evidence

- Probe script: `research/findings/raw/difference_readout_probe.py`
- Probe result JSON: `research/findings/raw/difference_readout_probe.json`
- Probe log: `research/findings/raw/difference_readout_probe.log`
- Design doc: `docs/plans/2026-05-21-predictive-coding-difference-readout-compositional-design.md`

## Pre-registered next step: storage-locus probe

The blocker is now localized to storage/reactivation. The immediate
cheap-first probe to localize it further -- does engram-tag
stimulation reactivate the bound adjective's POOL (the
`adjective_pool_*` regions), as opposed to its language-output code?

Probe: stimulate each engram tag; measure the firing rate of all
concept POOLS directly (via the validated `measure_pool_firing`-style
direct pool measurement), not the language-output cosine. For each
(noun, adjective) binding:
- If the bound adjective's pool fires strongest on tag stimulation:
  storage IS capturing the binding; the gap is the pool ->
  language-output readout pathway. Next work targets that pathway.
- If the bound adjective's pool does NOT fire strongest: the engram
  tag did not capture the adjective at encoding. Next work targets the
  encoding mechanism (the noun is cued first / more strongly; the tag
  may be noun-dominated).

Pre-registered decision rule (fixed): the probe is purely diagnostic
(no PASS/FAIL bar) -- it routes the next direction. It reuses
`measure_pool_firing` and `stimulate_tag` byte-unchanged; single seed
42; cached substrate; ~minutes. The outcome -- pathway gap vs encoding
gap -- determines the next biology-grounded mechanism to design.
