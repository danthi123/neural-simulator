---
type: finding
status: live
date: 2026-08-19
mechanism: graded-affect-bistable-ladder
board_task: 84
artifacts:
  - research/findings/raw/_affect_drives_chat/verify.json
  - research/findings/raw/_affect_drives_chat/full_default_config_AB_transcript.txt
runner: research/findings/raw/_affect_drives_chat/verify_affect_drives.py
---

# Affect DRIVES the live chat — the #81 graded-affect ladder read is LOAD-BEARING on what/how the brain responds (GO)

**Board #84 ("make graded affect load-bearing on the live conversation, not observe-only").** Verdict:
**INTEGRATION GO, verified through the REAL `/api/brain-chat` handler in-process** (numpy-CPU, NO `sim/` edit). The
#81 graded-affect bistable-LADDER (6/6-seed GO, `2026-08-19-graded-affect-attractor-GO.md`) shipped as a default-off
de-risk RUNNER, never in the live chat. This wires its NEURAL valence x arousal read onto the live conversational
brain AND makes it CHANGE the response: the felt affect state colors the affective EXPRESSION the reply leads with (a
graded warmth/curtness marker). This is the anti-hollow-integration counterpart to the observe-only faculties — the
affect READ is neural AND it demonstrably shapes the surface, and the surface change VANISHES under a neural lesion.

## What was wired (additive glue; `webapp/affect_drives_chat.py`; reuse-by-import; NO `sim/` edit)

- **The READ = the #81 mechanism.** Each turn the message's affective valence/arousal (the Gate-B
  `affect_production_organ.appraise_text` DR-2 learned valence — a language-comprehension boundary, like the SVO
  parser) is EMA-folded into a persistent per-session BODY-STATE (h = comfort/homeostasis, a = bodily arousal). A
  neutral turn HOLDS the prior body-state (cross-turn affect persistence). That body-state drives the #81 ladder via
  `read_body` (reused-by-import from `_graded_affect_attractor_derisk`), and the felt state is the ladder's OWN
  population read off `cp_firing_states`: mood = rate(V+ ladder) - rate(V- ladder); felt_arousal = rate(arousal
  ladder). The graded mood is binned into a Koulakov staircase LEVEL (-3..+3).
- **The COUPLING = a graded affective EXPRESSION lead.** VALENCE picks a graded warmth/curtness discourse marker
  prepended OUTERMOST to the answer surface ("Wonderful — <fact>", "Sure — <fact>", neutral -> NO lead,
  "Honestly — <fact>", "Frankly — <fact>"); AROUSAL picks the marker's emphasis (high felt-arousal -> "! ", else
  " — "). The marker is an honest EXPRESSION of the read state (prosody/tone-of-voice the body renders), NOT content.
- **Isolation + reversibility.** The ladder build (~0.4s, lazy/warm) and every read run on the workspace's PRIVATE
  RNG timeline with the host process-global RNG restored around them (the #77 global-RNG footgun), so enabling the
  module leaves the other response fields byte-identical. Default-ON anchor `_AFFECT_DRIVES_DEFAULT_ON=True`;
  `BRAIN_AFFECT_DRIVES=0` -> the block is fully skipped (no `affect_drives` key, no lead -> byte-identical oracle).

## (A) Affect TRACKS the conversation (through the real handler, PASS)

An emotionally-varied conversation moves the neural mood sensibly, and a neutral fact-probe HOLDS the induced mood
(persistence). mood is the ladder differential off `cp_firing_states`; lead is the graded marker. The four
distinct-affect turns (each an appraised message; mood/felt rounded from `verify.json` part_a):
<!--derived-->

| turn | message | mood | felt-arousal | level | lead |
| --- | --- | --- | --- | --- | --- |
| 0 | "what does the dog chase?" (baseline) | -0.003 | 0.000 | 0 | (none) |
| 1 | "I am so happy and joyful, this is wonderful" | +0.069 | 0.043 | +2 | "Gladly — " |
| 3 | "I feel so sad and afraid, everything is terrible" | -0.029 | 0.070 | -1 | "Hm! " |
| 5 | "this is a delight, I am glad and cheerful" | +0.064 | 0.074 | +2 | "Gladly! " |

The interleaved neutral fact-probes (turns 2, 4, 6 = "what does the dog chase?") each HOLD the immediately-preceding
induced mood UNCHANGED (persistence): turn 2 holds +0.069 / "Gladly — ", turn 4 holds -0.029 / "Hm! ", turn 6 holds
+0.064 / "Gladly! " — a neutral message adds no new appraisal, so the ladder keeps its latched state.
Baseline ~0, positive turns > +tol, negative < -tol, and felt-arousal RISES with sustained affective content
(0.000 -> 0.074). Both valence AND arousal move. Full-precision per-turn values are in the artifact
`research/findings/raw/_affect_drives_chat/verify.json`.

**Persistence shows up on the SURFACE (a feature, observed).** Because the mood is a persistent per-session state, a
NEUTRAL turn after an emotional exchange keeps the colored tone: after a negative exchange the brain answers an
unrelated fact-probe as "Frankly! The brain uses spikes." and even an abstain as "Frankly! I don't know about that."
— the content verdict (fact / abstain) is unchanged; only the tone carries the lingering mood. This is the (A)
persistence property becoming visible in what the brain says.

## (B) Affect DRIVES the response — the LOAD-BEARING proof (message FIXED, PASS)

Hold the message literally identical ("what does the dog chase?"), vary the brain's affect state (a mood induction
that sets the body-state directly; the neural ladder still reads it), and the reply DIFFERS in tone while the content
is identical:

| affect state | answer surface | content (abstained/recalled/verified) |
| --- | --- | --- |
| induced positive | **"Wonderful — The dog chases cat. The cat eats fish."** | (False, [dog,chase,cat], True) |
| induced negative | **"Frankly! The dog chases cat. The cat eats fish."** | (False, [dog,chase,cat], True) |

The lead differs (warm vs curt), the base sentence after the lead is byte-identical, and the content fields md5 are
identical — affect changes HOW it sounds, not WHICH fact is true.

**The difference VANISHES under the NEURAL lesion (the anti-hollow check).** `BRAIN_AFFECT_DRIVES_LESION=1` cuts the
interoceptive->ladder synapses (the #81 embodiment lesion): the neural mood collapses to 0.000 for BOTH the positive
and negative induction, so the staircase level is 0, the lead disappears, and both answers revert to the bare fact
**"The dog chases cat. The cat eats fish."** — identical, and equal to the coupling-off base. So the surface change
RIDES the SPIKING ladder read, not a host `if valence>0`: kill the neural read and the tone-difference is gone.

## (C) NO-REGRESSION on content (PASS)

- **Content affect-invariant.** For a recall / abstain / self panel, the content fields (abstained, recalled_svo,
  verified, source, brain, rich) have an IDENTICAL md5 across {coupling-off, coupling-on, induced-positive,
  induced-negative} on every turn. Affect never manufactures a fact, never flips an abstain, never changes the moat.
- **Byte-identical-off.** With `BRAIN_AFFECT_DRIVES=0` the response never carries an `affect_drives` key; and at a
  neutral affect (a fresh fact-probe reads mood ~0 -> no lead) the ON response MINUS the additive `affect_drives`
  key is byte-identical (md5) to the OFF response on every panel turn. The wiring adds nothing to the surface until
  the affect is non-neutral — the coupling only decorates, it never regresses.

## Fidelity note (the verify's organ isolation, honest)

(A) and (B) were confirmed on the FULL default-organ production config (every faculty on) AND the rich multi-sentence
answer path (rich=default) — the exact trajectory + the message-fixed warm/curt difference + the lesion collapse (the
rich base "The dog chases cat. The cat eats fish.") — see `full_default_config_AB_transcript.txt`. The committed
`verify.json` runs the SAME `/api/brain-chat` handler + the SAME recall/moat core but for a tractable in-process
A+B+C battery it (a) disables the OTHER heavy default-on organs (Gate-B affect = a 25k-neuron brain stepped per turn,
worldmodel, surprise, ...) — ORTHOGONAL to affect-drives, which reads its own #81 ladder and prepends the lead
regardless — and (b) routes through the STATELESS single-fact path (rich=False; base "The dog chases cat."), where
the lead is wired identically. Neither changes an affect-drives verdict. `AFFECT_VERIFY_FULL_CONFIG=1` runs without
the organ isolation.

## Honest residuals (named, ride existing burn-down items)

- **The affect READ is neural; the message->valence APPRAISAL is host** (a language-comprehension boundary, the SVO-
  parser boundary — DR-2 learned distributional valence gated by the Warriner salience norms). The felt read
  (body-state -> graded valence x arousal off `cp_firing_states`) and its embodiment dependence ARE the #81 neural
  mechanism (lesion-proven). The body-state VARIABLES (h, a) are the standard body boundary.
- **The level->EXPRESSION-MARKER map is a HOST conditioned-articulation scaffold** (the "mouth"): the affect that
  DRIVES it is the neural ladder read (load-bearing — the lesion collapses the marker), but the surface STRING for a
  given level is a host template. This is the owner-sanctioned articulation-crutch pattern (scaffold-ok-as-
  conditioned-articulation IF the faculty is load-bearing on the tone, which the lesion proves). A brain-native
  affective mouth (the marker emitted by a spiking prosody circuit) is the named next rung.
- **This module reads its OWN co-resident #81 ladder bridge**, run ALONGSIDE the recall composer, not merged onto the
  single recall bridge (the one-brain consolidation step, shared with the Gate-B affect burn-down).
- **Orthogonal to the Gate-B `BRAIN_AFFECT` path**, which independently colors prose-manner + forthcomingness off a
  different (2026-08-08 Stage-A) ladder. This module is the #81 interoceptive graded ladder driving a distinct,
  CPU-measurable, lesionable expression channel. Unifying the two affect reads onto one ladder is a follow-on.
- **Honesty boundary.** A functional graded core-affect state with a bodily cause and an honest functional read-out
  shaping tone; no claim of phenomenal experience.

## Reproduce

```
SIM_BACKEND=numpy python -u research/findings/raw/_affect_drives_chat/verify_affect_drives.py
AFFECT_VERIFY_FULL_CONFIG=1 SIM_BACKEND=numpy python -u research/findings/raw/_affect_drives_chat/verify_affect_drives.py
```
