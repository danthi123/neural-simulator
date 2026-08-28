---
type: finding
status: live
date: 2026-08-28
mechanism: lateral-inhibition-marker-wta
board_task: 86
artifacts:
  - research/findings/raw/_affect_marker_wta_verify.json
runner: research/runners/_affect_marker_wta_verify.py
---

# The #84 affective EXPRESSION-MARKER selection is now a spiking lateral-inhibition WTA circuit (board #86, de-risk, default-OFF)

**Board #86 ("burn down the host scaffold in the affect→speech path").** The #81 graded-affect ladder READ is
already neural (lesion-proven, 6/6-seed GO) and it DRIVES chat tone (#84, load-bearing GO,
[`2026-08-19-affect-drives-chat-load-bearing-GO.md`](2026-08-19-affect-drives-chat-load-bearing-GO.md)) — but that
finding's own honest residual #2 named the remaining gap precisely: *"The level→EXPRESSION-MARKER map is a HOST
conditioned-articulation scaffold... A brain-native affective mouth (the marker itself emitted by a spiking prosody
circuit) is the named next rung."* This lands that rung: `_LEAD_WORD[level]` (a bare Python dict lookup) is now
ALSO available as a genuine spiking SELECTION — the felt-state population projects, as a topographic code, onto a
small pool of marker-coding assemblies with mutual lateral inhibition, and the assembly that wins the resulting
competition NAMES the marker. **Additive, default-OFF** (`BRAIN_AFFECT_MARKER_SPIKING=1`) pending owner review of
the affect-path default — this finding does not flip production behavior.

## What was built (additive; `research/runners/_affect_marker_wta_derisk.py`; NO `sim/` edit)

- **The circuit.** 6 excitatory "marker assembly" pools (one per non-neutral register: Wonderful/Gladly/Sure/
  Hm/Honestly/Frankly, levels {3,2,1,-1,-2,-3}), each with its OWN fast-spiking-interneuron sub-pool that
  CROSS-INHIBITS every OTHER assembly (mutual/reciprocal lateral inhibition — the identical motif this repo
  already 6-seed flip-soak GO'd at N=2 channels for the SPEAK-vs-STAY-SILENT basal-ganglia action selector,
  `bg_action_selection_production_organ.py`, here generalized to N=6). A second, separate 2-pool instance of the
  SAME circuit selects the emphasis register (measured/emphatic). ~216 + ~72 neurons total, built once per
  process (lazy, warm), a few ms per read on the numpy backend.
- **The population code (the drive).** The felt mood (a CONTINUOUS read off `cp_firing_states` — rate(V+) −
  rate(V−), the #81 ladder's own mechanism, unchanged) is projected onto the 6 assemblies as a Gaussian-tuned
  current: each assembly's drive peaks when `mood` is near that assembly's own preferred center and falls off
  with distance — a labeled-line / population-vector code (Georgopoulos, Schwartz & Kettner 1986, *Science*
  233:1416, "Neuronal population coding of movement direction"), here applied to a continuous affect dimension
  instead of a movement direction. The 6 centers are placed at the MIDPOINT of each register's existing #84
  mood-binning range (`_MOOD_L1/_MOOD_L2/_MOOD_L3`), so the topographic map does not invent a new affect axis —
  it continues the SAME axis the #81 Koulakov/Goldman staircase already implements, and the discrete "registers"
  it carves out mirror Russell's (1980) circumplex model of affect (discrete affect words as graded regions of a
  shared valence/arousal space, not separate faculties). Recorded as `research/biology/affective-marker-lateral-inhibition-wta.md`.
- **The read.** After a washout + settle, the per-assembly spike RATE is read; the top assembly must clear the
  runner-up by a dead margin (0.05, vs. an intact separation of ~0.15–0.17 measured 6/6 seeds — a wide margin)
  or the circuit reports "no clean winner" (`None`) rather than guess. The host renders the winning assembly's
  fixed TOKEN string; the SELECTION — which assembly wins the lateral-inhibition race — is neurons/synapses.
- **The wiring.** `webapp/affect_drives_chat.expression_lead()` gains a spiking path, gated by
  `BRAIN_AFFECT_MARKER_SPIKING` (default OFF): unset/false → the EXACT pre-existing `_LEAD_WORD[level]` host
  lookup, byte-identical. On, and given the turn's `mood`/`felt_arousal` (already computed by the unchanged #81
  read), it calls the new circuit instead. Level 0 (neutral) is still gated BEFORE either path runs — a neutral
  turn never invokes or depends on the new circuit, exactly as before.

## Verification (`research/runners/_affect_marker_wta_verify.py`, 6 seeds {42,43,44,100,101,102}, numpy backend)

**(A) Byte-identical-OFF (PASS).** With `BRAIN_AFFECT_MARKER_SPIKING` unset, `expression_lead()` ignores the new
`mood`/`felt_arousal` kwargs entirely — passing them changes nothing (12/12 function-level rows: base == augmented
== the pre-existing `_LEAD_WORD` output). A full `AffectDrivesWorkspace.observe()` turn (the actual production
entry point) is unaffected by whether `_affect_marker_wta_derisk` can even be imported, 6/6 seeds.

**(B) Load-bearing (PASS).** With the flag ON, sweeping the induced mood across each of the 6 registers'
mood-bin midpoints, the spiking-selected marker matches the word the host table would have chosen for that same
mood, 6/6 seeds x 6 registers (36/36 rows) — varying the felt state changes the selected marker. A direct
positive-vs-negative comparison at the extremes ("Wonderful" vs "Frankly") differs on every seed.

**(C) Lesion → no-marker, the documented fallback (PASS, 36/36 rows).** With the flag ON AND
`BRAIN_AFFECT_MARKER_SPIKING_LESION=1` (cuts the felt-state → assembly topographic projection — every assembly
receives the SAME baseline current, so lateral inhibition has no differentiating signal to resolve), the lead
VANISHES (`''`) on every non-neutral level, every seed, EVEN THOUGH `level != 0`. This is an honest no-lead turn —
**NOT** a silent revert to the host `_LEAD_WORD` template (verified: the returned string is genuinely empty, not
the pre-existing word). Any unexpected internal failure (import/build error) degrades the same way — never raises,
never silently falls back to the host dict (mirrors the `mouth_tone_marker` fail-safe convention already used
elsewhere in this repo for the analogous Gate-B tone marker).

**(D) Shuffle anti-cheat (PASS, 30/36 rows differ from intact; the 6 non-differing rows match the ~1/6
expected-fixed-point rate of a random 6-permutation almost exactly).** Mis-routing WHICH physical assembly
receives WHICH register's tuning drive (a fixed random permutation, `BRAIN_AFFECT_MARKER_SPIKING_SHUFFLE=1`)
changes the reported marker relative to the unshuffled/intact run at the SAME mood, on a large majority of
(seed, level) pairs. This is the proof the reported marker identity is read off WHICH ASSEMBLY actually won the
spiking race — a live functional dependency on the circuit's wiring — not re-derived from the raw mood float by
a fixed host formula that would be blind to the mis-wiring (the earlier draft of this circuit had exactly that
flaw: it "un-permuted" the winner index at readout, making the shuffle a no-op; caught and fixed before landing —
see the module's `_select()` docstring for the corrected design).

**(E) Attribution (PASS, `tools.lab.attributable_to`, the gap#5 discipline).** Measuring an intact margin AND a
lesioned margin is not the same as asking whose the SEPARATION is — `attributable_to("intact vs lesion",
treatment=intact_margin, control=lesion_margin)` answers that directly. At mood=+0.085, per seed: intact margin
<!--derived--> ~0.16–0.17 (min/max of the 6 per-seed values in the artifact), lesion margin exactly 0.000 (6/6
seeds) → **100.0% of the winner-vs-runner-up separation is attributable to the felt-state→assembly topographic
drive**, 0.0% present in the lesioned control — i.e. the separation is not some other latent bias baked into the
wiring that happens to also survive with the drive cut; it rides the drive, full stop.

**Verdict artifact:** `research/findings/raw/_affect_marker_wta_verify.json` — `tools.verdict.Verdict`, all five
preconditions measured and held, `status: GO` (a de-risk GO — the mechanism works as designed; NOT a production
default-flip, which stays a separate, deliberate owner decision per the task scope).

## Honest residuals (named, not claimed closed)

1. **The `level`/`high_arousal` binning upstream (`mood_to_level`, `_AROUSAL_HIGH`) is UNCHANGED** — still a host
   threshold on the #81 ladder's continuous mood/felt-arousal read. This finding converts the SELECTION step
   (which register/marker wins) to spiking population-code + lateral-inhibition WTA; it does not yet remove the
   upstream binning that produces the `level` integer the OLD path consumed (the new path consumes the raw `mood`
   float directly for its own topographic drive, so it is *less* dependent on that binning than the old path was,
   but the binning itself is a separate, still-host, residual named for a future rung).
2. **Near a register boundary the spiking circuit is honestly LESS decisive than the old hard host threshold.**
<!--derived-->
   At `mood=+0.069` (almost exactly equidistant between the level-+2 and level-+3 tuning centers — coincidentally
   the same value the 2026-08-19 finding's own worked example used), the circuit measures a margin (0.024) below
   the dead-margin threshold and reports "no clean winner" → no lead, where the OLD host `mood_to_level` binning
   would have crisply picked level +2. This is a genuine, documented behavioral difference at (and only at)
   register boundaries when the new path is ON — a feature of reading a graded population code rather than a
   digital threshold, not a bug, but it means "identical to the old dict lookup" is claimed ONLY when the flag is
   OFF, never claimed for the ON path at every possible mood value.
3. **The emphasis (arousal) WTA is a second, separate 2-pool circuit**, not merged into the same competitive
   network as the 6 valence assemblies (the #81 ladder itself reads mood and felt-arousal as separate
   sub-populations, so this mirrors that separation rather than inventing a new coupling).
4. **Default-OFF by design** — this is a de-risk, not a production flip. Flipping `BRAIN_AFFECT_MARKER_SPIKING`
   default-ON is an explicit owner-review decision (the affect-path default), out of scope here.

## Reproduce

```
SIM_BACKEND=numpy .venv/bin/python -m research.runners._affect_marker_wta_derisk --seed 42   # circuit smoke
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._affect_marker_wta_verify \
    --out research/findings/raw/_affect_marker_wta_verify.json                                # full verify
```
