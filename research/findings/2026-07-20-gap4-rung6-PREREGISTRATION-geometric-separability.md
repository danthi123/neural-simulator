# gap#4 RUNG 6 — PRE-REGISTRATION: is the blocker GEOMETRIC? (filed BEFORE the run)

**Filed 2026-07-20 before any rung-6 result exists.** Seeds **1000-1005**, never used.

## The hypothesis, and why it is the only survivor

Six mechanism families are closed, and their unifying cause is measured: **adjacent-lag and field-forming synapses
are not separable by any quantity locally available at the synapse at update time** — eligibility magnitude 1.001x,
its rank (monotone in the same), overlap with the instructive signal (`IS` uniform across a cell), current weight
1.093x, and pointwise read-out transforms (which rescale without informing).

The precise cause: **at the tested geometry, field spacing (4 bins) EQUALS the measured backward shift (4-6 bins)**,
so both populations occupy the same lag — and lag is the only thing local eligibility encodes. That is a property of
the TASK GEOMETRY, not of the rule.

**Prediction: give the geometry room and the same local rule separates them.**

## Design — spacing is the SINGLE variable

On a 20-bin track no configuration gives both `spacing > shift` and >= 4 cells, so the track is lengthened to
**40 bins for BOTH arms**, holding track length constant:

| arm | spacing | cells | spacing vs shift (4-6) |
|---|---|---|---|
| **A (collision)** | 4 | 10 | **equal — the failing condition** |
| **B (separable)** | 8 | 5 | **greater — the hypothesis** |

Both arms run the SAME rule with the SAME band parameters. Nothing about the mechanism changes between them.

## PRE-REGISTERED PREDICTIONS

1. **P1 — the collision reproduces on the long track:** arm A shows adjacent contrast <= 1.35x (i.e. the deficit is
   a property of the geometry, not of the 20-bin track), on >= 5/6 seeds.
2. **P2 — separation appears when geometry permits:** arm B shows adjacent contrast **>= 1.60x**, on >= 5/6 seeds.
3. **P3 — far contrast is retained in both:** >= 2.0x, on >= 5/6 in each arm.
4. **P4 — stage 1 forms in both arms:** `map_ok = 1` on >= 5/6 in each (this is what killed both band attempts).

**FALSIFIED if** P2 fails — the geometric hypothesis is then wrong and the blocker is NOT the spacing/shift
collision, which would leave only a non-local instructive signal (Milstein's feedback-inhibition route).
Also falsified as a clean test if P1 fails, since arm A would then not reproduce the phenomenon being explained.

## ⚠️ Honest scope, stated in advance

- **This is a WEAKER claim than "the contrast problem is solved."** Confirming it shows the local rule CAN separate
  the two populations *when the geometry permits*, NOT that it solves adjacent-contrast at the geometry where the
  deficit was originally measured. I will not report a P2 pass as having fixed gap#4's blocker.
- The honest follow-up question, which this run does NOT answer: **is real hippocampal place-field spacing greater
  than the BTSP backward shift?** If yes, the 20-bin geometry was unrepresentative and this result matters. If no,
  the collision is biologically real and the geometric route is a lab artifact. That is a literature question and it
  is explicitly deferred, not assumed.
- Both arms are run in the SAME invocation so the comparison cannot drift.

## Cap

**One geometry.** If P2 fails I do not try a third spacing — the verdict becomes that the blocker is not geometric,
and the remaining route is the non-local instructive signal.
