# Biological spiking composition is ROBUST at 64-concept scale, multi-seed -- the boundary is lifted -- 2026-06-02

## What this is
The owner's goal is conversation built on the BRAIN-ANALOGUE mechanism (spiking composition computed by
actual neurons), NOT static engram-tag retrieval/ranking. The spiking bind/unbind was validated earlier this
session, but on the REAL deployed substrate it was a multi-seed BOUNDARY at the 160-tier (32 concepts/bridge):
mean ~0.80, lifted to ~0.917 only with temporal integration. This is the decisive test of whether the
biological mechanism becomes ROBUST at larger scale.

## Result -- 320-tier bridge (64 concepts/bridge), temporal-integration readout (stim=300), multi-seed
Spiking relational memory + wh-QA on the REAL 320-tier bridgeA (64 concepts, sparsity 0.007), captured codes,
n_trials=20:

| seed | REAL-code wh-QA | abstention control | synthetic wh-QA |
|------|----------------:|-------------------:|----------------:|
| 42 | 1.000 | 1.000 | 0.900 |
| 43 | 0.900 | 1.000 | 1.000 |
| 44 | 0.950 | 1.000 | 1.000 |
| **mean** | **0.950** | **1.000** | 0.967 |

All three seeds RESOLVE (>= 0.80, well above bar); abstention PERFECT every seed. between-concept cos 0.350;
cos(real, synthetic) 0.047.

## Why this is decisive (and scrutinized)
- It is a CLEAN multi-seed PASS (mean 0.95, all seeds >= 0.90), NOT a boundary like the 160-tier (0.80). The
  brain-analogue composition mechanism is robust at 64 concepts/bridge on the real deployed substrate.
- Abstention is PERFECT every seed -- the decisive anti-artifact control. A drive-echo / code-distinctness
  artifact CANNOT correctly abstain on unstored facts (it would clean up to some concept and answer wrongly).
  Perfect abstention + cos(real,synth)=0.047 (real codes genuinely differ from the idealized patterns)
  establish this is genuine composition, not an encoding echo.
- BETTER at 64 (320-tier) than 32 (160-tier): the 320-tier's sparser codes (sparsity 0.007 -> between-cos
  0.350) compose more cleanly than the 160-tier's denser codes. So scaling the vocab via sparser distributed
  codes HELPS the biological composition, not hurts it.
- The lever is biological: temporal integration (a longer readout window = sustained encoding) denoises the
  captured concept codes enough for the spiking cleanup to clear composition -- the same mechanism that
  lifted the 160-tier boundary, now carrying 64 concepts to a clean pass.

## Significance for the goal
This is the first demonstration that the biological spiking composition -- relational reasoning computed by
actual spiking neurons -- is ROBUST (multi-seed clean pass) at the 64-concept-per-bridge scale of the real
deployed 320-tier substrate. So a conversation's relational reasoning can run on the brain-analogue mechanism
(not static retrieval) at 4x the scale of the earlier 16-word relational demos. Integration demo:
compose_bio_conversation_320_demo.py (teaches SVO facts, answers wh-queries by spiking unbind + cleanup,
abstains on unknowns -- all spiking, at 64 concepts).

## Honest scope
- This is WITHIN-bridge (64 concepts). The full 320-concept space is 5 bridges that currently share seed-42
  sparse patterns, so a single GLOBAL 320-way spiking cleanup has duplicate codes (documented). Cross-bridge
  association currently uses the engram-tag mechanism. Full biological 320-way composition would need distinct
  per-bridge codes (the documented per-bridge-distinct-seed recovery path) -- a clean next step.
- The concept codes are still substantially given by the sparse encoding (the cheating-audit honest scope);
  the COMPOSITION on top is genuine (abstention-controlled), here shown robust at 64 concepts.

## Integration demo transcript (compose_bio_conversation_320_demo.py, 64 concepts, all spiking)
```
=== biological relational conversation @ 64 concepts (320-tier, backend=cupy) ===
  loaded bridge; capturing 64 real concept codes (temporal integration)...
  -- teaching facts (each stored as a spiking role(x)filler bind) --
    stored:  agent=apple  action=fish  patient=leaf
    stored:  agent=cat  action=tree  patient=cup
    stored:  agent=person  action=road  patient=sun
  -- asking (answers computed by spiking unbind + cleanup) --
    who fish leaf?  -> apple   (OK)
    what did apple fish?  -> leaf   (OK)
    who tree cup?  -> cat   (OK)
    what did cat tree?  -> cup   (OK)
    who road sun?  -> person   (OK)
    what did person road?  -> sun   (OK)
  -- abstention (a fact never taught) --
    who river dog?  -> (unknown -- correctly abstains)
  RESULT: 6/6 wh-answers correct via the spiking bind; abstains on unknown = True. Relational reasoning computed by spiking neurons at 64 concepts.
```
6/6 wh-answers correct via the spiking bind; correct abstention on an untaught fact. Relational reasoning at 64 concepts computed by spiking neurons.
