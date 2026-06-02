> **RESOLVED 2026-06-02 (the path this finding proposed is CONFIRMED):** distinct-flat codes (retrain bridges with distinct seeds) compose ROBUSTLY on structured facts -- multi-seed 1.000/1.000/1.000 at 192 concepts (seeds 42/43/44, incl. seed 42 where hierarchical = 0.000). See 2026-06-02-flat-distinct-RESOLVES-robust-cross-bridge-biological-composition.md. The nesting wall is avoided by removing the 2nd binding level.

# Honest retraction: full-320 biological composition via hierarchical bind hits the NESTING WALL (structured facts) -- 2026-06-02

## What was overclaimed
The full-320 spiking probe reported relational QA 1.000/1.000/0.950 (seeds 42/43/44) + perfect abstention,
and I wrote it up as "the brain-analogue conversational substrate scales to 320 concepts." That probe sampled
RANDOM fillers (3 random concepts from all 320 per fact). The integration demo, using STRUCTURED cross-bank
facts (noun agent / verb action / adjective patient) at seed 42, got 0/6 -- the opposite. Scrutiny resolved it.

## The truth (full-3-slot QA, same composition seeds)
| seed | RANDOM-filler facts | STRUCTURED (noun/verb/adj) facts |
|------|--------------------:|---------------------------------:|
| 42 | 1.000 | **0.000** |
| 43 | 1.000 | 0.950 |
| 44 | 0.950 | 1.000 |

On structured facts the composition is wildly SEED-DEPENDENT and CATASTROPHIC at seed 42 (0.000) -- the very
seed where random fillers scored 1.000. So the "RESOLVES" was a random-sampling artifact.

## Mechanism: the hierarchical bind stacks a 2nd binding level -> the documented nesting/SNR wall
To make the 5 shared-pattern bridges' 320 codes distinct without retraining, each concept was coded as
bridge_role (Hadamard) within_code. The relational composition then binds composition_role (Hadamard) that:
  composition_role (x) bridge_role (x) within_code  -- a 2-LEVEL nested bind.
The project already documented that flat nested / multi-hop binding hits an SNR wall (separate-storage is the
universal structure mechanism; flat nesting degrades). At some seeds the composition-role and bridge-role
vectors interfere (e.g. a composition role partially aligning with a bridge role), and the unbind cannot
cleanly recover the filler -> catastrophic failure for structured facts whose fillers come from specific
bridges (so the bridge-role structure is systematic, not averaged out as with random fillers). Recognition
(distinguishing the 320 distinct codes) is unaffected; it is the COMPOSITION that breaks.

## What still stands (honest)
- WITHIN-bridge 64-concept biological composition: ROBUST multi-seed (1.000/0.900/0.950, abstention 1.000).
  These are FLAT codes (no extra nesting level) -> no nesting wall. Real, validated.
- The hierarchical bind makes 320 codes DISTINCT (between-cos max 0.537) -> recognition over 320 is fine.
- The integration demo at a GOOD seed (43/44) would work; the headline "320 composition robust" does NOT.

## The honest path to robust full-320 biological composition
Use DISTINCT FLAT codes, not a 2nd nesting level: retrain the 5 bridges with distinct seeds (42-46) so the
320 codes are distinct WITHOUT the bridge-role bind -> the composition is a single-level bind (like the
within-bridge 64 that IS robust). That is a retrain (~1.5 hr) but it avoids the nesting wall. The
no-retrain hierarchical shortcut trades the duplicate-code problem for the nesting-interference problem.

## Discipline note
The random-filler probe + a positive abstention control were NOT sufficient -- they masked a structured-fact
catastrophe. The integration demo (a different, realistic fact distribution) exposed it. Lesson: validate the
REALISTIC input distribution (structured facts), not just random samples; a clean control on the wrong
distribution can still mislead.
