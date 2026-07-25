# A1 consolidation: the direct-binding REGRESSION is caused by the nmda_slow attractor self-loops (controlled NMDA bisect) — removing them recovers 6/8; the REAL A1 blocker (stranded compositional engram) is unchanged (2026-07-24)

## Context
A1 = the "compositional consolidation → lasting world-model" deferral-surpass. While working it, a de-risk found the
substrate's **direct binding** (a validated sanity check: cue a word → its own pool fires) had regressed to a
**stuck attractor** — every cue read out `adjective_pool_SMALL` (1/8), not its own pool. Before building any fix, the
cause was bisected. This run was a subagent's (`a2d8fd16`) that PARKED on a stalled Monitor; the controller read the
result off disk (`scratchpad/diagdirect.out`).

## Method — a clean one-variable NMDA bisect
Two arms, **identical except the `nmda_slow` self-loop augmentation**, both trained **960 events** (matched):
- `gnmda_add` : `global_nmda=True, skip_nmda=False` → +12 ca1→concept wires **+ 8 nmda_slow self-loops** (w=12, d=0.15)
- `gnmda_noadd`: `global_nmda=True, skip_nmda=True`  → +12 ca1→concept wires, **+0 nmda_slow self-loops**

## Result — the self-loops ARE the regression
| arm | nmda_slow self-loops | DIRECT binding | signature |
|-----|----|----|----|
| `gnmda_add`  | ON  | **1/8** | every cue → `adjective_pool_SMALL`, top_r 1.3–2.4 (one dominant winner) |
| `gnmda_noadd`| OFF | **6/8** | each cue → its own pool, top_r 0.1–0.3 (no runaway) |

⇒ **the `nmda_slow` recurrent self-excitation creates a runaway winner** (`adjective_pool_SMALL`) that captures every
readout; removing it restores binding. A controlled, large-effect, one-variable result (1/8 → 6/8).
- **Honest scope:** single-seed bisect (the effect is clear-cut, not a marginal generalization claim). 960 events is
  BELOW the direct-binding baseline curve (1600 ev → 89%, 12800 ev → 94%; `2026-05-21-DIRECT-BINDING-RECOVERS-...`), so
  6/8 is the *un-regressed trajectory at 960 ev*, not a ceiling — the point is the **relative** ON-vs-OFF gap, which is
  what the matched-event control isolates.

## The real A1 blocker is UNCHANGED (this only fixed a sanity-check regression)
In BOTH arms the **compositional tag-stim readout** (drive an engram tag → read the bound adjective off the concept
pools) sits at the **noise floor**: `selective 1/3, lifted 0/3`, r ≈ 0.0013–0.0025 (attractor ON *and* OFF). The +12
`ca1→concept` augmentation wires **stayed at w=0.01** (`ca1_concept_mean_w=0.0100`) — they did not potentiate during
consolidation. This is exactly the multiply-documented **2026-05-21 stranded-engram boundary**: the compositional
engram is hippocampal-only and `ca1→concept` is too weak to lift the tag-stimulated cortical pools
(`2026-05-21-storage-locus-probe-ROOT-CAUSE-compositional-engram-is-hippocampal-only-...`,
`2026-05-21-consolidation-probe-TERMINAL-...-no-ca1-to-concept-pool-consolidation-wire.md`).

## Verdict (per THE LAW — method banked, capability open, next lever named)
- **Direct-binding regression → CAUSE isolated** = the `nmda_slow` attractor self-loops. FIX for the sanity check:
  `skip_nmda` (or gate the self-loop excitation so it doesn't dominate). Banked.
- **A1 CAPABILITY (compositional consolidation) still OPEN** — the regression was a distractor; the genuine blocker is
  the stranded compositional engram, not the attractor self-loops.
- **NEXT LEVER (A1):** make the `ca1→concept` pathway **plastic** AND ensure consolidation replay **co-activates
  ca1 + the concept pools** (so the wire actually potentiates during NREM replay), then re-test the tag-stim
  compositional readout. This is the same boundary FAMILY as 2026-05-21 → the **research gate fires**: before building,
  deep-research the biology of hippocampal→cortical *schema* consolidation (systems consolidation / the ca1→cortex
  potentiation rule that lifts a tag-stimulated cortical assembly) and rank cheap-first mechanisms.

## Provenance
Bisect log: `scratchpad/diagdirect.out` (DIAG2 DONE). Baseline curve: `2026-05-21-DIRECT-BINDING-RECOVERS-with-longer-Phase-1-...`.
Stranded-engram boundary: the two 2026-05-21 storage-locus / consolidation-probe findings.
