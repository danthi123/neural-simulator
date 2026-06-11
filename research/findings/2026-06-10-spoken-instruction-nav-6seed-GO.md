# Spoken-instruction navigation — 6-seed: GO (unanimous) — the functional one-brain integration locked at multi-seed rigor

> **Result: GO, 6/6 seeds (42–47).** The agent follows a parsed spoken command to navigate on every seed —
> COUPLED command-following accuracy = **1.000** in all six, while every control (gate-closed, route-lesioned,
> word-scrambled, conv-alone) collapses to chance. This confirms the first functional integration of the
> navigation and conversational brains (`2026-06-10-spoken-instruction-nav-GO.md`, 3-seed) at the project's
> 6-seed standard.

## 6-seed table

| seed | COUPLED (acc-vs-commanded) | ISOLATED-NAV (gate closed) | LESION (route cut, parser firing) | SCRAMBLE (vs-commanded / vs-spoken) |
|---|---|---|---|---|
| 42 | **1.000** | 0.063 | 0.063 | 0.000 / 1.000 |
| 43 | **1.000** | 0.188 | 0.156 | 0.000 / 1.000 |
| 44 | **1.000** | 0.125 | 0.094 | 0.031 / 0.969 |
| 45 | **1.000** | 0.281 | 0.000 | 0.000 / 1.000 |
| 46 | **1.000** | 0.094 | 0.031 | 0.000 / 1.000 |
| 47 | **1.000** | 0.281 | 0.000 | 0.000 / 1.000 |

Chance = 0.25 (4 directions). ISOLATED-CONV parses correctly but has no body (cannot move) on every seed.

**Verdict = GO, all 6 seeds:** COUPLED follows commands (1.000 ≥ 0.5); COUPLED ≫ ISOLATED-NAV & LESION (margin
≥ 0.20); ISOLATED-NAV and LESION at chance (≤ 0.40); SCRAMBLE collapses vs-commanded while vs-spoken stays ~1.0;
ISOLATED-CONV parses but no body.

## Reading

The **LESION** control is decisive on every seed: with the parser still firing (the gate-open *signal* present)
but the `command_route` synapses cut, command-following collapses to chance — so the behavior rides the
**synaptic route** between the brains, not any ambient or host path. The **SCRAMBLE** control confirms the body
tracks the *content* of the spoken word, not a fixed schedule. The mechanism (the parser's firing opening a gate
on the learned word→action route) and the controls are documented in the 3-seed finding; this run extends the
result to 6 seeds without exception.

This locks the night's third milestone: navigation and conversation **functionally interact** on one bridge, and
the parsed command is a behaviorally load-bearing goal signal (resolving the standing "the nav reward isn't
load-bearing" residual). Runner: `research/runners/spoken_instruction_nav.py`. Raw:
`spoken_instruction_nav.json` (42/43/44) + `spoken_instruction_nav_seed454647.json` (45/46/47).
