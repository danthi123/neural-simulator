---
type: finding
status: positive
date: 2026-08-26
faculty: selective-attention (biased-competition multi-referent pronoun disambiguation)
mechanism: Wong-Wang / Desimone-Duncan lateral inhibition between held discourse-referent attractors (BiasedCompetitionContextBuffer, already wired into MultiTurnAgent)
runner: research/runners/_biased_competition_flip_soak.py
flag: BRAIN_BIASED_COMPETITION (default OFF)
seed-waiver: production WIRE-IN of the 2026-06-19 6-seed-validated faculty; the wire-in's flag-OFF byte-identity is STRUCTURAL (identical argument value) and seed-42 exact-compared, and load-bearing is the de-risk's own bias-lesion reproduced at seed 42; the 6-seed no-regression pool soak is the PARENT's default-on gate, not run here.
artifacts:
  - research/findings/raw/_biased_competition_prodflip/soak_seed42.json
verdict: GO (production wire-in; flag ships default-OFF; parent flips default-on after the 6-seed pool soak)
---

# Selective attention (biased competition) — production wire-in behind BRAIN_BIASED_COMPETITION (GO, flag default-OFF)

**seed-waiver:** this is a production WIRE-IN of a faculty already validated at 6 seeds (2026-06-19 de-risk, GO-arm
5/6). The wire-in's flag-OFF byte-identity is STRUCTURAL + seed-42 exact-compared; its load-bearing is the de-risk's
own bias-lesion reproduced at seed 42. The 6-seed no-regression soak is the parent's default-on gate (command below).

## What was wired

The validated organ — `BiasedCompetitionContextBuffer` (`research/runners/biased_competition_buffer.py`) — was
already importable and already wired into `MultiTurnAgent` behind its `enable_biased_competition` constructor flag.
But every LIVE build site hard-coded `enable_biased_competition=False`, so the faculty was dark in production. This
change routes the existing organ into the live pipeline behind ONE new env flag. NO `sim/` edit; the organ + its
lesion oracle are unchanged (reuse-by-import).

- **New organ (the gate):** `research/runners/biased_competition_prod.py` — `biased_competition_enabled()` reads
  `BRAIN_BIASED_COMPETITION` (unset/`0`/`false`/`off`/`no`/`""` -> `False`; `1`/`true`/`on`/`yes` -> `True`).
- **Guarded wiring (4 live build sites, 3 files):** each `enable_biased_competition=False` became
  `enable_biased_competition=_bc_enabled()` (import `biased_competition_enabled as _bc_enabled`), with a marked
  comment block:
  - `research/runners/brain_chat_tui.py` — `_load_self_knowledge` + `_build_tiny_demo` (console/TUI brains).
  - `research/runners/rich_answer_composer.py` — `_build_smoke_chat` (rich-answer smoke).
  - `research/runners/developed_brain_io.py` — `load_developed_brain` (the WEBAPP production loader:
    `webapp/server.py:_build_chat_brain` -> `load_developed_brain`, so the flag reaches the live chat brain here).
- **Flag default:** OFF. The parent flips the production default (to ON) after the 6-seed pool soak passes.

## The faculty (what changes when ON)

When ON, a bare pronoun over >=2 held discourse referents routes through the WTA biased competition (mutual
inhibition between the referent assemblies + a small CONTENT bias from the query verb's selectional restriction) and
binds the content-favored referent. Opposing-animacy pair {cat=animate, ball=inanimate}: "what does it eat?"
(animate-selecting) -> cat; "where does it roll?" (inanimate-selecting) -> ball. UNIQUE-referent turns (< 2 held)
never enter this path -> byte-identical to OFF. The no-confab moat is preserved (empty WM / content-silent verb ->
abstain). Mechanism + spiking-substrate validation: `2026-06-19-multireferent-biased-competition-derisk.md`
(GO-arm 5/6 seeds; anti-cheat controls — permuted-position, equal-salience — 6/6). Integration into MultiTurnAgent:
`2026-06-19-multireferent-integration-multiturnagent.md`. CI pin: `tests/test_multireferent_biased_competition.py`.

## Verification (seed 42; SIM_BACKEND=numpy; stub-free organ-level, no Qwen warm)

**A) WIRING.** Static: each of the 4 live sites passes `enable_biased_competition=_bc_enabled()` and imports the
organ; 0 remaining hardcoded-`False`. Dynamic: the wired expression drives `agent.enable_biased_competition` — flag
unset -> `False` (buffer never built, `bcw is None`); `BRAIN_BIASED_COMPETITION=1` -> `True`.

**B) FLAG-OFF BYTE-IDENTITY.** `biased_competition_enabled()` returns exactly `False` when the flag is unset — the
SAME literal every site passed before this change — so flag-OFF is byte-identical to today by CONSTRUCTION. Confirmed
in data: an ordinary (unique-referent) turn answers identically whether built with a hardcoded `False` or the wired
expression (`what does it eat` -> `fish` both), and the soak exact-compares the full ordinary-turn reply set
OFF-vs-ON: identical.

**C) LOAD-BEARING (the de-risk's own bias-lesion oracle).** Two held referents {cat, ball}; the CI-validated read
sequence (resolve `eat`, `what_does("it","eat")`, resolve `roll`). Both referents carry an `eat` fact so a wrong
bind returns a DIFFERENT non-None answer (fact-availability controlled).

| arm | resolve `eat` | resolve `roll` | reading |
|---|---|---|---|
| OFF (plain path) | None | None | the plain loop holds the SET, no ranked salience -> abstains on the 2-referent tie |
| ON (bias 2500 pA) | cat | **ball** | content flips the winner cat<->ball (selective attention resolves the pronoun) |
| LESION (bias 0 pA) | cat | **cat** | zeroing the bias reverts to the seed-dependent INTRINSIC attractor -> `roll` wrong-binds cat |

The bias is LOAD-BEARING: ON, the `roll` direction resolves `ball`; LESIONED, the feature-flip VANISHES and `roll`
reverts to the intrinsic winner (cat, == `eat`'s winner -> content ignored). Unrelated content is untouched: the
`eat` answer is `fish` in ON and LESION alike. The lesion holds by construction (the bias is a per-read injected
current; `biased_competition_bias_pA=0.0` means it is simply never injected; the buffer has no plasticity to regrow
it). This is the de-risk's §3 "bias-LESION breaks" row, reproduced through the wired production `_resolve` path.

## The soak (the parent's default-on gate)

`research/runners/_biased_competition_flip_soak.py` runs the conversation OFF vs ON per seed. GATE = NO-REGRESSION:
every ORDINARY turn byte-identical OFF vs ON (exact compare). FACULTY-LIVE (per-seed diagnostic): the triggered
multi-referent turn resolves the content-favored referent ON, differs from OFF, and the bias-lesion reverts it.
Seed 42: NO_REGRESSION=True, FACULTY_LIVE=True -> GO
(`research/findings/raw/_biased_competition_prodflip/soak_seed42.json`).

**Parent runs before flipping default-on:**
```
SIM_BACKEND=numpy python -m research.runners._biased_competition_flip_soak --seeds 42 43 44 100 101 102
```
Bar: 6/6 NO_REGRESSION (the flip is safe). FACULTY_LIVE is reported per-seed; the de-risk's GO-arm is 5/6 — the
seed-100 extreme-intrinsic-asymmetry case ABSTAINS on `roll` under both OFF and ON (moat-preserving, NOT a
regression), so a FACULTY_LIVE miss there is expected and does not gate the flip.

## Honest scope

- Verification here is seed 42 (organ-level, numpy). The wire-in's byte-identity is structural; the faculty itself
  is the 6-seed de-risk. The 6-seed no-regression soak is PENDING (parent-run) — do not read this as a 6-seed
  wire-in result.
- The content-bias SCORING (`content_bias_target` + animacy/verb-selection lexicons) remains a HOST scaffold flagged
  for conversion; the brain-based piece is the spiking competition + suppression. The learned-synaptic-map conversion
  (`SpikingFeatureCompat`, gap#3 A1) is a separate, already-validated follow-on and is not touched by this wire-in.
- Flag ships DEFAULT-OFF. This is a wired-but-default-off wire-in until the parent flips it after the pool soak; it
  is NOT yet integrated/production-default per docs/TERMS.md.
