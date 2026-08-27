---
type: finding
status: live
date: 2026-08-27
mechanism: mood-tone-spiking-readout
lane: E-language
artifacts:
  - research/findings/raw/_affect_tone_spiking_mouth_fix_verify.json
  - research/findings/raw/_spiking_mouth_recall_soak.json
runner: research/runners/_affect_tone_spiking_mouth_fix_verify.py, research/runners/_spiking_mouth_recall_soak.py
---

# Mood→tone spiking-mouth fix: heavy composer soak re-run and CONFIRMED GO — stays default-ON, with a stale-instrument bug found and fixed along the way

**One-line:** the mood→tone fix from `2026-08-27-affect-tone-coloring-restored-on-spiking-mouth.md` (branch
`research/affect-tone-spiking-mouth`, merged to `main`) lands GO on both required checks: the mood→tone coupling
itself (lightweight verify, 6/6 seeds) AND the heavy `BRAIN_SPIKING_MOUTH_RECALL` composer regression soak that the
original finding explicitly deferred ("Heavy-composer soak not run... machine was under severe memory pressure").
Re-running that soak first surfaced an unrelated, pre-existing instrument bug (found, root-caused, fixed, re-verified
GO 6/6) — not a regression from this fix. `BRAIN_SPIKING_MOUTH_MOOD` stays default-ON (`_MOUTH_MOOD_DEFAULT_ON = True`
in `spiking_mouth_recall_prod.py`, unchanged by this session — it shipped default-ON with the merged fix commit,
since it restores an already-default-ON coupling rather than introducing a new opt-in one).

## What this closes

The merged fix's own honest residual named exactly one open item: the heavy composer soak
(`research/runners/_spiking_mouth_recall_soak.py`, the flip-soak backing `BRAIN_SPIKING_MOUTH_RECALL`'s own
2026-08-26 wave-3 default-on flip) had not been re-run against the mood-tone addition. This session re-ran it
(`SIM_BACKEND=numpy`, seeds 42/43/44/100/101/102) as the GO-gate for keeping `BRAIN_SPIKING_MOUTH_MOOD` default-ON.

## First run: NO-GO 0/6 — investigated before being believed

The first re-run came back NO-GO on all 6 seeds, with `flag-OFF byte-identical` failing on every seed — a red flag,
since the mood-tone fix's own code (`_apply_mouth_mood_tone`) only ever runs AFTER `spiking_recall_surface`'s own
VERIFY passes, and is never reached at all when `BRAIN_SPIKING_MOUTH_RECALL` is off. A change confined to that path
should not have been able to break the flag-OFF arm. Per this repo's standing discipline (a NO-GO verdict is the
START of investigation, not the end), the result was checked against a **clean pre-merge `main` worktree**
(`69b65bd22`, before either of today's two branch merges) rather than accepted at face value: **the identical
NO-GO reproduced there too** — proof the failure predates and is independent of both branches landed this session.

## Root cause: the soak's own OFF-arm went stale when its subject's default flipped

`_spiking_mouth_recall_soak.py::_set_flag(False)` implements OFF as `os.environ.pop("BRAIN_SPIKING_MOUTH_RECALL",
None)` — correct when that flag defaulted OFF (its pre-2026-08-26 state: unset == OFF). The 2026-08-26 wave-3
landing flipped `_RECALL_MOUTH_DEFAULT_ON = True` in `spiking_mouth_recall_prod.py` **without updating this soak**,
so as of that flip `unset` reads as **ON**, not OFF. Confirmed directly (not inferred): with the var unset,
`recall_mouth_enabled()` returns `True`; forcing it to `chat.render()` with the var popped vs set to `"1"` gives
byte-identical output for all 17 transitive facts (`on_surface == off_surface` every time) — the soak's "OFF" arm
has been silently ON-vs-ON since the flip, collapsing every comparison it makes (flag-off-byte-identity, surface
re-authored, rate-read lesion, load-bearing coverage) to vacuous. This is why the ALREADY-COMMITTED artifact (banked
2026-08-26, `git show HEAD:research/findings/raw/_spiking_mouth_recall_soak.json`, GO 6/6, 15/17 load-bearing facts
per seed) still reads GO — it was produced BEFORE the wave-3 flip landed, when `pop()` still meant OFF; only a
fresh re-run after the flip exposes the staleness. Filed: `research/FAILURE_LOG.md` 2026-08-27 entry (general class:
any flip-soak using `os.environ.pop(FLAG)` as its OFF arm goes stale the moment that flag's production default
flips ON, with no error; a broader audit of other flip-soaks for the same pattern is spawned as a follow-up, not
done this session).

## Fix + re-verify

`_set_flag(False)` now writes an explicit `"0"` instead of popping (`research/runners/_spiking_mouth_recall_soak.py`).
Re-run, same 6 seeds:

| seed | GO | byte-identical(OFF) | load-bearing | no-regression | spiking-authored facts |
|---|---|---|---|---|---|
| 42 | True | True | True | True | 15/17 |
| 43 | True | True | True | True | 15/17 |
| 44 | True | True | True | True | 15/17 |
| 100 | True | True | True | True | 15/17 |
| 101 | True | True | True | True | 15/17 |
| 102 | True | True | True | True | 15/17 |

**VERDICT: GO (6/6 seeds).** The resulting artifact (`research/findings/raw/_spiking_mouth_recall_soak.json`) is
**byte-identical** to the artifact already committed on `main` from 2026-08-26 — only the provenance sidecar's
run_id/timestamp/git_sha differ. This is the clean, expected outcome: the corrected instrument reproduces exactly
what was banked before the flip, now genuinely re-verified against today's merged `main` (comprehension-animacy +
affect-tone-mood, both landed this session) rather than resting on a pre-flip artifact.

## Combined with the lightweight verify, both GO-gate legs now hold

- **Load-bearing + byte-identical-at-neutral** (the mood→tone coupling itself):
  `research/runners/_affect_tone_spiking_mouth_fix_verify.py`, 6/6 seeds GO, re-confirmed on the merged `main` HEAD
  in this session (unchanged from the original finding's numbers — neutral surface byte-identical, `level=+2`/`-2`
  sign-correct `'!'`/`'.'`, `BRAIN_SPIKING_MOUTH_MOOD=0` lesion reverts cleanly on all 6 seeds).
- **No chat-turn regression** (the composer it rides on): `_spiking_mouth_recall_soak.py`, 6/6 seeds GO (this
  finding).

Per the stated GO-gate ("no chat-turn regression across 6 seeds AND the mood→tone effect is load-bearing"), both
legs are satisfied. `BRAIN_SPIKING_MOUTH_MOOD` stays **default-ON** — no code change needed for the flip itself,
since the merged fix commit already shipped `_MOUTH_MOOD_DEFAULT_ON = True` (it restores an existing default-ON
Gate-B coupling onto a surface that pre-empted it, rather than introducing new opt-in behavior). The byte-identical
escape (`BRAIN_SPIKING_MOUTH_MOOD=0`) is unchanged and still reverts to the mood-blind surface.

## Honest scope

- This closes the ONE deferred item the original mood-tone finding named. It does not re-open or re-verify anything
  else about `BRAIN_SPIKING_MOUTH_RECALL` itself (that faculty's own 6-seed GO is UNCHANGED — same numbers,
  re-confirmed) — this session only fixed the SOAK's ability to correctly measure it going forward.
- The general instrument-staleness class (any flip-soak's `os.environ.pop`-as-OFF assumption going stale when its
  subject's default flips) is filed but NOT audited across other soaks this session — see the FAILURE_LOG entry.

## External context (deep-research gate, e-language lane)

<!--derived-->
Guo, Xu & Chua, "Emotional Prosody Control for Speech Generation" (arXiv:2111.04730), condition speech generation
on a continuous Arousal-Valence Prosody Control block. Our `mouth_tone_marker` (a rate-vs-rate spiking read that
selects a discrete `'!'`/`'.'` marker off a single mood-level scalar) is a minimal, single-scalar instance of the
same class of technique — an external affect signal selecting/modulating the surface render — which confirms the
pattern rather than surfacing a missed simpler fix, and is the named external reference for the richer
lexical/prosodic follow-on this finding's own honest residual already calls out ("tone realized as punctuation,
not a lexical/prosodic choice").
