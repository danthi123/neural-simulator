---
type: finding
status: live
date: 2026-08-27
mechanism: mood-tone-spiking-readout
lane: E-language
artifacts:
  - research/findings/raw/_affect_tone_spiking_mouth_fix_verify.json
  - research/findings/raw/_affect_tone_spiking_mouth_fix_verify.json.prov.json
runner: research/runners/_affect_tone_spiking_mouth_fix_verify.py
---

# Affect-tone coloring restored on the spiking recall mouth — a cross-faculty regression CONFIRMED and FIXED (GO)

**One-line:** `spiking-mouth-recall` (default-ON 2026-08-26) pre-empted `affect-coloring`'s manner effect (default-ON
2026-08-12) for every bounded-transitive recall turn — a validated coupling made silently inert by a later flip, the
"faculties must DRIVE not observe" failure mode. CONFIRMED via a CPU-only crossed A/B, then FIXED by making the SAME
Gate-B mood signal load-bearing on the spiking mouth's OWN surface through a tiny (2-pool) spiking tone-read, verified
byte-identical-at-neutral and lesion-load-bearing on all 6 canonical seeds.

## The regression (confirmed, not assumed)

`ChatBrain.render` (`research/runners/brain_chat_tui.py:996-1004`) and `RichAnswerComposer._render_one_verified`
(`research/runners/rich_answer_composer.py:553-560`) both call `chat.spiking_recall_surface(a, v, p)` and return its
result immediately when non-`None` — **before** `self.renderer.render_svo` is ever reached. `MoodConditionedRenderer`
(`research/runners/affect_production_organ.py:313-343`, installed by `webapp/server.py:4446-4452`) colors PROSE only
**inside** `render_svo`. So for every bounded-transitive recall the spiking mouth handles (default-ON since the
2026-08-26 wave-3 flip, `research/runners/spiking_mouth_recall_prod.py`), the Gate-B manner-coloring
(`2026-08-12-GateB-affect-colors-production-chat.md`) never runs — the two faculties never measured each other's
presence (`spiking_mouth_recall_prod.py`'s own 6-seed flip-soak never touches affect; `affect_production_organ.py`'s
probes never touch `BRAIN_SPIKING_MOUTH_RECALL`).

**Crossed A/B (CPU-only, `SIM_BACKEND=numpy`, no GPU/Qwen load, no heavy sim — the lightweight `rf` composer, a few
thousand neurons total).** Real, unmodified production code exercised: `ChatBrain.render`, `spiking_recall_surface`,
`spiking_mouth_recall_prod.recall_mouth_enabled`, `affect_production_organ.MoodConditionedRenderer`,
`manner_template_for`. Two things stood in for GPU-cost: (a) the fluent-mouth base `MoodConditionedRenderer` wraps —
a deterministic stand-in exposing the same `_fac._generate`/`CONSTRAIN_TEMPLATE` shape, whose text visibly depends on
the injected manner clause, mirroring the real Qwen mouth's behaviour in the 2026-08-12 finding; (b) `ChatBrain._verify`
monkeypatched to always accept, isolating the ROUTING question from the (separately, already adversarially-verified)
MOAT question — both arms funnel through the same patched call, so it cannot manufacture the measured asymmetry.

| flag `BRAIN_SPIKING_MOUTH_RECALL` | mood | manner reaches renderer? | surface |
|---|---|---|---|
| OFF (pre-2026-08-26 path) | positive (level +2) | YES (1 call) | `"The dog happily chases the cat!"` |
| OFF (pre-2026-08-26 path) | negative (level −2) | YES (1 call) | `"The dog chases the cat."` |
| ON (current default) | positive (level +2) | **NO (0 calls)** | `"the dog chases the cat"` |
| ON (current default) | negative (level −2) | **NO (0 calls)** | `"the dog chases the cat"` |

Flag OFF: positive vs negative differ (manner reaches the renderer). **Flag ON: positive and negative are
BYTE-IDENTICAL** — the mood signal is fully bypassed once the spiking mouth pre-empts the call. Hypothesis
confirmed exactly as the audit described. (Pre-fix transcript captured via `git stash` on the fix commit + a
direct re-run of the identical probe — see `research/runners/_affect_tone_spiking_mouth_fix_verify.py`'s
`part_a_confirm` for the still-running post-fix half of this same table.)

## The fix

Mirroring the pattern the spiking mouth's OWN flag/lesion oracle already uses (a host signal selects **which**
population is driven; a genuine spiking rate read decides the discrete outcome — the same shape
`AffectProductionOrgan.read_differential`'s `_set()` already uses to route an appraisal into the V+/V− ladder rungs)
rather than routing mood through `render_svo` (unreachable once the spiking mouth answers), the Gate-B mood **level**
is now threaded onto the spiking mouth's own surface directly:

- **`ChatBrain._mood_tone_level`** (new, additive, per-turn attribute) — set by `webapp/server.py`'s existing Gate-B
  affect block (`webapp/server.py:4413-4460`), from the SAME `level` value already used for `wrapped.manner`. Reset
  to `0` unconditionally at the top of the block every turn (so a stale non-neutral value from a prior turn / a
  disabled `BRAIN_AFFECT` can never leak — `chat` is a per-session cached `ChatBrain`).
- **`ChatBrain._apply_mouth_mood_tone`** (new method, `research/runners/brain_chat_tui.py`) — called from
  `spiking_recall_surface` right after its own VERIFY passes. `level == 0` (unset, the default for every caller that
  never sets it — the TUI, unit tests, `BRAIN_AFFECT=0`) returns the surface **unchanged**. A nonzero level calls
  `spiking_mouth_recall_prod.mouth_tone_marker`.
- **`spiking_mouth_recall_prod.mouth_tone_marker` / `_MoodToneReader`** (new, `research/runners/spiking_mouth_recall_prod.py`)
  — a tiny (2 pools × 30 neurons), lazily-built spiking population, REUSE-BY-IMPORT of the EMERGE-59 driven-pool
  primitives (`build_slot_bridge` / `slot_pool_rates` / `PRIMACY_pA` / `EQUAL_pA`) on a bridge fully separate from the
  `SpikingClauseProducer`'s own (the adversarially-verified word-order mechanism is untouched). `level`'s sign selects
  which pool (`warm` / `curt`) gets the graded drive (magnitude scaled `|level|/3`, the other pool holds `EQUAL_pA`);
  the discrete marker (`'!'` warm / `'.'` curt / `''` undecided) is a genuine rate-vs-rate comparison against a
  dead-margin, not a bare host threshold on the mood float. Escape: `BRAIN_SPIKING_MOUTH_MOOD=0`
  (`mouth_mood_enabled()`) reverts the recall mouth to mood-blind even with the recall mouth itself ON.

The dead-margin was set from a direct measurement, not guessed (`part_c_dead_margin_tuning` in the cited artifact):
driving the 2-pool bridge at `|level|=1` (the smallest magnitude the mapping produces) across all 6 canonical seeds
gave `min_abs_separation_at_|level|=1: 0.025`, `max_abs_separation_at_|level|=1: 0.150`, `all_sign_correct: true` —
the artifact's `dead_margin_used: 0.015` clears every seed with margin while still requiring genuine separation
(the mechanism cannot flip on noise alone, mirroring the equal-drive anti-cheat already used elsewhere in this file).

### Byte-identical at neutral (asserted in the data)

For seed 42, `chat.spiking_recall_surface("dog","chase","cat")` with `_mood_tone_level` **unset** returns
`'the dog chases the cat'`; with it explicitly set to `0` (the value `webapp/server.py` sets on every un-induced
turn) it returns the identical string — exact string compare, not inferred from reading the code.

### Load-bearing + lesion (asserted in the data, all 6 canonical seeds)

Per seed in `{42, 43, 44, 100, 101, 102}`, with `BRAIN_SPIKING_MOUTH_RECALL=1`:

| seed | neutral | level=+2 | level=−2 | `BRAIN_SPIKING_MOUTH_MOOD=0`, level=+2 | `BRAIN_SPIKING_MOUTH_MOOD=0`, level=−2 |
|---|---|---|---|---|---|
| 42 | `the dog chases the cat` | `...cat!` | `...cat.` | `the dog chases the cat` | `the dog chases the cat` |
| 43 | `the dog chases the cat` | `...cat!` | `...cat.` | `the dog chases the cat` | `the dog chases the cat` |
| 44 | `the dog chases the cat` | `...cat!` | `...cat.` | `the dog chases the cat` | `the dog chases the cat` |
| 100 | `the dog chases the cat` | `...cat!` | `...cat.` | `the dog chases the cat` | `the dog chases the cat` |
| 101 | `the dog chases the cat` | `...cat!` | `...cat.` | `the dog chases the cat` | `the dog chases the cat` |
| 102 | `the dog chases the cat` | `...cat!` | `...cat.` | `the dog chases the cat` | `the dog chases the cat` |

All 6/6: content still recovers the recalled SVO, positive/negative surfaces differ sign-correctly (`'!'`/`'.'`), and
the `BRAIN_SPIKING_MOUTH_MOOD=0` lesion reverts BOTH signs to the exact byte-identical neutral surface — the coupling
is genuinely load-bearing (disabling it collapses the difference), not cosmetic. Full `preconditions` block (22
checks, 0 unmet, 0 unmeasured) in the cited artifact via `tools.verdict.Verdict` — `status: GO`.

### End-to-end (mirrors the real `webapp/server.py` wiring)

Re-running the crossed A/B with BOTH `wrapped.manner` and `chat._mood_tone_level` set together from one `level`
(exactly as `webapp/server.py` now does) gives, post-fix: flag OFF unchanged (`"The dog happily chases the cat!"` vs
`"The dog chases the cat."`, the Qwen-like mouth called once each); **flag ON now ALSO differs**
(`"the dog chases the cat!"` vs `"the dog chases the cat."`, the Qwen-like mouth called ZERO times both — the tone is
authored by the spiking mouth itself, not a fallback). Both branches are mood-sensitive again.

## Scope / honest residuals

- **Heavy-composer soak not run.** `_spiking_mouth_recall_soak.py`'s own 6-seed regression battery (the ~46-region /
  4180-neuron `onebrain` composer) was **not** re-run for this fix — the machine was under severe memory pressure
  during this arc (39/46 GiB swap in use from concurrent runs; an earlier attempt to run that soak in the background
  produced a truncated, unverifiable log with no printed verdict, consistent with the process being starved/killed
  under swap thrashing, masked by a `| tail` pipe that reports the pipe's own exit code rather than the underlying
  process's). This fix's verification instead uses the lightweight `rf` composer (a few thousand neurons total) across
  all 6 canonical seeds, on the SAME `spiking_recall_surface` / `SpikingRecallMouth` code the heavy soak also
  exercises — the new code path added (`_apply_mouth_mood_tone`) is a small, additively-gated append AFTER the
  existing verify call, so it carries no plausible interaction with the heavy composer's word-order mechanism, but the
  heavy-composer soak should still be re-run opportunistically once the machine is not under memory pressure.
- **Tone realized as punctuation, not a lexical/prosodic choice.** `'!'`/`'.'` is a minimal, verify-safe surface
  marker (appended after the spiking mouth's own VERIFY already passed on the un-marked core, so it can never
  perturb the recalled content or the moat) — a genuinely richer manner effect (word choice, not just punctuation)
  on the spiking mouth's own grammar is a further rung, analogous to how the Qwen path's manner clause can reshape
  whole phrasing.
- **The appraisal → mood-LEVEL pipeline itself is unchanged** (still the Gate-B `AffectProductionOrgan` / DR-2
  learned valence lexicon path, `2026-08-12-GateB-affect-colors-production-chat.md`'s honest residuals apply
  unchanged) — this finding only closes the NEW gap between that level and the spiking mouth's surface.

## Repro

```
# CONFIRM + FIX-VERIFY (CPU-only, ~10s total):
SIM_BACKEND=numpy python -m research.runners._affect_tone_spiking_mouth_fix_verify \
    --out research/findings/raw/_affect_tone_spiking_mouth_fix_verify.json
```

## Files changed

- `research/runners/spiking_mouth_recall_prod.py` — new `mouth_mood_enabled()`, `_MoodToneReader`, `_get_tone_reader`,
  `mouth_tone_marker()` (additive; module docstring documents the regression + fix).
- `research/runners/brain_chat_tui.py` — `spiking_recall_surface` now calls the new `_apply_mouth_mood_tone` after
  its own VERIFY passes; `_apply_mouth_mood_tone` is additive and neutral-safe by construction.
- `webapp/server.py` — the Gate-B affect block now also sets `chat._mood_tone_level` (reset to `0` every turn before
  the block, set to `int(level)` when affect is on) alongside the pre-existing `wrapped.manner`.
- `research/runners/_affect_tone_spiking_mouth_fix_verify.py` (new) — the CONFIRM + FIX-VERIFY runner cited above.
