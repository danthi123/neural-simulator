---
type: finding
status: mixed
date: 2026-09-05
mechanism: superior-colliculus
lane: integration
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_rank24_quick_flips_verify.py --check enable-spiking-sc (byte-identical-off region
  count on the true default path, + a real bridge/step smoke with a validated warm-up/read protocol confirming
  whether the SC drives a directional cortex read-out when co-resident with the full default BG cascade +
  visual cortex); research/runners/_sc_orienting_flip_soak.py (the pre-existing 6-seed gate for the standalone
  production organ, re-run fresh); the full tests/test_visual_cortex.py + tests/test_g11_bg_runner_flags.py + 8
  sibling files re-run before/after as the no-regression suite.
runner: research/runners/_rank24_quick_flips_verify.py; research/runners/_sc_orienting_flip_soak.py
external: NO-EXTERNAL-NEEDED — a default-promotion attempt of an already 6-seed-CLOSED standalone mechanism
  (research/findings/2026-06-10-N1-spiking-superior-colliculus-CLOSED.md), not a new biological question.
artifacts:
  - research/findings/raw/_rank24_quick_flips/enable_spiking_sc_verify.json
  - research/findings/raw/sc_orienting/flip_soak_reverify_20260905.json
  - research/findings/raw/_rank24_quick_flips/postedit_pytest.txt
---
# Rank-24 quick flip "enable_spiking_sc default-on" — split verdict: the standalone production-organ flag is GO (kept), the g11_bg_runner.py library default is NOT flip-ready (tried, then reverted)

**One line.** The scaffold-retirement backlog's rank-24 entry names "flip enable_spiking_sc after a pool soak"
as a one-line flip. It maps to TWO separate flags in the tree. One (`sc_orienting_production_organ.py`'s
`BRAIN_SPIKING_SC_ORIENT`) is genuinely flip-ready and is now default-ON. The other (`g11_bg_runner.py`'s own
`enable_spiking_sc` library default) was tried, measured, and **reverted** — the scaffold-map's premise for it
does not hold, and a deeper audit than the map performed surfaces a real no-regression problem.

## Kept: `sc_orienting_production_organ.py`'s `BRAIN_SPIKING_SC_ORIENT` — GO, flipped to default-ON

This standalone production organ (2026-08-26-sc-orienting-production-wirein-GO.md) landed with
`BRAIN_SPIKING_SC_ORIENT` default OFF, its own docstring stating "the parent flips it after the pool soak" —
and a pre-built, purpose-made pool soak (`research/runners/_sc_orienting_flip_soak.py`, CPU/numpy, 6 seeds)
already existed to gate exactly that flip, unrun since 2026-08-26. Re-ran it fresh:

<!--derived-->
| gate | bar | measured (6-seed) | pass |
|---|---|---:|:---:|
| INTACT correct-cardinal (min) | >= 0.80 | 1.000 | yes |
| LESION correct-cardinal (max, `SC_SCRAMBLE=1`) | <= 0.45 (chance 0.25) | 0.333 | yes |
| INTACT embodied reach (min) | >= 0.80 | 1.000 | yes |
| LESION embodied reach (max) | <= 0.50 | 0.125 | yes |

`FLIP-GATE: GO`. Attribution: 83.3% of the mean correct-cardinal effect and 91.7% of the mean embodied-reach
effect are attributable to the intact retinotopic manipulation vs. the scrambled control
(`research/findings/raw/sc_orienting/flip_soak_reverify_20260905.json`). Flipped `sc_orient_enabled()`'s default
to `True` in `research/runners/sc_orienting_production_organ.py`.

**Important scope note, reported honestly rather than as a clean win.** `sc_orient_enabled()` currently has **no
caller anywhere in the tree** — this organ's own verify/soak scripts call `cardinal_battery(organ, lesion=...)`
/ `run_episode(organ, ..., lesion=...)` directly with an explicit `lesion` argument, never through this flag,
and no production path (including `webapp/server.py`) imports this organ yet ("off_byte_identical = N-A for
chat" in the landing finding — there was never a chat-path consumer). So this flip changes **no behavior
observable today**; it sets the correct default for whenever a future consumer wires this organ in, so that
consumer inherits the spiking SC by default. Zero risk (nothing reads the flag), modest present value (nothing
reads the flag).

## Reverted: `g11_bg_runner.py`'s library-default `enable_spiking_sc` kwarg

**What was tried:** `enable_spiking_sc: bool = False` → `True` at both function-signature sites
(`build_bg_brain_regions` and `run_moving_goal_episode`), leaving the CLI `--enable-spiking-sc` default
untouched (mirroring the `readout_source` library-vs-CLI split). **This is now REVERTED** — both sites are back
to `bool = False`, confirmed by `git diff` showing comment-only changes to this file.

**Why, in order of discovery:**

1. **The scaffold-map's own causal premise does not match the source.** Its location text claims the host
   reflex `sc_orienting_cardinal_from_image` runs "whenever --enable-visual-cortex runs without
   --enable-spiking-sc". It does not: that reflex call (the `if sc_orienting_reflex:` block) is gated by
   `sc_orienting_reflex: bool = False` — a THIRD, independent parameter. With plain `--enable-visual-cortex` and
   no other flags, **neither** the host reflex nor the spiking SC drove orienting before this flip (both gates
   were False) — the vision path built retina/V1/V2/IT but supplied no orienting signal at all. So the flip
   would not have retired an active host shortcut; it would only have ADDED a new default-on drive where none
   existed.

2. **A repo-wide audit (broader than the scaffold-map performed) found real callers that would silently
   change.** Beyond `text_train_embodied.py` (the one caller of `build_bg_brain_regions(enable_visual_cortex=True)`
   without an explicit `enable_spiking_sc` — but see below, this one turns out harmless), grepping every caller
   of `run_moving_goal_episode(...)` for `enable_visual_cortex=True` without an explicit `enable_spiking_sc=`
   turned up **7 existing research probes**: `_closure1_optionA_gate3.py`, `_closure1_optionA_gate3_fast.py`,
   `_homeostatic_g11bg_reuse_probe.py`, `_optionA_onestep_probe.py`, `_optionA_rng_vs_fp_probe.py`,
   `_merged_spiking_readout_navcmp.py`, `_tier3_spiking_living_loop_derisk.py`. Each of these calls the actual
   NAV STEP LOOP (not just the static region builder), which — unlike `text_train_embodied.py` — DOES install
   the SC's retinotopic wiring and inject its drive into the SAME `cortex_{N,E,S,W}` pools these probes' own
   mechanisms read for action selection. Flipping the library default would silently add an uninvited, imperfect
   (see next point) directional bias into narrow research probes that were written and validated with SC off.
   (`text_train_embodied.py` itself is confirmed harmless: it calls `build_bg_brain_regions` but never
   `install_spiking_sc_wiring`, so its SC regions would build with NO retinal input or cortex output wiring at
   all — extra disconnected neurons, not a new signal path. Its only effect is a shifted seeded-RNG draw
   sequence for regions built after them, already disclosed as acceptable.)

3. **A fresh, honest co-residence measurement shows the mechanism is not cleanly reliable in this configuration.**
   <!--derived--> The 2026-06-10 CLOSED result (SC/host 0.883, 12% better, scramble regresses 2.4x) — quoted from
   `research/findings/2026-06-10-N1-spiking-superior-colliculus-CLOSED.md`, not re-measured here — and today's
   `_sc_orienting_flip_soak.py` re-verify (6/6 GO) both measure the SC **in an ISOLATED minimal scaffold** (the
   nav-loop's own dedicated cascade, or the production organ's own quiet, plasticity-disabled, low-OU-noise
   bridge). Measuring it INSTEAD co-resident with the FULL default BG cascade + visual cortex — exactly what
   `build_bg_brain_regions(enable_visual_cortex=True, enable_spiking_sc=True)` builds — with a static two-goal
   directional read using the SAME validated warm-up(30)/read(160) window the production organ uses: **only 4/6
   seeds** produced the correct cardinal for both an east and a north goal (seeds 42/44/100/102 correct, 43/101
   wrong — and on the wrong seeds the SAME cortex pool won for BOTH goal directions, a systematic bias, not
   noise). This is a concrete instance of the "co-residence operating-point risk" flagged only abstractly in
   `2026-06-19-tier2-nav-spikeification-scoping.md` ("the SC bump 'starves' on the heterogeneity-OFF merged
   bridge"). `research/findings/raw/_rank24_quick_flips/enable_spiking_sc_verify.json` carries the full per-seed
   spike counts for both the naive (60-step) and the corrected (30+160-step) protocol.

**Net:** byte-identical-off for the true default path is solid (confirmed twice, 38 regions either way,
`enable_visual_cortex=False`). But "no-regression" fails for the 7 identified probes, and "load-bearing/correct
when on" is a genuine 4/6, not a clean GO. Per the task's own instruction to skip a flip that turns out not
flip-ready rather than force it, **this one is reverted** and banked here rather than landed quietly. It remains
a legitimate target for FRESH work — e.g. the "co-residence operating-point" fix the wall-reframe in CLAUDE.md
asks for (what homeostatic/competitive process does the isolated de-risk implicitly rely on that the co-resident
configuration lacks?) — not a config flip.

**Final no-regression confirmation (the actually-shipped state).** `git diff` on `g11_bg_runner.py` after the
revert shows comment-only changes (both `enable_spiking_sc: bool = False` sites restored verbatim). The 10-file
suite (`test_bridge_text_io`, `test_cluster_d`, `test_cluster_f`, `test_d1_d2_asymmetry`,
`test_distributed_motor_pop`, `test_e_inh_override`, `test_text_embeddings`, `test_td_critic_no_harm`,
`test_visual_cortex`, `test_g11_bg_runner_flags`) was run three times this session: (1) pre-edit baseline — **3
failed, 158 passed** in 1509s; (2) with the (since-reverted) library-default flip applied — same 3 pre-existing
failures, no new ones, through the point it was interrupted; (3) the final shipped state (library default
reverted + the organ flag flipped) — **3 failed, 158 passed** in 570s, the identical 3 pre-existing failures
(`test_protected_byte_untouched_across_td_critic_range`, `test_compartmentalized_da_kwarg_accepted`,
`test_spiking_snc_kwarg_accepted` — all on unrelated kwargs, none touching `enable_spiking_sc`/
`enable_visual_cortex`/`BRAIN_SPIKING_SC_ORIENT`) — see
`research/findings/raw/_rank24_quick_flips/postedit_pytest.txt`. Byte-identical pass/fail counts, run (1) vs (3).

## What neither flip touches

Neither flip changes the nav-loop's OTHER orienting shortcuts documented elsewhere in the ledger
(`heuristic_strength=1.0` default-on host Manhattan drive; the silent-commit argmax/RNG-tiebreak residuals on
`readout_source="spiking_wta"`, see the companion `--readout-source` note in
`research/runners/_rank24_quick_flips_verify.py`'s module docstring) — those are separate, larger, fresh-work
items, not part of this quick-flip package.

## Scope honesty

No `sim/` edit in either the kept or the reverted change (both are `research/runners/*.py` kwarg-default edits +
comments). No CLI-visible default changed anywhere. The kept flip is a promotion of an already-6-seed-CLOSED
mechanism to a currently-dead code path (zero present behavioral effect). The reverted flip's underlying
mechanism is NOT being characterized as broken — it is CLOSED and correct in its validated, isolated
configuration; what is newly documented is that the DEFAULT-PROMOTION specifically (which changes the
co-residence configuration for existing callers) is not yet safe to ship as a one-line flip.
