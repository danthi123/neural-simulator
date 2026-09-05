---
type: finding
status: live
date: 2026-09-05
mechanism: shared-spiking-salience-novelty-afferent
lane: integration
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: The 6-seed gate covers the shared organ's core load-bearing/lesion behavior AND all 3
  consumer-site wiring points (cheap, ~10s/seed total, numpy-CPU). The ONE addition beyond the
  6-seed gate -- a full end-to-end pass through value_choice_production_organ's REAL trained
  striosome_value critic (not just its upstream context function) -- is run ONCE at seed 42 only
  (research/findings/2026-08-21-da-gated-curiosity-threshold-wired-GO.md's own "a plumbing/
  attribution proof, not a stochastic effect size" scoping, for the identical reason: the critic's
  OWN sensitivity to its engagement input is a pre-existing, already-6-seed-GO'd mechanism this
  work does not modify -- research/findings/2026-07-23-value-critic-closure-RANK1-GO.md -- and a
  single trained build costs ~5 CPU-minutes).
verdict: GO
runner: research/runners/_shared_salience_afferent_derisk.py
artifacts:
  - research/findings/raw/_shared_salience_afferent/verify_6seed.json
  - research/findings/raw/_shared_salience_afferent/plumbing_seed42.json
external: NO-EXTERNAL-NEEDED -- couples two ALREADY in-repo, already-validated spiking mechanisms
  (the DR-1 curiosity ASK-pool novelty transduction, corr(gap,want) strongly positive, on-bridge 6-seed GO;
  and the #76/#79 spiking SNc engagement read, 6/6-seed GO). The biology (one shared salience/
  novelty afferent feeding several downstream consumers, and DA/arousal raising exploratory drive)
  is the SAME Aston-Jones-Cohen LC-NE adaptive-gain / tonic-DA vigor account already grounded for
  the sibling da-gated-curiosity-threshold coupling this wiring's root retires.
---

# ONE shared spiking novelty/salience afferent wired into the live chat turn -- retires 3 host formulas at their root (6-seed GO, default-OFF)

**Verdict: GO (wire-in de-risk, default-OFF -- NOT flipped on).** `research/coordination/scaffold_retirement_backlog.md` rank-4 named two already-independently-de-risked halves that had never been wired to each other or to the live turn: the curiosity organ's genuinely-spiking ASK-pool novelty transduction (`current_novelty_signal -> from_novelty -> excitability_drive -> cp_firing_states[ask]`, corr(gap,want)=+0.996 <!--derived--> reproduced 2026-08-12), and the SNc engagement afferent that three default-on chat couplings already read. This session generalizes the curiosity organ's spiking read from its two abstain-calibration anchors to an arbitrary continuous input (`CuriosityProductionOrgan.salience_of`, additive), and wires it (`research/runners/shared_salience_afferent.py`, `BRAIN_SHARED_SALIENCE`, default-OFF) into the THREE consumer sites that previously computed their own host novelty/salience formula in bare arithmetic:

1. **`webapp/da_mode_drives_chat.py::engagement_of()`** -- the message-engagement scalar that becomes the spiking SNc's afferent current. This is the ROOT of three downstream default-on consumers, because all three read the SAME `chat._last_da_drives["da_level"]` this workspace produces: da-mode-drives-response's engagement suffix, da-gated-encoding's write-magnitude gain (`webapp/da_encoding_drives_chat.py`), and da-gated-curiosity's crave-threshold gain (`webapp/da_curiosity_drives_chat.py`).
2. **`research/runners/bg_action_selection_production_organ.py::salience()`** -- the per-candidate SPEAK/STAY-SILENT salience bias handed to the two-channel basal-ganglia selector, the only live discrete motor decision in this codebase.
3. **`research/runners/value_choice_production_organ.py::default_context_fn()`** -- the per-candidate engagement context the learned `striosome_value` critic converts into a commit-by-value decision.

Each site keeps its OWN host sensory/environment-boundary raw-scalar computation unchanged (message novelty+richness, content-token count, fact-recency ratio -- each already a declared legitimate boundary, exactly like the SVO parser). What changes is what mediates that raw scalar on its way to the consumer: a genuine population of spiking neurons (the shared ASK-pool, excitability drive -> membrane integration -> spiking -> firing-rate readout) now sits in the path, in place of bare host arithmetic feeding the consumer directly.

## What was built

- **`research/runners/curiosity_production_organ.py::CuriosityProductionOrgan.salience_of(raw, lesion)`** (additive method, no existing code touched) -- the SAME `_read_want_raw`/calibration machinery `judge()` uses, generalized to an arbitrary raw scalar in `[0,1]` and reported as a salience NORMALIZED against the organ's own familiar/novel calibration anchors: `normalized = (want_hz(raw) - want_hz(FAMILIAR)) / (want_hz(NOVEL) - want_hz(FAMILIAR))`.
- **`research/runners/shared_salience_afferent.py`** (new module) -- `shared_salience_enabled()` (`BRAIN_SHARED_SALIENCE`, default-OFF), `shared_salience_lesioned()` (`BRAIN_SHARED_SALIENCE_LESION`), and `read_salience(raw, seed=42)`, which reads the SAME process-shared `curiosity_production_organ` singleton the follow-up-question faculty already builds (one substrate, two consumers -- no second bridge).
- **Three consumer-site edits**, each additive and flag-gated (`git diff` shows only new `if shared_salience_enabled(): ...` branches; the pre-existing `else` arithmetic is untouched):
  - `webapp/da_mode_drives_chat.py::DaModeDrivesWorkspace.observe()` -- when a turn carries content tokens, `turn_e` (the value that folds into the persistent EMA and then the SNc afferent) is read through the shared organ instead of used directly; an additive `shared_salience` trace key is attached ONLY when the flag is on (key-absence is the OFF proof, matching the sibling `curiosity_da`/`da_encoding` trace-key idiom).
  - `bg_action_selection_production_organ.py::salience()` -- the entry-gate boolean (`n == 0`, "is there any content at all") is UNCHANGED; the salience MAGNITUDE handed to the race on a content-empty turn is now the shared organ's read of the same raw content-count scalar.
  - `value_choice_production_organ.py::default_context_fn()` -- the recency/referent bookkeeping is UNCHANGED; the final per-candidate scalar handed to the critic is the shared organ's read of that context value.
- **`research/runners/_shared_salience_afferent_derisk.py`** (new de-risk/verify runner) -- the 6-seed gate below, plus the seed-42 plumbing proof.

No `sim/` edit (`git diff sim/` is empty) -- every neuron/synapse/pathway already existed in the DR-1 curiosity bridge.

## The 6-seed gate (42/43/44/100/101/102, numpy-CPU; artifact `verify_6seed.json`)

Each seed ran in its OWN subprocess (the curiosity organ's process-shared singleton is intentionally NOT seed-keyed -- production always runs one process at one seed -- so testing 6 seeds in-process would silently re-serve seed 42's cached build for every later seed; see the runner's docstring). **All 6 seeds pass all gates.**

### (A) The shared organ core -- load-bearing + lesion, generalized beyond the two binary anchors
Sweeping raw in `[0, .25, .5, .75, .95, 1.0]` at each seed:

| quantity | mean across 6 seeds | range |
|---|---|---|
| intact normalized-salience span | 1.051 <!--derived--> | 1.023 - 1.090 <!--derived--> |
| lesioned normalized-salience span | 0.0153 <!--derived--> | 0.007 - 0.026 <!--derived--> |
| corr(raw, intact normalized) | 0.981 <!--derived--> | 0.975 - 0.987 <!--derived--> |
| span attributable to the drive pathway (`tools.lab.attributable_to`) | 98.5% <!--derived--> | 97.4% - 99.3% <!--derived--> |

The lesion (`curiosity_excit_sensitivity=0`, `judge()`'s own anti-cheat, reused verbatim) is a STATIC build-time config value, not a plastic weight -- it cannot regrow within the read window (no `deliver_reward`/STDP pathway is exercised by a pure `current_novelty_signal` read), so the manipulation is verified to still hold at the moment of every measurement above. The `corr(raw, lesioned normalized)` figure is NOT reported as a meaningful correlation: with the span collapsed to ~1.5% of intact, the residual is pure OU-noise jitter, and its correlation sign flips seed to seed (-0.96, +0.56, -0.25, -0.86, -0.56, +0.07 <!--derived-->) exactly as an uncorrelated-noise residual should -- reported per `docs/TERMS.md`'s "selective" discipline (raw magnitudes + the fact that the residual is noise, not a ratio asserted alone).

### (B) Consumer 1 -- `da_mode_drives_chat` (root of 3 downstream default-on couplings)
On a fixed novel/rich message ("what does the dog chase around the yard today"):

| arm | self-produced DA level | mode |
|---|---|---|
| OFF (`shared_salience` key absent from the returned trace) | 0.7898 - 0.7912 <!--derived--> (range across seeds) | focus |
| ON | 0.7898 - 0.7965 <!--derived--> | focus |
| ON + LESION | **0.04616 <!--derived--> on every seed** | **rest** on every seed |

ON differs measurably from OFF on every seed (the shared afferent is genuinely in the path), and the LESION collapses the DA level to the SAME sub-tonic floor (0.046 <!--derived-->, REST) on all 6 seeds REGARDLESS of the message's actual novelty -- exactly the #76/#79 lesion signature this coupling's root shares with da-mode-drives-response's own established lesion proof, now caused by severing the shared afferent rather than silencing the SNc nucleus itself (a DISTINCT, upstream lesion point).

### (C) Consumer 2 -- `bg_action_selection_production_organ::salience()`
On a real message ("hello there friend", 3 content tokens): OFF returns the bare formula `(1.0, 0.0)` on every seed (exact, unchanged); ON returns `(1.04 - 1.09, 0.0)` <!--derived--> (varies per seed, tracking the shared organ's spiking read); LESION collapses back to `(0.0 - 0.005, 0.0)` <!--derived--> on every seed.

**Honest scope on this consumer.** The ONLY currently-reachable STAY-SILENT branch in `decide_action()` feeds `raw=0.0` (a content-empty turn, `n==0`) -- and at that exact anchor BOTH the intact and lesioned reads floor to ~0 (there is no novelty signal to lesion away at the familiar floor; a lesion of a novelty-DRIVE pathway cannot show its effect when the drive would not have fired anyway). The load-bearing + lesion-collapse proof above is therefore demonstrated on `salience()`'s general input range (any `n >= 1`), proving the wiring is genuine and mechanistically identical to the other two consumers, while the SPECIFIC pre-existing entry-gate anchor this project's `decide_action()` currently reaches is a floor case where intact and lesioned are not distinguishable. This is a property of the (unmodified) existing entry-gate design, not a defect in the wiring.

### (D) Consumer 3 -- `value_choice_production_organ::default_context_fn()`
Three candidates at recency ratios `[0.0, 0.5, 1.0]`:

| arm | engagement values | spread |
|---|---|---|
| OFF (exact, unchanged formula) | `[0.0, 0.5, 1.0]` on every seed | 1.0 |
| ON | `[0.0, 0.27-0.31, 1.01-1.06]` <!--derived--> | 1.01 - 1.06 <!--derived--> |
| ON + LESION | `[~0, ~0-0.025, ~0-0.025]` <!--derived--> | **0.000 - 0.025 <!--derived-->** |

`attributable_to` (`tools.lab`): **99.1% <!--derived--> mean (97.6% - 100.0% <!--derived--> range)** of the ON-arm spread is attributable to the shared drive pathway. Under lesion the cross-candidate GRADIENT the critic needs to be decisive collapses almost entirely -- qualitatively the same effect `ValueChoiceProductionOrgan.choose()`'s own pre-existing `BRAIN_VALUE_CHOICE_LESION` mean-pin lesion produces (a flat/near-flat fed value -> the WTA declines -> the turn reverts to abstain), now demonstrated one layer upstream, at the context the critic is handed.

## The seed-42 plumbing proof -- through the REAL production entry points (artifact `plumbing_seed42.json`)

`bg_action_selection_production_organ.decide_action("")` and `value_choice_production_organ.ValueChoiceProductionOrgan.choose()` (the REAL trained striosome_value critic, `value_train_trials=40`, the RANK-1 GO's own default -- build+train took 267.8s <!--derived--> on CPU, the reason this proof runs once, not 6x) were run end-to-end OFF / ON / ON+LESION at seed 42.

**BG decide_action("").** All three arms reach the SAME categorical decision (`STAY_SILENT`), consistent with -- not contradicting -- the honest floor-case scope named in (C) above: `silent_salience` reads `1.0` (OFF), `1.0` (ON), `0.9986` <!--derived--> (LESION) -- a genuine but tiny spiking-mediated difference at this floor anchor, and the exact spike-timing of the commit differs (`decision_step` 88 / 113 / 90 <!--derived-->) even though the winning channel does not. This is the predicted result, not a surprise: there is no novelty to lesion away at `raw=0`.

**value_choice's REAL trained critic -- the strongest result in this de-risk.** Three candidates (`cat`, `ball`, `shoe`) at recency `[0.0, 0.5, 1.0]`:

| arm | engagement fed to the critic | learned V (Hz) | fed spread (Hz) | WTA margin | **commit** |
|---|---|---|---|---|---|
| OFF | `[0.0, 0.5, 1.0]` | `[17.78, 28.75, 32.50]` | 14.72 | 4.0 | **shoe** |
| ON | `[0.013, 0.333, 1.086]` <!--derived--> | `[20.28, 18.06, 31.25]` <!--derived--> | 13.19 | 8.0 | **shoe** |
| ON + LESION | `[0.006, 0.0, 0.0]` <!--derived--> | `[15.97, 16.81, 19.44]` <!--derived--> | **3.47** <!--derived--> | 0.0 | **cat** |

ON reaches the SAME final commit as OFF here (`shoe`), but the learned V's the critic reads clearly differ (candidate `ball`'s V drops from 28.75 to 18.06 Hz -- the shared organ's nonlinear response to `e=0.333` <!--derived--> vs the bare `0.5`), so the wiring is demonstrably load-bearing on the REAL critic's readout, not merely on an upstream scalar that happens not to matter downstream. Under the LESION, severing the shared afferent does not just shrink the numbers -- it **collapses the fed-spread by 74% (13.19 -> 3.47 Hz) and FLIPS the final commit from `shoe` to `cat`**, exactly the same qualitative signature (a flattened value gradient reverting the decision) `ValueChoiceProductionOrgan`'s OWN pre-existing `BRAIN_VALUE_CHOICE_LESION` mean-pin control produces, now caused one layer upstream by severing the shared spiking afferent rather than pinning V to its mean directly. All three arms remain `decisive` (`fed_spread >= v_margin_hz=2.0` throughout), so this is a confident re-commit to a DIFFERENT patient, not a collapse into indecision.

## Anti-cheats
- **Additive + reversible.** `BRAIN_SHARED_SALIENCE` unset (default) -> every consumer's pre-existing host-arithmetic branch runs UNCHANGED; `da_mode_drives_chat`'s trace carries no `shared_salience` key at all (verified: `'shared_salience' not in info`, not merely a `None` value) and `bg_action_selection`/`value_choice`'s OFF-arm outputs are EXACT matches to the pre-existing formula, computed independently in the test from source rather than a hand-picked literal.
- **One flag, three sites, one root cause fixed.** Because da-mode-drives-response, da-gated-encoding and da-gated-curiosity all read the SAME `chat._last_da_drives["da_level"]`, retiring the ONE host formula at `engagement_of()` is sufficient to change the input all three depend on; this finding demonstrates that at the `da_mode` consumer directly (B above) and relies on the ALREADY-established fact (2026-08-19/2026-08-21 findings) that da-gated-encoding and da-gated-curiosity read the identical `da_level` field.
- **Load-bearing, not cosmetic.** Every consumer's returned value measurably differs ON vs OFF (B, C, D), and independently VANISHES toward a shared near-floor under the SAME lesion (severing `curiosity_excit_sensitivity`), attributed via `tools.lab.attributable_to` rather than merely reported alongside its control (97-100% <!--derived--> of the measured span/spread is attributable to the drive pathway across every check).
- **Never crashes a turn.** `shared_salience_afferent.read_salience()` catches every exception and degrades to a neutral, input-independent 0.5 (the same "never crash a turn" contract every sibling coupling in this codebase follows).

## Honest scope, terms, and open questions (per `docs/TERMS.md`)
- **NOT "closed" / NOT "on-by-default" / NOT "scaffold-retired".** This is a de-risked, WIRED (reachable from `/api/brain-chat` via the pre-existing, unmodified calls into these 3 files -- no `webapp/server.py` edit was needed), default-OFF coupling. The three host formulas this rank-4 item targets (`engagement_of`, `bg_action_selection.salience`'s bare arithmetic, `default_context_fn`'s bare recency ratio) all REMAIN the production default until `BRAIN_SHARED_SALIENCE` is flipped on; retirement is enabled, not yet executed by default. A default-ON flip needs its own no-regression soak on the live production default (mirroring how da-gated-curiosity/da-gated-encoding were flipped only after a dedicated soak), which this de-risk does not attempt.
- **The bg-action-selection floor-case residual** (honest scope C above) means a default-ON flip of THIS consumer specifically would change the exact numeric bias fed to the BG race only marginally at the one currently-reachable anchor; the flip's main effect there would be replacing hardcoded literals with a genuine (if near-floor) spiking read, not a behavioral shift. This is disclosed, not hidden.
- **The raw-scalar computations remain host** (message token-novelty/richness, content-token counting, fact-storage-order bookkeeping) -- each is a declared sensory/environment/episodic-memory-provenance boundary, not a cognitive computation, exactly as the SVO parser and vision percept are. What retires is the arithmetic that used to carry that scalar STRAIGHT to the consumer with zero neurons mediating it.
- **FUNCTIONAL correlate, NOT phenomenal** -- this reads + reports a spiking salience/novelty CORRELATE; it claims no subjective wanting or felt salience.
- **CO-RESIDENT.** The shared organ is its own small bridge (the DR-1 curiosity substrate), built once per process and reused across all 3 consumer sites in-process; it is not yet merged onto the single recall bridge (rides the one-brain consolidation burn-down, like the curiosity/affect/surprise/metacog organs already do).

## Files
`research/runners/curiosity_production_organ.py` (+`salience_of`, additive), `research/runners/shared_salience_afferent.py` (new), `webapp/da_mode_drives_chat.py`, `research/runners/bg_action_selection_production_organ.py`, `research/runners/value_choice_production_organ.py` (each +flag-gated branch), `research/runners/_shared_salience_afferent_derisk.py` (new verify runner). Artifacts: `research/findings/raw/_shared_salience_afferent/verify_6seed.json`, `research/findings/raw/_shared_salience_afferent/plumbing_seed42.json`.

## Citations
- Scaffold-retirement map (this de-risk's mandate): `research/coordination/scaffold_retirement_backlog.md` rank-4.
- Curiosity ASK-pool spiking crave-drive (reused verbatim): `research/findings/2026-07-30-lane-B-curiosity-DR1-onbridge-6seed-GO.md`, `research/findings/2026-07-23-DR1-curiosity-inversion-ONBRIDGE-spiking.md`, runner `research/runners/_curiosity_seek_learn_onbridge_derisk.py`.
- The DA-mode SNc read + its 3 downstream default-on consumers (reused verbatim): `research/findings/2026-08-19-neuromod-spiking-da-mode-GO.md`, `research/findings/2026-08-19-da-mode-drives-chat-load-bearing-GO.md`, `research/findings/2026-08-21-da-gated-curiosity-threshold-wired-GO.md` (the identical seed-waiver scoping this finding reuses), `research/findings/2026-08-21-da-gated-encoding-wired-into-chat-GO.md`.
- BG action-selection substrate (reused verbatim): `research/findings/2026-08-03-neural-vocal-selector-gateA-v2-4seed-GO.md`.
- Value-choice critic (reused verbatim, not modified): `research/findings/2026-07-23-value-critic-closure-RANK1-GO.md`.
- Attribution discipline: `tools/lab.py::attributable_to` (the gap#5 lesson).
