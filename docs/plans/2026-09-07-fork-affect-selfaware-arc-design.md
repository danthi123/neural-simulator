---
type: plan
status: live
date: 2026-09-07
---
# AGI-fork pivot — emergent AFFECT + SELF-AWARENESS arc (design)

> Owner steer 2026-09-07: the fork advances the MAIN project goals (genuine conversation, affective world-model, emotion, self-awareness, curiosity) with LOOSER biology. Design produced by workflow wkqqhz90l (6 agents). The main project ALREADY has affect/self-awareness/curiosity WIRED + on-by-default, but all three are driven by a HOST-PARSED-TEXT scalar and carry the identical BLOCKED:self-model-reward-residual tag. The fork substrate NATIVELY computes the missing substrate-native reward/prediction-error — that is the opening.

## GROUND 1 — main-project state (verify-first, do not duplicate)

# Main-project (spiking) state on AFFECT / SELF-AWARENESS / CURIOSITY

Ground-truthed from `/home/dant123/Projects/sim`: RAG search, git log, `research/findings/`, and `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`. Headline: **all three faculties already exist as wired, on-by-default, lesion-load-bearing organs in the live `/api/brain-chat` turn** — this is much further along than "de-risked." But every one of them is driven by a **host-derived scalar extracted from parsed text** (a language-comprehension boundary), never from the brain's own internally-generated prediction error/reward. That single fact is the fork's real opening.

## 1. AFFECT / emotional world-model — EXISTS, wired, on-by-default

- **Files/commits:** `research/runners/affect_production_organ.py`, `webapp/affect_drives_chat.py`, `research/runners/worldmodel_production_organ.py`. Findings: `2026-08-12-GateB-worldmodel-affective-forward-model-production-chat.md`, `2026-08-12-GateB-affect-colors-production-chat.md`, `2026-08-19-affect-drives-chat-load-bearing-GO.md`, `2026-08-19-graded-affect-attractor-GO.md`.
- **Read-out / metric:** a 2-channel spiking predictive-coding valence forward model — `state -> pred_{pos,neg}` — makes next-turn affect QUERYABLE ("what do you expect" -> two-pool spike-rate read, margin e.g. +411 Hz) and fires a genuinely-spiking surprise unit (`cp_firing_states[surprise]`) on prediction violation (24.3 Hz vs 0 Hz threshold 12.2). A separate co-resident "graded affect ladder" (Koulakov robust integrator, valence x arousal, Pearson +0.97/+0.95) colors tone/forthcomingness. All lesion-load-bearing (zeroing the transition/embodiment collapses the effect; content stays byte-identical).
- **Already solved — don't repeat:** the *prediction-error-drives-a-queryable-affect-forecast* mechanism (E2) is exactly the fork's "affect = reward + PE" idea, already built and wired on the discrete/SVO substrate. Don't re-derive this pattern from scratch — the lesion/query/violation-notice design is reusable as a template.
- **OPEN gap:** the valence APPRAISAL feeding all of this is `webapp/server.py`'s host SVO-parser reading the message text into a DR-2 learned-but-numpy distributional-valence lookup (Warriner-seeded label-propagation, r~0.81 held-out) — **not** derived from the brain's own prediction error. Ledger tag: `retire_status: BLOCKED:self-model-reward-residual` appears repeatedly (`da-mode`, `affect-drives-response`, `da-gated-curiosity-threshold`) — *"no standalone reward/value neuromodulator drives the live chat turn from the brain's own sensory stream."* This is the fork's opening: replace "host parses text -> valence scalar" with "substrate's own next-latent prediction error/reward head -> valence," which is the loosened-biology, tractable-function version of exactly what's missing here.

## 2. SELF-AWARENESS / metacognition & honesty — EXISTS, wired, on-by-default

- **Files/commits:** `research/runners/metacog_production_organ.py` (E1), `research/runners/self_schema_production_organ.py` (DR-3 authorship), `research/runners/source_provenance_honesty.py` (board #129, perceived-vs-generated). Findings: `2026-08-12-GateB-metacog-confidence-readout-production-chat.md`, `2026-08-26-DR3-self-schema-authorship-production-wirein-GO.md`, `2026-08-25-129-source-provenance-honesty-production-wirein-GO.md`; most recent refinement `b0d3c916a` (2026-09-06, self-schema authorship 5/6->6/6 via DG pattern-separation).
- **Read-out / metric:** a spiking workspace WTA "balance of evidence" — `|rate(asm1)-rate(asm0)|` off `cp_firing_states` — reads confidence of the answer about to be given; below a calibrated threshold it PREPENDS an honest hedge ("my decision-margin reads this as low-confidence..."). A dissociable authorship monitor (self vs. heard) does the same for generated hypotheses. Lesion (`BRAIN_METACOG_LESION=1`) flips a confident answer to hedged; flag-off is byte-identical. This is **exactly** the mission's "honest functional read-out" template — never a phenomenal claim, stated in every finding's closing line.
- **Already solved — don't repeat:** the whole "novelty/confidence -> honest hedge, never a phenomenal claim" framing, the lesion-load-bearing verification pattern, and the moat-safe wiring convention (qualify, never fabricate/flip) are all validated and reusable wholesale as the honesty-boundary deliverable pattern.
- **OPEN gap:** the EVIDENCE fed into the balance-of-evidence read is the host's own role-decode parse confidence (a *component* of confidence, not a full recall-vs-alternatives margin), and it is explicitly **NOT type-1/type-2 dissociable** — the architecturally-dissociable comparator (`margin_abs`) is seed-fragile and named as the still-open next rung (`2026-09-05-metacog-accumulation-to-bound-6seed-PARTIAL.md`, `2026-09-06-metacog-accumulation-to-bound-6seed-PARTIAL.md` — a sequential-sampling accumulate-to-bound confidence model, still PARTIAL as of yesterday). Same pattern as affect: the evidence signal is host-parsed, not substrate-native. This is precisely where the fork's own JEPA prediction-error magnitude ("this reads as novel, so I'm uncertain") could substitute for the host-derived evidence scalar — a genuinely substrate-native confidence readout, closing the named residual rather than duplicating it.

## 3. CURIOSITY — EXISTS, wired, on-by-default (crave-drive only)

- **Files/commits:** `research/runners/curiosity_production_organ.py`, `webapp/da_curiosity_drives_chat.py`. Findings: `2026-07-23-DR1-curiosity-inversion-6seed-GO.md` (the base mechanism), `2026-08-12-GateB-curiosity-followup-production-chat.md` (wired), `2026-09-05-rank10-curiosity-graded-novelty-familiarity-scaffold-derisk-GO.md` (graded upgrade, default-OFF), `2026-08-21-da-gated-curiosity-threshold-wired-GO.md`.
- **Read-out / metric:** on an ABSTAIN (the brain holds no answer -> a novelty scalar) drives the `from_novelty` neuromodulator -> excitability on a spiking ASK pool; wanting is read directly off `cp_firing_states[ask]` (corr(gap, want)=+0.996, lesion->0 asks). Above threshold, the brain APPENDS an honest follow-up question ("I haven't learned about X yet..."). DA/engagement (the brain's own spiking SNc tonic-DA level) gates the crave threshold (Aston-Jones-Cohen vigor account), also wired default-ON.
- **Already solved — don't repeat:** the curiosity-as-honest-follow-up-question pattern (crave, don't confabulate) and its DA-gating are done; the "binary novelty was a host constant" problem was *already identified and partially fixed* by the Bogacz-Brown graded familiarity gate (catalog D.04, anti-Hebbian projector) — reused from `2026-06-11-familiarity-gate-v320-GO.md`.
- **OPEN gap:** (1) the graded-novelty fix is de-risked GO but still **default-OFF** — not yet flipped into production; (2) novelty there is *lexical cue-fidelity* (clean vs. noisy draw of the same word-hash), not semantic/conceptual novelty, and is capacity-bounded (2×D=512 orthogonal directions); (3) the learning-progress SELECTOR (which of several topics to ask about) and the noisy-TV VETO remain unwired host formulas (LP is a numpy EMA proxy, seed-fragile on-substrate; the veto survives a critic lesion, i.e. it's not load-bearing) — same `BLOCKED:self-model-reward-residual` tag. This is the second and clearest fork opportunity: a JEPA-native prediction-error-based novelty signal is conceptually the exact upgrade this residual is waiting for, and would also naturally supply the LP-selector's missing substrate-native learning-progress signal (LP = shrinking prediction error over time on a topic/state).

## The one thread connecting all three opens

Every "BLOCKED" tag across affect (`affect-drives-response`, `da-mode`), curiosity (`da-gated-curiosity-threshold`), and metacog (`margin_abs`/accumulate-to-bound) points at the **same missing piece**: a genuinely substrate-native reward/prediction-error signal, as opposed to a host parser turning message text into a scalar. The main project has been circling this since 2026-08-19 without closing it because its substrate (RF phasor / SVO composer) doesn't compute prediction error as a primitive. **The fork's `sim/pcs_substrate.py` does** (JEPA next-latent prediction error + reward/value/SR heads, already validated 6/6 on the place-code emergence result). The board's own 07:20 anchor has already reasoned to this same conclusion independently (`"novelty/familiarity/confidence read-out from prediction-error"` / `"AFFECT/valence = reward + PE"`) — **this is confirmed correct and not yet duplicated anywhere in the main branch**; no existing main-project mechanism computes affect or curiosity-novelty from an intrinsic prediction-error signal. The load-bearing bar to reuse from the main project: vary state -> response differs, lesion -> difference vanishes (exactly the pattern every finding above already demonstrates) — apply the same discipline to the new PE-based probes, and use split-half stability / cyclic-shift nulls rather than an i.i.d. shuffle (the caveat already logged from the spatial-code work applies identically here).

## GROUND 2 — substrate affordances (signal->faculty map)

# Signal → Faculty Map: PCS Substrate Affordances for Affect / Self-Awareness / Curiosity

Read in full: `sim/pcs_substrate.py` (1392 lines) and `research/runners/_fork_pcs_emergence_derisk.py` (1428 lines), plus `research/runners/fork_pcs_world.py`'s interoceptive/reward code and its `TwoPoolDrive` import (`research/runners/_homeostatic_drive_rl_cheap_first_probe.py:59`).

## Signal inventory: what the substrate computes each step, and where

| Signal | Computed | Stored/exposed? | Frequency |
|---|---|---|---|
| Interoceptive drive `d_t` = [drive, deficit, energy, 1.0] | `fork_pcs_world.py:402-406` (`ForkPCSWorld.drive_afferent`), driven by `TwoPoolDrive` (AgRP/POMC push-pull, `_homeostatic_drive_rl_cheap_first_probe.py:59-77`) | Fed into recurrence via `P["W_d"] @ d_t` (`pcs_substrate.py:464` rate / `:482` spike) | every `observe()` step |
| Reward `r_t` | host, drive-reduction (`fork_pcs_world.py:469`) | passed to `learn(reward)` | every step |
| **Reward-prediction error** `rhat_t - r_t` | `pcs_substrate.py:731-736` inside `_window_forward` | only aggregated into `rl` → combined `loss`; not decomposed/exposed per-step | every TBPTT window (T=18) |
| **JEPA next-latent prediction error** `ehat_t - z_t` | `pcs_substrate.py:719-728` (`diff`, line 725) | only aggregated into `jl`; `jepa_terms` list (line 727) holds the raw per-t tensors but only inside the window's `cache`, never returned/persisted | every TBPTT window |
| **Actor-critic advantage** `adv = r_int - V(h_t)` | `pcs_substrate.py:623,628` in `learn()` | **local variable only** — computed, used, discarded; never stored on `self` | every step (when `value_weight>0`) |
| Value-head MC-return residual | `pcs_substrate.py:769-770` inside `_window_forward` | aggregated into `val_loss`; not exposed | every TBPTT window |
| **Learning-progress EMAs** `loss_fast`, `loss_slow`, `_last_lp = relu(loss_slow-loss_fast)` | `pcs_substrate.py:398-401` (init), `:636-645` (`_update_learning_progress`) | **stored on `self`**, persists across steps | updated every TBPTT window, **read every `learn()` call** |
| Policy distribution `self._last_probs` (entropy/confidence) | `pcs_substrate.py:580-594` in `act()` | stored, but overwritten next step; used only for the entropy-bonus gradient (`:656-658`) | every step |
| `self.last_pred_loss` (combined JEPA+reward+var[+value+sr+aux]) | `pcs_substrate.py:412` init, `:610` set | stored, read by the runner as a diagnostic and by `_update_learning_progress` | every TBPTT window |
| `eval_predictive_loss()` | `pcs_substrate.py:1113-1136` | offline/held-out evaluator, not a live per-step signal — used by the runner as the `object` faculty's behavioral metric | on-demand only |

## Faculty map

### 1. Affect / valence — signal: reward combined with prediction error
- **Best-grounded candidate**: the actor-critic **advantage** `adv = r_int - V(h_t)` (`pcs_substrate.py:628`), where `r_int = reward + curiosity_beta * LP` (`:623`). This is exactly "better/worse than expected," computed live, every step.
- **Complementary interoceptive half** (per the multimodal-affect memory: interoception = "have"): `d_t` from `TwoPoolDrive` is already an input every step (`fork_pcs_world.py:402-406` → `pcs_substrate.py:464/482`), so hedonic tone could combine `adv`'s sign/magnitude with the drive/deficit components of `d_t` for an arousal-like intensity term.
- **Raw reward-RPE** (`rhat_t - r_t`, `:731-736`) is the more classical dopamine-style signal but is currently **gradient-only** — it shapes `w_r`/recurrence weights slowly (`:861-866`) and is never read live.
- **Load-bearing status**: `adv` is **already load-bearing** — it directly forms `dlogits` in `_policy_update` (`:654`) every step an action was taken, i.e. it already drives the policy gradient. What's missing is *persistence and exposure*: it's a local variable, never written to `self`, so nothing downstream (a self-report, a decoder, a display) can currently read "how did that go." Turning it into a genuine affect faculty is plumbing (store `self.last_valence = adv`), not new mechanism. The raw RPE, by contrast, would need a live per-step reward-head evaluation added to `act()`/`learn()` (cheap — `w_r` already exists) before it's read-only no more.

### 2. Self-awareness / familiarity — signal: prediction error / confidence
- **Best-grounded candidate**: the `loss_fast`/`loss_slow` EMA pair (`pcs_substrate.py:398-401`, updated `:638-643`). `loss_fast` tracks current surprise, `loss_slow` tracks baseline surprise — a `loss_fast` vs `loss_slow` comparison is literally a familiarity readout: dropping below baseline = "getting more predictable/familiar," rising above = "getting more novel/uncertain." This is the *same* pair `_update_learning_progress` already computes (`LP = relu(loss_slow - loss_fast)`, `:645`), just currently one-sided (the `relu` throws away the "getting worse" half).
- **Raw perceptual novelty**: the undecomposed JEPA residual (`:719-728`) is the purest "how wrong was my expectation of the next view" signal but is presently invisible outside the window's local `cache`.
- **Secondary/different flavor — decision confidence**: `self._last_probs` policy entropy (computed every `act()`, `:580-594`) is a *metacognitive-confidence* signal distinct from predictive surprise (confidence about what-to-do vs. confidence about what-will-happen).
- **Load-bearing status**: currently **read-only for this purpose**. `loss_fast`/`loss_slow`/`last_pred_loss` only reach behavior indirectly through `LP → curiosity_beta → r_int → adv → policy gradient` — a real but coarse, one-step-removed, globally-scalar path (see next section). There is **no direct path** from "I am currently surprised" to a self-report or a confidence-gated action/utterance. Building one (e.g., feeding the surprise EMA back in as an additional recurrent input alongside `d_t`, so the core's own state carries "I am surprised" and that state causally shapes the next output) would close the loop the same way `d_t` already does for interoception — the substrate has the pieces but this specific feedback path does not exist yet. **Anti-pitfall**: per the steer's own caution, merely logging/decoding the raw JEPA residual without feeding it back would repeat the place-cell arc's decodable-but-not-load-bearing trap.

### 3. Curiosity / novelty-seeking — signal: learning progress (already implemented)
- This is the **most mature** of the three. `PCSConfig.curiosity_beta` (`:201-202`, default 0.1) weights `self._last_lp` into `r_int` (`:623`), which forms `adv` (`:628`) and trains the policy every step (`_policy_update`, `:647-662`). This is a genuine Oudeyer/Kaplan-style learning-progress intrinsic-motivation signal, derived purely from the substrate's own prediction dynamics (no host-designed novelty heuristic), and it is **demonstrably load-bearing**: the derisk runner's own pre-registered gate tests exactly this — curiosity-policy grid coverage vs. random-policy coverage, required ≥1.5× (`_fork_pcs_emergence_derisk.py:87` `CURIOSITY_RATIO`, `:492-497` `_coverage`).
- **Known limitation** (the substrate's own comment, `pcs_substrate.py:202`): LP is **global** (one scalar for the whole recent window), not state-conditioned — "a large global LP credited to arbitrary actions collapses the policy. Per-state LP is a next rung." So today's curiosity says "I am currently in a generally-improving-prediction regime," not "*this particular state* is novel, go there." The raw material for a state-conditioned version already exists (`jepa_terms`, the per-t residuals inside a window) and would only require tracking per-state (or per-`h_t`-region) surprise EMAs instead of one global pair — no new mechanism class needed, just finer-grained bookkeeping of a signal already computed.

## Summary table

| Faculty | Primary substrate signal | Code hook | Load-bearing today? |
|---|---|---|---|
| Affect/valence | actor-critic advantage `adv` (+ interoceptive `d_t`) | `pcs_substrate.py:623,628`; `fork_pcs_world.py:402-406` | **Yes** (drives policy every step) — needs persistence/exposure, not new mechanism |
| Self-awareness/familiarity | `loss_fast` vs `loss_slow` EMAs (JEPA residual) | `pcs_substrate.py:398-401,636-645`; raw residual `:719-728` | **No** — read-only except via the LP path; needs a feedback loop into `h_t` or output, analogous to how `d_t` already feeds in |
| Curiosity/novelty-seeking | learning progress `_last_lp` | `pcs_substrate.py:201-202,636-645,623`; validated by `_fork_pcs_emergence_derisk.py:87,492-497` | **Yes, already implemented and gated** — the one gap is state-conditioning (currently global, not per-context) |

Note: I did not propose or design any navigation-related extension — the SR/value-head residuals (`:499-502`, `:495-497`) share the same TD-error mechanism family and would generalize the same way, but per the steer, navigation is closed and out of scope here.

## DESIGN — emergent AFFECT/valence (FIRST target)

# Emergent Affect / Valence — Build-Ready Experiment Design

Grounded by reading `sim/pcs_substrate.py` (full) and `research/runners/_fork_pcs_emergence_derisk.py` + `fork_pcs_world.py` (targeted) in `/home/dant123/Projects/sim-agi-fork`. Reuses that file's proven anti-hollow machinery (`_ridge_weights`, `_r2_with_floors`/`_beats_floors`, `_lesion_dependency`, `_place_cell_metrics`) rather than reinventing it, and deviates from it in exactly the one place the owner flagged as unsound (the i.i.d. shuffle null).

## 1. The mechanism: two valence signals already half-built, currently discarded

Confirmed by reading the code directly (not assumed from the memory doc):

- **Reward-head RPE** — `self._reward_head(h_t, P)` (`pcs_substrate.py:492-493`) computes `rhat = w_r@h_t + b_r` every step. It is currently used **only inside the TBPTT window** (`_window_forward:701`, `_window_backward:861-865`) to form the gradient-only `rl` loss term. `w_r`/`b_r` are allocated **unconditionally** in `__init__` (`:318-319`) — this signal exists for every run that has ever been executed, no config change needed.
- **Actor-critic advantage** — `adv = r_int - self._last_value` (`learn():628`, only when `value_weight>0`) is computed and immediately used to form `dlogits` for the policy gradient (`_policy_update:654`), then **discarded** — never written to `self`. This is *already* load-bearing on the policy by construction; the gap is persistence, not mechanism.

**Design move**: expose both as first-class read-outs, with zero new parameters and zero new loss terms (so training dynamics are byte-identical — only added `self.` bookkeeping):

```python
# in act(), immediately after the existing self._last_value line (pcs_substrate.py:598):
self._last_rhat = float(to_host(self._reward_head(h_t, self.P)))

# in learn(), immediately after adv is computed (pcs_substrate.py:628) and after the
# existing reward-registration at the top of learn() (pcs_substrate.py:605):
if self._last_rhat is not None:
    self.last_valence_rpe = float(reward) - self._last_rhat   # signal A: always available
if self._last_value is not None:
    self.last_valence_adv = adv                                 # signal B: value_weight>0 only
```
`self.last_valence_rpe` / `self.last_valence_adv` initialize to `None`/`0.0` in `__init__` next to `self._last_lp`. This is the entire "mechanism" change — the emergence claim is about the *shared recurrent core*, not these two scalars: the population of `h_t` units whose activity the heads read is what must be shown to carry and drive valence, via decode+lesion below.

Sign convention (dopamine-RPE-consistent, Schultz 1997): **positive = better than expected, negative = worse than expected.** Note the base world's reward is `max(0, drive_before-drive_after)+shaping` (`fork_pcs_world.py:469`) — floored at zero — so signal A can already go negative (an anticipated-but-missed reward), but genuinely large negative *external* events don't exist yet. Section 2 fixes that.

## 2. World extension: a genuine bipolar hedonic event (default OFF, byte-identical when off)

Reading `fork_pcs_world.py:424-474`, reward is currently one-sided. Add to `WorldConfig`:

```python
hazard_enabled: bool = False
n_hazards: int = 1
hazard_cost: float = 0.4          # energy penalty on contact, same scale as eat_refill=0.5
hazard_type_idx: int = 4          # 5th, disjoint value in the EXISTING objects={(x,y):type_idx} channel
hazard_cooldown: int = 20         # steps of immunity after a hit (avoid degenerate repeat-punish loops)
```

In `reset()`: when enabled, place `n_hazards` fixed cells (same pattern as `self.larder`), rendered through the **already-generic** `objects` dict (`render_egocentric_crop`, `_draw_oriented_bar` — zero new rendering code, just one more `type_idx`). In `step()` (`fork_pcs_world.py:424`), after the existing `ate` block, add an **additive** penalty so the `ate`/base-reward line is untouched (preserves byte-identical when `hazard_enabled=False`):

```python
hit_hazard = cfg.hazard_enabled and self._hazard_cooldown_left == 0 and self.agent in self.hazard_cells
if hit_hazard:
    self.energy = max(0.0, self.energy - cfg.hazard_cost)
    self._hazard_cooldown_left = cfg.hazard_cooldown
# drive_after recomputed as today (line 464), now reflecting the extra depletion when hit
...
reward = cfg.reward_scale * max(0.0, drive_before - drive_after) + shaping - (cfg.hazard_scale if hit_hazard else 0.0)
```
Record `hit_hazard` in `info`/`labels()` — this is the **ground-truth negative-event label** the sign-correctness anti-cheat (G4 below) needs. This mirrors exactly the existing "byte-identical off-path" discipline used for `value_weight`/`sr_weight`/`aux_loc_weight`/`nav_required`.

## 3. Read-out (functional, honest)

Two live scalars, `last_valence_rpe` and `last_valence_adv`, reported as: *"the substrate's reward-prediction-error / advantage read-out is negative/positive here"* — never "the agent feels bad/good." Same phrasing discipline as the main-project's metacog hedge ("my decision-margin reads this as low-confidence") per the honesty-boundary precedent already validated there.

## 4. The sound metric — split-half stability + cyclic-shift null, NOT i.i.d. shuffle

`_place_cell_metrics` (`_fork_pcs_emergence_derisk.py:763-882`) is the house-proven pattern (Skaggs SI + split-half rate-map correlation + shuffle-null), reused for object-type tuning by `_categorical_tuning_metrics` (`:921+`). Its shuffle null is described in-file as permuting the label against firing while holding "occupancy `p_i` and overall `lam`" invariant — a **row-independent (i.i.d.) permutation**. That is exactly the invalid-null pattern the steer names, and it matters *here specifically* because hazard/eat events are temporally clustered (a hazard hit depresses drive/RPE for several subsequent steps as the deficit persists) — an i.i.d. shuffle destroys that autocorrelation on both sides and inflates apparent significance.

**Required deviation** (new small helper, `_cyclic_shift_null`, sibling to the existing shuffle code): for each of ~100 draws, circularly roll the valence-label time series by a random offset (autocorrelation and marginal distribution of both series preserved; only the cross-alignment destroyed), recompute the same tuning-curve statistic, and take the real value's percentile against *this* null instead.

Two-tier metric, both required (mirrors `_r2_with_floors`/`_beats_floors` + `_place_cell_metrics` side-by-side, exactly as the file already does for place):

- **(a) Population presence** — ridge-decode `valence_t` (both signals) from `h_t`, held-out R², beating `floor_untrained` / `floor_rawv1` / `floor_shuffle` by `FLOOR_MARGIN` (reuses `_ridge_weights`/`_r2_with_floors`/`_beats_floors` verbatim — this floor usage is population-level "does a decoder fit at all," the same category the file already treats as sound for value/position, not the per-unit significance test the steer's caveat targets).
- **(b) Per-unit soundness** — bin valence into 5 quantile bins, build a per-unit tuning curve (mean rate per bin) on temporal first-half vs second-half of the probe rollout (reuse `_place_cell_metrics`'s split-half machinery, retargeted from spatial bins to valence bins), correlate the two halves. A unit is valence-selective iff split-half correlation `> 0.30` (reuse `PLACE_SI_STABILITY_THRESH`) **and** its real SI exceeds the **cyclic-shift** null's 95th percentile. Report trained-core stability vs untrained-core stability, expecting a gap of the same order as the already-validated place result (~0.8 vs ~0.3).

## 5. The load-bearing test — necessity AND sufficiency

**Necessity (lesion, primary/must-pass)** — directly reuses `_lesion_dependency` (`:428-455`) and the `BEHAV_LESION_RATIO=1.5` bar unmodified. Select valence units by ridge-importance against `last_valence_adv` (same `_mask_from_imp` pattern used for place/object/permanence/value). Behavioral metric: **learned hazard-avoidance rate** — `P(step toward a hazard | agent within radius r)` on a frozen probe rollout, compared against (i) the pre-lesion trained rate and (ii) an equal-size **random-unit lesion** baseline (the file's existing shared control). Gate: valence-unit lesion degrades avoidance ≥1.5× the random-unit lesion's degradation.

**Sufficiency (injection, the novel piece this experiment adds — necessary because lesion alone still permits "load-bearing for some unrelated reason")** — new substrate method, same discipline as `set_lesion_mask`:

```python
def set_activation_clamp(self, mask, values):
    """mask: bool (n_hidden,); values: float (n_hidden,) applied where mask is True,
    to h_t everywhere it is read (heads, recurrence, policy) — same injection point as
    _apply_lesion. None clears it (default OFF, byte-identical)."""
```
Procedure: from the probe rollout, record the mean activation pattern of the top-k valence-important units during real negative-RPE steps (`pattern_neg`) and real positive-RPE steps (`pattern_pos`). On a **fresh, neutral** rollout (no hazard/food nearby), clamp those units to `pattern_neg` vs `pattern_pos` vs a random-unit-matched clamp immediately before `act()`, and measure the shift in the action distribution (entropy / avoidance-consistent-action probability). A genuinely load-bearing (not decorative) population shows a `pattern_neg`-clamp shift toward avoidance-consistent actions that a random-unit clamp does not reproduce.

**Anti-cheats (all required)**:
1. Random-unit-matched lesion control (§5, already the house pattern).
2. Decode-instrument vs behavioral-saliency-instrument dissociation — reuse `lesion_mode="both"`'s Jaccard-overlap check (`:477-483`) verbatim, retargeted to valence.
3. **Sign-correctness against ground truth** (new, enabled by §2's hazard label): decoded valence must be reliably negative at real `hit_hazard` steps and reliably positive at real `ate` steps (bootstrap CI excluding 0, both directions) — an independent check the decode isn't fitting noise that happens to clear the floors.
4. Core-lesion collapse — reuse `_core_lesion_presence` (`:704`) and `CORE_LESION_COLLAPSE=0.50` verbatim: zeroing `W_h` must collapse the valence decode toward floor, proving the signal lives in the trained recurrence, not a pass-through of `W_e`/heads alone.
5. Untrained-core reservoir control on the lesion metric too, not just the decode floor — a spurious-but-decodable random core should show lesion-ratio ~1.0× (no dependency), not ≥1.5×.

## 6. Protocol

New file `research/runners/_fork_pcs_affect_derisk.py`, sibling to (imports from, does not modify) `_fork_pcs_emergence_derisk.py`. Config: `units="rate"` (primary emergence arm), `value_weight=1.0` (for signal B / the actor-critic advantage), `aux_loc_weight=0` (navigation is closed — don't reopen it; keep this experiment orthogonal to position), everything else at the already-tuned-stable defaults (`n_hidden=512`, `alpha=0.2`, `lr=3e-4`, `grad_clip=1.0`, `grad_skip_factor=8.0`, `ema_rate=0.9999`). World: `hazard_enabled=True`, `n_hazards=1`, `hazard_cost=0.4`. Train online, sustained `explore_eps=0.15` (not decayed — need enough hazard *and* eat encounters throughout, matching the file's existing `explore_eps=0.1` probe convention), same step-count scale as `run_nav_ab`/`run_stability_ab` (40-60k). Then `sub.freeze()`, run a probe rollout (n≈3000, `explore_eps=0.1`, matching `n_behav` elsewhere) collecting `h_t`, `last_valence_rpe`, `last_valence_adv`, `ate`, `hit_hazard`, `reward`. **6 seeds**: 42/43/44/100/101/102 (house standard), GO requires ≥5/6 passing except the injection test (§7, G6) at ≥4/6 given its novel-plumbing risk.

## 7. Pre-registered GO gate

| # | Test | Bar | Reuses |
|---|---|---|---|
| G1 | Presence | ridge R² beats untrained/raw-V1/shuffle floors + `FLOOR_MARGIN` | `_r2_with_floors`/`_beats_floors` verbatim |
| G2 | Soundness | split-half stability > 0.30 AND real SI > cyclic-shift-null 95th pctile; trained ≥2× untrained | `_place_cell_metrics` pattern + **new** cyclic-shift null |
| G3 | Necessity | valence-unit lesion degrades hazard-avoidance ≥1.5× random-unit lesion | `_lesion_dependency`/`BEHAV_LESION_RATIO` verbatim |
| G4 | Sign-correctness | decoded valence sign matches `hit_hazard`(−)/`ate`(+) ground truth, CI excludes 0 | new, enabled by §2 |
| G5 | Core-dependence | `W_h`-zero lesion collapses decode R² ≥50% | `_core_lesion_presence`/`CORE_LESION_COLLAPSE` verbatim |
| G6 (stretch) | Sufficiency | `pattern_neg` clamp shifts actions toward avoidance ≥1.5× a random-unit clamp | new `set_activation_clamp` |

GO on the core faculty = G1∧G2∧G3∧G4∧G5 (≥5/6 seeds each). G6 reported separately; passing upgrades the claim from "necessary" to "necessary and sufficient."

## 8. Cheapest smoke that would falsify it early (minutes, not hours)

Before the 6-seed GPU run: numpy backend, 1 seed, `n_hidden=128`, ~5k steps, `explore_eps=0.3`, hazard world on.
1. **Wiring check** (~1 min): is `last_valence_adv`/`last_valence_rpe`'s sign correlated *at all* with `hit_hazard`/`ate` (Spearman `|ρ|>0.15`)? If ~0, stop — either the capture-order bug (RPE/adv captured at the wrong point relative to `learn()`'s reward registration) or the hazard penalty is too weak to move `reward` measurably; fix that before running anything expensive.
2. **Decode-exists check** (~5-10 min): run §4(a) ridge+floors once. If R² doesn't even loosely beat `floor_shuffle`, the population-presence claim is dead — don't spend GPU-hours on split-half stability or the lesion/injection battery.

Only after both pass on the 1-seed smoke does the full 6-seed run + G1-G6 battery get queued (GPU lane, `tools/gpu_queue.sh`, per the project's cost-routing discipline).

## Files touched

- `sim/pcs_substrate.py` — add `self.last_valence_rpe`/`self.last_valence_adv` persistence (§1) + `set_activation_clamp` method (§5/G6). No new params, no new loss terms, no config field required for the core signal.
- `research/runners/fork_pcs_world.py` — add `hazard_*` fields to `WorldConfig`, additive penalty in `step()`, hazard cell placement in `reset()` (§2). Default OFF → byte-identical to every existing artifact.
- `research/runners/_fork_pcs_affect_derisk.py` (new) — the experiment runner, importing and reusing `rollout`, `_ridge_weights`, `_r2_with_floors`, `_beats_floors`, `_lesion_dependency`, `_place_cell_metrics`'s split-half scaffolding from `_fork_pcs_emergence_derisk.py`, plus the new `_cyclic_shift_null` helper.

## DESIGN — emergent SELF-AWARENESS/familiarity (SECOND)

# Design: Emergent Familiarity/Confidence Monitor on the PCS Substrate

Grounded directly in `/home/dant123/Projects/sim-agi-fork/sim/pcs_substrate.py` (read in full, 1392 lines) and `/home/dant123/Projects/sim-agi-fork/research/runners/_fork_pcs_emergence_derisk.py` (read in full, 1427 lines). This is the substrate's **6th move**, additive and OFF-by-default, following the exact pattern of the 3rd–5th moves (value/SR/aux-loc heads) already in the file. It does **not** touch navigation reward/objective — position is used only as a diagnostic label, exactly as the existing place/value/permanence probes already do.

## 1. The signal: reuse the JEPA head one step early, not a new mechanism

The substrate already computes, once per TBPTT window, the exact residual `ehat_t - z_{t+1}` (`_window_forward`, lines 719–728) and then throws it away except as a scalar folded into the loss. The fix is exposure + feedback, not a new computation:

At `observe()` time (`pcs_substrate.py:510`), **before** running the recurrence, using the state `h_{t-1}` and action `a_{t-1}` that are already in scope:

```python
z_now    = tanh(W_enc_ema @ v1feat + b_enc_ema)                 # stop-grad encoding of THIS view
ehat_prev = W_pred @ h_prev_masked + W_pred_a @ a_prev + b_pred  # what I predicted, one step ago, I'd see now
phi_raw  = mean((ehat_prev - z_now) ** 2)                        # realized one-step prediction error, NOW
```

This is literally the model's own training residual, checked at the moment of violation, using zero new parameters for the raw signal — `W_pred`/`W_pred_a`/`b_pred`/`W_enc_ema` already exist. This is the strongest possible grounding: the "familiarity monitor" is not a bolt-on decoder, it's the substrate's own predictive-coding error read out live.

Then a per-step fast/slow EMA pair (identical math to `_update_learning_progress`, `pcs_substrate.py:636-645`, just applied per-step instead of per-window):

```python
phi_fast += selfmon_fast * (phi_raw - phi_fast)
phi_slow += selfmon_slow * (phi_raw - phi_slow)
novelty   = max(0, phi_raw - phi_slow) / (phi_slow + eps)   # relative-to-own-baseline surprise
confidence = 1 / (1 + kappa * novelty)
```

Using *relative* surprise (not raw error) is what makes this a familiarity signal rather than an irreducible-noise detector — inherited for free from the same anti-noisy-TV property already validated in the LP formula.

## 2. Closing the loop: feed it back in (biological anchor: interoceptive/error broadcast)

The grounding memo's own diagnosis is exact: `d_t` (drive) already feeds into the recurrence every step (`W_d`, `pcs_substrate.py:464`); nothing analogous exists for the substrate's own error signal. Fix: treat `novelty_t` as a fifth interoceptive-style channel.

**Config addition** (`PCSConfig`, after the `aux_loc_weight` block, `pcs_substrate.py:146`):
```python
selfmon_weight: float = 0.0   # 0 = OFF -> byte-identical (no W_phi, no rng draw, no extra tape field)
selfmon_fast:   float = 0.2
selfmon_slow:   float = 0.01
selfmon_kappa:  float = 2.0   # policy-temperature gain
```
**Params** (`__init__`, after the `W_loc` block, `pcs_substrate.py:358-360`): `if cfg.selfmon_weight > 0: self.P["W_phi"] = w((H, 1), 1)`.

**Forward** (`_core_forward_rate` line 464 and `_core_forward_spike` line 482): add `+ P["W_phi"][:,0] * phi_novelty` to `pre`/`inp`, exactly parallel to the `W_d @ d_t` term.

**Critical implementation choice for correctness**: `phi_novelty` is computed once at `observe()` time and **stored in the tape** (`step["phi_novelty"] = phi_novelty`, alongside `d`), then `_window_forward` *reads* it rather than recomputing it from `h_list[t-1]`. This avoids a same-step circular BPTT dependency and means the backward pass is a two-line copy of the existing `W_d` treatment (`_window_backward` rate branch, lines 933-936, and spike branch, lines 966-970): `grads["W_phi"] += outer(dpre, [tape[t]["phi_novelty"]])`. Because it's a stored stop-gradient constant (exactly like `d`, `a_prev`, `pos_target` already are), `gradcheck()` validates it for free — just add `"W_phi"` to the checked-param list (line 1203) and add `gradcheck("rate", selfmon_weight=1.0)` / `gradcheck("spike", tol=5e-2, selfmon_weight=1.0)` to `__main__` (mirroring lines 1381-1385 for aux-loc).

## 3. The behavioral channel — confidence-gated action commitment (the "hedge")

In `act()` (`pcs_substrate.py:580`), before softmax:
```python
if cfg.selfmon_weight > 0 and not self._selfmon_gate_lesion:
    tau = 1.0 + cfg.selfmon_kappa * self.last_novelty
    logits = logits / tau
```
High novelty → flatter softmax → lower max-prob / higher entropy → the agent's action selection becomes measurably less committed. This is the fork's non-linguistic analogue of the main-branch metacog organ's "prepend a hedge" — biologically anchored in the LC-NE/ACh unexpected-uncertainty literature (Yu & Dayan 2005; Behrens et al. 2007), which is exactly "surprise broadly flattens commitment," not a host-invented rule. Store `_last_logits_raw` (pre-division) alongside the existing `_last_logits`/`_last_probs` so post-hoc replay can recompute probabilities under alternative novelty traces without re-running the environment.

Two ablation flags, both pure inference-time toggles that never touch trained weights (same idiom as `set_lesion_mask`):
- `_selfmon_gate_lesion`: severs novelty → temperature (τ≡1).
- `_selfmon_feedback_lesion`: severs novelty → recurrence (feeds 0 into `W_phi` regardless of the real computed value).

## 4. Honest functional self-report

```python
def report_confidence(self) -> str:
    if self.cfg.selfmon_weight <= 0:
        return "self-monitor OFF"
    tag = "novel" if self.last_novelty > 1.0 else "familiar"
    return (f"my familiarity monitor reads this as {tag} "
            f"(one-step prediction error {self.last_novelty:.2f}x my recent baseline); "
            f"confidence set to {self.last_confidence:.2f}")
```
WYSIWYG by construction: the string reports exactly the scalar that drove the temperature — never a separate decorative decode, never a phenomenal claim.

## 5. Sound-metric validation (new runner: `research/runners/_fork_selfmon_derisk.py`, mirroring `_fork_pcs_emergence_derisk.py`)

**(a) Habituation curve.** From the probe rollout's position trace (already collected, same as `_place_cell_metrics`), bin steps by revisit count to the exact (x,y) cell. `HAB_RHO = Spearman(phi_raw, log(1+visit_count))`, expect **≤ -0.25**. This is the classic behavioral signature (Rescorla-Wagner-style error decay) — a system with no real memory of state cannot produce it.

**(b) Held-out novel-region discrimination (the decisive OOD test).** Confine the agent's *training* rollout to `x < 0.7*grid_size` (a ~3-line clamp on `nx` after the move step in `ForkPCSWorld.step`, or a runner-side clamp for the smoke — either way the world never trains on `x ≥ 0.7*grid_size`). Probe on the **full** grid. `DISCRIM_D = Cohen's-d(novelty | x≥0.7G vs novelty | x<0.7G)`, expect **≥ 0.5**, beating two floors:
   - *untrained-core floor*: replay the identical input sequence through a fresh random substrate (`replay_untrained`, already exists at line 288), compute the same phi/novelty formula off its own random `W_pred`/`W_enc_ema` — a random reservoir has no learned notion of "region," so its own D should be near-flat.
   - *raw-frame-diff floor*: `||v1feat_t - v1feat_{t-1}||` alone — kills "it's just noticing motion," since revisits to familiar cells can still have large frame-to-frame change.

**(c) Split-half stability** (direct reuse of the `_place_cell_metrics` pattern, lines 848-868): bin mean `phi_raw` by cell identity over the first vs second half of a long probe rollout, Pearson-correlate. Expect **≥ 0.35**. A pure-noise phi trace gives ~0.

**(d) Noisy-TV control** (irreducible-uncertainty floor): add one grid cell whose object type is redrawn uniformly at random on every visit (small `fork_pcs_world.py` addition, ~10 lines). Raw `phi_raw` there should stay permanently high, but `novelty_t` — because it's normalized against `phi_slow`, which tracks that same high baseline — must settle to **< 0.3×** its own first-10-visits average by the last 20% of training exposure. This proves the report is *relative familiarity*, not raw unpredictability.

## 6. Load-bearing test (the actual causal falsifier)

Three arms, computed by post-hoc replay over the SAME frozen probe's stored `(logits_raw_t, novelty_t)` pairs (no re-simulation needed):
- **Intact**: `tau_t = 1 + kappa*novelty_t`.
- **Gate-lesioned**: `tau_t ≡ 1` (control).
- **Cyclic-shift null** (K=64 draws, per the steer's explicit instruction to avoid an i.i.d. shuffle): `tau_t` computed from `novelty` rolled by a random circular time-offset — preserves the full marginal distribution and autocorrelation of the novelty trace, only destroys its alignment with *which moment* was actually surprising.

Metric: `Δ_conf = mean_max_prob(low-novelty tercile) − mean_max_prob(high-novelty tercile)`.

**GO requires**, reusing `tools.lab.attributable_to` and the `dependency_control`-style ratio/p95 reporting already standard in this arc:
1. `Δ_conf_intact` is present (not ~0);
2. `Δ_conf_intact` clears the cyclic-shift null: `> p95` **and** `≥ 3× mean` (same `ratio=3.0` default as `dependency_control`);
3. `attributable_to("selfmon gate", Δ_conf_intact, Δ_conf_gate_lesioned) ≥ 0.333` — matching this exact arc's existing `BEHAV_LESION_RATIO=1.5` convention (`1 − 1/1.5 = 0.333`), so the new mechanism, not a pre-existing confound between novelty and position-dependent policy confidence, owns at least a third of the effect. `attributable_to`'s own `warn_below=0.5` will print (not fail) a caveat between 0.333–0.5.

This is exactly the mission's bar: *state varies → response differs (1+2)*, *lesion → the difference is mostly attributable to the lesioned mechanism, not a confound (3)*.

## 7. Pre-registered GO gate

Calibrate all five numeric bars above on **one held-out pilot seed (e.g. 7)**, freeze them, then run the standing 6-seed set (42/43/44/100/101/102). **GO** iff all five clear on **≥ 5/6 seeds** (matching `SEEDS_REQUIRED_FRAC`). Config for the decisive run: `n_hidden=512, n_train=200_000, n_probe=8000, grid_size=24 (novel region x≥17), selfmon_weight=0.3, selfmon_kappa=2.0`, non-nav-required world, `units=rate` first (spike arm as a Day-N fork-thesis follow-up exactly like the existing convention), queued via `tools/gpu_queue.sh` (0 Claude tokens).

## 8. Cheapest falsifying smoke (run before spending any GPU time)

```
SIM_BACKEND=numpy python -m research.runners._fork_selfmon_derisk --smoke --out /tmp/fork_selfmon_smoke.json
```
Checks, in order of cheapness:
1. `gradcheck("rate"/"spike", selfmon_weight=1.0)` — catches an implementation bug in <1s before any behavioral claim.
2. Byte-identical-OFF: `selfmon_weight=0.0` (default) → identical `weight_hash()` to the pre-change substrate (mirrors `si_selftest`'s `off_no_state`/`identical_zero` checks) — proves true opt-in.
3. Miniature battery (n_hidden=64, n_train≈4000, tiny grid): every metric's **sign** must be correct (HAB_RHO<0, DISCRIM_D>0, Δ_conf_intact > Δ_conf_lesioned) even though none need to clear the full numeric bars yet. Wrong sign at this scale = stop, don't queue the GPU run.

## 9. Anti-cheats checklist

- Untrained-core floor and raw-frame-diff floor on the discrimination test (§5b).
- Cyclic-shift null, not i.i.d. shuffle, for the causal test (§6) — per the explicit steer, avoiding the place-cell arc's retracted instrument.
- `attributable_to` rather than a bare before/after diff for the lesion — isolates the mechanism's own contribution from a pre-existing novelty↔policy-confidence confound.
- Noisy-TV control (§5d) — rules out "the monitor just reports raw unpredictability."
- Held-out novel-*region* (a genuine train/probe split), not a natural-rarity proxy — avoids the risk that long training saturates visitation and leaves no real OOD tail.
- Split-half stability, not just presence, per the place-cell arc's own capacity-artifact lesson: decodability without stability was the exact prior failure mode.

## 10. Secondary/bonus (not gating)

Whether the recurrent-feedback path (`W_phi`, distinct from the temperature gate) itself measurably improves `eval_predictive_loss` — i.e., does knowing "I was just wrong" help future prediction. Report but don't gate on it, to avoid overclaiming a second faculty before it's separately earned. A novelty-gated online learning-rate multiplier (Lisman & Grace 2005, hippocampal novelty gates LTP) is a natural next-rung extension, deliberately deferred so the first GO stays on one crisp, falsifiable channel.

## Scope note

No navigation mechanism is touched or proposed. Position is used only as the diagnostic label for habituation/discrimination, identically to how the closed place/value/permanence probes already use it.

## ADVERSARIAL SKEPTIC — traps + first-target recommendation + falsifier

I have enough grounding in the actual substrate and the existing derisk battery. Returning the adversarial analysis.

---

# Adversarial feasibility review: emergent AFFECT vs SELF-AWARENESS on the PCS substrate

**TL;DR** — First target: **affect/valence**, but NOT the bare `adv` relabel (that is a tautology). The decisive, non-circular version is a **drive-anchored better/worse-than-expected readout**, tested by **double dissociation against the value faculty**. Cheapest falsifier: an **expectation-violation (reward-omission) probe on an already-trained checkpoint** — no retraining — that regresses out position+value and lesions against a value-matched control. Self-awareness is second: its signal is not load-bearing yet, so it requires building a feedback path before there is anything to lesion.

The structural difference from the place-cell arc that drives everything below: **there, the decode target (allocentric x,y) was EXTERNAL to the substrate.** Here, both candidate signals — `adv` (`pcs_substrate.py:628`) and `loss_fast/loss_slow` (`:398-401,642-645`) — are **quantities the substrate already computes for its own learning rule.** That inverts the main risk: the place-cell danger was "decodable but inert"; the affect/self-awareness danger is "**load-bearing by definition, therefore vacuous.**"

## Ranked traps (most decisive first)

**Trap 1 — The tautology / load-bearing-by-construction trap (THE dominant risk).** `adv` is *defined* as the policy-gradient multiplier: `dlogits = adv * (p - onehot)` (`:654`). So any lesion/ablation that shows "removing valence changes behavior" is testing that REINFORCE works, not that an affective faculty emerged. This masquerades as the strongest possible result (perfect load-bearing) while being empirically empty. The signal map even says building the faculty is "plumbing (`store self.last_valence = adv`), not new mechanism" — that is the confession. **Guard:** the load-bearing claim must route through a channel the signal was NOT already wired into (a persisted, read-back scalar fed into `h_t` via a new input weight, mirroring `W_d`), and the behavioral consequence must not be the REINFORCE update itself. If you cannot name a behavior that changes that does not pass through the existing gradient path, there is no faculty.

**Trap 2 — The relabel/exposure trap (no emergence).** Both signals already exist and already have their effect. "Building the faculty" collapses to `self.last_valence = adv` / exposing the EMAs. That is renaming a local variable, exactly the overclaim `docs/TERMS.md` exists to catch. **Guard:** a faculty claim requires something the substrate *grew through self-supervised training and did not have before* — demonstrated by a **base/untrained/reservoir control** where the signal is at floor and a trained substrate where it is present (the same base-control that legitimized the place code). Exposure alone is a documented plumbing change, never a finding.

**Trap 3 — The value/position collinearity trap (the affect-analog of the capacity-artifact floor).** In a foraging world, reward, drive-reduction, `V(h_t)` (`:495-497`), and distance-to-food are tightly collinear. A "valence decoder" reads high near food, low far away — but so does a plain distance/value decoder. You cannot distinguish an affective signal from a spatial/value one because they co-vary in the task. **Guard:** the test must isolate the **residual after value+position are regressed out** — i.e., *better/worse-than-expected*, which is precisely prediction error, not reward level. If the "valence" signal vanishes once `V` and position are partialled out, it was distance-to-goal re-expressed.

**Trap 4 — The noise-floor + wrong-null trap (self-awareness specifically).** `loss_fast − loss_slow` (`:645`) is a difference of two EMAs of a noisy loss; near convergence both sit at the noise floor and their difference is EMA-lag sampling noise. A "familiarity readout" can decode that noise. And the place-cell arc already proved the **i.i.d. shuffle is too weak a null.** **Guard:** use a **cyclic-shift / block-shuffle surrogate** of the loss (or of the surprise-condition labels) that preserves autocorrelation while destroying state-contingent structure; require the contingency to clear the block-shuffle floor, not the i.i.d. one.

**Trap 5 — The global-LP policy-collapse artifact (change ≠ function).** The substrate's own comment (`:202`, `:645`): LP is a single global scalar, and "a large global LP credited to arbitrary actions collapses the policy." If you make surprise state-conditioned and feed it back, a behavioral change you observe could be **collapse, not a working familiarity faculty.** **Guard:** require the fed-back signal to **improve (or at least appropriately structure) a held-out behavioral metric**, not merely to change behavior — per the standing rule "validate a signal by its FUNCTION, not a task that ignores it."

**Trap 6 — Shared-substrate lesion leakage (why "≥1.5× random" is insufficient here).** Every head reads the same `h_t`; `_apply_lesion` (`:449-452`) zeros units the value/place heads also read. A "valence-unit" lesion that degrades foraging may be damaging value/place, not affect. The runner's random-lesion control (`n_random_lesions=3`, `:415-418`) is a weak control against the *specific* alternative hypothesis (the value manifold). **Guard:** add a **value-matched (and place-matched) lesion arm** and require a **double dissociation**, not just faculty-vs-random.

## Recommended first target: AFFECT / VALENCE — but the drive-anchored version, not `adv`

Cheaper and more decisive than self-awareness, for concrete reasons:
- It has a **non-circular physiological anchor already in the recurrence**: interoceptive drive `d_t` (drive/deficit/energy) enters via `W_d` (`:464/482`) and is an *input*, not part of the learning rule. Hedonic tone = better/worse-than-expected **combined with drive state** is decodable and behaviorally anchored without touching the REINFORCE tautology.
- The derisk runner **already ships a full `value` faculty with an airtight behavioral-dependency lesion** (decode discounted future drive-reduction; lesion → reward-rate/approach degrades ≫ random, `:10-13, 428-452`). Valence is a small, well-scoped extension that **reuses that battery verbatim** — the value arm becomes the matched-lesion control for free (closes Trap 6).
- Self-awareness is harder/less decisive *first*: its signal is **not load-bearing at all** today (only via the coarse global-LP path), so you must **build the `h_t`-feedback loop before anything exists to lesion** — new mechanism, plus the Trap-4 null problem and the Trap-5 collapse risk. More surface area, more confounds, less near-term leverage.

Sharp caveat the skeptic must state out loud: the "cheap" valence version (relabel `adv`) is exactly the **vacuous** one (Traps 1–2). The decisive-and-cheap target is the **drive-anchored residual-valence readout tested by double dissociation against value** — cheap because it reuses existing machinery, decisive because it is the one framing that cannot be satisfied by the learning rule alone or by the value faculty.

## The single cheapest falsifier

**An expectation-violation (reward-omission) probe on an already-trained checkpoint — zero retraining.**

1. Take a trained agent (existing checkpoint). At a fixed set of states, deliver reward that is *as-expected*, *better-than-expected* (unexpected food), or *worse-than-expected* (expected food omitted), holding **position and drive-state approximately fixed** across conditions.
2. **Decode** a valence scalar from `h_t`, then **regress out position and `V(h_t)`.** Null = cyclic-shift/block-shuffle, not i.i.d.
3. **Lesion** the units carrying the *residual* valence signal; measure an expectation-modulated behavior (approach vigor / action selection).

**FALSIFIED if** the residual signal is at floor after partialling out value+position (Trap 3), **OR** its lesion degradation does not exceed **both** a random-lesion control **and** a value-head-unit lesion (Trap 6). If a value-lesion reproduces the same degradation, there is no separable affect faculty — it is the value system. This one probe needs no training, reuses `_lesion_dependency`/`_behavioral_saliency` (`:137-168, 428-452`), and hits the two most likely hollow modes (collinearity + shared-substrate leakage) simultaneously.

## Making the load-bearing test airtight

Reuse the runner's existing rig and add three affect-specific guards:

1. **Behavior, not decode.** The pass metric is a behavioral rollout read-out (`_lesion_dependency` already does rollout + `eval_predictive_loss(respect_lesion=True)`, `:433-434`). A decode R² is never the load-bearing result — that was the place-cell arc's core error.
2. **Two selection instruments** — decoding-selected AND behavioral-saliency-selected units (`lesion_mode="both"`, `:388-395`); their dissociation is itself the decodable≠load-bearing check.
3. **Random control, ≥5 draws** (bump from 3), require faculty-lesion ≥ 1.5× random (`BEHAV_LESION_RATIO`, `:84`).
4. **NEW — value-matched lesion arm + double dissociation.** Random units are too weak a control against the collinear value manifold. Require: valence-lesion hurts the expectation-modulated behavior more than value-lesion does, AND value-lesion hurts reward-rate/homing more than valence-lesion does. Without the cross-arm, "≥1.5× random" is satisfiable by any unit sitting in the value subspace.
5. **Recurrent-core lesion (zero `W_h`) collapse** (`:21`, `CORE_LESION_COLLAPSE`) — proves the signal lives in the recurrence/integration, not a feedforward read of the current view.
6. **Right null on the signal itself** — cyclic-shift/block-shuffle, not i.i.d. (the arc already proved i.i.d. is too weak).
7. **Base/untrained control** — signal at floor in a reservoir substrate, present after self-supervised training (the control that legitimized the place code); this is what separates *grown* from *relabeled* (Trap 2).
8. **Pre-register thresholds, 6/6 seeds** (`:25-35`), matching the project bar.
9. **For self-awareness when you reach it:** the load-bearing test MUST route the surprise scalar back into `h_t` via a new input channel mirroring `W_d` — a decode-only familiarity signal with no feedback is the decodable-but-inert trap by construction — and the behavioral consequence must be a **confidence-appropriate** change (e.g., more exploration/slowing under high surprise) that improves/structures behavior, since bare change is consistent with the global-LP collapse artifact (Trap 5). Honesty boundary stays a functional read-out only ("familiarity monitor reads novel → I report uncertain"), never a phenomenal claim.

**Files grounding this:** `/home/dant123/Projects/sim-agi-fork/sim/pcs_substrate.py` (recurrence `:462-490`, `adv` `:621-634`, LP EMAs `:636-645`, lesion `:449-452`, heads `:492-507`, `eval_predictive_loss` `:1113-1136`); `/home/dant123/Projects/sim-agi-fork/research/runners/_fork_pcs_emergence_derisk.py` (pre-registered gate `:25-37`, behavioral-saliency selection `:137-168`, random+faculty lesion dependency `:405-452`, floors/nulls `:181-199`).