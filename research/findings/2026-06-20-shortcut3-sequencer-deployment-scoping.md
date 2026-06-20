# Shortcut #3 — the production conversational sequencer: DEPLOYMENT scoping (retire the host `_scan`) + the K=32 routing-margin treatment (2026-06-20)

**Type:** READ-ONLY design / deployment scoping. NO code written, NO experiments run, NO GPU. One design document. Stayed
on `main`.

**The shortcut (#3):** the SHIPPED production `OneBrainComposer` (`research/runners/one_brain_composer.py`) answers its
who/what (and yes-no / describe / chain) queries with a host `_scan` — a Python `for`-over-stored-facts +
`if cue-matches then return-the-answer-role else continue/return-None` (`one_brain_composer.py:510-514`, and the
inlined twins at `query_patient:576-582`, `ask_yes_no:590-593`, `render_fact:602-609`). That cue-match COMPARISON +
answer/abstain ROUTING is a COGNITIVE control-flow op done by host bookkeeping = a brain-based-only shortcut
(`2026-06-20-shortcut-burndown-inventory.md` #3). The role labels it matches over are ALREADY spiking-cleaned (the #1
NEF WTA cleanup); the residual host op is the thin symbolic match + abstain decision.

**The status this doc corrects.** Per the owner's HARD rule (`feedback_close_all_shortcuts_before_capability`, hardened
CYCLE 329: "a boundary is a prompt to do MORE research + try NEW mechanisms until past it — never a stop; 'closed' =
CONVERTED-TO-SPIKING, host op RETIRED"), #3 is NOT closed until the host `_scan` is RETIRED in production. The
shortcut-burndown status (`2026-06-20-shortcut-burndown-status.md`) lists #3 as "IN PROGRESS: S0 GO; S1 GO; **S2 in
flight** (lift K=16→K=32); then S3 (fold), S4 (320-scale)." This doc establishes that **S2's first run is committed and
read NEGATIVE at K=32 by the CHEAPEST retreat ONLY (divnorm re-tune)** — and that the failure is a single-knob
true-match-rate margin squeeze (NOT a moat failure, NOT a cleanup failure), with **retreat-2 (the NEF-FS WTA pool)
already BUILT in the S2 runner but NEVER RUN.** So the honest current state is: the K=32 margin is an OPEN, finishable
point-neuron BUILD with the next mechanism already coded and one obvious threshold knob untried — it is NOT a
characterized boundary, and the production fold (S3/S4, retire `_scan`) has not started. This doc designs the remaining
work to RETIRE `_scan` at production scale.

---

## 0. TL;DR (the load-bearing answers)

- **Current state (what blocks the fold):** the on-bridge spiking sequencer is GO to **K=16** (S0 K-way generalization,
  6 seeds; S1 on-bridge divisive-norm wired in, host `scores_to_drive` peak-read RETIRED, D=128, 3 seeds — both `==host`
  + moat 0-FA). At the **production K=32**, the S2 margin sweep with the CHEAPEST retreat only (divnorm re-tune,
  `gain=0.11`) reads NEGATIVE on `==host` (2/3 seeds) — but the failure is **one present-cue dropping to abstain on ONE
  seed** because its correct-block match pool fired at `m4=0.116`, just below the `0.15` match threshold, while ALL 31
  other match pools fired exactly `0.000` (zero leak). **The moat held 3/3, 0 false-accepts, at K=32**; the cleanup was
  clean (`modes ex/xt/ms = 64/0/0`). So #3's fold is blocked ONLY by a true-match-rate margin squeeze in the larger
  K=32 inhibitory fabric — a calibration, not a boundary — with the next-strength mechanism already built.
- **The K=32 treatment (per the owner's rule — try NEW mechanisms, never accept K≤16):** the failure is diagnosed to a
  single mechanism (the K-way first-match priority WTA's larger inhibitory fan-out pulls the WINNER's firing below the
  fixed match threshold at K=32). Three named retreats, cheapest-first, all coded or trivial: **(R0)** raise the
  match-read threshold's HEADROOM by lowering it toward the measured true-match floor OR (equivalently) raise the
  decoded-line drive `hi_pA` so the winner clears threshold with margin — a one-line op-point change, untried; **(R1=retreat
  2, BUILT not RUN)** the NEF-FS lateral-inhibition WTA score pool (`build_wta_score_bridge` /`wta_drive`,
  `_phaseB_onebrain_sequencerK_k32_margin_derisk.py:122-190`) — sharpens the decoded drive to a single hard winner so the
  per-block match is cleaner and the priority WTA's winner stays high; **(R2=retreat 3)** the hierarchical 2-level match
  (coarse agent-cue pre-filter → only survivors run the full conjunction → far fewer parallel cascades into the K-way
  WTA, restoring the winner margin). The boundary is the spike-SNR of the 1-of-K=32 winner-take-all under fixed
  thresholding — and the SELECTION primitive is already proven at a HARDER scale (the #1 cleanup does 1-of-V=320), so
  the prediction is GO at K=32 once the winner margin is restored.
- **The deployment design (retire `_scan`):** promote the validated loop from the `_phaseB_*` / `_phaseC_*` runner
  subclass to an opt-in `integrated_loop=True` mode on the production `OneBrainComposer` (default OFF = byte-identical to
  today), routing the FIVE who/what-core methods (`query_patient`/`query_agent`/`ask_yes_no`/`render_fact`/`query_chain`)
  through the K-way sequencer instead of `_scan`. The composer ALREADY runs one persistent co-resident bridge with the
  cleanup in spikes; the sequencer is a second region-set on the SAME bridge (or a coupled score-bridge, as the de-risks
  used) driven by the cleanup result. The "multi-bridge arc" framing resolves to **one of two concrete options** (§2.2):
  (A) co-resident sequencer regions on the composer's existing bridge, or (B) a coupled `SequencerControl` score-bridge
  (what S1/S2 already build) wired as a composer member. Option B is the lower-risk first cut (it is literally the
  already-validated S1/S2 code, folded behind the flag).
- **The cheap-first de-risk:** the smallest test that the folded spiking sequencer answers the production who/what matrix
  `==host` INCLUDING the `is None` abstentions, at K up to 32 / D=128 / the 320 vocab — run on the REAL
  `OneBrainComposer` with `integrated_loop=True`, comparing every method's answer to the host `_scan` path on the SAME
  store, with the moat 0-FA as the HARD gate. Stage it: (S2-finish) the K=32 margin via R0→R1→R2; (S3) the opt-in fold +
  CI guard; (S4) the 320-scale production GO.
- **The moat is sacrosanct + NEVER weakened.** Every stage's GO requires 0 false-accepts; the K=32 failure analyzed here
  is a present-cue MISS (over-abstention), the SAFE direction — the moat has held 0-FA at every K, every seed, in every
  committed run. No proposed design trades the moat for a pass.
- **Effort + honest scope:** finishing #3 is a **days arc** (≈2–4 focused working days) — the K=32 margin is the only
  real unknown and its next mechanism (R1) is already coded; the fold (S3/S4) is reuse-by-import, NO `sim/` edit
  expected. The honest-negative form (if R0/R1/R2 all fail to restore the K=32 winner margin WITHOUT a moat breach) is a
  characterized partial conversion (on-bridge to K*, the runner's `--host-fallback-above K*` already implements it) — but
  per the owner's rule this is the LAST resort after the named mechanisms are exhausted, not the default, and current
  evidence (clean cleanup, zero leak, a single-knob winner-margin squeeze) predicts K=32 GO.

---

## 1. CURRENT STATE — what exactly the sequencer does, to what K it matches host, and what blocks the fold

### 1.1 What the on-bridge spiking sequencer does (the replacement for `_scan`)

The Phase-B sequencer (`2026-06-19-onebrain-sequencer-derisk.md`, GO 6/6 seeds) replaces the host `_scan`'s
`for/if/return` with a spiking basal-ganglia / thalamocortical control circuit on `cp_connections` (Izhikevich routes,
NO `sim/` edit):

1. **The op result (the cleanup, already spiking).** The REAL `OneBrainComposer` reconstructs + unbinds + cleans each
   stored block; its cleanup lands decisive per-role-per-word scores on the membrane (`block_cleanup_scores`,
   `_phaseB_onebrain_sequencer_derisk.py`, imported verbatim by every later stage). This is the op's spiking RESULT the
   sequencer conditions on — NOT a host string.
2. **The match, by GATED DISINHIBITION** (`couple_gate_to_pool`): the CUE word-line opens a per-word transmission gate,
   so the per-word match line fires iff block b's DECODED word == the CUE word; a per-block gated conjunction
   (`m{b}` fires iff agent AND action match). This is the genuinely-unproven point-neuron step, and it is GO — a
   weight-tuned coincidence-AND walled (pool-pulse over-drive, post-inhibitory rebound inverting the bias, network-state
   dependence), and gated disinhibition (the Logiaco-Abbott-Escola routing fabric) is the robust primitive
   (`2026-06-19-onebrain-sequencer-derisk.md:84-103`).
3. **The decision = the BG production rule over the K spiking match pools** (Spaun "BG action-selection IS cognitive
   control"): the lowest-index block with `m{b} > match_thresh` answers (first-match priority, == the host `_scan`'s
   "return the FIRST match"); none over threshold → abstain (the moat). The answer-vs-abstain + first-match priority is a
   K-way priority WTA: `ans{b}→inh{b}→{ans{j>b}} ∪ {abstain}`, the abstain channel a tonic default suppressed by ANY
   match (`2026-06-20-burndown-3-S0-kway-sequencer.md`, the K-generalization).

So the host `for got: if got==want: return` cue-match COMPARISON is gone (replaced by the spiking gated match, computed
for all blocks in parallel); the answer/abstain ROUTING is the production rule over the spiking match. **The decision is
read from the spiking match pools** — the legitimate body read (like the nav cascade reading which motor channel won),
NOT a host re-implementation of the cue-match.

### 1.2 To what K it matches host (the committed evidence)

| stage | what | result | source |
|---|---|---|---|
| S0 | generalize the sequencer builder K=2 → K-way (K-way priority WTA + K-way first-match rule), CPU/numpy, D=64 | **GO 6/6 seeds (42-47)** `==host` + moat 0-FA + lesion-safe + permuted-inverts at **K∈{2,4,8,16}** | `2026-06-20-burndown-3-S0-kway-sequencer.md`; `_phaseB_onebrain_sequencerK_derisk.py` |
| S1 | wire the on-bridge divisive-norm into the loop (RETIRE the host `scores_to_drive` peak-read), D=128 | **GO 3/3 at K∈{2,4,8}**; K=16 a mapped boundary (2/3); **moat 0-FA at EVERY K**; host `s.max()` GONE | `2026-06-20-burndown-3-S1-divnorm-in-loop.md`; `_phaseB_onebrain_sequencerK_divnorm_derisk.py` |
| S2 | the K=32 production margin sweep, retreat-1 (divnorm re-tune `gain=0.11`) ONLY, D=128, 3 seeds | K∈{2,4,8,16} **GO 3/3**; **K=32 NEGATIVE** (`eq 2/3`); **moat 3/3, FA_total=0 at K=32** | `_phaseB_onebrain_sequencerK_k32_margin.json` (`first_break_K=32, k_star=16`) |

**The committed on-bridge GO is K=16** (`k_star=16` in the S2 raw). The production deployed scale is K=32 (the
`OneBrainComposer` default `k_max=32`, `one_brain_composer.py:93`; the 320-scale demo stores up to 32 facts).

### 1.3 What specifically blocks the fold (the diagnosed K=32 failure — NOT a moat failure)

The S2 K=32 NEGATIVE was extracted at the seed level. The break is a SINGLE present-cue MISS on ONE seed:

```
seed 43, cue (sun, hop), correct block 4:
  m4 = 0.116   (the CORRECT block's match pool fired)
  m0..m3, m5..m31 = 0.000   (ZERO leak on all 31 other blocks)
  match_thresh = 0.15  ->  m4 below threshold  ->  decision = abstain  (over-abstention)
  cleanup modes (exact/extra/miss) = 64/0/0   (the divnorm cleanup is PERFECT at K=32 -- single argmax, no extra, no miss)
seeds 42, 44: eq_all=True, perm=True, moat 0-FA  (clean GO at K=32)
```

The diagnosis is unambiguous:
- **It is NOT a moat failure.** Moat 3/3, `FA_total=0` at K=32 across all three seeds. The failure direction is
  over-abstention (a present cue answered `None`), which is the SAFE direction — the moat is structurally protected.
- **It is NOT a cleanup / leak failure.** `modes 64/0/0` (the on-bridge divnorm lights ONLY the argmax word at K=32);
  the 31 non-matching blocks fire exactly `0.000` (no cross-block leak — the worst-leak pressure the scoping flagged did
  NOT materialize; the divnorm killed it).
- **It IS a true-match-rate margin squeeze.** The correct block's match pool fired `0.116` vs the fixed `0.15`
  threshold. As K grows from 16→32, the K-way priority WTA's inhibitory fabric (32 `inh{b}` interneurons each able to
  suppress the abstain channel and higher blocks) loads the winner's own answer pool slightly, pulling the winning
  `m{b}` firing fraction down. At K=16 the winner cleared 0.15; at K=32 on one seed it dipped to 0.116.

**So the fold is blocked by ONE knob (the winner's firing margin vs the fixed match threshold), not by the cleanup, not
by the moat, and not by a fundamental discrimination wall.** The "multi-bridge arc" framing that deferred S3 is a
mislabel of two distinct unfinished pieces: (i) the K=32 winner-margin (S2-finish, the only real risk, next mechanism
already coded), and (ii) the opt-in fold into the production composer (S3/S4, pure wire-in). Neither requires a new
bridge architecture beyond what S1/S2 already build.

---

## 2. THE DEPLOYMENT DESIGN — retire the host `_scan` in production

### 2.1 The seam: which production methods route through the sequencer

The five who/what-core methods are the SAME role-agnostic op at different cue-arities (verified
`one_brain_composer.py`):

| production method | line | host control-flow form | the cognitive op | sequencer realization |
|---|---|---|---|---|
| `query_patient(agent, action, …)` | 576-582 | `for i, got: if agent&action match: return …` | what: cue=(agent,action) → patient | the K-way match on 2 cue roles; body-read the patient role |
| `query_agent(action, patient)` | 584-585 | `return self._scan({action,patient}, "agent")` | who: cue=(action,patient) → agent | the SAME K-way match, different cue roles drive the lines |
| `ask_yes_no(agent, action, patient)` | 587-593 | `for got: if full-SVO match: return yes/no` else `unknown` | yes/no/unknown: cue=full SVO → polarity | a 3-cue-role gated conjunction (one more gated AND); `unknown`-on-miss IS abstain |
| `render_fact(agent, …)` | 595-609 | `for i, got: if agent match: return rendered fact` | describe: cue=(agent) → emit fact | 1-cue-role match; body-read the FULL row |
| `query_chain(cue, actions)` | (rides `query_patient`) | iterated `query_patient`; abstain on first miss | multi-hop what; moat at every hop | iterated sequencer; the moat holds per hop by composition |

The loop is ALREADY role-agnostic in the de-risks (`_phaseC_task2_wholeturn_loop.py`'s `block_role_scores(c, b, role_a,
role_x)` generalizes the K=2 `block_cleanup_scores`; `query_patient` and `query_agent` both run through one
`run_sequencer`, differing only in WHICH cleanup roles drive the cue lines). So the core (who/what) needs ZERO new
mechanism beyond the K-scaling; yes/no folds in by driving a THIRD cue role into the gated conjunction; describe folds in
by reading back the full matched row; chain composes for free.

### 2.2 The fold mechanism (resolving the "multi-bridge arc")

Promote the loop to an opt-in mode on the production `OneBrainComposer`:

```
OneBrainComposer(..., integrated_loop=True)   # default False = byte-identical to today's _scan path (the oracle/CPU)
```

When the flag is ON, the five methods route through the K-way sequencer; when OFF, they use the host `_scan` (kept as
the explicit oracle / numpy-CPU fallback, mirroring how `rf` is retained as the test oracle).

The composer ALREADY runs ONE persistent co-resident `SimulationBridge` with the cleanup in spikes (`enable_spiking_cleanup`
default-on, `2026-06-20-burndown-1-onebrain-spiking-cleanup.md`). The sequencer is added in one of two concrete ways —
**this is the "multi-bridge arc" demystified, and it is two design options, not an unbounded research arc:**

- **Option B (the lower-risk FIRST cut — recommended): a coupled `SequencerControl` score-bridge.** This is LITERALLY
  what S1/S2 already build (`build_sequencerK_bridge` + `build_divnorm_score_bridge`/`build_wta_score_bridge` +
  `run_sequencerK_with_drive`), folded behind the composer flag. The composer's cleanup result (the membrane scores,
  already produced) drives the divnorm score bridge → the decoded word-lines → the sequencer bridge → the production-rule
  decision; the composer reads the spiking decision (which block / abstain) and does the body-read. The sequencer +
  score bridges are separate `SimulationBridge` instances coupled by the score hand-off — exactly the validated S1/S2
  topology. **This is the cheapest path to retiring `_scan`: the already-GO-to-K=16 code, wired into the production
  composer behind a flag.** ("Multi-bridge" here just means "the composer's existing bridge + the sequencer/score
  bridges," all co-resident in process — not a distributed system.)
- **Option A (a later consolidation, if desired): co-resident sequencer regions on the composer's bridge.** The
  framework path is a wrapper around `inject_explicit_wiring` (`bridge.py:2196`), so the sequencer's match cascades +
  priority WTA can be appended as `BrainRegion`s on the composer's OWN bridge (disjoint neuron-index slices, the
  established merge pattern from the nav+conv unification). This is the TRUE one-bridge form but is a larger change; it
  is NOT required to retire `_scan` (Option B does that). Defer A to a consolidation pass after B's production GO.

**Recommendation: ship Option B behind `integrated_loop=True`, GO it at production scale, flip the default; pursue
Option A only as a later same-bridge consolidation.** Option B retires the host `_scan` (the #3 deliverable) with the
already-validated code.

### 2.3 NO `sim/` edit expected

Reuse-by-import throughout (matching S0/S1/S2, all NO `sim/` edit):
- the K-way sequencer (`build_sequencerK_bridge`, `wire_sequencerK_couplings`, `reset_sequencerK_state`,
  `run_sequencerK`) — `_phaseB_onebrain_sequencerK_derisk.py`;
- the on-bridge divisive-norm (`input_divisive_norm`, already in `sim/` from the PPMI work — `regions.py:240`,
  `config.py:440`, `bridge.py:6048`) — flipped from the runner, NO edit;
- the gated-disinhibition match (`couple_gate_to_pool`), the transmission gate (`set_transmission_gate` /
  `cp_transmission_gain`), the BG WTA template (`g11_bg_runner --bg-lateral-inhibition`, catalog A.04) — all on
  `cp_connections`, public APIs;
- the NEF-FS WTA score bridge (retreat 2) — `build_wta_score_bridge`/`wta_drive`, ALREADY in the S2 runner, runner-side
  score-bridge wiring, NO `sim/` edit;
- the conditional `rf_kick` tracker-mask edit the scoping flagged is CONFIRMED NOT needed for the sequencer (it gates
  Izhikevich routes, not RF routes; `2026-06-19-onebrain-sequencer-derisk.md:120-123`); flag for byte-review ONLY if a
  K=32 multi-op micro-schedule ever surfaces it (predicted: not needed).

---

## 3. THE K=32 ROUTING-MARGIN TREATMENT (per the owner's rule — try NEW mechanisms, NEVER accept K≤16)

The K=32 failure is a true-match-rate margin squeeze (§1.3): the WINNER's match pool fired `0.116` vs the fixed `0.15`
threshold on one seed, with ZERO leak elsewhere and the moat intact. This is the SAFE-direction failure (over-abstention)
and is one knob from GO. The retreats, cheapest-first, ALL coded or trivial — run them IN ORDER, stopping at the first
K=32 GO:

### R0 — the threshold / drive-margin re-calibration (cheapest, UNTRIED). 
The winner fired `0.116`; the threshold is `0.15`. The K=2 operating point's threshold did not transfer to K=32 because
the larger WTA fabric pulls the winner down. **Two equivalent one-line fixes, neither tried in the committed S2 run:**
(a) lower `match_thresh` toward the measured true-match floor with a safety band (e.g. `0.08`, still far above the
no-match `0.000` — the no-match floor is EXACTLY zero at K=32, so there is enormous headroom: the separation is
`0.116` vs `0.000`, not `0.116` vs a leaky `0.10`); OR (b) raise the decoded-line drive `hi_pA`
(`_phaseB_onebrain_sequencerK_derisk.py` `run_sequencerK`) so the winner's match pool fires harder and clears the
existing threshold. Because the no-match leak is zero (the divnorm killed it), **the threshold has a full `0.116`-wide
no-match margin to drop into** — this alone very likely restores K=32 GO. This is the FIRST thing to run; it was not in
the committed sweep (which fixed `match_thresh` at the K=2 value).

### R1 — the NEF-FS lateral-inhibition WTA score pool (retreat 2, BUILT in the S2 runner, NEVER RUN).
`build_wta_score_bridge`/`wta_drive` (`_phaseB_onebrain_sequencerK_k32_margin_derisk.py:122-190`) add a feed-forward
lateral-inhibition pool on the decoded word-lines (each word-pool excites a shared inhibitory pool; the inhibitory pool
suppresses every word-pool) ON TOP of the per-query divisive norm (Carandini-Heeger normalization + hard competition).
This drives a SINGLE hard winner per role, sharpening the per-block match so the winning `m{b}` fires higher and clears
threshold with margin. Run it via `--retreat wta`. This directly targets the winner-margin squeeze and is the named
production-strength mechanism. (It was coded specifically as the retreat-2 the scoping named; the committed run only
exercised retreat-1.)

### R2 — the hierarchical 2-level match (retreat 3, if R0+R1 insufficient).
A coarse pre-filter on the agent cue role gates which blocks run the full gated conjunction → far fewer parallel match
cascades feed the K-way WTA at once → the priority WTA's inhibitory fan-out shrinks → the winner margin recovers. A
sequencer re-wire, still NO `sim/` edit. This is the structural fallback if the op-point + WTA cannot restore the margin.

### Is it within the spiking WTA's proven range? — YES.
The #1 cleanup spiking WTA already does **1-of-V=320** (a 320-way argmax-by-firing, validated `==`host, moat 0-FA,
`2026-06-20-burndown-1-onebrain-spiking-cleanup.md`). Picking 1-of-32 blocks is a SMALLER selection. The match CASCADE
at K=32 produces a clean single winner (`64/0/0` cleanup, zero cross-block leak); the only residual is reading that
winner above a fixed threshold in the larger fabric — exactly what R0/R1 fix. **Prediction: K=32 GO after R0 (and
certainly after R1).** This is a calibration/competition treatment, NOT "accept K≤16."

### The honest-negative form (the LAST resort, NOT the default).
IF R0+R1+R2 all fail to restore the K=32 winner margin WITHOUT a moat breach (no current evidence suggests this — the
moat is structurally safe and the no-match floor is exactly zero), THEN the characterized deliverable is the per-K
margin table showing where the 1-of-K winner-margin squeezes, and the production loop runs the sequencer to K* and the
host `_scan` above it (the runner ALREADY implements this via `--host-fallback-above K*`). But per the owner's HARD rule
this is reached only after the named mechanisms are exhausted — and given a clean cleanup, zero leak, and a single-knob
winner margin, the expected outcome is K=32 GO, not a boundary.

---

## 4. THE CHEAP-FIRST DE-RISK + ANTI-CHEATS (the moat is the HARD gate)

### 4.1 The cheap-first de-risk (smallest test that the folded sequencer answers the who/what matrix `==host` incl. abstentions, at production scale)

Staged, each ending GREEN against the host `_scan` oracle:

- **S2-finish (the K=32 margin — the one real risk):** rerun `_phaseB_onebrain_sequencerK_k32_margin_derisk.py` with
  **R0** (`--retreat divnorm` + a lowered `match_thresh` / raised drive — a one-line op-point change), then **R1**
  (`--retreat wta`), K∈{2,4,8,16,32}, D=128, **6 seeds** (the standing rule for the noise-sensitive cascade). GO =
  `==host` on who/what + abstain on absent/cross at every K through 32, moat 0-FA. CPU/numpy (the exact-algebra parity
  oracle).
- **S3 (the opt-in fold + CI guard):** add `integrated_loop=True` (Option B) to `OneBrainComposer`; route the five
  methods through the sequencer when ON. The cheapest sufficient test: a NEW `tests/test_onebrain_integrated_loop.py`
  asserting K∈{2,8} who/what + yes-no + describe + the three `is None` moat abstentions with the flag ON (GPU-gated,
  skips gracefully like `test_onebrain_spiking_cleanup.py`), AND `test_one_brain_composer_agent.py` +
  `test_brain_conversational_agent.py` pass VERBATIM with the flag OFF (byte-unregressed).
- **S4 (the production 320-scale GO — retires `_scan` in production):** run
  `consolidated_320_conversation_demo.py --composer onebrain` with `integrated_loop=True` (D=128, V up to 320, k_max=32,
  the stream-learned cortex codes — the flagship production conversation), comparing every answer to the host `_scan`
  path on the SAME store. **3 seeds (42/43/44)**, the 320-scale demo precedent. GO = `==host` on the 320-scale
  who/what(/chain) matrix + moat 0-FA → flip `integrated_loop=True` as the production default (host `_scan` kept as
  `--host-scan` oracle / numpy-CPU fallback). GPU only.

### 4.2 The anti-cheats (the moat 0-false-accepts is SACROSANCT — never weakened)

The full battery, applied at every stage (all ALREADY in the S0/S1/S2 runners — reuse verbatim):

1. **Answer-identity vs the host `_scan`** across the FULL who/what matrix (present cues answer the right block; the
   patient/agent label `==` the host path on the same store), at K∈{2,4,8,16,32}, multi-seed. This is the `eq_all` /
   `==host` gate.
2. **The no-confab MOAT — the HARD gate — 0 false-accepts.** Every absent/cross cue abstains (the K-way WTA selects
   abstain; the emitted answer is `None`/`unknown`). **A single false-accept at any seed, any K, is a FAIL.** The moat
   is NEVER traded for an `==host` pass (`feedback_moat_not_hard_lossy_memory_ok`: kept where free — and here it is free,
   structurally protected by the gated match). The K=32 failure analyzed here is over-abstention (the safe direction),
   so the moat is not even at risk; the gate still asserts 0-FA explicitly at every K.
3. **Sequencer-LESION fails SAFE.** Sever the result→op conditioning (the decoded word-lines get zero drive) on every
   present cue → the match can't fire → the sequencer ABSTAINS, never confabulates a wrong block. (`lesion_fails_safe`.)
4. **Permuted-rule INVERTS.** Cyclic-shift the match→answer map (`m{b}→ans{(b+1)%K}`) → a present cue for block b routes
   to `ans{(b+1)%K}` — the decision follows the RULE applied to the spiking match, not a fixed scan order.
   (`permuted_inverts`.)
5. **The NO-DIVNORM (raw) control FAILS.** The same battery with the divisive-norm OFF (the raw un-normalized drive +
   the same placed threshold) breaks `==host` or the moat → the normalization is load-bearing. (`raw_fails`.)
6. **The K=32 routing-margin STRESS (provenance).** The K=32 fact set is maximal-stress by construction (8 actions each
   shared by 4 facts, so the shared-action cross-term is maximal); the per-K margin table reports `true-match rate` vs
   `worst no-match leak` vs threshold at each K — leakage/provenance is measured, not assumed. (The S2 runner already
   does this via `_cleanup_mode_counts` + the per-seed `m{b}` reads.)
7. **OFF == byte-identical** (the production regression guard): `integrated_loop=False` reproduces today's `_scan` path
   exactly; the full CI suite passes verbatim.
8. **Multi-seed:** 6 seeds for the noise-sensitive K-margin sweep (S2-finish); 3 seeds for the 320-scale production GO
   (the demo precedent).

---

## 5. THE BUILD PLAN (days, not weeks)

| stage | what | the GO check | effort | risk |
|---|---|---|---|---|
| **S2-finish** | the K=32 winner margin: **R0** (threshold/drive re-cal, untried) → **R1** (`--retreat wta`, BUILT not run) → R2 (hierarchical, if needed), K∈{2,4,8,16,32}, D=128, **6 seeds**, CPU | `==host` + moat 0-FA at K=32 through the named retreats; the per-K margin table | ~1–1.5 days | the ONLY real unknown; next mechanism already coded; predicted GO (clean cleanup, zero leak, one knob) |
| **S3** | fold into `OneBrainComposer` as `integrated_loop=True` (Option B, default-off) + route the 5 methods + CI guard | OFF==byte-identical (full suite verbatim); ON asserts K∈{2,8} who/what + yes-no + describe + moat, GPU-gated | ~1 day | low (reuse-by-import; the S1/S2 code behind a flag) |
| **S4** | the 320-scale production GO on `consolidated_320_conversation_demo --composer onebrain --integrated-loop`, **3 seeds**, GPU | `==host` on the 320-scale who/what(/chain) matrix + moat 0-FA → flip the production default; host `_scan` kept as the oracle | ~0.5–1 day (mostly GPU run time) | low |

**Total: ≈2.5–3.5 working days nominal.** The variance is the K=32 winner margin (S2-finish) — and per the owner's rule
that is finished by trying R0→R1→R2 (the next mechanisms, already coded), NOT by accepting K≤16. #3 is closed when S4
flips `integrated_loop=True` as the production default and the host `_scan` is retired to an explicit oracle/CPU
fallback.

**Ordering vs the other burndown items:** #3 is the ONE big remaining conversational arc; it is a days pass on proven
mechanisms; it retires the single most pervasive conversational host shortcut (the cue-match control flow under 5
production methods + the chain). It ranks ABOVE the nav reward/value dendritic frontier (#9) and the deferred FHRR-B
learned binder (owner-sequenced last). It touches disjoint files from the cheap nav default-flips, so it can run in
parallel with them.

---

## 6. SOURCES (file:line / finding verified)

- **The shortcut + the host op:** `research/runners/one_brain_composer.py` (`_scan`:510-514; `_read_blocks`:503-508;
  `query_patient`:576-582; `query_agent`:584-585; `ask_yes_no`:587-593; `render_fact`:595-609;
  `enable_spiking_cleanup`/`_select`/`_decode_batched_mem`:485-501; `k_max=32` default:93).
- **The proven sequencer + the K-generalization:** `research/runners/_phaseB_onebrain_sequencer_derisk.py` (the K=2
  kernel + `block_cleanup_scores` + `scores_to_drive`); `research/runners/_phaseB_onebrain_sequencerK_derisk.py` (the
  K-way builder/wiring/reset/rule); `research/runners/_phaseC_task2_wholeturn_loop.py` (the role-agnostic loop).
- **The committed evidence:** `2026-06-19-onebrain-sequencer-derisk.md` (Phase B GO 6/6, the gated-disinhibition fix +
  the K-margin scope note); `2026-06-20-burndown-3-S0-kway-sequencer.md` (S0 GO K∈{2,4,8,16}, 6 seeds);
  `2026-06-20-burndown-3-S1-divnorm-in-loop.md` (S1 GO K∈{2,4,8} D=128, host `scores_to_drive` retired, K=16 boundary);
  `research/findings/raw/_phaseB_onebrain_sequencerK_k32_margin.json` (S2 K=32 NEGATIVE by retreat-1 only:
  `first_break_K=32, k_star=16, gain=0.11`; per-seed: seed-43 cue (sun,hop) `m4=0.116` < `0.15`, all other `m=0.000`,
  moat 3/3 FA=0, cleanup `64/0/0`).
- **The K=32 retreats:** `research/runners/_phaseB_onebrain_sequencerK_k32_margin_derisk.py` (retreat-1 divnorm re-tune;
  **retreat-2 `build_wta_score_bridge`/`wta_drive`:122-190 — BUILT, NEVER RUN**; retreat-3 hierarchical, named;
  `--host-fallback-above K*` for the honest partial conversion).
- **The deployment + scope:** `2026-06-20-burndown-3-production-sequencer-scoping.md` (the original staged plan + the
  seam table + the K=32 boundary prediction + the NO-`sim/`-edit picture); `2026-06-20-shortcut-burndown-status.md` (#3
  IN PROGRESS, S2 in flight); `2026-06-20-shortcut-burndown-inventory.md` (#3 + #4(S5) + #1 buckets).
- **The selection-in-proven-range evidence:** `2026-06-20-burndown-1-onebrain-spiking-cleanup.md` (the 1-of-V=320
  spiking WTA cleanup, `==`host, moat 0-FA — a HARDER selection than 1-of-32).
- **The production path:** `research/runners/consolidated_320_conversation_demo.py` (`--composer onebrain` default; the
  320-scale 3-seed GO precedent); `tests/test_one_brain_composer_agent.py`, `tests/test_onebrain_spiking_cleanup.py`
  (the CI guards).
- **Reusable `sim/` primitives (NO edit):** `sim/regions.py:240` / `sim/config.py:440` / `sim/bridge.py:6048`
  (`input_divisive_norm`); `couple_gate_to_pool` / `set_transmission_gate` / `cp_transmission_gain` (`sim/bridge.py`);
  the BG WTA (`research/runners/g11_bg_runner.py`, `--bg-lateral-inhibition`, catalog A.04); `inject_explicit_wiring`
  (`sim/bridge.py:2196`, the Option-A co-residence path).

_Read-only deployment-scoping deliverable. No code written, no experiments run, no GPU. Every cited file:line + finding
verified against the source; the K=32 failure mode was extracted from the committed S2 raw at the seed level (a
true-match-rate margin squeeze, NOT a moat or cleanup failure; moat 0-FA held at K=32). The no-confab moat is the HARD
gate in every proposed stage and is NEVER weakened by any design here._
