# Ceiling-first for the spiking learn-W_in rung: the shipped rate reference had NO headroom (its task is too easy); a matched-difficulty rate map finds the headroom regime (K=30 cues overcomplete in an n=80 reservoir + noise → learn-W_in 0.63 vs fixed 0.25)

**Date:** 2026-07-12
**Status:** ✅ CEILING-FIRST CHECK — headroom regime located (rate level); the spiking learn_win arm is now runnable at a difficulty where a win is *possible*. Reuse-by-import, NO `sim/` edit, NO runner edit.
**Frontier:** the R3-reframe biological long-range path — *fixed spiking reservoir + e-prop-learned INPUT projection W_in (committed `enable_bdsp`) + local read-out* — the SPIKING realization (`_reslm_onbridge_learn_win_derisk.py`). The mechanism was already verified on spikes (BDSP moves W_in, `dw_rec≡0`, no weight transport), but the FUNCTIONAL win (learn beats fixed) was never shown.

## The problem this closes (why the spiking run was uninformative)

The spiking runner's own **rate reference** — the ceiling it uses to bound the spiking result — returned `RATE ref learn 1.000 vs fixed 1.000` at the shipped task config (n_cues=12, dist=3, n_pool=120). **Both arms at ceiling ⇒ zero headroom ⇒ the reference cannot say whether learning W_in helps at all.** Running the expensive spiking arms against a no-headroom ceiling is testing against a task with no signal to capture (the exact "run the ceiling early to bound the arc" discipline the skill now carries — `feedback_run_ceiling_early_and_keep_gpu_busy`).

**Root cause (read the code, don't guess):** the rate reference is DETERMINISTIC + NOISELESS — `build_task` with no jitter makes every example of a cue the *identical* token sequence `[cue_k] FILLER*dist [QUERY]`, and the leaky-tanh reservoir has no noise term. So the 12 cues map to 12 *distinct, fixed* reads that a ridge separates trivially at ANY dist (a fixed random projection of 12 one-hot codes into 120 dims is near-orthogonal; fading only shrinks magnitude, not distinctness). There is no genuine *collision* for learning W_in to overcome. The SPIKING version, in contrast, has real membrane / finite-spike-count noise → genuine collision — so the shipped rate ceiling is **not a like-for-like difficulty proxy** for the spiking task.

## What I did (reuse-by-import, `_reslm_rate_headroom_sweep.py`)

A rate map with **matched difficulty** — same leaky-tanh reservoir + the same input-synapse e-prop rule (broadcast random feedback, no weight transport), plus the difficulty knobs the spiking task actually has: within-class filler **jitter**, reservoir **state noise**, larger **K** (cues), smaller **pool**. Grid: dist ∈ {3,6,10,14,20} × jitter ∈ {0,3} × noise ∈ {0,0.05,0.15} × (n_cues,n_pool) ∈ {(12,120),(30,80)}, seeds 42+43, lr_in 0.05, 10 epochs. Reuses `build_task`; local noisy rate reference; no runner/`sim/` edit.

## Result — headroom OPENS, and it is CODE-CAPACITY (overcomplete) collision, not temporal fading

11/60 configs show genuine headroom (learn − fixed ≥ 0.10 AND fixed ≤ 0.90 = a real collision, not both-at-ceiling). The strongest, most robust:

| n_cues | n_pool | dist | jitter | noise | learn | fixed | margin | chance |
|---|---|---|---|---|---|---|---|---|
| **30** | **80** | **3** | 0 | **0.05** | **0.633** | **0.250** | **+0.383** | 0.033 |
| 30 | 80 | 3 | 3 | 0.05 | 0.667 | 0.283 | +0.383 | 0.033 |
| 12 | 120 | 3 | 3 | 0.15 | 0.250 | 0.042 | +0.208 | 0.083 |
| 12 | 120 | 3 | 3 | 0.05 | 0.917 | 0.750 | +0.167 | 0.083 |

**The decisive regime is K=30 cues in an n=80 reservoir + modest noise, at dist=3 (short).** So the collision is **code-capacity / overcompleteness** (30 codes packed into 80 noisy dims), NOT temporal fading depth — a fixed random W_in cannot keep 30 codes separable in 80 noisy dimensions (fixed 0.25), but a LEARNED W_in organizes them into separable directions (learn 0.63, ~2.5× fixed, both well above chance 0.033). This is exactly the **R3 thesis** — *the input representation is the memory/separation lever* — cleanly isolated from recurrence and from temporal fading (dist=3 is trivially within the reservoir's fading depth; noiseless K=30/n=80 gives fixed=1.000, so the collision is specifically **noise × overcompleteness**, structural, and thus should appear in the spiking reservoir regardless of its exact intrinsic-noise level).

Note the noise-dependence: at noise=0 every config is `no-collision` (fixed at ceiling); noise is what makes the packed codes collide. This matches why the spiking version (intrinsically noisy) should exhibit headroom where the noiseless rate reference did not.

## The spiking check (ran immediately) — the NOISE-based rate headroom did NOT transfer; the real lever is OVERLAPPING CODES

Ran the spiking `fixed_win` vs `learn_win` arms at the headroom difficulty (`--n-cues 30 --n-pool 80 --dist 3`, seed 42, 3 epochs). Result: **fixed 0.933, learn 0.933 — NO headroom on spikes.** The mechanism is clean (dw_win 0.014, **dw_rec 0.000** = recurrence frozen, `no_weight_transport True`, B_apical rises 0→0.027, input-lesion → 0.033 chance, label-scramble → 0.033 chance) — but the spiking reservoir *separates the 30 cues fine* (0.933), so there is nothing for learn-W_in to beat. **The noise-based rate headroom is a poor like-for-like proxy for the spiking regime** (the spiking reservoir's finite-spike-count noise at this operating point did not collide the cues the way rate noise=0.05 did).

**The deeper, decisive read (why NO capacity/noise knob is the right instrument):** the K-cue task is a **deterministic LOOKUP** — each cue is ONE distinct token, so a fixed random W_in gives distinct columns → distinct reads → a ridge separates them at ANY K and ANY dist (distinct random columns never collide deterministically; only *noise* or *shared structure* collides them). This is also why one-hot K-cue can NEVER reproduce the R3 headroom: on the real LM, learning the embedding helps because the same token appears in MANY contexts and the embedding must *organize* tokens so contexts are predictable — a structure-learning problem, not a lookup. **So the faithful, noise-independent instrument is OVERLAPPING SPARSE INPUT CODES** (the named rung): each cue = a sparse code over shared dims, so a fixed random W_in projects overlapping codes to overlapping (confusable) reservoir states, and a LEARNED W_in must actively de-correlate them — structural headroom that transfers to spikes.

## Next concrete action (building now)

**Ceiling-first for the overlapping-code instrument (cheap, rate, noise=0):** confirm that overlapping sparse codes give *structural* (deterministic) learn-vs-fixed headroom at rate level BEFORE building the spiking version. If yes → add an overlapping-code input option to the spiking runner and re-run the `learn_win` vs `fixed_win` gate. Honest scope: this is a *representation-organization* demonstration of learn-W_in (the R3 mechanism), the cleanest transferable isolation; distal (large-dist) decode is a separate rung.

## Files
- `research/runners/_reslm_rate_headroom_sweep.py` — the matched-difficulty headroom map (reuse-by-import).
- `raw/_reslm_headroom.json` — the full grid.
- Builds on: `2026-07-11-R3-REFRAME-...md` (the arc redirect), `2026-07-12-spiking-realization-scoping-...md` (the no-`sim/`-edit scoping).
