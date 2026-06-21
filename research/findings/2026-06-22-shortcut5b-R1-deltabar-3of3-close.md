# Shortcut #5b R1 — R1 (afferent selectivity) CLOSED 3/3; the secondary SNc-burst δ over-clamp is a precisely-characterized substrate boundary (2026-06-22)

**Task:** finish the #5b R1 close — per `2026-06-22-shortcut5b-R1-grid-frontend-derisk.md` (`1cd8bd66`) the
grid-cell front end already SURPASSED R1 (the value grades 3/3: V near/far 4.45× / 12.29× / 4.66× on seeds
42/43/44 vs the render's 1.0× R1-cap; place selectivity on real spikes place cos 0.137). The ONLY residual
was the secondary **SNc-burst-gap δ holding 2/3** — on seed 44 a stronger place volley over-fires the critic
(~257 Hz) → the SNc over-clamps (the GABA_B/GIRK conductance saturates → the SNc is silenced at BOTH near
AND far → 0/0) → that δ read inverts to 0.0. Two probe-level next moves were named: **(a) a graded-V-only δ
read** and **(b) a settling phase**.

## VERDICT — **#5b R1 (afferent selectivity) is CLOSED 3/3 (GO); the secondary SNc-burst δ over-clamp on seed 44 is a precisely-characterized, substrate-rooted boundary that resists FIVE mechanistically-distinct single-knob fixes — it does NOT reopen R1.**

**R1 — the wall the host-Gaussian `vs_place_context` scaffold was holding the line for — is broken 3/3.**
The grid front end converts the locally-degenerate render afferent (adjacent cos 0.99) into a decorrelated
metric → the self-org place pool carves locally-selective fields (real spikes, place cos 0.137) → the
SHIPPED graded plateau read-out grades the value selectively on **every** seed (V n/f **4.35 / 13.40 /
5.04×** this session, vs the render's ~1.0× R1-cap), INCLUDING the over-clamp seed 44 (5.04×). That is the
afferent-selectivity residual, and it is closed multi-seed.

**The secondary SNc-burst δ residual is genuinely substrate-rooted, not closable by a single knob, and I
ran it to ground (the SURPASS ISOLATE).** The honest finding is that the prior doc's two named moves do NOT
close it, and neither do three further mechanistically-distinct levers — because the root cause is upstream
of the read: the documented **CuPy place-code volley non-determinism** (the transpose-SpMV atomic scatter,
a 17–292 Hz critic-rate spread across seeds). On seed 44 the strong volley → a strong learned place→value
weight → the stage-B **weighted-plateau READ toggle** (which fires the critic from the LEARNED WEIGHT,
k_threshold in weight units) over-drives the critic to **~290 Hz regardless of any downstream knob** → the
SNc GABA_B subtraction over-clamps. The five levers and why each fails:

| lever | seed 44 | seeds 42/43 | why it fails |
|---|---|---|---|
| **(a) graded-V-only δ** (`delta_vnf` = graded plateau cond. near/far) | passes 5.6× | — | **CONTAMINATED**: the graded plateau grades with the FIXED structural place→value weights even WITHOUT learning, so the scramble LESION arm also grades it (`delta_vnf` scramble = 2.72 / **70.06** / 0.24 across seeds) → it does NOT collapse for the lesion → fails the scramble control. Not a valid δ. |
| **(b) settling window** (settle_steps=200 before the gap read) | still 0.0 | — | **INSUFFICIENT**: the over-clamp is regime-structural (the strong graded V over-drives the critic for as long as place is on), not a transient the critic relaxes out of. |
| **(c) fixed GIRK-cap** (`critic_gabab_max=1.0`, bounds g_gabab) | **FIXES** → gap 2.0, gabab_gap True | **BREAKS** → 42 gap 6.67→1.0, 43 True→False | **TRADES seeds**: the gentle seeds need g_gabab UNBOUNDED to subtract at their low critic rate; the hot seed needs it BOUNDED to not over-clamp. No single cap reconciles the opposite requirements. |
| **(d) critic-rate homeostasis** (fast-EMA intrinsic-threshold adaptation) | still 258 Hz / 0.0 | **PRESERVES** → 42 gap 5.0, 43 True | preserves the gentle seeds (unlike the cap) but **cannot pull down seed-44's READ-time rate** — the weighted-plateau toggle re-over-drives it regardless of the trained threshold (= the prior doc's Attempt 2). |
| **(e) lower graded-plateau strength** (the V_near magnitude at source) | strength 80/40: still 292 Hz / 0.0; strength 15 (V_near→16): **FIXES** → crit 66 Hz, gabab_gap True | strength 15 **BREAKS** → 42 crit→0 Hz (doesn't fire), gabab_gap False | **TRADES seeds (the same trade as the cap)**: a high strength is needed for the gentle seed's critic to fire AT ALL, but it over-drives the hot seed; a low strength un-clamps the hot seed but the gentle seed's critic then can't fire → no GABA_B → no gap. Opposite requirements, irreconcilable. (At intermediate strength 25, seed 44's authoritative read still over-clamps but the SETTLED-regime read `gabab_gap_graded` starts to recover — a promising lead for a settled-read close, below.) |

**Root cause (robust, the SURPASS ISOLATE):** the seed-44 place volley is so strong (the documented
non-determinism) that the stage-B weighted-plateau READ over-drives the critic into the over-clamp regime;
the gentle seeds need the OPPOSITE regime (enough drive for their critic to fire at all). The two ends of
the seed-variable critic-rate spread (17–292 Hz) require opposite settings of EVERY global knob tried
(g_gabab cap, graded strength, homeostasis target), so no single global knob holds the δ 3/3 — it always
trades the gentle seeds for the hot seed or vice-versa. **R1 is UNAFFECTED throughout (V n/f stays 4.51–5.04×
selective on every seed at every knob setting — the place value IS selective; only the SNc somatic readout
saturates on the hot seed).** ⇒ the host-Gaussian scaffold's retirement rests on R1 (the afferent-
selectivity wall, closed 3/3), with the secondary δ-readout over-clamp a precisely-characterized substrate
boundary downstream of it, not an R1 failure.

**The one promising lead (a deeper, non-single-global-knob close, NOT pursued to 3/3 here):** at lower
graded strength the SETTLED-regime read (`gabab_gap_graded`, `coincidence_weighted_drive=False` — NOT the
over-driving weighted-plateau toggle) recovers seed 44's δ (settled `gabab_gap_graded`=True at strength 25;
the authoritative weighted-plateau read still over-clamps there). A **strength + settled-regime read**
combination MIGHT hold 3/3 — but it requires (i) re-validating the settled read collapses for the controls
(it is closer to the graded-V read, so it must be checked it is not similarly structurally contaminated)
and (ii) finding a single strength that fires the gentle-seed critic in the settled regime without breaking
it — i.e. it still faces the gentle-vs-hot tension, just in the settled regime. It is a real next lead, not
a closed 3/3.

**NO `sim/` edit, NO `g11_bg_runner.py` edit.** All five levers are probe-level (reuse-by-import); the
GIRK-cap is the EXISTING `critic_gabab_max` kwarg (default 0 = off = byte-identical). The no-confab moat is
untouched (nav-only probe, array-disjoint from the composer's complex synapses).

---

## The 3-seed picture (R1 fix robust 3/3; the SNc-burst δ holds 2/3, the residual isolated to seed-44)

| seed | V n/f (R1 fix, robust 3/3) | SNc-burst δ gabab_gap (cap=0) | critic@read (Hz) | residual? |
|---|---|---|---|---|
| 42 | **4.35×** | **True** (gap 6.67) | 16.7 (gentle) | — |
| 43 | **13.40×** | **True** (near 0 / far 50) | 66.8 | — |
| 44 | **5.04×** | **False** (0 / 0 over-clamp) | **258–292 (over-fires)** | the isolated δ residual |

R1 (V n/f) is robustly selective on all 3 seeds (the close). The SNc-burst δ is present on the two
moderate-rate seeds and over-clamps only on seed 44, whose critic over-fires at the read.

## The GIRK-cap sweep on seed 44 (the one lever that touches the over-clamp — but trades the gentle seeds)

| `critic_gabab_max` | snc_pred NEAR | snc_unpred FAR | gabab_gap | seed 42 effect | seed 43 effect |
|---|---|---|---|---|---|
| 0 (off) | 0.0 | 0.0 | False (over-clamp) | gap 6.67 (True) | True |
| 0.5 | 30.0 | 25.0 | False (inverts) | — | — |
| **1.0** | **12.5** | **25.0** | **True** (gap 2.0) | gap 6.67→**1.0 (BROKEN)** | True→**False (BROKEN)** |

The cap that fixes seed 44 (1.0) removes the gentle-seed subtraction (their g_gabab is capped below the
level needed to subtract at their low critic rate). The opposite-regime requirement is irreducible.

## The control battery on the SNc-burst δ (the LOAD-BEARING discriminator — cap=0, the working seeds)

The SNc-burst δ (gabab_gap) is the only valid δ (the raw graded-V is structurally contaminated, above). It
collapses for every non-functional control because it requires the critic to FIRE and the value to
SUBTRACT:

| arm | SNc-burst δ (snc_gap) | gabab_gap | verdict |
|---|---|---|---|
| **grid** (TEST) | 6.67 (seed 42) | **True** | the RPE δ is present (on the working seeds) |
| **render** (NEGATIVE) | 1.0 | False | collapses (flat) |
| **scramble** (LESION) | 1.0 / 0.0 | False | collapses (critic silent → no learned V) |
| **no_learn** (floor) | 1.0 | False | collapses (no value-train) |
| **lesion** (graded OFF) | 1.0 | False | collapses (graded read-out off → V=0) |
| LESION (g_gabab mask zeroed) | ≤1.15 | — | collapses to ~1.0 (the in-arm anti-cheat) |

(The raw graded-V `delta_vnf` does NOT pass this battery — the scramble grades it up to 70× because the
graded plateau grades with the fixed structural weights, no learning required.)

## V n/f selectivity (the R1 fix itself) — reconfirmed 3/3 this session

| seed | grid V n/f (graded) | render V n/f (NEGATIVE) |
|---|---|---|
| 42 | 4.35× | 1.13× |
| 43 | 13.40× | 0.94× |
| 44 | 5.04× | 0.53× |

The grid front end converts the locally-degenerate render afferent into a selective place value on every
seed — the R1 fix, robust across all 3 seeds.

## sim/-edit + moat confirmation

- **NO `sim/` edit, NO `g11_bg_runner.py` edit.** All five δ-fix levers are in the standalone probe
  (`_n5_grid_frontend_onbridge_probe.py`, reuse-by-import): the δ reads (`delta_vnf`, `delta_snc_graded`,
  `delta_gabab`) read the bridge's existing arrays; `--critic-gabab-max` threads the EXISTING
  `critic_gabab_max` kwarg (default 0 = off = byte-identical); `--settle-steps` / `--critic-homeo-*` /
  `--graded-strength` are read-only probe knobs. #6's log-polar work continues to own `g11_bg_runner.py`.
- **The no-confab moat was NEVER weakened.** A nav-only probe (no conversational regions). The place/critic
  state (`cp_connections` / `cp_firing_states` / `cp_conductance_g_graded_plateau` / `cp_conductance_g_gabab`)
  is array-disjoint from the composer's complex `cp_rf_w_*` synapses. Preserved by construction.

## What this means + the next move

- **#5b R1 (the afferent-selectivity wall) is CLOSED 3/3.** The grid-cell front end (catalog D.07, the
  missing medial-EC metric) gives the self-org place pool a decorrelated input → locally-selective fields →
  a robustly selective place value on every seed. The host-Gaussian `vs_place_context` scaffold's retirement
  rests on this — R1 is the wall it was holding the line for, and it is broken multi-seed. Same resolution
  family as the conversation PPMI cortex / the B1 self-org RF: the fix is the right decorrelated INPUT
  representation, NOT point-neuron decorrelation and NOT a dendritic rewrite.
- **The secondary SNc-burst δ over-clamp on seed 44 is a precisely-characterized, substrate-rooted boundary**
  (the documented CuPy place-code volley non-determinism → the strong-volley seed's weighted-plateau read
  over-drives the critic → the SNc subtraction over-clamps; the gentle/hot regimes need opposite g_gabab,
  irreconcilable by a single knob). It does NOT reopen R1 (V n/f stays selective through the over-clamp).
  **The named next move (a deeper, non-single-knob fix, NOT this de-risk):** make the place code itself
  reproducible across draws — the documented `--deterministic-selforg` already fixes the FIELDS, but the
  volley STRENGTH still varies (the transpose-SpMV atomic scatter). A deterministic-scatter SpMV for the
  place→critic matvec (the same fix already applied to the GABA_B/graded-plateau matvecs via
  `deterministic_transpose_matvec`) would normalize the read-time critic rate across seeds, after which a
  single regime (no cap, or a single cap) would hold the δ 3/3. That is a `sim/`-level place-matvec change
  (gated behind the research gate), deliberately NOT taken here while #6 owns `g11_bg_runner.py` and R1 —
  the load-bearing residual — is already closed.
- **The validate-by-function caveat (carried from the prior doc):** the nav δ is INERT on immediate-reward
  nav (the #9 lesson), so neither R1 nor the δ changes navigation itself. The genuine downstream consumer is
  the deferred hidden-goal (Morris-water-maze) actor-critic spatial-credit arc, which uses the SAME `place →
  cortex_action` plastic pathway and where the selective place value IS load-bearing. R1 is now a solved
  afferent on a quantity that gates that arc.

## Files
- `research/runners/_n5_grid_frontend_onbridge_probe.py` — the on-bridge probe + the δ probes
  (`_read_graded_v_delta`: `delta_vnf` / `delta_snc_graded` / `delta_gabab`) + `--critic-gabab-max` (the
  GIRK-cap, threading the existing kwarg) + `--settle-steps` (move b) + the existing `--critic-homeo-*` /
  `--graded-strength` levers.
- `research/findings/raw/_n5_grid_onbridge_gradeddelta_allarms_seed{42,43,44}.json` — the cap=0 all-arms
  battery (R1 3/3; the SNc-burst δ 2/3; the delta_vnf contamination).
- `research/findings/raw/_n5_grid_onbridge_girkcap{0.5,1.0}_seed44.json`,
  `_n5_grid_onbridge_girkcap1.0_seed{42,43}.json` — the GIRK-cap (fixes 44, trades 42/43).
- `research/findings/raw/_n5_grid_onbridge_homeo_e02a01_seed{42,43,44}.json` — the homeostasis (preserves
  42/43, can't pull down 44).
- `research/findings/raw/_n5_grid_onbridge_gstr{40,...}_seed44.json` — the graded-strength sweep (the critic
  over-fires from the weighted-plateau toggle, not the graded current).
- `research/findings/raw/_gradeddelta_verdict.py` — the consolidated verdict printer.
- `research/findings/raw/_run_{gradeddelta_close,girkcap_seed44,girkcap_validate,homeo_settled}.sh` — drivers.

## Reproduce
```bash
# R1 fix (V n/f selective 3/3) + the SNc-burst δ (2/3, the residual on seed 44):
SIM_BACKEND=cupy python -m research.runners._n5_grid_frontend_onbridge_probe --seed 44 --all-arms \
    --readout-only --multi-goal --value-train-trials 40 --grid-drive-scale 2.5 --value-train-w-max 3 \
    --out research/findings/raw/_n5_grid_onbridge_gradeddelta_allarms_seed44.json
# the GIRK-cap (fixes seed 44 but trades the gentle seeds):
SIM_BACKEND=cupy python -m research.runners._n5_grid_frontend_onbridge_probe --seed 44 --arm grid \
    --readout-only --multi-goal --value-train-trials 40 --grid-drive-scale 2.5 --value-train-w-max 3 \
    --critic-gabab-max 1.0
# the consolidated verdict table:
python research/findings/raw/_gradeddelta_verdict.py
```
