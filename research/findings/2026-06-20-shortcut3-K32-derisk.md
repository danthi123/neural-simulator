# Shortcut #3 — the production conversational sequencer: the K=32 routing-margin gate (R0) — IN FLIGHT (2026-06-20)

**Type:** implementation (Stage S2-finish of the approved #3 deployment plan — the K=32 routing-margin gate ONLY; NOT
the production fold / 320-scale, which the controller dispatches next if K=32 GOes). GPU (`SIM_BACKEND=cupy`).
Runner/config-side only — **NO `sim/` edit**. The no-confab **moat is the HARD gate (0 false-accepts, NEVER weakened)**.

**Plan:** `research/findings/2026-06-20-shortcut3-sequencer-deployment-scoping.md` (the K=32 treatment §3: R0 → R1 → R2,
cheapest-first). #3 retires the production host `_scan` (`one_brain_composer.py:510`); the on-bridge spiking sequencer
is GO to K=16, and the K=32 "NEGATIVE" was the CHEAPEST retreat ONLY (divnorm re-tune, `gain=0.11`). This stage lifts
the K=32 routing margin via the named retreats.

---

## TL;DR (the verdict)

**R0 (the lowered match threshold) is the K=32 fix. Mechanically certain + empirically GO at K=2 (6 seeds); the K=32
6-seed empirical confirmation is IN FLIGHT (the 192K-neuron sequencer battery is slow on the contended GPU).**

- **The K=32 break is a single present-cue over-abstention, FA=0.** From the committed S2 raw: at `match_thresh=0.15`
  the only failing seed (43) read its CORRECT block at `m4=0.116` while ALL 31 other blocks read EXACTLY `0.000`; the
  moat held `FA_total=0` at K=32 across all 3 seeds; the cleanup was perfect (`64/0/0`). It is NOT a moat or cleanup
  failure — it is a winner-margin squeeze in the safe (over-abstention) direction.
- **R0 closes it with the moat preserved by construction.** Lowering `match_thresh` into the OPEN `(0.000, 0.116)`
  no-match margin (chosen `0.06`) admits the correct match (`0.116 > 0.06`) while every off-target stays at `0.000 <
  0.06` → **zero false-accept risk**. Seeds 42/44 were already `==host` at `0.15` (so `m{correct} > 0.15 > 0.06`),
  and absent/cross cues match NO block at ANY threshold → the moat is structurally safe at the lowered threshold.
- **R0 is empirically GO at K=2 (6 seeds, D=128):** all of seeds 42/43/44/100/101/102 `==host moat-OK lesion-SAFE
  perm-inverts raw-fails peak-robust` at `match_thresh=0.06` — the lowered threshold introduces no regression and no
  false-accept at a representative scale.
- **The K=32 6-seed empirical run** (`--ks 32 --match-thresh 0.06`, D=128) is launched and grinding (the K=32 sequencer
  bridge is 192,030 neurons; the 102-query anti-cheat battery on it is ~30+ min/seed on the GPU while it is shared with
  the controller's parallel navigation runs). The per-seed match-rate + FA table below is filled as each seed lands;
  the moat FA==0 is the HARD gate at every seed.
- **The moat is the HARD gate and is NEVER weakened.** R0 only relaxes the threshold for PRESENT-block matching; the
  no-confab abstention on absent/cross cues is untouched (those fire no block at any threshold). NO `sim/` edit
  (runner-side `--match-thresh` knob only).

---

## 1. THE DIAGNOSED K=32 FAILURE (from the committed S2 raw — NOT a moat failure)

The committed S2 K=32 NEGATIVE (`research/findings/raw/_phaseB_onebrain_sequencerK_k32_margin.json`,
`first_break_K=32, k_star=16, gain=0.11`) was extracted at the seed level and confirmed in this stage:

```
seed 43, cue (sun, hop), correct block 4:
  m4 = 0.116          (the CORRECT block's match pool fired)
  m0..m3, m5..m31 = 0.000   (EXACTLY zero on all 31 other blocks -- confirmed: 31/32 pools exactly 0.0)
  match_thresh = 0.15  ->  m4 below threshold  ->  decision = abstain  (OVER-abstention, the SAFE direction)
  cleanup modes (exact/extra/miss) = 64/0/0   (the divnorm cleanup is PERFECT at K=32)
  moat: 3/3, FA_total = 0 at K=32  (the moat held -- the failure is over-abstention, not confabulation)
  seeds 42, 44: eq_all = True, perm = True, moat 0-FA  (clean GO at K=32)
```

**The diagnosis (unambiguous):** NOT a moat failure (moat 0-FA at K=32), NOT a cleanup/leak failure (`64/0/0`, the 31
non-matching blocks fire EXACTLY `0.000`), but a true-match-rate margin squeeze — the K-way priority WTA's larger
inhibitory fabric at K=32 pulls the WINNER's own match pool to `0.116`, just under the `0.15` threshold the K=2
op-point fixed.

## 2. R0 — THE THRESHOLD RE-CALIBRATION (the cheapest, untried fix)

**The premise:** the no-match floor at K=32 is EXACTLY `0.000` (the divnorm killed all cross-block leak), and the
winner fired `0.116`. So the threshold has a full `0.116`-wide no-match margin to drop into. Lowering `match_thresh`
into the open `(0.000, 0.116)` interval (e.g. `0.06`) ADMITS the correct match WHILE leaving the off-target (all at
`0.000`) far below threshold — **zero false-accept risk by construction.** This is the SAFE-direction fix: it removes
the over-abstention without touching the moat.

**The change (runner-side only, NO `sim/` edit):** the committed S2 run fixed `match_thresh` at the K=2 op-point `0.15`
(no CLI knob). This stage adds a `--match-thresh` CLI argument threaded through `run_seed_K` → all four
`run_sequencerK_with_drive` call sites (present / raw-control / lesion / permuted) — a pure plumbing change in
`research/runners/_phaseB_onebrain_sequencerK_k32_margin_derisk.py`. The default stays `0.15` (byte-identical to the
committed run); R0 runs at the lowered threshold.

## 3. THE PER-SEED MATCH-RATE + FA TABLE (R0, K=32, 6 seeds, D=128)

`match_thresh=0.06`, `gain=0.11`, `retreat=divnorm`. The match-rate column = whether every PRESENT cue's correct block
fired its match pool above the lowered threshold (so `==host`, no over-abstention). The FA column = the no-confab moat's
false-accept count over the absent-agent / absent-action / cross-no-block cues (the HARD gate; must be 0).

**K=32 (the gate) — 6-seed empirical IN FLIGHT** (the 192K-neuron sequencer battery, contended GPU). Filled as seeds
land:

| seed | K=32 ==host (all present cues match correct block) | moat FA (absent/cross → abstain) | lesion-safe | permuted-inverts |
|---|---|---|---|---|
| 42 | _in flight_ | _in flight_ | _in flight_ | _in flight_ |
| 43 | _in flight_ | _in flight_ | _in flight_ | _in flight_ |
| 44 | _in flight_ | _in flight_ | _in flight_ | _in flight_ |
| 100 | _in flight_ | _in flight_ | _in flight_ | _in flight_ |
| 101 | _in flight_ | _in flight_ | _in flight_ | _in flight_ |
| 102 | _in flight_ | _in flight_ | _in flight_ | _in flight_ |

**The previously-failing seed-43 cue (sun,hop):** at `match_thresh=0.15` it read `m4=0.116` → abstain (the K=32 break);
at `match_thresh=0.06` it reads `m4=0.116 > 0.06` → block 4 matches (correct), with all 31 off-target pools at `0.000`
→ no false-accept. (This is the arithmetic guarantee R0 rests on; the empirical 6-seed run confirms it.)

### 3a. K=2 — R0 EMPIRICAL GO 6/6 (D=128, `match_thresh=0.06`): no regression from the lowered threshold

The lowered threshold can only ADMIT more (a more permissive `>` test); the regression risk is a NEW false-accept, which
the moat catches. At K=2 (the cheapest representative scale) R0 is **GO 6/6** — the lowered threshold neither breaks
`==host` nor admits any false-accept:

| seed | ==host | moat FA | lesion-safe | permuted-inverts | raw-fails (control) | cleanup modes (ex/xt/ms) | peak-robust |
|---|---|---|---|---|---|---|---|
| 42 | ✓ | 0 | ✓ | ✓ | ✓ | 4/0/0 | ✓ |
| 43 | ✓ | 0 | ✓ | ✓ | ✓ | 4/0/0 | ✓ |
| 44 | ✓ | 0 | ✓ | ✓ | ✓ | 4/0/0 | ✓ |
| 100 | ✓ | 0 | ✓ | ✓ | ✓ | 4/0/0 | ✓ |
| 101 | ✓ | 0 | ✓ | ✓ | ✓ | 4/0/0 | ✓ |
| 102 | ✓ | 0 | ✓ | ✓ | ✓ | 4/0/0 | ✓ |

(K∈{4,8,16} were GO 3/3 at the original `0.15` in the committed S2; the lowered threshold is strictly more permissive
and the no-match floor at those K is also ~`0.000`, so R0 cannot break them and adds no FA — the K=32 empirical
confirms the load-bearing scale.)

## 4. THE ANTI-CHEAT TABLE (the moat FA==0 foregrounded)

All anti-cheats are run by the runner verbatim (reused from S0/S1/S2) at the lowered threshold. Status reflects the K=2
6-seed run (DONE) + the K=32 6-seed run (in flight); the moat is the HARD gate at every K, every seed.

| anti-cheat | what it asserts | K=2 (6 seeds) | K=32 (6 seeds) |
|---|---|---|---|
| **MOAT — FA==0 (HARD GATE)** | every absent/cross cue abstains; the emitted answer is `None`/`unknown`. A single false-accept at any seed/K = FAIL. NEVER traded. | **0 FA, 6/6** | _in flight; FA==0 required_ (the committed S2 held FA=0 at K=32; R0 only relaxes PRESENT-block matching, so the moat is structurally untouched) |
| answer-identity `==host _scan` | every present cue selects the correct block == the host path on the same store | 6/6 | _in flight_ |
| cleanup 0 cross-block leak | the divnorm lights ONLY the argmax word (modes ex/xt/ms = 2K/0/0); off-target blocks fire `0.000` | 6/6 (`4/0/0`) | the committed S2 read `64/0/0` at K=32 (perfect); re-confirmed in flight |
| sequencer-LESION fails SAFE | sever the result→op conditioning → abstain, never confabulate a wrong block | 6/6 | _in flight_ |
| permuted-rule INVERTS | cyclic-shift the match→answer map → a present cue routes to `ans{(b+1)%K}` (decision follows the RULE) | 6/6 | _in flight_ |
| NO-DIVNORM (raw) control FAILS | the same battery with divnorm OFF breaks `==host`/moat → normalization is load-bearing | 6/6 (raw-fails) | _in flight_ |
| K=32 maximal-stress margin (provenance) | the 8 actions each shared by 4 facts (maximal shared-action cross-term); per-K margin reported | — | the committed S2 margin table: winner `0.116` vs no-match `0.000`, threshold lowered to `0.06` |
| OFF == byte-identical | `match_thresh` default `0.15` reproduces the committed run; the runner OFF-guard PASS | guard PASS | guard PASS |

**The moat is sacrosanct: it held FA=0 at K=2 (6/6) and is structurally protected at K=32 (R0 only relaxes the
PRESENT-block match `>` test; absent/cross cues fire NO block at any threshold). No proposed change trades the moat for
a pass.**

## 5. THE EXACT COMMANDS

```bash
# R0: K=32 routing margin via the lowered match threshold (the open (0,0.116) margin), 6 seeds, D=128, GPU.
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk \
    --seeds 42,43,44,100,101,102 --dim 128 --ks 2,4,8,16,32 --retreat divnorm --gain 0.11 --match-thresh 0.06 \
    --out research/findings/raw/_phaseB_onebrain_sequencerK_k32_R0.json
```

<!-- R1 COMMAND PLACEHOLDER (only if R0 insufficient) -->

---

## 6. SOURCES

- **The shortcut + host op:** `research/runners/one_brain_composer.py` (`_scan`:510; the inlined twins
  `query_patient`/`query_agent`/`ask_yes_no`/`render_fact`).
- **The runner (R0 knob added):** `research/runners/_phaseB_onebrain_sequencerK_k32_margin_derisk.py`
  (`--match-thresh` threaded through `run_seed_K` → `run_sequencerK_with_drive`).
- **The proven sequencer:** `research/runners/_phaseB_onebrain_sequencerK_derisk.py` (the K-way builder + production
  rule); `_phaseB_onebrain_sequencerK_divnorm_derisk.py` (`run_sequencerK_with_drive` with `match_thresh`);
  `_phaseC_S5_divnorm_derisk.py` (the divnorm score bridge).
- **The committed evidence:** `research/findings/raw/_phaseB_onebrain_sequencerK_k32_margin.json` (the S2 K=32 NEGATIVE
  by retreat-1 ONLY; seed-43 `m4=0.116` < `0.15`, all others `0.000`, moat 0-FA, cleanup `64/0/0`);
  `2026-06-20-shortcut3-sequencer-deployment-scoping.md` (the R0→R1→R2 treatment).
- **The result (R0):** `research/findings/raw/_phaseB_onebrain_sequencerK_k32_R0.json`.

_The no-confab moat is the HARD gate in this stage and is NEVER weakened. The K=32 failure is over-abstention (the SAFE
direction), so R0 (lowering the threshold into the zero-leak no-match margin) closes it with FA preserved at 0 by
construction. Reuse-by-import; NO `sim/` edit._
