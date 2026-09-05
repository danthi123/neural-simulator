---
status: live
type: finding
lane: laneC
date: 2026-09-05
integration_faculty: gnw-global-stop
---

# GNW STOP-trigger ACC/BG circuit — a spiking hyperdirect (ACC->STN->GPi) decision reading the SAME n_ignited/mm_peak afferents retires the host boolean-OR that decides WHETHER the (already-genuinely-spiking) global-workspace STOP fires. 6/6-seed GO on parity with the host decision, both afferents independently load-bearing, full lesion reverts to no-stop. De-risked + wired behind a DEFAULT-OFF flag (`BRAIN_GNW_STOP_TRIGGER_SPIKING`); flag-off byte-identical, 6/6.

**Date:** 2026-09-05 · **Backend:** CPU (numpy) · **Verdict:** **GO** (de-risk + production-hook-dispatch level, 6/6 seeds each) · **No `sim/` edit** (`git diff sim/` empty) · FUNCTIONAL correlate only; NO phenomenal claim.

**Files:** `research/runners/_gnw_acc_bg_stop_trigger_derisk.py` (NEW — the circuit build `build_stop_trigger_bridge`, the trial runner `run_trigger_trial`, the real-afferent readers `get_real_n_ignited`/`get_real_mm_peak`, `evaluate_seed`), `webapp/gnw_acc_bg_stop_trigger.py` (NEW — the default-OFF production glue: `_TriggerCircuit`, `detect_trigger_spiking`, `stop_trigger_spiking_enabled`, `stop_trigger_lesion_on`), `webapp/gnw_global_stop.py` (one additive branch at the top of `detect_trigger`, guarded by `stop_trigger_spiking_enabled()`), `research/runners/_gnw_acc_bg_stop_trigger_hook_verify.py` (NEW — the dispatch/byte-identical verification). **Artifacts:** `research/findings/raw/_gnw_acc_bg_stop_trigger_6seed.json` (the circuit GO gate), `research/findings/raw/_gnw_acc_bg_stop_trigger_hook_verify.json` (the production-dispatch verification).

**Builds on / reuses:** `research/runners/_gnw_rung_stn_stop_veto_derisk.py` (the ACC->STN->GPi hyperdirect chain shape and its biology citations — Frank 2006 "hold-your-horses"; Aron & Poldrack 2006 / Wessel & Aron 2017 broad fast STN reactive stop; Wei-Rubin-Wang 2015 STN-GPe dynamics — reused verbatim, NOT re-researched); `webapp/gnw_deliberation.py` (`conflict_gate`, the SAME `n_ignited` production reads); `webapp/gnw_thought_swap.py` (`ThoughtSwapWorkspace`, the SAME `mm_peak` mismatch-population firing production reads); `webapp/gnw_global_stop.py` (the STOP this trigger feeds — its clear/effector mechanism is UNTOUCHED by this finding). Scoped by `research/coordination/scaffold_retirement_backlog.md` rank-12 ("GNW STOP host boolean-OR").

## What this retires — the trigger, not the clear

`webapp/gnw_global_stop.py`'s `detect_trigger(chat)` decided whether to run the STOP with plain host Python:

```python
if isinstance(n_ign, (int, float)) and int(n_ign) >= 2: triggered = True     # delib conflict
if bool(swap.get("swapped")): triggered = True                              # topic-break
```

`n_ignited` (the deliberation `acc_conflict_gate`'s own spiking read of workspace co-ignition) and the swap
detector's mismatch-population firing (`mm_peak`) are EACH already genuine spiking read-outs of other organs — the
COMBINATION was not. The STOP's own CLEAR (the Tsodyks-Markram shared-recurrence depression, `WorkspaceDepression`
/ `run_conflict_stop`) and the deliberation's own re-entrant RETRY cycle count are ALREADY genuinely spiking and are
untouched here; this finding targets ONLY the trigger comparison, exactly as the roadmap scoped it.

## The retirement mechanism — an ACC/BG hyperdirect circuit reading the afferents as synaptic input

Two small feedforward relay pools (`delib_aff`, `mm_aff`, 40 excitatory Izhikevich neurons each, no internal
recurrence) are driven by a host-scaled current representing the SAME two afferents (`i_delib = 60 * n_ignited`,
`i_mm = 700 * mm_peak`) — the unavoidable stimulus-injection step every drive in this codebase uses (there is no
other channel to introduce a value into a spiking population). Both project via dense frozen `E_TO_E` synapses onto
a shared `acc` pool (80 neurons), which projects to `stn` (120) then `gpi` (200) — the reused hyperdirect chain.
ACC receives BOTH conflict signals as genuine synaptic input from two independent upstream populations and sums
them in its own membrane dynamics; the STOP-TRIGGER verdict is read off GPi's late-window firing rate crossing a
fixed threshold (`GPI_TRIGGER_THRESH=0.05`) — a rate→boolean read-out of the circuit's own spiking integration, the
same class of read-out every other spiking decision in this codebase already uses (`_ignited`, `acc_conflict_gate`).

This is a DE-RISK, not an effector: GPi has no downstream target here (the actual clearing stays the distributed-
overwrite workspace's own depression, unmodified). Reusing the ACC→STN→GPi SENSOR shape — already GO at the
sensor/selectivity level in the STN-veto de-risk — sidesteps that de-risk's own effector NO-GO by construction:
there is no localist attractor here that GPi needs to (and fails to) clear.

## Calibration — the veto file's ACC_STN_W/STN_GPI_W (25/25) saturate GPi's discriminating power here

Reusing the STN-veto de-risk's exact weights verbatim initially produced GARBAGE parity (1-2/3, non-monotonic
sweeps): a direct probe (`i_acc` injected straight into ACC, bypassing the afferent pools) showed GPi's rate
pinned at ~0.17-0.21 for ANY nonzero ACC/STN activity at all, regardless of magnitude — `stn_gpi_w=25` over a
dense 120→200 all-to-all fan-in saturates GPi from even a handful of firing STN neurons. That de-risk never needed
GPi's own rate to discriminate (it only needed GPi's DOWNSTREAM effect through a weak `gpi_ws_w=8` weight), so it
never had to avoid this. Reducing `stn_gpi_w` to 8.0 restores a clean separation (probed: baseline rate_gpi=0.000,
conflict-level rate_gpi~0.10-0.15). A second, independent miscalibration was OU noise: 20-30 pA (the value tuned
elsewhere to desynchronize a RECURRENT ATTRACTOR) drove enough noise-only ACC/STN/GPi firing that the dense
feedforward fan-in AMPLIFIED it into a substantial spurious baseline (rate_gpi~0.17 at zero afferent drive) —
this circuit has no recurrence anywhere, so there is no attractor to desynchronize, and `ou_noise_pA=5` removes the
artifact (probed byte-for-byte: baseline rate_gpi=0.000) while keeping a small background-noise realism. A third
recalibration: the relay pools' own knee sits near 75-100 pA of DIRECT injected current (probed), ~30-40x lower
than `IGNITE_PA=2500` (which ignites a RECURRENTLY-AMPLIFIED 80-neuron assembly, not a bare feedforward 40-neuron
pool) — `DELIB_CURRENT_SCALE=60`/`MM_CURRENT_SCALE=700` place a real solo `n_ignited=1` / matched `mm_peak` below
that knee and a real co-ignited `n_ignited=2` / mismatch-fired `mm_peak` clearly above it.

## GO GATE — the circuit (6 seeds 42/43/44/100/101/102, ALL hold)

Afferents are READ from the ACTUAL already-existing organs, not hand-picked: `webapp.gnw_deliberation.conflict_gate`
(the same function `chat._last_gnw_delib` is built from) for `n_ignited`, and a real
`webapp.gnw_thought_swap.ThoughtSwapWorkspace` (the same class `chat._last_swap_drives` is built from) for `mm_peak`.
mm_peak values below are rounded to 3 decimals for display; the exact per-seed floats are in the cited artifact's
`real_afferents.mm_peak_match` / `mm_peak_mismatch` fields.
<!--derived-->

| seed | real n_ignited (solo→conflict) | real mm_peak (match→mismatch) | PARITY vs host | delib-alone flips | mm-alone flips | full lesion reverts | per-pathway independent | determinism |
|---|---|---|---|---|---|---|---|---|
| 42  | 1→2 | 0.067→0.306 | 3/3 | yes | yes | yes | yes | yes |
| 43  | 1→2 | 0.044→0.294 | 3/3 | yes | yes | yes | yes | yes |
| 44  | 1→2 | 0.000→0.278 | 3/3 | yes | yes | yes | yes | yes |
| 100 | 1→2 | 0.022→0.300 | 3/3 | yes | yes | yes | yes | yes |
| 101 | 1→2 | 0.017→0.289 | 3/3 | yes | yes | yes | yes | yes |
| 102 | 1→2 | 0.056→0.278 | 3/3 | yes | yes | yes | yes | yes |

`parity 18/18 (3/3 × 6 seeds) · delib_sweep_flips 6/6 · mm_sweep_flips 6/6 · full-afferent-lesion reverts 6/6 ·
per-pathway-lesion independence 6/6 · determinism 6/6`. GO gate (per seed): parity >=2/3 (mirrors the
2026-08-26 flip-soak's own >=5/6-of-a-graded-readout bar), both sweeps flip OFF→ON, the full lesion reverts on
BOTH real-conflict scenarios, EACH afferent pathway alone remains independently sufficient, and build-twice
determinism holds. `attributable_to` credits 100% of the intact-vs-lesion GPi conflict-drive difference to the
afferent→ACC pathway on every seed (the lesion zeroes ONLY that link; the ACC→STN→GPi chain is untouched by it, so
a changed verdict is attributable to the AFFERENT pathway specifically, not a global kill-switch). <!--derived-->

## Load-bearing — both afferents independently, not just jointly

Each afferent ALONE flips the trigger from OFF to ON across its own range, holding the OTHER afferent at its
"no-conflict" baseline (n_ignited sweep 1→4 at a fixed matched `mm_peak`; `mm_peak` sweep match→mismatch at a fixed
solo `n_ignited=1`), on all 6 seeds. Zeroing BOTH afferent→ACC synapses (`afferent_lesion=True`) makes the trigger
NEVER fire on the SAME real conflict-indicating afferents that trigger it when intact — full reversion to "no
stop" — on all 6 seeds. Zeroing EACH pathway INDIVIDUALLY (`delib_lesion` / `mm_lesion`) leaves the OTHER pathway
independently sufficient on all 6 seeds: this is not one channel dominating a nominally-two-input circuit, both
are genuinely causal.

## GO GATE — the production-hook dispatch (6 seeds, `_gnw_acc_bg_stop_trigger_hook_verify.py`)

The de-risked circuit's own GO gate is not the same claim as "the production wire-in behaves correctly" (TERMS.md:
`wired` requires a call path from `/api/brain-chat`, and `byte-identical` must be asserted in the data). A second,
narrower verification exercises the ACTUAL `webapp.gnw_global_stop.detect_trigger` entry point on fake chats:

| seed | flag-off byte-identical vs frozen original logic | flag-on dispatch == direct call | flag-on real-afferent match vs host | lesion-via-flag reverts |
|---|---|---|---|---|
| 42–102 (all 6) | yes | yes | 3/3 | yes |

`flag_off_byte_identical 6/6 · dispatch_ok 6/6 · real_match 18/18 · lesion_reverts 6/6`. The flag-off path is
compared, per seed, against a FROZEN copy of the pre-edit boolean-OR (a tuple compare in the data, not inferred
from reading the diff) — confirming the added branch in `detect_trigger` never executes when
`BRAIN_GNW_STOP_TRIGGER_SPIKING` is unset. Re-running the PRE-EXISTING `_gnw_global_stop_flip_soak.py` (6/6 seeds,
unrelated to this change's flag) after this edit still returns GO 6/6 — no regression on the already-shipped
default-ON STOP clear. <!--derived-->

## Contract (additive, DEFAULT-OFF, reversible)

`BRAIN_GNW_STOP_TRIGGER_SPIKING` unset/0/false/off/no (DEFAULT) → `detect_trigger` runs its ORIGINAL, unmodified
host boolean-OR; this module is never imported into that path. An explicit 1/true/on/yes → `detect_trigger`
delegates the BOOLEAN decision to `detect_trigger_spiking`; `n_held`/`newcomer` (which content to clear / how to
label it — host bookkeeping, not the retired decision) are computed identically either way.
`BRAIN_GNW_STOP_TRIGGER_LESION=1` zeroes both afferent→ACC synapses in production exactly as the de-risk's own
`afferent_lesion` lever does standalone. The circuit runs on its own private RNG timeline (snapshotted/restored
around every read, the #77/#85/gnw-global-stop pattern) so enabling it cannot perturb the other response fields.
This is a DE-RISK + WIRED-BEHIND-A-FLAG landing, not a default-on flip — the parent decides whether/when to flip
`BRAIN_GNW_STOP_TRIGGER_SPIKING` on, per this codebase's standing de-risk → flip-soak → flip sequence.

## Honest residuals (named, not claimed closed)

1. The scalar→current CONVERSION (`i_delib = 60 * n_ignited`, `i_mm = 700 * mm_peak`) is host arithmetic — the SAME
   accepted "afferent drive" pattern the STN-veto sensor's own margin→i_acc conversion uses (its named residual
   #1). What moved from host to spiking is the COMBINATION (the OR), not this unavoidable stimulus-injection step.
2. The GPi-rate→boolean read-out threshold (`GPI_TRIGGER_THRESH=0.05`) is a fixed host constant — the same class
   of read-out every spiking decision in this codebase already uses (`_ignited`'s fraction-of-plateau threshold,
   `acc_conflict_gate`'s theta), not a re-hidden `if`.
3. `delib_aff`/`mm_aff`/`acc`/`stn`/`gpi` are hand-wired dense frozen populations (explicit wiring), not
   self-organized — inherited from the STN-veto de-risk's own residual #2.
4. This lands DEFAULT-OFF. The already-shipped STOP *clear* (`gnw-global-stop`, default-ON since 2026-08-26) is
   completely unaffected while the flag is off (verified 6/6, and the pre-existing flip-soak still passes 6/6
   after this edit) — this finding does not change the production ledger row's `scaffold_retired`/`on_by_default`
   status, which correctly still names the clearing-lead STRING template and the STD's conflict-boost-magnitude
   scaling (a DIFFERENT residual than the trigger, out of scope here) as the remaining host scaffolds for that row.
5. `n_held`/`newcomer` (how many held contents to clear, and their label) remain host bookkeeping in BOTH the
   original and the spiking-trigger path — only the BOOLEAN "should we stop" decision was in scope for rank-12.

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._gnw_acc_bg_stop_trigger_derisk \
    --seeds 42 43 44 100 101 102 --json research/findings/raw/_gnw_acc_bg_stop_trigger_6seed.json

SIM_BACKEND=numpy python -u -m research.runners._gnw_acc_bg_stop_trigger_hook_verify \
    --seeds 42 43 44 100 101 102 --json research/findings/raw/_gnw_acc_bg_stop_trigger_hook_verify.json
```
