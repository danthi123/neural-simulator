---
type: finding
status: live
date: 2026-09-02
mechanism: crossedge-surprise-metacog-read-isolation-fix
board: 108 / one-brain integration program (C2 rung)
seeds: [42, 43, 44, 100, 101, 102]
artifact: research/findings/raw/_crossedge_surprise_metacog_readfix_numpy6seed.json
---

# C2 (D2-surprise -> E1-metacog confidence): the READ's incomplete reset was the real defect — fixed, GO 6/6 numpy

**2026-09-02, numpy, 6 seeds (42/43/44/100/101/102), same runner
(`research/runners/_crossedge_surprise_metacog_derisk.py`), `--ablation plain` (the original, DEFAULT rate-Hebbian
mechanism — unchanged).** This closes the C2 residual two prior attempts left open
(`research/findings/2026-09-02-c2-metacog-error-gated-port-second-negative.md`, ⛔ PARTIAL retraction on its
Diagnosis section, see `docs/RETRACTED.md`) by fixing the metacog READ's incomplete reset-to-rest, not the
plasticity rule.

## Background — what two prior levers left unresolved

The original cupy 6-seed harvest (`research/findings/2026-09-02-integration-program-6seed-harvest-singlepool-GO-C1-GO-C2-NOGO.md`,
its own cupy artifact carrying the C2 sub-verdict) found C2 3/6 NO-GO: emergence passed on every
seed, but `interaction.PASS` failed on 3 — the LESIONED pool's own CONTRADICT-vs-CONFIRM read gap
(`delta_lesion`), which should sit near zero once the cross-edge is zeroed, instead exceeded the
`lesion_ratio=0.34` bound on some seeds. A first lever (plain rate-Hebbian, unchanged) was the original mechanism
itself. A second lever (porting sibling C1's error-gated three-factor update) REGRESSED robustness (6/6 -> 3/6 on
a controlled numpy A/B) rather than fixing it, and diagnosed the residual as a **lesion-baseline confound**: "the
random patient_asserted block's own non-learned connectivity leaking into the metacog read" — a hypothesized
STRUCTURAL cross-region synapse independent of the cross-edge.

## The real defect — found by direct instrumentation, not by another rule change

That structural hypothesis was tested directly and REFUTED. Enumerating every synapse in the built bridge between
`{cue, patient_expected, patient_asserted, surprise}` and `{member0, member1}`
(`np.isin(row, region_idx) & np.isin(col, member_idx)` over `cp_connections.tocoo()`, minus the declared
cross-edge mask) found **zero** — the entire `surprise -> member0/1` connectivity is the one declared `CrossEdge`
(23040 synapses, all inside the cross-edge mask). There is no base-connectivity leak to normalize against.

Snapshotting every `cp_*` bridge attribute immediately before and after one `read_confidence()` call, on an
otherwise-untouched pool, isolated the actual cause to 4 arrays `SurpriseMetacogPool._hard_reset()` never
restored: `cp_refractory_timers` and `cp_prev_firing_states` (the HARD firing gates — independent of membrane
potential, so a neuron mid-refractory at the end of one read/episode stays gated at the start of the next even
though `v`/`u` were reset) and `cp_neuron_activity_ema` / `cp_neuron_firing_thresholds` (the homeostatic
per-neuron EMA/threshold, participation-gated so it silently drifts on whichever neurons the immediately-prior
read/episode drove). `_hard_reset()` already restored membrane potential, recovery variable, every conductance,
firing state, and the Hebbian coactivity trace — but not these four.

**Direct proof this is the mechanism (seed 42, cross-edge fully zeroed, no training ever run — so nothing SHOULD
differ between reads):**

<!--derived-->
| check | pre-fix `_hard_reset()` | with the 4 arrays restored |
|---|---|---|
| `read_confidence('low')` called twice in a row | g0/g1 differ: (121.2957, 100.4195) then (120.9968, 101.0344) | identical: (120.0763, 99.3291) both times |
| low-then-high vs high-then-low (order reversed) | `high` reads differently depending which condition ran first: g1=102.10 (forward order) vs g1=85.40 (reversed) — order-DEPENDENT | identical regardless of order: g0=120.0763, g1=99.3291 in both orders |
| untrained + lesioned delta (`high.conf_signed - low.conf_signed`) | -0.00506 (seed 42) — matches the ALREADY-COMMITTED plain-ablation control's `delta_lesion=-0.0051` for seed 42 (`research/findings/raw/_crossedge_surprise_metacog_plain_numpy6seed_control.json`) | **0.0 exactly** |

(Numbers in this table are from ad-hoc instrumentation scripts run against the pre-fix and post-fix code on this
same runner/pool/seed, not from a JSON artifact — the JSON-cited numbers below are the ones the gate scores.)

## The fix — complete the reset, not the rule

Added `_EXTRA_RESET_ARRAYS = ("cp_refractory_timers", "cp_prev_firing_states", "cp_neuron_activity_ema",
"cp_neuron_firing_thresholds")` to `research/runners/_crossedge_surprise_metacog_derisk.py`. `__init__` snapshots
these 4 arrays at the SAME true-rest point `rest_v`/`rest_u` are already captured (after the 40-step zero-input
settle); `_hard_reset()` now restores all 4 to that snapshot, in addition to what it already reset. This is a
correction to what `_hard_reset()`'s own existing comment already claimed it did ("so each read/train starts from
the same substrate resting state"), not a new mechanism — it touches only this runner's own harness code (not
`sim/`), leaves the pool's wiring/construction untouched (`byte_off` is unaffected), and does not change the
plasticity rule (`--ablation plain`, the unconditional rate-Hebbian train(), is unchanged). Per the BRAIN-BASED
standard, this is legitimate: it is not a host-side correction of the metacog margin's OUTPUT value, it is making
the substrate the read is taken FROM start from a genuinely clean, identical state each time, as the harness
already intended.

## The result — 6/6 GO, `delta_lesion` exactly 0.0 on every seed

<!--derived-->
| seed | `drop_intact` (signed) | `drop_lesion` (signed) | `frac_attributable` | edge grown | GO |
|---|---|---|---|---|---|
| 42  | -0.1041 | 0.0 | 1.0 | 0.598 | true |
| 43  | -0.1151 | 0.0 | 1.0 | 0.604 | true |
| 44  | -0.1321 | 0.0 | 1.0 | 0.591 | true |
| 100 | -0.1366 | 0.0 | 1.0 | 0.561 | true |
| 101 | -0.0992 | 0.0 | 1.0 | 0.585 | true |
| 102 | -0.1071 | 0.0 | 1.0 | 0.627 | true |

(all from `research/findings/raw/_crossedge_surprise_metacog_readfix_numpy6seed.json`, `runs[*].confidence.{drop_intact,drop_lesion}`,
`runs[*].interaction.per_condition.high.frac_attributable`, `runs[*].emergence.grown.surprise_to_metacog_conf`,
`runs[*].GO`.)

**`GO: true`, `n_go: 6/6`** (`payload.GO`, `payload.n_go`, same artifact). `emergence.PASS` and `byte_off.PASS`
are true on all 6 seeds (unchanged from the pre-fix runs — this fix does not touch emergence or wiring).
`interaction.PASS` — the part that failed 3/6 on the original cupy NO-GO — is now true on all 6: every seed's
`drop_lesion` reads exactly `0.0` (not merely small) and `frac_attributable` is exactly `1.0` (100% of the
intact-vs-control shift is attributable to the cross-edge, 0% present in the lesioned control). The anti-cheat
(`other_block_w0_after < 5.0*W0`, unaffected by this fix) holds on all 6 seeds unchanged.

## Verdict — a real defect found and fixed, correcting a prior mischaracterization

This is a GO on the C2 cross-edge's numpy interaction gate (`--ablation plain`, unchanged mechanism), achieved by
fixing the READ's isolation rather than the plasticity rule — matching both this task's own framing and board
#82's independent conclusion ("fix the READ, not the write/separator"). It also corrects the prior finding's
specific diagnosis: there is no structural base-connectivity leak from a randomly-assigned block; the confound
was an incomplete reset primitive producing an order-dependent state leak between conditions. `docs/RETRACTED.md`
carries a PARTIAL retraction of that finding's Diagnosis section; its measured plain-vs-gated 6-seed comparison
numbers are unaffected and still stand.

**NOT established here:** a cupy confirmation (queued, see below — numpy and cupy have diverged before on this
exact edge, hence PARTIAL pending it) and production wiring (this remains a de-risk runner, no `sim/` edit, no
default flip). The fix is additive to the runner only; `--ablation gated` (the tested-and-rejected error-gated
port) is unaffected by this change and remains banked, not recommended, per the prior finding's own verdict.
