---
type: finding
status: no-go
date: 2026-08-07
mechanism: gateB-stage2l-commit-soft-WTA-normalization-refuted
backend: numpy
runner: research/runners/_vocal_gateb_stage2l_commit_normalization.py
builds-on: 2026-08-07-gateB-stage2k-commit-WTA-release-selects-730705-but-cannot-express-at-test.md
artifacts:
  - research/findings/raw/gateb_stage2l_commit_normalization/diag_730705.txt
  - research/findings/raw/gateb_stage2l_commit_normalization/smoke_730705_numpy.json
  - research/findings/raw/gateb_stage2l_commit_normalization/byte_numpy.json
---

# Gate B Stage 2l: a soft (non-latching) cortical commit WTA does NOT flip 730705 — direct measurement shows the thalamic drive itself favors action 0, relocating the residual from the commit competition to the BG-output readout

## Verdict (NO-GO for the commit mechanism; the Stage-2k diagnosis it built on was incomplete)

The Stage-2k finding located the 730705 residual at "a bistable cortical commit WTA that
overrides the BG (thalamic) action-selection signal" and named three commit-level surpasses
(divisive normalisation / thalamus-gated de-latching / reduced bistability). Stage 2l
operationalised the softest possible version of all three — scaling the `commit_fs_c ->
commit_{other}` lateral veto down to **0.0** (a fully de-latched, graded commit) — and
**measured that it does not flip 730705**, because the premise is wrong: on 730705 the
**thalamic drive favors action 0**, so no commit competition that reflects thalamic drive can
select action 1. Authoritative backend = **numpy**. Additive, default-OFF, byte-identical when
off (`--mode byte`, asserted); the Stage-2j/2k GO is unaffected.

## The decisive measurement (raw: `diag_730705.txt`)

After FULL FIX C+D training on 730705 (`proposal_1->str_d1_1` potentiated 40→110, str_d1_1
firing **286** >> str_d1_0 **104** — the D1 policy is correctly learned toward action 1),
the test-phase cascade and every commit-level intervention give:

| intervention (test, FIX D off) | str_d1 | gpi | thal | commit | motor | winner |
|---|---|---|---|---|---|---|
| none | [104, 286] | [0, 29] | **[273, 215]** | [452, 0] | [860, 0] | 0 |
| cut cross-inhib ×0.5 / ×0.25 | [104, 286] | [0, 29] | [273, 215] | [452, 0] | [860, 0] | 0 |
| cut cross-inhib ×0.0 (full de-latch) | [104, 286] | [0, 29] | [273, 215] | [452, **335**] | [860, **646**] | **0** |
| gpi_1 regulation proxy | [104, 286] | [0, 6] | [273, **242**] | [452, 0] | [860, 0] | 0 |
| gpi_1 reg + full de-latch | [104, 286] | [0, 6] | [273, 242] | [452, 384] | [860, **741**] | **0** |
| both gpi→thal removed (thal ceiling) | — | — | **[272, 258]** | — | — | — |

**thal_1 < thal_0 at every operating point.** Even the maximal de-latch lets commit_1 fire
(335) but below commit_0 (452) → motor [860, 646], action 0 still wins and the win is no
longer clean (loser 75% of winner). Even the pure-tonic thalamic ceiling (both `gpi->thal`
removed) is [272, 258] — a residual head-start for channel 0.

## Why thal_1 < thal_0 (the actual residual, quantified)

The correctly-learned striatal D1 policy (str_d1_1 fires ~2× str_d1_0) is **inverted before
the cortex by the BG output stage**:

1. **gpi_1 is heterogeneously hyperexcitable.** It rests at **−40 mV** (σ≈16) vs gpi_0 at
   **−61 mV** (σ≈0.5) and resists pausing: str_d1_1's 286 spikes only bring gpi_1 to 29
   (gpi_0 reaches 0 with a mere 104 str_d1_0 spikes). Even force-silencing gpi_1 caps thal_1
   at ~246 < thal_0 ~270. (GPi is the only heterogeneity-masked region here — 2×20 neurons —
   so this is an unlucky per-seed intrinsic draw, not a wiring asymmetry: `str_d1_c->gpi_c`
   weight sums are equal, 10795 vs 10782.)
2. **A thalamic initial-condition head-start.** Entering the onset window, thal_0 sits primed
   at **−45 mV** (leftover from baseline, when gpi_0 was already pausing) while thal_1 sits at
   **−61 mV** (at rest, clamped by the firing gpi_1). thal_0 therefore crosses first and the
   commit ignites on it. thal is NOT in the heterogeneity mask (k identical 1.6/1.6), so this
   is a dynamical state difference, not an intrinsic-parameter one.

The commit WTA then **faithfully** (if sharply) selects the higher thalamic drive = action 0.
It is not overriding a BG signal for action 1; there is no such signal at the thalamus.

## Contingency preserved (not a shortcut)

The same full de-latch on an untrained (acq_lesion) bridge also stays action 0 (motor
[860, 736] in the Stage-2k-diagnosis probe): the de-latch does not manufacture action 1, it
simply cannot express a policy the thalamus does not carry — consistent with the brain-based-
only bar and the acquisition-lesion attribution.

## Mechanism properties (additive, default-OFF, byte-identical when off — ASSERTED)

`--softwta-scale S` scales the standing `commit_fs_c -> commit_{other}` veto weight; `S=1.0`
(default) is a no-op → byte-identical to Stage 2k, applied via a build-time wrapper around
Stage-2k's own `run_condition` so 2k stays intact and the FIX-C calibration probe bridges are
not wrapped. `_assert_softwta_off_byte_identical` (`--mode byte`, seeds 730703/730705)
verifies `S=1.0` reproduces Stage 2k exactly — measured `all_byte_identical=true` (mismatch
`{}` on both seeds; `byte_numpy.json`), so the Stage-2j/2k GO is protected.

## Smoke result (raw: `smoke_730705_numpy.json`)

`--softwta-scale 0.0` (maximal de-latch, standing, on 730705): `test_rate_c1 = 0.000`,
`count_c1 = [40, 0]`, `D_contingent = 0.0`, `steer = False` — no flip
(`SMOKE_730705_test_rate_c1_flips = false`; Stage-2k base `test_rate_c1 = 0.000`,
`count_c1 = [37, 3]`). Applying the de-latch as a standing training+test property does not
recover action 1 at test, and during training it removes FIX D's release-based selections
(`count_c1` [37,3]→[40,0]) — the seed is unchanged, confirming the commit competition is not
the wall.

## Banked failing method + the new method (no-defer)

BANKED (refuted): a commit-level competition mechanism cannot flip 730705 — it is
NECESSARY-NOT-SUFFICIENT, downstream of the residual. NEW method (Stage 2m, **FIX E**): a
**GPi intrinsic-excitability homeostat** (Desai 1999 / Turrigiano 2011 — the direct analogue
of FIX C but on the GPi output pool) that regulates each GPi channel's baseline excitability
to a common set-point, so gpi_1 pauses fully under D1 drive AND the thalamic entry states
equalize — restoring the striatal D1 policy's expression through the BG output stage, which is
where the signal is actually lost. This addresses the ACTUAL residual rather than the commit
stage downstream of it. (Whether even a perfect GPi homeostat overcomes the residual ~6% thal
head-start is the Stage-2m smoke's verdict, not this note's.)

## Parent validation commands (numpy, orphan-proof)

```bash
export PYTHONPATH=$PWD SIM_BACKEND=numpy
# byte-identity when off (must be all_byte_identical=true -> GO protected):
.venv/bin/python -m research.runners._vocal_gateb_stage2l_commit_normalization --mode byte \
  --out research/findings/raw/gateb_stage2l_commit_normalization/byte_numpy.json
# honest-negative smoke on the held-out miss (SMOKE_730705_test_rate_c1_flips must be false):
.venv/bin/python -m research.runners._vocal_gateb_stage2l_commit_normalization --mode smoke \
  --softwta-scale 0.0 --smoke-seeds 730705 \
  --out research/findings/raw/gateb_stage2l_commit_normalization/smoke_730705_numpy.json
# dev no-regression under the soft-WTA (steer_passes should not fall below Stage-2k):
.venv/bin/python -m research.runners._vocal_gateb_stage2l_commit_normalization --mode seeds \
  --softwta-scale 0.0 --dev-seeds 730601 730602 730603 730604 730605 730606
# evidence dump (the cascade table above):
.venv/bin/python -m research.runners._vocal_gateb_stage2l_commit_normalization --mode diag \
  --diag-seed 730705
```
