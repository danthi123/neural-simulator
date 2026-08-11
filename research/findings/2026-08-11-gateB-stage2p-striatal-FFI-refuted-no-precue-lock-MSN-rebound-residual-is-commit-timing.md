---
type: finding
status: no-go
date: 2026-08-11
mechanism: gateB-stage2p-FIXG-striatal-feedforward-inhibition-downstate-homeostat
backend: numpy
runner: research/runners/_vocal_gateb_stage2p_striatal_ffi_downstate.py
builds-on: 2026-08-07-gateB-stage2o-learning-gated-commit-threshold-boundary.md
grounded-in:
  - research/findings/2026-08-07-gateB-stage2m-bg-output-homeostat-inverts-thalamus-but-necessary-not-sufficient.md
  - research/findings/2026-08-07-gateB-stage2l-commit-normalization-refuted-residual-relocated-to-BG-output-readout.md
  - research/findings/2026-08-07-gateB-stage2o-learning-gated-commit-threshold-boundary.md
artifacts:
  - research/findings/raw/gateb_stage2p_striatal_ffi_downstate/refute_730705.json
  - research/findings/raw/gateb_stage2p_striatal_ffi_downstate/smoke_numpy.json
---

# Gate B Stage 2p: FIX G (a striatal feedforward-inhibition / MSN down-state homeostat) is REFUTED at the mechanism level, and it CORRECTS Stage 2m's diagnosis — there is no str_d1 pre-cue baseline lock, hyperpolarising these MSNs makes them fire MORE (post-inhibitory rebound), and the true residual on 730705 is the commit WTA's ignition-TIMING race, not a striatal firing asymmetry. Gate B holds at ≥5/6.

## Verdict (NO-GO / UNDEFINED — the FFI manipulation never reaches its own mechanism variable)

The Stage-2p hypothesis was the direct STRIATAL analogue of Stage 2m's FIX E: attack the "channel-0-open
lock" at its SOURCE with a target-blind feedforward INHIBITION (a tonic hyperpolarising bias on the
baseline-over-active str_d1 channel, the PV+ FSI / MSN-down-state companion process) — because Stage 2m
recorded the lock as a str_d1 baseline firing asymmetry (str_d1_0 ~86 vs str_d1_1 ~0) that intrinsic-gain
`cp_izh_k` could not silence. Two decisive cheap probes (`--mode refute`, seconds, numpy) refute the premise,
so `tools.verdict.Verdict` returns **UNDEFINED** (`reaches` fails: str_d1 over-channel `before=86 after=86
moved=False`) rather than a fabricated negative. The calibrated FFI bias is **0.0** — no current reduces the
firing — so FIX G is INERT and byte-identical to the Stage-2k base.

## Two corrections to the Stage-2m diagnosis (raw: research/findings/raw/gateb_stage2p_striatal_ffi_downstate/refute_730705.json)

1. **THERE IS NO str_d1 PRE-CUE BASELINE LOCK.** At arousal=FALSE (true, no-cue baseline) `str_d1 = [0, 0]`.
   The "baseline str_d1_0 ~86" is the arousal=TRUE cue ONSET response (`[86, 0]`). So the head-start is not
   set by a quiescent-baseline firing asymmetry; str_d1_0 fires only WHILE the cue drives the striatum.
2. **FEEDFORWARD INHIBITION BACKFIRES ON THESE MSNs.** The IZH2007 striatal MSNs sit in a negative-b regime
   (`cp_izh_b ~ -2`), so a hyperpolarising current does NOT silence the over-active channel — it drives
   POST-INHIBITORY REBOUND and the channel fires MORE: ext `-400 pA -> 86`, `-2000 -> 129`, `-10000 -> 1783`.
   No intrinsic knob moved the 86-spike onset count either (measured: `k*0.1 -> 86`, `d_inc*0.1 -> 85`,
   `a*3 -> 86`, `vpeak+30 -> 84`, `b:=0/-4 -> 86`, `intrinsic_current -200 -> 87`, `ou_sigma=0 -> 86`).
   Feedforward inhibition is therefore the wrong knob for this substrate, and the "not k-reducible" wall of
   Stage 2m is deeper than recorded: it is not-reducible by ANY excitability knob AND not by inhibition.

## The real residual is UNCHANGED and it is DOWNSTREAM (raw: refute_730705.json, smoke_numpy.json)

With the trained policy correct (`str_d1 = [104, 286]`, channel 1 ~2.7x channel 0) and Stage 2m's FIX E
inverting the thalamic aggregate (`thal = [215, 228]`, thal_1 > thal_0), the commit still latches to channel
0: `commit = [388, 0]` (commit_1 NEVER ignites), `motor = [795, 0]` -> action 0, 0/8. `cp_izh_vr` is already
channel-symmetric (gpi -65/-65, thal -60/-60), so the gpi/thal entry-state asymmetry Stage 2l measured
(thal_0 primed at onset) is a DYNAMIC state — the TIMING of the gpi pause — not a resting-potential parameter;
equalising vr changes nothing (probed: identical cascade). This reproduces the Stage-2o boundary exactly: the
commit WTA cannot read out the real, learned thalamic advantage because its winner is decided by an ignition
RACE (whichever thalamic channel de-inhibits first), not by total drive.

## Anti-cheats (full train->test battery on 730705, numpy; raw: research/findings/raw/gateb_stage2p_striatal_ffi_downstate/smoke_numpy.json)

| quantity | STAGE2K base | FIX G on |
|---|---|---|
| `test_rate_c1` (trained target=1 expresses action 1) | 0.0 | 0.0 (no flip) |
| `test_rate_c0` (trained target=0 expresses action 0) | ~1.0 | 1.0 |
| `D_contingent` | 0.0 | 0.0 |
| `D_yoked` (yoked control — MUST stay non-contingent) | -0.10 | -0.10 |
| `steer` (D_contingent>=0.30 & gap>=0.20) | false | false |

- **Byte-identical when off = true** (Stage-2k GO protected; FIX G inert on this seed, bias 0.0).
- **Acquisition-lesion holds (PASSES):** on an UNTRAINED bridge action 1 does NOT win
  (`acq_lesion_action1_does_not_win=true`, `p_action0_target1=1.0`, `D_contingent_acq_lesion=0.0`) — the
  contingency is owned by D1 plasticity, and FIX G manufactures no policy.
- **The yoked control is non-contingent** (`D_yoked=-0.10`, essentially the same latch as contingent):
  730705's base is fully latched to action 0 regardless of the reward target, so there is no credit for
  FIX G to read out. `Verdict.control` correctly marks the |contingent-yoked| gap (0.10) below the 0.20 bar.

## Banked method + corrected next mechanism (named, not deferred)

BANKED (refuted): a striatal feedforward-inhibition / down-state homeostat cannot close 730705, because
(a) there is no pre-cue str_d1 baseline to silence and (b) hyperpolarising these negative-b MSNs drives
rebound (fires more, not less). The residual is not at the striatum: the learned D1 policy is correct and
FIX E already surfaces its advantage at the thalamus.

The ONE mechanism shown to flip 730705 legitimately (11/12, no de-latch) is Stage 2m's onset entry-state
equalisation. The next method is to realise it AS BIOLOGY rather than a host membrane reset: a spiking
TRN-like feedforward-inhibition pool that synchronises the thalamic onset each selection epoch (the in-model
`str_fsi` population — which here fires ~symmetrically, 643/663, and gates the MSNs via cross-channel
feedforward inhibition — is a template), removing the gpi-pause TIMING head-start at the thalamus so the
commit's ignition race reflects the higher (learned) thalamic drive rather than whichever channel
de-inhibits first. This targets the commit-timing residual directly, at the BG output where it is set,
without a commit de-latch that would pass unlearned drive. Gate B stands at ≥5/6 — a first-class result.

## Reproduce (numpy, orphan-proof)

```bash
export PYTHONPATH=$PWD SIM_BACKEND=numpy
# the decisive cheap refutation (no str_d1 pre-cue baseline; FFI rebound; thal inverts but commit latches):
.venv/bin/python -m research.runners._vocal_gateb_stage2p_striatal_ffi_downstate --mode refute --seed 730705
# full earned-verdict battery on the held-out miss (byte-identity + FIX_ON vs base + acq-lesion + Verdict):
.venv/bin/python -m research.runners._vocal_gateb_stage2p_striatal_ffi_downstate --mode smoke --smoke-seeds 730705
# cheap trained-cascade diagnostic (FIX G inert; FIX E+FIX G still commit-latched):
.venv/bin/python -m research.runners._vocal_gateb_stage2p_striatal_ffi_downstate --mode diag --diag-seed 730705 --fix-e
```
