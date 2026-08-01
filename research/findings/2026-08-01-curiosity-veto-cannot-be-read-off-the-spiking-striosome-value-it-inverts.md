---
type: finding
status: live
date: 2026-08-01
mechanism: curiosity-seek-learn
runner: research/runners/_curiosity_seek_learn_onbridge_derisk.py
artifacts:
  - research/findings/raw/_curiosity_seek_learn_onbridge_spikingveto.json
---

# The curiosity veto cannot be read off the spiking striosome VALUE — it inverts (honest negative, 6 seeds)

## Why this ran

The 2026-07-31 critic-lesion finding named a shortcut and its conversion target: DR-1's noisy-concept veto —
the load-bearing honesty anti-cheat (curious AND honest, declines the noisy-TV) — is computed by a HOST-side
Python ELP tracker (a TD low-pass fed by the SNc paired-subtraction read `snc_B - snc_A`), NOT by the spiking
substrate. It survived the GABA_B critic lesion 6/6, proving the striosome rate is not what it reads. The named
conversion target was "a spiking mechanism that computes the same decision FROM THE STRIOSOME ... rather than
from a Python running estimate." This de-risk tests the most direct realization: threshold the LEARNED SPIKING
STRIOSOME VALUE (`read_value(c)`, drift-free via the same wash-out) instead of the host tracker.

Additive, default-off: a `--spiking-veto` flag on the existing runner (77 insertions, one file, `sim/` untouched;
the default path is byte-identical — re-verified GO on the smoke).

## Result — the direct striosome-value veto FAILS, and it fails DIRECTIONALLY

**NO-GO, 0/6** (artifact `research/findings/raw/_curiosity_seek_learn_onbridge_spikingveto.json`, 6 seeds
42/43/44/100/101/102, numpy). The raw striosome value does not separate learnable from noisy. On 5/6 seeds it
INVERTS (noisy reads a HIGHER value than learnable, mean sep -10.8 Hz); on the one seed the sign is nominally
correct (+4.6 Hz) the margin is negligible. In every case the NOISY value sits ABOVE any floor that still passes
the learnable set, so the veto never fires (`noisy_vetoed=False`, `noisy-stops=False` on all six seeds).

| seed | v0 (Hz) | floor | learn V | noisy V | sep | SNc learn/noisy | noisy vetoed |
|---|---|---|---|---|---|---|---|
| 42  | 71.9  | 43.2 | 66.0 | 80.3 | -14.2 | 3.9 / 0.0  | False |
| 43  | 79.8  | 47.9 | 63.3 | 86.1 | -22.8 | 17.1 / 0.0 | False |
| 44  | 95.2  | 57.1 | 77.0 | 72.4 | +4.6  | 13.0 / 0.0 | False |
| 100 | 105.0 | 63.0 | 73.9 | 87.0 | -13.2 | 22.5 / 0.0 | False |
| 101 | 75.8  | 45.5 | 66.4 | 77.2 | -10.8 | 30.7 / 0.0 | False |
| 102 | 75.8  | 45.5 | 63.4 | 71.9 | -8.5  | 13.3 / 0.0 | False |

(v0 = fresh pre-learning striosome value; floor = 0.6 x v0; V = learned striosome value read after the run.)

The noisy striosome value sits above the floor on every seed (noisy learned 72-87 Hz vs floor 43-63 Hz), and it
climbs even higher under the critic lesion (99-185 Hz vs the 72-87 in the intact arm). The confound is the
reward-independent STDP drift on the plastic `cue -> striosome` pathway (the runner names it at `RPE_GAIN`):
absent a strong reward signal to gate it (noisy: r~0), direct STDP potentiates the cue->striosome weights, so a
noisy concept's value read climbs. The reward-gated correction that would pull a learnable value cleanly ABOVE it
is swamped. This is not a floor-tuning miss: no floor passes the learnable set while vetoing the higher noisy set.

## What DOES separate — and what a real conversion therefore needs

The clean separator is already in the loop and it is spiking: the **SNc reward burst** — learn-burst 4-31 Hz vs
**0.0 Hz** on noisy on every one of the six seeds. The host ELP tracker works precisely because it reads THAT (via the paired
subtraction `snc_B - snc_A`, which cancels the learned value V so the read isolates the reward r). The striosome
VALUE is the wrong substrate quantity to threshold; the SNc reward-PREDICTION-ERROR is the right one.

So the 07-31 conversion is NOT a signal-swap (read the striosome instead of the tracker) — that is refuted here.
A substrate-computed veto must read the **reward-OMISSION signal**: the striosome -> SNc GABA_B dip when a
predicted reward fails to arrive (a lateral-habenula / RMTg omission-detector is the biological form), gating the
ASK pool DOWN. That is a spiking circuit to build, not a threshold to move. This de-risk pins the wall and names
the build.

## What it does and does not mean

- It does **NOT** touch the DR-1 GO: the capability (ask about the learnable, decline the noisy) still holds
  6/6 on both backends. The veto's SIGNAL (SNc firing) is already a spiking read; what stays host-side is the
  paired-subtraction + TD low-pass + threshold arithmetic.
- It **narrows** the named shortcut: the residual host computation cannot be discharged by reading the striosome
  value. The striosome value read reported in DR-1 as "the critic moved" is confirmed NOT to be a clean
  reward-value readout (it moves, but not monotonically with learnability) — consistent with 07-31.
- The `critic-lesion-collapses-veto` gate I added is DEGENERATE under this negative (both the real and the
  critic-lesion arm fail to veto, so "collapse" trivially reads True) — it is only meaningful once a veto that
  actually fires exists; recorded for the future omission-circuit de-risk.

## Honest scope

6 seeds, numpy backend, real config (8 learnable / 4 noisy). A single veto formulation (a TD low-pass of the
striosome rate, floor = 0.6 x v0) — but the failure is directional (noisy > learnable), so it does not turn on
the floor. Says nothing about a spiking omission-detector veto, only that the direct striosome-value threshold is
not it.
