---
type: finding
status: no-go
date: 2026-08-19
runner: research/runners/_replay_dg_pattern_separation_popsetpoint.py
artifacts:
  - research/findings/raw/kwta_popsetpoint/popsetpoint_6seed.json
---

# A POPULATION-activity set-point STABILIZES DG sparsity (kills the dense-collapse the per-cell homeostat could not) but does NOT make two similar memories both-discriminable — because the both_win blocker is NOT in DG competition at all; it RE-LOCALIZES to the dg→answer readout/write

**Board #78** — "sharpen the memory-separator with population-level competition." Attacks the k-WTA-STABILITY
residual that two prior negatives mapped to a POPULATION set-point:
`2026-08-19-replay-separator-bridge-rebound-and-write-runaway-FIXED-single-recall-ceiling-kWTA-stability-residual.md`
(#71: rebound + write-runaway fixed; residual = k-WTA stability, one engram collapses dense and subsumes the other)
and `2026-08-19-kwta-stability-homeostat-NOGO.md` (#73: a per-CELL firing-rate homeostat is anti-sparse and made
it WORSE — it RECRUITS silenced cells; residual re-mapped to a POPULATION-competitive set-point).

## Verdict

**NO-GO on the board #78 headline bar (two SIMILAR memories both discriminable: both_win 0/6), but the negative is
DECISIVE and RE-LOCALIZES the residual — the two prior findings mislocalized it.** The population-activity set-point
does exactly what #73 said the per-cell homeostat could not: it holds total DG activity near a target k by adapting
the INHIBITORY pool's gain, so the dense-collapse that #71 blamed for the failure is ELIMINATED (dense-collapse 4/6
→ 0/6; every engram bounded to 36–65 of 200; symmetric; single-recall and scramble controls intact 6/6). **Yet
both_win stays 0/6.** A drive sweep then shows the killer directly: both_win is **0/24** across perforant-drive
levels that span engram sizes **3–88** and DG Jaccards **0.29–0.67**, all non-dense — the both_win failure is
**independent of DG density, engram size, and separation quality.** Therefore DG k-WTA competition is **not** the
load-bearing variable for the bar; the blocker sits DOWNSTREAM, in the dg→answer associative WRITE + the input→dg
REACTIVATION consistency — an anti-symmetric readout collapse in which one answer assembly wins BOTH probes at
near-noise answer rates. No `sim/` edit (the controller is a runner-side wrapper on the step; the selection stays
on-substrate). Deterministic (`cfg.seed`); the LESION arm reproduces the #71/#73 dense-collapse residual.

## What was built (the named mechanism, on-substrate selection, no `sim/` edit)

A POPULATION-activity set-point realized as a fast controller on **cumulative recruitment** — the number of DISTINCT
granules that have fired since the replay event began (this is exactly what the dense-collapse instrument measures:
an engram is the set of cells firing ≥1× during the event). The arc's diagnosis (probes below) showed the failure is
not instantaneous over-activity — instantaneous DG activity is already sparse (~10–15/step) at a reasonable operating
point — but that the winner set ROTATES across the 37-step window (Izhikevich adaptation + oscillating feedback
inhibition), so the CUMULATIVE engram runs dense. A set-point on instantaneous total activity never engages (activity
is below k every step). The controller instead ramps a depolarizing drive into the `dg_fs` basket pool (raising the
divisive shunting-inhibition gain) as cumulative recruitment approaches and exceeds k, blocking LATE recruits so the
early strongest-driven winners lock in; per-event reset on the silent settle gap. Driving the basket INPUT (not
scaling its OUTPUT gate) means the base competition-lesion (separator_off: fs→dg gate = 0) auto-neutralises the
controller, keeping the null clean. Wrapped around `bridge._run_one_simulation_step`, so every measurement path
(engram / consolidate / probe / direct-readout / scramble) inherits it, reusing the #71 runner by import (its
shunting reversal + transmission-gated write + all machinery). LESION arm (`pop_setpoint=False`) is a no-op wrapper →
byte-identical to #71. Biology: divisive normalization / population gain control (Carandini & Heeger 2012); the
`dg_fs` PV-basket IS the normalizer, its ~1–2% sparsity a k-of-N set-point on TOTAL activity (Marr 1971;
O'Reilly-McClelland 1994; Leutgeb 2007; Bakker 2008) — NOT a per-cell rate (Turrigiano 2011, the tool #73 banked).

## Results — 6 seeds (42/43/44/100/101/102), ON (set-point) vs LESION (#71) (`popsetpoint_6seed.json`)

<!--derived-->
_Numbers from the cited artifact._

| seed | ON sizes | ON dense | ON sel (m0,m1) | ON dgJ | LESION sizes | LES dense | LES sel (m0,m1) | single | scramble |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---:|---:|
| 42  | (53,44) | no  | (−0.36,+0.36) | 0.56 | (200,44)  | yes | (−0.17,+0.15) | +1.00 | −0.97 |
| 43  | (39,49) | no  | (+0.63,−0.55) | 0.47 | (39,55)   | no  | (+0.55,−0.59) | +1.00 | −1.00 |
| 44  | (50,65) | no  | (−0.03,+0.01) | 0.53 | (56,198)  | yes | (−0.11,+0.06) | +1.00 | −0.98 |
| 100 | (47,56) | no  | (+0.06,−0.02) | 0.56 | (199,60)  | yes | (+0.09,−0.13) | +0.97 | −0.97 |
| 101 | (44,36) | no  | (+0.12,−0.10) | 0.48 | (46,39)   | no  | (+0.36,−0.36) | +1.00 | −0.93 |
| 102 | (55,49) | no  | (+0.09,−0.06) | 0.60 | (199,68)  | yes | (+0.44,−0.44) | +1.00 | −1.00 |

**Pooled:** both_win **0/6** (ON and LESION); **dense-collapse ON 0/6 vs LESION 4/6** (the set-point removes it);
engram sizes bounded **6/6** (all ≤ 65 vs LESION up to 200); single-recall **6/6**; scramble-inverts **6/6**;
dissimilar both_win 2/6. The per-memory read stays ANTI-SYMMETRIC on the ON arm (whichever memory reads +x, the
other reads −x; |sel| up to 0.63 on seed 43 — sharp, not noise), i.e. one answer assembly wins BOTH probes.

**The set-point LANDS (dissociation, `control` check PASS):** it is the only mechanism in this arc that stabilises DG
sparsity — it eliminates the dense-collapse the per-cell homeostat (#73) made worse, and the LESION arm reproduces
the #71 dense-collapse (4/6). So the manipulation demonstrably changes the DG code; the verdict is DEFINED, not
UNDEFINED. It simply does not move both_win.

## The re-localization: both_win is INDEPENDENT of DG density/size/separation (both_win 0/24)

Sweeping the perforant DRIVE (`input_to_dg_weight` 40→70) in a feed-forward-inhibition regime (dg→fs low, shunting
reversal above vr) that produces NON-dense, SYMMETRIC engrams whose size scales cleanly with drive, set-point OFF —
the DG code the read receives is varied directly:

<!--derived-->
| drive | engram-size range (6 seeds) | DG Jaccard range | dense | both_win |
|---:|:--:|:--:|:--:|:--:|
| 40 | 3–9   | 0.29–0.67 | 0/6 | **0/6** |
| 50 | 13–24 | 0.32–0.50 | 0/6 | **0/6** |
| 60 | 30–55 | 0.43–0.60 | 0/6 | **0/6** |
| 70 | 50–88 | 0.49–0.63 | 0/6 | **0/6** |

**both_win = 0/24.** Even at drive 50 — engrams of ~13–24 cells (6–12% sparse), Jaccard 0.32–0.50, symmetric, no
dense-collapse (the exact "sparse, separated, bounded" code #71/#73 said was the goal) — every seed fails both_win
with the same anti-symmetric signature. If the blocker were DG competition/density, some cell here would pass. None
does. **This is the decisive evidence that the residual is NOT in the DG separator.**

## Where the residual actually is (direct-readout localization, arc probes)

Driving each WRITTEN engram DIRECTLY (bypassing input→dg) and reading the answer isolates the learned dg→answer
mapping from the reactivation dynamics:

- **The dg→answer WRITE is seed-dependently scrambled.** seed 42: m0-engram → a0=65 / a1=85 (picks a1, WRONG),
  m1-engram → a0=77 / a1=57 (picks a0, WRONG) — the learned map is BACKWARDS for BOTH memories, at ~sparse engrams.
- **The input→dg REACTIVATION is inconsistent with the written engram.** seed 43: the direct readout is CORRECT for
  BOTH (m0-engram → a0=85/a1=34; m1-engram → a0=29/a1=80), yet the input-driven PROBE still fails for m1 — driving
  m1's INPUT reactivates a set that reads as m0. The engram the input recreates ≠ the engram that was written.
- **Answer-side WTA is NOT the cause:** forcing the answer opponent-inhibition gate to 0 leaves the reads unchanged.
- Answer rates are near-noise (~0.03–0.11 spikes/step across 60 answer cells), so the read is a razor decision on a
  scrambled/inconsistent mapping.

So the both_win failure is an **anti-symmetric readout collapse in the associative WRITE + input→dg reactivation
consistency**, downstream of DG competition entirely. This CORRECTS #71 (which localised it to a k-WTA dense-collapse
— that is a real DG failure mode, now FIXED here, but it was a SYMPTOM, not the both_win cause) and #73 (which
proposed a per-cell then a population DG set-point — the population set-point works AT ITS JOB and still does not
close both_win).

## Levers tried this arc (all measured; the DG-side is exhausted)

Basket-drive population controller (instantaneous target — never engages; cumulative-recruitment target — bounds the
engram, both_win 0/6); scaling the fs→dg inhibitory GAIN directly (non-monotonic: w=6 dense-asymmetric, w=12 nearly
dead, w=24+ dense again — a feedback-inhibition OSCILLATION whose cumulative-ever-fired set is dense); shunting
reversal −63→−50 (the non-monotonicity is NOT rebound — it persists above vr; setting E_i above the driven-v makes
"inhibition" depolarising); feed-forward-only inhibition (dg→fs=0 → symmetric engrams, both_win still 0); perforant
drive 30→70 (engram 2→200, both_win 0/24); a DEVELOPED competitive input→dg projection via the engine's Oja /
Miller-MacKay `hebbian_mean_subtract` (first attempt UNSTABLE — mean-subtract drove the projection to 0, Oja blew it
to dense; needs tuning — see next mechanism). Every DG-side lever leaves the anti-symmetric readout intact.

## The mapped next mechanism (banks the DG-density METHOD; the capability stays open)

**Attack the dg→answer READOUT/WRITE + input→dg reactivation — NOT another DG inhibitory lever.** Population inhibitory
control is banked as SUFFICIENT for DG-sparsity stabilisation (a real capability it closed: dense-collapse gone,
engrams bounded, symmetric — where the per-cell homeostat failed) and INSUFFICIENT for both_win (wrong stage). The
residual's actual locus needs: (1) a cleaner one-shot HETEROassociative write that does not cross-contaminate the two
memories' answers (the soft-bound rate-window rule writes overlapping-engram granules to BOTH answers; a
competitive/normalised or one-shot BTSP-style write would keep m0→a0 and m1→a1 orthogonal in the weight matrix);
(2) consistent input→dg REACTIVATION so the probe recreates the WRITTEN engram (pattern completion / a DEVELOPED
competitive perforant projection — the engine's `hebbian_oja` and `hebbian_mean_subtract` are the on-substrate hooks;
this arc's quick attempt was unstable and is the next tuning target); possibly (3) a CA3-style autoassociator between
DG and answer to clean the retrieved pattern. This is a NEW arc (a readout/write de-risk), correctly scoped by this
negative — three DG-competition mechanisms (#71 k-WTA, #73 per-cell homeostat, #78 population set-point) have now
been shown to leave both_win untouched, so the next lever must not be a fourth DG-side one.

## Tracked scaffolds (host, not brain)

Inherited from the #71 runner: host-defined input patterns and answer assemblies; host reinstatement of each memory's
input AND answer during replay (hippocampal index / SWR trigger); scheduled down-states; the WRITE/READ
transmission-gate phase (host-scheduled sleep/wake gate); a rate-window Hebbian coactivity write; an argmax over
answer spike counts for MEASUREMENT only; a fixed random perforant projection and fixed FS anatomy (not developed).
NEW this runner: the population set-point CONTROLLER (a PI loop on cumulative DG recruitment → injected basket drive)
is host; the SELECTION (which granules survive the divisive shunting basket) is on-substrate spiking.

## Reproduce

    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_popsetpoint \
        --seeds 42 43 44 100 101 102 \
        --out research/findings/raw/kwta_popsetpoint/popsetpoint_6seed.json

The `--k-target/--kp/--ki/--integ-max/--drive-max` flags sweep the set-point operating point; the runner always runs
the LESION (`pop_setpoint=False`) arm for the dissociation and a drive-sweep re-localization block (set-point OFF).

## Sources

EXTERNAL-SEARCH-RAN: divisive normalization / population gain control vs per-cell rate homeostasis; heteroassociative
write cross-talk; DG pattern separation as a k-of-N competition (logged to the corpus-check record, 2026-08-19).

- Carandini, M., Heeger, D.J. (2012). Normalization as a canonical neural computation. Nat Rev Neurosci 13:51–62. —
  divisive normalization / population-gain set-point (the mechanism built here; it DOES stabilise DG sparsity).
- Turrigiano, G. (2011). Too many cooks? Intrinsic and synaptic homeostatic mechanisms. Annu Rev Neurosci 34:89–103.
  — per-cell rate homeostasis, the WRONG tool #73 banked (contrast).
- Marr 1971; O'Reilly & McClelland 1994; Leutgeb 2007; Bakker 2008 — DG sparse-expansive separation as k-of-N
  competition (adequate here: separation is NOT the both_win blocker).
- Oja 1982 (Oja's rule); Miller & MacKay 1994 (subtractive normalization) — the on-substrate competitive-development
  hooks (`hebbian_oja`, `hebbian_mean_subtract`) named for the next arc's developed input→dg projection.

Internal: builds on and CORRECTS
`2026-08-19-replay-separator-bridge-rebound-and-write-runaway-FIXED-single-recall-ceiling-kWTA-stability-residual.md`
(#71: dense-collapse was a symptom, fixed here) and `2026-08-19-kwta-stability-homeostat-NOGO.md` (#73: the population
set-point it mapped works at its job but does not close both_win — the residual is downstream, in the readout/write).
