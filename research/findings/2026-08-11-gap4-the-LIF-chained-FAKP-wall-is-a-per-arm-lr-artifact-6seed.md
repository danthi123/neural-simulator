---
type: finding
status: contributing
date: 2026-08-11
mechanism: deep-credit-on-spikes
lane: gap#4 ALL-IN (wave-2, redirect #1 — is the located wall real?)
verdict: 6-SEED — the 2026-08-02 "chained transport-free FA/KP COLLAPSE to majority-class at N>=3" wall on the LIF surrogate substrate is a PER-ARM LEARNING-RATE ARTIFACT, not a property of chained transport-free credit. At the shared lr the FA/KP arms sit at the majority-class floor (the reported "wall"); with a FAIR per-arm hidden lr (grid 0.005–0.02) both arms enter the regime at N=3 AND N=4, 6/6 seeds, beating the OPTIMAL-ridge frozen reservoir (numbers in the body table). This does NOT establish DEEP credit (XOR is depth-2-obligatory; entering != credit through obligatory depth) — it removes the LIF "wall" as a wall. The PRODUCTION Izhikevich substrate is separate and unresolved (2026-08-02: FA converges on LIF but not on Izhikevich) and needs the same fairness re-check.
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_gap4_perarm_tuned_fakp_baseline_derisk.py
artifacts:
  - research/findings/raw/_gap4_perarm_fakp/AGGREGATE.txt
  - research/findings/raw/_gap4_perarm_fakp/perarm_s42.json
  - research/findings/raw/_gap4_perarm_fakp/perarm_s43.json
  - research/findings/raw/_gap4_perarm_fakp/perarm_s44.json
  - research/findings/raw/_gap4_perarm_fakp/perarm_s100.json
  - research/findings/raw/_gap4_perarm_fakp/perarm_s101.json
  - research/findings/raw/_gap4_perarm_fakp/perarm_s102.json
corrects: research/findings/2026-08-02-gap4-depth-rescue-untestable-on-spikes-the-wall-is-upstream-deep-local-rule-does-not-enter-the-learning-regime.md
---

# gap#4 — the 2026-08-02 "chained FA/KP wall at N≥3" on the LIF surrogate substrate is a per-arm LEARNING-RATE artifact (6-seed)

## Why this ran (wave-2 redirect #1)

The wave-1 verification (workflow wrufiei6u → `2026-08-11-gap4-wave1-verification-corrected-...`) found that a skeptic
re-running the chained transport-free FA/KP arms at a fair per-arm learning rate got them to ENTER the regime at N=3/4,
where the 2026-08-02 findings had reported they COLLAPSE to majority-class — suggesting the "located wall" was a
learning-rate divergence, not a property of the credit rule. This runner turns that 1–2-seed skeptic re-run into a
proper 6-seed, multi-depth, per-arm-lr SWEEP so the wall's status is banked.

## Method

`_gap4_perarm_tuned_fakp_baseline_derisk.py` reuses `run_seed()` from the wall runner UNCHANGED (same forward init, same
arms, same task, same optimal-ridge reservoir floor); the ONLY swept variable is the chained-FA hidden lr (`lr_fa`) and
the KP hidden lr (`kp_lr`), over a grid {0.005, 0.01, 0.02, 0.05}. The output-arm lr stays 0.05 (matched to the
wall/DECOLLE runs). For each arm × depth × seed, take the BEST per-arm lr, and check ENTER-THE-REGIME (held-out >
majority + 0.03 AND > optimal-ridge reservoir + 0.03). Config matched to the DECOLLE 6-seed run (hidden 32, T 24,
epochs 200, subsample 2000, bptt-hidden 128, bptt-epochs 400). numpy/CPU. NO sim/ edit. `cfg.seed` seeding path
inherited from the wall runner.

## Result — 6 seeds (42/43/44/100/101/102), XOR held-out (`research/findings/raw/_gap4_perarm_fakp/AGGREGATE.txt`)

<!--derived-->
| N | majority | frozen-opt-ridge | FA best (lr) | FA enters | KP best (lr) | KP enters | FA @ shared lr=0.05 | KP @ shared lr=0.05 |
|---|----------|------------------|--------------|-----------|--------------|-----------|---------------------|---------------------|
| 2 | 0.524 | 0.609 | 0.881 (0.02) | 6/6 | 0.885 (0.02) | 6/6 | 0.839 | 0.878 |
| 3 | 0.524 | 0.623 | 0.856 (0.01) | 6/6 | 0.847 (0.01) | 6/6 | **0.500** | **0.500** |
| 4 | 0.524 | 0.615 | 0.844 (0.005) | 6/6 | 0.866 (0.005) | 6/6 | **0.500** | **0.500** |

<!--derived-->
**The reported "wall" reproduces exactly at the shared lr and vanishes at a fair one.** At N=3 and N=4 the chained
transport-free FA and KP arms sit at 0.500 == majority-class when run at the shared lr=0.05 (the 2026-08-02 "collapse to
majority-class / degenerate-dynamics fingerprint"). With a fair per-arm hidden lr (0.005–0.02) the SAME arms, on the
SAME forward init and task, leave majority-class and beat the OPTIMAL-ridge frozen reservoir by ~+0.23 — **6/6 seeds,
both arms, both wall-depths**. The best lr shrinks with depth (0.02 → 0.01 → 0.005), the signature of a plain
optimization step-size / gradient-scale mismatch compounding per hop, not a credit-assignment wall.

## What this settles — and what it explicitly does NOT

- **SETTLES:** on the LIF surrogate substrate, "the chained multi-hop transport-free rule does not get a deep (N≥3)
  spiking net into the learning regime" is FALSE — it does, when fairly tuned. The located wall (as measured on this
  substrate) was a per-arm learning-rate artifact. This CORRECTS the load-bearing claim of
  `2026-08-02-gap4-depth-rescue-untestable-...` (a correction banner is added to that file, pointing here).
- **DOES NOT establish DEEP credit.** XOR is depth-2-obligatory; leaving majority-class + beating the reservoir at N=3/4
  is ENTER-THE-REGIME, not credit through genuinely-obligatory depth (Q5 showed a depth-obligatory task is
  unconstructible here as a matched-width generalisation gate). Fairly-tuned FA/KP entering a depth-2 task is the SAME
  scope as the wave-1 FF/DECOLLE GOs — none of them tests deep credit. The deep-credit question remains OPEN.
- **DOES NOT touch the PRODUCTION substrate.** `2026-08-02-gap4-FA-convergence-...-6of6-LIF-converge-0of6-izhikevich`
  reports FA convergence 6/6 on the LIF surrogate but 0/6 on the Izhikevich on-bridge net — a genuine substrate
  difference. Whether THAT Izhikevich 0/6 is also a fairness/lr artifact or a real substrate wall is the open question
  (wave-2 redirect #3; the Q3 DRTP-on-Izhikevich lane's preliminary read is that no rule enters there at the shared lr —
  which must be re-checked with per-arm-lr fairness before it can be trusted, exactly as this finding did for LIF).

## The workflow lesson (why this matters beyond the number)

This is a textbook instance of the failure the project warns about: **a companion process (per-arm learning-rate
selection) was replaced with a constant (one shared lr across arms of very different gradient scale), and the constant
dominated the measurement** — producing a confident, reproducible "wall" that stood for ~10 days and framed an entire
assault. The instrument (the A/B at a fixed shared lr) was not broken; it was measuring lr-fairness, not the credit
rule. The fix was one fair sweep. **The wall was a verdict on a hyperparameter, not on the biology.**

## Reproduce

```
for S in 42 43 44 100 101 102; do SIM_BACKEND=numpy .venv/bin/python -m \
  research.runners._gap4_perarm_tuned_fakp_baseline_derisk --seeds $S --n-list 2 3 4 \
  --lr-grid 0.005 0.01 0.02 0.05 --epochs 200 --bptt-hidden 128 --bptt-epochs 400 \
  --out research/findings/raw/_gap4_perarm_fakp/perarm_s${S}.json & done; wait
SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap4_perarm_tuned_fakp_baseline_derisk \
  --aggregate "research/findings/raw/_gap4_perarm_fakp/perarm_s*.json"
```
