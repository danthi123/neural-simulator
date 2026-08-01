---
type: finding
status: contributing
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/realspikes/realspikes_credit_6seed_aggregate.json
---

# gap#4 crux: the unsupervised movable-plateau rule does NOT survive the port to REAL SPIKES — it degrades the real-spikes codon below the frozen reservoir — 6-seed NEGATIVE (beats frozen 1/6)

<!--derived-->
**One-line verdict:** the decisive test of the on-bridge spiking port. The unsupervised movable-plateau covariance
rule is gap#4's ONLY positive credit signal — but only at the RATE stand-in (a boolean-hold reset-read: beats a
frozen reservoir 5/6, dcs +0.139). With the degenerate-forward-pass boundary now cleared (the pre-gate: a REAL
spiking forward pass gives input-dependent, reproducible codons), this runs the SAME rule trained + read on real
spikes (pre-activity = real feature spike counts). It **does NOT survive**: the rule beats the frozen on-bridge
reservoir on only **1/6** seeds, `deep_credit_share > 0` on **1/6** (mean dcs **−0.063**). On 5/6 it *degrades* the
real-spikes codon **below the untrained reservoir** — and below frozen on TRAIN too (mean credit train 0.489 vs the
frozen it can't beat) — so it is not overfitting, it is actively worsening the representation. The one "win" (seed
102) is only an anomalously low frozen (0.148). The instrument is verified (pre-gate passed, the rule trains —
update mag nonzero, all anti-cheats clean, reproducibility 1.0), and an lr sweep on seed 42 finds no rescue
(credit < frozen at every lr in {0.001…0.02}). **The rate 5/6 was a rate-stand-in artifact; on real spikes the
only positive signal evaporates.** No `sim/` edit.

Artifact: `research/findings/raw/gap4/realspikes/realspikes_credit_6seed_aggregate.json` (backend numpy/CPU).
Runner: `research/runners/_gap4_realspikes_credit_derisk.py`.

## Result — 6 seeds {42,43,44,100,101,102}, real-spikes read (drive 1200, 30-step window)

<!--derived-->
| seed | CREDIT held-out | FROZEN held-out | deep_credit_share |
|---|---|---|---|
| 42 | 0.296 | 0.333 | −0.056 |
| 43 | 0.222 | 0.241 | −0.024 |
| 44 | 0.241 | 0.389 | −0.286 |
| 100 | 0.370 | 0.426 | −0.097 |
| 101 | 0.296 | 0.389 | −0.161 |
| 102 | 0.352 | 0.148 | +0.244 (frozen anomalously low) |

`beats frozen 1/6`, `dcs > 0 1/6`, mean dcs **−0.063**. Anti-cheats 6/6: oracle 0.975 (task learnable),
rate-reservoir 0.099 (op-point genuine), permuted-readout 0.111 (≈ chance), lesion 0.130 (≈ floor), no-transport
holds, reproducibility 1.0 (the load-bearing gate — the real read is reliable).

## Why the rule flips sign under the port (the anti-correlation, made consequential)

<!--derived-->
The pre-gate found the real-spikes column codons **anti-correlated** with the boolean-hold reset-read (−0.14 to
−0.38 per input) — the real spiking forward pass builds its own representation. The covariance rule sharpens each
column onto the feature conjunctions that most reliably co-drive its plateau. On the rate stand-in's low-CV
boolean-hold codon those conjunctions are class-discriminative, so sharpening helps (rate 5/6). On the real-spikes
codon the reliably-co-active structure is different, so the SAME sharpening moves columns toward conjunctions that
do not carry the inheritance signal — degrading held-out below the untrained reservoir. The rule is
representation-specific: it improved the stand-in and hurts the real thing.

## What this settles for the crux (and the honest next)

<!--derived-->
Consolidating the whole gap#4 movable-hidden arc on the **real** substrate: at the rate stand-in, three credit
routes were tried — unsupervised (5/6, the only positive), supervised DFA (overfits, null), DFC+Kolen-Pollack
(overfits, negative). Ported to REAL spikes, the one positive **evaporates** (this finding). So on the actual
mission substrate (real spiking forward pass), **no tested local credit rule — supervised or unsupervised —
improves the movable plateau hidden's held-out over a frozen reservoir.** The rate 5/6 headline was carried by the
reset-read stand-in, not by real spikes. This is a verdict on the rules tried at this read/operating point, NOT on
"deep credit on spikes" as a capability. The next lever is a rule matched to the REAL-spikes representation's
statistics — e.g. a spike-timing / plateau-coincidence rule that reads the real co-firing structure the covariance
rule mis-weights, rather than the rate-covariance form imported from the stand-in. That is a fresh mechanism arc; the
rate-imported rule is now closed on real spikes. No capability abandoned — a mapped boundary with the next
mechanism named, on the honest (real-spikes) substrate.
