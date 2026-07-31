---
type: finding
status: corrected
date: 2026-06-21
mechanism: k-way-sequencer
---

# Shortcut #3 — K=32 sequencer CAPABILITY verdict: the 0.15 NEGATIVE is a WRONG-THRESHOLD artifact (2026-06-21)

**Type:** SURPASS investigation (a NEGATIVE is a research prompt, never an exit). The #3 K=32 sequencer **capability**
verdict came back NEGATIVE on the cached OPTIMIZED verify
(`research/findings/raw/_phaseB_onebrain_sequencerK_k32_OPTIMIZED_verify.json`: `per_K.32.eq_n = 2/3`, `verdict
NEGATIVE`) — BUT the no-confab **moat HELD at scale** (`fa_total 0`, `moat_n 3/3`, `lesion_n 3/3`). That verify ran at
`match_thresh = 0.15`. The moat-safe PRODUCTION threshold from the R0 de-risk is **0.06**. This round ISOLATES which:
threshold artifact vs genuine K=32 capacity wall. Host-CPU (cache+parallel-fast); **NO `sim/` edit**; **moat NEVER
weakened**.

**Prior context:** `2026-06-20-shortcut3-K32-derisk.md` (R0: mechanically-certain threshold fix + R0 GO at K=2, 6 seeds;
the K=32 6-seed empirical confirmation was noted IN FLIGHT on the contended GPU). This doc is the dedicated K=32
capability confirmation at the production threshold, on host-CPU.

---

## TL;DR (the verdict)

**THRESHOLD-ARTIFACT GO. The 0.15 NEGATIVE is a wrong-threshold artifact, NOT a K=32 capacity wall.** At the R0
PRODUCTION threshold `match_thresh = 0.06` the K=32 battery is `eq_n 3/3` with `fa_total 0` (moat held). The single
failing row at 0.15 (seed 43, cue `monb`, correct block 4) read its CORRECT block at `m4 = 0.116` with ALL 64 other
blocks at EXACTLY `0.000` — so 0.15 abstained it (over-abstention, the SAFE direction) while 0.06 admits it with **zero
false-accept risk** (there is no competing non-zero block to spuriously fire). ⇒ #3 K=32 is GO at the production
threshold; the fold can proceed.

---

## Move 1 — ISOLATE + QUANTIFY: the 0.15-vs-0.06 contrast

### The failing seed/row (extracted from the committed 0.15 OPTIMIZED verify)

Seed 43 is the ONLY failing seed (`eq_all False`); seeds 42 and 44 are `eq_all True`. Seed 43 fails at **exactly one
row of 35** — row 4 (`blk4-present`, cue `monb`, host answer = block 4):

```
seed 43, row 4 (cue monb, host block 4):
  m4   = 0.116          <- the CORRECT block's spiking match-pool rate
  m0..m3, m5..m63 = 0.000   (EXACTLY zero on all 64 other blocks -- the only non-zero rate is the correct one)
  match_thresh = 0.15  ->  m4 (0.116) BELOW threshold  ->  sub = None, decision = abstain  (match_host_eq = False)
  every OTHER row (0-3, 5-34) matches host; the absent/cross moat rows abstain correctly (host = None)
```

This is the single-low-fidelity-code signature, NOT a capacity collapse: one concept code's match pool happened to land
at 0.116, inside the `(0.06, 0.15]` band, so a threshold stricter than production abstained it.

### Per-seed margins (re-derived from the cached `rates`, all 3 seeds)

| seed | min target-block rate (present rows) | max off-target rate (present rows) | max moat-row rate (absent/cross) | @0.06 capability | @0.06 moat |
|---|---|---|---|---|---|
| 42 | 0.182 | 0.000 | 0.000 | all rows > 0.06 → match | safe (0 fires) |
| 43 | **0.116** | 0.014 | 0.000 | all rows > 0.06 → match | safe (0 fires) |
| 44 | 0.196 | 0.000 | 0.000 | all rows > 0.06 → match | safe (0 fires) |

Every present row's target rate (min 0.116) clears 0.06; the largest off-target on any present row is 0.014 (far below
its target); and **every moat row's max rate is 0.000** → admitting matches at 0.06 carries zero false-accept risk. This
is exactly the R0 abstain/recall gap argument: the no-match floor is 0.000 and the weakest real signal is 0.116, so 0.06
sits cleanly inside the open gap.

### The threshold contrast

| match_thresh | eq_n (recall == host) | fa_total (moat) | verdict |
|---|---|---|---|
| **0.15** (the OPTIMIZED verify) | **2/3** (seed 43 over-abstains row 4) | **0** | NEGATIVE |
| **0.06** (R0 production) — STATIC re-derivation | **3/3** (all rows clear 0.06) | **0** (no block fires on any moat row) | GO |
| **0.06** (R0 production) — LIVE battery re-run | _PENDING (run in flight, host-CPU)_ | _PENDING_ | _PENDING_ |

The static re-derivation is conclusive on its own (the decision rule is `rate > match_thresh`; every relevant rate is
cached). The LIVE row is the byte-identical battery re-run at 0.06 (same `gain 0.11`, `sigma 1.0`, `retreat divnorm`,
`input_gain 1.0`, `peak_mults 0.1,1.0,10.0`, D=128, seeds 42/43/44 — the ONLY changed knob is `match_thresh 0.15 →
0.06`), filled below when it lands.

---

## Verdict

**(a) THRESHOLD-ARTIFACT GO.** The 0.15 NEGATIVE was a wrong-threshold artifact: a threshold stricter than the
moat-safe production threshold (0.06) abstained one in-gap real match (seed 43's `monb`, 0.116). At 0.06 the K=32
capability holds 3/3 with the moat held 0-FA. There is NO genuine K=32 capacity boundary — moves 2–4 (reframe / rank
cheap-first surpass / boundary verdict) are not triggered. **#3 K=32 is GO at the production threshold and the fold can
proceed.**

The moat was held at every seed at both thresholds (`fa_total 0`) and is untouched here — 0.06 only relaxes PRESENT-block
matching into the empty no-match gap; the no-confab abstention on absent/cross cues fires no block at ANY threshold.
**NO `sim/` edit** (runner-side `--match-thresh` knob only).
