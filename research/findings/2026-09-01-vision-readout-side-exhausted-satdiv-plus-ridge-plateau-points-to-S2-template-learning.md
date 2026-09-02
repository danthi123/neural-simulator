---
type: finding
status: live
date: 2026-09-01
mechanism: vision-lindiscrim-readout / satdiv-divisive-normalization + ridge-retune
lane: perception (board #135 / #75)
seeds: [42, 43, 100]   # EXPLORE seeds only (held-out 44/101/102 not yet run — see honest scope)
artifacts:
  - research/findings/raw/lanes/perception/vlin_satdiv_ridge0.05_explore.json
  - research/findings/raw/lanes/perception/vlin_satdiv_ridge0.1_explore.json
  - research/findings/raw/lanes/perception/vlin_satdiv_ridge0.25_explore.json
  - research/findings/raw/lanes/perception/vlin_satdiv_ridge0.5_explore.json
  - research/findings/raw/lanes/perception/vlin_satdiv_ridge1.0_explore.json
runner: research/runners/_vision_lindiscrim_readout_derisk.py
builds_on:
  - board #135 opsweep (affine s2-norm family exhausted)
  - research/vision-next-mechanism-scope (satdiv scoping + smoke, commit ad0648672)
---

# The vision identity-readout wall is READOUT-EXHAUSTED: the divisive-normalization ratio (satdiv) + a ridge re-tune both IMPROVE but PLATEAU short of the capability bar — the residual is the frozen random S2 template bank, not the readout

**One-line.** Following the board-#135 finding that every *affine* S2-normalization (none/submean/z/alpha) sits on
a saturation-vs-headroom cliff, the biologically-correct **semi-saturating divisive-normalization RATIO**
(`satdiv`, `R_i = drive_i^n / (sigma^n + Σ_j drive_j^n)`; Heeger 1992 / Carandini & Heeger 2012) was built and
does genuinely beat the `z` baseline (margin +0.24 vs +0.18, RATE ceiling 0.62 vs 0.465, 6/6 lb — these satdiv numbers are quoted from the scope smoke `ad0648672` + board #135 per builds_on, not this finding's ridge runs) <!--derived-->
— but it is **not a capability GO** (0/6 on the +0.10-over-V1-direct bar). Its own diagnostic
flagged the readout, not the norm, as the next lever (the config-C centroid read *beat* the `z`-tuned ridge). A
ridge re-tune sweep {0.05, 0.1, 0.25, 0.5, 1.0} on the 3 explore seeds now confirms that lever, too, **plateaus**:

**Artifact:** research/findings/raw/lanes/perception/vlin_satdiv_ridge0.25_explore.json (+ ridge {0.05, 0.1, 0.5, 1.0}, same dir; satdiv+ridge explore, seeds 42/43/100).

| ridge | verdict (explore 42/43/100) |
|---|---|
| 0.05 | PARTIAL — beats NOGO floor 1/3, load-bearing 3/3 |
| 0.10 | PARTIAL — beat 1/3, lb 3/3 |
| **0.25** | PARTIAL — **beat 2/3**, lb 3/3 |
| 0.50 | PARTIAL — beat 1/3, lb 3/3 |
| **1.0** | PARTIAL — **beat 2/3**, lb 3/3 |

Re-tuning the readout for satdiv's feature geometry lifts the best case to beat-2/3 (from the z-tuned readout's
lower score) but **no ridge value clears the capability bar on all seeds** — the load-bearing signal is real and
robust (lb 3/3 everywhere) yet the *separability* ceiling does not move enough. Readout-side improvement
(normalization form + ridge strength) is therefore **exhausted as a method**: it improves the margin without
crossing the bar.

## The NO-DEFER handoff: the residual is the FROZEN RANDOM S2 template bank

The mechanism this points to next (the wall-reframe "what does the real system run alongside this that we
replaced with a constant?"): the S2 template bank is **fixed and random** (template learning was ruled out of
scope back at #72). If the class information a frozen random bank carries is what caps separability, then **no
readout-side normalization — however biologically correct — can pass it**, which is exactly the plateau observed.
The next mechanism is therefore **activity-dependent (Hebbian / BCM) tuning of the S2 templates themselves** —
learn the intermediate features from the data instead of sampling them randomly (Olshausen & Field 1996 sparse
coding; a competitive/LCA step, Rozell et al.; cerebellar-style pattern separation with learned expansion,
Litwin-Kumar et al. 2017). A cheaper intermediate rung also remains: **competitive sparse coding (k-WTA / lateral
inhibition) at S2/C2** (k-WTA already exists at S1 but was never applied at S2), which #75b's own residual list
flagged as its missing piece.

## Honest scope / caveats

- **Explore seeds only (42/43/100).** The held-out seeds (44/101/102) were NOT run for the ridge sweep, so
  "beat 2/3" is an exploration signal, not a 6-seed verdict. Given every ridge value stays PARTIAL on the
  explore set, a full-6 confirm was not spent — the verdict "readout-side plateaus" holds either way, but a
  headline capability number would need the full 6.
- The RATE-ceiling gain could be partly a dynamic-range/variance artifact on 6 examples/class (the satdiv scope
  flagged this); the held-out discipline is what would catch it and was not run.
- `satdiv` is merged as an additive lever (default `z`, byte-identical unless `--s2-norm satdiv`), commit
  `ad0648672`. This finding adds no production wiring and flips no default — it is a lane characterization + a
  NO-DEFER pointer to the next mechanism (S2 template learning), not a capability landing.

**Next queued rung:** scope + de-risk activity-dependent S2 template learning (Hebbian/BCM or sparse-coding),
with a competitive k-WTA-at-S2 as the cheaper first step.
