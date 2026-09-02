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

## Update (same day): the cheaper first step (k-WTA-at-S2) is the best lever yet but STILL plateaus — confirming the templates must be LEARNED, not just sparsified

The competitive sparse-coding first rung — **k-WTA across the S2 template bank** (Foldiak 1991 / a hard-threshold
LCA approximation, Rozell et al. 2008; zero all but the top-`frac` of templates per patch-location, attacking the
diagnosed common-mode directly) — was built (`--s2-kwta-frac`, byte-identical-off, commit `bbe8ab27`) and is the
**best readout/sparse-coding lever in the whole arc**: on the 3 explore seeds it produced the first non-zero
`capability_go` count (1/3 at frac 0.25, vs 0/6 for satdiv/ridge/granule). But the **full-6-seed confirm holds the
same plateau**:

**Artifact:** research/findings/raw/lanes/perception/vlin_kwta0.25_6seed.json (+ vlin_kwta0.30_6seed.json).

| lever (full 6 seeds) | verdict |
|---|---|
| k-WTA frac 0.25 | PARTIAL — beats NOGO floor **3/6**, load-bearing **6/6** |
| k-WTA frac 0.30 | PARTIAL — beat 2/6, lb 6/6 |

beat-3/6 + lb-6/6 is a genuine step past satdiv/ridge (beat-2/6) — but it is **not** the ≥5/6 capability GO, and
the explore-set lead did NOT strengthen on the held seeds (the exact satdiv-style regression flagged as a risk).
**Conclusion tightened:** *rescaling* (satdiv) and *sparsifying* (k-WTA) the frozen random S2 bank both help and
both plateau — so the residual is confirmed to be the **information the frozen random templates carry**, not how
their responses are normalized or thresholded. The decisive next mechanism is therefore **BCM sliding-threshold
Hebbian LEARNING of the S2 templates** (Bienenstock, Cooper & Munro 1982) — already validated on this exact
substrate (the 2026-08-26 on-bridge BCM finding broke the identical V1 common-mode boundary 62×; `sim/config.py`
`hebbian_bcm`). k-WTA + BCM compose (sparsify + learn), so the k-WTA lever stays as a component. **Now de-risking
BCM S2-template learning directly** (honest risk: 6 examples/class is thin for a stable per-unit sliding
threshold — that is the live confound to design against).

**Next queued rung:** de-risk BCM sliding-threshold learning of the S2 templates (compose with the k-WTA lever).
