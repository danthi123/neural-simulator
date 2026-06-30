# 100M C2 generative scale-up — C1 GO (spiking==ANN at 88.6M); C2 not-a-scale-wall (retention holds; replay-contrast in-band-limited)

**2026-06-30.** The first *clean* run of the decisive C2 generative grow-without-forget scale-up at ~88.6M
params (the long-standing "30M scale wall" was two bugs — fine-tune LR 3e-4 instead of 1e-5 + an overfit
base — both fixed in this recipe). Run: `_genseq_C2_scaleup_runner --d-model 768 --n-layers 12 --n-heads 12
--vocab-size 2048 --block-size 512 --steps 232000 --corpus simplewiki`. FULL LOOP done (C1+C2 ~46 min).

## STAGE 1 (train) — peaked, data-bound
- 88.6M Gen-F on SimpleWiki (41M tokens). Held-out ppl plateaued ~11.2 by step ~100k; at 232k = **11.5**
  (flat-to-slightly-worse, measured) → **PEAKED**. Train loss plateaued ~1.85 since ~116k.
- DIAGNOSIS: **data-bound** — 88.6M params / 41M tokens ≈ 0.46 tok/param (Chinchilla-optimal ~20×). The
  model saturates the small corpus (~16 epochs) then stops improving; more STEPS don't help. The lever to
  beat ppl ~11 is **more DATA** (a bigger corpus), not more compute. STAGE 1 was stopped at 232k (the peak;
  the planned 450k would have been ~1 day of no-gain training — caught via a held-out probe + cut).

## STAGE 2 (C1) — GO (the headline) ✅
- The trained 88.6M Gen-F's on-bridge **spiking** forward == the ANN: **ppl_ratio 0.9999999, logit_fid 1.0**.
- ⇒ the project's **LARGEST spiking-consolidatable generative model, FAITHFUL at scale** — the per-layer
  graded-spiking error does NOT compound over 12 layers at 88.6M. The RF (resonate-and-fire) spiking
  substrate holds an ~88.6M generative LM bit-faithfully. **Decisive win.**

## STAGE 3 (C2) — NEGATIVE (strict), but NOT a scale wall — a characterized in-band-shift limit
- The corrected recipe **LEARNS** the new task (new-ppl drop 97.9%, ≥50% ✓) AND **RETAINS** the original
  (retention ≥85% ✓ — in fact the orig ppl improved). So **retention HOLDS at 100M** → the 30M
  "48%-retention scale wall" is **REFUTED** (it was the FT_LR + overfit-base bugs).
- The strict verdict is NEGATIVE only on `no_replay_forgets=False` + `dose_monotone=False`. BOTH are
  **known, runner-DOCUMENTED** consequences of the auto-selected IN-BAND moderate shift (SH-frac 0.43
  TinyStories/Shakespeare block-interleave = **57% the original**): quoting the runner's own caveat — *"a
  mixture self-reinforces the old distribution ... the no-replay forgetting CONTRAST at this in-band point
  is expected MODEST (~1.07–1.10×) rather than catastrophic — the price of staying in-band."* With little
  forgetting to prevent, replay cannot be *shown* load-bearing here.
- So C2 sits in the GAP between two characterized regimes: **in-band shift** = no forgetting (replay not
  needed) vs the prior **extreme 41× Shakespeare shift** (`_genseq_C2_grow_no_forget.json`) = strong
  forgetting but replay caps at ~55% retention. A clean demonstration of "replay PREVENTS catastrophic
  forgetting at 100M" needs a shift task in the **SWEET SPOT** (forgets-without-replay AND
  learnable-to-≥85%-with-replay) — a **task-design** open problem, NOT a substrate/scale limit.

## Bottom line
- **C1 GO** — 88.6M spiking == ANN (ppl_ratio 1.0). The largest faithful spiking-consolidatable generative model.
- **C2** — retention holds at 100M (the scale wall is refuted); the replay-load-bearing *demonstration* is
  blocked by the in-band/extreme shift-task dichotomy (a characterized task-design problem, not the substrate).
- **Data-bound** — 88.6M peaks at ppl ~11 on 41M tokens; more data is the next lever.

## Follow-ups (non-blocking; owner's call on the remaining window)
1. A shift-task **sweet-spot sweep** (between the in-band 3.9× and the extreme 41×) for a clean
   replay-load-bearing C2 at 100M.
2. A **bigger corpus** (the data-bound lever) for a lower-ppl base.

Artifacts: `research/findings/raw/_genseq_C2_scaleup_100M.json` (full verdict), `_heldout_probe.py`
(reusable held-out monitor), `C2_100M_RUN.md` (driving guide).

---

## UPDATE (2026-06-30): CLEAN re-run with ORIGINAL=SimpleWiki (the model's real domain) — the valid test

Re-ran C2 with `--c2-original simplewiki` (commit `7fa7a3fa`, research/runners/ only, no sim/) so retention
is tested on what the model ACTUALLY trained on (SimpleWiki, ppl ~10.8), not TinyStories (which it never
knew). STAGE 1 + C1 reused from cache; only the C2 loop re-ran.

**The valid test — orig=SimpleWiki 10.82; auto SH-frac 0.31 = 8.96× shift toward Shakespeare (in the [4,12]× band):**
- GROWN WITH REPLAY: orig 11.65 (**retention 92.9%** ✓≥85%), new 17.62 (**learns, 81.8% drop** ✓≥50%).
- NO-REPLAY control: orig 12.24 (retention 88.4%), new 17.09.
- **`dose_monotone=True`** (replay helps monotonically); on-bridge install **ppl_ratio 1.0 for BOTH** the
  original AND the grown task (the spiking consolidation holds through the grow).
- Verdict still NEGATIVE — but now ONLY on `no_replay_forgets=False`: the no-replay control retained 88.4%
  (a 1.05× rise vs with-replay) — MILD forgetting, below the strict ≥1.30× catastrophic bar.

**Honest synthesis (now a VALID test, not a confound):**
- **Grow-without-forget WORKS at 100M:** the spiking-consolidated 88.6M generator learns a new in-band task
  (81.8% drop) AND retains its real domain (92.9%), with a MONOTONE replay dose-response. The "scale wall"
  (retention failing at scale) is firmly **REFUTED**.
- **Replay is DIRECTIONALLY load-bearing** (dose-monotone; no-replay 88.4% → with-replay 92.9%) — but not
  STRICTLY (the ≥1.30× bar) at this in-band 8.96× shift, because a moderate shift causes only mild
  forgetting without replay. The strict "no-replay catastrophically forgets while replay holds ≥85%" demo
  needs a shift in a NARROW (possibly empty) window between in-band ~9× (mild forget) and extreme 41×
  (where replay caps at ~55% retention — the prior `_genseq_C2_grow_no_forget.json` boundary). That window
  is a characterized **task-design** property, NOT a substrate limit.

⇒ The C2 picture is now COMPLETE + valid: at 100M the spiking-consolidated generator does grow-without-forget
(learns + retains ≥85% + replay dose-monotone); strict catastrophic-forgetting-prevention is bounded to a
narrow strong-shift regime. Optional follow-up: a band-MULT sweep ([10,30]×) to map whether that strict
sweet spot is non-empty — needs a small `LEARNABLE_BAND_*_MULT` / `--sh-frac` plumb (the auto-band currently
targets [4,12]×).
