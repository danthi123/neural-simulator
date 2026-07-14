# The TRAINED selective channel over a FIXED reservoir robustly beats the memoryless BAG (+0.62→+1.13, 6-seed) and the LEARNED gate is the ingredient — but the "past-the-Ueda-bound / genuine-long-range / fluency-scaling" framings were OVERCLAIMED (adversarial-verify, 2 material refutations) and are RETRACTED below

## ⚠️ CORRECTION (2026-07-13, post-adversarial-verify — 3-skeptic Workflow, 2 material refutations)

A mandatory adversarial-verify (3 independent skeptics) found this finding's grander framings overclaimed. **SURVIVES / RETRACTED:**

- **SURVIVES (the robust core):** over a small FIXED echo-state reservoir, a TRAINED selective channel robustly adds ~+0.62 nats (V=120) up to **+1.13 (6-seed, V=600/nt=5000)** over the **memoryless BAG**, and the **LEARNED gate is the ingredient, NOT read-out capacity** — Skeptic A verified in code that the fixed-gate control has IDENTICAL read-out dimensionality yet HURT (−0.076) while training the same gate lifts +0.62 ("more params fit better" directly falsified). This is a real, 6-seed-robust result.
- **RETRACTED — "past the reservoir's Ueda-bound" (Skeptic B, material):** `m_res→+0.006` is **V=120-ONLY, where my fixed 300-neuron echo-state reservoir is WORSE than a bigram** (res_ce 3.14 vs bigram 2.64) — an under-capacity WEAK reservoir converging to the unordered bag, NOT the info-theoretic n-gram bound (at V=300 `m_res` stays +0.22 at nt=3200). The reservoir-scale Ueda-bound is a real documented result, but THIS runner's weak fixed reservoir does not demonstrate carrying "past" it.
- **RETRACTED — "deep-distributed / genuine long-range" (Skeptic C, material — a metric bug I made):** the per-depth `sel_lift = (bag−sel)−(bag−res) = res−sel` — **the bag CANCELS**, so it measures sel-vs-RESERVOIR CE, NOT a margin over the bag (my "bag-not-weakens-deep" self-check missed this cancellation). It is dominated by a shallow current-token readout fix (the clean `(1−λ)·Win·E[tok]` injection the degraded reservoir lacks at ALL depths), and the selective-SPECIFIC deep component is ~0 at d≥10 — exactly what the SIBLING frozen-coupling arc's fix/rand/noheld controls already walked back (sel<fix decays to +0.19, sel−rand −0.017 at d≥10). This runner omits those controls; its deep claim is not valid.
- **SCOPED — "grows with vocab = fluency direction" (Skeptic B):** the margin over the memoryless BAG grows with vocab, but that is confounded by **fixed-nt bigram starvation** (the ordered bigram is data-starved at higher V at fixed nt); against the ordered bigram in the DATA direction the advantage SHRINKS (V120 +0.31→+0.10, V300 +0.48→+0.38). So it is a robust order/gated-recency win over a WEAK fixed-reservoir+bag baseline, NOT a demonstrated fluency-scaling vs the proper ordered baseline.

**The valid long-range result is elsewhere and UNAFFECTED:** Rung 3 (`-RUNG3-selective-SSM-beats-reservoir-on-REAL-TEXT-deep-context.md`) shows the selective SSM beating the fixed reservoir AND the ordered BIGRAM at deep context on real text, 6/6, WITH the fix/detached/randgate controls — that is the load-bearing long-range demonstration, and the frozen+joint+on-bridge coupling (separately adversarially verified) stands. The overclaims are specific to THIS margin-over-bag scale probe.

**Airtight follow-on (per the skeptics):** re-run this runner's OWN no-hold (λ=0) + random-gate controls per-depth at d≥10, multi-seed, and compare against the ordered bigram (not just the order-blind bag), to establish (or bound) any genuine distal-holding component.

---

### (Original write-up below, superseded by the CORRECTION above for the retracted framings; the raw margin-over-bag numbers + the learned-gate-is-the-ingredient result remain accurate.)

# The TRAINED selective channel over a FIXED reservoir robustly lifts margin-over-bag (~+0.62, 5/5) where the FIXED gate HURT [FRAMING RETRACTED: see CORRECTION — the reservoir is weak, not at its Ueda-bound]

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_batched_scale_trained_selssm_derisk.py` · raw `research/findings/raw/_trainedsel_scale/`. numpy (GPU-capable batched reservoir); NO `sim/` edit.
**Status:** ✅ decisive + robust — the LEARNED gate is the scale-critical ingredient (settles the fixed-gate negative). Tractable + GPU-scalable via the batched infra.

## Why

The fixed-gate scale probe was a NEGATIVE (an untrained selective channel HURTS margin-over-bag → the learned gate is required). This trains the gate over a FIXED echo-state reservoir (batched-collected, fast) — cheap (no O(n²) reservoir e-prop) and GPU-scalable toward the validated regime — and asks whether a TRAINED selective lifts margin-over-bag where the fixed one hurt. Everything transport-free (read-out local delta; gate forward eligibility × fixed random feedback, no BPTT/transport); the SAME simple trainer for both arms (fair).

## Result (np=200, V=120, TinyStories; margin-over-bag = bag_ce − arm_ce)

Per-run (seeds × nt) + the 3-point scale trend (means):

| nt | seed | m_res (res−bag) | m_sel (res+trained-sel − bag) | sel_lift | sel−bigram (aggregate) |
|---|---|---|---|---|---|
| 800 | 42 | +0.153 | +0.753 | **+0.600** | +0.256 |
| 800 | 43 | +0.176 | +0.782 | **+0.606** | +0.293 |
| 800 | 44 | +0.160 | +0.828 | **+0.667** | +0.317 |
| 1600 | 42 | +0.072 | +0.677 | **+0.604** | +0.158 |
| 1600 | 43 | +0.083 | +0.718 | **+0.635** | +0.205 |
| 3200 | 42/43 (mean) | +0.006 | +0.628 | **+0.622** | — |

**Scale-trend means:** m_res **+0.163 → +0.077 → +0.006** (nt 800→1600→3200) · sel_lift **+0.624 → +0.620 → +0.622**.

- **`sel_lift` is ROCK-STABLE ~+0.62 across 3 scales / 4× data** — the TRAINED selective decisively + robustly + DATA-INVARIANTLY lifts margin-over-bag, where the FIXED gate HURT (−0.076). The LEARNED gate is the ingredient (a fixed hold was noise).
- **THE DECISIVE CLAIM — the trained selective pushes PAST the reservoir's Ueda-bound.** The FIXED reservoir's OWN margin over the bag DECAYS to ~ZERO with data (m_res +0.163 → +0.006 — this IS the reservoir-scale Ueda-bound the prior finding CLOSED as a negative: at scale the fixed reservoir's recurrent dynamics become worthless, the memoryless bag catches up). Yet res+trained-sel STILL beats the bag by +0.628 at nt=3200 — **entirely from the learned selective channel** (the reservoir contributes ~0). ⇒ where the reservoir alone is Ueda-bounded (decays to the n-gram floor), the trained selective channel supplies a DATA-INVARIANT ~+0.62-nat durable memory. **The selective mechanism is precisely what carries long-range past the reservoir bound at scale.**

## VOCAB-scaling toward the validated regime (V=120 → V=300) — the selective's value GROWS with vocab

**MONOTONIC vocab-scaling (V=120 → 300 → 600, nt≈1600, means):**

| V | sel_lift (over bag) | sel−bigram | m_res (fixed reservoir − bag) |
|---|---|---|---|
| 120 | ~+0.62 | ~+0.29 | decays +0.16→+0.006 with data |
| 300 | **+0.80** | **+0.45** | +0.36 (decays →+0.216 with data) |
| 600 | **+0.86** | **+0.77** | +0.65 |

- **`sel_lift` (margin-over-bag) GROWS MONOTONICALLY with vocab**: +0.62 → +0.80 → +0.86 (V 120→300→600). The trained selective adds MORE over the bag at larger vocab.
- **`sel−bigram` GROWS STRONGLY with vocab**: +0.29 → +0.45 → +0.77 (V 120→300→600). By V=600 the selective beats the bigram by +0.77 nats — and still climbing.
- **The past-Ueda-bound pattern HOLDS at every V** (m_res decays with data at V=120 and V=300).
- ⇒ **as vocab scales toward the validated regime (V=120→300→600→…→2000), the trained selective's advantage over BOTH the memoryless bag AND the bigram GROWS MONOTONICALLY** — directly resolving the a-1 null-discriminator concern (the bigram-margin was thin at small V because the deep signal was thin at toy scale; it GROWS with V, the fluency direction). The mechanism scales in the right direction with vocab AND is past-Ueda-bound-durable with data.

## DEEP-DISTRIBUTED — the lift is genuine long-range, not shallow (V=300 by-depth, adversarial-verify discipline)

Per-depth `margin_over_bag` lift (bag−sel minus bag−res) at V=300/nt=1600: **d4-5 +0.776, d6-9 +0.617, d≥10 +0.478** (tiny-smoke V=80: +0.63/+0.50/+0.55). The trained-selective's lift over the memoryless bag is STRONG at ALL deep depths INCLUDING the deepest tail (d≥10 +0.48) — the selective holds distal context the bag lacks at long range. NOT a shallow readout fix (it persists deep). This answers the frozen coupling's adversarial-verify shallow-concern for the decisive margin-over-bag result: over a fixed reservoir + trained selective, the lift is deep-distributed and long-range.

**Bag-not-weakens-deep check (pre-empts the "the bag just gets noisier deep, so anything beats it" confound):** the bag control's CE is FLAT-to-BETTER at deep depth — bag_ce 4.603 (d4-5) → 4.542 (d6-9) → **4.468 (d≥10)** — NOT exploding. So the deep sel_lift is genuine, not a bag-weakens-deep artifact. What DOES change with depth: sel_ce RISES slightly (3.476→3.581→3.773 — holding distal context gets harder), while res_ce stays flat (~4.25); that is why the sel_lift SHRINKS with depth (+0.78→+0.62→+0.48) yet stays substantial at d≥10. The one remaining cleaner isolation (a no-hold current-token-only "noheld" control at the deep tail, as built for the FROZEN coupling — where holding beat no-holding at deep) is a follow-on for this fixed-reservoir runner; the frozen coupling's noheld result already established the held-context contribution at depth for the same mechanism.

## 6-SEED robustness at the decisive scaled config (V=600, nt=5000, vectorized trainer, full dev+blind seed set)

Fanned across the full 6-seed set (42/43/44/100/101/102, core-saturated, 1-thread BLAS): **`sel_lift` mean +1.127 (min +1.084, max +1.167, 6/6 > 0, tight)**; `sel−bigram` +0.024..+0.100 (sel beats the bigram on all 6 seeds). The lift GREW from +0.86 (V=600/nt=1600) to +1.13 (V=600/nt=5000) — the data-scaling holds at 6-seed. ⇒ the decisive "the trained selective lifts margin-over-bag, scaling with data AND vocab" is a proper 6-seed generalization claim, not a single-seed indicator. (Realized via the validated vectorized GPU-scalable trainer, `_reslm_scale_trained_selssm_vectorized_derisk.py`.)

## ⇒ honest read (adversarial-verify + null-discriminator disciplined)

- **Robust decisive claim:** over a fixed reservoir, a TRAINED selective channel adds ~+0.62 nats over the memoryless bag, robustly across seeds AND data-robustly (holding as the reservoir's own margin decays). The LEARNED gate is the scale-critical ingredient. This settles the fixed-gate negative and is realized at the batched-scale-infra level (GPU-scalable).
- **NOT over-claimed:** the aggregate `sel−bigram` SHRINKS with data (+0.29→+0.18) — the bigram is a strong, fast-improving baseline at this tractable (null-discriminator) scale; sel still beats it on aggregate but the margin narrows. Per the a-1 null-discriminator finding + the adversarial-verify shallow-concern lesson, the aggregate-vs-bigram is NOT the deep-tail claim; margin-over-bag is the robust headline. The absolute deep-TAIL-vs-bigram win needs the validated-signal regime (23.7M/V=2000), the named GPU follow-on (this runner is the tractable, GPU-scalable path to it — the reservoir is batched; the gate+read-out training is the cheap part).

## Next
- Scale this runner toward the validated regime (larger V/data on GPU; vectorize the gate+read-out training loop for the V=2000 / large-nt run — the reservoir collection is already batched).
- A by-depth breakdown (does the trained-sel margin-over-bag concentrate at the deep tail, as the joint runner's did?).
- raw `research/findings/raw/_trainedsel_scale/*.json`.
