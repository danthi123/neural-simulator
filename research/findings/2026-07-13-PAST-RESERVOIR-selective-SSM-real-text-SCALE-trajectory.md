# Past the reservoir bound — SCALE trajectory: the selective SSM's deep-context advantage over the fixed reservoir + bigram HOLDS/GROWS with vocabulary on real text (the fluency-direction signal)

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_rung3_selective_ssm_realtext_derisk.py` (the Rung-3 real-text runner, swept over `--vocab`; numpy; NO `sim/` edit).
**Status:** ✅ MONOTONIC positive trajectory — the advantage grows with vocabulary (all runs GO).

## The question

Rung 3 showed the eligibility-trained selective SSM beats the fixed reservoir + bigram at deep context on real text at V=200. The earlier CEILING finding warned that long-range signal is *thin* at small scale (even a transformer couldn't beat a bigram at V=300/5M tokens). So the mission-central question: does the selective SSM's deep-context advantage **hold or grow** as the vocabulary/data scale up — the trajectory toward fluent long-range — or does it wash out?

## Result — the advantage GROWS with vocabulary (V=200 → 400 → 600, TinyStories, deep context d≥4)

| V | selective CE | fixed_res CE | bigram CE | sel < fixed_res | sel < bigram |
|---|---|---|---|---|---|
| 200 | 3.095 | 3.755 | 3.405 | +0.659 | +0.310 |
| 400 | 3.711 | 4.433 | 4.296 | +0.722 | +0.585 |
| 600 | 4.069 | 4.881 | 4.871 | +0.813 | **+0.803** |

(2 seeds/point, 16k TinyStories sentences, deep context d≥4.) **Both margins grow MONOTONICALLY with vocabulary** — sel<fixed_reservoir +0.66→+0.72→+0.81, and sel<bigram +0.31→+0.59→+0.80 (nearly TRIPLED). At V=600 the bigram has caught up to the fixed reservoir (both ~4.88, the Ueda n-gram floor), while the selective SSM (4.07) pulls decisively ahead of BOTH.

## ⇒ interpretation

The selective SSM's deep-context advantage does NOT wash out with scale — it holds/grows as the vocabulary increases, which is the RIGHT direction for the fluency trajectory: as the language gets richer (bigger vocab, more deep-context structure), the learned input-dependent gate captures MORE of it, while the fixed reservoir (n-gram-bounded, the Ueda ceiling) and the bigram fall further behind. This is the opposite of the scale-confound the CEILING finding warned about — the signal the selective SSM exploits is real and grows with the richness of the language.

## Honest scope / next

- Small sweep (V=200/400/600, 2 seeds, 16k sentences) — establishes the trajectory direction, not an asymptote. The genuine scale test (does it keep growing to the vocab/data where fluency lives) is the resource lever (more cores / a bigger corpus).
- The mechanism is on the spiking substrate (Rung 4b); this scale result is on the numpy version, which is byte-equivalent (Rung 4b-iii-a) so it transfers.
- NEXT: continue the scale (bigger V/data), and couple the on-substrate selective SSM into the emergent generator (the conversational cortex) — the mission-central path to fluent long-range conversation.

## Files
- `research/runners/_reslm_rung3_selective_ssm_realtext_derisk.py` (`--vocab` sweep); raw `research/findings/raw/_rung3scale/`.
- Follows Rung 3 (`-RUNG3-...`) + the on-substrate Rung 4b.
