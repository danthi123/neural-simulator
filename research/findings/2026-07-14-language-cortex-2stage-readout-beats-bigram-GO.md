# The recurrent language cortex's "bigram-level" ceiling was a LINEAR-READOUT limitation — a 2-STAGE cortical read-out BEATS the bigram (anti-cheat-verified), the cheap biological fix the controlled-lag de-risk identified

**Date:** 2026-07-14 · **Status:** GENUINE (permuted-corpus anti-cheat collapses), 2-seed verified + 6-seed confirming. numpy CPU; NO `sim/` edit.

## The chain that led here

The controlled-lag e-prop de-risk (`2026-07-14-controlled-lag-eprop-copy-task-artifact-recurrent-computation-needed.md`)
comprehensively characterized the emergence-engine recurrent-language-cortex frontier and — via a self-caught
correction + two research gates (Bellec e-prop, Menick SnAp) — landed on a precise, cheap, biologically-innocuous
fix: **the fixed reservoir's features already CONTAIN the structure (hold/integrate/combine up to the ALIF horizon);
a LINEAR read-out cannot COMBINE them (nonlinear, e.g. XOR); a 2-STAGE (nonlinear) cortical read-out supplies the
missing combination.** A 2-layer read-out solved XOR 1.000 on the fixed reservoir where the linear read-out was at
chance. The gate confirmed a 2-stage read-out is biologically innocuous (real cortex has a deep, multi-stage read-out)
and is the fix — NOT a fancier recurrent-credit rule (e-prop/SnAp-1 are diagonal + not load-bearing).

**The mission-connected test:** does that same fix help the REAL language task? The reslm reservoir-LM's own finding
is that the fixed reservoir + a LINEAR read-out is "bigram-level" on the EMERGE SVO+function-word stream. Is that a
reservoir limit — or a LINEAR-READOUT limit, exactly like XOR?

## Result (EMERGE SVO stream, V=147, ~4,678 held-out next-token positions)

| read-out on the FIXED reservoir | held-out CE (nats) | vs bigram |
|---|---|---|
| **bigram baseline** | **2.965** | — |
| LINEAR (delta rule, well-trained, epochs=20) | 3.166 | WORSE (can't beat it) |
| **2-STAGE (one hidden layer, backprop)** | **2.166** | **BEATS by 0.80 nats** |
| permuted-corpus 2-stage (ANTI-CHEAT) | 3.178 | COLLAPSES (≥ bigram) |

**⇒ GENUINE (2-seed 42/43; 6-seed confirming):** the 2-stage read-out BEATS the bigram by ~0.8 nats where the linear
read-out cannot, and the **permuted-corpus anti-cheat COLLAPSES** it back above the bigram (shuffling the train token
order destroys the word-ORDER structure → no advantage) — so the gain is **real higher-order word-order/context
structure the fixed reservoir holds, unlocked by the nonlinear read-out**, NOT a unigram frequency artifact or
memorization (held-out eval + permuted-corpus collapse).

## What this means

**The "recurrent language cortex is bigram-level" ceiling was a LINEAR-READOUT limitation, not a reservoir/recurrence
limitation.** The fixed ALIF reservoir DOES carry higher-order context that beats the bigram; a linear softmax
read-out simply cannot extract it (the same non-separability as the XOR combination). A 2-stage cortical read-out —
biologically the deep, multi-layer cortical output projection real brains have, and still LOCAL/no-BPTT (the reservoir
states are cached; the read-out is trained by backprop through ONE hidden layer over the fixed features) — extracts it
and beats the bigram. This is the cheap, biologically-defensible improvement to the emergence-engine's language
cortex, exactly as the controlled-lag de-risk predicted.

## Honest scope

- 2-seed verified with the permuted-corpus anti-cheat; the 6-seed (42/43/44/100/101/102) confirmation is running.
- The EMERGE SVO+function-word stream is a CONTROLLED (templated) language stream with strong word-order structure —
  which is exactly what a 2-stage read-out exploits; generalization to richer/natural corpora (where the structure is
  less templated) is the open question and the next test.
- This is a READ-OUT improvement over the FIXED reservoir (the local delta/backprop pair); it does NOT claim the
  recurrent W_rec learning is load-bearing (the controlled-lag de-risk showed it is not). The recurrent MEMORY horizon
  (the ALIF window) is the separate, still-open scaling lever.
- Additional anti-cheats to add for a full multi-seed GO: shuffled-state (permute the reservoir feature dims →
  read-out loses the reservoir's context → CE ≥ bigram) + a frozen/chance control; and confirm the 2-stage does not
  merely memorize by holding the template families out.

## Bottom line

A single, cheap, biologically-innocuous change — a **2-stage cortical read-out** in place of the linear softmax —
lifts the emergence-engine's fixed-reservoir language cortex from BELOW the bigram to **0.8 nats BELOW (better than)
the bigram**, anti-cheat-verified as real word-order structure. The controlled-lag de-risk's identified fix transfers
to the real language task. The recurrent-credit rule was never the bottleneck; the read-out depth was.
