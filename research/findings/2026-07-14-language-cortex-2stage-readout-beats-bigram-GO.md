# The recurrent language cortex's "bigram-level" ceiling was a LINEAR-READOUT limitation — a 2-STAGE cortical read-out BEATS the bigram (anti-cheat-verified), the cheap biological fix the controlled-lag de-risk identified

**Date:** 2026-07-14 · **Status:** GENUINE, **6-seed, DOUBLE-anti-cheat-verified** (42/43/44/100/101/102): 2-stage CE **2.174** vs bigram 2.962 vs linear 3.192; **BOTH anti-cheats collapse — permuted-corpus 3.175 (>bigram) AND shuffled-state 4.856 (≈ chance log(147)=4.99).** numpy CPU; NO `sim/` edit.

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
| permuted-corpus 2-stage (ANTI-CHEAT) | 3.175 | COLLAPSES (> bigram) |
| shuffled-state 2-stage (ANTI-CHEAT) | 4.856 | COLLAPSES HARD (≈ chance) |

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

## GENERALIZATION TEST — WikiText (natural corpus): the win is SCOPED to exploitable higher-order structure (2026-07-15)

The honest generalization check (per "run the ceiling early"): does the 2-stage read-out advantage hold on a NATURAL
corpus (WikiText), or is it specific to the templated EMERGE stream's strong word-order structure? 6-seed
(42/43/44/100/101/102), WikiText, V=300, ~5,092 held-out positions, well-sampled bigram (CE 3.484):

| | 2-stage CE | vs bigram 3.484 |
|---|---|---|
| seed 42 / 44 / 100 | 3.424 / 3.443 / 3.436 | beats by ~0.05 (anti-cheats collapse) |
| seed 43 / 101 / 102 | 3.550 / 3.524 / 3.573 | does NOT beat |
| **mean** | **~3.49** | **≈ bigram (not a robust beat)** |

**⇒ SCOPED.** On natural WikiText at tractable scale the 2-stage read-out is essentially BIGRAM-LEVEL (mean ≈ bigram,
±0.05, mixed across seeds), vs the robust 0.79-nat win on the TEMPLATED EMERGE stream. This CONFIRMS the owner's
ceiling finding (`feedback_run_ceiling_early_and_keep_gpu_busy`): natural language at tractable scale (few-M tokens /
V=300) is bigram-dominated — the higher-order signal is too thin for even a good model to beat a well-sampled bigram.
**The read-out DEPTH was the bottleneck for exploiting EXISTING (templated) structure; the SCALE/DATA (the thin
natural-language higher-order signal) is the bottleneck for natural language — the read-out depth does NOT overcome the
scale wall.** Not a wall to declare-and-stop: it re-localizes the natural-language frontier to (a) the scale/data lever
(more tokens → richer signal, but it races the bigram) and (b) the recurrent-cortex LEARNING frontier (a spiking
recurrent cortex that learns long-range structure — the genuinely-hard, field-open problem the controlled-lag de-risk
mapped, where e-prop's diagonal-RTRL is limited).

## Bottom line

A cheap, biologically-innocuous change — a **2-stage cortical read-out** in place of the linear softmax — robustly
lifts the fixed-reservoir language cortex to **0.8 nats better than the bigram on the templated EMERGE stream**
(6-seed, double-anti-cheat-verified: it exploits that stream's strong higher-order structure the linear read-out
couldn't). But on **natural WikiText at tractable scale it is bigram-level** — the read-out depth exploits existing
structure and does not overcome the documented scale/data wall. HONEST SCOPE: the fix is a genuine improvement where
exploitable higher-order structure exists; natural-language emergence at scale remains gated by the scale/data +
recurrent-learning frontier. The controlled-lag de-risk's fix transfers to structured language; the recurrent-credit
rule was never the bottleneck for that — the read-out depth was — and the natural-language ceiling is the scale wall.
