# Deep-credit-on-spikes external-literature scan (2024-26, adversarial Workflow): a GENUINE OPEN FRONTIER — the field has no scale-independent mechanism past our boundary; our committed K-normalized pool-k test is the correct next move; the new imports (DeepESN stack, Hebbian reshaping, ESPP, GLE, Dynamic-PC) are complementary scale-gated levers

**Date:** 2026-07-12
**Method:** a 4-agent adversarial deep-research Workflow (`wf_dd83a73e-74d`; 3 parallel lenses — external deep-credit-on-spikes lit / external reservoir-past-n-gram lit / our-own-D1-boundary — + a skeptical synthesizer; ~824k subagent tokens, 58 tool-uses, all citations agent-verified against arXiv/journal full text). Fired per the emergence-bar frontier (the R3↔generation convergence: the fixed-reservoir generator is n-gram-bounded; the path past it = learned representations / deep credit on spikes). **The controller reads the decision-critical sources itself before building** (see the separate Ueda-2025 read).
**Status:** scoping map — a genuine open frontier confirmed, with a ranked cheapest-first ladder + the near-equivalents skeptically ruled out. NO build yet.

## THE HONEST VERDICT (skeptical synthesizer, verbatim gist)
The 2024-26 field does NOT contain a mechanism that clears our boundary in a **scale-independent** way — a genuine open frontier. The boundary's two halves resolve differently:
- **(1) Per-sample spike-count credit noise** — the field has NOTHING past it: every genuinely-new local spiking rule (ESPP, TESS, S-TLLR, Forward-Forward) still reads FINITE per-sample spike counts, so none denoises per-sample credit better than population/ensemble averaging — which is **exactly our committed K-normalized pool-k de-risk** (`--lr 0.25/K --ff-w-init 4.5/K`; decisive read = does `corr(pooled-E, soma-rate)` rise with K). ⇒ importing an external rule would not address this half; pool-k IS the correct next move (it is running).
- **(2) Fading-memory / long-range** — our R3 reframe ALREADY localized this to input-representation + read-out (recurrent credit is the wrong bottleneck; training W_rec is counterproductive) and ALREADY produced the positive (frozen reservoir + e-prop-learned W_in + Kolen-Pollack feedback ≈ 78% of frozen-reservoir-BPTT, rate, 6-seed). **The single most decision-critical external fact — Ueda et al. 2025 (arXiv:2503.01724): a fixed reservoir ≈ n-gram until 16-65k units + ~100M words and STILL loses to a 512-unit LSTM — CONFIRMS our scale-confound** (no local mechanism beats a bigram at 5M-tok/V=300, so our reservoir long-range NEGATIVES are partly scale-bound, not mechanism failures).

⇒ the genuinely-new external imports are **COMPLEMENTARY input-representation levers**, worth cheap-first de-risks in the ranked order below, but **each is expected scale-gated per Ueda, and NONE is a silver bullet past the boundary.**

## The ranked cheapest-first de-risk ladder (all reuse-by-import, all with the ceiling-first + anti-cheat discipline baked in)
1. **DeepESN multi-timescale reservoir STACK** (staggered leak rates, only the read-out trained) — NEW vs our FLAT single OnBridgeLSM; NOT the ruled-out W_rec gradient, NOT the refuted dual-timescale ELIGIBILITY (this is a FORWARD-STATE multi-timescale representation). De-risk: OnBridgeLSM ×N with per-layer leak {0.9,0.5,0.1}, concat frozen states → existing read-out. Anti-cheats: leak-shuffle collapse · **size-matched flat-reservoir** control (must beat SAME unit budget, not a smaller one) · beat-bigram, CEILING-FIRST at TinyStories/WikiText-103. READ: **Ueda et al. arXiv:2503.01724** (the BLiMP scaling table) + Gallicchio-Micheli DeepESN + ICANN-2025 time-scales-in-DeepESN (doi:10.1007/978-3-032-04552-2_18).
2. **HAG — unsupervised Hebbian reservoir RESHAPING/GROWTH** (fire-together-wire-together sculpting of the recurrent wiring; ZERO gradient) — squarely the "structure must self-organize" directive; NOT the ruled-out W_rec *gradient* credit. De-risk: local co-activation growth on OnBridgeLSM's recurrence before the read-out. Anti-cheats: random-growth control · edge-count-matched frozen baseline · beat-bigram. READ: **Cazalets & Dambre, Nature Comms 2025, doi:10.1038/s41467-025-67137-1** (VERIFIED).
3. **ESPP — EchoSpike Predictive Plasticity** (fully-spiking, self-supervised local predictive+contrastive rule growing codes predictive-by-construction). De-risk: numpy ESPP layer on the token stream → same read-out. Anti-cheats: echo-shuffle collapse · head-to-head vs our PPMI/Hebbian codes at matched dim · beat-bigram. READ: **Rassmann et al., arXiv:2405.13976** (VERIFIED).
4. **GLE — Generalized Latent Equilibrium** (prospective-coding TEMPORAL credit: per-neuron temporal differentiation → a local three-factor rule online-approximates BPTT-through-TIME, no history buffer; per-neuron τ = multi-timescale) — the missing TEMPORAL half that composes with our dendritic (M2.6/D2) SPATIAL credit. Rate; needs a spiking port. READ: **Ellenberger/Haider/Jordan/Senn, Nature Comms 2025 (s41467-025-66666-z; arXiv:2403.16933)** (VERIFIED).
5. **Dynamic Predictive Coding** (higher levels predict lower-level TRANSITION DYNAMICS → an auto-grown timescale hierarchy where the top level encodes a whole sub-sequence). Rate; needs a spiking port. READ: **Jiang & Rao, PLOS Comp Biol 2024, 10.1371/journal.pcbi.1011801** (VERIFIED).

## Skeptically ruled out (near-equivalents to what we already own — avoids manufacturing novelty)
modern-Hopfield / key-value read = our VSA composer / fact-store; reservoir-DFA = our M2.6 feedback alignment; TESS / S-TLLR = our eligibility + feedback-alignment family (the same "generic partiality" we characterized); BDSP/Burstprop = already on-bridge.

## ⇒ Next actions (in priority)
1. **The credit-noise half:** our K-normalized pool-k test (running, `bbaxfaczk`) is the correct + only move the field offers — read its result.
2. **Read Ueda et al. 2025 (arXiv:2503.01724) MYSELF** — the decision-critical scale fact (maps whether the reservoir/scale path is viable at all, and the 16-65k-units/100M-words regime; it independently confirms the same-day data-scale finding).
3. **The long-range half:** the ranked complementary levers (DeepESN stack #1 = cheapest + reuse-by-import) — de-risk at genuine (TinyStories/WikiText-103) scale, expecting scale-gating; and/or the SPIKING realization of our own R3 positive (learn-W_in + KP) at scale.

## ⭐ CONTROLLER READ THE DECISION-CRITICAL SOURCE MYSELF (Ueda et al. 2025, arXiv:2503.01724) — and it CORRECTED the summary
Per the discipline (the Workflow LOCATES; the controller READS the load-bearing source), I read Ueda et al. myself. Exact numbers (BLiMP syntax, ~100M words BabyLM, single epoch; chance = 50%):
| model | BLiMP | note |
|---|---|---|
| ESN reservoir 1,024 units | 56.2% | |
| ESN 4,096 | 57.9% | beats GPT2-Scratch on val NLL from here |
| ESN 16,384 | 59.2% | **beats GPT2-Scratch Transformer (58.7%) on BLiMP** |
| ESN 65,536 | 60.5% | the reservoir's bounded ceiling |
| **LSTM 512 (gated)** | **67.8%** | best from-scratch; the ESN NEVER matches it |

Architecture: `Wrec`/`Win`/leaks FIXED, only `Wout` trained (low-rank + bias), NO BPTT — our exact setup. **The summary said "loses to a 512-unit LSTM" (TRUE) but MISSED the paper's headline positive: the fixed reservoir DOES beat a from-scratch Transformer on syntax at scale (≥16k units / 100M words)** — reading the source caught the over-negative framing.

**⇒ The precisely-corrected frontier map (load-bearing):**
1. Our reservoir generator being n-gram-bounded at n_pool=300 / ~1.7M words is **EXPECTED, not a mechanism failure** — we are 50-200× below Ueda's syntactic-capability scale (16k-65k units / 100M words). This THIRD-PARTY confirms the same-day data-scale finding + the R3 scale-confound.
2. The fixed reservoir reaches a **genuine but BOUNDED** syntactic ceiling (~60% BLiMP, barely above the from-scratch Transformer) at large scale — so SCALE is a real lever for the reservoir, up to that bounded ceiling, but reaching it needs a ~16k-neuron spiking reservoir + ~100M words (the Izhikevich CUDA-graph infra + a big corpus).
3. The genuine path PAST the reservoir's bounded ceiling toward real fluency (the LSTM's 67.8%) is **LEARNED RECURRENCE / gates = deep credit on spikes** — exactly our standing dendritic frontier (and exactly the LSTM-beats-ESN gap Ueda quantifies). The reservoir's fixed recurrence is the ceiling; learning it (the coarse-credit boundary) is the way up.

⇒ both paths are genuine + now precisely quantified: (a) scale the reservoir toward its bounded ~60%-BLiMP ceiling (infra-gated); (b) learned recurrence / deep credit to exceed it (the coarse-credit frontier). The DeepESN multi-timescale stack (#1 lever) is the CHEAP test of whether multi-timescale FORWARD state reaches the reservoir's ceiling with fewer units than raw scale.

## Files
Workflow `wf_dd83a73e-74d` (transcript + journal). Ueda et al. 2025 arXiv:2503.01724 (controller-read). Builds on the R3↔generation convergence (AUTONOMOUS_STATE 2026-07-12), the D1 pool-k finding (`2026-07-10-D1-onbridge-deep-credit-poolk-...`), the SCALE/SYNTHESIS generation findings.
