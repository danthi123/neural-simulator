---
type: finding
status: in-progress
date: 2026-09-05
mechanism: --recurrence deltanet (error-corrective delta-rule / Widrow-Hoff fast-weight write on the linattn substrate) — a WRITE-RULE fix to linattn's SAME KV trace, not a new content-addressing key
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42]
seed-waiver: the only run here is a labeled TINY CPU smoke (word-tokenizer, d_model=32, 1 epoch) that verifies the arm RUNS + is byte-identical-when-off + its anti-cheats collapse — NOT a generalization or a GO. The decisive wt103 direction-test (s43, then 6-seed if it lifts) is QUEUED on the GPU; no GO/NO-GO is claimed in this doc.
verdict: BUILT + CPU-smoke-verified (runs end-to-end; wkv/linattn byte-identical pre/post except wall-clock; both anti-cheats collapse); the decisive broad-domain (wikitext-103) direction-test is queued on the GPU for the controller to harvest — no fluency verdict yet
artifacts:
  - research/findings/raw/_deltanet_arm_numpy_smoke.json
  - research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json
---

# Delta-rule (deltanet) error-corrective write on linattn — BUILT, CPU-smoke-verified, wt103 direction-test queued

**Status: in-progress (BUILT, not yet adjudicated).** A new `--recurrence deltanet` arm is added to `research/runners/_emerge_wkv_lm_derisk.py`: it replaces linattn's additive Hebbian fast-weight write with the error-corrective **delta rule** (Widrow-Hoff; erase-before-write). It is additive, default-off, and byte-identical when off. This doc records the BUILD + the CPU smoke; it does **not** claim a fluency result — the decisive broad-domain test is queued on the GPU. No default is flipped.

## What this is (and, explicitly, what it is NOT)

`deltanet` is a **WRITE-RULE fix to linattn's SAME `D x D` fast-weight KV trace**. Query, key, value, feature map `phi`, learned per-channel decay `lam`, and the output gate are all IDENTICAL to `LinAttnLayer`; ONLY the line that updates the trace `M` changes:

- linattn (additive Hebbian):  `M_t = lam (*) M_{t-1} + phi(k_t) (x) v_t`
- deltanet (delta / Widrow-Hoff): `M_t = lam (*) M_{t-1} + beta * k_hat_t (x) (v_t - v_old_t)`, with `v_old_t = k_hat_t^T (lam (*) M_{t-1})` (the value CURRENTLY bound to the incoming key) and `k_hat = phi(k)/||phi(k)||_2`. Reading the just-written key back gives `v_t` exactly at `beta=1, ||k_hat||=1` — exact error correction. The read is `phi(q)^T M` (raw, as the whole DeltaNet family reads; see NORMALIZATION note in the class docstring).

**It is NOT a new content-addressing key**, and NOT a reproposal of the banked-and-exhausted content-addressing family (`assoc`/`assoc_t`/`learnkey`/`hippokey`, all of which lose to trigram). It is the same linattn fast-weight with erase-before-write. The point of the delta rule is a **bounded state norm under interference** (each write REPLACES, not ADDS, its key's binding), which directly targets linattn's measured failure mode.

## Distinct from the refuted 2026-07-15 edge5-rung3 delta-write (stated explicitly)

A delta-write WAS refuted once (`2026-07-15-edge5-rung3-delta-write-PARTIAL-error-correction-refuted`), so this distinction is load-bearing. That was a **DIFFERENT mechanism at a DIFFERENT scale**: a store-side ONE-SHOT ON-BRIDGE potentiate/depress for a spiking discourse BINDER (KV=4/8 value pools, P<=8 binds), refuted BECAUSE — in that finding's own words — the one-shot on-bridge write was *"too coarse to reproduce the numpy delta rule's ITERATIVE MATRIX error-correction"*, and the easy scale never triggered the error-correction (delta ~= additive). **This arm IS that iterative matrix delta**, applied per-step to the linattn fast-weight, at 13.5M-BPE-token LM scale. Not a refuted reproposal — the exact mechanism that finding said its coarse proxy could not realize.

## Why this is the high-confidence next mechanism (our record + external)

- **Our own record scoped it #1 and never built it.** `2026-07-15-emergence-engine-research-gate-...-delta-rule-fastweight-is-1` ranked *"DELTA-RULE fast-weight content-addressable store ... M += eta(v - M k)k^T, read v_hat = M q"* the cheap-first **#1** next mechanism (bio HIGH, drop-in) and cited exactly this: additive Hebbian *"SATURATES under interference; delta restores capacity and is biologically local"* (Schlag DeltaNet). It was scoped, not built.
- **The precondition it was gated on is now met.** `2026-07-13-input-magnitude-gating-...-NOT-content-selective-6seed` conditionally green-lit the content-selective / delta memory PENDING *"structured codes with informativeness structure ... AND/OR the validated GPU scale — NOT the toy one-hot regime"* (one-hot x random-W_in codes had no content-vs-filler structure to exploit). linattn's LEARNED BPE embeddings at 13.5M tokens on wikitext-103 are structured codes at real scale — the precondition.
- **Convergent across 3 independent external groups:** Gated DeltaNet (arXiv:2412.06464 — Wiki ppl 16.42 vs Mamba2 16.56 and vs a worse plain-DeltaNet, i.e. the gated delta rule is the lever), RWKV-7 (arXiv:2503.14456), DeltaNet parallelization (arXiv:2406.06484); Schlag et al. 2021 (arXiv:2102.11174) is the origin. All target the additive-linear-attention interference/unbounded-norm failure that linattn measured. <!--derived--> (external arXiv identifiers, not measurements from our artifacts)

## The measured premise it targets (confirmed against current main)

linattn's deployable spiking mouth CROSSES a fair interpolated trigram on the narrow simplewiki domain (+0.05 mean, 6/6; `2026-09-03-OPEN-FLUENCY-BREAKTHROUGH-linattn-...-6of6`) but FALLS BELOW the trigram on the BROAD wikitext-103 domain. From the byte-identical baseline artifact `research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json` (s43, argv in its `.prov.json`, git bbff50765), `margin_vs_trigram` by depth: d2 **-0.570**, d3 **-0.467**, d4-5 **-0.402**, d6-9 **-0.356**, d10-99 **-0.286**. Broad-domain fluency is the capability that retires Qwen, so this floor is the thing to lift. The failure signature (worsens exactly narrow->broad) is interference + unbounded memory-norm — the delta rule's target.

## Biology (brain-based-only)

The delta rule is error-correcting Widrow-Hoff: LOCAL and weight-transport-free (the correction `v_t - v_old` is the postsynaptic value cell's own activity minus the retrieved prediction, available at the synapse). Erase-before-write is realizable by short-term synaptic plasticity — the presynaptic key drive both reads the currently-bound value (facilitated transmission) and DEPRESSES the prior binding before the new value potentiates (presynaptic-driven short-term / heterosynaptic depression). It is framed as the spiking-portable write rule, not a host-only trick. Same fast-weight = STP anchor as linattn (Mongillo-Barak-Tsodyks 2008; Ba, Hinton et al. 2016).

## Build (additive, default-off, byte-identical-when-off)

- New nested class `DeltaNetLayer` in `_emerge_wkv_lm_derisk.py`, a structural sibling of `LinAttnLayer` (same module set: `Wq/Wk/Wv/Wr/Wo/w` + optional `Wg`), differing ONLY in the forward write rule. `beta` is a FIXED scalar (`--delta-beta`, default 1.0) so **zero extra parameters** over linattn.
- `--recurrence deltanet` choice; `self.deltanet_layers` built ONLY when selected (`... if RECUR == "deltanet" else nn.ModuleList()`), so it consumes ZERO init-RNG draws otherwise — the same construction the learnkey/hippokey arms use. New flags: `--delta-beta` (write strength) and `--delta-key-norm {l2,none}` (default `l2` = exact projection erase; `none` = the un-normalized literal formula, offered as an ablation). Reuses `--linattn-phi` and `--assoc-gate`.
- No `sim/` edit, no production edit, no default flipped.

## Verified by CPU smoke (tiny toy config; not a fluency result)

Tiny word-tokenizer config (`--d-model 32 --n-layers 2 --epochs 1 --n-sentences 500`, CPU) on wikitext.txt:
1. **Byte-identical when off:** `wkv` and `linattn` runs at seed 42 are IDENTICAL pre-edit vs post-edit — every `by_depth` NLL/margin/perm/mless value matches; the only diff is the `elapsed_s` wall-clock field. So adding the arm did not perturb any existing arm.
2. **deltanet runs end-to-end** (forward/backward/eval) — `research/findings/raw/_deltanet_arm_numpy_smoke.json`.
3. **Both anti-cheats collapse** on the deltanet run (genuine context + order use, even at toy scale): memoryless-collapse +0.060 (>0.05), permute-collapse +0.160 (>0.05). The runner's built-in per-arm GO gate already checks exactly this (`margin_vs_trigram > 0.02` AND both collapses `> 0.05`). <!--derived--> (collapse = wkv_memoryless/wkv_perm minus wkv, differences computed from the cited smoke artifact's per-depth values)

The toy scale is NOT decisive on fluency (both wkv and linattn also read negative margins there); the smoke's job is only build-correctness + genuineness of context use.

## Pre-registered GO bar for the queued wt103 direction-test

Run the deltanet arm on the SAME byte-identical wt103 config as the linattn baseline (only `--recurrence` swapped). Then:
- **DIRECTION-POSITIVE (warrants the 6-seed round):** deltanet s43 d10-99 `margin_vs_trigram` lifts CLEARLY off linattn's -0.286 floor (target Δ >= +0.05 vs the linattn baseline at the same depth) with BOTH anti-cheats collapsing (perm & mless > 0.05). Must beat plain linattn (d10-99 > -0.286).
- **PRIMARY GO (crosses the trigram, broad domain):** d10-99 `margin_vs_trigram > 0.02` AND both anti-cheats collapse, replicated 6-seed (42/43/44/100/101/102), mean > 0.02. Attribute the win to the write rule with the isolation control `--recurrence linattn --no-linattn-norm` (additive write + raw read; deltanet vs that isolates the delta write since both read raw).
- **NO REGRESSION:** a deltanet simplewiki run must stay >= linattn's +0.0505 mean (or at least positive) — queued as a second verify if the wt103 direction is positive. <!--derived--> (+0.0505 is the linattn 6-seed simplewiki mean quoted from 2026-09-03-OPEN-FLUENCY-BREAKTHROUGH-linattn-...-6of6)
- **NO-GO:** deltanet ~= the linattn floor or worse, or an anti-cheat fails to collapse (hollow). A NO-GO is an honest deliverable (it maps the write-rule's reach at this scale) and banks the method, per the law.

Single-seed s43 is only a labeled direction-test (decides whether to spend the 6-seed compute); the HEADLINE verdict requires 6-seed.

## Queued verify (controller to harvest)

Queued on the GPU (one brain-loading proc at a time; the queue is busy with critical-path work ahead of this). The controller writes the result into `research/findings/raw/` under the pending name `{OUTPUT}` = `_emerge_wkv_lm_deltanet_wt103_scale_s43` (shown as a placeholder because that output does not exist yet — it is a destination, not a cited result):

```
.venv/bin/python -m research.runners._emerge_wkv_lm_derisk --recurrence deltanet --uniform-decay \
  --batch 128 --tokenizer bpe --corpus data/corpus/wikitext103.txt --contiguous --max-len 40 \
  --max-eval-sents 4000 --epochs 4 --tok-cache --n-layers 2 --d-model 192 --n-sentences 3000000 \
  --max-train-sents 2500000 --seeds 43 --linattn-baseline-margin -0.286 \
  --json research/findings/raw/{OUTPUT}.json
```

## Files

- `research/runners/_emerge_wkv_lm_derisk.py` — `DeltaNetLayer` + `--recurrence deltanet` + `--delta-beta`/`--delta-key-norm` (additive, default-off, byte-identical-when-off).
- `research/findings/raw/_deltanet_arm_numpy_smoke.json` (+ `.prov.json`) — the CPU smoke.
- Baseline compared against: `research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json`.
