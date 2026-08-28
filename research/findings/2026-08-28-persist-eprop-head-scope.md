---
type: finding
status: live
date: 2026-08-28
mechanism: persistence + an opt-in load path for the e-prop LOCALLY-LEARNED WKV-mouth read-out head `W_hat`,
  closing residual #1 named by 2026-08-28-mouth-crutch-burndown-scope.md §4 and carried forward, unresolved, by
  2026-08-28-wkv-mouth-into-open-ended-WIRED-GO.md
verdict: COMPATIBLE. `W_hat[V,D]` (both eprop runners) is architecturally IDENTICAL to the checkpoint's native
  `head.weight` — same [1000,128] shape, same linear-map basis (`logits = W @ h + head_b` over the identical host
  feature `h = sigmoid(Wr@LN(emb))*(Wo_sp@[ap,an])`), so it is a literal drop-in for `WKVReadout.head_w`, the exact
  object `webapp/wkv_mouth_generator.py::generate()` reads through. Added `--save-w-hat` persistence to the
  batched-substrate eprop runner and an opt-in `BRAIN_WKV_MOUTH_LEARNED_HEAD` load path in the mouth generator;
  de-risked end-to-end at a deliberately tiny CPU/numpy scale (seed-waiver below): flag-off is byte-identical
  (hash-verified against the pre-edit path), flag-on loads + applies + generates, and the loader fails SAFE
  (falls back to the native head) on a missing file or a shape mismatch. The tiny-scale head itself carries NO
  quality claim (`sub_recov_ratio_mean=0.4569`, `go_count=0/1` at this toy B=2/1-epoch scale) — it exists only to
  prove the plumbing. Reproducing the real 0.8686-ratio head (already GO'd at 6-seed, `go_count=3/6` on the strict
  per-seed bar) needs the full 6-seed `--batch 48` GPU run with `--save-w-hat` added, named as the concrete next
  step.
seed-waiver: single-seed (42), CPU/numpy, deliberately-shrunk (`--batch 2 --epochs 1`, ~1600-neuron forward net)
  smoke — a MECHANICAL de-risk of the persist/load/swap plumbing only, not a quality or generalization claim. The
  quality claim (`sub_recov_ratio_mean=0.8686`, 6 seeds) is already established and is cited, not reproduced,
  from research/findings/2026-08-28-mouth-stale-coo-training-fix-fullscale-confirmation-GO.md.
lane: e-mouth-fluency / A1 (crutch-burndown, the residual #1 closure named by the 2026-08-28 wiring finding)
artifacts:
  - research/findings/raw/_persist_eprop_head_scope/eprop_batched_substrate_TINYSMOKE_seed42.json
  - research/findings/raw/_persist_eprop_head_scope/_wkv_eprop_learned_head_TINYSMOKE_seed42.npz
  - research/findings/raw/_mouth_stale_coo_training_fix/eprop_STALEFIX_6seed_frommain.json
  - research/findings/2026-08-28-mouth-crutch-burndown-scope.md
  - research/findings/2026-08-28-mouth-stale-coo-training-fix-fullscale-confirmation-GO.md
  - research/findings/2026-08-28-wkv-mouth-into-open-ended-WIRED-GO.md
  - bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed42.npz
  - research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py
  - webapp/wkv_mouth_generator.py
runner: research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py
---

# Persisting the e-prop-learned WKV-mouth read-out head — a scope + a de-risked load path

## 0. The question this scopes

The 2026-08-28 wiring finding
([`2026-08-28-wkv-mouth-into-open-ended-WIRED-GO.md`](2026-08-28-wkv-mouth-into-open-ended-WIRED-GO.md)) wired the
from-scratch WKV/SSM spiking mouth into `BRAIN_OPEN_ENDED`, but named an unresolved residual (its §"honest
residuals", item 1): the e-prop LOCALLY-LEARNED read-out head `W_hat` — confirmed at full scale to recover
`sub_recov_ratio_mean=0.8686` (min 0.8399) of the copied head's substrate recovery, 6 seeds
([`2026-08-28-mouth-stale-coo-training-fix-fullscale-confirmation-GO.md`](2026-08-28-mouth-stale-coo-training-fix-fullscale-confirmation-GO.md))
— was **never persisted to disk**. Both eprop runners (`_wkv_mouth_readout_eprop_learn_derisk.py`,
`_wkv_mouth_readout_eprop_batched_substrate_derisk.py`) train `W_hat` in memory and write only summary metrics;
`webapp/wkv_mouth_generator.py` therefore reads the checkpoint's own copied `head.weight` instead.

This rung asks, and answers: **is `W_hat` even the same kind of object as the checkpoint's native head — can it
be swapped in — or did the eprop runners train a readout for a different substrate/basis than the WKV mouth's
own generation path uses?** Then, if compatible, closes the gap: persist it, and de-risk loading it.

## 1. Compatibility — read, not assumed

**The checkpoint's native head** (`bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed42.npz`, read via
`research/runners/_wkv_fewspike_read_derisk.py::WKVReadout`):
- `head.weight` shape `(1000, 128)` float32 = `[V, D]`; `head.bias` shape `(1000,)`.
- `WKVReadout.logits(ap, an, tid)`: `state = concat(ap, an)` (`[256]`); `r_h = sigmoid(Wr @ LN(emb[tid]))`
  (`[128]`); `h = r_h * (Wo_sp @ state)` (`Wo_sp.weight` is `(128, 256)`, so `h` is `[128]=[D]`); returns
  `head_w @ h + head_b` — a plain `[V,D]@[D] + [V]` linear map.
- `webapp/wkv_mouth_generator.py::generate()` → `_free_gen()` calls exactly `ro.logits(ap, an, gen[-1])` every
  decode step, i.e. it is `ro.head_w` (nothing else) that decides the word distribution the few-spike Izhikevich
  soft-WTA then samples from.

**The eprop-learned `W_hat`** (`_wkv_mouth_readout_eprop_learn_derisk.py::_learn_hostlinear`,
`_wkv_mouth_readout_eprop_batched_substrate_derisk.py::_learn_substrate_batched`): initialized
`rng.standard_normal((V, D))` with `V, D = ro.V, ro.D` (`1000, 128` — read from the SAME checkpoint), trained
against the target label `target_t = argmax(head_w @ h_host + head_b)` (`head_w` used only to generate the
teaching label, never transported into the update — `no_transport=True` asserted), and its own recovery is scored
by `_eval_hostlinear`: `He @ W.T + hb` — the identical `W @ h + head_b` form, on the identical `h =
r_h*(Wo_sp@concat(ap,an))` feature (`_host_feat`, byte-identical code to `_free_gen`'s inputs). `head_b` is
explicitly kept COPIED (never learned) in both runners — only `W_hat` is locally-learned. The weight-cosine
diagnostic (`wcos = (W.flatten() @ head_w.flatten()) / (‖W‖‖head_w‖)`) requires the two matrices to share a shape
to even be computable, and both runners compute it directly against `ro.head_w`.

**Verdict: `W_hat` and `head_w` are the SAME object type** — both are `[V,D]=[1000,128]` linear maps over the
identical feature basis `h`, differing only in what filled the numbers. `W_hat` is a literal drop-in replacement
for `ro.head_w`; no basis translation, re-projection, or architecture bridge is needed. This is NOT the
`LearnedReadout`/`ComposedEndToEndRead` on-substrate Dale-split synapse path (a separate consumer of `W_hat` for
a different, substrate-native readout demo) — the mouth generator's `generate()` never touches that class; it
only ever calls `ro.logits`, i.e. the plain `head_w` linear map, which is exactly what got confirmed compatible.

## 2. What was built (two minimal, additive edits)

**`research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py`** — added `--save-w-hat <path>`
(`{seed}`-templated). When set, after `W_main`/`ratio` are computed for a seed, `np.savez`s `W_hat` (float32),
`head_b` (float32, the copied residual, for provenance/completeness — NOT meant to override the checkpoint's own
`head_b`), `seed`, `V`, `D`, `sub_recov_ratio`, `sub_learned_recov`, `sub_copied_recov`, `integrated_go`, and
`source_runner`. Not gated on the run's own `go` — a sub-GO head is still useful for de-risking the load path, and
the consumer sees the recorded ratio/verdict fields and can decide. No other line of the runner changed.

**`webapp/wkv_mouth_generator.py`** — a new opt-in flag `BRAIN_WKV_MOUTH_LEARNED_HEAD` (default OFF, same
truthy-string convention as the existing `BRAIN_OPEN_ENDED_WKV_MOUTH`), and `BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH`
(templated, defaults to `research/findings/raw/_wkv_eprop_learned_head_seed{seed}.npz` — the path a FUTURE full
6-seed `--save-w-hat` run would write to; nothing currently occupies that default path, see §4). `_apply_learned_
head(ro, seed)` loads the npz, checks `W_hat.shape == ro.head_w.shape`, and if it matches, replaces `ro.head_w`
in place (`head_b` untouched). `_get_readout`'s cache key became `(seed, learned_head_enabled())` so toggling the
flag never returns a stale cached object. On any failure (file missing, load error, shape mismatch) the loader
returns a reason string and the NATIVE head is left in place — it never raises, so this opt-in path cannot break
the existing default-off caller. `learned_head_status(seed)` exposes the last load's provenance for diagnostics.
Both `_get_readout` call sites' existing 3-tuple unpacking (`in_vocab_scope`, `generate`) are untouched.

## 3. De-risk — verified in the data, three states

Ran the batched-substrate eprop runner at a deliberately tiny CPU/numpy scale (`--batch 2 --epochs 1
--n-train-pos 16 --sub-hid-pop 2 --pop 1 --hid-pop 1`, seed 42; `~1592`–`8224`-neuron nets across the run's several
bridges) with `--save-w-hat`, DETACHED (`run_in_background`), completed in 12s
(`research/findings/raw/_persist_eprop_head_scope/eprop_batched_substrate_TINYSMOKE_seed42.json`):
`seed_hash_check.seeded=true` (build-twice-hash, the CLAUDE.md seed trap), `verify_first_all_ok=true`,
`forward_is_substrate_all=true` (`host_matmul_on_forward_max=0`), `sub_recov_ratio_mean=0.4569`,
`sub_learned_recov_mean=0.4206` vs `sub_copied_recov_mean=0.9205`, `go_count=0/1` — an honest, expected non-GO at
this toy scale (8 gradient steps total); the point was the saved artifact, not the quality.
`research/findings/raw/_persist_eprop_head_scope/_wkv_eprop_learned_head_TINYSMOKE_seed42.npz` (518KB) confirmed
on disk with `W_hat.shape=(1000,128)` float32, `head_b.shape=(1000,)`, `V=1000`, `D=128`.

Then, with `webapp/wkv_mouth_generator.py` imported directly (CPU/numpy, no server), three states measured by
SHA1 hash of `ro.head_w` and by actually calling `generate()`:

| state | `BRAIN_WKV_MOUTH_LEARNED_HEAD` | head_w hash | applied | generation |
|---|---|---|---|---|
| **OFF (default)** | unset | `4d5b0690d48feb44` (native) | n/a | `"once upon a time there was a little boy named tim saw a big red ball..."` — IDENTICAL prefix to the pre-edit-verified run |
| **ON, file present, shape OK** | `1` | `9c98196cd95bc30b` (learned, DIFFERENT) | `true` | `"once upon a time the big box box the big box and a big box..."` — well-formed in-vocab words, degenerate/repetitive (honest: an 8-step toy head, not the real one) |
| **ON, file missing** | `1` | `4d5b0690d48feb44` (native, fail-safe) | `false`, `reason=file_missing` | native-quality output unaffected |
| **ON, shape mismatch** (a `(10,5)` probe file) | `1` | `4d5b0690d48feb44` (native, fail-safe) | `false`, `reason=shape_mismatch:(10, 5)_vs_(1000, 128)` | native-quality output unaffected |

The OFF-path hash and generated-text prefix are byte-identical to the values measured immediately BEFORE this
change was made (same command, same seed, same prompt, run before either file was edited) — the flag genuinely
adds a new code path rather than perturbing the existing one. The learned-head hash differs from native, proving
the swap actually executes (not a silent no-op), and the two fail-safe cases confirm the loader cannot crash or
silently corrupt generation when the artifact is absent or malformed — the state a fresh checkout is in today,
since no real (non-smoke) head has been trained yet.

`_persist_eprop_head_scope/_wkv_eprop_learned_head_TINYSMOKE_seed42.npz` was deliberately saved OUTSIDE the
default consumption path (`_wkv_eprop_learned_head_seed{seed}.npz`) and under a `TINYSMOKE`-labelled name — so a
naive flip of `BRAIN_WKV_MOUTH_LEARNED_HEAD=1` today finds no file at the default path and fails safe to the
native head, rather than silently picking up this toy-quality artifact. The `ON, file present` row above was
measured via an explicit `BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH` override, not the default.

## 4. The concrete next step

Run the FULL `--batch 48 --seeds 42,43,44,100,101,102` batched-substrate eprop run (GPU-bound per that runner's
own docstring) with `--save-w-hat research/findings/raw/_wkv_eprop_learned_head_seed{seed}.npz` added — this now
persists the real `sub_recov_ratio_mean=0.8686` head at the default path `webapp/wkv_mouth_generator.py` already
looks for. Then: (a) flip `BRAIN_WKV_MOUTH_LEARNED_HEAD=1` for a qualitative A/B against the native head on
in-vocab prompts (self-NLL, coherence), through the SAME `verify_go` skeptic pass this repo requires before any
positive verdict lands; (b) decide, from that A/B, whether the learned head should become the wired path's
DEFAULT source (still behind `BRAIN_OPEN_ENDED_WKV_MOUTH`, itself default-OFF) or stay a documented opt-in given
the marginal `go_count=3/6` per-seed bar already on record. Neither this rung nor the prior wiring rung makes that
call — it is a genuine A/B on generation quality, not a plumbing question, and the plumbing (this rung) is what
was blocking it from being askable at all.
