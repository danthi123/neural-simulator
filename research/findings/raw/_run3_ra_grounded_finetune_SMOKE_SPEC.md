# run3 (83M WKV) RA grounded-render fine-tune — runner VERIFIED + launch spec (2026-07-23)

**Runner EXISTS** (built earlier this workflow, untracked): `research/runners/_run3_ra_grounded_finetune_derisk.py` (676 lines).
Ports the EMERGE-57 RA/format continuation-fine-tune to run3's arch, all **reuse-by-import**, **NO sim/ edit**:
- RA transitive-SVO QA frames from `_fluidconv_phase2_ra_finetune` (`_make_example`, `SEP`).
- EMERGE ability/exception/abstain frames + the `emerge_v3` inflection fix from `_emerge57_ra_refinetune_emerge_frames_derisk`.
- run3 `ChunkedWKV` build/load + frozen HF-BPE-16k tokenizer via `lm_train_lib` (`TrainConfig`/`build_model`/`_load_tokenizer`).
- Base ckpt: `bridges/lmtrain/run3/ckpt/best.pt` (83.2M, best_val_nll 3.987). Fine-tuned model → NEW path (run3 untouched).
- **Anti-forgetting**: per-batch `MixedSampler` interleaves raw run3 FineWeb-Edu windows (`tokens_train.npy`) with the grounded corpus.

## CPU smoke (tiny fresh 2.09M WKV, `--smoke --tiny-model`, ~10s wall) — pipeline + controls LIVE
- corpus build (900 frames → 26291 tok) → encode → MixedSampler (drew 11 QA + 5 FineWeb seq) → 8 train steps → pre/post ppl → render → moat → **VERDICT printed**.
- **Every anti-cheat control WIRED + INVOKED** (verified in the emitted JSON):
  - anti-forget: `fineweb_anti_forget_enabled=True`, `n_fineweb_seq_sampled=5`, FineWeb ppl pre/post measured (19482→19472).
  - learn: EMERGE-frame + RA-frame ppl pre/post measured.
  - **MOAT** (renderer-never-invoked-on-abstain, the load-bearing one): `n_abstain=2`, `moat_render_calls_on_abstains=0`, `n_model_invoked_on_abstain=0`.
  - confab guard (`no_other_member`) + double-inflection guard (`n_double_inflection=0`) live over the 6 answer probes.
- **fidelity 0.00 is EXPECTED garbage** (untrained tiny model rambles) — NOT a GO/negative claim. Smoke proves *wiring*, not the result.
- Safety: `--tiny-model` writes `run3_ra_grounded_ft.pt.tiny` — the real 332MB `run3_ra_grounded_ft.pt` (prior seed-42 derisk) is INTACT.

## Additive runner-only fixes made (NO sim/ edit)
1. Honest done-print: was hardcoding `FT_CKPT`; now prints the actual `ckpt_path` (the `.tiny` save was mislabeled).
2. Seed-suffixed derisk output paths (ckpt + JSON) so a 6-seed run never clobbers — seed 42 keeps historical names, 43/44/100/101/102 get `_seed{N}`; added `--out` override.

## Prior state (existing artifacts)
- `bridges/lmtrain/run3_ra_grounded_ft/run3_ra_grounded_ft.pt` (332MB) — a prior seed-42 fine-tune. A `--render-only` eval scored **render fidelity 0.67 (4/6), MOAT HELD** = BOUNDARY ("still rambles/confabulates on 2/6" — lever: more steps/n_emerge). Its ppl (forget/learn gates) were NOT recorded — the full `--derisk` (with ppl) still needs to run.

## LAUNCH SPEC — lane=local-3090, bounded GPU (during the current run3 PAUSE; ~19GB free)
```fish
for s in 42 43 44 100 101 102
  .venv/bin/python -m research.runners._run3_ra_grounded_finetune_derisk --derisk --device cuda --amp 1 \
    --steps 1200 --n-ra 12000 --n-emerge 14000 --mix-ratio 0.5 --batch-size 32 --seq-len 256 --lr 5e-5 \
    --seed $s --out research/findings/raw/_run3_ra_grounded_finetune_seed$s.json
end
```
- **ETA** ~6-8 min/seed (1200 steps @ ~0.21s/step uncompiled + ~1 min pre/post ppl); ~40-50 min for 6 seeds serially.
- **Per-seed GO** (the runner's OWN printed `VERDICT: GO`, `_summarize` line 504): `render_fidelity ≥ 0.85 AND n_double_inflection == 0 AND moat_render_calls_on_abstains == 0 AND n_model_invoked_on_abstain == 0 AND FineWeb forget_ratio ≤ 1.5 AND emerge_frame_learn_ratio < 1.0`. Aggregate: `VERDICT: GO` on ≥ 5/6 seeds (6-seed rule).
- run3 training is PAUSED (sentinel `bridges/lmtrain/run3/PAUSE`); a paused lm_train process idles at 3.4GB — the derisk co-fits.
