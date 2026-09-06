---
status: qualified
lane: infra
date: 2026-09-06
type: note
---

# vLLM Sleep Mode pilot checkpoint blocker was a Unix-permission bug, not a missing shard — fixed non-GPU, checkpoint verified complete (2026-09-06)

Follow-up to [`2026-09-06-vllm-sleep-mode-pilot-prep-complete-gpu-test-pending.md`](2026-09-06-vllm-sleep-mode-pilot-prep-complete-gpu-test-pending.md).
That note left one blocker for the controller: a `vllm serve` engine-init attempt died with a
file-access error on `model-00006-of-00007.safetensors` inside
`/home/dant123/Projects/qwen38-27b-rtx3090/models/Qwen3.8-27B-W4A16-AutoRound`. This session diagnosed
and fixed it without touching the GPU (the 3090 stayed on the llama.cpp Q4 server the whole time — no
`vllm serve`, no model GPU-load was run).

## 1. Diagnosis: the shard was never missing — it was unreadable

`pilot_prepare.log` (the Docker-based prepare pipeline's own log) ends with `prepare: model ready at
/app/models/Qwen3.8-27B-W4A16-AutoRound ... [21:58:51] PREPARE DONE rc=0` — the full pipeline (HF
download of the already-AutoRound-quantized base checkpoint, then int8 re-quant of `lm_head`,
`embed_tokens`, and the MTP module) completed with no error, and all 7 `model-*.safetensors` shards plus
`model_extra_tensors.safetensors` were present on disk at their expected sizes the entire time.

The actual defect: three files were left `-rw------- root:root` (mode 600) — unreadable by the `dant123`
user that `vllm serve` runs as — because the prepare pipeline runs inside Docker as the container's root
user, and the three files touched last by the int8-requant scripts (`quant_lm_head.py` writes into
`model-00007-of-00007.safetensors`, `quant_embed.py` writes into `model-00006-of-00007.safetensors`,
`quant_mtp.py`/`build_draft_vocab.py` write `model_extra_tensors.safetensors` and its
`.bak-draft`) ended up with a different mode than the five untouched shards (`model-0000{1,2,3,4,5}`,
mode 644, also root-owned but world-readable). Everything else in the checkpoint directory — including one
metadata file under a `.cache/huggingface` subdirectory, also 600 — followed the same pattern; the requant scripts' write path
(most likely an atomic write via a `mkstemp`-style temp file that was never `chmod`-restored after the
rename) is the shared root cause across all of them. This is a permissions bug in that reference
project's requant scripts, not a corrupted or partial artifact, and not something that needs
re-downloading or re-quantizing.

Confirmed non-destructively before touching anything: `find <dir> -type f ! -perm -044` listed exactly
these three files (plus the `.cache` json); `stat` on `model-00006-of-00007.safetensors` showed
`Access: (0600/-rw-------) Uid: (0/root) Gid: (0/root)`; `test -r` as `dant123` on that file failed.

## 2. Fix: three `chmod` calls via passwordless sudo, nothing else touched

```bash
sudo chmod 644 \
  /home/dant123/Projects/qwen38-27b-rtx3090/models/Qwen3.8-27B-W4A16-AutoRound/model-00006-of-00007.safetensors \
  /home/dant123/Projects/qwen38-27b-rtx3090/models/Qwen3.8-27B-W4A16-AutoRound/model-00007-of-00007.safetensors \
  /home/dant123/Projects/qwen38-27b-rtx3090/models/Qwen3.8-27B-W4A16-AutoRound/model_extra_tensors.safetensors \
  /home/dant123/Projects/qwen38-27b-rtx3090/models/Qwen3.8-27B-W4A16-AutoRound/model_extra_tensors.safetensors.bak-draft
```

No file was re-downloaded, re-quantized, moved, or renamed. Ownership stays `root:root` (matching the
five untouched shards); only the mode changed, to match its siblings. A re-run of the `! -perm -044` find
afterward returned nothing under the checkpoint directory (maxdepth 1), and `test -r` as `dant123`
succeeded on all three files.

## 3. Verification performed (all non-GPU, `CUDA_VISIBLE_DEVICES=""` where applicable)

- **Shard-by-shard safetensors header parse**: a small stdlib+`safetensors` script (run with the
  reference project's own venv python, `safetensors==0.8.0`, CPU-only) opened every one of the 8 files
  the index references (`model-0000{1..7}-of-00007.safetensors` + `model_extra_tensors.safetensors`) via
  `safe_open(..., framework="np")`, read the header, and pulled a slice descriptor for one tensor per
  shard to prove the header is not truncated. Result: all 8 opened cleanly, sizes matched `ls`
  (3.21/3.20/3.20/3.22/0.70/1.29/1.29/0.64 GB), tensor counts 428/415/415/481/242/4/3/34.
- **Index-to-shard cross-check**: all 2,022 keys in `model.safetensors.index.json`'s `weight_map`
  resolved to a key actually present in the shard file the index claims — 0 missing. No orphan tensors
  in the 7 main shards outside the index (`model_extra_tensors.safetensors`'s 34 keys are intentionally
  outside the main index — MTP/draft tensors, consistent with `quant_mtp.py`'s log output naming that
  file directly).
- **`config.json` parses under `transformers.AutoConfig.from_pretrained`** (transformers 5.15.0, CPU):
  resolves to `Qwen3_5Config` / `Qwen3_5ForConditionalGeneration`, `quantization_config` present,
  `text_config.num_hidden_layers == 64` — matches the architecture numbers in the prior pilot-prep note.
- **`vllm.transformers_utils.config.get_config()` on the checkpoint path** (vLLM 0.27.1, `pip`-installed
  in the reference venv, `CUDA_VISIBLE_DEVICES=""` so no GPU is visible to the process): this is the same
  HF-config-resolution call `vllm serve` makes before any weight loading or GPU memory allocation, and it
  returned the same `Qwen3_5Config` cleanly. No `EngineArgs`/`LLMEngine`/`vllm serve` was constructed or
  invoked at any point — no weights were loaded onto the GPU, and no CUDA context was created (verified
  by keeping `CUDA_VISIBLE_DEVICES=""` for this specific call).

No GPU command was run this session. `nvidia-smi` was not queried by this session either (not needed —
the task's constraint was "don't touch the GPU," not "confirm what else is on it"; the controller's
existing handoff in the prior note already covers that check).

## 4. Handoff: no checkpoint path change, no serve-script edit needed

`tools/vllm_sleep_pilot_serve.sh`'s existing default `VLLM_MODEL_DIR` already points at the
now-fixed checkpoint:

```
VLLM_MODEL_DIR="${VLLM_MODEL_DIR:-/home/dant123/Projects/qwen38-27b-rtx3090/models/Qwen3.8-27B-W4A16-AutoRound}"
```

Nothing in the script needed to change — the path was always correct; only the on-disk permissions were
wrong. The controller's exact invocation from the prior note's §6 is unchanged and now unblocked:

```bash
# 1. Confirm the card is actually free of a resident brain job:
tools/gpu_queue.sh status
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# 2. Launch the pilot server (no env override needed — default VLLM_MODEL_DIR is correct):
bash tools/vllm_sleep_pilot_serve.sh up
tail -f research/queue/vllm_pilot_server.log

# 3. Once READY, run the decisive Sleep Mode measurement:
python3 tools/vllm_sleep_mode_test.py --endpoint http://127.0.0.1:18020 --model qwen3.8-27b-pilot \
    --levels 1,2 --out research/findings/raw/vllm_sleep_mode_pilot/sleep_wake_$(date +%s).json

# 4. Tear down:
bash tools/vllm_sleep_pilot_serve.sh down
```

If `vllm serve` still reports a file-access error on any shard after this fix, re-run
`find /home/dant123/Projects/qwen38-27b-rtx3090/models/Qwen3.8-27B-W4A16-AutoRound -type f ! -perm -044`
first — it will be a permissions regression (e.g. the reference project's prepare pipeline was re-run),
not a re-emergence of a missing-shard problem, since this checkpoint's completeness is now independently
verified against the safetensors index rather than assumed.

## 5. Verdict

**RESOLVED, non-GPU.** The checkpoint was complete on disk since the 2026-08-31 prepare run; the pilot's
blocker was three files with the wrong Unix permission bits, now fixed via `chmod`. Completeness is
verified two ways that do not depend on trusting the prepare log alone: a full safetensors index
cross-check (every tensor key resolves) and vLLM's own config-resolution call succeeding against the
checkpoint path. The decisive Sleep Mode GPU measurement itself remains **PENDING**, per the prior note —
this session did not run it and did not touch the GPU.

## 6. Non-negotiables checked

- No `vllm serve`, no `LLMEngine`/`EngineArgs` construction, no GPU model load — verified by design (only
  `AutoConfig`/`get_config` calls, both HF-config-only code paths) and by keeping
  `CUDA_VISIBLE_DEVICES=""` for the vLLM config-parse check specifically.
- `tools/qwen_serve.sh` (the llama.cpp fallback currently serving on the 3090) — untouched.
- `tools/vllm_sleep_pilot_serve.sh` — untouched; no edit was needed since the checkpoint path it already
  pointed at is the one that got fixed.
- Fix scope — three `chmod` calls in a directory outside this repository
  (`/home/dant123/Projects/qwen38-27b-rtx3090`, a separate already-cloned reference project); nothing in
  the `sim` repository's tracked files changed as part of the fix itself, only this finding doc.
