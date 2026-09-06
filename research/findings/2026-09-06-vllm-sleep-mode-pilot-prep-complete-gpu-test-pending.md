---
status: qualified
lane: infra
date: 2026-09-06
type: note
---

# vLLM Sleep Mode pilot for Qwen3.8-27B on one RTX 3090 — non-GPU prep complete, decisive GPU test PENDING (2026-09-06)

Owner-approved pilot (not a cutover) of vLLM Sleep Mode (`/sleep`+`/wake_up`, in-place VRAM release) as a
possible faster alternative to `tools/qwen_serve.sh`'s kill+cold-reload dance for sharing the single 3090
between the local model and brain experiments. `tools/qwen_serve.sh` (llama.cpp, Q4_K_M GGUF, port 8033) is
UNTOUCHED and remains the proven fallback.

**Status of this note: every non-GPU claim below was verified this session (install log, `vllm serve
--help=all`, `config.json`, WebSearch). The Sleep Mode measurement itself — does it actually free VRAM cleanly
and wake in single-digit seconds on this hybrid checkpoint — has NOT been run, because the 3090 is currently
GPU-resident with a brain-experiment job (`_rank2_integrated_loop_webapp_thread_derisk`, confirmed via
`nvidia-smi --query-compute-apps`) and this pilot's own hard constraint is to never load a second model onto a
card a brain job owns. Section 6 hands the controller the exact commands to run once the card is confirmed
free.**

## 1. A reference project already existed for this exact hardware+model combo — use it, don't rebuild it

Before writing anything, a repo search turned up `/home/dant123/Projects/qwen38-27b-rtx3090`, already cloned
from `https://github.com/syv-ai/qwen38-27b-rtx3090` — an OSS project purpose-built for "Qwen3.8-27B on a single
RTX 3090 with vLLM," complete with 22 vLLM patches, a requantization pipeline, `verify.sh` as a quality gate,
and extremely detailed `docs/gotchas.md` (39+ numbered hard-won lessons). It contributed three load-bearing
things to this pilot without any of us re-deriving them the hard way:

- **The exact known-good pinned stack**: `docker/requirements.txt` pins `vllm==0.27.1` (torch 2.13.0/cu130,
  triton 3.7.1, transformers 5.15.0) as "the stack every number in the READMEs was measured on" — i.e. a
  version already confirmed to load and serve this specific model on this specific GPU class, not a guess.
- **An already-downloaded, already-requantized W4A16 checkpoint** at
  `models/Qwen3.8-27B-W4A16-AutoRound/` (compressed-tensors `pack-quantized` format, int4 weights/group_size
  128, int8 lm_head+embed) — vLLM's native quant format, so this pilot needed ZERO new HF downloads for the
  model weights. This is the AWQ-style quant the task asked us to identify/download; it was already local.
- **A measured `--language-model-only` weight-loading number** (`docs/gotchas.md` #9): 14.26 GiB resident with
  the vision tower dropped, vs 15.13 GiB with it loaded (the tower itself is ~0.86 GiB BF16, not the 2.7 GB an
  earlier note in that repo used to claim). `--language-model-only` is confirmed a STOCK vLLM 0.27.1 CLI flag
  (present in `vllm serve --help=all`, not something any of that repo's patches add), so this pilot uses it
  directly with no patching required.

This pilot does **not** apply that repo's 22 patches (they add speculative decoding / int8-activation /
hybrid-KV-pool-sizing optimizations, none of which Sleep Mode needs) and does **not** use its Docker/systemd
wrapping. It borrows the checkpoint, the pinned version numbers, and the gotchas as a risk map.

## 2. vLLM version + hybrid-arch + Sleep Mode support — verified this session

- **Installed and import-verified**: `vllm==0.27.1`, `torch==2.13.0+cu130`, `triton==3.7.1`,
  `transformers==5.15.0`, into a **fresh venv** at `/home/dant123/Projects/qwen38-27b-rtx3090/venv` (Python
  3.14.7, via `uv venv` + `uv pip install`) — separate from the sim's own `.venv`, per the task's constraint.
  `venv/bin/python -c "import vllm, torch"` succeeds; `torch.cuda.is_available()` reports `True` (driver
  detection only, no VRAM allocated by this check).
- **The model's architecture is `Qwen3_5ForConditionalGeneration` / `model_type: qwen3_5`** (read from the
  checkpoint's `config.json`, under that reference project's `models` dir, `Qwen3.8-27B-W4A16-AutoRound`
  subdirectory), which vLLM 0.27.1 loads without `--trust-remote-code`
  (native support; the model is "already the one in use" per the prior local-agent-stack-review note, and this
  vLLM version postdates the ~3-week-old hybrid-arch support window that note flagged as a risk).
- **Sleep Mode is present and stock**: `vllm serve --help=all` shows `--enable-sleep-mode` ("Enable sleep mode
  for the engine (only cuda and hip platforms are supported)"). The official docs
  (`docs.vllm.ai/en/latest/features/sleep_mode/`, fetched this session) additionally require the env var
  `VLLM_SERVER_DEV_MODE=1` for the `/sleep`, `/wake_up`, `/is_sleeping`, `/collective_rpc` endpoints to be
  registered at all — this is a documented "internal/dev" gate, not a bug, and both the CLI flag and the env
  var are baked into `tools/vllm_sleep_pilot_serve.sh`'s launch line.
- **Exact API** (from the same fetch): `POST /sleep?level=1|2`, `POST /wake_up[?tags=weights|kv_cache]`,
  `GET /is_sleeping`, and for level 2 specifically `POST /collective_rpc {"method":"reload_weights"}`
  afterwards. **Level 1** offloads weights to CPU RAM (backed up) and discards the KV cache — wake restores
  weights from the CPU copy, which is the fast path. **Level 2** discards weights AND KV cache with nothing
  backed up — wake needs the explicit `reload_weights` RPC, which re-reads the checkpoint and should be
  markedly slower. Neither vLLM's own docs nor the WebSearch round surfaced any documented interaction between
  Sleep Mode and the Mamba/Gated-DeltaNet recurrent-state cache (which vLLM's own docs describe elsewhere as "a
  separate, special-cased" pool from the ordinary KV cache) — **this is an open risk, not a confirmed problem**,
  and it is exactly why `tools/vllm_sleep_mode_test.py` checks the REAL `nvidia-smi` VRAM delta after `/sleep`
  rather than trusting the endpoint's 200 response (see §4).

## 3. A confirmed, load-bearing vLLM 0.27.1 bug on this exact GPU class — mitigation is mandatory, not optional

WebSearch surfaced `vllm-project/vllm#52682` (open, no merged fix as of this pilot): **CUDA-graph capture for
the Gated-DeltaNet hybrid-attention kernel hangs indefinitely at startup on Ampere GPUs** (reported on 4x RTX
A5000, compute capability 8.6 — **the same compute capability as our RTX 3090**), confirmed against vLLM
0.27.1. No crash, no error — the process just parks in graph capture forever. The documented workaround is
`--enforce-eager` (disables CUDA-graph capture and torch.compile; the reporter measured ~10s startup with it
vs. an indefinite hang without). `tools/vllm_sleep_pilot_serve.sh` therefore **always** passes
`--enforce-eager` and calls this out prominently in its own header comment — removing it to chase throughput
before that upstream issue closes risks wedging the GPU with a server that never becomes ready and never
releases the VRAM it already grabbed for weight loading.

Enforce-eager is compatible with Sleep Mode in principle (Sleep Mode operates on the weight/KV memory pools,
which exist independent of whether CUDA graphs are captured over them) — but this combination specifically has
not been run, so it is a stated assumption, not a measurement.

## 4. VRAM-fit analysis for the ~19-20GB budget (quantitative, from `config.json`)

Card: RTX 3090, 24576 MiB total (`nvidia-smi`, this session). Current non-brain VRAM floor observed this
session: desktop compositor + Vesktop + Claude Desktop GPU processes summed to ~1037 MiB of a 3299 MiB total
that also included a resident brain job's 596 MiB — i.e. desktop/monitor overhead alone was in the ~1.7-2.7 GiB
range at the moment of measurement, consistent with (a bit better than) the task's ~3.5-5GB planning figure.

**Architecture, from that checkpoint's `config.json` (`text_config` section)**: 64 hidden layers,
`full_attention_interval: 4` → 16 full-attention layers / 48 Gated-DeltaNet linear-attention layers (matches
the "3 linear : 1 full-attn, only 16/64 layers grow KV" framing in the prior stack-review note, now confirmed
against the actual checkpoint rather than assumed). Full-attention layers: `num_key_value_heads=4`,
`head_dim=256`. Linear-attention layers: `linear_num_value_heads=48`, `linear_key_head_dim=linear_value_head_dim=128`.

**Full-attention KV cost (the term that scales with context length)**: 2 (K+V) × 4 kv_heads × 256 head_dim × 16
layers = 32,768 bytes/token at 1-byte (fp8) KV, or 65,536 bytes/token at 2-byte (bf16/auto) KV.

**Gated-DeltaNet recurrent-state cost (FIXED per sequence, does not grow with context)**: 48 value_heads × 128
× 128 (key_dim × value_dim outer-product state) × 2 bytes(bf16) × 48 layers ≈ 72 MiB/sequence, plus a small
conv-state term (≈3 MiB) — on the order of 75 MiB total per resident sequence, negligible against a
multi-gigabyte KV budget at `max-num-seqs=1`. This is the numeric confirmation of "hybrid ≈4x cheaper than a
standard 32B": almost the entire per-token cost is concentrated in a quarter of the layers.

**The budget** (24576 MiB total, `--gpu-memory-utilization 0.80` → ~19,661 MiB ceiling, matching the task's
~19-20GB framing):

| component | MiB | source |
|---|---|---|
| weights, text-only (`--language-model-only`) | 14,602 | measured by the reference repo (docs/gotchas.md #9), 14.26 GiB |
| non-weight overhead | 420 | same source, ~0.41 GiB |
| activation-peak safety margin (eager mode) | 1,536 | estimate — see caveat below |
| **remaining for KV pool** | **~3,103** | 19,661 − 14,602 − 420 − 1,536 |
| `tools/vllm_sleep_pilot_serve.sh` default `--kv-cache-memory-bytes` | 2,867 (2.8 GiB) | deliberately below the ~3,103 ceiling for extra margin |

At fp8 KV (32,768 B/token): 2.8 GiB ÷ 32,768 ≈ **91,800 tokens** — inside the 64-100k target, with headroom.
At the bf16/auto fallback (65,536 B/token), the SAME byte budget gives ≈ **45,900 tokens** — below the 64k
floor. **This is the pilot's single biggest open unknown**: whether stock (unpatched) vLLM 0.27.1's fp8 KV-cache
quantization path actually supports this hybrid checkpoint's full-attention layers cleanly. `--kv-cache-dtype
fp8` is a stock, listed choice (`vllm serve --help=all` lists `fp8`, `fp8_e4m3`, `fp8_e5m2`,
`int8_per_token_head`, among others) but has not been exercised against this specific architecture in this
pilot. **Plan B if it errors or silently under-performs**: drop to `--kv-cache-dtype auto` (bf16) and
`--max-model-len 49152` (48k) to stay inside the same byte budget — both are one env var each
(`VLLM_KV_DTYPE`, `VLLM_MAX_MODEL_LEN`) on `tools/vllm_sleep_pilot_serve.sh`.

Caveat on the 1,536 MiB activation-peak line: this is an estimate, not a measurement for THIS config. The
reference repo's own gotcha #16 measured 1.09-1.96 GiB activation-peak swings between cold-cache starts of its
BATCH-mode config (higher concurrency, larger `--max-num-batched-tokens`); this pilot's single-user
(`--max-num-seqs 1`) eager-mode config should profile lower, but that has not been confirmed on this GPU. Also
note `--kv-cache-memory-bytes` (confirmed present and stock via `--help=all`) is used instead of tuning
`--gpu-memory-utilization` for the KV pool specifically, precisely because that repo's gotcha #16 documents
`gpu_memory_utilization`-derived KV sizing as noisy run-to-run — pinning bytes removes that variance from the
number this analysis most depends on.

## 5. What was built (both non-GPU, both verified without touching the GPU's compute allocator)

- **`tools/vllm_sleep_pilot_serve.sh`** — `up`/`down`/`restart`/`status`/`sleep [1|2]`/`wake`, analogous to
  `tools/qwen_serve.sh`'s interface. Bakes in `--enforce-eager`, `--language-model-only`,
  `--enable-sleep-mode` + `VLLM_SERVER_DEV_MODE=1`, `--kv-cache-dtype fp8`, `--kv-cache-memory-bytes` (2.8 GiB
  default), `--max-model-len 81920`, `--max-num-seqs 1`, all env-var-overridable. Uses the separate venv and
  the already-local checkpoint — no new downloads. **`up` REFUSES to launch while any `research.runners` /
  `webapp` python process is GPU-resident** (same detection pattern as `tools/gpu_queue.sh`'s
  `gpu_resident_brain_pids()`, independently re-implemented so this script has no runtime dependency on that
  one's internals) — verified working this session: with the current brain job resident, `up` printed
  `REFUSING to launch — GPU-resident brain process(es) found: 678811` and exited 1, exactly as intended.
  Override only via `VLLM_PILOT_FORCE=1`, documented as hand-only.
- **`tools/vllm_sleep_mode_test.py`** — stdlib-only (no dependency on the vllm venv), drives a running pilot
  server through: baseline inference → record real `nvidia-smi` VRAM → `POST /sleep?level=N` → poll
  `/is_sleeping` to confirm the transition → re-read VRAM and compute the freed fraction (flags `FAIL` if
  under 50% of the awake total was freed, catching a `/sleep` that returns 200 without the allocator actually
  releasing anything — the Mamba/GDN-state risk from §2) → `POST /wake_up` (+ `reload_weights` RPC for level
  2, timed separately since that is the step that actually re-reads the checkpoint) → confirm inference resumes
  and is coherent. Emits one JSON report per level with an explicit per-level `PASS`/`FAIL` and a
  `hits_3_6s_target` boolean (the task's own target figure) separate from the wider pilot pass bar (15s) used
  just to distinguish "worked slowly" from "broken." **Smoke-tested end-to-end this session against a throwaway
  mock HTTP server** (`/health`, `/is_sleeping`, `/sleep`, `/wake_up`, `/collective_rpc`,
  `/v1/chat/completions`) standing in for vLLM — confirmed the full control flow (both levels, the level-2
  reload-weights branch, JSON report writing, exit codes 0/1/2) executes correctly; the mock does not touch
  the GPU, so it correctly reported `FAIL` on the VRAM-freed check (nothing was actually freed) — this
  validates the harness's LOGIC, not the actual vLLM behavior, which is exactly the boundary this pilot could
  verify without the GPU. (The mock-server report itself is a scratch artifact from an ad hoc HTTP-client
  test, not a sim experiment — deliberately not committed as a `research/findings/raw/` artifact, since
  neither device/backend nor run-provenance apply to it and stamping fake ones would be worse than omitting
  it. The behavior it demonstrated is described in prose above; reproduce it with the throwaway mock server
  pattern described in `tools/vllm_sleep_mode_test.py`'s own module docstring if independent verification of
  the harness's control flow is wanted before the first real GPU run.)

## 6. Exact commands for the controller to run once the GPU is confirmed free

```bash
# 1. Confirm the card is actually free of a resident brain job (both checks — belt and suspenders):
tools/gpu_queue.sh status
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# 2. Launch the pilot server (refuses on its own if a brain job is resident — see §5 — but check anyway):
bash tools/vllm_sleep_pilot_serve.sh up
#    First start includes torch.compile/Triton JIT warmup even with --enforce-eager (kernel autotune for the
#    GDN layers) — budget several minutes, not seconds, for the FIRST `up`. Watch:
tail -f research/queue/vllm_pilot_server.log

# 3. Once READY (the `up` command itself blocks and prints this), run the decisive measurement:
python3 tools/vllm_sleep_mode_test.py --endpoint http://127.0.0.1:18020 --model qwen3.8-27b-pilot \
    --levels 1,2 --out research/findings/raw/vllm_sleep_mode_pilot/sleep_wake_$(date +%s).json

# 4. Tear down (frees ALL VRAM, not just what Sleep Mode itself would release):
bash tools/vllm_sleep_pilot_serve.sh down

# If fp8 KV errors at startup (§4 Plan B), retry step 2 with:
VLLM_KV_DTYPE=auto VLLM_MAX_MODEL_LEN=49152 bash tools/vllm_sleep_pilot_serve.sh up
```

Re-invoke this pilot (or hand the JSON report back) once step 3 has run — the verdict section below updates
from "PENDING" to the measured PASS/FAIL/partial result at that point, not before.

## 7. Verdict

**PENDING GPU TEST.** Every claim in §§1-5 is backed by an artifact this session actually produced or read
(the install log; `vllm serve --help=all`; the checkpoint's `config.json`; the two fetched
vLLM doc/issue pages; the `up`-refusal smoke test; the mock-server smoke test of the harness). **No claim is
made here about whether Sleep Mode actually frees VRAM cleanly or wakes in single-digit seconds on this
hybrid checkpoint** — that requires the GPU, which a resident brain job made unavailable for the duration of
this pilot, per the task's own hard constraint. Section 6 is the complete, ready-to-run handoff.

## 8. Non-negotiables checked

- `tools/qwen_serve.sh` — untouched (verified: not read-modified this session beyond the initial read for
  context; `git status` shows it clean).
- Separate env — `vllm==0.27.1` + its pinned stack installed into
  `/home/dant123/Projects/qwen38-27b-rtx3090/venv`, never into the sim's own `.venv`.
- No GPU job launched on a busy card — confirmed via `nvidia-smi --query-compute-apps` before any action, and
  `tools/vllm_sleep_pilot_serve.sh up`'s own guard independently refused when tested.
- `tools/gpu_queue.sh` respected — the handoff in §6 routes the actual GPU test through checking its `status`
  first, per the task's instruction to queue or wait rather than run around it.
