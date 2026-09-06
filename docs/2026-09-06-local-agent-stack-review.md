# Local agent stack review — harness + Qwen serving + GPU offload (2026-09-06)

Reference/decision note (not a science finding). Owner asked whether to switch the local-model
Claude-substitute from the in-house **Hermes** harness to **DeepSeek** or **Pi**, plus how to serve
Qwen3.8-27B on the single RTX 3090 and how to run the GPU auto-offload. Compiled from a 6-agent
web-verified research sweep. Full report artifact: https://claude.ai/code/artifact/15ec6d75-5659-48b5-b209-980ee83c02ef

## ⭐ REVISED after owner input (2026-09-06, same session)
The owner corrected/steered several points, which SHIFT the recommendation:
- **Offload is ORTHOGONAL to the harness** — it's a supervisor wrapping the model SERVER, not a Hermes-only
  capability. So it is NOT a reason to stay on Hermes; we keep + re-wrap it around any harness (cheap; Claude
  Code + Hermes are templates — don't fear the rebuild).
- **Owner requires ONE continuous, scrollable session** (like Claude Code), not many short sessions. Hermes v2's
  fresh-session-per-turn design fails this by construction. ⇒ this flips the call: **MIGRATE to OpenHands** (its
  LLMSummarizingCondenser + resume give one session with linear context) — now the LEAD, not "optional". Pi is the
  lightweight alt (sub-1k-token harness tax). OpenHands also doubles as a reusable local-automation platform the
  owner wants beyond this project.
- **VRAM budget is ~19-20GB, not 24** (3.5-5GB monitors, NO 2nd GPU coming). ⇒ **Q4_K_M, NOT Q5** (Q5 ~19-21GB
  leaves no room for KV+monitors). Q4_K_M ~15-16GB + Q8 KV ~2.6GB@80k + ~1GB overhead ≈ ~19GB (fits, tight);
  ~64-100k context; drop KV→Q4 for more room. Q4_K_M download (Unsloth Dynamic UD-Q4_K_M) STARTED this session.
- **Vision already OFF** (`qwen_serve.sh --no-mmproj`, llama.cpp text-only) — keep it off if migrating to vLLM.
- **vLLM Sleep Mode**: owner enthusiastic; pilot it (fast in-place offload). Offload dance stays essential (no 2nd GPU).

**Revised order:** (0) Q4_K_M + Q8 KV text-only [running] → (1) pilot vLLM Sleep Mode w/ Q4 → (2) prototype
OpenHands on the local endpoint + wrap our offload supervisor → (3) cut Hermes' role over if it holds; keep the
supervisor + gate/push conventions. Full report artifact updated (same URL above).

## Reframe (original)
The harness is NOT the weak link — the model QUANT is. Hermes' loop (`tools/hermes/loop.py`) is solid
and owns the one thing no alternative provides (GPU offload/reload). It drives Qwen3.8-27B at `Q2_0`
(~2-bit, `sdkyuan/qwen3.8-27B-qat-q2_0-gguf`), the real quality ceiling of the stack.

## Harness verdicts
- **DeepSeek — DON'T switch.** No DeepSeek model fits a single 3090 (all 250B+ MoE, need 90GB+ pooled).
  `dsh` harness (released 2026-08-13) is a 3-week preview with compaction bugs in exactly our
  long-context-local case; adds nothing on offload. Cloud API = a separate paid hard-task-fallback idea only.
- **Pi (earendil-works / M. Zechner) — optional experiment, not migration.** Nice minimal multi-provider
  loop; no VRAM/offload lifecycle, no sandbox — you'd rebuild Hermes' hard part regardless.
- **OpenHands = the real upgrade candidate if any** (not DeepSeek/Pi): autonomous headless loop + built-in
  context condenser (linear-cost summarization — the exact 80k-turn problem) + session resume. Optional.
- Study-only references: Nous "Hermes Agent" (name collision, NOT our Hermes — lineage context compression),
  mini-swe-agent (~100-line loop). Ruled out: Roo (archived), Open Interpreter (unsafe unattended),
  Aider/Cline/Continue/Kilo (IDE approval-per-step). No field harness solves the GPU-offload requirement.

## Serving Qwen3.8-27B on one 3090 (24GB)
- Qwen3.8-27B is a REAL current dense model (2026-08-14), already the one in use. Hybrid attention
  (3 Gated-DeltaNet linear : 1 full-attn; only 16/64 layers grow KV) => KV cache ~4x smaller than a
  standard 32B => 80k-token turns are viable on 24GB (the 2026-09-01 vLLM 30k wall was a standard model
  + un-quantized KV, not a hard ceiling).
- BIGGEST WIN: upgrade `Q2_0 -> Q4_K_M` (+`Q8 KV cache`). Q2 is degraded for code/reasoning; Q4_K_M keeps
  ~92-95% and fits comfortably (weights ~15-18GB + Q8 KV ~2.6GB @ 80k). Q5_K_M/AWQ (~19-21GB) if headroom.
- Backend: re-pilot vLLM for Sleep Mode (`/sleep`+`/wake_up`, ~3-6s in-place VRAM release) — purpose-built
  for shared-GPU offload, better than kill/cold-reload. Keep llama.cpp (Q5 + `--cache-type-k/v q8_0`) as the
  proven fallback. Stock vLLM + this hybrid arch is ~3wk old — expect rough edges; Sleep Mode is mainline.
- 2nd 3090 (48GB): removes the weights-vs-KV tension; dedicate GPU0=model, GPU1=experiments.

## GPU offload — current design is already correct
`gpu_queue.sh`+`qwen_serve.sh`+`loop.py` are hardened against the real incidents (double-daemon race;
reload-on-running-job -> OOM/bus-off). Research confirms explicit-unload + full-process-kill (a caching
allocator won't truly free in-process) + sentinel/flock. Incremental only: a `.last_gpu_result.json` harvest
pointer prepended to the turn prompt; a `GPU_PRIORITY` sentinel checked only between jobs (never preempt a
running experiment); a recovery test; delete the dead `qwen_supervisor.sh` (+ its stale "the crux" docs).

## Recommended order
1. Quant `Q2->Q4` + `Q8 KV` (biggest win, lowest effort) — sanity-check output on real repo tasks vs Q2.
2. Re-pilot vLLM Sleep Mode (measure wake time + hybrid-arch stability on native Linux).
3. Harden offload harvest + prune dead code.
4. Optionally evaluate OpenHands (its condenser is the draw).
5. Plan the 2nd-3090 one-card-per-role split.

## Caveats
Current model + DFlash2 drafter are community re-quants (provenance risk). The 150k+ context numbers are
one enthusiast repo (patched vLLM); Q8-KV/80-100k is well-supported. vLLM 3-6s wake is a general figure,
not Qwen3.8-27B-specific — validate before trusting the reload SLA. No committed eval exists of the current
Q2 model's actual dev-decision quality — generate one (Q2 vs Q4 on the same tasks) as part of step 1.
