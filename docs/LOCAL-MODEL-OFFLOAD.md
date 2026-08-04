# Local Model Offload

The project has an optional local drafting assistant on the RTX 3090. It is a
cost-control tool, not part of the simulated brain and not a source of
scientific authority.

## What It May Do

The worker accepts only three task classes:

- `documentation` for plain-language drafts and summaries;
- `research_synthesis` for organizing already-reviewed source notes; and
- `catalog_triage` for suggesting which existing references deserve human or
  primary-model review.

Its output is always provisional. A human or the primary engineering model
must read the cited sources before a claim enters the catalog, a finding, an
experiment specification, or a gate decision. The local model cannot edit
code, choose parameters, approve a gate, or silently fall back to a hosted
model.

## Installation And Service Lifetime

The prepared checkout is `/home/dant123/Projects/club-3090` and the selected
single-card recipe is Qwen3.6-27B GGUF. Downloading weights is separate from
starting the service. Hugging Face terms acceptance and a read token may be
required by the model publisher.

Start the service only when GPU experiments are not running, and keep the
wrapper process alive for as long as the service is needed:

```bash
bash tools/local_model_service.sh up
```

The wrapper holds `/tmp/sim-local-model-gpu0.lock`. The GPU lane dispatcher
uses the same lock, so queued experiments wait while the model is resident.
Use `down` to stop the service and release its VRAM:

```bash
bash tools/local_model_service.sh down
```

Do not start the compose file directly with `docker compose up -d`; that
bypasses the lease and can create GPU contention.

## Worker Contract

The worker is disabled in the committed configuration. Enable it only after
the service is intentionally running and the model ID in
`config/local_model_offload.json` matches `/v1/models`:

```bash
python tools/local_model_offload.py probe
python tools/local_model_offload.py run \
  --task documentation \
  --input /tmp/draft-request.md \
  --output /tmp/local-model-draft.json
```

Every successful artifact records the endpoint, served model ID, GPU state,
prompt hash, response hash, timestamps, and `review_required: true`.
Unavailable, rejected, and GPU-busy states are explicit machine-readable
results. No result is treated as completed unless the local endpoint, model
identity, GPU state, and exclusive lease all pass.

## RAG Boundary

RAG remains the project's authoritative locator for its own findings and the
local biology catalog. The local model may summarize material that has already
been retrieved, but it does not replace `tools/rag/search.sh`, source reading,
the research gate, or primary-source verification. RAG/index refresh failures
must remain visible rather than being hidden by a local-model answer.
