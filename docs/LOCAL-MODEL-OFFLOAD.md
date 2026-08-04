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

Start the service only through the foreground ownership broker, and keep that
process alive for as long as the service is needed:

```bash
python tools/local_model_offload.py broker
```

The broker acquires `/tmp/sim-local-model-gpu0.lock` before spawning Docker.
The GPU lane dispatcher uses the same lock, so broker startup fails while an
experiment owns GPU 0 and queued experiments wait while the model is resident.
The foreground service child inherits the broker's lease descriptor, so an
abrupt broker exit does not free GPU 0 while that child is still alive.
Stopping the foreground broker clears request authorization, runs the
configured compose `down` command, and only then releases the lease.

Do not start the old shell wrapper or compose file directly. They may still
hold or consume the GPU, but they cannot create the verified ownership record,
so all worker requests to them fail closed.

## Ownership State Machine

The service and experiment lanes share one exclusive kernel `flock`. A JSON
record alone never grants access. It must match the live broker's PID start
time, Linux boot ID, configured endpoint and lease path, and the PID reported
by `/proc/locks` for that exact lease inode.

| State | Event | Result |
|---|---|---|
| `absent` + lease free | `broker` starts | broker locks GPU, publishes `service_owned`, then starts Docker |
| `absent` + lease busy | `broker` starts | fails before Docker is spawned |
| `service_owned` + identity and lock match | worker request | request may use the service without reacquiring the lock |
| missing, malformed, stale, or mismatched owner | worker request | `unsafe_service`; endpoint is not contacted |
| stale owner + lease busy | `recover-owner` | `blocked_busy`; metadata is preserved because an experiment may own the GPU |
| stale owner + lease free | `recover-owner` | stale record is removed while holding the lease |
| broker exits | cleanup | authorization removed, service stopped, lease released, in that order |

CPU-only ownership inspection and recovery commands are:

```bash
python tools/local_model_offload.py owner-status
python tools/local_model_offload.py recover-owner
```

## Worker Contract

The worker is disabled in the committed configuration. Enable it before
starting the broker, after checking that the configured model ID matches the
recipe:

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

## Durable Queue And Handoff

Eligible drafting work can be staged while an experiment owns the GPU. These
commands only read and write local files; they do not probe the endpoint, start
the service, or inspect the GPU:

```bash
python tools/local_model_offload.py enqueue \
  --task documentation \
  --input /tmp/draft-request.md
python tools/local_model_offload.py list
python tools/local_model_offload.py recover
```

Tasks are stored under the configured `queue_directory` in atomic `pending`,
`running`, and `receipts` records. `recover` returns claims left behind by an
interrupted worker to the pending queue. Prompts are stored as plain text in
this local state directory, so secrets and private user data must not be
queued.

When the experiment lane is clear and the intentionally started service is
ready, process one item at a time:

```bash
python tools/local_model_offload.py process-next
```

A busy GPU, unavailable endpoint, or failed request produces a receipt and
returns the task to pending. A successful generation removes the claim but is
recorded as `awaiting_review`, never approved. Its output cannot inform code,
scientific conclusions, experiment design, parameter selection, or gates until
a human or higher-capability model reviews it.

## RAG Boundary

RAG remains the project's authoritative locator for its own findings and the
local biology catalog. The local model may summarize material that has already
been retrieved, but it does not replace `tools/rag/search.sh`, source reading,
the research gate, or primary-source verification. RAG/index refresh failures
must remain visible rather than being hidden by a local-model answer.
