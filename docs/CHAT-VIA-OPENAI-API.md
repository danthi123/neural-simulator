# Chat with the sim brain via the OpenAI-compatible API

The sim brain is exposed over the OpenAI `/v1/chat/completions` + `/v1/models` contract, so ANY standard LLM
client — Open WebUI, LibreChat, the `openai` Python library, `curl` — can talk to it with no custom UI. This
guide is the exact launch command + client config to point a client at a **developed knowledge brain** (the
21,777-fact Wikidata cortical long-term store) with the **no-confabulation moat** intact.

**Last updated:** 2026-08-21.

Two things ride each response:

- the **brain's reply** → `choices[0].message.content` (the conversation surface)
- the brain's honest **internal monologue** → `choices[0].message.reasoning_content` (the "thinking" panel most
  modern clients render collapsibly). Every line is a functional read-out of a real internal signal (mood,
  decision-margin, recall trace, the abstain), never a claim of phenomenal experience.

---

## Launch (CPU, GPU-free — knowledge + honesty on any machine)

The knowledge brain and the moat run on the CPU (numpy backend). Fluent PROSE additionally needs the Qwen mouth
(slow on CPU, fast on a CUDA GPU) — see the renderer note below; the DEFAULT CPU launch answers with a
correct, moat-verified **template** surface (the content is right; the prose is plain).

```bash
cd /home/dant123/Projects/sim
SIM_BACKEND=numpy \
BRAIN_GNW_BUS_HOST=1 BRAIN_GNW_2ORGAN=0 \
BRAIN_LTM_BUNDLE=/home/dant123/Projects/sim/research/findings/raw/_knowledge_bundle_wikidata_100k/ltm_store_partial \
BRAIN_CHAT_BUNDLE=/home/dant123/Projects/sim/research/findings/raw/_knowledge_bundle_wikidata_100k/chat_brain_bundle \
.venv/bin/python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8799
```

What each variable does:

- `SIM_BACKEND=numpy` — run on the CPU (portable, no GPU needed). Use `cupy` on a CUDA box for the fast path.
- `BRAIN_CHAT_BUNDLE=<dir>` — the developed-brain bundle the shim serves (a directory with `brain.json`).
  This is the new selector that makes the shim serve a DEVELOPED brain instead of the built-in `tiny-demo`.
- `BRAIN_LTM_BUNDLE=<dir>` — the bulk cortical LONG-TERM knowledge store (the 21,777-fact Wikidata bundle),
  installed as a routed sharded phasor store beside the brain's small conversation working-set. Reads check the
  working-set buffer first, then the routed LTM shard — sub-second at any scale.
- `BRAIN_GNW_BUS_HOST=1` + `BRAIN_GNW_2ORGAN=0` — route factual recall through the host FHRR gate (forward
  recall + the moat's abstain). The GNW multi-organ consensus combiners are calibrated on the small conversation
  working-set and cannot corroborate a 21k-fact LTM recall, so for a KNOWLEDGE-serving brain they are reverted to
  the host gate, which reaches the LTM AND keeps the no-confab moat. (See the caveat at the bottom.)

`GET /v1/models` then lists one model, id `sim-brain`.

---

## Client config

Point any OpenAI-compatible client at the server:

- **base_url:** `http://localhost:8799/v1`
- **model:** `sim-brain`
- **api_key:** any non-empty string — the server does not check it (e.g. `sk-noauth`).

The OpenAI `model` field may ALSO name a brain directly — `tiny-demo`, `self-knowledge`, or an absolute
developed-brain bundle path — which overrides `BRAIN_CHAT_BUNDLE` for that request. A generic id
(`sim-brain`, `gpt-4`, …) means "use the configured default brain", so any stock client works unchanged.

### `openai` Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8799/v1", api_key="sk-noauth")

r = client.chat.completions.create(
    model="sim-brain",
    messages=[{"role": "user", "content": "What is Canada?"}],
)
print(r.choices[0].message.content)
# -> the brain's reply, recalled from the 21k-fact LTM (Canada is a country)
# r.choices[0].message.reasoning_content carries the honest internal monologue
```

### `curl`

```bash
# list the model
curl -s http://localhost:8799/v1/models

# a known fact -> answered from the LTM
curl -s http://localhost:8799/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"sim-brain","messages":[{"role":"user","content":"What is Canada?"}]}'

# an unknown subject -> honest abstain (the no-confab moat), NOT a made-up answer
curl -s http://localhost:8799/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"sim-brain","messages":[{"role":"user","content":"What is a snarklebee?"}]}'
```

Verified responses (CPU, template renderer):

- **"What is Canada?"** → `content`: `"The canada isas country. …"`, with `reasoning_content` line
  `recalled from my store: canada isa country`. The fact comes from the 21,777-fact LTM.
- **"What is a snarklebee?"** → `content`: `"I don't know about that. …"`, with `reasoning_content` line
  `I have no grounded trace for this — I'm declining rather than confabulating`. The moat holds.

Streaming (`"stream": true`) is supported: the monologue streams on the `reasoning_content` delta channel first,
then the reply on `content`, then `[DONE]`. Clients that ignore `reasoning_content` still get a correct reply.

---

## CPU vs GPU (cupy), and fluent prose

- **Knowledge + honesty work on either backend.** The LTM recall and the no-confab moat are CPU-portable; they
  do not need a GPU.
- **Fluent PROSE (the off-bridge Qwen-0.5B mouth) is the only GPU-sensitive part.** On a CUDA box
  (`SIM_BACKEND=cupy`) the Qwen renderer runs on the GPU and is fast. On a GPU-less host it now runs on the CPU
  in float32 — real fluent generation, just slower (a single short render is seconds to tens of seconds; the
  first turn also pays a one-time model warm).
- **Opt into the CPU Qwen mouth** with `BRAIN_CHAT_RENDERER=qwen` in the launch env. Without it, a GPU-less host
  uses the GPU-free template renderer (correct content, plain prose). With it, the same knowledge + moat flow,
  now through fluent Qwen prose (slower per turn on CPU).

```bash
# GPU-less fluent-prose launch: add the Qwen mouth (slower per turn)
BRAIN_CHAT_RENDERER=qwen SIM_BACKEND=numpy BRAIN_GNW_BUS_HOST=1 BRAIN_GNW_2ORGAN=0 \
BRAIN_LTM_BUNDLE=.../ltm_store_partial BRAIN_CHAT_BUNDLE=.../chat_brain_bundle \
.venv/bin/python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8799
```

---

## Caveats (honest)

- **The prose surface for a Wikidata "instance-of" fact is plain.** These facts store the relation as the token
  `isa` (e.g. `canada isa country`), so the template renderer emits `"The canada isas country."` and the Qwen
  mouth's more natural `"Canada is a country"` is VERIFY-rejected (the moat treats the changed verb as drift and
  falls back to the raw triple). The CONTENT is correct and moat-verified either way; only the surface polish is
  affected. A clean subject-verb-object fact (e.g. `dog chases cat`) renders as fluent prose normally.
- **The GNW consensus faculties are OFF in this knowledge deployment** (`BRAIN_GNW_BUS_HOST=1`,
  `BRAIN_GNW_2ORGAN=0`), because their corroborating organs only know the small conversation working-set and
  cannot vote on a 21k-fact LTM recall — they would veto every LTM answer. Making those organs LTM-aware is a
  tracked follow-up; until then the knowledge path uses the host FHRR gate, which keeps the no-confab moat.
- **First turn is slow.** Building the developed brain + loading the LTM shards takes ~20s on CPU; with the Qwen
  mouth add the one-time model warm. Subsequent turns reuse the warm brain/renderer.
