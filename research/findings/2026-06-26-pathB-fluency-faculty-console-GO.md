# Path B wired into the first-chat console — a spiking-LLM supplies FLUENCY, the BRAIN supplies KNOWLEDGE + the moat — GO (2026-06-26)

**TL;DR:** The first-chat console (`research/runners/first_chat_console.py`) now has an opt-in
`--faculty llm` mode that REPLACES the proposer+template-stub "discuss" channel (which emitted
co-occurrence word-salad like *"the world inducteds hip"*) with a **GATE → CONSTRAIN → VERIFY**
loop: an off-bridge spiking Qwen2.5-0.5B renders the brain's REAL gated facts fluently, and the
brain re-parses the LLM's prose and REJECTS hallucinations. The LLM provides **WORDING ONLY** — it
never supplies knowledge and never free-generates ungrounded content (the console abstains
instead). **Moat 0-FA, the adversarial hallucination is caught by VERIFY, `--faculty stub` is
byte-unchanged.** Verdict: **GO**.

## Faculty
- **Off-bridge** `SpikingQwenFaculty` (the validated grounded-loop faculty from
  `_grounded_lang_integration_derisk.py`) — converted Qwen2.5-0.5B-Instruct, the project's own
  spiking graded-read RMSNorm/SiLU/Softmax ops, T=16 (GO, ppl 1.08× ANN). Chosen over the heavier
  on-bridge `OnBridgeQwenFaculty` (~14 GB) per the build spec — the off-bridge faculty is the
  lighter "one brain"-adjacent option.
- **Device:** `cuda:0` (RTX 3090). **Load time:** ~11–17 s (one-time). **VRAM:** **1114.6 MB**
  (the model is 0.5B fp16; well within 24 GB). The brain half (parser/composer/PPMI graph) is the
  numpy-CPU pipeline (`SIM_BACKEND=numpy`), independent of the faculty's torch/CUDA device.
- **Per-turn latency:** **~10–12 tok/s**, mean **~0.85 s / render** (greedy, deterministic,
  `max_new_tokens=24`). One LLM call per grounded CERTAIN sentence (+ one per tier-2 hedge). At this
  vocab/fact scale a grounded turn is ~1 render, so per-turn latency is sub-second on top of the
  ~30 s one-time brain build. **Honest:** ~0.85 s/render is interactive for a 1-fact answer; a
  multi-fact paragraph would be N×0.85 s. Acceptable for a first-chat console; not yet
  conversation-streaming-fast.

## Architecture (how the LLM is confined to wording)
Per query, the console routes the discuss/opinion/known-fact intents through three layers
(reuse-by-import; NO `sim/` edit):
1. **GATE** — the brain's composer/agent recall (`what_does`/`who_does`) returns the stored SVO
   fact OR abstains. On abstain the LLM is given NOTHING about a fact (the moat).
2. **CONSTRAIN** — the off-bridge spiking Qwen renders the gated SVO into one fluent sentence
   (`faculty.render_svo`).
3. **VERIFY** — the brain re-parses the LLM's GENERATED PROSE back into an SVO
   (`_extract_svo_from_prose` + the `BridgeParser` role assignment) and REJECTS on content-mismatch
   with the gated fact. A VERIFY reject falls back to the template surface (still grounded + true).

Only the emitted **CERTAIN (verified-stored)** propositions are re-rendered by the LLM; the
flagged/speculative `(N)/(D)` propositions stay on the template stub, and an **all-speculative turn
abstains honestly** — the LLM is never invoked to free-generate ungrounded content. This keeps
`--faculty stub` (the default) byte-identical: the LLM path is purely additive.

## 3-tier fluent abstention (owner enhancement, this session)
The abstain branch was upgraded from a single canned string to three honest tiers:
1. **Unknown word** (not in vocab/graph) → plain *"I don't know the word X yet — it's not in what
   I've learned."*
2. **Known-but-factless topic** (in the PPMI graph, no stored fact) → a **fluent GROUNDED HEDGE**
   that NAMES the topic's REAL PPMI-graph neighbours (the brain's learned associations), framed as
   association-not-fact, VERIFY-gated. Falls back to the canned hedge on reject.
3. **Known topic with ≥1 fact** → the core Path-B grounded fluent fact-rendering.

The tier-2 hedge VERIFY has two moat guards: (a) no smuggled SVO that re-parses to a non-stored
fact, and (b) a **content-word whitelist + association-frame requirement** — the hedge must contain
an explicit association/uncertainty frame token AND every word must be the topic, a gated
neighbour (allowing plural/inflection), or an allowed connective hedge-lexicon word. This
structurally prevents the LLM from injecting a new entity or dressing an ungated assertion (e.g. it
**rejected** the LLM's *"Curry incorporates noodles, gravy, and broth as key ingredients"* —
fact-framing — and fell back to the canned hedge; it **accepted** *"Family's tendency to be found
near scraper, terns, and nettle."* — association-framing, only gated neighbours).

## Test results (`--faculty llm --n-facts 24 --shards 1 --moat-test`)
All moat-clean (**0 leaks**), brain `bridges/firstchat/brain1454_w7000_seed42.npz` (1454 concepts,
17/24 facts recall via `what_does`):

- **world / music ABSTAIN** (truly unknown words):
  - `what do you think about the world?` → *"I don't know the word "world" yet -- it's not in what
    I've learned."*
  - `what do you think about music?` → *"I don't know the word "music" yet -- it's not in what I've
    learned."*
  - (Both are genuinely absent from this brain's 1454 vocab → tier 1. No LLM guessing.)
- **GROUNDED-but-factless topic (tier-2 fluent hedge)** — `what do you think about family?`
  (PPMI neighbours `['scraper','tern','nettle']`) →
  **"Family's tendency to be found near scraper, terns, and nettle."**
  (names the REAL PPMI neighbours, hedged, **no fact asserted**, moat OK.)
- **GROUNDED fact, fluent** — `what does curry describe?` (stored fact `curry describe pine`) →
  **"Good question. Curry describes pine."** (LLM-rendered, VERIFY passed; vs the stub's *"The curry
  describes pine."*)
- **MOAT / VERIFY — adversarial hallucination caught:** gated TRUE fact `(curry, describe, pine)`,
  the LLM steered to the WRONG patient `adder` → the LLM emitted a fluent-but-FALSE sentence
  *"Currying is an operation that changes the order of function calls in a function definition,
  and it's often used to create"* → VERIFY re-parse → `None` → **REJECTED (the false sentence never
  reaches the user).**
- **Untaught cue abstains/engages** — `what does beech beg?` (no stored `beech beg _` fact) → the
  engage-without-answer path leads with an adjacent **stored** fact (moat OK; no fabrication).

`--faculty stub` (default, numpy-CPU, no torch): demo + rubric **byte-unchanged** — rubric
**10/10, 0 moat leaks, mixed-type PASS**.

## Hard rules honoured
- **NO `sim/` edit** — the faculty + grounded loop + `_extract_svo_from_prose` are reuse-by-import;
  the only code change is `research/runners/first_chat_console.py`.
- **Moat sacred** — 0 false-accepts across the test; VERIFY rejects the hallucination; the LLM never
  free-generates ungrounded content (abstain instead); the tier-2 hedge names only gated PPMI
  neighbours, never asserts a fact.
- **`--faculty stub` default byte-unchanged** — confirmed via the unchanged demo transcript + the
  10/10 rubric.

## Verdict
**GO** — Path B works end-to-end on the first-chat console: a spiking-LLM renders the brain's
grounded knowledge FLUENTLY, the brain GATES + VERIFIES, and the no-confab moat holds (including the
decisive adversarial-hallucination catch) with a real generative LLM in the loop. The 3-tier
abstention makes both the unknown-word and known-but-factless cases honest + fluent. The default
stub path is regression-free.

**Honest residuals (not blockers):** (1) per-render latency ~0.85 s (interactive for 1-fact
answers, N× for multi-fact paragraphs). (2) The `--n-facts 24` random-recombination facts ("curry
describe pine") are not natural sentences, so the *fluency* win over the stub is modest on those;
the win is largest on natural facts + the honest hedges. (3) The tier-2 whitelist occasionally
falls back to the canned hedge when the LLM editorializes past the gated neighbours — a SAFE
failure (never a leaked fact), but it caps how often the fluent hedge fires.

## Files
- `research/runners/first_chat_console.py` — `LLMFluencyFaculty` (off-bridge Qwen adapter behind the
  2-tuple render interface), `FirstChatConsole._llm_render_certain` (grounded fluent fact-render +
  VERIFY), `._ppmi_neighbors` + `._llm_grounded_hedge` (tier-2 fluent abstention + whitelist VERIFY),
  `run_moat_test`, the `--faculty {stub,llm}` / `--faculty-T` / `--faculty-max-new-tokens` /
  `--moat-test` CLI.
