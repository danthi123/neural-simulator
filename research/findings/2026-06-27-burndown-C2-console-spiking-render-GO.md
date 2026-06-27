# Burndown C2 — the first-chat console's DEFAULT render word-ordering on SPIKES (GO, 2026-06-27)

**Verdict: GO.** The first-chat console's CERTAIN-fact sentence word ORDER is now produced by the VALIDATED
spiking competitive-queuing read-out (`NeuralSerialOrderRenderer`) **by default**, instead of a host f-string —
on the console's native numpy-CPU backend (and on cupy). The default `--rubric` stays **10/10, moat 0-leak,
VERDICT PASS** (byte-identical surfaces); the equal-drive + lesion anti-cheats FAIL (the neurons serialize, not
a host sort); the moat is structurally intact. **NO `sim/` edit; reuse-by-import; additive.** C2 of the Bucket-A
burndown (`2026-06-27-burndown-bucketA-build-plan.md`), following C1 (commit 324ba8a6).

## The exact gap C2 closes (verified in code)

The build-plan flagged C2 as "flip `enable_neural_render`," but on inspection that flag does NOT cover the
console's actual render path. The console's CERTAIN sentence surface is produced by the **fluency faculty's
`render_svo`**, not the agent's `describe()`/`what_does()`:

```
respond() -> dt.discuss(...) -> DiscursiveTurn._render_verify(prop, faculty)
          -> ct.render_and_verify(svo, faculty) -> faculty.render_svo(a, v, p)
          -> TemplateStubFaculty.render_svo == f"{det_a}{agent} {verb} {patient}."   # the host SVO word-ORDER
```

`enable_neural_render` only rewires `BrainConversationalAgent.describe()`/`what_does()` (the agent's own
`render_fact`/`query_patient` `order_fn=` wire) — which the console's primary certain path never calls. So the
console's certain word-ordering ("The dragonfly hums cod.") was a **host literal**, untouched by the flag.

**C2 therefore does BOTH** (the build-plan note "wire those too / report which remain host"):
1. **The faculty word-order → spikes (the load-bearing change):** the console builds a `SpikingOrderStubFaculty`
   (a `TemplateStubFaculty` subclass, `first_chat_console.py`) whose `render_svo` orders the 3 SVO slots
   `[agent, verb, patient]` via `NeuralSerialOrderRenderer.order([0,1,2])` (the per-pool spiking RATE ranking on a
   real `SimulationBridge`), then assembles the surface in the neural order. On the canonical SVO frame the
   spiking order == `[0,1,2]`, so the surface is **byte-identical** to the host f-string — but neurally produced.
   The asserted SVO it commits to is the canonical `[a,v,p]` (VERIFY checks content, not order — unchanged).
2. **`enable_neural_render=True`** on the agent the console builds (covers any path that reaches the agent's own
   `describe()`/`what_does()`, e.g. an inner-clause patient render), matching C1 / `consolidated_320`.

## The gate decision (corrected vs C1's GPU-gating)

C1 GPU-gated because it wired the renderer into the GPU `ArgStructureComposer` path. But the
`NeuralSerialOrderRenderer` itself is **backend-agnostic** — it builds a small 2-region/484-neuron
`SimulationBridge` that runs on the **numpy-CPU backend too** (~0.5s build, ~5ms/order; measured). And the
console is a numpy-CPU pipeline by default (`os.environ.setdefault("SIM_BACKEND","numpy")`).

**So C2's spiking order defaults ON on the console's NATIVE backend — it does NOT require `SIM_BACKEND=cupy`, and
must NOT use it:** forcing cupy breaks the UNRELATED `SpikingSpeakAccumulator.decide`
(`_value_salience_appraisal_derisk.py:235` does `np.asarray(b.cp_firing_states)`, a pre-existing numpy-only path
in the DiscursiveTurn speak decision — unrelated to word-rendering). This is a STRONGER result than GPU-only: the
flagship numpy console orders words on neurons by default. `--spiking-render off` keeps the host f-string (the
body-emission oracle / fastest path); `--faculty llm` (Path B) supplies its own fluent ordering (the
spiking-order stub is not used then; `enable_neural_render` still flips on).

## De-risk + anti-cheats (`_burndown_C2_console_spiking_render_derisk.py`, 3 seeds, numpy AND cupy → GO)

| seed | PARITY surface | PARITY asserted | real SVO-order | equal-drive (FAIL) | lesion (FAIL) | moat |
|---|---|---|---|---|---|---|
| 42 | 48/48 | 48/48 | 1.000 | 0.083 | 0.306 | OK |
| 43 | 48/48 | 48/48 | 1.000 | 0.028 | 0.528 | OK |
| 44 | 48/48 | 48/48 | 1.000 | 0.028 | 0.417 | OK |

(numpy backend; cupy is equivalent — equal-drive 0.028–0.333, all FAIL.)

- **PARITY (48/48 all seeds):** `SpikingOrderStubFaculty.render_svo` surface == the host f-string surface for the
  canonical SVO frame (byte-identical), and the asserted SVO == `[a,v,p]` (content unchanged → VERIFY unchanged).
- **EQUAL-DRIVE control FAILS (0.028–0.083 ≪ 1.000):** a flat primacy gradient (no agent>verb>patient gap) does
  NOT reproduce the SVO order → the NEURONS serialize via the gradient, not a host sort / pool bias. (The
  canonical anti-cheat of the validated mechanism; read with a random tie-break so no separation → a random
  order, not a stable echo of the input.)
- **LESION control FAILS (≤0.528, all < 1.000−0.30):** zero drive (unconditioned pools, no rate signal) → random
  order. Both controls failing → the serial order requires the primacy current into the spiking pools.
- **MOAT (structural):** the faculty asserts ONLY the 3 content tokens it is given (cannot add/drop/swap one);
  the spiking ORDER reorders those tokens, it can never fabricate one. Abstention is upstream (an unstored fact
  never reaches `render_svo`). The 48/48 asserted-SVO parity is the structural guarantee.

## The HARD GATE (all PASS)

| gate | result |
|---|---|
| rubric 10/10 BEFORE (host f-string) | 10/10, 0 leaks, PASS |
| rubric 10/10 AFTER (default = spiking order) | **10/10, 0 leaks, PASS** — `C2:` log fires; certain surfaces byte-identical |
| spiking render correct (== host order, not garbled) | parity 48/48; rubric surfaces unchanged |
| moat 0-FA | rubric 0 leaks; demo 0 leaks; de-risk moat OK |
| equal-drive / lesion degrades the order | equal-drive 0.028–0.083, lesion ≤0.528 (both FAIL) → neurons order |
| `--spiking-render off` = host oracle | no `C2:` log, byte-identical surfaces, 10/10 PASS |
| numpy default-off byte-identical (no regression) | the original numpy rubric is byte-unchanged |

### Live GPU/CPU transcript (default = spiking-ordered, the C2 conversion)

```
[console] C2: word-ordering on SPIKES (NeuralSerialOrderRenderer competitive-queuing; order==SVO ...)
[phatic ] YOU: hello!                       BRAIN: Hi! Ask me about what I know, or what I think about something.   [moat=OK]
[known  ] YOU: what does curry describe?    BRAIN: Good question. The curry describes pine.                          [C=1 moat=OK]   <- SVO order spiking
[engage ] YOU: what is family?              BRAIN: I don't have settled facts about family, but it tends to come up alongside scraper, tern, and nettle ...  [F=3 moat=OK]
[known  ] YOU: what does plate touch?       BRAIN: Good question. The plate touches autumn. I think maybe the plate floats casserole. ...                    [C=1 moat=OK]   <- SVO order spiking
[known  ] YOU: what does dragonfly hum?     BRAIN: Good question. The dragonfly hums cod. I think maybe the dragonfly flies nectar.                          [C=1 moat=OK]   <- SVO order spiking
[unknown] YOU: what is qwxzptl?             BRAIN: I don't know the word "qwxzptl" yet -- it's not in what I've learned.   [moat=OK]
  RUBRIC SCORE: 10/10   (moat leaks: 0)   VERDICT: PASS
```

## What is now SPIKING vs still HOST (honest)

**Now spiking (default on either backend):**
- The console's CERTAIN-fact sentence word ORDER (the SVO `[agent, verb, patient]` order) — the cognitive
  parallel→serial step the build-plan targets, via `SpikingOrderStubFaculty` → `NeuralSerialOrderRenderer`.
- The agent's `describe()`/`what_does()` word order (`enable_neural_render=True`) — covers any inner-clause
  render that reaches them.

**Still host — legitimately (the body's emission, or NOT a word-ordering operation, per BRAIN-BASED-ONLY):**
- The final `" ".join` + determiner/inflection (the body emitting the neurally-ordered words; KEEP, per the
  build-plan §1 "OneBrainComposer render_fact final join" row).
- The FLAGGED/hedge framing ("I don't have settled facts about X, but it comes up alongside …") — console-
  authored ASSOCIATIVE framing that NAMES the brain's real PPMI neighbours; not an SVO word-ordering (no
  parallel→serial conversion to neuralize). The cognition there is the PPMI graph + the spiking speak decision.
- The phatic replies (canned greetings), the chain-of-thought arrow string, the "Good question." discourse glue —
  non-claim discourse glue, not SVO word-ordering.

So C2's scope — the word-ORDERING of the certain SVO render — is converted to spikes. Sentence FLUENCY beyond
ordering is the Path-B LLM faculty (`--faculty llm`) or Bucket-B learned grammar (out of scope).

## Files

- `research/runners/first_chat_console.py` — added `SpikingOrderStubFaculty` (a `TemplateStubFaculty` subclass);
  `build_brain_on_codes(enable_spiking_order=None)` (default ON, native backend) builds it as the
  `CommunicableTurn` faculty + flips `enable_neural_render`; `--spiking-render {auto,on,off}` CLI (default auto=on).
- `research/runners/_burndown_C2_console_spiking_render_derisk.py` — the parity + equal-drive + lesion + moat
  de-risk (3 seeds, GO on numpy AND cupy). Artifact: `research/findings/raw/_burndown_C2_console_spiking_render.json`.
- `tests/test_first_chat_console_spiking_render.py` — CI guard (5 tests; runs on BOTH backends, unlike C1's
  GPU-only guard, because the renderer runs on numpy too). All pass.

## Next (per the build-plan sequence)

C3 — add `--composer onebrain` to the console (the flat who/what + chain-of-thought + yes/no + generation on the
persistent spiking `OneBrainComposer`), default onebrain on GPU / rf-numpy oracle on CPU. Then C4 (typed
verb-frame on the substrate).
