# SK-load latency RESOLVED + interact console COMPLETE + week-1 develop run (2026-06-24)

Closes the owner-chosen "keep digging on SK latency" thread and the capstone+console
arc. All webapp-only; no `sim/` edits.

## The big win: self-knowledge brain first-load ~9.8 min -> 0.7s (~800x)

The self-knowledge bundle (`bridges/developed/self_knowledge/brain`, 52 facts, 106
vocab) took ~9.8 min to load in the console. Profiling (`bwy27t6g6`: cProfile of
`load_developed_brain`) localized it precisely:

- `content_selection_spiking.SpikingLoopContextBuffer.__init__` called
  `bridge.set_pathway_weights` **144 times** = **681 s** (each call rebuilds the whole
  CSR: `tocsr` + `coosort` + `lexsort`). That working-memory loop is only needed for
  multi-turn anaphora, not for answering a who/what question.

Fixes (all reuse-by-import, `3bc96527` + `20b20f9a`):
1. **Lazy-defer the WM loop** — `MultiTurnAgent(defer_planner=True)`; the
   `SpikingLoopContextBuffer` + biased-competition buffer build on first referent
   write, not at load.
2. **Batch the 144 CSR rebuilds into 2** — `SpikingLoopContextBuffer.__init__` now
   accumulates all attractor edges and calls `set_pathway_weights` twice (c2d + d2c)
   instead of per-edge.
3. **Lazy parser** (`defer_parser=True`) — the `BridgeParser` train is only needed to
   comprehend a NEW taught sentence; skipped on load.
4. **Persist the composer KB composites** — `extract_kb_composites` / `_load_kb_composites`
   save the per-fact complex composites so load skips the per-fact resonate.

Direct re-measure (`b81rs1f0n`, SIM_BACKEND=cupy): `LOAD 0.0s n_kb=52 wm_deferred=True`,
first query 0.7s, warm query 0.47s, moat abstains on an untaught cue. ~800x.

## The residual first-turn cost was NOT the brain — it was the Qwen renderer

After the brain-load fix, the live console's *first* turn on the SK bundle still took
~58s. The `renderer` field gave it away: **"off-bridge Qwen-0.5B (spiking forward)"**.
That 58s is the Qwen-0.5B language model loading (downloads from HuggingFace Hub on
first use) — a one-time, per-process renderer cost, separate from the brain. Warm
turns are 1.7s.

Fix (`1d808b00`): **warm the Qwen at webapp startup** — a `@app.on_event('startup')`
daemon thread pre-builds the default ChatBrain's renderer via a process-wide warm
`QwenRenderer` singleton (double-checked locking, so the model loads exactly once even
if a first turn races it); `_build_chat_brain` reuses the singleton. Plus
`os.environ.setdefault('SIM_BACKEND','cupy')` when a CUDA GPU is present (numpy is
~20x slower for the brain ops; `SIM_BACKEND=numpy` still forces the CPU path). The
live log confirms it runs: `[webapp] startup: warming the off-bridge Qwen-0.5B
renderer` + the HF-Hub download. For a human opening the console, the warm finishes
in the background while they orient, so their first real turn is fast.

## Two Windows-console Unicode crashes fixed (prints now all-ASCII)

Logic-only validation can't see console-encoding. Caught on live restart:
- `f0d68f95`: a `->` (U+2192 arrow) in the cupy-default **module-load** print crashed
  the Windows charmap codec at import -> the webapp failed to boot. Replaced with ASCII.
- `7c524f44`: a `...` (U+2026 ellipsis) in the **startup-hook** warm print. This one
  degrades to a replacement char rather than crashing (uvicorn's stdout tolerates it
  by then), but cleaned for consistency. A scan confirms no non-ASCII remains in any
  `print()`.

## Live verification (run `bhnhjrjyx`, cupy)

- SK bundle: "how do you learn" -> "The brain learns words"; "what do you use" -> "The
  brain uses spikes to transmit information" — both `verified`, recalled SVO correct.
- tiny-demo: "what does the dog chase" -> "The dog chased the cat", `verified`.
- Moat: an untaught cue abstains (`None`).
- Warm chat turns: 1.7-5.5s.

Console: `localhost:8765` -> Interact -> pick `self_knowledge/brain` -> chat about the
project.

## Capstone payoff: week-1 develop-to-disk run (in flight, `b08132cle`)

`python -m research.runners._longitudinal_develop_loop_gpu --n-days 7 --save-bundle
bridges/developed/week1 --per-day-bundles` — the brain develops over a simulated week
(WAKE real-stream-cortex -> CONVERSE -> SLEEP replay+retention -> GROWTH -> PERSIST),
saving a self-contained bundle per day. The console `/api/brains` scan finds depth-2
per-day bundles and labels them `week1/day_<N> (day N)`, so the owner can load each day
and watch the brain grow (vocab + facts). ETA ~15.6 min (7 x ~2.2 min/day).

## Open items

- **Verify the week-1 run end-to-end** (pending `b08132cle`): confirm the 7 per-day
  bundles + final brain landed under `bridges/developed/week1/`, the console picker
  lists them, and day_0-vs-day_6 shows visible growth.
- **B3 — per-turn chat activity visualization** in the console (the "visualize brain"
  job): show the brain's spiking activity during a chat turn.
- **A2 — scale the develop horizon** (compressed-month / -year) once the week run is
  confirmed.
