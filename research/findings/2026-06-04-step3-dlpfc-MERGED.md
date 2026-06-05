# One-bridge unification step 3 — Task 2: dlPFC dialogue-planning loop MERGED onto the unified bridge (qualified) — 2026-06-04

**Verdict: QUALIFIED MERGE.** The dlPFC (dialogue-planning / "what to say next") working-memory loop now runs as
persistent index slices on the SAME `SimulationBridge` as the parser + composer, at the parser/composer timestep
**dt=1.0 ms**. On the unified bridge the merged `elaborate` reproduces **every criterion the dlPFC was originally
validated on** (content_selection_spiking, 6/6 seeds 2026-06-03): the pick is a DIRECT (1-hop) on-topic associate,
it abstains on an unconnected topic (the no-confab moat), it is deterministic, and a multi-turn elaboration stays in
the topic region. The one documented qualification: the merged path does NOT always reproduce the *separate dt=0.5
oracle's exact per-topic associate* — and that is a dt-resolution tie among **equidistant** direct neighbours, not a
capability loss. **All three conversational regions (parser, composer, dlPFC) are now one interacting brain on one
bridge.**

## What merged, and how (no `sim/` edit, no shared-module edit)

`UnifiedBrainBridge(enable_dlpfc=True)` (research/runners/unified_brain_bridge.py) adds the dlPFC `cortex_ctx`/
`dlpfc_wm` reverberatory loop as further persistent index slices on the unified bridge:

- **Per-region NMDA, isolated to the dlPFC slice.** One bridge has one global `enable_nmda`, but only the dlPFC
  slice gets NMDA receptors (the cluster-G per-region NMDA mask). The bridge log confirms it: `NMDA per-region mask:
  2 regions enabled (1200 neurons)` — `cortex_ctx` + `dlpfc_wm` only; the parser+composer slices stay NMDA-free.
- **dt=1.0, NMDA-dependent attractor weight ≈30** — the faithful merge regime from the step-3 Task-1 de-risk
  (`2026-06-04-step3-dlpfc-dt-survives.md`): the WM latch is genuinely NMDA-dependent at weight 30 (not the
  saturated weight 50 = AMPA ping-pong) and survives dt=1.0.
- **`elaborate` reuses the validated `SpikingSpreadingController` methods verbatim** (`_install_graph_edges`,
  `relevance_by_latency`, `turn_latency`, `_reset_wm`) against the shared-slice context — reuse-by-import, the
  association graph stays Python-built from the agent's own facts (scope clamp; making the fact→graph hand-off
  synaptic is a hypothetical step 4, out of scope).

## The one load-bearing fix found during verification: the dlPFC must run OU-OFF

The unified bridge runs OU background noise ON (σ=20 pA) for the parser+composer. But the dlPFC's VALIDATED
dialogue-planning config (`content_selection_spiking.SpikingSpreadingController`, lines 243–254) runs **OU OFF** —
its own validation finding states OU noise "tips the bistable concept attractors into spurious ON states" (Hopfield
spurious states), corrupting the latency-ranked selection; OU-off gave the 6/6-seed result. Running the merged dlPFC
with the inherited OU-on is therefore the *known-degraded* regime. The fix is principled, not a tuning hack:
`elaborate` toggles `core_config.enable_ou_process = False` for the duration of the dlPFC read (the parser+composer
are not active during an `elaborate` call — it drives and reads the dlPFC slice alone — so the toggle is local), and
`cfg.enable_ou_process` is read dynamically each step (`sim/bridge.py:5122`), so the toggle is clean.

## The characterization — the dlPFC's VALIDATED criterion, met on the unified bridge

Probe `research/findings/raw/_step3_dlpfc_char_probe.py`, seed 42, facts `dog go north / cat come south /
dog look river`. Merged = unified bridge dt=1.0 OU-off; oracle = the separate-dlPFC `BrainConversationalAgent` dt=0.5.

| topic | direct neighbours | merged pick | oracle pick | merged latency ranking (first-spike step) |
|---|---|---|---|---|
| dog   | go, look, north, river | **go** (direct) | look (direct) | `go=look=north=river=23` (**4-way EXACT TIE**) |
| cat   | come, south            | **come** (direct) | come (direct) | `come=south=22` |
| river | dog, look              | **dog** (direct) | dog (direct) | `dog=look=22` |

- **Every merged pick is a DIRECT neighbour** (the validated criterion). Only direct neighbours ever fire — for
  `dog`, the off-topic `cat/come/south` never spike, so "direct-neighbour" is a genuine discriminating bar, not
  trivially satisfiable.
- **Deterministic:** the merged latency ranking is identical across repeated runs (OU-off).
- **Abstention (no-confab moat):** merged returns `None` on the unconnected topic `apple` (so does the oracle).
- **Multi-turn coherence:** a 3-turn merged elaboration of `dog` = `['go', 'look', 'north']` — it rotates through
  dog's direct neighbours via the structured `SaidTrace` (inhibition-of-return) and stays inside the 2-hop topic
  region (the oracle's 6/6 multi-turn criterion).

## The qualification — exact-pick parity vs the validated function

The shared-bridge dlPFC does NOT always reproduce the dt=0.5 oracle's *exact* per-topic associate (it matches on
`cat`+`river`, differs on `dog`: merged `go` vs oracle `look`). The mechanism is precise and benign:

- Selection is a **rank-order (first-spike latency) code** — the earliest-firing associate wins; direct (1-hop)
  neighbours fire before indirect ones, unrelated concepts never fire.
- At **dt=0.5** (oracle), the finer timestep resolves the four equidistant direct neighbours of `dog` into distinct
  latencies (`look=6, north=6, river=7, go=12`) → `look` wins.
- At **dt=1.0** (merged — the de-risked faithful merge regime), the coarser timestep collapses all four into one
  step (`=23`) — a **4-way exact tie** — and the tie-break (dict order) picks `go`. Both `go` and `look` are
  equally-valid direct elaborations of `dog`; the function ("offer a relevant direct associate") is identical, only
  the tie-break among equidistant candidates differs.

This is why **exact-pick parity is the wrong faithfulness criterion** here: it tests a dt-resolution tie-break that
the dlPFC was never validated on. The dlPFC's published validation (content_selection_spiking line 376) is *"the
earliest-latency pick is a DIRECT neighbour 6/6"* and *"3-turn chains stay within the 2-hop topic region 6/6"* —
both met by the merge. The GATE test (`test_step3_dlpfc_merged_elaborate_matches_separate_path`) asserts that
validated functional criterion (direct-neighbour pick + abstention + determinism + multi-turn topic coherence + the
oracle also selecting from the same direct-neighbour set), not exact-pick identity.

## Biology-translatable insight

Rank-order / latency coding's *resolution* is bounded by the integration timestep. At the binding timestep (dt=1.0)
equidistant associates become temporally indistinguishable (they tie), so fine-ranking among equally-relevant
neighbours is lost — but coarse selection (direct vs indirect vs unrelated) and multi-turn rotation through the
neighbour set are preserved. Working-memory function (the NMDA latch, the spreading-activation selection) is
substrate-shareable at the binding timescale; the *precision* of temporal-code readout trades off against the
timestep. The dlPFC's dialogue-planning job does not require sub-step ranking of equally-relevant continuations, so
the merge is functionally complete; a task that DID require finely ranking equidistant associates would be the case
where a separate finer-dt dlPFC regime earns its cost.

## No regression (the standing gate)

With the NMDA dlPFC slice present on the unified bridge, the parser+composer are unchanged: comprehend→store→recall
→abstain hold, and the composer's FIXED coincidence-bind weights remain byte-identical to their design value
(W_COINC=320) after the parser's global-Hebbian training (the per-population plasticity-gate isolation from step 1,
re-asserted at full scale with the dlPFC slice and its NMDA mask added). The 10 on-brain tests + the unified-bridge
tests stay green.

## Artifacts
- Impl: `research/runners/unified_brain_bridge.py` — `enable_dlpfc`, `_wire_dlpfc`, `_SharedDlpfcContext`, the
  OU-off `elaborate`, and a `hear` convenience method (parse→store, mirroring `BrainConversationalAgent.hear`, so
  the agent runs on the unified bridge unchanged).
- Test: `tests/test_unified_brain_bridge.py::test_step3_dlpfc_merged_elaborate_matches_separate_path` — the GATE
  (validated functional criterion + no-regression).
- Probe: `research/findings/raw/_step3_dlpfc_char_probe.py` — the characterization table above.
- Backend: CuPy / RTX 3090 (spiking dynamics are GPU-bound).

## End-to-end capstone — the full conversation on ONE bridge (production D=2048)

`research/findings/raw/_unified_brain_capstone_demo.py` runs a scripted conversation exercising the FULL unified
API on a single **18,430-neuron** `UnifiedBrainBridge(enable_dlpfc=True)` at the production dimension **D=2048**,
seed 42, the real denoise64 concept codes:

```
COMPREHEND  heard 'dog go north' / 'cat come south' / 'dog look river'  -> roles assigned by the parser
WHO/WHAT    what does dog go? -> north | cat come? -> south | dog look? -> river
            who go north? -> dog | who come south? -> cat | who look river? -> dog          (6/6)
ABSTAIN     what does cat go? -> None | who go river? -> None                       (the no-confab moat)
YES/NO      does dog go north? -> yes | does dog go south? -> unknown               (bound polarity tag)
GENERATE    describe dog -> 'dog go north' | describe cat -> 'cat come south' | describe apple -> None
DIALOGUE    elaborate dog -> go | cat -> come | river -> dog | apple -> None         (dlPFC; abstains)
```

All three conversational regions — parser (comprehension), composer (bind/unbind recall, yes/no, generation),
dlPFC (dialogue planning) — interoperate on ONE interacting SimulationBridge. This is the capability the unit
tests validate, shown end-to-end at the operating dimension.

**Dimensional caveat (consistent with stage-1.5):** the same demo at proj_dim=64 degrades the composer's recall
(what/who mostly None; `describe dog -> 'dog hot look'` confabulated) while the dlPFC stays correct — the
correlated denoise64 codes (cos~0.80) need D=2048 to separate. The composer's recall is dimension-bound; the
dlPFC uses pattern-size assemblies (not the D-dim codes), so it is D-robust. The unified agent operates at
D=2048, as decided in stage 1.5.

## Status
**B step 3 = qualified MERGE → B structurally complete: parser + composer + dlPFC are one interacting bridge.** The
step-1 (shared substrate) + step-2 (comprehension routes composition in spikes via the gated latch) results stand;
step 3 brings the third region onto the same bridge with the dialogue-planning function reproduced and one honest,
characterized dt-resolution nuance.
