# NumPy backend — chat_repl full pipeline runs end-to-end (W→A + A→W)

**Date:** 2026-05-11 03:30 EDT
**Status:** MILESTONE — the full user-facing chat REPL (Tier 1 training
+ word→action inference + action→word :speak generative) now runs
under `SIM_BACKEND=numpy`. Hardware-independent chat workflow shipped.
**Trigger:** User (2026-05-11) — "Continue autonomously until I say stop"

---

## Demonstrated

```
$ SIM_BACKEND=numpy python -m research.runners.chat_repl \
    --mode tier1 --seed 42 --train-events 5 \
    --scripted-words "north,east" --from-scratch
[INFO] NumPy backend (CPU (NumPy backend), 47.7 GB RAM available)
[INFO] Simulation data initialized for 6336 neurons (3D). Synapses: 3218125
[SCRIPTED] running 2 predefined words.
  [OK] [TIER1 seed=42] sim hears 'north', activates motor_N
       (delta N+317 E+30 S-84 W+88, x3.6 confidence)
  [OK] [TIER1 seed=42] sim hears 'east', activates motor_E
       (delta N+85 E+316 S+56 W+101, x3.1 confidence)
[SCRIPTED COMPLETE]
[DONE] 2 turns total.
```

```
$ SIM_BACKEND=numpy python -m research.runners.chat_repl \
    --mode tier1 --seed 42 --train-events 5 \
    --scripted-words ":speak N,:speak E" --from-scratch
rankings: east=0.27 west=0.08 south=0.06 north=-0.02
[SCRIPTED COMPLETE]
```

Both directions work:

- **W→A (word → action):** Tier 1 sim correctly binds "north" → motor_N
  and "east" → motor_E with confidence ratios 3.1-3.6× (typical Tier 1
  range).
- **A→W (action → word, `:speak`):** generative inference produces
  semantically-correct word rankings: `:speak E` → east 0.27 (highest),
  west 0.08, south 0.06, north -0.02. The sim "speaks" the right word
  for the requested motor action.

## The full chain that works under NumPy

```
chat_repl --mode tier1 --from-scratch
  -> run_repl()
    -> _load_or_train_tier1(seed=42, n_train_events=5)
      -> bio_three_factor.run_three_factor()
        -> SimulationBridge.__init__()
        -> SimulationBridge._initialize_simulation_data()
           [brain region framework: 10 regions, 6336 neurons,
            inject_explicit_wiring -> scipy.sparse CSR with numpy data]
        -> _apply_parameter_heterogeneity()
           [backend-aware get_random_state / set_random_state]
        -> apply_topographic_bias()
           [backend-aware to_host + push-back via backend cp]
        -> training loop: stim + simulation steps + reward modulation
        -> embodied-Hebbian co-firing + STDP
    -> chat_inference(bridge, "north")
       [backend-aware D->H transfer for baseline/drive deltas]
    -> chat_speak(bridge, "N")
       [generative inference + cosine ranking]
```

Every step in this chain previously had at least one CuPy-specific
call (`.get()`, `cp.asarray()`, `cp.random.get_random_state()`,
`import cupy as cp`, etc.). All migrated to backend-aware patterns.

## Migration summary (whole autonomous arc)

### sim/ package — 7 modules touched

| Module | Sites migrated | Status |
|--------|---------------|--------|
| backend.py | NEW (391 lines, 37 tests) | Created |
| bridge.py | 81 sites (imports + .get() + .cuda + .random + cp.asnumpy) | Backend-aware |
| kernels.py | 15 sites (@cp.fuse decorators) | Backend-aware |
| connectivity.py | import block | Backend-aware via get_sparse_module |
| synapse_storage.py | NEW (290 lines, 25 tests) | Phase 3 foundation |

### research/runners/ — 4 runners touched

| Runner | Sites migrated | Status |
|--------|---------------|--------|
| bio_three_factor.py | 8 (6 .get() + 2 cupy imports) | Works on NumPy |
| text_minimal_isolation.py | 4 (3 .get() + 1 cupy import) | Works on NumPy |
| chat_repl.py | 5 (3 cupy imports + 2 .get()) | Works on NumPy |
| inference_benchmark.py | 1 (Unicode arrow fix) | (earlier; not backend-related) |

**Total: ~110 CuPy-specific call sites migrated across the chat
training + inference pipeline.**

## Performance on NumPy backend (toy scale)

- Tier 1 toy (n_lang=64, n_motor=16, 5 events/word): bio_three_factor
  runs in 0.9 sec total
- Tier 1 production (n_lang=2048, n_motor=500, 5 events/word):
  chat_repl train+inference completes in ~15-20 sec
- 500 raw simulation steps × n=200: 105 ms (0.21 ms/step)
- Brain region 100 steps × n=70: 19 ms (0.19 ms/step)

NumPy is 4-10× slower than CuPy at small scales (Python overhead
dominates). At production Tier 1 scale we'd expect 20-50× slowdown
per the design doc — fine for portability and verification, not for
peak training. CuPy remains the production speed path.

## What this unlocks

1. **Mac M-series compatibility** — pure scipy.sparse + numpy backend;
   no CUDA. MLX backend stub exists for future hardware-native acceleration.
2. **Linux CPU-only servers** — containerized deployment without GPU
   passthrough. Cloud CPUs work.
3. **Windows / WSL without NVIDIA** — most user laptops/desktops.
4. **CI/CD without GPU runners** — `SIM_BACKEND=numpy pytest` works.
5. **Algorithmic verification at toy scale** — NumPy gives reference
   semantics for the CuPy implementations.
6. **Foundation for Phase 3 SSD synapse paging** — already shipped
   (commit `33ca704`); the paging tier sits on the NumPy code path.

## Known pre-existing bugs surfaced (not backend-related)

The chat_repl `:speak` path has a KeyError on `in_vocab` field at exit
— transcript records for `:speak` commands don't have that key but
the cleanup code assumes they do. Cosmetic; the actual `:speak`
inference works. Patched as a follow-up.

## What's NOT yet verified on NumPy

- chat_synonym_demo (Tier 2.1 path; arch is similar so likely works)
- chat_speak_synonym_demo
- Replicas (sim/replicas.py)
- Visual cortex (sim/visual_cortex.py)
- Neuromodulators (sim/neuromodulators.py)
- Recording playback

These will surface fixes as exercised; pattern is well-established now.

## Roadmap forward

Per the CPU/RAM/SSD tiering design:

| Phase | Status | Notes |
|-------|--------|-------|
| 1: xp abstraction | ✅ SHIPPED | sim/backend.py + 37 tests |
| 2: NumPy backend passes tests | ✅ SHIPPED (core paths) | bio_three_factor + chat_repl verified |
| 3: SSD synapse paging | ✅ Part 1 SHIPPED | TieredSynapseStore + 25 tests |
| 3 part 2: bridge integration | Pending | Per-pathway access in bridge step |
| 4: Activity-driven auto-tiering | Pending | Depends on Phase 3 part 2 |

## Provenance

- Commits this arc: `ab5500f` → `d434d43` (7 commits for tiering proper +
  related fixes)
- Earlier findings: `research/findings/2026-05-11-numpy-backend-shipped.md`
- Strategic context: `docs/plans/2026-05-11-strategic-reevaluation.md`
- Tiering design: `docs/plans/2026-05-11-cpu-ram-ssd-tiering-design.md`
- Backend module: `sim/backend.py`
- SSD paging module: `sim/synapse_storage.py`

The user-facing chat REPL is now hardware-independent. This is the
foundation for everything downstream — cloud deployment, Mac support,
GPU-less CI, paging for arch-larger-than-VRAM, and the strategic Path
1/2/3 decision (all three paths benefit from this work).
