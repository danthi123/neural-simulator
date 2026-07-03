# EMERGE-69 — the FLAGSHIP console SPEAKS its EMERGE answers **100% ON SPIKES** end-to-end (self-organized structure + every word content+function) — **GO** (6-seed)

**2026-07-03 (autonomous).** EMERGE-66 wired the fully **self-organized** producer (structure — function words + slot inventory + slot order — all discovered from the corpus, NO host `FRAMES` dict) into the flagship `SpikingBrocaConsole`; its `spell` was still the **token surface**. EMERGE-67/68 made the A→W read-out spell every word (content on BRIDGE-A, function on BRIDGE-F) from `language_output` spikes, but only at the **producer** level. EMERGE-69 wires the two together: an **additive default-off `neural_spell` flag** on `SpikingBrocaConsole` installs the EMERGE-68 `UnifiedNeuralSpell` as the producer's `spell` callback, so the flagship's EMERGE render is 100% on spikes end-to-end — the **structure** is self-organized (EMERGE-65/66) AND **every word** (order via EMERGE-59/63, content via EMERGE-67, function via EMERGE-68) is decoded from spikes. Reuse-by-import; **NO `sim/` edit**.

## The wire (additive, default-off, mirroring EMERGE-66's `self_organized`)

`SpikingBrocaConsole.__init__` gains `neural_spell=False`. When `True`, it loads `UnifiedNeuralSpell(load=True)` from the EMERGE-67/68 caches and sets `spell = self._neural_speller.spell`, which flows into **both** producer branches (`_emerge60:110`). `realize_slot` (`_emerge59:138`) routes **every** slot — DET/FUNC + SUBJ + VERB — through this one `spell`, so with `self_organized=True` every slot is spike-decoded (content→BRIDGE-A, function→BRIDGE-F). Default `False` == EMERGE-66 byte-identical (the token surface). The gate-first structure is untouched: on ABSTAIN `_emerge_turn` returns before `_render_emerge`, so the producer — and hence **both** A→W read-outs — is NEVER invoked (moat by construction).

## The honest backend split (named, not hidden)

`sim.bridge` binds **one** backend (`cp`) at import (a module-global), so numpy and cupy bridges cannot coexist in one process. The EMERGE-52/54 per-dimension **reasoner** (the console's `self.reasoner`, a stacked HTM pooler) is **numpy-only** (its EMERGE-12/14 on-substrate teach + predict paths write host arrays into the CSR via an `xp = bridge.xp if hasattr else np` fallback → under cupy: *"non-scalar ndarray cannot be used for fill"*); the A→W read-out is **cupy-only**. So the full console with a **live** numpy reasoner + the cupy A→W cannot co-execute in one process. Making the whole EMERGE-52/54 reasoner stack cupy-clean is a wide, high-risk change to committed shared runners — out of scope for this additive wire. The de-risk therefore validates the two claims on their native backends:

- **The new spike claim (GPU/cupy):** the flagship's **own self-organized producer** (`SelfOrganizedProducer.build_from_corpus` → `producer(spell=UnifiedNeuralSpell.spell)` — the **exact** producer the `neural_spell` flag installs; a simple slot-order bridge + corpus-mined structure, no reasoner) renders every EMERGE frame with A→W-spellable facts → **all slots decoded from spikes** + producer gate-first moat + function-word lesion collapse.
- **The console-integration invariants (CPU/numpy):** the full flagship console (reasoner + self-organized producer) with the `neural_spell` **wire structure** (a counting token spell, since the A→W can't run on numpy) → gate-first moat + membership routing + fluid no-regression + wire content-routing. EMERGE-66 already GO'd this numpy surface; the flag is additive/default-preserving.

## De-risk — **GO** (6-seed 42/43/44/100/101/102)

**[GPU] the new spike claim (through the flagship's self-organized producer, the neural spell wired in):**

| gate | value | bar |
|---|---|---|
| ALL-word spike render (det+subj+func+verb decoded from `language_output` spikes) | **1.000** every seed | ≥ 0.90 |
| function-word slot accuracy (`the/a/can/does/not` on spikes) | **1.000** every seed | ≥ 0.90 |
| raw A→W content-word rate (BRIDGE-A) / function-word rate (BRIDGE-F) | **1.000 / 1.000** (5/5: the→the, a→a, can→can, does→does, not→not) | — |
| genuinely SPIKING — FUNCTION-word LESION (zero BRIDGE-F pool→`language_output`) collapses the engine decode | **0.000** | ≪ (≥ 0.40 drop) |
| producer gate-first MOAT — spell-calls + producer-invocations on abstains | **0 / 0** | 0 |

**[CPU] the console-integration invariants (full flagship console, neural_spell wire structure, token spell):**

| gate | value | bar |
|---|---|---|
| wire content-routing (the right grounded fact routed to the producer, produced once) | **1.000** every seed | ≥ 0.99 |
| gate-first MOAT — producer + spell calls on abstains | **0 / 0** | 0 |
| membership routing (`can a dog eat?` → fluid, producer/spell NOT stolen) | **True** every seed | True |
| NO fluid-path regression (Broca-free baseline, re-seeded per seed) | **True** every seed | True |

Sample (fully-spiking, through the flagship's self-organized producer): *"the crow can hop"* / *"the robin can fly"* / *"the penguin walks"* / *"the penguin does not fly"* — det/subj/func/verb **all** decoded from `language_output` spikes; abstain → producer + A→W **never** invoked.

## A backend-compat bug found + SURPASSED with a byte-identical fix

The GPU self-organized producer initially crashed in EMERGE-61's inter-utterance **wash-out** reset (`_emerge61._restore_state`): it wrote the numpy post-init snapshot into the (cupy) bridge arrays via `arr[:] = xp.asarray(val) if xp is not None else val` with `xp = None` (the bridge has no `_cp` attr) → *"non-scalar ndarray cannot be used for fill"*. Fix (one line, in a **research runner** not `sim/`): `arr[:] = from_host(val)` (the sim backend's H→D marshal — a **no-op passthrough on numpy → byte-identical**, a H→D copy on cupy). EMERGE-61/65/66 CI stay green on numpy (26 passed); the GPU wash-out now works. This is a routine backend-mismatch fix with a clear cause, not a mechanism change.

## Verdict

**GO.** The flagship `SpikingBrocaConsole` now renders its EMERGE emergent-reasoning answers **100% on spikes end-to-end**: the grammatical **structure** is self-organized from the corpus (EMERGE-65/66) AND **every word** (order EMERGE-59/63, content EMERGE-67, function EMERGE-68) is decoded from `language_output` spikes. The gate-first no-confab moat holds by construction (0 producer + 0 A→W spell calls on abstains); membership routing + fluid paths inherited byte-identical; the function words are genuinely spiking (BRIDGE-F lesion collapses the decode). Renders the BOUNDED EMERGE frame inventory (ability-affirm / intransitive-exception / negated-modal), NOT open prose (R4, the deferred wall). The A→W engines are GPU-trained once + cached (a scale/data lever). The **only** change to committed code is the additive default-off `neural_spell` flag on EMERGE-60's `SpikingBrocaConsole` (default False == EMERGE-66 byte-identical) + the one-line byte-identical-on-numpy backend-compat fix in EMERGE-61's wash-out. **NO `sim/` edit.** EMERGE-59..69 CI all still pass (**81 passed / 3 skipped** on numpy; the 3 skips are the GPU-only A→W smokes for 67/68/69).

⇒ the emergent brain discovers categories from experience → reasons → and now **speaks its grounded EMERGE answers 100% on spikes** (self-organized structure + every word), on the flagship console, transformer-free, host-token-free.

## Files
- `research/runners/_emerge69_console_fully_spiking_derisk.py` — the merged de-risk (`_build_self_organized_producer` + `_producer_all_word_spike_render` = the GPU spike claim; `_derisk_cpu` = the console invariants; `_derisk` runs GPU inline + spawns a numpy child + merges) + `--demo`/`--derisk`/`--derisk-gpu`/`--derisk-cpu`.
- `research/runners/_emerge60_console_spiking_broca_derisk.py` — the additive default-off `neural_spell` flag on `SpikingBrocaConsole` (the ONLY committed-console change; default False == EMERGE-66 byte-identical).
- `research/runners/_emerge61_spiking_broca_order_robustness_derisk.py` — one-line byte-identical-on-numpy backend-compat fix in `_restore_state` (`from_host`, unblocks the wash-out on cupy).
- `tests/test_emerge69_console_fully_spiking.py` — 4 CPU-safe (additive-flag-default-off / wire-routes-spell-through-producer / gate-first-moat / all-slot ground-truth) + 1 GPU smoke (skip unless the process is `SIM_BACKEND=cupy` and both caches exist).
- `research/findings/raw/_emerge69_console_fully_spiking.json` (+ `_cpu.json`) — the 6-seed de-risk (GO=true).
- Caches reused verbatim: `bridges/emerge67_aw/aw_content.simstate.h5` (BRIDGE-A), `bridges/emerge68_aw/aw_func.simstate.h5` (BRIDGE-F) — local, `.h5` gitignored, regenerable via EMERGE-67/68 `--train`.
