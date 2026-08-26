---
type: finding
status: live
date: 2026-08-25
lane: infra
mechanism: da-mode-cupy-backend-interop
runner: research/runners/_neuromod_spiking_da_mode_derisk.py
artifacts:
  - research/findings/raw/da_axis_cupy_interop/repro_before_fix.json
  - research/findings/raw/da_axis_cupy_interop/verify_after_fix_cupy.json
  - research/findings/raw/da_axis_cupy_interop/verify_ensure_restore_cupy.json
  - research/findings/raw/da_axis_cupy_interop/verify_via_brain_chat_endpoint.json
  - research/findings/raw/da_axis_cupy_interop/numpy_unchanged_before_after.json
seed-waiver: single seed (42) is sufficient here — this is a deterministic backend-interop bug fix (an array-module mismatch that either throws or does not; a substrate-state restore that either round-trips or does not), not a statistical generalization claim. Every comparison below is exact (diff / equality), not seed-sensitive.
---

# The DA axis silently died on the production `/api/brain-chat` cupy path every turn — FIXED (device-correct array module, both backends verified)

**2026-08-25.** On `SIM_BACKEND=cupy` (the owner's real production path), the entire board-#76 spiking
dopamine-mode faculty wired into `/api/brain-chat` (`webapp/da_mode_drives_chat.py`: the DA-mode suffix,
DA-gated encoding, DA-gated curiosity) errored on **every single turn** and was **silently swallowed** by a
bare `except Exception`. `da_drives.reason` read `"error:ValueError: non-scalar numpy.ndarray cannot be used
for fill"`. Three default-ON faculties were **inert on the one backend the owner actually runs** — a textbook
instance of the silent-failure class this task exists to close: a "done" faculty (wired, on-by-default,
board-#76 6/6-seed GO) that does nothing on the path that matters.

## Root cause — two bugs, one shared cause: hardcoded `numpy` in a backend-agnostic substrate

Board #76's `_neuromod_spiking_da_mode_derisk.py` was written and GO'd entirely under
`SIM_BACKEND=numpy` (`os.environ.setdefault("SIM_BACKEND", "numpy")`, line 68) and never exercised on cupy.
Two of its functions hardcoded `np` regardless of what backend the substrate `sb` (built by
`_perturb_and_measure_derisk.build`, which DOES respect `SIM_BACKEND`) actually used:

1. **`make_manager(sb)`** called `mgr.initialize(sb.core_config.num_neurons, np)` — the neuromodulator
   manager's own array module (`self._cp` in `sim/neuromodulators.py`) was hardcoded numpy, so
   `compute_excitability_drive_per_neuron()` always returned a plain numpy array regardless of `sb`'s backend.
2. **`_base_current`** built its static input template with `np.zeros`/`np.asarray`.
3. **`measure_self_driven`**'s live loop then did `sb.cp_external_input_current[:] = base` (and
   `base + np.asarray(drive, ...)`) every step. On a cupy-backed substrate (`SIM_BACKEND=cupy`),
   `sb.cp_external_input_current` is a **cupy** array; assigning a numpy array into a `[:]` slice of it raises
   cupy's own fill-path error.

**Confirmed the exact mechanism directly** (not inferred from reading the code — `docs/TERMS.md` "byte-identical"
discipline applied to the throw site too):

```
>>> cupy.zeros(5)[:] = numpy.ones(5)
ValueError: non-scalar numpy.ndarray cannot be used for fill
```

**Exact throw site, confirmed via full uncaught traceback** (repro below):
`research/runners/_neuromod_spiking_da_mode_derisk.py:164`, inside `measure_self_driven`, at
`sb.cp_external_input_current[:] = base` — reached on the *very first* simulation step of *every* DA read (the
DA concentration starts exactly at baseline, so `compute_excitability_drive_per_neuron()` returns `None` on
step 0 and the `drive is None` branch fires first), called from
`webapp/da_mode_drives_chat.py:309` (`DaModeDrivesWorkspace._read_da_level` →
`_DA.measure_self_driven(...)`), itself called from `observe()`'s bare `except Exception as e: info["reason"]
= f"error:{type(e).__name__}: {e}"` — which is exactly what swallowed it in production.

**A SECOND, independently-confirmed latent bug, named in the task and verified separately**: `_ensure()`'s
post-build state snapshot (`webapp/da_mode_drives_chat.py`) filtered `cp_*` attributes with
`isinstance(x, np.ndarray)`. On a cupy substrate every `cp_*` attribute is a **cupy** array, so this filter is
false for all of them — the snapshot dict came back **empty** (measured: 0 entries, was 33 after the fix), and
`_restore()`'s `for k, v in self._snapshot.items(): ...[:] = v` loop was a **silent no-op**. This does not
crash (nothing throws), so it would have stayed invisible even after fixing bug #1: every DA read on cupy
would silently violate the module's own documented contract ("every read is a deterministic function of THIS
turn's afferent, history-independent") by carrying over the substrate's dynamic state (membrane potentials,
adaptation variables, …) turn to turn.

Both bugs share one cause: code written and validated only against `SIM_BACKEND=numpy`, reused unchanged
against a cupy substrate. This is the SAME class of bug as the 2026-06-24 `brain_conversational_agent._bridge_xp`
webapp fix (same exact exception string, same root cause — a numpy array assigned into a cupy `[:]` slice) and
the 2026-07-03 EMERGE-70 finding (`sim.bridge` binds its module-level `cp` once at import; a caller that
assumes `SIM_BACKEND` or `get_backend()` reflects the substrate's actual backend is wrong the moment anything
else in the process has touched the global cache).

## The fix (Option A — device-correct, minimal, no `sim/` edit)

Both files gained a helper that derives the array module from the substrate's OWN array, not from
`SIM_BACKEND`/`get_backend()` (a process-global sticky cache `sim.bridge` does not itself consult after
import) — the exact pattern already established in `research/runners/brain_conversational_agent.py._bridge_xp`:

```python
def _bridge_xp(sb):
    try:
        import cupy as _cp
        return _cp.get_array_module(sb.cp_external_input_current)
    except Exception:
        return np
```

- `research/runners/_neuromod_spiking_da_mode_derisk.py`: `make_manager` now calls
  `mgr.initialize(sb.core_config.num_neurons, _bridge_xp(sb))` instead of hardcoded `np`; `_base_current`
  builds its template on `_bridge_xp(sb)`; `measure_self_driven`'s per-step assignment casts `drive` through
  `xp.asarray(drive, dtype=xp.float64)` instead of `np.asarray`. The rates/SNc-firing accumulation half
  (already correctly using `to_host()` to bring `cp_firing_states` to numpy before accumulating) is untouched.
- `webapp/da_mode_drives_chat.py`: added `_is_ndarray(x)` (True for numpy OR cupy ndarrays) and changed
  `_ensure()`'s snapshot comprehension to `{k: getattr(sb, k).copy() for k in dir(sb) if k.startswith("cp_")
  and _is_ndarray(getattr(sb, k, None))}` — `.copy()` (available on both array types) keeps each snapshot
  entry on its OWN backend, so `_restore()`'s same-device `[:] = v` assignment is unchanged and correct on
  either backend.

**Why Option A over Option B (force this isolated read to numpy):** infeasible, not just non-minimal.
`sim/bridge.py` binds its module-level `cp` **once, at import time** of `sim.bridge` — not per-call. In the
production webapp process, `sim.bridge` is imported once under `SIM_BACKEND=cupy` and its `cp` name is cupy for
the rest of the process's life; flipping the `SIM_BACKEND` env var later does **nothing** to `sim.bridge`'s
already-bound `cp`, so any substrate `PM.build()` constructs would still be cupy-backed regardless. Forcing
numpy would additionally require mutating the process-global `sim.backend` cache (shared across every
concurrent chat session in a multi-threaded server) around each read — a real race-condition risk this DA
workspace's own `threading.Lock` cannot cover, since the lock is per-workspace, not process-wide. Option A
touches nothing global.

## VERIFY

### 1. Reproduced on cupy — full traceback captured (pre-fix)

`research/findings/raw/da_axis_cupy_interop/repro_cupy_bug.py` calls `DaModeDrivesWorkspace.observe(message)`
both (1) as production runs it (exception swallowed into `reason`) and (2) uncaught, for the real traceback.
Run on `SIM_BACKEND=cupy` against the pre-fix code (`git stash` of the fix commit):

```
info = {'acted': False, ..., 'da_level': 0.0, 'mode': 'rest',
        'reason': 'error:ValueError: non-scalar numpy.ndarray cannot be used for fill', 'seed': 42}
CONFIRMED: da_drives.reason = 'error:ValueError: non-scalar numpy.ndarray cannot be used for fill'
```
Uncaught traceback (`research/findings/raw/da_axis_cupy_interop/repro_before_fix_cupy_traceback.log`) bottoms
out at `research/runners/_neuromod_spiking_da_mode_derisk.py:164, in measure_self_driven:
sb.cp_external_input_current[:] = base` → `cupy/_core/core.pyx:1002 _ndarray_base.fill` →
`ValueError: non-scalar numpy.ndarray cannot be used for fill` — matches the diagnostic-observed reason
**verbatim**. Artifact: `research/findings/raw/da_axis_cupy_interop/repro_before_fix.json`.

### 2(a) — `observe_turn` now returns `acted:True`, no error, on cupy

`research/findings/raw/da_axis_cupy_interop/verify_fix_cupy.py`, post-fix, `SIM_BACKEND=cupy`:

```
{'acted': True, 'da_level': 0.8773571795323781, 'snc_firing': 0.13833333333333322, 'mode': 'focus',
 'lead': ' — worth going further here.', 'reason': 'engaged', 'seed': 42}
```

### 2(b) — da_level VARIES with engagement (load-bearing on input)

Two fresh workspaces (seed 42, avoiding EMA cross-turn confound), one empty message, one rich/novel message:

| turn | message | da_level | mode |
|---|---|---|---|
| low-engagement | `""` | 0.04616293556102311 | rest |
| high-engagement | "tell me something surprising and unusual about deep sea bioluminescent creatures and their behavior" | 0.8965045558416256 | focus |

`da_level` rises **0.04616293556102311 → 0.8965045558416256** (rounds to 0.046 -> 0.897, delta 0.850) <!--derived--> and the mode crosses rest → focus with richer/novel input. Artifact: `verify_after_fix_cupy.json`.

### 2(c) — the LESION collapses da_level to the floor regardless of engagement

Same rich/novel message as 2(b)'s high-engagement turn, `BRAIN_DA_DRIVES_LESION=1` (silences the spiking SNc
nucleus, the #76 anti-cheat-2 lesion):

```
{'acted': True, 'lesioned': True, 'da_level': 0.04616293556102311, 'mode': 'rest',
 'lead': '', 'reason': 'lesion_collapsed'}
```

`da_level` collapses back to **0.04616293556102311 — the exact same floor value as the unrelated
low-engagement turn in 2(b)** — on the *identical* rich/novel message that alone produced 0.8965045558416256
unlesioned. The engagement suffix vanishes (`lead == ''`). This is the brain-based load-bearing proof:
silencing the SNc nucleus severs the coupling even though the world input (an engaging message) is unchanged.

### 2(d) bonus — through the real `/api/brain-chat` handler (in-process, no HTTP layer)

`research/findings/raw/da_axis_cupy_interop/verify_via_brain_chat_endpoint.py` calls
`webapp.server.brain_chat(BrainChatRequest(...))` directly (the same technique
`tests/test_production_chat_gpu_smoke.py` uses), fresh `SIM_BACKEND=cupy` subprocess,
`renderer="stub"` (GPU-free mouth, avoids the ~58s Qwen load). The `research/findings/2026-08-25-integrated-
conversational-state-diagnostic.md` TestClient harness named in the task does not exist on this branch (likely
still in flight on another agent's branch), so this is a self-built equivalent:

- Turn 1 (empty message): HTTP 200, `da_drives` absent (an early abstain path short-circuits before the DA
  block on an empty message — unrelated to this fix, confirmed separately in 2(b)/2(c) at the workspace level).
- Turn 2 (rich/novel message): HTTP 200, `da_drives.reason == 'engaged'` (no `error:`), **and the live answer
  text itself carries the DA-mode engagement suffix**:
  `"I don't know about that. My curiosity is piqued — I haven't learned about surprising yet: what can you
  tell me about surprising? — worth going further here."` — the trailing `— worth going further here.` is
  `da_mode_suffix("focus")`, appended by the exact production code path (`webapp/server.py:5130-5131`) the
  owner's real turns go through. Artifact: `verify_via_brain_chat_endpoint.json`.

### 2(e) — the second latent bug (`_ensure`/`_restore` no-op) independently confirmed fixed

`research/findings/raw/da_axis_cupy_interop/verify_ensure_restore_cupy.py`: pre-fix, the post-build snapshot
dict size on cupy is **0** (asserted, raises). Post-fix, it is **33** entries, and three consecutive reads at
the identical afferent (`afferent_override=900.0`) on the SAME persistent workspace are byte-identical
(`da_level=0.9002284628483648`, `snc_firing=0.14250000000000013`, all three reads) — confirming `_restore()`
now correctly resets the cupy substrate's dynamic state each turn (history-independence, as documented).
Artifact: `verify_ensure_restore_cupy.json`.

### 3 — numpy path unchanged (board-#76 6/6-seed-GO regression guard)

`research/findings/raw/da_axis_cupy_interop/numpy_unchanged_check.py` run via `git stash` / `git stash pop`
across the fix commit (identical script, identical seed=42, three fixed afferents spanning rest/focus/arousal
+ one direct runner-level call) — stdout **diff'd byte-for-byte, zero differences**:

| afferent (pA) | da_level (before == after) | mode |
|---|---|---|
| 0.0 | 0.04616293556102311 | rest |
| 800.0 | 0.8801292495026654 | focus |
| 1300.0 | 1.2386777669861797 | arousal |

Artifact: `numpy_unchanged_before_after.json`. The numpy path is unaffected because `_bridge_xp(sb)` resolves
to `numpy` on a numpy-backed substrate (the `cupy.get_array_module` call only distinguishes cupy vs numpy
arrays; on a numpy `sb` every subsequent `xp.*` call is the exact call the old hardcoded-`np` code made).

## Verdict

The root cause was exactly as traced: two functions in the board-#76 runner hardcoded `np` against a
substrate whose actual backend depends on `SIM_BACKEND`, so on the production cupy path every DA read threw
`ValueError: non-scalar numpy.ndarray cannot be used for fill` on its first step and was silently swallowed.
A second, non-crashing bug in the webapp's own state-snapshot filter meant that even fixing bug #1 alone would
have left the per-turn read silently history-dependent on cupy. **Both are fixed** by deriving the array
module from the substrate's own arrays (`cupy.get_array_module`, the same pattern already used in
`brain_conversational_agent.py`) rather than from the global `SIM_BACKEND` knob, which `sim.bridge` does not
itself re-consult after import. Verified: the production entry point (`observe_turn`) now runs error-free on
cupy, `da_level` is load-bearing on message engagement (0.04616293556102311 → 0.8965045558416256, i.e. rest
to focus) <!--derived-->, the SNc lesion collapses it back to the exact same floor, the engagement
suffix now visibly appears in real `/api/brain-chat` answers on the owner's actual backend, cross-turn
history-independence holds (33 snapshot entries, byte-identical repeated reads), and the numpy path — board
#76's own 6/6-seed GO — is unaffected (byte-identical, diffed).

## Files
- `research/runners/_neuromod_spiking_da_mode_derisk.py` — `_bridge_xp(sb)` helper added; `make_manager`,
  `_base_current`, `measure_self_driven` now device-correct.
- `webapp/da_mode_drives_chat.py` — `_is_ndarray(x)` helper added; `_ensure()`'s snapshot filter now captures
  cupy arrays too.
- `research/findings/raw/da_axis_cupy_interop/` — repro script + traceback log, 3 verify scripts, 5 JSON
  result artifacts (this finding's evidence).
