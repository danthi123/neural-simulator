---
type: finding
status: live
mechanism: gnw-stop-trigger-accbg-circuit production-flip verify (rank-12 GNW STOP host boolean-OR)
lane: laneC
date: 2026-09-05
integration_faculty: gnw-global-stop
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_gnw_stop_trigger_production_flip_verify.json
  - research/findings/raw/_gnw_global_stop_flip_soak.json
runner: research/runners/_gnw_stop_trigger_production_flip_verify.py
---

# GNW STOP-trigger ACC/BG circuit — PRODUCTION-FLIP verified GO: `BRAIN_GNW_STOP_TRIGGER_SPIKING` runs DEFAULT-ON, 6/6 seeds, two real bugs found along the way and an adversarial check on which one actually mattered

**Verdict: GO (6/6 seeds).** The rank-12 de-risk
([`2026-09-05-gnw-stop-trigger-accbg-circuit-derisk-GO.md`](2026-09-05-gnw-stop-trigger-accbg-circuit-derisk-GO.md))
and its production-dispatch hook-verify were already 6/6 GO with the flag wired but DEFAULT-OFF. This finding does
the flip-specific work those two did not: it flips `stop_trigger_spiking_enabled()`'s own default to ON, re-verifies
NO REGRESSION and LOAD-BEARING-NOT-HOLLOW through the real production entry points on 6 seeds, and — in the course
of that verification — found two genuine bugs the de-risk's own scope never had reason to exercise, then ran an
adversarial A/B check on its OWN first hypothesis before writing this up: only ONE of the two bugs (Bug A, a stale
test fixture, confirmed production-unreachable) turned out to be the actual, sole, complete cause of the observed
regression signal; the other (Bug B, a real production RNG-isolation gap) was found first, fixed, kept on its own
merits, and then PROVEN — by reverting it while keeping Bug A fixed — to not have been load-bearing for that
signal at all. Both are documented in full below because getting the causal story right, not just landing on a
green run, is the deliverable.

## What changed

`webapp/gnw_acc_bg_stop_trigger.py::stop_trigger_spiking_enabled()` now mirrors `gnw_global_stop.stop_enabled()`'s
own default-ON style exactly: unset → `True` (delegate to the spiking ACC/BG circuit); an explicit falsy
(`0`/`false`/`off`/`no`/`''`) → `False` (the original host boolean-OR, byte-identical opt-out). No other file's
*behavior* changes — `webapp/gnw_global_stop.py::detect_trigger` already had its additive branch from the de-risk
wire-in; this commit only touches its RNG isolation (see Bug B).

## GATE 1 — NO REGRESSION (6/6 seeds)

| check | what it proves | result |
|---|---|---|
| bare-unset resolves `True`, asserted in the data | the flip is real, not aspirational | 6/6 |
| explicit `=0` still byte-identical to the frozen original host boolean-OR | the opt-out escape hatch survives the flip | 6/6 |
| bare-unset byte-identical to explicit `=1` on real organ-sourced afferents | the new default IS the audited ON path | 6/6 |
| the SIBLING `gnw-global-stop` STOP-*clear* flip-soak (default-ON since 2026-08-26), re-run verbatim with this flip's default now live underneath it | the already-shipped, unrelated STOP-clear mechanism is unharmed | 6/6 GO (see `research/findings/raw/_gnw_global_stop_flip_soak.json` — content BYTE-IDENTICAL to the pre-flip 2026-08-26 artifact once Bug A below was fixed: same n_pre/n_post/boost on every seed, same verdict) |
| cross-faculty regression battery (`onebrain_regression_battery.run_regression_battery`, reused verbatim — the REAL `/api/brain-chat` handler on the GPU-free `tiny-demo` brain, ~26-38 other default-ON faculties' DECISION fields compared ON-vs-OFF) | "every other faculty stays alive" on an actual battery, not this mechanism's own fixtures | **launched, not gating this verdict** — see note below |

**Battery status note.** This machine was running several other Track-1 verification agents concurrently (composer-bundle, integrated-loop, shared-salience, metacog, appraisal — all visible in `ps` at the time), and the battery's ON-arm worker alone was still mid-flight past 12 minutes of heavy multi-tens-of-thousands-of-neuron organ builds (comprehension/xedge) when this finding was written; it was NOT killed and left running (its runner writes its own artifact under `_gnw_stop_trigger_flip_battery` in the usual raw-findings directory when it finishes), but its
completion is not required for this GO: GATE 1's other four checks already establish no-regression through this
mechanism's own production entry points and its directly-coupled sibling organ, and NONE of the ~26-38 battery
probe turns' faculties read `chat._last_gnw_stop`, `gnw_global_stop`, or either `BRAIN_GNW_STOP_TRIGGER_*` flag —
there is no code path by which this flip could reach them. If the battery result later disagrees with that
reasoning, that is new information and should be treated as such, not retrofitted into this verdict.

## GATE 2 — LOAD-BEARING, NOT HOLLOW (the crux; 6/6 seeds)

The existing hook-verify already proved the internal `detect_trigger` boolean is load-bearing. This finding extends
that proof one layer further OUT, to the actual observable a `/api/brain-chat` caller sees: `gnw_global_stop.observe_turn`'s
own `acted` field, which is exactly the boolean that gates whether `resp["gnw_stop"]` is attached to a real JSON
response (`webapp/server.py`, both the rich and single-fact paths). Real, organ-sourced afferents
(`_gnw_acc_bg_stop_trigger_derisk.get_real_n_ignited`/`get_real_mm_peak`) were swept through the REAL production
entry point at the shipped default:

| axis | intact (varies OFF→ON) | afferent→ACC LESIONED (`BRAIN_GNW_STOP_TRIGGER_LESION=1`) | seeds |
|---|---|---|---|
| n_ignited alone (1→4, mm_peak held at real match) | `acted` flips False→True | `acted` NEVER True across the same sweep | 6/6 |
| mm_peak alone (match→mismatch, n_ignited held at real solo) | `acted` flips False→True | `acted` NEVER True across the same sweep | 6/6 |
| `n_hollow` (lesioned calls where `acted` fired anyway) | — | **0/12 on every seed** | 6/6 |
| attribution of the intact-vs-lesioned key-attachment rate to the afferent→ACC pathway | — | **1.0 on every seed** (`tools.lab.attributable_to`) | 6/6 |

`n_hollow==0` is measured directly (the count of lesioned-arm calls where the surface key attached anyway), not
inferred — the anti-hollow bar this campaign names ("byte-identical whether-varied-or-not = HOLLOW") is the
observable itself, at 6/6 seeds.

## Bug A (test-only, fixed) — THE actual cause: a stale swap-only fixture in the sibling flip-soak, unreachable in production

**Found by:** re-running the sibling `_gnw_global_stop_flip_soak.py` with the flag flipped ON (GATE 1's own
no-regression check) — at seed 42 (then confirmed on all 6) the COUPLING sub-check's `swap_only` call
(`_chat(swapped=True, topic="weather")`) never triggered under the spiking path, so `lead_present_on_stop` read
False even though the delib-conflict call succeeded. Tracing it down: this fixture sets `swapped=True` with NO
`mm_peak` key at all — harmless while `detect_trigger` was a host `n_ignited>=2 or swapped` (which never reads
`mm_peak`), but exposed once the spiking circuit reads `mm_peak` as its OWN synaptic afferent: an absent key reads
as `mm_peak=0.0`, and `detect_trigger_spiking`'s "nothing to read" bail-out (`n_ignited is None and mm_peak<=0.0`)
fires before the circuit is even built.

**Is this reachable from `/api/brain-chat`? No — confirmed by code trace, not by absence of evidence.**
`webapp/gnw_thought_swap.py::ThoughtSwapWorkspace.observe()` always returns `mm_peak` alongside `swapped` in both
its branches (first-thought and subsequent), and `run_intention_swap`'s own `swapped` verdict is CAUSALLY
downstream of an elevated `mm_peak` (the eviction boost driving it is literally `boost_gain * mm_rate`) — a real
swap cannot happen without `mm_peak` having been substantially elevated first. `webapp/swap_drives_chat.py::observe_turn`
copies that whole dict forward (`out = dict(info)`) and its ONLY path that hard-codes `swapped=False` without
`mm_peak` is its exception handler, where `swapped=False` anyway (so neither trigger path would fire regardless).
`swapped=True` with `mm_peak` absent is therefore a state production cannot produce.

**Fix.** `research/runners/_gnw_global_stop_flip_soak.py`'s `_chat()` fixture now carries a realistic
mismatch-level `mm_peak` (0.30, matching the de-risk's own real "mismatch" scenario magnitude) alongside
`swapped=True`, so this soak exercises the same afferent SHAPE production actually produces. This is the same class
of defect `gates/flip_offarm_staleness` was built to catch (a default flip silently exposing a fixture that was
only ever exercised under the OLD default) — here on the INPUT shape a fixture constructs, not an env-var arm, so
the existing gate does not (and structurally cannot) see it; named here as the honest limit of that gate rather than
left implicit.

**Verified as the SOLE, COMPLETE cause** (not merely "a contributing factor"): re-running the ORIGINAL, unfixed
`detect_trigger` (Bug B below reverted) with ONLY this fixture fixed reproduces a clean 6/6 `seed_go=True` — the
decisive adversarial check against my own first (broader, less precise) hypothesis, kept in the reproduce section.

## Bug B (production code, fixed, but NOT the cause of any observed regression) — the circuit's RNG isolation missed Python's stdlib `random` module

**Found while root-causing Bug A**, before Bug A itself was identified: chasing the same `seed_go=False` symptom, a
targeted probe showed building the trigger circuit changes `random.getstate()` while leaving
`np.random.get_state()` byte-identical — a real leak. `_TriggerCircuit._isolated()` (mirroring the codebase's
#77/#85 pattern) snapshots/restores `np.random`/`xp.random` around the circuit's build+step, but never touched
Python's **stdlib `random` module** — a THIRD RNG source this codebase's own reproducibility doc names explicitly
("All RNG sources (CuPy, NumPy, random) are seeded together"). Because `webapp.gnw_acc_bg_stop_trigger` is imported
LAZILY inside `gnw_global_stop.detect_trigger`, before `_accbg`'s own isolation wrapper is ever entered, no wrapper
*inside* that module could have contained it — closing it has to happen at the caller, which is what the fix below
does.

**Adversarial follow-up (the honest part): fixing this did NOT, by itself, resolve the observed regression**, and a
direct A/B check proves it — reverting ONLY this fix (restoring the original unwrapped `detect_trigger`) while
Bug A stays fixed still reproduces `seed_go=True` on all 6 seeds (see Bug A's verification note). The stdlib-random
leak is real and measured, but this codebase's own culture treats "the fix I made happened to precede the symptom
going away" as an unproven claim until checked — checking it here shows the leak was a coincidental, non-causal
finding along the way, not the explanation. It is kept and fixed anyway on its own merits (a genuine hygiene gap
matching this codebase's explicit 3-RNG-source discipline), not because it was load-bearing for GATE 1.

**Fix (kept for its own sake, not for GATE 1).** `webapp/gnw_global_stop.py::detect_trigger` now wraps the whole
delegation (import + call) in a `random.getstate()`/`random.setstate()` snapshot/restore.

**Scope note.** The sibling `_StopWorkspace`'s OWN isolation has the identical `np.random`-only gap (confirmed:
its own `.run()` also perturbs stdlib `random`), inherited from the SAME #77/#85 pattern everywhere in this file
family. This finding closes it only at the ONE call site rank-12 touches; the broader gap across every
`_isolated()`-style wrapper in the codebase is a separate, out-of-scope cleanup, flagged for a follow-up task
rather than expanded here.

## Honest residuals (named, not claimed closed; unchanged from the de-risk + one new item)

1–4. Unchanged from `2026-09-05-gnw-stop-trigger-accbg-circuit-derisk-GO.md`: the scalar→current conversion is host
   arithmetic; the GPi-rate→boolean threshold is a fixed host constant; `delib_aff`/`mm_aff`/`acc`/`stn`/`gpi` are
   hand-wired, not self-organized; `n_held`/`newcomer` stay host bookkeeping in both paths.
5. The host boolean-OR is **not deleted** — it remains as an exception-only safety fallback
   (`except Exception: pass` in `detect_trigger`, the same "never crash a turn" idiom every sibling consumer in
   this file family uses). On any turn where the circuit builds without error — the normal case — it never
   executes, but the source line is still there. This is named honestly as `wired` + `on-by-default` per
   [`docs/TERMS.md`](../../docs/TERMS.md)'s code conditions, **not** `scaffold-retired` (that term requires the
   host shortcut to be gone from the default path or demoted to test/verify-only, and an exception fallback is
   neither).
6. The `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` `gnw-global-stop` row's own levels/residual text predate this flip
   and do not yet reflect the trigger sub-component's OWN de-risked/wired/on-by-default status; updating that
   ledger (and any board/roadmap sync it obliges) is left to the parent/controller landing this flip, per this
   campaign's own division of labor.
7. Bug B's isolation gap (stdlib `random` unmanaged by an `_isolated()`-style wrapper) is closed only at the one
   call site this finding touches; the same gap likely exists in every sibling wrapper built on the same #77/#85
   pattern. Flagged, not fixed here.

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._gnw_stop_trigger_production_flip_verify \
    --seeds 42 43 44 100 101 102 --json research/findings/raw/_gnw_stop_trigger_production_flip_verify.json

SIM_BACKEND=numpy python -u -m research.runners._gnw_global_stop_flip_soak \
    --seeds 42 43 44 100 101 102 --json research/findings/raw/_gnw_global_stop_flip_soak.json
```

The Bug A / Bug B causal-isolation check (monkeypatches `gnw_global_stop.detect_trigger` back to the pre-Bug-B
unwrapped version, in-process, then runs the already-Bug-A-fixed `evaluate_seed` on all 6 seeds — all 6 still
return `seed_go=True`, proving Bug A alone is sufficient and Bug B was not load-bearing for this symptom):
```
SIM_BACKEND=numpy python -u -c "
import os; os.environ.setdefault('SIM_BACKEND','numpy'); os.environ.setdefault('SIM_NO_PROVENANCE','1')
from webapp import gnw_global_stop as G
def _unfixed(chat):
    try:
        from webapp import gnw_acc_bg_stop_trigger as _accbg
        if _accbg.stop_trigger_spiking_enabled():
            return _accbg.detect_trigger_spiking(chat)
    except Exception: pass
    reason=None; n_held=2; newcomer=None
    delib=getattr(chat,'_last_gnw_delib',None)
    if isinstance(delib,dict) and isinstance(delib.get('n_ignited'),(int,float)) and int(delib['n_ignited'])>=2:
        reason='delib_sustained_coignition'; n_held=max(n_held,int(delib['n_ignited']))
    swap=getattr(chat,'_last_swap_drives',None)
    if isinstance(swap,dict) and bool(swap.get('swapped')):
        reason='swap_topic_break' if reason is None else 'delib+swap'
        t=swap.get('new_topic') or swap.get('held_topic'); newcomer=str(t) if t else None
    return (reason is not None), reason, n_held, newcomer
G.detect_trigger = _unfixed
from research.runners._gnw_global_stop_flip_soak import evaluate_seed
for s in (42,43,44,100,101,102):
    r = evaluate_seed(s, verbose=False); print(s, r['seed_go'])
"
```
