---
type: finding
status: go
date: 2026-08-27
mechanism: production-gate-hollow-flip-fix
lane: one-brain / integration-honesty
artifacts:
  - research/findings/raw/_gnw_three_organ/production_verify_realvocab_numpy.json
runner: research/runners/_gnw_three_organ_bus_verify.py
---

# A hollow production flip on the honesty spine, fixed: GNW three-organ bus was `on_by_default: YES` in the ledger but INERT in production — plus a clean 36-gate systemic audit for the same class

**One-line:** the GNW three-genuinely-distinct-organs consensus bus was flipped `on_by_default: YES` on
2026-08-21 (owner-approved, over-veto fixed, 6-seed-composed GO), and the PI-ledger + `webapp/server.py`'s outer
gate + the flip finding all read default-ON — but a SECOND, inner gate `gnw_three_organ_bus.py::three_organ_enabled()`
was left defaulting `""` (OFF), so on the production default (`BRAIN_GNW_3ORGAN` unset) the outer block ran and the
inner gate silently installed nothing. The faculty the ledger CLAIMED was live had done nothing in production since
the flip landed. Found while auditing a separate flip-soak staleness class; fixed (inner fallback `""`→`"1"`);
verified genuinely live by default; and a systemic 36-gate audit of every other `on_by_default: YES` faculty found
NO other real instance of this class.

## The bug (a two-gate "hollow flip")

`BRAIN_GNW_3ORGAN` is read in two places with DIFFERENT defaults:
- `webapp/server.py` (the outer hook): `os.environ.get("BRAIN_GNW_3ORGAN", "1")` — default **ON**; when unset it
  enters the block and calls `install_three_organ_gate(chat)`.
- `webapp/gnw_three_organ_bus.py::three_organ_enabled()` (the inner master switch): `os.environ.get("BRAIN_GNW_3ORGAN", "")`
  — default **OFF**; `install_three_organ_gate` re-checks it at line 320 and `return False` (installs nothing).

So with the flag genuinely unset — the production default — the outer gate opens and the inner gate vetoes: the
3-organ workspace is never installed. The 2026-08-21 flip updated the outer gate + the PI-ledger row
(`on_by_default: YES`, finding `2026-08-21-gnw-three-organ-realvocab-flip-GO`) but MISSED this inner gate, so the
ledger's `on_by_default: YES` has been FALSE in production since the flip. This is the exact "hollow checkbox
integration" the goal (a brain measured on the INTEGRATED PRODUCTION SYSTEM) exists to prevent.

## The fix + verification

Artifact: `research/findings/raw/_gnw_three_organ/production_verify_realvocab_numpy.json` (the GO verify run,
numpy seed 42, through the real production ChatBrain).

Aligned the inner gate to the intent (default-ON), matching the outer gate + ledger + the flip finding:
`three_organ_enabled()` fallback `""` → `"1"` (`webapp/gnw_three_organ_bus.py`). Verified:

- **Gate-level:** with the flag genuinely unset (production default) `three_organ_enabled()` now returns **True**
  (the faculty installs by default); `BRAIN_GNW_3ORGAN=0` still returns **False** (the byte-identical escape is
  preserved); `=1` True.
- **Mechanism (through the REAL production ChatBrain + handler, `_gnw_three_organ_bus_verify.py`, numpy seed 42):**
  (B) NO-REGRESSION — every legitimate recall commits (incl. the 2 the old toy veto abstained); (C) GENUINE
  NON-COMPREHENSION VETO — `veto_ok`, `reverts_under_lesion`, `known_control_ok` all True; (D) MOAT holds; the
  low-comprehension veto is 100% ATTRIBUTABLE to organ C's spiking participation (intact vs lesion). These
  substantive checks pass with the organ explicitly on — the flip does not change them, it just makes the default
  path actually reach them.

### The verify runner itself had the same stale-OFF-arm bug (found + fixed here)

The first re-run came back UNDEFINED, not because of the fix but because `_gnw_three_organ_bus_verify.py` set its
OWN "OFF" arm via `os.environ.pop("BRAIN_GNW_3ORGAN", None)` (lines 132, 141) — the identical unset-as-OFF
assumption the 2026-08-27 flip-soak audit was about. Once the faculty defaults ON, `unset` reads ON, so the
runner's `install-off-is-noop` and `install-on-installs` probes both broke (the off-probe installed; the on-probe
then hit the idempotent already-installed short-circuit). Fixed the runner's OFF arms to explicit `="0"` (the true
byte-identical-escape state) — the same fix pattern as the 5 soaks in the sibling audit. A neat corollary: the
verify runner had been implicitly RELYING on the hollow flip (unset==OFF); making the flip genuine exposed it.

A SECOND instrument flaw then surfaced (a NO-GO, not UNDEFINED): the runner's (A) byte-identity panel included a
STOCHASTIC generative probe (`"what might a dog do"` — the generative-DRAW WTA draws a fresh sample each call, so
its output drifts with RNG position between the two panel evaluations REGARDLESS of the 3-organ code, which
provably delegates to `orig_gate` when off). Byte-identity-testing a stochastic output is a category error; the
deterministic panel (all recalls + moat + `"what are you"`) was already byte-identical. Excluded the generative
probe from the byte-identity comparison (its delegation is covered structurally). With BOTH instrument flaws
fixed, the runner returns a clean **GO**: `runtime_flip_off_matches_two_organ=True`, `install_noop_when_off=True`,
no_regression/moat/veto all pass, attributable 1.0 — through the REAL production ChatBrain.

## Bug 2 (same audit): a stale docstring, not a behavior bug

`webapp/continuous_engine.py::ideation_enabled()` reads `os.environ.get("BRAIN_CONTINUOUS_IDEATE", "1")` (default
ON) — the owner-approved 2026-08-21 flip (commit `c7654d52a`, "flip … continuous-ideation default-ON"), confirmed
by the function's own inline comment. Only the docstring's stale "Default-OFF anchor" first line contradicted it.
Corrected the docstring to default-ON; behavior was already correct.

## Systemic audit: is the hollow-flip class widespread? (No.)

A 9-agent workflow audited all 36 gate reads across every PI-ledger `on_by_default: YES` faculty for the same
class (ledger claims ON, real production gate defaults OFF). Result: **35 GENUINE_ON, 1 false positive, 0 other
real instances.** The lone flag it surfaced — `BRAIN_GNW_BUS` on `gnw-bus-shadow` — reads OFF everywhere, but that
faculty is default-ON via `install_bus_gate` with escape `BRAIN_GNW_BUS_HOST=1`; `BRAIN_GNW_BUS` is only the
observability/debug-attach flag, correctly default-OFF (the audit agent mis-identified the observability flag as
the activation gate). So GNW-3organ was the only genuine hollow flip, and it is now closed.

## Honest scope

- This makes the GNW 3-organ workspace genuinely live-by-default for the first time since 2026-08-21 — it COMPLETES
  an owner-approved, already-GO-verified flip that a missed second gate had silently blocked; it does not decide a
  new default (the decision was the owner's on 2026-08-21).
- Candidate MECHANICAL gate (not built this session, logged in `research/FAILURE_LOG.md`): a check that a module's
  own `_enabled()`/`*_on()` reader agrees with any OUTER gate that calls into it before installing, + a
  docstring-vs-fallback-literal consistency check on every `os.environ.get(FLAG, "<literal>")`. Either would have
  caught this at commit time.
