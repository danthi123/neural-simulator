---
type: finding
status: live
lane: load-bearing
date: 2026-09-23
verdict: NO-GO (5/6) — the pre-registered ALL-6-seed gate is not met; seed 101 alone misses, by design, because of a uniform cliff-safety ceiling
---

# Prospective-memory LIVE operating-point homeostat: 5/6 seeds clear the pre-registered gate, seed 101 misses by a uniform safety margin (2026-09-23)

Branch `research/lbf-live-homeostat`. Direct honest-residual follow-on of
`research/findings/2026-09-23-operating-point-stabilizer-go-6seed.md` (merged to main @f48fce12), which named its
own gap explicitly: *"a genuinely continuous/live homeostat (rather than this static, precomputed 6-seed table) is
a named follow-on"*. This build replaces the static `CALIBRATED_FAC_G` lookup dict with a genuine LIVE, in-loop
Turrigiano-style integral set-point controller (`research/runners/_pmem_live_homeostat_derisk.py`) that reads the
prospective-memory pool's own running N=3 coincidence output and adjusts the facilitation gain `fac_g` toward a
fixed target, with no per-seed table anywhere in the control loop. **This is NOT a merge candidate** (task
instruction, honored): the branch stays open, carrying its own honest verdict.

## A discovery this build made before it could be built safely
The parent build's coarse grid (`6000, 7000, 8000, 9000, 10000, 11000`) reported seed 44's cliff as "somewhere in
[9000, 10000]" (9000 safe at 0.2183, 10000 collapsed at 0.1506). A LIVE, continuous controller can probe at ANY
gain, not just six pre-picked ones — so before freezing the safety ceiling, this build ran a finer pilot scan at
seed 44 (the pre-documented hardest/cliffiest seed) between the two -- an AD HOC pilot scan, not a committed
artifact (reproduce with the one-liner in "Files" below).
**9100 pA still reads 0.2144 (fine); 9200 pA reads 0.1911 — a 0.023 drop in ONE HUNDRED pA.** <!--derived-->
The true edge is far tighter and far more abrupt than the parent's grid resolution could show, and it is
seed-44-SPECIFIC: the identical scan at seeds 100 and 101 stayed flat (0.283–0.296) all the way through 9400 pA. <!--derived-->
The FIRST version of this controller
used `G_CEILING=9500` (the midpoint of the parent's assumed-safe window) — it would have walked straight into
seed 44's collapse on its very first climb. The ceiling was tightened to **`G_CEILING=9000`** (the LAST grid point
the parent build had already validated safe) BEFORE the 6-seed run, not after seeing any 6-seed result — the same
"fix the controller from a principled argument, not a fit to the outcome" discipline the task itself required.

## The mechanism (additive; NO sim/ edit; reuse-by-import of the committed GO facilitation substrate)
A Turrigiano-style slow integral set-point controller (Turrigiano 2011; Desai, Rutherford & Turrigiano 1999 — the
SAME citations the static build used, now given the LIVE dynamics those citations actually describe). For a seed,
from an initial gain `g_0`:
1. MEASURE the pool's OWN N=3 production-protocol intact coincidence read `rel_i` at the CURRENT gain `g_i`, via a
   fresh, self-consistent build (`F._n3_arm` — the only correct way to evaluate a candidate `fac_g` on this
   substrate, per the parent build's own finding).
2. `error_i = REL_TARGET(0.30) - rel_i`.
3. `raw_delta = K_I(20000) * error_i`.
4. `delta_i = clip(raw_delta, -MAX_STEP(500), +MAX_STEP(500))` — a rate limit independent of the (unknown, possibly
   locally steep) plant slope.
5. `g_{i+1} = clip(g_i + delta_i, G_FLOOR(6000), G_CEILING(9000))` — the floor is the shipped constant (a
   homeostat already at target does not regress below what shipped); the ceiling is the cliff-safety bound above.
6. Repeat until `|g_{i+1}-g_i| < 25` pA for 3 consecutive iterations (settled — an interior point or a safely
   pinned bound) or 16 iterations are exhausted.
`K_I`/`MAX_STEP` were checked for smooth, non-oscillating convergence on seed 44 alone (a stability argument, not a
result fit) before being frozen and applied UNCHANGED to all 6 seeds. Every seed is run from **two** different
inits — `G_INIT_LOW=6000` and `G_INIT_MID=8000` — both inside the safe interior (never at a bound).

## Result — 6-seed (`_pmem_live_homeostat_derisk.py --derisk`, numpy CPU, memcap 12)
<!--derived from research/findings/raw/_pmem_live_homeostat.json-->
| seed | low-init final `fac_g` | mid-init final `fac_g` | same set-point | converged `rel` | lesion `rel` | load-bearing | frozen N=5 silence | static-table margin | **live-converged margin** | meets static |
|---|---|---|---|---|---|---|---|---|---|---|
| 42  | 6000 | 6000 | yes | 0.3439 | 0.0278 | True | pass (0.0472) | +0.1439 | +0.1439 | **yes** |
| 43  | 6000 | 6000 | yes | 0.3361 | 0.0206 | True | pass (0.0483) | +0.1361 | +0.1361 | **yes** |
| 44  | 9000 | 9000 | yes | 0.2183 | 0.0000 | True | pass (0.0467) | +0.0183 | +0.0183 | **yes** |
| 100 | 9000 | 9000 | yes | 0.2956 | 0.0178 | True | pass (0.0475) | +0.0956 | +0.0956 | **yes** |
| 101 | 9000 | 9000 | yes | 0.2889 | 0.0000 | True | pass (0.0456) | +0.0922 | **+0.0889** | **NO** |
| 102 | 6000 | 6000 | yes | 0.3133 | 0.0211 | True | pass (0.0408) | +0.1133 | +0.1133 | **yes** |

**5/6 seeds meet-or-beat the static table's own margin exactly, load-bearing 6/6, silence held 6/6, cliff-safe 6/6,
same-set-point-from-two-inits 6/6.** Seed 101 alone falls **0.0033 short** <!--derived--> (+0.0889 live vs +0.0922 static, 0.0922-0.0889): the
static table's own search reached that seed's margin at `fac_g=11000` (inside the parent's coarser grid, which
never measured whether 10000/11000 were safe FOR THAT SEED specifically — it only knew 44's cliff), while this
build's SINGLE, uniform, evidence-based safety ceiling (9000, fixed for every seed to respect seed 44's
newly-discovered edge) never lets any seed's controller climb that high. **Per the pre-registered gate (task's own
wording: GO iff ALL 6 seeds clear it), this is a NO-GO** — 5/6 is not 6/6. Full per-seed trajectories, evals and
the Verdict block (12 preconditions, `tools.verdict.Verdict`) are in the cited artifact; the runner's own printed
verdict reads `verdict_status: UNDEFINED` (the project's `Verdict.decide()` reports UNDEFINED, not a bare NO-GO,
whenever any registered precondition is unmet — here the "meets static margin, 6/6" precondition is the one that
did not hold; this is a genuinely MEASURED shortfall, not an instrument failure, and is reported as such in prose).

## The convergence trajectory (proving genuine live convergence, not a lookup)
Seed 44, low-init (identical shape at mid-init, converging to the same 9000): <!--derived, see artifact-->
```
iter  0  fac_g=6000.0  rel=0.2111  error=+0.0889
iter  1  fac_g=6500.0  rel=0.2150  error=+0.0850
iter  2  fac_g=7000.0  rel=0.2078  error=+0.0922   (the dip the parent build also found)
iter  3  fac_g=7500.0  rel=0.2089  error=+0.0911
iter  4  fac_g=8000.0  rel=0.2139  error=+0.0861
iter  5  fac_g=8500.0  rel=0.2172  error=+0.0828
iter  6  fac_g=9000.0  rel=0.2183  error=+0.0817   (ceiling reached; ERROR STILL POSITIVE -- target unreachable)
iter  7..9  fac_g=9000.0 (pinned)  streak -> 3      CONVERGED at the ceiling, never past it
```
Seed 100's low-init trajectory (target also unreachable) is the SAME shape but with genuinely different step sizes
(the integral law is already in its fine/proportional regime from iteration 0, since seed 100's baseline error was
already small: steps of 322/212/266/... pA, not the 500 pA bang-bang ceiling) — direct evidence the controller
reads and reacts to each seed's OWN measured error, not a fixed step schedule. <!--derived, see artifact-->

## Anti-cheats (each implemented + measured)
1. **Genuinely live, not a table.** `prove_it_is_live(seed=44)`: this module's globals contain no
   `CALIBRATED_FAC_G` / `stabilized_fac_g_for_seed` (`no_table_import=True`), and re-running the identical
   (seed, init) trajectory TWICE reproduces it EXACTLY (`rerun_identical_trajectory=True`) — the value is computed
   fresh each time from a deterministic live read, not memoised from a shipped dict.
2. **Two different inits, same set-point.** `G_INIT_LOW=6000` and `G_INIT_MID=8000` converge to the identical
   final `fac_g` on **6/6** seeds (`n_same_setpoint=6`) — for the three above-target seeds (42/43/102) this is a
   genuine DESCENT from 8000 back down to the floor (not a trivial "stayed put"); for the three below-target seeds
   (44/100/101) both inits climb to the SAME safety ceiling. An init-dependent final gain would mean drift, not
   regulation; none was observed.
3. **Cliff-safe.** `max(all fac_g visited, every seed, every init) = 9000.0 <= G_CEILING(9000) < CLIFF_EDGE_S44
   (9150, this build's own pilot bracket) < 10000` (the parent's documented collapse) — asserted directly off the
   trajectories, not assumed. The controller never entered the collapse zone even where the target was
   unreachable and integral windup pulled every below-target seed toward the bound (see the seed-44 trace above:
   error is still `+0.0817` at the moment it stops climbing — pinned, not satisfied).
4. **Load-bearing preserved.** 6/6 (structurally guaranteed by the Mg-block-gated facilitation current requiring
   the held assembly's own firing; lesion `rel` is 0.0000–0.0278 on all 6 seeds, well under `FIRE_THR=0.20`).
5. **Byte-identical default-off**, asserted in the data. `BRAIN_PMEM_LIVE_HOMEOSTAT` unset → the production hook
   (`pmem_live_homeostat_enabled()`) is never called; a LIVE WIRING CHECK (both directions, seed 44) confirms:
   with `BRAIN_PMEM_LIVE_HOMEOSTAT=1` AND `BRAIN_PMEM_OP_STABILIZER=1` both set, the LIVE controller takes
   PRIORITY (`fac_g=9000.0`, `run_live_homeostat` called); with only `BRAIN_PMEM_OP_STABILIZER=1`, the static table
   path runs unchanged (`fac_g=9000.0`, no live call); with **neither** flag set, the constructor call carries NO
   `fac_g` kwarg at all — the identical call as before this build. Separately, this run's own low-init iteration-0
   read (measured at `g=FAC_G_DEFAULT`, THIS process) is EXACT-COMPARED against the static stabilizer's
   INDEPENDENTLY-committed grid-point-0 read (a different process, a different day): all 6 seeds match to 4 decimal
   places (`default_off_exact_compare.ok: true`).
6. **Determinism.** The re-run in anti-cheat #1 IS the determinism check (identical seed → identical trajectory,
   full precision, not just the final value).

## Default-OFF wiring (`research/runners/prospective_memory_production_organ.py`)
New `pmem_live_homeostat_enabled()` (env `BRAIN_PMEM_LIVE_HOMEOSTAT`, default-OFF) gates a NEW branch in both
`_ensure_pm` call sites, checked BEFORE the existing `pmem_op_stabilizer_enabled()` branch (so LIVE takes priority
when both flags are set — verified above): ON, it calls `live_fac_g_for_seed(seed)`
(`_pmem_live_homeostat_derisk.py`), which runs `run_live_homeostat(seed, G_INIT_LOW)` the first time THIS seed is
needed in the process and CACHES the result (`_LIVE_CACHE`, the identical per-process calibration-caching pattern
the homeostat bias / plateau theta already use elsewhere in this codebase) — nothing ships pre-computed; the cache
is empty until the process actually runs the convergence loop itself. OFF (both flags unset, the shipped default),
no `fac_g` kwarg is passed at all — the SAME code path as every prior build.

## Honest residual (banked, not hidden)
The pre-registered ALL-6-seed gate is **NOT met**: seed 101's live-converged margin (+0.0889) sits 3.6% below the
static table's own committed margin (+0.0922) because this build's single, uniform, evidence-based safety ceiling
(9000, sized to respect seed 44's own newly-discovered edge at ~9150) is more conservative than the static table's
per-seed-searched top grid point (11000 for seed 101 specifically — a value the parent grid never independently
validated as safe for the OTHER below-target seeds, only for 101 itself, at the coarse resolution it used). This
is the expected, quantified shape of the trade this build's own instructions anticipated: **a uniform mechanism
applied identically to every seed cannot match a table that was allowed to pick a DIFFERENT top value per seed**,
and this build declines to break that uniformity (a per-seed-tuned ceiling would be exactly the kind of hidden
per-seed hack CLAUDE.md's "uniform mechanism, not a per-seed tune" standard forbids). The residual is precisely
sized: 0.0033 of `rel` (0.0922-0.0889) <!--derived-->, on one seed, entirely attributable to the ceiling's conservatism, not to any other part of
the controller (5/6 seeds match the static table's margin EXACTLY, to 4 decimal places, because their optimum sits
at or below 9000 anyway).

**Named next controller** (per the task's own instruction: bank the miss, name the next method, do not tune to
force a pass). The gap is not the CONTROL LAW — it is that a single global ceiling cannot capture a per-seed
cliff location that this build only measured for ONE seed (44). The principled next step is not a bigger ceiling
(that is exactly the tuning-to-force-a-pass this build was told not to do) but a LIVE CLIFF DETECTOR: extend the
integral step with a local-derivative safety check (if `rel` drops by more than some fraction between two
consecutive live reads at the SAME direction of travel, treat that as a detected cliff and clamp `G_CEILING` down
for THAT seed from then on, rather than assuming one global number is safe everywhere) — a homeostat that finds
its OWN safe ceiling live, the direct generalization of the fixed-ceiling design here, banked for a follow-on
build rather than retrofitted into this one to manufacture a 6/6.

## Files
- `research/runners/_pmem_live_homeostat_derisk.py` (new): the live controller, the 6-seed de-risk, the
  anti-cheats, `selftest()`.
- `research/runners/prospective_memory_production_organ.py`: `pmem_live_homeostat_enabled()` (default-OFF) + two
  call sites in `_ensure_pm`, checked before the existing op-stabilizer branch.
- Artifacts: `research/findings/raw/_pmem_live_homeostat.json` (+ `.prov.json`), reads
  `research/findings/raw/_pmem_operating_point_stabilizer.json` (the static table's own committed margins — the
  GO-gate's comparison baseline).
- Pilot scans (not committed as artifacts; reproduce with `_pmem_facilitation_derisk._n3_arm(seed, fac_on=True,
  lesion=False, N=3, fac_g=<g>, fac_U=F.FAC_U, fac_tau_F_steps=F.FAC_TAU_F_STEPS)` at `g in
  {9100,9200,9300,9400,9500}`): the finer seed-44 cliff-edge scan that set `G_CEILING`, and the two-init
  seed-44-only convergence pilot that set `K_I`/`MAX_STEP` before the 6-seed run.

## Honesty
Functional read-out only — a spiking coincidence read against a fixed release threshold, hardened by a live,
bounded integral controller reading its own output. No claim of phenomenal experience. This finding's own headline
NO-GO is reported in full alongside the 5/6 partial result it rests on, per `docs/TERMS.md`'s GO condition ("the
gate's OWN verdict is positive — never a metric lifted out of a run whose verdict was negative") and the task's
own instruction that an honest NO-GO is a METHOD verdict, not a capability to defer.
