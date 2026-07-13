# EMERGE graded-confidence wired into the TALKABLE console — 12-seed GO (2026-07-13)

**Headline.** The validated graded-confidence completion (the 2026-07-08 frontier gate's #1 open
piece; `2026-07-13-EMERGE-graded-confidence-completion-12seed-GO.md`) is now wired INTO the
emergent console the owner actually talks to. The EMERGE-31 experiential console
(`_emerge31_experiential_console.py`) LEARNS categories from observed co-occurrence and answers by
inheritance, but its `ask_can` was a **hard binary** — "Yes, a X can P." / "I don't know whether a
X can P." This wire-in replaces that binary membership test with the graded 2-hop apical-drive +
margin read, so a spoken question gets a **three-level graded answer plus the intrinsic moat**:

```
you> can a robin fly?          brain> Yes, a robin can fly.              (CONFIDENT)
you> can a bat fly?            brain> A bat can probably fly.            (HEDGED)
you> can a trout fly?          brain> I don't know whether a trout can fly.  (ABSTAIN)
you> can a wolpertinger fly?   brain> I don't know what a wolpertinger is.   (MOAT — never observed)
```

This extends the no-confab moat **from a hard abstain to a graded hedge** — the concrete step toward
open-domain conversation (a co-observed concept with weak/contested evidence hedges instead of
falsely asserting or flatly refusing). The confidence is **not hand-coded**: it EMERGES from the
strength of the learned co-occurrence drive (Rogers-McClelland graded distributed completion; the
graded read is Bogacz-Brown evidence-margin confidence).

## Mechanism (reuse-by-import; NO `sim/` edit)

`GradedExperientialConsole(ExperientialConsole)` in
`research/runners/_emerge_graded_confidence_console_derisk.py` overrides only `ask_can`:

1. **Unknown concept** (`member not in self._cols`) → MOAT ("I don't know what a X is") — the
   intrinsic no-confab moat, unchanged.
2. **Direct exemplar** (the property was taught to this member) → CONFIDENT (a stated fact).
3. **2-hop inheritance:** prime the member → its emergent context cells → prime those → read the
   apical drive on the asked property and the **margin over the strongest competing taught
   property**. `drive ≤ FLOOR` → ABSTAIN; `margin > CONF_MARGIN` → CONFIDENT (the property
   dominates); else → HEDGED (present but contested). `CONF_MARGIN = 30` is the SAME apical-drive
   scale as the validated completion probe (`CONF_TH(-10) − FLOOR(-40)`), validated on the console
   here.

**One mechanism subtlety diagnosed + fixed (a0 on the substrate):** the console's `observe(member,
context)` runs a full potentiation block per call, and the committed HTM three-term kernel
*depresses inactive* — so calling `observe("bat","nest")` then `observe("bat","cave")` (two
sequential blocks) ERASES bat↔nest. A genuinely category-ambiguous concept is co-observed with both
contexts *interleaved over time*, not one block then the other. `observe_mixed(member, contexts)`
strictly alternates the contexts across epochs → balanced ~50/50 evidence → the ambiguous member
lands HEDGED (mirroring the validated probe's per-epoch-alternating ambiguous member). This is the
faithful temporal model of ambiguity, not a hack.

## Scenario (emergent — no hand-coded confidence)

Observe robin/sparrow with "nest" (bird context), mole with "cave" (mammal context), **bat with
BOTH "nest" AND "cave" interleaved** (category-ambiguous), trout with "river" (fish, no property).
Teach "a sparrow can fly" (binds fly to the bird context) + "a mole can walk". Then ask. The
confidence of each answer falls out of how strongly the queried concept's learned context drives the
asked property, contested by competing properties.

## Result — 12-seed GO (standard 42/43/44/100/101/102 + FRESH 7/8/9/10/11/12)

GO gate per seed: robin=CONFIDENT (+correct) AND bat=HEDGED AND trout=ABSTAIN AND wolpertinger=MOAT,
with PERMUTED destroying the confident answer AND LESION driving all-abstain.

- **htm arm — UNANIMOUS across all 12 seeds:** robin=CONFIDENT, bat=HEDGED, trout=ABSTAIN,
  wolpertinger=MOAT. The graded read + moat is fully robust.
- **Anti-cheats:** LESION (coincidence off → no apical drive) → robin=ABSTAIN every seed. PERMUTED
  (every member co-occurs equally with all contexts → no category structure) → robin=HEDGED (i.e.
  ≠ CONFIDENT — fly no longer dominates, contested by the now-equal walk evidence) every seed.

### Control-validity fix (the 2026-07-02 lesson, applied)

The first pass used a random single member→context permutation and failed on **3/12 seeds** (44,
102, 9) — the permutation coincidentally left robin→nest in a 3-context space, so
`perm.robin=CONFIDENT`. This is exactly the small-space fixed-random-control unreliability
documented in `2026-07-02-anti-cheat-control-validity-methodology.md`. Fixed by gating on
**deterministic input-destruction** instead: in the permuted arm EVERY member co-occurs equally with
ALL contexts (mirroring the validated completion-probe's per-epoch permutation) → no category
structure can survive → robin drops from CONFIDENT to HEDGED (≠ CONFIDENT) reliably on every seed.

## Scope / honest notes

- Validated at the console's small demonstrated concept inventory (a handful of members / 3 contexts
  / 2 properties). Real-corpus-stream threshold tuning (calibrating CONFIDENT vs HEDGED on a larger,
  noisier co-occurrence stream) and scaling the inventory are the follow-ons.
- The graded margin threshold (`CONF_MARGIN=30`) is inherited from the validated completion probe's
  apical-drive scale (same `build_pool_bridge` machinery) and confirmed to transfer here; on a
  differently-scaled stream it would be re-calibrated (exposed as a constructor param).
- NO `sim/` edit anywhere (reuse-by-import of the committed HTM pool-bridge + EMERGE-31 console).
  The no-confab moat is preserved by construction (unknown concept → MOAT before any graded read).

## Files

- `research/runners/_emerge_graded_confidence_console_derisk.py` — `GradedExperientialConsole` +
  the 12-seed de-risk (`--demo` prints the transcript).
- Builds on: `_emerge_graded_confidence_completion_derisk.py` (the validated mechanism),
  `_emerge31_experiential_console.py` (the talkable console), `_emerge14_stageC_onbridge_learning_derisk.py`
  (`build_pool_bridge` + the committed 3-term kernel).
