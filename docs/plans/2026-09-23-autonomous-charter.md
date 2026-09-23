# Autonomous Charter — the measurable medium-term goal for weeks-long hands-off operation

**Status: DRAFT for owner review (2026-09-23). Not active until the owner pastes the `/goal` line (§7) + enables auto mode.**
This charter defines a *measurable* medium-term end-state so Claude can run for weeks toward it, the owner speaking
only to (a) free local compute for gaming or (b) get progress reports. It does NOT redefine the mission's
non-negotiables (brain-based-only · ONE brain · emergent-not-hand-built · NO-DEFER · speed<faithfulness ·
honesty-boundary · 6-seed · gates AUTHORITATIVE) — it operationalizes a slice of them into a checkable target.

---

## 1. Why a charter (and the honest limitation of `/goal`)
`/goal` keeps turns running until a small fast model (Haiku) judges a condition met — but that evaluator **only reads
what Claude surfaced in the conversation; it runs no commands and reads no files.** It is a WEAK check. So the
completion condition below is written to make the **project's own gates + verify-go + 6-seed the real verifier** — the
`/goal` evaluator merely keeps the loop turning; truth is enforced by machinery that runs independently of what Claude
"says." (This session already had adversarial-verify catch 3 overclaims — that discipline is the backbone, not the
Haiku check.)

## 2. The measurable medium-term goal (owner, 2026-09-23)
> A simulated brain that can **learn and grow**, **feel**, and **express (load-bearing) all other major human brain
> faculties**, while **conversing fluently and open-endedly**, running on a **single 'one brain' spiking substrate**,
> with allowances for **crutches specifically impossible to surmount in a reasonable span** (e.g. the Qwen mouth).

This is the achievable medium-term target — NOT full scaffold-retirement (the end-game, blocked on the deep
neural-render wall; only 6/71 rows retired today) and NOT "consciousness" (unmeasurable by design + honesty-boundary).

## 3. DONE conditions (measurable, ledger-grounded, gate-verified)
The authoritative source is `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (CLASS PI gate cross-checks every row vs live
source). "The target set" = the major-faculty rows EXCEPT the §4 allowed crutches. Each condition is provable from
artifacts Claude surfaces (ledger flags, gate output, 6-seed findings) — re-derive exact counts at check time, do not
hardcode.

- **D1 — Express (load-bearing), the #1 metric.** Every target-set faculty is lesion-verified **load-bearing at the
  adequate probe across all 6 seeds** (42/43/44/100/101/102), each a verify-go GO committed both remotes. Target:
  robust-core (load-bearing in ALL 6 seeds) → **≥ 24 / 26** on the load-bearing battery (from 22 today), reported
  Option-C (paired with the thin-probe number). A faculty whose honest verdict is a characterized NO-GO with the next
  method banked counts as *resolved-for-now*, not a blocker (NO-DEFER).
- **D2 — On-by-default.** Every target-set faculty reads `on_by_default: YES` in the ledger (from ~55/71 today), each
  flip passing the auto-flip 6-seed byte-identical gate + verify-go. (Production default-flips that change how the
  brain *speaks* to a user are an owner-fork — see §5.)
- **D3 — One brain.** The core cognitive faculties (the 4 cortical organs + affect + memory + metacog + the target
  set that can) run on the **shared single spiking pool** (`BRAIN_ONEBRAIN_SINGLE_POOL` family), validated
  answer-preserving 6-seed — i.e. cross-region synaptic interaction on ONE substrate, not co-residency.
- **D4 — Converse fluently + open-ended.** The integrated conversation battery passes 6-seed for: fluent open-ended
  replies (BRAIN_OPEN_ENDED path), honest (moat holds — no fabrication), and affect-colored (feel is expressed, D5).
  The **Qwen mouth is the declared, allowed articulation scaffold** (owner-ratified permanent mouth) — its presence
  does NOT block DONE.
- **D5 — Feel.** Affect is lesion-verified load-bearing over the reply (the §8 arc) — the brain's affect read
  provably changes the reply's content/tone, 6-seed. (Affect→tone via additive-bias is a characterized NO-GO; the
  brain-based neural-coupling method is under test — this dimension may take real work; a banked NO-GO + next method
  is progress, not failure.)
- **D6 — Learn and grow.** Continuous learning (learn-through-use / novelty / surprise-driven) is lesion-verified
  load-bearing + on-by-default — the brain measurably changes from use, 6-seed.

**Overall DONE = D1–D6 all hold, each gate-verified + 6-seed + committed both remotes, with the ledger + a summary
finding demonstrating it.** Partial credit is tracked per-dimension in the board.

## 4. Allowed crutches (EXEMPT — do not block DONE, do not force-retire)
- **The Qwen mouth** — owner-ratified permanent conditioned-articulation scaffold (spiking-mouth-fluency CLOSED as
  falsified). Its neural burn-down (`BRAIN_WKV_MOUTH_*`) is tracked but NOT required for medium-term DONE.
- **World / body / clock / initial-education curriculum** — LEGITIMATE host (never owes retirement).
- **Any faculty whose neural replacement is BLOCKED on a genuinely multi-year wall** (e.g. the deep neural-render
  frontier) — its host crutch is allowed for the medium-term goal PROVIDED the faculty is still load-bearing (D1) +
  on-by-default (D2). Each such exemption is listed explicitly in the board with its blocking wall, so the allowance
  is honest and auditable, not a silent pass.

## 5. STOP-and-flag (owner-reserved — Claude must NOT do these autonomously)
On hitting any of these, Claude does NOT proceed — it records the decision in the board + surfaces it in the next
progress report, and keeps working OTHER lanes meanwhile (never blocks the whole goal on one fork):
- **Any spend** (AWS instances/grids, paid APIs).
- **Production default-flips that change the user-visible reply** (how the brain speaks) — propose, don't flip.
- **The AGI-first fork / relaxing a non-negotiable constraint.**
- **Anything honesty-boundary-adjacent** — never assert phenomenal experience; a self-report is always a functional
  read-out. If a claim would even approach "felt/conscious," STOP.
- **Publishing / outward-facing actions**, deleting data, or anything hard to reverse.
- **Freeing local compute for gaming** — on the owner's word (or detected GPU contention), renice/pause heavy lanes.

## 6. Operating discipline (how Claude runs the goal)
- **Verification is the gates, not the /goal evaluator.** No GO/merge without verify-go (adversarial) + 6-seed + the
  pre-commit gates. NO-GO is banked with the next method named (NO-DEFER); it counts as progress.
- **Cost-routing:** CPU→pool, GPU→gpu_queue (one brain-loading GPU proc at a time), multi-seed→--seeds, AGENTS=genuine
  builds only, model-tiered. Commit BOTH remotes via `tools/push_both.sh`, never `--no-verify`.
- **One heavy brain-lane per resource** (RAM/swap-thrash lesson; mem_ok gates it). RSS-based proc checks, never a
  pgrep-`-f` pattern that self-matches + gets killed.
- **Compaction-durable:** the GAP_CLOSURE board CURRENT STATE + live_state stay current every cycle so weeks of
  compaction never lose the thread. A state-checking heartbeat stays armed.
- **Progress reports:** a concise report at each `/goal` check-in (30min→1h→2h cadence) + a daily summary — landings,
  metric movement, open forks awaiting the owner, honest residuals.

## 7. The `/goal` line to paste (owner action — Claude cannot self-issue it)
Enable **auto mode** first (so turns run unattended), then paste:

```
/goal Advance the neural-simulator medium-term goal per docs/plans/2026-09-23-autonomous-charter.md: D1 load-bearing robust-core >=24/26 6-seed (Option-C reported), D2 target-set on_by_default, D3 core faculties on the shared one-brain pool, D4 fluent+honest+affect-colored open-ended conversation battery 6-seed (Qwen mouth allowed), D5 affect load-bearing over the reply, D6 continuous-learning load-bearing+on. Every claimed win MUST pass verify-go (adversarial) + 6-seed + the pre-commit gates + be committed to BOTH remotes; a characterized NO-GO with the next method banked counts as progress (NO-DEFER). STOP-and-flag every owner-reserved fork (spend, user-visible production flip, AGI-fork, honesty-boundary, publish) per charter §5 and keep other lanes moving. Keep the GAP_CLOSURE board durable through compaction. Report at each check-in. Not-done until D1-D6 all hold gate-verified; stop and report if genuinely blocked on an owner-fork with no other lane to advance.
```

## 8. Honest caveats (so this is set up with eyes open)
- **`/goal` is built for hours, not weeks.** It survives resume + usage-limit auto-pause, but an **unrecoverable
  clear** (auth failure [desktop app handles this], exhausted credit, context-overflow auto-compaction can't clear,
  model-unavailable) drops the goal — the owner then re-pastes the `/goal` line. Expect this occasionally over weeks.
  This is a legitimate reason to "speak" (fits the owner's model).
- **Research ≠ compile-until-green.** Some dimensions (D5 feel-expression, D6 learn-grow at scale, D3 full one-brain)
  are open problems that may take a long time or yield honest NO-GOs. The condition counts banked-NO-GO-with-next-
  method as progress so the evaluator does not call "impossible" prematurely — but the owner should expect the metric
  to move in steps, not monotonically.
- **The weak evaluator is why §6 leans on the gates.** If the gates were ever bypassed, the /goal loop could drift
  into overclaiming. The gates are AUTHORITATIVE and `--no-verify` is banned — that is the guardrail.
- **Weeks of autonomy still benefits from the light morning touch** the owner already wants: a 30-second glance at the
  flagged owner-forks turns a potential multi-day stall into a redirect.
