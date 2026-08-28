---
type: finding
status: negative
date: 2026-08-28
verdict: After the cupy build-crash fix, the one-brain d6-WM->comprehension cross-edge production-default FLIP re-verify is STILL NO-GO (FLIP_VERIFY_GO=False) — but for a precise, addressable reason, not the earlier crash. arm_A flag-off outputs exactly match baseline (n_match=4/4, exact compare) + arm_C no-regression PASS (moat held, learn_wiring_live=True) + the crash-fix WORKED (the shared pool now primes on cupy; visibility improved 0/4 -> 2/4 grown-focus seeds; seed42 shipped-visible; lesion-attributable=1.0, all-seeds-lesion-revert=True). arm_B visible-on-real-traffic FAILS: n_visible_grown_focus=2 of 4, n_hollow=2. The residual is a POSITIONAL-BINDING mismatch: the real handler's WM focus is the positional CAND_POOLS[0]=w0, so a seed's edge-drive is visible ONLY when w0 happens to be a GROWN role for that seed (true for 2/4 grown-focus seeds). BRAIN_ONEBRAIN_XEDGE stays default-OFF; the next lever aligns the handler focus-position with the grown-role position.
mechanism: 6-seed cupy re-verify of BRAIN_ONEBRAIN_XEDGE flip through the real handler, AFTER the build-crash fix (a8c8a2d18)
lane: onebrain-integration-xedge-flip
artifacts:
  - research/findings/raw/_xedge_flip_verify/flip_verify_cupy_6seed_strengthened.json
runner: research/runners/_xedge_flip_production_verify.py
---

# One-brain xedge flip re-verify (post crash-fix) — still NO-GO, but the residual is a precise positional-binding mismatch, not hollowness

Artifact: `research/findings/raw/_xedge_flip_verify/flip_verify_cupy_6seed_strengthened.json` (6 seeds 42/43/44/100/101/102, `SIM_BACKEND=cupy`, `b_edge=learn`, through the REAL `/api/brain-chat` handler on the production tiny-demo brain, AFTER the `_snapshot_rest` host->device build-crash fix).

## What the crash-fix settled (it was necessary + real)

The prior "hollow" NO-GO ([`2026-08-28-onebrain-xedge-production-default-flip-NO-GO`](2026-08-28-onebrain-xedge-production-default-flip-NO-GO.md)) was an INSTRUMENT ARTIFACT — a cupy build crash disabled the shared pool (`primed=false`), so `n_visible=0` measured a crash. This re-verify, with the fix, confirms the fix worked:

- **arm_A byte-identical-off: PASS** — with the flag OFF the handler outputs EXACTLY MATCH baseline on an exact per-turn compare (`n_match=4/4`, `diffs=[]`).
- **arm_C no-regression: PASS** (`moat_well_held_ok=True`, `learn_wiring_live=True`).
- The pool now PRIMES on cupy; `flip_fraction_attributable_to_crossedge=1.0`, `all_seeds_lesion_revert=True`; the shipped seed 42 is visible. Visibility rose from **0/4 -> 2/4** grown-focus seeds.

## Why it is STILL NO-GO — the positional-binding residual

- **`FLIP_VERIFY_GO = False`** because **arm_B visible-on-real-traffic FAILS**: `n_visible_grown_focus=2` of 4 (`per_seed` visible = `[T,F,F,T,F,F]` for seeds 42/43/44/100/101/102), `n_hollow=2`.
- Root cause (artifact's own note): through the real handler the WM focus is the POSITIONAL `CAND_POOLS[0]=w0`. The edge-drive is visible on a seed only when `w0` is a GROWN role for that seed — true for 2 of the 4 grown-focus seeds (42, 100; where `w0`=agent), false for the other 2 (their grown role sits at a different candidate position). So the drive IS present and lesion-attributable; it just doesn't land on the position the handler reads for 2/4 seeds. This is a POSITIONAL-BINDING mismatch, NOT a fundamental hollowness (the edge is genuinely load-bearing when the positions align).

## Decision + next lever (NO-DEFER)

Per the owner's standing rule (flip default-ON only on a genuine non-hollow GO), `n_hollow=2` and `n_visible_grown_focus=2/4` mean the flip is NOT applied — `BRAIN_ONEBRAIN_XEDGE` + `_LEARN` stay DEFAULT-OFF. This is NOT a retraction: the cross-edge is load-bearing + lesion-attributable + moat-safe + byte-identical-off; only the production-default flip is NO-GO, now for a precise reason. NEXT LEVER: align the handler's WM focus-position with the grown-role position — either route the edge drive through the role the handler actually reads (`content_role`/`net_lean`) regardless of which candidate POSITION grew, or make the WM-focus selection follow the grown role rather than the fixed positional `CAND_POOLS[0]`. Only re-attempt the flip once all 4 grown-focus seeds show `visible=True` (`n_hollow=0`) through the real handler. (My earlier VOID-banner optimism that a clean GO "looks likely" was premature — the crash-fix was necessary but not sufficient; the positional residual is the real remaining work.)

## Sources (external mechanism for the next lever)

(Kriete, Noelle, Cohen & O'Reilly 2013, PNAS; PMID 24062434; full DOI in the external-search log) — position-invariant variable binding via PFC/basal-ganglia **INDIRECTION**: a role-filler binding is gated through an indirection pointer rather than tied to a fixed slot/position. This is the biological template for the residual here — the WM held-referent should bind to the comprehension ROLE via an indirection/gating step so the edge-drive lands on the role the handler reads, INVARIANT to which candidate position (w0/w1/w2) the grown role occupies. That converts the 2/4 positional coincidence into a position-invariant read.
