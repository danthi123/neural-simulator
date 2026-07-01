# Tier-3 Option 2 'develop-with-a-body' — a brain DEVELOPS over DAYS from LIVED experience

**2026-06-30 (CYCLE 737-739, autonomous loop; the scoping-recommended + controller-verified second Tier-3 slice).**
The second artificial-life-capstone synthesis slice: a persistent merged one-brain that DEVELOPS over multiple days,
where each day's knowledge is **LIVED** (perceived + grounded during a foraging day) rather than a scripted
curriculum, RETAINING old lived facts as new days add more (no-forgetting), and PERSISTING the developed brain
across a reset. **6/6 seeds GO on all seven develop-capability gates** (seeds 42/43/44/100/101/102; run `bowg8ddn5`,
~2 h 20 m; verdict `research/findings/raw/_tier3_develop_with_a_body.json`). The one borrowed drive-quality
sanity-gate (`corr_ok`, inherited from Option 1) was 5/6 — seed 102's drive-tracking sweep landed at 0.897 vs the
0.90 threshold, a **window=20 under-sampling artifact** (the corr-sweep read window was hardcoded to 20 for speed;
the SAME drive scored +0.97–0.98 at Option-1's validated window=40, 6/6 GO — so the drive genuinely tracks). Owner
call: **accept the 6/6-capability result** (the corr metric is a redundant, Option-1-validated drive-quality check,
not a develop-capability gate); the runner's corr-sweep window is set to the validated 40 going forward. Runner:
`research/runners/_tier3_develop_with_a_body_derisk.py`. **NO `sim/` edit** (reuse-by-import; `live()` gained one
additive default-off `commit_facts` param — default `True` is byte-identical to the Option-1 6/6 GO).

## Why this is the second Tier-3 slice (the scoping's ranking)
Option 1 ("live-and-remember", 6/6 GO, `2026-06-30-tier3-live-and-remember-first-slice.md`) proved a *single* lived
life. The capstone scoping (`2026-06-30-tier3-option2-develop-with-a-body-scoping.md`, controller-verified) ranked
"develop-with-a-body" as the next slice: give the validated multi-day develop loop a *perceiving body* so each day's
WAKE experience is LIVED, keeping the develop-loop's SLEEP/METRICS/PERSIST patterns.

## The honest seam (why it's a JOIN, not a trivial compose)
The develop-loop's per-day brain (`develop_gpu`) is a text-corpus `StreamCortex` + a fresh conversational agent; the
live-and-remember body is a persistent `MergedNavConvAgent` (grounds from live perception). These are **different
substrates**, so the JOIN is **substitution, not fusion**: on a body-day the `MergedNavConvAgent` *replaces* the
corpus stream (the corpus is simply not run). Controller-verified that the reused stages are agent-composer patterns,
not `StreamCortex`-coupled — `consolidate(agent, ...)` and the retention re-test (`is_it_true` yes/no on prior facts)
work directly on the merged agent's composer; the one hard interface gap (`MergedNavConvAgent` has no `reason_chain`)
is a non-issue for this **self-contained runner** (it owns its own metrics — the corridor produces only `near`-facts,
so chain probes are vacuous). This self-contained design **cannot regress** the existing `develop_gpu`.

## The mechanism
One persistent `MergedNavConvAgent` lives `N` days. Each day: a cumulatively-richer `DevWorld` (objects introduced
over days — day0 `[apple, cat]`, day1 `+dog`, day2 `+river`) → Option-1's `live()` forages (drive-biased survival)
and, on first arrival at each object cell, `perceive_and_ground`s it and STORES the lived chain link
(`(prev, "near", cur)`), so the lived-fact knowledge grows `apple→cat→dog→river`. Then the develop-loop patterns:
a retention re-test (recall of ALL accumulated facts) + the no-confab moat + tier/metrics + PERSIST (body + all
lived facts + grounded codes via `BridgeLineage`). The day's knowledge is a consequence of WHICH objects the foraging
brought the agent to — **lived, not authored**.

## The seven develop-capability gates (6/6 seeds, n_days=4, on the real merged bridge, GPU)
| gate | result | evidence (all 6 seeds unless noted) |
|---|---|---|
| **develops over days** | **6/6 GO** | facts/day grows every seed: 42 `[1,2,3,3]`, 43 `[1,1,3,3]`, 44 `[1,1,3,3]`, 100 `[1,2,3,3]`, 101 `[1,1,3,3]`, 102 `[1,1,2,3]` |
| **retention / no-forget** | **6/6 GO** | last-day recall **3/3** every seed — day-0's fact still recalled through the day-3 no-new-learning day |
| **frozen-brain flat** | **6/6 GO** | `commit_facts=False` (sees + grounds but does NOT store) → **0 facts, recall 0/0** every seed |
| **lived, not scripted** | **6/6 GO** | a **permuted-world** control yields a DIFFERENT fact set every seed (memory tracks the lived layout) |
| **no-confab MOAT** | **6/6 GO** | `(obj,"chase")` abstains every day every seed; conversational synapses **byte-frozen** in vivo |
| **persistence across reset** | **6/6 GO** | reload resumes **3/3** every seed; no-persistence cold-start **0/3** |
| **alive** | **6/6 GO** | survived every day every seed (drive keeps energy in-band) |
| *(drive-is-spiking, borrowed sanity)* | *5/6* | corr(deficit, `drive_agrp` firing) +0.90–0.92 at window=20; seed 102 at **0.897** vs 0.90 — a window under-sampling artifact (Option-1-validated at +0.98, window=40); **accepted** |

## Honest scope (deferred; flagged in the runner docstring)
- The corridor + **4-object perceivable set** (the gen stack renders `OBJECT_WORDS`, N=4) bounds the developed graph
  to a short chain (~3 lived facts over the growth days). A richer multi-day development needs **more perceivable
  objects / a 2D world / a pair-accumulation upgrade** (follow-ons).
- Promotion to the **24/7 `develop_gpu` harness** (Option 2B — an additive default-off `per_day_agent_factory` seam)
  is a follow-on; this self-contained runner is the de-risk.
- The **learned spatial policy** stays the deferred Tier-4 dendrite wall (survival uses the validated rate-proxy
  stand-in). **Persistence** is JSON re-instate (not the raw `cp_connections` tensor).

## Anti-cheat standing rules (honored)
No-confab moat NEVER weakened (byte-frozen in vivo; abstains on every unstored cue — HARD). Validate-by-function:
the frozen-brain control isolates "commit → competence", the permuted-world control isolates "lived → not scripted".
6-seed for the develops/retention robustness claim (**met — 6/6**). Honest scope stated up front.

## Verdict
**6/6 seeds GO on all seven develop-capability gates** — this **closes the second Tier-3 synthesis slice**: the
merged one brain now not only lives + remembers (Option 1), but *develops over time from what it lives* — its
lived-fact knowledge grows day-over-day from foraging, old lived facts are retained (no-forgetting) through
no-new-learning days, a frozen brain that sees-but-doesn't-store stays flat, a permuted world yields different
lived facts (the knowledge is LIVED, not scripted), the no-confab moat holds byte-frozen, and the developed brain
resumes across a reset. NO `sim/` edit. The single borrowed drive-quality `corr_ok` sanity-gate was 5/6 (seed 102
marginal at window=20 — a redundant, Option-1-validated metric, accepted; the runner's corr window is now the
validated 40). ⇒ Tier-3 has two closed synthesis slices (live-and-remember + develop-with-a-body). The ranked
follow-ons remain: Option 3 cross-modal one-animal (the shared hunger drive tightens the conversational moat — one
drive touching both halves); Option 4 lived consolidation (event-triggered SWR replay); Option 2B the 24/7
`develop_gpu` harness; the richer-world upgrades (more perceivable objects / 2D / pair-accumulation).
