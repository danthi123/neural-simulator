# Tier-3 Option 2 'develop-with-a-body' — a brain DEVELOPS over DAYS from LIVED experience

**2026-06-30 (CYCLE 737-739, autonomous loop; the scoping-recommended + controller-verified second Tier-3 slice).**
The second artificial-life-capstone synthesis slice: a persistent merged one-brain that DEVELOPS over multiple days,
where each day's knowledge is **LIVED** (perceived + grounded during a foraging day) rather than a scripted
curriculum, RETAINING old lived facts as new days add more (no-forgetting), and PERSISTING the developed brain
across a reset. **Seed-42 (smoke) full GO — all 8 checks; the full 6-seed is [PENDING — bg bowg8ddn5, ETA ~2 h;
verdict `research/findings/raw/_tier3_develop_with_a_body.json`].** Runner:
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

## The eight gates (seed-42 smoke, n_days=3, on the real merged bridge, GPU)
| gate | result | evidence |
|---|---|---|
| **develops over days** | GO | facts/day **[1, 2, 3]** (knowledge grows from experience) |
| **retention / no-forget** | GO | last-day recall **3/3** (day-0's fact still recalled after 2 more days) |
| **frozen-brain flat** | GO | `commit_facts=False` (sees + grounds but does NOT store) → **0 facts, recall 0/0** |
| **lived, not scripted** | GO | a **permuted-world** control yields a DIFFERENT fact set (memory tracks the lived layout) |
| **no-confab MOAT** | GO | `(obj,"chase")` abstains every day (**2/2, 3/3, 4/4**); conversational synapses **byte-frozen** |
| **persistence across reset** | GO | reload resumes **3/3**; no-persistence cold-start **0/3** |
| **alive** | GO | survived every day (drive keeps energy in-band) |
| **drive-is-spiking** | GO | corr(deficit, `drive_agrp` firing) **+0.92** |

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
6-seed for the develops/retention robustness claim (PENDING). Honest scope stated up front.

## Verdict
The seed-42 smoke is a full 8/8 GO — a brain that develops over days from LIVED experience, retaining old lived
knowledge, moat intact, NO `sim/` edit. **[6-seed robustness PENDING — bowg8ddn5.]** On a 6/6 GO this closes the
second Tier-3 synthesis slice (the merged one brain now not only lives + remembers, but *develops over time from
what it lives*). The ranked follow-ons remain (Option 3 cross-modal one-animal; Option 4 lived consolidation;
Option 2B the 24/7 harness; the richer-world upgrades).
