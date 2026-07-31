---
type: finding
status: live
date: 2026-07-31
mechanism: curiosity-seek-learn
runner: research/runners/_curiosity_seek_learn_onbridge_derisk.py
artifacts:
  - research/findings/raw/lanes_2026-07-31/AGG_curiosity_critic_lesion_6seed.json
---

# The curiosity veto survives its critic lesion at 6/6 seeds — so the striosome is not what computes it

**An anti-cheat that was listed and never run.** DR-1 curiosity is banked 6/6 GO on both backends, and its
finding lists `critic_lesion` under ANTI-CHEATS as "(reported)" — then never reports it. It had **zero hits
across every `_curiosity_*.json` on record**. This runs it.

## Result

With the GABA_B striosome critic lesioned, the noisy-concept veto **still fires on 6 of 6 seeds**
(`vetoed=True`, seeds 42/43/44/100/101/102). Aggregate:
`research/findings/raw/lanes_2026-07-31/AGG_curiosity_critic_lesion_6seed.json`, built through
`tools.verdict.Verdict` so it carries the preconditions that earned it, alongside the six per-seed logs it
reads. That is the opposite of the runner's own printed prediction.

## What it does and does not mean

**It does NOT refute the DR-1 GO.** The capability — asking about what can be learned and declining
unlearnable noise — still holds. What moves is the account of *which component produces it*.

**It localises the veto away from the striosome.** A lesion that removes the value-subtraction pathway
entirely leaves the veto intact at every seed, so the striosome rate is not the quantity the veto is
reading. The DR-1 finding's own honest-scope note already said as much in passing — that the veto is a
**host-side ELP tracker**, not the striosome rate — and this measures it rather than leaving it as a
caveat. The two readings agree, which is the reassuring outcome; had the veto collapsed, the caveat would
have been wrong.

**The consequence is a brain-based-only one.** Under the project's standing bar, a veto computed by a host
tracker is a documented shortcut, not a faculty: the *brain* is not doing it, the simulation's bookkeeping
is. This result promotes that from a caveat inside a GO to a named, measured shortcut with its own
conversion target — a spiking mechanism that computes the same decision from the striosome (or from
whatever the substrate can actually supply) rather than from a Python running estimate.

## Also confirmed, same batch

False-belief at `helper_pa 4000` returned **GO at 6 seeds** — the midpoint of an axis whose endpoints were
both banked (5000 → GO 6/6, 3000 → PARTIAL 4/6), which would mean the GO has **margin** rather than sitting
knife-edge on the drive.

**That result is HELD, not banked, and deliberately.** `gates/verdict_preconditions` — built hours earlier
today — refused its artifact: it asserts `GO` while carrying no `preconditions` block, so nothing records
what earned it. The gate is right, and overriding it on the day it was written would make it decorative. I
could not honestly reconstruct the preconditions after the fact either; that is precisely the fabrication
the block exists to prevent. It stands re-runnable under `tools.verdict.Verdict`, and the artifact stays in
`g5s_out/` until then.

## Honest scope

Six seeds, numpy backend, single lesion arm. This says the striosome is not load-bearing *for the veto*; it
says nothing about the striosome's role elsewhere in DR-1, and nothing about whether a spiking replacement
for the host tracker is achievable — only that one is owed.
