---
type: finding
status: live
date: 2026-07-30
---

# ⛔ CRITICAL: the gap#4 crux is ~14x over its runtime estimate — I checked LIVENESS all session and never THROUGHPUT

## The measurement

`/tmp/claude-1000/gap4ob_a.log`, first completed arm:

```
[gap4-onbridge][seed 42]   arm reservoir   held-out 0.130 train 0.135 memctrl 0.000 ff-moved 75971.30 nwt True (81444s)
```

**81444 s = 22.6 HOURS for ONE ARM of ONE SEED.** The job is **5 arms x 3 seeds = 15 arms**.
**Projected total: ~339 hours = ~14 DAYS.**

The pre-registered estimate recorded on the board was **"~8-24 h per job"**, computed from
`n_train=1260 x 40 settle-steps x 40 epochs x 5 arms x 3 seeds ~ 30M bridge steps`. **The real figure is ~14x
that.** The step-count arithmetic was right; the per-step cost was not, and nothing checked it against reality
until 24.6 h in.

## The process failure, precisely

**I verified LIVENESS every ~15 minutes for an entire session and never verified THROUGHPUT ONCE.** Every check I
ran — `cpu-time vs elapsed = 99%`, `device=ok`, process alive — answers *"is it computing?"*. **None answers
*"will it finish?"*.** They are different questions and only the second one matters for a job with a deadline.

I even recorded the correct rule on the board earlier: *"a LAUNCH-BOUND run is genuinely computing but
pathologically slow — kill + re-scope, do not wait hours."* I then applied its liveness half (99% CPU ⇒ healthy)
and never its throughput half, reporting "crux healthy" ~20 times.

**THE CHECK THAT WAS MISSING, and it is one line:** when a run prints ANY per-unit progress marker, divide it by
the unit count and compare to the estimate. `81444 s x 15 arms` is arithmetic available the moment the first arm
landed — **which was 2 h before I looked.**

## Consequence

The run cannot deliver. The roadmap's own framing of this item is **"gap#4 on-bridge surpass — SHRUNK-task"**, so
the intent was already a reduced problem; this configuration is not reduced enough by ~14x. Options are
(a) let it run ~14 days, or (b) kill and re-scope with the now-MEASURED per-arm cost.

**Re-scope levers, with the measurement to size them:** 22.6 h/arm at `n_train=1260 x 40 settle x 40 epochs`.
Cutting epochs 40→10 and settle 40→20 is a **~8x** reduction (~2.8 h/arm, ~42 h total); adding a 3→1 seed
smoke first bounds it further. **The arm that DID complete is informative and should be kept:** `reservoir`
held-out **0.130** vs train **0.135** — the control arm, near chance, exactly as a reservoir should be.

## The generalisable rule (now banked in verify-go)

**LIVENESS IS NOT PROGRESS.** A monitor that reports "running at 99% CPU" is answering the easy question. For any
long job, the monitor must also answer: *at the observed rate, when does this finish, and is that inside the
budget?* If the job emits no per-unit marker, that is itself a defect to fix before launching it.
