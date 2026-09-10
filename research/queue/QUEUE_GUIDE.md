# Research queue — owner controls (start / pause / stop / top-up)

**Purpose (2026-09-10).** A deep queue of **non-Claude compute** you start and control yourself — zero Claude usage
while it runs. It fills two independent lanes and is built so a gaming pause wastes almost no compute.

- **GPU lane** (local 3090) — *pausable for gaming.* One job at a time, VRAM-contention-safe.
- **Pool lane** (mini-PCs, CPU) — *runs straight through gaming, never needs pausing.*

Every queued job is a **single self-contained unit** (one seed, or one seed's full scaling curve) that writes its own
result file. So pausing kills only the **one** unit in flight (losing at most its progress), never a whole batch — and
that killed unit is automatically re-queued to re-run on resume.

---

## The three commands you need

```bash
# ▶ START the queue (do this once, when you want compute running)
bash tools/gpu_queue.sh start && bash tools/game.sh off
```

```bash
# ⏸ PAUSE for gaming (frees the 3090 immediately; the pool keeps running)
bash tools/game.sh on
```

```bash
# ▶ RESUME after gaming (the GPU picks the re-queued job back up)
bash tools/game.sh off
```

`game.sh on` also unloads anything holding the card and sets a flag that survives reboot — so if you reboot mid-break
it stays paused. `game.sh off` clears it. **Nothing auto-starts; you are always in control.**

---

## Check status any time

```bash
bash tools/game.sh status        # pause state + VRAM + is the pool running
bash tools/gpu_queue.sh status   # GPU: current job, queue depth, dispatcher up/down
bash tools/pool_queue.sh list    # pool: depth + contents
```

## Top up when it runs low

```bash
bash tools/stock_research_queue.sh   # re-stage; skips every cell already run (safe to re-run)
```

If a pool node is unreachable after a reboot, re-provision it once: `bash tools/pool_provision.sh` (then top up).

## Fully stop (end of the week)

```bash
bash tools/game.sh on            # pause everything GPU
bash tools/gpu_queue.sh stop     # stop the dispatcher (queue file is preserved for later)
```
The pool keeps draining its queue on its own; leave it, or `pkill -f pool_autodispatch` to halt it too.

---

## What's queued (the science, in priority order)

**GPU lane**
1. **gap#4 decisive** — the in-engine self-predicting-interneuron microcircuit, 6 single-seed runs (the verdict you've
   been waiting on; may legitimately come back UNDEFINED under the interpretability gate — that's a valid result).
2. **Mouth token-scaling sweep** (your decided #1 mouth fork) — pushes token supply into the large-token "does the
   curve bend?" regime on the now-local **fineweb_edu** (quality corpus) + **wikitext103**, across model sizes and all
   6 seeds. The prior run was flat only because it never left the data-starved region; this is the real test.

**Pool lane**
3. **Affect-opponent competition-strength gain sweep** — Rank-7's own named next rung (the gains were never tuned for
   this convergence; literature says specialization reliability tracks competition strength). CPU, runs during gaming.

Results land in `research/findings/raw/gap4/`, `research/findings/raw/_gencortex_scaling/`, and
`research/findings/raw/_affect_gain_sweep/` — one JSON per cell. When you're back with Claude, "continue" and the
per-cell artifacts get aggregated + written up.
