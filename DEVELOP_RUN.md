# 3-Day Self-Driven Develop Run

Run the artificial-life brain-development loop unattended (no Claude needed). The brain
hears the **TinyStories** corpus day by day, grows its vocabulary + facts (with zero
catastrophic forgetting — sleep replay consolidates each day), and saves a per-day "brain"
you can chat with.

## TL;DR

```powershell
.\scripts\develop.ps1 start      # start (or resume) -- leave it running
.\scripts\develop.ps1 status     # day / vocab / facts (safe anytime, no GPU)
.\scripts\develop.ps1 pause      # stop cleanly at the next day boundary
.\scripts\develop.ps1 resume     # continue from where it stopped
```

That's the whole workflow. Everything below is detail.

## What it does

Each simulated "day": the brain HEARS corpus windows (its Hebbian synapses learn the day's
word meanings) → CONVERSES (stores the day's facts) → SLEEPS (replay consolidation, which
*prevents* forgetting) → GROWS (tier promotion) → PERSISTS to disk. A few minutes per day on
the 3090; over 3 days it covers years of simulated development.

Everything lives under `bridges\developed\run3day\`:
- `lineage\` — the persistent brain state ("resume" continues from here)
- `bundles\day_<N>\` — a self-contained brain snapshot for day N (chat with it; see below)
- `PAUSE` — a sentinel file; when present, the run stops at the next day boundary

## Start / pause / resume / stop

- **Start / resume** — `.\scripts\develop.ps1 start` (or `resume`). Auto-detects the saved
  state and continues; the first run begins at day 0. Runs in the foreground, printing
  per-day progress.
- **Pause** — `.\scripts\develop.ps1 pause` (from another terminal). The run finishes the
  current day, persists it, and stops. **Zero completed work is lost.** Pressing **Ctrl-C**
  in the run window does the same.
- **Resume** — `.\scripts\develop.ps1 resume`. Removes the pause and continues from the last
  completed day.
- **Status** — `.\scripts\develop.ps1 status`. Prints day / vocab / facts. Uses the CPU (no
  GPU), so it's safe to run anytime — even while the GPU run is going.

### Run it in the background (close the terminal)
```powershell
Start-Process pwsh -ArgumentList '-NoProfile','-File','scripts\develop.ps1','start'
```
Then `pause` / `status` from any terminal.

### Gaming / need the GPU back
`.\scripts\develop.ps1 pause` — it stops at the next day boundary (a minute or two), freeing
the GPU. When you're done, `.\scripts\develop.ps1 resume`. No VRAM is held while paused.

## Chat with the developing brain

Each day is saved as a bundle under `bridges\developed\run3day\bundles\day_<N>\`. Load one in
the **webapp dashboard's brain picker** (the Interact tab lists developed-brain bundles) and
chat with the brain at that stage. Compare an early day vs a later day to *hear* it develop —
more words, more facts, steadier answers.

(The `first_chat_console` CLI loads a single `--brain <file.npz>`; the day bundles are
directories, so the dashboard picker is the way to load them.)

## Tuning (optional)

Extra flags after `start` pass straight through to the runner:
```powershell
.\scripts\develop.ps1 start --max-windows-per-day 4000   # hear more corpus per day (slower, richer)
```
- `--max-windows-per-day` (default 2500) — more = the brain hears more corpus each day.
- `--corpus-path PATH` — a different plain-text corpus shard (default: TinyStories).

## If something goes wrong

- "PAUSE file present, not starting" → run `resume` (or delete
  `bridges\developed\run3day\PAUSE`).
- Check progress without disturbing the GPU run → `status` (CPU-only).
- A crash mid-day loses only the in-progress day; `resume` continues from the last completed
  (persisted) day — the lineage is written at the end of every day, before the pause check.

## Under the hood

`scripts\develop.ps1` → `research/runners/develop_run.py` → the validated `develop_gpu`
day-loop (`research/runners/_longitudinal_develop_loop_gpu.py`). The wrapper only adds
persistence (a stable lineage), the pause sentinel, and per-day bundles — no change to the
simulator or the validated loop.
