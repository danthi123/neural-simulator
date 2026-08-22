# FOUR-DAY AUTONOMOUS QUEUE — loaded 2026-08-22, HARVEST at the Tuesday 2026-08-26 ~11:00 EST usage reset

This is the durable, compaction-proof record of the headless work loaded to run on FREE (non-Claude) compute over
the ~4 days while the weekly Claude usage limit is spent down. It cost zero agent tokens to run. Read this file
first at the Tuesday harvest, then act on the HARVEST CHECKLIST at the bottom.

> **Author's note (verify-first):** every runner here was `--help`-verified to import + parse, and the flagship
> (#75 vision R-STDP) + both knowledge runners were smoke-run end-to-end before queueing. Output-flag mismatches
> were caught and fixed (`--json` vs `--out`) before commit. Nothing here is fabricated; a job that still fails at
> runtime logs `DONE(rc!=0)` and the dispatcher moves on.

## ⚠️ RESTOCK 2026-08-22 ~11:45 EDT — the first queue drained in ~2h; now AUTO-REFILLING (no-drain guaranteed)

The initial 22 de-risks were 30–180 s each (small networks), so the queue cleared by ~05:48 and the GPU idled.
Fixed:

- **AUTO-REFILL ENQUEUER** — `tools/gpu_queue_autofill.sh` + `gpu-queue-autofill.timer` (system, every 15 min).
  When `gpu.queue` drops below 8 it ENQUEUES a fresh batch of genuine LONG jobs (the old `gpu-queue-refill.timer`
  only *restarted* the dispatcher — that is why it stayed idle). This makes wall-clock prediction unnecessary: the
  GPU cannot sit idle more than ~15 min before it is refilled, at any job speed. Verified firing (cycles 1–2 auto).
  Check: `tail research/queue/autofill.log`; `systemctl list-timers gpu-queue-autofill`.
- **Backbone = the LONGITUDINAL CONTINUOUS-LIFE loop** (`_longitudinal_develop_loop_gpu --n-days 60`, fresh seed
  each cycle) — the owner's #1 strategic priority (make the brain CONTINUOUS), and genuinely long: ~50 s/day
  steady-state (brain plateaus) ⇒ ~1.5–1.7 h per loop, and it is **crash-RESUMABLE** (resumes on reboot). Each
  refill cycle adds 3 such loops + a `persistent_living_loop_derisk` soak + a vision #75 cell ≈ ~3 h of GPU work.
  The autofill ALSO tops up the pool (5 CPU lanes) each cycle.
- **Wall-clock:** at restock the queue held 13 jobs (~8 longitudinal loops @ ~1.7 h + persistent/vision/order/gap2
  ≈ 15–20 h queued), and the autofill adds ~3 h whenever depth < 8. Net: **self-sustaining until the Tue reset.**
- **15k knowledge core soak came back `go: True` (6/6 seeds, 0 byte-identity mismatches, 0 confab)** — the curated
  bundle is a proven no-regression `BRAIN_LTM_BUNDLE`; the default-on flip is the owner/harvest call (step 3 below).
- **Fixed the 3 rc=1 failures** (were open-wall de-risks): `_replay_cortical_consolidation_gate_v6_order_stdp`
  (#130) is SEED-LOCKED → re-queued at its CALIBRATION seeds `412 413`; `_laneC_source_monitor_competitive_encoding_gate`
  (#129) is seed-locked AND its cupy path has a non-scalar-fill bug → re-staged on the POOL (numpy) at its DECISIVE
  seeds `700–705`; `_gap2_spiking_deltarule_binder_derisk` (#132) hard-coded a missing `scale787` codes path →
  added `--codes-path` + recursive fallback + `--out` verdict, verified `go=True`, re-queued 6-seed.
- **Knowledge 100k build** launched detached (`sim-data/knowledge_bundles/wikidata_100k`, log `build100k.log`) —
  a real 100k-fact wikidata bundle + reduced soak; the O(V·D) latency-at-scale follow-on to the 15k core.
- **Still DEFERRED to the harvest** (were not tractable to add safely this pass): the gap5 self-ignition
  DECOUPLED-store next rung (needs the igniting config read from `gap5_ignition_sweep_6seed.json`), and the
  DA-encoding on-substrate spiking synaptic-scaling rule (a new runner to build). Both are noted in step 5.

## 0. WHERE THE CODE + STATE LIVES (read this before anything else)

The **primary checkout `/home/dant123/Projects/sim` is on the STALE branch `research/gap4-axon-capd-derisk`**
(1146 commits behind `main`; it carries 163 real uncommitted files — DO NOT `git checkout`/`reset` it, DO NOT
sweep those into a commit). All current work lives on `main`. This session did its build/queue work from a
dedicated worktree so the queued jobs run CURRENT code:

| thing | location | branch |
|---|---|---|
| primary checkout (queue files, systemd services read here) | `/home/dant123/Projects/sim` | `research/gap4-axon-capd-derisk` (stale) |
| **run worktree** (all queued GPU jobs `cd` here; new runners live here) | `/home/dant123/Projects/sim-worktrees/four-day-queue` | `research/four-day-autonomous-queue` (off `main`) |
| this doc + the new runners (pushed to origin+gitea) | on `research/four-day-autonomous-queue` | commits `c62dc386…` onward |
| curated knowledge bundle (OUT of repo, persists) | `/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k` | n/a |
| raw wikidata5m source | `/home/dant123/Projects/sim-data/wikidata5m` | n/a |

The run worktree has a `.venv` symlink to the primary `.venv` (jobs use the absolute path
`/home/dant123/Projects/sim/.venv/bin/python` anyway). `sim`/`research` resolve from the worktree (CWD), verified.

## 1. CRASH-RESILIENCE (TASK 1 — done + verified)

The 3090 falls off the bus under load (reboot-only). Everything below survives a crash→reboot unattended:

| unit | kind | role | check | restart |
|---|---|---|---|---|
| `gpu-queue-dispatch.service` | system, enabled, Restart=always | runs `tools/gpu_queue.sh __daemon`; resumes `research/queue/gpu.queue` on boot | `systemctl status gpu-queue-dispatch` | `sudo systemctl restart gpu-queue-dispatch` |
| `gpu-queue-refill.timer` | system, every 10 min | idempotent `systemctl start` of the dispatcher (belt-and-suspenders vs a stopped/limit-exhausted daemon) | `systemctl list-timers gpu-queue-refill` | `sudo systemctl start gpu-queue-refill.timer` |
| `pool-dispatch.service` | user, enabled, **linger=yes** | feeds `research/queue/pool.queue` to idle pool nodes; now survives reboot (linger enabled this session) | `systemctl --user status pool-dispatch` | `systemctl --user restart pool-dispatch` |
| `gpu-train-watchdog.service` | system (pre-existing) | reboots the box on GPU-off-the-bus; respects the lmtrain `PAUSE` sentinel | `systemctl status gpu-train-watchdog` | (leave as-is) |

- Unit files are version-controlled at `tools/systemd/` (on `main` + this branch). Reinstall = `sudo cp
  tools/systemd/gpu-queue-*.{service,timer} /etc/systemd/system/ && sudo systemctl daemon-reload`.
- **Restart-safety guard** added to `gpu_queue.sh`: the daemon WAITS on a still-alive `$RUNNING` pid instead of
  double-starting a second GPU job after a crash+Restart or a stop+restart. Verified: `kill -9` MainPID →
  systemd restarted it in ~18s, the in-flight job survived, no double-start.
- **lmtrain is PAUSED** (`bridges/lmtrain/run3/PAUSE` present); watchdog + `lm_train_run start` both respect it, so
  a reboot does NOT resume LM training and does NOT contend for the GPU. Leave the PAUSE in place during the window.
- `gpu_queue.sh status` shows current job + depth + dispatcher state. `gpu_queue.sh pause --now` / `resume` for gaming.

## 2. KNOWLEDGE CORE BUNDLE (TASK 2 — owner #1 priority; BUILT, soak in flight)

The knowledge-scale infra is already GREEN on `main` (TieredFactStore + `BRAIN_LTM_BUNDLE`, byte-identical, scales
to 100k). The only open piece was WHICH bundle ships (owner-UX). Built a **curated CORE** (not the raw 5M dump),
per the owner guard "depth not breadth, a brain you communicate with, not a fancy plastic RAG":

- **Runner:** `research.runners._knowledge_core_curate` — from wikidata5m keep facts whose subject+object are the
  top-connected entities and whose relation is top-frequency → a dense shared-vocab core; entities mapped to a
  cruft-rejecting cleanest-alias token; **genuine resonate bind (fast=False)**; persisted via
  `ShardedPhasorStore.save`.
- **Built artifact:** `/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k/` — **15,000 facts,
  vocab 7,032, 75 shards, ship_ready=True** (report in `<bundle>/curation_report.json` +
  `research/findings/raw/_knowledge_core/curate_report.json`).
- **Soak (gates the flip):** `research.runners._knowledge_core_bundle_soak --bundle <dir>` → 6-seed byte-identity
  vs the plain-flat unsharded oracle + no-confab moat + first-match recall, PLUS the real production load path
  `load_developed_brain(ltm_bundle=<dir>)`. Verdict → `research/findings/raw/_knowledge_core/core_bundle_soak_verdict.json`
  (in the run worktree). Smoke (300 facts) was GO 2/2, 0 mismatches, moat clean, ~60ms latency, prod-path GO.
- Build+soak launched detached; combined log `/home/dant123/Projects/sim-data/knowledge_bundles/build_soak.log`.
- **The default-on FLIP is intentionally NOT done** (owner-UX call, per the task) — see the harvest checklist.
- **Caveat for the harvest:** vocab 7,032 may put warm recall latency around ~0.5–1.5 s (the O(V·D) codebook
  cleanup grows with vocab; the flip-soak finding notes >~20k entities exceeds 1 s). For a snappier default,
  rebuild with `--top-entities 4000 --n-facts 12000` (→ ~3–4k vocab, sub-second). The current bundle is a valid
  v1; latency is a UX tuning knob, NOT a correctness gate (the soak GO is byte-identity/moat/recall).

## 3. DEEP GPU + POOL DE-RISK QUEUE (TASK 3)

### GPU queue (`research/queue/gpu.queue`, sequential, one 3090 job at a time; jobs `cd` the run worktree, cupy)
Outputs land in `sim-worktrees/four-day-queue/research/findings/raw/four_day/`. A job is a GO per its own Verdict
(`go: true` in the JSON) / the runner's printed GO gate.

| lane | runner | jobs | what a GO looks like |
|---|---|---|---|
| **Vision #75** R-STDP sparse readout | `_vision_rstdp_readout_derisk` | 2D op-point sweep: n_s2 ∈ {24,32,48,64,96,128,192,256}@ep30, {48,64,96,128}@ep60, {64,128}@ep100 — 14 cells, 6-seed each → `vrstdp_ns2<N>_ep<E>_6seed.json` | learned spiking readout held-decode clears the NOGO floor (~0.34) AND beats the random-readout control (learning load-bearing on spikes) at 6/6 |
| **Memory** self-ignition | `_gap5_dg_detonator_ignition_derisk` | 1 (6-seed) → `gap5_dg_ignition_6seed.json` | DG detonator ignites pattern completion at 6/6 (the D5 s102 self-ignition floor) |
| **Memory** ignition sweep | `_gap5_ignition_sweep_probe` | 1 (6-seed) → `gap5_ignition_sweep_6seed.json` | maps the k-of-N / formation-floor operating point |
| **Memory** order consolidation #130 | `_replay_cortical_consolidation_gate_v6_order_stdp` | 1 (6-seed) → `replay_v6_order_stdp_6seed.json` | order-sensitive replay consolidation beats the order-shuffled control 6/6 |
| **Source-monitoring #129** | `_laneC_source_monitor_competitive_encoding_gate` | 1 (6-seed, `--json`) → `laneC_source_competitive_6seed.json` | source margins clear AND the preregistered no-harm control passes on all seeds |
| **Binder #132** | `_gap2_spiking_deltarule_binder_derisk` | 1 (6-seed, prints to gpu_queue.log — no output flag) | spiking delta-rule binder recalls role→filler above control 6/6 |

Plus 3 pre-existing jobs at the queue head from the **DA-encoding lever2 agent** (its own worktree
`scratchpad/wt-lever2`) — do NOT touch; that agent owns them.

### Pool (`research/queue/pool.queue`, parallel across pool40/41/42, numpy CPU) — source-monitoring mechanism compare
Genuinely-open lane (board CURRENT STATE: v2 competitive cleared margins but 1 seed failed the no-harm control).
Staged the OTHER mechanism variants at 6-seed (competitive is on the GPU): `_laneC_source_monitor_attractor_competition`,
`_..._attractor_joint`, `_..._conjunctive_tag`, `_laneC_plastic_source_memory_derisk`. These are FAST (~30–60 s
each, 65k-neuron sims) so the pool drains quickly then idles — restock at harvest if you want more CPU work.
**Pool outputs live ON THE NODES** at `~/derisk-pool/sim/research/findings/raw/four_day/`; retrieve with:
`for n in pool40 pool41 pool42; do rsync -e ssh $n:derisk-pool/sim/research/findings/raw/four_day/ research/findings/raw/four_day/; done`
(the `_laneC_..._competitive` one writes with `--json`). Note: `tools/pool_queue.sh` validates against the primary
checkout (stale) so it REFUSES current runners — stage via the WORKTREE copy
`sim-worktrees/four-day-queue/tools/pool_queue.sh` (its ROOT has the code; queue path is the shared absolute one).

## 4. HARVEST CHECKLIST — Tuesday 2026-08-26 ~11:00 EST (do these in order)

1. **Re-anchor:** read this file, then `GAP_CLOSURE_MISSION.md` CURRENT STATE + the MASTER ROADMAP. Note the
   primary checkout is on the stale gap4 branch; current work is on `main` + `research/four-day-autonomous-queue`.
2. **Resilience sanity:** `systemctl status gpu-queue-dispatch gpu-train-watchdog`; `systemctl --user status
   pool-dispatch`; `bash tools/gpu_queue.sh status`. If the box rebooted, confirm the dispatcher auto-resumed the
   queue (it should have). Check `/var/tmp/gpu_train_watchdog/` for a `STRANDED` marker (genuine dead GPU).
3. **Knowledge bundle (owner #1):** read `research/findings/raw/_knowledge_core/core_bundle_soak_verdict.json`
   (in the run worktree). **If `go: true`** → the curated core is a proven no-regression `BRAIN_LTM_BUNDLE`.
   Decide with the owner whether to ship it as the default; the FLIP is to default `BRAIN_LTM_BUNDLE` to the bundle
   dir in `webapp/server.py` (currently unset→off, byte-identical), keeping unset→off as the escape. Confirm the
   recall latency in the verdict is acceptable; if not, rebuild with `--top-entities 4000` (see §2 caveat) and
   re-soak. Write the finding (status live) + sync the board (#133) + the ledger (`tiered-knowledge-ltm`).
4. **GPU sweep results:** read `research/findings/raw/four_day/vrstdp_*.json`. The best (n_s2, epochs) cell that
   clears the NOGO floor AND beats the random-readout control at 6/6 CLOSES board #75 (fully-spiking object
   "which" readout via R-STDP). If none clears it, the sweep MAPS the operating point (still a deliverable).
   Write the finding; if GO, that's the fully-spiking vision readout — wire toward production.
5. **Other GPU lanes:** read `gap5_dg_ignition_*`, `gap5_ignition_sweep_*`, `replay_v6_order_stdp_*`,
   `laneC_source_competitive_*`, and the gap2 binder result (in `research/queue/gpu_queue.log`, grep the DONE line).
   Each GO advances its board item (#71 self-ignition→6/6, #130 order consolidation, #129 source monitoring, #132
   binder). Promote 3-seed indicators that came back clean to findings.
6. **Pool results:** rsync the pool outputs (§3 command). The source-monitoring mechanism that clears margins AND
   the no-harm control across 6 seeds is the one to build on for #129.
7. **Provenance:** every runner auto-stamps argv/SHA/env sidecars; trust but spot-check.
8. **Restock** the GPU queue + pool if compute time remains (`tools/gpu_queue.sh add`, worktree `pool_queue.sh add`).
