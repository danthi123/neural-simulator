# C3 curiosity learning-progress — methods 2 and 3 (driver proposal, 2026-10-18)

Source: one MiMo Pro delegate run (RULING models), read-only over the record; driver persisted it
for NEEDS `c3-methods`. Priority: RULING priorities-1016 (3), dev seeds 1-8, default off, in parallel.

## Method 2 — "topic inputs, background noise off" (NOT runnable today)

- Runner: `research.runners._curiosity_lp_trace_derisk` (branch `claude-build/curiosity-lp-derisk`;
  plan `docs/drafts/curiosity-lp/PLAN.md`).
- "Topic inputs": already the module's presentation — the topic's slice of `topic_aff` is direct-driven
  at `novelty*WORD_DRIVE` (`_curiosity_lp_trace_derisk.py:423-428`).
- "Background noise off": `cfg.enable_ou_process = False` (the habituation organ's setting,
  `spiking_novelty_habituation_organ.py:113`; the 2026-06-03 item-8 fix,
  `2026-06-03-content-selection-milestone2-seed-robustness-RESOLVED.md:74-77`).
- Gap: the bridge builder leaves the CoreSimConfig OU default ON (per-neuron `ou_seed` at
  `:290-291`; the `:289-313` block never sets `enable_ou_process`), and the CLI (`:813-822`) has no
  OU flag. That OU background firing through the 400 ms gaps is the recorded NO-GO cause
  (`PLAN.md:194-196`).
- Minimal fix (additive, default-off, module-only, no `sim/` edit, frozen `W_TOPIC_ASK=80` and all
  bars/arms kept): ONE flag `--no-ou` setting `cfg.enable_ou_process = False` in `_build_bridge`
  (in-repo pattern: `_emerge_wkv_tokensdr_writefidelity_derisk.py:75` argparse, wired at `:40`).

  ```bash
  .venv/bin/python -u -m research.runners._curiosity_lp_trace_derisk \
      --battery --no-ou --seeds 1 --out research/findings/raw/_curiosity_lp/m2_dev/s1.json
  ```
- Until `--no-ou` lands, the only runnable variant (same command minus the flag) reproduces the
  method-1 NO-GO — must NOT be re-queued.

## Method 3 — per-topic learning-progress SLOPE gate (new default-off runner)

The brake becomes a per-topic LP slope instead of method 2's exposure-count depression trace. Same
bridge/presentation/battery; per topic, progress = spiking `reward_us->snc` DA phasic above SNc tonic
after a teach turn (the DR-1 read already reused, `PLAN.md:46-47`) plus the want read off the spiking
ASK pool; slope = fast minus slow
(`2026-08-02-laneB-curiosity-learning-progress-slope-CPU-proxy-6seed-GO-next-onbridge-realization.md:31-43`).
A topic whose slope stays ~0 (probed, never answered) accumulates that finding's omission veto, which
raises its slice->ASK threshold: asked on the first probe (B1/B1b), silent by the 4th (B2); a
positive-slope topic is veto-protected (slow-but-improving keeps asking); an answered topic reads
familiar and is never presented again (B3).

- Biology: learning progress as intrinsic reward (Oudeyer/Kaplan/Hafner 2007; Gottlieb et al. 2013;
  Kidd & Hayden 2015); infants withdrawing from unlearnable sequences (Kidd et al. 2012); midbrain DA
  signalling information arrival (Bromberg-Martin & Hikosaka 2009) — `PLAN.md:58-61`.
- Why not method 2: STP depression counts PRESENTATIONS and is blind to progress — a topic being
  learned is silenced exactly like one yielding nothing (the identifiability limit the
  `2026-08-02-laneB-curiosity-reserve-rescue-REFUTED` finding names).
- Why not the existing plan: `PLAN.md:33` declined the phasic-minus-tonic EMA as the trace (host numpy
  EMA, 1/6 on-bridge promotion); method 3 takes the record's other arm — the spiking DA read plus the
  veto gate (`2026-08-07-laneB-curiosity-learning-progress-MAXIMIZING-selection`) — with fast/slow
  state a DECLARED host scaffold in the same class as the module's own DA-gated restore
  (`PLAN.md:79-81`), rung: spiking uptake pools.
- Shape: new `_`-prefixed module `research/runners/_curiosity_lp_slope_gate_derisk.py`, never imported
  by `webapp/` or the production organ, no edit of `curiosity_production_organ.py` or `sim/`, dev
  seeds 1-8 only. Same bars B1-B9 with the slope/veto lesion (gate removed) as the counted lesion arm
  replacing `lesion_habituation` (B5 clause retargets to the slope lesion).

  ```bash
  .venv/bin/python -u -m research.runners._curiosity_lp_slope_gate_derisk \
      --battery --seeds 1 --out research/findings/raw/_curiosity_lp_slope/m3_dev/s1.json
  ```

## Queue mechanics (dev seeds 1-8)

Dev seeds need no prereg and no pool canary (`tools/pool_queue.sh` demands `prereg <path>` in the
reason for gate seeds only; canary gate covers gate-seed lines). Provision first:
`touch research/queue/provision.req/<full-40-hex-sha>` of the commit carrying the build (the
`--no-ou` flag; the new module). Then one `pool_queue.sh add` line per seed, shape:

```bash
bash tools/pool_queue.sh add 'cd ~/derisk-pool/revisions/<sha> && mkdir -p <outparent> \
  && bash tools/mem_ok.sh 9 && bash tools/memcap.sh 9 -- env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 .venv/bin/python -u -m <runner> --battery [--no-ou] --seeds N --out <out>' \
  --checked 'priorities-1016 (3) C3 LP method {2|3} ... dev seed N/8, no prereg (dev seed); mem_gb=9 max_age_h=48'
```

Queue AFTER the priorities-1016 (1)-(2) lines (RULING pool-order-1013); lane-stock-1016's B-Curiosity
smoke (ldc-gpe-tonic outcome-timing pilot) is a different line and stays first smoke.

## NEEDS (Claude, §2)

Claude registers method 3's slope-gate bars (B5 lesion clause retargets to the slope/veto lesion) and
confirms m2 keeps the frozen `W_TOPIC_ASK=80` with the one `--no-ou` flag; per `judgment-to-claude-1016`,
is a per-method scorer needed, or do the `PLAN.md:129-151` bars read the per-seed `--battery` json as-is?