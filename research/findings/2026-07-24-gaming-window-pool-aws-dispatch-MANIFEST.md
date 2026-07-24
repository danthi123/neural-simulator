# Gaming-window dispatch MANIFEST — pool + AWS runs live, independent of the dispatching session (2026-07-24 ~00:20 EDT)

**Purpose:** the owner rebooted the main box (CachyOS → Windows) to game for ~2–4 hr. Runs were dispatched to the
**mini-PC pool** + **AWS** so they run WITHOUT this session. This doc lets ANY fresh session collect + continue.
On the owner's "I'm back", run the COLLECT commands below, then the PENDING full-workflow tasks.

## Access (both survive the reboot)
- **Pool:** `ssh pool40` / `pool41` / `pool42` (aliases in `~/.ssh/config`: user `node`, key `~/.ssh/id_ed25519`).
  Remote repo: `derisk-pool/sim/`. Results under `derisk-pool/sim/research/findings/raw/`.
- **AWS:** keys PRESERVED to `~/.ssh/aws-train/aws-267m-run4.pem` + `aws-162m-run5.pem` (the `/tmp` originals are wiped
  by the reboot). 267M `ubuntu@54.88.59.2` (run4_d2048); 162M `ubuntu@3.239.187.212` (run5_d1536). Instance IDs
  `i-0c7c01b8409ac6463` / `i-045aaf797ab8329b9` (`aws ec2 describe-instances`).

## What's running on the POOL (3 nodes × 12 cores, numpy, detached via setsid/nohup)
Each node runs `pool_queue.sh` (in `derisk-pool/sim/`): WAVE1 = gap#4 deconf (2 seeds, ~2 min, FAST) → WAVE2 = gap#5
**ramp-phase-advance readout** 6-seed (the slow spiking frontier de-risk, the window-filler). pool42 also runs the
gap#5 completion probe (AXIS-B ff_basket).
- pool40 → seeds 42,43 ; pool41 → 44,100 ; pool42 → 101,102 + completion-probe.
- **Completion sentinel per node:** `derisk-pool/sim/research/findings/raw/QUEUE_DONE_<hostname>.txt`.
- **Outputs on the nodes:** deconf `research/findings/raw/gap4/pool_deconf_s<seed>.json(.log)`; ramp
  `research/findings/raw/gap5_r4/pool_ramp_s<seed>.json(.log)`; probe `.../gap5_r4/pool_completion_ff20.json`.

### COLLECT from the pool (run on return)
```bash
cd ~/Projects/sim
for h in pool40 pool41 pool42; do
  rsync -az "$h:derisk-pool/sim/research/findings/raw/gap4/pool_deconf_s*.json" research/findings/raw/gap4/ 2>/dev/null
  rsync -az "$h:derisk-pool/sim/research/findings/raw/gap5_r4/pool_ramp_s*.json" research/findings/raw/gap5_r4/ 2>/dev/null
  rsync -az "$h:derisk-pool/sim/research/findings/raw/gap5_r4/pool_completion_ff20.json" research/findings/raw/gap5_r4/ 2>/dev/null
  ssh $h "ls -la derisk-pool/sim/research/findings/raw/QUEUE_DONE_* 2>/dev/null; tail -2 derisk-pool/sim/research/findings/raw/queue.log"
done
```
Known state at dispatch: deconf **4/6 collected GO-with-nuance** (see below); ramp running all 3 nodes; pool42 deconf
101,102 + probe were re-launched last (verify they wrote `.json`, not just `.log` — a stale-code failure writes a
`.log` with `No module named ...` and an instant QUEUE_DONE; if so, re-sync that node and re-run its queue).

## What's running on AWS (survives reboot; tmux session `train` on each instance)
- **267M** run4_d2048: at dispatch step ~99,500 / val_ppl ~59.1 / 815M tok. **162M** run5_d1536: step ~62,500 /
  val_ppl ~67.6 / 512M tok. Both climbing.
- **Check:** `ssh -i ~/.ssh/aws-train/aws-267m-run4.pem ubuntu@54.88.59.2 "ls -t ~/sim/bridges/lmtrain/run4_d2048/*.log | head -1 | xargs tail -3"` (same for 162m with its key + `run5_d1536`).
- **Collect a checkpoint:** `rsync -az -e "ssh -i ~/.ssh/aws-train/aws-267m-run4.pem" ubuntu@54.88.59.2:~/sim/bridges/lmtrain/run4_d2048/ bridges/lmtrain/run4_d2048/` (the local mirror was stale at 10:13; the rsync-back loop may have stopped — pull fresh on return).
- NOTE: if these are SPOT instances they can be reclaimed; `describe-instances` first.

## PENDING full-workflow tasks (do on return, after collecting)
1. **gap#4 de-confounded credit — write the finding (the headline is NUANCED, not a blanket GO).** Pattern across
   seeds 42/43/44/100 (+ collect 101,102): at **full data (frac=1.0) bdsp ≈ reservoir** (seed 44 frac1.0 GO=False:
   bdsp 0.793 vs RES 0.795); at **scarce data (frac 0.10, 0.05) directed credit BEATS the frozen reservoir** every
   seed (bdsp>res True), with ALL de-confound controls passing (sel>0.20, bdsp>shufE, shufY<chance, permB). ⇒ the
   directed-credit advantage is **DATA-EFFICIENCY (low-data regime)**, robust to the input-selectivity +
   credit-scramble + shuffled-target confounds. Honest, not overstated. Commit + push both remotes.
2. **gap#5 ramp-phase-advance readout — read the 6-seed verdict** (`pool_ramp_s*.json`). This is the NEW Buzsáki-ramp
   mechanism (see `2026-07-23-gap5-phase-precession-research-gate-Buzsaki-ramp-mechanism.md`): does the forward chain
   occupy monotonically-advancing theta phases (order 0<1<2), with shuffled-store / reverse-cue / basket-off
   collapsing? GO → write finding + scale; NEGATIVE → it's a METHOD verdict (per THE LAW), bank + next method.
3. **gap#5 completion probe (AXIS-B ff_basket)** — read `pool_completion_ff20.json`.
4. **AWS ladder** — pull fresh checkpoints; the 267M val_ppl ticked 58.5→59.2 (watch for plateau/overfit vs run3's ~55).

## Already committed THIS session (context)
- Curiosity on-bridge `from_novelty` 6-seed CPU GO (`27edcf08`); the `from_novelty` sim edit (`25f23162`).
- CLAUDE.md trim 3767→1550 + When-Compacting (`dd60d0cf`); local-RAG/catalog path repair (`ee1a20d0`);
  adversarial-verify finding corrections (`d1a5633e`); gap#5 phase-precession research gate.

## Anti-footgun notes learned this session
- **NEVER `pkill -f <pattern>`** where the pattern appears in the running command — it self-matches + kills the shell
  (exit 144, happened twice). Kill by explicit PID.
- **rsync excludes must be inline-quoted args** (`--exclude '.venv*'`), NOT a shell variable with globs (it mangles →
  `rsync syntax error` → the node runs stale code → `No module named ...` → instant fake "QUEUE_DONE").
- The pool `node` account = user `node` (not `derisk`, which is a placeholder in dispatch.sh's header comment).
