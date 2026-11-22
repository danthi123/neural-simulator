# HANDOFF (overwrite each session; <= 40 lines)

- **TASK:** 12-14 milestone, multi-week life with nightly sleep (board #286). Design
  `brain/research/2026-11-19-multiweek-life-design.md` (§5 bars, §13 registration). Older state: `brain/experiments/results/`.
- **TYPE:** build · **DRIVER:** claude · **TIMEBOX:** Stage-2 methods by 12-08, 2 days each; read/land 12-10 to 12-14.
- **STATE:** Stage 1 read 11-21 (no capability passes; defects: language drift, echo, stock leak, never-told keys).
  D2-PR read 11-22: PR-l/PR-x UNDEFINED (engagement); next = D2-PR2 below.
- **LIVE: D2-PR2 DEV screen** (prereg `brain/research/2026-11-22-replay-priority-2-prereg.md`, built c91c0c200, `loss_a`
  = L_e**0.6): PR-a s319 pool41, s320 pool42 (same nodes as B, CPU fp32, started ~17:15 UTC 11-22, `outLIFE/lifePRa_s<S>`);
  s321 pool40 node-mxl1462drl (started by hand ~13:20 UTC 11-22; the watcher's pgrep matched its own ssh line). Read with `d2pr_bars.py` (results/2026-11-22-d2pr-dev-read-1.5b/) + §7 blind read. Lever 3 of 3.
- **K-P DEV READ DONE (11-22, `results/2026-11-22-kp-dev-read-1.5b.md`, 4 refuters): KEEP at DEV (gate-overridden).**
  Held-out KL n15 0.118x B (0.32 vs 2.72), NLL 2.21 vs 4.51; every numeric guard passes (day-1-2 window n5-15; n1-4 fails
  s320); leak 0.04 vs 0.29; recall up (once-told F 19/34 vs 7/34). COSTS to register as TEST bars: self-name "Dan"
  8/18 vs 0/18; stock-refusal collapse nights 59/1158 rows vs 0 (abstention rise: any flip needs a ruling). Device
  confound (K-P fp16 AWS, B fp32 pool) not explanatory per refuter; B-fp16 lives `lifeBg_s<S>` launching on AWS
  (`_scratch/stage2/launch_bg.sh`, state `.aws_brain.bg<S>`). Gate replications s319/s320/s321 repeat s316 (s320 0.85 vs B 2.93); W runs lifeKPc_s319 (K-P CPU fp32 life,
  `_scratch/kpcpu/`), closing the device confound on one seed.
- **BUILT (31de906ca, not frozen): TEST split v2** (design `brain/research/2026-11-22-fresh-test-split-design.md`, 4c9c0504f): age ladder
  (12 items/seed, 22 rows at age>=10), slot-matched never-told floor + FM-v2 bar, `--split test2`, `--age-ladder` OFF.
  Workflow builds validator, blind bank writer, code, opus review, fix (uncommitted in worktree until reviewed).
- **NEXT STEP:** (1) Land split v2 (review diff, hash-freeze bank + cues, one DEV 1.5B check). (2) Read B-fp16 vs K-P
  when AWS lands. (3) D2-PR2 read when PR-a lands. (4) Prereg the six-seed TEST on v2: B, K-P (+ D2-PR2 if it passes),
  with self-name and abstention bars named. (5) Label-shuffle null for `life_bars.py`.
- **OWNER WORDS (2026-11-02, verbatim):** "Do all, ensure Hermes and Pi also follow our new methodology. And one
  caveat, I may not noticeably correct you so I don't want that as a benchmark of improvement. All I really care
  about is us making meaningful progress on this project, which has stalled for way too long as I continue to spend
  money on AI to drive it."
- **OWNER WORDS (2026-11-20, verbatim):** "Looks like the GPU fell off the bus earlier? I won't be home to reboot until
  Monday so continue with CPU, pool, and AWS until then. The MTALLM Claude session also may need AWS for GPU access, I
  am preapproving that access as needed so don't ask for permission."
- **OPEN:** AWS cap for 11-22 only: $150, aim < $100 (owner); remove after 11-22 to restore $50 (drop-in `~/.config/systemd/user/aws-guard.service.d/cap-override.conf`). Split v2 built (31de906ca), not frozen: DEV row check `lifeAL_s322` on AWS (`launch_al_now.sh`), then hash-freeze. W GPU off the bus until Mon 11-23 (CPU, pools, AWS only; AWS shared with MTALLM, keep cap headroom). s102 g6
  i-008ed35fe857dce55 STOPPED (state `research/queue/.aws_brain.s102-stopped-20261117`), retry pull then terminate. RAG
  index stale.
- **FORCED SHUTDOWN 11-22 ~17:50 UTC (display dead).** On return:
  (1) remove `~/.config/systemd/user/aws-guard.service.d/cap-override.conf` ($150 was 11-22 only) + daemon-reload;
  (2) pull AWS (instances self-stop when done; results stay on disk): `AWS_BRAIN_STATE=research/queue/.aws_brain.<x>
  bash tools/aws_brain_batch.sh pull` for x = bg320, bgr321, al322 (bg319 already pulled, batch 20261122-0832).
  bgr321 and al322 share local batch dir 20261122-1027 (same-minute launch): pull each explicitly, check both JSONs.
- **AFTER THE REBOOT (owner runs `bash ~/Projects/sim/_scratch/gpu-unlock-go.sh`, then powers on):** check
  `journalctl -k -b | grep gpu_unlock`, nvidia-smi, one gpu_queue daemon only (two = card off the bus), llama-swap VRAM;
  then the K-P identity gate (prereg `brain/research/2026-11-23-kp-identity-prereg.md`, 59881a228) on GPU fp32:
  `tools/gpu_queue.sh add 'DEVICE=cuda bash _scratch/wt-mixday/_scratch/kpid/run_gate.sh > _scratch/wt-mixday/_scratch/kpid/run_gate.log 2>&1'`;
  restart the attention watcher; resume or stop `_scratch/kpcpu/run.sh` (K-P CPU fp32 life s319, resumable).
- **DELEGATED:** Pi: lane items. Hermes: [weekly] is Claude-on-AWS while W is dedicated.
