# GPU crash — diagnosis, recovery, and what Claude changed (2026-07-22)

**Symptom (recurring):** the RTX 3090 "falls off the bus" mid-training — `torch.AcceleratorError: CUDA error:
unspecified launch failure` (cudaErrorLaunchFailure); `nvidia-smi` then reports *"No devices were found"*.

## Diagnosis (2026-07-22 crash, ~step 626000)

- **Kernel log root cause:** `NVRM: GPU0 _scrubWaitAndSave: Timed out ... [NV_ERR_TIMEOUT]` +
  `API_GPU_ATTACHED_SANITY_CHECK failed`. The GPU **hung at the firmware/core level** (memory-scrub jobs timed out; the
  driver's attached-sanity check fails). This is NOT a driver glitch — the GPU core stopped responding.
- **PCIe state:** the device stays enumerated (`lspci` still shows the 3090) but `current_link_speed = Unknown`,
  `current_link_width = 63` (garbage = config-space reads failing) — the PCIe link is down.
- **⇒ For THIS class (hung core, sanity-check failing), a REBOOT is required.** Module reload / PCIe remove-rescan cannot
  revive a wedged core, and `lact` holding `/dev/nvidia*` open would make a `remove` HANG. Do not attempt remove/rescan
  while lactd is running.

## Likely cause of the RECURRENCE (the preventable part)

`lactd` (LACT GPU control tool, enabled service) was enforcing **power_cap = 390 W** on the 3090 (above the 350 W stock)
and re-applying it every 5 s. A 3090 at 390 W under sustained training load — with GDDR6X transient power spikes + VRAM
heat that the *edge*-temperature fan curve doesn't see — is a classic cause of repeated "hang under load". 

## What Claude changed (2026-07-22, reversible)

1. **Lowered the LACT power cap 390 → 300 W** (`/etc/lact/config.yaml`, the `power_cap:` line only; fan curve untouched).
   Backup at `/etc/lact/config.yaml.bak-preclaude-2026-07-22`. Takes effect on the next boot (lactd re-applies on start).
   - **Revert:** `sudo cp /etc/lact/config.yaml.bak-preclaude-2026-07-22 /etc/lact/config.yaml && sudo systemctl restart lactd`
   - **Tune:** edit the `power_cap:` value (e.g. 320 or 350) — raise if stable and you want the perf back; lower if it
     still crashes. A 3090 at 300 W loses ~10 % perf vs 350 W but is much more stable under sustained load.
2. **Installed + enabled `lmtrain-resume.service`** (`/etc/systemd/system/`) — on boot it waits ≤150 s for the GPU, then
   runs `lm_train_run resume --root bridges/lmtrain/run3`, retrying on failure (≤4×/30 min so no infinite crash-loop),
   logging to `bridges/lmtrain/run3/boot_resume.log`. So the training auto-resumes after any reboot.
   - **Disable auto-resume:** `sudo systemctl disable --now lmtrain-resume.service`
   - **Watch it after boot:** `tail -f bridges/lmtrain/run3/boot_resume.log` or `systemctl status lmtrain-resume`
   - **Manual pause/resume still works:** `lm_train_run pause --root bridges/lmtrain/run3` (+ `systemctl stop
     lmtrain-resume` so it doesn't restart), then `resume`.

## Recovery procedure

- **This crash (hung core):** reboot. The checkpoint is safe + bit-exact resumable; `lmtrain-resume.service` resumes it
  automatically on boot. (Claude rebooted the machine on 2026-07-22 after committing everything.)
- **A future crash that is only a driver glitch** (rare — GPU still responds to `nvidia-smi` but a process wedged):
  `tools/gpu_recover.sh` attempts a no-reboot recovery (stop lactd → free /dev/nvidia* → reload nvidia modules → restart
  lactd). It REFUSES to run if the core is hung (sanity-check failing), where only a reboot helps.

## 2026-09-26: it did crash with the 300 W cap in force

Same signature (`_scrubWaitAndSave ... [NV_ERR_TIMEOUT]`, then `GPU lost from the bus`) at ~11:07, during sustained
local-LLM inference (Qwen3.8-27B, 128K context, ~23 GB resident; generation is memory-bandwidth bound, which heats the
GDDR6X far more than the core). The journal kept only the last ~40 s before shutdown: the NVRM error flood rotated the
earlier entries out, so the first failure was not recorded. LACT fan control was OFF (`fan_control_enabled: false`).
Mitigations: owner applied `sudo nvidia-smi -lgc 210,1800` (verified: core pinned at 1800 MHz under load; lost on
reboot). Then, persistently in LACT (verified in /etc/lact/config.yaml, re-applied by lactd at boot): locked core
clocks 210-1800 MHz, fan control ON with LACT's default curve (30% at 40 C, 50% at 60 C, 75% at 70 C, 100% at 80 C,
keyed on the GPU edge temperature), power cap 300 W. LACT also reads the VRAM temperature: if it passes ~95 C under
local-LLM load, add a -500 MHz VRAM offset. Still optional: a +100 MHz core offset under the 1800 lock (mild
undervolt) once stable, and a larger journald size (`SystemMaxUse`). If it crashes again: the LTS kernel, then the PSU.

## If it STILL crashes after the 300 W cap (points to deeper cause — needs your hands-on attention)

1. **Undervolt** (best 3090 stability fix): set a VF curve in LACT (e.g. ~1800–1900 MHz @ ~875–900 mV) — less heat + less
   power at the same clocks. Needs care; do it interactively in the LACT GUI.
2. **VRAM thermals:** the fan curve keys on `edge` temp; GDDR6X junction can be 30–40 °C hotter. Consider a more
   aggressive fan curve or check memory-junction temp under load.
3. **Kernel/driver:** currently `linux-cachyos 7.1.3` + `nvidia-open 610.43.03`. The LTS kernel
   (`linux-cachyos-lts 6.18.38`, also installed) + its nvidia-open is often more stable for GPU compute — try booting it
   if crashes persist.
4. **Hardware:** repeated hangs under load can also be PSU (3090 transient spikes) or a degrading card — if software
   mitigations don't hold, suspect power delivery.
