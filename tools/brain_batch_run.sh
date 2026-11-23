#!/usr/bin/env bash
# brain_batch_run.sh -- run a list of brain/ experiment jobs one at a time on a compute node (a mini-PC or an AWS GPU
# instance); used by tools/aws_brain_batch.sh and by hand on the mini-PCs (owner 2026-11-02: new-core work only).
# Layout on the node: ~/brainv2/{code/brain, models/*.pth, jobs.txt, results/}. jobs.txt: one job per line,
#   <name> <brain.experiments.sleep_probe args...>          (the default module), or
#   <name> brain.<module> <args...>                         (any brain/ module that takes --out and EXTRA's args)
# Paths in args are relative to ~/brainv2/code (e.g. --model ../models/<file>.pth, --home ../work/<name>). Lines
# starting with # are skipped. A job whose result exists (<name>.json, or a non-empty <name>.json.json: brain.eval.life
# writes <out>.json, refreshed after every day) is skipped, so a rerun never restarts it from scratch; a part-done life
# is resumed only by a job line with --resume. PY picks the interpreter,
# EXTRA adds args to every job (e.g. "--device cuda"). If ~/brainv2/env.sh exists it is sourced first.
cd ~/brainv2/code || exit 1
# ~/brainv2/env.sh (written by aws_brain_batch.sh launch: AWS_AMI_ID, AWS_BRAIN_ENV) is exported for every job.
[ -f ~/brainv2/env.sh ] && . ~/brainv2/env.sh
PY="${PY:-$HOME/brainv2/venv/bin/python}"
while read -r name args; do
  [ -z "$name" ] || [ "${name#\#}" != "$name" ] && continue
  mod=brain.experiments.sleep_probe
  case "$args" in brain.*) mod=${args%% *}; args=${args#"$mod"} ;; esac
  out=~/brainv2/results/$name.json
  { [ -s "$out" ] || [ -s "$out.json" ]; } && continue
  echo "$(date '+%F %T') start $name"
  # shellcheck disable=SC2086
  "$PY" -u -m "$mod" $args ${EXTRA:-} --out "$out" > ~/brainv2/results/"$name".log 2>&1
  rc=$?   # captured first: inside the echo, $(date) would reset $? to 0 (every job logged rc 0, 11-14)
  echo "$(date '+%F %T') end $name rc $rc"
done < ~/brainv2/jobs.txt
echo "$(date '+%F %T') all done"
