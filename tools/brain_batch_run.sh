#!/usr/bin/env bash
# brain_batch_run.sh -- run a list of brain/ experiment jobs one at a time on a compute node (a mini-PC or an AWS GPU
# instance); used by tools/aws_brain_batch.sh and by hand on the mini-PCs (owner 2026-11-02: new-core work only).
# Layout on the node: ~/brainv2/{code/brain, models/*.pth, jobs.txt, results/}. jobs.txt: one job per line,
#   <name> <brain.experiments.sleep_probe args...>          (the default module), or
#   <name> brain.<module> <args...>                         (any brain/ module that takes --out and EXTRA's args)
# Paths in args are relative to ~/brainv2/code (e.g. --model ../models/<file>.pth, --home ../work/<name>). Lines
# starting with # are skipped. A job whose result exists is skipped, so a rerun resumes. PY picks the interpreter,
# EXTRA adds args to every job (e.g. "--device cuda").
cd ~/brainv2/code || exit 1
PY="${PY:-$HOME/brainv2/venv/bin/python}"
while read -r name args; do
  [ -z "$name" ] || [ "${name#\#}" != "$name" ] && continue
  mod=brain.experiments.sleep_probe
  case "$args" in brain.*) mod=${args%% *}; args=${args#"$mod"} ;; esac
  out=~/brainv2/results/$name.json
  [ -s "$out" ] && continue
  echo "$(date '+%F %T') start $name"
  # shellcheck disable=SC2086
  "$PY" -u -m "$mod" $args ${EXTRA:-} --out "$out" > ~/brainv2/results/"$name".log 2>&1
  echo "$(date '+%F %T') end $name rc $?"
done < ~/brainv2/jobs.txt
echo "$(date '+%F %T') all done"
