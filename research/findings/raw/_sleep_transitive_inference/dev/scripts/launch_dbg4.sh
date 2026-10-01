#!/usr/bin/env bash
D=/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/sti
cd /home/dant123/Projects/sim/.claude/worktrees/sleep-transitive-inference
for spec in "$@"; do
  set -- $spec; tag=$1; shift
  (SIM_BACKEND=numpy timeout 7200 /home/dant123/Projects/sim/.venv/bin/python $D/dbg4.py $tag "$@" 2>&1 | grep --line-buffered "^\[[a-z]" > $D/dbg4_$tag.log) &
done
wait
