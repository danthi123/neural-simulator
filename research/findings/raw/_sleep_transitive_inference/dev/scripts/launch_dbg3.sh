#!/usr/bin/env bash
D=/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/sti
cd /home/dant123/Projects/sim/.claude/worktrees/sleep-transitive-inference
seed=$1; shift
for spec in "$@"; do
  set -- $spec; tag=$1; shift
  (SIM_BACKEND=numpy timeout 7200 /home/dant123/Projects/sim/.venv/bin/python $D/dbg3.py $tag $seed "$@" 2>&1 | grep --line-buffered "^\[" > $D/dbg3_$tag.log) &
done
wait
