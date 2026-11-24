#!/bin/bash
cd /home/dant123/Projects/sim/_scratch/wt-mixday || exit 1
E="ssh -o IdentitiesOnly=yes"; export SSH_AUTH_SOCK=
D=brain-lane/flip_repo/_scratch/flip/cascade
rsync -a -e "$E" _scratch/flip/cascade/probe3.py pool40:$D/ || exit 1
$E pool40 "cd ~/brain-lane/flip_repo && pgrep -f probe3.py; rm -f $HOME/nothing; rm -f _scratch/flip/cascade/done.txt; (nohup bash _scratch/flip/cascade/run_pool.sh 319 320 321 >/dev/null 2>&1 &); sleep 100; cat _scratch/flip/cascade/done.txt; tail -5 _scratch/flip/cascade/probe_s319.log; pgrep -af probe3.py | cut -c1-100" </dev/null
