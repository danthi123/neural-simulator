#!/bin/bash
cd /home/dant123/Projects/sim/_scratch/wt-mixday || exit 1
export SSH_AUTH_SOCK=
C=brain-lane/flip_repo/_scratch/flip/cascade
ssh -o IdentitiesOnly=yes pool40 "cat ~/$C/done.txt; pgrep -af probe3.py | cut -c1-80" </dev/null
rsync -a -e "ssh -o IdentitiesOnly=yes" pool40:$C/out_s*.json pool40:$C/probe_s*.log pool40:$C/done.txt _scratch/flip/cascade/
