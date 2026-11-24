#!/bin/bash
# W side: ship brain/ (HEAD, clean), the probe, the banked life JSONs and the final M1 trunk homes to pool40 and start.
W=/home/dant123/Projects/sim/_scratch/wt-mixday; cd $W || exit 1
E="ssh -o IdentitiesOnly=yes"; export SSH_AUTH_SOCK=
D=brain-lane/flip_repo/_scratch/flip/cascade
$E pool40 "mkdir -p ~/$D/homes/s319 ~/$D/homes/s320 ~/$D/homes/s321" </dev/null || exit 1
rsync -a --delete --exclude experiments/results --exclude __pycache__ -e "$E" brain/ pool40:brain-lane/flip_repo/brain/ || exit 1
rsync -a -e "$E" _scratch/flip/cascade/probe3.py _scratch/flip/cascade/run_pool.sh _scratch/brainv2/aws/20261123-1552/lifeM1_s319.json.json _scratch/brainv2/aws/20261123-1632/lifeM1_s320.json.json _scratch/brainv2/aws/20261123-1713/lifeM1_s321.json.json pool40:$D/ || exit 1
for x in 1552:319 1632:320 1713:321; do
  rsync -a --delete -e "$E" _scratch/brainv2/aws/20261123-${x%:*}/work/lifeM1_s${x#*:}/trunk/ pool40:$D/homes/s${x#*:}/trunk/ || exit 1
done
$E pool40 "cd ~/brain-lane/flip_repo && du -sh _scratch/flip/cascade/homes/*; uptime; (nohup bash _scratch/flip/cascade/run_pool.sh 319 320 321 >/dev/null 2>&1 &); sleep 45; cat _scratch/flip/cascade/done.txt; tail -5 _scratch/flip/cascade/probe_s319.log; pgrep -af probe3.py | cut -c1-100" </dev/null
