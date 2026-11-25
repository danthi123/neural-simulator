#!/bin/bash
# W side: ship brain/ (HEAD, clean) + memcap + one KR life and its night snapshots per node, start the replay.
W=/home/dant123/Projects/sim/_scratch/wt-mixday; cd $W || exit 1
E="ssh -o IdentitiesOnly=yes"; export SSH_AUTH_SOCK=
for x in pool40:319:1219 pool41:320:1624 pool42:321:1704; do IFS=: read N S B <<<"$x"
  D=brain-lane/route_repo/_scratch/route/kr; A=_scratch/brainv2/aws/20261124-$B
  $E $N "mkdir -p ~/$D ~/brain-lane/route_repo/tools" </dev/null || exit 1
  rsync -a --delete --exclude experiments/results --exclude __pycache__ -e "$E" brain/ $N:brain-lane/route_repo/brain/ || exit 1
  rsync -a -e "$E" tools/memcap.sh $N:brain-lane/route_repo/tools/ || exit 1
  rsync -a -e "$E" _scratch/route/kr/run_node.sh $A/lifeKR_s$S.json.json $N:$D/ || exit 1
  rsync -a --delete -e "$E" $A/work/lifeKR_s$S/nightsnap/ $N:$D/snaps_s$S/ || exit 1
  $E $N "cd ~/brain-lane/route_repo && (nohup bash ~/$D/run_node.sh $S >/dev/null 2>&1 &); sleep 5; cat ~/$D/done_s$S.txt" </dev/null
done
