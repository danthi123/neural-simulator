#!/bin/bash
# W side: ship brain/ (HEAD, clean), the 11-25 replay output and the final trunk per node; start the ceiling run.
W=/home/dant123/Projects/sim/_scratch/wt-mixday; cd $W || exit 1
[ -z "$(git status --porcelain -- brain/)" ] || { echo "brain/ dirty: commit first"; exit 1; }
E="ssh -o IdentitiesOnly=yes"; export SSH_AUTH_SOCK=
for x in pool40:319:1219 pool41:320:1624 pool42:321:1704; do IFS=: read N S B <<<"$x"
  D=brain-lane/route_repo/_scratch/route/kr; A=_scratch/brainv2/aws/20261124-$B
  rsync -a --delete --exclude experiments/results --exclude __pycache__ -e "$E" brain/ $N:brain-lane/route_repo/brain/ || exit 1
  rsync -a -e "$E" _scratch/route/ceil/run_node.sh brain/experiments/results/2026-11-25-routing-screen/replay_s$S.json $N:$D/ || exit 1
  rsync -a --delete -e "$E" $A/work/lifeKR_s$S/trunk/ $N:$D/trunk_s$S/ || exit 1
  $E $N "(nohup bash ~/$D/run_node.sh $S >/dev/null 2>&1 &); sleep 5; cat ~/$D/ceil_done_s$S.txt" </dev/null
done
