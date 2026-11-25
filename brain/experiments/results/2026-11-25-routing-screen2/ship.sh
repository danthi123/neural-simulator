#!/bin/bash
# W side: ship brain/ (HEAD, clean) to each node (snapshots, trunks, life JSONs already there from the ceiling run); start.
W=/home/dant123/Projects/sim/_scratch/wt-mixday; cd $W || exit 1
[ -z "$(git status --porcelain -- brain/)" ] || { echo "brain/ dirty: commit first"; exit 1; }
E="ssh -o IdentitiesOnly=yes"; export SSH_AUTH_SOCK=
for x in pool40:319 pool41:320 pool42:321; do N=${x%:*}; S=${x#*:}; D=brain-lane/route_repo/_scratch/route/kr
  rsync -a --delete --exclude experiments/results --exclude __pycache__ -e "$E" brain/ $N:brain-lane/route_repo/brain/ || exit 1
  rsync -a -e "$E" _scratch/route/s2/run_node.sh $N:$D/s2_run_node.sh || exit 1
  $E $N "ls ~/$D/trunk_s$S/episodes | wc -l; (nohup bash ~/$D/s2_run_node.sh $S >/dev/null 2>&1 &); sleep 5; cat ~/$D/s2diag_done_s$S.txt" </dev/null
done
