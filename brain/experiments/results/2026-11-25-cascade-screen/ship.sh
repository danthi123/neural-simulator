#!/bin/bash
# W side: ship brain/ (HEAD, clean) to each node and start the cascade screen, one seed per node.
W=/home/dant123/Projects/sim/_scratch/wt-mixday; cd $W || exit 1
[ -z "$(git status --porcelain -- brain/)" ] || { echo "brain/ dirty: commit first"; exit 1; }
H=$(git rev-parse HEAD); E="ssh -o IdentitiesOnly=yes"; export SSH_AUTH_SOCK=
for x in pool40:319 pool41:320 pool42:321; do N=${x%:*}; S=${x#*:}; D=brain-lane/route_repo/_scratch/route/kr
  rsync -a --delete --exclude experiments/results --exclude __pycache__ -e "$E" brain/ $N:brain-lane/route_repo/brain/ || exit 1
  echo $H | $E $N "cat > ~/brain-lane/route_repo/COMMIT" || exit 1
  rsync -a -e "$E" _scratch/route/casc/run_node.sh $N:$D/casc_run_node.sh || exit 1
  $E $N "(nohup bash ~/$D/casc_run_node.sh $S >/dev/null 2>&1 &); sleep 5; cat ~/$D/casc_done_s$S.txt" </dev/null
done
