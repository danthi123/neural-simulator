#!/bin/bash
# W side: ship brain/ (HEAD, clean) and start the voice ceiling. bash ship.sh smoke | full
W=/home/dant123/Projects/sim/_scratch/wt-mixday; cd $W || exit 1
[ -z "$(git status --porcelain -- brain/)" ] || { echo "brain/ dirty: commit first"; exit 1; }
H=$(git rev-parse HEAD); E="ssh -o IdentitiesOnly=yes"; export SSH_AUTH_SOCK=; M=$1
L="pool40:319 pool41:320 pool42:321"; [ "$M" = smoke ] && L="pool40:319"
for x in $L; do N=${x%:*}; S=${x#*:}; D=brain-lane/route_repo/_scratch/route/kr
  rsync -a --delete --exclude experiments/results --exclude __pycache__ -e "$E" brain/ $N:brain-lane/route_repo/brain/ || exit 1
  echo $H | $E $N "cat > ~/brain-lane/route_repo/COMMIT" || exit 1
  rsync -a -e "$E" brain/experiments/results/2026-11-26-voice-ceiling/run_node.sh $N:$D/vc_run_node.sh || exit 1
  if [ "$M" = smoke ]; then A="--nights 6 --arms T"; else A=""; fi
  $E $N "(nohup bash ~/$D/vc_run_node.sh $S $M $A >/dev/null 2>&1 &); sleep 5; cat ~/$D/vc_done_${M}_s$S.txt" </dev/null
done
