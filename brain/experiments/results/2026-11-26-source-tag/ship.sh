#!/bin/bash
# W side: ship brain/ (HEAD, clean) to pool40 and start. bash ship.sh smoke | full   (full chains s319 -> s320 -> s321)
W=/home/dant123/Projects/sim/_scratch/wt-mixday; cd $W || exit 1
[ -z "$(git status --porcelain -- brain/)" ] || { echo "brain/ dirty: commit first"; exit 1; }
H=$(git rev-parse HEAD); E="ssh -o IdentitiesOnly=yes"; export SSH_AUTH_SOCK=; M=$1; N=pool40; D=brain-lane/route_repo/_scratch/route/kr
rsync -a --delete --exclude experiments/results --exclude __pycache__ -e "$E" brain/ $N:brain-lane/route_repo/brain/ || exit 1
echo $H | $E $N "cat > ~/brain-lane/route_repo/COMMIT" || exit 1
rsync -a -e "$E" brain/experiments/results/2026-11-26-source-tag/run_node.sh $N:$D/st_run_node.sh || exit 1
if [ "$M" = smoke ]; then
  $E $N "(nohup bash ~/$D/st_run_node.sh 319 smoke --smoke >/dev/null 2>&1 &); sleep 5; cat ~/$D/st_done_smoke_s319.txt" </dev/null
else
  $E $N "(nohup bash -c 'for s in 319 320 321; do bash ~/$D/st_run_node.sh \$s full; done' >/dev/null 2>&1 &); sleep 5; cat ~/$D/st_done_full_s319.txt" </dev/null
fi
