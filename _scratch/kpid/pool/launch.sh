#!/bin/bash
# W side: ship brain/ (committed 9fd474dde), gate.py, drift_read.py and the lifeS1F JSONs to each pool, start one seed.
W=/home/dant123/Projects/sim/_scratch/wt-mixday; cd $W || exit 1
E="ssh -o IdentitiesOnly=yes"
go() { h=$1; s=$2; src=$3
  sha=$(sha256sum $src/cortex_adapters.pt | cut -d' ' -f1)
  SSH_AUTH_SOCK= $E $h "mkdir -p ~/brain-lane/kpid_repo/_scratch/kpid ~/brain-lane/kpid_repo/_scratch/stage2/read" </dev/null
  SSH_AUTH_SOCK= rsync -a --delete --exclude experiments/results --exclude __pycache__ -e "$E" brain/ $h:brain-lane/kpid_repo/brain/
  SSH_AUTH_SOCK= rsync -a -e "$E" _scratch/kpid/gate.py _scratch/kpid/drift_read.py _scratch/kpid/pool/run_pool_gate.sh $h:brain-lane/kpid_repo/_scratch/kpid/
  SSH_AUTH_SOCK= rsync -a -e "$E" _scratch/stage2/read/lifeS1F_s$s.json $h:brain-lane/kpid_repo/_scratch/stage2/read/
  SSH_AUTH_SOCK= timeout 30 $E $h "cd ~/brain-lane/kpid_repo && (nohup bash _scratch/kpid/run_pool_gate.sh $s $sha >/dev/null 2>&1 &); sleep 20; cat _scratch/kpid/pool_cpu/done.txt; pgrep -af gate.py | grep -v pgrep | cut -c1-90" </dev/null
}
go pool41 319 /home/dant123/Projects/sim/_scratch/kpgate_w/src_s319
go pool42 320 /home/dant123/Projects/sim/_scratch/kpgate_w/src_s320
go pool40 321 $W/_scratch/kpid/src_s321
