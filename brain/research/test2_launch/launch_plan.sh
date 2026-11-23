#!/usr/bin/env bash
# TEST v2 launch plan (prereg brain/research/2026-11-23-stage2-test2-prereg.md sections 2, 3; Amendment T2 arms B, KP,
# PR = D2-PR2). 18 lives, one per AWS g5.xlarge (A10G, CUDA fp16), one state file per life
# (research/queue/.aws_brain.t2<arm><seed>). Written 11-23; NOT run. Usage (from the repo root):
#   bash brain/research/test2_launch/launch_plan.sh plan     # print the schedule and the worst-case spend per UTC day (default)
#   bash brain/research/test2_launch/launch_plan.sh go D1    # D1 = UTC date of day 1 (YYYY-MM-DD, a parameter; prereg
#                                                            # Amendment T3); launches each wave at its time, then
#                                                            # syncs/pulls every 5 min until every life is pulled
#   bash brain/research/test2_launch/launch_plan.sh poll     # one sync/pull pass over the t2 state files
# The job files sit next to this script (in the freeze commit); the log and the .launched_ markers go to
# _scratch/stage2/test2/ (gitignored) so brain/ stays clean. `go` resolves the AMI once, logs it and pins it
# (AWS_BRAIN_AMI) on every launch, so all 18 lives run one image; AWS_BRAIN_AMI_RECORD=1 writes it to the life provenance.
# Daily cap $50/UTC day at ~$1.01/h g5. Every life is costed at its WORST case: dead-man AWS_BRAIN_MAX_MIN (prereg
# per-life timing +30%) + grace 15 min + 15 min boot before the dead-man is armed. A wave more than DRIFT_MAX min late
# is not launched (a late start could push spend into a day it was not planned for); rerun `go` to re-plan it.
# Never terminates an instance whose job started: success is read from the instance's own run.log ("start <name>"),
# never from the local launch command's exit code or timeout (the 11-23 bug).
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1   # repo root
T2=brain/research/test2_launch; RUN=_scratch/stage2/test2; mkdir -p "$RUN"
SUBS=(subnet-0928cbecfe33fcbc1 subnet-010008ef274fac445 subnet-0c9d3dcf31a419f6d subnet-04d90e0c156fd8e16 subnet-076cae9f29280d90a)
RATE=1.01; CAP=50; GRACE=15; BOOT=15; DRIFT_MAX=45
KEYFILE=/home/dant123/.ssh/aws-train/claude-gpu-1785524741.pem; REGION="${AWS_REGION:-us-east-1}"
SSH=(ssh -i "$KEYFILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15)
# Per-life timing (prereg section 3): B ~4.4 h, K-P ~6.5 h, D2-PR2 ~4.5 h (estimated as B). +30%, rounded up to 5 min.
maxmin() { case $1 in B) echo 345;; KP) echo 510;; PR) echo 355;; esac; }
# Waves: <day 1..3> <HH:MM UTC> <arm> <seed>. <= 6 instances at once (Amendment T3; g5 vCPU quota checked first).
WAVES="1 00:10 KP 42
1 00:10 KP 43
1 00:10 KP 44
1 00:10 KP 100
1 00:10 KP 101
1 22:00 B 42
2 00:10 KP 102
2 00:10 B 43
2 00:10 B 44
2 00:10 B 100
2 00:10 B 101
2 16:00 PR 42
3 00:10 B 102
3 00:10 PR 43
3 00:10 PR 44
3 00:10 PR 100
3 00:10 PR 101
3 00:10 PR 102"
st_of() { echo "research/queue/.aws_brain.t2$1$2"; }
get() { awk -F= -v k="$2" '$1==k{print $2}' "$1" 2>/dev/null; }
log() { echo "$(date -u '+%F %H:%M') $*" | tee -a "$RUN/launch_plan.log"; }

plan() {   # worst-case instance-hours per UTC day, split at midnight; with each start also shifted DRIFT_MAX late
  echo "arm seed  day start  max_min  worst_h"
  echo "$WAVES" | while read -r d t arm s; do
    echo "$arm $s $d $t $(maxmin "$arm") $(awk -v m="$(maxmin "$arm")" -v g=$GRACE -v b=$BOOT 'BEGIN{printf "%.2f", (m+g+b)/60}')"
  done | column -t
  for drift in 0 $DRIFT_MAX; do
    echo "$WAVES" | while read -r d t arm s; do echo "$d $t $(maxmin "$arm")"; done | awk -v r=$RATE -v cap=$CAP -v g=$GRACE \
      -v b=$BOOT -v dr=$drift '{split($2,a,":"); s=($1-1)*1440+a[1]*60+a[2]+dr; e=s+$3+g+b
        for (m=s; m<e; m++) h[int(m/1440)+1]+=1/60; if (int((e-1)/1440)+1>n) n=int((e-1)/1440)+1}
      END{tot=0; for (i=1;i<=n;i++){c=h[i]*r; tot+=c; printf "start drift %2d min: UTC day %d  worst %.1f g5-h  $%.2f %s\n", dr, i, h[i], c, (c>cap?"OVER CAP":"<= cap")}
          printf "start drift %2d min: total worst $%.2f (expected about $95-105 at real timings)\n", dr, tot; bad=0
          for (i=1;i<=n;i++) if (h[i]*r>cap) bad=1; exit bad}' || { echo "⛔ plan exceeds the cap"; return 1; }
  done
}

# started <state> <name>: STARTED | STOPPED (instance stopped, no start) | NOT_STARTED (ssh ok: no run.log start, no
# run.sh; the [b] keeps pgrep from matching this remote shell's own command line) | UNKNOWN (ssh failed). STOPPED does
# not read run.log: launch_life acts on it only for an instance it created itself (fresh=1).
started() {
  local iid ip s r; iid=$(get "$1" instance)
  s=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null)
  [ "$s" = stopped ] && { echo STOPPED; return; }
  ip=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" --query 'Reservations[].Instances[].PublicIpAddress' --output text 2>/dev/null)
  r=$("${SSH[@]}" ubuntu@"$ip" "grep -q ' start $2\$' ~/brainv2/run.log 2>/dev/null && echo STARTED || { pgrep -f '[b]rainv2/run.sh' >/dev/null && echo RUNSH || echo NOT_STARTED; }" 2>/dev/null)
  case "$r" in STARTED) echo STARTED;; NOT_STARTED) echo NOT_STARTED;; RUNSH) echo UNKNOWN;; *) echo UNKNOWN;; esac
}

K0=0
launch_life() {   # <arm> <seed>; returns 0 when the job started, 1 to skip (flagged), 2 to abort the plan
  local arm=$1 s=$2 st name jobs k sub out v i fresh
  st=$(st_of "$arm" "$s"); name=life${arm}_s$s; jobs=$T2/jobs_${arm}_s$s.txt
  fresh=0   # 1 only for an instance this call created; an existing one (finished, dead-man stopped) is never terminated
  for k in 0 1 2 3 4; do
    sub=${SUBS[$(( (K0 + k) % 5 ))]}
    if [ ! -f "$st" ]; then
      fresh=1
      out=$(AWS_BRAIN_STATE=$st AWS_BRAIN_SUBNET=$sub AWS_BRAIN_MAX_MIN=$(maxmin "$arm") AWS_BRAIN_GRACE_MIN=$GRACE \
            AWS_BRAIN_AMI="$AMI" AWS_BRAIN_AMI_RECORD=1 AWS_BRAIN_ENV="$ENVV" \
            timeout 2400 bash tools/aws_brain_batch.sh launch "$jobs" 2>&1)
      log "$name $sub rc $?: $(echo "$out" | grep -E 'launched|running|⛔|A10G|Insufficient' | tr '\n' ' ')"
      if [ ! -f "$st" ]; then   # nothing was created (run-instances failed, or refused before it)
        echo "$out" | grep -q 'aws_budget' && { log "$name refused by the daily cap: plan aborted"; return 2; }
        continue                 # InsufficientInstanceCapacity or another run-instances failure: next zone
      fi
    fi
    for i in 1 2 3 4 5 6; do      # the instance exists: read its own run.log, never the local exit code
      v=$(started "$st" "$name"); [ "$v" = STARTED ] && { K0=$(( (K0 + k + 1) % 5 )); log "$name STARTED ($(get "$st" instance))"; return 0; }
      [ "$v" = STOPPED ] && break; sleep 60
    done
    if [ $fresh = 1 ] && { [ "$v" = STOPPED ] || [ "$v" = NOT_STARTED ]; }; then   # our new instance, job never started
      log "$name $v on $(get "$st" instance): terminate, next zone"
      AWS_BRAIN_STATE=$st timeout 900 bash tools/aws_brain_batch.sh terminate >/dev/null 2>&1
      [ -f "$st" ] && { log "$name terminate failed: look at $st"; return 1; }
      continue
    fi
    [ $fresh = 0 ] && { log "$name $v on $(get "$st" instance): existing instance, look by hand"; return 1; }
    log "$name UNKNOWN on $(get "$st" instance) (ssh failed or run.sh alive without a start): left as is, look by hand"; return 1
  done
  log "$name no zone had capacity"; return 1
}

poll() {   # sync each running life; pull (with the home and keep_bank.json, section 3) once its job ended
  local st b iid s
  for st in research/queue/.aws_brain.t2*; do
    case "$st" in *.done-*) continue;; esac; [ -f "$st" ] || continue
    b=$(get "$st" batch); iid=$(get "$st" instance)
    AWS_BRAIN_STATE=$st timeout 600 bash tools/aws_brain_batch.sh sync >/dev/null 2>&1
    s=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null)
    if grep -q "all done" "_scratch/brainv2/aws/$b/run.log" 2>/dev/null; then
      grep -q " end .* rc 0$" "_scratch/brainv2/aws/$b/run.log" || log "$st: job ended rc != 0 (see run.log)"
      log "$st pull: $(AWS_BRAIN_PULL_WORK=1 AWS_BRAIN_STATE=$st timeout 3600 bash tools/aws_brain_batch.sh pull 2>&1 | tail -1)"
    elif [ "$s" = stopped ]; then   # dead-man or cap stop mid-life: never auto-terminate; resume on g5 is a decision
      log "$st STOPPED before 'all done' ($iid): left on its disk; resume decision by hand (section 3: --resume on g5 only)"
    fi
  done
}

case "${1:-plan}" in
plan) plan ;;
poll) poll ;;
go)
  D1="${2:?usage: launch_plan.sh go YYYY-MM-DD}"; T0=$(date -u -d "$D1 00:00" +%s) || exit 1
  plan >/dev/null || exit 1
  [ -z "$(git status --porcelain -- brain/)" ] || { echo "⛔ brain/ is dirty: the registered lives need a clean freeze commit"; exit 1; }
  AMI=$(aws ec2 describe-images --region "$REGION" --owners amazon --filters \
    "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.* (Ubuntu 22.04)*" "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)   # resolved once: every life on one image
  [[ "$AMI" == ami-* ]] || { echo "⛔ could not resolve the Deep Learning AMI"; exit 1; }
  ENVV="BRAIN_CODE_COMMIT=$(git rev-parse HEAD) BRAIN_CODE_CLEAN=1"; log "go D1=$D1 commit $(git rev-parse HEAD) ami $AMI"
  while :; do
    left=0
    while read -r d t arm s; do
      [ -f "$RUN/.launched_$arm$s" ] && continue; left=1
      due=$(( T0 + (d - 1) * 86400 + $(date -u -d "1970-01-01 $t" +%s) )); now=$(date -u +%s)
      [ "$now" -lt "$due" ] && continue
      if [ $(( now - due )) -gt $(( DRIFT_MAX * 60 )) ]; then log "life${arm}_s$s: $(( (now - due) / 60 )) min late, not launched (rm $RUN/.launched_$arm$s and re-run go to re-plan)"; touch "$RUN/.launched_$arm$s"; continue; fi
      launch_life "$arm" "$s" < /dev/null; rc=$?
      [ $rc = 2 ] && exit 1; touch "$RUN/.launched_$arm$s"
    done <<< "$WAVES"
    poll
    ls research/queue/.aws_brain.t2* 2>/dev/null | grep -qv '\.done-' || [ $left = 1 ] || { log "every life launched and pulled"; exit 0; }
    sleep 300
  done ;;
*) echo "usage: launch_plan.sh {plan|go YYYY-MM-DD|poll}"; exit 2 ;;
esac
