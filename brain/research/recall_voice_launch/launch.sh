#!/usr/bin/env bash
# Recall-voice DEV launcher (prereg brain/research/2026-11-23-recall-voice-dev-prereg.md section 1). One life per AWS
# g5.xlarge, state research/queue/.aws_brain.rv<M><seed>, AMI pinned to the lifeKP image, BRAIN_CODE_COMMIT = HEAD.
#   bash brain/research/recall_voice_launch/launch.sh plan <M1|M2|all>   # projected worst-case spend vs today's cap
#   bash brain/research/recall_voice_launch/launch.sh go   <M1|M2|all>   # launch (prereg: M1 on 11-24, M2 on 11-25 UTC)
# Refuses: brain/ dirty; HEAD without life_kw.py, its test, the prereg or this folder; another rv arm still live;
# today's spend + worst case > the $50/UTC-day cap. Zone rotation on InsufficientInstanceCapacity (any run-instances
# failure moves to the next subnet). Never terminates an instance it did not just create; success is read from the
# instance's own run.log ("start <name>"), never from the local launch exit code (the launch_plan.sh pattern).
# Pull/sync afterwards with tools/aws_brain_batch.sh (AWS_BRAIN_STATE=<state>); resume only via life_kw, same extra.
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1   # repo root
RV=brain/research/recall_voice_launch; RUN=_scratch/stage2/rv; mkdir -p "$RUN"
SUBS=(subnet-0928cbecfe33fcbc1 subnet-010008ef274fac445 subnet-0c9d3dcf31a419f6d subnet-04d90e0c156fd8e16 subnet-076cae9f29280d90a)
AMI=ami-012ba162b9cd2729c   # the lifeKP image (research/queue/.aws_brain.kp319/320/321.done-*)
RATE=1.01; CAP=50; GRACE=15; BOOT=15; SEEDS="319 320 321"
KEYFILE=/home/dant123/.ssh/aws-train/claude-gpu-1785524741.pem; REGION="${AWS_REGION:-us-east-1}"
SSH=(ssh -i "$KEYFILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15)
maxmin() { case $1 in M1) echo 600;; M2) echo 510;; esac; }   # dead-man per life (prereg section 5)
st_of() { echo "research/queue/.aws_brain.rv$1$2"; }
get() { awk -F= -v k="$2" '$1==k{print $2}' "$1" 2>/dev/null; }
log() { echo "$(date -u '+%F %H:%M') $*" | tee -a "$RUN/launch.log"; }

arms_of() { case "$1" in M1|M2) echo "$1";; all) echo "M1 M2";; *) echo "usage: launch.sh {plan|go} {M1|M2|all}" >&2; exit 2;; esac; }

plan() {   # worst case = dead-man + grace + boot per life, all counted on today (conservative); exit 1 if over the cap
  local today worst=0 a
  today=$(bash tools/aws_budget.sh status 2>/dev/null | grep -o 'spend_today≈\$[0-9.]*' | tr -dc '0-9.')
  [ -n "$today" ] || { echo "⛔ could not read today's spend (tools/aws_budget.sh status)"; return 1; }
  for a in $ARMS; do
    worst=$(awk -v w="$worst" -v m="$(maxmin "$a")" -v g=$GRACE -v b=$BOOT -v r=$RATE 'BEGIN{printf "%.2f", w+3*(m+g+b)/60*r}')
    echo "$a: 3 lives x $(maxmin "$a")+$GRACE+$BOOT min at \$$RATE/h"
  done
  awk -v t="$today" -v w="$worst" -v c=$CAP 'BEGIN{printf "spend today $%.2f + projected worst $%.2f = $%.2f (cap $%d) %s\n", t, w, t+w, c, (t+w>c?"OVER CAP":"<= cap"); exit (t+w>c)}' \
    || { echo "⛔ projected spend exceeds the \$$CAP/UTC-day cap"; return 1; }
}

# started <state> <name>: STARTED | STOPPED | NOT_STARTED | UNKNOWN (copied from test2_launch/launch_plan.sh)
started() {
  local iid ip s r; iid=$(get "$1" instance)
  s=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null)
  [ "$s" = stopped ] && { echo STOPPED; return; }
  ip=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" --query 'Reservations[].Instances[].PublicIpAddress' --output text 2>/dev/null)
  r=$("${SSH[@]}" ubuntu@"$ip" "grep -q ' start $2\$' ~/brainv2/run.log 2>/dev/null && echo STARTED || { pgrep -f '[b]rainv2/run.sh' >/dev/null && echo RUNSH || echo NOT_STARTED; }" 2>/dev/null)
  case "$r" in STARTED) echo STARTED;; NOT_STARTED) echo NOT_STARTED;; *) echo UNKNOWN;; esac
}

K0=0
launch_life() {   # <arm> <seed>; 0 started, 1 skip (flagged), 2 abort
  local arm=$1 s=$2 st name jobs k sub out v i fresh
  st=$(st_of "$arm" "$s"); name=life${arm}_s$s; jobs=$RV/jobs_${arm}_s$s.txt
  [ -f "$st" ] && { log "$name: $st already exists (an instance this run did not create): not touched"; return 1; }
  fresh=0
  for k in 0 1 2 3 4; do
    sub=${SUBS[$(( (K0 + k) % 5 ))]}
    if [ ! -f "$st" ]; then
      fresh=1
      out=$(AWS_BRAIN_STATE=$st AWS_BRAIN_SUBNET=$sub AWS_BRAIN_MAX_MIN=$(maxmin "$arm") AWS_BRAIN_GRACE_MIN=$GRACE \
            AWS_BRAIN_AMI="$AMI" AWS_BRAIN_AMI_RECORD=1 AWS_BRAIN_ENV="$ENVV" \
            timeout 2400 bash tools/aws_brain_batch.sh launch "$jobs" 2>&1)
      log "$name $sub rc $?: $(echo "$out" | grep -E 'launched|running|⛔|A10G|Insufficient' | tr '\n' ' ')"
      if [ ! -f "$st" ]; then
        echo "$out" | grep -q 'aws_budget' && { log "$name refused by the daily cap: aborted"; return 2; }
        continue                 # InsufficientInstanceCapacity or another run-instances failure: next zone
      fi
    fi
    for i in 1 2 3 4 5 6; do
      v=$(started "$st" "$name"); [ "$v" = STARTED ] && { K0=$(( (K0 + k + 1) % 5 )); log "$name STARTED ($(get "$st" instance))"; return 0; }
      [ "$v" = STOPPED ] && break; sleep 60
    done
    if [ $fresh = 1 ] && { [ "$v" = STOPPED ] || [ "$v" = NOT_STARTED ]; }; then   # our new instance, job never started
      log "$name $v on $(get "$st" instance): terminate, next zone"
      AWS_BRAIN_STATE=$st timeout 900 bash tools/aws_brain_batch.sh terminate >/dev/null 2>&1
      [ -f "$st" ] && { log "$name terminate failed: look at $st"; return 1; }
      continue
    fi
    log "$name $v on $(get "$st" instance): left as is, look by hand"; return 1
  done
  log "$name no zone had capacity"; return 1
}

MODE="${1:-plan}"; ARMS=$(arms_of "${2:-}") || exit 2
case "$MODE" in
plan) plan ;;
go)
  [ -z "$(git status --porcelain -- brain/)" ] || { echo "⛔ brain/ is dirty: launch only from a clean commit"; exit 1; }
  for f in brain/experiments/life_kw.py brain/tests/test_life_kw.py brain/research/2026-11-23-recall-voice-dev-prereg.md \
           $RV/launch.sh $RV/jobs_M1_s319.txt $RV/jobs_M2_s319.txt; do
    git cat-file -e "HEAD:$f" 2>/dev/null || { echo "⛔ HEAD lacks $f (prereg section 1): commit before launch"; exit 1; }
  done
  for a in M1 M2; do case " $ARMS " in *" $a "*) continue;; esac
    ls research/queue/.aws_brain.rv"$a"* 2>/dev/null | grep -qv '\.done-' && { echo "⛔ arm $a still has a live state file: one arm per UTC day"; exit 1; }
  done
  plan || exit 1
  ENVV="BRAIN_CODE_COMMIT=$(git rev-parse HEAD)"; log "go $ARMS commit $(git rev-parse HEAD) ami $AMI"
  for a in $ARMS; do for s in $SEEDS; do
    launch_life "$a" "$s" < /dev/null; [ $? = 2 ] && exit 1
  done; done ;;
*) echo "usage: launch.sh {plan|go} {M1|M2|all}"; exit 2 ;;
esac
