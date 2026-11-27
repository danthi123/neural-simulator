#!/usr/bin/env bash
# aws_brain_batch.sh -- run brain/ experiment jobs on ONE AWS GPU instance that stops itself (owner 2026-11-02, speed
# lever A on the existing AWS lane: six-seed runs in about an hour instead of a night on the mini-PCs, and safe while
# the workstation is off, e.g. on gaming nights).
#   bash tools/aws_brain_batch.sh launch <jobs.txt> [instance-type]   # default g5.xlarge (A10G 24 GB, ~$1/h)
#   bash tools/aws_brain_batch.sh status | sync | pull | terminate
# Optional env: AWS_BRAIN_ENV="K=V ..." (exported for every job via ~/brainv2/env.sh), AWS_BRAIN_AMI=ami-... (pin the
# image instead of the newest Deep Learning AMI), AWS_BRAIN_AMI_RECORD=1 (export AWS_AMI_ID in env.sh for life.py
# provenance), AWS_BRAIN_UPLOAD (extra inputs), AWS_BRAIN_PULL_WORK=1 (pull also copies ~/brainv2/work/, e.g. life
# homes). All unset = the old behaviour exactly (no env.sh is written).
# sync: while the instance runs, copy the results so far (no start, no terminate). Poll with `sync`, not `status`:
# a stopped g6 cannot always restart (InsufficientInstanceCapacity, 2026-11-16), and the results live on its disk.
# jobs.txt: the tools/brain_batch_run.sh format ("<name> <sleep_probe args>" or "<name> brain.<module> <args>",
# models as ../models/<file>.pth; # lines are comments). A batch longer than 4 h needs AWS_BRAIN_MAX_MIN raised;
# every job gets --device cuda. Results land in _scratch/brainv2/aws/<batch>/ on `pull`.
#
# Safety lives on the AWS side, so nothing here has to stay up:
#   - launched with --instance-initiated-shutdown-behavior stop;
#   - a dead-man `shutdown -h +AWS_BRAIN_MAX_MIN` (default 240) is armed before any job starts;
#   - the runner ends with `shutdown -h +AWS_BRAIN_GRACE_MIN` (default 0 = now), whether the jobs passed or failed.
# A stopped instance bills only its disk (cents a day) until `pull` copies the results and terminates it. The daily
# spend cap gates the launch (tools/aws_budget.sh check), and the workstation's aws-guard timer (budget enforce + idle
# stop, which now also counts brain jobs as work) runs whenever the workstation is up. The AWS keys stay on the
# workstation: the server's rules keep credentials out of its agent-writable checkout.
set -uo pipefail
REGION="${AWS_REGION:-us-east-1}"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT" || exit 1
STATE="${AWS_BRAIN_STATE:-$ROOT/research/queue/.aws_brain}"   # override to run several batches at once (one state file each)
KEYNAME=claude-gpu-1785524741
KEYFILE=/home/dant123/.ssh/aws-train/claude-gpu-1785524741.pem
SUBNET="${AWS_BRAIN_SUBNET:-subnet-0928cbecfe33fcbc1}"   # us-east-1d; on InsufficientInstanceCapacity pass another AZ default subnet (aws ec2 describe-subnets --filters Name=default-for-az,Values=true)
MODELS=/home/dant123/models/rwkv7
MAX_MIN="${AWS_BRAIN_MAX_MIN:-240}"
GRACE="${AWS_BRAIN_GRACE_MIN:-0}"   # minutes the instance stays up after the jobs end, so a pull loop can copy results without a restart (a stopped g5/g6 often cannot restart: InsufficientInstanceCapacity 2026-11-16, 11-21)
SSH=(ssh -i "$KEYFILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=15)
get() { awk -F= -v k="$1" '$1==k{print $2}' "$STATE" 2>/dev/null; }
ip_of() { aws ec2 describe-instances --region "$REGION" --instance-ids "$1" --query 'Reservations[].Instances[].PublicIpAddress' --output text; }
state_of() { aws ec2 describe-instances --region "$REGION" --instance-ids "$1" --query 'Reservations[].Instances[].State.Name' --output text; }

case "${1:-status}" in
launch)
  JOBS="${2:?usage: aws_brain_batch.sh launch <jobs.txt> [instance-type]}"; TYPE="${3:-g5.xlarge}"
  [ -s "$JOBS" ] || { echo "no jobs in $JOBS"; exit 1; }
  [ -f "$STATE" ] && { echo "⛔ a brain batch is already recorded in $STATE ($(get instance)); pull/terminate it first"; exit 1; }
  bash "$ROOT/tools/aws_budget.sh" check "$TYPE" || { echo "⛔ refused by tools/aws_budget.sh (daily cap)"; exit 1; }
  # Optional AWS_BRAIN_ENV="K1=V1 K2=V2": exported for every job (written to ~/brainv2/env.sh, which run.sh sources;
  # no spaces inside a value). A BRAIN_CODE_COMMIT in it must be the HEAD that git archive ships. Unset = unchanged.
  for kv in ${AWS_BRAIN_ENV:-}; do
    [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "⛔ AWS_BRAIN_ENV entry '$kv' is not K=V"; exit 1; }
    [[ "$kv" == BRAIN_CODE_COMMIT=* && "${kv#*=}" != "$(git rev-parse HEAD)" ]] \
      && { echo "⛔ AWS_BRAIN_ENV BRAIN_CODE_COMMIT is not HEAD ($(git rev-parse HEAD)), which is what ships"; exit 1; }
  done
  # Off-the-shelf image with the NVIDIA driver and PyTorch (/opt/pytorch), newest Ubuntu 22.04 build, unless
  # AWS_BRAIN_AMI pins one (a multi-day batch of lives must run one image).
  AMI="${AWS_BRAIN_AMI:-$(aws ec2 describe-images --region "$REGION" --owners amazon --filters \
    "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.* (Ubuntu 22.04)*" "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)}"
  [[ "$AMI" == ami-* ]] || { echo "could not find the Deep Learning AMI"; exit 1; }
  MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '\n'); [ -n "$MYIP" ] || { echo "no public IP"; exit 1; }
  VPC=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  SG=$(aws ec2 create-security-group --region "$REGION" --group-name "claude-brain-$(date +%s)-$$" \
    --description "temp SSH for brain batch" --vpc-id "$VPC" --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" --protocol tcp --port 22 --cidr "${MYIP}/32" >/dev/null
  IID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEYNAME" \
    --security-group-ids "$SG" --subnet-id "$SUBNET" --instance-initiated-shutdown-behavior stop \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=claude-brain-batch},{Key=Project,Value=neural-sim}]' \
    --query 'Instances[0].InstanceId' --output text)
  [[ "$IID" == i-* ]] || { echo "run-instances failed; deleting SG $SG"; aws ec2 delete-security-group --region "$REGION" --group-id "$SG"; exit 1; }
  BATCH="$(date +%Y%m%d-%H%M)-$$"
  { echo "# LIVE brain batch launched $(date '+%F %T %Z') type=$TYPE ami=$AMI -- pull: tools/aws_brain_batch.sh pull"
    echo "instance=$IID"; echo "sg=$SG"; echo "batch=$BATCH"; echo "jobs=$JOBS"; [ -n "${AWS_BRAIN_AMI_RECORD:-}" ] && echo "ami=$AMI"; } > "$STATE"
  echo "[aws-brain] launched $IID ($TYPE), waiting for it to boot..."
  aws ec2 wait instance-status-ok --region "$REGION" --instance-ids "$IID"
  IP=$(ip_of "$IID"); R=(ubuntu@"$IP")
  # Arm the dead-man first, so even a failed provisioning step cannot leave it running.
  "${SSH[@]}" "${R[@]}" "sudo shutdown -h +$MAX_MIN" || { echo "⛔ could not arm the dead-man; stopping"; aws ec2 stop-instances --region "$REGION" --instance-ids "$IID" >/dev/null; exit 1; }
  "${SSH[@]}" "${R[@]}" "mkdir -p ~/brainv2/code ~/brainv2/models ~/brainv2/results && /opt/pytorch/bin/python -m pip -q install rwkv==0.8.32 && /opt/pytorch/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'" \
    || { echo "⛔ provisioning failed; stopping the instance"; aws ec2 stop-instances --region "$REGION" --instance-ids "$IID" >/dev/null; exit 1; }
  git archive HEAD brain | "${SSH[@]}" "${R[@]}" "tar -x -C ~/brainv2/code"
  # ~/brainv2/env.sh (run.sh sources it), only when AWS_BRAIN_ENV or AWS_BRAIN_AMI_RECORD is set: AWS_AMI_ID
  # (life.py provenance; record-only) if AWS_BRAIN_AMI_RECORD, then AWS_BRAIN_ENV. Both unset = no env.sh (unchanged).
  if [ -n "${AWS_BRAIN_ENV:-}" ] || [ -n "${AWS_BRAIN_AMI_RECORD:-}" ]; then
    { [ -n "${AWS_BRAIN_AMI_RECORD:-}" ] && printf 'export AWS_AMI_ID=%q\n' "$AMI"
      for kv in ${AWS_BRAIN_ENV:-}; do printf 'export %s=%q\n' "${kv%%=*}" "${kv#*=}"; done; } \
      | "${SSH[@]}" "${R[@]}" "cat > ~/brainv2/env.sh" \
      || { echo "⛔ could not write ~/brainv2/env.sh; stopping"; aws ec2 stop-instances --region "$REGION" --instance-ids "$IID" >/dev/null; exit 1; }
  fi
  # Optional extra inputs before any job starts: AWS_BRAIN_UPLOAD="local=>remote;..." (remote relative to ~/brainv2;
  # a local dir with a trailing / copies its contents). Unset = unchanged behaviour.
  if [ -n "${AWS_BRAIN_UPLOAD:-}" ]; then
    IFS=';' read -ra UPS <<< "$AWS_BRAIN_UPLOAD"
    for u in "${UPS[@]}"; do
      l=${u%%=>*}; r=${u#*=>}
      "${SSH[@]}" "${R[@]}" "mkdir -p ~/brainv2/$(dirname "$r")" && rsync -a -e "${SSH[*]}" "$l" "${R[0]}:brainv2/$r" \
        || { echo "⛔ upload of $l failed; stopping"; aws ec2 stop-instances --region "$REGION" --instance-ids "$IID" >/dev/null; exit 1; }
    done
  fi
  for m in $(grep -o 'rwkv7-[A-Za-z0-9._-]*\.pth' "$JOBS" | sort -u); do
    rsync -a -e "${SSH[*]}" "$MODELS/$m" "${R[0]}:brainv2/models/" || { echo "⛔ upload of $m failed; stopping"; aws ec2 stop-instances --region "$REGION" --instance-ids "$IID" >/dev/null; exit 1; }
  done
  scp -q -i "$KEYFILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    "$JOBS" "${R[0]}:brainv2/jobs.txt"
  scp -q -i "$KEYFILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    "$ROOT/tools/brain_batch_run.sh" "${R[0]}:brainv2/run.sh"
  "${SSH[@]}" "${R[@]}" "chmod +x ~/brainv2/run.sh; nohup bash -c 'PY=/opt/pytorch/bin/python EXTRA=\"--device cuda\" ~/brainv2/run.sh >> ~/brainv2/run.log 2>&1; sudo shutdown -h +$GRACE' >/dev/null 2>&1 < /dev/null &"
  echo "[aws-brain] running $(grep -vc '^\(#\|$\)' "$JOBS") job(s) on $IID; it stops itself when done (dead-man ${MAX_MIN} min)."
  echo "[aws-brain] later: bash tools/aws_brain_batch.sh pull"
  ;;
status)
  [ -f "$STATE" ] || { echo "no brain batch recorded"; exit 0; }
  IID=$(get instance); S=$(state_of "$IID"); echo "$IID $S (batch $(get batch))"
  [ "$S" = running ] && "${SSH[@]}" ubuntu@"$(ip_of "$IID")" "tail -3 ~/brainv2/run.log" 2>/dev/null
  ;;
sync)
  [ -f "$STATE" ] || { echo "no brain batch recorded"; exit 0; }
  IID=$(get instance); S=$(state_of "$IID"); echo "$IID $S (batch $(get batch))"
  [ "$S" = running ] || exit 0
  OUT="$ROOT/_scratch/brainv2/aws/$(get batch)"; mkdir -p "$OUT"
  rsync -a -e "${SSH[*]}" ubuntu@"$(ip_of "$IID")":brainv2/results/ ubuntu@"$(ip_of "$IID")":brainv2/run.log "$OUT/" 2>/dev/null \
    && tail -1 "$OUT/run.log"
  ;;
pull)
  [ -f "$STATE" ] || { echo "no brain batch recorded"; exit 1; }
  IID=$(get instance); S=$(state_of "$IID")
  if [ "$S" = stopped ]; then
    aws ec2 start-instances --region "$REGION" --instance-ids "$IID" >/dev/null \
      || { echo "⛔ could not start $IID (capacity?); results stay on its disk. Retry pull later."; exit 1; }
    aws ec2 wait instance-status-ok --region "$REGION" --instance-ids "$IID"   # sshd up; a fixed sleep 30 was not enough (2026-11-16)
  fi
  OUT="$ROOT/_scratch/brainv2/aws/$(get batch)"; mkdir -p "$OUT"
  rsync -a -e "${SSH[*]}" ubuntu@"$(ip_of "$IID")":brainv2/results/ ubuntu@"$(ip_of "$IID")":brainv2/run.log "$OUT/" \
    || { echo "⛔ pull failed; instance left as is ($IID)"; exit 1; }
  # AWS_BRAIN_PULL_WORK=1: also copy ~/brainv2/work/ (life homes, keep_bank.json) before any terminate. Unset = unchanged.
  if [ "${AWS_BRAIN_PULL_WORK:-}" = 1 ]; then
    rsync -a -e "${SSH[*]}" ubuntu@"$(ip_of "$IID")":brainv2/work/ "$OUT/work/" \
      || { echo "⛔ pull of brainv2/work failed; instance left as is ($IID)"; exit 1; }
  fi
  echo "[aws-brain] results in $OUT:"; tail -3 "$OUT/run.log"
  if grep -q "all done" "$OUT/run.log" || [ "$S" = stopped ]; then exec bash "$0" terminate; fi
  echo "[aws-brain] jobs still running; it will stop itself. Pull again later."
  ;;
resume)
  # A batch the dead-man stopped mid-job: start it, re-arm the dead-man, rerun (run.sh skips jobs with a result).
  [ -f "$STATE" ] || { echo "no brain batch recorded"; exit 1; }
  IID=$(get instance); S=$(state_of "$IID")
  [ "$S" = stopped ] || { echo "⛔ $IID is $S, not stopped; wait for it to stop, or pull"; exit 1; }
  bash "$ROOT/tools/aws_budget.sh" check "$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" --query 'Reservations[].Instances[].InstanceType' --output text)" \
    || { echo "⛔ refused by tools/aws_budget.sh (daily cap)"; exit 1; }
  aws ec2 start-instances --region "$REGION" --instance-ids "$IID" >/dev/null \
    || { echo "⛔ could not start $IID (capacity?); retry later"; exit 1; }
  aws ec2 wait instance-status-ok --region "$REGION" --instance-ids "$IID"
  R=(ubuntu@"$(ip_of "$IID")")
  "${SSH[@]}" "${R[@]}" "sudo shutdown -h +$MAX_MIN" || { echo "⛔ could not arm the dead-man; stopping"; aws ec2 stop-instances --region "$REGION" --instance-ids "$IID" >/dev/null; exit 1; }
  "${SSH[@]}" "${R[@]}" "nohup bash -c 'PY=/opt/pytorch/bin/python EXTRA=\"--device cuda\" ~/brainv2/run.sh >> ~/brainv2/run.log 2>&1; sudo shutdown -h +$GRACE' >/dev/null 2>&1 < /dev/null &"
  echo "[aws-brain] resumed $IID (dead-man ${MAX_MIN} min); later: bash tools/aws_brain_batch.sh pull"
  ;;
terminate)
  [ -f "$STATE" ] || { echo "no brain batch recorded"; exit 1; }
  IID=$(get instance); SG=$(get sg)
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$IID" >/dev/null
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$IID"
  aws ec2 delete-security-group --region "$REGION" --group-id "$SG"
  mv "$STATE" "$STATE.done-$(date +%s)"; echo "[aws-brain] terminated $IID, deleted $SG"
  ;;
*) echo "usage: aws_brain_batch.sh {launch <jobs.txt> [type]|status|sync|pull|resume|terminate}"; exit 2 ;;
esac
