#!/usr/bin/env bash
# aws_brain_batch.sh -- run brain/ experiment jobs on ONE AWS GPU instance that stops itself (owner 2026-11-02, speed
# lever A on the existing AWS lane: six-seed runs in about an hour instead of a night on the mini-PCs, and safe while
# the workstation is off, e.g. on gaming nights).
#   bash tools/aws_brain_batch.sh launch <jobs.txt> [instance-type]   # default g5.xlarge (A10G 24 GB, ~$1/h)
#   bash tools/aws_brain_batch.sh status | pull | terminate
# jobs.txt: the tools/brain_batch_run.sh format ("<name> <sleep_probe args>" or "<name> brain.<module> <args>",
# models as ../models/<file>.pth; # lines are comments). A batch longer than 4 h needs AWS_BRAIN_MAX_MIN raised;
# every job gets --device cuda. Results land in _scratch/brainv2/aws/<batch>/ on `pull`.
#
# Safety lives on the AWS side, so nothing here has to stay up:
#   - launched with --instance-initiated-shutdown-behavior stop;
#   - a dead-man `shutdown -h +AWS_BRAIN_MAX_MIN` (default 240) is armed before any job starts;
#   - the runner ends with `shutdown -h now`, whether the jobs passed or failed.
# A stopped instance bills only its disk (cents a day) until `pull` copies the results and terminates it. The daily
# spend cap gates the launch (tools/aws_budget.sh check), and the workstation's aws-guard timer (budget enforce + idle
# stop, which now also counts brain jobs as work) runs whenever the workstation is up. The AWS keys stay on the
# workstation: the server's rules keep credentials out of its agent-writable checkout.
set -uo pipefail
REGION="${AWS_REGION:-us-east-1}"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT" || exit 1
STATE="$ROOT/research/queue/.aws_brain"
KEYNAME=claude-gpu-1785524741
KEYFILE=/home/dant123/.ssh/aws-train/claude-gpu-1785524741.pem
SUBNET=subnet-0928cbecfe33fcbc1
MODELS=/home/dant123/models/rwkv7
MAX_MIN="${AWS_BRAIN_MAX_MIN:-240}"
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
  # Off-the-shelf image with the NVIDIA driver and PyTorch (/opt/pytorch), newest Ubuntu 22.04 build.
  AMI=$(aws ec2 describe-images --region "$REGION" --owners amazon --filters \
    "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.* (Ubuntu 22.04)*" "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
  [[ "$AMI" == ami-* ]] || { echo "could not find the Deep Learning AMI"; exit 1; }
  MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '\n'); [ -n "$MYIP" ] || { echo "no public IP"; exit 1; }
  VPC=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  SG=$(aws ec2 create-security-group --region "$REGION" --group-name "claude-brain-$(date +%s)" \
    --description "temp SSH for brain batch" --vpc-id "$VPC" --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" --protocol tcp --port 22 --cidr "${MYIP}/32" >/dev/null
  IID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEYNAME" \
    --security-group-ids "$SG" --subnet-id "$SUBNET" --instance-initiated-shutdown-behavior stop \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=claude-brain-batch},{Key=Project,Value=neural-sim}]' \
    --query 'Instances[0].InstanceId' --output text)
  [[ "$IID" == i-* ]] || { echo "run-instances failed; deleting SG $SG"; aws ec2 delete-security-group --region "$REGION" --group-id "$SG"; exit 1; }
  BATCH="$(date +%Y%m%d-%H%M)"
  { echo "# LIVE brain batch launched $(date '+%F %T %Z') type=$TYPE ami=$AMI -- pull: tools/aws_brain_batch.sh pull"
    echo "instance=$IID"; echo "sg=$SG"; echo "batch=$BATCH"; echo "jobs=$JOBS"; } > "$STATE"
  echo "[aws-brain] launched $IID ($TYPE), waiting for it to boot..."
  aws ec2 wait instance-status-ok --region "$REGION" --instance-ids "$IID"
  IP=$(ip_of "$IID"); R=(ubuntu@"$IP")
  # Arm the dead-man first, so even a failed provisioning step cannot leave it running.
  "${SSH[@]}" "${R[@]}" "sudo shutdown -h +$MAX_MIN" || { echo "⛔ could not arm the dead-man; stopping"; aws ec2 stop-instances --region "$REGION" --instance-ids "$IID" >/dev/null; exit 1; }
  "${SSH[@]}" "${R[@]}" "mkdir -p ~/brainv2/code ~/brainv2/models ~/brainv2/results && /opt/pytorch/bin/python -m pip -q install rwkv==0.8.32 && /opt/pytorch/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'" \
    || { echo "⛔ provisioning failed; stopping the instance"; aws ec2 stop-instances --region "$REGION" --instance-ids "$IID" >/dev/null; exit 1; }
  git archive HEAD brain | "${SSH[@]}" "${R[@]}" "tar -x -C ~/brainv2/code"
  for m in $(grep -o 'rwkv7-[A-Za-z0-9._-]*\.pth' "$JOBS" | sort -u); do
    rsync -a -e "${SSH[*]}" "$MODELS/$m" "${R[0]}:brainv2/models/" || { echo "⛔ upload of $m failed; stopping"; aws ec2 stop-instances --region "$REGION" --instance-ids "$IID" >/dev/null; exit 1; }
  done
  scp -q -i "$KEYFILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    "$JOBS" "${R[0]}:brainv2/jobs.txt"
  scp -q -i "$KEYFILE" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    "$ROOT/tools/brain_batch_run.sh" "${R[0]}:brainv2/run.sh"
  "${SSH[@]}" "${R[@]}" "chmod +x ~/brainv2/run.sh && nohup bash -c 'PY=/opt/pytorch/bin/python EXTRA=\"--device cuda\" ~/brainv2/run.sh >> ~/brainv2/run.log 2>&1; sudo shutdown -h now' >/dev/null 2>&1 &"
  echo "[aws-brain] running $(grep -vc '^\(#\|$\)' "$JOBS") job(s) on $IID; it stops itself when done (dead-man ${MAX_MIN} min)."
  echo "[aws-brain] later: bash tools/aws_brain_batch.sh pull"
  ;;
status)
  [ -f "$STATE" ] || { echo "no brain batch recorded"; exit 0; }
  IID=$(get instance); S=$(state_of "$IID"); echo "$IID $S (batch $(get batch))"
  [ "$S" = running ] && "${SSH[@]}" ubuntu@"$(ip_of "$IID")" "tail -3 ~/brainv2/run.log" 2>/dev/null
  ;;
pull)
  [ -f "$STATE" ] || { echo "no brain batch recorded"; exit 1; }
  IID=$(get instance); S=$(state_of "$IID")
  if [ "$S" = stopped ]; then
    aws ec2 start-instances --region "$REGION" --instance-ids "$IID" >/dev/null
    aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"; sleep 30
  fi
  OUT="$ROOT/_scratch/brainv2/aws/$(get batch)"; mkdir -p "$OUT"
  rsync -a -e "${SSH[*]}" ubuntu@"$(ip_of "$IID")":brainv2/results/ ubuntu@"$(ip_of "$IID")":brainv2/run.log "$OUT/" \
    || { echo "⛔ pull failed; instance left as is ($IID)"; exit 1; }
  echo "[aws-brain] results in $OUT:"; tail -3 "$OUT/run.log"
  if grep -q "all done" "$OUT/run.log" || [ "$S" = stopped ]; then exec bash "$0" terminate; fi
  echo "[aws-brain] jobs still running; it will stop itself. Pull again later."
  ;;
terminate)
  [ -f "$STATE" ] || { echo "no brain batch recorded"; exit 1; }
  IID=$(get instance); SG=$(get sg)
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$IID" >/dev/null
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$IID"
  aws ec2 delete-security-group --region "$REGION" --group-id "$SG"
  mv "$STATE" "$STATE.done-$(date +%s)"; echo "[aws-brain] terminated $IID, deleted $SG"
  ;;
*) echo "usage: aws_brain_batch.sh {launch <jobs.txt> [type]|status|pull|terminate}"; exit 2 ;;
esac
