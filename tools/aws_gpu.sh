#!/usr/bin/env bash
# Drive the AWS GPU lane. STATE LIVES HERE, not in my memory -- the 2026-07-25 lane lost its key to a /tmp
# reboot and needed an owner-granted IAM recovery. Key and instance id are recorded durably.
#
#   bash tools/aws_gpu.sh launch [instance-type]   # default g5.xlarge (AWS_GPU_TYPE overrides) -- gated by
#                                                   # tools/aws_budget.sh check (owner-approved 2026-09-23 cap)
#   bash tools/aws_gpu.sh status | ip | ssh | stop | terminate
REGION="${AWS_REGION:-us-east-1}"
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT" || exit 1
STATE="$ROOT/research/queue/.aws_gpu"

if [ "${1:-status}" = "launch" ]; then
  TYPE="${2:-${AWS_GPU_TYPE:-g5.xlarge}}"
  # Hard pre-launch refusal -- owner-approved 2026-09-23 daily spend cap, enforced by tooling not memory.
  bash "$ROOT/tools/aws_budget.sh" check "$TYPE" || { echo "⛔ aws_gpu launch: refused by tools/aws_budget.sh (daily cap) -- see above"; exit 1; }
  [ -f "$STATE" ] && { live=$(awk -F= '/^instance=/{print $2}' "$STATE" 2>/dev/null)
    echo "⛔ an instance is already recorded ($live) in $STATE -- stop/terminate it first"; exit 1; }
  # Same proven AMI/key/subnet as tools/aws_cpu_launch.sh (verified-working in this account) -- only the
  # instance type differs. This AMI is plain Ubuntu 22.04 with NO NVIDIA driver preinstalled; tools/
  # aws_provision.sh (unchanged, not owned by this script) already asserts cp.cuda.runtime.getDeviceCount()
  # > 0 and REFUSES before any GPU work is dispatched if the driver isn't present, so a driver-less launch
  # is a bounded/caught failure, not a silent one. If it fires, relaunch on an AWS Deep Learning AMI instead.
  AMI=ami-05a3e9423ae4d7a19
  KEYNAME=claude-gpu-1785524741
  KEYFILE=/home/dant123/.ssh/aws-train/claude-gpu-1785524741.pem
  SUBNET=subnet-0928cbecfe33fcbc1
  MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '\n')
  [ -n "$MYIP" ] || { echo "could not determine my public IP"; exit 1; }
  VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region "$REGION")
  echo "[aws-gpu] my IP=$MYIP  VPC=$VPC  type=$TYPE"
  SG=$(aws ec2 create-security-group --group-name "claude-gpu-verify-$(date +%s)" \
    --description "temp SSH for GPU lane" --vpc-id "$VPC" --region "$REGION" --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 \
    --cidr "${MYIP}/32" --region "$REGION" >/dev/null
  echo "[aws-gpu] created SG $SG (ssh from ${MYIP}/32)"
  IID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" \
    --key-name "$KEYNAME" --security-group-ids "$SG" --subnet-id "$SUBNET" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=60,VolumeType=gp3,DeleteOnTermination=true}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=claude-gpu-verify},{Key=Project,Value=neural-sim}]' \
    --query 'Instances[0].InstanceId' --output text)
  { echo "# LIVE GPU instance launched $(date '+%F %T %Z') type=$TYPE -- TERMINATE when done: tools/aws_gpu.sh terminate"
    echo "# then delete SG: aws ec2 delete-security-group --group-id $SG --region $REGION"
    echo "instance=$IID"; echo "region=$REGION"; echo "key=$KEYFILE"; echo "sg=$SG"; } > "$STATE"
  echo "[aws-gpu] launched $IID (recorded in $STATE). Waiting for running+status-ok…"
  aws ec2 wait instance-status-ok --instance-ids "$IID" --region "$REGION"
  IP=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION" --query 'Reservations[].Instances[].PublicIpAddress' --output text)
  echo "[aws-gpu] RUNNING. IP=$IP"
  echo "[aws-gpu] ssh: ssh -i $KEYFILE -o StrictHostKeyChecking=no ubuntu@$IP"
  exit 0
fi

[ -f "$STATE" ] || { echo "no AWS GPU lane recorded in $STATE"; exit 1; }
IID=$(awk -F= '/^instance=/{print $2}' "$STATE"); KEY=$(awk -F= '/^key=/{print $2}' "$STATE")
case "${1:-status}" in
  status) aws ec2 describe-instances --instance-ids "$IID" \
            --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,PublicIpAddress]' --output text ;;
  ip)     aws ec2 describe-instances --instance-ids "$IID" \
            --query 'Reservations[].Instances[].PublicIpAddress' --output text ;;
  ssh)    IP=$(aws ec2 describe-instances --instance-ids "$IID" --query 'Reservations[].Instances[].PublicIpAddress' --output text)
          echo "ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$IP" ;;
  stop)   aws ec2 stop-instances --instance-ids "$IID" --query 'StoppingInstances[].CurrentState.Name' --output text ;;
  terminate) aws ec2 terminate-instances --instance-ids "$IID" --query 'TerminatingInstances[].CurrentState.Name' --output text ;;
  *) echo "usage: aws_gpu.sh {launch [instance-type]|status|ip|ssh|stop|terminate}"; exit 2 ;;
esac
