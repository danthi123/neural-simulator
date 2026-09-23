#!/usr/bin/env bash
# aws_cpu_launch.sh — launch ONE high-RAM CPU instance (r7i.4xlarge, 128GB) for the integrated brain-chat
# no-regression verify batch (RAM-blocked on the 46GB dev box). CPU/numpy only — no GPU, no cupy.
# Records the instance to research/queue/.aws_gpu IMMEDIATELY so it can NEVER be lost/leaked. Root volume
# DeleteOnTermination=true. SSH scoped to THIS box's current IP only. Idempotent-ish: refuses if one is already recorded.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
STATE="$ROOT/research/queue/.aws_gpu"
REGION=us-east-1
AMI=ami-05a3e9423ae4d7a19            # Ubuntu 22.04 x86_64, us-east-1 (newest as of 2026-09-15)
TYPE=r7i.4xlarge                     # 16 vCPU / 128 GiB
KEYNAME=claude-gpu-1785524741
KEY=/home/dant123/.ssh/aws-train/claude-gpu-1785524741.pem
SUBNET=subnet-0928cbecfe33fcbc1      # default-VPC subnet, us-east-1d

live=$(awk -F= '/^instance=/{print $2}' "$STATE" 2>/dev/null)
[ -n "$live" ] && { echo "⛔ an instance is already recorded ($live) — terminate it first (tools/aws_gpu.sh terminate)"; exit 1; }

# Hard pre-launch refusal — owner-approved 2026-09-23 daily spend cap, enforced by tooling not memory.
bash "$ROOT/tools/aws_budget.sh" check "$TYPE" || { echo "⛔ aws_cpu_launch: refused by tools/aws_budget.sh (daily cap) — see above"; exit 1; }

MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '\n')
[ -n "$MYIP" ] || { echo "could not determine my public IP"; exit 1; }
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region $REGION)
echo "[aws] my IP=$MYIP  VPC=$VPC  type=$TYPE"

SG=$(aws ec2 create-security-group --group-name "claude-cpu-verify-$(date +%s)" \
  --description "temp SSH for CPU verify batch" --vpc-id "$VPC" --region $REGION --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 \
  --cidr "${MYIP}/32" --region $REGION >/dev/null
echo "[aws] created SG $SG (ssh from ${MYIP}/32)"

IID=$(aws ec2 run-instances --region $REGION --image-id $AMI --instance-type $TYPE \
  --key-name $KEYNAME --security-group-ids "$SG" --subnet-id "$SUBNET" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=60,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=claude-cpu-verify},{Key=Project,Value=neural-sim}]' \
  --query 'Instances[0].InstanceId' --output text)

# RECORD DURABLY, immediately (before anything can interrupt) — this is the anti-leak anchor.
# A refused launch (e.g. VcpuLimitExceeded: the account's 32-vCPU quota = two r7i.4xlarge) returns an EMPTY id; the
# script used to record it, wait on "", and leave the SG orphaned (2026-09-23). Stop and clean up instead.
if [ -z "$IID" ] || [ "$IID" = "None" ]; then
  echo "⛔ aws_cpu_launch: run-instances returned no instance id (quota/capacity?) — deleting SG $SG and aborting" >&2
  aws ec2 delete-security-group --group-id "$SG" --region $REGION >/dev/null 2>&1 || true
  exit 1
fi
{ echo "# LIVE CPU verify-batch instance launched $(date '+%F %T %Z') — TERMINATE when done: tools/aws_gpu.sh terminate"
  echo "# then delete SG: aws ec2 delete-security-group --group-id $SG --region $REGION"
  echo "instance=$IID"; echo "region=$REGION"; echo "key=$KEY"; echo "sg=$SG"; } > "$STATE"
echo "[aws] launched $IID (recorded in $STATE). Waiting for running+status-ok…"
aws ec2 wait instance-status-ok --instance-ids "$IID" --region $REGION
IP=$(aws ec2 describe-instances --instance-ids "$IID" --region $REGION --query 'Reservations[].Instances[].PublicIpAddress' --output text)
echo "[aws] RUNNING. IP=$IP"
echo "[aws] ssh: ssh -i $KEY -o StrictHostKeyChecking=no ubuntu@$IP"
