#!/usr/bin/env bash
# Drive the AWS GPU lane. STATE LIVES HERE, not in my memory -- the 2026-07-25 lane lost its key to a /tmp
# reboot and needed an owner-granted IAM recovery. Key and instance id are recorded durably.
#
#   bash tools/aws_gpu.sh status | ip | ssh | stop | terminate
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT" || exit 1
STATE="$ROOT/research/queue/.aws_gpu"
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
  *) echo "usage: aws_gpu.sh {status|ip|ssh|stop|terminate}"; exit 2 ;;
esac
