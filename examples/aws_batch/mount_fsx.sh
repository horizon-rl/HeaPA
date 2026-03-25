#!/usr/bin/env bash

set -e

MOUNTPOINT="/fsx-sfai"

# Check if already mounted
if mountpoint -q "$MOUNTPOINT"; then
  echo "FSx is already mounted at $MOUNTPOINT"
  exit 0
fi

# Detect region via IMDSv2
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
REGION="us-west-2"
echo "Detected region: $REGION"

# Map region to FSx IP and mount name
case "$REGION" in
  ap-south-1) FSX_IP="172.16.1.71";  MOUNT_NAME="aeschbev" ;;
  us-west-2)  FSX_IP="172.16.17.66"; MOUNT_NAME="q6rwzb4v" ;;
  *) echo "Region $REGION not configured."; exit 1 ;;
esac

# Mount FSx
mkdir -p "$MOUNTPOINT"
mount -t lustre -o flock,_netdev "${FSX_IP}@tcp:/${MOUNT_NAME}" "$MOUNTPOINT"
echo "Mounted FSx at $MOUNTPOINT"