#!/usr/bin/env bash
# shellcheck disable=SC1090
# =========================
# Credit: Mingyu Zhao (SFAI-Verl/commit: c2e0c3ff)
# =========================

# Usage:
# # In package root directory:
# $ bash examples/aws_batch/job_launcher.sh greenland
 
set -eu
 
REGION=us-west-2 
GRS_NUM_NODES=0
USER=$(whoami)
VERL_NUM_NODES=2
SCHEDULER=$1  # local/aws_batch/greenland

ENABLE_EFA_HEALTHCHECK=False
RANDOM_TAG=$(printf '%s' $(echo "$RANDOM" | md5sum) | cut -c 1-24)
RANDOM_ID=$(printf '%s' $(echo "$RANDOM" | md5sum) | cut -c 1-10)
export JOB_NAME="${USER}-trainingjob-${RANDOM_ID}"

TRAIN_CMD="bash examples/rloo_trainer/run_qwen7b_math_gsm8k_megatron.sh"
RAY_START_CMD="bash /root/code/examples/aws_batch/start_ray_cluster.sh ${VERL_NUM_NODES} &&"

setup_scheduler_greenland() {
    echo "Setting up Scheduler greenland configuration..."
    AWS_SETUP="aws configure set --profile 'greenland' 'credential_source' 'EcsContainer'; \
        aws configure set --profile 'greenland' 'role_arn' 'arn:aws:iam::684288478426:role/GreenlandCrossAccountAccessRole'; \
        aws configure set --profile 'greenland' 'region' '${REGION}'; \
        aws configure set --profile 'greenland' s3.preferred_transfer_client crt; \
        aws configure set --profile 'greenland' s3.target_bandwidth 100Gb/s; \
        aws configure set --profile 'greenland' s3.max_concurrent_requests 32; \
        export AWS_PROFILE=greenland; "
 
    INITIATIVE_ID=Rufus-post-training
    INSTANCE_TYPE=p5en.48xlarge
    IS_PRODUCTION=false
    ROLE=arn:aws:iam::684288478426:role/GreenlandCrossAccountAccessRole
}
 
# Function to setup scheduler aws_batch variables
setup_scheduler_aws_batch() {
    echo "Setting up Scheduler aws_batch configuration..."
    AWS_SETUP=""
    DOCKER_IMAGE_TAG=nemo-nile-runner
    USE_BATCH=true
    JOB_QUEUE=FS-P5EN_48XL-Training-us-west-2d
    JOB_PRIORITY=0 # range 0-9999 (avoid use large number)
    SHARE_IDENTIFIER=Normal 
}
 
# Submit job based on scheduler selection
case $SCHEDULER in
    greenland)
        setup_scheduler_greenland
        ;;
    aws_batch)
        setup_scheduler_aws_batch
        ;;
    local)
        AWS_SETUP=""
        ;;
    *)
        echo "Error: Invalid scheduler name '$SCHEDULER'"
        exit 1
        ;;
esac
 
aws ecr get-login-password --region "${REGION}" | \
docker login --username AWS --password-stdin 684288478426.dkr.ecr."${REGION}".amazonaws.com

RUNCMD="${AWS_SETUP} ${RAY_START_CMD} ${TRAIN_CMD}"
echo "${RUNCMD}"
 
source ./examples/aws_batch/nile-runner-aws_batch_node_property_docker.sh "${SCHEDULER}"