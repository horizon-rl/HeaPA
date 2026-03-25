#!/usr/bin/env bash
set -xeuo pipefail

############################################
# Scheduler + AWS/ECR + cluster bootstrap  #
############################################

REGION=us-east-1
USER="$(whoami)"
GRS_NUM_NODES=0
VERL_NUM_NODES=1
SCHEDULER="${1:-local}"  # local/aws_batch/greenland

ENABLE_EFA_HEALTHCHECK=False
RANDOM_TAG=$(printf '%s' "$(echo "$RANDOM" | md5sum)" | cut -c 1-24)
RANDOM_ID=$(printf '%s' "$(echo "$RANDOM" | md5sum)" | cut -c 1-10)
export JOB_NAME="wwq-${RANDOM_ID}"

# (Kept; not used directly by DAPO)
CONFIG_FILE="grpo_mini_dragon_grs_greenland.yaml"
GRS_RUN_CMD="serve run deployment_configs/qwen3-32B-no-think.yaml"
export GRS_ECR_IMAGE=684288478426.dkr.ecr.us-east-1.amazonaws.com/sfai-intern-repo:weifann_deepresearcher_verl_0.8.5_qwen3

project_name='DAPO_heap_Qwen25_7B_instruct_Query'
exp_name='DAPO-Qwen25-7B-instruct-20250911-test-obx'
# Instance storage for checkpoints (we'll reuse for DAPO CKPTS_DIR default)
export CHECKPOINT_DIR="/fsx-sfai/deficated-fsx-data-repo-pretraining-gl-us-west-2/home/wwq/${project_name}/${exp_name}"

# Ray cluster start (kept)
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

    INITIATIVE_ID=Rufus-SDB
    INSTANCE_TYPE=p4de.24xlarge
    IS_PRODUCTION=false
    ROLE=arn:aws:iam::684288478426:role/GreenlandCrossAccountAccessRole
}

setup_scheduler_aws_batch() {
    echo "Setting up Scheduler aws_batch configuration..."
    AWS_SETUP=""
    DOCKER_IMAGE_TAG=nemo-nile-runner
    USE_BATCH=true
    JOB_QUEUE=FS-P5EN_48XL-Training-us-west-2d
    JOB_PRIORITY=0
    SHARE_IDENTIFIER=Normal
}

case "$SCHEDULER" in
  greenland) setup_scheduler_greenland ;;
  aws_batch) setup_scheduler_aws_batch ;;
  local)     AWS_SETUP="" ;;
  *) echo "Error: Invalid scheduler name '$SCHEDULER'"; exit 1 ;;
esac

aws ecr get-login-password --region "${REGION}" | \
docker login --username AWS --password-stdin 684288478426.dkr.ecr."${REGION}".amazonaws.com

############################################
# DAPO job settings
############################################


adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 6))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=512
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=4
train_prompt_mini_bsz=32

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://127.0.0.1:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=1

# Paths / Model
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
# Use instance storage by default (safe across schedulers); override if you want.
CKPTS_DIR=${CKPTS_DIR:-"/fsx-sfai/deficated-fsx-data-repo-pretraining-gl-us-west-2/home/wwq/${project_name}/${exp_name}"}

TRAIN_FILE="['/fsx-sfai/deficated-fsx-data-repo-pretraining-gl-us-west-2/home/wwq/verl/data/dapo-math-17k-no_chinese.parquet']"
TEST_FILE="['/fsx-sfai/deficated-fsx-data-repo-pretraining-gl-us-west-2/home/wwq/verl/data/aime-2024-960.parquet','/fsx-sfai/deficated-fsx-data-repo-pretraining-gl-us-west-2/home/wwq/verl/data/aime-2025-960.parquet','/fsx-sfai/deficated-fsx-data-repo-pretraining-gl-us-west-2/home/wwq/verl/data/amc23.parquet','/fsx-sfai/deficated-fsx-data-repo-pretraining-gl-us-west-2/home/wwq/verl/data/math-500.parquet','/fsx-sfai/deficated-fsx-data-repo-pretraining-gl-us-west-2/home/wwq/verl/data/minerva-math.parquet']"

# Algorithm rollout params
temperature=1.0
top_p=1.0
top_k=-1        # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=4

############################################
# Compose the actual TRAIN_CMD invoked by runner
############################################

# If running on Greenland, mount FSx; otherwise skip
if [[ "$SCHEDULER" == "greenland" ]]; then
  # Ref: NileX greenland scheduler docs (kept)
  MOUNT_FSX_CMD="mkdir -p /fsx-sfai && mount -t lustre -o flock,_netdev 172.16.17.66@tcp:/q6rwzb4v /fsx-sfai && echo 'FSX mounted at /fsx-sfai' && "
else
  MOUNT_FSX_CMD=""
fi

# Build the Ray job submit command as one long command line
TRAIN_CMD="
cd /fsx-sfai/deficated-fsx-data-repo-pretraining-gl-us-west-2/home/wwq/verl && \
ray job submit --no-wait \
--runtime-env=\"${RUNTIME_ENV}\" \
--working-dir \"${WORKING_DIR}\" \
--address \"${RAY_ADDRESS}\" \
-- python3 -m recipe.dapo.main_dapo_augment \
data.train_files=${TRAIN_FILE} \
data.val_files=${TEST_FILE} \
data.prompt_key=prompt \
data.truncation='left' \
data.max_prompt_length=${max_prompt_length} \
data.max_response_length=${max_response_length} \
data.gen_batch_size=${gen_prompt_bsz} \
data.train_batch_size=${train_prompt_bsz} \
actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
algorithm.adv_estimator=${adv_estimator} \
algorithm.use_kl_in_reward=${use_kl_in_reward} \
algorithm.kl_ctrl.kl_coef=${kl_coef} \
actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
actor_rollout_ref.actor.clip_ratio_c=10.0 \
algorithm.filter_groups.enable=${enable_filter_groups} \
algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
algorithm.filter_groups.metric=${filter_groups_metric} \
+dynamic_data.enable=True \
+dynamic_data.max_pool_size=1000000 \
+dynamic_data.teacher_model=\"gpt-5-mini\" \
+dynamic_data.uniform_reward=0.5 \
+dynamic_data.seed_cap=1000000 \
+dynamic_data.eviction_policy=\"reject_new\" \
+augmentation.enable=True \
+augmentation.num_per_prompt=1 \
+augmentation.temperature=1 \
+augmentation.max_retries=3 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
actor_rollout_ref.model.path=\"${MODEL_PATH}\" \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.optim.lr=1.5e-6 \
actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
actor_rollout_ref.actor.optim.weight_decay=0.1 \
actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
actor_rollout_ref.actor.entropy_coeff=0 \
actor_rollout_ref.actor.grad_clip=1.0 \
actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
actor_rollout_ref.rollout.enable_chunked_prefill=True \
actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
actor_rollout_ref.rollout.temperature=${temperature} \
actor_rollout_ref.rollout.top_p=${top_p} \
actor_rollout_ref.rollout.top_k=\"${top_k}\" \
actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
actor_rollout_ref.rollout.val_kwargs.do_sample=True \
actor_rollout_ref.rollout.val_kwargs.n=1 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
reward_model.reward_manager=dapo \
reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
reward_model.overlong_buffer.len=${overlong_buffer_len} \
reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
trainer.logger='[\"console\",\"wandb\"]' \
trainer.project_name=\"${project_name}\" \
trainer.experiment_name=\"${exp_name}\" \
trainer.n_gpus_per_node=8 \
trainer.nnodes=\"${NNODES}\" \
trainer.val_before_train=False \
trainer.test_freq=100 \
trainer.save_freq=100 \
trainer.total_epochs=1 \
trainer.default_local_dir=\"${CKPTS_DIR}\" \
trainer.resume_mode=auto"

# Compose full runtime command
if [[ "$SCHEDULER" == "greenland" ]]; then
    RUNCMD="${AWS_SETUP} ${MOUNT_FSX_CMD} ${RAY_START_CMD} ${TRAIN_CMD}"
else
    RUNCMD="${AWS_SETUP} ${RAY_START_CMD} ${TRAIN_CMD}"
fi

echo "========== RUNCMD =========="
echo "${RUNCMD}"
echo "============================"

# Hand off to Nile runner wrapper (kept)
source ./examples/aws_batch/nile-runner-aws_batch_node_property_docker.sh "${SCHEDULER}"
