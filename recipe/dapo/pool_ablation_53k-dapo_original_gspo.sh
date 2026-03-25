#!/usr/bin/env bash
set -xeuo pipefail

# ---- experiment names --------------------------------------------------------
project_name='Pool-Ablation'
exp_name='originalGSPO_Qwen25-7B-ins_53k-dapo'

# ---- algo knobs --------------------------------------------------------------
adv_estimator=grpo
loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.0003 # as recommended by the paper, see Sec. 5.1
clip_ratio_high=0.0004 # as recommended by the paper, see Sec. 5.1

# Filter-groups expects "seq_final_reward" or "seq_reward" by default
enable_filter_groups=True
filter_groups_metric=seq_final_reward
max_num_gen_batches=10

# ---- lengths & buffers -------------------------------------------------------
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 5))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0
rollout_engine=vllm
rollout_mode=sync # can be async to speedup large scale xps

# ---- batching ----------------------------------------------------------------
train_prompt_bsz=512
n_resp_per_prompt=4
train_prompt_mini_bsz=128
ppo_mini_batch_size=128 # maintain 4 mini-batches as recommended by the paper, see Sec. 5.1
ppo_micro_batch_size_per_gpu=8 # setup depending on your GPU memory

# ---- rollout / sampling ------------------------------------------------------
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=4

# ---- Ray / paths -------------------------------------------------------------
RAY_ADDRESS=${RAY_ADDRESS:-"http://127.0.0.1:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=1

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
CKPTS_DIR=${CKPTS_DIR:-"/home/greenland-user/ckpts/${project_name}/${exp_name}"}
mkdir -p "${CKPTS_DIR}"

TRAIN_FILE="['/home/greenland-user/verl/data/pool_ablation_set2_53k-dapo-math.parquet']"
TEST_FILE="['/home/greenland-user/verl/data/aime-2024-960.parquet','/home/greenland-user/verl/data/aime-2025-960.parquet','/home/greenland-user/verl/data/amc23.parquet','/home/greenland-user/verl/data/math-500.parquet','/home/greenland-user/verl/data/minerva-math.parquet']"

# Optional seed reward map (only if you have one)
SEED_REWARD_MAP="/path/to/seed_reward_map.jsonl"

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
--working-dir "${WORKING_DIR}" \
--address "${RAY_ADDRESS}" \
-- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1.5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.entropy_checkpointing=true \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=false \
    trainer.test_freq=100 \
    trainer.save_freq=200 \
    trainer.total_epochs=20 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto