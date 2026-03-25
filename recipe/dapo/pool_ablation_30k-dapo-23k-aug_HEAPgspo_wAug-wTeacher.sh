#!/usr/bin/env bash
set -xeuo pipefail

# ---- experiment names --------------------------------------------------------
project_name='Pool-Ablation'
exp_name='HEAPGSPO_Qwen25-7B-ins_30k-dapo-23k-aug_wAug-wTeacher'

# ---- algo knobs --------------------------------------------------------------
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.0003
clip_ratio_high=0.0004
loss_agg_mode="seq-mean-token-mean"

# Filter-groups expects "seq_final_reward" or "seq_reward" by default
enable_filter_groups=True
filter_groups_metric=seq_final_reward
max_num_gen_batches=20

# ---- lengths & buffers -------------------------------------------------------
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 5))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# ---- batching ----------------------------------------------------------------
train_prompt_bsz=528
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=4
train_prompt_mini_bsz=48

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
NNODES=2

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
CKPTS_DIR=${CKPTS_DIR:-"/home/greenland-user/ckpts/${project_name}/${exp_name}"}
mkdir -p "${CKPTS_DIR}"

TRAIN_FILE="['/home/greenland-user/verl/data/pool_ablation_set3_1400k-dapo-math_53k-augmentation.parquet']"
TEST_FILE="['/home/greenland-user/verl/data/aime-2024-960.parquet','/home/greenland-user/verl/data/aime-2025-960.parquet','/home/greenland-user/verl/data/amc23.parquet','/home/greenland-user/verl/data/math-500.parquet','/home/greenland-user/verl/data/minerva-math.parquet']"

# Optional seed reward map (only if you have one)
SEED_REWARD_MAP="/path/to/seed_reward_map.jsonl"

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
--working-dir "${WORKING_DIR}" \
--address "${RAY_ADDRESS}" \
-- python3 -m recipe.dapo.main_dapo_augment_teacher_extraction \
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
actor_rollout_ref.actor.policy_loss.loss_mode="gspo" \
actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
actor_rollout_ref.actor.clip_ratio_c=10.0 \
algorithm.filter_groups.enable=${enable_filter_groups} \
algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
algorithm.filter_groups.metric=${filter_groups_metric} \
+dynamic_data.enable=True \
+dynamic_data.max_pool_size=1500000 \
+dynamic_data.teacher_model="gpt-5-mini" \
+dynamic_data.init_mode="uniform" \
+dynamic_data.seed_cap=1500000 \
+dynamic_data.sampling_mode="medium_only" \
+dynamic_data.origin_reward_dump_path="${CKPTS_DIR}/origin_rewards.jsonl" \
+dynamic_data.uniform_reward=0.3 \
+dynamic_data.seed_reward_map="${SEED_REWARD_MAP}" \
+dynamic_data.moderate_band=0.25 \
+dynamic_data.return_fraction=0.20 \
+dynamic_data.min_keep=1 \
+dynamic_data.max_visits=3 \
+dynamic_data.ema_alpha=0.3 \
+augmentation.enable=True \
+augmentation.num_per_prompt=2 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
actor_rollout_ref.model.path="${MODEL_PATH}" \
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
actor_rollout_ref.rollout.top_k="${top_k}" \
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
trainer.logger='["console","wandb"]' \
trainer.project_name="${project_name}" \
trainer.experiment_name="${exp_name}" \
trainer.n_gpus_per_node=8 \
trainer.nnodes="${NNODES}" \
trainer.val_before_train=False \
trainer.test_freq=200 \
trainer.save_freq=200 \
trainer.total_epochs=1 \
trainer.default_local_dir="${CKPTS_DIR}" \
trainer.resume_mode=auto
