#!/usr/bin/env bash
set -xeuo pipefail

# ---- experiment names --------------------------------------------------------
project_name='HEAP-GSPO-Main'
exp_name='Qwen25-7B-ins_gspo-heap-aug-teacher-16rollout'

# ---- GSPO-specific knobs -----------------------------------------------------
adv_estimator=grpo
loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"

# ---- model / engine ----------------------------------------------------------
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
rollout_engine=vllm
rollout_mode=sync                # use 'async' for large-scale exps
gpu_memory_utilization=0.80
offload=false                    # small model; offloading slows training

# ---- lengths & buffers -------------------------------------------------------
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 5))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# ---- data --------------------------------------------------------------------
shuffle_dataset=true
first_time_dataset_prep=true

# ---- batching ----------------------------------------------------------------
train_batch_size=512
gen_prompt_bsz=$((train_batch_size * 2))
n_resp_per_prompt=8
ppo_mini_batch_size=128            # 4 mini-batches (512 / 128)
ppo_micro_batch_size_per_gpu=8     # tune per GPU memory

# ---- rollout / sampling ------------------------------------------------------
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
gen_tp=4

# ---- clipping (GSPO paper Sec. 5.1) ------------------------------------------
clip_ratio_low=0.0003
clip_ratio_high=0.0004

# ---- Ray / paths -------------------------------------------------------------
RAY_ADDRESS=${RAY_ADDRESS:-"http://127.0.0.1:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=2

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=${CKPTS_DIR:-"/home/greenland-user/ckpts/${project_name}/${exp_name}"}
mkdir -p "${CKPTS_DIR}"

TRAIN_FILE="['/home/greenland-user/verl/data/fullset_14k-dapo-math_34k-augmentation.parquet']"
TEST_FILE="['/home/greenland-user/verl/data/aime-2024-960.parquet']"

# Optional seed reward map (only if you have one)
SEED_REWARD_MAP="/path/to/seed_reward_map.jsonl"

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
--working-dir "${WORKING_DIR}" \
--address "${RAY_ADDRESS}" \
-- python3 -m recipe.gspo.main_gspo_augment_heap_teacher_extraction \
data.train_files=${TRAIN_FILE} \
data.val_files=${TEST_FILE} \
data.prompt_key=prompt \
data.truncation='left' \
data.max_prompt_length=${max_prompt_length} \
data.max_response_length=${max_response_length} \
data.gen_batch_size=${gen_prompt_bsz} \
data.train_batch_size=${train_batch_size} \
data.shuffle_dataset=${shuffle_dataset} \
data.first_time_dataset_prep=${first_time_dataset_prep} \
algorithm.adv_estimator=${adv_estimator} \
algorithm.loss_mode=${loss_mode} \
actor_rollout_ref.actor.loss_mode=${loss_mode} \
actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
actor_rollout_ref.actor.clip_ratio_c=10.0 \
actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
+dynamic_data.enable=True \
+dynamic_data.teacher_model="gpt-5-mini" \
+augmentation.enable=True \
+augmentation.num_per_prompt=1 \
+augmentation.pool.enable=True \
+augmentation.pool.max_size=300000 \
+augmentation.pool.low_fraction=0.5 \
+augmentation.pool.mixed_easy_medium=False \
+augmentation.pool.sample_per_step=32 \
+augmentation.pool.snapshot_every=200 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
actor_rollout_ref.model.path="${MODEL_PATH}" \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
actor_rollout_ref.actor.entropy_coeff=0 \
actor_rollout_ref.actor.grad_clip=1.0 \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
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
actor_rollout_ref.rollout.name=${rollout_engine} \
actor_rollout_ref.rollout.mode=${rollout_mode} \
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
trainer.test_freq=1000 \
trainer.save_freq=50 \
trainer.total_training_steps=2000 \
trainer.default_local_dir="${CKPTS_DIR}" \
trainer.resume_mode=auto
