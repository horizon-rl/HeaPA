set -x

project_name='Pool-Ablation'
exp_name='originalGRPO_Qwen25-7B-ins_30k-dapo-23k-aug-v2'

TRAIN_FILE="['/home/greenland-user/verl/data/pool_ablation_set1_30k-dapo-math_23k-augmentation.parquet']"
TEST_FILE="['/home/greenland-user/verl/data/aime-2024-960.parquet','/home/greenland-user/verl/data/aime-2025-960.parquet','/home/greenland-user/verl/data/amc23.parquet','/home/greenland-user/verl/data/math-500.parquet','/home/greenland-user/verl/data/minerva-math.parquet']"

RAY_ADDRESS=${RAY_ADDRESS:-"http://127.0.0.1:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

CKPTS_DIR=${CKPTS_DIR:-"/home/greenland-user/ckpts/${project_name}/${exp_name}"}
mkdir -p "${CKPTS_DIR}"

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
--working-dir "${WORKING_DIR}" \
--address "${RAY_ADDRESS}" \
-- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=5120 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1.5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_epochs=10 $@
