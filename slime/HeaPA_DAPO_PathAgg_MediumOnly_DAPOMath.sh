#!/usr/bin/env bash
# run_heapa.sh — Launch HeaPA training with slime
#
# Mirrors scripts/HeaPA_DAPO_PathAgg_MediumOnly_DAPOMath.sh (verl variant).
# Key settings: Qwen3-4B, DAPO-Math-14k, medium-only pool, teacher augmentation,
#               16 responses/prompt, token-mean loss, 500 training steps.
#
# Usage:
#   export WANDB_API_KEY=<key>
#   bash slime/run_heapa.sh

set -euo pipefail

# ─── Experiment names ────────────────────────────────────────────────────────
project_name='HEAP-DAPO-Qwen3-4B-20260322'
exp_name='PathAgg_MediumOnly_17kdapo_wAugTeacherRewardRefresh_16r20k'

# ─── Paths ───────────────────────────────────────────────────────────────────
RAY_DATA_HOME="${RAY_DATA_HOME:-${PWD}}"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
PROMPT_DATA="${PROMPT_DATA:-${RAY_DATA_HOME}/data/dapo-math-14k-no_chinese-unique.parquet}"
EVAL_DATA="${EVAL_DATA:-${RAY_DATA_HOME}/data/aime-2024-960.parquet}"
SAVE_PATH="${SAVE_PATH:-${PWD}/exp/${project_name}/${exp_name}}"
LOAD_PATH="${LOAD_PATH:-}"          # set to resume from checkpoint

REPO_ROOT="${REPO_ROOT:-${RAY_DATA_HOME}}"
SLIME_ROOT="${SLIME_ROOT:-/root/slime}"
MEGATRON_LM_ROOT="${MEGATRON_LM_ROOT:-/root/Megatron-LM}"

export PYTHONPATH="${REPO_ROOT}:${SLIME_ROOT}:${MEGATRON_LM_ROOT}:${PYTHONPATH:-}"

# ─── Cluster ─────────────────────────────────────────────────────────────────
NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
# gen_tp=8: one sglang engine using all 8 GPUs for tensor-parallel rollout
ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-8}"

# ─── Batching ────────────────────────────────────────────────────────────────
# gen_prompt_bsz=512 in verl (rollout batch)
ROLLOUT_BATCH="${ROLLOUT_BATCH:-512}"
# n_resp_per_prompt=16
N_SAMPLES="${N_SAMPLES:-16}"
# train_prompt_bsz=256 in verl (PPO global batch)
GLOBAL_BATCH="${GLOBAL_BATCH:-4096}"

# ─── Lengths ─────────────────────────────────────────────────────────────────
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-20480}"

# ─── HeaPA pool & teacher ────────────────────────────────────────────────────
POOL_MAX_SIZE="${POOL_MAX_SIZE:-1000000}"
LOW_FRACTION="${LOW_FRACTION:-0.5}"
# medium_only mode: do NOT pass --heapa-mixed-sampling
TEACHER_ENABLED="${TEACHER_ENABLED:-true}"
TEACHER_MODEL="${TEACHER_MODEL:-gpt-5-nano}"
TEACHER_WORKERS="${TEACHER_WORKERS:-4}"
TEACHER_HARD_LO="${TEACHER_HARD_LO:-0.1}"
TEACHER_HARD_HI="${TEACHER_HARD_HI:-0.7}"

# Use a dedicated temp-dir so Ray always starts with a clean GCS state
RAY_TEMP_DIR="/tmp/ray_heapa"
rm -rf "${RAY_TEMP_DIR}"
mkdir -p "${RAY_TEMP_DIR}"

# ─── Start Ray head ──────────────────────────────────────────────────────────
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --temp-dir "${RAY_TEMP_DIR}" \
    --num-gpus "${NUM_GPUS_PER_NODE}" \
    --disable-usage-stats

# ─── Build CLI args ──────────────────────────────────────────────────────────
CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --ref-load "${MODEL_PATH}"
    --save "${SAVE_PATH}"
)
if [ -n "${LOAD_PATH}" ]; then
    CKPT_ARGS+=(--load "${LOAD_PATH}")
fi

# HeaPA data source and rollout function (medium-only: no --heapa-mixed-sampling)
# --heapa-* args are pre-parsed by slime/train.py before slime's Megatron parser runs.
HEAPA_ARGS=(
    --data-source-path slime.heapa_data_source.HeaPADataSource
    --rollout-function-path slime.heapa_rollout.generate_rollout
    --heapa-pool-max-size "${POOL_MAX_SIZE}"
    --heapa-low-fraction "${LOW_FRACTION}"
    --heapa-teacher-hard-lo "${TEACHER_HARD_LO}"
    --heapa-teacher-hard-hi "${TEACHER_HARD_HI}"
)
if [ "${TEACHER_ENABLED}" = "true" ]; then
    HEAPA_ARGS+=(
        --heapa-teacher-enabled
        --heapa-teacher-model "${TEACHER_MODEL}"
        --heapa-teacher-workers "${TEACHER_WORKERS}"
    )
fi

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type dapo
    --reward-key score
    --num-rollout 500
    --rollout-batch-size "${ROLLOUT_BATCH}"
    --n-samples-per-prompt "${N_SAMPLES}"
    --rollout-max-prompt-len "${MAX_PROMPT_LEN}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --over-sampling-batch-size "$(( ROLLOUT_BATCH * 2 ))"
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --global-batch-size "${GLOBAL_BATCH}"
)

EVAL_ARGS=(
    --eval-interval 10
    --eval-prompt-data math "${EVAL_DATA}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len "${MAX_RESPONSE_LEN}"
    --eval-top-p 0.7
    --eval-top-k 1
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 1e-6
    --kl-loss-type low_var_kl
    --eps-clip 0.2
    --eps-clip-high 0.28
    --loss-agg-mode token-mean
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-warmup-iters 0
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.999
)

SGLANG_ARGS=(
    --rollout-num-gpus "${ROLLOUT_GPUS}"
    --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
    --sglang-chunked-prefill-size 128
    --sglang-mem-fraction-static 0.8
    --sglang-enable-metrics
)

PERF_ARGS=(
    --tensor-model-parallel-size 8
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --use-dynamic-batch-size
    --max-tokens-per-gpu "$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    # --accumulate-allreduce-grads-in-fp32
    # --attention-softmax-in-fp32
    --attention-backend flash
    --actor-num-nodes "${NUM_ACTOR_NODES}"
    --actor-num-gpus-per-node "${NUM_GPUS_PER_NODE}"
    --colocate
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --megatron-to-hf-mode bridge
)

MISC_ARGS=(
    --save-interval 10
    --wandb-project "${project_name}"
    --wandb-exp-name "${exp_name}"
    --use-wandb
    --wandb-key "${WANDB_API_KEY}"
    --wandb-group slime-HeaPA
)

# ─── Launch ──────────────────────────────────────────────────────────────────
RUNTIME_ENV=$(python3 -c "
import json, os
env = {
    'PYTHONPATH': '${REPO_ROOT}:${SLIME_ROOT}:${MEGATRON_LM_ROOT}',
    'SLIME_ROOT': '${SLIME_ROOT}',
    'CUDA_DEVICE_MAX_CONNECTIONS': '1',
    'MASTER_ADDR': '${MASTER_ADDR}',
    'NCCL_NVLS_ENABLE': '1',
    'WANDB_MODE': 'online',
}
for k in ['WANDB_API_KEY', 'WANDB_ENTITY', 'OPENAI_API_KEY']:
    if os.environ.get(k):
        env[k] = os.environ[k]
print(json.dumps({'env_vars': env}))
")

# Load model architecture args for Qwen3-4B (populates MODEL_ARGS array)
# shellcheck source=/dev/null
source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"
# qwen3-4B.sh sets rotary_base=1000000 but Qwen3-4B-Instruct-2507 uses rope_theta=5000000
MODEL_ARGS+=(--rotary-base 5000000)

ray job submit \
    --runtime-env-json "${RUNTIME_ENV}" \
    --no-wait \
    -- python3 "${REPO_ROOT}/slime/train.py" \
        "${CKPT_ARGS[@]}" \
        "${HEAPA_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" \
        "${EVAL_ARGS[@]}" \
        "${GRPO_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" \
        "${SGLANG_ARGS[@]}" \
        "${PERF_ARGS[@]}" \
        "${MODEL_ARGS[@]}" \
        "${MISC_ARGS[@]}"

echo "HeaPA training job submitted. Monitor with: ray job logs --follow <job_id>"
