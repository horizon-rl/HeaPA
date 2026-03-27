# HeaPA: Difficulty-Aware Heap Sampling and On-Policy Query Augmentation for LLM Reinforcement Learning

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2601.22448-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2601.22448)
[![Notion](https://img.shields.io/badge/Notion-000?logo=notion&logoColor=fff)](https://weiqiwang.notion.site/HeaPA-Difficulty-Aware-Heap-Sampling-and-On-Policy-Query-Augmentation-for-LLM-Reinforcement-Learnin-2c0be69e159180058fcfd3c75c0f4dc9?pvs=73)


This repository contains the official implementation of **HeaPA**, an intern project by Weiqi, which explores:
- **Heap-based difficulty-aware sampling** of training queries, and  
- **On-policy query augmentation** for improving the efficiency of reinforcement learning (RL) on large language models (LLMs).


## 1. Dependencies

This codebase is built on top of **[VERL](https://github.com/volcengine/verl)**.

1. Install VERL and its dependencies by following the official installation guide:  
   https://verl.readthedocs.io/en/latest/start/install.html
2. Additionally, install the `openai` Python package:
   ```bash
   pip install openai
   ```

Make sure your environment can successfully run basic VERL examples before running HeaPA experiments.

## 2. Data Preparation

All data preparation scripts are located in the `data/` directory.

We provide three scripts:

* **DAPO training dataset**

  * `data/prepare_DAPO_training_dataset.py`
    Collects and formats the **DAPO-Math** training dataset used for RL training.

* **OpenR1 training dataset**

  * `data/prepare_OpenR1_training_dataset.py`
    Collects and filters the **OpenR1-Math** training dataset into the format expected by our HeaPA pipeline.

* **Evaluation benchmarks**

  * `data/prepare_eval_benchmarks.py`
    Prepares the evaluation suites (e.g., AIME24, AIME25, MATH500, MinervaMath, AMC23, GPQA, OlympiadBench) into a unified eval format.

Each script has inline comments describing the exact datasets used and the output file formats (typically Parquet). Read and run them individually depending on which training/evaluation setup you want to reproduce.

## 3. HeaPA Training Recipes

All training recipes are located in `recipe/dapo`.

### Main HeaPA recipe

Our primary training entrypoint is:

- `recipe/dapo/dapo_ray_trainer_HeaPA.py`  
  This script runs the standard **HeaPA** pipeline with:
  - difficulty-aware heap-based query sampling, and  
  - on-policy query augmentation.

### Variants

We also provide two variants for ablations and comparisons:

- `recipe/dapo/dapo_ray_trainer_HeaPA_multi_heap.py`  
  Uses the **multi-heap** variant of HeaPA, where queries are partitioned into multiple heaps (e.g., by difficulty bands) for more fine-grained sampling.

- `recipe/dapo/dapo_ray_trainer_prioritized_sampling_baseline.py`  
  Replaces heap sampling with a simpler **prioritized sampling** baseline, which serves as a comparison point against HeaPA’s heap-based sampler.

Each script comes with its own configuration options and comments; see the file-level docstrings and argument definitions for usage details.

Here’s a filled-in / polished version of that section based on your script:

## 4. Launching HeaPA Training

We provide an example launch script in `scripts/` that shows how to run HeaPA end-to-end. Below are hyperparameters you can play with:

### Backbone model: Qwen2.5 vs Qwen3

```bash
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
```

To switch backbone:

* **Qwen3** → e.g. `Qwen/Qwen3-4B-Instruct-2507`
* **Qwen2.5** → e.g. `Qwen/Qwen2.5-7B-Instruct`

Just update `MODEL_PATH` and (optionally) `project_name` / `exp_name` to reflect the model.

### Dataset: DAPO vs OpenR1 (or custom)

```bash
TRAIN_FILE="['./data/dapo-math-14k-no_chinese-unique.parquet']"

data.train_files=${TRAIN_FILE}
```

* To switch to **OpenR1** training data, point `TRAIN_FILE` to the parquet produced by `prepare_OpenR1_training_dataset.py`.
* To add or mix datasets, use a Python-list string:
  `TRAIN_FILE="['path1.parquet','path2.parquet']"`.

### GRPO vs DAPO

```bash
# DAPO
loss_agg_mode="token-mean"
enable_filter_groups=True
```

```bash
# GRPO
loss_agg_mode="seq-mean-token-mean"
enable_filter_groups=False
```

This is the main place to flip between GRPO/DAPO-style training while keeping the rest of the HeaPA machinery fixed.

#### 5. HeaPA-specific knobs: dynamic data & augmentation

```bash
+dynamic_data.enable=True
+dynamic_data.max_pool_size=1000000
+dynamic_data.teacher_model="gpt-5-nano"
+dynamic_data.init_mode="uniform"
+dynamic_data.seed_cap=1000000
+dynamic_data.sampling_mode="medium_only"
+dynamic_data.refresh_reward="path_aggregation"
+dynamic_data.obx_checkpoints_upload=True

+augmentation.enable=True
+augmentation.num_per_prompt=1
```

* `dynamic_data.enable=True` turns on the **on-policy query augmentation / heap pool**.
* `dynamic_data.teacher_model` controls the teacher used to generate new queries.
* `dynamic_data.sampling_mode` is the key difficulty-curriculum knob:

  * `medium_only` → HeaPA’s "medium-difficulty focused" sampling.
  * `easy_medium_mixed` → HeaPA’s mixed sampling from easy and medium boundaries.
* `dynamic_data.refresh_reward` chooses how to recompute rewards for the pool:

  * `path_aggregation` → HeaPA’s path-agg style reward refresh with number-of-children weighted path aggregation.
  * `children_aggregation` → HeaPA’s child-agg style reward refresh.
* `augmentation.enable=True` with `augmentation.num_per_prompt` controls how many new augmented problems are generated per seed query.

Disabling HeaPA (for a plain baseline) is as simple as setting:

```bash
+dynamic_data.enable=False
+augmentation.enable=False
```

and/or using the `prioritized_sampling_baseline` recipe.

#### 6. Lengths, batch sizes, and rollout controls

Some of the main training shape/throughput parameters:

```bash
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))
train_prompt_bsz=512
train_prompt_mini_bsz=64
n_resp_per_prompt=16
temperature=1.0
top_p=1.0
val_top_p=0.7
```

* `max_prompt_length`, `max_response_length` → total context for training and rollout.
* `train_prompt_bsz` / `train_prompt_mini_bsz` → global vs mini-batch size for PPO.
* `n_resp_per_prompt` → number of rollouts per prompt (e.g., 16).
* `temperature`, `top_p`, `top_k` → sampling settings for rollouts and validation.

All other flags are reasonable defaults reproduced from our experiments and can be tuned as needed for your hardware and desired training regime.

---

## 5. HeaPA Training with Slime (Megatron + SGLang backend)

The `slime/` directory contains an alternative training stack that replaces VERL's FSDP actor with **Megatron-LM** and uses **SGLang** for rollout generation.  This is the recommended path for large-scale runs on clusters where the slime Docker image is available.

### 5.1 Prerequisites

| Requirement | Notes |
|---|---|
| Docker image | `slimerl/slime:nightly-dev-20260202c` (or later nightly) |
| Slime root | `/root/slime` inside the container |
| Megatron-LM | `/root/Megatron-LM` inside the container |
| Python packages | `openai` (for teacher augmentation) |

The container already has `megatron-core`, `megatron-bridge`, and SGLang installed.

### 5.2 File structure

We do not include all files from slime but add some core files. You need to enter the slime docker manully and run the following script.

```
slime/
├── __init__.py               # package marker
├── train.py                  # entry-point wrapper (pre-parses HeaPA args,
│                             #   patches slime data utils, delegates to /root/slime/train.py)
├── heapa_core.py             # ThreadSafeQueryPool + QueryRecord (no VERL dependency)
├── heapa_teacher.py          # SlimeTeacherAnnotator (OpenAI API)
├── heapa_data_source.py      # HeaPADataSource — slime DataSource ABC implementation
├── heapa_rollout.py          # generate_rollout — wraps slime sglang rollout + pool update hook
└── HeaPA_DAPO_PathAgg_MediumOnly_DAPOMath.sh              # end-to-end launch script
```

### 5.3 Data format

The training and eval parquets must follow the **verl dataset format**:

| Column | Type | Description |
|---|---|---|
| `prompt` | list of dicts | Chat-template messages, e.g. `[{"role": "user", "content": "..."}]` |
| `reward_model` | dict | Must contain `"ground_truth"` key with the reference answer |
| `data_source` | str | Dataset name tag (e.g. `"dapo_math"`) |

The `label` column is **not** required — `slime/heapa_data_source.py` automatically extracts it from `reward_model["ground_truth"]` at load time.

Use the preparation scripts in `data/` to produce correctly formatted parquets.

### 5.4 Launching a training run

1. **Start your docker environment**:
  Remember to setup a file mapping, e.g., `-v /tmp/instance_storage:/tmp/instance_storage` so that you can copy the code files into your docker environment.
  ```
  docker run --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --privileged --network host \
    --name=slime \
    -v /tmp/instance_storage:/tmp/instance_storage \
    --mount type=tmpfs,destination=/tmpfs,tmpfs-mode=1777 \
    --device /dev/infiniband/uverbs0:/dev/infiniband/uverbs0:rwm \
    --device /dev/infiniband/uverbs1:/dev/infiniband/uverbs1:rwm \
    --device /dev/infiniband/uverbs2:/dev/infiniband/uverbs2:rwm \
    --device /dev/infiniband/uverbs3:/dev/infiniband/uverbs3:rwm \
    --entrypoint /bin/bash -it slimerl/slime:nightly-dev-20260202c
  ```

2. **Set required environment variables**:
   ```bash
   export WANDB_API_KEY=<your_key>
   export OPENAI_API_KEY=<your_key>   # only needed when teacher augmentation is enabled
   ```

3. **Copy slime files into your docker**:
  Docker environment is using a separate file system so you need to manully copy the slime folder into it.
   ```bash
   cp -r /path/to/HeapAugment-DAPO-local /path/to/HeapAugment-DAPO
   cd /path/to/HeapAugment-DAPO
   ```

4. **Edit path overrides** in `slime/HeaPA_DAPO_PathAgg_MediumOnly_DAPOMath.sh` (or pass as env vars):
   ```bash
   RAY_DATA_HOME=/path/to/HeapAugment-DAPO
   MODEL_PATH=/path/to/Qwen3-4B-Instruct
   PROMPT_DATA=/path/to/dapo-math-14k-no_chinese-unique.parquet
   EVAL_DATA=/path/to/aime-2024-960.parquet
   SAVE_PATH=/path/to/experiment/checkpoints
   ```

5. **Submit**:
   ```bash
   bash slime/HeaPA_DAPO_PathAgg_MediumOnly_DAPOMath.sh
   ```
   The script submits a Ray job and exits immediately (`--no-wait`).  Monitor with:
   ```bash
   ray job logs --follow <job_id>
   ```

### 5.5 Key HeaPA knobs

All HeaPA arguments are prefixed `--heapa-` and pre-parsed by `slime/train.py` before the Megatron argument parser runs.

| Env var | CLI flag | Default | Description |
|---|---|---|---|
| `POOL_MAX_SIZE` | `--heapa-pool-max-size` | `1000000` | Max queries kept in heap pool |
| `LOW_FRACTION` | `--heapa-low-fraction` | `0.5` | Fraction of pool in the low (hard) partition |
| — | `--heapa-mixed-sampling` | off | Enable mixed easy+medium sampling (omit for medium-only) |
| `TEACHER_ENABLED` | `--heapa-teacher-enabled` | off | Enable teacher augmentation via OpenAI API |
| `TEACHER_MODEL` | `--heapa-teacher-model` | `gpt-5-nano` | OpenAI model used for teacher |
| `TEACHER_WORKERS` | `--heapa-teacher-workers` | `4` | Async worker threads for teacher calls |
| `TEACHER_HARD_LO` | `--heapa-teacher-hard-lo` | `0.1` | Lower reward bound for medium-difficulty gate |
| `TEACHER_HARD_HI` | `--heapa-teacher-hard-hi` | `0.7` | Upper reward bound for medium-difficulty gate |

**Medium-only mode** (default): teacher is triggered for queries whose average reward falls in `(TEACHER_HARD_LO, TEACHER_HARD_HI)`.
**Mixed sampling mode**: pass `--heapa-mixed-sampling` to also draw from the easy (high-reward) partition.

### 5.6 PYTHONPATH

When running outside the provided `HeaPA_DAPO_PathAgg_MediumOnly_DAPOMath.sh`, make sure the following are all on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/HeapAugment-DAPO:/root/slime:/root/Megatron-LM:${PYTHONPATH}
```

---

## 📖 Citation

```
@misc{wang2026heapa,
      title={HeaPA: Difficulty-Aware Heap Sampling and On-Policy Query Augmentation for LLM Reinforcement Learning}, 
      author={
         {Weiqi Wang} and
         {Xin Liu} and
         {Binxuan Huang} and
         {Hejie Cui} and
         {Rongzhi Zhang} and
         {Changlong Yu} and
         {Shuowei Jin} and
         {Jingfeng Yang} and
         {Qingyu Yin} and
         {Zhengyang Wang} and
         {Zheng Li} and
         {Yifan Gao} and
         {Priyanka Nigam} and
         {Bing Yin} and
         {Lihong Li} and
         {Yangqiu Song}
      },
      year={2026},
      eprint={2601.22448},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
}
```
