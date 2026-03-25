"""
Prepare a unified repeated evaluation set from multiple math benchmarks and
save it as a Parquet file.

For each dataset (AIME24, AIME25, MATH-500, MinervaMath, AMC23, GPQA,
OlympiadBench), we:
  - load the Hugging Face dataset,
  - repeat each test example N times (for evaluation with multiple samples),
  - wrap each problem into a standard prompt template,
  - store ground-truth answers and some extra metadata,
  - concatenate everything into a single pandas DataFrame,
  - save the result to a Parquet file,
  - print one example per dataset for sanity-checking.

Note:
  - The field names (e.g. "problem", "solution", "answer", "question") differ
    across datasets and are handled case-by-case below.
  - Some datasets store answers inside LaTeX \boxed{}; we strip these where
    necessary to match the evaluation style (rule-lighteval/MATH_V2).
"""

import pandas as pd
from datasets import load_dataset

# ======================================================================
# Global configuration
# ======================================================================

# Number of times to repeat each test question (e.g., for multi-sample eval)
N_REPEATS = 16

# Output parquet file path
OUTPUT_PATH = "./merged_math_dapo_eval_repeat16.parquet"

# Common prompt template used for all math problems
PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    'Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
    "{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)

# Container that will hold all evaluation entries (dicts)
total_eval_dataset = []

# ======================================================================
# AIME24
# ======================================================================

aime24 = load_dataset("math-ai/aime24")
for i in range(len(aime24["test"])):
    for j in range(N_REPEATS):
        # Many AIME-style solutions are wrapped in \boxed{}, so we strip that.
        gt = (
            aime24["test"][i]["solution"]
            .replace("\\boxed{", "")
            .replace("}", "")
        )

        total_eval_dataset.append(
            {
                "data_source": "math_dapo",
                "prompt": [
                    {
                        "content": PROMPT_TEMPLATE.format(
                            problem=aime24["test"][i]["problem"]
                        ),
                        "role": "user",
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "ground_truth": gt,
                    "style": "rule-lighteval/MATH_V2",
                },
                "extra_info": {
                    "index": f"aime24-question{i}-sample{j}",
                    "raw_problem": str(aime24["test"][i]["problem"]),
                    "raw_answer": str(aime24["test"][i]["solution"]),
                    "split": "test",
                    "dataset_name": "aime24",
                },
            }
        )

# ======================================================================
# AIME25
# ======================================================================

aime25 = load_dataset("math-ai/aime25")
for i in range(len(aime25["test"])):
    for j in range(N_REPEATS):
        # AIME25 uses "answer" directly as the ground-truth field.
        gt = aime25["test"][i]["answer"]

        total_eval_dataset.append(
            {
                "data_source": "math_dapo",
                "prompt": [
                    {
                        "content": PROMPT_TEMPLATE.format(
                            problem=aime25["test"][i]["problem"]
                        ),
                        "role": "user",
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "ground_truth": gt,
                    "style": "rule-lighteval/MATH_V2",
                },
                "extra_info": {
                    "index": f"aime25-question{i}-sample{j}",
                    "raw_problem": str(aime25["test"][i]["problem"]),
                    "raw_answer": str(aime25["test"][i]["answer"]),
                    "split": "test",
                    "dataset_name": "aime25",
                },
            }
        )

# ======================================================================
# MATH-500
# ======================================================================

math500 = load_dataset("HuggingFaceH4/MATH-500")
for i in range(len(math500["test"])):
    for j in range(N_REPEATS):
        # math500 has "problem" and "answer" fields, and a "solution" which
        # may contain a boxed answer. We keep answer as ground truth and
        # store a cleaned version of the solution in raw_answer.
        gt = math500["test"][i]["answer"]
        raw_solution_clean = (
            str(math500["test"][i]["solution"])
            .replace("$\\boxed{", "Answer: ")
            .replace("}.$", "")
        )

        total_eval_dataset.append(
            {
                "data_source": "math_dapo",
                "prompt": [
                    {
                        "content": PROMPT_TEMPLATE.format(
                            problem=math500["test"][i]["problem"]
                        ),
                        "role": "user",
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "ground_truth": gt,
                    "style": "rule-lighteval/MATH_V2",
                },
                "extra_info": {
                    "index": f"math500-question{i}-sample{j}",
                    "raw_problem": str(math500["test"][i]["problem"]),
                    "raw_answer": raw_solution_clean,
                    "split": "test",
                    "dataset_name": "math500",
                },
            }
        )

# ======================================================================
# MinervaMath
# ======================================================================

minervamath = load_dataset("math-ai/minervamath")
for i in range(len(minervamath["test"])):
    for j in range(N_REPEATS):
        gt = minervamath["test"][i]["answer"]

        total_eval_dataset.append(
            {
                "data_source": "math_dapo",
                "prompt": [
                    {
                        "content": PROMPT_TEMPLATE.format(
                            problem=minervamath["test"][i]["question"]
                        ),
                        "role": "user",
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "ground_truth": gt,
                    "style": "rule-lighteval/MATH_V2",
                },
                "extra_info": {
                    "index": f"minervamath-question{i}-sample{j}",
                    "raw_problem": str(minervamath["test"][i]["question"]),
                    "raw_answer": str(minervamath["test"][i]["answer"]),
                    "split": "test",
                    "dataset_name": "minervamath",
                },
            }
        )

# ======================================================================
# AMC23
# ======================================================================

amc23 = load_dataset("math-ai/amc23")
for i in range(len(amc23["test"])):
    for j in range(N_REPEATS):
        gt = amc23["test"][i]["answer"]

        total_eval_dataset.append(
            {
                "data_source": "math_dapo",
                "prompt": [
                    {
                        "content": PROMPT_TEMPLATE.format(
                            problem=amc23["test"][i]["question"]
                        ),
                        "role": "user",
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "ground_truth": gt,
                    "style": "rule-lighteval/MATH_V2",
                },
                "extra_info": {
                    "index": f"amc23-question{i}-sample{j}",
                    "raw_problem": str(amc23["test"][i]["question"]),
                    "raw_answer": str(amc23["test"][i]["answer"]),
                    "split": "test",
                    "dataset_name": "amc23",
                },
            }
        )

# ======================================================================
# GPQA
# ======================================================================

gpqa = load_dataset("math-ai/gpqa")
for i in range(len(gpqa["test"])):
    for j in range(N_REPEATS):
        # GPQA solutions may also use \boxed{}, so we strip that.
        gt = (
            gpqa["test"][i]["solution"]
            .replace("\\boxed{", "")
            .replace("}", "")
        )

        total_eval_dataset.append(
            {
                "data_source": "math_dapo",
                "prompt": [
                    {
                        "content": PROMPT_TEMPLATE.format(
                            problem=gpqa["test"][i]["problem"]
                        ),
                        "role": "user",
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "ground_truth": gt,
                    "style": "rule-lighteval/MATH_V2",
                },
                "extra_info": {
                    "index": f"gpqa-question{i}-sample{j}",
                    "raw_problem": str(gpqa["test"][i]["problem"]),
                    "raw_answer": str(gpqa["test"][i]["solution"]),
                    "split": "test",
                    "dataset_name": "gpqa",
                },
            }
        )

# ======================================================================
# OlympiadBench
# ======================================================================

olympiadbench = load_dataset("math-ai/olympiadbench")
for i in range(len(olympiadbench["test"])):
    for j in range(N_REPEATS):
        # final_answer is a list; join with comma for ground truth.
        gt = ", ".join(olympiadbench["test"][i]["final_answer"])

        total_eval_dataset.append(
            {
                "data_source": "math_dapo",
                "prompt": [
                    {
                        "content": PROMPT_TEMPLATE.format(
                            problem=olympiadbench["test"][i]["question"]
                        ),
                        "role": "user",
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "ground_truth": gt,
                    "style": "rule-lighteval/MATH_V2",
                },
                "extra_info": {
                    "index": f"olympiadbench-question{i}-sample{j}",
                    "raw_problem": str(olympiadbench["test"][i]["question"]),
                    "raw_answer": str(olympiadbench["test"][i]["final_answer"]),
                    "split": "test",
                    "dataset_name": "olympiadbench",
                },
            }
        )

# ======================================================================
# Convert to DataFrame and save
# ======================================================================

total_eval_dataset = pd.DataFrame(total_eval_dataset)
total_eval_dataset.to_parquet(OUTPUT_PATH, index=False)

print(
    "Finished preparing merged eval set parquet! Total samples:",
    len(total_eval_dataset),
)

# ======================================================================
# Sanity-check: print one example per dataset
# ======================================================================

datasets = [
    "aime24",
    "aime25",
    "math500",
    "minervamath",
    "amc23",
    "gpqa",
    "olympiadbench",
]

for name in datasets:
    # Filter rows whose extra_info["dataset_name"] == name
    sample = total_eval_dataset[
        total_eval_dataset["extra_info"].apply(
            lambda x: x["dataset_name"] == name
        )
    ].iloc[0]

    print("=" * 80)
    print(f"Dataset: {name}")
    print(f"Index: {sample['extra_info']['index']}")
    print(f"Prompt:\n{sample['prompt'][0]['content']}")
    print(f"\nGround Truth:\n{sample['reward_model']['ground_truth']}")
    print(f"\nExtra Info:\n{sample['extra_info']}")
