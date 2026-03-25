"""
Filter and reformat the OpenR1-Math-220k dataset into a Parquet file
compatible with the math_dapo evaluation format.

Pipeline:
  1. Load the "open-r1/OpenR1-Math-220k" Hugging Face dataset (train split).
  2. Filter out:
       - problems whose question text has >= 500 whitespace-separated tokens,
       - problems whose question or answer contains any non-ASCII characters
         (treated as "non-English").
  3. For the remaining examples, construct:
       - a unified prompt template instructing step-by-step reasoning
         and final answer format,
       - a reward_model with the ground-truth answer and style tag
         "rule-lighteval/MATH_V2",
       - extra_info with metadata (uuid, raw text, etc.).
  4. Save the resulting list of examples as a Parquet file.
"""

from datasets import load_dataset
import pandas as pd
from tqdm import trange
import re

# ======================================================================
# Helpers
# ======================================================================

# Consider "English-only" as strictly ASCII:
# Any character outside the range \x00-\x7F is treated as non-English.
_non_english_re = re.compile(r"[^\x00-\x7F]")


def has_non_english_char(text: str) -> bool:
    """
    Return True if `text` contains any non-ASCII (non-English) character.

    Args:
        text: Input string to check. `None` or empty string is treated as English-only.

    Returns:
        bool: True if any character falls outside ASCII range, False otherwise.
    """
    if not text:
        return False
    return bool(_non_english_re.search(text))


# ======================================================================
# Load OpenR1 dataset
# ======================================================================

# Requires authentication for gated datasets:
# e.g., run `huggingface-cli login` in your shell beforehand.
ds = load_dataset("open-r1/OpenR1-Math-220k", "default")["train"]

# Output container for transformed examples
total_dataset_after_transformation = []

# Counters for filtered-out samples
skip_length = 0
skip_non_english = 0

# Shared prompt template used for all problems
PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    'Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
    "{question}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)

# ======================================================================
# Main transformation loop
# ======================================================================

for i in trange(len(ds), desc="Processing OpenR1-Math-220k"):
    question = ds[i]["problem"]
    answer = ds[i]["answer"]

    # 1) Filter by length (in whitespace-delimited tokens).
    #    This approximates very long questions and removes them.
    if len(question.split()) >= 500:
        skip_length += 1
        continue

    # 2) Filter out problems/solutions that contain non-English
    #    (i.e., any non-ASCII) characters.
    if has_non_english_char(question) or has_non_english_char(answer):
        skip_non_english += 1
        continue

    # 3) Construct the standardized entry
    total_dataset_after_transformation.append(
        {
            "data_source": "math_dapo",
            "prompt": [
                {
                    "content": PROMPT_TEMPLATE.format(question=question),
                    "role": "user",
                }
            ],
            "ability": "math",
            "reward_model": {
                # Directly use "answer" field as ground truth
                "ground_truth": answer,
                "style": "rule-lighteval/MATH_V2",
            },
            "extra_info": {
                # Preserve original metadata for traceability
                "index": str(ds[i]["uuid"]),
                "raw_problem": str(ds[i]["problem"]),
                "raw_answer": str(ds[i]["solution"]),
                "split": str(ds[i]["source"]),
                "dataset_name": "OpenR1-Math-default",
            },
        }
    )

# ======================================================================
# Reporting and saving
# ======================================================================

print("Number of skipped samples due to length:", skip_length)
print("Number of skipped samples due to non-English chars:", skip_non_english)
print("Number of remaining samples:", len(total_dataset_after_transformation))

# Convert to DataFrame and write to Parquet
total_openR1_math_dataset = pd.DataFrame(total_dataset_after_transformation)
output_path = "./OpenR1-math-default-noRR.parquet"
total_openR1_math_dataset.to_parquet(output_path, index=False)

print("Full set written to:", output_path)
print("Total entries:", len(total_openR1_math_dataset))
