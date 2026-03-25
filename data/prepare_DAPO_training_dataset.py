"""
Filter DAPO-Math dataset to remove problems containing CJK (Chinese/Japanese/Korean)
characters in the prompt text, then deduplicate by `extra_info["index"]` and save.

Steps:
  1. Load:
       - AIME 2024 parquet (currently unused in this script, but kept for consistency).
       - DAPO-Math 17k parquet.
  2. Detect rows where the problem text (extracted from the prompt) or the raw prompt
     contains CJK characters using a regex over CJK code ranges.
  3. Filter out those rows (keep only rows with *no* CJK).
  4. Report:
       - total rows
       - kept rows (no CJK)
       - removed rows (has CJK)
       - number of unique `extra_info["index"]` in the kept set.
  5. Ensure each `extra_info["index"]` appears at most once (deduplication).
  6. Save the filtered, deduplicated dataset to two parquet files:
       - "./dapo-math-14k-no_chinese-unique.parquet"
       - "./dapo-math-14k-no_chinese.parquet"
     (Note: as written, both files contain the same deduplicated content.)
"""

import pandas as pd
import re

# ----------------------------------------------------------------------
# Load original data
# ----------------------------------------------------------------------

# Use the original release from the DAPO recipe repo
# AIME24 is read here but not used below; you can remove it if unnecessary.
aime24 = pd.read_parquet("./aime-2024.parquet")
dapo_math = pd.read_parquet("./dapo-math-17k.parquet")

# ----------------------------------------------------------------------
# CJK detector (Han ideographs + common CJK punctuation/fullwidth forms)
# ----------------------------------------------------------------------

CJK_RE = re.compile(
    r"[\u3400-\u4DBF"   # CJK Unified Ideographs Extension A
    r"\u4E00-\u9FFF"    # CJK Unified Ideographs
    r"\u3000-\u303F"    # CJK Symbols and Punctuation
    r"\uFF00-\uFFEF"    # Halfwidth and Fullwidth Forms
    r"\U00020000-\U0002A6DF"  # CJK Unified Ideographs Extension B
    r"\U0002A700-\U0002CEAF"  # CJK Unified Ideographs Extensions C–E
    r"\U0002B740-\U0002B81F"  # CJK Unified Ideographs Extension F
    r"\U0002B820-\U0002CEAF"  # CJK Unified Ideographs Extensions G (overlaps)
    r"]"
)


def contains_cjk(s: str) -> bool:
    """
    Return True if string `s` contains at least one CJK character.

    Args:
        s: Input string to check.

    Returns:
        bool: True if any character matches CJK_RE, False otherwise.
    """
    if not isinstance(s, str):
        return False
    return bool(CJK_RE.search(s))


def extract_problem_text(content: str) -> str:
    """
    Extract the problem statement from a prompt string.

    The prompt is assumed to follow a template that includes:
      "... where $Answer is the answer to the problem.\n\n{problem}\n\nRemember ..."
    We try to:
      1. Split after the fixed prefix,
      2. Split before the "Remember..." footer,
      3. Strip the result.

    If extraction fails to produce a non-empty substring, fall back to the full content.

    Args:
        content: Full prompt text.

    Returns:
        str: Extracted problem text or the original content if extraction fails.
    """
    if not isinstance(content, str):
        return ""
    # Try to isolate the problem between the template markers
    s = content.split("where $Answer is the answer to the problem.\n\n")[-1]
    s = s.split("\n\nRemember to put your answer on its own line")[0]
    s = s.strip()
    # If extraction didn't shorten to anything meaningful, return full content
    return s if s else content


def row_has_chinese(row) -> bool:
    """
    Check if a DAPO-Math row's prompt contains any CJK characters.

    The `prompt` field is usually:
        - a list of dicts with a 'content' key, or
        - a single dict with 'content', or
        - something else (fallback: str(row["prompt"])).

    We:
      - collect all prompt text segments,
      - extract the "problem" portion via `extract_problem_text`,
      - check both the extracted problem and the raw content for CJK.

    Args:
        row: A pandas Series representing one row of the dataset.

    Returns:
        bool: True if any CJK character is found, False otherwise.
    """
    p = row.get("prompt", "")
    texts = []

    # Typical case: list of message dicts
    if isinstance(p, list):
        for item in p:
            if isinstance(item, dict) and "content" in item:
                texts.append(item["content"])
    # Single dict
    elif isinstance(p, dict) and "content" in p:
        texts.append(p["content"])
    # Fallback: cast entire prompt to string
    else:
        texts.append(str(p))

    # Check both raw text and extracted problem view
    for t in texts:
        problem = extract_problem_text(t)
        if contains_cjk(problem) or contains_cjk(t):
            return True

    return False


# ----------------------------------------------------------------------
# Filter rows with CJK in prompts
# ----------------------------------------------------------------------

# mask_keep is True for rows that DO NOT have Chinese/CJK characters
mask_keep = ~dapo_math.apply(row_has_chinese, axis=1)
dapo_math_filtered = dapo_math[mask_keep].reset_index(drop=True)

print(f"Total rows: {len(dapo_math)}")
print(f"Kept (no Chinese): {len(dapo_math_filtered)}")
print(f"Removed (has Chinese): {len(dapo_math) - len(dapo_math_filtered)}")

# ----------------------------------------------------------------------
# Analyze unique indices in the kept subset
# ----------------------------------------------------------------------

total_index = {}
for i in dapo_math_filtered.index:
    idx = dapo_math_filtered["extra_info"].iloc[i]["index"]
    total_index[idx] = 1

print(f"Unique indices in kept set: {len(total_index)}")

# ----------------------------------------------------------------------
# Deduplicate so each index appears at most once
# ----------------------------------------------------------------------

dapo_math_filtered_unique = []
seen_indices = set()

for i in dapo_math_filtered.index:
    idx = dapo_math_filtered["extra_info"].iloc[i]["index"]
    if idx not in seen_indices:
        dapo_math_filtered_unique.append(dapo_math_filtered.iloc[i])
        seen_indices.add(idx)

dapo_math_filtered = pd.DataFrame(dapo_math_filtered_unique).reset_index(drop=True)
print(f"After ensuring unique indices, total rows: {len(dapo_math_filtered)}")

# ----------------------------------------------------------------------
# Save filtered datasets
# ----------------------------------------------------------------------
# NOTE: In the current script, both files below are written from the same
#       deduplicated `dapo_math_filtered` DataFrame. If you want one file
#       to contain pre-dedup data, you should save before overwriting the
#       variable with the unique-only version.

# Save filtered DAPO-Math with unique indices (no Chinese)
dapo_math_filtered.to_parquet(
    "./dapo-math-14k-no_chinese-unique.parquet",
    index=False,
)

# Save again to an alternate filename (currently identical content)
dapo_math_filtered.to_parquet(
    "./dapo-math-14k-no_chinese.parquet",
    index=False,
)
