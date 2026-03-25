# eval_avg_at_k_multi.py
# Dependencies: pandas, pyarrow, tqdm
# Optional: sympy (for exact numeric equivalence)

import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from tqdm import tqdm
import pandas as pd

try:
    from sympy import sympify
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False


# ----------------------- CONFIG: EDIT THIS BLOCK -----------------------
# If GENERATIONS is non-empty, the script will ONLY evaluate those files.
# If GENERATIONS is empty, the script will scan ROOT_DIRS (or CWD if ROOT_DIRS
# is also empty) and automatically find all "generations.parquet" files.
GENERATIONS = [
    # e.g. "/home/greenland-user/eval_results/SomeRun/global_step_120/generations.parquet",
]

# Root directories to search when GENERATIONS is empty.
ROOT_DIRS = [
    "./eval_results/",
]

K_LIST = [16]  # e.g. [1, 2, 4, 8, 16, 32]
# ----------------------------------------------------------------------


# ----------------------- VERIFICATION HELPERS -----------------------

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx: right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string."""
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left): -1]


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""),
    ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","), ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km", "units", "\\ldots", "sue",
    "points", "feet", "minutes", "digits", "cents", "degrees", "cm", "gm", "pounds", "meters", "meals",
    "edges", "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}",
    r"\;", r",\!", "{,}", '"', "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def _numeric_equal(a_str: str, b_str: str) -> bool:
    """Optional numeric equivalence via sympy or float fallback."""
    if _HAS_SYMPY:
        try:
            a = sympify(a_str)
            b = sympify(b_str)
            if a.equals(b):
                return True
            return float(a.evalf()) == float(b.evalf())
        except Exception:
            pass
    try:
        return float(a_str) == float(b_str)
    except Exception:
        return False


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> Tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria."""
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    if pred == gt or _numeric_equal(pred, gt):
        return True, pred
    return False, pred


def verify(solution_str: str, answer: str) -> Tuple[bool, str]:
    """Verify if the solution is correct (Minerva-style)."""
    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def compute_score(solution_str: str, ground_truth: str) -> Dict[str, Any]:
    """Compute a simple reward & acc."""
    # Limit tail for speed; answers appear at the end.
    solution_str = solution_str[-300:]
    correct, pred = verify(solution_str, ground_truth)
    return {"score": 1.0 if correct else -1.0, "acc": bool(correct), "pred": pred}


# ----------------------- PARSING & METRICS -----------------------

_INDEX_RE = re.compile(
    r"(?P<dataset>[A-Za-z0-9_]+)-question(?P<q>\d+)-sample(?P<s>\d+)")


def parse_index(extra_info: Dict[str, Any]) -> Tuple[str, int, int]:
    """Parse dataset, question_id, sample_id from extra_info['index'] (fallback to dataset_name)."""
    idx = (extra_info or {}).get("index", "") or ""
    m = _INDEX_RE.search(idx)
    dataset_name = (extra_info or {}).get("dataset_name")
    if m:
        dataset = dataset_name or m.group("dataset")
        q = int(m.group("q"))
        s = int(m.group("s"))
        return dataset, q, s
    # Fallbacks
    dataset = dataset_name or "unknown"
    q = int((extra_info or {}).get("question_id", 0))
    s = int((extra_info or {}).get("sample_id", 0))
    return dataset, q, s


def extract_row_fields(row: pd.Series) -> Tuple[str, int, int, str, str]:
    """
    Return (dataset, question_id, sample_id, solution_str, ground_truth).
    - solution_str is taken from 'responses' (string or list; use first item if list)
    - ground_truth from reward_model.ground_truth or extra_info.raw_answer
    """
    extra = row.get("extra_info", {}) or {}
    rw = row.get("reward_model", {}) or {}
    responses = row.get("responses", []) or []

    if (not extra or not rw) and len(row.index) == 1 and isinstance(row.iloc[0], dict):
        record = row.iloc[0]
        extra = record.get("extra_info", extra)
        rw = record.get("reward_model", rw)
        responses = record.get("responses", responses)

    dataset, q_id, s_id = parse_index(extra)

    solution_str = responses[0]

    gt = rw.get("ground_truth")
    if gt is None:
        gt = (extra or {}).get("raw_answer", "")

    return dataset, q_id, s_id, solution_str, str(gt)


def compute_dataset_metrics(ans_by_q: Dict[int, List[Tuple[int, bool]]]) -> Dict[str, Any]:
    """
    ans_by_q: q_id -> list of (sample_id, correct_bool)
    Returns:
      {
        "avg_at_k": {k: float},
        "pass_at_k": {k: float},
        "num_questions": int
      }
    """
    num_q = len(ans_by_q)
    if num_q == 0:
        return {"avg_at_k": {k: float("nan") for k in K_LIST},
                "pass_at_k": {k: float("nan") for k in K_LIST},
                "num_questions": 0}

    avg_sums = {k: 0.0 for k in K_LIST}
    pass_sums = {k: 0.0 for k in K_LIST}

    for _, items in ans_by_q.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        y = np.array([1.0 if c else 0.0 for _, c in items_sorted], dtype=float)
        n = len(y)
        for k in K_LIST:
            kk = min(k, n)
            if kk == 0:
                continue
            yk = y[:kk]
            avg_sums[k] += float(yk.mean())
            pass_sums[k] += float(1.0 if (yk.max() > 0.0) else 0.0)

    avg_at_k = {k: avg_sums[k] / num_q for k in K_LIST}
    pass_at_k = {k: pass_sums[k] / num_q for k in K_LIST}
    return {"avg_at_k": avg_at_k, "pass_at_k": pass_at_k, "num_questions": num_q}


def format_metrics_table(per_dataset: Dict[str, Dict[str, Dict[int, float]]],
                         overall: Dict[str, Dict[int, float]],
                         counts: Dict[str, int]) -> pd.DataFrame:
    """Build a neat DataFrame for printing."""
    rows = []
    for ds, metrics in sorted(per_dataset.items()):
        row = {"dataset": ds, "#Q": counts.get(ds, 0)}
        for k in K_LIST:
            row[f"avg@{k}"] = metrics["avg_at_k"][k]
        for k in K_LIST:
            row[f"pass@{k}"] = metrics["pass_at_k"][k]
        rows.append(row)

    overall_row = {"dataset": "OVERALL", "#Q": sum(counts.values())}
    for k in K_LIST:
        overall_row[f"avg@{k}"] = overall["avg_at_k"][k]
    for k in K_LIST:
        overall_row[f"pass@{k}"] = overall["pass_at_k"][k]
    rows.append(overall_row)

    df = pd.DataFrame(rows)
    for col in df.columns:
        if col.startswith("avg@") or col.startswith("pass@"):
            df[col] = df[col].map(
                lambda x: np.nan if pd.isna(x) else round(float(x), 3))
    return df


def normalize_gt(gt: str) -> str:
    """Normalize ground-truth consistently (handles boxed forms)."""
    try:
        if "\\boxed{" in gt:
            bx = last_boxed_only_string(gt)
            if bx is not None:
                gt = remove_boxed(bx)
        return normalize_final_answer(gt)
    except Exception:
        return str(gt)


def build_predictions_table(
    preds_by_q: Dict[str, Dict[int, List[Tuple[int, str, bool]]]],
    gt_by_q: Dict[str, Dict[int, str]],
    max_k: int = 32,
) -> pd.DataFrame:
    """
    Build a wide table with per-question predictions.
    Columns: dataset, question_id, num_attempts, ground_truth, pred_1..pred_32, correct_1..correct_32
    """
    rows = []
    for ds in sorted(preds_by_q.keys()):
        for q_id in sorted(preds_by_q[ds].keys()):
            # sort by sample_id
            items = sorted(preds_by_q[ds][q_id], key=lambda x: x[0])
            preds = [p for (_, p, _c) in items][:max_k]
            flags = [int(c) for (_sid, _p, c) in items][:max_k]
            row = {
                "dataset": ds,
                "question_id": q_id,
                "num_attempts": len(items),
                "ground_truth": gt_by_q.get(ds, {}).get(q_id, ""),
            }
            for i in range(max_k):
                row[f"pred_{i+1}"] = preds[i] if i < len(preds) else ""
            for i in range(max_k):
                row[f"correct_{i+1}"] = flags[i] if i < len(flags) else ""
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------- MAIN EVAL LOOP -----------------------

def evaluate_file(parquet_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert Path(parquet_path).exists(), f"File not found: {parquet_path}"
    df = pd.read_parquet(parquet_path)

    # dataset -> q_id -> list of (sample_id, correct_bool)
    dataset_ans: Dict[str, Dict[int, List[Tuple[int, bool]]]
                      ] = defaultdict(lambda: defaultdict(list))
    # dataset -> q_id -> list of (sample_id, pred_str, correct_bool)
    dataset_preds: Dict[str, Dict[int, List[Tuple[int, str, bool]]]] = defaultdict(
        lambda: defaultdict(list))
    # dataset -> q_id -> normalized GT
    dataset_gt: Dict[str, Dict[int, str]] = defaultdict(dict)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating rows"):
        dataset, q_id, s_id, solution, gt_raw = extract_row_fields(row)
        score = compute_score(solution, gt_raw)
        correct = bool(score["acc"])
        pred_str = score["pred"]

        dataset_ans[dataset][q_id].append((s_id, correct))
        dataset_preds[dataset][q_id].append((s_id, pred_str, correct))
        # set GT once (normalized)
        if q_id not in dataset_gt[dataset]:
            dataset_gt[dataset][q_id] = normalize_gt(gt_raw)

    # per-dataset metrics
    per_dataset_metrics: Dict[str, Dict[str, Dict[int, float]]] = {}
    counts: Dict[str, int] = {}
    all_questions_flat: Dict[int, List[Tuple[int, bool]]] = {}
    global_q_counter = 0

    for ds, ans_by_q in dataset_ans.items():
        per_metrics = compute_dataset_metrics(ans_by_q)
        per_dataset_metrics[ds] = {
            "avg_at_k": per_metrics["avg_at_k"],
            "pass_at_k": per_metrics["pass_at_k"],
        }
        counts[ds] = per_metrics["num_questions"]

        for _q_id, items in ans_by_q.items():
            all_questions_flat[global_q_counter] = items
            global_q_counter += 1

    overall_metrics = compute_dataset_metrics(all_questions_flat)

    metrics_table = format_metrics_table(
        per_dataset_metrics, overall_metrics, counts)
    preds_table = build_predictions_table(dataset_preds, dataset_gt, max_k=32)
    return metrics_table, preds_table


def _short_name(path: str) -> str:
    """Make a concise run name from the path."""
    p = Path(path)
    # parent.name is the run folder (e.g. Baseline-DAPO-...__global_step_200)
    return p.parent.name or p.stem


def discover_generation_files() -> List[Path]:
    """
    Decide which generations.parquet files to evaluate.

    Priority:
      1) If GENERATIONS is non-empty: use exactly those paths.
      2) Else: scan ROOT_DIRS (or CWD if ROOT_DIRS is empty) recursively
         and pick every 'generations.parquet'.
    """
    # Manual mode: user explicitly listed files
    if GENERATIONS:
        files = [Path(p).expanduser() for p in GENERATIONS]
        return files

    # Auto-discovery mode
    roots: List[Path] = []
    if ROOT_DIRS:
        roots = [Path(r).expanduser() for r in ROOT_DIRS]
    else:
        roots = [Path.cwd()]

    found: List[Path] = []
    seen_dirs = set()

    for root in roots:
        if not root.exists():
            print(f"[WARN] Root dir not found: {root}")
            continue
        print(f"[INFO] Scanning root: {root}")
        for gen_path in root.rglob("generations.parquet"):
            run_dir = gen_path.parent
            key = str(run_dir.resolve())
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
            found.append(gen_path)

    # Sort for nicer printing: by parent directory name
    found.sort(key=lambda p: p.parent.name)
    return found


def main():
    gen_files = discover_generation_files()
    if not gen_files:
        print("[WARN] No generations.parquet files found. "
              "Check ROOT_DIRS or GENERATIONS config.")
        return

    print("=" * 88)
    print("Discovered generations.parquet files to evaluate:")
    for fp in gen_files:
        print(" -", fp)
    print("=" * 88)

    # For aggregated CSV: one row per run, columns = datasets' avg at max K
    aggregated_rows: List[Dict[str, Any]] = []
    max_k = max(K_LIST)  # use the largest K
    avg_suffix = f"_avg@{max_k}"

    for fpath in gen_files:
        fpath_str = str(fpath)
        run_name = _short_name(fpath_str)

        print("=" * 88)
        print(f"Evaluating: {fpath_str}")
        try:
            metrics_df, preds_df = evaluate_file(fpath_str)
        except Exception as e:
            print(f"[ERROR] Failed on {fpath_str}: {e}")
            continue

        # Print metrics nicely
        with pd.option_context('display.max_columns', None, 'display.width', 200):
            print(metrics_df.to_string(index=False))

        # Save metrics and per-question predictions for this run
        out_metrics = Path(fpath_str).with_name(
            f"{_short_name(fpath_str)}__avgk_results.csv")
        out_preds = Path(fpath_str).with_name(
            f"{_short_name(fpath_str)}__per_question_preds.csv")
        try:
            metrics_df.to_csv(out_metrics, index=False)
            print(f"Saved metrics: {out_metrics}")
        except Exception as e:
            print(f"[WARN] Could not save metrics CSV: {e}")
        try:
            preds_df.to_csv(out_preds, index=False)
            print(f"Saved per-question predictions: {out_preds}")
        except Exception as e:
            print(f"[WARN] Could not save per-question CSV: {e}")

        # --------- Build aggregated row for this run (one row per folder) ---------
        agg_row: Dict[str, Any] = {"run": run_name}

        # metrics_df has rows per dataset + one OVERALL row
        for _, mrow in metrics_df.iterrows():
            ds_name = str(mrow["dataset"])
            avg_col = f"avg@{max_k}"

            if avg_col in mrow:
                agg_row[f"{ds_name}{avg_suffix}"] = mrow[avg_col]

            # Also store #Q per dataset (optional)
            if "#Q" in mrow:
                agg_row[f"{ds_name}_#Q"] = mrow["#Q"]

        aggregated_rows.append(agg_row)
        # -------------------------------------------------------------------------

    # After all runs, write the aggregated CSV
    if aggregated_rows:
        agg_df = pd.DataFrame(aggregated_rows)

        # Compute mean over all benchmark datasets' avg@K (exclude OVERALL)
        bench_cols = [
            c for c in agg_df.columns
            if c.endswith(avg_suffix) and not c.startswith("OVERALL")
        ]
        if bench_cols:
            mean_col = f"avg_all_benchmarks{avg_suffix}"
            agg_df[mean_col] = agg_df[bench_cols].astype(float).mean(axis=1)

            # Sort by this mean, descending
            agg_df = agg_df.sort_values(by=mean_col, ascending=False)

            # Column order: run, mean, then others sorted
            other_cols = [c for c in agg_df.columns if c not in ("run", mean_col)]
            cols = ["run", mean_col] + sorted(other_cols)
        else:
            # Fallback if something weird happens
            cols = ["run"] + sorted([c for c in agg_df.columns if c != "run"])

        agg_df = agg_df[cols]

        if ROOT_DIRS:
            agg_out_path = Path(ROOT_DIRS[0]).expanduser() / "aggregated_eval_results.csv"
        else:
            agg_out_path = Path.cwd() / "aggregated_eval_results.csv"

        try:
            agg_df.to_csv(agg_out_path, index=False)
            print("=" * 88)
            print(f"Saved aggregated results: {agg_out_path}")
        except Exception as e:
            print(f"[WARN] Could not save aggregated CSV: {e}")

    print("=" * 88)
    print("Done.")


if __name__ == "__main__":
    main()
