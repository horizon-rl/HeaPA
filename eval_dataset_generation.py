#!/usr/bin/env python3
# Edit the CONFIG block only. No CLI args needed.

import datetime as dt
import subprocess
from pathlib import Path
import re
import json
import os
import glob
import shlex
import shutil

# ===================== CONFIG (edit me) =====================
DATA_PATH = "./data/merged_math_dapo_eval_repeat16.parquet"
BASE_OUT = "./eval_results/"

# Put your model checkpoints here. Each entry can be:
# - A run root dir containing multiple global_step_* subfolders, e.g.:
#   "/.../ChildrenAgg_EasyMedium_17kdapo_wAugTeacherRewardRefresh_16r20k"
#   (the script will auto-pick the largest global_step_* number)
# - A specific checkpoint dir, e.g.:
#   "/.../ChildrenAgg_EasyMedium_17kdapo_wAugTeacherRewardRefresh_16r20k/global_step_110"
# - A local HF model dir
# - A HuggingFace hub id like "Qwen/Qwen2.5-7B-7B-Instruct"
# - A glob pattern for local paths, e.g.:
#   "./ckpts/HEAP-DAPO-Qwen25-7B-2025Nov14/*GRPO*"
MODELS = [
    "./ckpts/HEAP-DAPO-Qwen25-7B-2025Nov14/Heap_Ablation_PathAgg_MediumOnly_17kdapo_wAugTeacherRewardRefresh_16r20k/",
    # Example of pattern usage:
    # "./ckpts/HEAP-DAPO-Qwen25-7B-2025Nov14/*GRPO*",
]

# Hydra overrides / hyperparams
NNODES = 1
NGPUS_PER_NODE = 8
TRUST_REMOTE_CODE = True       # real boolean
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.7
PROMPT_LEN = 2048
RESPONSE_LEN = 20 * 1024
TP_SIZE = 4
DP_SIZE = (NGPUS_PER_NODE // TP_SIZE)
GPU_MEM_UTIL = 0.85

# Extra hydra args (optional), e.g. 'rollout.seed=42 data.n_samples=1'
EXTRA = ""
# =================== END CONFIG =============================



def utc_ts() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def make_tag(model_id: str) -> str:
    raw = model_id.rstrip("/")
    parts = [seg for seg in raw.split("/") if seg]
    if parts and parts[-1] == "actor" and len(parts) >= 3:
        base = f"{parts[-3]}__{parts[-2]}"   # <model_dir>__global_step_X
    elif len(parts) >= 2:
        base = f"{parts[-2]}__{parts[-1]}"
    else:
        base = parts[0] if parts else "model"
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)


def _is_hf_id(s: str) -> bool:
    raw = s.strip()
    return (not raw.startswith("/")) and ("://" not in raw) and (raw.count("/") == 1)


def _has_hf_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists():
        return True
    # sharded safetensors
    if glob.glob(str(path / "model-*.safetensors")):
        return True
    return False


def _expand_model_specs(specs: list[str]) -> list[str]:
    """
    Expand MODELS entries:

    - HF hub ids are kept as-is.
    - Local entries containing glob wildcards (*?[]) are expanded with glob.glob().
    - Plain local paths are kept as-is.
    - Results are deduplicated in-order.
    """
    expanded: list[str] = []
    for spec in specs:
        # HF hub id -> no expansion
        if _is_hf_id(spec):
            expanded.append(spec)
            continue

        # Glob pattern for local paths
        if any(ch in spec for ch in "*?[]"):
            matches = sorted(glob.glob(spec))
            if not matches:
                print(f"[WARN] Pattern had no matches: {spec}")
            else:
                print(f"[GLOB] {spec} -> {len(matches)} matches:")
                for m in matches:
                    print(f"       {m}")
                expanded.extend(matches)
        else:
            expanded.append(spec)

    # Deduplicate while preserving order
    seen = set()
    unique: list[str] = []
    for s in expanded:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _resolve_model_spec(model_spec: str) -> str:
    """
    Resolve a model spec into a concrete path or HF id:

    - If it's a HF hub id -> return as-is.
    - If it's a dir named global_step_<N> -> use that dir.
    - If it's a run root dir containing global_step_* subdirs ->
      pick the one with the largest N.
    - If it's a local HF export dir (has HF weights) -> use that dir.
    """
    if _is_hf_id(model_spec):
        return model_spec

    p = Path(model_spec)

    if not p.exists():
        raise FileNotFoundError(f"Model spec does not exist: {p}")

    if not p.is_dir():
        raise NotADirectoryError(f"Model spec is not a directory: {p}")

    # If it's already an HF-exported model dir, just use it
    if _has_hf_weights(p):
        print(f"[RESOLVE] {p} appears to be an HF model dir, using as-is.")
        return str(p)

    # If the dir itself is global_step_<N>, treat it as the checkpoint root
    m = re.match(r"global_step_(\d+)$", p.name)
    if m:
        print(f"[RESOLVE] {p} is a checkpoint dir, using as-is.")
        return str(p)

    # Otherwise, treat it as a run root: pick the largest global_step_* subdir
    candidates = []
    for child in p.iterdir():
        if not child.is_dir():
            continue
        m = re.match(r"global_step_(\d+)$", child.name)
        if m:
            step = int(m.group(1))
            candidates.append((step, child))

    if not candidates:
        raise RuntimeError(
            f"No global_step_* checkpoints found under {p} "
            "(and it is not an HF model dir)."
        )

    best_step, best_dir = max(candidates, key=lambda x: x[0])
    print(f"[RESOLVE] {p} -> latest checkpoint {best_dir} (step {best_step})")
    return str(best_dir)


def _merge_fsdp_if_needed(model_path: str, merged_root: Path, tag: str) -> str:
    """
    If model_path looks like a VERL FSDP ckpt (…/global_step_X[/actor]),
    merge shards into an HF dir under merged_root/<tag> and return that path.
    If it's already HF (id or local), return as-is.
    """
    if _is_hf_id(model_path):
        return model_path  # Hub id

    src = Path(model_path)
    if not src.exists():
        raise FileNotFoundError(f"Model path not found: {src}")

    # Determine actor dir and ckpt root
    actor_dir = src / "actor" if (src / "actor").is_dir() else (
        src if src.name == "actor" else None
    )
    if actor_dir is None or not actor_dir.is_dir():
        # Might already be an HF export on disk
        if _has_hf_weights(src):
            return str(src)
        # Otherwise nothing to merge, let it error early
        return str(src)

    # If target already merged, reuse it
    merged_dir = merged_root / tag
    if _has_hf_weights(merged_dir) and (merged_dir / "config.json").exists():
        print(f"[MERGE] Reusing existing merged HF model: {merged_dir}")
        return str(merged_dir)

    merged_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3",
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(merged_dir),
    ]
    print(f"[MERGE] {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)

    if not _has_hf_weights(merged_dir):
        raise RuntimeError(
            f"Merge completed but no HF weights found in {merged_dir}"
        )
    return str(merged_dir)


def build_cmd(model_path: str, out_dir: Path) -> list[str]:
    out_file = out_dir / "generations.parquet"
    cmd = [
        "python3",
        "-m",
        "verl.trainer.main_generation",
        f"trainer.nnodes={NNODES}",
        f"trainer.n_gpus_per_node={NGPUS_PER_NODE}",
        f"data.path={DATA_PATH}",
        "data.prompt_key=prompt",
        "data.n_samples=1",
        f"data.output_path={str(out_file)}",
        f"model.path={model_path}",
        f"+model.trust_remote_code={str(TRUST_REMOTE_CODE)}",
        f"rollout.temperature={TEMPERATURE}",
        f"rollout.top_k={TOP_K}",
        f"rollout.top_p={TOP_P}",
        f"+rollout.data_parallel_size={DP_SIZE}",
        f"rollout.prompt_length={PROMPT_LEN}",
        f"rollout.response_length={RESPONSE_LEN}",
        f"rollout.tensor_model_parallel_size={TP_SIZE}",
        f"rollout.gpu_memory_utilization={GPU_MEM_UTIL}",
        "rollout.enable_chunked_prefill=False",
        "rollout.max_num_batched_tokens=200000",
    ]
    if EXTRA.strip():
        cmd.extend(shlex.split(EXTRA))
    return cmd


def run_one(cmd: list[str], out_dir: Path, model_path: str) -> int:
    if out_dir.exists():
        raise FileExistsError(f"Output folder already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    (out_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    meta = {
        "tag": out_dir.name,
        "model_path": model_path,
        "data_path": DATA_PATH,
        "timestamp_utc": utc_ts(),
        "params": {
            "nnodes": NNODES,
            "n_gpus_per_node": NGPUS_PER_NODE,
            "trust_remote_code": TRUST_REMOTE_CODE,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "prompt_length": PROMPT_LEN,
            "response_length": RESPONSE_LEN,
            "tp_size": TP_SIZE,
            "gpu_memory_utilization": GPU_MEM_UTIL,
            "extra": EXTRA,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    log_path = out_dir / "run.log"
    print("=" * 70)
    print("[START]", dt.datetime.now().isoformat())
    print("CMD  :", (out_dir / "command.txt").read_text().strip())
    print("OUT  :", out_dir)
    print("=" * 70)

    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
        rc = proc.wait()

    if rc == 0:
        (out_dir / "DONE.ok").touch()
        print("[DONE ]", dt.datetime.now().isoformat(), "rc=0")
    else:
        print("[ERROR]", dt.datetime.now().isoformat(), f"rc={rc}")
    return rc


def main():
    base_out = Path(BASE_OUT)
    base_out.mkdir(parents=True, exist_ok=True)
    merged_cache = base_out / ".merged_hf"   # keep merged HF copies here
    merged_cache.mkdir(parents=True, exist_ok=True)

    if not MODELS:
        raise SystemExit(
            "Edit MODELS in the CONFIG block with your checkpoint paths."
        )

    # Expand glob patterns (e.g., *GRPO*) before resolving
    expanded_models = _expand_model_specs(MODELS)
    if not expanded_models:
        raise SystemExit(
            "After expanding patterns, no valid MODELS remain. "
            "Check your paths/patterns."
        )

    for spec in expanded_models:
        # 0) resolve run-root vs checkpoint vs HF id
        resolved_ckpt_or_hf = _resolve_model_spec(spec)

        # 1) build a tag based on the resolved path/id
        tag = make_tag(resolved_ckpt_or_hf)

        # 2) decide output directory and handle existing folders
        out_dir = base_out / tag
        gen_file = out_dir / "generations.parquet"

        if out_dir.exists():
            if gen_file.exists():
                # Folder already has generations.parquet → keep it and skip this model
                print(f"[SKIP] {out_dir} already has generations.parquet, skipping.")
                continue
            else:
                # Folder exists but no generations.parquet → nuke and re-run
                print(
                    f"[CLEAN] {out_dir} exists but has no generations.parquet; "
                    "removing and regenerating."
                )
                shutil.rmtree(out_dir)

        # 3) ensure we have an HF model (merge if needed)
        hf_model_path = _merge_fsdp_if_needed(resolved_ckpt_or_hf, merged_cache, tag)

        # 4) build command and run
        cmd = build_cmd(hf_model_path, out_dir)
        rc = run_one(cmd, out_dir, resolved_ckpt_or_hf)
        if rc != 0:
            raise SystemExit(rc)

    print("All sequential inference runs finished.")


if __name__ == "__main__":
    main()