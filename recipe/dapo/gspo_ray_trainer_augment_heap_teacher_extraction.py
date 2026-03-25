# Standard library
from __future__ import annotations

import collections
import itertools
import logging
import os
import re
import threading
import time
import uuid
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Third-party
import numpy as np
import ray
import torch
from tqdm import tqdm

# OpenAI SDK (no hardcoded key; uses env var OPENAI_API_KEY)
from openai import OpenAI

# VERL core
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.utils.metric import reduce_metrics
from verl.trainer.ppo.reward import compute_reward, compute_reward_async

# Import your base PPO trainer + utility helpers from verl
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# =============================================================================
# Augmentation logger (flush-safe, thread-safe)
# =============================================================================

class AugmentationLogger:
    """Logs augmentation + annotation data for offline analysis."""

    def __init__(self, log_dir: str, experiment_name: str, max_buffer_size: int = 500):
        self.log_dir = Path(log_dir) / "augmentation_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.augmentation_file = self.log_dir / \
            f"{experiment_name}_{self.timestamp}_augmentations.jsonl"
        self.annotation_file = self.log_dir / \
            f"{experiment_name}_{self.timestamp}_annotations.jsonl"
        self.pool_snapshot_file = self.log_dir / \
            f"{experiment_name}_{self.timestamp}_pool_snapshots.jsonl"

        self.augmentation_buffer: List[dict] = []
        self.annotation_buffer:  List[dict] = []
        self.pool_snapshot_buffer: List[dict] = []
        self.max_buffer_size = max_buffer_size
        self._lock = threading.Lock()

        meta = {
            "experiment_name": experiment_name,
            "timestamp": self.timestamp,
            "files": {
                "augmentations": str(self.augmentation_file),
                "annotations": str(self.annotation_file),
                "pool_snapshots": str(self.pool_snapshot_file),
            },
        }
        with open(self.log_dir / f"{experiment_name}_{self.timestamp}_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _flush_list(self, items: List[dict], path: Path):
        if not items:
            return
        try:
            with open(path, "a") as f:
                for r in items:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to flush {path.name}: {e}")

    def log_augmentation(self, record: Dict[str, Any]):
        record = dict(record)
        record["timestamp"] = datetime.now().isoformat()
        record["id"] = str(uuid.uuid4())
        with self._lock:
            self.augmentation_buffer.append(record)
            if len(self.augmentation_buffer) >= self.max_buffer_size:
                self._flush_list(self.augmentation_buffer,
                                 self.augmentation_file)
                self.augmentation_buffer.clear()

    def log_annotation(self, record: Dict[str, Any]):
        record = dict(record)
        record["timestamp"] = datetime.now().isoformat()
        record["id"] = str(uuid.uuid4())
        with self._lock:
            self.annotation_buffer.append(record)
            if len(self.annotation_buffer) >= self.max_buffer_size:
                self._flush_list(self.annotation_buffer, self.annotation_file)
                self.annotation_buffer.clear()

    def log_pool_snapshot(self, pool_metrics: Dict[str, Any], sample: List[Dict[str, Any]] | None = None):
        snap = {
            "timestamp": datetime.now().isoformat(),
            "metrics": dict(pool_metrics),
            "sample": (sample or [])[:10],
        }
        with self._lock:
            self.pool_snapshot_buffer.append(snap)
            if len(self.pool_snapshot_buffer) >= max(10, self.max_buffer_size // 10):
                self._flush_list(self.pool_snapshot_buffer,
                                 self.pool_snapshot_file)
                self.pool_snapshot_buffer.clear()

    def flush_all(self):
        with self._lock:
            self._flush_list(self.augmentation_buffer, self.augmentation_file)
            self._flush_list(self.annotation_buffer,  self.annotation_file)
            self._flush_list(self.pool_snapshot_buffer,
                             self.pool_snapshot_file)
            self.augmentation_buffer.clear()
            self.annotation_buffer.clear()
            self.pool_snapshot_buffer.clear()


# =============================================================================
# Simple record (compatible with the pool)
# =============================================================================

@dataclass
class QueryRecord:
    raw_prompt_data: Any
    input_ids: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    gt: Optional[object] = None
    reward: float = 0.5
    est_reward: Optional[float] = None
    meta: dict = field(default_factory=dict)

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_text: Optional[str] = None
    augmented_text: Optional[str] = None
    teacher_response: Optional[str] = None
    creation_time: str = field(
        default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "reward": self.reward,
            "est_reward": self.est_reward,
            "gt": self.gt if isinstance(self.gt, (int, float, str)) else str(self.gt),
            "original_text": self.original_text,
            "augmented_text": self.augmented_text,
            "teacher_response": self.teacher_response,
            "creation_time": self.creation_time,
            "meta": dict(self.meta),
        }


# =============================================================================
# Thread-safe heap pool (your original design; lightly trimmed)
# =============================================================================

class ThreadSafeQueryPool:
    """
    Two-partition pool:
      - low_heap stores medium/low items as a max-heap by reward (negated key),
      - high side uses twin heaps (min + max) with lazy deletion.

    sample_batch(k): returns "medium" by default (mix from fronts), or mixed easy/medium
    if mixed_easy_medium is enabled.
    """

    def __init__(
        self,
        max_size: int = 30000,
        low_fraction: float = 0.5,
        rng: Optional[np.random.Generator] = None,
        cleanup_frequency: int = 1000,
        mixed_easy_medium: bool = False,
    ):
        import heapq  # local import to keep namespace tight
        self._heapq = heapq

        self._lock = threading.RLock()
        self._max_size = max(1, int(max_size))
        self._low_fraction = float(np.clip(low_fraction, 0.05, 0.95))
        self._cleanup_frequency = cleanup_frequency
        self._operations_count = 0
        self._mixed_easy_medium = bool(mixed_easy_medium)

        self._low_heap: List[Tuple[float, int, QueryRecord]] = []
        self._high_heap_min: List[Tuple[float, int, QueryRecord]] = []
        self._high_heap_max: List[Tuple[float, int, QueryRecord]] = []
        self._active_high: set[int] = set()

        self._seq = itertools.count()
        self._rng = rng if rng is not None else np.random.default_rng()

        self._total_added = 0
        self._total_sampled = 0
        self._total_evicted = 0

    # --- admin/metrics ---
    def size(self) -> int:
        with self._lock:
            return len(self._low_heap) + len(self._active_high)

    def capacity_remaining(self) -> int:
        with self._lock:
            return max(0, self._max_size - (len(self._low_heap) + len(self._active_high)))

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_size": self.size(),
                "low_heap_size": len(self._low_heap),
                "high_heap_size": len(self._active_high),
                "capacity_remaining": self.capacity_remaining(),
                "total_added": self._total_added,
                "total_sampled": self._total_sampled,
                "total_evicted": self._total_evicted,
            }

    def set_mixed_easy_medium(self, enabled: bool) -> None:
        with self._lock:
            self._mixed_easy_medium = bool(enabled)

    # --- core ops ---
    def add_many(self, items: List[QueryRecord]):
        if not items:
            return
        for it in items:
            it.reward = float(it.reward)
        with self._lock:
            for it in items:
                total = len(self._low_heap) + len(self._active_high)
                if total < self._max_size:
                    self._push_high_unlocked(it)
                    self._rebalance_low_high_unlocked(local_only=True)
                else:
                    self._insert_when_full_unlocked(it)
            self._total_added += len(items)
            self._operations_count += len(items)
            if self._operations_count >= self._cleanup_frequency:
                self._deep_clean_heaps_unlocked()
                self._operations_count = 0
            self._evict_to_capacity_unlocked()

    def sample_batch(self, k: int) -> List[QueryRecord]:
        if k <= 0:
            return []
        with self._lock:
            n_total = len(self._low_heap) + len(self._active_high)
            if n_total == 0:
                return []
            actual_k = min(k, n_total)

            if self._mixed_easy_medium:
                # half easy (global highest), half medium
                want_easy = actual_k // 2
                easy = []
                for _ in range(want_easy):
                    popped = self._pop_high_max_unlocked()
                    if popped is None:
                        break
                    _, _, it = popped
                    easy.append(it)
                need_medium = actual_k - len(easy)
                medium = self._sample_medium_k_unlocked(need_medium)
                chosen = easy + medium
                self._rebalance_low_high_unlocked(local_only=True)
            else:
                chosen = self._sample_medium_k_unlocked(actual_k)

            self._total_sampled += len(chosen)
            return [self._copy_record_unlocked(r) for r in chosen]

    # --- internals ---
    def _push_low_unlocked(self, it: QueryRecord):
        self._heapq.heappush(
            self._low_heap, (-float(it.reward), next(self._seq), it))

    def _push_high_unlocked(self, it: QueryRecord):
        seq = next(self._seq)
        reward = float(it.reward)
        self._heapq.heappush(self._high_heap_min, (reward, seq, it))
        self._heapq.heappush(self._high_heap_max, (-reward, seq, it))
        self._active_high.add(seq)

    def _pop_high_min_unlocked(self):
        self._clean_high_min_top_unlocked()
        if not self._high_heap_min:
            return None
        reward, seq, it = self._heapq.heappop(self._high_heap_min)
        if seq in self._active_high:
            self._active_high.remove(seq)
            return (reward, seq, it)
        return self._pop_high_min_unlocked()

    def _pop_high_max_unlocked(self):
        self._clean_high_max_top_unlocked()
        while self._high_heap_max:
            neg_r, seq, it = self._heapq.heappop(self._high_heap_max)
            if seq in self._active_high:
                self._active_high.remove(seq)
                return (-float(neg_r), seq, it)
        return None

    def _clean_high_min_top_unlocked(self):
        while self._high_heap_min and (self._high_heap_min[0][1] not in self._active_high):
            self._heapq.heappop(self._high_heap_min)

    def _clean_high_max_top_unlocked(self):
        while self._high_heap_max and (self._high_heap_max[0][1] not in self._active_high):
            self._heapq.heappop(self._high_heap_max)

    def _deep_clean_heaps_unlocked(self):
        # rebuild heaps to purge stale entries
        mh, MH = [], []
        for r, s, it in self._high_heap_min:
            if s in self._active_high:
                mh.append((r, s, it))
        for r, s, it in self._high_heap_max:
            if s in self._active_high:
                MH.append((r, s, it))
        self._high_heap_min = mh
        self._high_heap_max = MH
        self._heapq.heapify(self._high_heap_min)
        self._heapq.heapify(self._high_heap_max)

    def _pop_candidates_unlocked(self, take: int):
        cands: List[QueryRecord] = []
        origins: List[str] = []
        want_high = True
        while take > 0 and (self._low_heap or self._active_high):
            pulled = False
            if want_high and self._active_high:
                popped = self._pop_high_min_unlocked()
                if popped is not None:
                    _, _, it = popped
                    cands.append(it)
                    origins.append("high")
                    take -= 1
                    pulled = True
            elif (not want_high) and self._low_heap:
                _, _, it = self._heapq.heappop(self._low_heap)
                cands.append(it)
                origins.append("low")
                take -= 1
                pulled = True
            if not pulled:
                if self._active_high:
                    popped = self._pop_high_min_unlocked()
                    if popped is not None:
                        _, _, it = popped
                        cands.append(it)
                        origins.append("high")
                        take -= 1
                elif self._low_heap:
                    _, _, it = self._heapq.heappop(self._low_heap)
                    cands.append(it)
                    origins.append("low")
                    take -= 1
                else:
                    break
            want_high = not want_high
        return cands, origins

    def _sample_medium_k_unlocked(self, k: int) -> List[QueryRecord]:
        if k <= 0:
            return []
        n_total = len(self._low_heap) + len(self._active_high)
        if n_total == 0:
            return []
        actual_k = min(k, n_total)
        if actual_k >= n_total:
            candidates, origins = self._pop_candidates_unlocked(n_total)
            chosen_idx = list(range(len(candidates)))
        else:
            take = min(2 * actual_k, n_total)
            candidates, origins = self._pop_candidates_unlocked(take)
            chosen_idx = self._rng.choice(len(candidates), size=min(
                actual_k, len(candidates)), replace=False).tolist() if candidates else []
        chosen = [candidates[i] for i in chosen_idx]
        keep_mask = np.ones(len(candidates), dtype=bool)
        for idx in chosen_idx:
            if idx < len(keep_mask):
                keep_mask[idx] = False
        to_reinsert = [c for i, c in enumerate(
            candidates) if i < len(keep_mask) and keep_mask[i]]
        origins_back = [origins[i] for i in range(
            len(origins)) if i < len(keep_mask) and keep_mask[i]]
        for rec, origin in zip(to_reinsert, origins_back):
            if origin == "low":
                self._push_low_unlocked(rec)
            else:
                self._push_high_unlocked(rec)
        self._rebalance_low_high_unlocked(local_only=True)
        return chosen

    def _global_max_reward_unlocked(self) -> Optional[float]:
        cand = []
        if self._low_heap:
            cand.append(-self._low_heap[0][0])
        self._clean_high_max_top_unlocked()
        if self._high_heap_max:
            cand.append(-self._high_heap_max[0][0])
        return max(cand) if cand else None

    def _remove_high_heap_max_unlocked(self) -> bool:
        self._clean_high_max_top_unlocked()
        while self._high_heap_max:
            _, seq, _ = self._heapq.heappop(self._high_heap_max)
            if seq in self._active_high:
                self._active_high.remove(seq)
                self._total_evicted += 1
                return True
        return False

    def _insert_when_full_unlocked(self, it: QueryRecord) -> bool:
        gmax = self._global_max_reward_unlocked()
        if gmax is not None and float(it.reward) > gmax:
            return False  # reject too-easy
        r_low = (-self._low_heap[0][0]) if self._low_heap else None
        self._clean_high_min_top_unlocked()
        r_high = self._high_heap_min[0][0] if self._high_heap_min else None

        if r_low is None and r_high is None:
            self._push_high_unlocked(it)
        elif r_high is None:
            self._push_high_unlocked(it)
        elif r_low is None:
            self._push_low_unlocked(it)
        else:
            if float(it.reward) <= r_high:
                self._push_low_unlocked(it)
            else:
                self._push_high_unlocked(it)

        if not self._remove_high_heap_max_unlocked():
            if self._low_heap:
                self._heapq.heappop(self._low_heap)
                self._total_evicted += 1
        self._rebalance_low_high_unlocked(local_only=True)
        return True

    def _rebalance_low_high_unlocked(self, local_only: bool = False):
        total = len(self._low_heap) + len(self._active_high)
        if total == 0:
            return
        target_low = int(round(self._low_fraction * total))
        current_low = len(self._low_heap)
        if abs(current_low - target_low) <= 1:
            return
        if current_low < target_low:
            to_move = min(target_low - current_low, len(self._active_high))
            if local_only:
                to_move = min(to_move, 4)
            for _ in range(to_move):
                popped = self._pop_high_min_unlocked()
                if popped is None:
                    break
                _, _, it = popped
                self._push_low_unlocked(it)
        else:
            to_move = min(current_low - target_low, len(self._low_heap))
            if local_only:
                to_move = min(to_move, 4)
            for _ in range(to_move):
                if not self._low_heap:
                    break
                _, _, it = self._heapq.heappop(self._low_heap)
                self._push_high_unlocked(it)

    def _evict_to_capacity_unlocked(self):
        total = len(self._low_heap) + len(self._active_high)
        if total <= self._max_size:
            return
        need = total - self._max_size
        evicted = 0
        while need > 0 and self._active_high:
            if self._remove_high_heap_max_unlocked():
                evicted += 1
            need -= 1
        while need > 0 and self._low_heap:
            self._heapq.heappop(self._low_heap)
            evicted += 1
            self._total_evicted += 1
            need -= 1

    @staticmethod
    def _copy_record_unlocked(it: QueryRecord) -> QueryRecord:
        return QueryRecord(
            raw_prompt_data=it.raw_prompt_data,
            input_ids=it.input_ids.clone() if it.input_ids is not None else None,
            attention_mask=it.attention_mask.clone() if it.attention_mask is not None else None,
            position_ids=it.position_ids.clone() if it.position_ids is not None else None,
            gt=it.gt,
            reward=it.reward,
            est_reward=it.est_reward,
            meta=dict(it.meta or {}),
            record_id=it.record_id,
            original_text=it.original_text,
            augmented_text=it.augmented_text,
            teacher_response=it.teacher_response,
            creation_time=it.creation_time,
        )

    def get_sample_for_logging(self, n: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            samples = []
            # sample a few from low heap
            for i in range(min(n//2, len(self._low_heap))):
                _, _, record = self._low_heap[i]
                samples.append(record.to_dict())
            # and a few from active high (min top)
            hs = []
            for reward, seq, record in self._high_heap_min:
                if seq in self._active_high:
                    hs.append(record.to_dict())
                    if len(hs) >= n//2:
                        break
            samples.extend(hs)
            return samples


# =============================================================================
# Async teacher annotator
# =============================================================================

def _sanitize_non_tensor_batch(dp: DataProto) -> DataProto:
    """Ensure non_tensor_batch arrays are 1D and length-aligned."""
    try:
        n = len(dp.batch["input_ids"])
    except Exception:
        return dp

    def _as_array(v, n):
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                return np.repeat(v.reshape(1), n)
            if len(v) == n and v.ndim == 1:
                return v
            if len(v) == 1:
                return np.repeat(v, n, axis=0)
            out = np.empty(n, dtype=object)
            for i in range(n):
                out[i] = v[i] if i < len(v) else v[-1]
            return out
        if isinstance(v, (list, tuple)):
            if len(v) == n:
                return np.asarray(v, dtype=object)
            if len(v) == 1:
                return np.asarray(list(v) * n, dtype=object)
            out = np.empty(n, dtype=object)
            for i in range(n):
                out[i] = v[i] if i < len(v) else v[-1]
            return out
        if isinstance(v, dict):
            return np.asarray([v] * n, dtype=object)
        return np.asarray([v] * n, dtype=object)

    for k, v in list(dp.non_tensor_batch.items()):
        dp.non_tensor_batch[k] = _as_array(v, n)
    return dp


class AsyncTeacherAnnotator(threading.Thread):
    """
    Takes DataProto, cleans/solves with JSON-mode, returns only solvable items.
    """

    def __init__(
        self,
        trainer_ref: "RayPPOAugTrainer",
        poll_interval: float = 0.1,
        max_queue: int = 20000,
        api_timeout: float = 30.0,
        model_name: str = "gpt-5-mini",
        max_retries: int = 1,
        retry_delay: float = 1.0,
        augmentation_logger: Optional[AugmentationLogger] = None,
    ):
        super().__init__(daemon=True)
        self.trainer_ref = trainer_ref
        self.queue: "deque[DataProto]" = deque(maxlen=max_queue)
        self._queue_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.poll_interval = poll_interval
        self.api_timeout = api_timeout
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._running = True
        self._shutdown_lock = threading.Lock()
        self.augmentation_logger = augmentation_logger

        self.client = OpenAI()
        self.executor = ThreadPoolExecutor(max_workers=32)

        self._processed_count = 0
        self._error_count = 0
        self._api_call_count = 0

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(re.search(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]', text or ""))

    def enqueue_aug(self, aug_proto: DataProto) -> bool:
        with self._queue_lock:
            if len(self.queue) >= self.queue.maxlen:
                return False
            self.queue.append(aug_proto)
            return True

    def shutdown(self):
        with self._shutdown_lock:
            if not self._running:
                return
            self._running = False
            self.stop_event.set()
            try:
                self.executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass

    def get_metrics(self) -> Dict[str, Any]:
        with self._queue_lock:
            qs = len(self.queue)
        return {
            "queue_size": qs,
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "api_call_count": self._api_call_count,
        }

    def _make_extract_and_solve_prompt(self, original: str, generation: str) -> str:
        return (
            "You are a math data cleaner and solver.\n"
            "TASKS:\n"
            "1) From GENERATION ONLY, extract a single, self-contained math problem statement. "
            "   Remove prefaces, code fences, 'Answer:' lines, and any commentary.\n"
            "   Do NOT copy from ORIGINAL; if nothing clean remains, return empty and mark unsolvable.\n"
            "2) If a clean question exists and is well-posed, compute ONLY the final numeric answer.\n"
            "3) Estimate relative difficulty vs ORIGINAL on a 0.75–1.33 scale (1.0=same).\n"
            "4) violations := subset of [preamble, answer_leak, code_fence, non_math, missing_question, domain_shift].\n\n"
            "Return exactly ONE JSON object:\n"
            '{"clean":"<string>","solvable":true|false,"answer":"<string or null>",'
            '"difficulty":<number>,"violations":["<strings>"]}\n\n'
            f"ORIGINAL:\n{original}\n\nGENERATION:\n{generation}"
        )

    def _call_extract_solve_with_retry(self, original: str, generation: str) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                self._api_call_count += 1
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system",
                            "content": "You are a careful math data cleaner and solver."},
                        {"role": "user", "content": self._make_extract_and_solve_prompt(
                            original, generation)},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=4096,
                    timeout=self.api_timeout,
                )
                content = ""
                finish_reason = None
                if getattr(resp, "choices", None):
                    ch = resp.choices[0]
                    finish_reason = getattr(ch, "finish_reason", None)
                    msg = getattr(ch, "message", None)
                    if msg is not None:
                        content = (getattr(msg, "content", "") or "").strip()

                text = content
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?", "",
                                  text, flags=re.I).strip()
                    if text.endswith("```"):
                        text = text[:-3].strip()
                obj = json.loads(text)

                clean = str(obj.get("clean") or "").strip()
                solvable = bool(obj.get("solvable", False))
                ans = obj.get("answer", None)
                try:
                    diff = float(obj.get("difficulty", 1.0))
                except Exception:
                    diff = 1.0
                if not np.isfinite(diff):
                    diff = 1.0
                diff = float(np.clip(diff, 0.75, 1.33))
                viol = obj.get("violations", [])
                if not isinstance(viol, list):
                    viol = []

                return {
                    "ok": True,
                    "clean": clean,
                    "solvable": solvable,
                    "answer": (str(ans) if isinstance(ans, str) else None) if solvable else None,
                    "difficulty": diff,
                    "violations": viol,
                    "raw": content,
                    "finish_reason": finish_reason,
                }
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    return {"ok": False, "error": str(e), "raw": ""}

    def run(self):
        while self._running and not self.stop_event.is_set():
            with self._queue_lock:
                aug_proto = self.queue.popleft() if self.queue else None
            if aug_proto is None:
                time.sleep(self.poll_interval)
                continue
            try:
                self._process_augmented_proto(aug_proto)
                self._processed_count += 1
            except Exception as e:
                logger.error(f"Teacher annotator error: {e}", exc_info=True)
                self._error_count += 1

    def _process_augmented_proto(self, aug_proto: DataProto):
        est = aug_proto.non_tensor_batch.get("policy/est_reward", None)
        if est is None:
            return
        _sanitize_non_tensor_batch(aug_proto)
        est_list = list(est)
        idxs = [i for i, v in enumerate(est_list) if isinstance(
            v, (int, float)) and np.isfinite(v)]
        if not idxs:
            return
        sub_proto = aug_proto[idxs]

        original_texts = list(
            sub_proto.non_tensor_batch.get("original_text", []))
        augmented_texts = list(
            sub_proto.non_tensor_batch.get("raw_prompt_data", []))
        N = min(len(original_texts), len(augmented_texts), len(idxs))

        clean_questions = [""] * N
        teacher_solvable = [False] * N
        teacher_answers = [None] * N
        teacher_raw_resp = [""] * N
        teacher_difficulty = [1.0] * N
        teacher_violations = [[] for _ in range(N)]

        futures = []
        for i in range(N):
            q_text = str(augmented_texts[i])
            if self._contains_cjk(q_text):
                teacher_solvable[i] = False
                teacher_raw_resp[i] = "[filtered:cjk]"
                continue
            orig = str(original_texts[i]) if i < len(original_texts) else ""
            futures.append((i, self.executor.submit(
                self._call_extract_solve_with_retry, orig, q_text), q_text))

        for i, fut, q_text in futures:
            try:
                r = fut.result(timeout=self.api_timeout * 2)
                if not r.get("ok", False):
                    teacher_solvable[i] = False
                    teacher_answers[i] = None
                    teacher_raw_resp[i] = r.get("raw", "")
                else:
                    clean_questions[i] = r["clean"]
                    teacher_difficulty[i] = r["difficulty"]
                    teacher_violations[i] = r["violations"]
                    if clean_questions[i] and ("non_math" not in teacher_violations[i]) \
                       and ("missing_question" not in teacher_violations[i]) \
                       and ("domain_shift" not in teacher_violations[i]):
                        teacher_solvable[i] = bool(r["solvable"])
                        teacher_answers[i] = r["answer"] if r["solvable"] else None
                    else:
                        teacher_solvable[i] = False
                        teacher_answers[i] = None
                    teacher_raw_resp[i] = r["raw"]

                if self.augmentation_logger:
                    self.augmentation_logger.log_annotation({
                        "original_text": original_texts[i] if i < len(original_texts) else None,
                        "augmented_text": q_text,
                        "cleaned_question": clean_questions[i],
                        "estimated_reward": float(est_list[idxs[i]]) if i < len(idxs) else None,
                        "teacher_model": self.model_name,
                        "solvable": teacher_solvable[i],
                        "teacher_answer": teacher_answers[i],
                        "teacher_raw_response": teacher_raw_resp[i],
                        "teacher_difficulty": teacher_difficulty[i],
                        "violations": teacher_violations[i],
                        "api_calls": self._api_call_count,
                    })
            except TimeoutError:
                teacher_solvable[i] = False
                teacher_raw_resp[i] = "[timeout]"
            except Exception as e:
                teacher_solvable[i] = False
                teacher_raw_resp[i] = f"[error]{e}"

        sub_proto.non_tensor_batch["teacher/gt"] = teacher_answers
        sub_proto.non_tensor_batch["teacher/solvable"] = np.asarray(
            teacher_solvable, dtype=bool)
        sub_proto.non_tensor_batch["teacher/raw_responses"] = teacher_raw_resp
        sub_proto.non_tensor_batch["teacher/difficulty"] = np.asarray(
            teacher_difficulty, dtype=float)
        sub_proto.non_tensor_batch["teacher/violations"] = np.asarray(
            teacher_violations, dtype=object)

        sub_proto.non_tensor_batch["raw_prompt_data"] = np.asarray(
            clean_questions, dtype=object)
        try:
            enc = self.trainer_ref._tokenize_texts(clean_questions)
            sub_proto.batch["input_ids"] = enc["input_ids"]
            sub_proto.batch["attention_mask"] = enc["attention_mask"]
            sub_proto.batch["position_ids"] = enc["position_ids"]
        except Exception as e:
            logger.warning(
                f"Retokenization failed; using original tensors. {e}")

        try:
            est_arr = np.asarray(sub_proto.non_tensor_batch.get(
                "policy/est_reward", []), dtype=float)
            dif_arr = np.asarray(teacher_difficulty, dtype=float)
            if est_arr.size == dif_arr.size and est_arr.size:
                eps = 1e-6
                safe_d = np.maximum(dif_arr, eps)
                adj = np.where(est_arr >= 0, est_arr /
                               safe_d, est_arr * safe_d)
                sub_proto.non_tensor_batch["policy/est_reward"] = np.clip(
                    adj, -1.0, 1.0)
        except Exception:
            pass

        _sanitize_non_tensor_batch(sub_proto)
        keep = np.asarray([i for i, ok in enumerate(
            teacher_solvable) if ok], dtype=np.int64)
        if keep.size == 0:
            return
        self.trainer_ref.submit_teacher_batch(sub_proto[keep])


# =============================================================================
# PPO trainer with augmentation + teacher annotation + HEAP POOL (no DAPO filter)
# =============================================================================

class RayPPOAugTrainer(RayPPOTrainer):
    """
    PPO dataflow intact.
    - build numeric augmentations from current batch prompts
    - add them to a heap-based difficulty pool
    - sample medium-difficulty items (or mixed easy/medium) each step
    - send sampled items to teacher annotator
    - integrate teacher-validated items via dataset hook or logging
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._annotated_inbox: "deque[DataProto]" = deque(maxlen=20000)
        self._inbox_lock = getattr(
            torch.multiprocessing, "RLock", threading.RLock)()

        self._aug_metrics = dict(
            enqueued=0,                 # sent to teacher
            # came back from teacher (before integration)
            teacher_submitted=0,
            teacher_integrated=0,
            queue_full_events=0,
            pool_added=0,               # added into pool
            pool_sampled=0,             # sampled from pool
        )

        try:
            self.augmentation_logger = AugmentationLogger(
                log_dir=self.config.trainer.default_local_dir,
                experiment_name=self.config.trainer.experiment_name,
            )
        except Exception:
            self.augmentation_logger = None

        # ----- Pool config -----
        aug_cfg = getattr(self.config, "augmentation", {}) or {}
        pool_cfg = aug_cfg.get("pool", {}) or {}
        self._pool_enabled = bool(pool_cfg.get("enable", True))
        if self._pool_enabled:
            self.query_pool = ThreadSafeQueryPool(
                max_size=int(pool_cfg.get("max_size", 30000)),
                low_fraction=float(pool_cfg.get("low_fraction", 0.5)),
                mixed_easy_medium=bool(
                    pool_cfg.get("mixed_easy_medium", False)),
            )
            self._pool_sample_per_step = int(
                pool_cfg.get("sample_per_step", 32))
            self._pool_snapshot_every = int(
                pool_cfg.get("snapshot_every", 200))
        else:
            self.query_pool = None
            self._pool_sample_per_step = 0
            self._pool_snapshot_every = 0

        self._last_pool_snapshot_step = 0
        self.teacher_annotator: Optional[AsyncTeacherAnnotator] = None

    # --------------------------
    # Teacher lifecycle
    # --------------------------

    def init_workers(self):
        super().init_workers()

    def _start_teacher_pipeline(self):
        if self.teacher_annotator is not None:
            return
        enable = bool(getattr(self.config, "augmentation", {}).get("enable", True)) or \
            bool(getattr(self.config, "dynamic_data", {}).get("enable", False))
        if not enable:
            return
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning(
                "OPENAI_API_KEY not set; teacher annotator disabled.")
            return
        teacher_model = getattr(
            getattr(self.config, "dynamic_data", {}), "teacher_model", "gpt-5-mini")
        self.teacher_annotator = AsyncTeacherAnnotator(
            trainer_ref=self,
            model_name=teacher_model,
            augmentation_logger=self.augmentation_logger,
        )
        self.teacher_annotator.start()

    def _cleanup_teacher_pipeline(self):
        if self.teacher_annotator is not None:
            try:
                self.teacher_annotator.shutdown()
                self.teacher_annotator.join(timeout=10.0)
            except Exception:
                pass
            self.teacher_annotator = None
        if self.augmentation_logger is not None:
            try:
                self.augmentation_logger.flush_all()
            except Exception:
                pass

    # called by annotator thread
    def submit_teacher_batch(self, dp: DataProto):
        with self._inbox_lock:
            self._annotated_inbox.append(dp)
        try:
            n = len(dp.batch["input_ids"])
        except Exception:
            n = 0
        self._aug_metrics["teacher_submitted"] += n

    def _drain_teacher_inbox(self):
        packets: List[DataProto] = []
        with self._inbox_lock:
            while self._annotated_inbox:
                packets.append(self._annotated_inbox.popleft())

        if not packets:
            return

        integrated = 0
        for dp in packets:
            consumed = False
            if hasattr(self.train_dataset, "on_augmented_batch"):
                try:
                    self.train_dataset.on_augmented_batch(dp=dp)
                    consumed = True
                except Exception as e:
                    logger.warning(f"on_augmented_batch hook failed: {e}")

            if self.augmentation_logger and not consumed:
                try:
                    texts = self.tokenizer.batch_decode(
                        dp.batch["input_ids"], skip_special_tokens=True)
                    N = len(texts)
                    for i in range(min(N, 8)):
                        self.augmentation_logger.log_annotation({
                            "event": "teacher_integrated",
                            "augmented_text": texts[i],
                            "record_id": str(dp.non_tensor_batch.get("record_ids", [""] * N)[i]),
                            "estimated_reward": float(dp.non_tensor_batch.get("policy/est_reward", [np.nan] * N)[i]),
                            "solvable": bool(dp.non_tensor_batch.get("teacher/solvable", [True] * N)[i]),
                        })
                except Exception:
                    pass
            try:
                integrated += len(dp.batch["input_ids"])
            except Exception:
                pass

        self._aug_metrics["teacher_integrated"] += integrated

    # --------------------------
    # Augmentation helpers
    # --------------------------
    def _pad_2d_right(self, t: torch.Tensor, tgt_len: int, pad_val: int = 0) -> torch.Tensor:
        if not torch.is_tensor(t) or t.dim() < 2:
            return t
        cur = t.size(1)
        if cur == tgt_len:
            return t
        if cur > tgt_len:
            return t[:, :tgt_len]
        pad_right = tgt_len - cur
        return torch.nn.functional.pad(t, (0, pad_right), value=pad_val)

    def _pad_value_for(self, key: str) -> int:
        if "input_ids" in key:
            return int(getattr(self.tokenizer, "pad_token_id", 0))
        if "attention_mask" in key or key.endswith("mask"):
            return 0
        if "position_ids" in key:
            return 0
        return 0

    def _align_seq_len_for_union(self, left: DataProto, right: DataProto):
        # Harmonize sequence lengths for overlapping 2D tensors (dim=1)
        shared = set(left.batch.keys()).intersection(set(right.batch.keys()))
        for k in list(shared):
            ta, tb = left.batch[k], right.batch[k]
            if torch.is_tensor(ta) and torch.is_tensor(tb) and ta.dim() >= 2 and tb.dim() >= 2:
                tgt = max(ta.size(1), tb.size(1))
                padv = self._pad_value_for(k)
                left.batch[k] = self._pad_2d_right(ta, tgt, padv)
                right.batch[k] = self._pad_2d_right(tb, tgt, padv)

    def _tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        if not texts:
            raise ValueError("Cannot tokenize empty text list")
        max_len = int(getattr(self.config.data, "max_prompt_length", 2048))
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=max_len, return_tensors="pt")
        attention_mask = enc["attention_mask"]
        position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min_(0)
        return {"input_ids": enc["input_ids"], "attention_mask": attention_mask, "position_ids": position_ids}

    def _make_numeric_variants(self, originals: List[str], k: int) -> Dict[str, List[str]]:
        import random
        out: Dict[str, List[str]] = {}
        for src in originals:
            variants = []
            nums = list(re.finditer(r"(?<![A-Za-z_.])(?:\d+(?:\.\d+)?)", src))
            for _ in range(k):
                if not nums:
                    variants.append(src)
                    continue
                s = list(src)
                for m in random.sample(nums, k=min(2, len(nums))):
                    a, b = m.span()
                    val = m.group(0)
                    try:
                        x = float(val)
                    except Exception:
                        continue
                    delta = max(0.01, abs(x) * 0.05)
                    new = x + random.choice([-1, 1]) * delta
                    s[a:b] = list(f"{new:.3g}")
                variants.append("".join(s))
            out[src] = variants
        return out

    def _estimate_rewards_from_difficulty(self, base_reward: float, diffs: List[float]) -> List[float]:
        est = []
        for d in diffs:
            d = float(np.clip(d, 0.75, 1.33)) if np.isfinite(d) else 1.0
            est.append(float(np.clip((base_reward / d)
                       if base_reward >= 0 else (base_reward * d), -1.0, 1.0)))
        return est

    def _extract_prompt_texts(self, dp: DataProto) -> List[str]:
        ids = dp.batch.get("prompts", None)
        if ids is None:
            ids = dp.batch.get("input_ids", None)
        if ids is None:
            return []
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def _build_aug_proto(
        self,
        new_texts: List[str],
        est_rewards: List[float],
        original_texts: List[str],
        record_ids: Optional[List[str]] = None,
    ) -> DataProto:
        enc = self._tokenize_texts(new_texts)
        if record_ids is None:
            record_ids = [str(uuid.uuid4()) for _ in new_texts]
        nt = {
            "raw_prompt_data": np.asarray(new_texts, dtype=object),
            "original_text": np.asarray(original_texts, dtype=object),
            "policy/est_reward": np.asarray(est_rewards, dtype=float),
            "record_ids": np.asarray(record_ids, dtype=object),
            "origin": np.asarray(["augmented"] * len(new_texts), dtype=object),
            "is_augmented": np.asarray([True] * len(new_texts), dtype=bool),
            "data_source": np.asarray(["math_dapo"] * len(new_texts), dtype=object),
        }
        td = DataProto.from_single_dict(
            {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "position_ids": enc["position_ids"]}
        ).batch
        return DataProto(batch=td, non_tensor_batch=nt, meta_info={})

    def _augment_from_batch(self, src_batch: DataProto, num_per_prompt: int = 1) -> Optional[DataProto]:
        if num_per_prompt <= 0:
            return None
        src_texts_all = self._extract_prompt_texts(src_batch)
        uniq = list(dict.fromkeys(
            [s for s in src_texts_all if s and s.strip()]))
        if not uniq:
            return None

        variants_map = self._make_numeric_variants(uniq, k=num_per_prompt)
        new_texts, diffs, originals_for_new = [], [], []
        for src, alts in variants_map.items():
            def _count_nums(s: str) -> int:
                return len(re.findall(r"(?<![A-Za-z_.])(?:\d+(?:\.\d+)?)", s))
            a = max(1, _count_nums(src))
            for t in alts:
                b = max(1, _count_nums(t))
                new_texts.append(t)
                originals_for_new.append(src)
                diffs.append(float(np.clip(b / a, 0.75, 1.33)))

        base_reward = 0.5
        est_rewards = self._estimate_rewards_from_difficulty(
            base_reward, diffs)
        dp = self._build_aug_proto(new_texts, est_rewards, originals_for_new)

        if self.augmentation_logger:
            for t, o, r in zip(new_texts[:8], originals_for_new[:8], est_rewards[:8]):
                self.augmentation_logger.log_augmentation(
                    {"original_text": o, "augmented_text": t,
                        "estimated_reward": float(r)}
                )
        return dp

    # ----- pool glue -----
    def _dp_to_records(self, dp: DataProto) -> List[QueryRecord]:
        texts = list(dp.non_tensor_batch.get("raw_prompt_data", []))
        origs = list(dp.non_tensor_batch.get(
            "original_text", [""] * len(texts)))
        est = list(dp.non_tensor_batch.get(
            "policy/est_reward", [0.5] * len(texts)))
        recs: List[QueryRecord] = []
        for t, o, e in zip(texts, origs, est):
            r = float(e) if isinstance(e, (int, float, np.floating)) else 0.5
            recs.append(
                QueryRecord(
                    raw_prompt_data=t,
                    input_ids=None,
                    attention_mask=None,
                    position_ids=None,
                    reward=r,
                    est_reward=r,
                    meta={"source": "augment"},
                    original_text=o,
                    augmented_text=str(t),
                )
            )
        return recs

    def _records_to_dataproto(self, recs: List[QueryRecord]) -> DataProto:
        new_texts = [r.augmented_text or (
            str(r.raw_prompt_data) if r.raw_prompt_data is not None else "") for r in recs]
        original_texts = [r.original_text or "" for r in recs]
        est_rewards = [
            float(r.est_reward if r.est_reward is not None else r.reward) for r in recs]
        record_ids = [r.record_id for r in recs]
        return self._build_aug_proto(new_texts, est_rewards, original_texts, record_ids=record_ids)

    # --------------------------
    # Training loop (PPO flow intact)
    # --------------------------

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        from verl.utils.debug import marked_timer

        tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                tracker.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        self._start_teacher_pipeline()

        progress_bar = tqdm(total=self.total_training_steps,
                            initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.config.global_profiler.steps is not None
            and self.global_steps in self.config.global_profiler.steps
        )
        next_step_profile = False

        try:
            for epoch in range(self.config.trainer.total_epochs):
                for batch_dict in self.train_dataloader:
                    metrics: Dict[str, Any] = {}
                    timing_raw: Dict[str, float] = {}

                    # integrate teacher outputs if any
                    self._drain_teacher_inbox()

                    with marked_timer("start_profile", timing_raw):
                        self._start_profiling(
                            (not prev_step_profile and curr_step_profile)
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )

                    # ---- base PPO flow (unchanged) ----
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                    gen_batch = self._get_gen_batch(batch)
                    gen_batch.meta_info["global_steps"] = self.global_steps
                    gen_batch = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )

                    is_last_step = self.global_steps >= self.total_training_steps

                    with marked_timer("step", timing_raw):
                        # rollout
                        with marked_timer("gen", timing_raw, color="red"):
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(
                                    gen_batch)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(
                                    gen_batch)
                            timing_raw.update(
                                gen_batch_output.meta_info.get("timing", {}))
                            gen_batch_output.meta_info.pop("timing", None)

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            if self.reward_fn is None:
                                raise ValueError(
                                    "A reward_fn is required for REMAX.")
                            with marked_timer("gen_max", timing_raw, color="purple"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                if not self.async_rollout_mode:
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(
                                        gen_baseline_batch)
                                else:
                                    gen_baseline_output = self.async_rollout_manager.generate_sequences(
                                        gen_baseline_batch)
                                batch = batch.union(gen_baseline_output)
                                reward_baseline_tensor = self.reward_fn(
                                    batch).sum(dim=-1)
                                batch.pop(batch_keys=list(
                                    gen_baseline_output.batch.keys()))
                                batch.batch["reward_baselines"] = reward_baseline_tensor
                                del gen_baseline_batch, gen_baseline_output

                        batch = batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                        )
                        batch = batch.union(gen_batch_output)

                        if "response_mask" not in batch.batch:
                            batch.batch["response_mask"] = compute_response_mask(
                                batch)

                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        batch.meta_info["global_token_num"] = torch.sum(
                            batch.batch["attention_mask"], dim=-1).tolist()

                        # rewards
                        with marked_timer("reward", timing_raw, color="yellow"):
                            if self.use_rm and "rm_scores" not in batch.batch:
                                reward_tensor = self.rm_wg.compute_rm_score(
                                    batch)
                                batch = batch.union(reward_tensor)

                            reward_extra_infos_dict: Dict[str, List] = {}
                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(
                                    data=batch, reward_fn=self.reward_fn)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(
                                    batch, self.reward_fn)

                        # log probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(
                                batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            metrics.update(
                                {"actor/entropy": float(entropy_agg.detach().item())})
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                        # ref policy
                        if self.use_reference_policy:
                            with marked_timer("ref", timing_raw, color="olive"):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                        batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(
                                        batch)
                                batch = batch.union(ref_log_prob)

                        # critic values
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        # advantages
                        with marked_timer("adv", timing_raw, color="brown"):
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(
                                    future_reward)  # type: ignore
                            batch.batch["token_level_scores"] = reward_tensor
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True)
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                        # --------- AUGMENT -> POOL -> SAMPLE MEDIUM -> TEACHER ----------
                        try:
                            aug_cfg = getattr(
                                self.config, "augmentation", {}) or {}
                            if bool(aug_cfg.get("enable", True)):
                                k_per_prompt = int(
                                    aug_cfg.get("num_per_prompt", 1))
                                aug_dp = self._augment_from_batch(
                                    batch, num_per_prompt=k_per_prompt)
                                if aug_dp is not None and len(aug_dp.batch["input_ids"]) > 0:
                                    if self._pool_enabled and self.query_pool is not None:
                                        recs = self._dp_to_records(aug_dp)
                                        self.query_pool.add_many(recs)
                                        self._aug_metrics["pool_added"] += len(
                                            recs)

                                        # sample medium difficulty from pool
                                        to_sample = int(
                                            self._pool_sample_per_step)
                                        if to_sample > 0 and self.teacher_annotator is not None:
                                            sampled = self.query_pool.sample_batch(
                                                to_sample)
                                            if sampled:
                                                dp_to_teacher = self._records_to_dataproto(
                                                    sampled)
                                                if self.teacher_annotator.enqueue_aug(dp_to_teacher):
                                                    n_sent = len(sampled)
                                                    self._aug_metrics["pool_sampled"] += n_sent
                                                    self._aug_metrics["enqueued"] += n_sent
                                                else:
                                                    self._aug_metrics["queue_full_events"] += 1
                                        # occasional snapshot
                                        if self.augmentation_logger and self._pool_snapshot_every > 0:
                                            if (self.global_steps - self._last_pool_snapshot_step) >= self._pool_snapshot_every:
                                                self._last_pool_snapshot_step = self.global_steps
                                                self.augmentation_logger.log_pool_snapshot(
                                                    self.query_pool.get_metrics(),
                                                    sample=self.query_pool.get_sample_for_logging(
                                                        10),
                                                )
                                    else:
                                        # fallback: no pool, enqueue all aug to teacher
                                        if self.teacher_annotator is not None:
                                            if self.teacher_annotator.enqueue_aug(aug_dp):
                                                self._aug_metrics["enqueued"] += len(
                                                    aug_dp.batch["input_ids"])
                                            else:
                                                self._aug_metrics["queue_full_events"] += 1
                        except Exception as e:
                            logger.debug(
                                f"Augmentation/pool enqueue skipped: {e}")
                        # -----------------------------------------------------------------

                        # critic update
                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, color="pink"):
                                critic_output = self.critic_wg.update_critic(
                                    batch)
                            metrics.update(reduce_metrics(
                                critic_output.meta_info["metrics"]))

                        # actor update
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with marked_timer("update_actor", timing_raw, color="red"):
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(
                                    batch)
                            metrics.update(reduce_metrics(
                                actor_output.meta_info["metrics"]))

                    # validation checkpoint / metrics
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
                    esi_close = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    with marked_timer("stop_profile", timing_raw):
                        next_step_profile = (
                            self.config.global_profiler.steps is not None
                            and (self.global_steps + 1) in self.config.global_profiler.steps
                        )
                        self._stop_profiling(
                            curr_step_profile and not next_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )
                        prev_step_profile = curr_step_profile
                        curr_step_profile = next_step_profile

                    steps_duration = float(timing_raw.get("step", 0.0))
                    self.max_steps_duration = max(
                        self.max_steps_duration, steps_duration)

                    # metrics incl. pool
                    pool_metrics = self.query_pool.get_metrics() if (
                        self._pool_enabled and self.query_pool) else {}
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                            "augmentation/enqueued": self._aug_metrics["enqueued"],
                            "augmentation/teacher_submitted": self._aug_metrics["teacher_submitted"],
                            "augmentation/teacher_integrated": self._aug_metrics["teacher_integrated"],
                            "augmentation/queue_full_events": self._aug_metrics["queue_full_events"],
                            "pool/size": pool_metrics.get("total_size", 0),
                            "pool/low_heap": pool_metrics.get("low_heap_size", 0),
                            "pool/high_heap": pool_metrics.get("high_heap_size", 0),
                            "pool/added": self._aug_metrics["pool_added"],
                            "pool/sampled": self._aug_metrics["pool_sampled"],
                        }
                    )
                    metrics.update(compute_data_metrics(
                        batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(
                        batch=batch, timing_raw=timing_raw))
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                    tracker.log(data=metrics, step=self.global_steps)
                    progress_bar.update(1)
                    self.global_steps += 1

                    if is_last_step:
                        progress_bar.close()
                        return
        finally:
            self._cleanup_teacher_pipeline()
